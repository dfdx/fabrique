from typing import Dict
from functools import partial

import flax
import jax
import jax.lax as lax
import jax.numpy as jnp


def debug_while_loop(cond_fun, body_fun, init_val):
    val = init_val
    while cond_fun(val):
        val = body_fun(val)
    return val


@flax.struct.dataclass
class GreedyState:
    cur_len: jnp.ndarray
    sequences: jnp.ndarray
    running_token: jnp.ndarray
    is_sent_finished: jnp.ndarray
    start_pos: int
    cache: jax.Array


@flax.struct.dataclass
class SampleState:
    cur_len: jnp.ndarray
    sequences: jnp.ndarray
    running_token: jnp.ndarray
    is_sent_finished: jnp.ndarray
    prng_key: jnp.ndarray
    start_pos: int
    cache: jax.Array


@flax.struct.dataclass
class BeamSearchState:
    cur_len: jnp.ndarray
    running_sequences: jnp.ndarray
    running_scores: jnp.ndarray
    sequences: jnp.ndarray
    scores: jnp.ndarray
    is_sent_finished: jnp.ndarray
    start_pos: int
    cache: jax.Array


def greedy(
    model,
    variables: Dict,
    prompt_tokens: jax.Array,
    pad_token_id: int,
    eos_token_id: int,
    max_length: int = 512,
):
    bsz, cur_len = prompt_tokens.shape

    eos_token_id = jnp.array(eos_token_id, dtype=jnp.int32)  # type: ignore
    pad_token_id = jnp.array(pad_token_id, dtype=jnp.int32)  # type: ignore
    cur_len = jnp.array(cur_len)  # type: ignore

    # per batch-item holding current token in loop.
    sequences = jnp.full((bsz, max_length), pad_token_id, dtype=jnp.int32)
    sequences = lax.dynamic_update_slice(sequences, prompt_tokens, (0, 0))

    # per batch-item state bit indicating if sentence has finished.
    is_sent_finished = jnp.zeros((bsz,), dtype=jnp.bool_)

    params = variables["params"]

    # initialize state
    state = GreedyState(
        cur_len=cur_len,
        sequences=sequences,
        running_token=prompt_tokens,
        is_sent_finished=is_sent_finished,
        start_pos=0,
        cache=variables["cache"],
        # model_kwargs={}
    )

    def greedy_search_cond_fn(state):
        """state termination condition fn."""
        has_reached_max_length = state.cur_len == max_length
        all_sequence_finished = jnp.all(state.is_sent_finished)
        finish_generation = jnp.logical_or(
            has_reached_max_length, all_sequence_finished
        )
        return ~finish_generation

    def greedy_search_body_fn(state):
        """state update fn."""
        print(">>>>>>>>>> compiling body function <<<<<<<<<<<<<<")
        variables = {"params": params, "cache": state.cache}
        logits, v_upd = model.apply(
            variables, state.running_token, state.start_pos, mutable=("cache",)
        )
        next_token_logits = logits[:, -1]

        next_token = jnp.argmax(next_token_logits, axis=-1)

        next_token = (
            next_token * ~state.is_sent_finished + pad_token_id * state.is_sent_finished
        )
        next_is_sent_finished = state.is_sent_finished | (next_token == eos_token_id)
        next_token = next_token[:, None]

        next_sequences = lax.dynamic_update_slice(
            state.sequences, next_token, (0, state.cur_len)
        )
        # next_start_pos = state.start_pos + state.running_token.shape[-1]
        next_start_pos = state.cur_len
        next_cache = v_upd["cache"]
        return GreedyState(
            cur_len=state.cur_len + 1,
            sequences=next_sequences,
            running_token=next_token,
            is_sent_finished=next_is_sent_finished,
            start_pos=next_start_pos,
            cache=next_cache,
        )

    # The very first prompt often has sequence length > 1, so run outside of `lax.while_loop` to comply with TPU
    if prompt_tokens.shape[1] > 1:
        state = greedy_search_body_fn(state)

    print(greedy_search_body_fn)
    print(id(greedy_search_body_fn))
    state = lax.while_loop(greedy_search_cond_fn, greedy_search_body_fn, state)
    # state = debug_while_loop(greedy_search_cond_fn, greedy_search_body_fn, state)

    return state.sequences



def sample_top_p(rng, probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (jax.Array): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        jax.Array: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    # probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_idx = jnp.flip(jnp.argsort(probs), axis=-1)
    probs_sort = jnp.take(probs, probs_idx)
    probs_sum = jnp.cumsum(probs_sort, axis=-1)
    mask = probs_sum - probs_sort > p
    # probs_sort = probs_sort.at[mask].set(0.0)
    probs_sort = jnp.where(mask, probs_sort, 0)
    probs_sort = probs_sort / probs_sort.sum(axis=-1, keepdims=True)
    # next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = jax.random.categorical(rng, jnp.log(probs_sort), axis=-1)
    # next_token = torch.gather(probs_idx, -1, next_token)
    next_token = probs_idx[jnp.arange(probs.shape[0]), next_token]
    return next_token


def sample(
    model,
    variables: dict,
    prompt_tokens: jax.Array,
    pad_token_id: int,
    eos_token_id: int,
    max_length: int = 512,
    prng_key: jax.Array | None = None,
):
    prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

    batch_size, cur_len = prompt_tokens.shape

    eos_token_id = jnp.array(eos_token_id, dtype=jnp.int32)
    pad_token_id = jnp.array(pad_token_id, dtype=jnp.int32)
    cur_len = jnp.array(cur_len)

    # per batch-item holding current token in loop.
    sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
    sequences = lax.dynamic_update_slice(sequences, prompt_tokens, (0, 0))

    # per batch-item state bit indicating if sentence has finished.
    is_sent_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

    # initialize state
    state = SampleState(
        cur_len=cur_len,
        sequences=sequences,
        running_token=prompt_tokens,
        is_sent_finished=is_sent_finished,
        prng_key=prng_key,
        start_pos=0,
        cache=variables["cache"],
    )

    def sample_search_cond_fn(state):
        """state termination condition fn."""
        has_reached_max_length = state.cur_len == max_length
        all_sequence_finished = jnp.all(state.is_sent_finished)
        finish_generation = jnp.logical_or(has_reached_max_length, all_sequence_finished)
        return ~finish_generation

    def sample_search_body_fn(state):
        """state update fn."""
        prng_key, prng_key_next = jax.random.split(state.prng_key)
        # model_outputs = model(state.running_token, params=params, **state.model_kwargs)
        logits, v_upd = model.apply(
            variables, state.running_token, state.start_pos, mutable=("cache",)
        )

        next_token_logits = logits[:, -1]

        # apply top_p, top_k, temperature
        # next_token_logits = logits_warper(next_token_logits, next_token_logits, state.cur_len)

        # next_token = jax.random.categorical(prng_key, next_token_logits, axis=-1)
        next_token = sample_top_p(prng_key, next_token_logits, 0.9)

        next_token = next_token * ~state.is_sent_finished + pad_token_id * state.is_sent_finished
        next_is_sent_finished = state.is_sent_finished | (next_token == eos_token_id)
        next_token = next_token[:, None]

        next_sequences = lax.dynamic_update_slice(state.sequences, next_token, (0, state.cur_len))

        next_start_pos = state.cur_len
        next_cache = v_upd["cache"]

        return SampleState(
            cur_len=state.cur_len + 1,
            sequences=next_sequences,
            running_token=next_token,
            is_sent_finished=next_is_sent_finished,
            prng_key=prng_key_next,
            start_pos=next_start_pos,
            cache=next_cache,
        )

    # The very first prompt often has sequence length > 1, so run outside of `lax.while_loop` to comply with TPU
    if prompt_tokens.shape[1] > 1:
        state = sample_search_body_fn(state)


    state = lax.while_loop(sample_search_cond_fn, sample_search_body_fn, state)

    return state.sequences


################################################################


def main():
    from fabrique.llama import Llama
    from fabrique.generation import greedy
    model_id = "meta-llama/Meta-Llama-3-8B"
    kwargs = {"max_seq_len": 512, "max_batch_size": 1}
    llama = Llama.from_pretrained(model_id, **kwargs)
    model, variables = llama.model, llama.variables

    prompt = """Once upon a time"""
    prompt_tokens = llama.tokenizer.encode(prompt).ids
    prompt_tokens = jnp.asarray(prompt_tokens).reshape(1, -1)
    sequences = sample(
        model,
        variables,
        prompt_tokens,
        pad_token_id=llama.hf_config["eos_token_id"],
        eos_token_id=llama.hf_config["eos_token_id"],
        max_length=llama.model.args.max_seq_len,
    )
    out = llama.tokenizer.decode(sequences[0])
    print(out)