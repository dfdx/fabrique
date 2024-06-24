from functools import partial
from typing import Dict

import flax
import jax
import jax.lax as lax
import jax.numpy as jnp
from flax import nnx


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
    model_state: nnx.State
    static: nnx.GraphDef
    pad_token_id: jnp.ndarray  # doesn't change
    eos_token_id: jnp.ndarray  # doesn't change
    max_length: jnp.ndarray  # doesn't change


@flax.struct.dataclass
class SampleState:
    cur_len: jnp.ndarray
    sequences: jnp.ndarray
    running_token: jnp.ndarray
    is_sent_finished: jnp.ndarray
    prng_key: jnp.ndarray
    start_pos: int
    cache: jax.Array


def greedy_search_cond_fn(state):
    """state termination condition fn."""
    has_reached_max_length = state.cur_len == state.max_length
    all_sequence_finished = jnp.all(state.is_sent_finished)
    finish_generation = jnp.logical_or(has_reached_max_length, all_sequence_finished)
    return ~finish_generation


def greedy_search_body_fn(state):
    """state update fn."""
    model_state = state.model_state
    static = state.static
    model = nnx.merge(static, model_state)
    logits = model(state.running_token, state.start_pos)
    next_token_logits = logits[:, -1]

    next_token = jnp.argmax(next_token_logits, axis=-1)

    pad_token_id = state.pad_token_id
    eos_token_id = state.eos_token_id
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
    # next_cache = v_upd["cache"]

    _, next_model_state = nnx.split(model, ...)
    return state.replace(
        cur_len=state.cur_len + 1,
        sequences=next_sequences,
        running_token=next_token,
        is_sent_finished=next_is_sent_finished,
        start_pos=next_start_pos,
        model_state=next_model_state,
    )

    # return GreedyState(
    #     cur_len=state.cur_len + 1,
    #     sequences=next_sequences,
    #     running_token=next_token,
    #     is_sent_finished=next_is_sent_finished,
    #     start_pos=next_start_pos,
    #     # cache=next_cache,
    #     model_state=next_model_state,
    #     static=static,
    # )


def greedy(
    model,
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

    static, model_state = nnx.split(model, ...)

    # initialize state
    state = GreedyState(
        cur_len=cur_len,  # type: ignore
        sequences=sequences,
        running_token=prompt_tokens,
        is_sent_finished=is_sent_finished,
        start_pos=0,
        model_state=model_state,
        static=static,
        pad_token_id=pad_token_id,  # type: ignore
        eos_token_id=eos_token_id,  # type: ignore
        max_length=max_length,  # type: ignore
    )

    # The very first prompt often has sequence length > 1, so run outside of `lax.while_loop` to comply with TPU
    if prompt_tokens.shape[1] > 1:
        state = greedy_search_body_fn(state)
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
    raise Exception("NNX version of sample() is not implemented yet")
    prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

    batch_size, cur_len = prompt_tokens.shape

    eos_token_id = jnp.array(eos_token_id, dtype=jnp.int32)  # type: ignore
    pad_token_id = jnp.array(pad_token_id, dtype=jnp.int32)  # type: ignore
    cur_len = jnp.array(cur_len)  # type: ignore

    # per batch-item holding current token in loop.
    sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
    sequences = lax.dynamic_update_slice(sequences, prompt_tokens, (0, 0))

    # per batch-item state bit indicating if sentence has finished.
    is_sent_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

    # initialize state
    state = SampleState(
        cur_len=cur_len,  # type: ignore
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
        finish_generation = jnp.logical_or(
            has_reached_max_length, all_sequence_finished
        )
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

        next_token = (
            next_token * ~state.is_sent_finished + pad_token_id * state.is_sent_finished
        )
        next_is_sent_finished = state.is_sent_finished | (next_token == eos_token_id)
        next_token = next_token[:, None]

        next_sequences = lax.dynamic_update_slice(
            state.sequences, next_token, (0, state.cur_len)
        )

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


def example():
    from fabrique.models.llm import LLM

    model_id = "microsoft/Phi-3-mini-128k-instruct"
    kwargs = {
        "max_seq_len": 512,
        "max_batch_size": 1,
    }
    llm = LLM.from_pretrained(model_id, **kwargs)
    model, tokenizer, hf_config = llm.model, llm.tokenizer, llm.hf_config

    # prompt = """{"name": "Thomas", "surname": "Anderson", "age":"""
    prompt = """<|user|>\nWrite a long poem about Disney Land<|end|>\n<|assistant|>"""
    prompt_tokens = tokenizer.encode(prompt).ids
    prompt_tokens = jnp.asarray(prompt_tokens).reshape(1, -1)

    sequences = greedy(
        model,
        prompt_tokens,
        pad_token_id=hf_config["eos_token_id"],
        eos_token_id=hf_config["eos_token_id"],
        max_length=model.args.max_seq_len,
    )
    out = tokenizer.decode(sequences[0])
    print(out)

    pad_token_id = hf_config["pad_token_id"]
    eos_token_id = hf_config["eos_token_id"]
    max_length = 512
