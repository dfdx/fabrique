from functools import partial
from typing import Dict

import flax
import jax
import jax.lax as lax
import jax.numpy as jnp
from flax import nnx

from fabrique.models.llama.model import Transformer


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

    static, model_state = nnx.split(model)

    # initialize state
    state = GreedyState(
        cur_len=cur_len,  # type: ignore
        sequences=sequences,
        running_token=prompt_tokens,
        is_sent_finished=is_sent_finished,
        start_pos=0,
        model_state=model_state
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
        model = nnx.merge(static, model_state)
        logits = model(state.running_token, state.start_pos)
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
        # next_cache = v_upd["cache"]
        _, next_model_state = nnx.split(model)
        return GreedyState(
            cur_len=state.cur_len + 1,
            sequences=next_sequences,
            running_token=next_token,
            is_sent_finished=next_is_sent_finished,
            start_pos=next_start_pos,
            # cache=next_cache,
            model_state=next_model_state,
        )

    # The very first prompt often has sequence length > 1, so run outside of `lax.while_loop` to comply with TPU
    if prompt_tokens.shape[1] > 1:
        state = greedy_search_body_fn(state)
    state = lax.while_loop(greedy_search_cond_fn, greedy_search_body_fn, state)
    # state = debug_while_loop(greedy_search_cond_fn, greedy_search_body_fn, state)

    return state.sequences



################################################################


def main():
    from fabrique.models.llama import Llama
    from fabrique.models.llama.model_nnx import Transformer, ModelArgs
    from tokenizers import Tokenizer

    model_id = "meta-llama/Meta-Llama-3-8B"
    kwargs = {
        "max_seq_len": 512,
        "max_batch_size": 1,
        "dtype": jnp.float16,
        "param_dtype": jnp.float16,
    }
    llama = Llama.from_pretrained(model_id, **kwargs)
    # model, variables = llama.model, llama.variables

    tokenizer = Tokenizer.from_pretrained(model_id)
    model = Transformer(ModelArgs(**kwargs, vocab_size=tokenizer.get_vocab_size()))




    prompt = """{"name": "Thomas", "surname": "Anderson", "age":"""
    prompt_tokens = tokenizer.encode(prompt).ids
    prompt_tokens = jnp.asarray(prompt_tokens).reshape(1, -1)

    out = model(prompt_tokens, 0)
    out.argmax(axis=-1).shape

    sequences = greedy(
        model,
        prompt_tokens,
        pad_token_id=llama.hf_config["eos_token_id"],
        eos_token_id=llama.hf_config["eos_token_id"],
        max_length=llama.model.args.max_seq_len,
    )
    out = llama.tokenizer.decode(sequences[0])
    print(out)

    pad_token_id = llama.hf_config["eos_token_id"]
    eos_token_id = llama.hf_config["eos_token_id"]
    max_length = llama.model.args.max_seq_len
