from typing import Optional, Dict

import jax
import jax.numpy as jnp
import jax.lax as lax
import flax




@flax.struct.dataclass
class GreedyState:
    cur_len: jnp.ndarray
    sequences: jnp.ndarray
    running_token: jnp.ndarray
    is_sent_finished: jnp.ndarray
    start_pos: int
    cache: jax.Array
    # model_kwargs: Dict[str, jnp.ndarray]


def _greedy_search(
    model,
    variables: Dict,
    prompt_tokens: jax.Array,
    max_length: Optional[int] = None,  # TOOD: really optional?
    pad_token_id: Optional[int] = None,  # TOOD: really optional?
    eos_token_id: Optional[int] = None,  # TOOD: really optional?
):
    bsz, cur_len = prompt_tokens.shape

    eos_token_id = jnp.array(eos_token_id, dtype=jnp.int32 if eos_token_id is not None else None)
    pad_token_id = jnp.array(pad_token_id, dtype=jnp.int32)
    cur_len = jnp.array(cur_len)

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
        finish_generation = jnp.logical_or(has_reached_max_length, all_sequence_finished)
        return ~finish_generation

    def greedy_search_body_fn(state):
        """state update fn."""
        variables = {"params": params, "cache": state.cache}
        logits, v_upd = model.apply(variables, state.running_token, state.start_pos, mutable=("cache",))
        next_token_logits = logits[:, -1]

        # TODO: put cache from v_upd to the state

        next_token = jnp.argmax(next_token_logits, axis=-1)

        next_token = next_token * ~state.is_sent_finished + pad_token_id * state.is_sent_finished
        next_is_sent_finished = state.is_sent_finished | (next_token == eos_token_id)
        next_token = next_token[:, None]

        next_sequences = lax.dynamic_update_slice(state.sequences, next_token, (0, state.cur_len))
        next_start_pos = state.start_pos + state.running_token.shape[-1]
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

    state = lax.while_loop(greedy_search_cond_fn, greedy_search_body_fn, state)

    return state.sequences


################################################################

from fabrique.llama import Llama

# MODEL_DIR = "/home/devpod/.cache/huggingface/hub/models--microsoft--Phi-3-mini-128k-instruct/snapshots/f10fb29b79f038c78229ab4dcd9234a9666a770f/"
MODEL_DIR = "/home/devpod/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/1448453bdb895762499deb4176c1dd83b145fac1/"

PROMPT = """<|begin_of_text|>Tell me a story"""


def main():
    model_dir = MODEL_DIR
    kwargs = {"max_seq_len": 512, "max_batch_size": 1}
    llama = Llama(model_dir, **kwargs)
    tokenizer, model, variables = llama.tokenizer, llama.model, llama.variables

    prompt_tokens = tokenizer.encode(PROMPT).ids
    prompt_tokens = jnp.asarray(prompt_tokens).reshape(1, -1)

    rng = jax.random.PRNGKey(925)
    logits, v_upd = model.apply(variables, prompt_tokens, 0, mutable=("cache",))

    jit_apply = jax.jit(model.apply, static_argnames=("mutable",))
    logits, v_upd = jit_apply(variables, prompt_tokens, 0, mutable=("cache",))

    ids = jnp.argmax(logits, axis=-1)[0]
    tokenizer.decode(ids)

    bos_token_id = 128000
    eos_token_id = 128001
    pad_token_id = eos_token_id
    max_length = 512

    sequences = _greedy_search(model, variables, prompt_tokens, max_length, pad_token_id, eos_token_id)
    tokenizer.decode(sequences[0])