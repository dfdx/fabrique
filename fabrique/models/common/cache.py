import jax
import jax.numpy as jnp
from jax import lax
from flax import nnx
from flax.linen.attention import combine_masks


class KVCache(nnx.Variable):
    pass


def concatenate_to_cache(cache_k: KVCache, cache_v: KVCache, xk: jax.Array, xv: jax.Array, xq: jax.Array, attn_mask: jax.Array, start_pos: int):
    """
    Take projected key & value states from a single input token and concatenates the states to cached
    states from previous steps.
    """
    *batch_dims, max_length, num_heads, depth_per_head = cache_k.value.shape
    # indices are [starting_batch, start_pos_in_seq, starting_head, starting_pos_in_head]
    indices = (0,) * len(batch_dims) + (start_pos, 0, 0)
    # note: keys and values now have length == max_length, i.e. may be longer than xk/xv
    keys = lax.dynamic_update_slice(cache_k.value, xk, indices)
    values = lax.dynamic_update_slice(cache_v.value, xv, indices)
    cache_k.value = keys
    cache_v.value = values
    num_updated_cache_vectors = xq.shape[1]
    # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
    pad_mask = jnp.broadcast_to(
        jnp.arange(max_length) < start_pos + num_updated_cache_vectors,
        tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
    )
    attn_mask = combine_masks(pad_mask, attn_mask)
    return keys, values, attn_mask