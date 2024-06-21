import jax
import jax.numpy as jnp


def repeat_kv(x: jax.Array, n_rep: int) -> jax.Array:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return jnp.tile(x[:, :, :, jnp.newaxis, :], (1, 1, 1, n_rep, 1)).reshape(
        bs, slen, n_kv_heads * n_rep, head_dim
    )
