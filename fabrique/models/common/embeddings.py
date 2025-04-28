import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np


def create_sinusoidal_positions(num_pos, dim, theta=10000):
    inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2) / dim))
    freqs = np.einsum("i , j -> i j", np.arange(num_pos), inv_freq).astype("float32")

    emb = np.concatenate((freqs, freqs), axis=-1)
    return np.concatenate((np.sin(emb), np.cos(emb)), axis=-1)



def _llama_rope_scaling(
    inv_freq: jax.Array,
    factor=8,
    low_freq_factor=1,
    high_freq_factor=4,
    old_context_len=8192
):
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * jnp.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = jnp.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = jnp.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
    return inv_freq_llama


def create_llama_sinusoidal_positions(
    num_pos,
    dim,
    theta=10000,
    factor=8,
    low_freq_factor=1,
    high_freq_factor=4,
    old_context_len=8192
):
    # original inv_freq
    inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2) / dim))

    # Llama specifics
    inv_freq_llama = _llama_rope_scaling(
        inv_freq,
        factor=factor,
        low_freq_factor=low_freq_factor,
        high_freq_factor=high_freq_factor,
        old_context_len=old_context_len
    )

    # original emb calculation, cont'd
    freqs = np.einsum("i , j -> i j", np.arange(num_pos), inv_freq_llama).astype("float32")
    emb = np.concatenate((freqs, freqs), axis=-1)
    return np.concatenate((np.sin(emb), np.cos(emb)), axis=-1)





def rotate_half(tensor):
    """Rotates half the hidden dims of the input."""
    rotate_half_tensor = jnp.concatenate(
        (-tensor[..., tensor.shape[-1] // 2 :], tensor[..., : tensor.shape[-1] // 2]),
        axis=-1,
    )
    return rotate_half_tensor


def apply_rotary_pos_emb(xq: jax.Array, xk: jax.Array, sincos, start_pos: int):
    dtype = xq.dtype
    assert len(xq.shape) >= 4  # (*bs, seqlen, n_heads, head_dim)
    seqlen = xq.shape[-3]

    sincos_slice = lax.dynamic_slice(sincos, (start_pos, 0), (seqlen, sincos.shape[-1]))
    sincos_slice = sincos_slice[jnp.newaxis, :, jnp.newaxis, :]
    sin_pos, cos_pos = jnp.split(sincos_slice, 2, axis=-1)
    xq = (xq * cos_pos) + (rotate_half(xq) * sin_pos)
    xk = (xk * cos_pos) + (rotate_half(xk) * sin_pos)
    xq = jnp.asarray(xq, dtype=dtype)
    xk = jnp.asarray(xk, dtype=dtype)
    return xq, xk
