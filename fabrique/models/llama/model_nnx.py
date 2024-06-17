import json
import math
from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.linen.attention import combine_masks
from jax import lax

from fabrique.utils import print_var


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_hidden_size: int = 14336
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    use_cache: bool = True

    @staticmethod
    def from_file(config_file: str, **kwargs):
        """
        Load ModelArgs from a Hugginface config file.
        """
        with open(config_file) as fp:
            config = json.load(fp)
        args = ModelArgs(
            dim=config["hidden_size"],
            n_layers=config["num_hidden_layers"],
            n_heads=config["num_attention_heads"],
            n_kv_heads=config["num_key_value_heads"],
            vocab_size=config["vocab_size"],
            # multiple_of=
            # ffn_dim_multiplier=
            ffn_hidden_size=config["intermediate_size"],
            norm_eps=config["rms_norm_eps"],
            # max_batch_size=
            max_seq_len=config["max_position_embeddings"],
        )
        for k, v in kwargs.items():
            setattr(args, k, v)
        return args


class RMSNorm(nnx.Module):

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        param_dtype: jnp.dtype = jnp.float32,

    ):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input array.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        self.dim = dim
        self.eps = eps
        self.param_dtype = param_dtype
        self.weight = nnx.Param(jnp.ones(self.dim, dtype=self.param_dtype))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input array.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array: The normalized array.

        """
        return x * jax.lax.rsqrt(
            jnp.power(x, 2).mean(axis=-1, keepdims=True) + self.eps
        )

    def __call__(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array: The output array after applying RMSNorm.

        """
        output = self._norm(x.astype("float32")).astype(x.dtype)
        return output * self.weight


def create_sinusoidal_positions(num_pos, dim):
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    freqs = np.einsum("i , j -> i j", np.arange(num_pos), inv_freq).astype("float32")

    emb = np.concatenate((freqs, freqs), axis=-1)
    # out = np.concatenate((np.sin(emb)[:, None, :], np.cos(emb)[:, None, :]), axis=-1)
    # return jnp.array(out[:, :, :num_pos])
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


def repeat_kv(x: jax.Array, n_rep: int) -> jax.Array:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return jnp.tile(x[:, :, :, jnp.newaxis, :], (1, 1, 1, n_rep, 1)).reshape(
        bs, slen, n_kv_heads * n_rep, head_dim
    )


class KVCache(nnx.Variable):
    pass


class Attention(nnx.Module):
    """
    Multi-head attention module.

    Args:
        args (ModelArgs): Model configuration parameters.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Dense): Linear transformation for queries.
        wk (Dense): Linear transformation for keys.
        wv (Dense): Linear transformation for values.
        wo (Dense): Linear transformation for output.
        cache_k (jax.Array): Cached keys for attention.
        cache_v (jax.Array): Cached values for attention.
    """

    def __init__(self, args: ModelArgs, rngs: nnx.Rngs):
        self.args = args
        self.n_heads = args.n_heads
        self.n_kv_heads = (
            self.n_heads if args.n_kv_heads is None else args.n_kv_heads
        )

        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = self.args.dim // self.n_heads

        dense = partial(
            nnx.Linear,
            use_bias=False,
            dtype=self.args.dtype,
            param_dtype=self.args.param_dtype,
            kernel_init=jax.nn.initializers.normal(0.02),  # 0.02 - initializer range,
            rngs=rngs
        )
        self.wq = dense(args.dim, self.n_heads * self.head_dim)
        self.wk = dense(args.dim, self.n_kv_heads * self.head_dim)
        self.wv = dense(args.dim, self.n_kv_heads * self.head_dim)
        self.wo = dense(self.n_heads * self.head_dim, self.args.dim)
        # if use_cache == False, we still create the variable to keep the same structure
        # but set its length to zero
        cache_len = self.args.max_seq_len if self.args.use_cache else 0
        cache_shape = (args.max_batch_size, cache_len, self.n_kv_heads, self.head_dim)
        self.cache_k = KVCache(jnp.zeros(cache_shape, args.param_dtype))
        self.cache_v = KVCache(jnp.zeros(cache_shape, args.param_dtype))

    def _concatenate_to_cache(self, xk, xv, xq, attn_mask, start_pos: int):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        *batch_dims, max_length, num_heads, depth_per_head = self.cache_k.value.shape
        # indices are [starting_batch, start_pos_in_seq, starting_head, starting_pos_in_head]
        indices = (0,) * len(batch_dims) + (start_pos, 0, 0)
        # note: keys and values now have length == max_length, i.e. may be longer than xk/xv
        keys = lax.dynamic_update_slice(self.cache_k.value, xk, indices)
        values = lax.dynamic_update_slice(self.cache_v.value, xv, indices)
        self.cache_k.value = keys
        self.cache_v.value = values
        num_updated_cache_vectors = xq.shape[1]
        # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
        pad_mask = jnp.broadcast_to(
            jnp.arange(max_length) < start_pos + num_updated_cache_vectors,
            tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
        )
        attn_mask = combine_masks(pad_mask, attn_mask)
        return keys, values, attn_mask

    def __call__(
        self,
        x: jax.Array,
        start_pos: int,
        sincos: jax.Array,
        full_causal_mask: jax.Array,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (jax.Array): Input array.
            start_pos (int): Starting position for caching.
            sincos (jax.Array): Precomputed frequency array.
            full_causal_mask (jax.Array): Causal mask of size (max_seq_len x max_seq_len).

        Returns:
            jax.Array: Output array after attention.
        """
        bsz, seq_len, _ = x.shape
        q_len = seq_len
        max_kv_len = self.args.max_seq_len if self.args.use_cache else q_len

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.reshape(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.reshape(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_pos_emb(xq, xk, sincos, start_pos)

        causal_mask = jax.lax.dynamic_slice(
            full_causal_mask, (0, 0, start_pos, 0), (1, 1, q_len, max_kv_len)
        )

        # print_var("xk before cache", xk)
        # print_var("xv before cache", xv)
        # print_var("cache_k", self.cache_k.value)

        mask = causal_mask
        if self.args.use_cache:
            # shape of kv after concatenating to the cache is
            # [bs, max_seq_len, n_heads, head_dim]
            xk, xv, mask = self._concatenate_to_cache(xk, xv, xq, mask, start_pos)

        # print_var("xk after cache", xk)
        # print_var("xv after cache", xv)

        # repeat k/v heads if n_kv_heads < n_heads
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = jnp.moveaxis(xq, 1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = jnp.moveaxis(xk, 1, 2)
        xv = jnp.moveaxis(xv, 1, 2)

        scores = jnp.matmul(xq, jnp.moveaxis(xk, 2, 3)) / math.sqrt(self.head_dim)

        if mask is not None:  # should we even allow mask to be None?
            # so far we used mask with 1s to mean "attend" and 0s to mean "ignore"
            # to apply it to scores, we convert it to 0s and -inf accordingly
            imask = jnp.where(mask == 0, -jnp.inf, 0)
            scores = scores + imask  # (bs, n_heads, q_len, kv_len)
        scores = nnx.softmax(scores.astype("float32"), axis=-1).astype(xq.dtype)

        # output = jnp.matmul(scores, xv)  # (bs, n_heads, q_len, head_dim)??
        # output = jnp.moveaxis(output, 1, 2).ravel().reshape(bsz, seq_len, -1)
        output = jnp.einsum(
            "...hqk,...hkd->...qhd", scores, xv
        )  # (bs, q_len, n_heads, head_dim)
        output = output.reshape(output.shape[:2] + (self.args.dim,))

        return self.wo(output)


class FeedForward(nnx.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(params=0),
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (Linear): Linear transformation for the first layer.
            w2 (Linear): Linear transformation for the second layer.
            w3 (Linear): Linear transformation for the third layer.

        """
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.dtype = dtype
        self.param_dtype = param_dtype
        # hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if self.ffn_dim_multiplier is not None:
            hidden_dim = int(self.ffn_dim_multiplier * hidden_dim)
        hidden_dim = self.multiple_of * (
            (hidden_dim + self.multiple_of - 1) // self.multiple_of
        )
        linear = partial(nnx.Linear, use_bias=False, param_dtype=self.param_dtype, dtype=self.dtype, rngs=rngs)
        self.w1 = linear(dim, hidden_dim)
        self.w2 = linear(hidden_dim, dim)
        self.w3 = linear(dim, hidden_dim)

    def __call__(self, x):
        return self.w2(nnx.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nnx.Module):

    def __init__(self, args: ModelArgs, rngs: nnx.Rngs):
        """
        Initialize a TransformerBlock.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        self.args = args
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args, rngs=rngs)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.ffn_hidden_size,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            dtype=self.args.dtype,
            param_dtype=self.args.param_dtype,
            rngs=rngs,
        )
        self.attention_norm = RMSNorm(
            args.dim, eps=args.norm_eps, param_dtype=args.param_dtype
        )
        self.ffn_norm = RMSNorm(
            args.dim, eps=args.norm_eps, param_dtype=args.param_dtype
        )

    def __call__(
        self,
        x: jax.Array,
        start_pos: int,
        sincos: jax.Array,
        full_causal_mask: jax.Array,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (jax.Array): Input array.
            start_pos (int): Starting position for attention caching.
            sincos (jax.Array): Precomputed freqs_ciscosine and sine frequencies.
            mask (jax.Array, optional): Masking tensor for attention. Defaults to None.

        Returns:
            jax.Array: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention(
            self.attention_norm(x), start_pos, sincos, full_causal_mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nnx.Module):

    def __init__(self, args: ModelArgs, rngs: nnx.Rngs = nnx.Rngs(params=0)):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (nn.Embed): Token embeddings.
            layers (list): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (nn.Dense): Linear layer for final output.
            sincos (jax.Array): Precomputed cosine and sine frequencies.

        """
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        self.tok_embeddings = nnx.Embed(
            num_embeddings=args.vocab_size, features=args.dim,
            dtype=args.dtype, param_dtype=args.param_dtype, rngs=rngs
        )

        self.layers = [
            TransformerBlock(args, rngs=rngs)
            for _ in range(args.n_layers)
        ]

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nnx.Linear(args.dim, args.vocab_size, use_bias=False, rngs=rngs)

        self.sincos = create_sinusoidal_positions(
            args.max_seq_len, args.dim // args.n_heads
        )
        self.causal_mask = nnx.make_causal_mask(
            jnp.ones((1, args.max_seq_len), dtype="bool"), dtype="bool"
        )

    def __call__(self, tokens: jax.Array, start_pos: int):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (jax.Array): Input token indices.
            start_pos (int): Starting position for attention caching.

        Returns:
            jax.Array: Output logits after applying the Transformer model.
        """
        h = self.tok_embeddings(tokens)
        # print_var("h after embeddings", h)
        for i, layer in enumerate(self.layers):
            h = layer(h, start_pos, self.sincos, self.causal_mask)
            # print_var(f"h after layer {i}", h)
        h = self.norm(h)
        # print_var(f"h after self.norm()", h)
        output = self.output(h).astype("float32")
        # print_var(f"output", output)
        return output

