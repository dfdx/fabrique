import json
import math
from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

from fabrique.models.common.cache import KVCache, concatenate_to_cache
from fabrique.models.common.embeddings import apply_rotary_pos_emb, create_sinusoidal_positions
from fabrique.models.common.norm import RMSNorm
from fabrique.models.common.utils import repeat_kv


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


class Attention(nnx.Module):
    """
    Multi-head attention module.

    Args:
        args (ModelArgs): Model configuration parameters.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.
        cache_k (jax.Array): Cached keys for attention.
        cache_v (jax.Array): Cached values for attention.
    """

    def __init__(self, args: ModelArgs, rngs: nnx.Rngs):
        self.args = args
        self.n_heads = args.n_heads
        self.n_kv_heads = self.n_heads if args.n_kv_heads is None else args.n_kv_heads

        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = self.args.dim // self.n_heads

        dense = partial(
            nnx.Linear,
            use_bias=False,
            dtype=self.args.dtype,
            param_dtype=self.args.param_dtype,
            kernel_init=jax.nn.initializers.normal(0.02),  # 0.02 - initializer range,
            rngs=rngs,
        )
        op_size = self.n_heads * self.head_dim + 2 * (self.n_kv_heads * self.head_dim)
        self.wqkv = dense(args.dim, op_size)
        self.wo = dense(self.n_heads * self.head_dim, self.args.dim)
        # if use_cache == False, we still create the variable to keep the same structure
        # but set its length to zero
        cache_len = self.args.max_seq_len if self.args.use_cache else 0
        cache_shape = (args.max_batch_size, cache_len, self.n_kv_heads, self.head_dim)
        self.cache_k = KVCache(jnp.zeros(cache_shape, args.param_dtype))
        self.cache_v = KVCache(jnp.zeros(cache_shape, args.param_dtype))

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
            start_pos (int): Starting position for computations. Items before this
                position are taken from cache.
            sincos (jax.Array): Precomputed frequency array.
            full_causal_mask (jax.Array): Causal mask of size (max_seq_len x max_seq_len).

        Returns:
            jax.Array: Output array after attention.
        """
        bsz, seq_len, _ = x.shape
        q_len = seq_len
        max_kv_len = self.args.max_seq_len if self.args.use_cache else q_len

        qkv = self.wqkv(x)
        query_pos = self.n_heads * self.head_dim
        xq = qkv[..., :query_pos]
        xk = qkv[..., query_pos : query_pos + self.n_kv_heads * self.head_dim]
        xv = qkv[..., query_pos + self.n_kv_heads * self.head_dim :]

        xq = xq.reshape(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.reshape(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_pos_emb(xq, xk, sincos, start_pos)

        causal_mask = jax.lax.dynamic_slice(
            full_causal_mask, (0, 0, start_pos, 0), (1, 1, q_len, max_kv_len)
        )

        mask = causal_mask
        if self.args.use_cache:
            # shape of kv after concatenating to the cache is
            # [bs, max_seq_len, n_heads, head_dim]
            xk, xv, mask = concatenate_to_cache(self.cache_k, self.cache_v, xk, xv, xq, mask, start_pos)

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
        linear = partial(
            nnx.Linear,
            use_bias=False,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )
        self.w1 = linear(dim, 2 * hidden_dim)  # gate + up projection
        self.w2 = linear(hidden_dim, dim)      # down projection

    def __call__(self, x):
        gate_up = self.w1(x)
        gate, up = jnp.split(gate_up, 2, axis=-1)
        return self.w2(up * nnx.silu(gate))


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
            full_causal_mask (jax.Array, optional): Causal mask of size (max_seq_len x max_seq_len).

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
            args (ModelArgs): Model configuration parameters.

        Attributes:
            args (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (nn.Embed): Token embeddings.
            layers (list): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (Linear): Linear layer for final output.
            sincos (jax.Array): Precomputed cosine and sine frequencies.

        """
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        self.tok_embeddings = nnx.Embed(
            num_embeddings=args.vocab_size,
            features=args.dim,
            dtype=args.dtype,
            param_dtype=args.param_dtype,
            rngs=rngs,
        )

        self.layers = [TransformerBlock(args, rngs=rngs) for _ in range(args.n_layers)]

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
        for i, layer in enumerate(self.layers):
            h = layer(h, start_pos, self.sincos, self.causal_mask)
        h = self.norm(h)
        output = self.output(h).astype("float32")
        return output


###########################################################

def main():
    from tokenizers import Tokenizer
    from fabrique.loading import load_from_pretrained
    from fabrique.models.phi.load_rules import RULES
    model_id = "microsoft/Phi-3-mini-128k-instruct"
    model_args = {"max_seq_len": 512, "max_batch_size": 1}

    tokenizer, model, hf_config = load_from_pretrained(Tokenizer, ModelArgs, Transformer, RULES, model_id, **model_args)
    tokens = tokenizer.encode("Once upon a time").ids
    tokens = jnp.array(tokens).reshape(1, -1)
    out = model(tokens, 0).argmax(axis=-1).reshape(-1)
    tokenizer.decode(out)

