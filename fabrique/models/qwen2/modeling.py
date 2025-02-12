import json
from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

from fabrique.models.common.cache import concatenate_to_cache
from fabrique.models.common.embeddings import apply_rotary_pos_emb
from fabrique.models.llama import modeling as llama
from fabrique.utils import check_and_update_fields


# Qwen2 closely follows Llama architecture with a few exceptions. Thus, instead
# of repeating the code we inherit from Llama classes and add comments about
# the differences. A few differences to be aware off:
#
# 1. In Qwen2's attentions, input projections DO have bias.
# 2. Some sources report that Qwen2 uses sliding window attention,
#    but I couldn't find information about it in the technical report [1]
#     and in the HF config the use_sliding_window option is set to false [2].
#    Thus, sliding window is omitted here.
# 3. Similarly, some sources report use of dropout in attention,
#    but in the same HF config its rate is set to 0, and there's no
#    public information about specific parameters used during training.
#
#
# [1]: https://arxiv.org/pdf/2407.10671
# [2]: https://huggingface.co/Qwen/Qwen2-7B/blob/main/config.json



# TODO: add tests for qwen2



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
        return check_and_update_fields(args, **kwargs)



class Attention(llama.Attention):
    """
    Multi-head attention module.
    """

    def __init__(self, args: ModelArgs, rngs: nnx.Rngs):
        super().__init__(args, rngs)

        dense = partial(
            nnx.Linear,
            dtype=self.args.dtype,
            param_dtype=self.args.param_dtype,
            kernel_init=jax.nn.initializers.normal(0.02),  # 0.02 - initializer range,
            rngs=rngs,
        )
        # unlike in Llama, Qwen uses bias in incoming transformations
        self.wq = dense(args.dim, self.n_heads * self.head_dim, use_bias=True)
        self.wk = dense(args.dim, self.n_kv_heads * self.head_dim, use_bias=True)
        self.wv = dense(args.dim, self.n_kv_heads * self.head_dim, use_bias=True)
        # note: self.wo doesn't use bias and so is the same as in Llama


    # def __call__(
    #     self,
    #     x: jax.Array,
    #     start_pos: int,
    #     sincos: jax.Array,
    #     full_causal_mask: jax.Array,
    # ):
    #     """
    #     Forward pass of the attention module.

    #     Args:
    #         x (jax.Array): Input array.
    #         start_pos (int): Starting position for computations. Items before this
    #             position are taken from cache.
    #         sincos (jax.Array): Precomputed frequency array.
    #         full_causal_mask (jax.Array): Causal mask of size (max_seq_len x max_seq_len).

    #     Returns:
    #         jax.Array: Output array after attention.
    #     """
    #     bsz, seq_len, _ = x.shape
    #     q_len = seq_len
    #     max_kv_len = self.args.max_seq_len if self.args.use_cache else q_len

    #     xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

    #     xq = xq.reshape(bsz, seq_len, self.n_heads, self.head_dim)
    #     xk = xk.reshape(bsz, seq_len, self.n_kv_heads, self.head_dim)
    #     xv = xv.reshape(bsz, seq_len, self.n_kv_heads, self.head_dim)

    #     xq, xk = apply_rotary_pos_emb(xq, xk, sincos, start_pos)

    #     causal_mask = jax.lax.dynamic_slice(
    #         full_causal_mask, (0, 0, start_pos, 0), (1, 1, q_len, max_kv_len)
    #     )

    #     mask = causal_mask
    #     if self.args.use_cache:
    #         # shape of kv after concatenating to the cache is
    #         # [bs, max_seq_len, n_heads, head_dim]
    #         xk, xv, mask = concatenate_to_cache(
    #             self.cache_k, self.cache_v, xk, xv, xq, mask, start_pos
    #         )

    #     output = jax.nn.dot_product_attention(xq, xk, xv, mask=mask)
    #     output = output.reshape(output.shape[:2] + (self.args.dim,))

    #     return self.wo(output)


class TransformerBlock(llama.TransformerBlock):

    def __init__(self, args: ModelArgs, rngs: nnx.Rngs):
        super().__init__(args, rngs)
        # replace Llama attention with the implementation in this file
        self.attention = Attention(args, rngs=rngs)


class Transformer(llama.Transformer):

    def __init__(self, args: ModelArgs, rngs: nnx.Rngs = nnx.Rngs(params=0)):
        super().__init__(args, rngs)
        for i in range(len(self.layers)):
            # replace Llama blocks by the blocks in this file, one by one
            self.layers[i] = TransformerBlock(args, rngs)


# def main():
#     args = ModelArgs(dim=12, n_heads=2, max_seq_len=32)
#     rngs = nnx.Rngs(0)
#     self = Attention(args, rngs)
#     sincos = create_sinusoidal_positions(args.max_seq_len, args.dim // args.n_heads)
#     start_pos = 0
#     full_causal_mask = nnx.make_causal_mask(
#         jnp.ones((1, args.max_seq_len), dtype="bool"), dtype="bool"
#     )
#     x = jax.random.normal(rngs(), (2, 8, 12))
#     y = nnx.jit(self)(x, start_pos, sincos, full_causal_mask)
