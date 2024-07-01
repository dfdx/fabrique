import json
import math
from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

from fabrique.models.common.cache import KVCache, concatenate_to_cache
from fabrique.models.common.embeddings import (
    apply_rotary_pos_emb,
    create_sinusoidal_positions,
)
from fabrique.models.common.norm import RMSNorm
from fabrique.models.common.utils import repeat_kv
from fabrique.utils import print_var


@dataclass
class ModelArgs:
    dim: int = 768                # name in hf: hidden_size
    n_layers: int = 12            # num_hidden_layers
    n_heads: int = 12             # num_attention_heads
    vocab_size: int = 30522
    type_vocab_size: int = 2
    ffn_hidden_size: int = 3072   # intermediate_size
    hidden_dropout_rate=0.1
    attn_weight_dropout_rate=0.1
    classifier_dropout=None
    max_seq_len: int = 512         # max_position_embeddings
    initializer_range: float = 0.02
    norm_eps: float = 1e-12        # layer_norm_eps
    pad_token_id: int = 0
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
            vocab_size=config["vocab_size"],
            type_vocab_size=config["type_vocab_size"],
            ffn_hidden_size=config["intermediate_size"],
            hidden_dropout_rate=config["hidden_dropout_prob"],
            attn_weight_dropout_rate=config["attention_probs_dropout_prob"],
            classifier_dropout=config["classifier_dropout"],
            norm_eps=config["layer_norm_eps"],
            max_seq_len=config["max_position_embeddings"],
            pad_token_id=config["pad_token_id"]
        )
        for k, v in kwargs.items():
            setattr(args, k, v)
        return args


class Embeddings(nnx.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, args: ModelArgs, rngs: nnx.Rngs = nnx.Rngs(params=0)):
        self.args = args
        embedding = partial(
            nnx.Embed,
            embedding_init=jax.nn.initializers.normal(stddev=args.initializer_range),
            param_dtype=args.param_dtype,
            dtype=args.dtype,
            rngs=rngs,
        )
        self.word_embeddings = embedding(args.vocab_size, args.dim)
        self.position_embeddings = embedding(args.max_seq_len, args.dim)
        self.token_type_embeddings = embedding(args.vocab_size, args.dim)

        self.norm = nnx.LayerNorm(
            num_features=args.dim,
            epsilon=args.norm_eps,
            param_dtype=args.param_dtype,
            dtype=args.dtype,
            rngs=rngs
        )
        self.dropout = nnx.Dropout(rate=args.hidden_dropout_rate)

    def __call__(self, tokens, token_types, position_ids, deterministic: bool = True):
        inputs_embeds = self.word_embeddings(tokens.astype("i4"))
        position_embeds = self.position_embeddings(position_ids.astype("i4"))
        token_type_embeddings = self.token_type_embeddings(token_types.astype("i4"))

        x = inputs_embeds + token_type_embeddings + position_embeds

        x = self.norm(x)
        x = self.dropout(x, deterministic=deterministic)
        return x


# def attention_weights_dropout(attn_weights: jax.Array, deterministic: bool, dropout_rate: float, broadcast_dropout: bool, dtype: jnp.dtype):
#     # from flax.linen.attention.dot_product_attention_weights()
#     # https://github.com/google/flax/blob/main/flax/linen/attention.py#L136-L146
#     if not deterministic and dropout_rate > 0.0:
#         keep_prob = 1.0 - dropout_rate
#         if broadcast_dropout:
#             # dropout is broadcast across the batch + head dimensions
#             dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
#             keep = jax.random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore
#         else:
#             keep = jax.random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)  # type: ignore
#         multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
#         attn_weights = attn_weights * multiplier
#     return attn_weights


class Attention(nnx.Module):
    """
    Multi-head attention module.

    Args:
        args (ModelArgs): Model configuration parameters.

    Attributes:
        n_heads (int): Number of heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.
    """

    def __init__(self, args: ModelArgs, rngs: nnx.Rngs):
        self.args = args
        self.n_heads = args.n_heads
        self.head_dim = self.args.dim // self.n_heads

        dense = partial(
            nnx.Linear,
            args.dim, args.dim,
            dtype=self.args.dtype,
            param_dtype=self.args.param_dtype,
            kernel_init=jax.nn.initializers.normal(args.initializer_range),
            rngs=rngs,
        )
        self.wq = dense()
        self.wk = dense()
        self.wv = dense()
        self.wo = dense()
        self.attn_weight_dropout = nnx.Dropout(
            args.attn_weight_dropout_rate,
            broadcast_dims=(0, 1),   # batch and head dims
            rngs=rngs,
        )
        self.attn_output_dropout = nnx.Dropout(
            args.hidden_dropout_rate,
            rngs=rngs,
        )
        self.norm = nnx.LayerNorm(
            num_features=args.dim,
            epsilon=args.norm_eps,
            param_dtype=args.param_dtype,
            dtype=args.dtype,
            rngs=rngs
        )

    def __call__(
        self,
        x: jax.Array,
        mask: jax.Array | None,
        deterministic: bool = True,
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
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.reshape(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.reshape(bsz, seq_len, self.n_heads, self.head_dim)
        xv = xv.reshape(bsz, seq_len, self.n_heads, self.head_dim)

        xq = jnp.moveaxis(xq, 1, 2)  # (bs, n_heads, seqlen, head_dim)
        xk = jnp.moveaxis(xk, 1, 2)
        xv = jnp.moveaxis(xv, 1, 2)


        scores = jnp.matmul(xq, jnp.moveaxis(xk, 2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            # so far we used mask with 1s to mean "attend" and 0s to mean "ignore"
            # to apply it to scores, we convert it to 0s and -inf accordingly
            imask = jnp.where(mask == 0, -jnp.inf, 0)
            scores = scores + imask  # (bs, n_heads, q_len, kv_len)
        scores = nnx.softmax(scores.astype("float32"), axis=-1).astype(xq.dtype)

        scores = self.attn_weight_dropout(scores, deterministic=deterministic)

        output = jnp.einsum("...hqk,...hkd->...qhd", scores, xv) # (bs, q_len, n_heads, head_dim)
        output = output.reshape(output.shape[:2] + (self.args.dim,)) # (bs, q_len, dim)

        # post-attention processing
        output = self.wo(output)
        output = self.attn_output_dropout(output, deterministic=deterministic)
        output = self.norm(output + x)

        return output



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
        self.w2 = linear(hidden_dim, dim)  # down projection

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

    from fabrique.loading import from_pretrained
    from fabrique.models.phi.load_rules import RULES

    model_id = "google-bert/bert-base-uncased"
    model_args = {"max_seq_len": 512, "max_batch_size": 1}

    args = ModelArgs()
    self = Attention(args, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (1, 5, 768))
    mask = jnp.ones((x.shape[0], x.shape[1]))
    self(x, mask)


    from transformers.models.bert.modeling_flax_bert import FlaxBertAttention
    from transformers.models.bert.configuration_bert import BertConfig

    hf_attn = FlaxBertAttention(BertConfig())
    variables = hf_attn.init(jax.random.key(0), x, mask, None)
    variables["params"]["self"]["query"]["kernel"] = self.wq.kernel.value
    variables["params"]["self"]["query"]["bias"] = self.wq.bias.value
    variables["params"]["self"]["key"]["kernel"] = self.wk.kernel.value
    variables["params"]["self"]["key"]["bias"] = self.wk.bias.value
    variables["params"]["self"]["value"]["kernel"] = self.wv.kernel.value
    variables["params"]["self"]["value"]["bias"] = self.wv.bias.value
    variables["params"]["output"]["dense"]["kernel"] = self.wo.kernel.value
    variables["params"]["output"]["dense"]["bias"] = self.wo.bias.value
    hf_attn.apply(variables, x, mask, None)



    tokenizer, model, hf_config = load_from_pretrained(
        Tokenizer, ModelArgs, Transformer, RULES, model_id, **model_args
    )
    tokens = tokenizer.encode("Once upon a time").ids
    tokens = jnp.array(tokens).reshape(1, -1)
    out = model(tokens, 0).argmax(axis=-1).reshape(-1)
    tokenizer.decode(out)
