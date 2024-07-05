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
    segment_vocab_size: int = 2
    ffn_hidden_size: int = 3072   # intermediate_size
    hidden_dropout_rate: float = 0.1
    attn_weight_dropout_rate: float =0.1
    classifier_dropout: float | None = None
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
            segment_vocab_size=config["type_vocab_size"],
            ffn_hidden_size=config["intermediate_size"],
            hidden_dropout_rate=config["hidden_dropout_prob"],
            attn_weight_dropout_rate=config["attention_probs_dropout_prob"],
            classifier_dropout=config.get("classifier_dropout"),
            norm_eps=config["layer_norm_eps"],
            max_seq_len=config["max_position_embeddings"],
            pad_token_id=config["pad_token_id"]
        )
        for k, v in kwargs.items():
            setattr(args, k, v)
        return args


class Embeddings(nnx.Module):
    """Construct the embeddings from word, position and segment embeddings."""

    def __init__(self, args: ModelArgs, rngs: nnx.Rngs = nnx.Rngs(params=0)):
        self.args = args
        embedding = partial(
            nnx.Embed,
            embedding_init=jax.nn.initializers.normal(stddev=args.initializer_range),
            param_dtype=args.param_dtype,
            dtype=args.dtype,
            rngs=rngs,
        )
        self.position_embeddings = embedding(args.max_seq_len, args.dim)
        self.token_embeddings = embedding(args.vocab_size, args.dim)
        self.segment_embeddings = embedding(args.segment_vocab_size, args.dim)

        self.norm = nnx.LayerNorm(
            num_features=args.dim,
            epsilon=args.norm_eps,
            param_dtype=args.param_dtype,
            dtype=args.dtype,
            rngs=rngs
        )
        self.dropout = nnx.Dropout(rate=args.hidden_dropout_rate)

    def __call__(self, tokens, segment_tokens, position_ids, deterministic: bool = True):
        token_embeddings = self.token_embeddings(tokens.astype("i4"))
        position_embeds = self.position_embeddings(position_ids.astype("i4"))
        segment_embeddings = self.segment_embeddings(segment_tokens.astype("i4"))

        x = token_embeddings + segment_embeddings + position_embeds

        x = self.norm(x)
        x = self.dropout(x, deterministic=deterministic)
        return x


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
        # TODO: update docstring
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
        args: ModelArgs,
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
        # TODO: update docstring
        linear = partial(
            nnx.Linear,
            kernel_init=jax.nn.initializers.normal(args.initializer_range),
            param_dtype=args.param_dtype,
            dtype=args.dtype,
            rngs=rngs
        )
        self.w1 = linear(args.dim, args.ffn_hidden_size)
        self.w2 = linear(args.ffn_hidden_size, args.dim)
        self.norm = nnx.LayerNorm(
            num_features=args.dim,
            epsilon=args.norm_eps,
            param_dtype=args.param_dtype,
            dtype=args.dtype,
            rngs=rngs
        )
        self.dropout = nnx.Dropout(rate=args.hidden_dropout_rate)


    def __call__(self, x, deterministic=True):
        out = self.w1(x)
        out = nnx.gelu(out, approximate=False)
        out = self.w2(out)
        out = self.dropout(out, deterministic=deterministic)
        out = self.norm(out + x)
        return out


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
        # TODO: update docstring
        self.args = args
        self.attention = Attention(args, rngs=rngs)
        self.feed_forward = FeedForward(args, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array | None, deterministic: bool = True):
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
        # TODO: update docstring
        h = self.attention(x, mask, deterministic=deterministic)
        out = self.feed_forward(h)
        return out


class Pooler(nnx.Module):

    def __init__(self, args: ModelArgs, rngs: nnx.Rngs):
        self.w = nnx.Linear(
            args.dim,
            args.dim,
            kernel_init=jax.nn.initializers.normal(args.initializer_range),
            param_dtype=args.param_dtype,
            dtype=args.dtype,
            rngs=rngs,
        )

    def __call__(self, x):
        out = x[:, 0]
        out = self.w(out)
        return nnx.tanh(out)


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
        # TODO: update docstring
        self.args = args
        self.embeddings = Embeddings(args, rngs=rngs)
        self.layers = [TransformerBlock(args, rngs=rngs) for _ in range(args.n_layers)]
        self.pooler = Pooler(args, rngs=rngs)

    def __call__(
            self,
            tokens: jax.Array,
            mask: jax.Array | None = None,
            segments: jax.Array | None = None,
            pool: bool = False,
            deterministic = True,
    ):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (jax.Array): Input token indices.
            start_pos (int): Starting position for attention caching.

        Returns:
            jax.Array: Output logits after applying the Transformer model.
        """
        # TODO: update docstring
        if mask is None:
            mask = jnp.ones_like(tokens)
        if segments is None:
            segments = jnp.zeros_like(tokens)
        position_ids = jnp.broadcast_to(jnp.arange(tokens.shape[-1]), tokens.shape)
        h = self.embeddings(tokens, segments, position_ids, deterministic=deterministic)
        for i, layer in enumerate(self.layers):
            h = layer(h, mask, deterministic=deterministic)
        if pool:
            h = self.pooler(h)
        output = h.astype("float32")
        return output


###########################################################


def test_bert():
    from fabrique.loading import from_pretrained

    model_id = "google-bert/bert-base-uncased"

    tokenizer, model, _hf_config = from_pretrained(model_id)
    tokens = tokenizer.encode("Once upon a time").ids
    tokens = jnp.array(tokens).reshape(1, -1)
    out = model(tokens)

    # from huggingface model
    expected_slice = jnp.asarray(
        [ 0.06543218, -0.35718504, -0.3604128 ,  0.0042748 ,  0.23516399,
        -0.09302475, -0.2931913 ,  0.6673709 , -0.02174259, -0.38075727]
    )
    assert jnp.allclose(out[0, 0, :10], expected_slice)
    assert out.sum() == -80.01314
    assert out.var() == 0.3342754

    mask = jnp.ones_like(tokens)
    mask = mask.at[0, -3:].set(0)
    segment_tokens = jnp.zeros_like(tokens)
    out = model(tokens, mask, segment_tokens)



def main():
    import logging, fabrique.loading
    logging.getLogger(fabrique.loading.__name__).setLevel(logging.WARNING)

    from tokenizers import Tokenizer

    from fabrique.loading import from_pretrained

    model_id = "google-bert/bert-base-uncased"
    repo_id = model_id
    model_args = {"max_seq_len": 512}

    tokenizer, model, hf_config = from_pretrained(model_id, **model_args);


    args = ModelArgs(**model_args)
    model = Transformer(args)





    self = Attention(args, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (1, 5, 768))
    mask = jnp.ones((x.shape[0], x.shape[1]))
    self(x, mask)


    from transformers.models.bert.configuration_bert import BertConfig
    config = BertConfig()


    from transformers.models.bert.modeling_flax_bert import FlaxBertAttention

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


    from transformers.models.bert.modeling_flax_bert import FlaxBertIntermediate, FlaxBertOutput

    ff = FeedForward(args)
    ff(x)
    hf_int = FlaxBertIntermediate(config)
    int_variables = hf_int.init(jax.random.key(0), x)
    int_variables["params"]["dense"]["kernel"] = ff.w1.kernel.value
    int_variables["params"]["dense"]["bias"] = ff.w1.bias.value
    hf_res = hf_int.apply(int_variables, x)
    hf_output = FlaxBertOutput(config)
    out_variables = hf_output.init(jax.random.key(0), hf_res, x)
    out_variables["params"]["dense"]["kernel"] = ff.w2.kernel.value
    out_variables["params"]["dense"]["bias"] = ff.w2.bias.value
    hf_res = hf_output.apply(out_variables, hf_res, x)


    from transformers.models.bert.modeling_flax_bert import FlaxBertLayer

    block = TransformerBlock(args, nnx.Rngs(params=0))
    block(x, mask)

    hf_block = FlaxBertLayer(config)
    variables = hf_block.init(jax.random.key(0), x, mask, None)
    variables["params"]["attention"]["self"]["query"]["kernel"] = block.attention.wq.kernel.value
    variables["params"]["attention"]["self"]["query"]["bias"] = block.attention.wq.bias.value
    variables["params"]["attention"]["self"]["key"]["kernel"] = block.attention.wk.kernel.value
    variables["params"]["attention"]["self"]["key"]["bias"] = block.attention.wk.bias.value
    variables["params"]["attention"]["self"]["value"]["kernel"] = block.attention.wv.kernel.value
    variables["params"]["attention"]["self"]["value"]["bias"] = block.attention.wv.bias.value
    variables["params"]["attention"]["output"]["dense"]["kernel"] = block.attention.wo.kernel.value
    variables["params"]["attention"]["output"]["dense"]["bias"] = block.attention.wo.bias.value
    variables["params"]["intermediate"]["dense"]["kernel"] = block.feed_forward.w1.kernel.value
    variables["params"]["intermediate"]["dense"]["bias"] = block.feed_forward.w1.bias.value
    variables["params"]["output"]["dense"]["kernel"] = block.feed_forward.w2.kernel.value
    variables["params"]["output"]["dense"]["bias"] = block.feed_forward.w2.bias.value
    variables["params"]["output"]["LayerNorm"]["scale"] = block.feed_forward.norm.scale.value
    variables["params"]["output"]["LayerNorm"]["bias"] = block.feed_forward.norm.bias.value

    hf_block.apply(variables, x, mask, None)


    from transformers import FlaxAutoModel

    tokenizer, model, hf_config = from_pretrained(model_id, **model_args)
    tokens = tokenizer.encode("Once upon a time").ids
    tokens = jnp.array(tokens).reshape(1, -1)
    out = model(tokens)

    # from huggingface model
    expected_slice = jnp.asarray(
        [ 0.06543218, -0.35718504, -0.3604128 ,  0.0042748 ,  0.23516399,
        -0.09302475, -0.2931913 ,  0.6673709 , -0.02174259, -0.38075727]
    )
    assert jnp.allclose(out[0, 0, :10], expected_slice)
    assert out.sum() == -80.01314
    assert out.var() == 0.3342754

    out_pooled = model(tokens, pool=True)

    hf_model = FlaxAutoModel.from_pretrained(model_id)
    hf_out = hf_model(tokens)

    assert jnp.allclose(out, hf_out.last_hidden_state)
    # assert jnp.allclose(out_pooled, hf_out.pooler_output)

    hf_model(tokens, attention_mask=None, token_type_ids=segment_tokens).last_hidden_state
    model(tokens, mask=None, segments=segment_tokens)



    # my_shortcuts = [{'command': 'IPython:auto_suggest.accept', 'new_keys': ['end']}]
    # %config TerminalInteractiveShell.shortcuts = my_shortcuts