
import jax
import jax.numpy as jnp
import torch
from transformers.models.llama import modeling_llama as pt
from fabrique.llama.model import (
    # precompute_freqs_cis,
    # view_as_complex,
    # view_as_real,
    create_sinusoidal_positions,
    apply_rotary_pos_emb,
    repeat_kv,
    ModelArgs,
    RMSNorm,
    Attention,
    FeedForward,
    TransformerBlock,
    Transformer,
)
from tests.torch_utils import pt2jax, jax2pt, fill_pytorch


# reduces memory per layer from 2.2Gb to 0.8Gb
MODEL_ARGS = ModelArgs(max_batch_size=1, max_seq_len=512)


# args = ModelArgs(
#             dim=config["hidden_size"],
#             n_layers=config["num_hidden_layers"],
#             n_heads=config["num_attention_heads"],
#             n_kv_heads=config["num_key_value_heads"],
#             vocab_size=config["vocab_size"],
#             # multiple_of=
#             # ffn_dim_multiplier=
#             norm_eps=config["rms_norm_eps"],
#             # max_batch_size=
#             max_seq_len=config["max_position_embeddings"],
#         )


def args2config(args: ModelArgs):
    from transformers.models.llama.modeling_llama import LlamaConfig
    return LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.dim,
        intermediate_size=args.ffn_hidden_size,
        num_hidden_layers=args.n_layers,
        num_attention_heads=args.n_heads,
        num_key_value_heads=args.n_kv_heads,
        # hidden_act="silu",
        max_position_embeddings=args.max_seq_len,
        # initializer_range=0.02,
        rms_norm_eps=args.norm_eps,
        use_cache=args.use_cache,
        # pad_token_id=None,
        # bos_token_id=1,
        # eos_token_id=2,
        # pretraining_tp=1,
        # tie_word_embeddings=False,
        # rope_theta=10000.0,
        # rope_scaling=None,
        # attention_bias=False,
        # attention_dropout=0.0,
        # mlp_bias=False,
    )


# def compare_with_pytorch(module, pt_module, rng, *args):
#     # jax
#     variables = module.init(rng, *args)
#     out = module.apply(variables, *args)
#     # pytorch
#     pt_args = map(jax2pt, args)
#     pt_out = pt_module(*pt_args)
#     assert jnp.all(out == pt2jax(pt_out))


# def test_rmsnorm():
#     from gendo.llama.model_pt import RMSNorm as PtRMSNorm

#     batch_size, dim = (3, 4)
#     rng = jax.random.PRNGKey(925)
#     x = jax.random.normal(rng, (batch_size, dim))
#     compare_with_pytorch(RMSNorm(dim), PtRMSNorm(dim), rng, x)


# def test_precompute_freqs_cis():
#     from transformers.models.llama.modeling_llama import precompute_freqs_cis as pt_precompute_freqs_cis

#     res = precompute_freqs_cis(32, 8)
#     pt_res = pt_precompute_freqs_cis(32, 8)
#     assert jnp.allclose(res, pt2jax(pt_res))



def test_apply_rotary_embeddings():
    args = MODEL_ARGS
    rng = jax.random.PRNGKey(71)
    rng_q, rng_k = jax.random.split(rng, 2)

    batch_dim, seq_len, n_heads, head_dim = 1, 5, args.n_heads, args.dim // args.n_heads
    xq = jax.random.normal(rng_q, (batch_dim, seq_len, n_heads, head_dim))
    xk = jax.random.normal(rng_k, (batch_dim, seq_len, n_heads, head_dim))

    # jax
    sincos = create_sinusoidal_positions(args.max_seq_len, args.dim // args.n_heads)
    out = apply_rotary_pos_emb(xq, xk, sincos, start_pos=0)

    # torch
    pt_rot_emb = pt.LlamaRotaryEmbedding(args.dim // args.n_heads, args.max_seq_len)
    pt_cos, pt_sin = pt_rot_emb(jax2pt(xq), torch.arange(seq_len).reshape(1, -1))
    # note: for torch, we move dimensions
    # (bs, seq_len, n_heads, head_dim) -> (bs, n_heads, seq_len, head_dim)
    pt_xq, pt_xk = jax2pt(xq).transpose(1, 2), jax2pt(xk).transpose(1, 2)
    pt_out = pt.apply_rotary_pos_emb(pt_xq, pt_xk, pt_cos, pt_sin)

    # test that sin and cos are calculated the same way
    assert jnp.allclose(pt2jax(pt_cos)[0, :, :], sincos[:seq_len, head_dim:])
    assert jnp.allclose(pt2jax(pt_sin)[0, :, :], sincos[:seq_len, :head_dim])
    # test that values are the same after RoPE
    # note: we move dimensions back
    assert jnp.allclose(out[0], pt2jax(pt_out[0].transpose(1, 2)), atol=1e-5)

    # test with start_pos > 0
    start_pos = 3
    out = apply_rotary_pos_emb(xq, xk, sincos, start_pos=start_pos)
    pt_cos, pt_sin = pt_rot_emb(jax2pt(xq), torch.arange(start_pos, start_pos + seq_len).reshape(1, -1))
    pt_xq, pt_xk = jax2pt(xq).transpose(1, 2), jax2pt(xk).transpose(1, 2)
    pt_out = pt.apply_rotary_pos_emb(pt_xq, pt_xk, pt_cos, pt_sin)
    assert jnp.allclose(out[0], pt2jax(pt_out[0].transpose(1, 2)), atol=1e-5)



def test_repeat_kv():
    rng = jax.random.PRNGKey(134)
    x = jax.random.normal(rng, (5, 4, 3, 2))
    out = repeat_kv(x, 6)
    pt_x = jax2pt(x).transpose(1, 2)
    pt_out = pt.repeat_kv(pt_x, 6)
    assert jnp.allclose(out, pt2jax(pt_out.transpose(1, 2)))


def test_attention():
    args = MODEL_ARGS
    bsz, seqlen, dim = (1, 5, args.dim)
    rng = jax.random.PRNGKey(925)
    x = jax.random.normal(rng, (bsz, seqlen, dim))

    sincos = create_sinusoidal_positions(args.max_seq_len, args.dim // args.n_heads)
    attn = Attention(args)
    variables = attn.init(rng, x, 0, sincos, None)
    attn = attn.bind(variables)
    out, _v_upd = attn.apply(variables, x, 0, sincos, None, mutable=["cache"])

    pt_config = args2config(args)
    pt_attn = pt.LlamaAttention(pt_config)
    params = variables["params"]
    pt_attn.q_proj.weight.data = jax2pt(params["wq"]["kernel"].T)
    pt_attn.k_proj.weight.data = jax2pt(params["wk"]["kernel"].T)
    pt_attn.v_proj.weight.data = jax2pt(params["wv"]["kernel"].T)
    pt_attn.o_proj.weight.data = jax2pt(params["wo"]["kernel"].T)

    pt_x = jax2pt(x)
    pt_out = pt_attn(pt_x, position_ids=torch.arange(pt_x.shape[1]).unsqueeze(0))[0]
    assert jnp.allclose(pt2jax(pt_out), out, atol=5e-2)


def test_feedforward():
    from gendo.llama.model_pt import FeedForward as PtFeedForward

    init_pseudo_distributed()

    args = MODEL_ARGS
    bsz, seqlen, dim = (1, 5, args.dim)
    rng = jax.random.PRNGKey(925)
    x = jax.random.normal(rng, (bsz, seqlen, dim))

    # freqs_cis = precompute_freqs_cis(128, seqlen)
    ff = FeedForward(args.dim, args.dim // 2, args.multiple_of, args.ffn_dim_multiplier)
    variables = ff.init(rng, x)
    out = ff.apply(variables, x)

    pt_ff = PtFeedForward(
        args.dim, args.dim // 2, args.multiple_of, args.ffn_dim_multiplier
    )
    params = variables["params"]
    pt_ff.w1.weight.data = jax2pt(params["w1"]["kernel"].T)
    pt_ff.w2.weight.data = jax2pt(params["w2"]["kernel"].T)
    pt_ff.w3.weight.data = jax2pt(params["w3"]["kernel"].T)

    pt_x = jax2pt(x)
    pt_out = pt_ff(pt_x)
    assert jnp.allclose(pt2jax(pt_out), out, atol=1e-2)


def test_transformerblock():
    from gendo.llama.model_pt import TransformerBlock as PtTransformerBlock

    init_pseudo_distributed()

    args = MODEL_ARGS
    bsz, seqlen, dim = (1, 5, args.dim)
    rng = jax.random.PRNGKey(925)
    x = jax.random.normal(rng, (bsz, seqlen, dim))
    freqs_cis = precompute_freqs_cis(128, seqlen)
    block = TransformerBlock(0, args)
    variables = block.init(rng, x, 0, freqs_cis, None)
    out, variable_updates = block.apply(variables, x, 0, freqs_cis, None, mutable=["cache"])

    pt_block = PtTransformerBlock(0, args)
    fill_pytorch(pt_block, variables["params"])

    pt_x = jax2pt(x)
    pt_freqs_cis = jax2pt(freqs_cis)
    pt_out = pt_block(pt_x, 0, pt_freqs_cis, None)
    assert jnp.allclose(pt2jax(pt_out), out, atol=1e-2)
    assert jnp.allclose(
        pt2jax(pt_block.attention.cache_k), variable_updates["cache"]["attention"]["cache_k"], atol=1e-2
    )
    assert jnp.allclose(
        pt2jax(pt_block.attention.cache_v), variable_updates["cache"]["attention"]["cache_v"], atol=1e-2
    )


def test_transformer():
    from gendo.llama.tokenizer import Tokenizer
    from gendo.llama.model_pt import Transformer as PtTransformer

    init_pseudo_distributed()

    args = ModelArgs(max_batch_size=1, max_seq_len=512)
    tokenizer = Tokenizer(model_path="/data/llama/tokenizer.model")
    args.vocab_size = tokenizer.n_words
    tokens = tokenizer.encode("frankenstein walks into a bar", False, False)
    tokens = jnp.asarray(tokens).reshape(1, -1)
    rng = jax.random.PRNGKey(925)
    model = Transformer(args)
    variables = model.init(rng, tokens, 0)
    model = model.bind(variables)
    out, _variable_updates = model.apply(variables, tokens, 0, mutable=["cache"])

    pt_tokens = jax2pt(tokens)
    pt_model = PtTransformer(args)
    fill_pytorch(pt_model, variables["params"])
    pt_out = pt_model(pt_tokens, 0)
    assert jnp.allclose(pt2jax(pt_out), out, atol=1e-2)
