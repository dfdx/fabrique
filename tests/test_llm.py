from datetime import datetime
import jax
import jax.numpy as jnp

from fabrique import LLM, ChatMessage


def test_inference():
    # model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    # model_id = "microsoft/Phi-3.5-mini-instruct"
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    kwargs = {
        "dtype": jnp.bfloat16,
        # limit cache size
        "max_seq_len": 32,
        "max_batch_size": 2,
        # "ffn_hidden_size": 1024,  # TODO (before commiting): remove
    }
    llm = LLM.from_pretrained(model_id, **kwargs)
    key = jax.random.key(94)

    prompts = ["What is the meaning of life?", "Write a tanku"]
    chats = [[ChatMessage(role="user", content=prompt)] for prompt in prompts]

    result = llm.generate(prompts[0], max_length=32, prng_key=key)
    assert isinstance(result, str)

    result = llm.generate(prompts, max_length=32, prng_key=key)
    assert isinstance(result, list)

    result = llm.generate(chats[0], max_length=32, prng_key=key)
    assert isinstance(result, ChatMessage)

    result = llm.generate(chats, max_length=32, prng_key=key)
    assert isinstance(result, list)



############################################################################

import torch


def j2t(x, device=torch.device("cpu")):
    # return torch.as_tensor(x.__array__(), device=device)
    return torch.from_dlpack(x)



def main_debug_pt():
    from transformers import AutoModelForCausalLM, LlamaModel

    t_tokens = torch.tensor([[128000,   3923,    374,    279,   7438,    315,   2324,     30]])
    t_m = AutoModelForCausalLM.from_pretrained( "meta-llama/Llama-3.2-1B-Instruct", device_map="auto")
    t_out = t_m(t_tokens.to(t_m.device)).logits

    print("Done.")


# main_debug_pt()

def diff(x, t_x):
    c_x = j2t(x).to(t_x.device)
    return (c_x - t_x).max().item()


def main3():
    from flax import nnx

    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    llm = LLM.from_pretrained(model_id, max_seq_len=32, max_batch_size=1, dtype=jnp.bfloat16)
    prompt = "What is the meaning of life?"
    tokens, padding_mask = llm.tokenizer(prompt)
    padding_mask = padding_mask.astype(bool)
    attn_mask = nnx.make_causal_mask(tokens)

    m = llm.model
    out = m(tokens, 0, padding_mask=padding_mask)


    import torch
    import transformers as tfm
    from transformers import AutoModelForCausalLM, LlamaModel

    t_m = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
    t_tokens, t_padding_mask = j2t(tokens, t_m.device), j2t(padding_mask, t_m.device)
    attention_mask = j2t(attn_mask, t_m.device).to(bool)  # causal mask!!!
    t_out = t_m(t_tokens, attention_mask=t_padding_mask).logits


    tok_emb = m.tok_embeddings(tokens)
    t_tok_emb = t_m.model.embed_tokens(t_tokens)

    # # jax - apply layer 0
    # h0 = m.layers[0](tok_emb, 0)

    # # torch - apply layer 0
    inputs_embeds = t_tok_emb
    hidden_states = inputs_embeds
    past_key_values = tfm.DynamicCache()
    past_seen_tokens = 0
    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
    )
    position_ids = position_ids = cache_position.unsqueeze(0)
    # causal_mask = t_m.model._update_causal_mask(
    #     attention_mask, inputs_embeds, cache_position, past_key_values, False
    # )
    position_embeddings = t_m.model.rotary_emb(inputs_embeds, position_ids)

    # t_h0 = t_m.model.layers[0].forward(
    #     inputs_embeds,
    #     attention_mask=causal_mask,
    #     position_ids=position_ids,
    #     past_key_value=past_key_values,
    #     output_attentions=False,
    #     use_cache=False,
    #     cache_position=cache_position,
    #     position_embeddings=position_embeddings,
    # )[0]


    # TODO
    # create an utility to check divergence when applying 2 equivalent layers to a sample input?


    # DIVERGENCE!
    # explanation of (0.0078 == 2 ** -7) difference:
    # https://chatgpt.com/c/680abe75-fb28-800a-b61f-e458a4421a12
    x = jax.random.normal(jax.random.key(0), (1, 8, 2048), dtype=jnp.bfloat16)
    hidden_states = j2t(x, t_m.device)
    diff(m.layers[0].attention.wq(x), t_m.model.layers[0].self_attn.q_proj(hidden_states))


    # NOTE: setting precision (in model code!) to "highest" fixes the divergence
    x = jax.random.normal(jax.random.key(0), (1, 8, 2048), dtype=jnp.bfloat16)
    hidden_states = j2t(x, t_m.device)
    diff(m.layers[0].attention.wk(x), t_m.model.layers[0].self_attn.k_proj(hidden_states))

    x = jax.random.normal(jax.random.key(0), (1, 8, 2048), dtype=jnp.bfloat16)
    hidden_states = j2t(x, t_m.device)
    j_r = m.layers[0].attention(x, 0)
    t_r = t_m.model.layers[0].self_attn(hidden_states, position_embeddings, attention_mask)[0]
    diff(j_r, t_r)

    j_r = m.layers[0](x, 0)
    t_r = t_m.model.layers[0](hidden_states, attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
    diff(j_r, t_r)


    # TODO: make rope_theta configurable for all models
    # setting precision="highest" fixed divergence in xk and xv, but introduced in xq
    # perhaps we want to make precision configurable, but in this case none of these
    # isses seems to be the root cause.

    # TODO: next hypothesis - errors due to precision accumulate over layers
    # => check errors after layer 1, layer 2, etc


    # jax - attention
    from fabrique.models.common.embeddings import apply_rotary_pos_emb
    from fabrique.models.common.utils import padding_to_attention_mask
    from fabrique.models.common.cache import concatenate_to_cache

    x = jax.random.normal(jax.random.key(0), (1, 8, 2048), dtype=jnp.bfloat16)
    self = m.layers[0].attention
    j_self = self
    start_pos = 0

    bsz, seq_len, _ = x.shape
    q_len = seq_len
    kv_len = self.args.max_seq_len if self.args.use_cache else seq_len

    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

    xq = xq.reshape(bsz, seq_len, self.n_heads, self.head_dim)
    xk = xk.reshape(bsz, seq_len, self.n_kv_heads, self.head_dim)
    xv = xv.reshape(bsz, seq_len, self.n_kv_heads, self.head_dim)

    xq, xk = apply_rotary_pos_emb(xq, xk, self.sincos.value, start_pos)

    # apply masks. note: masks have shape (bsz, q_len, kv_len)
    # kv_len depends on the use of cache - see its definition above
    mask = jax.lax.dynamic_slice(
        self.full_causal_mask.value, (0, start_pos, 0), (1, q_len, kv_len)
    )
    mask = jnp.broadcast_to(mask, (bsz, *mask.shape[1:]))
    if padding_mask is not None:
        pad_attn_mask = padding_to_attention_mask(padding_mask, shape=mask.shape)
        mask = nnx.combine_masks(mask, pad_attn_mask).astype(bool)  # type: ignore

    if self.args.use_cache:
        # shape of kv after concatenating to the cache is
        # [bs, max_seq_len, n_heads, head_dim]
        xk, xv, mask = concatenate_to_cache(
            self.cache_k, self.cache_v, xk, xv, xq, mask, start_pos
        )

    output = jax.nn.dot_product_attention(xq, xk, xv, mask=mask[:, None, :, :])
    output = output.reshape(output.shape[:2] + (self.args.dim,))
    output = self.wo(output)

    # torch - attention
    hidden_states = j2t(x).to(t_m.device)
    self = t_m.model.layers[0].self_attn
    t_self = self

    past_key_value = past_key_values

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = tfm.models.llama.modeling_llama.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface = tfm.models.llama.modeling_llama.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask.to(bool),
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        # **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)





    # jax - layer
    x = tok_emb
    h_norm1 = m.layers[0].attention_norm(x)
    h_attn1 = m.layers[0].attention(
        h_norm1,
        0,
        padding_mask=padding_mask.astype(bool),
    )
    h = x + h_attn1
    out = h + m.layers[0].feed_forward(m.layers[0].ffn_norm(h))


    # torch - layer
    inputs_embeds = t_tok_emb
    hidden_states = inputs_embeds
    residual = hidden_states

    layer = t_m.model.layers[0]
    hidden_states = layer.input_layernorm(hidden_states)   # NOTE: hidden states differ by 1.2e-07

    # Self Attention
    hidden_states, self_attn_weights = layer.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_values,
        output_attentions=False,
        use_cache=False,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        # **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = residual + hidden_states



def main2():
    kwargs = {
        'dim': 4096,
        'n_layers': 32,
        'n_heads': 32,
        'n_kv_heads': 8,
        'vocab_size': 128256,
        'multiple_of': 256,
        'ffn_hidden_size': 14336,
        'ffn_dim_multiplier': None,
        'norm_eps': 1e-05,
        'max_batch_size': 2,
        'max_seq_len': 32,
        'dtype': jax.numpy.bfloat16,
        'param_dtype': jax.numpy.bfloat16,
        'use_cache': False
    }
    from equilibrium.utils.inspection import print_size
    from fabrique.models.llama.modeling import Transformer, ModelArgs
    m = Transformer(ModelArgs(**kwargs))




def main():
    from flax import nnx
    # model_id = "microsoft/Phi-3.5-mini-instruct"
    model_id = "microsoft/Phi-4-mini-instruct"
    kwargs = {
        "dtype": jnp.bfloat16,
        # limit cache size
        "max_seq_len": 32,
        "max_batch_size": 2,
    }
    llm = LLM.from_pretrained(model_id, **kwargs)
    key = jax.random.key(94)

    prompts = ["What is the meaning of life?", "Write a tanku"]
    prompt_tokens, mask = llm.tokenizer(prompts, padding_length=16)

    # jax.profiler.save_device_memory_profile("memory.prof")

    model = llm.model

    pad_token_id = llm.tokenizer.special_tokens.pad_id
    eos_token_id = llm.tokenizer.special_tokens.eos_id
    max_length: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    prng_key: jax.Array = jax.random.key(0)
    debug = False

    jax.profiler.start_trace(f"/tmp/tensorboard/model_call")
    model(tokens, 0).block_until_ready()


    nnx.jit(type(model).__call__)(model, tokens, 0)
    jax.profiler.stop_trace()

    # pip install tensorflow tensorboard-plugin-profile



    # wget https://go.dev/dl/go1.24.2.linux-arm64.tar.gz
    # sudo rm -rf /usr/local/go && sudo tar -C /usr/local -xzf go1.24.2.linux-arm64.tar.gz
    # export PATH=$PATH:/usr/local/go/bin
    # go install github.com/google/pprof@latest
    # sudo apt install -y graphviz
    # /home/devpod/go/bin/pprof --web memory.prof