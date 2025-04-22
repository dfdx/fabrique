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


def j2t(x):
    import torch
    import numpy as np
    return torch.tensor(np.array(x))


def main3():
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    llm = LLM.from_pretrained(model_id, max_seq_len=32, max_batch_size=1)
    prompt = "What is the meaning of life?"
    tokens, mask = llm.tokenizer(prompt)

    m = llm.model
    out = m(tokens, 0)


    from transformers import AutoModelForCausalLM

    t_m = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    t_tokens, t_mask = j2t(tokens), j2t(mask)
    t_out = t_m(t_tokens)


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