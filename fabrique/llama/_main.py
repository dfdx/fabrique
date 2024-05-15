import os
from functools import partial

import jax
import jax.numpy as jnp
import jax.tree_util as tree_util
from flax import linen as nn
from safetensors.flax import load, load_file
from tokenizers import Tokenizer

from fabrique.llama.loading import RULES as LLAMA_RULES
from fabrique.llama.model import ModelArgs, Transformer
from fabrique.loading import load_params

# BASE_DIR = "/home/devpod/.cache/huggingface/hub/models--microsoft--Phi-3-mini-128k-instruct/snapshots/f10fb29b79f038c78229ab4dcd9234a9666a770f/"
MODEL_DIR = "/home/devpod/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/1448453bdb895762499deb4176c1dd83b145fac1/"
# TOKENIZER_PATH = MODEL_DIR + "tokenizer.json"
# CONFIG_PATH = MODEL_DIR + "config.json"


def main():
    model_dir = MODEL_DIR
    kwargs = {"max_seq_len": 512, "max_batch_size": 1}
    llama = Llama(model_dir, **kwargs)
    tokenizer, model, variables = llama.tokenizer, llama.model, llama.variables

    tokens = tokenizer.encode("Hello, my name is").ids
    tokens = jnp.asarray(tokens).reshape(1, -1)

    rng = jax.random.PRNGKey(925)
    logits, v_upd = model.apply(variables, tokens, 0, mutable=("cache",))

    jit_apply = jax.jit(model.apply, static_argnames=("mutable",))
    logits, v_upd = jit_apply(variables, tokens, 0, mutable=("cache",))

    ids = jnp.argmax(logits, axis=-1)[0]
    tokenizer.decode(ids)


#     {
#   "architectures": [
#     "LlamaForCausalLM"
#   ],
#   "attention_bias": false,
#   "attention_dropout": 0.0,
#   "bos_token_id": 128000,
#   "eos_token_id": 128001,
#   "hidden_act": "silu",
#   "hidden_size": 4096,
#   "initializer_range": 0.02,
#   "intermediate_size": 14336,
#   "max_position_embeddings": 8192,
#   "model_type": "llama",
#   "num_attention_heads": 32,
#   "num_hidden_layers": 32,
#   "num_key_value_heads": 8,
#   "pretraining_tp": 1,
#   "rms_norm_eps": 1e-05,
#   "rope_scaling": null,
#   "rope_theta": 500000.0,
#   "tie_word_embeddings": false,
#   "torch_dtype": "bfloat16",
#   "transformers_version": "4.40.0.dev0",
#   "use_cache": true,
#   "vocab_size": 128256
# }
