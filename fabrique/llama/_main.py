import os
from functools import partial

import jax
import jax.numpy as jnp
import jax.tree_util as tree_util
from flax import linen as nn
from safetensors.flax import load, load_file
from tokenizers import Tokenizer

from fabrique.llama.model import ModelArgs, Transformer
from fabrique.llama.loading import RULES as LLAMA_RULES
from fabrique.loading import load_variables

# BASE_DIR = "/home/devpod/.cache/huggingface/hub/models--microsoft--Phi-3-mini-128k-instruct/snapshots/f10fb29b79f038c78229ab4dcd9234a9666a770f/"
MODEL_DIR = "/home/devpod/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/1448453bdb895762499deb4176c1dd83b145fac1/"
# TOKENIZER_PATH = MODEL_DIR + "tokenizer.json"
# CONFIG_PATH = MODEL_DIR + "config.json"



class Llama:

    def __init__(self, model_dir: str, **kwargs):
        config_file = os.path.join(model_dir, "config.json")
        tokenizer_file = os.path.join(model_dir, "tokenizer.json")
        self.tokenizer = Tokenizer.from_file(tokenizer_file)
        args = ModelArgs.from_file(config_file, **kwargs)
        self.model = Transformer(args)
        self.variables = load_variables(LLAMA_RULES, model_dir)


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


def main():


    model.apply(variables, tokens, 0, mutable=("cache",))
