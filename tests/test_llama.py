from functools import partial

import jax
import jax.numpy as jnp
import jax.tree_util as tree_util
from flax import linen as nn
from tokenizers import Tokenizer

from fabrique.llama import Llama
from fabrique.llama.model import ModelArgs, Transformer

MODEL_DIR = "/home/devpod/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/1448453bdb895762499deb4176c1dd83b145fac1/"
TOKENIZER_PATH = "/home/devpod/.cache/huggingface/hub/models--microsoft--Phi-3-mini-128k-instruct/snapshots/f10fb29b79f038c78229ab4dcd9234a9666a770f/tokenizer.json"


def test_jit_and_cache():
    args = ModelArgs(max_batch_size=1, max_seq_len=512)
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    args.vocab_size = tokenizer.get_vocab_size()
    tokens = tokenizer.encode("frankenstein walks into a bar").ids
    tokens = jnp.asarray(tokens).reshape(1, -1)
    rng = jax.random.PRNGKey(925)
    model = Transformer(args)
    variables = model.init(rng, tokens, 0)

    model = model.bind(variables, mutable=("cache",))

    jit_apply = jax.jit(model.apply, static_argnames=("mutable",))
    logits, v_upd = jit_apply(variables, tokens, 0, mutable=("cache",))
    assert isinstance(logits, jax.Array)
    assert isinstance(v_upd, dict)


def test_generate():
    model_dir = MODEL_DIR
    kwargs = {"max_seq_len": 512, "max_batch_size": 1}
    llama = Llama(model_dir, **kwargs)
    prompt = "I will tell you a story about"
    full = llama.generate(prompt)
    assert isinstance(full, str)
