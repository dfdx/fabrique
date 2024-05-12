from functools import partial

import jax
import jax.numpy as jnp
import jax.tree_util as tree_util
from flax import linen as nn
from tokenizers import Tokenizer

from fabrique.llama.model import ModelArgs, Transformer

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
    logits, _var_updates = jit_apply(variables, tokens, 0, mutable=("cache",))
