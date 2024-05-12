from functools import partial

import jax
import jax.numpy as jnp
import jax.tree_util as tree_util
from flax import linen as nn
from tokenizers import Tokenizer
from safetensors.flax import load, load_file

from fabrique.llama.model import ModelArgs, Transformer

# TOKENIZER_PATH = "/data/llama/tokenizer.model"
BASE_DIR = "/home/devpod/.cache/huggingface/hub/models--microsoft--Phi-3-mini-128k-instruct/snapshots/f10fb29b79f038c78229ab4dcd9234a9666a770f/"
TOKENIZER_PATH = BASE_DIR + "tokenizer.json"



def main():
    path = BASE_DIR + "model-00001-of-00002.safetensors"
    variables = load_file(path)



def main2():
    args = ModelArgs(max_batch_size=1, max_seq_len=512)
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    args.vocab_size = tokenizer.get_vocab_size()
    tokens = tokenizer.encode("frankenstein walks into a bar").ids
    tokens = jnp.asarray(tokens).reshape(1, -1)
    rng = jax.random.PRNGKey(925)
    model = Transformer(args)
    variables = model.init(rng, tokens, 0)
    # variables["params"] = tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), variables["params"])

    model = model.bind(variables, mutable=("cache",))
    freqs_cis = model.freqs_cis
    start_pos = 2

    with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        jax.jit(model.apply, static_argnames=("mutable",))(
            variables, tokens, 0, mutable=("cache",)
        )

    jit_apply = jax.jit(model.apply, static_argnames=("mutable",))
    jit_apply(variables, tokens, 0, mutable=("cache",))

    model.apply(variables, tokens, 0, mutable=("cache",))
    causal_mask = jax.lax.dynamic_slice(
        self.causal_mask, (0, 0, start_pos, 0), (1, 1, q_len, kv_len)
    )
    mask = causal_mask

    self = model
    _bsz, seqlen = tokens.shape
    x = self.tok_embeddings(tokens)
    # mask = nn.make_causal_mask(jnp.ones((1, seqlen)))   # TODO: untested

    self = model.layers[0].attention

    tokens = tokenizer.encode("hello, he says").ids
    tokens = jnp.asarray(tokens).reshape(1, -1)
    h = model.tok_embeddings(tokens)
    x = h

