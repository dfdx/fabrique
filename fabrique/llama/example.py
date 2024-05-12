# This is an example of a single call of a transformer (using randomly initialized parameters).
# See generation.py for a an example of generation using this transformer.
# Note: you will need Llama weights, see the official instruction:
# https://github.com/facebookresearch/llama#download

from functools import partial

import jax
import jax.numpy as jnp
import jax.tree_util as tree_util
from tokenizers import Tokenizer

from fabrique.llama.model import ModelArgs, Transformer

# TOKENIZER_PATH = "/data/llama/tokenizer.model"
TOKENIZER_PATH = "/home/devpod/.cache/huggingface/hub/models--microsoft--Phi-3-mini-128k-instruct/snapshots/f10fb29b79f038c78229ab4dcd9234a9666a770f/tokenizer.json"


def main():
    args = ModelArgs(max_batch_size=1, max_seq_len=512)
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    args.vocab_size = tokenizer.get_vocab_size()
    token_ids = tokenizer.encode("frankenstein walks into a bar").ids
    token_ids = jnp.asarray(token_ids).reshape(1, -1)
    rng = jax.random.PRNGKey(925)
    model = Transformer(args)
    variables = model.init(rng, token_ids, 0)
    variables = tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), variables)
    # note: we make start_pos static to make JIT happy (specifically, in jnp.triu),
    # but it leads to re-compilation each new value; I'd be happy to find a better way
    jit_apply = partial(jax.jit, static_argnums=(2,), static_argnames=("mutable",))(
        model.apply
    )
    # cache is updated during the call, so we get both - logits and updated cache values
    logits, _variable_updates = jit_apply(variables, token_ids, 0, mutable=("cache",))
    print(logits)

    # TODO: mask has shape (1, 1, 8, 8), but keys & values after caching have length of 10
    # TODO: migrate to HF cache, understand correct shapes


if __name__ == "__main__" and "__file__" in globals():
    main()
