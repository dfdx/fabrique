from functools import partial
import jax
import jax.numpy as jnp
import jax.tree_util as tree_util
from tokenizers import Tokenizer
from fabrique.llama.model import Transformer, ModelArgs

# TOKENIZER_PATH = "/data/llama/tokenizer.model"
TOKENIZER_PATH = "/home/devpod/.cache/huggingface/hub/models--microsoft--Phi-3-mini-128k-instruct/snapshots/f10fb29b79f038c78229ab4dcd9234a9666a770f/tokenizer.json"


def main():
    args = ModelArgs(max_batch_size=1, max_seq_len=512)
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    args.vocab_size = tokenizer.get_vocab_size()
    tokens = tokenizer.encode("frankenstein walks into a bar").ids
    tokens = jnp.asarray(tokens).reshape(1, -1)
    rng = jax.random.PRNGKey(925)
    model = Transformer(args)
    variables = model.init(rng, tokens, 0)
    # variables["params"] = tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), variables["params"])

    # note: we make start_pos static to make JIT happy (specifically, in jnp.triu),
    # but it leads to re-compilation each new value; I'd be happy to find a better way
    # jit_apply = partial(jax.jit, static_argnums=(2,), static_argnames=("mutable",))(model.apply)
    # cache is updated during the call, so we get both - logits and updated cache values
    # logits, _variable_updates = jit_apply(variables, tokens, 0, mutable=("cache",))

    model = model.bind(variables, mutable=("cache",))
    freqs_cis = model.freqs_cis
    start_pos = 2

    jax.jit(model.apply, static_argnames=("mutable",))(variables, tokens, 0, mutable=("cache",))

    self = model
    self = model.layers[0].attention



# In Meta's implementation, mask is padded with zeros for cached values in Transformer.forward().
# Also, start_pos is passed explicitely.
#
# In HF Flax implementation, they take cache length from variables["cache"]["cache_index"] and
# the mask is padded directly in _concatenate_to_cache().


# In HF Flax implementation, they initialize cache_index to 0 and then always increase it,
# and also shift the causal mask to account for the cache [1]. But how do they reset it
# for the new x?
# [1]: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_flax_llama.py#L286


# I think we need to drop cache_index and pass start_pos explicitely, yet shift the mask
# as in the link above.
# Questions:
# 1. How do me manage start_pos from outside?
# 2. Maybe just reset cache_index and keep the current implementation?
#
# Also need to figure out mask format


# TODO: check that mask after _concatenate_to_cache() has correct format (0 vs -inf)

if __name__ == "__main__" and "__file__" in globals():
    main()