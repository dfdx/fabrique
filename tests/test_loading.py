import jax
import jax.numpy as jnp
from tokenizers import Tokenizer

from fabrique.llama.loading import RULES as LLAMA_RULES
from fabrique.llama.model import ModelArgs, Transformer
from fabrique.loading import load_params

# TODO: don't require the model to be on the disk
MODEL_DIR = "/home/devpod/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/1448453bdb895762499deb4176c1dd83b145fac1/"
TOKENIZER_PATH = MODEL_DIR + "tokenizer.json"
CONFIG_PATH = MODEL_DIR + "config.json"


def test_loading():
    model_dir = MODEL_DIR

    args = ModelArgs.from_file(CONFIG_PATH, max_batch_size=1, max_seq_len=512)
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    args.vocab_size = tokenizer.get_vocab_size()
    tokens = tokenizer.encode("frankenstein walks into a bar").ids
    tokens = jnp.asarray(tokens).reshape(1, -1)
    rng = jax.random.PRNGKey(925)
    model = Transformer(args)
    variables = model.init(rng, tokens, 0)
    ref = variables["params"]

    params = load_params(LLAMA_RULES, model_dir)
    eq = jax.tree.map(lambda p, r: p.shape == r.shape, params, ref)
    assert jax.tree.all(eq)
