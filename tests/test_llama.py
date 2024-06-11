import jax
import jax.numpy as jnp
from tokenizers import Tokenizer

from fabrique.models.llama import Llama
from fabrique.models.llama.model import ModelArgs, Transformer

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
# TODO: get rid of absolute paths
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
    kwargs = {"max_seq_len": 32, "max_batch_size": 1}
    llama = Llama.from_pretrained(MODEL_ID, **kwargs)
    prompt = "I will tell you a story about"
    result = llama.generate(prompt)
    assert isinstance(result, str)
    expected = (
        "I will tell you a story about a man who was a great leader. "
        + "He was a man who was always looking for ways to improve himself "
        + "and his team"
    )
    assert result == expected
