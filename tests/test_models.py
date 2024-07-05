import jax.numpy as jnp

from fabrique.models.llm import LLM


def load_and_check(model_id: str, revision: str, prompt: str, expected: str):
    kwargs = {"max_seq_len": 32, "max_batch_size": 1}
    llm = LLM.from_pretrained(model_id, revision=revision, **kwargs)
    result = llm.generate(prompt)
    assert isinstance(result, str)
    assert result == expected


def test_llama():
    model_id = "meta-llama/Meta-Llama-3-8B"
    prompt = "Once upon a time"
    revision = None
    expected = (
        "Once upon a time, there was a little girl "
        + "who loved to read. She loved to read so much "
        + "that she would read anything she could get her hands"
    )
    load_and_check(model_id, revision, prompt, expected)


def test_phi():
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    revision = "c1358f8"
    prompt = "Once upon a time"
    expected = (
        "Once upon a time, in a small town nestled between "
        + "rolling hills, there lived a young woman named Emily. "
        + "Emily was known for her vibr"
    )
    load_and_check(model_id, revision, prompt, expected)


def test_bert():
    from fabrique.loading import from_pretrained

    model_id = "google-bert/bert-base-uncased"

    tokenizer, model, _hf_config = from_pretrained(model_id)
    tokens = tokenizer.encode("Once upon a time").ids
    tokens = jnp.array(tokens).reshape(1, -1)
    out = model(tokens)

    # from huggingface model
    expected_slice = jnp.asarray(
        [
            0.06543218,
            -0.35718504,
            -0.3604128,
            0.0042748,
            0.23516399,
            -0.09302475,
            -0.2931913,
            0.6673709,
            -0.02174259,
            -0.38075727,
        ]
    )
    assert jnp.allclose(out[0, 0, :10], expected_slice)
    assert out.sum() == -80.01314
    assert out.var() == 0.3342754

    mask = jnp.ones_like(tokens)
    mask = mask.at[0, -3:].set(0)
    segments = jnp.zeros_like(tokens)
    out = model(tokens, mask, segments)

    expected_slice = jnp.asarray(
        [
            -0.2556453,
            0.12366249,
            -0.70183635,
            0.26455373,
            -0.14038257,
            -0.27187574,
            -0.35302785,
            0.5766474,
            -0.55448484,
            -0.573969,
        ]
    )
    assert jnp.allclose(out[0, 0, :10], expected_slice)
    assert out.sum() == -50.359936
    assert out.var() == 0.25323534
