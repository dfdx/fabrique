import jax
import jax.numpy as jnp

from fabrique import LLM


def load_and_check(model_id: str, revision: str, prompt: str, expected: str):
    kwargs = {"max_seq_len": 32, "max_batch_size": 1}  # limit cache size
    llm = LLM.from_pretrained(model_id, revision=revision, **kwargs)
    key = jax.random.key(818)
    result = llm.generate(prompt, new_only=False, max_length=32, prng_key=key)
    assert isinstance(result, str)
    assert result == expected


def test_llama():
    model_id = "meta-llama/Meta-Llama-3-8B"
    prompt = "Once upon a time"
    revision = None
    expected = "Once upon a time the British government established a board charged with running the country, the British economy and the welfare of her citizens.\nThis board was duly composed"
    load_and_check(model_id, revision, prompt, expected)


def test_phi():
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    revision = "c1358f8"
    prompt = "Once upon a time"
    expected = "Once upon a time in the magical land of Wordspell, every word had its unique spell, and words were connected with pathways of letters that dan"
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
    atol = 5e-2
    assert jnp.allclose(out[0, 0, :10], expected_slice, atol=atol)
    assert jnp.allclose(out.sum(), -80.01314, atol=atol)
    assert jnp.allclose(out.var(), 0.3342754, atol=atol)

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
    print(out[0, 0, :10] - expected_slice)
    print(jnp.abs(out[0, 0, :10] - expected_slice) < atol)
    assert jnp.allclose(out[0, 0, :10], expected_slice, atol=atol)
    assert jnp.allclose(out.sum(), -50.359936, atol=atol)
    assert jnp.allclose(out.var(), 0.25323534, atol=atol)