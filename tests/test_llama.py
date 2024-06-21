from fabrique.models.llama import Llama

MODEL_ID = "meta-llama/Meta-Llama-3-8B"


def test_generate():
    kwargs = {"max_seq_len": 32, "max_batch_size": 1}
    llama = Llama.from_pretrained(MODEL_ID, **kwargs)
    prompt = "Once upon a time"
    result = llama.generate(prompt)
    assert isinstance(result, str)
    expected = (
        "Once upon a time, there was a little girl "
        + "who loved to read. She loved to read so much "
        + "that she would read anything she could get her hands"
    )
    assert result == expected
