from fabrique.models.llm import LLM


def load_and_check(model_id: str, prompt: str, expected: str):
    kwargs = {"max_seq_len": 32, "max_batch_size": 1}
    llm = LLM.from_pretrained(model_id, **kwargs)
    result = llm.generate(prompt)
    assert isinstance(result, str)
    assert result == expected


def test_llama():
    model_id = "meta-llama/Meta-Llama-3-8B"
    prompt = "Once upon a time"
    expected = (
        "Once upon a time, there was a little girl "
        + "who loved to read. She loved to read so much "
        + "that she would read anything she could get her hands"
    )
    load_and_check(model_id, prompt, expected)


def test_phi():
    model_id = "microsoft/Phi-3-mini-128k-instruct"
    prompt = "Once upon a time"
    expected = (
        "Once upon a time, in a small town, there lived a young "
        + "girl named Lily. Lily was a curious and adventurous "
        + "girl who loved expl"
    )
    load_and_check(model_id, prompt, expected)
