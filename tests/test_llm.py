import jax
import jax.numpy as jnp
from fabrique import LLM, ChatMessage


def test_inference():
    model_id = "meta-llama/Meta-Llama-3-8B"
    kwargs = {"max_seq_len": 32, "max_batch_size": 1, "dtype": jnp.bfloat16}  # limit cache size
    llm = LLM.from_pretrained(model_id,  **kwargs)
    key = jax.random.key(94)

    result = llm.generate("What is the meaning of life?", new_only=False, max_length=32, prng_key=key)
    assert isinstance(result, str)   # smoke test

    messages = [ChatMessage(role="user", content="What is the meaning of life?")]
    result = llm.generate(messages, new_only=False, max_length=32, prng_key=key)
    assert isinstance(result, ChatMessage)