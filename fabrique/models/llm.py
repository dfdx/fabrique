import jax
import jax.numpy as jnp
from flax import nnx

from fabrique.generation import sample
from fabrique.loading import from_pretrained


class LLM:

    def __init__(self, tokenizer, model, hf_config: dict):
        self.tokenizer = tokenizer
        self.model = model
        self.hf_config = hf_config
        self.rngs = nnx.Rngs(0)

    @staticmethod
    def from_pretrained(repo_id: str, revision: str | None = None, **model_args):
        tokenizer, model, hf_config = from_pretrained(
            repo_id, revision=revision, **model_args
        )
        return LLM(tokenizer, model, hf_config)

    def generate(
            self,
            prompt: str,
            new_only: bool = True,
            max_length: int = 4096,
            temperature: float = 1.0,
            top_p: float = 1.0,
            top_k: int = 50,
            prng_key: jax.Array | None = None
        ):
        prompt_tokens = self.tokenizer.encode(prompt).ids
        prompt_tokens = jnp.asarray(prompt_tokens).reshape(1, -1)
        sequences = sample(
            self.model,
            prompt_tokens,
            pad_token_id=self.hf_config["eos_token_id"],
            eos_token_id=self.hf_config["eos_token_id"],
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            prng_key=prng_key if prng_key is not None else self.rngs(),
        )
        start = prompt_tokens.shape[1] if new_only else 0
        return self.tokenizer.decode(sequences[0][start:])
