import jax.numpy as jnp

from fabrique.generation import greedy
from fabrique.loading import from_pretrained


class LLM:

    def __init__(self, tokenizer, model, hf_config: dict):
        self.tokenizer = tokenizer
        self.model = model
        self.hf_config = hf_config

    @staticmethod
    def from_pretrained(repo_id: str, **model_args):
        tokenizer, model, hf_config = from_pretrained(repo_id, **model_args)
        return LLM(tokenizer, model, hf_config)

    def generate(self, prompt: str):
        prompt_tokens = self.tokenizer.encode(prompt).ids
        prompt_tokens = jnp.asarray(prompt_tokens).reshape(1, -1)
        sequences = greedy(
            self.model,
            prompt_tokens,
            pad_token_id=self.hf_config["eos_token_id"],
            eos_token_id=self.hf_config["eos_token_id"],
            max_length=self.model.args.max_seq_len,
        )
        return self.tokenizer.decode(sequences[0])
