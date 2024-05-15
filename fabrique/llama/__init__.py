import json
import os

import jax
import jax.numpy as jnp
from tokenizers import Tokenizer

from fabrique.generation import greedy
from fabrique.llama.loading import RULES
from fabrique.llama.model import ModelArgs, Transformer
from fabrique.loading import load_params


class Llama:

    def __init__(self, model_dir: str, **kwargs):
        config_file = os.path.join(model_dir, "config.json")
        with open(config_file) as fp:
            self.hf_config = json.load(fp)

        tokenizer_file = os.path.join(model_dir, "tokenizer.json")
        rng = jax.random.PRNGKey(925)

        self.tokenizer = Tokenizer.from_file(tokenizer_file)
        example_tokens = self.tokenizer.encode("Llama walks into a bar").ids
        example_tokens = jnp.asarray(example_tokens).reshape(1, -1)

        args = ModelArgs.from_file(config_file, **kwargs)
        self.model = Transformer(args)
        self.variables = self.model.init(rng, example_tokens, 0)
        load_params(RULES, model_dir, out=self.variables["params"])

    def generate(self, prompt: str, seed: int = 0):
        prompt_tokens = self.tokenizer.encode(prompt).ids
        prompt_tokens = jnp.asarray(prompt_tokens).reshape(1, -1)
        sequences = greedy(
            self.model,
            self.variables,
            prompt_tokens,
            pad_token_id=self.hf_config["eos_token_id"],
            eos_token_id=self.hf_config["eos_token_id"],
            max_length=self.model.args.max_seq_len,
        )
        return self.tokenizer.decode(sequences[0])


def main():
    MODEL_DIR = "/home/devpod/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/1448453bdb895762499deb4176c1dd83b145fac1/"
    model_dir = MODEL_DIR
    kwargs = {"max_seq_len": 512, "max_batch_size": 1}
    self = Llama(model_dir, **kwargs)
    prompt = "I will tell you a story about"
