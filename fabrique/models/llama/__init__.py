import json
import os

import jax.numpy as jnp
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

from fabrique.generation import greedy
from fabrique.loading import update_model_from_safe
from fabrique.models.llama.loading import RULES
from fabrique.models.llama.model import ModelArgs, Transformer


class Llama:

    def __init__(self, tokenizer, model, hf_config: dict):
        self.tokenizer = tokenizer
        self.model = model
        self.hf_config = hf_config

    @staticmethod
    def from_file(model_dir: str, **model_args):
        config_file = os.path.join(model_dir, "config.json")
        with open(config_file) as fp:
            hf_config = json.load(fp)
        tokenizer_file = os.path.join(model_dir, "tokenizer.json")
        tokenizer = Tokenizer.from_file(tokenizer_file)
        args = ModelArgs.from_file(config_file, **model_args)
        model = Transformer(args)
        update_model_from_safe(model, RULES, model_dir)
        return Llama(tokenizer, model, hf_config)  # type: ignore

    @staticmethod
    def from_pretrained(repo_id: str, **model_args):
        model_dir = snapshot_download(repo_id, repo_type="model")
        return Llama.from_file(model_dir, **model_args)

    def generate(self, prompt: str, seed: int = 0):
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
