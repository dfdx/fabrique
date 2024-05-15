import os
import jax
import jax.numpy as jnp
from tokenizers import Tokenizer
from fabrique.llama.model import ModelArgs, Transformer
from fabrique.llama.loading import RULES
from fabrique.loading import load_params



class Llama:

    def __init__(self, model_dir: str, **kwargs):
        config_file = os.path.join(model_dir, "config.json")
        tokenizer_file = os.path.join(model_dir, "tokenizer.json")
        rng = jax.random.PRNGKey(925)

        self.tokenizer = Tokenizer.from_file(tokenizer_file)
        example_tokens = self.tokenizer.encode("Llama walks into a bar").ids
        example_tokens = jnp.asarray(example_tokens).reshape(1, -1)

        args = ModelArgs.from_file(config_file, **kwargs)
        self.model = Transformer(args)
        self.variables = self.model.init(rng, example_tokens, 0)
        load_params(RULES, model_dir, out=self.variables["params"])



def main():
    MODEL_DIR = "/home/devpod/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/1448453bdb895762499deb4176c1dd83b145fac1/"
    model_dir = MODEL_DIR
    kwargs = {"max_seq_len": 512, "max_batch_size": 1}
    self = Llama.__new__(Llama)