import re
from typing import Callable, Dict, List
from functools import partial
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.tree_util as tree_util
from flax import linen as nn
from tokenizers import Tokenizer
from safetensors.flax import load, load_file

from fabrique.llama.model import ModelArgs, Transformer

# BASE_DIR = "/home/devpod/.cache/huggingface/hub/models--microsoft--Phi-3-mini-128k-instruct/snapshots/f10fb29b79f038c78229ab4dcd9234a9666a770f/"
BASE_DIR = "/home/devpod/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/1448453bdb895762499deb4176c1dd83b145fac1/"
TOKENIZER_PATH = BASE_DIR + "tokenizer.json"


@dataclass
class Rule:
    safe_pattern: str
    fab_pattern: str
    converter: Callable | None = None

    @property
    def safe_regexp(self):
        pat = self.safe_pattern
        pat = pat.replace(".", "\.")
        pat = pat.replace("{n}", "(?P<n>\d+)")
        pat = "^" + pat + "$"
        return re.compile(pat)


R = Rule

RULES = [
    R("model.embed_tokens.weight", "tok_embeddings.embedding"),
    R("model.layers.{n}.input_layernorm.weight", "layers_{n}.attention_norm.weight")

]



def maybe_apply_rule(rule: Rule, safe_key: str, safe_val):
    if m := re.match(rule.safe_regexp, safe_key):
        path = rule.fab_pattern.format(**m.groupdict())
        val = rule.converter(safe_val) if rule.converter else safe_val
        return path, val
    else:
        return None


def convert_safetensor(safe_key: str, safe_val):
    for rule in RULES:
        path_val = maybe_apply_rule(rule, safe_key, safe_val)
        if path_val:
            return path_val
    raise ValueError(f"Couldn't find a rule for safetensors key {safe_key} " +
                     f"and value of shape {safe_val.shape}")


def set_nested(nested: Dict, keys: List[str], val):
    dct = nested
    for subkey in keys[:-1]:
        if subkey not in dct:
            dct[subkey] = {}
        dct = dct[subkey]
    dct[keys[-1]] = val
    return nested


def convert_params(flat: Dict[str, jax.Array]):
    params = []
    for safe_key, safe_val in flat.items():
        path, val = convert_safetensor(safe_key, safe_val)
        fab_keys = path.split(".")




def main():
    args = ModelArgs(max_batch_size=1, max_seq_len=512)
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    args.vocab_size = tokenizer.get_vocab_size()
    tokens = tokenizer.encode("frankenstein walks into a bar").ids
    tokens = jnp.asarray(tokens).reshape(1, -1)
    rng = jax.random.PRNGKey(925)
    model = Transformer(args)
    variables = model.init(rng, tokens, 0)
    params = variables["params"]

    # path = BASE_DIR + "model-00001-of-00002.safetensors"
    path = BASE_DIR + "model-00001-of-00004.safetensors"
    flat = load_file(path)

    self = RULES[1]
    rule = RULES[1]
    safe_key = "model.layers.0.input_layernorm.weight"