import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import jax
import safetensors.flax as st
from flax import nnx
from tqdm import tqdm

from huggingface_hub import snapshot_download

from fabrique.utils import set_nested_attr



@dataclass
class RuleIgnore:
    pass


IGNORE = RuleIgnore()


@dataclass
class ConversionRule:
    safe_pattern: str
    fab_pattern: str | RuleIgnore
    converter: Callable | None = None

    @property
    def safe_regexp(self):
        pat = self.safe_pattern
        pat = pat.replace(".", "\\.")
        pat = pat.replace("{n}", "(?P<n>\\d+)")
        pat = "^" + pat + "$"
        return re.compile(pat)


def maybe_apply_rule(rule: ConversionRule, safe_key: str, safe_val):
    if m := re.match(rule.safe_regexp, safe_key):
        if isinstance(rule.fab_pattern, RuleIgnore):
            return "", IGNORE
        path = rule.fab_pattern.format(**m.groupdict())
        val = rule.converter(safe_val) if rule.converter else safe_val
        return path, val
    else:
        return None


def convert_safetensor(rules: List[ConversionRule], safe_key: str, safe_val):
    for rule in rules:
        path_val = maybe_apply_rule(rule, safe_key, safe_val)
        if path_val:
            return path_val
    raise ValueError(
        f"Couldn't find a rule for safetensors key {safe_key} "
        + f"and value of shape {safe_val.shape}"
    )


# def safe2jax(rules: List[ConversionRule], flat: Dict[str, jax.Array]):
#     params: Dict[str, Any] = {}
#     for safe_key, safe_val in flat.items():
#         path, val = convert_safetensor(rules, safe_key, safe_val)
#         if val is not IGNORE:
#             fab_keys = path.split(".")
#             # Flax NNX uses dicts with INT keys for lists (e.g. layers)
#             fab_keys = [int(key) if key.isnumeric() else key for key in fab_keys]
#             set_nested(params, fab_keys, val)
#     return params


# # TODO: not needed in NNX world?
# def load_params(rules: List[ConversionRule], model_dir: str, out=None):
#     """
#     Load Flax variables from a Huggingface model directory
#     """
#     with open(os.path.join(model_dir, "model.safetensors.index.json")) as fp:
#         index = json.load(fp)
#     safe_files_ = set(index["weight_map"].values())
#     safe_files = [os.path.join(model_dir, filename) for filename in safe_files_]
#     params = out or {}
#     for path in tqdm(safe_files):
#         flat = st.load_file(path)
#         new_params = safe2jax(rules, flat)
#         print(path)
#         update_tree(params, new_params)
#     return params


# def update_state_from_safe(state: Dict, rules: List[ConversionRule], model_dir: str):
#     """
#     Load Flax variables from a Huggingface model directory
#     """
#     with open(os.path.join(model_dir, "model.safetensors.index.json")) as fp:
#         index = json.load(fp)
#     safe_files_ = set(index["weight_map"].values())
#     safe_files = [os.path.join(model_dir, filename) for filename in safe_files_]
#     for path in tqdm(safe_files):
#         flat = st.load_file(path)
#         new_params = safe2jax(rules, flat)
#         update_tree(state, new_params)


def apply_rules(
    model: nnx.Module, rules: List[ConversionRule], flat: Dict[str, jax.Array]
):
    for safe_key, safe_val in flat.items():
        path, val = convert_safetensor(rules, safe_key, safe_val)
        if val is not IGNORE:
            fab_keys = path.split(".")
            fab_keys += ["value"]  # set to the .value field
            set_nested_attr(model, fab_keys, val)


def update_model_from_safe(
    model: nnx.Module, rules: List[ConversionRule], model_dir: str
):
    """
    Update Flax NNX model from a Huggingface model directory
    """
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as fp:
        index = json.load(fp)
    safe_files_ = set(index["weight_map"].values())
    safe_files = [os.path.join(model_dir, filename) for filename in safe_files_]
    for path in tqdm(safe_files):
        flat = st.load_file(path)
        apply_rules(model, rules, flat)


def load_from_file(
    TokC,
    ModArgC,
    ModC,
    rules,
    model_dir: str,
    **model_args
):
    """
    Load a model from a Huggingface model directory.

    Args:
        TokC (type): Tokenizer class.
        ModArgsC (type): Model args class.
        ModC (type): Model class.
        rules (List[ConversionRule]): Rules to load params.
        model_dir (str): Path to the Huggingface model directory.
        **model_args: Keyword arguments to overwrite defaults in ModArgsC().

    Returns:
        Tuple of (tokenizer, model, hf_config).
    """
    config_file = os.path.join(model_dir, "config.json")
    with open(config_file) as fp:
        hf_config = json.load(fp)
    tokenizer_file = os.path.join(model_dir, "tokenizer.json")
    tokenizer = TokC.from_file(tokenizer_file)
    args = ModArgC.from_file(config_file, **model_args)
    model = ModC(args)
    update_model_from_safe(model, rules, model_dir)
    return tokenizer, model, hf_config


def load_from_pretrained(
    TokC,
    ModArgC,
    ModC,
    rules,
    repo_id: str,
    **model_args
):
    """
    Load a model from a Huggingface Hub.

    Args:
        TokC (type): Tokenizer class.
        ModArgsC (type): Model args class.
        ModC (type): Model class.
        rules (List[ConversionRule]): Rules to load params.
        repo_id (str): Repo/model ID on Hugginface Hub.
        **model_args: Keyword arguments to overwrite defaults in ModArgsC().

    Returns:
        Tuple of (tokenizer, model, hf_config).
    """
    model_dir = snapshot_download(repo_id, repo_type="model")
    return load_from_file(TokC, ModArgC, ModC, rules, model_dir, **model_args)