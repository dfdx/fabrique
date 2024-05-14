import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, List

import jax
import safetensors.flax as st


@dataclass
class ConversionRule:
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


def maybe_apply_rule(rule: ConversionRule, safe_key: str, safe_val):
    if m := re.match(rule.safe_regexp, safe_key):
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


def set_nested(nested: Dict, keys: List[str], val):
    dct = nested
    for subkey in keys[:-1]:
        if subkey not in dct:
            dct[subkey] = {}
        dct = dct[subkey]
    dct[keys[-1]] = val
    return nested


def safe2jax(rules: List[ConversionRule], flat: Dict[str, jax.Array]):
    params = {}
    for safe_key, safe_val in flat.items():
        path, val = convert_safetensor(rules, safe_key, safe_val)
        fab_keys = path.split(".")
        set_nested(params, fab_keys, val)
    return params


# from https://stackoverflow.com/a/7205107/365872
def merge_dicts_recur(a: dict, b: dict, path=[]):
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dicts_recur(a[key], b[key], path + [str(key)])
            elif a[key] != b[key]:
                raise Exception("Conflict at " + ".".join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a


def load_variables(rules: List[ConversionRule], model_dir: str):
    """
    Load Flax variables from a Huggingface model directory
    """
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as fp:
        index = json.load(fp)
    safe_files = set(index["weight_map"].values())
    safe_files = [os.path.join(model_dir, filename) for filename in safe_files]
    params = {}
    for path in safe_files:
        flat = st.load_file(path)
        new_params = safe2jax(rules, flat)
        params = merge_dicts_recur(params, new_params)
    return {"params": params}
