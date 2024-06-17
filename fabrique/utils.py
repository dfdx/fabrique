import re
from typing import Dict, List
from jax import tree_util
from flax.core import FrozenDict


AnyDict = Dict | FrozenDict
DictOrList = List | AnyDict


def size_gb(variables: dict):
    from math import prod

    bytes = sum([prod(x.shape) * 4 for x in tree_util.tree_leaves(variables)])
    return bytes / (1024**3)


def update_tree(a: Dict, b: Dict):
    """
    Update tree a with keys from tree b.

    Semantics of this operation is the same as dict.update(), but update_tree()
    also works recursively.
    """
    for key in b:
        if key in a and isinstance(a[key], dict) and isinstance(b[key], dict):
            update_tree(a[key], b[key])
        else:
            a[key] = b[key]


def eachindex(x: DictOrList):
    if isinstance(x, List):
        return list(range(len(x)))
    elif isinstance(x, AnyDict):
        return list(x.keys())


def hasindex(x: DictOrList, idx):
    if isinstance(x, List):
        return 0 <= idx < len(x)
    elif isinstance(x, AnyDict):
        return idx in x


def update_tree(a: DictOrList, b: DictOrList):
    """
    Update tree a with keys from tree b.

    Semantics of this operation is the same as dict.update(), but update_tree()
    also works recursively.
    """
    for key in eachindex(b):
        if hasindex(a, key) and isinstance(a[key], DictOrList) and isinstance(b[key], DictOrList):
            update_tree(a[key], b[key])
        else:
            a[key] = b[key]


def print_var(name: str, x):
    """
    Print some of the array properties. Useful for print-based debuging
    (which you normally shouldn't do)
    """
    if x is not None:
        print(f"{name}: mean={x.mean()}, shape={x.shape}, dtype={x.dtype}")
    else:
        print(f"{name} = {x}")


LIST_KEY_REGEXP = r"^([a-zA-Z0-9]+)\[(\d+)\]"


def set_nested(nested: Dict, keys: List[str], val):
    """
    Set value into the nested dict according to the path of keys.

    Example:

        set_nested({}, ["a", "b", "c"], 42)
        # ==> {'a': {'b': {'c': 42}}}

        set_nested({"a": {"d": 54}}, ["a", "b[3]", "c"], 42)
        # ==> {'a': {'d': 54, 'b': [None, None, None, {'c': 42}]}}
    """
    dct = nested
    for key in keys[:-1]:
        index = None
        is_list_match = None
        if isinstance(key, str):
            re.match(LIST_KEY_REGEXP, key)
        if is_list_match:
            # sub-object is a list
            key, index = is_list_match.groups()
            index = int(index)
            if key not in dct:
                dct[key] = []
            lst = dct[key]
            # if list is too short, extend it
            lst.extend([None for _ in range(index + 1 - len(lst))])
            if lst[index] is None:
                lst[index] = {}
            dct = lst[index]
        else:
            # sub-object is a dict
            if key not in dct:
                dct[key] = {}
            dct = dct[key]
    dct[keys[-1]] = val
    return nested


def set_nested_attr(nested_obj, fields: List[str], val):
    """
    Set attribute value according to the path of fields.

    Like set_nested(), but for object attributes.
    """
    obj = nested_obj
    for field in fields[:-1]:
        index = None
        is_list_match = None
        if isinstance(field, str):
            is_list_match = re.match(LIST_KEY_REGEXP, field)
        if is_list_match:
            # sub-object is a list
            field, index = is_list_match.groups()
            index = int(index)
            assert hasattr(obj, field), f"Object of type {type(obj)} doesn't have attribute {field}"
            # if field not in obj:
            #     # dct[key] = []
            #     setattr(obj, field, [])
            lst = getattr(obj, field)
            assert index < len(lst), f"Trying to set {type(obj)}.{field}[{index}], but the list only has length of {len(lst)}"
            # if list is too short, extend it
            # lst.extend([None for _ in range(index + 1 - len(lst))])
            # if lst[index] is None:
            #     lst[index] = {}
            obj = lst[index]
        else:
            # sub-object is a dict
            # if field not in obj:
            #     obj[field] = {}
            obj = getattr(obj, field)
    setattr(obj, fields[-1], val)
    return nested_obj


def cache_layout(model, layer_id=0):
    x = model.layers[layer_id].attention.cache_k.value
    flags = x.sum(axis=2)[0, :, 0] != 0
    return flags.astype(int)


# def int_dicts_to_lists(nested: Dict):
#     """
#     Given a nested dict, convert subdicts with only integer keys
#     into corresponding lists.
#     """
#     def dict_to_list(dct: Dict):
#         idxs = [int(key) for key in dct.keys()]
#         lst = [{} for _ in range(max(idxs) + 1)]
#         for key, idx in zip(dct.keys(), idxs):
#             lst[idx] = dct[key]
#         return lst
#     for key in nested:
#         dct = nested[key]
#         if not isinstance(dct, Dict):
#             continue
#         if all(subkey.isnumeric() for subkey in dct.keys()):
#             lst = dict_to_list(dct)
#             nested[key] = lst
#             # apply recursively to each item in the new list
#             for item in lst:
#                 if isinstance(item, Dict):
#                     int_dicts_to_lists(item)
#         else:
#             # apply recursively to each value of the dct
#             for val in dct.values():
#                 if isinstance(val, Dict):
#                     int_dicts_to_lists(val)


# def int_dicts_to_lists(x):
#     """
#     Given a nested structure x, identify all dicts that have only numeric
#     keys and convert them to the corresponding lists.

#     Example:

#     ```
#     nested = {
#         "params": {
#             "layers": {
#                 "0": {"weights": 0},
#                 "1": {"weights": 10},
#                 "2": {"weights": 20},
#             }
#         },
#         "cache": 42,
#     }
#     int_dicts_to_lists(nested)
#     ```
#     produces:
#     ```
#     expected = {
#         "params": {
#             "layers": [
#                 {"weights": 0},
#                 {"weights": 10},
#                 {"weights": 20},
#             ]
#         },
#         "cache": 42,
#     }
#     ```
#     """
#     if isinstance(x, AnyDict):
#         if all(key.isnumeric() for key in x.keys()):
#             # convert dict to list
#             keys = list(x.keys())
#             idxs = [int(key) for key in keys]
#             lst = [{} for _ in range(max(idxs) + 1)]
#             for key, idx in zip(keys, idxs):
#                 # note: apply function recursively
#                 lst[idx] = int_dicts_to_lists(x[key])
#             return lst
#         else:
#             dct = {key: int_dicts_to_lists(val) for key, val in x.items()}
#             return type(x)(dct)  # convert to the original dict type
#     elif isinstance(x, List):
#         return [int_dicts_to_lists(item) for item in x]
#     else:
#         return x