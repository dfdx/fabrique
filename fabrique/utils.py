from jax import tree_util


def size_gb(variables: dict):
    from math import prod

    bytes = sum([prod(x.shape) * 4 for x in tree_util.tree_leaves(variables)])
    return bytes / (1024**3)


def update_tree(a: dict, b: dict):
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


def print_var(name: str, x):
    """
    Print some of the array properties. Useful for print-based debuging
    (which you normally shouldn't do)
    """
    if x is not None:
        print(f"{name}: mean={x.mean()}, shape={x.shape}, dtype={x.dtype}")
    else:
        print(f"{name} = {x}")
