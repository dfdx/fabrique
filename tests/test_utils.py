from fabrique.utils import update_tree, set_nested


def test_update_tree():
    t1 = {"params": {"a": 1, "b": 2}}
    t2 = {"params": {"b": 20, "c": 3}, "cache": {"k": 100}}
    update_tree(t1, t2)
    assert t1 == {"params": {"a": 1, "b": 20, "c": 3}, "cache": {"k": 100}}


def test_set_nested():
    nested = {}
    assert set_nested(nested, ["a", "b", "c"], 42) == {'a': {'b': {'c': 42}}}

    nested = {"a": {"d": 54}}
    assert set_nested(nested, ["a", "b", "c"], 42) == {'a': {'d': 54, 'b': {'c': 42}}}

    nested = {"a": {"d": 54}}
    assert set_nested(nested, ["a", "3", "c"], 42) == {'a': {'d': 54, 'b': {3: {'c': 42}}}}

    nested = {"a": {"d": 54}}
    assert set_nested(nested, ["a", "b[3]", "c"], 42) == {'a': {'d': 54, 'b': [None, None, None, {'c': 42}]}}

    nested = {"a": {"d": 54}}
    assert set_nested(nested, ["a", "b[12]", "c"], 42) == {'a': {'d': 54, 'b': [None] * 12 + [{'c': 42}]}}


# def test_int_dicts_to_lists():
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
#     assert int_dicts_to_lists(nested) == expected