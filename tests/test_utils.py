from fabrique.utils import update_tree


def test_update_tree():
    t1 = {"params": {"a": 1, "b": 2}}
    t2 = {"params": {"b": 20, "c": 3}, "cache": {"k": 100}}
    update_tree(t1, t2)
    assert t1 == {"params": {"a": 1, "b": 20, "c": 3}, "cache": {"k": 100}}
