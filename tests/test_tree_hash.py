import pytest

import triton_collections.tree_hash as tree_hash


class FakeValueHash:
    def __init__(self, value):
        self.value = value

    def __xor__(self, other):
        return (self.value, other)


@pytest.mark.parametrize(
    ("hash_fn", "mix_name"),
    [
        (tree_hash.tree_hash_32, "murmur_mix32"),
        (tree_hash.tree_hash_64, "murmur_mix64"),
    ],
)
def test_tree_hash_delegates_to_sum_and_murmur_mix(monkeypatch, hash_fn, mix_name):
    calls = []

    def fake_sum(child_hashes):
        calls.append(("sum", child_hashes))
        return "summed-child-hashes"

    def fake_mix(key, seed):
        calls.append(("mix", key, seed))
        return {"key": key, "seed": seed}

    monkeypatch.setattr(tree_hash.tl, "sum", fake_sum)
    monkeypatch.setattr(tree_hash, mix_name, fake_mix)

    result = hash_fn.fn(FakeValueHash("value-hash"), "children", 7)

    assert result == {"key": ("value-hash", "summed-child-hashes"), "seed": 7}
    assert calls == [
        ("sum", "children"),
        ("mix", ("value-hash", "summed-child-hashes"), 7),
    ]


def test_tree_hash_jit_signatures_expose_expected_arguments():
    assert tree_hash.tree_hash_32.arg_names == ["value_hash", "child_hashes", "seed"]
    assert tree_hash.tree_hash_64.arg_names == ["value_hash", "child_hashes", "seed"]


@pytest.mark.parametrize(
    ("hash_fn", "mix_name"),
    [
        (tree_hash.tree_hash_32, "murmur_mix32"),
        (tree_hash.tree_hash_64, "murmur_mix64"),
    ],
)
def test_tree_hash_is_invariant_to_child_hash_order(monkeypatch, hash_fn, mix_name):
    def fake_mix(key, seed):
        return (key, seed)

    monkeypatch.setattr(tree_hash.tl, "sum", sum)
    monkeypatch.setattr(tree_hash, mix_name, fake_mix)

    first_order = hash_fn.fn(FakeValueHash("value-hash"), [1, 2, 3, 4], 7)
    second_order = hash_fn.fn(FakeValueHash("value-hash"), [4, 1, 3, 2], 7)

    assert first_order == second_order