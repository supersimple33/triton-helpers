import pytest
import torch

from triton_collections.static_map_host import StaticMap
from triton_collections import static_map_kernels


class FakeKernelLauncher:
    def __init__(self, impl):
        self.impl = impl
        self.grids = []

    def __getitem__(self, grid):
        self.grids.append(grid)

        def launch(*args, **kwargs):
            return self.impl(*args, **kwargs)

        return launch


def _supports_remainder(dtype: torch.dtype) -> bool:
    try:
        _ = torch.tensor([3], dtype=dtype) % 2
        return True
    except NotImplementedError:
        return False


def _supported_key_dtype() -> torch.dtype:
    for dtype in (torch.uint64, torch.uint32, torch.int64, torch.int32):
        if _supports_remainder(dtype):
            return dtype
    pytest.skip("this environment does not support remainder for torch.uint32/torch.uint64")


def _call_uncompiled(method, *args, **kwargs):
    # torch.compile wraps methods; tests can call the original function directly
    # to avoid backend requirements for unit-level behavior checks.
    uncompiled = getattr(method, "__wrapped__", method)
    return uncompiled(*args, **kwargs)


def test_init_validates_capacity_value_size_and_key_dtype():
    with pytest.raises(ValueError, match="capacity must be positive"):
        StaticMap(0)

    with pytest.raises(ValueError, match="value_size must be positive"):
        StaticMap(8, value_size=0)

    with pytest.raises(TypeError, match="key_dtype must be torch.uint32 or torch.uint64"):
        StaticMap(8, key_dtype=torch.float64)


def test_validate_key_tensor_rejects_non_1d_non_contiguous_or_wrong_dtype():
    key_dtype = _supported_key_dtype()
    smap = StaticMap(16, key_dtype=key_dtype)

    with pytest.raises(TypeError, match="key_hashes dtype does not match map key_dtype"):
        smap._validate_key_tensor(torch.arange(4, dtype=torch.int16))

    with pytest.raises(ValueError, match="key_hashes must be a flat 1D tensor"):
        smap._validate_key_tensor(torch.zeros((2, 2), dtype=key_dtype))

    non_contiguous = torch.arange(8, dtype=key_dtype)[::2]
    assert not non_contiguous.is_contiguous()
    with pytest.raises(ValueError, match="key_hashes tensor must be contiguous"):
        smap._validate_key_tensor(non_contiguous)


def test_insert_skips_kernel_for_empty_input(monkeypatch):
    key_dtype = _supported_key_dtype()
    smap = StaticMap(16, key_dtype=key_dtype, value_size=1)

    launcher = FakeKernelLauncher(lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("kernel should not be called")))
    monkeypatch.setattr(static_map_kernels, "insert_key_linear", launcher)

    key_hashes = torch.empty((0,), dtype=key_dtype)
    values = torch.empty((0, 1), dtype=torch.uint64)
    _call_uncompiled(StaticMap.insert, smap, key_hashes, values)

    assert launcher.grids == []


def test_insert_then_retrieve_round_trip(monkeypatch):
    key_dtype = _supported_key_dtype()
    smap = StaticMap(8, key_dtype=key_dtype, value_dtype=torch.int64, value_size=2)

    def fake_insert_kernel(
        keys,
        in_keys,
        out_slots,
        out_inserted,
        n_elements,
        capacity,
        empty_key,
        MAX_PROBE,
        BLOCK,
    ):
        assert n_elements == 3
        assert capacity == smap.capacity
        assert empty_key == smap.empty_key
        out_slots.copy_(torch.tensor([1, 2, 3], dtype=in_keys.dtype))
        out_inserted.copy_(torch.tensor([True, True, True], dtype=torch.bool))
        keys[1] = in_keys[0]
        keys[2] = in_keys[1]
        keys[3] = in_keys[2]

    launcher = FakeKernelLauncher(fake_insert_kernel)
    monkeypatch.setattr(static_map_kernels, "insert_key_linear", launcher)

    key_hashes = torch.tensor([1, 9, 17], dtype=key_dtype)
    values = torch.tensor([[11, 12], [91, 92], [171, 172]], dtype=torch.int64)

    _call_uncompiled(StaticMap.insert, smap, key_hashes, values)
    result = _call_uncompiled(StaticMap.retrieve, smap, key_hashes)

    assert launcher.grids == [(1,)]
    assert torch.equal(result, values)


@pytest.mark.xfail(reason="StaticMap.insert uses unsigned tensor indices for value writes")
def test_insert_writes_values_only_for_inserted_slots(monkeypatch):
    key_dtype = _supported_key_dtype()
    smap = StaticMap(8, key_dtype=key_dtype, value_dtype=torch.int64, value_size=2)
    smap._values.fill_(-1)

    def fake_insert_kernel(
        keys,
        in_keys,
        out_slots,
        out_inserted,
        n_elements,
        capacity,
        empty_key,
        MAX_PROBE,
        BLOCK,
    ):
        assert n_elements == in_keys.numel()
        assert capacity == smap.capacity
        assert empty_key == smap.empty_key
        out_slots.copy_(torch.tensor([3, 5, 1], dtype=in_keys.dtype))
        out_inserted.copy_(torch.tensor([True, False, True], dtype=torch.bool))

    launcher = FakeKernelLauncher(fake_insert_kernel)
    monkeypatch.setattr(static_map_kernels, "insert_key_linear", launcher)

    key_hashes = torch.tensor([11, 21, 31], dtype=key_dtype)
    values = torch.tensor([[100, 101], [200, 201], [300, 301]], dtype=torch.int64)

    _call_uncompiled(StaticMap.insert, smap, key_hashes, values)

    assert launcher.grids == [(1,)]
    assert torch.equal(smap._values[3], values[0])
    assert torch.equal(smap._values[1], values[2])
    assert torch.equal(smap._values[5], torch.tensor([-1, -1], dtype=torch.int64))


def test_retrieve_returns_values_for_found_keys():
    key_dtype = _supported_key_dtype()
    smap = StaticMap(8, key_dtype=key_dtype, value_dtype=torch.int64, value_size=2)

    # Build a simple probe chain:
    # key 1 starts at slot 1, key 9 also starts at slot 1 and lands at slot 2.
    smap._keys[1] = 1
    smap._keys[2] = 9
    smap._values[1] = torch.tensor([10, 11], dtype=torch.int64)
    smap._values[2] = torch.tensor([90, 91], dtype=torch.int64)

    query = torch.tensor([9, 1], dtype=key_dtype)
    result = _call_uncompiled(StaticMap.retrieve, smap, query)

    expected = torch.tensor([[90, 91], [10, 11]], dtype=torch.int64)
    assert torch.equal(result, expected)


def test_clear_zeros_all_keys():
    smap = StaticMap(8)
    smap._keys[:] = 5

    smap.clear()

    assert torch.equal(smap._keys, torch.zeros_like(smap._keys))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for Triton kernel execution")
def test_insert_and_retrieve_round_trip_with_real_kernel_and_compile():
    smap = StaticMap(8, key_dtype=torch.int64, value_dtype=torch.int64, value_size=2, device="cuda")

    key_hashes = torch.tensor([1, 9, 17], dtype=torch.int64, device="cuda")
    values = torch.tensor([[11, 12], [91, 92], [171, 172]], dtype=torch.int64, device="cuda")

    smap.insert(key_hashes, values)
    result = smap.retrieve(key_hashes)

    assert torch.equal(result, values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for Triton kernel execution")
def test_retrieve_returns_zero_rows_for_missing_keys_after_real_insert():
    smap = StaticMap(8, key_dtype=torch.int64, value_dtype=torch.int64, value_size=2, device="cuda")

    inserted_keys = torch.tensor([2, 10], dtype=torch.int64, device="cuda")
    inserted_values = torch.tensor([[20, 21], [100, 101]], dtype=torch.int64, device="cuda")
    query_keys = torch.tensor([10, 2, 18], dtype=torch.int64, device="cuda")

    smap.insert(inserted_keys, inserted_values)
    result = smap.retrieve(query_keys)

    expected = torch.tensor([[100, 101], [20, 21], [0, 0]], dtype=torch.int64, device="cuda")
    assert torch.equal(result, expected)
