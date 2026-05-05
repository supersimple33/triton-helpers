"""Triton kernels for a simple static hash map.

These kernels implement a fixed-capacity, open-addressing hash map with
linear probing and insert-or-assign semantics. Keys are expected to be
pre-hashed by the caller.
"""

import triton
import triton.language as tl


@triton.jit
def _reserve_slots_linear(
    keys_ptr,
    in_keys,
    slots,
    valid,
    capacity,
    empty_key,
    MAX_PROBE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Reserves slots for the given keys using linear probing."""
    active = valid
    slot_indices = tl.zeros((BLOCK,), dtype=slots.dtype) # safe since we only write if this succeeds
    for i in range(MAX_PROBE):
        idxs = (slots + i) % capacity

        # TODO: remove and use masking instead
        slot_keys = tl.load(keys_ptr + idxs, mask=active, other=empty_key)
        cmp = tl.where(active, empty_key, slot_keys)
        val = tl.where(active, in_keys, slot_keys)

        prev = tl.atomic_cas(keys_ptr + idxs, cmp, val) # empty_key, keys, mask=active)
        hit = (prev == in_keys) | (prev == empty_key)
        write = active & hit
        slot_indices = tl.where(write, idxs, slot_indices)
        active = active & (~hit)
    return slot_indices, active

@triton.jit
def _find_slots_linear(
    keys_ptr,
    in_keys,
    slots,
    valid,
    capacity,
    empty_key,
    MAX_PROBE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Finds slots for the given keys using linear probing."""
    active = valid
    slot_indices = tl.zeros((BLOCK,), dtype=slots.dtype) # safe since non overwritten are flagged as not found
    for i in range(MAX_PROBE):
        idxs = (slots + i) % capacity

        slot_keys = tl.load(keys_ptr + idxs, mask=active, other=empty_key)
        hit = active & (slot_keys == in_keys)
        slot_indices = tl.where(hit, idxs, slot_indices)
        active = active & (slot_keys != empty_key)
    return slot_indices, valid & (~active)

@triton.jit
def static_map_insert_or_assign(
    keys_ptr,
    values_ptr,
    in_keys_ptr,
    in_values_ptr,
    n_elements,
    capacity,
    empty_key,
    MAX_PROBE: tl.constexpr,
    BLOCK: tl.constexpr,
    VALUE_SIZE: tl.constexpr,
):
    """Inserts or assigns key-value pairs into the map."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    in_keys = tl.load(in_keys_ptr + offsets, mask=mask, other=empty_key)
    valid = mask & (in_keys != empty_key) # host should ensure that in_keys != empty_key
    slots = in_keys % capacity

    slot_indices, active = _reserve_slots_linear(keys_ptr, in_keys, slots, valid, capacity, empty_key, MAX_PROBE, BLOCK) # type: ignore

    tl.device_assert(tl.max(active) == 0, "static_map_insert_or_assign failed to reserve slots for all keys")

    value_range = tl.arange(0, VALUE_SIZE)
    in_values = tl.load(in_values_ptr + offsets[:, None] * VALUE_SIZE + value_range[None, :], mask=valid[:, None], other=0)

    tl.store(values_ptr + slot_indices[:, None] * VALUE_SIZE + value_range[None, :], in_values, mask=valid[:, None])

@triton.jit
def static_map_retrieve(
    keys_ptr,
    values_ptr,
    in_keys_ptr,
    out_values_ptr,
    out_found_ptr,
    n_elements,
    capacity,
    empty_key,
    MAX_PROBE: tl.constexpr,
    BLOCK: tl.constexpr,
    VALUE_SIZE: tl.constexpr,
):
    """Retrieves values for the given keys from the map."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    in_keys = tl.load(in_keys_ptr + offsets, mask=mask, other=empty_key)
    valid = mask & (in_keys != empty_key) # host should ensure that in_keys != empty_key

    slots = in_keys % capacity

    slot_indices, found = _find_slots_linear(keys_ptr, in_keys, slots, valid, capacity, empty_key, MAX_PROBE, BLOCK) # type: ignore

    tl.store(out_found_ptr + offsets, found.to(out_found_ptr.dtype), mask=valid)

    value_range = tl.arange(0, VALUE_SIZE)
    values = tl.load(values_ptr + slot_indices[:, None] * VALUE_SIZE + value_range[None, :], mask=found[:, None], other=0)
    tl.store(out_values_ptr + offsets[:, None] * VALUE_SIZE + value_range[None, :], values, mask=found[:, None])
