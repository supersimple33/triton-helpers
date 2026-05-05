"""Triton kernels for a simple static hash map.

These kernels implement a fixed-capacity, open-addressing hash map with
linear probing and insert-or-assign semantics. Keys are expected to be
pre-hashed by the caller.
"""

import triton
import triton.language as tl

@triton.jit
def find_or_insert_key_linear(
    keys_ptr,
    in_keys_ptr,
    out_slots_ptr,
    n_elements,
    capacity,
    empty_key,
    MAX_PROBE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Finds slots for the given keys using linear probing."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    in_keys = tl.load(in_keys_ptr + offsets, mask=mask, other=empty_key)
    valid = mask & (in_keys != empty_key) # host should ensure that in_keys != empty_key
    slots = in_keys % capacity

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

    tl.device_assert(tl.max(active) == 0, "static_map_insert_or_assign failed to reserve slots for all keys")

    tl.store(out_slots_ptr + offsets, slot_indices, mask=mask)

@triton.jit
def insert_key_linear(
    keys_ptr,
    in_keys_ptr,
    out_slots_ptr,
    out_inserted_ptr,
    n_elements,
    capacity,
    empty_key,
    MAX_PROBE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Inserts the given keys using linear probing. Skips keys that are already present."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    in_keys = tl.load(in_keys_ptr + offsets, mask=mask, other=empty_key)
    valid = mask & (in_keys != empty_key) # host should ensure that in_keys != empty_key
    slots = in_keys % capacity

    active = valid
    slot_indices = tl.zeros((BLOCK,), dtype=slots.dtype) # safe since we report inserted
    for i in range(MAX_PROBE):
        idxs = (slots + i) % capacity

        slot_keys = tl.load(keys_ptr + idxs, mask=active, other=empty_key)
        hit = active & (slot_keys == empty_key)
        slot_indices = tl.where(hit, idxs, slot_indices)
        active = active & ~hit & (slot_keys != in_keys)
    inserted = valid & (~active)

    tl.store(out_inserted_ptr + offsets, inserted.to(out_inserted_ptr.dtype), mask=mask)
    tl.store(out_slots_ptr + offsets, slot_indices, mask=mask)

@triton.jit
def find_key_linear(
    keys_ptr,
    in_keys_ptr,
    out_slots_ptr,
    out_found_ptr,
    n_elements,
    capacity,
    empty_key,
    MAX_PROBE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Finds slots for the given keys using linear probing."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    in_keys = tl.load(in_keys_ptr + offsets, mask=mask, other=empty_key)
    valid = mask & (in_keys != empty_key) # host should ensure that in_keys != empty_key
    slots = in_keys % capacity

    active = valid
    slot_indices = tl.zeros((BLOCK,), dtype=slots.dtype) # safe since non overwritten are flagged as not found
    for i in range(MAX_PROBE):
        idxs = (slots + i) % capacity

        slot_keys = tl.load(keys_ptr + idxs, mask=active, other=empty_key)
        hit = active & (slot_keys == in_keys)
        slot_indices = tl.where(hit, idxs, slot_indices)
        active = active & (slot_keys != empty_key)
    found = valid & (~active)

    tl.store(out_found_ptr + offsets, found.to(out_found_ptr.dtype), mask=mask)
    tl.store(out_slots_ptr + offsets, slot_indices, mask=mask)
