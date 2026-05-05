"""Host-side wrapper for Triton static hash map kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import triton
import triton.language as tl

from triton_collections import static_map_kernels


@dataclass
class StaticMapConfig:
    max_probe: tl.constexpr = tl.constexpr(16)
    block_size: tl.constexpr = tl.constexpr(256)


class StaticMap:
    """Fixed-capacity, open-addressing hash map backed by Triton kernels.

    Keys are expected to be pre-hashed by the caller. Values are stored as
    fixed-size payloads represented by a flat 1D tensor of length
    n_elements * value_size.
    """

    def __init__(
        self,
        capacity: int,
        *,
        key_dtype: torch.dtype = torch.uint64,
        value_dtype: torch.dtype = torch.uint64,
        value_size: int = 1,
        device: torch.device | str | None = None,
        config: StaticMapConfig | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self._device = torch.device(device) if device is not None else None
        self._key_dtype = key_dtype
        self._value_dtype = value_dtype
        self._capacity = int(capacity)
        self._config = config or StaticMapConfig()
        if value_size <= 0:
            raise ValueError("value_size must be positive")

        if key_dtype not in (torch.uint32, torch.uint64):
            raise TypeError("key_dtype must be torch.uint32 or torch.uint64")

        self._empty_key = 0
        self._value_size = int(value_size)

        self._keys = torch.zeros(
            self._capacity,
            dtype=self._key_dtype,
            device=self._device,
        )
        self._values = torch.empty(
            (self._capacity, self._value_size),
            dtype=self._value_dtype,
            device=self._device,
        )

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def empty_key(self) -> int:
        return self._empty_key

    def clear(self) -> None:
        self._keys.zero_()

    @torch.compile
    def insert(self, key_hashes: torch.Tensor, in_values: torch.Tensor) -> None:
        self._validate_kv_tensors(key_hashes, in_values)
        if torch.any(key_hashes == self._empty_key):
            raise ValueError("cannot insert empty key sentinel")

        n_elements = key_hashes.numel()
        if n_elements == 0:
            return
        
        slots = torch.empty_like(key_hashes)
        inserted = torch.empty(n_elements, dtype=torch.bool, device=self._device)

        grid = (triton.cdiv(n_elements, self._config.block_size),)
        static_map_kernels.insert_key_linear[grid](
            self._keys,
            key_hashes,
            slots,
            inserted,
            n_elements,
            self._capacity,
            self._empty_key,
            MAX_PROBE=self._config.max_probe,
            BLOCK=self._config.block_size,
        )

        self._values[slots[inserted]] = in_values[inserted]


    def retrieve(self, key_hashes: torch.Tensor) -> torch.Tensor:
        self._validate_key_tensor(key_hashes)

        n_elements = key_hashes.numel()
        if n_elements == 0:
            return torch.empty((0, self._value_size), dtype=self._value_dtype, device=self._device)

        result = torch.empty((n_elements, self._value_size), dtype=self._value_dtype, device=self._device)
        found = torch.zeros(n_elements, dtype=torch.bool, device=self._device)

        grid = (triton.cdiv(n_elements, self._config.block_size),)
        static_map_kernels.static_map_retrieve[grid](
            self._keys,
            self._values,
            key_hashes,
            result,
            found,
            n_elements,
            self._capacity,
            self._empty_key,
            MAX_PROBE=self._config.max_probe,
            BLOCK=self._config.block_size,
            VALUE_SIZE=self._value_size, # type: ignore
        )

        return result

    def _validate_key_tensor(self, key_hashes: torch.Tensor) -> None:
        if key_hashes.dtype != self._key_dtype:
            raise TypeError("key_hashes dtype does not match map key_dtype")
        if key_hashes.device != self._keys.device:
            raise ValueError("key_hashes must be on the same device as the map")
        if key_hashes.ndim != 1:
            raise ValueError("key_hashes must be a flat 1D tensor")
        if not key_hashes.is_contiguous():
            raise ValueError("key_hashes tensor must be contiguous")

    def _validate_kv_tensors(self, key_hashes: torch.Tensor, values: torch.Tensor) -> None:
        self._validate_key_tensor(key_hashes)
        if values.dtype != self._value_dtype:
            raise TypeError("values dtype does not match map value_dtype")
        if values.device != self._values.device:
            raise ValueError("values must be on the same device as the map")
        if values.ndim != 2:
            raise ValueError("values must be a 2D tensor")
        if values.shape[1] != self._value_size:
            raise ValueError(f"values second dimension must match map value_size of {self._value_size}")
        if values.shape[0] != key_hashes.numel():
            raise ValueError("key_hashes and values must have the same number of elements")
        if not values.is_contiguous():
            raise ValueError("values tensor must be contiguous")
