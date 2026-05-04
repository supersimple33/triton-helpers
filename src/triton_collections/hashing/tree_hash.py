import random

import triton
import triton.language as tl

from triton_collections.hashing.murmur3 import murmur_mix32, murmur_mix64

@triton.jit
def tree_hash_32(
        value_hash: tl.tensor,
        child_hashes: tl.tensor,
        seed: tl.constexpr = tl.constexpr(42)
) -> tl.tensor:
    """Computes a hash for a tensor based on its value and the hashes of its children."""

    h = value_hash ^ tl.sum(child_hashes)
    return murmur_mix32(h, seed)

@triton.jit
def tree_hash_64(
        value_hash: tl.tensor,
        child_hashes: tl.tensor,
        seed: tl.constexpr = tl.constexpr(42)
) -> tl.tensor:
    """Computes a hash for a tensor based on its value and the hashes of its children."""

    h = value_hash ^ tl.sum(child_hashes)
    return murmur_mix64(h, seed)
    
    
    