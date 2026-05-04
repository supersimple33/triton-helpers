import triton
import triton.language as tl

@triton.jit
def murmur_mix32(key: tl.tensor, seed: tl.constexpr = tl.constexpr(42)):
    """MurmurHash3 32-bit mix function."""
    h = key.to(tl.uint32) ^ seed
    h ^= h >> 16
    h *= 0x85ebca6b
    h ^= h >> 13
    h *= 0xc2b2ae35
    h ^= h >> 16
    return h

@triton.jit
def murmur_mix64(key: tl.tensor, seed: tl.constexpr = tl.constexpr(42)):
    """MurmurHash3 64-bit mix function."""
    h = key.to(tl.uint64) ^ seed
    h ^= h >> 33
    h *= 0xff51afd7ed558ccd
    h ^= h >> 33
    h *= 0xc4ceb9fe1a85ec53
    h ^= h >> 33
    return h