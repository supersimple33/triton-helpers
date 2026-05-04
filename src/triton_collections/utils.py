import triton
import triton.language as tl

@triton.jit
def rotl32(x: tl.tensor, r: tl.constexpr):
    """Rotates a uint32 tensor left by r bits."""
    return (x << r) | (x >> (32 - r))

@triton.jit
def rotl64(x: tl.tensor, r: tl.constexpr):
    """Rotates a uint64 tensor left by r bits."""
    # Note: Assumes x is uint64
    return (x << r) | (x >> (64 - r))