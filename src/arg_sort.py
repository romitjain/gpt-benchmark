import torch
import triton
import triton.language as tl
import triton.language.core as core

@triton.jit
def _compare_and_swap(x, ids, flip, i: core.constexpr, n_dims: core.constexpr):

    n_outer: core.constexpr = x.numel >> n_dims
    shape: core.constexpr = [n_outer * 2**i, 2, 2**(n_dims - i - 1)]

    # Reshape x into [n_outer * 2^i, 2, 2^(n_dims - i - 1)] 
    # This groups elements into pairs along the second dimension.
    y = core.reshape(x, shape)
    y_ids = core.reshape(ids, shape)

    # Generate a mask to isolate left and right halves
    mask = core.arange(0, 2)[None, :, None]
    # left and right halves
    left = core.broadcast_to(core.sum(y * (1 - mask), 1)[:, None, :], shape)
    right = core.broadcast_to(core.sum(y * mask, 1)[:, None, :], shape)

    left_idx = core.broadcast_to(core.sum(y_ids * (1 - mask), 1)[:, None, :], shape)
    right_idx = core.broadcast_to(core.sum(y_ids * mask, 1)[:, None, :], shape)

    left = core.reshape(left, x.shape)
    right = core.reshape(right, x.shape)
    left_idx = core.reshape(left_idx, ids.shape)
    right_idx = core.reshape(right_idx, ids.shape)

    # Convert to int type for XOR swapping
    idtype = core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    # If descending: cond = left < right
    # flip encodes order, here just treat flip=1 for descending.
    # Condition: we want descending, so we swap if left < right
    cond = (left < right) ^ flip  # If flip=1 (descending), cond = left < right

    ret = ix ^ core.where(cond, ileft ^ iright, core.zeros_like(ix))
    new_ids = ids ^ core.where(cond, left_idx ^ right_idx, core.zeros_like(ids))

    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(x, ids, stage: core.constexpr, flip: core.constexpr, n_dims: core.constexpr):
    # Perform 'stage' rounds of compare and swap
    for i in core.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def argsort(x, ids, dim: core.constexpr = None):
    # We'll assume we are sorting the last dimension and the size is power-of-two.
    _dim: core.constexpr = len(x.shape) - 1 if dim is None else dim
    core.static_assert(_dim == len(x.shape) - 1, "only minor dimension supported")

    n_dims: core.constexpr = core._log2(x.shape[_dim])
    # For descending, flip = 1
    flip = 1

    for i in core.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge(x, ids, i, flip, n_dims)
    return x, ids


if __name__ == "__main__":
    # Example:
    N = 16  # Must be a power of two
    x = torch.rand(N, device='cuda')  # random values
    ids = torch.arange(N, device='cuda', dtype=torch.int32)

    print("Before sort:", x)

    x_sorted, ids_sorted = argsort(x, ids, N)
    print("After sort:", x_sorted)
    print("ids:", ids_sorted)

    print(torch.sort(x, descending=True)[0])

    # Verify correctness:
    # x_sorted should be in descending order
    assert torch.all(x_sorted == torch.sort(x, descending=True)[0])

    print("Check passed: array is sorted in descending order")
