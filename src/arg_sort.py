"""
Copied from https://github.com/triton-lang/triton/issues/3698
"""

import torch
import triton
import triton.language as tl
import triton.language.core as core
from triton.language.standard import _log2, sum, zeros_like


@triton.jit
def _compare_and_swap(x, ids, flip, i: core.constexpr, n_dims: core.constexpr):
    n_outer: core.constexpr = x.numel >> n_dims
    shape: core.constexpr = [n_outer * 2**i, 2, 2**(n_dims - i - 1)]
    y = core.reshape(x, shape)
    # slice left/right with 'stride' 2**(n_dims - i - 1)
    mask = core.arange(0, 2)[None, :, None]
    left = core.broadcast_to(sum(y * (1 - mask), 1)[:, None, :], shape)
    right = core.broadcast_to(sum(y * mask, 1)[:, None, :], shape)
    left = core.reshape(left, x.shape)
    right = core.reshape(right, x.shape)

    # idx
    y_idx = core.reshape(ids, shape)
    left_idx = core.broadcast_to(sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = core.broadcast_to(sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = core.reshape(left_idx, x.shape)
    right_idx = core.reshape(right_idx, x.shape)

    # actual compare-and-swap
    idtype = core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth,
                                signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) ^ flip

    ret = ix ^ core.where(cond, ileft ^ iright, zeros_like(ix))

    new_ids = ids ^ core.where(cond, left_idx ^ right_idx, zeros_like(ids))

    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(x, ids, stage: core.constexpr, order: core.constexpr,
                   n_dims: core.constexpr):
    '''
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    '''
    n_outer: core.constexpr = x.numel >> n_dims
    core.static_assert(stage <= n_dims)
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if order == 2:
        shape: core.constexpr = [
            n_outer * 2**(n_dims - 1 - stage), 2, 2**stage
        ]
        flip = core.reshape(
            core.broadcast_to(core.arange(0, 2)[None, :, None], shape),
            x.shape)
    else:
        flip = order
    # perform `stage` rounds of `compare-and-swap`
    for i in core.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def argsort(x,
            ids,
            dim: core.constexpr = None,
            descending: core.constexpr = core.CONSTEXPR_0):
    # handle default dimension or check that it is the most minor dim
    _dim: core.constexpr = len(x.shape) - 1 if dim is None else dim
    core.static_assert(_dim == len(x.shape) - 1,
                       "only minor dimension is currently supported")
    # iteratively run bitonic merge-sort steps
    n_dims: core.constexpr = _log2(x.shape[_dim])

    for i in core.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge(x, ids, i, 2 if i < n_dims else descending,
                                n_dims)
    return x, ids


@triton.jit
def sort_kernel(
    # Pointers to matrices
    x_ptr,
    o_ptr,
    id_ptr,
    stride_m,
    stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    m_offset = pid_m * stride_m * BLOCK_M
    k_off = tl.arange(0, BLOCK_N)

    x_ptrs = x_ptr + m_offset + (tl.arange(0, BLOCK_M)[:, None] * stride_m +
                                 k_off[None, :])

    # shape: [BLOCK_M, BLOCK_N]
    x = tl.load(x_ptrs)
    ids = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (BLOCK_M, BLOCK_N))

    # o, ids = argsort(x, ids, 1, True)
    o, ids = argsort(x, ids, 1, False)

    o_ptrs = o_ptr + m_offset + (tl.arange(0, BLOCK_M)[:, None] * stride_m +
                                 k_off[None, :])
    id_ptrs = id_ptr + m_offset + (tl.arange(0, BLOCK_M)[:, None] * stride_m +
                                   k_off[None, :])
    tl.store(o_ptrs, o)
    tl.store(id_ptrs, ids)

def argsort_triton(x):
    o = torch.empty_like(x)
    ids = torch.empty(x.shape, dtype=torch.int64, device='cuda')

    BLOCK_M = 1
    BLOCK_N = triton.next_power_of_2(x.shape[-1])

    grid = (
        triton.cdiv(x.shape[0], BLOCK_M),
        triton.cdiv(x.shape[1], BLOCK_N),
    )

    k = sort_kernel[grid](x, o, ids, x.stride(0), x.stride(1), BLOCK_M, BLOCK_N)

    return o, ids

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[4 ** i for i in range(2, 8)],
        line_arg='provider',
        line_vals=[
            'triton',
            'torch',
        ],
        line_names=[
            "Triton",
            "Torch (native)",
        ],
        styles=[('blue', '-'), ('green', '-')],
        ylabel="GB/s",
        plot_name="Performance",
        args={'B': 1},
    ))
def benchmark(B, N, provider):
    x = torch.randn(B, N, device='cuda:0', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]

    print(f'bench for {B, N, provider}')

    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: argsort_triton(x), quantiles=quantiles)
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.sort(x, 1, False), quantiles=quantiles)

    def gbps(ms): return 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == '__main__':

    x = torch.randn((1, 2048), dtype=torch.float16, device='cuda')
    o, ids = argsort_triton(x)

    print('result: ')
    print(o)
    print(ids, ids.dtype)

    ref_o, ref_ids = torch.sort(x, 1, False)
    print('ref: ')
    print(ref_o)
    print(ref_ids, ref_ids.dtype)

    assert torch.allclose(o, ref_o), "Sorted arrays are not same"
    assert torch.sum(ref_o != ref_ids), "Sorted IDs are not the same"

    benchmark.run(show_plots=True, print_data=True)
