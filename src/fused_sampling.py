import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'group_sz': 2}, num_warps=2),
        triton.Config({'group_sz': 2}, num_warps=4),
        triton.Config({'group_sz': 2}, num_warps=8),
        triton.Config({'group_sz': 2}, num_warps=16),
        triton.Config({'group_sz': 4}, num_warps=2),
        triton.Config({'group_sz': 4}, num_warps=4),
        triton.Config({'group_sz': 4}, num_warps=8),
        triton.Config({'group_sz': 4}, num_warps=16),
        triton.Config({'group_sz': 8}, num_warps=2),
        triton.Config({'group_sz': 8}, num_warps=4),
        triton.Config({'group_sz': 8}, num_warps=8),
        triton.Config({'group_sz': 8}, num_warps=16),
    ],
    key=['group_sz']
)
@triton.jit
def fused_softmax_sampling_kernel(
    logits_ptr,
    input_batch_stride,
    output_ptr,
    output_batch_stride,
    vocab_size,
    exponential_ptr,
    group_sz: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    block_start = batch_idx * input_batch_stride
    offsets = tl.arange(0, BLOCK_SIZE)

    # Load logits
    logits = tl.load(logits_ptr + block_start + offsets, mask=offsets < vocab_size, other=-float("inf"))
    expo = tl.load(exponential_ptr + block_start + offsets, mask=offsets < vocab_size, other=1)

    # Softmax
    logits = logits - tl.max(logits, axis=0)
    exp_logits = tl.exp(logits)
    probs = exp_logits / tl.sum(exp_logits, axis=0)

    # Sampling
    sampled_index = tl.argmax(tl.div_rn(probs,expo), axis=-1)

    tl.store(output_ptr + batch_idx * output_batch_stride, sampled_index)


def fused_softmax_sampling(logits: torch.Tensor, out: torch.Tensor):
    """
    This function performs the sampling operation using fused kernels.
    """
    assert logits.is_contiguous(), "Logits must be contiguous"
    assert logits.device.type == "cuda", "Logits must be on CUDA"
    assert logits.ndim == 2, "Logits must be 2D, (batch_size, vocab_size)"

    batch_size, vocab = logits.shape
    exponential = torch.empty_like(logits).exponential_(1)

    grid = (batch_size, )
    BLOCK_SIZE = triton.next_power_of_2(vocab)

    fused_softmax_sampling_kernel[grid](
        logits_ptr=logits,
        input_batch_stride=logits.stride(0),
        output_ptr=out,
        output_batch_stride=out.stride(0),
        vocab_size=vocab,
        exponential_ptr=exponential,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out


def torch_sample(logits, q=None):
    probs = torch.nn.functional.softmax(logits, dim=-1)

    if q is None:
        q = torch.empty_like(probs).exponential_(1)

    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['V'],
        x_vals=[128, 512, 2048, 4096, 16000, 32000, 42000, 50000, 64000],
        line_arg='provider',
        line_vals=[
            'triton',
            'torch',
        ],
        line_names=[
            "Triton",
            "Torch",
        ],
        styles=[('blue', '-'), ('green', '-')],
        ylabel="GB/s",
        plot_name="Performance",
        args={'B': 1},
    ))
def benchmark(B, V, provider):
    x = torch.randn(B, V, device='cuda:0')
    out = torch.empty((B,), device=logits.device).to(dtype=torch.int32)

    quantiles = [0.5, 0.2, 0.8]

    print(f'bench for {B, V, provider}')

    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fused_softmax_sampling(x, out), quantiles=quantiles)
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_sample(x), quantiles=quantiles)

    def gbps(ms): return 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == '__main__':

    B, V = 1, 50000
    logits = torch.randn(B, V, device='cuda:0')
    out = torch.empty((B, ), device=logits.device).to(dtype=torch.int32)

    sample_idx = fused_softmax_sampling(logits=logits, out=out)
    sample_torch_idx = torch_sample(logits).squeeze(-1)

    # assert torch.allclose(sample_idx, sample_torch_idx), f'{sample_idx}, {sample_torch_idx}'
    benchmark.run(show_plots=True, print_data=True)
