import torch
import flashinfer
import triton
from typing import Optional
from torch.profiler import profile, record_function, ProfilerActivity

from src.fused_sampling import fused_softmax_sampling

torch.library.define(
    "flashinfer::sampling",
    "(Tensor logits, Tensor top_k) -> Tensor",
)

@torch.library.impl("flashinfer::sampling", "cuda")
def custom_func(logits, top_k):
    return flashinfer.sampling.top_k_mask_logits(logits, top_k)

@torch.library.register_fake("flashinfer::sampling")
def custom_func_abstract(logits, top_k):
    return torch.empty_like(logits)

def torch_sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)

    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int), probs

def triton_sampling(logits, out: torch.Tensor = None, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

    if out is None:
        out = torch.empty((1, ), device=logits.device)

    return fused_softmax_sampling(logits, out), None

def flash_sample(logits: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)
    # out = torch.zeros(logits.shape[-1], device=logits.device).to(dtype=torch.int32)

    if top_k is not None:
        logits = torch.ops.flashinfer.sampling(logits, top_k)

    # return fused_softmax_sampling(logits, out)
    probs = torch.nn.functional.softmax(logits, dim=-1)

    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int), probs

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[4 ** i for i in range(2, 9)],
        line_arg='provider',
        line_vals=[
            'flash',
            'torch',
        ],
        line_names=[
            "Flash",
            "Torch (native)",
        ],
        styles=[('blue', '-'), ('green', '-')],
        ylabel="GB/s",
        plot_name="Performance",
        args={'B': 1},
    ))
def benchmark(B, N, provider):
    x = torch.randn(B, N, device='cuda:0', dtype=torch.float16)
    top_k = 50

    quantiles = [0.5, 0.2, 0.8]

    print(f'bench for {B, N, provider}')

    with torch.no_grad():
        if provider == 'flash':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: flash_sample(x, 0.5, top_k), quantiles=quantiles, warmup=50, rep=1000)
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_sample(x, 0.5, top_k), quantiles=quantiles, warmup=50, rep=1000)

    def gbps(ms): return 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms), gbps(max_ms), gbps(min_ms)
    # return ms, min_ms, max_ms

def compile_fn():
    global torch_sample
    global flash_sample

    torch_sample = torch.compile(torch_sample, mode="reduce-overhead", fullgraph=True)
    flash_sample = torch.compile(flash_sample, mode="reduce-overhead", fullgraph=True)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--profiling", action="store_true")
    args = parser.parse_args()

    if args.compile:
        compile_fn()

    benchmark.run(show_plots=True, print_data=True)

    logits = torch.randn(1, 50000).to("cuda")
    top_k = 50

    if args.profiling:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            # with_stack=True
        ) as prof:
            with torch.no_grad():
                with record_function("sampling"):
                    for _ in range(10):
                        _ = torch_sample(logits, 0.5, top_k)

        prof.export_chrome_trace(f"torch_sample_{args.compile}.json")

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            # with_stack=True
        ) as prof:
            with torch.no_grad():
                with record_function("sampling"):
                    for _ in range(10):
                        _ = flash_sample(logits, 0.5, top_k)

        prof.export_chrome_trace(f"flash_sample_{args.compile}.json")
