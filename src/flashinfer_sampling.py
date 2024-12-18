import torch
import numpy as np
import flashinfer
from typing import Optional
from torch.profiler import profile, record_function, ProfilerActivity

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

def multinomial_sample_one_no_sync(probs_sort):
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        # logits = flashinfer.sampling.top_k_mask_logits(logits, top_k)
        logits = torch.ops.flashinfer.sampling(logits, top_k)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits, temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)

    return idx_next, probs

def time_sample(sample_fn, logits, top_k):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad():
        idx_next, probs = sample_fn(logits, 0.5, top_k)
    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--profiling", action="store_true")
    args = parser.parse_args()

    if args.compile:
        sample = torch.compile(sample, mode="reduce-overhead", fullgraph=True)

    top_k = torch.tensor([10], dtype=torch.int32, device='cuda')
    # warmup
    for _ in range(10):
        logits = torch.randn(1, 50000).to("cuda")
        idx_next, probs = sample(logits, 0.5, top_k)

    inf_time = []

    for _ in range(100):
        logits = torch.randn(1, 50000).to("cuda")
        inf_time.append(time_sample(sample, logits, top_k))
    print(f'compile: {args.compile}', np.mean(inf_time))

    if args.profiling:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_stack=True
        ) as prof:
            logits = torch.randn(1, 50000).to("cuda")
            with torch.no_grad():
                with record_function("sampling"):
                    for _ in range(5):
                        _, _ = sample(logits, 0.5, 10)

        prof.export_chrome_trace(f"sampling_{args.compile}.json")
