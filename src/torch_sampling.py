import torch
import numpy as np
from typing import Optional
from torch.profiler import profile, record_function, ProfilerActivity

def multinomial_sample_one_no_sync(probs_sort):
    # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

def time_sample(sample_fn):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad():
        logits = torch.randn(1, 10000).to("cuda:1")
        idx_next, probs = sample_fn(logits, 0.5, 10)
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
        # warmup
        logits = torch.randn(1, 10000).to("cuda:1")
        for _ in range(10):
            idx_next, probs = sample(logits, 0.5, 10)

    inf_time = []
    for _ in range(100):
        inf_time.append(time_sample(sample))
    print(f'compile: {args.compile}', np.mean(inf_time))

    logits = torch.randn(1, 10000).to("cuda:1")
    if args.profiling:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_stack=True
        ) as prof:
            with torch.no_grad():
                with record_function("sampling"):
                    for _ in range(5):
                        _, _ = sample(logits, 0.5, 10)

        prof.export_chrome_trace(f"sampling_{args.compile}.json")
