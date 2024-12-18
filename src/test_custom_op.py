import torch
import flashinfer

with torch.device("cuda"):
    logits = torch.randn((1, 50000), dtype=torch.bfloat16)
    top_k = torch.tensor([10], dtype=torch.int32)

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

assert torch.allclose(
    flashinfer.sampling.top_k_mask_logits(logits, top_k),
    torch.ops.flashinfer.sampling(logits, top_k),
)
torch.compile(torch.ops.flashinfer.sampling, fullgraph=True)(logits, top_k)
