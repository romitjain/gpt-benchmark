import torch
import triton
import triton.language as tl

@triton.jit
def fused_topk_softmax_sampling_kernel(
    logits_ptr,
    input_batch_stride,
    output_ptr,
    output_batch_stride,
    temp,
    top_k,
    vocab_size,
    BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    block_start = batch_idx * input_batch_stride
    offsets = tl.arange(0, BLOCK_SIZE)

    # Load logits
    logits = tl.load(logits_ptr + block_start + offsets, mask=offsets < vocab_size, other=-float("inf"))

    # Temperature scaling
    logits = tl.div_rn(logits, temp)

    # Top-K selection
    # TODO: Implement fused top-k selection

    # Softmax
    logits = logits - tl.max(logits, axis=0)

    exp_logits = tl.exp(logits)
    probs = exp_logits / tl.sum(exp_logits, axis=0)

    # Sampling
    cumulative_sum = tl.cumsum(probs, axis=0)
    rand = tl.rand()

    # Find the first cumulative sum bin that exceeds U
    sampled_index = tl.where(cumulative_sum >= rand, offsets, vocab_size)
    sampled_index = tl.min(sampled_index, axis=0)  # Find the first index

    if offsets[0] == 0:
        tl.store(output_ptr + batch_idx * output_batch_stride, sampled_index)

def fused_topk_softmax_sampling(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
):
    """
    This function performs the sampling operation using fused kernels.
    """
    assert logits.is_contiguous(), "Logits must be contiguous"
    assert logits.device.type == "cuda", "Logits must be on CUDA"
    assert logits.ndim == 2, "Logits must be 2D, (batch_size, vocab_size)"

    B, V = logits.shape
    out = torch.empty((B, 1), device=logits.device, dtype=torch.int32)

    grid = (B, )
    BLOCK_SIZE = triton.next_power_of_2(V)

    fused_topk_softmax_sampling_kernel[grid](
        logits, logits.stride(0),
        out, out.stride(0),
        max(temperature, 1e-5), top_k, V,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out
