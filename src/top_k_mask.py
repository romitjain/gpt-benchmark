import torch
import triton
import triton.language as tl

@triton.jit
def top_k_mask_logits_kernel(
    logits_ptr, masked_ptr, top_k, n_cols, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    row_start = row_idx * n_cols
    logits = tl.load(logits_ptr + row_start + col_offsets, mask=col_offsets < n_cols)
    topk_values = tl.sort(logits, descending=True)[:top_k]
    threshold = topk_values[-1]

    mask = logits >= threshold
    masked_logits = tl.where(mask, logits, -float("inf"))
    tl.store(masked_ptr + row_start + col_offsets, masked_logits, mask=col_offsets < n_cols)

def top_k_mask_logits(logits, top_k):
    logits = logits.contiguous()
    masked = torch.empty_like(logits)
    n_rows, n_cols = logits.shape
    grid = (n_rows,)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    top_k_mask_logits_kernel[grid](
        logits, masked, top_k, n_cols, BLOCK_SIZE=BLOCK_SIZE
    )
    return masked
