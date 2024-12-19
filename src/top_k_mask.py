"""
While this is accurate, it is extremely slow due to:

1. We are depending on tl.sort, which works fast for smaller ranges but extremely slow for larger ranges.
2. We are storing the sorted results back to the HBM and then loading it back to get the topk index

This complete process is pretty slow and can not beat topk of torch.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def top_k_mask_logits_kernel(
    logits_ptr, masked_ptr, intermediate_ptr, n_cols, top_k: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    row_start = row_idx * n_cols
    logits = tl.load(logits_ptr + row_start + col_offsets, mask=col_offsets < n_cols)

    sorted_logits = tl.sort(logits, descending=True)
    tl.store(intermediate_ptr + row_start + col_offsets, sorted_logits, mask=col_offsets < n_cols)
    load_mask = col_offsets > top_k
    topk_threshold = tl.load(intermediate_ptr + row_start + col_offsets, mask=load_mask, other=-float("inf"))
    topk_threshold = tl.max(topk_threshold)

    mask = logits >= topk_threshold
    masked_logits = tl.where(mask, logits, -float("inf"))
    tl.store(masked_ptr + row_start + col_offsets, masked_logits, mask=col_offsets < n_cols)

def top_k_mask_logits(logits, top_k):
    masked = torch.empty_like(logits)
    intermediate = torch.empty_like(logits)

    n_rows, n_cols = logits.shape
    grid = (n_rows,)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    top_k_mask_logits_kernel[grid](
        logits, masked, intermediate, n_cols, top_k, BLOCK_SIZE=BLOCK_SIZE
    )
    return masked

if __name__ == '__main__':
    batch_size, vocab_size = 1, 2000
    logits = torch.randn(batch_size, vocab_size).to(device='cuda:0', dtype=torch.float32)
    top_k = 5

    print(logits, torch.sort(logits, descending=True)[0])
    print('----------------------------')

    masked = top_k_mask_logits(logits, top_k)
    print(masked)
    print('----------------------------')

    print(torch.topk(logits, top_k))
    print('----------------------------')
