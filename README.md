# indri-benchmark

Making small indri as fast as possible

This is an effort at making small models run as fast as possible! Here, running faster means running at highest throughput possible for the batch size of 1 and lowest time to fist token.

- Baseline numbers on 4090 and A100
- Why is it so slow?
  - What is the expected?
  - Why is it lower than expected?
  - How do verify your hunch?
- Special case with smaller models vs larger models
  - Why?
- How to go about speeding it up?
  - Remove dead code
  - CUDA graphs
  - Sampling with custom kernels
  - Logit processor with custom kernels
- Final results


## Baseline numbers

### 4090

Let's first evaluate what speed do we get for small models on consumer grade GPUs. For this test, I am selecting Nvidia RTX 4090 as the GPU and gpt2 (smallest size) as the model.

For simplicity, I am selecting batchsize as 1.

## Why is it slow?

The model is running on an average at x tok/s. Let's do some quick math to understand what should be the theoretical speed of running such a model on 4090 GPU.

<!-- some math to prove it should run at >3k tok/s -->

So what's the reason for this model running slower than expected?

