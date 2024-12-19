import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, asdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

import gpt

torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

@dataclass
class Metrics:
    bs: list[int] = field(default_factory=list)
    input_toks: list[int] = field(default_factory=list)
    output_toks: list[int] = field(default_factory=list)
    ttft: list[float] = field(default_factory=list)
    throughput: list[float] = field(default_factory=list)
    subset: list[str] = field(default_factory=list)

class BenchmarkResults:
    def __init__(self):
        self.gpt_metrics = Metrics()
        self.hf_metrics = Metrics()
        self.vllm_metrics = Metrics()
        self.tgi_metrics = Metrics()


def benchmark_vllm(model, prompt, runs, sampling_params):
    ttfts = []
    throughputs = []

    with torch.no_grad():
        for r in range(runs):
            out = model.generate(prompt, sampling_params=sampling_params)
            if r != 0:
                ttfts.append(out[0].metrics.first_token_time-out[0].metrics.first_scheduled_time)
                decoding_time = out[0].metrics.finished_time-out[0].metrics.first_token_time
                decoding_tokens = len(out[0].outputs[0].token_ids)
                throughputs.append(decoding_tokens/decoding_time)

    return 1000*sum(ttfts)/len(ttfts), sum(throughputs)/len(throughputs)


def benchmark_tgi():
    pass


def benchmark_hf(model, tokenizer, prompt, runs, max_new_tokens):
    ttfts = []
    throughputs = []
    prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    attention_mask = tokenizer(prompt, return_tensors="pt").attention_mask.to(device)

    with torch.no_grad():
        for r in range(runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            outputs = model(prompt_tokens)
            end_event.record()
            end_event.synchronize()

            first_tok, _ = gpt.sample(
                outputs.logits[:,-1,:],
                temperature=model.generation_config.temperature,
                top_k=model.generation_config.top_k
            )

            ttft = start_event.elapsed_time(end_event) / 1000.0

            new_prompt_tokens = torch.hstack((prompt_tokens, first_tok.unsqueeze(0)))
            new_mask_tokens = torch.hstack((attention_mask, torch.ones((1, 1), device=device)))

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            out = model.generate(
                **{'input_ids': new_prompt_tokens, 'attention_mask': new_mask_tokens},
                max_new_tokens=max_new_tokens,
            )
            end_event.record()
            end_event.synchronize()

            inference_time = start_event.elapsed_time(end_event) / 1000.0
            throughput = (out.shape[-1] - prompt_tokens.shape[-1]) / inference_time

            if r != 0:
                ttfts.append(ttft)
                throughputs.append(throughput)

    return 1000*sum(ttfts)/len(ttfts), sum(throughputs)/len(throughputs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def bench():
    args = parse_args()
    global device
    device = args.device
    # Set the environment variable to specify the GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[1]
    print(f"Using GPU: {args.device}, {os.environ['CUDA_VISIBLE_DEVICES']}")

    with open("long_prompt.txt", "r") as f:
        prompt = f.read()

    num_paras = prompt.count("\n\n")
    print(f"Number of paragraphs: {num_paras}")

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    prompt_tokens = tokenizer(prompt, return_tensors="pt")
    warmup_tokens = prompt_tokens.input_ids[0, :512]
    warmup_prompt = tokenizer.decode(warmup_tokens)
    warmup_prompt_copy = tokenizer.decode(warmup_tokens)
    warmup_tokens = warmup_tokens.unsqueeze(0).to(device)
    warmup_mask = prompt_tokens.attention_mask[:, :512].to(device)

    ## HF
    print('Loading HF model...')
    hf_model = AutoModelForCausalLM.from_pretrained(
        "openai-community/gpt2",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    ).to(device=device)

    hf_model.generation_config.temperature = args.temperature
    hf_model.generation_config.top_k = args.top_k

    hf_model.eval()
    hf_model = torch.compile(hf_model)

    with torch.no_grad():
        _ = hf_model(warmup_tokens)
        _ = hf_model.generate(
            **{'input_ids': warmup_tokens, 'attention_mask': warmup_mask},
            max_new_tokens=512,
            do_sample=True
        )

    ## Custom GPT
    print("Loading Custom GPT model...")
    # topk = torch.tensor([args.top_k], dtype=torch.int32, device=device)
    gpt_gen = gpt.warmup(
        model=args.model,
        prompt=warmup_prompt_copy,
        max_new_tokens=512,
        temperature=args.temperature,
        top_k=args.top_k,
        compile=True,
        profiling=False,
        device=device,
    )

    ## vLLM
    print('Loading vLLM model...')
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=512, top_k=args.top_k)
    vllm_model = LLM(
        "openai-community/gpt2",
        gpu_memory_utilization=0.25,
        device=device,
        dtype=torch.bfloat16
    )
    with torch.no_grad():
        for _ in range(5):
            _ = vllm_model.generate(warmup_prompt, sampling_params=sampling_params)

    bench_results = BenchmarkResults()

    # input_toks are variable, output_toks are fixed
    input_toks_range = [i for i in range(16, 992, 32)]
    output_toks = 32
    runs = 5

    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=output_toks, top_k=args.top_k)

    for input_toks in input_toks_range:
        fixed_prompt = tokenizer.decode(prompt_tokens.input_ids[0, :input_toks])
        print(f"Running GPT benchmark for input tokens: {input_toks}, output tokens: {output_toks}")

        # ttft, throughput = benchmark_hf(hf_model, tokenizer, fixed_prompt, runs, output_toks)

        # bench_results.hf_metrics.bs.append(args.batch_size)
        # bench_results.hf_metrics.input_toks.append(input_toks)
        # bench_results.hf_metrics.output_toks.append(output_toks)
        # bench_results.hf_metrics.ttft.append(ttft)
        # bench_results.hf_metrics.throughput.append(throughput)
        # bench_results.hf_metrics.subset.append("ttft")

        ttft, throughput = gpt_gen(prompt=fixed_prompt, runs=runs, output_toks=output_toks)

        bench_results.gpt_metrics.bs.append(args.batch_size)
        bench_results.gpt_metrics.input_toks.append(input_toks)
        bench_results.gpt_metrics.output_toks.append(output_toks)
        bench_results.gpt_metrics.ttft.append(ttft)
        bench_results.gpt_metrics.throughput.append(throughput)
        bench_results.gpt_metrics.subset.append("ttft")
        print(ttft, throughput)

        # ttft, throughput = benchmark_vllm(vllm_model, fixed_prompt, runs, sampling_params)

        # bench_results.vllm_metrics.bs.append(args.batch_size)
        # bench_results.vllm_metrics.input_toks.append(input_toks)
        # bench_results.vllm_metrics.output_toks.append(output_toks)
        # bench_results.vllm_metrics.ttft.append(ttft)
        # bench_results.vllm_metrics.throughput.append(throughput)
        # bench_results.vllm_metrics.subset.append("ttft")

    # input toks are fixed, output toks are variable
    input_toks = 32
    output_toks_range = [i for i in range(16, 992, 32)]
    fixed_prompt = tokenizer.decode(prompt_tokens.input_ids[0, :input_toks])

    for output_toks in output_toks_range:
        print(f"Running GPT benchmark for input tokens: {input_toks}, output tokens: {output_toks}")

        # ttft, throughput = benchmark_hf(hf_model, tokenizer, fixed_prompt, runs, output_toks)

        # bench_results.hf_metrics.bs.append(args.batch_size)
        # bench_results.hf_metrics.input_toks.append(input_toks)
        # bench_results.hf_metrics.output_toks.append(output_toks)
        # bench_results.hf_metrics.ttft.append(ttft)
        # bench_results.hf_metrics.throughput.append(throughput)
        # bench_results.hf_metrics.subset.append("throughput")

        ttft, throughput = gpt_gen(prompt=fixed_prompt, runs=runs, output_toks=output_toks)

        bench_results.gpt_metrics.bs.append(args.batch_size)
        bench_results.gpt_metrics.input_toks.append(input_toks)
        bench_results.gpt_metrics.output_toks.append(output_toks)
        bench_results.gpt_metrics.ttft.append(ttft)
        bench_results.gpt_metrics.throughput.append(throughput)
        bench_results.gpt_metrics.subset.append("throughput")
        print(ttft, throughput)

        # sampling_params = SamplingParams(temperature=args.temperature, max_tokens=output_toks, top_k=args.top_k)
        # ttft, throughput = benchmark_vllm(vllm_model, fixed_prompt, runs, sampling_params)

        # bench_results.vllm_metrics.bs.append(args.batch_size)
        # bench_results.vllm_metrics.input_toks.append(input_toks)
        # bench_results.vllm_metrics.output_toks.append(output_toks)
        # bench_results.vllm_metrics.ttft.append(ttft)
        # bench_results.vllm_metrics.throughput.append(throughput)
        # bench_results.vllm_metrics.subset.append("throughput")

    gpt_df = pd.DataFrame(asdict(bench_results.gpt_metrics))
    gpt_df["model"] = "gpt"

    hf_df = pd.DataFrame(asdict(bench_results.hf_metrics))
    hf_df["model"] = "hf"

    vllm_df = pd.DataFrame(asdict(bench_results.vllm_metrics))
    vllm_df["model"] = "vllm"

    results_df = pd.concat([gpt_df, hf_df, vllm_df], ignore_index=True)
    print(results_df)
    results_df.to_csv('results.csv', index=False)

    return results_df

def plot_results(df):
    plt.figure(figsize=(15, 10))

    plt.subplot(1, 2, 1)
    for model in df[df['subset'] == 'ttft']['model'].unique():
        model_data = df[(df['subset'] == 'ttft') & (df['model'] == model)]
        plt.plot(model_data['input_toks'], model_data['ttft'], marker='o', label=model)

    plt.title('Time to First Token (TTFT) by Input Tokens (BS = 1, output tokens = 32)', fontsize=14)
    plt.xlabel('Input Tokens', fontsize=12)
    plt.ylabel('Time to First Token (ms)', fontsize=12)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    plt.subplot(1, 2, 2)
    df.loc[:, 'throughput'] = df['throughput'].apply(lambda x: 4000 if x > 4000 else x)

    for model in df[df['subset'] == 'throughput']['model'].unique():
        model_data = df[(df['subset'] == 'throughput') & (df['model'] == model)]
        # Clip values above 4000
        plt.plot(model_data['output_toks'], model_data['throughput'], marker='o', label=model)

    plt.title('Throughput by Output Tokens (BS = 1, input tokens = 32)', fontsize=14)
    plt.xlabel('Output Tokens', fontsize=12)
    plt.ylabel('Throughput (tokens/s) (Values above 4000 are clipped)', fontsize=12)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('performance_plots.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    df = bench()
    plot_results(df)
