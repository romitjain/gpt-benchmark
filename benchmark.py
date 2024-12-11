import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch
import argparse
import pandas as pd
from typing import Optional
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

import gpt

device = 'cuda:0'

@dataclass
class Metrics:
    bs: list[int] = field(default_factory=list)
    input_toks: list[int] = field(default_factory=list)
    output_toks: list[int] = field(default_factory=list)
    ttft: list[float] = field(default_factory=list)
    throughput: list[float] = field(default_factory=list)

class BenchmarkResults:
    def __init__(self): 
        self.gpt_metrics = Metrics()
        self.hf_metrics = Metrics()
        self.vllm_metrics = Metrics()
        self.tgi_metrics = Metrics()

def benchmark_vllm(
        model,
        prompt,
        sampling_params
    ):
    with torch.no_grad():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
    
        start_event.record()
        out = model.generate(
            prompt,
            sampling_params=sampling_params
        )
        end_event.record()
        torch.cuda.synchronize()

        print(out.shape)
        inference_time = start_event.elapsed_time(end_event) / 1000.0

    return (out.shape[-1]-prompt.shape[-1])/inference_time

def benchmark_tgi():
    pass

def benchmark_hf(
        model,
        tokenizer,
        prompt,
        max_new_tokens
    ):
    with torch.no_grad():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        outputs = model(prompt)
        end_event.record()
        torch.cuda.synchronize()

        import pdb; pdb.set_trace()

        ttft = start_event.elapsed_time(end_event) / 1000.0

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        out = model.generate(
            prompt+tokenizer.decode(outputs[-1]),
            max_new_tokens=max_new_tokens,
        )
        end_event.record()
        torch.cuda.synchronize()

        print(out.shape)
        inference_time = start_event.elapsed_time(end_event) / 1000.0

    return ttft, (out.shape[-1]-prompt.shape[-1])/inference_time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()

def main():
    args = parse_args()

    with open('long_prompt.txt', 'r') as f:
        prompt = f.read()

    num_paras = prompt.count('\n\n')
    print(f"Number of paragraphs: {num_paras}")

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    prompt_tokens = tokenizer(prompt, return_tensors='pt')
    warmup_tokens = prompt_tokens.input_ids[0, :512]
    warmup_prompt = tokenizer.decode(warmup_tokens)
    warmup_tokens = warmup_tokens.unsqueeze(0).to(device)

    ## HF
    # print('Loading HF model...')
    # hf_model = AutoModelForCausalLM.from_pretrained(
    #     "openai-community/gpt2",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="sdpa"
    # ).to(device=device)

    # hf_model.generation_config.temperature = args.temperature
    # hf_model.generation_config.top_k = args.top_k

    # hf_model.eval()
    # hf_model = torch.compile(hf_model)

    # for _ in range(5):
    #     with torch.no_grad():
    #         _ = hf_model.generate(
    #             warmup_tokens,
    #             max_new_tokens=512,
    #             do_sample=True
    #         )

    # ## vLLM
    # print('Loading vLLM model...')
    # sampling_params = SamplingParams(temperature=args.temperature, top_k=args.top_k, max_tokens=512)
    # vllm_model = LLM(
    #     "openai-community/gpt2",
    #     gpu_memory_utilization=0.25,
    #     device=device,
    #     dtype=torch.bfloat16
    # )
    # for _ in range(5):
    #     with torch.no_grad():
    #         _ = vllm_model.generate(warmup_prompt, sampling_params=sampling_params)

    ## Custom GPT
    print('Loading Custom GPT model...')
    gpt_gen = gpt.warmup(
        model=args.model,
        prompt=warmup_prompt,
        max_new_tokens=512,
        temperature=args.temperature,
        top_k=args.top_k,
        compile=True,
        profiling=False,
        device=device
    )

    bench_results = BenchmarkResults()

    # input_toks are variable, output_toks are fixed
    input_toks_range = [16, 32, 64, 128, 256, 512]
    output_toks = 512

    for input_toks in input_toks_range:
        fixed_prompt = tokenizer.decode(prompt_tokens.input_ids[0, :input_toks])

        for _ in range(10):
            print(f'Running GPT benchmark for input tokens: {input_toks}, output tokens: {output_toks}')

            ttft, throughput = gpt_gen(prompt=fixed_prompt, output_toks=256)

            bench_results.gpt_metrics.bs.append(args.batch_size)
            bench_results.gpt_metrics.input_toks.append(input_toks)
            bench_results.gpt_metrics.output_toks.append(output_toks)
            bench_results.gpt_metrics.ttft.append(ttft)
            bench_results.gpt_metrics.throughput.append(throughput)

            # throughput = benchmark_hf(
            #     hf_model,
            #     tokenizer,
            #     fixed_prompt,
            #     output_toks
            # )

            # bench_results.hf_metrics.bs.append(args.batch_size)
            # bench_results.hf_metrics.input_toks.append(input_toks)
            # bench_results.hf_metrics.output_toks.append(output_toks)
            # bench_results.hf_metrics.throughput.append(throughput)

            # sampling_params = SamplingParams(temperature=args.temperature, top_k=args.top_k, max_tokens=output_toks)

            # throughput = benchmark_vllm(
            #     vllm_model,
            #     fixed_prompt,
            #     sampling_params
            # )

            # bench_results.vllm_metrics.bs.append(args.batch_size)
            # bench_results.vllm_metrics.input_toks.append(input_toks)
            # bench_results.vllm_metrics.output_toks.append(output_toks)
            # bench_results.vllm_metrics.throughput.append(throughput)

    import pdb; pdb.set_trace()
    bench_df = pd.DataFrame(bench_results)
    print(bench_df)
    bench_df.to_csv('benchmark_results.csv', index=False)

if __name__ == "__main__":
    main()
