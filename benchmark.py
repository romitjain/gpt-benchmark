import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import argparse
from dataclasses import dataclass

@dataclass
class Metrics:
    bs: list[int]
    input_toks: list[int]
    output_toks: list[int]
    ttft: list[float]
    throughput: list[float]

class BenchmarkResults:
    def __init__(self):
        self.metrics = {
            'custom_gpt': Metrics(),
            'vllm': Metrics(),
            'tgi': Metrics(),
            'hf': Metrics()
        }

def warmup_gpt_custom():
    pass

def benchmark_gpt_custom():
    pass

def benchmark_vllm():
    pass

def benchmark_tgi():
    pass

def benchmark_hf():
    pass

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

    # input_toks are variable, output_toks are fixed
    input_toks_range = [16, 32, 64, 128, 256, 512]
    output_toks = 512

    model_gpt = warmup_gpt_custom(
        model_id=args.model,
        device=args.device,
        max_input_toks=max(input_toks_range),
        max_output_toks=output_toks,
        prompt=prompt,
        temperature=args.temperature,
        top_k=args.top_k
    )

    for input_toks in input_toks_range:
        ttft, throughput = model_gpt(
            prompt,
            input_toks,
            output_toks,
            temperature=args.temperature,
            top_k=args.top_k
        )
        print(f"Input tokens: {input_toks}, TTF: {ttft:.2f} seconds, throughput: {throughput:.2f} tokens/second")
    # input_toks are fixed, output_toks are variable

    benchmark_gpt_custom(args, prompt)
    benchmark_vllm(args, prompt)
    benchmark_tgi(args, prompt)
