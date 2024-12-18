import torch
from functools import wraps
from statistics import mean

def gpu_timer(warmup=25, repeat=100):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Warmup
            for _ in range(warmup):
                func(*args, **kwargs)
            
            # Time multiple runs
            times = []
            for _ in range(repeat):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                torch.cuda.synchronize()
                start.record()
                result = func(*args, **kwargs)
                end.record()
                end.synchronize()
                
                times.append(start.elapsed_time(end))
            
            print(f"{func.__name__}: {mean(times):.4f}ms")

            return result
        return wrapper
    return decorator

# Example:
@gpu_timer()
def sample(logits):
    return torch.nn.functional.softmax(logits, dim=-1)

if __name__ == '__main__':
    logits = torch.randn(1, 10000)
    _ = sample(logits)