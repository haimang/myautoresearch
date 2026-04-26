import mlx.core as mx

def get_peak_memory_mb() -> float:
    return mx.get_peak_memory() / (1024 * 1024)

def clear_cache():
    mx.clear_cache()
