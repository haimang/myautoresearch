import mlx.core as mx

def get_peak_memory_mb() -> float:
    return mx.metal.get_peak_memory() / (1024 * 1024) if hasattr(mx.metal, "get_peak_memory") else mx.get_peak_memory() / (1024 * 1024)

def clear_cache():
    if hasattr(mx.metal, "clear_cache"):
        mx.metal.clear_cache()
    else:
        mx.clear_cache()
