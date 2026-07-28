"""Throughput accounting behind /health."""

import threading

_lock = threading.Lock()
_stats = {
    "total_requests": 0,
    "total_tokens": 0,
    "total_latency_s": 0.0,
    "last_tok_per_s": 0.0,
    "peak_tok_per_s": 0.0,
}


def record(n_tokens: int, elapsed_s: float) -> float:
    """Fold one inference call into the running stats; return its tok/s."""
    tok_per_s = n_tokens / elapsed_s if elapsed_s > 0 else 0.0
    with _lock:
        _stats["total_requests"] += 1
        _stats["total_tokens"] += n_tokens
        _stats["total_latency_s"] += elapsed_s
        _stats["last_tok_per_s"] = tok_per_s
        _stats["peak_tok_per_s"] = max(_stats["peak_tok_per_s"], tok_per_s)
    return tok_per_s


def throughput() -> dict | None:
    """The /health throughput block, or None before the first request."""
    with _lock:
        snapshot = dict(_stats)
    if not snapshot["total_requests"]:
        return None
    latency = snapshot["total_latency_s"]
    average = snapshot["total_tokens"] / latency if latency > 0 else None
    return {
        "total_requests": snapshot["total_requests"],
        "total_tokens": snapshot["total_tokens"],
        "last_tok_per_s": round(snapshot["last_tok_per_s"], 1),
        "avg_tok_per_s": round(average, 1) if average else None,
        "peak_tok_per_s": round(snapshot["peak_tok_per_s"], 1),
    }
