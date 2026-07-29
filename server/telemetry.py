"""Throughput accounting: the /health block, and the per-request headers.

Timing is on-prem-only -- api.jina.ai does not report it -- so it travels in
`X-Jina-*` response headers and never in the body. That is what lets every
Jina-native response byte-match the public API with no carve-outs, while an
operator with no metrics stack can still read throughput off a curl.
"""

import contextvars
import threading

# Set by the middleware before the endpoint runs, mutated by record(). A box
# rather than a plain value because Starlette runs the endpoint in a child
# task: the context is copied in, so a variable set afterwards would not be
# visible here, but mutating the object it already holds is.
_measure: contextvars.ContextVar[dict] = contextvars.ContextVar("jina_measure")

_lock = threading.Lock()
_stats = {
    "total_requests": 0,
    "total_tokens": 0,
    "total_latency_s": 0.0,
    "last_tok_per_s": 0.0,
    "peak_tok_per_s": 0.0,
    "forwards": 0,
    "batched_rows": 0,
    "peak_batch_rows": 0,
}


def begin() -> dict:
    """Open a per-request measurement box and hand it to the caller."""
    box: dict = {}
    _measure.set(box)
    return box


def record(n_tokens: int, elapsed_s: float) -> float:
    """Fold one inference call into the running stats; return its tok/s."""
    tok_per_s = n_tokens / elapsed_s if elapsed_s > 0 else 0.0
    box = _measure.get(None)
    if box is not None:
        box["tok_per_s"] = tok_per_s
        box["elapsed_ms"] = elapsed_s * 1000
    with _lock:
        _stats["total_requests"] += 1
        _stats["total_tokens"] += n_tokens
        _stats["total_latency_s"] += elapsed_s
        _stats["last_tok_per_s"] = tok_per_s
        _stats["peak_tok_per_s"] = max(_stats["peak_tok_per_s"], tok_per_s)
    return tok_per_s


def record_forward(rows: int) -> None:
    """Fold one forward pass into the running stats.

    Separate from ``record`` because the two count different things: a request
    is what a caller sent, a forward is what the GPU actually ran. Under
    batching one forward serves many requests, and the ratio between them is
    the only way to tell a batcher that is coalescing from one that is merely
    running every request through a worker thread one at a time. Both look
    identical in tok/s until the box is loaded.
    """
    with _lock:
        _stats["forwards"] += 1
        _stats["batched_rows"] += rows
        _stats["peak_batch_rows"] = max(_stats["peak_batch_rows"], rows)


def throughput() -> dict | None:
    """The /health throughput block, or None before the first request."""
    with _lock:
        snapshot = dict(_stats)
    if not snapshot["total_requests"]:
        return None
    latency = snapshot["total_latency_s"]
    average = snapshot["total_tokens"] / latency if latency > 0 else None
    block = {
        "total_requests": snapshot["total_requests"],
        "total_tokens": snapshot["total_tokens"],
        "last_tok_per_s": round(snapshot["last_tok_per_s"], 1),
        "avg_tok_per_s": round(average, 1) if average else None,
        "peak_tok_per_s": round(snapshot["peak_tok_per_s"], 1),
    }
    forwards = snapshot["forwards"]
    if forwards:
        # Only the embedding path runs through the batcher, so a reranker or a
        # chat image reports no forwards and these keys stay absent rather than
        # claiming a batch size of zero.
        block["forwards"] = forwards
        block["avg_batch_rows"] = round(snapshot["batched_rows"] / forwards, 1)
        block["peak_batch_rows"] = snapshot["peak_batch_rows"]
    return block
