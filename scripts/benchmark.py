#!/usr/bin/env python3
"""
Jina v5 Throughput Benchmark
Measures real tok/s for all v5 embedding models on CPU and GPU.
"""

import os
import time
import json
import argparse
import platform
import subprocess
import sys
import tempfile
import shutil
from typing import Optional

# Sentences of varying lengths to simulate real-world mixed input
BENCHMARK_CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Semantic search enables finding relevant content based on meaning rather than exact keyword matching.",
    "In recent years, transformer-based models have revolutionized natural language processing tasks.",
    "Jina AI provides state-of-the-art embedding models optimized for retrieval and semantic similarity.",
    "Air-gapped environments require all dependencies to be bundled and deployed without internet access.",
    "Hello world.",
    "This is a test.",
    "Machine learning models can be deployed on edge devices with limited computational resources.",
    "The NVIDIA L4 GPU offers 30.3 TFLOPS of FP16 Tensor performance in a low-power form factor suitable for inference workloads.",
    "Embeddings are dense vector representations that capture semantic meaning of text.",
    "Fine-tuning large language models requires significant computational resources and carefully curated datasets.",
    "Python is a versatile programming language widely used in data science and machine learning applications.",
    "Docker containers provide a consistent environment for deploying applications across different platforms.",
    "Vector databases enable efficient similarity search over millions of high-dimensional embeddings.",
    "The attention mechanism in transformer models allows the model to focus on relevant parts of the input sequence.",
    "Deep learning has achieved remarkable results in computer vision, speech recognition, and natural language processing.",
    "Retrieval-augmented generation combines the strengths of retrieval systems and generative language models.",
    "OK.",
    "Batch processing improves GPU utilization by processing multiple inputs simultaneously.",
    "The softmax function converts raw logits into probability distributions over the vocabulary.",
    "Quantization reduces model size and inference latency by using lower-precision arithmetic.",
    "Knowledge distillation transfers knowledge from a large teacher model to a smaller student model.",
    "Contrastive learning trains models to embed similar items close together and dissimilar items far apart.",
    "Sparse attention mechanisms reduce the quadratic complexity of standard self-attention.",
    "Multi-modal models process and align information from different modalities such as text and images.",
    "Yes.",
    "No.",
    "The embedding dimension determines the capacity of the model to encode semantic information.",
    "Normalized embeddings have unit length and enable cosine similarity to be computed efficiently.",
    "Cross-encoder models score pairs of texts jointly for higher accuracy at the cost of throughput.",
]

MODELS = [
    ("jinaai/jina-embeddings-v5-text-nano", "239M"),
    ("jinaai/jina-embeddings-v5-text-small", "677M"),
    ("jinaai/jina-embeddings-v5-omni-nano", "1.04B"),
    ("jinaai/jina-embeddings-v5-omni-small", "1.74B"),
]

# Models that need tokenizer_config patching (TokenizersBackend not registered in transformers)
NEEDS_TOKENIZER_PATCH = {
    "jinaai/jina-embeddings-v5-omni-nano",
}


def get_cpu_info():
    try:
        result = subprocess.run(["lscpu"], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if "Model name" in line:
                return line.split(":")[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def get_gpu_info():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "No GPU detected"


def resolve_model_path(model_id: str) -> str:
    """Download model if needed, patch tokenizer if required, return local path."""
    if model_id in NEEDS_TOKENIZER_PATCH:
        from huggingface_hub import snapshot_download

        repo_path = snapshot_download(model_id)
        tc_path = os.path.join(repo_path, "tokenizer_config.json")
        if os.path.exists(tc_path):
            with open(tc_path) as f:
                tc = json.load(f)
            if tc.get("tokenizer_class") == "TokenizersBackend":
                tc["tokenizer_class"] = "PreTrainedTokenizerFast"
                with open(tc_path, "w") as f:
                    json.dump(tc, f, indent=2)
                print(
                    f"  Patched tokenizer_config: TokenizersBackend -> PreTrainedTokenizerFast"
                )
        return repo_path
    return model_id


def apply_optimizations(model, device: str):
    """Apply runtime optimizations to a loaded SentenceTransformer."""
    import torch
    import os

    if device == "cpu":
        n_threads = int(os.environ.get("OMP_NUM_THREADS", "") or os.cpu_count() or 4)
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(max(1, n_threads // 2))
        print(f"  CPU threads: {n_threads}")

    elif device == "cuda":
        torch.backends.cudnn.benchmark = True

        # FP16: halves memory bandwidth, ~1.5-2x throughput
        model.half()
        print("  Applied: FP16 (model.half())")

        # torch.compile: fuses ops, reduce-overhead mode uses CUDA graphs
        try:
            first_module = model._first_module()
            if hasattr(first_module, "auto_model"):
                first_module.auto_model = torch.compile(
                    first_module.auto_model,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
                print("  Applied: torch.compile(reduce-overhead)")
        except Exception as e:
            print(f"  torch.compile skipped: {e}")

    return model


def benchmark_model(
    model_id: str,
    device: str,
    batch_size: int = 32,
    duration_s: float = 10.0,
    optimize: bool = True,
):
    """Benchmark a model, returns tok/s."""
    import torch

    print(f"\n  Loading {model_id} on {device}...")

    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    model_path = resolve_model_path(model_id)

    model = SentenceTransformer(
        model_path,
        trust_remote_code=True,
        device=device,
    )

    if optimize:
        print("  Applying optimizations...")
        model = apply_optimizations(model, device)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"  Warning: tokenizer load failed ({e}), using word-split fallback")
        tokenizer = None

    def count_tokens(texts):
        if tokenizer is not None:
            enc = tokenizer(
                texts, add_special_tokens=True, truncation=True, max_length=512
            )
            return sum(len(ids) for ids in enc["input_ids"])
        return sum(len(t.split()) for t in texts)

    # Autocast context for GPU
    def make_ctx():
        if device == "cuda":
            return torch.autocast("cuda", dtype=torch.float16)
        import contextlib

        return contextlib.nullcontext()

    # Build batches from the corpus (cycle to fill batch)
    corpus = BENCHMARK_CORPUS
    batches = []
    for i in range(0, max(batch_size * 10, len(corpus)), batch_size):
        batch = [corpus[j % len(corpus)] for j in range(i, i + batch_size)]
        batches.append(batch)

    # Warm up: 5 batches (more warmup to let torch.compile settle)
    print(f"  Warming up (5 batches)...")
    for b in batches[:5]:
        with torch.no_grad(), make_ctx():
            model.encode(
                b, task="retrieval", convert_to_numpy=True, normalize_embeddings=True
            )

    # Steady-state measurement: run for exactly duration_s seconds
    print(f"  Measuring for {duration_s}s...")
    total_tokens = 0
    batch_idx = 0
    t_start = time.perf_counter()
    t_end = t_start + duration_s

    while time.perf_counter() < t_end:
        batch = batches[batch_idx % len(batches)]
        with torch.no_grad(), make_ctx():
            model.encode(
                batch,
                task="retrieval",
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        total_tokens += count_tokens(batch)
        batch_idx += 1

    elapsed = time.perf_counter() - t_start
    tok_per_s = total_tokens / elapsed

    avg_tokens = total_tokens / batch_idx if batch_idx > 0 else 0
    print(
        f"  Done: {batch_idx} batches | {total_tokens:,} tokens | {elapsed:.1f}s | {tok_per_s:,.0f} tok/s"
    )
    print(f"  Avg tokens/batch: {avg_tokens:.1f}")

    # Cleanup
    del model
    import gc

    gc.collect()
    try:
        import torch

        if device == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass

    return tok_per_s


def main():
    parser = argparse.ArgumentParser(description="Jina v5 Throughput Benchmark")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Steady-state measurement duration in seconds",
    )
    parser.add_argument("--device", choices=["cpu", "cuda", "both"], default="both")
    parser.add_argument(
        "--models", nargs="+", default=None, help="Subset of models to benchmark"
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable runtime optimizations (baseline mode)",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Test multiple batch sizes (e.g. 32 64 128)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Jina v5 Embedding Throughput Benchmark")
    print("=" * 70)

    cpu_info = get_cpu_info()
    gpu_info = get_gpu_info()
    print(f"CPU: {cpu_info}")
    print(f"GPU: {gpu_info}")
    print(f"Batch size: {args.batch_size}")
    print(f"Duration: {args.duration}s steady-state")

    import torch

    has_gpu = torch.cuda.is_available()
    print(f"CUDA available: {has_gpu}")

    models_to_run = MODELS
    if args.models:
        models_to_run = [(m, p) for m, p in MODELS if any(x in m for x in args.models)]

    results = {}  # model_id -> {"cpu": tok_s, "gpu": tok_s}

    devices = []
    if args.device in ("cpu", "both"):
        devices.append("cpu")
    if args.device in ("cuda", "both") and has_gpu:
        devices.append("cuda")

    optimize = not args.no_optimize
    batch_sizes_to_test = args.batch_sizes if args.batch_sizes else [args.batch_size]

    for model_id, params in models_to_run:
        results[model_id] = {"params": params, "cpu": None, "gpu": None}
        for device in devices:
            result_key = "gpu" if device == "cuda" else "cpu"
            best_tok_s = None
            best_bs = args.batch_size
            for bs in batch_sizes_to_test:
                print(
                    f"\n[{model_id}] ({params}) on {device.upper()} | batch_size={bs} | optimize={optimize}"
                )
                try:
                    tok_s = benchmark_model(
                        model_id, device, bs, args.duration, optimize=optimize
                    )
                    if best_tok_s is None or tok_s > best_tok_s:
                        best_tok_s = tok_s
                        best_bs = bs
                except Exception as e:
                    print(f"  ERROR: {e}")
            results[model_id][result_key] = best_tok_s
            if len(batch_sizes_to_test) > 1:
                print(f"  Best batch size: {best_bs} -> {int(best_tok_s):,} tok/s")

    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # CPU info short
    cpu_short = cpu_info
    if "Intel" in cpu_info:
        parts = cpu_info.split()
        try:
            idx = next(
                (i for i, p in enumerate(parts) if p in ("Platinum", "Gold", "Silver")),
                -1,
            )
            if idx >= 0 and idx + 1 < len(parts):
                cpu_short = f"Intel Xeon {parts[idx]} {parts[idx+1]}"
            elif "@" in cpu_info:
                cpu_short = "Intel Xeon @ " + cpu_info.split("@")[1].strip()
        except Exception:
            pass

    print(f"\nHardware:")
    print(f"  CPU: {cpu_info}")
    print(f"  GPU: {gpu_info}")

    print(f"\n{'Model':<45} {'CPU tok/s':>12} {'GPU tok/s':>12}")
    print("-" * 70)
    for model_id, data in results.items():
        cpu_val = f"{int(data['cpu']):,}" if data["cpu"] is not None else "N/A"
        gpu_val = f"{int(data['gpu']):,}" if data["gpu"] is not None else "N/A"
        short_name = model_id.split("/")[-1]
        print(f"{short_name:<45} {cpu_val:>12} {gpu_val:>12}")

    # Output machine-readable JSON
    output = {
        "hardware": {
            "cpu": cpu_info,
            "gpu": gpu_info,
        },
        "config": {
            "batch_size": args.batch_size,
            "duration_s": args.duration,
        },
        "results": results,
    }
    with open("/tmp/benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to /tmp/benchmark_results.json")

    # Also print raw numbers for README update
    print("\n--- README TABLE FORMAT ---")
    print(f"CPU: {cpu_short} | GPU: NVIDIA L4 (30.3 TFLOPS FP16)")
    print()
    print("| Model | CPU tok/s | GPU tok/s |")
    print("|-------|-----------|-----------|")
    print(f"|       | {cpu_short} | NVIDIA L4 (30.3 TFLOPS FP16) |")
    for model_id, data in results.items():
        short_name = model_id.split("/")[-1]
        cpu_val = f"{int(data['cpu']):,}" if data["cpu"] is not None else "N/A"
        gpu_val = f"{int(data['gpu']):,}" if data["gpu"] is not None else "N/A"
        print(f"| {short_name} | {cpu_val} | {gpu_val} |")


if __name__ == "__main__":
    main()
