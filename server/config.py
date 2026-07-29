import logging
import os
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    model_id: str
    short_model_id: str
    device: str
    offline: bool
    dtype: str
    cpu_autocast: str
    cpu_threads: int
    license_key: str
    port: int
    batching: bool
    batch_tokens: int
    batch_wait_ms: float


def _detect_device() -> str:
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _init() -> Settings:
    model_id = os.environ.get("JINA_MODEL_ID", "")
    offline = os.environ.get("JINA_OFFLINE", "0") == "1"
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    device = _detect_device()
    logger.info(f"Using device: {device}")

    omp_env = os.environ.get("OMP_NUM_THREADS", "")
    cpu_threads = int(omp_env) if omp_env and omp_env != "0" else (os.cpu_count() or 4)

    # Boost matmul precision: uses TF32 on Ampere/Ada/Hopper GPUs, ~1.2x faster on L4
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(max(1, cpu_threads // 2))

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    return Settings(
        model_id=model_id,
        short_model_id=model_id.split("/")[-1] if model_id else "",
        device=device,
        offline=offline,
        dtype=os.environ.get("JINA_DTYPE", "float16").lower(),
        cpu_autocast=os.environ.get("JINA_CPU_AUTOCAST", "auto").lower(),
        cpu_threads=cpu_threads,
        license_key=os.environ.get("JINA_LICENSE_KEY", ""),
        port=int(os.environ.get("PORT", 8080)),
        # Off unless the image bakes it on. The batching path is the same
        # contract either way -- it only changes how rows are grouped into
        # forwards -- but it is a different concurrency model, so a plain
        # :cpu / :gpu image keeps the one-request-at-a-time behaviour it
        # has always had.
        batching=os.environ.get("JINA_BATCH", "0").lower() in ("1", "on", "true"),
        # 0 means size it against real VRAM at startup rather than guess.
        batch_tokens=int(os.environ.get("JINA_BATCH_TOKENS", "0")),
        batch_wait_ms=float(os.environ.get("JINA_BATCH_WAIT_MS", "5")),
    )


settings = _init()
