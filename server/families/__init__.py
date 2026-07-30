"""Model-family polymorphism: one class per catalog ``runtime`` value.

A family owns three things: how its model loads, how it turns an API task
string into encode kwargs, and how it runs its one verb. ``engine`` owns
everything around that -- lifecycle, wrong-endpoint guards, timing, token
counting -- so adding a model means adding a class here and touching no HTTP
code.

Selection is by ``spec.runtime``, never by matching on the model id: the old
substring predicates made ``jina-reranker-v3`` match ``jina-reranker-v3.5``.
"""

import inspect
import logging
from abc import ABC, abstractmethod
from contextlib import nullcontext

import torch

from config import settings

logger = logging.getLogger(__name__)

# Prompt keys that name a role rather than a task. v4 and v5 key their prompts
# dict by these alone, so they carry no information about which tasks exist.
_PROMPT_ROLES = frozenset({"query", "passage", "document"})


class Family(ABC):
    """One model runtime."""

    verb: str  # which engine verb this model answers: embed / rerank / generate
    kind: str  # human name, rendered into the wrong-endpoint message
    endpoint: str  # where a misrouted caller should have gone instead

    def __init__(self, spec):
        self.spec = spec
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.accepts_task_kwarg = False
        self.cpu_bf16 = False

    @abstractmethod
    def _load(self) -> None:
        """Populate ``self.model`` (and ``self.processor`` where applicable)."""

    def load(self) -> None:
        self._load()
        self._detect_task_kwarg()
        self._load_tokenizer()
        self._finalize_cpu_bf16()

    @property
    def encode_kwargs(self) -> frozenset[str]:
        """Extra ``encode()`` kwargs this loaded model genuinely consumes.

        sentence-transformers routes ``encode(**kwargs)`` to each module's
        ``forward()`` filtered by ``module_kwargs`` -- a map it builds from the
        modules' own signatures -- and silently DROPS anything undeclared. So a
        request for ``late_chunking`` on a model that has no such parameter
        would look like a success and return ordinary embeddings. Reading the
        allowlist is what lets the route refuse truthfully instead.
        """
        declared = getattr(self.model, "module_kwargs", None) or {}
        return frozenset(name for names in declared.values() for name in names)

    @property
    def known_tasks(self) -> frozenset[str]:
        """Base task names this model recognises, read off the model itself.

        Not from the catalog's ``tasks`` list: that field is documentation and
        has drifted -- it says ``techqa`` for code-embeddings where both the
        public API (``qa.query``) and the model's own prompts (``qa_query``)
        say ``qa``.

        Two prompt-key conventions exist and only one of them names tasks. v3
        keys by task (``retrieval.query``), code-embeddings by task and role
        (``nl2code_query``), but v4 and v5 key by role alone (``query`` /
        ``passage`` / ``document``) -- which says nothing about which tasks the
        model has. Reading those as task names rejected every valid task on v4
        and both v5 families, so role words are dropped and the real names come
        off the underlying HF config.

        A PEFT adapter model is a third convention: it names each adapter after
        the task that adapter implements, and its wrapper config carries no
        ``task_names`` at all. v5 loads one adapter per task and switches per
        forward, so those names are its whole task vocabulary and the only place
        it is written down.

        Empty means the model exposes nothing to check against, and the task is
        passed through rather than guessed at.
        """
        names = {
            str(name)
            for name in (getattr(self._auto_config(), "task_names", None) or ())
        }
        names.update(
            str(name) for name in (getattr(self._auto_model(), "peft_config", None) or ())
        )
        for key in getattr(self.model, "prompts", None) or ():
            base = str(key).partition(".")[0].partition("_")[0]
            if base not in _PROMPT_ROLES:
                names.add(base)
        return frozenset(names)

    def _auto_model(self):
        """The transformer ``SentenceTransformer`` wraps, if there is one."""
        first = getattr(self.model, "_first_module", None)
        module = first() if callable(first) else None
        return getattr(module, "auto_model", None)

    def _auto_config(self):
        """The HF config of the underlying transformer.

        ``SentenceTransformer`` has no ``.config`` of its own, so reading one
        off it silently yields nothing.
        """
        return getattr(self._auto_model(), "config", None)

    def autocast(self):
        """Autocast context for encode().

        Honours JINA_DTYPE: float32 -> no autocast (some models -- notably
        jinaai/jina-bert variants with ALiBi -- produce NaN under fp16 due to
        attention-score overflow).
        """
        if settings.device == "cpu":
            # bf16 has fp32's exponent range, so no ALiBi/fp16-style overflow NaN risk.
            if self.cpu_bf16:
                return torch.autocast("cpu", dtype=torch.bfloat16)
            return nullcontext()
        if settings.device != "cuda":
            return nullcontext()
        if settings.dtype in ("float16", "fp16", "half"):
            return torch.autocast("cuda", dtype=torch.float16)
        if settings.dtype in ("bfloat16", "bf16"):
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _detect_task_kwarg(self) -> None:
        """ST 3.4+ exposes ``encode(..., **kwargs)`` and routes them to each
        module's ``forward()`` filtered by its declared kwargs list
        (``custom_st.Transformer`` for v3 / code-embeddings declares
        ``["task"]``, so passing ``task=`` is required for LoRA adapter
        routing). ST 2.7 has no ``**kwargs`` and rejects unknown ones.
        """
        if not hasattr(self.model, "encode"):
            return
        try:
            params = inspect.signature(self.model.encode).parameters
        except (ValueError, TypeError):
            return
        self.accepts_task_kwarg = "task" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        logger.info(f"MODEL.encode() accepts task= kwarg: {self.accepts_task_kwarg}")

    def _load_tokenizer(self) -> None:
        """Best-effort: only used for real tok/s measurement, so a model whose
        tokenizer will not load standalone still serves."""
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.spec.hf_repo, trust_remote_code=True
            )
            logger.info("Tokenizer loaded for real tok/s measurement")
        except Exception as e:
            logger.warning(
                f"Could not load tokenizer ({e}), will use word-split approximation"
            )

    def _finalize_cpu_bf16(self) -> None:
        """Decide whether to use CPU bf16 autocast, once the model is loaded.

        In "auto" mode (default), enable bf16 only when the CPU has native bf16
        support AND a probe encode succeeds with cos-sim >= 0.99 vs fp32 on this
        specific model -- so models that error or degrade under bf16 (e.g.
        xlm-roberta-flash rotary) transparently keep fp32.

        The trick is not universal: EuroBERT (v5) gets ~2-2.5x lossless, but the
        xlm-roberta-flash family (v3, clip text tower) raises an ``index_put``
        dtype mismatch in its rotary path under autocast. JINA_CPU_AUTOCAST=bf16
        forces it on (skip probe), =off forces it off. No effect on GPU/MPS.
        """
        if settings.device != "cpu":
            return
        if settings.cpu_autocast in ("off", "0", "fp32", "float32"):
            logger.info("CPU bf16 autocast: disabled (JINA_CPU_AUTOCAST=off)")
            return
        forced = settings.cpu_autocast in ("bf16", "bfloat16", "on", "1")
        if not forced and not _cpu_supports_bf16():
            logger.info(
                "CPU bf16 autocast: disabled (no native bf16: needs AMX or AVX512_BF16)"
            )
            return
        if forced:
            self.cpu_bf16 = True
            logger.info(
                "CPU bf16 autocast: ENABLED (JINA_CPU_AUTOCAST=bf16, probe skipped)"
            )
            return
        if not hasattr(self.model, "encode"):
            logger.info(
                "CPU bf16 autocast: disabled (model not probe-compatible; "
                "force with JINA_CPU_AUTOCAST=bf16)"
            )
            return
        self.cpu_bf16 = self._probe_bf16()

    def _probe_bf16(self) -> bool:
        import numpy as np

        probe = [
            "The quick brown fox jumps over the lazy dog.",
            "Semantic search finds relevant content by meaning.",
        ]
        base_kwargs = dict(
            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        # Find an encode signature that works in fp32. Task names differ across
        # the fleet (v5: "retrieval", v3: "retrieval.passage", omni: default_task
        # set at load so no kwarg). Only "task must be specified"-type errors
        # advance the fallback; a RuntimeError here is a real failure and aborts
        # the probe.
        reference, chosen = None, None
        for task in ("retrieval", "retrieval.passage", None):
            kwargs = dict(base_kwargs)
            if task is not None and self.accepts_task_kwarg:
                kwargs["task"] = task
            try:
                with torch.inference_mode():
                    reference = self.model.encode(probe, **kwargs)
                chosen = kwargs
                break
            except (ValueError, TypeError):
                continue
            except Exception as e:
                logger.info(
                    f"CPU bf16 autocast: disabled (fp32 probe encode failed: {type(e).__name__})"
                )
                return False
        if chosen is None:
            logger.info(
                "CPU bf16 autocast: disabled (could not find a working probe encode signature)"
            )
            return False

        # Same signature under bf16 autocast: must not error and must stay accurate.
        try:
            with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
                candidate = self.model.encode(probe, **chosen)
            cos = float(
                np.mean(
                    np.sum(reference * candidate, axis=1)
                    / (
                        np.linalg.norm(reference, axis=1)
                        * np.linalg.norm(candidate, axis=1)
                        + 1e-9
                    )
                )
            )
        except Exception as e:
            logger.info(
                f"CPU bf16 autocast: disabled (model errors under bf16: {type(e).__name__})"
            )
            return False
        if cos >= 0.99:
            logger.info(f"CPU bf16 autocast: ENABLED (probe cos-sim={cos:.5f})")
            return True
        logger.info(
            f"CPU bf16 autocast: disabled (probe cos-sim={cos:.5f} < 0.99, keeping fp32)"
        )
        return False


def _cpu_supports_bf16() -> bool:
    """True if the CPU has native bf16 matmul (AMX on Sapphire Rapids+, or
    AVX512_BF16 on Cooper Lake / Zen4). On these, bf16 autocast is a 1.8-2.5x
    throughput win on encoders at batch with cos-sim ~0.9999 vs fp32 (measured
    on jina-embeddings-v5-text-nano). Without the flag, bf16 matmul is emulated
    and can be slower, so we keep fp32 there."""
    try:
        with open("/proc/cpuinfo") as f:
            flags = f.read()
    except Exception:
        return False
    return "amx_bf16" in flags or "avx512_bf16" in flags


def family_for(spec) -> Family:
    """Instantiate the family that serves ``spec``. Imports are deferred so a
    reranker image never pulls in pylate and a CPU image never touches vLLM."""
    from .embedding import (
        EmbeddingsV3Family,
        EmbeddingsV4Family,
        EmbeddingsV5Family,
        EmbeddingsV5OmniFamily,
        SentenceTransformerFamily,
    )
    from .generation import TextChatFamily, VlmFamily
    from .reranking import ColbertFamily, CrossEncoderFamily, JinaRankingFamily

    return {
        "sentence_transformer": SentenceTransformerFamily,
        "embeddings_v3": EmbeddingsV3Family,
        "embeddings_v4": EmbeddingsV4Family,
        "embeddings_v5_text": EmbeddingsV5Family,
        "embeddings_v5_omni": EmbeddingsV5OmniFamily,
        "cross_encoder": CrossEncoderFamily,
        "jina_ranking": JinaRankingFamily,
        "colbert": ColbertFamily,
        "vlm": VlmFamily,
        "text_chat": TextChatFamily,
    }[spec.runtime](spec)
