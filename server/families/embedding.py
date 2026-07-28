"""Embedding families: everything served at /v1/embeddings.

They differ only in how a request's ``task`` string reaches the model, so the
subclasses override ``resolve_task`` (and, for v5-omni, how the resolved task
is applied) and nothing else.
"""

import logging
from typing import Optional

import torch

import tasks
from config import settings

from . import Family

logger = logging.getLogger(__name__)


class SentenceTransformerFamily(Family):
    """Stock sentence-transformers load path: b-en-v1, v2, code-embeddings, clip."""

    verb, kind, endpoint = "embed", "embedding", "/v1/embeddings"

    # Passed to the underlying transformer ctor. Only v5-omni needs it: other
    # multimodal models (clip-v1/v2, v4, reranker-m0) use older custom_st code
    # that forwards model_kwargs straight to the model ctor, which rejects
    # unknown kwargs.
    model_kwargs: Optional[dict] = None

    def _load(self) -> None:
        from sentence_transformers import SentenceTransformer

        extra = {"model_kwargs": self.model_kwargs} if self.model_kwargs else {}
        self.model = SentenceTransformer(
            self.spec.hf_repo,
            trust_remote_code=True,
            device=settings.device,
            **extra,
        )

    def load(self) -> None:
        super().load()
        if settings.device == "cuda":
            self._optimize_cuda()

    @property
    def prompts(self) -> Optional[dict]:
        return getattr(self.model, "prompts", None)

    def resolve_task(self, task: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        """Plain models (v1/v2/b-en-v1) and code-embeddings: pass task through.

        Code-embeddings has no LoRA -- its standard sentence_transformers
        Transformer ignores unknown kwargs, so ``task=`` is a no-op and the
        prompt routing happens entirely via prompts/prompt_name. v1/v2 have
        ``accepts_task_kwarg`` False, so encode never sees ``task=`` at all.
        """
        task = task or tasks.default_task(self.spec.family)
        return task, tasks.map_prompt_name(task, self.prompts)

    def encode(
        self,
        inputs: list,
        task: Optional[str],
        prompt_name: Optional[str],
        *,
        normalized: bool = True,
        extra: Optional[dict] = None,
    ):
        """Prompt routing applies whenever the model exposes a matching prompts
        entry: v5-omni "Query: "/"Document: " (without it, last-token pooling on
        the LLaMA/Qwen text tower drifts -- cos ~0.16 on nano); v4
        "Query: "/"Passage: " on the retrieval/code LoRAs; code-embeddings
        "Find the most relevant ..."/"Candidate ..." per task family.
        """
        kwargs = {"convert_to_numpy": True, "normalize_embeddings": normalized}
        kwargs.update(self._task_kwargs(task))
        kwargs.update(extra or {})
        if kwargs.get("return_multivector"):
            # Per-token output is ragged -- one matrix per input, each with its
            # own token count -- so there is nothing for numpy to stack.
            kwargs["convert_to_numpy"] = False
        # Only forward prompt_name when every input is a string: ST prepends
        # ``prompts[prompt_name]`` to each input verbatim and would raise on
        # PIL.Image / BytesIO / fused tuple items. Pure-multimodal calls keep
        # working via the model's internal default.
        if prompt_name is not None and all(isinstance(x, str) for x in inputs):
            kwargs["prompt_name"] = prompt_name
        return self.model.encode(inputs, **kwargs)

    def _task_kwargs(self, task: Optional[str]) -> dict:
        """v3/v4/code-embeddings (ST 3.4+ with ``**kwargs``) take the task via
        ``encode(task=...)``. ``task is None`` -- v3 with no task -- means "skip
        the kwarg" so encode falls through to its native no-task path."""
        if self.accepts_task_kwarg and task is not None:
            return {"task": task}
        return {}

    def _optimize_cuda(self) -> None:
        if settings.dtype in ("float16", "fp16", "half"):
            self.model.half()
            logger.info("Model converted to FP16 (JINA_DTYPE=float16)")
        elif settings.dtype in ("bfloat16", "bf16"):
            self.model.bfloat16()
            logger.info("Model converted to BF16 (JINA_DTYPE=bfloat16)")
        else:
            logger.info(f"Running in FP32 (JINA_DTYPE={settings.dtype})")

        # torch.compile fuses ops for ~10-30% additional speedup, but
        # xlm-roberta-flash models (jina-embeddings-v3 family, jina-clip-* text
        # tower) mutate an internal rotary _cos_cached tensor inside the forward
        # pass; CUDA Graphs (reduce-overhead) sees it as constant and raises
        # "accessing tensor output of CUDAGraphs that has been overwritten by a
        # subsequent run" the first time the captured shape changes (e.g. a
        # different task prompt length). Detect those and skip compile.
        try:
            module = self.model._first_module()
            if not hasattr(module, "auto_model"):
                return
            implementation = getattr(type(module.auto_model), "__module__", "") or ""
            if "xlm-roberta-flash-implementation" in implementation:
                logger.info(
                    "torch.compile skipped: xlm-roberta-flash rotary cache is "
                    "incompatible with CUDA Graphs"
                )
                return
            module.auto_model = torch.compile(
                module.auto_model, mode="reduce-overhead", fullgraph=False
            )
            logger.info("torch.compile(reduce-overhead) applied to encoder")
        except Exception as e:
            logger.warning(f"torch.compile skipped: {e}")


class EmbeddingsV3Family(SentenceTransformerFamily):
    def resolve_task(self, task: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        # v3 no-task: prod runs the raw base xlm-roberta with no LoRA and no
        # prefix when ``task`` is omitted (verified against api.jina.ai --
        # prod_no_task vs prod_task=retrieval.passage cos~0.63). Skip the generic
        # default so we mirror that; otherwise the default ``retrieval`` would
        # map to retrieval.passage below and encode with the retrieval LoRA plus
        # the "Represent the document..." prefix.
        if not task:
            return None, None
        # Resolve the prompt against the MAPPED task, because v3's prompts are
        # keyed by task name. Without this the suffix-less ``retrieval`` alias
        # skips the prompt lookup and encodes without v3's required
        # "Represent the document for retrieval: " prefix -- previously observed
        # as cos~0.92 vs prod for retrieval.passage.
        mapped = tasks.V3_TASKS.get(task, "retrieval.passage")
        return mapped, tasks.map_prompt_name(mapped, self.prompts)


class EmbeddingsV4Family(SentenceTransformerFamily):
    def resolve_task(self, task: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        # v4's config.task_names is ["retrieval", "text-matching", "code"]; the
        # .query/.passage suffix would fail its task validator, so strip to the
        # base. The suffix is carried into prompt_name instead -- v4's prompts
        # dict has "query"/"passage" keys.
        task = task or tasks.default_task(self.spec.family)
        return task.partition(".")[0], tasks.map_prompt_name(task, self.prompts)


class EmbeddingsV5Family(SentenceTransformerFamily):
    def resolve_task(self, task: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        # v5 (text and omni): the custom encode only accepts the bare base task.
        # For omni the .query/.passage suffix is forwarded via prompt_name;
        # v5-text has no prompts map so prompt_name stays None.
        task = task or tasks.default_task(self.spec.family)
        return tasks.V5_TASKS.get(task, "retrieval"), tasks.map_prompt_name(
            task, self.prompts
        )


class EmbeddingsV5OmniFamily(EmbeddingsV5Family):
    """v5-omni-nano / v5-omni-small.

    Their custom_st module picks the LoRA adapter from a ``default_task``
    attribute on the module rather than an ``encode(task=...)`` kwarg, because
    st 3.4.1 does not forward ``task=`` to ``forward()``. It is seeded at load
    so the model always has a valid task, then re-pointed per request.
    """

    model_kwargs = {"default_task": "retrieval"}

    def _load(self) -> None:
        super()._load()
        # Force custom_st._build_eval_image_prompt into its bare-prompt fallback
        # so image inputs emit `<|vision_start|><|image_pad|><|vision_end|>`
        # directly; chat-template wrapping shifts last-token pooling and drops
        # image cos vs api.jina.ai from ~1.0 to ~0.90 (issue #23).
        processor = getattr(self.model[0], "processor", None)
        if processor is None:
            return
        processor.chat_template = None
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.chat_template = None

    def _task_kwargs(self, task: Optional[str]) -> dict:
        for module in self.model.modules():
            if hasattr(module, "default_task"):
                module.default_task = task
                break
        return {}
