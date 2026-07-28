"""Reranking families: everything served at /v1/rerank."""

import logging
from abc import abstractmethod
from typing import Any, Optional

import torch

from config import settings

from . import Family

logger = logging.getLogger(__name__)


def document_text(document: Any) -> str:
    """A request document is either a bare string or ``{"text": ...}``."""
    return document if isinstance(document, str) else document.get("text", "")


class RerankFamily(Family):
    verb, kind, endpoint = "rerank", "reranker", "/v1/rerank"

    @abstractmethod
    def rank(
        self, query: str, documents: list[str], top_n: Optional[int]
    ) -> list[tuple[int, float]]:
        """Score ``documents`` against ``query``.

        Returns ``(original_index, relevance_score)`` pairs, already sorted
        best-first with ``top_n`` applied.
        """

    def render_document(self, document: Any) -> Any:
        """The shape of ``results[].document``: echo what the caller sent.

        Measured on api.jina.ai, this is per-family and not a global rule. The
        CrossEncoder family mirrors the caller's type -- a bare string in gives
        a bare string out, ``{"text": ...}`` in gives an object out -- so the
        base class echoes and ``JinaRankingFamily`` overrides. ColBERT has no
        public rerank oracle (the public API also lists it under embeddings) and
        is treated as CrossEncoder-family by analogy.
        """
        return document


class CrossEncoderFamily(RerankFamily):
    """sentence-transformers CrossEncoder: reranker v1 (tiny/turbo/base),
    v2-base-multilingual, m0."""

    def _load(self) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(
            self.spec.hf_repo, trust_remote_code=True, device=settings.device
        )
        tokenizer = self.model.tokenizer
        if tokenizer.pad_token is None:
            # qwen3-based rerankers need a pad token for batched inference.
            tokenizer.pad_token = tokenizer.eos_token
            if hasattr(self.model.model, "config"):
                self.model.model.config.pad_token_id = tokenizer.eos_token_id
            logger.info("Set pad_token = eos_token for reranker")

    def rank(
        self, query: str, documents: list[str], top_n: Optional[int]
    ) -> list[tuple[int, float]]:
        # convert_to_tensor=True keeps the model's native dtype (e.g. bf16 for
        # jina-reranker-v2-base-multilingual); cast to fp32 before numpy because
        # numpy has no bf16 dtype.
        scores = self.model.predict(
            [[query, document] for document in documents],
            convert_to_numpy=False,
            convert_to_tensor=True,
        )
        if hasattr(scores, "float"):
            scores = scores.float().detach().cpu().numpy()
        ranked = sorted(
            ((i, float(s)) for i, s in enumerate(scores)),
            key=lambda scored: scored[1],
            reverse=True,
        )
        return ranked[:top_n] if top_n else ranked


class JinaRankingFamily(RerankFamily):
    """jina-reranker-v3 / v3.5: Qwen3-based listwise rerankers with a custom
    JinaForRanking class (auto_map.AutoModel -> modeling.JinaForRanking, a
    Qwen3ForCausalLM subclass with a 1024->512->256 MLP projector) exposing its
    own ``.rerank(query, documents)``. They are NOT sentence-transformers
    CrossEncoders. v3.5 is a drop-in upgrade: same class, same interface,
    per-item truncation baked into its own rerank().
    """

    def render_document(self, document: Any) -> Any:
        """Always an object, whatever the caller sent -- measured on
        api.jina.ai for both v3 and v3.5, with string and object input."""
        return {"text": document_text(document)}

    def _load(self) -> None:
        from transformers import AutoModel

        # Native dtype is bf16; keep it on cuda, use fp32 on cpu because generic
        # x86_64 has no bf16 SIMD.
        dtype = torch.bfloat16 if settings.device == "cuda" else torch.float32
        # Pinned, not left to the transformers default: eager attention holds the
        # whole (heads, N, N) score matrix, which at this model's 16 heads is
        # 34 GB for a 32k-token block and 137 GB for a 64k one -- and listwise
        # reranking packs documents until the block fills. sdpa never
        # materialises it. The default resolves to sdpa today, but "today"
        # spans transformers 4.44 to 5.7 across this catalogue, and a silent
        # fallback would surface as an unexplained OOM rather than a config
        # mistake. Logged below for the same reason.
        model = AutoModel.from_pretrained(
            self.spec.hf_repo,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        self.model = model.to(settings.device).eval()
        attn = getattr(model.config, "_attn_implementation", "unknown")
        logger.info(
            f"Loaded as JinaForRanking: {self.spec.hf_repo} dtype={dtype} attn={attn}"
        )

    def rank(
        self, query: str, documents: list[str], top_n: Optional[int]
    ) -> list[tuple[int, float]]:
        # Its own block-wise listwise rerank() already returns documents sorted
        # by relevance_score descending with top_n applied -- no manual sort.
        return [
            (int(item["index"]), float(item["relevance_score"]))
            for item in self.model.rerank(query, documents, top_n=top_n)
        ]


class ColbertFamily(RerankFamily):
    kind = "ColBERT (late-interaction)"

    def _load(self) -> None:
        from pylate import models as pylate_models

        # PyLate wraps sentence-transformers and handles ColBERT-specific Q/D
        # markers, query expansion, and the 128-dim projection head.
        self.model = pylate_models.ColBERT(
            model_name_or_path=self.spec.hf_repo,
            trust_remote_code=True,
            device=settings.device,
        )

    def rank(
        self, query: str, documents: list[str], top_n: Optional[int]
    ) -> list[tuple[int, float]]:
        # Late interaction: encode query and documents as token-level
        # multi-vectors and score via MaxSim. pylate.rank.rerank takes a
        # nested-by-query layout (one inner list per query); with a single query
        # we pass a 1-element outer list and pop result[0] back out.
        # NOTE: encode() with a nested-list documents arg internally
        # torch.stack()s per-doc embeddings without padding and crashes on
        # variable-length docs (jina-colbert returns (n_tokens, 128) per doc,
        # n_tokens varies). So encode documents as a flat list and wrap the
        # returned list-of-tensors for rerank.
        from pylate import rank as pylate_rank

        queries_embeddings = self.model.encode(
            [query], is_query=True, convert_to_tensor=True
        )
        documents_embeddings = self.model.encode(
            documents, is_query=False, convert_to_tensor=True
        )
        ranked = pylate_rank.rerank(
            documents_ids=[list(range(len(documents)))],
            queries_embeddings=queries_embeddings,
            documents_embeddings=[documents_embeddings],
        )
        scored = [(int(item["id"]), float(item["score"])) for item in ranked[0]]
        return scored[:top_n] if top_n else scored
