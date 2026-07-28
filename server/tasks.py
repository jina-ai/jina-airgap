"""Task-string vocabulary shared by the embedding families.

The API-level task string carries two pieces of information that different
model families consume in different ways:

1. The base task name (``retrieval``, ``text-matching``, ``code``, ...) picks
   the LoRA adapter (v3, v4) or the omni model's ``default_task``.
   Code-embeddings has no LoRA -- it ignores the base task entirely.
2. The ``.query`` / ``.passage`` suffix picks the prompt prefix that
   sentence-transformers' ``encode()`` prepends, looked up in the model's
   ``prompts`` dict from ``config_sentence_transformers.json``. The actual
   prompt KEY differs per model (v4 uses ``query``/``passage``; v5-omni uses
   ``query``/``document``; code-embeddings uses ``{base}_query`` /
   ``{base}_document``), so it is resolved data-driven by
   :func:`map_prompt_name` -- never a guessed key, which would raise
   ``ValueError: Prompt name 'X' not found in...``.

Each family applies these two rules its own way in ``families.embedding``;
what lives here is the shared vocabulary those rules are written against.
"""

from typing import Optional

# Default task when the caller omits ``task``, keyed by the catalog ``family``.
# Matches prod /v1/embeddings behaviour (probed against api.jina.ai):
#
#   - v5 (text-nano, text-small, omni-nano, omni-small) and v4 ->
#     ``text-matching`` (prod no-task == prod task=text-matching at cos 1.0000
#     for v5-text/omni-nano, 0.9998 for omni-small/v4).
#   - code-embeddings (0.5b / 1.5b) -> ``nl2code.query`` (prod no-task ==
#     prod task=nl2code.query at cos 1.0000).
#   - everything else -> ``retrieval`` (back-compat for v3 / v1 / v2 / b-en-v1).
#
# Without this alignment, no-task local vs prod cos for the affected families
# sits at 0.70 - 0.90 instead of >0.99. (v5-text was previously missing here
# and defaulted to ``retrieval``, giving cos ~0.69 vs prod.)
_DEFAULTS = {
    "embeddings-v4": "text-matching",
    "embeddings-v5": "text-matching",
    "code-embeddings": "nl2code.query",
}

# v3's task vocabulary IS ``retrieval.query`` / ``retrieval.passage``, so the
# suffix is preserved rather than collapsed. Unknown tasks fall back to
# ``retrieval.passage``.
V3_TASKS = {
    "retrieval": "retrieval.passage",
    "retrieval.query": "retrieval.query",
    "retrieval.passage": "retrieval.passage",
    "text-matching": "text-matching",
    "classification": "classification",
    "clustering": "text-matching",
    "separation": "separation",
}

# v5's custom encode only accepts the bare base task, so the .query/.passage
# suffix is collapsed here and carried into ``prompt_name`` instead.
V5_TASKS = {
    "retrieval": "retrieval",
    "retrieval.query": "retrieval",
    "retrieval.passage": "retrieval",
    "text-matching": "text-matching",
    "classification": "classification",
    "clustering": "clustering",
}


def default_task(family: str) -> str:
    return _DEFAULTS.get(family, "retrieval")


def map_prompt_name(task: str, prompts: Optional[dict]) -> Optional[str]:
    """Pick the matching ST prompt_name for ``task`` from the model's prompts dict.

    Different model families use different key conventions:
      - v4: ``{"query": ..., "passage": ...}``
      - v5-omni: ``{"query": ..., "document": ...}``
      - code-embeddings: ``{"nl2code_query": ..., "nl2code_document": ...,
        "qa_query": ..., "code2code_query": ..., ...}``

    Mapping rules:

      - ``{base}.query`` -> first hit of ``{base}_query``, ``query``
      - ``{base}.passage`` -> first hit of ``{base}_document``, ``document``,
        ``passage``
      - no suffix (bare task like ``text-matching``, ``classification``,
        ``retrieval``): prod uses each model's own canonical encode default,
        which differs by family -- v5-omni's ``JinaEmbeddingsV5OmniModel.encode``
        defaults ``prompt_name="document"`` (prepends ``"Document: "``); v4's
        ``JinaEmbeddingsV4Model.encode`` defaults to ``"query"`` (prepends
        ``"Query: "``). We mirror that fallback when the model prompts dict has
        the matching key. Code-embeddings has neither key, return None.
    """
    if not prompts or not task:
        return None
    # v3 keys its prompts dict by the task name itself (e.g.
    # ``"retrieval.passage": "Represent the document for retrieval: "``) rather
    # than the standard ST ``query``/``passage`` keys. Try the literal task
    # first; for other families this is a no-op because their task names -- e.g.
    # ``"retrieval.passage"`` for v4, ``"nl2code.query"`` for code-embeddings --
    # are not present as prompts keys, so the suffix logic below still applies.
    if task in prompts:
        return task
    base, _, suffix = task.partition(".")
    if suffix == "query":
        return next((k for k in (f"{base}_query", "query") if k in prompts), None)
    if suffix == "passage":
        return next(
            (k for k in (f"{base}_document", "document", "passage") if k in prompts),
            None,
        )
    return next((k for k in ("document", "query") if k in prompts), None)
