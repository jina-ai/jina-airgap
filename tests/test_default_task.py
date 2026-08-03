"""
Unit tests for _default_task() — the no-task embedding default per model family.

Asserts on-prem parity with prod api.jina.ai defaults. No server, no network,
no model weights (imports the app module, which pulls torch/transformers, but
never loads a model).

Run:
  python tests/test_default_task.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
try:
    import app  # noqa: E402
except ImportError as e:
    # server/app.py pulls torch and transformers. Under pytest, skip rather than
    # break collection for the whole suite when they are not installed.
    if "pytest" not in sys.modules:
        raise
    import pytest

    pytest.skip(f"server deps not installed ({e})", allow_module_level=True)

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

_results = []


def check(name, cond):
    _results.append(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def default_for(short_id):
    app.MODEL_ID = f"jinaai/{short_id}" if short_id else ""
    return app._default_task()


def main():
    print("_default_task() unit tests\n")
    original = app.MODEL_ID
    try:
        # --- the fix: v5-text must default to text-matching (was retrieval) ---
        check("v5-text-nano -> text-matching",
              default_for("jina-embeddings-v5-text-nano") == "text-matching")
        check("v5-text-small -> text-matching",
              default_for("jina-embeddings-v5-text-small") == "text-matching")

        # --- regression guards: text-matching family stays put ---
        check("v5-omni-nano -> text-matching",
              default_for("jina-embeddings-v5-omni-nano") == "text-matching")
        check("v5-omni-small -> text-matching",
              default_for("jina-embeddings-v5-omni-small") == "text-matching")
        check("v4 -> text-matching",
              default_for("jina-embeddings-v4") == "text-matching")

        # --- code-embeddings default ---
        check("code-embeddings-0.5b -> nl2code.query",
              default_for("jina-code-embeddings-0.5b") == "nl2code.query")
        check("code-embeddings-1.5b -> nl2code.query",
              default_for("jina-code-embeddings-1.5b") == "nl2code.query")

        # --- everything else falls through to retrieval ---
        check("v3 -> retrieval",
              default_for("jina-embeddings-v3") == "retrieval")
        check("v2-base-en -> retrieval",
              default_for("jina-embeddings-v2-base-en") == "retrieval")
        check("empty MODEL_ID -> retrieval", default_for("") == "retrieval")
    finally:
        app.MODEL_ID = original

    print()
    total, passed = len(_results), sum(_results)
    print(f"{PASS if passed == total else FAIL}: {passed}/{total}")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
