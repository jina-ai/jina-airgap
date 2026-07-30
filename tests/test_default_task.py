"""
Unit tests for tasks.default_task() — the no-task embedding default per model family.

Asserts on-prem parity with prod api.jina.ai defaults, resolved through the
catalog exactly as the server does. No server, no network, no model weights.

Run:
  python tests/test_default_task.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
import tasks  # noqa: E402
from catalog import spec_for  # noqa: E402

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

_results = []


def check(name, cond):
    _results.append(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def default_for(short_id):
    return tasks.default_task(spec_for(short_id).family)


def main():
    print("default_task() unit tests\n")

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
    check("unknown family -> retrieval", tasks.default_task("") == "retrieval")

    # --- v5 must not rewrite a task it does not recognise ---
    # check_task rejects it upstream; if anything ever gets past that, the
    # model's own validator has to see the value the caller actually sent.
    # Substituting a valid task here answered `text_matching` -- one wrong
    # character -- with retrieval vectors and a 200.
    v5 = tasks.v5_task
    for bad in ("text_matching", "not-a-real-task", "Retrieval.Query"):
        check(f"v5 leaves {bad!r} for the model to reject", v5(bad) == bad)
    check("v5 collapses .query to the bare task", v5("retrieval.query") == "retrieval")
    check("v5 collapses .passage to the bare task",
          v5("retrieval.passage") == "retrieval")
    for good in ("retrieval", "text-matching", "clustering", "classification"):
        check(f"v5 passes {good!r} through", v5(good) == good)

    print()
    total, passed = len(_results), sum(_results)
    print(f"{PASS if passed == total else FAIL}: {passed}/{total}")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
