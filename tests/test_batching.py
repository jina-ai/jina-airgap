#!/usr/bin/env python3
"""Batcher unit tests: no model, no GPU, no server.

Everything that can go wrong in a batcher goes wrong in the plumbing, not in
the arithmetic -- rows scattered to the wrong caller, a batch that mixes two
tasks, one bad input taking down unrelated requests, a coalescing window that
taxes an idle server. All of that is observable with a stub encode, and none of
it needs weights, so it is checked here rather than discovered on a GPU.

    python3 tests/test_batching.py
"""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

try:
    import numpy as np

    import batching  # noqa: E402
    import telemetry  # noqa: E402
    from batching import MAX_ROWS, Batcher  # noqa: E402
except ImportError as exc:
    # server/batching.py imports torch. CI installs neither it nor the weights,
    # so skip the module there rather than break collection for the suite.
    if "pytest" not in sys.modules:
        raise
    import pytest

    pytest.skip(f"batcher deps not installed ({exc})", allow_module_level=True)

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
_results = []


def check(name, cond, detail=""):
    _results.append(bool(cond))
    print(f"  [{PASS if cond else FAIL}] {name}{('  ' + detail) if detail else ''}")


class Recorder:
    """Stub encode: echoes each input as a vector and records the call shapes."""

    def __init__(self, delay=0.0):
        self.calls = []  # (key, [inputs])
        self.delay = delay
        self.lock = threading.Lock()

    def __call__(self, key, inputs, merge=True):
        if self.delay:
            time.sleep(self.delay)
        with self.lock:
            self.calls.append((key, list(inputs)))
        # Row i encodes to a vector whose first element is the input's number,
        # so a mis-scatter is visible rather than plausible.
        return np.array([[float(text.split("-")[1]), 0.0] for text in inputs])


def key(task="retrieval", normalized=True):
    return (task, None, normalized, (), None)


def _forward_counts(since=(0, 0)):
    """(forwards, rows) recorded so far, less a previous reading. The counters
    are process-global, so every assertion here is on a delta."""
    return (
        telemetry._stats["forwards"] - since[0],
        telemetry._stats["batched_rows"] - since[1],
    )


def submit_many(batcher, jobs):
    """Fire each job from its own thread, as the threadpool would."""
    out = {}
    errs = {}

    def run(name, inputs, k):
        try:
            out[name] = batcher.submit(inputs, k, [len(i) for i in inputs])
        except BaseException as exc:  # noqa: BLE001
            errs[name] = exc

    threads = [threading.Thread(target=run, args=j) for j in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out, errs


def main():
    print("Batcher unit tests\n")

    # --- scatter correctness: every caller gets its own rows, in order ---
    rec = Recorder(delay=0.05)
    b = Batcher(rec, wait_ms=20, startup=lambda: 100_000)
    jobs = [
        ("a", [f"x-{i}" for i in range(3)], key()),
        ("b", [f"x-{i}" for i in range(100, 105)], key()),
        ("c", [f"x-{i}" for i in range(200, 202)], key()),
    ]
    out, errs = submit_many(b, jobs)
    check("no errors on the happy path", not errs, str(errs))
    check("a gets its 3 rows in order", [v[0] for v in out["a"]] == [0, 1, 2])
    check("b gets its 5 rows in order", [v[0] for v in out["b"]] == [100, 101, 102, 103, 104])
    check("c gets its 2 rows in order", [v[0] for v in out["c"]] == [200, 201])

    # --- coalescing: three concurrent same-key jobs share forwards ---
    check(
        "3 concurrent jobs used fewer than 3 forwards",
        len(rec.calls) < 3,
        f"forwards={len(rec.calls)}",
    )

    # --- homogeneity: a differing key must never share a forward ---
    rec2 = Recorder(delay=0.05)
    b2 = Batcher(rec2, wait_ms=20, startup=lambda: 100_000)
    submit_many(
        b2,
        [
            ("q", [f"x-{i}" for i in range(4)], key(task="retrieval")),
            ("p", [f"x-{i}" for i in range(10, 14)], key(task="text-matching")),
            ("n", [f"x-{i}" for i in range(20, 24)], key(normalized=False)),
        ],
    )
    keys_per_call = {id(c): c[0] for c in rec2.calls}
    check("every forward carries exactly one key", len(keys_per_call) == len(rec2.calls))
    distinct = {k for k, _ in rec2.calls}
    check("all three keys were served", len(distinct) == 3, f"keys={len(distinct)}")

    # --- idle latency: a lone request must not pay the coalescing window ---
    rec3 = Recorder()
    b3 = Batcher(rec3, wait_ms=500, startup=lambda: 100_000)
    start = time.monotonic()
    b3.submit(["x-1"], key(), [4])
    solo_ms = (time.monotonic() - start) * 1000
    check(
        "lone request does not wait out the 500 ms window",
        solo_ms < 100,
        f"{solo_ms:.1f} ms",
    )

    # --- token budget: packing must not exceed rows x longest ---
    rec4 = Recorder()
    b4 = Batcher(rec4, wait_ms=0, startup=lambda: 40)
    b4.submit([f"x-{i}" for i in range(20)], key(), [10] * 20)
    worst = max(len(inputs) * 10 for _, inputs in rec4.calls)
    check("no forward exceeds the token budget", worst <= 40, f"worst={worst}")
    check("all 20 rows were encoded", sum(len(i) for _, i in rec4.calls) == 20)

    # --- row cap holds even when the budget would allow more ---
    rec5 = Recorder()
    b5 = Batcher(rec5, wait_ms=0, startup=lambda: 10_000_000)
    b5.submit([f"x-{i}" for i in range(MAX_ROWS + 50)], key(), [1] * (MAX_ROWS + 50))
    check(
        "no forward exceeds MAX_ROWS",
        max(len(i) for _, i in rec5.calls) <= MAX_ROWS,
        f"max={max(len(i) for _, i in rec5.calls)}",
    )

    # --- error isolation: a failing key must not fail an unrelated one ---
    def selective(k, inputs, merge=True):
        if k[0] == "poison":
            raise ValueError("bad task")
        return np.array([[float(t.split("-")[1]), 0.0] for t in inputs])

    b6 = Batcher(selective, wait_ms=20, startup=lambda: 100_000)
    out6, errs6 = submit_many(
        b6,
        [
            ("good", [f"x-{i}" for i in range(3)], key()),
            ("bad", [f"x-{i}" for i in range(9)], key(task="poison")),
        ],
    )
    check("the failing job raises to its own caller", isinstance(errs6.get("bad"), ValueError))
    check("the healthy job still returns", [v[0] for v in out6.get("good", [])] == [0, 1, 2])

    # --- OOM backoff: split in half, shrink the budget, still return everything ---
    state = {"fail_above": 8}

    def oom_once(k, inputs, merge=True):
        if len(inputs) > state["fail_above"]:
            raise torch_oom()
        return np.array([[float(t.split("-")[1]), 0.0] for t in inputs])

    b7 = Batcher(oom_once, wait_ms=0, startup=lambda: 10_000_000)
    before = b7._budget
    counted = _forward_counts()
    got = b7.submit([f"x-{i}" for i in range(32)], key(), [1] * 32)
    forwards, rows = _forward_counts(since=counted)
    check("OOM backoff still returns every row", [v[0] for v in got] == list(range(32)))
    check("OOM backoff shrank the token budget", b7._budget < before, f"{before} -> {b7._budget}")
    # 32 rows, nothing above 8 survives: 32 -> 16+16 -> four forwards of 8. The
    # abandoned attempts must not be counted, and neither must their rows.
    check("only the forwards that ran are counted", forwards == 4, f"forwards={forwards}")
    check("counted rows match the rows encoded", rows == 32, f"rows={rows}")

    # --- forward accounting: /health has to show the real batch shape ---
    rec9 = Recorder(delay=0.05)
    b9 = Batcher(rec9, wait_ms=20, startup=lambda: 100_000)
    counted = _forward_counts()
    submit_many(b9, [(str(i), [f"x-{i}"], key()) for i in range(6)])
    forwards, rows = _forward_counts(since=counted)
    check(
        "every forward is counted exactly once",
        forwards == len(rec9.calls),
        f"counted={forwards} actual={len(rec9.calls)}",
    )
    check("six single-row jobs account for six rows", rows == 6, f"rows={rows}")

    # --- a wrong-length model return is caught, not silently mis-scattered ---
    b8 = Batcher(lambda k, i, m: np.zeros((len(i) - 1, 2)), wait_ms=0,
                 startup=lambda: 100_000)
    try:
        b8.submit(["x-1", "x-2", "x-3"], key(), [1, 1, 1])
        check("short model return is rejected", False, "no error raised")
    except RuntimeError as exc:
        check("short model return is rejected", "one result per input" in str(exc))

    print()
    total, passed = len(_results), sum(_results)
    print(f"{PASS if passed == total else FAIL}: {passed}/{total}")
    sys.exit(0 if passed == total else 1)


def torch_oom():
    import torch

    return torch.cuda.OutOfMemoryError("simulated")


if __name__ == "__main__":
    main()
