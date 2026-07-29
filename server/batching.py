"""Cross-request batching for the embedding path.

One image serves one model, so every concurrent request wants the same weights
-- and a GPU that runs them one request at a time is idle for most of its
width. A single worker thread owns the model, coalesces whatever is in flight
into length-sorted, token-budgeted forwards, and scatters the rows back.

Two invariants make this safe rather than merely fast:

* **One thread touches the model, always.** Callers block on an ``Event``; they
  never reach ``encode`` themselves, and even the startup warm-up runs on the
  worker. That last part is not tidiness: ``torch.compile(mode="reduce-overhead")``
  keeps its CUDA-graph tree in thread-local storage, so a graph captured during
  a warm-up on the startup thread cannot be replayed from the worker -- it trips
  an assertion on the first real request. Requests that cannot be merged still
  go through the worker, alone, so there is never a second thread on the GPU.
* **A batch is homogeneous in everything that changes the forward pass.**
  Task, prompt prefix, normalisation and the native extras are part of the
  grouping key, so merging can only ever change how rows are grouped, never
  what is computed. ``dimensions`` is deliberately *not* in the key: matryoshka
  truncation happens after the forward, per item.

The window is adaptive. The reference implementation this is modelled on waited
its full coalescing window on every request, including the first one on an idle
server -- 5 ms, which on a warm nano model costs more than the forward it is
trying to amortise. Here an empty queue means no contention and the forward
starts immediately; the window only opens once a second request is already
waiting, which is the only situation where there is anything to coalesce.
"""

import logging
import queue
import threading
import time
from typing import Any, Callable, Optional

import numpy as np
import torch

import telemetry

logger = logging.getLogger(__name__)

# Ceiling on rows per forward, independent of the token budget: a batch of ten
# thousand one-token inputs fits any budget and still builds tensors no kernel
# wants. Not an env knob -- it is a guard, not a tuning dial.
MAX_ROWS = 512

# How many rows the worker will accumulate across jobs before it stops
# collecting and starts packing. Above this the extra latency of waiting is not
# repaid by better packing.
COLLECT_ROWS = MAX_ROWS * 4

# Fraction of VRAM the startup probe aims to fill when sizing the token budget.
# The remainder absorbs allocator fragmentation and the activation spike of a
# batch longer than the probe's.
VRAM_TARGET = 0.90

# Floor for the OOM backoff, and only for that: below this, halving again buys
# nothing that splitting the chunk has not already bought. It is deliberately
# not applied to the configured budget -- an operator who sets a small
# JINA_BATCH_TOKENS on a memory-tight host means it, and silently raising it
# would turn their deliberate cap into an OOM.
MIN_TOKENS = 2048


class _Job:
    """One caller's inputs, and the Event it is blocked on."""

    __slots__ = ("inputs", "key", "lengths", "results", "error", "event")

    def __init__(self, inputs: list, key: tuple, lengths: list[int]):
        self.inputs = inputs
        self.key = key
        self.lengths = lengths
        self.results: list = [None] * len(inputs)
        self.error: Optional[BaseException] = None
        self.event = threading.Event()


class Batcher:
    """Serialises model access through one worker thread, batching what it can.

    ``encode`` is called as ``encode(key, inputs) -> sequence`` on the worker
    thread and must return one result per input, in order. It is the caller's
    job to put everything that changes the forward pass into ``key``.
    """

    def __init__(
        self,
        encode: Callable[[tuple, list], Any],
        wait_ms: float,
        merge: bool = True,
        startup: Optional[Callable[[], int]] = None,
    ):
        self._encode = encode
        self._budget = 1
        self._wait_s = max(0.0, wait_ms) / 1000.0
        self._merge = merge
        self._startup = startup
        self._ready = threading.Event()
        self._startup_error: Optional[BaseException] = None
        self._queue: "queue.Queue[Optional[_Job]]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="model-worker", daemon=True)
        self._thread.start()
        # Block until the worker has finished warming up, so the server does not
        # report ready while the first request would still pay for compilation.
        self._ready.wait()
        if self._startup_error is not None:
            raise self._startup_error
        logger.info(
            f"Batcher started: merge={merge} token_budget={self._budget} "
            f"max_rows={MAX_ROWS} wait_ms={wait_ms}"
        )

    @property
    def merges(self) -> bool:
        return self._merge

    def submit(self, inputs: list, key: tuple, lengths: list[int]) -> list:
        """Enqueue and block until this job's rows are filled.

        Blocking is deliberate: the routes are synchronous handlers, so FastAPI
        runs them in its threadpool and the event loop stays free. The pool's
        own size is the backpressure -- an unbounded queue in front of a single
        GPU would just convert overload into memory growth.
        """
        job = _Job(inputs, key, lengths)
        self._queue.put(job)
        job.event.wait()
        if job.error is not None:
            raise job.error
        return job.results

    def _run(self) -> None:
        try:
            if self._startup is not None:
                self._budget = max(self._startup(), 1)
        except BaseException as exc:  # noqa: BLE001 - surfaced to load()
            self._startup_error = exc
        finally:
            self._ready.set()
        if self._startup_error is not None:
            return
        while True:
            first = self._queue.get()
            if first is None:
                return
            jobs = self._collect(first)
            groups: dict = {}
            for job in jobs:
                groups.setdefault(job.key, []).append(job)
            for key, group in groups.items():
                try:
                    self._process(key, group)
                except BaseException as exc:  # noqa: BLE001 - a job must never hang
                    for job in group:
                        self._fail(job, exc)

    def _collect(self, first: _Job) -> list[_Job]:
        if not self._merge:
            # Worker-only mode: the point is that one thread owns the model, not
            # that forwards are full. Callers still get exactly the forward they
            # would have got calling encode themselves.
            return [first]
        jobs = [first]
        rows = len(first.inputs)
        # Whatever is already queued is free to take -- no waiting involved.
        while rows < COLLECT_ROWS:
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                break
            if job is None:
                self._queue.put(None)
                break
            jobs.append(job)
            rows += len(job.inputs)
        # Only hold the door open if something was already waiting. On an idle
        # server this branch never runs and the request pays nothing.
        if len(jobs) > 1 and self._wait_s > 0:
            deadline = time.monotonic() + self._wait_s
            while rows < COLLECT_ROWS:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    job = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if job is None:
                    self._queue.put(None)
                    break
                jobs.append(job)
                rows += len(job.inputs)
        return jobs

    def _process(self, key: tuple, jobs: list[_Job]) -> None:
        # (job, index-within-that-job) for every row, flattened.
        slots, inputs, lengths = [], [], []
        for job in jobs:
            for index, item in enumerate(job.inputs):
                slots.append((job, index))
                inputs.append(item)
                lengths.append(job.lengths[index])

        # Ascending length: padding is charged at the longest member of a
        # chunk, so neighbours of similar length waste the least compute.
        if not self._merge:
            chunks = [list(range(len(inputs)))]
        else:
            # Ascending length: padding is charged at the longest member of a
            # chunk, so neighbours of similar length waste the least compute.
            order = sorted(range(len(inputs)), key=lambda i: lengths[i])
            chunks = self._pack(order, lengths)
        for chunk in chunks:
            rows = [slots[i] for i in chunk]
            try:
                encoded = self._encode_with_backoff(
                    key, [inputs[i] for i in chunk], self._merge
                )
            except BaseException as exc:  # noqa: BLE001 - isolate to this chunk
                for job, _ in rows:
                    self._fail(job, exc)
                continue
            for position, (job, index) in enumerate(rows):
                job.results[index] = encoded[position]

        for job in jobs:
            if job.error is None and not job.event.is_set():
                job.event.set()

    def _pack(self, order: list[int], lengths: list[int]) -> list[list[int]]:
        """Greedy fill by padded cost: rows x longest-row-in-chunk."""
        chunks: list[list[int]] = []
        current: list[int] = []
        longest = 0
        for index in order:
            candidate = max(longest, lengths[index])
            over_budget = (len(current) + 1) * candidate > self._budget
            if current and (over_budget or len(current) + 1 > MAX_ROWS):
                chunks.append(current)
                current, longest = [index], lengths[index]
            else:
                current.append(index)
                longest = candidate
        if current:
            chunks.append(current)
        return chunks

    def _encode_with_backoff(self, key: tuple, inputs: list, merge: bool):
        try:
            encoded = _as_rows(self._encode(key, inputs, merge), len(inputs))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(inputs) == 1:
                raise
            # The probe that sized the budget used one shape; real traffic
            # found a worse one. Shrink permanently rather than rediscovering
            # the same cliff on every batch.
            self._budget = max(min(MIN_TOKENS, self._budget), int(self._budget * 0.7))
            logger.warning(
                f"CUDA OOM at {len(inputs)} rows -- token budget now {self._budget}, splitting"
            )
            middle = len(inputs) // 2
            head = self._encode_with_backoff(key, inputs[:middle], merge)
            tail = self._encode_with_backoff(key, inputs[middle:], merge)
            # The halves count themselves; counting here as well would report
            # one forward that never ran, at a row count no kernel ever saw.
            return list(head) + list(tail)
        telemetry.record_forward(len(inputs))
        return encoded

    def _fail(self, job: _Job, exc: BaseException) -> None:
        if job.error is None:
            job.error = exc
        job.event.set()


def _as_rows(encoded: Any, expected: int) -> list:
    """One result per input, whatever shape the model returned.

    A single media item comes back as a flat vector rather than a one-row
    matrix, and ``late_chunking`` / ``return_multivector`` return a list of
    per-token matrices instead of an array. Both index the same way once the
    flat case is promoted.
    """
    if isinstance(encoded, np.ndarray) and encoded.ndim == 1:
        encoded = encoded[None, :]
    if len(encoded) != expected:
        raise RuntimeError(
            f"Model returned {len(encoded)} results for {expected} inputs; "
            "batching assumes one result per input"
        )
    return list(encoded)


def autotune_budget(probe: Callable[[int], None], context: int) -> int:
    """Largest token budget that keeps a probe forward inside VRAM_TARGET.

    Doubles the row count against a fixed-length input until either the target
    fraction of VRAM is reserved or the allocator refuses, then backs off. On
    anything but CUDA there is no VRAM to size against and a fixed budget is
    used instead.
    """
    if not torch.cuda.is_available():
        return 8192
    torch.cuda.empty_cache()
    total = torch.cuda.get_device_properties(0).total_memory
    target = int(total * VRAM_TARGET)
    length = min(context or 512, 512)
    rows, best = 16, length * 8
    while rows <= MAX_ROWS:
        try:
            torch.cuda.reset_peak_memory_stats()
            probe(rows)
            torch.cuda.synchronize()
            reserved = torch.cuda.max_memory_reserved()
            best = rows * length
            logger.info(f"autotune rows={rows} len={length} reserved={reserved / 1e9:.2f}GB")
            if reserved >= target:
                break
            rows *= 2
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            best = max(length, (rows // 2) * length)
            logger.info(f"autotune OOM at rows={rows}, backing off to {best} tokens")
            break
    torch.cuda.empty_cache()
    budget = max(4096, int(best * 0.85))
    logger.info(f"Autotuned token budget: {budget} (~{budget // length} rows at {length} tokens)")
    return budget
