"""Bounded batches of null searches in persistent, single-threaded workers."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from multiprocessing import get_context
from numbers import Integral
import os
from threading import Lock

import numpy as np
from threadpoolctl import threadpool_limits

from ..line_search.candidate_regions import (
    return_raw_lines, DEFAULT_MIN_RELATIVE_PROMINENCE,
    validate_min_relative_prominence,
)
from .synthetic_spectra import (
    _validate_simulation_count, _prepare_null_context,
    _draw_null_sample, _finish_null_sample,
)


_SPAWN_LOCK = Lock()
_THREAD_VARIABLES = (
    "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "OMP_NUM_THREADS", "BLIS_NUM_THREADS",
)


@contextmanager
def _worker_thread_environment():
    # Accelerate on macOS caches this setting and is not controlled by
    # threadpoolctl. Spawned interpreters must inherit it before importing
    # NumPy/SciPy. Restore the parent's environment as soon as submit returns.
    with _SPAWN_LOCK:
        previous = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
        try:
            os.environ.update({name: "1" for name in _THREAD_VARIABLES})
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _submit_batch(pool, batch):
    # ProcessPoolExecutor spawns any needed worker synchronously on submit.
    # Apply on every submission in case a worker has not been started yet.
    with _worker_thread_environment():
        return pool.submit(_search_batch, batch)


def validate_parallel_options(n_jobs, batch_size):
    for name, value in (("n_jobs", n_jobs), ("simulation_batch_size", batch_size)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")


def _search_sample(context, max_sigma_line, sim, sample, search_lines=True,
                   min_relative_prominence=DEFAULT_MIN_RELATIVE_PROMINENCE):
    null = _finish_null_sample(context, sample)
    if not search_lines:
        return null
    candidates = return_raw_lines(
        null.energy, null.values, null.uncertainties, null.ylines, null.baseline,
        response_sigma=null.response_sigma, max_sigma_line=max_sigma_line,
        min_relative_prominence=min_relative_prominence,
    )
    candidates["sim"] = sim
    return candidates


def _initialize_worker(context, max_sigma_line, search_lines=True,
                       min_relative_prominence=DEFAULT_MIN_RELATIVE_PROMINENCE):
    global _worker_context, _worker_max_sigma, _worker_thread_limit, _worker_search_lines
    global _worker_min_relative_prominence
    _worker_context = context
    _worker_max_sigma = max_sigma_line
    _worker_search_lines = search_lines
    _worker_min_relative_prominence = min_relative_prominence
    # Keep the controller alive for the worker's lifetime. It limits native
    # BLAS/OpenMP pools, so two processes do not each use many numeric threads.
    _worker_thread_limit = threadpool_limits(limits=1)


def _search_batch(batch):
    return [_search_sample(_worker_context, _worker_max_sigma, sim, sample,
                           _worker_search_lines, _worker_min_relative_prominence)
            for sim, sample in batch]


def iter_parallel_null_candidates(spectrum_full, base_full, fit_en1, fit_en2,
                                  config, *, window_width=None):
    """Yield ordered null candidate tables with bounded parallel batches."""
    yield from _iter_parallel_null_results(
        spectrum_full, base_full, fit_en1, fit_en2, config,
        window_width=window_width, search_lines=True,
    )


def iter_parallel_null_realizations(spectrum_full, base_full, fit_en1, fit_en2,
                                    config, *, noise_model=None, window_width=None):
    """Yield prepared null spectra for diagnostics, without fitting lines."""
    yield from _iter_parallel_null_results(
        spectrum_full, base_full, fit_en1, fit_en2, config,
        noise_model=noise_model, window_width=window_width, search_lines=False,
    )


def _iter_parallel_null_results(spectrum_full, base_full, fit_en1, fit_en2,
                                config, *, noise_model=None, window_width=None,
                                search_lines=True):
    """Yield one spectrum or candidate table per simulation, including empty searches.

    Draws are made in simulation order in the parent to retain the serial RNG
    stream. Workers rebin, estimate baselines and search lines. Shared numeric
    context is sent once per worker, and at most twice the worker count of
    batches is pending. Results are consumed in simulation order, regardless
    of completion order. Any worker error aborts the run.
    """
    validate_parallel_options(config.n_jobs, config.simulation_batch_size)
    validate_min_relative_prominence(config.min_relative_prominence)
    n_sim = _validate_simulation_count(config.num_synthetic_simulations)
    if n_sim == 0:
        return
    context = _prepare_null_context(
        spectrum_full, base_full, fit_en1, fit_en2, config,
        noise_model=noise_model, window_width=window_width,
    )
    rng = np.random.default_rng(config.synthetic_seed)
    batch_size = int(config.simulation_batch_size)
    workers = min(int(config.n_jobs), (n_sim + batch_size - 1) // batch_size)
    if workers == 1:
        for sim in range(n_sim):
            yield _search_sample(context, config.max_sigma_line, sim,
                                 _draw_null_sample(context, rng), search_lines,
                                 config.min_relative_prominence)
        return

    def batches():
        for first in range(0, n_sim, batch_size):
            yield [(sim, _draw_null_sample(context, rng))
                   for sim in range(first, min(first + batch_size, n_sim))]

    pending = deque()
    pool = ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn"),
        initializer=_initialize_worker,
        initargs=(context, config.max_sigma_line, search_lines,
                  config.min_relative_prominence),
    )
    try:
        remaining = iter(batches())
        for _ in range(2 * workers):
            batch = next(remaining, None)
            if batch is None:
                break
            pending.append(_submit_batch(pool, batch))
        while pending:
            tables = pending.popleft().result()
            batch = next(remaining, None)
            if batch is not None:
                pending.append(_submit_batch(pool, batch))
            yield from tables
    finally:
        # Also handles an interrupted notebook cell or a failed fit. Running
        # batches finish, but queued work is cancelled instead of discarded.
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
