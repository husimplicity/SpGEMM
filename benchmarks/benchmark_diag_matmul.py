#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import scipy.sparse as sp

from scipy_sparse_opt.diag_fastpath import disable_diag_fastpath, enable_diag_fastpath


def _build_case(n: int, density: float, fmt: str, seed: int):
    rng = np.random.default_rng(seed)
    a = sp.random(n, n, density=density, format=fmt, random_state=rng, dtype=np.float64)
    d = sp.diags(rng.random(n), format="dia")
    return a, d


def _time_operation(fn, repeats: int, warmups: int):
    for _ in range(warmups):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return samples


def _max_abs_sparse_diff(a, b):
    diff = (a - b).data
    return 0.0 if diff.size == 0 else float(np.max(np.abs(diff)))


def _run_one_case(fmt: str, n: int, density: float, repeats: int, warmups: int, seed: int):
    a, d = _build_case(n=n, density=density, fmt=fmt, seed=seed)

    disable_diag_fastpath()
    base_times = _time_operation(lambda: a @ d, repeats=repeats, warmups=warmups)
    baseline = a @ d

    enable_diag_fastpath()
    opt_times = _time_operation(lambda: a @ d, repeats=repeats, warmups=warmups)
    optimized = a @ d
    disable_diag_fastpath()

    max_abs_diff = _max_abs_sparse_diff(baseline, optimized)
    base_median = statistics.median(base_times)
    opt_median = statistics.median(opt_times)
    speedup = base_median / opt_median

    return {
        "format": fmt,
        "n": n,
        "density": density,
        "nnz": int(a.nnz),
        "baseline_median_s": base_median,
        "optimized_median_s": opt_median,
        "speedup": speedup,
        "max_abs_diff": max_abs_diff,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SciPy sparse @ diagonal multiplication with and without fastpath patch.",
    )
    parser.add_argument("--n", type=int, default=10000, help="Square matrix size.")
    parser.add_argument("--density", type=float, default=0.01, help="Input sparse density.")
    parser.add_argument("--repeats", type=int, default=8, help="Timing repeats.")
    parser.add_argument("--warmups", type=int, default=2, help="Warmup iterations.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    args = parser.parse_args()

    rows = [
        _run_one_case("csr", args.n, args.density, args.repeats, args.warmups, args.seed),
        _run_one_case("csc", args.n, args.density, args.repeats, args.warmups, args.seed),
    ]

    print("SciPy Sparse @ DIA Benchmark")
    print(
        "format  n      density   nnz      baseline_med(ms)  optimized_med(ms)  speedup(x)  max_abs_diff"
    )
    for r in rows:
        print(
            f"{r['format']:>4}  {r['n']:>6}  {r['density']:<8.5f}  {r['nnz']:>7}  "
            f"{r['baseline_median_s'] * 1e3:>16.3f}  {r['optimized_median_s'] * 1e3:>17.3f}  "
            f"{r['speedup']:>9.2f}  {r['max_abs_diff']:.3e}"
        )


if __name__ == "__main__":
    main()
