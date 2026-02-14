#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import scipy.sparse as sp

from scipy_sparse_opt.diag_fastpath import disable_diag_fastpath, enable_diag_fastpath


def _median_runtime(fn, repeats: int, warmups: int) -> float:
    for _ in range(warmups):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def _max_abs_sparse_diff(a, b) -> float:
    diff = (a - b).data
    return 0.0 if diff.size == 0 else float(np.max(np.abs(diff)))


def _make_random_monomial_csr(n: int, rng: np.random.Generator):
    rows = np.arange(n, dtype=np.int32)
    cols = rng.permutation(n).astype(np.int32, copy=False)
    data = rng.random(n) + 0.1
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def _run_case(a: sp.csr_matrix, b: sp.csr_matrix, repeats: int, warmups: int):
    disable_diag_fastpath()
    base_med = _median_runtime(lambda: a @ b, repeats=repeats, warmups=warmups)
    baseline = a @ b

    enable_diag_fastpath()
    opt_med = _median_runtime(lambda: a @ b, repeats=repeats, warmups=warmups)
    optimized = a @ b
    disable_diag_fastpath()

    return {
        "baseline_med_s": base_med,
        "optimized_med_s": opt_med,
        "speedup": base_med / opt_med,
        "max_abs_diff": _max_abs_sparse_diff(baseline, optimized),
        "nnz_a": int(a.nnz),
        "nnz_b": int(b.nnz),
        "nnz_out": int(baseline.nnz),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark random sparse matmul before/after fastpath patch.",
    )
    parser.add_argument("--n", type=int, default=10000, help="Square matrix size")
    parser.add_argument("--density-a", type=float, default=0.01, help="Density of left matrix A")
    parser.add_argument(
        "--density-b",
        type=float,
        default=0.01,
        help="Density of random right matrix B (non-monomial case)",
    )
    parser.add_argument("--repeats", type=int, default=8, help="Timing repeats")
    parser.add_argument("--warmups", type=int, default=2, help="Warmup rounds")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    a = sp.random(
        args.n,
        args.n,
        density=args.density_a,
        format="csr",
        random_state=rng,
        dtype=np.float64,
    )
    b_random = sp.random(
        args.n,
        args.n,
        density=args.density_b,
        format="csr",
        random_state=rng,
        dtype=np.float64,
    )
    b_monomial = _make_random_monomial_csr(args.n, rng)

    random_stats = _run_case(a, b_random, repeats=args.repeats, warmups=args.warmups)
    monomial_stats = _run_case(a, b_monomial, repeats=args.repeats, warmups=args.warmups)

    print("Random Sparse Matmul Benchmark")
    print("case                     n      nnz(A)    nnz(B)    nnz(C)    base(ms)   opt(ms)   speedup  max_abs_diff")
    print(
        f"{'random-random':<24} {args.n:>6}  {random_stats['nnz_a']:>8}  {random_stats['nnz_b']:>8}  "
        f"{random_stats['nnz_out']:>8}  {random_stats['baseline_med_s'] * 1e3:>8.3f}  "
        f"{random_stats['optimized_med_s'] * 1e3:>8.3f}  {random_stats['speedup']:>7.2f}  "
        f"{random_stats['max_abs_diff']:.3e}"
    )
    print(
        f"{'random-monomial':<24} {args.n:>6}  {monomial_stats['nnz_a']:>8}  {monomial_stats['nnz_b']:>8}  "
        f"{monomial_stats['nnz_out']:>8}  {monomial_stats['baseline_med_s'] * 1e3:>8.3f}  "
        f"{monomial_stats['optimized_med_s'] * 1e3:>8.3f}  {monomial_stats['speedup']:>7.2f}  "
        f"{monomial_stats['max_abs_diff']:.3e}"
    )


if __name__ == "__main__":
    main()
