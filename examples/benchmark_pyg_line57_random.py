from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import scipy.sparse as sp

from scipy_sparse_opt import sparse_matmul


def _median_runtime(fn, repeats: int, warmups: int) -> float:
    for _ in range(warmups):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def _make_random_bipartite(num_users: int, num_items: int, density: float, seed: int) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)

    # Keep 1.0 edge weights to match the line-57 semantics in the PyG example.
    mat = sp.random(
        num_users,
        num_items,
        density=density,
        format="csr",
        dtype=np.float64,
        random_state=rng,
        data_rvs=lambda n: np.ones(n, dtype=np.float64),
    )
    mat.sum_duplicates()
    mat.eliminate_zeros()
    return mat


def _max_abs_diff(a: sp.spmatrix, b: sp.spmatrix) -> float:
    diff = (a - b).data
    return 0.0 if diff.size == 0 else float(np.max(np.abs(diff)))


def _run_case(
    num_users: int,
    num_items: int,
    density: float,
    repeats: int,
    warmups: int,
    seed: int,
    num_threads: int,
) -> dict[str, float]:
    mat = _make_random_bipartite(num_users, num_items, density, seed)
    left = mat.T.tocsr()

    scipy_sec = _median_runtime(lambda: left @ mat, repeats=repeats, warmups=warmups)
    opt_sec = _median_runtime(
        lambda: sparse_matmul(left, mat, kernel="cpp", num_threads=num_threads),
        repeats=repeats,
        warmups=warmups,
    )

    baseline = left @ mat
    optimized = sparse_matmul(left, mat, kernel="cpp", num_threads=num_threads)

    return {
        "users": float(num_users),
        "items": float(num_items),
        "density": density,
        "nnz": float(mat.nnz),
        "scipy_ms": scipy_sec * 1e3,
        "opt_ms": opt_sec * 1e3,
        "speedup": scipy_sec / opt_sec,
        "max_abs_diff": _max_abs_diff(baseline, optimized),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark line-57 equivalent in PyG bipartite_sage_unsup.py: comat = mat.T @ mat",
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-threads", type=int, default=0)
    args = parser.parse_args()

    cases = [
        (50_000, 20_000, 2.0e-4),
        (80_000, 30_000, 1.5e-4),
        (120_000, 40_000, 1.0e-4),
    ]

    print("Benchmarking random sparse bipartite matrices for line-57 operation (A^T A)")
    print("users\titems\tdensity\tnnz\tscipy_ms\topt_ms\tspeedup\tmax_abs_diff")

    rows = []
    for i, (num_users, num_items, density) in enumerate(cases):
        row = _run_case(
            num_users=num_users,
            num_items=num_items,
            density=density,
            repeats=args.repeats,
            warmups=args.warmups,
            seed=args.seed + i,
            num_threads=args.num_threads,
        )
        rows.append(row)
        print(
            f"{int(row['users'])}\t{int(row['items'])}\t{row['density']:.2e}\t"
            f"{int(row['nnz'])}\t{row['scipy_ms']:.3f}\t{row['opt_ms']:.3f}\t"
            f"{row['speedup']:.3f}\t{row['max_abs_diff']:.2e}"
        )

    geo_speedup = statistics.geometric_mean([r["speedup"] for r in rows])
    print(f"\nGeometric-mean speedup: {geo_speedup:.3f}x")


if __name__ == "__main__":
    main()
