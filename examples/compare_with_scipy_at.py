#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import scipy.sparse as sp

from scipy_sparse_opt import benchmark_against_scipy


def _print_report(report: dict):
    print(
        f"path={report['path']:<20} "
        f"scipy={report['scipy_ms']:.3f}ms  optimized={report['optimized_ms']:.3f}ms  "
        f"speedup={report['speedup']:.2f}x  max_abs_diff={report['max_abs_diff']:.3e}"
    )


def main():
    parser = argparse.ArgumentParser(description="Compare scipy '@' and scipy-sparse-opt library.")
    parser.add_argument("--n", type=int, default=20000, help="Square matrix size")
    parser.add_argument("--density-a", type=float, default=0.001, help="Density for left sparse matrix A")
    parser.add_argument("--density-b", type=float, default=0.001, help="Density for right sparse matrix B")
    parser.add_argument("--kernel", choices=["auto", "cpp", "scipy"], default="cpp", help="Kernel mode")
    parser.add_argument("--num-threads", type=int, default=8, help="Thread count for cpp kernel (0=auto)")
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

    print("Compare scipy '@' vs scipy-sparse-opt")
    print(f"n={args.n}, density(A)={args.density_a}, density(B)={args.density_b}, kernel={args.kernel}, threads={args.num_threads}")
    random_report = benchmark_against_scipy(
        a, b_random, repeats=args.repeats, warmups=args.warmups, kernel=args.kernel, num_threads=args.num_threads
    )
    _print_report(random_report)


if __name__ == "__main__":
    main()
