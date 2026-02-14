#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import scipy.sparse as sp


def _median_runtime(fn, repeats: int, warmups: int) -> float:
    for _ in range(warmups):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def _dense_fallback(a: sp.csr_matrix, b: sp.csr_matrix):
    return sp.csr_matrix(a.toarray() @ b.toarray())


def _row_singleton_general(a: sp.csr_matrix, b: sp.csr_matrix):
    k, n = b.shape
    counts = np.diff(b.indptr)
    if np.any(counts > 1):
        raise ValueError("B is not row-singleton")

    row_ptrs = b.indptr[:-1]
    has = counts == 1
    col_map = np.zeros(k, dtype=b.indices.dtype)
    scale = np.zeros(k, dtype=np.result_type(a.dtype, b.dtype))
    pos = row_ptrs[has]
    col_map[has] = b.indices[pos]
    scale[has] = b.data[pos]

    mapped_cols = col_map[a.indices]
    mapped_scale = scale[a.indices]
    mapped_data = a.data * mapped_scale
    c = sp.csr_matrix((mapped_data, mapped_cols, a.indptr), shape=(a.shape[0], n))
    c.sum_duplicates()
    c.eliminate_zeros()
    return c


def _monomial_fast(a: sp.csr_matrix, b: sp.csr_matrix):
    pos = b.indptr[:-1]
    col_map = b.indices[pos]
    scale = b.data[pos]
    return sp.csr_matrix((a.data * scale[a.indices], col_map[a.indices], a.indptr), shape=a.shape)


def _make_random_monomial(n: int, rng: np.random.Generator):
    rows = np.arange(n, dtype=np.int32)
    cols = rng.permutation(n).astype(np.int32, copy=False)
    data = rng.random(n) + 0.1
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def _make_random_row_singleton(n: int, rng: np.random.Generator):
    rows = np.arange(n, dtype=np.int32)
    cols = rng.integers(0, n, size=n, dtype=np.int32)
    data = rng.random(n) + 0.1
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def main():
    parser = argparse.ArgumentParser(description="Compare candidate sparse matmul optimization strategies.")
    parser.add_argument("--n", type=int, default=4000, help="Square matrix size")
    parser.add_argument("--density-a", type=float, default=0.01, help="Density of left random sparse matrix A")
    parser.add_argument("--repeats", type=int, default=8, help="Timing repeats")
    parser.add_argument("--warmups", type=int, default=2, help="Warmup rounds")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
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
    b_monomial = _make_random_monomial(args.n, rng)
    b_row_singleton = _make_random_row_singleton(args.n, rng)

    rows = []
    rows.append(("baseline scipy (A@B_monomial)", _median_runtime(lambda: a @ b_monomial, args.repeats, args.warmups)))
    rows.append(("dense fallback", _median_runtime(lambda: _dense_fallback(a, b_monomial), args.repeats, args.warmups)))
    rows.append(
        (
            "row-singleton general",
            _median_runtime(lambda: _row_singleton_general(a, b_row_singleton), args.repeats, args.warmups),
        )
    )
    rows.append(("monomial remap fastpath", _median_runtime(lambda: _monomial_fast(a, b_monomial), args.repeats, args.warmups)))

    print("Strategy Candidate Benchmark")
    print("strategy                     median(ms)")
    for name, sec in rows:
        print(f"{name:<28} {sec * 1e3:>10.3f}")


if __name__ == "__main__":
    main()
