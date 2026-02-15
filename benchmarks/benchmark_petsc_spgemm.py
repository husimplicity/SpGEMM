#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc

from scipy_sparse_opt import sparse_matmul


@dataclass
class Case:
    n: int
    density: float


def _median_runtime(fn, repeats: int, warmups: int) -> float:
    for _ in range(warmups):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def _max_abs_sparse_diff(a: sp.csr_matrix, b: sp.csr_matrix) -> float:
    diff = (a - b).data
    return 0.0 if diff.size == 0 else float(np.max(np.abs(diff)))


def _to_csr_canonical(x: sp.spmatrix) -> sp.csr_matrix:
    csr = x.tocsr(copy=False)
    if not csr.has_canonical_format:
        csr.sum_duplicates()
    csr.sort_indices()
    return csr


def _csr_to_petsc(a: sp.csr_matrix) -> PETSc.Mat:
    indptr = np.asarray(a.indptr, dtype=PETSc.IntType)
    indices = np.asarray(a.indices, dtype=PETSc.IntType)
    data = np.asarray(a.data, dtype=np.float64)
    mat = PETSc.Mat().createAIJ(size=a.shape, csr=(indptr, indices, data), comm=PETSc.COMM_SELF)
    mat.assemble()
    return mat


def _petsc_to_csr(mat: PETSc.Mat) -> sp.csr_matrix:
    indptr, indices, data = mat.getValuesCSR()
    return sp.csr_matrix(
        (
            np.array(data, copy=True),
            np.array(indices, copy=True),
            np.array(indptr, copy=True),
        ),
        shape=mat.getSize(),
    )


def _parse_cases(raw: str) -> list[Case]:
    cases = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        n_s, d_s = item.split(":")
        cases.append(Case(n=int(n_s), density=float(d_s)))
    if not cases:
        raise ValueError("No valid cases parsed from --cases")
    return cases


def _run_case(
    case: Case,
    *,
    repeats: int,
    warmups: int,
    seed: int,
    num_threads: int,
) -> dict[str, float | int | str]:
    rng = np.random.default_rng(seed)
    a = sp.random(
        case.n,
        case.n,
        density=case.density,
        format="csr",
        random_state=rng,
        dtype=np.float64,
    )
    b = sp.random(
        case.n,
        case.n,
        density=case.density,
        format="csr",
        random_state=rng,
        dtype=np.float64,
    )
    a = _to_csr_canonical(a)
    b = _to_csr_canonical(b)

    baseline = _to_csr_canonical(a @ b)
    optimized, meta = sparse_matmul(a, b, kernel="cpp", num_threads=num_threads, return_meta=True)
    optimized = _to_csr_canonical(optimized)

    a_p = _csr_to_petsc(a)
    b_p = _csr_to_petsc(b)
    c_p = a_p.matMult(b_p)
    petsc_once = _to_csr_canonical(_petsc_to_csr(c_p))

    max_diff_opt = _max_abs_sparse_diff(baseline, optimized)
    max_diff_petsc = _max_abs_sparse_diff(baseline, petsc_once)

    def scipy_once():
        _ = a @ b

    def opt_once():
        _ = sparse_matmul(a, b, kernel="cpp", num_threads=num_threads)

    def petsc_full_once():
        c_tmp = a_p.matMult(b_p)
        c_tmp.destroy()

    scipy_sec = _median_runtime(scipy_once, repeats=repeats, warmups=warmups)
    opt_sec = _median_runtime(opt_once, repeats=repeats, warmups=warmups)
    petsc_sec = _median_runtime(petsc_full_once, repeats=repeats, warmups=warmups)

    c_reuse = a_p.matMult(b_p)
    petsc_reuse_sec = _median_runtime(
        lambda: a_p.matMult(b_p, result=c_reuse),
        repeats=repeats,
        warmups=warmups,
    )

    c_reuse.destroy()
    c_p.destroy()
    a_p.destroy()
    b_p.destroy()

    return {
        "n": case.n,
        "density": case.density,
        "nnz_a": int(a.nnz),
        "nnz_b": int(b.nnz),
        "nnz_c": int(baseline.nnz),
        "path": str(meta["path"]),
        "scipy_ms": scipy_sec * 1e3,
        "opt_ms": opt_sec * 1e3,
        "petsc_ms": petsc_sec * 1e3,
        "petsc_reuse_ms": petsc_reuse_sec * 1e3,
        "opt_speedup_vs_scipy": scipy_sec / opt_sec,
        "petsc_speedup_vs_scipy": scipy_sec / petsc_sec,
        "opt_vs_petsc": petsc_sec / opt_sec,
        "max_diff_opt": max_diff_opt,
        "max_diff_petsc": max_diff_petsc,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark random CSR SpGEMM: SciPy @ vs scipy_sparse_opt vs PETSc.")
    parser.add_argument(
        "--cases",
        type=str,
        default="12000:0.0006,20000:0.00045,28000:0.00036",
        help="Comma-separated list like '12000:0.0006,20000:0.00045'",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Timing repeats")
    parser.add_argument("--warmups", type=int, default=1, help="Warmup rounds")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--num-threads", type=int, default=12, help="Thread count for scipy_sparse_opt cpp kernel")
    args = parser.parse_args()

    cases = _parse_cases(args.cases)

    print("Random CSR SpGEMM Benchmark (SciPy vs scipy_sparse_opt vs PETSc)")
    print(f"cases={args.cases}")
    print(f"repeats={args.repeats}, warmups={args.warmups}, seed={args.seed}, num_threads={args.num_threads}")
    print(
        "n        dens      nnz(A)    nnz(B)    nnz(C)    "
        "scipy(ms)  opt(ms)   petsc(ms)  petsc_reuse(ms)  "
        "opt/scipy  petsc/scipy  opt/petsc  maxdiff(opt)  maxdiff(petsc)  path"
    )
    for case in cases:
        row = _run_case(
            case,
            repeats=args.repeats,
            warmups=args.warmups,
            seed=args.seed,
            num_threads=args.num_threads,
        )
        print(
            f"{row['n']:>6}  {row['density']:<8.6f}  "
            f"{row['nnz_a']:>8}  {row['nnz_b']:>8}  {row['nnz_c']:>8}  "
            f"{row['scipy_ms']:>9.3f}  {row['opt_ms']:>8.3f}  {row['petsc_ms']:>10.3f}  {row['petsc_reuse_ms']:>15.3f}  "
            f"{row['opt_speedup_vs_scipy']:>9.3f}  {row['petsc_speedup_vs_scipy']:>11.3f}  {row['opt_vs_petsc']:>9.3f}  "
            f"{row['max_diff_opt']:.3e}     {row['max_diff_petsc']:.3e}     {row['path']}"
        )


if __name__ == "__main__":
    main()
