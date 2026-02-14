from __future__ import annotations

import statistics
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse import issparse

from .diag_fastpath import (
    _extract_right_monomial_csr,
    _is_square_main_diagonal_dia_matrix,
    _right_multiply_monomial_csr,
    _right_scale_by_diagonal,
)

try:
    from ._spgemm_cpp import (
        csr_spgemm_f64_i32 as _csr_spgemm_f64_i32,
        csr_spgemm_f64_i64 as _csr_spgemm_f64_i64,
    )
except Exception:  # pragma: no cover - optional binary extension
    _csr_spgemm_f64_i32 = None
    _csr_spgemm_f64_i64 = None


def _fastpath_sparse_matmul(a, b):
    if not (issparse(a) and issparse(b)):
        return None, None

    if (
        getattr(a, "format", None) == "csr"
        and getattr(a, "ndim", None) == 2
        and getattr(b, "ndim", None) == 2
        and a.shape[1] == b.shape[0]
        and getattr(a, "has_canonical_format", True)
    ):
        monomial = _extract_right_monomial_csr(b)
        if monomial is not None:
            col_map, scale = monomial
            return _right_multiply_monomial_csr(a, b, col_map, scale), "csr_right_monomial"

    if (
        getattr(a, "format", None) in {"csr", "csc"}
        and _is_square_main_diagonal_dia_matrix(b)
        and getattr(a, "ndim", None) == 2
        and a.shape[1] == b.shape[0]
    ):
        result = _right_scale_by_diagonal(a, b)
        if result is not None:
            return result, "right_diagonal_dia"

    return None, None


def _cpp_spgemm_csr(a, b, *, num_threads: int):
    if _csr_spgemm_f64_i32 is None or _csr_spgemm_f64_i64 is None:
        return None
    if not (issparse(a) and issparse(b)):
        return None
    if getattr(a, "format", None) != "csr" or getattr(b, "format", None) != "csr":
        return None
    if getattr(a, "ndim", None) != 2 or getattr(b, "ndim", None) != 2:
        return None
    if a.shape[1] != b.shape[0]:
        return None

    a64 = a.astype(np.float64, copy=False)
    b64 = b.astype(np.float64, copy=False)
    a_indptr = np.asarray(a64.indptr)
    a_indices = np.asarray(a64.indices)
    b_indptr = np.asarray(b64.indptr)
    b_indices = np.asarray(b64.indices)
    a_data = np.asarray(a64.data, dtype=np.float64)
    b_data = np.asarray(b64.data, dtype=np.float64)

    if (
        a_indptr.dtype == np.int32
        and a_indices.dtype == np.int32
        and b_indptr.dtype == np.int32
        and b_indices.dtype == np.int32
    ):
        out_indptr, out_indices, out_data = _csr_spgemm_f64_i32(
            a_indptr,
            a_indices,
            a_data,
            int(a64.shape[0]),
            int(a64.shape[1]),
            b_indptr,
            b_indices,
            b_data,
            int(b64.shape[1]),
            int(num_threads),
        )
    else:
        out_indptr, out_indices, out_data = _csr_spgemm_f64_i64(
            np.asarray(a_indptr, dtype=np.int64),
            np.asarray(a_indices, dtype=np.int64),
            a_data,
            int(a64.shape[0]),
            int(a64.shape[1]),
            np.asarray(b_indptr, dtype=np.int64),
            np.asarray(b_indices, dtype=np.int64),
            b_data,
            int(b64.shape[1]),
            int(num_threads),
        )
    return sp.csr_matrix((out_data, out_indices, out_indptr), shape=(a.shape[0], b.shape[1]))


def sparse_matmul(a, b, *, kernel: str = "auto", num_threads: int = 0, return_meta: bool = False):
    """Matrix multiply with optional C++ SpGEMM kernel and sparse fastpaths.

    Parameters
    ----------
    kernel : {"auto", "cpp", "scipy"}
        "auto": structure fastpaths, then C++ CSR kernel, then SciPy fallback.
        "cpp": force C++ CSR kernel when supported, otherwise SciPy fallback.
        "scipy": direct ``a @ b``.
    num_threads : int
        Thread count for C++ kernel. 0 means auto-detect.
    """
    if kernel not in {"auto", "cpp", "scipy"}:
        raise ValueError("kernel must be one of: 'auto', 'cpp', 'scipy'")

    result = None
    selected_path = "fallback_scipy"

    if kernel == "auto":
        fast_result, path = _fastpath_sparse_matmul(a, b)
        if fast_result is not None:
            result = fast_result
            selected_path = path

    if result is None and kernel in {"auto", "cpp"}:
        cpp_result = _cpp_spgemm_csr(a, b, num_threads=num_threads)
        if cpp_result is not None:
            result = cpp_result
            selected_path = "cpp_csr_spgemm"

    if result is None:
        result = a @ b
        selected_path = "fallback_scipy"

    if return_meta:
        return result, {"path": selected_path}
    return result


def _median_runtime(fn, repeats: int, warmups: int) -> float:
    for _ in range(warmups):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def _max_abs_diff(a, b) -> float:
    if issparse(a) and issparse(b):
        diff = (a - b).data
        return 0.0 if diff.size == 0 else float(np.max(np.abs(diff)))

    dense_diff = np.asarray(a) - np.asarray(b)
    return float(np.max(np.abs(dense_diff)))


def benchmark_against_scipy(
    a,
    b,
    *,
    repeats: int = 8,
    warmups: int = 2,
    kernel: str = "auto",
    num_threads: int = 0,
):
    """Benchmark optimized matmul against plain SciPy ``a @ b``."""
    scipy_sec = _median_runtime(lambda: a @ b, repeats=repeats, warmups=warmups)
    optimized_sec = _median_runtime(
        lambda: sparse_matmul(a, b, kernel=kernel, num_threads=num_threads),
        repeats=repeats,
        warmups=warmups,
    )
    baseline = a @ b
    optimized, meta = sparse_matmul(a, b, kernel=kernel, num_threads=num_threads, return_meta=True)
    return {
        "path": meta["path"],
        "scipy_ms": scipy_sec * 1e3,
        "optimized_ms": optimized_sec * 1e3,
        "speedup": scipy_sec / optimized_sec,
        "max_abs_diff": _max_abs_diff(baseline, optimized),
    }
