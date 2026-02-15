from __future__ import annotations

import numpy as np
from scipy.sparse._compressed import _cs_matrix
from scipy.sparse._sputils import upcast

_ORIGINAL_MATMUL_SPARSE = _cs_matrix._matmul_sparse
_PATCH_ENABLED = False


def _is_square_main_diagonal_dia_matrix(x) -> bool:
    if getattr(x, "format", None) != "dia":
        return False
    if getattr(x, "ndim", None) != 2:
        return False
    if x.shape[0] != x.shape[1]:
        return False
    offsets = np.asarray(x.offsets)
    return offsets.size == 1 and int(offsets[0]) == 0


def _right_scale_by_diagonal(self: _cs_matrix, other):
    out_dtype = upcast(self.dtype, other.dtype)
    diag = np.asarray(other.data[0])
    if diag.shape[0] != other.shape[1]:
        diag = np.asarray(other.diagonal())

    if self.format == "csr":
        scaled_data = self.data.astype(out_dtype, copy=False) * diag[self.indices]
    elif self.format == "csc":
        col_nnz = np.diff(self.indptr)
        scaled_data = self.data.astype(out_dtype, copy=False) * np.repeat(diag, col_nnz)
    else:
        return None

    return self.__class__(
        (scaled_data, self.indices.copy(), self.indptr.copy()),
        shape=(self.shape[0], other.shape[1]),
        copy=False,
    )


def _extract_right_monomial_csr(other):
    if getattr(other, "format", None) != "csr":
        return None
    if getattr(other, "ndim", None) != 2:
        return None

    n_rows, _ = other.shape
    if n_rows == 0:
        return None
    # Necessary condition for row-monomial CSR: exactly one nonzero per row.
    # This avoids scanning row_nnz on common random matrices.
    if getattr(other, "nnz", None) != n_rows:
        return None

    row_nnz = np.diff(other.indptr)
    if row_nnz.shape[0] != n_rows or not np.all(row_nnz == 1):
        return None

    row_pos = other.indptr[:-1]
    col_map = np.asarray(other.indices[row_pos])
    _, n_cols = other.shape
    if col_map.size > n_cols:
        return None
    col_counts = np.bincount(col_map, minlength=n_cols)
    if np.any(col_counts > 1):
        return None

    scale = np.asarray(other.data[row_pos])
    return col_map, scale


def _right_multiply_monomial_csr(self: _cs_matrix, other, col_map, scale):
    out_dtype = upcast(self.dtype, other.dtype)
    mapped_indices = col_map[self.indices]
    scaled_data = self.data.astype(out_dtype, copy=False) * scale[self.indices]
    return self.__class__(
        (scaled_data, mapped_indices, self.indptr.copy()),
        shape=(self.shape[0], other.shape[1]),
        copy=False,
    )


def _patched_matmul_sparse(self, other):
    if (
        getattr(self, "format", None) == "csr"
        and getattr(self, "ndim", None) == 2
        and getattr(other, "ndim", None) == 2
        and self.shape[1] == other.shape[0]
        and getattr(self, "has_canonical_format", True)
    ):
        monomial = _extract_right_monomial_csr(other)
        if monomial is not None:
            col_map, scale = monomial
            return _right_multiply_monomial_csr(self, other, col_map, scale)

    if (
        _is_square_main_diagonal_dia_matrix(other)
        and getattr(self, "ndim", None) == 2
        and self.shape[1] == other.shape[0]
    ):
        fast_result = _right_scale_by_diagonal(self, other)
        if fast_result is not None:
            return fast_result

    return _ORIGINAL_MATMUL_SPARSE(self, other)


def enable_diag_fastpath() -> None:
    global _PATCH_ENABLED
    if _PATCH_ENABLED:
        return
    _cs_matrix._matmul_sparse = _patched_matmul_sparse
    _PATCH_ENABLED = True


def disable_diag_fastpath() -> None:
    global _PATCH_ENABLED
    if not _PATCH_ENABLED:
        return
    _cs_matrix._matmul_sparse = _ORIGINAL_MATMUL_SPARSE
    _PATCH_ENABLED = False


def is_diag_fastpath_enabled() -> bool:
    return _PATCH_ENABLED
