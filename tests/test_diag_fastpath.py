import numpy as np
import scipy.sparse as sp

from scipy_sparse_opt.diag_fastpath import (
    _extract_right_monomial_csr,
    disable_diag_fastpath,
    enable_diag_fastpath,
)


def _make_inputs(n: int = 128, density: float = 0.15, seed: int = 7):
    rng = np.random.default_rng(seed)
    a_csr = sp.random(n, n, density=density, format="csr", random_state=rng)
    a_csc = a_csr.tocsc()
    d = rng.random(n)
    diag = sp.diags(d, format="dia")
    return a_csr, a_csc, diag


def test_csr_diag_fastpath_matches_baseline():
    a_csr, _, diag = _make_inputs()
    disable_diag_fastpath()
    baseline = a_csr @ diag

    enable_diag_fastpath()
    optimized = a_csr @ diag
    disable_diag_fastpath()

    assert optimized.format == baseline.format
    np.testing.assert_allclose((optimized - baseline).data, 0.0, rtol=0.0, atol=0.0)


def test_csc_diag_fastpath_matches_baseline():
    _, a_csc, diag = _make_inputs()
    disable_diag_fastpath()
    baseline = a_csc @ diag

    enable_diag_fastpath()
    optimized = a_csc @ diag
    disable_diag_fastpath()

    assert optimized.format == baseline.format
    np.testing.assert_allclose((optimized - baseline).data, 0.0, rtol=0.0, atol=0.0)


def test_enable_disable_is_idempotent():
    a_csr, _, diag = _make_inputs(n=64, density=0.2, seed=11)
    disable_diag_fastpath()
    baseline = a_csr @ diag

    enable_diag_fastpath()
    enable_diag_fastpath()
    optimized = a_csr @ diag
    disable_diag_fastpath()
    disable_diag_fastpath()

    np.testing.assert_allclose((optimized - baseline).data, 0.0, rtol=0.0, atol=0.0)


def test_non_diagonal_path_is_unchanged():
    rng = np.random.default_rng(23)
    a = sp.random(96, 96, density=0.2, format="csr", random_state=rng)
    b = sp.random(96, 96, density=0.2, format="csr", random_state=rng)

    disable_diag_fastpath()
    baseline = a @ b

    enable_diag_fastpath()
    optimized = a @ b
    disable_diag_fastpath()

    np.testing.assert_allclose((optimized - baseline).data, 0.0, rtol=0.0, atol=0.0)


def _make_random_monomial_csr(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = np.arange(n, dtype=np.int32)
    cols = rng.permutation(n).astype(np.int32, copy=False)
    data = rng.random(n) + 0.1
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def test_extract_right_monomial_csr_map():
    b = _make_random_monomial_csr(64, seed=5)
    extracted = _extract_right_monomial_csr(b)
    assert extracted is not None
    col_map, scale = extracted

    expected_cols = b.indices[b.indptr[:-1]]
    expected_scale = b.data[b.indptr[:-1]]
    np.testing.assert_array_equal(col_map, expected_cols)
    np.testing.assert_allclose(scale, expected_scale)


def test_extract_right_monomial_csr_rejects_non_monomial():
    rng = np.random.default_rng(9)
    b = sp.random(64, 64, density=0.1, format="csr", random_state=rng)
    assert _extract_right_monomial_csr(b) is None


def test_random_monomial_sparse_path_matches_baseline():
    rng = np.random.default_rng(29)
    a = sp.random(256, 256, density=0.1, format="csr", random_state=rng)
    b = _make_random_monomial_csr(256, seed=13)

    disable_diag_fastpath()
    baseline = a @ b

    enable_diag_fastpath()
    optimized = a @ b
    disable_diag_fastpath()

    np.testing.assert_allclose((optimized - baseline).data, 0.0, rtol=0.0, atol=0.0)
