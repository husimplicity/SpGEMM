import numpy as np
import scipy.sparse as sp

from scipy_sparse_opt.lib import benchmark_against_scipy, sparse_matmul


def _make_random_monomial_csr(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = np.arange(n, dtype=np.int32)
    cols = rng.permutation(n).astype(np.int32, copy=False)
    data = rng.random(n) + 0.1
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def test_sparse_matmul_matches_scipy_for_random_sparse():
    rng = np.random.default_rng(10)
    a = sp.random(256, 256, density=0.08, format="csr", random_state=rng)
    b = sp.random(256, 256, density=0.08, format="csr", random_state=rng)
    expected = a @ b

    actual, meta = sparse_matmul(a, b, kernel="scipy", return_meta=True)

    assert meta["path"] == "fallback_scipy"
    np.testing.assert_allclose((actual - expected).data, 0.0, rtol=0.0, atol=0.0)


def test_sparse_matmul_uses_monomial_fastpath():
    rng = np.random.default_rng(11)
    a = sp.random(256, 256, density=0.08, format="csr", random_state=rng)
    b = _make_random_monomial_csr(256, seed=12)
    expected = a @ b

    actual, meta = sparse_matmul(a, b, return_meta=True)

    assert meta["path"] == "csr_right_monomial"
    np.testing.assert_allclose((actual - expected).data, 0.0, rtol=0.0, atol=0.0)


def test_benchmark_against_scipy_produces_consistent_report():
    rng = np.random.default_rng(20)
    a = sp.random(512, 512, density=0.02, format="csr", random_state=rng)
    b = _make_random_monomial_csr(512, seed=21)

    report = benchmark_against_scipy(a, b, repeats=5, warmups=1)

    assert report["path"] == "csr_right_monomial"
    assert report["max_abs_diff"] == 0.0
    assert report["scipy_ms"] > 0.0
    assert report["optimized_ms"] > 0.0
    assert report["speedup"] > 0.0


def test_sparse_matmul_cpp_kernel_matches_scipy_random_random():
    rng = np.random.default_rng(33)
    a = sp.random(300, 300, density=0.06, format="csr", random_state=rng, dtype=np.float64)
    b = sp.random(300, 300, density=0.06, format="csr", random_state=rng, dtype=np.float64)
    expected = a @ b

    actual, meta = sparse_matmul(a, b, kernel="cpp", num_threads=2, return_meta=True)

    assert meta["path"] == "cpp_csr_spgemm"
    np.testing.assert_allclose((actual - expected).data, 0.0, rtol=1e-12, atol=1e-12)


def test_sparse_matmul_cpp_kernel_fallback_when_not_supported():
    rng = np.random.default_rng(35)
    a = sp.random(100, 100, density=0.1, format="csc", random_state=rng, dtype=np.float64)
    b = sp.random(100, 100, density=0.1, format="csc", random_state=rng, dtype=np.float64)
    expected = a @ b

    actual, meta = sparse_matmul(a, b, kernel="cpp", num_threads=2, return_meta=True)

    assert meta["path"] == "fallback_scipy"
    np.testing.assert_allclose((actual - expected).data, 0.0, rtol=0.0, atol=0.0)


def test_sparse_matmul_cpp_kernel_handles_zero_work_case():
    a = sp.csr_matrix((128, 128), dtype=np.float64)
    b = sp.random(128, 128, density=0.1, format="csr", random_state=np.random.default_rng(44), dtype=np.float64)

    actual, meta = sparse_matmul(a, b, kernel="cpp", num_threads=8, return_meta=True)

    assert meta["path"] == "cpp_csr_spgemm"
    assert actual.nnz == 0
    assert actual.shape == (128, 128)


def test_sparse_matmul_cpp_i32_kernel_returns_i32_indices():
    rng = np.random.default_rng(55)
    a = sp.random(128, 96, density=0.1, format="csr", random_state=rng, dtype=np.float64)
    b = sp.random(96, 80, density=0.1, format="csr", random_state=rng, dtype=np.float64)
    a.indices = a.indices.astype(np.int32, copy=False)
    a.indptr = a.indptr.astype(np.int32, copy=False)
    b.indices = b.indices.astype(np.int32, copy=False)
    b.indptr = b.indptr.astype(np.int32, copy=False)

    actual, meta = sparse_matmul(a, b, kernel="cpp", num_threads=2, return_meta=True)
    expected = a @ b

    assert meta["path"] == "cpp_csr_spgemm"
    assert actual.indices.dtype == np.int32
    assert actual.indptr.dtype == np.int32
    np.testing.assert_allclose((actual - expected).data, 0.0, rtol=1e-12, atol=1e-12)
