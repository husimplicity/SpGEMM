# SpGEMM

A lightweight library that accelerates sparse matrix multiplication with a generic CSR SpGEMM kernel and targeted sparse fastpaths.

## Install

```bash
pip install -e .
```

## Quick Usage

```python
from scipy_sparse_opt import sparse_matmul

c = sparse_matmul(a, b, kernel="cpp", num_threads=0)
```

## Test

```bash
python -m pytest -q tests
```

## Benchmarks

### SciPy `@` vs optimized kernel

```bash
python examples/compare_with_scipy_at.py \
  --n 20000 --density-a 0.001 --density-b 0.001 \
  --kernel cpp --num-threads 8
```

### SciPy `@` vs optimized kernel vs PETSc (random CSR)

```bash
python benchmarks/benchmark_petsc_spgemm.py \
  --cases 12000:0.0006,20000:0.00045,28000:0.00036 \
  --repeats 3 --warmups 1 --num-threads 12
```

### Taobao line-57 style operation (real data subgraph)

```bash
python examples/benchmark_taobao_line57_subgraph.py \
  --root third_party/pytorch_geometric/data/Taobao \
  --sample-users 220000 --sample-items 350000 \
  --repeats 3 --warmups 1 --num-threads 12
```

## Notes

- `test.py` is a local ad-hoc script for manual profiling and is not part of the library API.
- For PETSc comparisons, result reuse (`matMult(..., result=C)`) and `-matmatmult_via` choice can change timings significantly.
