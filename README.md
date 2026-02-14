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

## Example Benchmark

```bash
python examples/compare_with_scipy_at.py --n 20000 --density-a 0.001 --density-b 0.001 --kernel cpp --num-threads 8
```
