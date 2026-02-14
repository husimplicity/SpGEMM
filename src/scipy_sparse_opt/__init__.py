from .diag_fastpath import (
    disable_diag_fastpath,
    enable_diag_fastpath,
    is_diag_fastpath_enabled,
)
from .lib import benchmark_against_scipy, sparse_matmul

__all__ = [
    "benchmark_against_scipy",
    "disable_diag_fastpath",
    "enable_diag_fastpath",
    "is_diag_fastpath_enabled",
    "sparse_matmul",
]
