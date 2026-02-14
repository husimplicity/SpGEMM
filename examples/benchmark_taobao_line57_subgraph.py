from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.datasets import Taobao

from scipy_sparse_opt import sparse_matmul


def _median_runtime(fn, repeats: int, warmups: int) -> tuple[float, list[float]]:
    for _ in range(warmups):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), samples


def _build_subgraph_matrix(
    data,
    sample_users: int,
    sample_items: int,
    seed: int,
) -> sp.csr_matrix:
    row = data["user", "item"].edge_index[0]
    col = data["user", "item"].edge_index[1]
    num_users = data["user"].num_nodes
    num_items = data["item"].num_nodes

    rng = torch.Generator().manual_seed(seed)
    user_pick = torch.randperm(num_users, generator=rng)[:sample_users]
    user_mask = torch.zeros(num_users, dtype=torch.bool)
    user_mask[user_pick] = True

    mask_u = user_mask[row]
    row_u = row[mask_u]
    col_u = col[mask_u]

    active_items = torch.unique(col_u)
    if active_items.numel() > sample_items:
        item_pick = torch.randperm(active_items.numel(), generator=rng)[:sample_items]
        chosen_items = active_items[item_pick]
    else:
        chosen_items = active_items

    item_mask = torch.zeros(num_items, dtype=torch.bool)
    item_mask[chosen_items] = True
    mask_ui = item_mask[col_u]
    row_f = row_u[mask_ui]
    col_f = col_u[mask_ui]

    u_ids, u_inv = torch.unique(row_f, sorted=True, return_inverse=True)
    i_ids, i_inv = torch.unique(col_f, sorted=True, return_inverse=True)
    mat = sp.csr_matrix(
        (
            np.ones(row_f.numel(), dtype=np.float64),
            (u_inv.cpu().numpy(), i_inv.cpu().numpy()),
        ),
        shape=(int(u_ids.numel()), int(i_ids.numel())),
    )
    mat.sum_duplicates()
    mat.eliminate_zeros()
    return mat


def _max_abs_diff(a: sp.spmatrix, b: sp.spmatrix) -> float:
    diff = (a - b).data
    return 0.0 if diff.size == 0 else float(np.max(np.abs(diff)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark line-57 equivalent op (A^T A) on Taobao real-data subgraphs.",
    )
    parser.add_argument(
        "--root",
        default="/Users/huxleyhu/Documents/New project/third_party/pytorch_geometric/data/Taobao",
    )
    parser.add_argument("--sample-users", type=int, default=220_000)
    parser.add_argument("--sample-items", type=int, default=350_000)
    parser.add_argument("--seed", type=int, default=20260217)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--one-shot", action="store_true")
    args = parser.parse_args()

    t_load0 = time.perf_counter()
    data = Taobao(args.root)[0]
    t_load = time.perf_counter() - t_load0

    t_build0 = time.perf_counter()
    mat = _build_subgraph_matrix(
        data=data,
        sample_users=args.sample_users,
        sample_items=args.sample_items,
        seed=args.seed,
    )
    t_build = time.perf_counter() - t_build0
    left = mat.T.tocsr()

    if args.one_shot:
        _ = left @ mat
        _ = sparse_matmul(left, mat, kernel="cpp", num_threads=args.num_threads)

        t0 = time.perf_counter()
        baseline = left @ mat
        scipy_sec = time.perf_counter() - t0

        t1 = time.perf_counter()
        optimized = sparse_matmul(left, mat, kernel="cpp", num_threads=args.num_threads)
        opt_sec = time.perf_counter() - t1
    else:
        scipy_sec, scipy_runs = _median_runtime(
            lambda: left @ mat,
            repeats=args.repeats,
            warmups=args.warmups,
        )
        opt_sec, opt_runs = _median_runtime(
            lambda: sparse_matmul(left, mat, kernel="cpp", num_threads=args.num_threads),
            repeats=args.repeats,
            warmups=args.warmups,
        )
        print(f"scipy_runs_s={scipy_runs}")
        print(f"opt_runs_s={opt_runs}")
        baseline = left @ mat
        optimized = sparse_matmul(left, mat, kernel="cpp", num_threads=args.num_threads)

    print(f"dataset_load_s={t_load:.3f}")
    print(f"subgraph_build_s={t_build:.3f}")
    print(f"mat_shape={mat.shape}")
    print(f"mat_nnz={mat.nnz}")
    print(f"scipy_ms={scipy_sec * 1e3:.3f}")
    print(f"opt_ms={opt_sec * 1e3:.3f}")
    print(f"speedup={scipy_sec / opt_sec:.3f}")
    print(f"max_abs_diff={_max_abs_diff(baseline, optimized):.3e}")


if __name__ == "__main__":
    main()
