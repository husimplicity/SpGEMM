#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {
// Base threshold: small/medium row work favors per-row hash accumulator.
constexpr std::int64_t kHashWorkThreshold = 4096;
// Start applying width-aware heuristics once B is wide.
constexpr std::int64_t kWideColsForAdaptiveHash = 200000;
// Ultra-wide region where dense SPA is usually memory-bandwidth bound.
constexpr std::int64_t kUltraWideCols = 750000;
// In ultra-wide case, extend hash coverage to larger row_work.
constexpr std::int64_t kUltraHashWorkThreshold = 65536;
// For wide matrices, keep dense path only for very heavy rows.
constexpr std::int64_t kHashVsDenseRatio = 8;
// Thread count target derived from total symbolic work.
constexpr std::int64_t kWorkPerThreadTarget = 300000;
// Ultra-wide thread caps are split by total work to avoid over/under-subscription.
constexpr std::int64_t kUltraWideHighWorkThreshold = 1000000000;
constexpr int kUltraWideLowWorkThreadCap = 8;
constexpr int kUltraWideHighWorkThreadCap = 12;

static inline std::uint64_t hash_col(std::int64_t j, std::uint64_t mask) {
    return (static_cast<std::uint64_t>(j) * 11400714819323198485ull) & mask;
}

// Path selection:
// - hash path for low/medium row_work and most ultra-wide rows
// - dense marks+accum only for very heavy rows
// This avoids allocating/scanning huge dense buffers on typical sparse rows.
static inline bool should_use_hash_path(std::int64_t row_work, std::int64_t n_cols_b) {
    if (row_work <= 0) {
        return false;
    }
    if (row_work <= kHashWorkThreshold) {
        return true;
    }
    if (n_cols_b >= kUltraWideCols && row_work <= kUltraHashWorkThreshold) {
        return true;
    }
    if (n_cols_b >= kWideColsForAdaptiveHash && row_work <= (n_cols_b / kHashVsDenseRatio)) {
        return true;
    }
    return false;
}

template <typename IndexT>
static void symbolic_count_range(
    const IndexT* a_indptr,
    const IndexT* a_indices,
    const IndexT* b_indptr,
    const IndexT* b_indices,
    std::int64_t row_start,
    std::int64_t row_end,
    std::int64_t n_cols_b,
    const std::int64_t* prefix_work,
    std::int64_t* row_counts) {
    // Dense symbolic state is allocated lazily, because many workloads never
    // enter dense path after adaptive hash selection.
    std::vector<std::int32_t> marks;
    std::vector<std::int64_t> touched_cols;
    touched_cols.reserve(128);
    std::int32_t mark_id = 1;

    std::vector<std::int64_t> hash_keys;
    std::vector<std::int64_t> touched_slots;
    touched_slots.reserve(256);

    for (std::int64_t i = row_start; i < row_end; ++i) {
        const std::int64_t row_work = prefix_work[i + 1] - prefix_work[i];
        if (should_use_hash_path(row_work, n_cols_b)) {
            // Open-addressing hash set for unique column counting.
            std::int64_t table_size = 16;
            while (table_size < row_work * 2) {
                table_size <<= 1;
            }
            if (static_cast<std::int64_t>(hash_keys.size()) < table_size) {
                hash_keys.assign(static_cast<std::size_t>(table_size), -1);
            }
            touched_slots.clear();
            const std::uint64_t mask = static_cast<std::uint64_t>(table_size - 1);

            const std::int64_t a_begin = static_cast<std::int64_t>(a_indptr[i]);
            const std::int64_t a_end = static_cast<std::int64_t>(a_indptr[i + 1]);
            for (std::int64_t ap = a_begin; ap < a_end; ++ap) {
                const std::int64_t k = static_cast<std::int64_t>(a_indices[ap]);
                const std::int64_t b_begin = static_cast<std::int64_t>(b_indptr[k]);
                const std::int64_t b_end = static_cast<std::int64_t>(b_indptr[k + 1]);
                for (std::int64_t bp = b_begin; bp < b_end; ++bp) {
                    const std::int64_t j = static_cast<std::int64_t>(b_indices[bp]);
                    std::uint64_t slot = hash_col(j, mask);
                    while (true) {
                        const std::int64_t key = hash_keys[static_cast<std::size_t>(slot)];
                        if (key == -1) {
                            hash_keys[static_cast<std::size_t>(slot)] = j;
                            touched_slots.push_back(static_cast<std::int64_t>(slot));
                            break;
                        }
                        if (key == j) {
                            break;
                        }
                        slot = (slot + 1) & mask;
                    }
                }
            }
            row_counts[i] = static_cast<std::int64_t>(touched_slots.size());
            for (const std::int64_t slot_i : touched_slots) {
                hash_keys[static_cast<std::size_t>(slot_i)] = -1;
            }
        } else {
            // Dense marks path for very heavy rows.
            if (marks.empty()) {
                marks.assign(static_cast<std::size_t>(n_cols_b), 0);
            }
            touched_cols.clear();
            const std::int64_t a_begin = static_cast<std::int64_t>(a_indptr[i]);
            const std::int64_t a_end = static_cast<std::int64_t>(a_indptr[i + 1]);
            for (std::int64_t ap = a_begin; ap < a_end; ++ap) {
                const std::int64_t k = static_cast<std::int64_t>(a_indices[ap]);
                const std::int64_t b_begin = static_cast<std::int64_t>(b_indptr[k]);
                const std::int64_t b_end = static_cast<std::int64_t>(b_indptr[k + 1]);
                for (std::int64_t bp = b_begin; bp < b_end; ++bp) {
                    const std::int64_t j = static_cast<std::int64_t>(b_indices[bp]);
                    if (marks[j] != mark_id) {
                        marks[j] = mark_id;
                        touched_cols.push_back(j);
                    }
                }
            }
            row_counts[i] = static_cast<std::int64_t>(touched_cols.size());
            ++mark_id;
            if (mark_id == INT_MAX) {
                std::fill(marks.begin(), marks.end(), 0);
                mark_id = 1;
            }
        }
    }
}

template <typename IndexT>
static void numeric_fill_range(
    const IndexT* a_indptr,
    const IndexT* a_indices,
    const double* a_data,
    const IndexT* b_indptr,
    const IndexT* b_indices,
    const double* b_data,
    std::int64_t row_start,
    std::int64_t row_end,
    std::int64_t n_cols_b,
    const std::int64_t* prefix_work,
    const IndexT* out_indptr,
    IndexT* out_indices,
    double* out_data) {
    // Dense numeric buffers are allocated lazily for the same reason as symbolic.
    std::vector<std::int32_t> marks;
    std::vector<double> accum;
    std::vector<std::int64_t> touched_cols;
    touched_cols.reserve(128);
    std::int32_t mark_id = 1;

    std::vector<std::int64_t> hash_keys;
    std::vector<double> hash_vals;
    std::vector<std::int64_t> touched_slots;
    touched_slots.reserve(256);

    for (std::int64_t i = row_start; i < row_end; ++i) {
        const std::int64_t start = static_cast<std::int64_t>(out_indptr[i]);
        const std::int64_t end = static_cast<std::int64_t>(out_indptr[i + 1]);
        const std::int64_t row_work = prefix_work[i + 1] - prefix_work[i];

        if (should_use_hash_path(row_work, n_cols_b)) {
            // Hash accumulator: avoids touching O(n_cols_b) memory on wide rows.
            std::int64_t table_size = 16;
            while (table_size < row_work * 2) {
                table_size <<= 1;
            }
            if (static_cast<std::int64_t>(hash_keys.size()) < table_size) {
                hash_keys.assign(static_cast<std::size_t>(table_size), -1);
                hash_vals.assign(static_cast<std::size_t>(table_size), 0.0);
            }
            touched_slots.clear();
            const std::uint64_t mask = static_cast<std::uint64_t>(table_size - 1);

            const std::int64_t a_begin = static_cast<std::int64_t>(a_indptr[i]);
            const std::int64_t a_end = static_cast<std::int64_t>(a_indptr[i + 1]);
            for (std::int64_t ap = a_begin; ap < a_end; ++ap) {
                const std::int64_t k = static_cast<std::int64_t>(a_indices[ap]);
                const double a_val = a_data[ap];
                const std::int64_t b_begin = static_cast<std::int64_t>(b_indptr[k]);
                const std::int64_t b_end = static_cast<std::int64_t>(b_indptr[k + 1]);
                for (std::int64_t bp = b_begin; bp < b_end; ++bp) {
                    const std::int64_t j = static_cast<std::int64_t>(b_indices[bp]);
                    const double prod = a_val * b_data[bp];
                    std::uint64_t slot = hash_col(j, mask);
                    while (true) {
                        const std::int64_t key = hash_keys[static_cast<std::size_t>(slot)];
                        if (key == -1) {
                            hash_keys[static_cast<std::size_t>(slot)] = j;
                            hash_vals[static_cast<std::size_t>(slot)] = prod;
                            touched_slots.push_back(static_cast<std::int64_t>(slot));
                            break;
                        }
                        if (key == j) {
                            hash_vals[static_cast<std::size_t>(slot)] += prod;
                            break;
                        }
                        slot = (slot + 1) & mask;
                    }
                }
            }

            std::int64_t pos = start;
            for (const std::int64_t slot_i : touched_slots) {
                const std::size_t s = static_cast<std::size_t>(slot_i);
                out_indices[pos] = static_cast<IndexT>(hash_keys[s]);
                out_data[pos] = hash_vals[s];
                hash_keys[s] = -1;
                ++pos;
            }
            if (pos != end) {
                throw std::runtime_error("numeric fill mismatch in hash path");
            }
        } else {
            // Dense SPA accumulator for extremely heavy rows.
            if (marks.empty()) {
                marks.assign(static_cast<std::size_t>(n_cols_b), 0);
                accum.resize(static_cast<std::size_t>(n_cols_b));
            }
            touched_cols.clear();
            const std::int64_t a_begin = static_cast<std::int64_t>(a_indptr[i]);
            const std::int64_t a_end = static_cast<std::int64_t>(a_indptr[i + 1]);
            for (std::int64_t ap = a_begin; ap < a_end; ++ap) {
                const std::int64_t k = static_cast<std::int64_t>(a_indices[ap]);
                const double a_val = a_data[ap];
                const std::int64_t b_begin = static_cast<std::int64_t>(b_indptr[k]);
                const std::int64_t b_end = static_cast<std::int64_t>(b_indptr[k + 1]);
                for (std::int64_t bp = b_begin; bp < b_end; ++bp) {
                    const std::int64_t j = static_cast<std::int64_t>(b_indices[bp]);
                    const double prod = a_val * b_data[bp];
                    if (marks[j] != mark_id) {
                        marks[j] = mark_id;
                        accum[j] = prod;
                        touched_cols.push_back(j);
                    } else {
                        accum[j] += prod;
                    }
                }
            }

            std::int64_t pos = start;
            for (const std::int64_t j : touched_cols) {
                out_indices[pos] = static_cast<IndexT>(j);
                out_data[pos] = accum[j];
                ++pos;
            }
            if (pos != end) {
                throw std::runtime_error("numeric fill mismatch in dense path");
            }

            ++mark_id;
            if (mark_id == INT_MAX) {
                std::fill(marks.begin(), marks.end(), 0);
                mark_id = 1;
            }
        }
    }
}

template <typename IndexT>
static py::tuple csr_spgemm_f64_impl(
    py::array_t<IndexT, py::array::c_style | py::array::forcecast> a_indptr_arr,
    py::array_t<IndexT, py::array::c_style | py::array::forcecast> a_indices_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> a_data_arr,
    std::int64_t a_rows,
    std::int64_t a_cols,
    py::array_t<IndexT, py::array::c_style | py::array::forcecast> b_indptr_arr,
    py::array_t<IndexT, py::array::c_style | py::array::forcecast> b_indices_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> b_data_arr,
    std::int64_t b_cols,
    int num_threads) {
    if (a_rows < 0 || a_cols < 0 || b_cols < 0) {
        throw std::invalid_argument("negative shape is not allowed");
    }
    if (a_indptr_arr.ndim() != 1 || a_indices_arr.ndim() != 1 || a_data_arr.ndim() != 1 ||
        b_indptr_arr.ndim() != 1 || b_indices_arr.ndim() != 1 || b_data_arr.ndim() != 1) {
        throw std::invalid_argument("all CSR arrays must be 1D");
    }
    if (a_indptr_arr.shape(0) != a_rows + 1) {
        throw std::invalid_argument("a_indptr length must be a_rows + 1");
    }
    if (b_indptr_arr.shape(0) != a_cols + 1) {
        throw std::invalid_argument("b_indptr length must be a_cols + 1");
    }
    if (a_indices_arr.shape(0) != a_data_arr.shape(0)) {
        throw std::invalid_argument("a_indices and a_data must have same length");
    }
    if (b_indices_arr.shape(0) != b_data_arr.shape(0)) {
        throw std::invalid_argument("b_indices and b_data must have same length");
    }

    const auto* a_indptr = static_cast<const IndexT*>(a_indptr_arr.data());
    const auto* a_indices = static_cast<const IndexT*>(a_indices_arr.data());
    const auto* a_data = static_cast<const double*>(a_data_arr.data());
    const auto* b_indptr = static_cast<const IndexT*>(b_indptr_arr.data());
    const auto* b_indices = static_cast<const IndexT*>(b_indices_arr.data());
    const auto* b_data = static_cast<const double*>(b_data_arr.data());

    if (num_threads <= 0) {
        num_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (num_threads <= 0) {
            num_threads = 1;
        }
    }
    if (a_rows == 0) {
        py::array_t<IndexT> out_indptr(a_rows + 1);
        py::array_t<IndexT> out_indices(0);
        py::array_t<double> out_data(0);
        auto* out_indptr_ptr = static_cast<IndexT*>(out_indptr.mutable_data());
        out_indptr_ptr[0] = 0;
        return py::make_tuple(out_indptr, out_indices, out_data);
    }
    num_threads = std::max(1, std::min(num_threads, static_cast<int>(a_rows)));

    // prefix_work[i+1]-prefix_work[i] is row i's symbolic effort:
    // sum_{k in A[i,:]} nnz(B[k,:]).
    std::vector<std::int64_t> prefix_work(static_cast<std::size_t>(a_rows + 1), 0);
    for (std::int64_t i = 0; i < a_rows; ++i) {
        std::int64_t row_work = 0;
        const std::int64_t a_begin = static_cast<std::int64_t>(a_indptr[i]);
        const std::int64_t a_end = static_cast<std::int64_t>(a_indptr[i + 1]);
        for (std::int64_t ap = a_begin; ap < a_end; ++ap) {
            const std::int64_t k = static_cast<std::int64_t>(a_indices[ap]);
            row_work += static_cast<std::int64_t>(b_indptr[k + 1]) - static_cast<std::int64_t>(b_indptr[k]);
        }
        prefix_work[static_cast<std::size_t>(i + 1)] = prefix_work[static_cast<std::size_t>(i)] + row_work;
    }
    const std::int64_t total_work = prefix_work.back();
    if (total_work == 0) {
        py::array_t<IndexT> out_indptr(a_rows + 1);
        py::array_t<IndexT> out_indices(0);
        py::array_t<double> out_data(0);
        auto* out_indptr_ptr = static_cast<IndexT*>(out_indptr.mutable_data());
        for (std::int64_t i = 0; i <= a_rows; ++i) {
            out_indptr_ptr[i] = 0;
        }
        return py::make_tuple(out_indptr, out_indices, out_data);
    }

    const int thread_limit_by_work =
        static_cast<int>(std::max<std::int64_t>(1, total_work / kWorkPerThreadTarget));
    num_threads = std::max(1, std::min(num_threads, thread_limit_by_work));
    if (b_cols >= kUltraWideCols) {
        // Extra guard for ultra-wide outputs:
        // lower cap for moderate work, higher cap for very high work.
        const int ultra_cap = (total_work >= kUltraWideHighWorkThreshold) ? kUltraWideHighWorkThreadCap
                                                                           : kUltraWideLowWorkThreadCap;
        num_threads = std::max(1, std::min(num_threads, ultra_cap));
    }
    num_threads = std::max(1, std::min(num_threads, static_cast<int>(a_rows)));

    // Work-balanced row partitioning by prefix_work, not by row count.
    std::vector<std::int64_t> boundaries(static_cast<std::size_t>(num_threads + 1), 0);
    boundaries[0] = 0;
    boundaries[static_cast<std::size_t>(num_threads)] = a_rows;
    std::size_t search_from = 0;
    for (int t = 1; t < num_threads; ++t) {
        const std::int64_t target = (total_work * t) / num_threads;
        const auto it = std::lower_bound(
            prefix_work.begin() + static_cast<std::ptrdiff_t>(search_from),
            prefix_work.end(),
            target);
        std::size_t row = static_cast<std::size_t>(std::distance(prefix_work.begin(), it));
        if (row < static_cast<std::size_t>(boundaries[static_cast<std::size_t>(t - 1)])) {
            row = static_cast<std::size_t>(boundaries[static_cast<std::size_t>(t - 1)]);
        }
        if (row > static_cast<std::size_t>(a_rows)) {
            row = static_cast<std::size_t>(a_rows);
        }
        boundaries[static_cast<std::size_t>(t)] = static_cast<std::int64_t>(row);
        search_from = row;
    }

    std::vector<std::int64_t> row_counts(static_cast<std::size_t>(a_rows), 0);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(num_threads - 1));

    // Phase 1: symbolic count (compute nnz per output row).
    for (int t = 0; t < num_threads; ++t) {
        const std::int64_t row_start = boundaries[static_cast<std::size_t>(t)];
        const std::int64_t row_end = boundaries[static_cast<std::size_t>(t + 1)];
        if (t == num_threads - 1) {
            symbolic_count_range<IndexT>(
                a_indptr,
                a_indices,
                b_indptr,
                b_indices,
                row_start,
                row_end,
                b_cols,
                prefix_work.data(),
                row_counts.data());
        } else {
            workers.emplace_back(
                [&, row_start, row_end]() {
                    symbolic_count_range<IndexT>(
                        a_indptr,
                        a_indices,
                        b_indptr,
                        b_indices,
                        row_start,
                        row_end,
                        b_cols,
                        prefix_work.data(),
                        row_counts.data());
                });
        }
    }
    for (auto& th : workers) {
        th.join();
    }

    py::array_t<IndexT> out_indptr(a_rows + 1);
    auto* out_indptr_ptr = static_cast<IndexT*>(out_indptr.mutable_data());
    out_indptr_ptr[0] = 0;
    std::int64_t total_nnz = 0;
    for (std::int64_t i = 0; i < a_rows; ++i) {
        total_nnz += row_counts[static_cast<std::size_t>(i)];
        if constexpr (std::is_same_v<IndexT, std::int32_t>) {
            // i32 kernel keeps output index arrays in int32 to cut index memory
            // bandwidth on large problems; guard against overflow.
            if (total_nnz > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
                throw std::overflow_error("nnz exceeds int32 capacity in i32 kernel");
            }
        }
        out_indptr_ptr[i + 1] = static_cast<IndexT>(total_nnz);
    }

    py::array_t<IndexT> out_indices(total_nnz);
    py::array_t<double> out_data(total_nnz);
    auto* out_indices_ptr = static_cast<IndexT*>(out_indices.mutable_data());
    auto* out_data_ptr = static_cast<double*>(out_data.mutable_data());

    workers.clear();
    workers.reserve(static_cast<std::size_t>(num_threads - 1));
    // Phase 2: numeric fill (write indices/data).
    for (int t = 0; t < num_threads; ++t) {
        const std::int64_t row_start = boundaries[static_cast<std::size_t>(t)];
        const std::int64_t row_end = boundaries[static_cast<std::size_t>(t + 1)];
        if (t == num_threads - 1) {
            numeric_fill_range<IndexT>(
                a_indptr,
                a_indices,
                a_data,
                b_indptr,
                b_indices,
                b_data,
                row_start,
                row_end,
                b_cols,
                prefix_work.data(),
                out_indptr_ptr,
                out_indices_ptr,
                out_data_ptr);
        } else {
            workers.emplace_back(
                [&, row_start, row_end]() {
                    numeric_fill_range<IndexT>(
                        a_indptr,
                        a_indices,
                        a_data,
                        b_indptr,
                        b_indices,
                        b_data,
                        row_start,
                        row_end,
                        b_cols,
                        prefix_work.data(),
                        out_indptr_ptr,
                        out_indices_ptr,
                        out_data_ptr);
                });
        }
    }
    for (auto& th : workers) {
        th.join();
    }

    return py::make_tuple(out_indptr, out_indices, out_data);
}

}  // namespace

PYBIND11_MODULE(_spgemm_cpp, m) {
    m.doc() = "Generic CSR x CSR SpGEMM kernel (float64) with symbolic/numeric phases.";
    m.def(
        "csr_spgemm_f64_i32",
        &csr_spgemm_f64_impl<std::int32_t>,
        py::arg("a_indptr"),
        py::arg("a_indices"),
        py::arg("a_data"),
        py::arg("a_rows"),
        py::arg("a_cols"),
        py::arg("b_indptr"),
        py::arg("b_indices"),
        py::arg("b_data"),
        py::arg("b_cols"),
        py::arg("num_threads") = 0);
    m.def(
        "csr_spgemm_f64_i64",
        &csr_spgemm_f64_impl<std::int64_t>,
        py::arg("a_indptr"),
        py::arg("a_indices"),
        py::arg("a_data"),
        py::arg("a_rows"),
        py::arg("a_cols"),
        py::arg("b_indptr"),
        py::arg("b_indices"),
        py::arg("b_data"),
        py::arg("b_cols"),
        py::arg("num_threads") = 0);
}
