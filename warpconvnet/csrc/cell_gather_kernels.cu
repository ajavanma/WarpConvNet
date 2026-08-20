// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Capped cell-neighbourhood gather for point-transformer set abstraction.
//
// ONE kernel family serves both rules. They share the whole 3x3x3 cell-block
// traversal and differ only in the compile-time flag USE_RADIUS:
//
//   USE_RADIUS = false  -> voxel_block_gather: no distance test at all.
//   USE_RADIUS = true   -> capped_ball_query: true d^2 <= radius^2.
//
// Both return the `nsample` LOWEST point indices of the neighbourhood in
// ascending order -- the same selection rule as `_select_first_k` in
// models/ppat/model.py. Taking the first `nsample` in cell traversal order
// would implement a different neighbourhood rule.
//
// Work assignment: one warp per query. Lanes 0..26 each own one of the 27 cell
// offsets and walk that cell's points, which the host has sorted so that every
// cell's members appear in ascending point index. Each pop of the merge is a
// warp-wide min over the 27 lane heads, so the output falls out in ascending
// order with no sort and the traversal stops after `nsample` accepted hits.
//
// Uses only __shfl_xor_sync / __ballot_sync (sm_35+, no inline PTX), so it
// carries no arch guard and cannot break the sm_90a CI wheel builds.

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "cuhash/hash_functions.cuh"
#include "cuhash/hash_table.cuh"

using namespace cuhash;

namespace {

constexpr int kWarpSize = 32;
constexpr int kNumCellOffsets = 27;
constexpr int kSentinel = 0x7FFFFFFF;

// Butterfly min over the whole warp. Every lane leaves with the same value.
__device__ __forceinline__ int warp_min(int v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    int other = __shfl_xor_sync(0xFFFFFFFFu, v, offset);
    v = other < v ? other : v;
  }
  return v;
}

// Advance this lane's cursor to the next point of its cell that qualifies, and
// return that point's global index (kSentinel when the cell is exhausted). The
// cursor is left ON the returned element -- the caller steps past it only once
// the element has actually been popped.
template <bool USE_RADIUS>
__device__ __forceinline__ int refill(const float *__restrict__ points,
                                      const int *__restrict__ sorted_order,
                                      int &cursor,
                                      int cell_end,
                                      int N,
                                      float qx,
                                      float qy,
                                      float qz,
                                      float radius_sq) {
  while (cursor < cell_end) {
    const int pi = sorted_order[cursor];
    if (pi < 0 || pi >= N) {
      ++cursor;
      continue;
    }
    if (!USE_RADIUS) return pi;
    // Compute d^2 directly from coordinate differences. This can differ by a
    // few ulp at the radius boundary from an expanded -2ab + |a|^2 + |b|^2
    // implementation because the floating-point operation order differs.
    const float dx = qx - points[3 * pi + 0];
    const float dy = qy - points[3 * pi + 1];
    const float dz = qz - points[3 * pi + 2];
    if (dx * dx + dy * dy + dz * dz <= radius_sq) return pi;
    ++cursor;
  }
  return kSentinel;
}

template <bool USE_RADIUS>
__global__ void cell_gather_kernel(const float *__restrict__ points,
                                   const float *__restrict__ queries,
                                   const int *__restrict__ query_batch,
                                   const int *__restrict__ ref_offsets,
                                   const int *__restrict__ sorted_order,
                                   const int *__restrict__ cell_starts,
                                   const int *__restrict__ cell_counts,
                                   const uint64_t *__restrict__ keys,
                                   const int *__restrict__ values,
                                   int *__restrict__ out,
                                   int M,
                                   int N,
                                   int B,
                                   int num_cells,
                                   int nsample,
                                   float cell_size,
                                   float radius_sq,
                                   uint32_t capacity_mask) {
  extern __shared__ int smem[];

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int query = blockIdx.x * (blockDim.x / kWarpSize) + warp_in_block;
  if (query >= M) return;

  int *picked = smem + warp_in_block * nsample;
  int *dst = out + static_cast<int64_t>(query) * nsample;

  const float qx = queries[3 * query + 0];
  const float qy = queries[3 * query + 1];
  const float qz = queries[3 * query + 2];
  const int b = query_batch[query];

  // The Python wrapper constructs query_batch, but the low-level binding is
  // also callable directly. Fail closed rather than indexing ref_offsets with
  // an invalid batch id; checking here avoids a device-to-host synchronization.
  if (b < 0 || b >= B) {
    for (int k = lane; k < nsample; k += kWarpSize) dst[k] = -1;
    return;
  }

  // floorf(q / cell_size) must match the host's torch.floor(q / cell_size)
  // bit-for-bit, or a query lands in a different cell than its own points.
  const float cxf = floorf(qx / cell_size);
  const float cyf = floorf(qy / cell_size);
  const float czf = floorf(qz / cell_size);
  if (!isfinite(cxf) || !isfinite(cyf) || !isfinite(czf) || cxf < kCoordMin || cxf > kCoordMax ||
      cyf < kCoordMin || cyf > kCoordMax || czf < kCoordMin || czf > kCoordMax) {
    for (int k = lane; k < nsample; k += kWarpSize) dst[k] = -1;
    return;
  }
  const int cx = static_cast<int>(cxf);
  const int cy = static_cast<int>(cyf);
  const int cz = static_cast<int>(czf);

  // Lanes 0..26 take one cell each; 27..31 sit out as permanently exhausted
  // lists so they contribute only kSentinel to every min.
  int cursor = 0;
  int cell_end = 0;
  if (lane < kNumCellOffsets) {
    const int dx = lane / 9 - 1;
    const int dy = (lane / 3) % 3 - 1;
    const int dz = lane % 3 - 1;
    const int64_t nx = static_cast<int64_t>(cx) + dx;
    const int64_t ny = static_cast<int64_t>(cy) + dy;
    const int64_t nz = static_cast<int64_t>(cz) + dz;
    // pack_key_4d masks each spatial coordinate to 18 bits. Do not let a
    // neighbour just outside that range wrap from +131072 to -131072 (or the
    // reverse) and turn a far-away cell into a false neighbour.
    if (nx >= kCoordMin && nx <= kCoordMax && ny >= kCoordMin && ny <= kCoordMax &&
        nz >= kCoordMin && nz <= kCoordMax) {
      const uint64_t packed =
          pack_key_4d(b, static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(nz));
      const int cell_id = packed_search(keys, values, packed, capacity_mask);
      if (cell_id >= 0 && cell_id < num_cells) {
        const int start = cell_starts[cell_id];
        const int count = cell_counts[cell_id];
        if (start >= 0 && start <= N && count >= 0 && count <= N - start) {
          cursor = start;
          cell_end = start + count;
        }
      }
    }
  }

  int head = refill<USE_RADIUS>(points, sorted_order, cursor, cell_end, N, qx, qy, qz, radius_sq);

  // 27-way merge. `m` is warp-uniform, so the loop bound and the break are
  // uniform too and every lane reaches each __shfl_xor_sync.
  int found = 0;
  for (int k = 0; k < nsample; ++k) {
    const int m = warp_min(head);
    if (m == kSentinel) break;
    if (lane == 0) picked[k] = m;
    if (head == m) {
      // Point indices are globally unique and each cell is visited by exactly
      // one lane, so precisely one lane takes this branch.
      ++cursor;
      head = refill<USE_RADIUS>(points, sorted_order, cursor, cell_end, N, qx, qy, qz, radius_sq);
    }
    found = k + 1;
  }
  __syncwarp();

  // Short groups repeat their first hit; empty groups take the batch-local 0.
  const int batch_start = ref_offsets[b];
  const int fill = found > 0 ? picked[0] : (batch_start >= 0 && batch_start < N ? batch_start : -1);
  for (int k = lane; k < nsample; k += kWarpSize) {
    dst[k] = k < found ? picked[k] : fill;
  }
}

}  // namespace

void coords_cell_gather(torch::Tensor points,
                        torch::Tensor queries,
                        torch::Tensor query_batch,
                        torch::Tensor ref_offsets,
                        torch::Tensor sorted_order,
                        torch::Tensor cell_starts,
                        torch::Tensor cell_counts,
                        torch::Tensor keys,
                        torch::Tensor values,
                        torch::Tensor out,
                        int num_cells,
                        int nsample,
                        float cell_size,
                        float radius_sq,
                        bool use_radius,
                        int capacity) {
  TORCH_CHECK(nsample > 0, "nsample must be positive, got ", nsample);
  TORCH_CHECK(std::isfinite(cell_size) && cell_size > 0.0f,
              "cell_size must be finite and positive, got ",
              cell_size);
  TORCH_CHECK(!use_radius || (std::isfinite(radius_sq) && radius_sq > 0.0f),
              "radius_sq must be finite and positive when use_radius is true, got ",
              radius_sq);

  TORCH_CHECK(points.defined() && points.is_cuda(), "points must be a CUDA tensor");
  const auto device = points.device();
  auto check_cuda_device = [&](const torch::Tensor &tensor, const char *name) {
    TORCH_CHECK(tensor.defined() && tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.device() == device, name, " must be on ", device, ", got ", tensor.device());
  };
  check_cuda_device(queries, "queries");
  check_cuda_device(query_batch, "query_batch");
  check_cuda_device(ref_offsets, "ref_offsets");
  check_cuda_device(sorted_order, "sorted_order");
  check_cuda_device(cell_starts, "cell_starts");
  check_cuda_device(cell_counts, "cell_counts");
  check_cuda_device(keys, "keys");
  check_cuda_device(values, "values");
  check_cuda_device(out, "out");

  TORCH_CHECK(points.dim() == 2 && points.size(1) == 3 && points.is_contiguous() &&
                  points.scalar_type() == at::kFloat,
              "points must be a contiguous (N, 3) float32 tensor");
  TORCH_CHECK(queries.dim() == 2 && queries.size(1) == 3 && queries.is_contiguous() &&
                  queries.scalar_type() == at::kFloat,
              "queries must be a contiguous (M, 3) float32 tensor");

  const int64_t N64 = points.size(0);
  const int64_t M64 = queries.size(0);
  TORCH_CHECK(N64 <= std::numeric_limits<int>::max() && M64 <= std::numeric_limits<int>::max(),
              "points and queries must fit in int32 indexing");
  const int N = static_cast<int>(N64);
  const int M = static_cast<int>(M64);

  TORCH_CHECK(query_batch.dim() == 1 && query_batch.numel() == M && query_batch.is_contiguous() &&
                  query_batch.scalar_type() == at::kInt,
              "query_batch must be a contiguous (M,) int32 tensor");
  TORCH_CHECK(ref_offsets.dim() == 1 && ref_offsets.numel() >= (M > 0 ? 2 : 1) &&
                  ref_offsets.numel() <= kBatchMax + 2 && ref_offsets.is_contiguous() &&
                  ref_offsets.scalar_type() == at::kInt,
              "ref_offsets must be a contiguous (B+1,) int32 tensor with B in [0, ",
              kBatchMax + 1,
              "]");
  const int B = static_cast<int>(ref_offsets.numel() - 1);
  TORCH_CHECK(sorted_order.dim() == 1 && sorted_order.numel() == N &&
                  sorted_order.is_contiguous() && sorted_order.scalar_type() == at::kInt,
              "sorted_order must be a contiguous (N,) int32 tensor");
  TORCH_CHECK(num_cells >= 0, "num_cells must be non-negative, got ", num_cells);
  TORCH_CHECK(cell_starts.dim() == 1 && cell_starts.numel() == num_cells &&
                  cell_starts.is_contiguous() && cell_starts.scalar_type() == at::kInt,
              "cell_starts must be a contiguous (num_cells,) int32 tensor");
  TORCH_CHECK(cell_counts.dim() == 1 && cell_counts.numel() == num_cells &&
                  cell_counts.is_contiguous() && cell_counts.scalar_type() == at::kInt,
              "cell_counts must be a contiguous (num_cells,) int32 tensor");
  TORCH_CHECK(capacity > 0 && (capacity & (capacity - 1)) == 0,
              "hash table capacity must be a power of two, got ",
              capacity);
  TORCH_CHECK(keys.dim() == 1 && keys.numel() == capacity && keys.is_contiguous() &&
                  keys.scalar_type() == at::kLong,
              "keys must be a contiguous (capacity,) int64 tensor");
  TORCH_CHECK(values.dim() == 1 && values.numel() == capacity && values.is_contiguous() &&
                  values.scalar_type() == at::kInt,
              "values must be a contiguous (capacity,) int32 tensor");
  TORCH_CHECK(out.dim() == 2 && out.size(0) == M && out.size(1) == nsample && out.is_contiguous() &&
                  out.scalar_type() == at::kInt,
              "out must be a contiguous (M, nsample) int32 tensor");

  const c10::cuda::CUDAGuard device_guard(device);
  if (M == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  // Each warp keeps its `nsample` picks in shared memory. Shrink the block
  // rather than overrun the 48 KB static limit when a caller asks for a very
  // large nsample -- silently truncating the buffer would corrupt neighbouring
  // warps' output instead of failing.
  const int kMaxSharedInts = 48 * 1024 / static_cast<int>(sizeof(int));
  TORCH_CHECK(nsample <= kMaxSharedInts,
              "nsample=",
              nsample,
              " exceeds the ",
              kMaxSharedInts,
              " ints of shared memory available to a single warp");
  int warps_per_block = kMaxSharedInts / nsample;
  warps_per_block = warps_per_block > 8 ? 8 : warps_per_block;
  const int threads = warps_per_block * kWarpSize;
  const int blocks = (M + warps_per_block - 1) / warps_per_block;
  const size_t shmem = static_cast<size_t>(warps_per_block) * nsample * sizeof(int);
  const uint32_t capacity_mask = static_cast<uint32_t>(capacity - 1);

  auto launch = [&](auto use_radius_tag) {
    constexpr bool kUseRadius = decltype(use_radius_tag)::value;
    cell_gather_kernel<kUseRadius><<<blocks, threads, shmem, stream>>>(
        points.data_ptr<float>(),
        queries.data_ptr<float>(),
        query_batch.data_ptr<int>(),
        ref_offsets.data_ptr<int>(),
        sorted_order.data_ptr<int>(),
        cell_starts.data_ptr<int>(),
        cell_counts.data_ptr<int>(),
        reinterpret_cast<const uint64_t *>(keys.data_ptr<int64_t>()),
        values.data_ptr<int>(),
        out.data_ptr<int>(),
        M,
        N,
        B,
        num_cells,
        nsample,
        cell_size,
        radius_sq,
        capacity_mask);
  };

  if (use_radius) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
