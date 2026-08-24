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
// A separate cell_nearest_k kernel below scans bounded Chebyshev shells and
// maintains the true nearest k in a fixed per-query heap. It is independent of
// hash insertion and cell traversal order, and reports when the shell budget
// cannot prove exactness. Point index remains the tie-breaker for equal distances.
//
// Uses CUDA warp intrinsics only (sm_35+, no inline PTX), so it carries no
// architecture guard and cannot break the sm_90a CI wheel builds.

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <torch/extension.h>

#include <cfloat>
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

// Status bits written by cell_nearest_k_kernel.  A zero status means that the
// returned neighbours are the exact k nearest points in the query's batch.
// Callers may safely fall back to dense kNN for every non-zero status.
constexpr int kCellNearestInvalid = 1;
constexpr int kCellNearestBudgetExhausted = 2;
constexpr int kCellNearestUnderfilled = 4;
constexpr int kCellNearestCoordClipped = 8;
constexpr int kMaxNearestK = 64;
constexpr int kMaxShellBudget = 64;

struct NearestCandidate {
  float distance_sq;
  int index;
};

// A total order makes the selected set independent of cell traversal and
// thread scheduling: distance first, then global point index for exact ties.
__device__ __forceinline__ bool candidate_better(const NearestCandidate &a,
                                                 const NearestCandidate &b) {
  return a.distance_sq < b.distance_sq || (a.distance_sq == b.distance_sq && a.index < b.index);
}

__device__ __forceinline__ bool candidate_worse(const NearestCandidate &a,
                                                const NearestCandidate &b) {
  return candidate_better(b, a);
}

__device__ __forceinline__ void heap_sift_up(NearestCandidate *heap, int position) {
  while (position > 0) {
    const int parent = (position - 1) >> 1;
    if (!candidate_worse(heap[position], heap[parent])) break;
    const NearestCandidate tmp = heap[parent];
    heap[parent] = heap[position];
    heap[position] = tmp;
    position = parent;
  }
}

__device__ __forceinline__ void heap_sift_down(NearestCandidate *heap, int size, int position) {
  while (true) {
    const int left = 2 * position + 1;
    if (left >= size) break;
    const int right = left + 1;
    int worse_child = left;
    if (right < size && candidate_worse(heap[right], heap[left])) worse_child = right;
    if (!candidate_worse(heap[worse_child], heap[position])) break;
    const NearestCandidate tmp = heap[position];
    heap[position] = heap[worse_child];
    heap[worse_child] = tmp;
    position = worse_child;
  }
}

__device__ __forceinline__ void heap_insert(NearestCandidate *heap,
                                            int &size,
                                            int capacity,
                                            const NearestCandidate &candidate) {
  if (size < capacity) {
    heap[size] = candidate;
    heap_sift_up(heap, size);
    ++size;
    return;
  }
  if (!candidate_better(candidate, heap[0])) return;
  heap[0] = candidate;
  heap_sift_down(heap, size, 0);
}

// In-place max-heap sort.  The result is ascending in the same deterministic
// (distance, point-index) order used for selection.
__device__ __forceinline__ void heap_sort(NearestCandidate *heap, int size) {
  for (int end = size - 1; end > 0; --end) {
    const NearestCandidate tmp = heap[0];
    heap[0] = heap[end];
    heap[end] = tmp;
    heap_sift_down(heap, end, 0);
  }
}

// Lower bound on the distance to every cell outside the completed Chebyshev
// shell.  The subtraction makes the bound conservative with respect to the
// float division used to assign points to cells.  We require a strict
// kth-distance comparison so an unseen equal-distance point with a lower index
// cannot change the deterministic result.
__device__ __forceinline__ float unvisited_distance_lower_bound_sq(
    float qx, float qy, float qz, int cx, int cy, int cz, int shell, float cell_size) {
  const float lower_x = static_cast<float>(cx - shell) * cell_size;
  const float lower_y = static_cast<float>(cy - shell) * cell_size;
  const float lower_z = static_cast<float>(cz - shell) * cell_size;
  const float upper_x = static_cast<float>(cx + shell + 1) * cell_size;
  const float upper_y = static_cast<float>(cy + shell + 1) * cell_size;
  const float upper_z = static_cast<float>(cz + shell + 1) * cell_size;

  float bound = fminf(fminf(qx - lower_x, upper_x - qx),
                      fminf(fminf(qy - lower_y, upper_y - qy), fminf(qz - lower_z, upper_z - qz)));
  float scale = fmaxf(1.0f, fabsf(cell_size));
  scale = fmaxf(scale, fabsf(qx));
  scale = fmaxf(scale, fabsf(qy));
  scale = fmaxf(scale, fabsf(qz));
  scale = fmaxf(scale, fabsf(lower_x));
  scale = fmaxf(scale, fabsf(lower_y));
  scale = fmaxf(scale, fabsf(lower_z));
  scale = fmaxf(scale, fabsf(upper_x));
  scale = fmaxf(scale, fabsf(upper_y));
  scale = fmaxf(scale, fabsf(upper_z));
  bound = fmaxf(0.0f, bound - 8.0f * FLT_EPSILON * scale);
  return bound * bound;
}

// Fused, order-independent voxel-shell nearest-k.  One warp owns one query.
// A group of 32 lanes resolves 32 cells in parallel, prefix-flattens their
// point ranges, and evaluates those points in full-warp batches.  Lane zero
// maintains only the best k candidates in a shared max heap; no M x candidate
// buffer is ever materialized.
__global__ void cell_nearest_k_kernel(const float *__restrict__ points,
                                      const float *__restrict__ queries,
                                      const int *__restrict__ query_batch,
                                      const int *__restrict__ ref_offsets,
                                      const int *__restrict__ sorted_order,
                                      const int *__restrict__ cell_starts,
                                      const int *__restrict__ cell_counts,
                                      const uint64_t *__restrict__ keys,
                                      const int *__restrict__ values,
                                      int *__restrict__ out_indices,
                                      float *__restrict__ out_distances,
                                      int *__restrict__ out_counts,
                                      int *__restrict__ out_status,
                                      int *__restrict__ out_visited,
                                      int M,
                                      int N,
                                      int B,
                                      int num_cells,
                                      int k,
                                      float cell_size,
                                      int max_shell,
                                      uint32_t capacity_mask) {
  extern __shared__ NearestCandidate shared_heaps[];

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int query = blockIdx.x * (blockDim.x / kWarpSize) + warp_in_block;
  if (query >= M) return;

  NearestCandidate *heap = shared_heaps + warp_in_block * kMaxNearestK;
  int *dst_indices = out_indices + static_cast<int64_t>(query) * k;
  float *dst_distances = out_distances + static_cast<int64_t>(query) * k;

  const float qx = queries[3 * query + 0];
  const float qy = queries[3 * query + 1];
  const float qz = queries[3 * query + 2];
  const int b = query_batch[query];

  const float cxf = floorf(qx / cell_size);
  const float cyf = floorf(qy / cell_size);
  const float czf = floorf(qz / cell_size);
  const bool valid_query = b >= 0 && b < B && isfinite(qx) && isfinite(qy) && isfinite(qz) &&
                           isfinite(cxf) && isfinite(cyf) && isfinite(czf) && cxf >= kCoordMin &&
                           cxf <= kCoordMax && cyf >= kCoordMin && cyf <= kCoordMax &&
                           czf >= kCoordMin && czf <= kCoordMax;

  int batch_start = 0;
  int batch_end = 0;
  if (valid_query) {
    batch_start = ref_offsets[b];
    batch_end = ref_offsets[b + 1];
  }
  const bool valid_batch =
      valid_query && batch_start >= 0 && batch_start <= batch_end && batch_end <= N;
  if (!valid_batch) {
    for (int position = lane; position < k; position += kWarpSize) {
      dst_indices[position] = -1;
      dst_distances[position] = CUDART_INF_F;
    }
    if (lane == 0) {
      out_counts[query] = 0;
      out_visited[query] = 0;
      out_status[query] = kCellNearestInvalid | kCellNearestUnderfilled;
    }
    return;
  }

  const int cx = static_cast<int>(cxf);
  const int cy = static_cast<int>(cyf);
  const int cz = static_cast<int>(czf);
  const int batch_size = batch_end - batch_start;

  int heap_size = 0;
  int visited = 0;
  int status = 0;
  bool exact = false;

  for (int shell = 0; shell <= max_shell; ++shell) {
    const int side = 2 * shell + 1;
    const int inner_side = 2 * shell - 1;
    const int64_t side_sq = static_cast<int64_t>(side) * side;
    // Enumerate only the shell boundary, not the entire cube followed by an
    // interior predicate.  The three disjoint regions are the two X faces,
    // the two Y faces without their X edges, and the two Z faces without X/Y
    // edges.  For shell > 0 this is 24*shell^2 + 2 cells, keeping a search
    // through max_shell O(max_shell^3), rather than O(max_shell^4).
    const int64_t x_face_cells = 2 * side_sq;
    const int64_t y_face_cells = shell > 0 ? 2 * static_cast<int64_t>(inner_side) * side : 0;
    const int64_t z_face_cells = shell > 0 ? 2 * static_cast<int64_t>(inner_side) * inner_side : 0;
    const int64_t shell_cells = shell == 0 ? 1 : x_face_cells + y_face_cells + z_face_cells;

    for (int64_t group_base = 0; group_base < shell_cells; group_base += kWarpSize) {
      const int64_t ordinal = group_base + lane;
      int cell_start = 0;
      int cell_count = 0;
      bool malformed = false;
      bool clipped = false;

      if (ordinal < shell_cells) {
        int ox = 0;
        int oy = 0;
        int oz = 0;
        if (shell > 0 && ordinal < x_face_cells) {
          const int face = static_cast<int>(ordinal / side_sq);
          const int rem = static_cast<int>(ordinal % side_sq);
          ox = face == 0 ? -shell : shell;
          oy = rem / side - shell;
          oz = rem % side - shell;
        } else if (shell > 0 && ordinal < x_face_cells + y_face_cells) {
          const int64_t local = ordinal - x_face_cells;
          const int face_span = inner_side * side;
          const int face = static_cast<int>(local / face_span);
          const int rem = static_cast<int>(local % face_span);
          ox = rem / side - (shell - 1);
          oy = face == 0 ? -shell : shell;
          oz = rem % side - shell;
        } else if (shell > 0) {
          const int64_t local = ordinal - x_face_cells - y_face_cells;
          const int face_span = inner_side * inner_side;
          const int face = static_cast<int>(local / face_span);
          const int rem = static_cast<int>(local % face_span);
          ox = rem / inner_side - (shell - 1);
          oy = rem % inner_side - (shell - 1);
          oz = face == 0 ? -shell : shell;
        }

        const int64_t nx = static_cast<int64_t>(cx) + ox;
        const int64_t ny = static_cast<int64_t>(cy) + oy;
        const int64_t nz = static_cast<int64_t>(cz) + oz;
        if (nx < kCoordMin || nx > kCoordMax || ny < kCoordMin || ny > kCoordMax ||
            nz < kCoordMin || nz > kCoordMax) {
          clipped = true;
        } else {
          const uint64_t packed =
              pack_key_4d(b, static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(nz));
          const int cell_id = packed_search(keys, values, packed, capacity_mask);
          if (cell_id >= num_cells) {
            malformed = true;
          } else if (cell_id >= 0) {
            const int start = cell_starts[cell_id];
            const int count = cell_counts[cell_id];
            if (start < 0 || start > N || count < 0 || count > N - start) {
              malformed = true;
            } else {
              cell_start = start;
              cell_count = count;
            }
          }
        }
      }

      if (__any_sync(0xFFFFFFFFu, clipped)) status |= kCellNearestCoordClipped;
      if (__any_sync(0xFFFFFFFFu, malformed)) status |= kCellNearestInvalid;

      // Inclusive prefix sum of the 32 cell lengths.  It lets every lane map
      // a flattened point ordinal back to its owning cell in five shuffles.
      int prefix_end = cell_count;
#pragma unroll
      for (int offset = 1; offset < kWarpSize; offset <<= 1) {
        const int other = __shfl_up_sync(0xFFFFFFFFu, prefix_end, offset);
        if (lane >= offset) prefix_end += other;
      }
      const int prefix_begin = prefix_end - cell_count;
      const int group_points = __shfl_sync(0xFFFFFFFFu, prefix_end, kWarpSize - 1);

      for (int point_base = 0; point_base < group_points; point_base += kWarpSize) {
        const int flat_point = point_base + lane;
        bool valid_point = flat_point < group_points;
        int point_index = -1;
        float distance_sq = CUDART_INF_F;
        bool bad_point = false;

        // All lanes must participate in every shuffle named by the full-warp
        // mask, including lanes beyond group_points.  Their mapped owner is
        // ignored below because valid_point is false.
        int lo = 0;
        int hi = kWarpSize - 1;
#pragma unroll
        for (int step = 0; step < 5; ++step) {
          const int mid = (lo + hi) >> 1;
          const int end = __shfl_sync(0xFFFFFFFFu, prefix_end, mid);
          if (flat_point < end) {
            hi = mid;
          } else {
            lo = mid + 1;
          }
        }
        const int owner = lo;
        const int owner_start = __shfl_sync(0xFFFFFFFFu, cell_start, owner);
        const int owner_begin = __shfl_sync(0xFFFFFFFFu, prefix_begin, owner);

        if (valid_point) {
          point_index = sorted_order[owner_start + flat_point - owner_begin];
          if (point_index < batch_start || point_index >= batch_end) {
            bad_point = true;
            valid_point = false;
          } else {
            const float dx = qx - points[3 * point_index + 0];
            const float dy = qy - points[3 * point_index + 1];
            const float dz = qz - points[3 * point_index + 2];
            distance_sq = dx * dx + dy * dy + dz * dz;
            if (!isfinite(distance_sq)) {
              bad_point = true;
              valid_point = false;
            }
          }
        }

        if (__any_sync(0xFFFFFFFFu, bad_point)) status |= kCellNearestInvalid;
        const uint32_t valid_mask = __ballot_sync(0xFFFFFFFFu, valid_point);
        visited += __popc(valid_mask);

        // Every lane publishes one candidate through shuffles.  Lane zero is
        // the sole heap writer, so no lock or scheduling-dependent atomic
        // ordering is involved.
#pragma unroll
        for (int source_lane = 0; source_lane < kWarpSize; ++source_lane) {
          NearestCandidate candidate;
          candidate.distance_sq = __shfl_sync(0xFFFFFFFFu, distance_sq, source_lane);
          candidate.index = __shfl_sync(0xFFFFFFFFu, point_index, source_lane);
          if (lane == 0 && (valid_mask & (1u << source_lane)) != 0) {
            heap_insert(heap, heap_size, k, candidate);
          }
        }
        __syncwarp();
        // Make lane zero's heap_size visible as a warp-uniform value before
        // the next candidate batch or shell-level exactness test.
        heap_size = __shfl_sync(0xFFFFFFFFu, heap_size, 0);
      }
    }

    if (visited >= batch_size) {
      exact = true;
      break;
    }
    if (heap_size == k) {
      float kth_distance_sq = lane == 0 ? heap[0].distance_sq : CUDART_INF_F;
      kth_distance_sq = __shfl_sync(0xFFFFFFFFu, kth_distance_sq, 0);
      const float lower_bound_sq =
          unvisited_distance_lower_bound_sq(qx, qy, qz, cx, cy, cz, shell, cell_size);
      if (kth_distance_sq < lower_bound_sq) {
        exact = true;
        break;
      }
    }
  }

  if (!exact) status |= kCellNearestBudgetExhausted;
  if (heap_size < k) status |= kCellNearestUnderfilled;

  if (lane == 0) heap_sort(heap, heap_size);
  __syncwarp();
  for (int position = lane; position < k; position += kWarpSize) {
    if (position < heap_size) {
      dst_indices[position] = heap[position].index;
      dst_distances[position] = heap[position].distance_sq;
    } else {
      dst_indices[position] = -1;
      dst_distances[position] = CUDART_INF_F;
    }
  }
  if (lane == 0) {
    out_counts[query] = heap_size;
    out_visited[query] = visited;
    out_status[query] = status;
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

void coords_cell_nearest_k(torch::Tensor points,
                           torch::Tensor queries,
                           torch::Tensor query_batch,
                           torch::Tensor ref_offsets,
                           torch::Tensor sorted_order,
                           torch::Tensor cell_starts,
                           torch::Tensor cell_counts,
                           torch::Tensor keys,
                           torch::Tensor values,
                           torch::Tensor out_indices,
                           torch::Tensor out_distances,
                           torch::Tensor out_counts,
                           torch::Tensor out_status,
                           torch::Tensor out_visited,
                           int num_cells,
                           int k,
                           float cell_size,
                           int max_shell,
                           int capacity) {
  TORCH_CHECK(k > 0 && k <= kMaxNearestK, "k must be in [1, ", kMaxNearestK, "], got ", k);
  TORCH_CHECK(max_shell >= 0 && max_shell <= kMaxShellBudget,
              "max_shell must be in [0, ",
              kMaxShellBudget,
              "] and is inclusive, got ",
              max_shell);
  TORCH_CHECK(std::isfinite(cell_size) && cell_size > 0.0f,
              "cell_size must be finite and positive, got ",
              cell_size);

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
  check_cuda_device(out_indices, "out_indices");
  check_cuda_device(out_distances, "out_distances");
  check_cuda_device(out_counts, "out_counts");
  check_cuda_device(out_status, "out_status");
  check_cuda_device(out_visited, "out_visited");

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

  TORCH_CHECK(out_indices.dim() == 2 && out_indices.size(0) == M && out_indices.size(1) == k &&
                  out_indices.is_contiguous() && out_indices.scalar_type() == at::kInt,
              "out_indices must be a contiguous (M, k) int32 tensor");
  TORCH_CHECK(out_distances.dim() == 2 && out_distances.size(0) == M &&
                  out_distances.size(1) == k && out_distances.is_contiguous() &&
                  out_distances.scalar_type() == at::kFloat,
              "out_distances must be a contiguous (M, k) float32 tensor");
  auto check_output_vector = [&](const torch::Tensor &tensor, const char *name) {
    TORCH_CHECK(tensor.dim() == 1 && tensor.numel() == M && tensor.is_contiguous() &&
                    tensor.scalar_type() == at::kInt,
                name,
                " must be a contiguous (M,) int32 tensor");
  };
  check_output_vector(out_counts, "out_counts");
  check_output_vector(out_status, "out_status");
  check_output_vector(out_visited, "out_visited");

  const c10::cuda::CUDAGuard device_guard(device);
  if (M == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  constexpr int kWarpsPerBlock = 8;
  constexpr int kThreads = kWarpsPerBlock * kWarpSize;
  const int blocks = (M + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const size_t shmem =
      static_cast<size_t>(kWarpsPerBlock) * kMaxNearestK * sizeof(NearestCandidate);
  const uint32_t capacity_mask = static_cast<uint32_t>(capacity - 1);

  cell_nearest_k_kernel<<<blocks, kThreads, shmem, stream>>>(
      points.data_ptr<float>(),
      queries.data_ptr<float>(),
      query_batch.data_ptr<int>(),
      ref_offsets.data_ptr<int>(),
      sorted_order.data_ptr<int>(),
      cell_starts.data_ptr<int>(),
      cell_counts.data_ptr<int>(),
      reinterpret_cast<const uint64_t *>(keys.data_ptr<int64_t>()),
      values.data_ptr<int>(),
      out_indices.data_ptr<int>(),
      out_distances.data_ptr<float>(),
      out_counts.data_ptr<int>(),
      out_status.data_ptr<int>(),
      out_visited.data_ptr<int>(),
      M,
      N,
      B,
      num_cells,
      k,
      cell_size,
      max_shell,
      capacity_mask);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
