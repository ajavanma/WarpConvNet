# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capped cell-neighbourhood gather for point-transformer set abstraction.

Two rules over the same 3x3x3 cell-block traversal (one CUDA kernel family,
switched by a compile-time flag):

- ``voxel_block_gather`` -- the query's 3x3x3 cell block with **no distance
  test at all**. This is an approximate rule whose cell size controls the patch's
  spatial scale.
- ``capped_ball_query`` -- the same traversal plus the true
  ``d^2 <= radius^2`` test. This implements the PointNet++ ball geometry.

Both return the ``nsample`` **lowest point indices** of the neighbourhood in
ascending order. For the distance-free rule that is not an arbitrary tie-break:
it is exactly the rule ``ppat._select_first_k`` implements. Filling a capped
group in cell order instead would bias the selection toward one part of the block.

Both are capped: the traversal stops after ``nsample`` accepted hits, so unlike
``warpconvnet.geometry.coords.search.search_results.radius_search`` it does not
materialise neighbours that will be discarded after applying the cap.
"""

import math
from numbers import Integral, Real
from typing import Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor

import warpconvnet._C as _C
from warpconvnet.geometry.coords.search.packed_hashmap import PackedHashTable

__all__ = ["voxel_block_gather", "capped_ball_query"]


def _validated_offsets(
    offsets: Tensor,
    *,
    name: str,
    expected_end: int,
    device: torch.device,
    strictly_increasing: bool = False,
) -> Tensor:
    """Return canonical int32 offsets after validating the packed-batch contract."""
    if not isinstance(offsets, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if offsets.ndim != 1 or offsets.numel() == 0:
        raise ValueError(f"{name} must be a non-empty 1D tensor")
    if offsets.dtype == torch.bool or offsets.is_floating_point() or offsets.is_complex():
        raise TypeError(f"{name} must have an integer dtype, got {offsets.dtype}")

    offsets64 = offsets.to(dtype=torch.int64, device=device).contiguous()
    values = offsets64.tolist()
    if values[0] != 0:
        raise ValueError(f"{name} must start at 0")
    if values[-1] != expected_end:
        raise ValueError(f"{name} must end at {expected_end}")
    if any(right < left for left, right in zip(values, values[1:])):
        raise ValueError(f"{name} must be monotonically non-decreasing")
    if strictly_increasing and any(right == left for left, right in zip(values, values[1:])):
        raise ValueError("every ref group must contain at least one point")
    return offsets64.to(dtype=torch.int32)


@torch.no_grad()
def _cell_gather(
    points: Float[Tensor, "N 3"],  # noqa: F821
    ref_offsets: Int[Tensor, "B+1"],  # noqa: F821
    queries: Float[Tensor, "M 3"],  # noqa: F821
    query_offsets: Int[Tensor, "B+1"],  # noqa: F821
    cell_size: float,
    nsample: int,
    radius_sq: Optional[float] = None,
) -> Int[Tensor, "M nsample"]:  # noqa: F821
    """Shared driver: build the cell list, then run the capped 27-way merge.

    ``radius_sq`` of ``None`` selects the distance-free rule; a value selects the
    exact ``d^2 <= radius_sq`` rule.
    """
    if not isinstance(points, Tensor) or not isinstance(queries, Tensor):
        raise TypeError("points and queries must be torch.Tensor objects")
    if not points.is_cuda or not queries.is_cuda:
        raise ValueError("cell gather is CUDA-only")
    if points.device != queries.device:
        raise ValueError(
            f"points and queries must be on the same CUDA device, got "
            f"{points.device} and {queries.device}"
        )
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {tuple(points.shape)}")
    if queries.ndim != 2 or queries.shape[1] != 3:
        raise ValueError(f"queries must have shape (M, 3), got {tuple(queries.shape)}")
    if not points.is_floating_point() or not queries.is_floating_point():
        raise TypeError(
            f"points and queries must have floating-point dtypes, got "
            f"{points.dtype} and {queries.dtype}"
        )

    if isinstance(cell_size, bool) or not isinstance(cell_size, Real):
        raise TypeError(f"cell_size must be a real number, got {type(cell_size).__name__}")
    cell_size = float(cell_size)
    if not math.isfinite(cell_size) or cell_size <= 0:
        raise ValueError(f"cell_size must be finite and positive, got {cell_size}")
    if isinstance(nsample, bool) or not isinstance(nsample, Integral):
        raise TypeError(f"nsample must be an integer, got {type(nsample).__name__}")
    nsample = int(nsample)
    if nsample <= 0:
        raise ValueError(f"nsample must be positive, got {nsample}")
    if radius_sq is not None:
        radius_sq = float(radius_sq)
        if not math.isfinite(radius_sq) or radius_sq <= 0:
            raise ValueError(f"radius_sq must be finite and positive, got {radius_sq}")

    device = points.device
    N = points.shape[0]
    M = queries.shape[0]
    int32_max = torch.iinfo(torch.int32).max
    if N > int32_max or M > int32_max:
        raise ValueError(f"cell gather supports at most {int32_max} points and queries")

    # Keep every allocation and launch on the points device without changing the
    # caller's current device after this function returns.
    with torch.cuda.device(device):
        ref_offsets = _validated_offsets(
            ref_offsets,
            name="ref_offsets",
            expected_end=N,
            device=device,
            strictly_increasing=True,
        )
        query_offsets = _validated_offsets(
            query_offsets, name="query_offsets", expected_end=M, device=device
        )
        if query_offsets.numel() != ref_offsets.numel():
            raise ValueError("ref_offsets and query_offsets must describe the same batch count")

        B = ref_offsets.numel() - 1
        if B > PackedHashTable.BATCH_MAX + 1:
            raise ValueError(f"at most {PackedHashTable.BATCH_MAX + 1} batches are supported")

        points = points.float().contiguous()
        queries = queries.float().contiguous()
        ref_counts = ref_offsets[1:] - ref_offsets[:-1]
        query_counts = query_offsets[1:] - query_offsets[:-1]

        out = torch.empty(M, nsample, dtype=torch.int32, device=device)
        if M == 0:
            return out

        batch_ids = torch.arange(B, device=device, dtype=torch.int32)
        pt_batch = torch.repeat_interleave(batch_ids, ref_counts.long())
        q_batch = torch.repeat_interleave(batch_ids, query_counts.long()).contiguous()

        # The batch id shares the packed uint64 key with the cell coordinate, so a
        # neighbour cell of batch b can never resolve to a point of batch b' -- the
        # kernel needs no separate batch test.
        pt_cells = torch.floor(points / cell_size).int()
        q_cells = torch.floor(queries / cell_size)
        q_cells_min, q_cells_max = torch.stack((q_cells.amin(), q_cells.amax())).tolist()
        if not math.isfinite(q_cells_min) or not math.isfinite(q_cells_max):
            raise ValueError("query coordinates must be finite")
        if q_cells_min < PackedHashTable.COORD_MIN or q_cells_max > PackedHashTable.COORD_MAX:
            raise ValueError(
                "query cell coordinate out of packed range "
                f"[{PackedHashTable.COORD_MIN}, {PackedHashTable.COORD_MAX}]: "
                f"got [{int(q_cells_min)}, {int(q_cells_max)}] (cell_size={cell_size})"
            )

        coords4 = torch.cat([pt_batch.unsqueeze(1), pt_cells], dim=1)
        table = PackedHashTable.from_coords(coords4, device=device)

        # The table hands every point the index of whichever point won the insert
        # CAS for its cell -- an arbitrary but consistent label in [0, N). Which
        # point wins is not deterministic, but the labels are only bucket names:
        # the output is the nsample lowest point indices, which does not depend on
        # them. Hence bit-identical results across runs despite the racing insert.
        cell_ids = table.search(coords4)
        sorted_cell_ids, sorted_order = torch.sort(cell_ids, stable=True)
        num_cells = table.num_entries  # == N; labels are point indices, not ranks

        # A stable sort leaves each cell's members in ascending point index, which
        # is what lets the kernel merge 27 already-sorted lists.
        unique_ids, counts = torch.unique_consecutive(sorted_cell_ids, return_counts=True)
        cell_starts = torch.zeros(num_cells, dtype=torch.int32, device=device)
        cell_counts = torch.zeros(num_cells, dtype=torch.int32, device=device)
        seg_starts = torch.zeros(len(counts), dtype=torch.int32, device=device)
        torch.cumsum(counts[:-1].int(), dim=0, out=seg_starts[1:])
        cell_starts[unique_ids.long()] = seg_starts
        cell_counts[unique_ids.long()] = counts.int()

        _C.coords.cell_gather(
            points,
            queries,
            q_batch,
            ref_offsets,
            sorted_order.int().contiguous(),
            cell_starts,
            cell_counts,
            table.keys_tensor,
            table.values_tensor,
            out,
            num_cells,
            nsample,
            cell_size,
            radius_sq if radius_sq is not None else 0.0,
            radius_sq is not None,
            table.capacity,
        )
        return out


def voxel_block_gather(
    points: Float[Tensor, "N 3"],  # noqa: F821
    ref_offsets: Int[Tensor, "B+1"],  # noqa: F821
    queries: Float[Tensor, "M 3"],  # noqa: F821
    query_offsets: Int[Tensor, "B+1"],  # noqa: F821
    cell_size: float,
    nsample: int,
) -> Int[Tensor, "M nsample"]:  # noqa: F821
    """Distance-free neighbourhood gather: the query's 3x3x3 cell block, capped.

    Returns the ``nsample`` lowest point indices of the 3x3x3 block of grid cells
    around each query, ascending. Short groups repeat their first entry; empty
    groups fill with the batch-local index 0. Indices are global (into
    ``points``), so subtract ``ref_offsets[b]`` for batch-local ones.

    ``cell_size`` is a first-class hyperparameter, not a quantity to derive from a
    radius. It depends on corpus normalization and belongs in the run record; a
    checkpoint trained under one cell size must be evaluated with the same value.
    """
    return _cell_gather(points, ref_offsets, queries, query_offsets, cell_size, nsample)


def capped_ball_query(
    points: Float[Tensor, "N 3"],  # noqa: F821
    ref_offsets: Int[Tensor, "B+1"],  # noqa: F821
    queries: Float[Tensor, "M 3"],  # noqa: F821
    query_offsets: Int[Tensor, "B+1"],  # noqa: F821
    radius: float,
    nsample: int,
) -> Int[Tensor, "M nsample"]:  # noqa: F821
    """Exact ball query, capped at ``nsample`` hits: the reference selection.

    Returns the ``nsample`` lowest-index points with ``d^2 <= radius^2`` per query,
    ascending -- the PointNet++ rule -- via a capped 27-way merge over a cell list
    at cell size ``radius``, so it never fetches more neighbours than it keeps.
    Short groups repeat their first entry; empty groups fill with the batch-local
    index 0.

    ``d^2`` is computed directly from coordinate differences. The dense torch path
    uses the ``-2ab + |a|^2 + |b|^2`` expansion instead. Floating-point
    cancellation in that expansion can cause an ulp-level membership difference
    at the radius boundary; the focused tests compare this direct predicate with
    a float64 reference.
    """
    if isinstance(radius, bool) or not isinstance(radius, Real):
        raise TypeError(f"radius must be a real number, got {type(radius).__name__}")
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError(f"radius must be finite and positive, got {radius}")
    radius_sq = radius * radius
    if not math.isfinite(radius_sq):
        raise ValueError(f"radius squared must be finite, got {radius_sq}")
    return _cell_gather(
        points,
        ref_offsets,
        queries,
        query_offsets,
        radius,
        nsample,
        radius_sq=radius_sq,
    )
