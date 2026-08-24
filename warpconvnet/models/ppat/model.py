# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenShape PointBERT -- the Point Patch Transformer (PPAT) shape encoder.

OpenShape: Scaling Up 3D Shape Representation Towards Open-World Understanding,
Liu et al., NeurIPS 2023 -- https://arxiv.org/abs/2305.10764

What OpenShape names ``PointBERT`` in its configs is a Point Patch Transformer: a
single PointNet++ set-abstraction layer turns ``N`` input points into ``patches``
patch tokens, a pre-norm ViT-style transformer mixes those tokens together with a
learned class token, and the class token is linearly projected into the CLIP
embedding space that OpenShape aligns against.

Module and parameter names mirror the reference implementation
(https://github.com/Colin97/OpenShape_code, ``src/models/ppat.py``) so the
released checkpoints load with no key remapping beyond stripping the training
wrapper prefix -- see ``load_openshape_pointbert``.

Inputs follow OpenShape's convention: channels-first ``(B, C, N)`` tensors whose
first three feature channels are the XYZ coordinates and whose remaining channels
are per-point RGB in ``[0, 1]``. Align the input's gravity axis with the selected
checkpoint before calling ``normalize_point_cloud``: ``vitg14-rgb`` uses z-up,
while ``vitl14-rgb`` and ``vitb32-rgb`` use y-up. Normalization then centers the
shape and scales it to the unit ball.
"""

import math
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from warpconvnet.geometry.types.points import Points
from warpconvnet.ops.sampling import farthest_point_sampling

__all__ = [
    "Attention",
    "FeedForward",
    "OPENSHAPE_VARIANTS",
    "PointNetSetAbstraction",
    "PointPatchTransformer",
    "PreNorm",
    "ProjectedPointPatchTransformer",
    "Transformer",
    "build_openshape_pointbert",
    "farthest_point_sample",
    "index_points",
    "load_openshape_pointbert",
    "normalize_point_cloud",
    "parse_neighborhood_rule",
    "query_ball_point",
]

FPSBackend = Literal["auto", "warp", "torch"]


# ---------------------------------------------------------------------------
# Point sampling / grouping primitives (PointNet++ semantics)
# ---------------------------------------------------------------------------


def normalize_point_cloud(
    xyz: Float[Tensor, "... N 3"], eps: float = 1e-6
) -> Float[Tensor, "... N 3"]:
    """Center a point cloud and scale it into the unit ball.

    This is the preprocessing OpenShape applies before feeding a shape to the
    encoder. It preserves the coordinate frame: rotate XYZ to the variant's
    ``OPENSHAPE_VARIANTS[variant]["up_axis"]`` before calling it. Degenerate
    clouds (every point at the same location) collapse to zeros rather than
    blowing up.

    Args:
        xyz: point coordinates, batched or not, with points along dim ``-2``.
        eps: radius below which the cloud is treated as degenerate.

    Returns:
        The normalized coordinates, same shape as ``xyz``.
    """
    xyz = xyz - xyz.mean(dim=-2, keepdim=True)
    scale = xyz.norm(dim=-1).amax(dim=-1, keepdim=True).unsqueeze(-1)
    return torch.where(scale < eps, torch.zeros_like(xyz), xyz / scale.clamp_min(eps))


def index_points(
    points: Float[Tensor, "B N C"], idx: Int[Tensor, "B ..."]  # noqa: F821
) -> Tensor:
    """Gather ``points`` along the point dimension with per-batch indices.

    Args:
        points: ``(B, N, C)`` point data.
        idx: ``(B, S)`` or ``(B, S, K)`` indices into ``N``.

    Returns:
        ``(B, S, C)`` or ``(B, S, K, C)`` gathered data.
    """
    batch_shape = [points.shape[0]] + [1] * (idx.ndim - 1)
    batch_idx = torch.arange(points.shape[0], device=points.device).view(batch_shape)
    return points[batch_idx.expand_as(idx), idx]


def _farthest_point_sample_torch(
    xyz: Float[Tensor, "B N 3"], npoint: int, start_idx: int = 0
) -> Int[Tensor, "B npoint"]:  # noqa: F821
    """Pure-PyTorch greedy farthest point sampling (device agnostic)."""
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), torch.finfo(xyz.dtype).max, dtype=xyz.dtype, device=device)
    farthest = torch.full((B,), start_idx, dtype=torch.long, device=device)
    batch_idx = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest].unsqueeze(1)
        distance = torch.minimum(distance, (xyz - centroid).pow(2).sum(-1))
        farthest = distance.argmax(-1)
    return centroids


def _farthest_point_sample_warp(
    xyz: Float[Tensor, "B N 3"], npoint: int
) -> Int[Tensor, "B npoint"]:  # noqa: F821
    """Farthest point sampling through the warpconvnet CUDA kernel."""
    B, N, _ = xyz.shape
    # The CUDA kernel reads ``float*`` unconditionally. Convert here instead of
    # letting fp16/bf16 storage be reinterpreted as float32.
    packed = xyz.reshape(B * N, 3).float().contiguous()
    offsets = torch.arange(0, B * N + 1, N, device=xyz.device, dtype=torch.int32)
    idx = farthest_point_sampling(packed, offsets, npoint).long().view(B, npoint)
    # The kernel returns indices into the packed buffer; make them batch-local.
    return idx - torch.arange(B, device=xyz.device).view(B, 1) * N


def farthest_point_sample(
    xyz: Float[Tensor, "B N 3"],
    npoint: int,
    backend: FPSBackend = "auto",
    start_idx: int = 0,
) -> Int[Tensor, "B npoint"]:  # noqa: F821
    """Sample ``npoint`` farthest points per batch item.

    Args:
        xyz: ``(B, N, 3)`` coordinates.
        npoint: number of centroids to select.
        backend: ``"warp"`` uses the compiled warpconvnet kernel, ``"torch"`` the
            portable greedy loop, ``"auto"`` picks the kernel on CUDA.
        start_idx: seed point for the ``"torch"`` backend. The reference
            implementation seeds randomly (via DGL); a fixed seed keeps
            inference reproducible.

    Returns:
        ``(B, npoint)`` indices into the point dimension.
    """
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must have shape (B, N, 3); got {tuple(xyz.shape)}")
    if not torch.is_floating_point(xyz):
        raise TypeError(f"xyz must be floating point; got {xyz.dtype}")
    B, N, _ = xyz.shape
    if B == 0 or N == 0:
        raise ValueError(
            f"xyz must contain at least one batch item and one point; got {tuple(xyz.shape)}"
        )
    if isinstance(npoint, bool) or not isinstance(npoint, int) or npoint <= 0:
        raise ValueError(f"npoint must be a positive integer; got {npoint!r}")
    if npoint > N:
        raise ValueError(f"npoint ({npoint}) cannot exceed the number of input points ({N})")
    if backend not in ("auto", "warp", "torch"):
        raise ValueError(f"Unknown FPS backend {backend!r}; use 'auto', 'warp', or 'torch'")
    if backend == "auto":
        backend = "warp" if xyz.is_cuda else "torch"
    if backend == "warp" and not xyz.is_cuda:
        raise ValueError("The 'warp' FPS backend requires xyz on a CUDA device")
    if backend == "torch" and (
        isinstance(start_idx, bool) or not isinstance(start_idx, int) or not 0 <= start_idx < N
    ):
        raise ValueError(f"start_idx must be an integer in [0, {N}); got {start_idx!r}")

    if backend == "warp":
        return _farthest_point_sample_warp(xyz, npoint)
    if backend == "torch":
        return _farthest_point_sample_torch(xyz, npoint, start_idx=start_idx)
    raise AssertionError(f"unreachable FPS backend {backend!r}")


def _square_distance(
    src: Float[Tensor, "B N C"], dst: Float[Tensor, "B M C"]
) -> Float[Tensor, "B N M"]:
    """Pairwise squared distances via the PointNet++ expansion ``|a|^2 - 2ab + |b|^2``.

    Kept in the expansion form (rather than ``torch.cdist``) so ball-query
    membership matches the reference implementation bit for bit.
    """
    return (
        -2 * torch.matmul(src, dst.transpose(-1, -2))
        + src.pow(2).sum(-1).unsqueeze(-1)
        + dst.pow(2).sum(-1).unsqueeze(-2)
    )


def _query_ball_point_dense(
    radius: float,
    nsample: int,
    xyz: Float[Tensor, "B N 3"],
    new_xyz: Float[Tensor, "B S 3"],
) -> Int[Tensor, "B S nsample"]:  # noqa: F821
    inside = _square_distance(new_xyz, xyz) <= radius**2  # (B, S, N)
    # A stable sort on the "outside the ball" flag lists in-radius points in
    # ascending index order, which is exactly what the reference obtains by
    # sorting `[i if inside else N]`.
    order = torch.argsort((~inside).to(torch.uint8), dim=-1, stable=True)
    group_idx = order[..., :nsample]
    counts = inside.sum(-1, keepdim=True)
    rank = torch.arange(nsample, device=xyz.device).view(1, 1, -1)
    # Reference pads short groups by repeating the first in-radius neighbor.
    return torch.where(rank < counts, group_idx, group_idx[..., :1].expand_as(group_idx))


#: Rules that select the ``nsample`` lowest-index points inside an L2 ball. ``cumsum`` and
#: ``dense`` reproduce the reference distance expansion exactly. ``warp`` and
#: ``ball_cuda`` compute distances directly, so membership can differ by an ulp at the
#: radius boundary even though they implement the same geometric rule.
EXACT_NEIGHBORHOOD_RULES = ("cumsum", "dense", "warp", "ball_cuda")

#: Rules that deliberately select a different neighborhood from the published model.
#: Their geometry or sampling policy is part of a checkpoint's effective configuration.
APPROX_NEIGHBORHOOD_RULES = ("random", "linf", "l1", "voxel", "voxel_cuda", "knn")


def parse_neighborhood_rule(spec: str) -> Dict[str, Union[str, float, bool, None]]:
    """Normalise a neighbourhood-rule spec into the fields a run record has to carry.

    The spec is ``name`` or ``name:parameter`` -- ``"cumsum"``, ``"voxel:0.1"``,
    ``"l1:1.732"``. The returned dict is what gets logged, and it always names the
    parameter explicitly rather than leaving it to be recomputed from ``radius``. Voxel
    cell size changes the patch's spatial scale, so a run record that omits it does not
    identify the model configuration that produced the result.

    ``"auto"`` resolves to ``"cumsum"``, the exact default.

    Raises:
        ValueError: on an unknown rule, on ``voxel`` with no cell size, or on a
            parameter given to a rule that takes none.
    """
    name, sep, arg = spec.partition(":")
    if name == "auto":
        name = "cumsum"
    known = set(EXACT_NEIGHBORHOOD_RULES) | set(APPROX_NEIGHBORHOOD_RULES)
    if name not in known:
        raise ValueError(f"Unknown neighbourhood rule {spec!r}; choose from {sorted(known)}")

    if name in ("voxel", "voxel_cuda"):
        if not arg:
            raise ValueError(
                f"The {name!r} rule needs an explicit cell size, e.g. '{name}:0.1'. It is a "
                "first-class hyperparameter, not a function of the ball radius. Deriving it "
                "silently would hide a checkpoint-relevant change in patch scale."
            )
        cell_size = float(arg)
        if not math.isfinite(cell_size) or cell_size <= 0:
            raise ValueError(f"The {name!r} cell size must be finite and positive; got {arg!r}")
        return {"rule": name, "exact": False, "cell_size": cell_size, "radius_scale": None}

    if name in ("linf", "l1"):
        # These shapes have different volumes from the sphere at equal radius, so the scale
        # is how a comparison is held at a matched receptive field rather than confounding
        # "different shape" with "different size".
        radius_scale = float(arg or 1.0)
        if not math.isfinite(radius_scale) or radius_scale <= 0:
            raise ValueError(f"The {name!r} radius scale must be finite and positive; got {arg!r}")
        return {
            "rule": name,
            "exact": False,
            "cell_size": None,
            "radius_scale": radius_scale,
        }

    if sep:
        raise ValueError(f"The {name!r} rule takes no parameter; got {spec!r}")
    return {
        "rule": name,
        "exact": name in EXACT_NEIGHBORHOOD_RULES,
        "cell_size": None,
        "radius_scale": None,
    }


def _select_first_k(mask: Tensor, nsample: int) -> Int[Tensor, "B S nsample"]:  # noqa: F821
    """Given a ``(B, S, N)`` membership mask, take the ``nsample`` lowest indices per query.

    Shared by every mask-based ball-query variant. Uses the cumsum rank rather than a sort
    (see ``_query_ball_point_cumsum``); short groups repeat the rank-0 entry and empty
    groups yield 0, matching the reference.
    """
    B, S, _ = mask.shape
    rank = torch.cumsum(mask, dim=-1, dtype=torch.int32) - 1
    keep = mask & (rank < nsample)
    first = torch.zeros(B, S, dtype=torch.long, device=mask.device)
    fb, fs, fn = torch.nonzero(mask & (rank == 0), as_tuple=True)
    first[fb, fs] = fn
    out = first.unsqueeze(-1).expand(B, S, nsample).clone()
    kb, ks, kn = torch.nonzero(keep, as_tuple=True)
    out[kb, ks, rank[kb, ks, kn].long()] = kn
    return out


def _ball_mask(
    metric: str, radius: float, xyz: Tensor, new_xyz: Tensor, voxel: Optional[float] = None
) -> Tensor:
    """``(B, S, N)`` membership mask under one of several neighbourhood shapes.

    ``l2`` is the exact sphere the reference uses. The others are cheaper shapes that a
    kernel could evaluate without a sphere test:

    * ``linf`` -- axis-aligned CUBE of half-width ``radius``. Circumscribes the sphere, so
      it admits points out to ``sqrt(3)*radius`` at the corners.
    * ``l1`` -- OCTAHEDRON. Inscribed in the sphere, so it is strictly smaller.
    * ``voxel`` -- the 3x3x3 block of grid cells around the query's own cell, i.e. what a
      hash-table lookup returns with no distance test at all. Unlike the others this is
      GRID-anchored, not query-centred: the neighbourhood shifts depending on where the
      query sits inside its cell, which is the price of removing the test entirely.

    Note the shapes have different volumes at equal ``radius``, so a raw comparison
    confounds "different selection rule" with "different receptive field". Calibrate by
    scaling ``radius`` per metric before reading anything into a score difference.
    """
    if metric == "voxel":
        c = voxel if voxel is not None else radius
        gq = torch.floor(new_xyz / c)  # (B, S, 3)
        gp = torch.floor(xyz / c)  # (B, N, 3)
        return (gq.unsqueeze(2) - gp.unsqueeze(1)).abs().max(-1).values <= 1.0
    d = new_xyz.unsqueeze(2) - xyz.unsqueeze(1)  # (B, S, N, 3)
    if metric == "linf":
        return d.abs().amax(-1) <= radius
    if metric == "l1":
        return d.abs().sum(-1) <= radius
    raise ValueError(f"unknown metric {metric!r}")


def _query_ball_point_knn(
    nsample: int, xyz: Tensor, new_xyz: Tensor
) -> Int[Tensor, "B S nsample"]:  # noqa: F821
    """Select the ``nsample`` nearest support points, ignoring the radius entirely.

    This is not equivalent to the published ball rule. It generally produces tighter
    patches with a smaller spatial extent, so it is an explicit experimental backend
    rather than a drop-in replacement for checkpoint inference.
    """
    dist = _square_distance(new_xyz, xyz)
    return dist.topk(nsample, dim=-1, largest=False).indices


def _query_ball_point_random(
    radius: float, nsample: int, xyz: Tensor, new_xyz: Tensor, seed: int = 0
) -> Int[Tensor, "B S nsample"]:  # noqa: F821
    """A RANDOM ``nsample`` of the in-radius points.

    The control that decides how much precision the ball query needs at all: if scores hold
    up under a random in-radius subset, then *which* neighbours are chosen carries no
    information and the kernel can be as crude as we like.
    """
    inside = _square_distance(new_xyz, xyz) <= radius**2
    g = torch.Generator(device=xyz.device).manual_seed(seed)
    key = torch.rand(inside.shape, device=xyz.device, generator=g)
    key = key.masked_fill(~inside, 2.0)  # in-radius sorts first
    idx = key.argsort(dim=-1)[..., :nsample]
    counts = inside.sum(-1, keepdim=True).clamp(min=1)
    rank = torch.arange(nsample, device=xyz.device).view(1, 1, -1)
    return torch.where(rank < counts, idx, idx[..., :1].expand_as(idx))


def _query_ball_point_cumsum(
    radius: float,
    nsample: int,
    xyz: Float[Tensor, "B N 3"],
    new_xyz: Float[Tensor, "B S 3"],
) -> Int[Tensor, "B S nsample"]:  # noqa: F821
    """Ball query without the sort. Bit-identical to ``_query_ball_point_dense``.

    The dense path argsorts the whole ``N`` axis per query purely to list in-radius points
    in ascending index order. But "rank among the in-radius entries, in index order" is a
    **cumulative sum over the mask** -- O(N) instead of a full O(N log N) stable sort, and
    it never materialises the ``(B, S, N)`` int64 index tensor that the argsort returns.

    The selection is identical by construction: ``cumsum(inside) - 1`` gives each in-radius
    point its position in ascending index order, which is exactly what the stable argsort
    on ``~inside`` produces. Short groups pad by repeating the rank-0 entry, and empty
    groups yield 0, matching the reference in both cases.
    """
    inside = _square_distance(new_xyz, xyz) <= radius**2  # (B, S, N)
    rank = torch.cumsum(inside, dim=-1, dtype=torch.int32) - 1
    keep = inside & (rank < nsample)

    # The rank-0 entry is the reference's padding value; 0 where the ball is empty.
    B, S = keep.shape[0], keep.shape[1]
    first = torch.zeros(B, S, dtype=torch.long, device=xyz.device)
    fb, fs, fn = torch.nonzero(inside & (rank == 0), as_tuple=True)
    first[fb, fs] = fn

    out = first.unsqueeze(-1).expand(B, S, nsample).clone()
    kb, ks, kn = torch.nonzero(keep, as_tuple=True)
    out[kb, ks, rank[kb, ks, kn].long()] = kn
    return out


def _query_ball_point_warp(
    radius: float,
    nsample: int,
    xyz: Float[Tensor, "B N 3"],
    new_xyz: Float[Tensor, "B S 3"],
) -> Int[Tensor, "B S nsample"]:  # noqa: F821
    """Ball query via warpconvnet's cell-list radius search. **Lower memory, NOT faster.**

    This uses the same lowest-index selection and padding rule as
    ``_query_ball_point_dense`` without building a ``(B, S, N)`` distance matrix. The
    radius search computes distances directly, so an ulp-level boundary case may differ
    from the expanded distance formula used by the reference-compatible dense path.
    """
    from warpconvnet.geometry.coords.search.radius import batched_radius_search

    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    dev = xyz.device

    points = xyz.reshape(B * N, 3).contiguous()
    queries = new_xyz.reshape(B * S, 3).contiguous()
    ref_off = torch.arange(0, B * N + 1, N, device=dev, dtype=torch.int32)
    qry_off = torch.arange(0, B * S + 1, S, device=dev, dtype=torch.int32)

    nbr_idx, _, nbr_split = batched_radius_search(points, ref_off, queries, qry_off, radius)
    nbr_split = nbr_split.long()
    counts = nbr_split[1:] - nbr_split[:-1]  # (B*S,)
    total = int(nbr_split[-1].item())

    if total == 0:  # degenerate: radius smaller than the closest point everywhere
        return torch.zeros(B, S, nsample, dtype=torch.long, device=dev)

    # Which query each neighbour belongs to, and the batch-local point index.
    qid = torch.repeat_interleave(torch.arange(B * S, device=dev), counts)
    local = nbr_idx.long() - (qid // S) * N

    # Sorting by (query, local index) reproduces the dense path's stable argsort ordering:
    # in-radius points listed in ascending index order. The cell-list returns them in cell
    # order, so this is required for equivalence, not tidiness.
    order = torch.argsort(qid * N + local)
    qid, local = qid[order], local[order]

    rank = torch.arange(total, device=dev) - nbr_split[:-1].repeat_interleave(counts)
    keep = rank < nsample

    # First in-radius neighbour per query -- the reference's padding value.
    first = torch.zeros(B * S, dtype=torch.long, device=dev)
    has = counts > 0
    first[has] = local[rank == 0]

    out = first.unsqueeze(1).expand(B * S, nsample).clone()
    out[qid[keep], rank[keep]] = local[keep]
    return out.view(B, S, nsample)


def _query_ball_point_cell(
    radius: float,
    nsample: int,
    xyz: Float[Tensor, "B N 3"],
    new_xyz: Float[Tensor, "B S 3"],
    cell_size: Optional[float] = None,
) -> Int[Tensor, "B S nsample"]:  # noqa: F821
    """Ball query via the capped cell-gather CUDA kernels. **The fast path.**

    ``cell_size=None`` runs the exact ``d^2 <= radius^2`` rule; a cell size runs the
    distance-free 3x3x3 block gather. Both return the ``nsample`` lowest point indices
    ascending, so the distance-free one is bit-identical to ``voxel:cell`` rather than
    merely selecting points from the same cells. Tests pin that ordering because filling
    a capped group in cell order would bias it toward one part of the block.

    Unlike ``warp`` this path stops its 27-way merge after ``nsample`` accepted hits.
    Recorded development timings and their scope are documented in the PPAT README.
    """
    from warpconvnet.ops.cell_gather import capped_ball_query, voxel_block_gather

    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    dev = xyz.device
    ref_offsets = torch.arange(B + 1, dtype=torch.int32, device=dev) * N
    query_offsets = torch.arange(B + 1, dtype=torch.int32, device=dev) * S
    points = xyz.reshape(B * N, 3).contiguous()
    queries = new_xyz.reshape(B * S, 3).contiguous()
    if cell_size is None:
        idx = capped_ball_query(points, ref_offsets, queries, query_offsets, radius, nsample)
    else:
        idx = voxel_block_gather(points, ref_offsets, queries, query_offsets, cell_size, nsample)
    # The kernels index `points` globally; the rest of this module is batch-local.
    return (idx.view(B, S, nsample).long() - ref_offsets[:-1].long().view(B, 1, 1)).contiguous()


def query_ball_point(
    radius: float,
    nsample: int,
    xyz: Float[Tensor, "B N 3"],
    new_xyz: Float[Tensor, "B S 3"],
    chunk_size: Optional[int] = None,
    backend: str = "auto",
) -> Int[Tensor, "B S nsample"]:  # noqa: F821
    """Group up to ``nsample`` points within ``radius`` of every query point.

    Args:
        radius: ball radius.
        nsample: maximum neighbors per ball; short groups repeat their first
            neighbor, over-full groups keep the ``nsample`` lowest point indices.
        xyz: ``(B, N, 3)`` support points.
        new_xyz: ``(B, S, 3)`` query points.
        chunk_size: split the query dimension into chunks of this size to bound
            the ``(B, S, N)`` distance matrix. ``None`` computes it in one shot.
            Ignored by the ``"warp"`` backend, which never builds that matrix.
        backend: which implementation to use. ``"auto"`` selects ``"cumsum"``, the
            portable reference-compatible default. The other exact rules use the same
            geometry but direct-distance CUDA paths can differ at an ulp-level radius
            boundary. Approximate rules select a different neighbourhood and are parsed
            by ``parse_neighborhood_rule``. Set ``PointPatchTransformer.neighborhood``
            when configuring a model so the rule appears in its configuration report.

    Returns:
        ``(B, S, nsample)`` indices into the support points.
    """
    if isinstance(radius, bool) or not isinstance(radius, (int, float)):
        raise ValueError(f"radius must be a finite positive number; got {radius!r}")
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError(f"radius must be a finite positive number; got {radius!r}")
    if isinstance(nsample, bool) or not isinstance(nsample, int) or nsample <= 0:
        raise ValueError(f"nsample must be a positive integer; got {nsample!r}")
    if chunk_size is not None and (
        isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0
    ):
        raise ValueError(f"chunk_size must be a positive integer or None; got {chunk_size!r}")
    if not isinstance(xyz, Tensor) or not isinstance(new_xyz, Tensor):
        raise TypeError("xyz and new_xyz must be torch tensors")
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must have shape (B, N, 3); got {tuple(xyz.shape)}")
    if new_xyz.ndim != 3 or new_xyz.shape[-1] != 3:
        raise ValueError(f"new_xyz must have shape (B, S, 3); got {tuple(new_xyz.shape)}")
    if not torch.is_floating_point(xyz) or not torch.is_floating_point(new_xyz):
        raise TypeError(
            f"xyz and new_xyz must be floating point; got {xyz.dtype} and {new_xyz.dtype}"
        )
    B, N, _ = xyz.shape
    if B == 0 or N == 0:
        raise ValueError(
            f"xyz must contain at least one batch item and one point; got {tuple(xyz.shape)}"
        )
    if new_xyz.shape[1] == 0:
        raise ValueError("new_xyz must contain at least one query point per batch item")
    if new_xyz.shape[0] != B:
        raise ValueError(f"xyz and new_xyz batch sizes must match; got {B} and {new_xyz.shape[0]}")
    if xyz.device != new_xyz.device:
        raise ValueError(
            f"xyz and new_xyz must share a device; got {xyz.device} and {new_xyz.device}"
        )
    if xyz.dtype != new_xyz.dtype:
        raise ValueError(
            f"xyz and new_xyz must share a dtype; got {xyz.dtype} and {new_xyz.dtype}"
        )
    if nsample > N:
        raise ValueError(f"nsample ({nsample}) cannot exceed the number of support points ({N})")

    cfg = parse_neighborhood_rule(backend)
    backend = cfg["rule"]
    # Approximate variants, for the published-weights compatibility sweep. `voxel:C` sets the
    # cell size, `linf:S` / `l1:S` scale the radius so a shape can be volume-calibrated
    # against the sphere rather than compared at an unequal receptive field.
    if backend == "voxel":
        return _select_first_k(
            _ball_mask("voxel", radius, xyz, new_xyz, voxel=cfg["cell_size"]), nsample
        )
    if backend == "voxel_cuda":
        return _query_ball_point_cell(radius, nsample, xyz, new_xyz, cell_size=cfg["cell_size"])
    if backend == "ball_cuda":
        return _query_ball_point_cell(radius, nsample, xyz, new_xyz)
    if backend in ("linf", "l1"):
        return _select_first_k(
            _ball_mask(backend, radius * cfg["radius_scale"], xyz, new_xyz), nsample
        )
    if backend == "knn":
        return _query_ball_point_knn(nsample, xyz, new_xyz)
    if backend == "random":
        return _query_ball_point_random(radius, nsample, xyz, new_xyz)
    if backend == "cumsum":
        S = new_xyz.shape[1]
        if chunk_size is None or chunk_size >= S:
            return _query_ball_point_cumsum(radius, nsample, xyz, new_xyz)
        return torch.cat(
            [
                _query_ball_point_cumsum(radius, nsample, xyz, new_xyz[:, s : s + chunk_size])
                for s in range(0, S, chunk_size)
            ],
            dim=1,
        )
    if backend == "warp":
        return _query_ball_point_warp(radius, nsample, xyz, new_xyz)
    S = new_xyz.shape[1]  # "dense"; `parse_neighborhood_rule` has rejected anything else
    if chunk_size is None or chunk_size >= S:
        return _query_ball_point_dense(radius, nsample, xyz, new_xyz)
    return torch.cat(
        [
            _query_ball_point_dense(radius, nsample, xyz, new_xyz[:, s : s + chunk_size])
            for s in range(0, S, chunk_size)
        ],
        dim=1,
    )


def _supercat(tensors: List[Tensor], dim: int) -> Tensor:
    """Broadcasting ``torch.cat`` -- the ``torch_redstone.supercat`` the reference uses.

    Lower-rank operands are right-aligned, then every dimension except ``dim`` is
    broadcast to the common size before concatenating.
    """
    ndim = max(t.ndim for t in tensors)
    tensors = [t.reshape(*([1] * (ndim - t.ndim)), *t.shape) for t in tensors]
    shape = [max(t.size(i) for t in tensors) for i in range(ndim)]
    shape[dim] = -1
    return torch.cat([t.expand(shape) for t in tensors], dim=dim)


class _Permute(nn.Module):
    """Stateless permutation.

    Stands in for the ``torch_redstone.Lambda`` the reference puts at index 1 of
    ``PointPatchTransformer.lift``; keeping a module here is what makes the
    trailing ``LayerNorm`` load from the checkpoint's ``lift.2.*`` keys.
    """

    def __init__(self, *dims: int) -> None:
        super().__init__()
        self.dims = tuple(dims)

    def forward(self, x: Tensor) -> Tensor:
        return x.permute(*self.dims)

    def extra_repr(self) -> str:
        return f"dims={self.dims}"


class PointNetSetAbstraction(nn.Module):
    """Single-scale PointNet++ set abstraction: FPS centroids, ball query, shared MLP, max pool.

    Args:
        npoint: default number of centroids.
        radius: ball-query radius.
        nsample: maximum neighbors per ball.
        in_channel: input feature channels, including the 3 relative-XYZ channels.
        mlp: output channels of the shared point-wise MLP.
        fps_backend: see ``farthest_point_sample``.
        ball_query_chunk: see ``query_ball_point``.
        neighborhood: neighbourhood-rule spec, parsed by ``parse_neighborhood_rule``.
            Defaults to the exact ball. Validated here rather than at the first forward,
            so a typo'd rule fails when the model is built.
    """

    def __init__(
        self,
        npoint: int,
        radius: float,
        nsample: int,
        in_channel: int,
        mlp: List[int],
        fps_backend: FPSBackend = "auto",
        ball_query_chunk: Optional[int] = None,
        neighborhood: str = "cumsum",
    ) -> None:
        super().__init__()
        if isinstance(npoint, bool) or not isinstance(npoint, int) or npoint <= 0:
            raise ValueError(f"npoint must be a positive integer; got {npoint!r}")
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError(f"radius must be finite and positive; got {radius!r}")
        if isinstance(nsample, bool) or not isinstance(nsample, int) or nsample <= 0:
            raise ValueError(f"nsample must be a positive integer; got {nsample!r}")
        if ball_query_chunk is not None and (
            isinstance(ball_query_chunk, bool)
            or not isinstance(ball_query_chunk, int)
            or ball_query_chunk <= 0
        ):
            raise ValueError(
                "ball_query_chunk must be a positive integer or None; " f"got {ball_query_chunk!r}"
            )
        if fps_backend not in ("auto", "warp", "torch"):
            raise ValueError(
                f"Unknown fps_backend {fps_backend!r}; use 'auto', 'warp', or 'torch'"
            )
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.fps_backend = fps_backend
        self.ball_query_chunk = ball_query_chunk
        self.neighborhood = neighborhood  # validates and derives, via the property below
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    @property
    def neighborhood(self) -> str:
        """The neighbourhood-rule spec the ball query runs under.

        A property rather than a plain attribute so the parsed form cannot go stale: it is
        re-derived on every assignment, and assigning an invalid rule raises here instead
        of at the next forward. Mutating this post-construction is legitimate (the
        published-weights sweep does exactly that), but it must not be possible to mutate
        it into a state where ``neighborhood_config`` -- and so the run record built from
        it -- describes a rule other than the one being executed.
        """
        return self._neighborhood

    @neighborhood.setter
    def neighborhood(self, spec: str) -> None:
        self._neighborhood_config = parse_neighborhood_rule(spec)
        self._neighborhood = spec

    @property
    def neighborhood_config(self) -> Dict[str, Union[str, float, bool, None]]:
        """The parsed rule, always in sync with ``neighborhood`` by construction."""
        return dict(self._neighborhood_config)

    def forward(
        self,
        xyz: Float[Tensor, "B 3 N"],
        features: Float[Tensor, "B D N"],
        npoint: Optional[int] = None,
    ) -> Tuple[Float[Tensor, "B 3 S"], Float[Tensor, "B D2 S"]]:
        """Abstract ``N`` points into ``npoint`` patch features.

        Args:
            xyz: ``(B, 3, N)`` coordinates.
            features: ``(B, D, N)`` per-point features.
            npoint: overrides ``self.npoint`` for this call.

        Returns:
            ``(B, 3, S)`` centroid coordinates and ``(B, mlp[-1], S)`` patch features.
        """
        npoint = self.npoint if npoint is None else npoint
        xyz = xyz.permute(0, 2, 1)  # (B, N, 3)
        features = features.permute(0, 2, 1)  # (B, N, D)

        fps_idx = farthest_point_sample(xyz, npoint, backend=self.fps_backend)
        new_xyz = index_points(xyz, fps_idx)  # (B, S, 3)
        group_idx = query_ball_point(
            self.radius,
            self.nsample,
            xyz,
            new_xyz,
            chunk_size=self.ball_query_chunk,
            backend=self.neighborhood,
        )
        grouped_xyz = index_points(xyz, group_idx) - new_xyz.unsqueeze(2)
        grouped = torch.cat([grouped_xyz, index_points(features, group_idx)], dim=-1)

        grouped = grouped.permute(0, 3, 2, 1)  # (B, 3 + D, nsample, S)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            grouped = F.relu(bn(conv(grouped)))
        return new_xyz.permute(0, 2, 1), grouped.max(dim=2).values


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------


class PreNorm(nn.Module):
    """LayerNorm applied to the input of ``fn``."""

    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return self.fn(self.norm(x), *args, **kwargs)


class FeedForward(nn.Module):
    """Two-layer GELU MLP."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Attention(nn.Module):
    """Multi-head self attention with an optional relative positional bias.

    When ``rel_pe`` is set, a small conv net maps the pairwise centroid offsets to
    a scalar bias added to the pre-softmax logits. None of the released OpenShape
    checkpoints enable it.

    Args:
        attn_backend: ``"sdpa"`` (default) routes through
            ``torch.nn.functional.scaled_dot_product_attention``, which dispatches to
            the flash / memory-efficient kernels and never materialises the ``(B, heads,
            N, N)`` attention matrix. ``"naive"`` is the reference formulation --
            ``matmul`` / ``softmax`` / ``matmul`` -- kept so the equivalence test can
            compare the two directly. Both are the same function of the same weights;
            neither is a checkpoint property.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        rel_pe: bool = False,
        attn_backend: str = "sdpa",
    ) -> None:
        super().__init__()
        if attn_backend not in ("sdpa", "naive"):
            raise ValueError(f"Unknown attn_backend {attn_backend!r}; use 'sdpa' or 'naive'")
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.attn_backend = attn_backend

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Identity()
            if (heads == 1 and dim_head == dim)
            else nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        )

        self.rel_pe = rel_pe
        if rel_pe:
            self.pe = nn.Sequential(nn.Conv2d(3, 64, 1), nn.ReLU(), nn.Conv2d(64, 1, 1))

    def forward(
        self, x: Float[Tensor, "B N D"], centroid_delta: Float[Tensor, "B 3 N N"]
    ) -> Float[Tensor, "B N D"]:
        B, N, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, N, self.heads, self.dim_head).transpose(1, 2) for t in qkv)

        # (B, 1, N, N) -- one shared bias across heads, broadcast into the logits.
        pe = self.pe(centroid_delta) if self.rel_pe else None

        if self.attn_backend == "sdpa":
            # The reference scales the SUM: `(qk + pe) * scale`. SDPA scales only the
            # logits and then ADDS the mask: `qk * scale + attn_mask`. So the equivalent
            # mask is `pe * scale`, not `pe` -- passing `pe` is a silent accuracy change
            # with no error anywhere. `test_sdpa_matches_naive[True]` pins this with a
            # non-zero `pe` (the same test at `pe = 0` would pass either way), and
            # `test_real_sdpa_branch_passes_the_scaled_mask` asserts on the mask this line
            # actually hands to SDPA rather than on a copy of it.
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None if pe is None else (pe * self.scale).to(q.dtype),
                # The reference applies `self.dropout` to the post-softmax weights, which
                # is exactly SDPA's `dropout_p`. nn.Dropout is a no-op in eval mode, so the
                # training-mode gate has to be reproduced explicitly here.
                dropout_p=self.dropout.p if self.training else 0.0,
                scale=self.scale,
            )
        else:
            dots = (torch.matmul(q, k.transpose(-1, -2)) + (0 if pe is None else pe)) * self.scale
            out = torch.matmul(self.dropout(self.attend(dots)), v)

        out = out.transpose(1, 2).reshape(B, N, self.heads * self.dim_head)
        return self.to_out(out)


class Transformer(nn.Module):
    """Stack of pre-norm attention / feed-forward residual blocks."""

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        rel_pe: bool = False,
        attn_backend: str = "sdpa",
    ) -> None:
        super().__init__()
        if attn_backend not in ("sdpa", "naive"):
            raise ValueError(f"Unknown attn_backend {attn_backend!r}; use 'sdpa' or 'naive'")
        self.layers = nn.ModuleList(
            nn.ModuleList(
                [
                    PreNorm(dim, Attention(dim, heads, dim_head, dropout, rel_pe, attn_backend)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout)),
                ]
            )
            for _ in range(depth)
        )

    def forward(self, x: Tensor, centroid_delta: Tensor) -> Tensor:
        for attn, ff in self.layers:
            x = attn(x, centroid_delta) + x
            x = ff(x) + x
        return x


# ---------------------------------------------------------------------------
# Point Patch Transformer
# ---------------------------------------------------------------------------


def _to_channels_first(
    points: Union[Points, Float[Tensor, "B C N"]],
    features: Optional[Float[Tensor, "B C N"]],
) -> Tuple[Float[Tensor, "B 3 N"], Float[Tensor, "B C N"]]:
    """Normalize the three accepted input forms into ``(xyz, features)``.

    Accepted forms:
      * ``Points`` -- packed warpconvnet geometry with a uniform point count per
        batch item; coordinates become ``xyz`` and its features the input features.
      * ``features`` alone, ``(B, C, N)`` -- ``xyz`` is taken from the first three
        channels, matching the OpenShape inference code.
      * ``xyz, features`` -- both ``(B, 3, N)`` / ``(B, C, N)``, matching the
        OpenShape training code.
    """
    if isinstance(points, Points):
        if features is not None:
            raise ValueError("Pass features inside the Points object, not as a second argument")
        counts = points.offsets.diff()
        if counts.numel() == 0:
            raise ValueError("PointPatchTransformer needs a non-empty Points batch")
        if bool((counts <= 0).any()):
            raise ValueError(
                "PointPatchTransformer needs at least one point per batch item; "
                f"got {counts.tolist()}"
            )
        if not bool((counts == counts[0]).all()):
            raise ValueError(
                "PointPatchTransformer needs a uniform point count per batch item; "
                f"got {counts.tolist()}. Resample the batch to a fixed size first."
            )
        B, N = points.batch_size, int(counts[0])
        coordinates = points.coordinate_tensor
        point_features = points.feature_tensor
        if coordinates.ndim != 2 or coordinates.shape != (B * N, 3):
            raise ValueError(
                f"Points coordinates must have shape ({B * N}, 3); got {tuple(coordinates.shape)}"
            )
        if point_features.ndim == 2 and point_features.shape[0] == B * N:
            features_bnc = point_features.view(B, N, -1)
        elif (
            point_features.ndim == 3
            and point_features.shape[0] == B
            and point_features.shape[1] >= N
        ):
            # Uniform padded features are also a valid Points representation.
            features_bnc = point_features[:, :N, :]
        else:
            raise ValueError(
                "Points features must be packed (B*N, C) or padded (B, M, C) with M >= N; "
                f"got {tuple(point_features.shape)} for B={B}, N={N}"
            )
        if features_bnc.shape[-1] == 0:
            raise ValueError("Points features must contain at least one channel")
        if not torch.is_floating_point(coordinates) or not torch.is_floating_point(point_features):
            raise TypeError(
                "Points coordinates and features must be floating point; "
                f"got {coordinates.dtype} and {point_features.dtype}"
            )
        if coordinates.device != point_features.device:
            raise ValueError(
                "Points coordinates and features must share a device; "
                f"got {coordinates.device} and {point_features.device}"
            )
        # Coordinates determine FPS and neighbourhood membership, so preserve
        # their precision and make mixed-dtype Points features explicit here.
        features_bnc = features_bnc.to(dtype=coordinates.dtype)
        xyz = coordinates.view(B, N, 3).permute(0, 2, 1)
        feats = features_bnc.permute(0, 2, 1)
        return xyz, feats

    if not isinstance(points, Tensor):
        raise TypeError(f"points must be a Tensor or Points object; got {type(points).__name__}")
    if points.ndim != 3:
        raise ValueError(f"points must be a 3-D channels-first tensor; got {tuple(points.shape)}")
    if not torch.is_floating_point(points):
        raise TypeError(f"points must be floating point; got {points.dtype}")
    if points.shape[0] == 0 or points.shape[2] == 0:
        raise ValueError(
            f"points must contain at least one batch item and one point; got {tuple(points.shape)}"
        )
    if features is None:
        if points.shape[1] < 3:
            raise ValueError(
                "features-only input needs at least three XYZ channels; " f"got {points.shape[1]}"
            )
        return points[:, :3], points

    if not isinstance(features, Tensor):
        raise TypeError(f"features must be a Tensor; got {type(features).__name__}")
    if points.shape[1] != 3:
        raise ValueError(
            f"separate xyz input must have shape (B, 3, N); got {tuple(points.shape)}"
        )
    if features.ndim != 3:
        raise ValueError(
            f"features must be a 3-D channels-first tensor; got {tuple(features.shape)}"
        )
    if not torch.is_floating_point(features):
        raise TypeError(f"features must be floating point; got {features.dtype}")
    if features.shape[1] == 0:
        raise ValueError("features must contain at least one channel")
    if points.shape[0] != features.shape[0] or points.shape[2] != features.shape[2]:
        raise ValueError(
            "xyz and features must share batch and point dimensions; "
            f"got {tuple(points.shape)} and {tuple(features.shape)}"
        )
    if points.device != features.device:
        raise ValueError(
            f"xyz and features must share a device; got {points.device} and {features.device}"
        )
    if points.dtype != features.dtype:
        raise ValueError(
            f"xyz and features must share a dtype; got {points.dtype} and {features.dtype}"
        )
    return points, features


class PointPatchTransformer(nn.Module):
    """OpenShape's ``PointBERT`` encoder: patchify with PointNet++, mix with a transformer.

    Args:
        dim: transformer width.
        depth: number of transformer blocks.
        heads: attention heads.
        mlp_dim: feed-forward hidden width.
        sa_dim: set-abstraction output width.
        patches: number of patch tokens (set-abstraction centroids).
        prad: patch (ball-query) radius.
        nsamp: maximum points grouped per patch.
        in_dim: input feature channels -- 6 for the released ``xyz + rgb`` models.
        dim_head: per-head width.
        rel_pe: enable the relative positional bias (off in every released model).
        patch_dropout: number of patches dropped during training.
        fps_backend: see ``farthest_point_sample``.
        ball_query_chunk: see ``query_ball_point``.
        neighborhood: which neighbourhood rule the set abstraction selects patches with,
            e.g. ``"cumsum"`` (default, the exact ball) or ``"voxel:0.1"``. See
            ``parse_neighborhood_rule``, and ``ppat_config_report`` for why it lives here.
        attn_backend: see ``Attention``. ``"sdpa"`` by default; ``"naive"`` is the
            reference formulation, kept only for the equivalence test.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        sa_dim: int,
        patches: int,
        prad: float,
        nsamp: int,
        in_dim: int = 3,
        dim_head: int = 64,
        rel_pe: bool = False,
        patch_dropout: int = 0,
        fps_backend: FPSBackend = "auto",
        ball_query_chunk: Optional[int] = None,
        neighborhood: str = "cumsum",
        attn_backend: str = "sdpa",
    ) -> None:
        super().__init__()
        if isinstance(patches, bool) or not isinstance(patches, int) or patches <= 0:
            raise ValueError(f"patches must be a positive integer; got {patches!r}")
        if (
            isinstance(patch_dropout, bool)
            or not isinstance(patch_dropout, int)
            or not 0 <= patch_dropout < patches
        ):
            raise ValueError(
                f"patch_dropout must be an integer in [0, {patches}); got {patch_dropout!r}"
            )
        if attn_backend not in ("sdpa", "naive"):
            raise ValueError(f"Unknown attn_backend {attn_backend!r}; use 'sdpa' or 'naive'")
        if isinstance(in_dim, bool) or not isinstance(in_dim, int) or in_dim <= 0:
            raise ValueError(f"in_dim must be a positive integer; got {in_dim!r}")
        self.patches = patches
        self.patch_dropout = patch_dropout
        self.in_dim = in_dim
        self.sa = PointNetSetAbstraction(
            npoint=patches,
            radius=prad,
            nsample=nsamp,
            in_channel=in_dim + 3,
            mlp=[64, 64, sa_dim],
            fps_backend=fps_backend,
            ball_query_chunk=ball_query_chunk,
            neighborhood=neighborhood,
        )
        self.lift = nn.Sequential(
            nn.Conv1d(sa_dim + 3, dim, 1),
            _Permute(0, 2, 1),
            nn.LayerNorm([dim]),
        )
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, 0.0, rel_pe, attn_backend
        )

        # Everything in the report that a forward pass cannot change. The neighbourhood
        # fields are deliberately NOT captured here -- see `ppat_config_report`.
        self._config_report_static = {
            "radius": prad,
            "nsample": nsamp,
            "patches": patches,
            "patch_dropout": patch_dropout,
            # Not a checkpoint property -- the two backends are the same function -- but
            # it is what an unexpected speed or a 3rd-decimal drift would be traced to.
            "attn_backend": attn_backend,
            "fps_backend": fps_backend,
        }

    @property
    def ppat_config_report(self) -> Dict:
        """What the run record needs in order to identify the model that produced a number.

        The neighbourhood rule is part of what a checkpoint MEANS, and nothing else in the
        record would say which one ran: the variant name, the parameter count and the
        state-dict keys are byte-identical across all of them. A model trained under
        ``voxel:0.1`` and scored as if it were the exact ball would look like an ordinary
        result. ``cell_size`` is here for the same reason one level down: ``"voxel"`` alone
        does not identify the geometry or the patch scale.

        Computed on each access from ``self.sa`` rather than snapshotted at construction.
        A snapshot is a report that can disagree with the model: the sweep mutates
        ``sa.neighborhood`` between scores, and a cached dict would then keep reporting the
        rule the model was *built* with while a different one executed -- a wrong run record
        that reads as an authoritative one, which is worse than no record at all.
        """
        return {
            "neighborhood": self.sa.neighborhood,
            **{f"neighborhood_{k}": v for k, v in self.sa.neighborhood_config.items()},
            **self._config_report_static,
        }

    def forward(
        self,
        points: Union[Points, Float[Tensor, "B C N"]],
        features: Optional[Float[Tensor, "B C N"]] = None,
    ) -> Float[Tensor, "B D"]:
        """Encode a batch of point clouds into class-token embeddings.

        See ``_to_channels_first`` for the accepted input forms.

        Returns:
            ``(B, dim)`` class-token features.
        """
        xyz, feats = _to_channels_first(points, features)
        if feats.shape[1] != self.in_dim:
            raise ValueError(
                f"Expected {self.in_dim} input feature channels, got {feats.shape[1]}. "
                "Released OpenShape models require all six XYZ+RGB channels even when XYZ "
                "is also passed separately or stored as Points coordinates."
            )
        npoint = self.patches - (self.patch_dropout if self.training else 0)
        centroids, patch_feats = self.sa(xyz, feats, npoint=npoint)

        x = self.lift(torch.cat([centroids, patch_feats], dim=1))  # (B, S, dim)
        x = _supercat([self.cls_token, x], dim=-2)  # (B, S + 1, dim)
        centroids = _supercat([centroids.new_zeros(1), centroids], dim=-1)  # (B, 3, S + 1)

        centroid_delta = centroids.unsqueeze(-1) - centroids.unsqueeze(-2)
        return self.transformer(x, centroid_delta)[:, 0]


class ProjectedPointPatchTransformer(nn.Module):
    """A ``PointPatchTransformer`` plus the linear head into CLIP embedding space."""

    def __init__(self, ppat: PointPatchTransformer, proj: nn.Module) -> None:
        super().__init__()
        self.ppat = ppat
        self.proj = proj

    @property
    def ppat_config_report(self) -> Dict:
        """Forward the encoder's config report to the object the trainer actually holds.

        ``build_openshape_pointbert`` returns this wrapper for every variant with a
        projection head, i.e. for the two the trainer normally runs. A report the trainer
        cannot reach is a report that silently never gets logged.
        """
        return self.ppat.ppat_config_report

    def forward(
        self,
        points: Union[Points, Float[Tensor, "B C N"]],
        features: Optional[Float[Tensor, "B C N"]] = None,
    ) -> Float[Tensor, "B D"]:
        """Encode point clouds and project to the CLIP embedding dimension."""
        return self.proj(self.ppat(points, features))


# ---------------------------------------------------------------------------
# Released OpenShape checkpoints
# ---------------------------------------------------------------------------

#: Architecture and input orientation of every released OpenShape PointBERT model.
#: ``out_dim`` is the CLIP text-encoder width the shape embedding is aligned to
#: (``None`` = no projection head). ``up_axis`` records the gravity convention used
#: to train that checkpoint: B32 and L14 are Y-up, while bigG is Z-up.
#: Normalization and the model forward pass do not rotate inputs. Checkpoint
#: layout is detected at load time by ``_match_state_dict``.
OPENSHAPE_VARIANTS: Dict[str, Dict] = {
    "openshape-pointbert-vitb32-rgb": {
        "ppat": dict(
            dim=512, depth=12, heads=8, mlp_dim=1024, sa_dim=128, patches=64, prad=0.4, nsamp=256
        ),
        "out_dim": None,
        "up_axis": "y",
    },
    "openshape-pointbert-vitl14-rgb": {
        "ppat": dict(
            dim=512, depth=12, heads=8, mlp_dim=1024, sa_dim=128, patches=64, prad=0.4, nsamp=256
        ),
        "out_dim": 768,
        "up_axis": "y",
    },
    "openshape-pointbert-vitg14-rgb": {
        "ppat": dict(
            dim=512, depth=12, heads=8, mlp_dim=1536, sa_dim=256, patches=384, prad=0.2, nsamp=64
        ),
        "out_dim": 1280,
        "up_axis": "z",
    },
}


def build_openshape_pointbert(
    variant: str = "openshape-pointbert-vitg14-rgb",
    in_dim: int = 6,
    **kwargs,
) -> Union[PointPatchTransformer, ProjectedPointPatchTransformer]:
    """Instantiate a released OpenShape PointBERT architecture without weights.

    Args:
        variant: key of ``OPENSHAPE_VARIANTS``. It also selects the input frame:
            Y-up for B32/L14 and Z-up for bigG. Rotate before normalization.
        in_dim: input feature channels; the released models use 6 (``xyz + rgb``).
        **kwargs: forwarded to ``PointPatchTransformer`` (e.g. ``fps_backend``,
            ``ball_query_chunk``).

    Returns:
        A ``ProjectedPointPatchTransformer``, or a bare
        ``PointPatchTransformer`` for variants trained without a projection head.
    """
    if variant not in OPENSHAPE_VARIANTS:
        raise KeyError(f"Unknown variant {variant!r}; choose from {sorted(OPENSHAPE_VARIANTS)}")
    spec = OPENSHAPE_VARIANTS[variant]
    ppat = PointPatchTransformer(**spec["ppat"], in_dim=in_dim, **kwargs)
    if spec["out_dim"] is None:
        return ppat
    return ProjectedPointPatchTransformer(ppat, nn.Linear(spec["ppat"]["dim"], spec["out_dim"]))


def _match_state_dict(checkpoint: Dict, model: nn.Module) -> Dict[str, Tensor]:
    """Find the part of a checkpoint whose keys are exactly the model's parameters.

    The released checkpoints disagree on layout: the ViT-bigG-14 one nests the
    weights under ``state_dict`` behind a DDP ``module.`` prefix, the others are
    flat behind ``pc_encoder.``, and both carry extra entries (``tau``, optimizer
    state) that a strict load would reject. Checkpoints written by
    ``examples/train/openshape_pointbert.py`` are plain and unprefixed. Rather than
    hardcode each case, search the plausible containers and prefixes and accept the
    one whose key set matches the model exactly -- which also validates the choice.
    """
    containers = [checkpoint]
    if isinstance(checkpoint, dict):
        containers += [
            checkpoint[key]
            for key in ("state_dict", "model", "model_state_dict")
            if isinstance(checkpoint.get(key), dict)
        ]

    want = set(model.state_dict())
    for container in containers:
        if not isinstance(container, dict):
            continue
        prefixes = [""] + sorted({k.split(".")[0] for k in container if "." in k})
        for prefix in prefixes:
            stripped = (
                dict(container)
                if not prefix
                else {
                    k[len(prefix) + 1 :]: v
                    for k, v in container.items()
                    if k.startswith(prefix + ".")
                }
            )
            if set(stripped) == want:
                return stripped

    sample = list(checkpoint)[:6] if isinstance(checkpoint, dict) else type(checkpoint).__name__
    raise KeyError(
        f"No sub-dict of the checkpoint matches the model's {len(want)} parameters. "
        f"Checkpoint top-level keys: {sample}"
    )


def load_openshape_pointbert(
    variant: str = "openshape-pointbert-vitg14-rgb",
    checkpoint_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs,
) -> Union[PointPatchTransformer, ProjectedPointPatchTransformer]:
    """Build a released OpenShape PointBERT and load its published weights.

    The checkpoints live on the Hugging Face Hub under the ``OpenShape``
    organization; ``huggingface_hub`` is only imported when a download is needed.

    Args:
        variant: key of ``OPENSHAPE_VARIANTS``; also selects the architecture
            and input frame (Y-up for B32/L14, Z-up for bigG). The loader does
            not rotate coordinates.
        checkpoint_path: local checkpoint; downloaded from the Hub when omitted.
            Both the published layouts and plain ``model.state_dict()`` dumps work.
        device: device to move the model to.
        **kwargs: forwarded to ``build_openshape_pointbert``.

    Returns:
        The model in ``eval`` mode with the weights loaded strictly.
    """
    if checkpoint_path is None:
        from huggingface_hub import hf_hub_download

        checkpoint_path = hf_hub_download(f"OpenShape/{variant}", "model.pt")

    # All supported checkpoint layouts contain tensors and built-in containers only.
    # Keep remote Hub files on PyTorch's restricted weights-only unpickler instead of
    # allowing arbitrary pickle code to execute during a normal model download.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_openshape_pointbert(variant, **kwargs)
    model.load_state_dict(_match_state_dict(checkpoint, model))
    model.eval()
    if device is not None:
        model.to(device)
    return model
