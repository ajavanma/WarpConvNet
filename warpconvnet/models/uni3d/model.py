# SPDX-FileCopyrightText: Copyright (c) 2022 BAAI-Vision
# SPDX-FileCopyrightText: Copyright (c) 2023 BAAI-Vision
# SPDX-FileCopyrightText: Copyright (c) 2023 Ross Wightman
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""Native Uni3D point-cloud encoders.

The implementation follows and modifies the released BAAI Uni3D inference architecture while
depending only on PyTorch and WarpConvNet at runtime.  Parameter names and tensor
shapes intentionally match the official ``timm==0.9.7`` models so released
DeepSpeed checkpoints can be loaded strictly without a conversion step.

The original BAAI-Vision MIT notice is distributed in ``LICENSE`` beside this
file. The native implementation and later modifications are also licensed under
WarpConvNet's Apache-2.0 terms.

All released scales share the same point tokenizer: farthest-point sampling picks
512 centers, unsorted 64-nearest-neighbor search forms patches, and a two-stage
PointNet maps each patch to a transformer token.  The transformer architecture is
scale dependent: tiny/small use packed SwiGLU, base/large use unfused Q/K/V and a
normalized split SwiGLU, and giant uses the original EVA GELU MLP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from warpconvnet.ops.sampling import farthest_point_sampling

__all__ = [
    "CELL_KNN_CELL_SIZE",
    "CELL_KNN_MAX_SHELL",
    "FPSBackend",
    "NeighborBackend",
    "NeighborSearch",
    "UNI3D_MODEL_REVISION",
    "UNI3D_SOURCE_REVISION",
    "UNI3D_UP_AXIS",
    "UNI3D_VARIANTS",
    "Uni3D",
    "Uni3DArchitecture",
    "Uni3DGrouping",
    "Uni3DPointEncoder",
    "Uni3DVariant",
    "build_uni3d",
    "cell_knn_indices",
    "farthest_point_indices",
    "knn_indices",
    "resolve_uni3d_variant",
    "square_distance",
]

FPSBackend = Literal["auto", "warp", "torch"]
NeighborBackend = Literal["knn", "cell-knn"]
NeighborSearch = Callable[[Tensor, Tensor, int], Tensor]
MLPKind = Literal["packed_swiglu", "split_swiglu", "gelu"]

CELL_KNN_CELL_SIZE = 0.10
CELL_KNN_MAX_SHELL = 4

#: Every released Uni3D scale uses the same Y-up input frame. The upstream
#: preprocessing treats coordinate 1 as gravity and applies yaw augmentation
#: around Y. Centering and unit-ball scaling do not rotate the input frame.
UNI3D_UP_AXIS: Literal["y"] = "y"

UNI3D_MODEL_REVISION = "3d8233b76aa350d72f6213ecd2123c2026b42355"
UNI3D_SOURCE_REVISION = "64e03c3c42c196e8cb5ed03857810af9fc9ac39c"


@dataclass(frozen=True)
class Uni3DArchitecture:
    """Shape-changing fields of one released Uni3D scale."""

    scale: str
    timm_model: str
    width: int
    depth: int
    heads: int
    image_size: int
    qkv_fused: bool
    mlp_kind: MLPKind
    mlp_hidden: int
    visual_classes: int
    point_encoder_dim: int = 512
    output_dim: int = 1024
    num_groups: int = 512
    group_size: int = 64


@dataclass(frozen=True)
class Uni3DVariant:
    """Architecture, input frame, and Hugging Face metadata for released weights."""

    name: str
    architecture: Uni3DArchitecture
    filename: str
    size: int
    sha256: str
    training: str
    repo_id: str = "BAAI/Uni3D"
    revision: str = UNI3D_MODEL_REVISION
    up_axis: Literal["y"] = UNI3D_UP_AXIS


_ARCHITECTURES: dict[str, Uni3DArchitecture] = {
    "ti": Uni3DArchitecture(
        scale="ti",
        timm_model="eva02_tiny_patch14_224",
        width=192,
        depth=12,
        heads=3,
        image_size=224,
        qkv_fused=True,
        mlp_kind="packed_swiglu",
        mlp_hidden=512,
        visual_classes=0,
    ),
    "s": Uni3DArchitecture(
        scale="s",
        timm_model="eva02_small_patch14_224",
        width=384,
        depth=12,
        heads=6,
        image_size=224,
        qkv_fused=True,
        mlp_kind="packed_swiglu",
        mlp_hidden=1024,
        visual_classes=0,
    ),
    "b": Uni3DArchitecture(
        scale="b",
        timm_model="eva02_base_patch14_448",
        width=768,
        depth=12,
        heads=12,
        image_size=448,
        qkv_fused=False,
        mlp_kind="split_swiglu",
        mlp_hidden=2048,
        visual_classes=1000,
    ),
    "l": Uni3DArchitecture(
        scale="l",
        timm_model="eva02_large_patch14_448",
        width=1024,
        depth=24,
        heads=16,
        image_size=448,
        qkv_fused=False,
        mlp_kind="split_swiglu",
        mlp_hidden=2730,
        visual_classes=1000,
    ),
    "g": Uni3DArchitecture(
        scale="g",
        timm_model="eva_giant_patch14_560",
        width=1408,
        depth=40,
        heads=16,
        image_size=560,
        qkv_fused=True,
        mlp_kind="gelu",
        mlp_hidden=6144,
        visual_classes=1000,
    ),
}


def _variant(
    name: str,
    scale: str,
    filename: str,
    size: int,
    sha256: str,
    training: str,
) -> Uni3DVariant:
    return Uni3DVariant(name, _ARCHITECTURES[scale], filename, size, sha256, training)


# Every model checkpoint in the official repository at UNI3D_MODEL_REVISION.
UNI3D_VARIANTS: Mapping[str, Uni3DVariant] = {
    "uni3d-ti": _variant(
        "uni3d-ti",
        "ti",
        "modelzoo/uni3d-ti/model.pt",
        12_835_148,
        "6ff7e821e6997cb814b529e6dc3b224ad8349500471c8517e491685a718688e0",
        "ensembled",
    ),
    "uni3d-ti-no-lvis": _variant(
        "uni3d-ti-no-lvis",
        "ti",
        "modelzoo/uni3d-ti-no-lvis/model.pt",
        12_835_148,
        "4acc1369542311ba3c813b1e5160c897c04eb5a06bd1f0ab47e8c289e0b43719",
        "ensembled without LVIS",
    ),
    "uni3d-s": _variant(
        "uni3d-s",
        "s",
        "modelzoo/uni3d-s/model.pt",
        45_713_996,
        "9a6342fda2245f00fa0200eb7fc667ee67fee9a953c82961e04d242130a3350f",
        "ensembled",
    ),
    "uni3d-s-no-lvis": _variant(
        "uni3d-s-no-lvis",
        "s",
        "modelzoo/uni3d-s-no-lvis/model.pt",
        45_713_996,
        "b0198692d98b2cde7b8ad19ffdba81a2d07553781189861eced37a4bdd2720ac",
        "ensembled without LVIS",
    ),
    "uni3d-b": _variant(
        "uni3d-b",
        "b",
        "modelzoo/uni3d-b/model.pt",
        178_008_217,
        "bb533f918957e939f0e500376bef5166437883443da10fd9adccde0d1b512fba",
        "ensembled",
    ),
    "uni3d-b-no-lvis": _variant(
        "uni3d-b-no-lvis",
        "b",
        "modelzoo/uni3d-b-no-lvis/model.pt",
        178_008_217,
        "da0da5bc124f2e8e38bf754de0ebce7075a1eb9dbb0d468027adbf6bc8974a9f",
        "ensembled without LVIS",
    ),
    "uni3d-l": _variant(
        "uni3d-l",
        "l",
        "modelzoo/uni3d-l/model.pt",
        614_861_649,
        "540a3cafac52c251cbda1844d109bc598f142b4f5a2118db87cd722b01800c20",
        "ensembled",
    ),
    "uni3d-l-no-lvis": _variant(
        "uni3d-l-no-lvis",
        "l",
        "modelzoo/uni3d-l-no-lvis/model.pt",
        614_861_649,
        "b3bbdb3eb51426c6cdf7d7d43460c7cb7dabdcfef5bd390655796791883c0268",
        "ensembled without LVIS",
    ),
    "uni3d-g": _variant(
        "uni3d-g",
        "g",
        "modelzoo/uni3d-g/model.pt",
        2_034_909_361,
        "aa1f163bb8c34d7eb4bbb7187e8aaf78879b29df02d08acfbeb0671e2a218c3f",
        "ensembled",
    ),
    "uni3d-g-no-lvis": _variant(
        "uni3d-g-no-lvis",
        "g",
        "modelzoo/uni3d-g-no-lvis/model.pt",
        2_034_909_361,
        "6ab35a98481d9862c98ebab1d2c25102ce968b214a6324d9f38fa8091f33babc",
        "ensembled without LVIS",
    ),
    "uni3d-g-lvis": _variant(
        "uni3d-g-lvis",
        "g",
        "modelzoo/uni3d-g/lvis/model.pt",
        2_034_909_361,
        "f0d8ecd047935b037fcc40bdfd66bef632631a968c37fef0f90f1351641573c3",
        "LVIS task checkpoint",
    ),
    "uni3d-g-modelnet40": _variant(
        "uni3d-g-modelnet40",
        "g",
        "modelzoo/uni3d-g/mnet40/model.pt",
        2_034_909_361,
        "bcbcd6bea00e55d031b083ff06d4783e3d23bff0f02f23f076ed98db58b8a468",
        "ModelNet40 task checkpoint",
    ),
    "uni3d-g-scanobjnn": _variant(
        "uni3d-g-scanobjnn",
        "g",
        "modelzoo/uni3d-g/scanobjnn/model.pt",
        2_034_909_361,
        "d2b26da86c71ce48383a915a3f320b89843c774064edad2728324ef71145149b",
        "ScanObjectNN task checkpoint",
    ),
}

_VARIANT_ALIASES = {
    "ti": "uni3d-ti",
    "tiny": "uni3d-ti",
    "uni3d-tiny": "uni3d-ti",
    "ti-no-lvis": "uni3d-ti-no-lvis",
    "tiny-no-lvis": "uni3d-ti-no-lvis",
    "s": "uni3d-s",
    "small": "uni3d-s",
    "uni3d-small": "uni3d-s",
    "s-no-lvis": "uni3d-s-no-lvis",
    "small-no-lvis": "uni3d-s-no-lvis",
    "b": "uni3d-b",
    "base": "uni3d-b",
    "uni3d-base": "uni3d-b",
    "b-no-lvis": "uni3d-b-no-lvis",
    "base-no-lvis": "uni3d-b-no-lvis",
    "l": "uni3d-l",
    "large": "uni3d-l",
    "uni3d-large": "uni3d-l",
    "l-no-lvis": "uni3d-l-no-lvis",
    "large-no-lvis": "uni3d-l-no-lvis",
    "g": "uni3d-g",
    "giant": "uni3d-g",
    "uni3d-giant": "uni3d-g",
    "g-no-lvis": "uni3d-g-no-lvis",
    "giant-no-lvis": "uni3d-g-no-lvis",
    "uni3d-g-mnet40": "uni3d-g-modelnet40",
}


def resolve_uni3d_variant(variant: str) -> Uni3DVariant:
    """Resolve a canonical variant name or a scale alias."""
    name = _VARIANT_ALIASES.get(variant.lower(), variant.lower())
    try:
        return UNI3D_VARIANTS[name]
    except KeyError as exc:
        choices = ", ".join(UNI3D_VARIANTS)
        raise ValueError(f"Unknown Uni3D variant {variant!r}; choose from {choices}") from exc


def square_distance(src: Tensor, dst: Tensor) -> Tensor:
    """Official expanded squared-distance formula for ``(B, S, C)`` and ``(B, N, C)``."""
    if src.ndim != 3 or dst.ndim != 3 or src.shape[0] != dst.shape[0]:
        raise ValueError("src and dst must be rank-3 tensors with the same batch size")
    if src.shape[-1] != dst.shape[-1]:
        raise ValueError("src and dst must have the same coordinate width")
    return (
        -2 * torch.matmul(src, dst.transpose(1, 2))
        + src.pow(2).sum(-1, keepdim=True)
        + dst.pow(2).sum(-1).unsqueeze(1)
    )


def knn_indices(xyz: Tensor, centers: Tensor, k: int) -> Tensor:
    """Select the official unsorted top-k nearest neighbors.

    ``sorted=False`` follows the released implementation. The patch PointNet is
    permutation-invariant, so the selected set matters but its returned order does not.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer; got {k!r}")
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must have shape (B, N, 3); got {tuple(xyz.shape)}")
    if centers.ndim != 3 or centers.shape[0] != xyz.shape[0] or centers.shape[-1] != 3:
        raise ValueError(
            f"centers must have shape (B, S, 3) for B={xyz.shape[0]}; got {tuple(centers.shape)}"
        )
    if k > xyz.shape[1]:
        raise ValueError(f"k ({k}) cannot exceed the number of points ({xyz.shape[1]})")
    return torch.topk(
        square_distance(centers, xyz), k, dim=-1, largest=False, sorted=False
    ).indices


@torch.no_grad()
def cell_knn_indices(
    xyz: Tensor,
    centers: Tensor,
    k: int,
    *,
    cell_size: float = CELL_KNN_CELL_SIZE,
    max_shell: int = CELL_KNN_MAX_SHELL,
    exact_fallback: bool = True,
) -> Tensor:
    """Select nearest neighbors with the fused cell-list backend.

    The CUDA kernel scans voxel shells while maintaining nearest ``k`` points.
    With ``exact_fallback=True`` (the default), queries whose shell budget does
    not certify a complete global result fall back to the official dense kNN.
    Thus density changes affect performance, not whether valid indices are
    returned. Setting ``exact_fallback=False`` accepts an uncertified but full
    nearest set (status bit 1); invalid, clipped, or underfilled results still
    raise instead of leaking sentinel indices into the model.

    Certified kernel results use direct squared distances after canonicalizing
    coordinates to FP32. The released selector instead uses an expanded matrix
    formula, so roundoff at a tied 64th-neighbor boundary can select a different
    but geometrically equivalent set. Keep ``neighbor_search="knn"`` when exact
    released-implementation parity is required.

    The released exact ``knn_indices`` path remains Uni3D's default. This
    backend is CUDA-only and requires a current WarpConvNet native extension.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer; got {k!r}")
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must have shape (B, N, 3); got {tuple(xyz.shape)}")
    if centers.ndim != 3 or centers.shape[0] != xyz.shape[0] or centers.shape[-1] != 3:
        raise ValueError(
            f"centers must have shape (B, S, 3) for B={xyz.shape[0]}; "
            f"got {tuple(centers.shape)}"
        )
    if k > xyz.shape[1]:
        raise ValueError(f"k ({k}) cannot exceed the number of points ({xyz.shape[1]})")
    if xyz.device != centers.device:
        raise ValueError("xyz and centers must be on the same device")
    if xyz.dtype != centers.dtype or not torch.is_floating_point(xyz):
        raise TypeError("xyz and centers must have the same floating-point dtype")
    if not xyz.is_cuda:
        raise ValueError("The 'cell-knn' neighbor backend requires CUDA tensors")
    if not isinstance(exact_fallback, bool):
        raise TypeError(f"exact_fallback must be bool; got {type(exact_fallback).__name__}")
    if centers.shape[1] == 0:
        return torch.empty(
            xyz.shape[0],
            0,
            k,
            dtype=torch.long,
            device=xyz.device,
        )

    from warpconvnet.ops.cell_gather import cell_nearest_k

    batch, points, _ = xyz.shape
    queries = centers.shape[1]
    ref_offsets = torch.arange(
        0,
        batch * points + 1,
        points,
        dtype=torch.int32,
        device=xyz.device,
    )
    query_offsets = torch.arange(
        0,
        batch * queries + 1,
        queries,
        dtype=torch.int32,
        device=xyz.device,
    )
    result = cell_nearest_k(
        xyz.reshape(batch * points, 3),
        ref_offsets,
        centers.reshape(batch * queries, 3),
        query_offsets,
        k,
        cell_size=cell_size,
        max_shell=max_shell,
    )
    global_indices = result.indices.long().view(batch, queries, k)
    batch_starts = ref_offsets[:-1].long().view(batch, 1, 1)
    local_indices = global_indices - batch_starts
    status = result.status.view(batch, queries)

    if exact_fallback:
        fallback_mask = status.ne(0)
        if bool(fallback_mask.any()):
            # The fallback is intentionally query-selective. Sparse or unusual
            # clouds do not force the common, certified queries through B*S*N
            # dense distance materialization.
            for batch_index in range(batch):
                row_mask = fallback_mask[batch_index]
                failed_centers = centers[batch_index : batch_index + 1, row_mask]
                if failed_centers.shape[1] == 0:
                    continue
                exact = knn_indices(xyz[batch_index : batch_index + 1], failed_centers, k).squeeze(
                    0
                )
                local_indices[batch_index, row_mask] = exact
    else:
        # Bit 1 is the deliberate approximation: k points were found, but the
        # finite shell budget could not prove they are globally nearest. The
        # other bits may contain -1 or incomplete results and are never safe to
        # pass into PyTorch advanced indexing.
        unsafe_status = status.bitwise_and(~2)
        if bool(unsafe_status.ne(0).any()):
            observed = sorted(set(unsafe_status[unsafe_status.ne(0)].tolist()))
            raise RuntimeError(
                "cell-knn returned invalid, underfilled, or coordinate-clipped "
                f"queries with status values {observed}; increase max_shell or enable "
                "exact_fallback"
            )
    return local_indices


def _farthest_point_indices_torch(xyz: Tensor, count: int) -> Tensor:
    batch, points, _ = xyz.shape
    selected = torch.empty(batch, count, dtype=torch.long, device=xyz.device)
    min_distance = torch.full(
        (batch, points), torch.finfo(xyz.dtype).max, dtype=xyz.dtype, device=xyz.device
    )
    farthest = torch.zeros(batch, dtype=torch.long, device=xyz.device)
    batch_idx = torch.arange(batch, device=xyz.device)
    for index in range(count):
        selected[:, index] = farthest
        center = xyz[batch_idx, farthest].unsqueeze(1)
        min_distance = torch.minimum(min_distance, (xyz - center).pow(2).sum(-1))
        farthest = min_distance.argmax(-1)
    return selected


def farthest_point_indices(
    xyz: Tensor,
    count: int,
    *,
    backend: FPSBackend = "auto",
) -> Tensor:
    """Return batch-local FPS indices, using WarpConvNet on CUDA."""
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must have shape (B, N, 3); got {tuple(xyz.shape)}")
    if not torch.is_floating_point(xyz):
        raise TypeError(f"xyz must be floating point; got {xyz.dtype}")
    batch, points, _ = xyz.shape
    if batch == 0 or points == 0:
        raise ValueError("xyz must contain at least one batch item and one point")
    if isinstance(count, bool) or not isinstance(count, int) or not 0 < count <= points:
        raise ValueError(f"count must be an integer in [1, {points}]; got {count!r}")
    if backend not in ("auto", "warp", "torch"):
        raise ValueError(f"Unknown FPS backend {backend!r}; use 'auto', 'warp', or 'torch'")
    if backend == "auto":
        backend = "warp" if xyz.is_cuda else "torch"
    if backend == "warp":
        if not xyz.is_cuda:
            raise ValueError("The 'warp' FPS backend requires a CUDA tensor")
        packed = xyz.reshape(batch * points, 3).float().contiguous()
        offsets = torch.arange(0, batch * points + 1, points, device=xyz.device, dtype=torch.int32)
        global_indices = farthest_point_sampling(packed, offsets, count).long().view(batch, count)
        return global_indices - torch.arange(batch, device=xyz.device).view(batch, 1) * points
    return _farthest_point_indices_torch(xyz, count)


class Uni3DGrouping(nn.Module):
    """FPS centers plus a factored, replaceable neighborhood selector."""

    def __init__(
        self,
        num_groups: int = 512,
        group_size: int = 64,
        *,
        fps_backend: FPSBackend = "auto",
        neighbor_search: NeighborBackend | NeighborSearch = "knn",
    ) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.fps_backend = fps_backend
        if neighbor_search == "knn":
            self.neighbor_search: NeighborSearch = knn_indices
            self.neighbor_backend = "knn"
        elif neighbor_search == "cell-knn":
            self.neighbor_search = cell_knn_indices
            self.neighbor_backend = "cell-knn"
        elif callable(neighbor_search):
            self.neighbor_search = neighbor_search
            self.neighbor_backend = getattr(neighbor_search, "__name__", "custom")
        else:
            raise ValueError("neighbor_search must be 'knn', 'cell-knn', or a callable")
        self._validate_neighbor_indices = self.neighbor_search not in (
            knn_indices,
            cell_knn_indices,
        )

    def forward(self, xyz: Tensor, color: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError(f"xyz must have shape (B, N, 3); got {tuple(xyz.shape)}")
        if color.shape != xyz.shape:
            raise ValueError(f"color must have shape {tuple(xyz.shape)}; got {tuple(color.shape)}")
        if xyz.device != color.device:
            raise ValueError("xyz and color must be on the same device")

        center_idx = farthest_point_indices(xyz, self.num_groups, backend=self.fps_backend)
        batch_idx = torch.arange(xyz.shape[0], device=xyz.device).view(-1, 1)
        centers = xyz[batch_idx, center_idx]
        neighbor_idx = self.neighbor_search(xyz, centers, self.group_size)
        expected = (xyz.shape[0], self.num_groups, self.group_size)
        if tuple(neighbor_idx.shape) != expected:
            raise ValueError(
                f"neighbor_search returned {tuple(neighbor_idx.shape)}; expected {expected}"
            )
        if neighbor_idx.dtype not in (torch.int32, torch.int64):
            raise TypeError("neighbor_search must return integer indices")
        if neighbor_idx.device != xyz.device:
            raise ValueError("neighbor_search indices must be on the input device")
        # The built-in top-k is range-safe by construction. Custom search methods
        # commonly use -1 as an internal sentinel, so fail clearly before PyTorch
        # silently treats it as the final point. The validation sync is kept off
        # the released hot path.
        if self._validate_neighbor_indices and bool(
            ((neighbor_idx < 0) | (neighbor_idx >= xyz.shape[1])).any()
        ):
            raise ValueError(f"neighbor_search indices must be in [0, {xyz.shape[1]})")

        gather_batch = torch.arange(xyz.shape[0], device=xyz.device).view(-1, 1, 1)
        neighborhood = xyz[gather_batch, neighbor_idx]
        neighborhood_color = color[gather_batch, neighbor_idx]
        neighborhood = neighborhood - centers.unsqueeze(2)
        features = torch.cat((neighborhood, neighborhood_color), dim=-1)
        return neighborhood.contiguous(), centers.contiguous(), features.contiguous()


class Uni3DPatchEncoder(nn.Module):
    """The released two-stage PointNet patch tokenizer."""

    def __init__(self, encoder_channel: int = 512) -> None:
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(6, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, encoder_channel, 1),
        )

    def forward(self, point_groups: Tensor) -> Tensor:
        if point_groups.ndim != 4 or point_groups.shape[-1] != 6:
            raise ValueError(
                f"point_groups must have shape (B, G, K, 6); got {tuple(point_groups.shape)}"
            )
        batch, groups, points, _ = point_groups.shape
        point_groups = point_groups.reshape(batch * groups, points, 6)
        feature = self.first_conv(point_groups.transpose(2, 1))
        global_feature = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat((global_feature.expand(-1, -1, points), feature), dim=1)
        feature = self.second_conv(feature)
        global_feature = torch.max(feature, dim=2)[0]
        return global_feature.reshape(batch, groups, self.encoder_channel)


class _PackedSwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, hidden * 2)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(0.0)
        self.norm = nn.Identity()
        self.fc2 = nn.Linear(hidden, width)
        self.drop2 = nn.Dropout(0.0)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.fc1(x).chunk(2, dim=-1)
        x = self.act(gate) * value
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return self.drop2(x)


class _SplitSwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.fc1_g = nn.Linear(width, hidden)
        self.fc1_x = nn.Linear(width, hidden)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(0.0)
        self.norm = nn.LayerNorm(hidden, eps=1e-6)
        self.fc2 = nn.Linear(hidden, width)
        self.drop2 = nn.Dropout(0.0)
        self.drop = nn.Dropout(0.0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.fc1_g(x)) * self.fc1_x(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return self.drop2(x)


class _GELUMLP(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(0.0)
        self.norm = nn.Identity()
        self.fc2 = nn.Linear(hidden, width)
        self.drop2 = nn.Dropout(0.0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return self.drop2(x)


class _EVAAttention(nn.Module):
    def __init__(self, width: int, heads: int, *, qkv_fused: bool) -> None:
        super().__init__()
        if width % heads:
            raise ValueError(f"width ({width}) must be divisible by heads ({heads})")
        self.num_heads = heads
        self.head_dim = width // heads
        self.scale = self.head_dim**-0.5
        if qkv_fused:
            self.qkv: nn.Linear | None = nn.Linear(width, width * 3, bias=False)
            self.q_proj = self.k_proj = self.v_proj = None
            self.q_bias: nn.Parameter | None = nn.Parameter(torch.zeros(width))
            self.register_buffer("k_bias", torch.zeros(width), persistent=False)
            self.v_bias: nn.Parameter | None = nn.Parameter(torch.zeros(width))
        else:
            self.qkv = None
            self.q_proj = nn.Linear(width, width, bias=True)
            self.k_proj = nn.Linear(width, width, bias=False)
            self.v_proj = nn.Linear(width, width, bias=True)
            self.q_bias = self.k_bias = self.v_bias = None
        self.attn_drop = nn.Dropout(0.0)
        self.norm = nn.Identity()
        self.proj = nn.Linear(width, width)
        self.proj_drop = nn.Dropout(0.0)

    def materialize_nonpersistent_buffers(self) -> None:
        """Recreate timm's non-persistent zero K bias after a meta-device load."""
        if self.q_bias is not None:
            self.k_bias = torch.zeros_like(self.q_bias)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, width = x.shape
        if self.qkv is not None:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
            qkv = F.linear(x, self.qkv.weight, qkv_bias)
            qkv = qkv.reshape(batch, tokens, 3, self.num_heads, self.head_dim)
            query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        else:
            query = (
                self.q_proj(x)
                .reshape(batch, tokens, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            key = (
                self.k_proj(x)
                .reshape(batch, tokens, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            value = (
                self.v_proj(x)
                .reshape(batch, tokens, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
        x = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(batch, tokens, width)
        x = self.norm(x)
        x = self.proj(x)
        return self.proj_drop(x)


class _EVABlock(nn.Module):
    def __init__(self, architecture: Uni3DArchitecture) -> None:
        super().__init__()
        width = architecture.width
        self.norm1 = nn.LayerNorm(width, eps=1e-6)
        self.attn = _EVAAttention(width, architecture.heads, qkv_fused=architecture.qkv_fused)
        self.drop_path1 = nn.Identity()
        self.norm2 = nn.LayerNorm(width, eps=1e-6)
        if architecture.mlp_kind == "packed_swiglu":
            self.mlp = _PackedSwiGLU(width, architecture.mlp_hidden)
        elif architecture.mlp_kind == "split_swiglu":
            self.mlp = _SplitSwiGLU(width, architecture.mlp_hidden)
        elif architecture.mlp_kind == "gelu":
            self.mlp = _GELUMLP(width, architecture.mlp_hidden)
        else:  # pragma: no cover - guarded by the frozen built-in registry
            raise ValueError(f"Unknown MLP kind {architecture.mlp_kind!r}")
        self.drop_path2 = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        return x + self.drop_path2(self.mlp(self.norm2(x)))


class _PatchEmbedParameters(nn.Module):
    """Unused image projection retained solely for official checkpoint compatibility."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, width, kernel_size=14, stride=14)


class _EVAVisual(nn.Module):
    """The subset of timm EVA used by Uni3D, with all checkpoint parameters retained."""

    def __init__(self, architecture: Uni3DArchitecture) -> None:
        super().__init__()
        width = architecture.width
        patch_count = (architecture.image_size // 14) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.pos_embed = nn.Parameter(torch.zeros(1, patch_count + 1, width))
        self.patch_embed = _PatchEmbedParameters(width)
        self.pos_drop = nn.Dropout(0.0)
        self.blocks = nn.ModuleList([_EVABlock(architecture) for _ in range(architecture.depth)])
        self.norm = nn.Identity()
        self.fc_norm = nn.LayerNorm(width, eps=1e-6)
        self.head_drop = nn.Dropout(0.0)
        self.head = (
            nn.Linear(width, architecture.visual_classes)
            if architecture.visual_classes > 0
            else nn.Identity()
        )


class Uni3DPointEncoder(nn.Module):
    """Point tokenizer, EVA transformer, and projection into EVA-CLIP-E space."""

    def __init__(
        self,
        architecture: Uni3DArchitecture,
        *,
        fps_backend: FPSBackend = "auto",
        neighbor_search: NeighborBackend | NeighborSearch = "knn",
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.trans_dim = architecture.width
        self.embed_dim = architecture.output_dim
        self.group_size = architecture.group_size
        self.num_group = architecture.num_groups
        self.group_divider = Uni3DGrouping(
            num_groups=self.num_group,
            group_size=self.group_size,
            fps_backend=fps_backend,
            neighbor_search=neighbor_search,
        )
        self.encoder_dim = architecture.point_encoder_dim
        self.encoder = Uni3DPatchEncoder(self.encoder_dim)
        self.encoder2trans = nn.Linear(self.encoder_dim, self.trans_dim)
        self.trans2embed = nn.Linear(self.trans_dim, self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim)
        )
        self.patch_dropout = nn.Identity()
        self.visual = _EVAVisual(architecture)

    def forward(self, xyz: Tensor, colors: Tensor) -> Tensor:
        _, centers, features = self.group_divider(xyz, colors)
        tokens = self.encoder2trans(self.encoder(features))
        cls_tokens = self.cls_token.expand(tokens.shape[0], -1, -1)
        cls_pos = self.cls_pos.expand(tokens.shape[0], -1, -1)
        positions = torch.cat((cls_pos, self.pos_embed(centers)), dim=1)
        x = torch.cat((cls_tokens, tokens), dim=1) + positions
        x = self.patch_dropout(x)
        x = self.visual.pos_drop(x)
        for block in self.visual.blocks:
            x = block(x)
        x = self.visual.norm(x[:, 0, :])
        x = self.visual.fc_norm(x)
        return self.trans2embed(x)


class Uni3D(nn.Module):
    """A released Uni3D point-cloud model with official input/output semantics."""

    def __init__(
        self,
        architecture: Uni3DArchitecture,
        *,
        fps_backend: FPSBackend = "auto",
        neighbor_search: NeighborBackend | NeighborSearch = "knn",
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.up_axis: Literal["y"] = UNI3D_UP_AXIS
        self.logit_scale = nn.Parameter(torch.ones(()) * math.log(1 / 0.07))
        self.point_encoder = Uni3DPointEncoder(
            architecture, fps_backend=fps_backend, neighbor_search=neighbor_search
        )

    def materialize_nonpersistent_buffers(self) -> None:
        """Initialize buffers absent from official state dictionaries."""
        for block in self.point_encoder.visual.blocks:
            block.attn.materialize_nonpersistent_buffers()

    def encode_pc(self, point_cloud: Tensor) -> Tensor:
        """Encode channels-last ``(B, N, 6)`` XYZ+RGB input in the Y-up frame.

        All released Uni3D scales use Y-up. The caller must rotate other input
        frames before centering and unit-ball normalization; this model does not
        rotate coordinates.
        """
        if point_cloud.ndim != 3 or point_cloud.shape[-1] != 6:
            raise ValueError(
                f"point_cloud must have shape (B, N, 6); got {tuple(point_cloud.shape)}"
            )
        if not torch.is_floating_point(point_cloud):
            raise TypeError(f"point_cloud must be floating point; got {point_cloud.dtype}")
        xyz = point_cloud[:, :, :3].contiguous()
        colors = point_cloud[:, :, 3:].contiguous()
        return self.point_encoder(xyz, colors)

    def forward(
        self,
        point_cloud: Tensor,
        text: Tensor | None = None,
        image: Tensor | None = None,
    ) -> Tensor | dict[str, Tensor | None]:
        """Encode points, or preserve the official multimodal wrapper when peers are supplied."""
        point_embedding = self.encode_pc(point_cloud)
        if text is None and image is None:
            return point_embedding
        return {
            "text_embed": text,
            "pc_embed": point_embedding,
            "image_embed": image,
            "logit_scale": self.logit_scale.exp(),
        }


def build_uni3d(
    variant: str = "uni3d-s",
    *,
    fps_backend: FPSBackend = "auto",
    neighbor_search: NeighborBackend | NeighborSearch = "knn",
) -> Uni3D:
    """Build an uninitialized released Uni3D architecture.

    ``variant`` accepts canonical checkpoint names as well as scale aliases such
    as ``"small"`` or ``"g"``.  All checkpoints at a scale share one architecture.
    """
    config = resolve_uni3d_variant(variant)
    model = Uni3D(
        config.architecture,
        fps_backend=fps_backend,
        neighbor_search=neighbor_search,
    )
    model.uni3d_variant = config.name
    return model
