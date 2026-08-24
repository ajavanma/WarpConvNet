# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch

from warpconvnet.models.uni3d import (
    CELL_KNN_CELL_SIZE,
    CELL_KNN_MAX_SHELL,
    UNI3D_MODEL_REVISION,
    UNI3D_SOURCE_REVISION,
    UNI3D_UP_AXIS,
    UNI3D_VARIANTS,
    Uni3DGrouping,
    build_uni3d,
    cell_knn_indices,
    farthest_point_indices,
    knn_indices,
    load_uni3d,
    resolve_uni3d_variant,
    square_distance,
)


def test_registry_covers_every_released_checkpoint() -> None:
    assert UNI3D_SOURCE_REVISION == "64e03c3c42c196e8cb5ed03857810af9fc9ac39c"
    assert set(UNI3D_VARIANTS) == {
        "uni3d-ti",
        "uni3d-ti-no-lvis",
        "uni3d-s",
        "uni3d-s-no-lvis",
        "uni3d-b",
        "uni3d-b-no-lvis",
        "uni3d-l",
        "uni3d-l-no-lvis",
        "uni3d-g",
        "uni3d-g-no-lvis",
        "uni3d-g-lvis",
        "uni3d-g-modelnet40",
        "uni3d-g-scanobjnn",
    }
    assert {variant.revision for variant in UNI3D_VARIANTS.values()} == {UNI3D_MODEL_REVISION}
    assert UNI3D_UP_AXIS == "y"
    assert {variant.up_axis for variant in UNI3D_VARIANTS.values()} == {UNI3D_UP_AXIS}
    assert len({variant.filename for variant in UNI3D_VARIANTS.values()}) == len(UNI3D_VARIANTS)


@pytest.mark.parametrize(
    ("alias", "scale"),
    [("tiny", "ti"), ("s", "s"), ("base", "b"), ("l", "l"), ("giant", "g")],
)
def test_scale_aliases(alias: str, scale: str) -> None:
    assert resolve_uni3d_variant(alias).architecture.scale == scale


def test_every_scale_exposes_y_up_input_contract() -> None:
    for alias in ("tiny", "small", "base", "large", "giant"):
        with torch.device("meta"):
            model = build_uni3d(alias)
        assert model.up_axis == UNI3D_UP_AXIS


@pytest.mark.parametrize(
    ("variant", "width", "depth", "heads", "attention_key", "mlp_key", "pos_tokens"),
    [
        ("ti", 192, 12, 3, "attn.qkv.weight", "mlp.fc1.weight", 257),
        ("s", 384, 12, 6, "attn.qkv.weight", "mlp.fc1.weight", 257),
        ("b", 768, 12, 12, "attn.q_proj.weight", "mlp.fc1_g.weight", 1025),
        ("l", 1024, 24, 16, "attn.q_proj.weight", "mlp.fc1_g.weight", 1025),
        ("g", 1408, 40, 16, "attn.qkv.weight", "mlp.fc1.weight", 1601),
    ],
)
def test_all_scale_key_shapes_on_meta(
    variant: str,
    width: int,
    depth: int,
    heads: int,
    attention_key: str,
    mlp_key: str,
    pos_tokens: int,
) -> None:
    with torch.device("meta"):
        model = build_uni3d(variant)
    state = model.state_dict()
    prefix = "point_encoder.visual."
    assert model.point_encoder.trans_dim == width
    assert len(model.point_encoder.visual.blocks) == depth
    assert model.point_encoder.visual.blocks[0].attn.num_heads == heads
    assert state[prefix + "pos_embed"].shape == (1, pos_tokens, width)
    assert prefix + "blocks.0." + attention_key in state
    assert prefix + "blocks.0." + mlp_key in state
    assert state["point_encoder.trans2embed.weight"].shape == (1024, width)
    assert model.point_encoder.num_group == 512
    assert model.point_encoder.group_size == 64


def test_scale_specific_checkpoint_layouts() -> None:
    with torch.device("meta"):
        small = build_uni3d("s")
        base = build_uni3d("b")
        giant = build_uni3d("g")
    small_keys = small.state_dict()
    base_keys = base.state_dict()
    giant_keys = giant.state_dict()
    assert "point_encoder.visual.head.weight" not in small_keys
    assert base_keys["point_encoder.visual.head.weight"].shape == (1000, 768)
    assert giant_keys["point_encoder.visual.blocks.0.mlp.fc1.weight"].shape == (
        6144,
        1408,
    )
    assert "point_encoder.visual.blocks.0.mlp.norm.weight" not in giant_keys
    assert base_keys["point_encoder.visual.blocks.0.mlp.norm.weight"].shape == (2048,)


def test_square_distance_and_unsorted_topk_match_reference() -> None:
    xyz = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]]])
    centers = torch.tensor([[[0.9, 0.0, 0.0], [3.9, 0.0, 0.0]]])
    expected_distance = (
        -2 * torch.matmul(centers, xyz.transpose(1, 2))
        + torch.sum(centers**2, -1).view(1, 2, 1)
        + torch.sum(xyz**2, -1).view(1, 1, 4)
    )
    assert torch.equal(square_distance(centers, xyz), expected_distance)
    expected = torch.topk(expected_distance, 2, -1, largest=False, sorted=False).indices
    assert torch.equal(knn_indices(xyz, centers, 2), expected)


def test_torch_fps_is_deterministic_and_starts_at_zero() -> None:
    xyz = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    assert torch.equal(farthest_point_indices(xyz, 3, backend="torch"), torch.tensor([[0, 2, 1]]))


def test_grouping_accepts_factored_neighbor_selector() -> None:
    calls = []

    def first_neighbors(xyz: torch.Tensor, centers: torch.Tensor, k: int) -> torch.Tensor:
        calls.append((tuple(xyz.shape), tuple(centers.shape), k))
        return torch.arange(k).view(1, 1, k).expand(xyz.shape[0], centers.shape[1], -1)

    xyz = torch.arange(18, dtype=torch.float32).reshape(1, 6, 3)
    colors = torch.ones_like(xyz)
    grouping = Uni3DGrouping(2, 3, fps_backend="torch", neighbor_search=first_neighbors)
    neighborhood, centers, features = grouping(xyz, colors)
    assert calls == [((1, 6, 3), (1, 2, 3), 3)]
    assert neighborhood.shape == (1, 2, 3, 3)
    assert centers.shape == (1, 2, 3)
    assert features.shape == (1, 2, 3, 6)
    assert torch.equal(features[..., 3:], torch.ones(1, 2, 3, 3))


def test_grouping_exposes_cell_knn_as_an_opt_in_backend() -> None:
    reference = Uni3DGrouping(2, 3, fps_backend="torch")
    accelerated = Uni3DGrouping(2, 3, fps_backend="torch", neighbor_search="cell-knn")

    assert reference.neighbor_backend == "knn"
    assert reference.neighbor_search is knn_indices
    assert accelerated.neighbor_backend == "cell-knn"
    assert accelerated.neighbor_search is cell_knn_indices
    assert not accelerated._validate_neighbor_indices


def test_every_scale_accepts_cell_knn_configuration_on_meta() -> None:
    for variant in ("ti", "s", "b", "l", "g"):
        with torch.device("meta"):
            model = build_uni3d(variant, neighbor_search="cell-knn")
        assert model.point_encoder.group_divider.neighbor_backend == "cell-knn"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_grouping_cell_knn_runs_through_the_native_kernel() -> None:
    import warpconvnet._C as _C

    if not hasattr(_C.coords, "cell_nearest_k"):
        pytest.skip("cell-nearest-k CUDA kernel is not built")

    generator = torch.Generator(device="cuda").manual_seed(29)
    xyz = torch.rand(2, 128, 3, generator=generator, device="cuda")
    point_ids = torch.arange(128, device="cuda", dtype=xyz.dtype).view(1, 128, 1)
    color = point_ids.expand(2, -1, 3).contiguous()
    exact = Uni3DGrouping(8, 64, fps_backend="torch", neighbor_search="knn")
    fused = Uni3DGrouping(8, 64, fps_backend="torch", neighbor_search="cell-knn")

    with torch.inference_mode():
        _, exact_centers, exact_features = exact(xyz, color)
        neighborhood, fused_centers, fused_features = fused(xyz, color)

    torch.testing.assert_close(fused_centers, exact_centers, rtol=0, atol=0)
    torch.testing.assert_close(
        fused_features[..., 3].sort(dim=-1).values,
        exact_features[..., 3].sort(dim=-1).values,
        rtol=0,
        atol=0,
    )
    assert neighborhood.shape == (2, 8, 64, 3)
    assert torch.isfinite(neighborhood).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cell_knn_exact_fallback_is_query_selective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from warpconvnet.ops import cell_gather

    xyz = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.1, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
        ],
        device="cuda",
    )
    centers = xyz[:, [0, 2]]
    expected = knn_indices(xyz, centers, 2)
    captured: dict[str, object] = {}

    def fake_cell_nearest_k(
        points: torch.Tensor,
        ref_offsets: torch.Tensor,
        queries: torch.Tensor,
        query_offsets: torch.Tensor,
        k: int,
        *,
        cell_size: float,
        max_shell: int,
    ) -> cell_gather.CellNearestKResult:
        captured.update(cell_size=cell_size, max_shell=max_shell)
        batch, query_count = 2, 2
        local = expected.clone()
        local[0, 1] = torch.tensor([0, 1], device=xyz.device)
        starts = ref_offsets[:-1].long().view(batch, 1, 1)
        indices = (local + starts).reshape(batch * query_count, k).int()
        status = torch.tensor([0, 2, 0, 0], dtype=torch.int32, device=xyz.device)
        return cell_gather.CellNearestKResult(
            indices,
            torch.zeros_like(indices, dtype=torch.float32),
            torch.full((4,), k, dtype=torch.int32, device=xyz.device),
            status,
            torch.full((4,), 80, dtype=torch.int32, device=xyz.device),
        )

    monkeypatch.setattr(cell_gather, "cell_nearest_k", fake_cell_nearest_k)
    actual = cell_knn_indices(xyz, centers, 2)

    torch.testing.assert_close(actual.sort(-1).values, expected.sort(-1).values)
    assert captured == {
        "cell_size": CELL_KNN_CELL_SIZE,
        "max_shell": CELL_KNN_MAX_SHELL,
    }

    approximate = cell_knn_indices(xyz, centers, 2, exact_fallback=False)
    assert not torch.equal(approximate[0, 1].sort().values, expected[0, 1].sort().values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cell_knn_without_fallback_rejects_unsafe_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from warpconvnet.ops import cell_gather

    xyz = torch.zeros(1, 2, 3, device="cuda")

    def underfilled(*args, **kwargs) -> cell_gather.CellNearestKResult:
        del args, kwargs
        device = xyz.device
        return cell_gather.CellNearestKResult(
            torch.tensor([[0, -1]], dtype=torch.int32, device=device),
            torch.tensor([[0.0, torch.inf]], device=device),
            torch.tensor([1], dtype=torch.int32, device=device),
            torch.tensor([4], dtype=torch.int32, device=device),
            torch.tensor([1], dtype=torch.int32, device=device),
        )

    monkeypatch.setattr(cell_gather, "cell_nearest_k", underfilled)
    with pytest.raises(RuntimeError, match="underfilled"):
        cell_knn_indices(xyz, xyz[:, :1], 2, exact_fallback=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cell_knn_accepts_an_empty_query_axis() -> None:
    xyz = torch.randn(2, 4, 3, device="cuda")
    indices = cell_knn_indices(xyz, xyz[:, :0], 2)

    assert indices.shape == (2, 0, 2)
    assert indices.dtype == torch.long
    assert indices.device == xyz.device


@pytest.mark.parametrize("bad_index", [-1, 6])
def test_grouping_rejects_custom_neighbor_sentinels(bad_index: int) -> None:
    # A callable may happen to use the display name "knn"; validation is based
    # on function identity, not that name.
    def knn(xyz: torch.Tensor, centers: torch.Tensor, k: int) -> torch.Tensor:
        return torch.full(
            (xyz.shape[0], centers.shape[1], k),
            bad_index,
            device=xyz.device,
            dtype=torch.long,
        )

    xyz = torch.randn(1, 6, 3)
    grouping = Uni3DGrouping(2, 3, fps_backend="torch", neighbor_search=knn)
    with pytest.raises(ValueError, match=r"must be in \[0, 6\)"):
        grouping(xyz, torch.ones_like(xyz))


def test_tiny_cpu_forward_needs_no_reference_runtime() -> None:
    model = build_uni3d("tiny", fps_backend="torch").eval()
    model.point_encoder.group_divider.num_groups = 4
    model.point_encoder.group_divider.group_size = 3
    generator = torch.Generator().manual_seed(11)
    points = torch.randn(2, 8, 6, generator=generator)
    points[:, :, 3:] = points[:, :, 3:].sigmoid()
    with torch.inference_mode():
        embedding = model(points)
        wrapped = model(points, text=torch.ones(2, 1024), image=torch.zeros(2, 1024))
    assert embedding.shape == (2, 1024)
    assert torch.isfinite(embedding).all()
    assert wrapped["pc_embed"].shape == (2, 1024)
    assert wrapped["logit_scale"].ndim == 0


@pytest.mark.parametrize("variant", ["uni3d-ti", "uni3d-s", "uni3d-b"])
def test_cached_checkpoint_strict_load_if_available(variant: str) -> None:
    config = UNI3D_VARIANTS[variant]
    cache = (
        Path.home()
        / ".cache/huggingface/hub/models--BAAI--Uni3D/snapshots"
        / config.revision
        / config.filename
    )
    if not cache.is_file():
        pytest.skip(f"{variant} checkpoint is not cached")
    model = load_uni3d(variant, checkpoint_path=cache, dtype=None)
    assert model.uni3d_variant == variant
    assert next(model.parameters()).dtype == torch.float16
    assert not model.training
    for block in model.point_encoder.visual.blocks:
        if block.attn.q_bias is not None:
            assert not block.attn.k_bias.is_meta
            assert block.attn.k_bias.device.type == "cpu"
            assert block.attn.k_bias.dtype == torch.float16
            assert torch.count_nonzero(block.attn.k_bias) == 0


def test_cached_tiny_checkpoint_reduced_cpu_forward_if_available() -> None:
    config = UNI3D_VARIANTS["uni3d-ti"]
    cache = (
        Path.home()
        / ".cache/huggingface/hub/models--BAAI--Uni3D/snapshots"
        / config.revision
        / config.filename
    )
    if not cache.is_file():
        pytest.skip("uni3d-ti checkpoint is not cached")
    model = load_uni3d("tiny", checkpoint_path=cache, dtype=torch.float32)
    model.point_encoder.group_divider.num_groups = 4
    model.point_encoder.group_divider.group_size = 3
    points = torch.randn(1, 8, 6)
    points[:, :, 3:] = points[:, :, 3:].sigmoid()
    with torch.inference_mode():
        embedding = model(points)
    assert embedding.shape == (1, 1024)
    assert torch.isfinite(embedding).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_warp_fps_matches_portable_reference() -> None:
    generator = torch.Generator(device="cuda").manual_seed(7)
    xyz = torch.randn(2, 128, 3, generator=generator, device="cuda")
    expected = farthest_point_indices(xyz, 32, backend="torch")
    actual = farthest_point_indices(xyz, 32, backend="warp")
    assert torch.equal(actual, expected)
