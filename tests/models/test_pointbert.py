# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import warpconvnet.models.ppat.model as ppat_model
from warpconvnet.geometry.types.points import Points
from warpconvnet.models.ppat import (
    OPENSHAPE_VARIANTS,
    PointPatchTransformer,
    ProjectedPointPatchTransformer,
    build_openshape_pointbert,
    farthest_point_sample,
    index_points,
    load_openshape_pointbert,
    normalize_point_cloud,
    query_ball_point,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def small_ppat():
    """A cheap PPAT with the released models' input layout (xyz + rgb)."""
    torch.manual_seed(0)
    return (
        PointPatchTransformer(
            dim=32,
            depth=2,
            heads=2,
            mlp_dim=64,
            sa_dim=16,
            patches=8,
            prad=0.4,
            nsamp=16,
            in_dim=6,
            dim_head=8,
        )
        .to(DEVICE)
        .eval()
    )


@pytest.fixture
def batch():
    """(B, 6, N) channels-first xyz + rgb, normalized the way OpenShape does."""
    torch.manual_seed(0)
    B, N = 2, 512
    xyz = normalize_point_cloud(torch.rand(B, N, 3) * 2 - 1)
    rgb = torch.rand(B, N, 3)
    return torch.cat([xyz, rgb], dim=-1).permute(0, 2, 1).contiguous().to(DEVICE)


# ---------------------------------------------------------------------------
# Sampling / grouping primitives
# ---------------------------------------------------------------------------


def test_normalize_point_cloud():
    xyz = torch.rand(3, 100, 3) * 10 + 5
    out = normalize_point_cloud(xyz)
    assert torch.allclose(out.mean(dim=-2), torch.zeros(3, 3), atol=1e-5)
    assert torch.allclose(out.norm(dim=-1).amax(dim=-1), torch.ones(3), atol=1e-5)
    # A degenerate cloud collapses to zeros instead of dividing by ~0.
    assert torch.equal(normalize_point_cloud(torch.zeros(1, 10, 3)), torch.zeros(1, 10, 3))


@pytest.mark.parametrize("backend", ["torch", "warp"])
def test_farthest_point_sample(backend):
    if backend == "warp" and not torch.cuda.is_available():
        pytest.skip("warp FPS backend needs CUDA")
    torch.manual_seed(0)
    xyz = torch.rand(3, 256, 3, device=DEVICE)
    idx = farthest_point_sample(xyz, 32, backend=backend)
    assert idx.shape == (3, 32)
    assert idx.min() >= 0 and idx.max() < 256
    for b in range(3):
        assert idx[b].unique().numel() == 32, "FPS must not repeat points"


def test_farthest_point_sample_covers_clusters():
    """FPS on well-separated clusters must hit every cluster before repeating one."""
    centers = torch.tensor([[0.0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]])
    xyz = (centers.repeat_interleave(50, 0) + torch.rand(200, 3) * 0.1).unsqueeze(0).to(DEVICE)
    idx = farthest_point_sample(xyz, 4, backend="torch")[0]
    assert set((idx // 50).tolist()) == {0, 1, 2, 3}


@pytest.mark.parametrize(
    "xyz,npoint,match",
    [
        (torch.empty(0, 8, 3), 1, "at least one batch item"),
        (torch.empty(2, 0, 3), 1, "at least one batch item"),
        (torch.rand(2, 8, 3), 0, "positive integer"),
        (torch.rand(2, 8, 3), 9, "cannot exceed"),
    ],
)
def test_farthest_point_sample_rejects_unsafe_sizes(xyz, npoint, match):
    with pytest.raises(ValueError, match=match):
        farthest_point_sample(xyz, npoint, backend="torch")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_warp_fps_converts_low_precision_storage_to_float32(monkeypatch, dtype):
    seen = {}

    def fake_fps(points, offsets, npoint):
        seen["points"] = points
        B = offsets.numel() - 1
        N = points.shape[0] // B
        return torch.cat(
            [torch.arange(npoint, dtype=torch.int32) + batch * N for batch in range(B)]
        )

    monkeypatch.setattr(ppat_model, "farthest_point_sampling", fake_fps)
    xyz = torch.rand(2, 8, 3).to(dtype)
    idx = ppat_model._farthest_point_sample_warp(xyz, 4)

    assert seen["points"].dtype == torch.float32
    assert idx.tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]


def test_farthest_point_sample_validates_backend_and_start_index():
    xyz = torch.rand(1, 8, 3)
    with pytest.raises(ValueError, match="requires xyz on a CUDA device"):
        farthest_point_sample(xyz, 4, backend="warp")
    with pytest.raises(ValueError, match="Unknown FPS backend"):
        farthest_point_sample(xyz, 4, backend="unknown")
    with pytest.raises(ValueError, match="start_idx must be an integer"):
        farthest_point_sample(xyz, 4, backend="auto", start_idx=8)


def test_query_ball_point_semantics():
    # 6 support points on a line; query at the origin with radius 2.5 keeps 0..2.
    xyz = torch.tensor([[[float(i), 0.0, 0.0] for i in range(6)]], device=DEVICE)
    new_xyz = torch.tensor([[[0.0, 0.0, 0.0]]], device=DEVICE)

    # Room to spare: short groups repeat the first (lowest-index) neighbor.
    idx = query_ball_point(2.5, 5, xyz, new_xyz)
    assert idx.tolist() == [[[0, 1, 2, 0, 0]]]

    # Over-full groups keep the nsample lowest point indices, matching PointNet++.
    idx = query_ball_point(2.5, 2, xyz, new_xyz)
    assert idx.tolist() == [[[0, 1]]]

    # Chunking the query dimension must not change the result.
    many = torch.rand(2, 37, 3, device=DEVICE)
    support = torch.rand(2, 128, 3, device=DEVICE)
    ref = query_ball_point(0.5, 8, support, many)
    assert torch.equal(ref, query_ball_point(0.5, 8, support, many, chunk_size=7))


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True])
def test_query_ball_point_rejects_invalid_chunk_size(chunk_size):
    xyz = torch.rand(1, 8, 3)
    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        query_ball_point(0.5, 4, xyz, xyz[:, :2], chunk_size=chunk_size)


@pytest.mark.parametrize("radius", [0.0, -0.1, float("nan"), float("inf"), True])
def test_query_ball_point_rejects_invalid_radius(radius):
    xyz = torch.rand(1, 8, 3)
    with pytest.raises(ValueError, match="radius must be a finite positive number"):
        query_ball_point(radius, 4, xyz, xyz[:, :2])


@pytest.mark.parametrize("nsample", [0, -1, 1.5, True])
def test_query_ball_point_rejects_invalid_nsample(nsample):
    xyz = torch.rand(1, 8, 3)
    with pytest.raises(ValueError, match="nsample must be a positive integer"):
        query_ball_point(0.5, nsample, xyz, xyz[:, :2])


def test_query_ball_point_validates_tensor_contract():
    xyz = torch.rand(1, 8, 3)
    with pytest.raises(ValueError, match="cannot exceed the number of support points"):
        query_ball_point(0.5, 9, xyz, xyz[:, :2])
    with pytest.raises(ValueError, match="xyz must have shape"):
        query_ball_point(0.5, 4, xyz.transpose(1, 2), xyz[:, :2])
    with pytest.raises(ValueError, match="new_xyz must have shape"):
        query_ball_point(0.5, 4, xyz, torch.rand(1, 3, 2))
    with pytest.raises(TypeError, match="must be floating point"):
        query_ball_point(0.5, 4, xyz.long(), xyz[:, :2].long())
    with pytest.raises(ValueError, match="batch sizes must match"):
        query_ball_point(0.5, 4, xyz, torch.rand(2, 2, 3))
    with pytest.raises(ValueError, match="share a dtype"):
        query_ball_point(0.5, 4, xyz, xyz[:, :2].double())
    with pytest.raises(ValueError, match="share a device"):
        query_ball_point(0.5, 4, xyz, torch.empty(1, 2, 3, device="meta"))
    with pytest.raises(ValueError, match="at least one batch item and one point"):
        query_ball_point(0.5, 1, torch.empty(1, 0, 3), torch.empty(1, 0, 3))
    with pytest.raises(ValueError, match="at least one query point"):
        query_ball_point(0.5, 1, xyz, torch.empty(1, 0, 3))


def test_index_points():
    points = torch.arange(2 * 5 * 3, dtype=torch.float32).view(2, 5, 3)
    idx2d = torch.tensor([[4, 0], [1, 3]])
    assert torch.equal(index_points(points, idx2d)[0, 0], points[0, 4])
    idx3d = idx2d.unsqueeze(-1).expand(-1, -1, 4)
    assert index_points(points, idx3d).shape == (2, 2, 4, 3)


# ---------------------------------------------------------------------------
# Model plumbing
# ---------------------------------------------------------------------------


def _small_ppat_kwargs():
    return dict(
        dim=32,
        depth=1,
        heads=2,
        mlp_dim=64,
        sa_dim=16,
        patches=8,
        prad=0.4,
        nsamp=16,
        in_dim=6,
        dim_head=8,
    )


@pytest.mark.parametrize(
    "override,match",
    [
        ({"patches": 0}, "patches must be a positive integer"),
        ({"patch_dropout": -1}, "patch_dropout must be an integer"),
        ({"patch_dropout": 8}, "patch_dropout must be an integer"),
        ({"nsamp": 0}, "nsample must be a positive integer"),
        ({"prad": 0.0}, "radius must be finite and positive"),
        ({"prad": float("nan")}, "radius must be finite and positive"),
        ({"prad": float("inf")}, "radius must be finite and positive"),
        ({"ball_query_chunk": 0}, "ball_query_chunk must be a positive integer"),
        ({"fps_backend": "unknown"}, "Unknown fps_backend"),
        ({"attn_backend": "unknown"}, "Unknown attn_backend"),
        ({"in_dim": 0}, "in_dim must be a positive integer"),
    ],
)
def test_ppat_rejects_invalid_sampling_configuration(override, match):
    kwargs = _small_ppat_kwargs()
    kwargs.update(override)
    with pytest.raises(ValueError, match=match):
        PointPatchTransformer(**kwargs)


def test_ppat_rejects_more_patches_than_input_points():
    model = PointPatchTransformer(**_small_ppat_kwargs(), fps_backend="torch").eval()
    with pytest.raises(ValueError, match="cannot exceed the number of input points"):
        model(torch.rand(1, 6, 7))


def test_tensor_input_contract_is_validated_before_model_ops():
    model = PointPatchTransformer(**_small_ppat_kwargs(), fps_backend="torch").eval()
    with pytest.raises(ValueError, match="3-D channels-first"):
        model(torch.rand(6, 8))
    with pytest.raises(TypeError, match="points must be floating point"):
        model(torch.ones(1, 6, 8, dtype=torch.int64))
    with pytest.raises(ValueError, match="at least three XYZ channels"):
        model(torch.rand(1, 2, 8))
    with pytest.raises(ValueError, match="at least one batch item and one point"):
        model(torch.empty(1, 6, 0))


def test_separate_xyz_and_features_contract_is_validated():
    xyz = torch.rand(2, 3, 16)
    with pytest.raises(ValueError, match="separate xyz input must have shape"):
        ppat_model._to_channels_first(torch.rand(2, 4, 16), torch.rand(2, 6, 16))
    with pytest.raises(ValueError, match="share batch and point dimensions"):
        ppat_model._to_channels_first(xyz, torch.rand(2, 6, 15))
    with pytest.raises(ValueError, match="share a dtype"):
        ppat_model._to_channels_first(xyz, torch.rand(2, 6, 16, dtype=torch.float64))
    with pytest.raises(ValueError, match="share a device"):
        ppat_model._to_channels_first(xyz, torch.empty(2, 6, 16, device="meta"))


def test_model_reports_feature_channel_mismatch_clearly():
    model = PointPatchTransformer(**_small_ppat_kwargs(), fps_backend="torch").eval()
    xyz = torch.rand(1, 3, 16)
    rgb_only = torch.rand(1, 3, 16)

    with pytest.raises(ValueError, match=r"require all six XYZ\+RGB channels"):
        model(xyz, rgb_only)


def test_points_input_rejects_empty_batches(small_ppat):
    empty_batch = Points(
        torch.empty(0, 3), torch.empty(0, 6), offsets=torch.tensor([0], dtype=torch.int32)
    )
    with pytest.raises(ValueError, match="non-empty Points batch"):
        small_ppat(empty_batch)

    empty_item = Points(
        torch.empty(0, 3), torch.empty(0, 6), offsets=torch.tensor([0, 0], dtype=torch.int32)
    )
    with pytest.raises(ValueError, match="at least one point per batch item"):
        small_ppat(empty_item)


def test_points_input_accepts_uniform_padded_features():
    B, N = 2, 16
    coordinates = torch.rand(B * N, 3)
    features = torch.rand(B, N, 6)
    offsets = torch.tensor([0, N, B * N], dtype=torch.int32)
    points = Points(coordinates, features, offsets=offsets)

    xyz, feats = ppat_model._to_channels_first(points, None)
    assert xyz.shape == (B, 3, N)
    assert feats.shape == (B, 6, N)


def test_forward_input_forms_agree(small_ppat, batch):
    """features-only, (xyz, features), and Points must all encode identically."""
    with torch.no_grad():
        from_feats = small_ppat(batch)
        from_pair = small_ppat(batch[:, :3].contiguous(), batch)

        B, N = batch.shape[0], batch.shape[2]
        rowwise = batch.permute(0, 2, 1)
        pc = Points(
            [rowwise[b, :, :3].contiguous() for b in range(B)],
            [rowwise[b].contiguous() for b in range(B)],
        ).to(DEVICE)
        from_points = small_ppat(pc)

    assert from_feats.shape == (batch.shape[0], 32)
    assert torch.allclose(from_feats, from_pair, atol=1e-5)
    assert torch.allclose(from_feats, from_points, atol=1e-5)


def test_points_input_rejects_ragged_batches(small_ppat):
    coords = [torch.rand(100, 3), torch.rand(120, 3)]
    feats = [torch.rand(100, 6), torch.rand(120, 6)]
    pc = Points(coords, feats).to(DEVICE)
    with pytest.raises(ValueError, match="uniform point count"):
        small_ppat(pc)


def test_patch_dropout_only_in_training(batch):
    torch.manual_seed(0)
    model = PointPatchTransformer(
        dim=32,
        depth=1,
        heads=2,
        mlp_dim=64,
        sa_dim=16,
        patches=8,
        prad=0.4,
        nsamp=16,
        in_dim=6,
        dim_head=8,
        patch_dropout=3,
    ).to(DEVICE)

    model.eval()
    with torch.no_grad():
        model(batch)
    assert model.sa.npoint == 8, "eval must not mutate the configured patch count"

    model.train()
    model(batch).sum().backward()
    assert model.cls_token.grad is not None


def test_backward_reaches_all_parameters(small_ppat, batch):
    small_ppat.train()
    small_ppat(batch).sum().backward()
    missing = [n for n, p in small_ppat.named_parameters() if p.grad is None]
    assert missing == []


def test_relative_positional_encoding(batch):
    torch.manual_seed(0)
    model = (
        PointPatchTransformer(
            dim=32,
            depth=1,
            heads=2,
            mlp_dim=64,
            sa_dim=16,
            patches=8,
            prad=0.4,
            nsamp=16,
            in_dim=6,
            dim_head=8,
            rel_pe=True,
        )
        .to(DEVICE)
        .eval()
    )
    with torch.no_grad():
        assert model(batch).shape == (batch.shape[0], 32)


# ---------------------------------------------------------------------------
# Published OpenShape checkpoints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", sorted(OPENSHAPE_VARIANTS))
def test_published_state_dict_layout(variant):
    """Module tree must match the reference so published weights load unmapped.

    Guards the non-obvious parts: the ``_Permute`` placeholder that keeps
    ``lift``'s LayerNorm at index 2, and the PreNorm(``norm``/``fn``) nesting.
    """
    model = build_openshape_pointbert(variant)
    keys = set(model.state_dict())
    spec = OPENSHAPE_VARIANTS[variant]
    assert spec["up_axis"] == ("z" if variant == "openshape-pointbert-vitg14-rgb" else "y")
    prefix = "" if spec["out_dim"] is None else "ppat."

    for suffix in [
        "cls_token",
        "sa.mlp_convs.0.weight",
        "sa.mlp_bns.2.running_mean",
        "lift.0.weight",
        "lift.2.weight",  # LayerNorm sits at index 2, after the permute placeholder
        "transformer.layers.0.0.norm.weight",
        "transformer.layers.0.0.fn.to_qkv.weight",
        "transformer.layers.0.0.fn.to_out.0.bias",
        "transformer.layers.11.1.fn.net.3.weight",
    ]:
        assert prefix + suffix in keys, suffix
    assert not any(k.endswith(".pe.0.weight") for k in keys), "rel_pe must stay off"

    if spec["out_dim"] is None:
        assert isinstance(model, PointPatchTransformer)
        assert "proj.weight" not in keys
    else:
        assert isinstance(model, ProjectedPointPatchTransformer)
        assert model.proj.out_features == spec["out_dim"]


@pytest.mark.parametrize(
    "wrap",
    [
        pytest.param(lambda sd: sd, id="plain"),
        pytest.param(
            lambda sd: {"state_dict": {f"module.{k}": v for k, v in sd.items()}},
            id="nested-module",
        ),  # vitg14 layout
        pytest.param(
            lambda sd: {**{f"pc_encoder.{k}": v for k, v in sd.items()}, "tau": torch.tensor(1.0)},
            id="flat-pc_encoder-plus-extras",
        ),  # vitb32 / vitl14 layout
        pytest.param(
            lambda sd: {"model": sd, "optimizer": {"junk": 1}, "epoch": 3},
            id="training-checkpoint",
        ),
    ],
)
def test_checkpoint_layout_autodetect(tmp_path, wrap):
    """Every checkpoint layout in play must load without hand-written key mapping."""
    variant = "openshape-pointbert-vitb32-rgb"
    reference = build_openshape_pointbert(variant)
    path = tmp_path / "ckpt.pt"
    torch.save(wrap(reference.state_dict()), path)

    loaded = load_openshape_pointbert(variant, checkpoint_path=str(path))
    for (name, a), (_, b) in zip(loaded.state_dict().items(), reference.state_dict().items()):
        assert torch.equal(a, b), name


def test_checkpoint_layout_rejects_mismatch(tmp_path):
    """A checkpoint for a different architecture must fail loudly, not partially load."""
    path = tmp_path / "ckpt.pt"
    torch.save(build_openshape_pointbert("openshape-pointbert-vitb32-rgb").state_dict(), path)
    with pytest.raises(KeyError, match="matches the model"):
        load_openshape_pointbert("openshape-pointbert-vitg14-rgb", checkpoint_path=str(path))


@pytest.mark.slow
@pytest.mark.parametrize("variant", sorted(OPENSHAPE_VARIANTS))
def test_load_published_checkpoint(variant):
    """Download the released weights and load them strictly, then encode a shape."""
    huggingface_hub = pytest.importorskip("huggingface_hub")
    try:
        checkpoint_path = huggingface_hub.hf_hub_download(f"OpenShape/{variant}", "model.pt")
    except Exception as exc:  # offline, or the Hub is unreachable
        pytest.skip(f"could not fetch {variant}: {exc}")

    # Keep download failures skippable, but let checkpoint-layout and strict-load
    # regressions fail this test instead of being mislabeled as an offline machine.
    model = load_openshape_pointbert(variant, checkpoint_path=checkpoint_path, device=DEVICE)

    assert not model.training
    torch.manual_seed(0)
    xyz = normalize_point_cloud(torch.rand(1, 10000, 3) * 2 - 1)
    feats = torch.cat([xyz, torch.rand(1, 10000, 3)], dim=-1)
    with torch.no_grad():
        out = model(feats.permute(0, 2, 1).contiguous().to(DEVICE))

    spec = OPENSHAPE_VARIANTS[variant]
    assert out.shape == (1, spec["out_dim"] or spec["ppat"]["dim"])
    assert torch.isfinite(out).all()
