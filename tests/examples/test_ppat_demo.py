# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
import torch

from examples import ppat_demo


def test_default_asset_is_deterministic_normalized_xyz_rgb():
    pytest.importorskip("trimesh")
    first = ppat_demo.mesh_to_ppat_input(ppat_demo.DEFAULT_MESH, num_points=512, seed=7)
    second = ppat_demo.mesh_to_ppat_input(ppat_demo.DEFAULT_MESH, num_points=512, seed=7)

    assert torch.equal(first, second)
    assert first.shape == (1, 6, 512)
    xyz = first[0, :3].transpose(0, 1)
    rgb = first[0, 3:].transpose(0, 1)
    assert torch.allclose(xyz.mean(0), torch.zeros(3), atol=1e-6)
    assert torch.allclose(xyz.norm(dim=1).max(), torch.tensor(1.0), atol=1e-6)
    assert torch.all((0.0 <= rgb) & (rgb <= 1.0))


def test_axis_alignment_maps_gravity_to_positive_z():
    xyz = np.array([[2.0, 3.0, 5.0]], dtype=np.float32)
    assert np.array_equal(ppat_demo.align_to_z_up(xyz, "z"), xyz)
    assert np.array_equal(
        ppat_demo.align_to_z_up(xyz, "y"), np.array([[2.0, -5.0, 3.0]], dtype=np.float32)
    )
    assert np.array_equal(
        ppat_demo.align_to_z_up(xyz, "x"), np.array([[-5.0, 3.0, 2.0]], dtype=np.float32)
    )


def test_mesh_sampling_rejects_negative_seed():
    with pytest.raises(ppat_demo.DemoError, match="--seed must be non-negative"):
        ppat_demo.sample_mesh_surface(ppat_demo.DEFAULT_MESH, num_points=8, seed=-1)


def test_category_parser_reads_literal_without_executing_file(tmp_path: Path):
    sentinel = tmp_path / "must_not_exist"
    category_file = tmp_path / "lvis.py"
    category_file.write_text(
        "import pathlib\n"
        f"pathlib.Path({str(sentinel)!r}).touch()\n"
        "categories = ['chair', 'table']\n",
        encoding="utf-8",
    )

    assert ppat_demo.parse_category_file(category_file) == ["chair", "table"]
    assert not sentinel.exists()


def test_category_parser_rejects_executable_value(tmp_path: Path):
    category_file = tmp_path / "lvis.py"
    category_file.write_text("categories = list(('chair',))\n", encoding="utf-8")
    with pytest.raises(ppat_demo.DemoError, match="not a literal list"):
        ppat_demo.parse_category_file(category_file)


def test_select_and_rank_cosine_similarities():
    categories = ["table", "chair", "lamp"]
    all_features = torch.tensor([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]])
    labels, features = ppat_demo.select_label_features(
        ["chair", "table", "lamp"], categories, all_features
    )
    ranked = ppat_demo.rank_cosine_similarities(torch.tensor([[2.0, 0.0]]), labels, features)

    assert [label for label, _ in ranked] == ["chair", "table", "lamp"]
    assert [score for _, score in ranked] == pytest.approx([1.0, 0.0, -1.0])


def test_cli_defaults_describe_released_checkpoint_contract():
    from warpconvnet.models.ppat import OPENSHAPE_VARIANTS

    args = ppat_demo.build_parser().parse_args([])
    assert args.mesh == ppat_demo.DEFAULT_MESH
    assert args.num_points == 10_000
    assert args.up_axis == "z"
    assert args.device == "auto"
    assert args.labels[0] == "chair"
    assert OPENSHAPE_VARIANTS[ppat_demo.PPAT_VARIANT]["up_axis"] == args.up_axis
