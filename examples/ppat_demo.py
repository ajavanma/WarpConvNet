# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run OpenShape PPAT on a mesh and compare it with familiar text labels.

The default example is deliberately small to download: OpenShape's PPAT weights
are compared with the prompt-averaged OpenCLIP ViT-bigG-14 text features that
OpenShape publishes for the LVIS category names.  Loading a multi-gigabyte CLIP
image tower just to encode a handful of labels is therefore unnecessary.

Install the optional dependencies and run from the repository root::

    pip install -e '.[demo]'
    python examples/ppat_demo.py

The bundled chair and the ``openshape-pointbert-vitg14-rgb`` checkpoint are
z-up.  For a y-up mesh, pass ``--up-axis y`` so the demo rotates it to z-up
before normalization.
"""

from __future__ import annotations

import argparse
import ast
import gc
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


PPAT_VARIANT = "openshape-pointbert-vitg14-rgb"
TEXT_FEATURE_REPO = "OpenShape/openshape-demo-support"
TEXT_FEATURE_PATH = "openshape/demo/lvis_cats.pt"
TEXT_CATEGORY_PATH = "openshape/demo/lvis.py"
DEFAULT_MESH = Path(__file__).resolve().parent / "assets" / "ppat_demo_chair.obj"
DEFAULT_LABELS = (
    "chair",
    "table",
    "sofa",
    "airplane",
    "car_(automobile)",
    "guitar",
    "lamp",
)


class DemoError(RuntimeError):
    """An expected, user-actionable demo failure."""


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise DemoError(f"Invalid device {requested!r}.") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise DemoError("CUDA was requested, but this PyTorch installation cannot use CUDA.")
    if device.type == "cuda" and device.index is not None:
        if device.index < 0 or device.index >= torch.cuda.device_count():
            raise DemoError(
                f"CUDA device {device.index} does not exist; found {torch.cuda.device_count()} "
                "CUDA device(s)."
            )
    if device.type not in {"cpu", "cuda"}:
        raise DemoError("This demo currently supports CPU and CUDA devices.")
    return device


def _as_triangle_mesh(mesh_path: Path):
    try:
        import trimesh
    except ImportError as exc:
        raise DemoError(
            "Mesh loading needs trimesh. Install the demo dependencies with "
            "`pip install -e '.[demo]'`."
        ) from exc

    if not mesh_path.is_file():
        raise DemoError(f"Mesh does not exist: {mesh_path}")
    try:
        loaded = trimesh.load(mesh_path, force="scene", process=False)
        if isinstance(loaded, trimesh.Scene):
            if not loaded.geometry:
                raise DemoError(f"Mesh scene contains no geometry: {mesh_path}")
            # ``to_geometry`` applies scene transforms before concatenating.
            # ``dump(concatenate=True)`` is the equivalent API in older trimesh.
            if hasattr(loaded, "to_geometry"):
                loaded = loaded.to_geometry()
            else:  # pragma: no cover - compatibility with older trimesh
                loaded = loaded.dump(concatenate=True)
    except DemoError:
        raise
    except Exception as exc:
        raise DemoError(f"Could not load mesh {mesh_path}: {exc}") from exc

    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise DemoError(f"File does not contain a non-empty triangle mesh: {mesh_path}")
    return loaded


def _rgb_for_samples(mesh, face_index: np.ndarray, barycentric: np.ndarray) -> np.ndarray:
    """Sample mesh colors, or use OpenShape's neutral 0.4 fill when none exist."""
    visual = mesh.visual
    if getattr(visual, "kind", None) == "texture":
        try:
            visual = visual.to_color()
        except Exception:
            visual = mesh.visual

    rgb: np.ndarray | None = None
    if getattr(visual, "kind", None) == "vertex":
        vertex_colors = np.asarray(visual.vertex_colors)
        if vertex_colors.ndim == 2 and len(vertex_colors) == len(mesh.vertices):
            triangle_colors = vertex_colors[np.asarray(mesh.faces)[face_index], :3]
            rgb = (triangle_colors * barycentric[..., None]).sum(axis=1)
    elif getattr(visual, "kind", None) == "face":
        face_colors = np.asarray(visual.face_colors)
        if face_colors.ndim == 2 and len(face_colors) == len(mesh.faces):
            rgb = face_colors[face_index, :3]

    if rgb is None:
        return np.full((len(face_index), 3), 0.4, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.size and float(rgb.max()) > 1.0:
        rgb /= 255.0
    return np.clip(rgb, 0.0, 1.0)


def sample_mesh_surface(
    mesh_path: Path, num_points: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Area-sample deterministic XYZ and RGB values from a triangle mesh."""
    if num_points < 1:
        raise DemoError("--num-points must be positive.")
    if seed < 0:
        raise DemoError("--seed must be non-negative.")
    mesh = _as_triangle_mesh(mesh_path)
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    if not np.isfinite(triangles).all():
        raise DemoError(f"Mesh has non-finite vertex coordinates: {mesh_path}")

    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = np.linalg.norm(cross, axis=1) * 0.5
    total_area = float(area.sum())
    if not np.isfinite(total_area) or total_area <= 0.0:
        raise DemoError(f"Mesh has no positive-area faces: {mesh_path}")

    generator = np.random.default_rng(seed)
    face_index = generator.choice(len(triangles), size=num_points, p=area / total_area)
    # sqrt(u) makes the samples uniform over each selected triangle.
    uv = generator.random((num_points, 2))
    root_u = np.sqrt(uv[:, :1])
    barycentric = np.concatenate(
        [1.0 - root_u, root_u * (1.0 - uv[:, 1:]), root_u * uv[:, 1:]], axis=1
    )
    xyz = (triangles[face_index] * barycentric[..., None]).sum(axis=1).astype(np.float32)
    rgb = _rgb_for_samples(mesh, face_index, barycentric)
    return xyz, rgb


def align_to_z_up(xyz: np.ndarray, up_axis: str) -> np.ndarray:
    """Rotate x-up or y-up points to the checkpoint's z-up convention."""
    xyz = np.asarray(xyz, dtype=np.float32)
    if up_axis == "z":
        return xyz.copy()
    aligned = np.empty_like(xyz)
    if up_axis == "y":
        # +90 degrees about x: (x, y, z) -> (x, -z, y).
        aligned[:, 0] = xyz[:, 0]
        aligned[:, 1] = -xyz[:, 2]
        aligned[:, 2] = xyz[:, 1]
        return aligned
    if up_axis == "x":
        # -90 degrees about y: (x, y, z) -> (-z, y, x).
        aligned[:, 0] = -xyz[:, 2]
        aligned[:, 1] = xyz[:, 1]
        aligned[:, 2] = xyz[:, 0]
        return aligned
    raise DemoError(f"Unknown up axis {up_axis!r}; choose x, y, or z.")


def mesh_to_ppat_input(
    mesh_path: Path,
    num_points: int = 10_000,
    seed: int = 0,
    up_axis: str = "z",
    checkpoint_up_axis: str = "z",
) -> Tensor:
    """Load a mesh as normalized, channels-first ``(1, 6, N)`` XYZ+RGB."""
    from warpconvnet.models.ppat import normalize_point_cloud

    if checkpoint_up_axis != "z":
        raise DemoError(
            "This demo's axis conversion currently targets z-up, but the selected checkpoint "
            f"declares {checkpoint_up_axis}-up."
        )
    xyz, rgb = sample_mesh_surface(mesh_path, num_points, seed)
    xyz_tensor = torch.from_numpy(align_to_z_up(xyz, up_axis))
    xyz_tensor = normalize_point_cloud(xyz_tensor)
    features = torch.cat([xyz_tensor, torch.from_numpy(rgb)], dim=-1)
    return features.transpose(0, 1).unsqueeze(0).contiguous()


def parse_category_file(path: Path) -> list[str]:
    """Read only the literal ``categories`` assignment from OpenShape's Python file.

    The downloaded file is parsed as data.  It is never imported or executed.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise DemoError(f"Could not parse OpenShape category file {path}: {exc}") from exc

    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == "categories" for target in targets
        ):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError) as exc:
            raise DemoError("OpenShape's `categories` value is not a literal list.") from exc
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise DemoError("OpenShape's `categories` value is not a list of strings.")
        return value
    raise DemoError("OpenShape category file has no literal `categories` assignment.")


def load_openshape_text_features() -> tuple[list[str], Tensor]:
    """Download OpenShape's prompt-averaged ViT-bigG LVIS text features."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise DemoError(
            "Model downloads need huggingface-hub. Install the demo dependencies with "
            "`pip install -e '.[demo]'`."
        ) from exc

    try:
        feature_path = Path(hf_hub_download(TEXT_FEATURE_REPO, TEXT_FEATURE_PATH))
        category_path = Path(hf_hub_download(TEXT_FEATURE_REPO, TEXT_CATEGORY_PATH))
        features = torch.load(feature_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise DemoError(f"Could not download OpenShape's CLIP text features: {exc}") from exc

    categories = parse_category_file(category_path)
    if not isinstance(features, Tensor) or features.ndim != 2:
        raise DemoError("OpenShape text feature file does not contain one 2-D tensor.")
    if features.shape != (len(categories), 1280):
        raise DemoError(
            "OpenShape category and text feature files disagree: "
            f"{len(categories)} names versus tensor shape {tuple(features.shape)}."
        )
    if not torch.isfinite(features).all():
        raise DemoError("OpenShape text features contain non-finite values.")
    return categories, features.float()


def select_label_features(
    requested: Sequence[str], categories: Sequence[str], features: Tensor
) -> tuple[list[str], Tensor]:
    """Select a short, ordered candidate list from the published LVIS features."""
    index = {name: position for position, name in enumerate(categories)}
    missing = [name for name in requested if name not in index]
    if missing:
        examples = ", ".join(DEFAULT_LABELS[:4])
        raise DemoError(
            "These labels are not exact OpenShape LVIS category names: "
            f"{', '.join(missing)}. Examples: {examples}."
        )
    indices = torch.tensor([index[name] for name in requested], dtype=torch.long)
    return list(requested), features.index_select(0, indices)


def rank_cosine_similarities(
    shape_feature: Tensor, labels: Sequence[str], text_features: Tensor
) -> list[tuple[str, float]]:
    """Return labels from largest to smallest shape-text cosine similarity."""
    if shape_feature.ndim == 2 and shape_feature.shape[0] == 1:
        shape_feature = shape_feature[0]
    if shape_feature.ndim != 1:
        raise DemoError(f"Expected one shape embedding, got {tuple(shape_feature.shape)}.")
    if text_features.ndim != 2 or text_features.shape[0] != len(labels):
        raise DemoError("Text feature rows must match the label list.")
    if shape_feature.shape[0] != text_features.shape[1]:
        raise DemoError(
            f"Shape embedding width {shape_feature.shape[0]} does not match text width "
            f"{text_features.shape[1]}."
        )
    shape_feature = F.normalize(shape_feature.float(), dim=0)
    text_features = F.normalize(text_features.float(), dim=1)
    scores = text_features @ shape_feature
    order = scores.argsort(descending=True).tolist()
    return [(labels[i], float(scores[i])) for i in order]


def display_label(label: str) -> str:
    """Turn an LVIS identifier into a friendlier terminal label."""
    if label == "car_(automobile)":
        return "car"
    return label.replace("_", " ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify a mesh by comparing PPAT and OpenCLIP text embeddings.",
        epilog=(
            "The first run downloads about 394 MB: PPAT weights plus OpenShape's small "
            "prompt-averaged text-feature bundle. CUDA is recommended; CPU inference works "
            "but is much slower."
        ),
    )
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH, help="OBJ, PLY, or GLB mesh")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=list(DEFAULT_LABELS),
        metavar="LVIS_LABEL",
        help="exact OpenShape LVIS category names to compare",
    )
    parser.add_argument(
        "--num-points", type=int, default=10_000, help="surface samples (default: 10000)"
    )
    parser.add_argument("--seed", type=int, default=0, help="mesh-sampling seed (default: 0)")
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or a CUDA device such as cuda:1"
    )
    parser.add_argument(
        "--up-axis",
        choices=("x", "y", "z"),
        default="z",
        help="mesh gravity axis before rotation to the checkpoint's z-up convention",
    )
    return parser


def run(args: argparse.Namespace) -> list[tuple[str, float]]:
    if not args.labels:
        raise DemoError("Provide at least one --labels value.")
    from warpconvnet.models.ppat import OPENSHAPE_VARIANTS

    variant_spec = OPENSHAPE_VARIANTS[PPAT_VARIANT]
    checkpoint_up_axis = variant_spec["up_axis"]
    minimum_points = variant_spec["ppat"]["patches"]
    if args.num_points < minimum_points:
        raise DemoError(
            f"--num-points must be at least {minimum_points} for the ViT-bigG PPAT checkpoint."
        )
    device = _resolve_device(args.device)
    print(f"Mesh: {args.mesh}")
    print(
        f"Sampling: {args.num_points:,} points, seed={args.seed}, input {args.up_axis}-up "
        f"-> checkpoint {checkpoint_up_axis}-up"
    )
    print(f"Device: {device}")
    if device.type == "cpu":
        print("Note: CUDA is strongly recommended; CPU inference can take several minutes.")

    points = mesh_to_ppat_input(
        args.mesh, args.num_points, args.seed, args.up_axis, checkpoint_up_axis
    ).to(device)
    print(f"Loading {PPAT_VARIANT} from Hugging Face ...")
    try:
        from warpconvnet.models.ppat import load_openshape_pointbert

        # The portable FPS path is deterministic and also works when an installed
        # WarpConvNet CUDA extension was built for a different GPU architecture.
        model = load_openshape_pointbert(
            "openshape-pointbert-vitg14-rgb", device=device, fps_backend="torch"
        )
    except Exception as exc:
        raise DemoError(f"Could not load the published PPAT checkpoint: {exc}") from exc

    print("Encoding the mesh ...")
    try:
        with torch.inference_mode():
            shape_feature = model(points)
    except Exception as exc:
        raise DemoError(f"PPAT inference failed: {exc}") from exc
    shape_feature = shape_feature.float().cpu()
    del model, points
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("Loading OpenShape's prompt-averaged OpenCLIP ViT-bigG-14 text features ...")
    categories, all_text_features = load_openshape_text_features()
    labels, text_features = select_label_features(args.labels, categories, all_text_features)
    ranked = rank_cosine_similarities(shape_feature, labels, text_features)

    print("\nRanked labels (cosine similarity; higher is a closer text match):")
    width = max(len(display_label(label)) for label in labels)
    for position, (label, score) in enumerate(ranked, start=1):
        print(f"  {position:>2}. {display_label(label):<{width}}  {score:+.4f}")
    print(f"\nTop prediction: {display_label(ranked[0][0])}")
    return ranked


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DemoError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
