# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download and strictly load official Uni3D checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path
import torch

from warpconvnet.models.uni3d.model import (
    FPSBackend,
    NeighborBackend,
    NeighborSearch,
    Uni3D,
    build_uni3d,
    resolve_uni3d_variant,
)

__all__ = ["download_uni3d_checkpoint", "load_uni3d"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_uni3d_checkpoint(variant: str = "uni3d-s", *, verify: bool = False) -> Path:
    """Download a checkpoint from the pinned official Hugging Face revision."""
    config = resolve_uni3d_variant(variant)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "Downloading Uni3D weights requires huggingface-hub; install WarpConvNet's demo extra."
        ) from exc
    path = Path(
        hf_hub_download(
            repo_id=config.repo_id,
            filename=config.filename,
            revision=config.revision,
        )
    )
    actual_size = path.stat().st_size
    if actual_size != config.size:
        raise RuntimeError(
            f"Uni3D checkpoint has {actual_size} bytes; expected {config.size} for {config.name}"
        )
    if verify:
        actual_hash = _sha256(path)
        if actual_hash != config.sha256:
            raise RuntimeError(
                f"Uni3D checkpoint SHA-256 is {actual_hash}; expected {config.sha256}"
            )
    return path


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Uni3D checkpoint must be a dictionary")
    state = checkpoint.get("module", checkpoint)
    if not isinstance(state, dict) or not state:
        raise RuntimeError("Uni3D checkpoint has no non-empty 'module' state dictionary")
    if all(isinstance(key, str) and key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise RuntimeError("Uni3D state dictionary must contain only string-to-tensor entries")
    return state


def load_uni3d(
    variant: str = "uni3d-s",
    *,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = torch.float32,
    fps_backend: FPSBackend = "auto",
    neighbor_search: NeighborBackend | NeighborSearch = "knn",
    verify_download: bool = False,
) -> Uni3D:
    """Strictly load released Uni3D weights without timm or pointnet2_ops.

    The architecture is first created on the meta device, so loading giant does
    not allocate an extra randomly initialized 4 GB model.  Official checkpoints
    store half-precision tensors; ``dtype=torch.float32`` matches the release
    inference path, while ``dtype=None`` preserves checkpoint precision.
    """
    config = resolve_uni3d_variant(variant)
    target_device = torch.device("cpu" if device is None else device)
    if dtype is not None and not torch.is_floating_point(torch.empty((), dtype=dtype)):
        raise TypeError(f"dtype must be floating point or None; got {dtype}")
    path = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else download_uni3d_checkpoint(config.name, verify=verify_download)
    )
    if not path.is_file():
        raise FileNotFoundError(f"Uni3D checkpoint does not exist: {path}")
    state = _checkpoint_state(path)
    with torch.device("meta"):
        model = build_uni3d(
            config.name,
            fps_backend=fps_backend,
            neighbor_search=neighbor_search,
        )
    try:
        model.load_state_dict(state, strict=True, assign=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint {path} is not strictly compatible with {config.name}: {exc}"
        ) from exc
    model.materialize_nonpersistent_buffers()
    if dtype is None:
        model = model.to(device=target_device)
    else:
        model = model.to(device=target_device, dtype=dtype)
    model.eval()
    model.uni3d_checkpoint = str(path)
    return model
