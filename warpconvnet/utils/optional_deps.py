# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Accessors for optional third-party dependencies.

``torch_scatter`` is declared an OPTIONAL extra
(``pyproject.toml`` ``[project.optional-dependencies]``: ``torch-scatter``) and is
documented as a separate install step in the README and
``docs/getting_started/installation.md``. It is deliberately kept out of the hard
dependencies: it compiles against an exact torch build, so making it mandatory
would slow every release build and export a fragile source compile to every user.

Four modules nonetheless imported it at module scope, and all four are reachable
from ``import warpconvnet`` (``__init__`` -> ``_compile`` -> ...), which made a
declared-optional package mandatory in practice: a clean environment following the
README's *Install from PyPI* path got something that raised ``ModuleNotFoundError``
on import. It never showed up in development because ``torch_scatter`` is present
on every machine here.

The shim below keeps those call sites byte-identical. When ``torch_scatter`` is
installed, ``segment_csr`` IS the real function — no wrapper, no overhead on a hot
reduction path. When it is absent, it is a placeholder that raises at CALL time
with the install command, so the package imports fine and only the features that
actually reduce are unavailable.

This mirrors how the sibling optional extra is handled — see
``nn/functional/flash_attn_utils.py`` (``try: import flash_attn / except
ImportError: flash_attn = None``) — and how ``nn/functional/bilateral.py`` already
imports ``segment_csr`` lazily inside the function that uses it.
"""

from typing import Any

__all__ = ["segment_csr", "HAS_TORCH_SCATTER", "require_torch_scatter"]

_INSTALL_HINT = (
    "torch_scatter is an optional WarpConvNet dependency and is not installed.\n"
    "Install it with:\n"
    "    pip install git+https://github.com/rusty1s/pytorch_scatter.git\n"
    "See the installation section of the README. It is kept optional because it "
    "compiles against your exact torch build, which would otherwise slow every "
    "install and break on a torch mismatch."
)

try:
    from torch_scatter import segment_csr  # noqa: F401

    HAS_TORCH_SCATTER = True
except ImportError:  # pragma: no cover - exercised only without the optional extra
    HAS_TORCH_SCATTER = False

    def segment_csr(*args: Any, **kwargs: Any) -> Any:
        """Placeholder for ``torch_scatter.segment_csr``; raises when called.

        Deliberately raises ``ImportError`` rather than returning something
        wrong-but-plausible: a silently degraded reduction would surface as a
        numerical discrepancy far from its cause.
        """
        raise ImportError(_INSTALL_HINT)


def require_torch_scatter() -> None:
    """Raise if the optional extra is missing.

    For call sites that want to fail before doing expensive setup, or that route
    to ``segment_csr`` only on some branches (e.g. a CUDA path) and would
    otherwise appear to work on CPU and fail later.
    """
    if not HAS_TORCH_SCATTER:
        raise ImportError(_INSTALL_HINT)
