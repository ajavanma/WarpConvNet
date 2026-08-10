# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Autotune candidates for the Blackwell (sm_100) deep-pipe forward tiles.

Owned by the CUTLASS/GEMM workstream. ``algo_params.py`` consumes this with a
two-line hook::

    from .algo_params_sm100 import _AB_MASK_SM100
    ...
    params.extend(_AB_MASK_SM100)

The tiles are ``MaskGemm_forward_sm100_deep_pipe`` (hand-written, see
``csrc/mask_gemm/include/MaskGemm_forward_sm100_deep_pipe.h``). They differ from
the incumbent forward tiles in exactly one way: the (kernel offset, k-tile)
product is flattened into ONE cp.async pipeline, so the pipeline fills and
drains once per output tile instead of once per active kernel offset, and the
submanifold identity offset rides that pipeline instead of a fully serialized
``cp_async_wait<0>``-per-k-tile shortcut.

They are compiled ONLY into builds carrying an accelerated ``10.0a`` target
(``compile_archs=(100,)`` in ``csrc/mask_gemm/tile_metadata.py``), because the
useful stage counts need B200's 232,448 B opt-in shared-memory budget; the
python arch gate in ``detail/tile_metadata.py`` filters them out everywhere
else with no change needed there.

Constraints that the pool below encodes:
  * MaskWords == 1 only (K <= 32). The binding raises for mask_words > 1.
  * fp16->fp16 and bf16->bf16 only (no f32 output instantiation).
  * 64x128 tiles want C_out >= 128; 64x64 tiles want C_out >= 64. That is
    already enforced by ``TileMetadata.handles_c_out``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

_HAS_MASK_GEMM = False
try:  # pragma: no cover - import probe, mirrors algo_params.py
    import warpconvnet._C as _test_C

    _HAS_MASK_GEMM = hasattr(_test_C, "mask_gemm")
except ImportError:  # pragma: no cover
    pass


# tile_id -> (tile_m, tile_n, num_stages, smem_bytes_fp16)
SM100_DEEP_TILES: Dict[int, Tuple[int, int, int, int]] = {
    1000: (64, 64, 6, 49_408),
    1001: (64, 128, 6, 74_112),
    1002: (64, 128, 4, 49_408),
    1003: (64, 128, 8, 98_816),
    1004: (64, 64, 10, 82_176),
    1005: (64, 128, 6, 74_112),  # 256 threads (2x4 warps)
    1006: (64, 128, 4, 49_680),  # minBlocks=3 -> 160 regs, 3 CTAs/SM
    1007: (64, 64, 4, 33_104),   # minBlocks=4 -> 124 regs, 4 CTAs/SM
    1008: (64, 64, 4, 33_104),   # minBlocks=3 -> 150 regs, 3 CTAs/SM
    1009: (64, 128, 3, 37_392),  # minBlocks=3 -> 159 regs, 3 CTAs/SM
}


def _mk(tile_ids) -> List[Tuple[str, Dict[str, Any]]]:
    if not _HAS_MASK_GEMM:
        return []
    return [("mask_gemm", {"tile_id": t}) for t in tile_ids]


# Wide-channel candidates (C_out >= 128), ordered by the measured B200 sweep
# (surface, N=500k, C=256, fp16, kernel ms): 1009 5.2174 < 1006 5.2794 <
# 1002 6.1004 < 1001 6.1193 < incumbent tile 3 6.8998. Achieved blocks/SM is
# the ordering variable (3 for 1009/1006, 2 for 1002/1001), not pipeline depth
# -- 1009 has the SHALLOWEST pipeline of the four. 1003 (8 stages, 2 CTAs/SM)
# and 1005 (8 warps, 1 CTA/SM) are compiled but deliberately NOT pooled: they
# lost in every measured cell.
_AB_MASK_SM100_WIDE = _mk((1009, 1006, 1002, 1001))

# Narrow-channel candidates (64 <= C_out < 128). Same sweep, C=64:
# 1007 0.6833 < 1008 0.8095 < 1000 0.8225 < incumbent tile 41 1.0570 ms,
# i.e. blocks/SM 4 > 3 > 3. 1004 (10 stages, 2 CTAs/SM) is compiled but not
# pooled: it lost to 1000 (6 stages, 3 CTAs/SM) in every cell.
_AB_MASK_SM100_NARROW = _mk((1007, 1008, 1000))

# Everything, for the generic hook.
_AB_MASK_SM100 = _AB_MASK_SM100_WIDE + _AB_MASK_SM100_NARROW

__all__ = [
    "SM100_DEEP_TILES",
    "_AB_MASK_SM100",
    "_AB_MASK_SM100_NARROW",
    "_AB_MASK_SM100_WIDE",
]
