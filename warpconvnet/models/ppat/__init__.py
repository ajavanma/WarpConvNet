# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PPAT -- the Point Patch Transformer shape encoder.

What OpenShape's configs call ``PointBERT`` is a Point Patch Transformer, not Point-BERT: one
PointNet++ set-abstraction layer turns ``N`` points into patch tokens, a pre-norm ViT mixes them
with a class token, and the class token is projected into CLIP space.

See ``README.md`` in this directory for measured performance, the neighbourhood-rule options and
what each of them costs.

The underscore-prefixed names re-exported below are imported by sibling encoders and by the test
suite. They are re-exported so every caller uses one import path, but they remain private API and
may change. The list was derived by parsing every ``ImportFrom`` of this package across the repo,
not by hand -- a hand-written list missed four of them and broke collection.
"""

from warpconvnet.models.ppat.model import (
    APPROX_NEIGHBORHOOD_RULES,
    EXACT_NEIGHBORHOOD_RULES,
    OPENSHAPE_VARIANTS,
    Attention,
    FeedForward,
    FPSBackend,
    PointNetSetAbstraction,
    PointPatchTransformer,
    PreNorm,
    ProjectedPointPatchTransformer,
    Transformer,
    _ball_mask,
    _match_state_dict,
    _query_ball_point_cumsum,
    _select_first_k,
    _square_distance,
    _to_channels_first,
    build_openshape_pointbert,
    farthest_point_sample,
    index_points,
    load_openshape_pointbert,
    normalize_point_cloud,
    parse_neighborhood_rule,
    query_ball_point,
)

__all__ = [
    "APPROX_NEIGHBORHOOD_RULES",
    "Attention",
    "EXACT_NEIGHBORHOOD_RULES",
    "FPSBackend",
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
