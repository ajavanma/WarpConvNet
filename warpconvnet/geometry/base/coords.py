# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import torch
from torch import Tensor
from .batched import BatchedTensor


from warpconvnet.geometry.coords.search.search_results import RealSearchResult
from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates

# Debug switch: verify every handed-off batch-indexed coordinate tensor against
# a fresh recomputation. Off by default (it costs a full rebuild + a compare).
_CHECK_BIC_HANDOFF = os.environ.get("WARPCONVNET_CHECK_BIC_HANDOFF", "0") == "1"


class Coords(BatchedTensor):
    """Base class for coordinates."""

    @property
    def num_spatial_dims(self):
        return self.batched_tensor.shape[1]  # tensor does not have batch index

    def neighbors(
        self,
        query_coords: "Coords",
        search_args: dict,
    ) -> "RealSearchResult":
        """
        Find the neighbors of the query_coords in the current coordinates.

        Args:
            query_coords: The coordinates to search for neighbors
            search_args: Arguments for the search
        """
        raise NotImplementedError

    @property
    def batch_indexed_coordinates(self) -> Tensor:
        """The [N, D+1] (batch, *spatial) form of these coordinates, memoized.

        Stock rebuilt this tensor on *every* access. A sparse convolution reads
        it at least once per layer, so a 56-layer PTv3 backbone paid 56 rebuilds
        of an identical tensor; measured at 0.178 ms each (B=1, N=14000), i.e.
        35% of a 0.512 ms ``mask_gemm`` conv call.

        The memo is keyed on the *identity* of ``batched_tensor`` and
        ``offsets``, so any rebind (``_to``, ``GridCoords._ensure_initialized``,
        a new instance) misses and recomputes. Coordinate tensors are treated as
        immutable throughout the codebase: the two in-place ``bic[:, 1:] = ...``
        sites (``Voxels.from_dense``/``to_dense``) both operate on a ``.clone()``.
        """
        ct = self.batched_tensor
        offsets = self.offsets
        memo = self.__dict__.get("_bic_memo")
        if memo is not None and memo[0] is ct and memo[1] is offsets:
            return memo[2]
        value = batch_indexed_coordinates(ct, offsets)
        self.__dict__["_bic_memo"] = (ct, offsets, value)
        return value

    def _set_batch_indexed_coordinates(self, batch_indexed: Tensor) -> None:
        """Register an already-materialised batch-indexed form of these coords.

        Callers that *constructed* this object out of an ``[N, D+1]`` tensor they
        already hold (``spatially_sparse_conv``, ``sparse_reduce``) can hand it
        over instead of having the next layer rebuild it. Purely an optimization:
        the value must equal what the property would compute. Mismatched shape /
        dtype / device is ignored rather than trusted.
        """
        ct = self.batched_tensor
        if (
            batch_indexed.dim() == 2
            and batch_indexed.shape[0] == ct.shape[0]
            and batch_indexed.shape[1] == ct.shape[1] + 1
            and batch_indexed.dtype == ct.dtype
            and batch_indexed.device == ct.device
        ):
            if _CHECK_BIC_HANDOFF:
                expected = batch_indexed_coordinates(ct, self.offsets)
                assert torch.equal(
                    expected, batch_indexed
                ), "batch-indexed coordinate handoff does not match the recomputed value"
            self.__dict__["_bic_memo"] = (ct, self.offsets, batch_indexed)
