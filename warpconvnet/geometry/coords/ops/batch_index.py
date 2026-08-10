# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, Optional, Union
from jaxtyping import Float, Int

import numpy as np
import math
import os

import torch
from torch import Tensor

import warpconvnet._C as _C


@torch.inference_mode()
def batch_index_from_offset(
    offsets: Int[Tensor, "B+1"],
    device: Optional[Union[str, torch.device]] = None,
) -> Int[Tensor, "N"]:  # type: ignore
    """
    Generates batch indices for a contiguous range of elements defined by offsets.
    `offsets` has B+1 elements, defining B batches.
    Output has N = offsets[B] elements.

    Args:
        offsets: B+1 boundary offsets.
        device: Device to build the result on. Defaults to ``offsets.device``
            (usually the CPU, per the ``BatchedTensor`` contract). Callers that
            immediately move the result to the GPU should pass the target device
            instead: the CPU ``repeat_interleave`` opens an OpenMP parallel
            region for every ``B > 1`` (``native/Repeat.cpp::compute_cpu`` ->
            ``at::parallel_for(..., grain_size=1)`` over B), which in a
            thread-oversubscribed container costs tens of milliseconds of pure
            host stall. See ``batch_indexed_coordinates`` for the measurements.
    """
    assert len(offsets) > 1, "offsets must have at least two elements. [0, N] for batch size 1"
    if device is None:
        device = offsets.device
    # Read the output size off the *source* offsets while they are still on the
    # host: free there, a device sync once they have been moved. Without it the
    # CUDA repeat_interleave sizes its output with cumsum[-1].item().
    out_size = int(offsets[-1]) if offsets.device.type == "cpu" else None
    offsets = offsets.to(device=device, dtype=torch.long)
    count = offsets[1:] - offsets[:-1]
    batch = torch.arange(len(count), device=device, dtype=torch.long).repeat_interleave(
        count, output_size=out_size
    )
    return batch


@torch.inference_mode()
def batch_index_from_indices(
    indices: Int[Tensor, "N_indices"],  # type: ignore
    offsets: Int[Tensor, "B_plus_1"],  # type: ignore
    device: Optional[str] = None,
    threads: int = 256,
) -> Int[Tensor, "N_indices"]:  # type: ignore
    """
    Finds batch indices for given `indices` based on `offsets`.
    `offsets` has B+1 elements, defining B batches.
    Output has N_indices elements.
    """
    assert isinstance(indices, torch.Tensor), "indices must be a torch.Tensor"
    assert isinstance(offsets, torch.Tensor), "offsets must be a torch.Tensor"

    _dev = device
    if _dev is None:
        if indices.is_cuda:
            _dev = str(indices.device)
        elif offsets.is_cuda:
            _dev = str(offsets.device)
        else:
            raise ValueError("At least one tensor must be on CUDA if device is not specified.")

    if not indices.is_cuda or str(indices.device) != _dev:
        indices = indices.to(_dev)
    if not offsets.is_cuda or str(offsets.device) != _dev:
        offsets = offsets.to(_dev)

    indices = indices.contiguous().int()
    offsets = offsets.contiguous().int()

    M_len = offsets.shape[0]  # Length of offsets array, M_len = B + 1
    N_indices = indices.shape[0]

    if N_indices == 0:
        return torch.empty(0, dtype=torch.int32, device=_dev)
    if M_len == 0:  # No offsets defined, cannot determine batch
        raise ValueError("Offsets cannot be empty.")
    if M_len == 1:  # Only one offset value, e.g. offsets=[limit]. All indices < limit are batch 0.
        return torch.zeros(N_indices, dtype=torch.int32, device=_dev)

    batch_index_buffer = torch.empty(N_indices, dtype=torch.int32, device=_dev)

    _C.coords.find_first_gt_bsearch(
        offsets,
        M_len,
        indices,
        N_indices,
        batch_index_buffer,
    )

    return batch_index_buffer


@torch.inference_mode()
def batch_indexed_coordinates(
    batched_coords: Float[Tensor, "N 3"],  # noqa: F821
    offsets: Int[Tensor, "B + 1"],  # noqa: F821
) -> Float[Tensor, "N 4"]:  # noqa: F821
    """Prepend the batch index column to ``batched_coords``.

    Value-identical to ``cat([batch_index_from_offset(offsets), batched_coords], 1)``
    but materialises the batch column on ``batched_coords.device``.

    Why: ``offsets`` lives on the CPU by the ``BatchedTensor`` contract, so the
    stock formulation ran ``torch.repeat_interleave`` on the **CPU**. ATen's CPU
    kernel (``native/Repeat.cpp::compute_cpu``) calls ``at::parallel_for`` with
    ``grain_size=1`` over ``numel(repeats) == B``: for ``B == 1`` that takes the
    serial path, for every ``B > 1`` it opens an OpenMP parallel region and forks
    ``torch.get_num_threads()`` OS threads. Inside a CPU-limited container
    (measured: 144 threads over a 16-CPU cgroup quota) that fork/join storm
    descheduled the host thread and starved the CUDA driver, making a
    ``SparseConv3d`` call jump from 2.717 ms at B=1 to 80.637 ms at B=2 --- a
    30x cliff that is purely host-side (the GPU sat idle). Building the column
    on the GPU removes the CPU parallel region entirely.

    ``output_size=`` is load-bearing: without it the CUDA ``repeat_interleave``
    does ``cumsum[-1].item()`` to size its output, which is a device sync.
    """
    assert len(offsets) > 1, "offsets must have at least two elements. [0, N] for batch size 1"
    device = batched_coords.device
    num_points = batched_coords.shape[0]
    num_batches = len(offsets) - 1

    # Validate BEFORE dispatching to repeat_interleave. `output_size=` is a
    # promise to the CUDA kernel: if it disagrees with cumsum(counts)[-1] the
    # kernel raises a DEVICE-SIDE ASSERT, which kills the CUDA context for the
    # rest of the process and misattributes itself to whatever runs next. Stock
    # raised a clean, recoverable RuntimeError here, so the check keeps the
    # failure mode a host-side error at the actual call site.
    if offsets.device.type == "cpu":
        assert (
            int(offsets[-1]) == num_points
        ), f"Offsets {offsets} does not match the number of points {num_points}"

    if num_batches == 1:
        # Every row belongs to batch 0. Skips the repeat_interleave and the
        # offsets host-to-device copy altogether.
        batch_index = torch.zeros(
            (num_points, 1), dtype=batched_coords.dtype, device=device
        )
    else:
        offsets_dev = offsets.to(device=device, dtype=torch.long)
        counts = offsets_dev[1:] - offsets_dev[:-1]
        batch_index = (
            torch.arange(num_batches, device=device, dtype=torch.long)
            .repeat_interleave(counts, output_size=num_points)
            .to(batched_coords)
            .unsqueeze(1)
        )
    return torch.cat([batch_index, batched_coords], dim=1)


@torch.inference_mode()
def offsets_from_batch_index_consecutive(
    batch_index: Int[Tensor, "N"],  # noqa: F821
) -> Int[Tensor, "B + 1"]:  # noqa: F821
    """
    Given a list of batch indices [0, 0, 1, 1, 2, 2, 2, 3, 3],
    return the offsets [0, 2, 4, 7, 9].
    """
    assert batch_index.ndim == 1, "batch_index must be a 1D tensor"
    assert len(batch_index) > 0, "batch_index must not be empty"
    # Derive offsets (assuming out_indices_batch_indexed[:, 0] is sorted by batch)
    unique_b_idx, counts = torch.unique_consecutive(batch_index, return_counts=True)
    # Basic check if any points returned for all batches up to max batch_idx
    # This logic for offsets needs to be robust for empty batches.
    out_offsets_cpu = [0]
    max_batch_idx_present = unique_b_idx[-1].item()
    temp_counts = torch.zeros(max_batch_idx_present + 1, dtype=counts.dtype)
    temp_counts[unique_b_idx.cpu()] = counts.cpu()
    out_offsets_cpu.extend(torch.cumsum(temp_counts, dim=0).cpu().tolist())
    return torch.IntTensor(out_offsets_cpu)


@torch.inference_mode()
def offsets_from_batch_index(
    batch_index: Int[Tensor, "N"],  # noqa: F821
    num_batches: Optional[int] = None,
) -> Int[Tensor, "B + 1"]:  # noqa: F821
    """
    Given a list of batch indices [0, 0, 1, 1, 2, 2, 2, 3, 3],
    return the offsets [0, 2, 4, 7, 9].

    Args:
        batch_index: 1D tensor of batch indices.
        num_batches: Total number of batches. If provided, ensures the returned
            offsets have exactly num_batches + 1 elements, even when trailing
            batches are empty.
    """
    minlength = num_batches if num_batches is not None else 0
    counts = torch.bincount(batch_index, minlength=minlength)
    counts = counts.cpu()
    # Get the offsets by cumsum
    offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32),
            counts.cumsum(dim=0),
        ],
        dim=0,
    )
    return offsets


@torch.inference_mode()
def offsets_from_offsets(
    offsets: Int[Tensor, "B+1"],  # noqa: F821
    sorted_indices: Int[Tensor, "N"],  # noqa: F821
    device: Optional[str] = None,
) -> Int[Tensor, "B+1"]:  # noqa: F821
    """
    Given a sorted indices, return a new offsets that selects batch indices using the indices.
    """
    B = offsets.shape[0] - 1
    if B == 1:
        new_offsets = torch.IntTensor([0, len(sorted_indices)])
    else:
        batch_index = batch_index_from_offset(offsets)
        if device is not None:
            batch_index = batch_index.to(device)
            sorted_indices = sorted_indices.to(device)
        else:
            # if no device is specified, use the device of sorted_indices
            batch_index = batch_index.to(sorted_indices.device)
        _, batch_counts = torch.unique_consecutive(batch_index[sorted_indices], return_counts=True)
        batch_counts = batch_counts.cpu()
        new_offsets = torch.cat((batch_counts.new_zeros(1), batch_counts.cumsum(dim=0)))
    return new_offsets
