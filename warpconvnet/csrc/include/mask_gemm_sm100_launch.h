// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// HAND-WRITTEN (NOT warpgemm-generated). Safe to edit.
//
// Launch declarations for the Blackwell (sm_100) deep-pipe forward mask-GEMM
// tiles. Kept out of mask_gemm_kernels_common.h so that the new tiles live in
// their own translation unit (warpconvnet/csrc/mask_gemm_kernels_fwd_sm100.cu)
// and an incremental rebuild of them does not recompile the 100+ incumbent
// forward instantiations.
//
// The whole family is compiled ONLY when the build carries an accelerated
// 10.0a target (build_arch.py: -DWARPCONVNET_SM100_ENABLED). On any other
// build the tiles are absent and the binding arms below are compiled out, so
// dispatching tile_id 1000-1009 raises the usual "Unsupported tile_id".

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace warpconvnet {
namespace cute_gemm {

// TileTag selects the (M, N, K) config from CuteTileConfig; NumStages is the
// pipeline depth (smem/stage = (tM + tN) * tK * sizeof(ElementInput));
// MinBlocks is the __launch_bounds__ minimum-blocks-per-SM register cap.
template <typename ElemIn, typename TileTag, typename ElemOut, int NumStages, int MinBlocks>
int launch_mask_gemm_fwd_sm100_deep(const void *a,
                                    const void *b,
                                    void *d,
                                    const int *pt,
                                    const uint32_t *pm,
                                    const int *ms,
                                    int N_in,
                                    int N_out,
                                    int C_in,
                                    int C_out,
                                    int K,
                                    float alpha,
                                    int groups,
                                    int identity_offset,
                                    cudaStream_t stream);

// Introspection for the validation harness: static shared-memory footprint of
// a (TileTag, NumStages) instantiation, and the occupancy the driver reports
// for it. Returns -1 if the query fails.
template <typename ElemIn, typename TileTag, typename ElemOut, int NumStages, int MinBlocks>
int sm100_deep_smem_bytes();

template <typename ElemIn, typename TileTag, typename ElemOut, int NumStages, int MinBlocks>
int sm100_deep_max_active_blocks();

}  // namespace cute_gemm
}  // namespace warpconvnet
