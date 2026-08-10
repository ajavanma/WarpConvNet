// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// HAND-WRITTEN (NOT warpgemm-generated). Safe to edit.
//
// Blackwell-only translation unit: instantiations of the deep, cross-offset
// persistent forward mask-GEMM pipeline (tile ids 1000-1004).
//
// Compiled to an empty TU unless the build carries an accelerated 10.0a
// target, which is the only spelling that defines WARPCONVNET_SM100_ENABLED
// (build_arch.py:209).

#include "mask_gemm_kernels_common.h"

#if defined(WARPCONVNET_SM100_ENABLED)

#include "include/mask_gemm_sm100_launch.h"
#include "include/wcn_sm100_tiles.h"
#include "mask_gemm/include/MaskGemm_forward_sm100_deep_pipe.h"

namespace warpconvnet {
namespace cute_gemm {

// One definition, parameterised on (ElemIn, TileTag, ElemOut, NumStages).
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
                                    cudaStream_t stream) {
  using Config = CuteTileConfig<ElemIn, TileTag>;
  using Kernel = MaskGemm_forward_sm100_deep_pipe<Config, ElemOut, 1, NumStages, MinBlocks>;
  constexpr int TileM = cute::size<0>(typename Config::TileShape{});
  constexpr int TileN = cute::size<1>(typename Config::TileShape{});
  if (N_out == 0 || C_in == 0 || C_out == 0) return 0;
  int m_tiles = (N_out + TileM - 1) / TileM;
  int n_tiles = (C_out + TileN - 1) / TileN;
  dim3 grid(m_tiles * n_tiles, 1, groups);
  size_t smem = Kernel::SharedStorageSize;
  if (smem > 48 * 1024) {
    // B200 opt-in ceiling is 232,448 B/SM; an 8-stage 64x128x32 fp16 tile
    // needs ~98 KiB. Both attributes are set: MaxDynamicSharedMemorySize is
    // the hard grant, the carveout is the scheduler hint that lets the
    // driver actually hand out the >100 KiB partition.
    if (cudaFuncSetAttribute(mask_gemm_kernel_entry<Kernel>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem)) != cudaSuccess)
      return -1;
    cudaFuncSetAttribute(mask_gemm_kernel_entry<Kernel>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         100);
  }
  mask_gemm_kernel_entry<Kernel><<<grid, Kernel::MaxThreadsPerBlock, smem, stream>>>(
      (const ElemIn *)a,
      (const ElemIn *)b,
      (ElemOut *)d,
      pt,
      pm,
      ms,
      N_in,
      N_out,
      C_in,
      C_out,
      K,
      alpha,
      C_in * groups,
      C_out * groups,
      identity_offset);
  return 0;
}

template <typename ElemIn, typename TileTag, typename ElemOut, int NumStages, int MinBlocks>
int sm100_deep_smem_bytes() {
  using Config = CuteTileConfig<ElemIn, TileTag>;
  using Kernel = MaskGemm_forward_sm100_deep_pipe<Config, ElemOut, 1, NumStages, MinBlocks>;
  return static_cast<int>(Kernel::SharedStorageSize);
}

template <typename ElemIn, typename TileTag, typename ElemOut, int NumStages, int MinBlocks>
int sm100_deep_max_active_blocks() {
  using Config = CuteTileConfig<ElemIn, TileTag>;
  using Kernel = MaskGemm_forward_sm100_deep_pipe<Config, ElemOut, 1, NumStages, MinBlocks>;
  size_t smem = Kernel::SharedStorageSize;
  if (smem > 48 * 1024) {
    if (cudaFuncSetAttribute(mask_gemm_kernel_entry<Kernel>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem)) != cudaSuccess)
      return -1;
  }
  int blocks = -1;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks, mask_gemm_kernel_entry<Kernel>, Kernel::MaxThreadsPerBlock, smem) !=
      cudaSuccess)
    return -1;
  return blocks;
}

// --- Explicit instantiations -------------------------------------------------
// Tile ids (see csrc/mask_gemm/tile_metadata.py):
//   1000 : 64x64x32   NumStages=6   (49 KiB smem)
//   1001 : 64x128x32  NumStages=6   (74 KiB smem)
//   1002 : 64x128x32  NumStages=4   (49 KiB smem)
//   1003 : 64x128x32  NumStages=8   (98 KiB smem)
//   1004 : 64x64x32   NumStages=10  (82 KiB smem)
//   1005 : 64x128x32  NumStages=6, EIGHT warps / 256 threads (74 KiB smem)
//   1006 : 64x128x32  NumStages=4, __launch_bounds__ minBlocks=3
//   1007 : 64x64x32   NumStages=4, __launch_bounds__ minBlocks=4
//   1008 : 64x64x32   NumStages=4, __launch_bounds__ minBlocks=3
//   1009 : 64x128x32  NumStages=3, __launch_bounds__ minBlocks=3
#define WCN_INST_SM100_DEEP(ElemIn, TileTag, NumStages, MinBlocks)                        \
  template int                                                                             \
  launch_mask_gemm_fwd_sm100_deep<ElemIn, gemm::TileTag, ElemIn, NumStages, MinBlocks>(    \
      const void *,                                                                        \
      const void *,                                                                        \
      void *,                                                                              \
      const int *,                                                                         \
      const uint32_t *,                                                                    \
      const int *,                                                                         \
      int,                                                                                 \
      int,                                                                                 \
      int,                                                                                 \
      int,                                                                                 \
      int,                                                                                 \
      float,                                                                               \
      int,                                                                                 \
      int,                                                                                 \
      cudaStream_t);                                                                       \
  template int sm100_deep_smem_bytes<ElemIn, gemm::TileTag, ElemIn, NumStages, MinBlocks>();      \
  template int sm100_deep_max_active_blocks<ElemIn, gemm::TileTag, ElemIn, NumStages, MinBlocks>();

#define WCN_INST_SM100_DEEP_BOTH_DTYPES(TileTag, NumStages, MinBlocks)     \
  WCN_INST_SM100_DEEP(cutlass::half_t, TileTag, NumStages, MinBlocks)      \
  WCN_INST_SM100_DEEP(cutlass::bfloat16_t, TileTag, NumStages, MinBlocks)

// Iteration 1 (depth sweep at the default 1-block launch bound).
WCN_INST_SM100_DEEP_BOTH_DTYPES(Tile64x64x32, 6, 1)      // 1000
WCN_INST_SM100_DEEP_BOTH_DTYPES(Tile64x128x32, 6, 1)     // 1001
WCN_INST_SM100_DEEP_BOTH_DTYPES(Tile64x128x32, 4, 1)     // 1002
WCN_INST_SM100_DEEP_BOTH_DTYPES(Tile64x128x32, 8, 1)     // 1003
WCN_INST_SM100_DEEP_BOTH_DTYPES(Tile64x64x32, 10, 1)     // 1004
WCN_INST_SM100_DEEP_BOTH_DTYPES(Tile64x128x32_8W, 6, 1)  // 1005 (256 threads)

// Iteration 2 (occupancy pivot). Measured blocks/SM on B200 was 3 (1000) >
// 2 (1001/1002/1003) > 1 (1005), and kernel time was monotone in it while
// depth beyond 4 stages was neutral -> force the register cap instead.
WCN_INST_SM100_DEEP_BOTH_DTYPES(Tile64x128x32, 4, 3)     // 1006 (regs <= 170)
WCN_INST_SM100_DEEP_BOTH_DTYPES(Tile64x64x32, 4, 4)      // 1007 (regs <= 128)
WCN_INST_SM100_DEEP_BOTH_DTYPES(Tile64x64x32, 4, 3)      // 1008 (regs <= 170)
WCN_INST_SM100_DEEP_BOTH_DTYPES(Tile64x128x32, 3, 3)     // 1009 (min smem, regs <= 170)

#undef WCN_INST_SM100_DEEP_BOTH_DTYPES
#undef WCN_INST_SM100_DEEP

}  // namespace cute_gemm
}  // namespace warpconvnet

#endif  // WARPCONVNET_SM100_ENABLED
