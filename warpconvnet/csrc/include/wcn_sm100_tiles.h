// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// HAND-WRITTEN (NOT warpgemm-generated). Safe to edit.
//
// wcn-only tile tags for the Blackwell deep-pipe forward family. Follows the
// same extension pattern as wcn_pcoff_tiles.h: declare a tag in
// warpconvnet::gemm, specialize CuteTileConfig in warpconvnet::cute_gemm.
//
// The one config that is NOT just a re-tag of a canonical base is the
// EIGHT-WARP 64x128x32 variant. Motivation: the incumbent forward configs all
// use a 2x2 warp layout (128 threads), so only 4 warps issue the sparse
// A-gather. Standalone microbenchmarks of the same
// `cp.async.ca.shared.global.L2::128B` instruction reach 12.5-16.3 TB/s with
// 4-8 issuing warps while the kernel achieves 1.09 TB/s, so issue width is a
// candidate co-factor alongside pipeline depth. Doubling the warp count also
// halves the per-thread fp32 accumulator (64 -> 32 registers), which is the
// single biggest register consumer in the mainloop.

#pragma once

// clang-format off
#include "cute/tensor.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/layout.hpp"
// clang-format on
#include "cute_gemm_config.h"
#include "gemm_mma_tiles.h"

namespace warpconvnet {
namespace gemm {

// 64x128x32, F32 accumulator, 2x4 warp layout (256 threads).
struct Tile64x128x32_8W {};

}  // namespace gemm

namespace cute_gemm {

#define WCN_DEFINE_SM100_8W_CONFIG(Elem, MmaOp)                                              \
  template <>                                                                                \
  struct CuteTileConfig<Elem, gemm::Tile64x128x32_8W> {                                      \
    using ElementInput = Elem;                                                               \
    using TileShape = cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<32>>;             \
    /* 2x4 warps; the permutation tiler keeps the atom's 16x8x16 shape and     */            \
    /* gives each warp a 32x32x16 slab, so the CTA tile is covered by 2 M- and */            \
    /* 1 N-iteration of the 32x128 warp-group footprint.                       */            \
    using TiledMma = cute::TiledMMA<cute::MMA_Atom<MmaOp>,                                   \
                                    cute::Layout<cute::Shape<cute::_2, cute::_4, cute::_1>>, \
                                    cute::Tile<cute::_32, cute::_64, cute::_16>>;            \
    using SmemLayoutAtomA = SmemLayoutAtomA_FP16;                                            \
    using SmemLayoutAtomB = SmemLayoutAtomB_FP16;                                            \
    using SmemCopyAtomA = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, ElementInput>;            \
    using SmemCopyAtomB = cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, ElementInput>;            \
    using GmemTiledCopyA = void;                                                             \
    using GmemTiledCopyB = void;                                                             \
    static constexpr int NumStages = 2;                                                      \
    static constexpr int AlignmentA = 4;                                                     \
    static constexpr int AlignmentB = 4;                                                     \
    static constexpr bool UseCpAsyncGatherA = false;                                         \
  };

WCN_DEFINE_SM100_8W_CONFIG(cutlass::half_t, cute::SM80_16x8x16_F32F16F16F32_TN)
WCN_DEFINE_SM100_8W_CONFIG(cutlass::bfloat16_t, cute::SM80_16x8x16_F32BF16BF16F32_TN)

#undef WCN_DEFINE_SM100_8W_CONFIG

}  // namespace cute_gemm
}  // namespace warpconvnet
