/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
#pragma once

// Single source of truth for the DSv3 fused-FFN v2 task family's SMEM region
// layouts + the cross-task meta-tensor layout (Step 3a of the V2 migration:
// the v1 ffn_full_megakernel expressed as a v2 task CHAIN:
//   rmsnorm_v2 -> dsv3_ffn_router_quant_v2 -> dsv3_ffn_topk_sigmoid_v2
//   -> dsv3_ffn_w13_gemv_v2 -> dsv3_ffn_silu_quant_v2 -> dsv3_ffn_w2_gemv_v2).
//
// Host-safe: included by src/kernel/task_register.cc (regions for the
// planner) AND by the device-side dsv3_ffn_v2.cuh (typed views). The region
// ordinals here are the contract for task_desc->smem_region_offset(N).
//
// Shapes are the DSv3 TP8 EP2 per-rank decode shapes, kept in lock-step with
// kernel::ffn_full_megakernel_sm100 (static_asserts in dsv3_ffn_v2.cuh).

#include "mirage/kernel/task_register.h"

namespace kernel {
namespace dsv3_ffn_v2 {

// ---- problem shapes (mirror ffn_full_megakernel_sm100; device side asserts
// equality against the v1 constants) --------------------------------------
inline constexpr int HIDDEN = 7168;
inline constexpr int W13_N = 1024;
inline constexpr int W2_K = 512;
inline constexpr int W2_N = 7168;
inline constexpr int GRP = 128;
inline constexpr int KG1 = HIDDEN / GRP; // 56
inline constexpr int KG2 = W2_K / GRP;   // 4
inline constexpr int MAX_ACTIVE = 8;
inline constexpr int ROUTER_N = 256;
inline constexpr int RKSPLIT = 4;
inline constexpr int SH_GU_N = 512;
inline constexpr int SH_DN_K = 256;
inline constexpr int KG_SHDN = SH_DN_K / GRP; // 2

// ---- meta tensor (int32[META_INTS], the T2->T3 chain edge) ----------------
// [0]  active_count (0..MAX_ACTIVE)
// [1]  magic 0xD5F3
// [2..9]   active_experts[8]  (LOCAL expert ids, slots >= active_count = 0)
// [10..17] active_weights[8]  (fp32 bits; normalized * routed_scaling_factor)
// [18..23] reserved (0)
inline constexpr int META_INTS = 24;
inline constexpr int META_MAGIC = 0xD5F3;
inline constexpr int META_OFF_COUNT = 0;
inline constexpr int META_OFF_MAGIC = 1;
inline constexpr int META_OFF_EXPERTS = 2;
inline constexpr int META_OFF_WEIGHTS = 10;

inline constexpr int align_up_16(int n) {
  return (n + 15) & ~15;
}

// ---- per-warp cp.async ring geometry (bytes) ------------------------------
// Sized as the max over every dgemv path a warp of the task can run:
//   router: router_partial_cpa<4,4>       -> 4 stages x 32 uint4     =  2 KB
//   w13:    dgemv_cpa16_h2<8,4>           -> 8x32x4 uint4            = 16 KB
//           dgemv_cpa16_h2<4,2> (sharedGU)-> 4x32x2 uint4            =  4 KB
//   w2:     dgemv_cpa16_h2<16,2> / <8,2>  -> 16x32x2 uint4           = 16 KB
//           dgemv_cpa<4,3> (sharedDN)     -> 4x32x3 uint32           = 1.5 KB
inline constexpr int RQ_RING_BYTES_PER_WARP = 4 * 32 * 16;   // 2048
inline constexpr int GEMV_RING_BYTES_PER_WARP = 16 * 1024;   // 16384

// ---- T1 router_quant regions ----------------------------------------------
// [0] NORM: staged rmsnorm_out (bf16[HIDDEN] = 14336 B)   -> 1 page
// [1] RING: nwarps * RQ_RING_BYTES_PER_WARP (packs sub-page)
inline constexpr int RQ_REGION_NORM = 0;
inline constexpr int RQ_REGION_RING = 1;

inline ::mirage::runtime::TaskSmemInfo make_router_quant_smem_info(int nwarps) {
  int const norm_bytes = HIDDEN * 2;
  int const ring_bytes = nwarps * RQ_RING_BYTES_PER_WARP;
  ::mirage::runtime::TaskSmemInfo info{norm_bytes + ring_bytes,
                                       /*alignment=*/1024,
                                       {}};
  info.regions.push_back({"rq_norm", norm_bytes, 1024, /*page_count=*/1,
                          /*can_pack=*/false, /*release_step=*/2,
                          /*contiguous=*/true});
  info.regions.push_back({"rq_ring", ring_bytes, 1024, /*page_count=*/1,
                          /*can_pack=*/true, /*release_step=*/2,
                          /*contiguous=*/true});
  return info;
}

// ---- T2 topk regions --------------------------------------------------------
// One packed region: s_sig f32[256] | s_biased f32[256] | s_gscore f32[8]
// | s_gactw f32[8] | s_top8v f32[64] | s_gsel i32[8] | s_gacte i32[8]
// | s_top8i i32[64]  (each sub-array 16B aligned; offsets fixed below).
inline constexpr int TK_REGION_WORK = 0;
inline constexpr int TK_OFF_SIG = 0;
inline constexpr int TK_OFF_BIASED = TK_OFF_SIG + 256 * 4;
inline constexpr int TK_OFF_GSCORE = TK_OFF_BIASED + 256 * 4;
inline constexpr int TK_OFF_GACTW = TK_OFF_GSCORE + align_up_16(8 * 4);
inline constexpr int TK_OFF_TOP8V = TK_OFF_GACTW + align_up_16(8 * 4);
inline constexpr int TK_OFF_GSEL = TK_OFF_TOP8V + 64 * 4;
inline constexpr int TK_OFF_GACTE = TK_OFF_GSEL + align_up_16(8 * 4);
inline constexpr int TK_OFF_TOP8I = TK_OFF_GACTE + align_up_16(8 * 4);
// SMEM-resident meta (the folded w13_topk publishes the routing here instead
// of / in addition to the GMEM meta tensor).
inline constexpr int TK_OFF_META = TK_OFF_TOP8I + 64 * 4;
// nwarps=7 cross-role tag-flags (u64[4]: [0] ACT_READY, [1..3] HELPER_DONE).
inline constexpr int TK_OFF_FLAGS = TK_OFF_META + align_up_16(META_INTS * 4);
inline constexpr int TK_WORK_BYTES = TK_OFF_FLAGS + 4 * 8;

inline ::mirage::runtime::TaskSmemInfo make_topk_smem_info() {
  ::mirage::runtime::TaskSmemInfo info{TK_WORK_BYTES, /*alignment=*/1024, {}};
  info.regions.push_back({"tk_work", TK_WORK_BYTES, 1024, /*page_count=*/1,
                          /*can_pack=*/true, /*release_step=*/2,
                          /*contiguous=*/true});
  return info;
}

// ---- T3 w13_gemv regions ----------------------------------------------------
// [0] ACT: a_fp8 u8[HIDDEN] @0 | a_scale f32[KG1] @HIDDEN  (7168+224=7392 B)
// [1] RING: nwarps * 16 KB (page_count = nwarps)
inline constexpr int W13_REGION_ACT = 0;
inline constexpr int W13_REGION_RING = 1;
inline constexpr int W13_ACT_SCALE_OFF = HIDDEN; // bytes; 7168 % 16 == 0

inline ::mirage::runtime::TaskSmemInfo make_w13_smem_info(int nwarps) {
  int const act_bytes = HIDDEN + KG1 * 4;
  int const ring_bytes = nwarps * GEMV_RING_BYTES_PER_WARP;
  ::mirage::runtime::TaskSmemInfo info{act_bytes + ring_bytes,
                                       /*alignment=*/1024,
                                       {}};
  info.regions.push_back({"w13_act", act_bytes, 1024, /*page_count=*/1,
                          /*can_pack=*/false, /*release_step=*/2,
                          /*contiguous=*/true});
  info.regions.push_back({"w13_ring", ring_bytes, 1024,
                          /*page_count=*/nwarps, /*can_pack=*/false,
                          /*release_step=*/2, /*contiguous=*/true});
  return info;
}

// ---- T4 silu_quant: GMEM-only, no SMEM regions ------------------------------
inline ::mirage::runtime::TaskSmemInfo make_silu_quant_smem_info() {
  return ::mirage::runtime::TaskSmemInfo{/*size=*/0, /*alignment=*/1, {}};
}

// ============================================================================
// Folded 3-op chain (router_quant_rms -> w13_topk -> w2_silu): SMEM layouts.
// ============================================================================

// ---- T1' router_quant_rms regions ------------------------------------------
// [0] NORM: staged hidden -> in-place normed bf16[HIDDEN] (14336 B) followed
//     by the 4-float block-reduce scratch + the nwarps=7 tag-flags.
// [1] RING: nwarps * RQ_RING_BYTES_PER_WARP
inline constexpr int RQR_OFF_RED = HIDDEN * 2; // bytes; float[4] warp partials
inline constexpr int RQR_OFF_FLAGS = RQR_OFF_RED + align_up_16(4 * 4);
inline constexpr int RQR_NORM_BYTES = RQR_OFF_FLAGS + 4 * 8;

inline ::mirage::runtime::TaskSmemInfo
    make_router_quant_rms_smem_info(int nwarps) {
  int const ring_bytes = nwarps * RQ_RING_BYTES_PER_WARP;
  ::mirage::runtime::TaskSmemInfo info{RQR_NORM_BYTES + ring_bytes,
                                       /*alignment=*/1024,
                                       {}};
  info.regions.push_back({"rqr_norm", RQR_NORM_BYTES, 1024, /*page_count=*/1,
                          /*can_pack=*/false, /*release_step=*/2,
                          /*contiguous=*/true});
  info.regions.push_back({"rqr_ring", ring_bytes, 1024, /*page_count=*/1,
                          /*can_pack=*/true, /*release_step=*/2,
                          /*contiguous=*/true});
  return info;
}

// ---- T2' w13_topk regions ----------------------------------------------------
// [0] ACT (same as w13)  [1] RING (nwarps pages)  [2] TK work (packed)
inline constexpr int W13TK_REGION_ACT = 0;
inline constexpr int W13TK_REGION_RING = 1;
inline constexpr int W13TK_REGION_TK = 2;

inline ::mirage::runtime::TaskSmemInfo make_w13_topk_smem_info(int nwarps) {
  int const act_bytes = HIDDEN + KG1 * 4;
  int const ring_bytes = nwarps * GEMV_RING_BYTES_PER_WARP;
  ::mirage::runtime::TaskSmemInfo info{act_bytes + ring_bytes + TK_WORK_BYTES,
                                       /*alignment=*/1024,
                                       {}};
  info.regions.push_back({"w13tk_act", act_bytes, 1024, /*page_count=*/1,
                          /*can_pack=*/false, /*release_step=*/2,
                          /*contiguous=*/true});
  info.regions.push_back({"w13tk_ring", ring_bytes, 1024,
                          /*page_count=*/nwarps, /*can_pack=*/false,
                          /*release_step=*/2, /*contiguous=*/true});
  info.regions.push_back({"w13tk_tk", TK_WORK_BYTES, 1024, /*page_count=*/1,
                          /*can_pack=*/true, /*release_step=*/2,
                          /*contiguous=*/true});
  return info;
}

// ---- T3' w2_silu regions -----------------------------------------------------
// [0] ACT (same layout as w2: i_fp8|i_scale|si_fp8|si_scale, but COMPUTED by
//     the in-task silu instead of cp.async-staged)
// [1] RING (nwarps pages). The first MAX_ACTIVE*W13_N*4 + SH_GU_N*4 bytes
//     (34 KB <= 4*16 KB) double as the y13/sg staging area during the silu
//     phase; the GEMV reuses the ring after the post-silu consumer barrier.
// make_w2_silu_smem_info (== make_w2_smem_info) is defined after the w2
// section below.
inline constexpr int W2S_RING_Y13_OFF = 0;
inline constexpr int W2S_RING_SG_OFF = MAX_ACTIVE * W13_N * 4; // 32768
static_assert(W2S_RING_SG_OFF + SH_GU_N * 4 <= 4 * GEMV_RING_BYTES_PER_WARP,
              "y13/sg staging must fit the 4 consumer ring slices");

// ---- T5 w2_gemv regions ------------------------------------------------------
// [0] ACT: i_fp8 u8[8*512] @0 | i_scale f32[8*4] @4096 | si_fp8 u8[256] @4224
//          | si_scale f32[2] @4480   (4488 B)
// [1] RING: nwarps * 16 KB
inline constexpr int W2_REGION_ACT = 0;
inline constexpr int W2_REGION_RING = 1;
inline constexpr int W2_ACT_ISCALE_OFF = MAX_ACTIVE * W2_K;             // 4096
inline constexpr int W2_ACT_SIFP8_OFF =
    W2_ACT_ISCALE_OFF + MAX_ACTIVE * KG2 * 4;                           // 4224
inline constexpr int W2_ACT_SISCALE_OFF = W2_ACT_SIFP8_OFF + SH_DN_K;   // 4480
// nwarps=7 cross-role tag-flags (u64[4]); 16B-aligned tail.
inline constexpr int W2_ACT_FLAGS_OFF =
    align_up_16(W2_ACT_SISCALE_OFF + KG_SHDN * 4);                      // 4496
inline constexpr int W2_ACT_BYTES = W2_ACT_FLAGS_OFF + 4 * 8;           // 4528

inline ::mirage::runtime::TaskSmemInfo make_w2_smem_info(int nwarps) {
  int const ring_bytes = nwarps * GEMV_RING_BYTES_PER_WARP;
  ::mirage::runtime::TaskSmemInfo info{W2_ACT_BYTES + ring_bytes,
                                       /*alignment=*/1024,
                                       {}};
  info.regions.push_back({"w2_act", W2_ACT_BYTES, 1024, /*page_count=*/1,
                          /*can_pack=*/false, /*release_step=*/2,
                          /*contiguous=*/true});
  info.regions.push_back({"w2_ring", ring_bytes, 1024,
                          /*page_count=*/nwarps, /*can_pack=*/false,
                          /*release_step=*/2, /*contiguous=*/true});
  return info;
}

inline ::mirage::runtime::TaskSmemInfo make_w2_silu_smem_info(int nwarps) {
  return make_w2_smem_info(nwarps); // identical region shapes
}

// ============================================================================
// FUSION-LADDER EXPERIMENT (scratch/v2_ffn_fuse): Rung A 2-op chain
// (w13_rqr_topk -> w2_silu) and Rung B 1-op megakernel-shape (ffn_mega).
// ============================================================================

// Extra shape constants needed host-side by the packs (device side
// static_asserts these against the v1 kernel's constants).
inline constexpr int E_LOCAL = 128;
inline constexpr int NB1 = W13_N / GRP;   // 8
inline constexpr int NB2 = W2_N / GRP;    // 56
inline constexpr int KG_SHGU = HIDDEN / GRP;  // 56
inline constexpr int NB_SHGU = SH_GU_N / GRP; // 4
inline constexpr int NB_SHDN = W2_N / GRP;    // 56

// ---- Rung A: w13_rqr_topk regions ------------------------------------------
// [0] NORM (RQR layout: staged hidden -> in-place normed + reduce scratch;
//     the RQR flag tail is UNUSED here — Rung A's flags live in the TK region)
// [1] ACT (w13 layout: a_fp8 | a_scale — COMPUTED by the in-task quant)
// [2] RING (nwarps pages)
// [3] TK: [s_inter f32[1024] @0 | TK work @4096 | flags u64[8] @A_TK_OFF_FLAGS]
//     flags: [0] NORM_READY  [1..3] RDONE (helper router+quant done)
//            [4] META_READY  [5..7] epilogue helper-done
//            (mac_task_epilogue is handed &flags[4] so it touches [5..7])
inline constexpr int A_REGION_NORM = 0;
inline constexpr int A_REGION_ACT = 1;
inline constexpr int A_REGION_RING = 2;
inline constexpr int A_REGION_TK = 3;
inline constexpr int A_TK_OFF_INTER = 0; // f32[ROUTER_N*RKSPLIT] = 4096 B
inline constexpr int A_TK_OFF_WK = ROUTER_N * RKSPLIT * 4;
inline constexpr int A_TK_OFF_FLAGS = A_TK_OFF_WK + align_up_16(TK_WORK_BYTES);
inline constexpr int A_TK_BYTES = A_TK_OFF_FLAGS + 8 * 8;

inline ::mirage::runtime::TaskSmemInfo make_w13_rqr_topk_smem_info(int nwarps) {
  int const act_bytes = HIDDEN + KG1 * 4;
  int const ring_bytes = nwarps * GEMV_RING_BYTES_PER_WARP;
  ::mirage::runtime::TaskSmemInfo info{
      RQR_NORM_BYTES + act_bytes + ring_bytes + A_TK_BYTES,
      /*alignment=*/1024,
      {}};
  info.regions.push_back({"artk_norm", RQR_NORM_BYTES, 1024, /*page_count=*/1,
                          /*can_pack=*/false, /*release_step=*/2,
                          /*contiguous=*/true});
  info.regions.push_back({"artk_act", act_bytes, 1024, /*page_count=*/1,
                          /*can_pack=*/false, /*release_step=*/2,
                          /*contiguous=*/true});
  info.regions.push_back({"artk_ring", ring_bytes, 1024,
                          /*page_count=*/nwarps, /*can_pack=*/false,
                          /*release_step=*/2, /*contiguous=*/true});
  info.regions.push_back({"artk_tk", A_TK_BYTES, 1024, /*page_count=*/1,
                          /*can_pack=*/true, /*release_step=*/2,
                          /*contiguous=*/true});
  return info;
}

// ---- Rung B: ffn_mega regions ------------------------------------------------
// [0] NORM (RQR layout, flag tail unused)  [1] ACT (w13 layout, quant-computed)
// [2] RING (nwarps pages; W2-phase y13/sg staging borrows the first 34 KB =
//     consumer slices, exactly like w2_silu)  [3] TK+flags  [4] W2ACT (w2
//     layout: i_fp8|i_scale|si_fp8|si_scale, silu-computed; flag tail unused)
// flags u64[16]: [0] NORM_READY  [1..3] PH1 (helper router done)  [4] GO1
//   [5] META_READY  [6..8] PH2 (helper W13 done)  [9] GO2  [10] SILU_READY
//   [11] (base for epilogue: mac_task_epilogue gets &flags[11] -> [12..14])
inline constexpr int M_REGION_NORM = 0;
inline constexpr int M_REGION_ACT = 1;
inline constexpr int M_REGION_RING = 2;
inline constexpr int M_REGION_TK = 3;
inline constexpr int M_REGION_W2ACT = 4;
inline constexpr int M_TK_OFF_WK = 0;
inline constexpr int M_TK_OFF_FLAGS = align_up_16(TK_WORK_BYTES);
inline constexpr int M_TK_BYTES = M_TK_OFF_FLAGS + 16 * 8;

inline ::mirage::runtime::TaskSmemInfo make_ffn_mega_smem_info(int nwarps) {
  int const act_bytes = HIDDEN + KG1 * 4;
  int const ring_bytes = nwarps * GEMV_RING_BYTES_PER_WARP;
  ::mirage::runtime::TaskSmemInfo info{
      RQR_NORM_BYTES + act_bytes + ring_bytes + M_TK_BYTES + W2_ACT_BYTES,
      /*alignment=*/1024,
      {}};
  info.regions.push_back({"mega_norm", RQR_NORM_BYTES, 1024, /*page_count=*/1,
                          /*can_pack=*/false, /*release_step=*/2,
                          /*contiguous=*/true});
  info.regions.push_back({"mega_act", act_bytes, 1024, /*page_count=*/1,
                          /*can_pack=*/false, /*release_step=*/2,
                          /*contiguous=*/true});
  info.regions.push_back({"mega_ring", ring_bytes, 1024,
                          /*page_count=*/nwarps, /*can_pack=*/false,
                          /*release_step=*/2, /*contiguous=*/true});
  info.regions.push_back({"mega_tk", M_TK_BYTES, 1024, /*page_count=*/1,
                          /*can_pack=*/true, /*release_step=*/2,
                          /*contiguous=*/true});
  info.regions.push_back({"mega_w2act", W2_ACT_BYTES, 1024, /*page_count=*/1,
                          /*can_pack=*/true, /*release_step=*/2,
                          /*contiguous=*/true});
  return info;
}

// ---- Rung B GMEM pack layouts (host allocates; device + harness share) ------
// xfer pack (f32 element offsets): the two in-op all-to-alls. inter and y13
// lifetimes overlap ACROSS tasks (a task can still read inter while another
// writes y13) so they get disjoint storage.
inline constexpr int MEGA_XFER_OFF_INTER_F = 0;                    // [1024]
inline constexpr int MEGA_XFER_OFF_Y13_F = ROUTER_N * RKSPLIT;     // [8*1024]
inline constexpr int MEGA_XFER_OFF_SG_F =
    MEGA_XFER_OFF_Y13_F + MAX_ACTIVE * W13_N;                      // [512]
inline constexpr int MEGA_XFER_FLOATS = MEGA_XFER_OFF_SG_F + SH_GU_N;

// scales pack (f32 element offsets): w13_s | wgu_s | w2_s | wdn_s.
inline constexpr int MEGA_SC_OFF_W13 = 0;
inline constexpr int MEGA_SC_OFF_WGU = MEGA_SC_OFF_W13 + E_LOCAL * NB1 * KG1;
inline constexpr int MEGA_SC_OFF_W2 = MEGA_SC_OFF_WGU + NB_SHGU * KG_SHGU;
inline constexpr int MEGA_SC_OFF_WDN = MEGA_SC_OFF_W2 + E_LOCAL * NB2 * KG2;
inline constexpr int MEGA_SC_FLOATS = MEGA_SC_OFF_WDN + NB_SHDN * KG_SHDN;

// artifacts pack (BYTE offsets, each 16B-aligned; task-0-published compare
// artifacts — not dataflow):
//   rmsnorm_out bf16[HIDDEN] | a_fp8 u8[HIDDEN] | a_scale f32[KG1]
//   | logits bf16[ROUTER_N] | meta i32[META_INTS] | i_fp8 u8[8*512]
//   | i_scale f32[8*4] | si_fp8 u8[256] | si_scale f32[2]
inline constexpr int MEGA_ART_OFF_RMSNORM = 0;
inline constexpr int MEGA_ART_OFF_AFP8 =
    align_up_16(MEGA_ART_OFF_RMSNORM + HIDDEN * 2);
inline constexpr int MEGA_ART_OFF_ASCALE =
    align_up_16(MEGA_ART_OFF_AFP8 + HIDDEN);
inline constexpr int MEGA_ART_OFF_LOGITS =
    align_up_16(MEGA_ART_OFF_ASCALE + KG1 * 4);
inline constexpr int MEGA_ART_OFF_META =
    align_up_16(MEGA_ART_OFF_LOGITS + ROUTER_N * 2);
inline constexpr int MEGA_ART_OFF_IFP8 =
    align_up_16(MEGA_ART_OFF_META + META_INTS * 4);
inline constexpr int MEGA_ART_OFF_ISCALE =
    align_up_16(MEGA_ART_OFF_IFP8 + MAX_ACTIVE * W2_K);
inline constexpr int MEGA_ART_OFF_SIFP8 =
    align_up_16(MEGA_ART_OFF_ISCALE + MAX_ACTIVE * KG2 * 4);
inline constexpr int MEGA_ART_OFF_SISCALE =
    align_up_16(MEGA_ART_OFF_SIFP8 + SH_DN_K);
inline constexpr int MEGA_ART_BYTES =
    align_up_16(MEGA_ART_OFF_SISCALE + KG_SHDN * 4);

} // namespace dsv3_ffn_v2
} // namespace kernel
