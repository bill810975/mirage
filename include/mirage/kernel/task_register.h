/* Copyright 2023-2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "mirage/persistent_kernel/runtime_header.h"
#include "mirage/threadblock/graph.h"
#include <string>
#include <vector>

namespace mirage {
namespace runtime {

struct TaskSmemRegion {
  std::string name;
  int size = 0;
  int alignment = 1;
  int page_count = -1;
  bool can_pack = false;
  int release_step = 0;
  bool contiguous = true;
};

struct TaskSmemInfo {
  int size = 0;
  int alignment = 1;
  std::vector<TaskSmemRegion> regions;
};

struct TaskRoleVariantCode {
  // Optional: op-declared body that the controller runs (single-thread)
  // once per published instruction, before role warps wake. Use to
  // mbar_init slots in runtime_smem->dynamic_semaphores[slot][i] that
  // the role bodies will arrive/wait on.
  std::string init_semaphores;
  std::string loader;
  std::string launcher;
  std::string consumer;
  std::string storer;

  // Phase 3.5: page-lifecycle hooks. Defaults wire every task into the
  // generic page protocol (every task arrives every page exactly once).
  //   - auto_loader_page_lifecycle: codegen prepends every loader body
  //     with a lane-parallel "wait every page; for pages this task does
  //     not use, finish them immediately." If the user has no loader
  //     body, codegen emits one anyway with just this prefix.
  //   - auto_consumer_finish: codegen appends every consumer body with
  //     a runtime_finish_region_range_pages over the task's regions
  //     (i.e. the pages this task uses get released here). Set false
  //     for tasks that release pages incrementally inside their body
  //     (e.g. linear's per-stage release in 3.5b).
  bool auto_loader_page_lifecycle = true;
  bool auto_consumer_finish = true;
};

class TaskRegister {
public:
  static TaskRegister *singleton;
  TaskRegister();

public:
  static TaskRegister *get_instance();
  int register_embedding_task(threadblock::Graph const &bgraph,
                              std::vector<int> const &params);
  int register_rmsnorm_task(threadblock::Graph const &bgraph,
                            std::vector<int> const &params);
  int register_rmsnorm_linear_task(threadblock::Graph const &bgraph,
                                   std::vector<int> const &params);
  int register_attention_task(threadblock::Graph const &bgraph,
                              std::vector<int> const &params);
  int register_paged_attention_task(threadblock::Graph const &bgraph,
                                    std::vector<int> const &params);
  int register_single_batch_extend_attention_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_linear_task(threadblock::Graph const &bgraph,
                           std::vector<int> const &params,
                           bool with_residual);
  int register_silu_mul_task(threadblock::Graph const &bgraph,
                             std::vector<int> const &params);
  int register_identity_task(threadblock::Graph const &bgraph,
                             std::vector<int> const &params);
  int register_silu_mul_linear_with_residual_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_argmax_partial_task(threadblock::Graph const &bgraph,
                                   std::vector<int> const &params);
  int register_argmax_reduce_task(threadblock::Graph const &bgraph,
                                  std::vector<int> const &params);
  int register_reduction_task(threadblock::Graph const &bgraph,
                              std::vector<int> const &params);
  int register_find_ngram_partial_task(threadblock::Graph const &bgraph,
                                       std::vector<int> const &params);
  int register_find_ngram_global_task(threadblock::Graph const &bgraph,
                                      std::vector<int> const &params);
  int register_target_verify_greedy_task(threadblock::Graph const &bgraph,
                                         std::vector<int> const &params);
  // Hopper tasks
  int register_linear_hopper_task(threadblock::Graph const &bgraph,
                                  std::vector<int> const &params,
                                  bool with_residual);
  int register_paged_attention_hopper_task(threadblock::Graph const &bgraph,
                                           std::vector<int> const &params);
  int register_rmsnorm_hopper_task(threadblock::Graph const &bgraph,
                                   std::vector<int> const &params);
  int register_linear_swapAB_hopper_task(threadblock::Graph const &bgraph,
                                         std::vector<int> const &params,
                                         bool with_residual);
  int register_linear_cutlass_hopper_task(threadblock::Graph const &bgraph,
                                          std::vector<int> const &params,
                                          bool with_residual);
  int register_silu_mul_hopper_task(threadblock::Graph const &bgraph,
                                    std::vector<int> const &params);
  int register_embedding_hopper_task(threadblock::Graph const &bgraph,
                                     std::vector<int> const &params);
  int register_moe_linear_sm90_task(threadblock::Graph const &bgraph,
                                    std::vector<int> const &params,
                                    bool w13_linear);
  int register_splitk_linear_swapAB_hopper_task(
      threadblock::Graph const &bgraph,
      std::vector<int> const &params,
      bool with_residual);
  int register_paged_attention_split_kv_hopper_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  // SM100 tasks
  int register_splitk_linear_sm100_task(threadblock::Graph const &bgraph,
                                        std::vector<int> const &params,
                                        bool with_residual);
  int register_linear_sm100_task(threadblock::Graph const &bgraph,
                                 std::vector<int> const &params,
                                 bool with_residual);
  // v2 linear: hand-written swapAB kernel in blackwell_v2/linear_sm100_v2.cuh.
  // Uses one TMA descriptor per op (full W & A), runtime tile_idx from
  // task_metadata.task_offset. Requires grid_dim[0] = N / BLOCK_M(=128).
  int register_linear_sm100_v2_task(threadblock::Graph const &bgraph,
                                    std::vector<int> const &params,
                                    bool with_residual);
  // v3 linear: same math as v2 but driven by typed Channel + TmemChannel
  // primitives (blackwell_v2/linear_sm100_v3.cuh + channel.cuh). Encapsulated
  // drain replaces the v2 role-warp re-init workaround.
  int register_linear_sm100_v3_task(threadblock::Graph const &bgraph,
                                    std::vector<int> const &params,
                                    bool with_residual);
  // v2 dispatch variants for non-linear tasks. Emit same kernel calls as the
  // v1 versions, but register under TASK_X_V2 enums so the whole pipeline
  // goes through v2 codegen (no mixed v1/v2 dispatch in the task graph).
  int register_rmsnorm_hopper_v2_task(threadblock::Graph const &bgraph,
                                      std::vector<int> const &params);
  int register_silu_mul_v2_task(threadblock::Graph const &bgraph,
                                std::vector<int> const &params);
  int register_embedding_v2_task(threadblock::Graph const &bgraph,
                                 std::vector<int> const &params);
  int register_paged_attention_sm100_v2_task(threadblock::Graph const &bgraph,
                                             std::vector<int> const &params);
  int register_argmax_partial_sm100_v2_task(threadblock::Graph const &bgraph,
                                            std::vector<int> const &params);
  int register_argmax_reduce_sm100_v2_task(threadblock::Graph const &bgraph,
                                           std::vector<int> const &params);
  // DSv3 fused-FFN block as a v2 task chain (Step 3a of the V2 migration).
  // See tasks/blackwell_v2/dsv3_ffn_v2.cuh for the chain layout.
  int register_dsv3_ffn_router_quant_v2_task(threadblock::Graph const &bgraph,
                                             std::vector<int> const &params);
  int register_dsv3_ffn_topk_sigmoid_v2_task(threadblock::Graph const &bgraph,
                                             std::vector<int> const &params);
  int register_dsv3_ffn_w13_gemv_v2_task(threadblock::Graph const &bgraph,
                                         std::vector<int> const &params);
  int register_dsv3_ffn_silu_quant_v2_task(threadblock::Graph const &bgraph,
                                           std::vector<int> const &params);
  int register_dsv3_ffn_w2_gemv_v2_task(threadblock::Graph const &bgraph,
                                        std::vector<int> const &params);
  // Folded 3-op variant (router_quant_rms -> w13_topk -> w2_silu).
  int register_dsv3_ffn_router_quant_rms_v2_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_dsv3_ffn_w13_topk_v2_task(threadblock::Graph const &bgraph,
                                         std::vector<int> const &params);
  int register_dsv3_ffn_w2_silu_v2_task(threadblock::Graph const &bgraph,
                                        std::vector<int> const &params);
  // DSv3 fused-ATTN block as a v2 task chain (Step 3b of the V2 migration).
  // See tasks/blackwell_v2/dsv3_attn_v2.cuh for the chain layout.
  int register_dsv3_attn_p0_qkva_v2_task(threadblock::Graph const &bgraph,
                                         std::vector<int> const &params);
  int register_dsv3_attn_qb_rope_kv_v2_task(threadblock::Graph const &bgraph,
                                            std::vector<int> const &params);
  int register_dsv3_attn_mla_fused_v2_task(threadblock::Graph const &bgraph,
                                            std::vector<int> const &params);
  int register_dsv3_attn_mla_partial_v2_task(threadblock::Graph const &bgraph,
                                             std::vector<int> const &params);
  int register_dsv3_attn_mla_merge_v2_task(threadblock::Graph const &bgraph,
                                           std::vector<int> const &params);
  int register_dsv3_attn_wuv_v2_task(threadblock::Graph const &bgraph,
                                     std::vector<int> const &params);
  int register_dsv3_attn_oproj_v2_task(threadblock::Graph const &bgraph,
                                       std::vector<int> const &params);
  int register_paged_attention_sm100_task(threadblock::Graph const &bgraph,
                                          std::vector<int> const &params);
  int register_argmax_partial_sm100_task(threadblock::Graph const &bgraph,
                                         std::vector<int> const &params);
  int register_argmax_reduce_sm100_task(threadblock::Graph const &bgraph,
                                        std::vector<int> const &params);
  int register_sampling_sm100_task(threadblock::Graph const &bgraph,
                                   std::vector<int> const &params);
  int register_tensor_init_task(threadblock::Graph const &bgraph,
                                std::vector<int> const &params);
  int register_elementwise_add_sm100_task(threadblock::Graph const &bgraph,
                                          std::vector<int> const &params);
  int register_softmax_gather_sm100_task(threadblock::Graph const &bgraph,
                                         std::vector<int> const &params);
  int register_mtp_verify_probabilistic_task(threadblock::Graph const &bgraph,
                                             std::vector<int> const &params);
  int register_mtp_float_scatter_task(threadblock::Graph const &bgraph,
                                      std::vector<int> const &params);
  int register_prob_extract_sm100_task(threadblock::Graph const &bgraph,
                                       std::vector<int> const &params);
  int register_prob_scatter_sm100_task(threadblock::Graph const &bgraph,
                                       std::vector<int> const &params);
  int register_moe_topk_softmax_sm100_task(threadblock::Graph const &bgraph,
                                           std::vector<int> const &params);
  int register_moe_topk_sigmoid_sm100_task(threadblock::Graph const &bgraph,
                                           std::vector<int> const &params);
  int register_moe_linear_sm100_task(threadblock::Graph const &bgraph,
                                     std::vector<int> const &params,
                                     bool w13_linear);
  int register_moe_fp8_sm100_task(threadblock::Graph const &bgraph,
                                  std::vector<int> const &params,
                                  bool w13_linear);
  int register_moe_silu_mul_task(threadblock::Graph const &bgraph,
                                 std::vector<int> const &params);
  int register_moe_mul_sum_add_sm100_task(threadblock::Graph const &bgraph,
                                          std::vector<int> const &params);
  int register_paged_attention_split_kv_sm100_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_paged_attention_split_kv_merge_sm100_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_mla_decode_sm100_task(threadblock::Graph const &bgraph,
                                     std::vector<int> const &params);
  int register_mla_reduce_sm100_task(threadblock::Graph const &bgraph,
                                     std::vector<int> const &params);
  int register_mla_prefill_sm100_task(threadblock::Graph const &bgraph,
                                      std::vector<int> const &params);
  int register_mla_prefill_absorbed_sm100_task(threadblock::Graph const &bgraph,
                                               std::vector<int> const &params);
  int register_mla_prefill_tp8_sm100_task(threadblock::Graph const &bgraph,
                                          std::vector<int> const &params);
  int register_mla_prefill_tp8_chunked_sm100_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_mla_prefill_tp8_chunked_splitk_sm100_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_mla_prefill_tp8_chunked_reduce_sm100_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_mla_unified_sm100_task(threadblock::Graph const &bgraph,
                                      std::vector<int> const &params);
  int register_mla_mtp_decode_sm100_task(threadblock::Graph const &bgraph,
                                         std::vector<int> const &params);
  int register_mla_mtp_reduce_sm100_task(threadblock::Graph const &bgraph,
                                         std::vector<int> const &params);
  // MLA-MTP TP variants (ferret-derived, no-PDL). Each: TP=2/4/8, with paired
  // reduce. Differ in NUM_HEADS (64/32/16); TP=4 also splits V across two
  // CTAs (z=2); TP=8 takes Q_LEN_real (Q_LEN padded to even).
  int register_mla_mtp_decode_tp2_sm100_task(threadblock::Graph const &bgraph,
                                             std::vector<int> const &params);
  int register_mla_mtp_decode_tp4_sm100_task(threadblock::Graph const &bgraph,
                                             std::vector<int> const &params);
  int register_mla_mtp_decode_tp8_sm100_task(threadblock::Graph const &bgraph,
                                             std::vector<int> const &params);
  // Unified TP2/TP4/TP8 split-KV reduce (one TASK_MLA_MTP_DECODE_TP_REDUCE
  // enum; tp in {2, 4, 8} picks the device function at graph-build time).
  int register_mla_mtp_decode_tp_reduce_sm100_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params, int tp);
  int register_quantize_fp8_sm100_task(threadblock::Graph const &bgraph,
                                       std::vector<int> const &params,
                                       bool scale_ue8m0);
  int register_linear_fp8_sm100_task(threadblock::Graph const &bgraph,
                                     std::vector<int> const &params,
                                     bool with_residual);
  int register_linear_fp8_swapAB_sm100_task(threadblock::Graph const &bgraph,
                                            std::vector<int> const &params,
                                            bool with_residual);
  int register_splitk_linear_fp8_swapAB_sm100_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_linear_fp8_bmm_sm100_task(threadblock::Graph const &bgraph,
                                         std::vector<int> const &params);
  int register_linear_fp8_bmm_dense_sm100_task(threadblock::Graph const &bgraph,
                                               std::vector<int> const &params);
  // Unified dense FP8 GEMM family (one TASK_FP8_GEMM_DENSE_SM100 enum).
  // `mediumm` picks the smallm/mediumm tile flavor; the *_fp8out_* fn picks
  // the epilogue-UE8M0-quantize flavor; decode_splitk is the split-K
  // decode variant. All register variants under the same task type.
  int register_fp8_gemm_dense_sm100_task(threadblock::Graph const &bgraph,
                                         std::vector<int> const &params,
                                         bool mediumm);
  // ferret BF16 CUDA-core GEMV for DSv3 router gate. params: [M,N,K,
  // num_workers]. default-OFF lever (MPK_DSV3_ROUTER_GEMV).
  int register_dsv3_router_gate_gemv_sm100_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  // fine-N dense GEMM (mediumm body @ BN=16, NS=6). default-OFF
  // MPK_DSV3_DENSE_FINEN.
  int register_fp8_gemm_dense_finen_sm100_task(threadblock::Graph const &bgraph,
                                               std::vector<int> const &params);
  int register_fp8_gemm_dense_decode_splitk_sm100_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_fp8_gemm_dense_fp8out_sm100_task(
      threadblock::Graph const &bgraph,
      std::vector<int> const &params,
      bool mediumm);
  int register_fused_rmsnorm_quantize_fp8_sm100_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_fp8_group_gemm_smallm_sm100_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_fp8_group_gemm_largem_sm100_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_fp8_group_gemm_largem_compact_sm100_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_fp8_group_gemm_largem_compact_fused_sm100_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_ffn_full_megakernel_sm100_task(threadblock::Graph const &bgraph,
                                              std::vector<int> const &params);
  int register_dsv3_dense_mlp_fused_sm100_task(threadblock::Graph const &bgraph,
                                               std::vector<int> const &params);
  int register_attn_block_megakernel_sm100_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_moe_permute_sm100_task(threadblock::Graph const &bgraph,
                                      std::vector<int> const &params);
  int register_moe_unpermute_sm100_task(threadblock::Graph const &bgraph,
                                        std::vector<int> const &params);
  int register_transpose_scale_sm100_task(threadblock::Graph const &bgraph,
                                          std::vector<int> const &params);
  int register_assemble_q_decode_sm100_task(threadblock::Graph const &bgraph,
                                            std::vector<int> const &params);
  int register_mla_kv_append_sm100_task(threadblock::Graph const &bgraph,
                                        std::vector<int> const &params);
  int register_mla_kv_gather_sm100_task(threadblock::Graph const &bgraph,
                                        std::vector<int> const &params);
  int register_mla_kv_gather_split_sm100_task(threadblock::Graph const &bgraph,
                                              std::vector<int> const &params);
  int register_mla_kv_gather_unified_sm100_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_deepseek_mla_rope_q_sm100_task(threadblock::Graph const &bgraph,
                                              std::vector<int> const &params);
  int register_deepseek_mla_rope_q_fused_sm100_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_deepseek_mla_rope_q_split_sm100_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_deepseek_mla_rope_k_sm100_task(threadblock::Graph const &bgraph,
                                              std::vector<int> const &params);
  // MTP tasks
  int register_mtp_verify_strict_task(threadblock::Graph const &bgraph,
                                      std::vector<int> const &params);
  int register_mtp_accept_commit_task(threadblock::Graph const &bgraph,
                                      std::vector<int> const &params);
  int register_mtp_token_scatter_task(threadblock::Graph const &bgraph,
                                      std::vector<int> const &params);
  int register_mtp_prepare_verify_task(threadblock::Graph const &bgraph,
                                       std::vector<int> const &params);
  int register_mtp_build_embed_input_task(threadblock::Graph const &bgraph,
                                          std::vector<int> const &params);
  // Eagle3 tasks
  int register_copy_task(threadblock::Graph const &bgraph,
                         std::vector<int> const &params);
  int register_concat_task(threadblock::Graph const &bgraph,
                           std::vector<int> const &params);
  int register_eagle3_d2t_remap_task(threadblock::Graph const &bgraph,
                                     std::vector<int> const &params);
  int register_eagle3_commit_task(threadblock::Graph const &bgraph,
                                  std::vector<int> const &params);
  // Eagle3 tasks end
  // SM100 tasks end
  // Multi-GPU tasks
  int register_nvshmem_allgather_strided_put_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_nvshmem_tile_allreduce_task(threadblock::Graph const &bgraph,
                                           std::vector<int> const &params);
  int register_nvshmem_global_argmax_task(threadblock::Graph const &bgraph,
                                          std::vector<int> const &params);
  // Multi-GPU tasks end
  int register_task_variant(TaskType type, std::string const &code);
  void register_v2_task_role_variant(TaskType type,
                                     int variant_id,
                                     TaskRoleVariantCode code);

  // Register the total SMEM bytes a (TaskType, variant_id) consumes. Current
  // v2 codegen publishes this as metadata while keeping executable task bases
  // at offset 0. A future per-SM allocator can use these sizes for placement
  // once every task body supports non-zero offsets.
  void register_variant_smem_size(TaskType type, int variant_id, int size);
  void register_variant_smem_info(TaskType type,
                                  int variant_id,
                                  TaskSmemInfo info);
  TaskSmemInfo get_variant_smem_info(TaskType type, int variant_id) const;
  int get_variant_smem_size(TaskType type, int variant_id) const;

public:
  std::map<TaskType, std::vector<std::string>> all_task_variants;
  std::map<TaskType, std::vector<TaskRoleVariantCode>>
      all_v2_task_role_variants;
  std::map<TaskType, std::vector<TaskSmemInfo>> all_task_variant_smem_infos;
};

} // namespace runtime
} // namespace mirage
