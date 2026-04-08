"""DeepSeek V3 model builder for Mirage MPK with MTP support.

Architecture: 61 decoder layers with MLA attention and MoE MLP.
- Layers 0-2: Dense MLP (DeepseekV2MLP)
- Layers 3-60: MoE MLP (256 experts, top-8, + shared experts)
- MLA: 128 Q heads, 1 KV head after weight absorption, head_dim=576 (512+64)
- Optional MTP: 1 predictor layer for multi-token prediction

Weight absorption: at load time, kv_b_proj is absorbed into q_b_proj so that
runtime only needs compressed KV cache [c_latent(512), k_pe(64)] = 576 dims.
"""

import torch
from typing import Optional

from ..utils import grid_for_rmsnorm_linear_layer
from ..graph_builder import GraphBuilder, MirageModelConfig
from ...persistent_kernel import PersistentKernel
from ...model_registry import register_model_builder
from ....core import bfloat16, float8_e4m3, float32, int64


# DeepSeek V3 architecture constants
HIDDEN_SIZE = 7168
NUM_LAYERS = 61
NUM_Q_HEADS = 128         # total Q heads
Q_LORA_RANK = 1536        # q_a_proj output dim
KV_LORA_RANK = 512        # c_latent dim
QK_NOPE_HEAD_DIM = 128    # per-head nope dim
QK_ROPE_HEAD_DIM = 64     # per-head rope dim
V_HEAD_DIM = 128          # per-head value dim (before absorption)
QK_HEAD_DIM_TOTAL = 576   # 512 latent + 64 rope (after absorption)
V_HEAD_DIM_TOTAL = 512    # latent dim only (after absorption)
INTERMEDIATE_SIZE = 18432       # Dense MLP intermediate (layers 0-2)
MOE_INTERMEDIATE_SIZE = 2048    # Per-expert intermediate (routed + shared)
NUM_EXPERTS = 256
NUM_EXPERTS_PER_TOK = 8
NUM_SHARED_EXPERTS = 1
FIRST_MOE_LAYER = 3
VOCAB_SIZE = 129280
RMS_NORM_EPS = 1e-6


@register_model_builder("deepseek-v3", "DeepSeek-V3", "deepseek-ai/DeepSeek-V3")
class DeepSeekV3Builder(GraphBuilder):
    def __init__(self, mpk: PersistentKernel, weights: Optional[dict] = None):
        super().__init__(mpk, weights)
        self.max_num_pages = mpk.max_num_pages
        self.page_size = mpk.page_size
        self.world_size = mpk.world_size
        self.input_tokens = mpk.meta_tensors["input_tokens"]
        self.output_tokens = mpk.meta_tensors["output_tokens"]
        self.rank = mpk.mpi_rank
        self.max_num_batched_tokens = mpk.max_num_batched_tokens

        # DeepSeek V3 dimensions
        self.hidden_size = HIDDEN_SIZE
        self.num_layers = NUM_LAYERS
        self.num_q_heads = NUM_Q_HEADS
        self.num_local_q_heads = NUM_Q_HEADS // self.world_size
        self.qk_head_dim = QK_HEAD_DIM_TOTAL  # 576 after absorption
        self.v_head_dim = V_HEAD_DIM_TOTAL     # 512 after absorption
        self.q_lora_rank = Q_LORA_RANK
        self.kv_lora_rank = KV_LORA_RANK
        self.intermediate_size = INTERMEDIATE_SIZE // self.world_size
        self.moe_intermediate_size = MOE_INTERMEDIATE_SIZE // self.world_size

        # MTP config
        self.mtp_config = getattr(mpk, 'spec_decode_config', None)

    def build_from_model(self, model_name: str, model_path: str = None):
        raise NotImplementedError(
            "DeepSeek V3 is too large for direct HuggingFace loading. "
            "Use build_from_config() with pre-converted weights."
        )

    def build_from_config(self, model_config: MirageModelConfig, layer_indices: list = None):
        """Build from pre-processed config with absorbed weights.

        Args:
            layer_indices: If provided, only build these specific layer indices.
        """
        self.ckv_kpe_cache = model_config.k_cache  # [num_layers, num_pages, page_size, 576]
        self.position_embeddings = model_config.position_embeddings

        self.build_from_dict(
            model_config.state_dict,
            model_config.with_lm_head,
            layer_indices=layer_indices,
        )

    def _fp8_linear(self, input_bf16, weight_fp8, weight_scale, output,
                     grid_dim, block_dim, residual=None):
        """Quantize BF16 input → FP8, then run FP8 GEMM.

        If weight is BF16 (no scale), falls back to BF16 linear.
        Handles the quantize → gemm pipeline automatically.
        """
        mbt = self.max_num_batched_tokens
        reduction_size = weight_fp8.dim(1) if weight_fp8.num_dims == 2 else weight_fp8.dim(-1)
        group_size = 128
        num_groups = (reduction_size + group_size - 1) // group_size

        # Allocate quantize output buffers (reusable)
        if not hasattr(self, '_fp8_input_buf') or self._fp8_input_buf.dim(1) != reduction_size:
            self._fp8_input_buf = self.mpk.new_tensor(
                dims=(mbt, reduction_size), dtype=bfloat16,  # placeholder dtype, actual is fp8
                name=f"fp8_input_{reduction_size}",
                io_category="cuda_tensor",
            )
            self._fp8_scale_buf = self.mpk.new_tensor(
                dims=(mbt, num_groups), dtype=bfloat16,  # placeholder, actual is uint32
                name=f"fp8_scale_{reduction_size}",
                io_category="cuda_tensor",
            )

        # Step 1: Quantize input BF16 → FP8
        self.mpk.quantize_fp8_layer(
            input=input_bf16,
            output_fp8=self._fp8_input_buf,
            output_scale=self._fp8_scale_buf,
            grid_dim=(mbt, 1, 1),
            block_dim=(128, 1, 1),
        )

        # Step 2: FP8 GEMM
        if residual is not None:
            self.mpk.linear_fp8_with_residual_layer(
                input_fp8=self._fp8_input_buf,
                input_scale=self._fp8_scale_buf,
                weight_fp8=weight_fp8,
                weight_scale=weight_scale,
                residual=residual,
                output=output,
                grid_dim=grid_dim,
                block_dim=block_dim,
            )
        else:
            self.mpk.linear_fp8_layer(
                input_fp8=self._fp8_input_buf,
                input_scale=self._fp8_scale_buf,
                weight_fp8=weight_fp8,
                weight_scale=weight_scale,
                output=output,
                grid_dim=grid_dim,
                block_dim=block_dim,
            )

    def _new_intermediate_tensors(self):
        """Allocate intermediate computation buffers."""
        mbt = self.max_num_batched_tokens

        # RMSNorm output
        self.rmsnorm_out = self.mpk.new_tensor(
            dims=(mbt, self.hidden_size),
            dtype=bfloat16,
            name="rmsnorm_out",
            io_category="cuda_tensor",
        )

        # MLA projections
        # q_a output: [batch, q_lora_rank]
        self.q_a_out = self.mpk.new_tensor(
            dims=(mbt, self.q_lora_rank),
            dtype=bfloat16,
            name="q_a_out",
            io_category="cuda_tensor",
        )
        # q_b output (after absorption): [batch, num_local_q_heads * qk_head_dim]
        self.q_nope_pe = self.mpk.new_tensor(
            dims=(mbt, self.num_local_q_heads * self.qk_head_dim),
            dtype=bfloat16,
            name="q_nope_pe",
            io_category="cuda_tensor",
        )
        # kv_a output split: c_latent [batch, 512] and k_pe [batch, 64]
        # We use two separate linear layers instead of one 576-dim output,
        # so we can apply kv_a_layernorm to c_latent only.
        self.c_latent_out = self.mpk.new_tensor(
            dims=(mbt, self.kv_lora_rank),  # [batch, 512]
            dtype=bfloat16,
            name="c_latent_out",
            io_category="cuda_tensor",
        )
        self.k_pe_out = self.mpk.new_tensor(
            dims=(mbt, QK_ROPE_HEAD_DIM),  # [batch, 64]
            dtype=bfloat16,
            name="k_pe_out",
            io_category="cuda_tensor",
        )
        # Combined KV entry after layernorm: [batch, 576]
        self.kv_combined = self.mpk.new_tensor(
            dims=(mbt, self.qk_head_dim),  # [batch, 576]
            dtype=bfloat16,
            name="kv_combined",
            io_category="cuda_tensor",
        )
        # Attention output: [batch, num_local_q_heads * v_head_dim]
        self.attn_out = self.mpk.new_tensor(
            dims=(mbt, self.num_local_q_heads * self.v_head_dim),
            dtype=bfloat16,
            name="attn_out",
            io_category="cuda_tensor",
        )
        # O projection output (same as hidden_size)
        self.attn_proj_out = self.mpk.new_tensor(
            dims=(mbt, self.hidden_size),
            dtype=bfloat16,
            name="attn_proj_out",
            io_category="cuda_tensor",
        )

        # MLP intermediates
        # Dense MLP: gate+up = 2 * intermediate_size
        self.mlp_mid = self.mpk.new_tensor(
            dims=(mbt, 2 * self.intermediate_size),
            dtype=bfloat16,
            name="mlp_mid",
            io_category="cuda_tensor",
        )
        self.silu_mul_out = self.mpk.new_tensor(
            dims=(mbt, self.intermediate_size),
            dtype=bfloat16,
            name="silu_mul_out",
            io_category="cuda_tensor",
        )
        self.mlp_out = self.mpk.new_tensor(
            dims=(mbt, self.hidden_size),
            dtype=bfloat16,
            name="mlp_out",
            io_category="cuda_tensor",
        )

        # AllReduce buffer
        if self.world_size > 1:
            self.allreduce_buf = self.mpk.new_tensor(
                dims=(self.world_size, mbt, self.hidden_size),
                dtype=bfloat16,
                name="allreduce_buf",
                io_category="nvshmem_tensor",
            )
            self.allreduce_out = self.mpk.new_tensor(
                dims=(mbt, self.hidden_size),
                dtype=bfloat16,
                name="allreduce_out",
                io_category="cuda_tensor",
            )

        # Argmax
        self.argmax_part_value = self.mpk.new_tensor(
            dims=(mbt, self.mpk.num_workers),
            dtype=bfloat16,
            name="argmax_part_value",
            io_category="cuda_tensor",
        )
        self.argmax_part_index = self.mpk.new_tensor(
            dims=(mbt, self.mpk.num_workers),
            dtype=int64,
            name="argmax_part_index",
            io_category="cuda_tensor",
        )

    def _safe_attach(self, tensor, name):
        """Attach tensor. FP8 is now natively supported in core.pyx."""
        return self.mpk.attach_input(torch_tensor=tensor, name=name)

    def _attach_fp8_weight(self, state_dict, key, name):
        """Attach FP8 weight + its packed UE8M0 scale_inv."""
        w = self._safe_attach(state_dict[key], name)
        s = self._safe_attach(state_dict[f"{key}_scale_inv"], f"{name}_scale")
        return w, s

    def _build_mla_attention_layer(self, layer_idx: int, state_dict: dict):
        """Build MLA attention for one decoder layer (FP8 weights)."""
        prefix = f"model.layers.{layer_idx}."
        attn = f"{prefix}self_attn."

        # Step 1: q_a_proj (FP8)
        w_q_a, s_q_a = self._attach_fp8_weight(
            state_dict, f"{attn}q_a_proj.weight", f"layer_{layer_idx}_q_a_proj")
        self._fp8_linear(self.rmsnorm_out, w_q_a, s_q_a, self.q_a_out,
                         grid_dim=(grid_for_rmsnorm_linear_layer(w_q_a.dim(0)), 1, 1),
                         block_dim=(128, 1, 1))

        # Step 2: q_a_layernorm (BF16 norm weight)
        w_q_a_ln = self.mpk.attach_input(
            torch_tensor=state_dict[f"{attn}q_a_layernorm.weight"],
            name=f"layer_{layer_idx}_q_a_layernorm")
        self.mpk.rmsnorm_layer(
            input=self.q_a_out, weight=w_q_a_ln, output=self.q_a_out,
            grid_dim=(self.max_num_batched_tokens, 1, 1), block_dim=(128, 1, 1))

        # Step 3: q_b_proj absorbed (FP8)
        w_q_b, s_q_b = self._attach_fp8_weight(
            state_dict, f"{attn}q_b_proj.weight", f"layer_{layer_idx}_q_b_proj")
        self._fp8_linear(self.q_a_out, w_q_b, s_q_b, self.q_nope_pe,
                         grid_dim=(grid_for_rmsnorm_linear_layer(w_q_b.dim(0)), 1, 1),
                         block_dim=(128, 1, 1))

        # Step 4: kv_a_proj split into c_latent and k_pe (FP8)
        # Split the [576, hidden] weight into [512, hidden] and [64, hidden]
        # Also split the scale_inv accordingly
        kv_a_w = state_dict[f"{attn}kv_a_proj_with_mqa.weight"]
        kv_a_s = state_dict[f"{attn}kv_a_proj_with_mqa.weight_scale_inv"]
        # Determine scale row split based on output dim ratio
        scale_rows_total = kv_a_s.shape[0]
        latent_ratio = self.kv_lora_rank / (self.kv_lora_rank + QK_ROPE_HEAD_DIM)
        scale_rows_latent = round(scale_rows_total * latent_ratio)

        w_kv_latent = self._safe_attach(
            kv_a_w[:self.kv_lora_rank].contiguous(),
            f"layer_{layer_idx}_kv_a_latent")
        s_kv_latent = self._safe_attach(
            kv_a_s[:scale_rows_latent].contiguous(),
            f"layer_{layer_idx}_kv_a_latent_scale")
        # kv_a_rope: output=64, not 128-aligned for FP8 GEMM. Use BF16 dequant.
        kv_rope_w_fp8 = kv_a_w[self.kv_lora_rank:].contiguous()
        kv_rope_s = kv_a_s[scale_rows_latent:].contiguous()
        # Dequantize FP8 weight to BF16 for this small projection
        kv_rope_w_bf16 = (kv_rope_w_fp8.float() * kv_rope_s.float().repeat_interleave(
            128, dim=-1)[:, :kv_rope_w_fp8.shape[-1]]).to(torch.bfloat16)
        w_kv_rope = self.mpk.attach_input(
            torch_tensor=kv_rope_w_bf16,
            name=f"layer_{layer_idx}_kv_a_rope")

        self._fp8_linear(self.rmsnorm_out, w_kv_latent, s_kv_latent, self.c_latent_out,
                         grid_dim=(grid_for_rmsnorm_linear_layer(self.kv_lora_rank), 1, 1),
                         block_dim=(128, 1, 1))
        # BF16 linear for rope projection (64 output dims, not FP8-aligned)
        self.mpk.linear_layer(
            input=self.rmsnorm_out, weight=w_kv_rope, output=self.k_pe_out,
            grid_dim=(grid_for_rmsnorm_linear_layer(QK_ROPE_HEAD_DIM), 1, 1),
            block_dim=(128, 1, 1))

        # Step 5: kv_a_layernorm on c_latent ONLY
        w_kv_a_ln = self.mpk.attach_input(
            torch_tensor=state_dict[f"{attn}kv_a_layernorm.weight"],
            name=f"layer_{layer_idx}_kv_a_layernorm")
        self.mpk.rmsnorm_layer(
            input=self.c_latent_out, weight=w_kv_a_ln, output=self.c_latent_out,
            grid_dim=(self.max_num_batched_tokens, 1, 1), block_dim=(128, 1, 1))

        # Step 6: MLA paged attention
        cache = self.mpk.attach_input(
            torch_tensor=self.ckv_kpe_cache[layer_idx],
            name=f"layer_{layer_idx}_ckv_kpe_cache")
        self.mpk.paged_mla_layer(
            q_nope_pe=self.q_nope_pe, ckv_kpe_cache=cache,
            c_latent_new=self.c_latent_out, k_pe_new=self.k_pe_out,
            output=self.attn_out,
            grid_dim=(self.mpk.max_num_batched_requests, 1, 1),
            block_dim=(128, 1, 1),
            num_q_heads=self.num_local_q_heads,
            qk_head_dim=self.qk_head_dim, v_head_dim=self.v_head_dim)

        # Step 7: O projection (FP8)
        w_o, s_o = self._attach_fp8_weight(
            state_dict, f"{attn}o_proj.weight", f"layer_{layer_idx}_o_proj")
        self._fp8_linear(self.attn_out, w_o, s_o, self.attn_proj_out,
                         grid_dim=(self.hidden_size // 128, 128 * 128 // self.hidden_size, 1),
                         block_dim=(256, 1, 1))

    def _build_dense_mlp(self, layer_idx: int, state_dict: dict):
        """Build dense MLP for layers 0-2 (FP8 weights)."""
        prefix = f"model.layers.{layer_idx}."

        w_gate_up, s_gate_up = self._attach_fp8_weight(
            state_dict, f"{prefix}mlp.gate_up_proj.weight",
            f"layer_{layer_idx}_gate_up_proj")
        self._fp8_linear(self.rmsnorm_out, w_gate_up, s_gate_up, self.mlp_mid,
                         grid_dim=(grid_for_rmsnorm_linear_layer(w_gate_up.dim(0)), 1, 1),
                         block_dim=(128, 1, 1))
        self.mpk.silu_mul_layer(
            input=self.mlp_mid, output=self.silu_mul_out,
            grid_dim=(self.intermediate_size // 64, 1, 1), block_dim=(128, 1, 1))
        w_down, s_down = self._attach_fp8_weight(
            state_dict, f"{prefix}mlp.down_proj.weight",
            f"layer_{layer_idx}_down_proj")
        self._fp8_linear(self.silu_mul_out, w_down, s_down, self.mlp_out,
                         grid_dim=(self.hidden_size // 128, 128 * 128 // self.hidden_size, 1),
                         block_dim=(256, 1, 1))

    def _build_moe_mlp(self, layer_idx: int, state_dict: dict):
        """Build MoE MLP for layers 3-60.

        Uses existing MoE task infrastructure:
        moe_topk_softmax → moe_w13_linear → silu_mul → moe_w2_linear → mul_sum_add
        """
        prefix = f"model.layers.{layer_idx}.mlp."

        # Router
        w_gate = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}gate.weight"],
            name=f"layer_{layer_idx}_moe_gate",
        )

        # MoE routing tensors
        moe_topk_weights = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, NUM_EXPERTS_PER_TOK),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_moe_topk_weights",
            io_category="cuda_tensor",
        )
        moe_routing_indices = self.mpk.new_tensor(
            dims=(NUM_EXPERTS, self.max_num_batched_tokens),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_moe_routing_indices",
            io_category="cuda_tensor",
        )
        moe_mask = self.mpk.new_tensor(
            dims=(NUM_EXPERTS + 1, 1),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_moe_mask",
            io_category="cuda_tensor",
        )

        # Router logits → topk routing
        router_logits = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, NUM_EXPERTS),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_router_logits",
            io_category="cuda_tensor",
        )
        self.mpk.linear_layer(
            input=self.rmsnorm_out,
            weight=w_gate,
            output=router_logits,
            grid_dim=(grid_for_rmsnorm_linear_layer(w_gate.dim(0)), 1, 1),
            block_dim=(128, 1, 1),
        )

        # Initialize MoE output tensor
        moe_output = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.hidden_size),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_moe_output",
            io_category="cuda_tensor",
        )
        self.mpk.tensor_init_layer(
            input=moe_output,
            dummy_input=self.rmsnorm_out,
            dummy_output=self.rmsnorm_out,
            grid_dim=(self.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )

        # TopK sigmoid routing (DeepSeek V3: scoring_func=sigmoid)
        # e_score_correction_bias is added to sigmoid scores for routing selection
        bias_key = f"{prefix}gate.e_score_correction_bias"
        w_bias = self.mpk.attach_input(
            torch_tensor=state_dict[bias_key],
            name=f"layer_{layer_idx}_moe_gate_bias",
        )
        self.mpk.moe_topk_sigmoid_routing_layer(
            input=router_logits,
            bias=w_bias,
            output=(moe_topk_weights, moe_routing_indices, moe_mask),
            grid_dim=(1, 1, 1),
            block_dim=(256, 1, 1),  # 8 warps required by topk kernel
        )

        # Expert W1+W3 (gate + up projection) — FP8
        w_experts_w13 = self._safe_attach(
            state_dict[f"{prefix}experts.w13.weight"],
            f"layer_{layer_idx}_experts_w13")
        s_experts_w13 = self._safe_attach(
            state_dict[f"{prefix}experts.w13.weight_scale_inv"],
            f"layer_{layer_idx}_experts_w13_scale")
        # Quantize input for MoE FP8
        mbt = self.max_num_batched_tokens
        moe_input_fp8 = self.mpk.new_tensor(
            dims=(mbt, self.hidden_size), dtype=bfloat16,
            name=f"layer_{layer_idx}_moe_input_fp8", io_category="cuda_tensor",
        )
        moe_input_scale = self.mpk.new_tensor(
            dims=(mbt, self.hidden_size // 128), dtype=bfloat16,
            name=f"layer_{layer_idx}_moe_input_scale", io_category="cuda_tensor",
        )
        self.mpk.quantize_fp8_layer(
            input=self.rmsnorm_out,
            output_fp8=moe_input_fp8,
            output_scale=moe_input_scale,
            grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1),
        )

        moe_mid = self.mpk.new_tensor(
            dims=(mbt, NUM_EXPERTS_PER_TOK, 2 * self.moe_intermediate_size),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_moe_mid",
            io_category="cuda_tensor",
        )
        self.mpk.moe_w13_fp8_layer(
            input_fp8=moe_input_fp8,
            input_scale=moe_input_scale,
            weight_fp8=w_experts_w13,
            weight_scale=s_experts_w13,
            moe_routing_indices=moe_routing_indices,
            moe_mask=moe_mask,
            output=moe_mid,
            grid_dim=(NUM_EXPERTS, 1, 1),
            block_dim=(128, 1, 1),
        )

        # SiLU activation
        moe_silu_out = self.mpk.new_tensor(
            dims=(mbt, NUM_EXPERTS_PER_TOK, self.moe_intermediate_size),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_moe_silu",
            io_category="cuda_tensor",
        )
        self.mpk.moe_silu_mul_layer(
            input=moe_mid, output=moe_silu_out,
            grid_dim=(mbt * NUM_EXPERTS_PER_TOK, 1, 1),
            block_dim=(128, 1, 1),
        )

        # Expert W2 (down projection) — FP8
        # Quantize 3D silu_out [batch, topk, intermediate] — quantize kernel
        # flattens to [batch*topk, intermediate] internally
        w_experts_w2 = self._safe_attach(
            state_dict[f"{prefix}experts.w2.weight"],
            f"layer_{layer_idx}_experts_w2")
        s_experts_w2 = self._safe_attach(
            state_dict[f"{prefix}experts.w2.weight_scale_inv"],
            f"layer_{layer_idx}_experts_w2_scale")
        moe_silu_fp8 = self.mpk.new_tensor(
            dims=(mbt, NUM_EXPERTS_PER_TOK, self.moe_intermediate_size),
            dtype=float8_e4m3,
            name=f"layer_{layer_idx}_moe_silu_fp8",
            io_category="cuda_tensor",
        )
        moe_silu_scale = self.mpk.new_tensor(
            dims=(mbt, NUM_EXPERTS_PER_TOK, self.moe_intermediate_size // 128),
            dtype=float32,
            name=f"layer_{layer_idx}_moe_silu_scale",
            io_category="cuda_tensor",
        )
        self.mpk.quantize_fp8_layer(
            input=moe_silu_out,
            output_fp8=moe_silu_fp8,
            output_scale=moe_silu_scale,
            grid_dim=(mbt * NUM_EXPERTS_PER_TOK, 1, 1),
            block_dim=(128, 1, 1),
        )
        moe_down_out = self.mpk.new_tensor(
            dims=(mbt, NUM_EXPERTS_PER_TOK, self.hidden_size),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_moe_down",
            io_category="cuda_tensor",
        )
        self.mpk.moe_w2_fp8_layer(
            input_fp8=moe_silu_fp8,
            input_scale=moe_silu_scale,
            weight_fp8=w_experts_w2,
            weight_scale=s_experts_w2,
            moe_routing_indices=moe_routing_indices,
            moe_mask=moe_mask,
            output=moe_down_out,
            grid_dim=(NUM_EXPERTS, 1, 1),
            block_dim=(128, 1, 1),
        )

        # ---- Shared Expert (1 expert, TP parallel, same as dense MLP) ----
        # Shared expert runs on ALL tokens independently of routing.
        # Its output is added to the residual before the routed expert reduction:
        #   final = sum(routed * weights) + (residual + shared_expert_out)
        shared_prefix = f"{prefix}shared_experts."

        # gate_proj + up_proj fused (FP8)
        # Concatenate gate and up FP8 weights + scales
        shared_gate_w = state_dict[f"{shared_prefix}gate_proj.weight"]
        shared_up_w = state_dict[f"{shared_prefix}up_proj.weight"]
        shared_gate_s = state_dict[f"{shared_prefix}gate_proj.weight_scale_inv"]
        shared_up_s = state_dict[f"{shared_prefix}up_proj.weight_scale_inv"]
        w_shared_gate_up = self._safe_attach(
            torch.cat([shared_gate_w, shared_up_w], dim=0),
            f"layer_{layer_idx}_shared_expert_gate_up")
        s_shared_gate_up = self._safe_attach(
            torch.cat([shared_gate_s, shared_up_s], dim=0),
            f"layer_{layer_idx}_shared_expert_gate_up_scale")
        shared_mid = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, 2 * self.moe_intermediate_size),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_shared_mid",
            io_category="cuda_tensor",
        )
        self._fp8_linear(self.rmsnorm_out, w_shared_gate_up, s_shared_gate_up,
                         shared_mid,
                         grid_dim=(grid_for_rmsnorm_linear_layer(
                             w_shared_gate_up.dim(0)), 1, 1),
                         block_dim=(128, 1, 1))

        # silu_mul
        shared_silu_out = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.moe_intermediate_size),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_shared_silu",
            io_category="cuda_tensor",
        )
        self.mpk.silu_mul_layer(
            input=shared_mid, output=shared_silu_out,
            grid_dim=(self.moe_intermediate_size // 64, 1, 1),
            block_dim=(128, 1, 1))

        # down_proj with residual (FP8): shared_residual = self.x + shared_down(shared_silu)
        w_shared_down, s_shared_down = self._attach_fp8_weight(
            state_dict, f"{shared_prefix}down_proj.weight",
            f"layer_{layer_idx}_shared_expert_down")
        shared_residual = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.hidden_size),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_shared_residual",
            io_category="cuda_tensor",
        )
        self._fp8_linear(shared_silu_out, w_shared_down, s_shared_down,
                         shared_residual,
                         grid_dim=(self.hidden_size // 64, 1, 1),
                         block_dim=(128, 1, 1),
                         residual=self.x)

        # Final: moe_output = sum(routed_experts * weights) + shared_residual
        # where shared_residual = original_hidden + shared_expert_output
        self.mpk.moe_mul_sum_add_layer(
            input=moe_down_out,
            weight=moe_topk_weights,
            residual=shared_residual,
            output=moe_output,
            grid_dim=(self.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )
        self.mlp_out = moe_output

    def _build_mtp_decoder_layer(self, state_dict: dict, prefix: str):
        """Build one MTP decoder layer (same structure as main model layer).

        The MTP block is a full DeepseekV2DecoderLayer with its own weights.
        It shares the same architecture: input_layernorm → MLA → post_norm → MLP.
        """
        # Input layernorm
        w_norm = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}input_layernorm.weight"],
            name="mtp_block_input_layernorm",
        )
        self.mpk.rmsnorm_layer(
            input=self.mtp_x, weight=w_norm, output=self.rmsnorm_out,
            grid_dim=(self.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )

        # MLA attention (same structure as main model, own weights)
        self._build_mla_attention_layer_with_prefix(prefix, state_dict)

        # Residual (attn output is in self.attn_proj_out)
        self.mtp_x = self.attn_proj_out

        # AllReduce after attention
        if self.world_size > 1:
            self.mpk.allreduce_layer(
                input=self.attn_proj_out, buffer=self.allreduce_buf,
                output=self.allreduce_out,
                grid_dim=(self.hidden_size // 64, 1, 1),
                block_dim=(128, 1, 1),
            )
            self.mtp_x = self.allreduce_out

        # Post-attention layernorm
        w_post_norm = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}post_attention_layernorm.weight"],
            name="mtp_block_post_attn_layernorm",
        )
        self.mpk.rmsnorm_layer(
            input=self.mtp_x, weight=w_post_norm, output=self.rmsnorm_out,
            grid_dim=(self.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )

        # MLP: DeepSeek V3 MTP block uses MoE MLP (same as main layers 3-60)
        # Check if MoE weights exist, fallback to dense
        mlp_gate_key = f"{prefix}mlp.gate.weight"
        if mlp_gate_key in state_dict:
            self._build_moe_mlp_with_prefix(prefix, state_dict)
        else:
            self._build_dense_mlp_with_prefix(prefix, state_dict)

        self.mtp_x = self.mlp_out
        if self.world_size > 1:
            self.mpk.allreduce_layer(
                input=self.mlp_out, buffer=self.allreduce_buf,
                output=self.allreduce_out,
                grid_dim=(self.hidden_size // 64, 1, 1),
                block_dim=(128, 1, 1),
            )
            self.mtp_x = self.allreduce_out

    def _build_mla_attention_layer_with_prefix(self, prefix: str, state_dict: dict):
        """Build MLA attention using a custom weight prefix (FP8, for MTP reuse)."""
        attn = f"{prefix}self_attn."

        # q_a_proj (FP8)
        w_q_a, s_q_a = self._attach_fp8_weight(
            state_dict, f"{attn}q_a_proj.weight", f"mtp_{attn}q_a_proj")
        self._fp8_linear(self.rmsnorm_out, w_q_a, s_q_a, self.q_a_out,
                         grid_dim=(grid_for_rmsnorm_linear_layer(w_q_a.dim(0)), 1, 1),
                         block_dim=(128, 1, 1))

        w_q_a_ln = self.mpk.attach_input(
            torch_tensor=state_dict[f"{attn}q_a_layernorm.weight"],
            name=f"mtp_{attn}q_a_layernorm")
        self.mpk.rmsnorm_layer(
            input=self.q_a_out, weight=w_q_a_ln, output=self.q_a_out,
            grid_dim=(self.max_num_batched_tokens, 1, 1), block_dim=(128, 1, 1))

        # q_b_proj (FP8)
        w_q_b, s_q_b = self._attach_fp8_weight(
            state_dict, f"{attn}q_b_proj.weight", f"mtp_{attn}q_b_proj")
        self._fp8_linear(self.q_a_out, w_q_b, s_q_b, self.q_nope_pe,
                         grid_dim=(grid_for_rmsnorm_linear_layer(w_q_b.dim(0)), 1, 1),
                         block_dim=(128, 1, 1))

        # kv_a_proj split (FP8)
        kv_a_w = state_dict[f"{attn}kv_a_proj_with_mqa.weight"]
        kv_a_s = state_dict[f"{attn}kv_a_proj_with_mqa.weight_scale_inv"]
        scale_rows_total = kv_a_s.shape[0]
        latent_ratio = self.kv_lora_rank / (self.kv_lora_rank + QK_ROPE_HEAD_DIM)
        scale_rows_latent = round(scale_rows_total * latent_ratio)

        w_kv_latent = self._safe_attach(
            kv_a_w[:self.kv_lora_rank].contiguous(), f"mtp_{attn}kv_a_latent")
        s_kv_latent = self._safe_attach(
            kv_a_s[:scale_rows_latent].contiguous(), f"mtp_{attn}kv_a_latent_scale")
        # kv_a_rope: dequant to BF16 (output=64, not 128-aligned for FP8)
        kv_rope_w_fp8 = kv_a_w[self.kv_lora_rank:].contiguous()
        kv_rope_s = kv_a_s[scale_rows_latent:].contiguous()
        kv_rope_w_bf16 = (kv_rope_w_fp8.float() * kv_rope_s.float().repeat_interleave(
            128, dim=-1)[:, :kv_rope_w_fp8.shape[-1]]).to(torch.bfloat16)
        w_kv_rope = self.mpk.attach_input(
            torch_tensor=kv_rope_w_bf16, name=f"mtp_{attn}kv_a_rope")

        self._fp8_linear(self.rmsnorm_out, w_kv_latent, s_kv_latent, self.c_latent_out,
                         grid_dim=(grid_for_rmsnorm_linear_layer(self.kv_lora_rank), 1, 1),
                         block_dim=(128, 1, 1))
        self.mpk.linear_layer(
            input=self.rmsnorm_out, weight=w_kv_rope, output=self.k_pe_out,
            grid_dim=(grid_for_rmsnorm_linear_layer(QK_ROPE_HEAD_DIM), 1, 1),
            block_dim=(128, 1, 1))

        w_kv_a_ln = self.mpk.attach_input(
            torch_tensor=state_dict[f"{attn}kv_a_layernorm.weight"],
            name=f"mtp_{attn}kv_a_layernorm")
        self.mpk.rmsnorm_layer(
            input=self.c_latent_out, weight=w_kv_a_ln, output=self.c_latent_out,
            grid_dim=(self.max_num_batched_tokens, 1, 1), block_dim=(128, 1, 1))

        # MTP attention uses its own KV cache
        self.mpk.paged_mla_layer(
            q_nope_pe=self.q_nope_pe, ckv_kpe_cache=self.mtp_ckv_kpe_cache_tensor,
            c_latent_new=self.c_latent_out, k_pe_new=self.k_pe_out,
            output=self.attn_out,
            grid_dim=(self.mpk.max_num_batched_requests, 1, 1),
            block_dim=(128, 1, 1),
            num_q_heads=self.num_local_q_heads,
            qk_head_dim=self.qk_head_dim, v_head_dim=self.v_head_dim)

        # o_proj (FP8)
        w_o, s_o = self._attach_fp8_weight(
            state_dict, f"{attn}o_proj.weight", f"mtp_{attn}o_proj")
        self._fp8_linear(self.attn_out, w_o, s_o, self.attn_proj_out,
                         grid_dim=(self.hidden_size // 128, 128 * 128 // self.hidden_size, 1),
                         block_dim=(256, 1, 1))

    def _build_dense_mlp_with_prefix(self, prefix: str, state_dict: dict):
        """Build dense MLP using a custom weight prefix (FP8, for MTP reuse)."""
        mlp_prefix = f"{prefix}mlp."

        w_gate_up, s_gate_up = self._attach_fp8_weight(
            state_dict, f"{mlp_prefix}gate_up_proj.weight",
            f"mtp_{mlp_prefix}gate_up_proj")
        self._fp8_linear(self.rmsnorm_out, w_gate_up, s_gate_up, self.mlp_mid,
                         grid_dim=(grid_for_rmsnorm_linear_layer(w_gate_up.dim(0)), 1, 1),
                         block_dim=(128, 1, 1))
        self.mpk.silu_mul_layer(
            input=self.mlp_mid, output=self.silu_mul_out,
            grid_dim=(self.intermediate_size // 64, 1, 1), block_dim=(128, 1, 1))
        w_down, s_down = self._attach_fp8_weight(
            state_dict, f"{mlp_prefix}down_proj.weight",
            f"mtp_{mlp_prefix}down_proj")
        self._fp8_linear(self.silu_mul_out, w_down, s_down, self.mlp_out,
                         grid_dim=(self.hidden_size // 128, 128 * 128 // self.hidden_size, 1),
                         block_dim=(256, 1, 1))

    def _build_moe_mlp_with_prefix(self, prefix: str, state_dict: dict):
        """Build MoE MLP using a custom weight prefix (FP8, for MTP reuse)."""
        mlp_prefix = f"{prefix}mlp."
        mbt = self.max_num_batched_tokens

        # Router (BF16 — gate.weight is BF16)
        w_gate = self.mpk.attach_input(
            torch_tensor=state_dict[f"{mlp_prefix}gate.weight"],
            name=f"mtp_{mlp_prefix}gate")
        moe_topk_weights = self.mpk.new_tensor(
            dims=(mbt, NUM_EXPERTS_PER_TOK), dtype=bfloat16,
            name="mtp_moe_topk_weights", io_category="cuda_tensor")
        moe_routing_indices = self.mpk.new_tensor(
            dims=(NUM_EXPERTS, mbt), dtype=bfloat16,
            name="mtp_moe_routing_indices", io_category="cuda_tensor")
        moe_mask = self.mpk.new_tensor(
            dims=(NUM_EXPERTS + 1, 1), dtype=bfloat16,
            name="mtp_moe_mask", io_category="cuda_tensor")
        router_logits = self.mpk.new_tensor(
            dims=(mbt, NUM_EXPERTS), dtype=bfloat16,
            name="mtp_router_logits", io_category="cuda_tensor")
        self.mpk.linear_layer(
            input=self.rmsnorm_out, weight=w_gate, output=router_logits,
            grid_dim=(grid_for_rmsnorm_linear_layer(w_gate.dim(0)), 1, 1),
            block_dim=(128, 1, 1))

        moe_output = self.mpk.new_tensor(
            dims=(mbt, self.hidden_size), dtype=bfloat16,
            name="mtp_moe_output", io_category="cuda_tensor")
        self.mpk.tensor_init_layer(
            input=moe_output, dummy_input=self.rmsnorm_out,
            dummy_output=self.rmsnorm_out,
            grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1))

        w_gate_bias = self.mpk.attach_input(
            torch_tensor=state_dict[f"{mlp_prefix}gate.e_score_correction_bias"],
            name=f"mtp_{mlp_prefix}gate_bias")
        self.mpk.moe_topk_sigmoid_routing_layer(
            input=router_logits, bias=w_gate_bias,
            output=(moe_topk_weights, moe_routing_indices, moe_mask),
            grid_dim=(1, 1, 1), block_dim=(256, 1, 1))

        # Expert W13 (FP8)
        w_w13, s_w13 = self._attach_fp8_weight(
            state_dict, f"{mlp_prefix}experts.w13.weight",
            f"mtp_{mlp_prefix}experts_w13")
        moe_input_fp8 = self.mpk.new_tensor(
            dims=(mbt, self.hidden_size), dtype=bfloat16,
            name="mtp_moe_input_fp8", io_category="cuda_tensor")
        moe_input_scale = self.mpk.new_tensor(
            dims=(mbt, self.hidden_size // 128), dtype=bfloat16,
            name="mtp_moe_input_scale", io_category="cuda_tensor")
        self.mpk.quantize_fp8_layer(
            input=self.rmsnorm_out, output_fp8=moe_input_fp8,
            output_scale=moe_input_scale,
            grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1))

        moe_mid = self.mpk.new_tensor(
            dims=(mbt, NUM_EXPERTS_PER_TOK, 2 * self.moe_intermediate_size),
            dtype=bfloat16, name="mtp_moe_mid", io_category="cuda_tensor")
        self.mpk.moe_w13_fp8_layer(
            input_fp8=moe_input_fp8, input_scale=moe_input_scale,
            weight_fp8=w_w13, weight_scale=s_w13,
            moe_routing_indices=moe_routing_indices, moe_mask=moe_mask,
            output=moe_mid, grid_dim=(NUM_EXPERTS, 1, 1), block_dim=(128, 1, 1))

        moe_silu_out = self.mpk.new_tensor(
            dims=(mbt, NUM_EXPERTS_PER_TOK, self.moe_intermediate_size),
            dtype=bfloat16, name="mtp_moe_silu", io_category="cuda_tensor")
        self.mpk.moe_silu_mul_layer(
            input=moe_mid, output=moe_silu_out,
            grid_dim=(mbt * NUM_EXPERTS_PER_TOK, 1, 1), block_dim=(128, 1, 1))

        # Expert W2 (FP8) — quantize 3D silu_out first
        w_w2, s_w2 = self._attach_fp8_weight(
            state_dict, f"{mlp_prefix}experts.w2.weight",
            f"mtp_{mlp_prefix}experts_w2")
        mtp_silu_fp8 = self.mpk.new_tensor(
            dims=(mbt, NUM_EXPERTS_PER_TOK, self.moe_intermediate_size),
            dtype=float8_e4m3, name="mtp_moe_silu_fp8", io_category="cuda_tensor")
        mtp_silu_scale = self.mpk.new_tensor(
            dims=(mbt, NUM_EXPERTS_PER_TOK, self.moe_intermediate_size // 128),
            dtype=float32, name="mtp_moe_silu_scale", io_category="cuda_tensor")
        self.mpk.quantize_fp8_layer(
            input=moe_silu_out, output_fp8=mtp_silu_fp8,
            output_scale=mtp_silu_scale,
            grid_dim=(mbt * NUM_EXPERTS_PER_TOK, 1, 1), block_dim=(128, 1, 1))
        moe_down_out = self.mpk.new_tensor(
            dims=(mbt, NUM_EXPERTS_PER_TOK, self.hidden_size),
            dtype=bfloat16, name="mtp_moe_down", io_category="cuda_tensor")
        self.mpk.moe_w2_fp8_layer(
            input_fp8=mtp_silu_fp8, input_scale=mtp_silu_scale,
            weight_fp8=w_w2, weight_scale=s_w2,
            moe_routing_indices=moe_routing_indices, moe_mask=moe_mask,
            output=moe_down_out, grid_dim=(NUM_EXPERTS, 1, 1), block_dim=(128, 1, 1))

        # Shared expert (FP8)
        sp = f"{mlp_prefix}shared_experts."
        shared_gate_w = state_dict[f"{sp}gate_proj.weight"]
        shared_up_w = state_dict[f"{sp}up_proj.weight"]
        shared_gate_s = state_dict[f"{sp}gate_proj.weight_scale_inv"]
        shared_up_s = state_dict[f"{sp}up_proj.weight_scale_inv"]
        w_s_gu = self._safe_attach(
            torch.cat([shared_gate_w, shared_up_w], dim=0), f"mtp_{sp}gate_up")
        s_s_gu = self._safe_attach(
            torch.cat([shared_gate_s, shared_up_s], dim=0), f"mtp_{sp}gate_up_scale")
        shared_mid = self.mpk.new_tensor(
            dims=(mbt, 2 * self.moe_intermediate_size), dtype=bfloat16,
            name="mtp_shared_mid", io_category="cuda_tensor")
        self._fp8_linear(self.rmsnorm_out, w_s_gu, s_s_gu, shared_mid,
                         grid_dim=(grid_for_rmsnorm_linear_layer(w_s_gu.dim(0)), 1, 1),
                         block_dim=(128, 1, 1))
        shared_silu = self.mpk.new_tensor(
            dims=(mbt, self.moe_intermediate_size), dtype=bfloat16,
            name="mtp_shared_silu", io_category="cuda_tensor")
        self.mpk.silu_mul_layer(
            input=shared_mid, output=shared_silu,
            grid_dim=(self.moe_intermediate_size // 64, 1, 1), block_dim=(128, 1, 1))
        w_s_down, s_s_down = self._attach_fp8_weight(
            state_dict, f"{sp}down_proj.weight", f"mtp_{sp}down_proj")
        shared_residual = self.mpk.new_tensor(
            dims=(mbt, self.hidden_size), dtype=bfloat16,
            name="mtp_shared_residual", io_category="cuda_tensor")
        self._fp8_linear(shared_silu, w_s_down, s_s_down, shared_residual,
                         grid_dim=(self.hidden_size // 64, 1, 1),
                         block_dim=(128, 1, 1), residual=self.mtp_x)

        self.mpk.moe_mul_sum_add_layer(
            input=moe_down_out, weight=moe_topk_weights,
            residual=shared_residual, output=moe_output,
            grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1))
        self.mlp_out = moe_output

    def _build_mtp_layer(self, state_dict: dict):
        """Build MTP predictor layer.

        Architecture (from vLLM's DeepSeekMultiTokenPredictorLayer):
        1. embed(draft_token) → enorm
        2. hnorm(previous_hidden_states)
        3. eh_proj(cat[enorm_out, hnorm_out]) → via split: W1@e + W2@h
        4. Full decoder layer (MLA attention + dense MLP)
        5. Shared LM head → draft logits → argmax → draft_token_ids[step]

        Draft steps are statically unrolled at compile time.
        MTP layer weights recycle via modulo: step_idx % num_mtp_layers.
        """
        if self.mtp_config is None:
            return

        from ...speculative import MTPConfig
        if not isinstance(self.mtp_config, MTPConfig):
            return

        num_draft_steps = self.mtp_config.num_speculative_tokens
        # Checkpoint stores MTP layer at model.layers.{num_hidden_layers}
        # (e.g., model.layers.61 for DeepSeek V3 with 61 main layers)
        mtp_layer_idx = self.num_layers  # 61
        mtp_prefix = f"model.layers.{mtp_layer_idx}."
        # The transformer block weights use the same prefix (no mtp_block sub-prefix)
        mtp_block_prefix = mtp_prefix

        # ---- Shared weights ----
        # embed_tokens and lm_head are shared with main model (already attached)

        # MTP-specific weights: enorm, hnorm, eh_proj
        w_enorm = self.mpk.attach_input(
            torch_tensor=state_dict[f"{mtp_prefix}enorm.weight"],
            name="mtp_enorm_weight",
        )
        w_hnorm = self.mpk.attach_input(
            torch_tensor=state_dict[f"{mtp_prefix}hnorm.weight"],
            name="mtp_hnorm_weight",
        )

        # eh_proj: [hidden_size, 2*hidden_size] → split into W1 (embed) + W2 (hidden)
        eh_proj_full = state_dict[f"{mtp_prefix}eh_proj.weight"]
        w_eh_proj_1 = self.mpk.attach_input(
            torch_tensor=eh_proj_full[:, :self.hidden_size].contiguous(),
            name="mtp_eh_proj_embed",
        )
        w_eh_proj_2 = self.mpk.attach_input(
            torch_tensor=eh_proj_full[:, self.hidden_size:].contiguous(),
            name="mtp_eh_proj_hidden",
        )

        # ---- MTP KV cache (separate from main model) ----
        mtp_ckv_kpe_cache = torch.zeros(
            (self.mpk.max_num_pages, self.mpk.page_size, self.qk_head_dim),
            dtype=torch.bfloat16, device="cuda",
        )
        self.mtp_ckv_kpe_cache_tensor = self.mpk.attach_input(
            torch_tensor=mtp_ckv_kpe_cache,
            name="mtp_ckv_kpe_cache",
        )

        # ---- Intermediate tensors ----
        mbt = self.max_num_batched_tokens
        mtp_embed_out = self.mpk.new_tensor(
            dims=(mbt, self.hidden_size), dtype=bfloat16,
            name="mtp_embed_out", io_category="cuda_tensor",
        )
        mtp_enorm_out = self.mpk.new_tensor(
            dims=(mbt, self.hidden_size), dtype=bfloat16,
            name="mtp_enorm_out", io_category="cuda_tensor",
        )
        mtp_hnorm_out = self.mpk.new_tensor(
            dims=(mbt, self.hidden_size), dtype=bfloat16,
            name="mtp_hnorm_out", io_category="cuda_tensor",
        )
        mtp_proj_out = self.mpk.new_tensor(
            dims=(mbt, self.hidden_size), dtype=bfloat16,
            name="mtp_proj_out", io_category="cuda_tensor",
        )

        # Draft token ID buffers
        draft_token_ids = self.mpk.new_tensor(
            dims=(mbt, 1), dtype=int64,
            name="mtp_draft_token_ids", io_category="cuda_tensor",
        )

        # Collect all draft token IDs for verification
        all_draft_ids = self.mpk.new_tensor(
            dims=(mbt, num_draft_steps), dtype=int64,
            name="mtp_all_draft_ids", io_category="cuda_tensor",
        )

        # ---- Shared embed weight reference (saved during build_from_dict) ----
        w_embed = self.w_embed

        # ---- Save main model state ----
        main_hidden_states = self.x  # After all 61 layers + final norm

        # ---- Draft generation loop (statically unrolled) ----
        for step in range(num_draft_steps):
            # 1. Get draft token: step 0 from main argmax, step 1+ from prev MTP
            draft_input = self.argmax_out_dtensor if step == 0 else draft_token_ids

            # 2. Embed draft token (shared embed_tokens weight)
            self.mpk.embed_layer(
                input=draft_input, weight=w_embed, output=mtp_embed_out,
                grid_dim=(1, 1, 1), block_dim=(128, 1, 1), input_source=1,
            )

            # 3. enorm(embed_out)
            self.mpk.rmsnorm_layer(
                input=mtp_embed_out, weight=w_enorm, output=mtp_enorm_out,
                grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1),
            )

            # 4. hnorm(previous_hidden_states)
            hidden_input = main_hidden_states if step == 0 else self.mtp_x
            self.mpk.rmsnorm_layer(
                input=hidden_input, weight=w_hnorm, output=mtp_hnorm_out,
                grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1),
            )

            # 5. eh_proj: output = W1 @ enorm_out + W2 @ hnorm_out
            self.mpk.linear_layer(
                input=mtp_enorm_out, weight=w_eh_proj_1, output=mtp_proj_out,
                grid_dim=(grid_for_rmsnorm_linear_layer(w_eh_proj_1.dim(0)), 1, 1),
                block_dim=(128, 1, 1),
            )
            self.mpk.linear_with_residual_layer(
                input=mtp_hnorm_out, weight=w_eh_proj_2,
                residual=mtp_proj_out, output=mtp_proj_out,
                grid_dim=(self.hidden_size // 64, 1, 1),
                block_dim=(128, 1, 1),
            )

            # 6. Full MTP decoder layer (MLA attention + MLP, own weights)
            self.mtp_x = mtp_proj_out
            self._build_mtp_decoder_layer(state_dict, mtp_block_prefix)

            # 7. Final norm → shared lm_head → argmax → draft_token_ids
            # shared_head.norm is the MTP's output norm
            # Checkpoint key: model.layers.61.shared_head.norm.weight
            w_mtp_norm = self.mpk.attach_input(
                torch_tensor=state_dict.get(
                    f"{mtp_prefix}shared_head.norm.weight",
                    state_dict["model.norm.weight"],  # fallback to main model norm
                ),
                name=f"mtp_step{step}_norm",
            )
            self.mpk.rmsnorm_layer(
                input=self.mtp_x, weight=w_mtp_norm, output=self.rmsnorm_out,
                grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1),
            )

            # Shared lm_head (saved during build_from_dict)
            w_lm_head = self.w_lm_head
            padded_vocab_size = 129280
            lm_head_out = self.mpk.new_tensor(
                dims=(mbt, padded_vocab_size), dtype=bfloat16,
                name=f"mtp_step{step}_logits", io_category="cuda_tensor",
            )
            self.mpk.linear_layer(
                input=self.rmsnorm_out, weight=w_lm_head, output=lm_head_out,
                grid_dim=(grid_for_rmsnorm_linear_layer(padded_vocab_size), 1, 1),
                block_dim=(128, 1, 1),
            )

            # Argmax → draft_token_ids
            self.mpk.argmax_partial_layer(
                input=lm_head_out,
                output=(self.argmax_part_value, self.argmax_part_index),
                grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1),
            )
            self.mpk.argmax_reduce_layer(
                input=(self.argmax_part_value, self.argmax_part_index),
                output=draft_token_ids,
                grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1),
            )

            # Scatter this step's draft token into the collection buffer
            self.mpk.mtp_token_scatter_layer(
                src=draft_token_ids,
                dst=all_draft_ids,
                grid_dim=(1, 1, 1),
                block_dim=(128, 1, 1),
                batch_size=mbt,
                num_slots=num_draft_steps,
                slot_idx=step,
            )

        # ---- Prepare verify: write draft tokens to sequence buffer ----
        # This sets up input for the next iteration's verification forward:
        # tokens[request, step+1] = main_token, tokens[request, step+2..K+1] = drafts
        # Note: these meta tensors must be attached as DTensors for the task graph
        tokens_buf_raw = self.mpk.meta_tensors.get("tokens", None)
        step_raw = self.mpk.meta_tensors.get("step", None)
        num_new_raw = self.mpk.meta_tensors.get("num_new_tokens", None)

        if tokens_buf_raw is not None and step_raw is not None:
            d_tokens_buf = self.mpk.attach_input(
                torch_tensor=tokens_buf_raw, name="mtp_tokens_buffer")
            d_step = self.mpk.attach_input(
                torch_tensor=step_raw, name="mtp_step")
            d_num_new = self.mpk.attach_input(
                torch_tensor=num_new_raw, name="mtp_num_new_tokens")
            self.mpk.mtp_prepare_verify_layer(
                main_token=self.argmax_out_dtensor,
                draft_tokens=all_draft_ids,
                tokens_buffer=d_tokens_buf,
                step=d_step,
                num_new_tokens=d_num_new,
                grid_dim=(self.mpk.max_num_batched_requests, 1, 1),
                block_dim=(128, 1, 1),
                num_draft_tokens=num_draft_steps,
                max_seq_len=self.mpk.max_seq_length,
            )

        # ---- Verification + Accept/Commit ----
        # After target model re-runs on draft tokens (managed by scheduler),
        # the target token IDs are available. Wire up verification here.
        target_token_ids = self.mpk.new_tensor(
            dims=(mbt, num_draft_steps + 1), dtype=int64,
            name="mtp_target_token_ids", io_category="cuda_tensor",
        )
        accepted_count = self.mpk.new_tensor(
            dims=(mbt, 1), dtype=int64,
            name="mtp_accepted_count", io_category="cuda_tensor",
        )
        verified_output_tokens = self.mpk.new_tensor(
            dims=(mbt, num_draft_steps + 1), dtype=int64,
            name="mtp_verified_output", io_category="cuda_tensor",
        )

        # Select verification method
        method = self.mtp_config.rejection_sample_method
        if method == "strict":
            self.mpk.mtp_verify_strict_layer(
                draft_token_ids=all_draft_ids,
                target_token_ids=target_token_ids,
                accepted_count=accepted_count,
                output_tokens=verified_output_tokens,
                grid_dim=(mbt, 1, 1),
                block_dim=(128, 1, 1),
                num_draft_tokens=num_draft_steps,
            )
        # TODO: add probabilistic and synthetic verify paths

        # Accept/commit: update position and output final tokens
        step_raw = self.mpk.meta_tensors.get("step", None)
        if step_raw is not None:
            current_position = self.mpk.attach_input(
                torch_tensor=step_raw, name="mtp_accept_step")
            new_position = self.mpk.new_tensor(
                dims=(mbt, 1), dtype=int64,
                name="mtp_new_position", io_category="cuda_tensor",
            )
            final_output = self.mpk.new_tensor(
                dims=(mbt, num_draft_steps + 1), dtype=int64,
                name="mtp_final_output", io_category="cuda_tensor",
            )
            num_new = self.mpk.new_tensor(
                dims=(mbt, 1), dtype=int64,
                name="mtp_num_new_tokens", io_category="cuda_tensor",
            )
            self.mpk.mtp_accept_commit_layer(
                accepted_count=accepted_count,
                output_tokens=verified_output_tokens,
                current_position=current_position,
                new_position=new_position,
                final_output=final_output,
                num_new_tokens=num_new,
                grid_dim=(mbt, 1, 1),
                block_dim=(128, 1, 1),
                num_draft_tokens=num_draft_steps,
            )

    def build_layers(self, state_dict: dict, layer_indices: list = None):
        """Build decoder layers.

        Args:
            layer_indices: If provided, only build these specific layer indices
                          (e.g., [0, 3] for 1 dense + 1 MoE). If None, build all.
        """
        if layer_indices is None:
            layer_indices = list(range(self.num_layers))
        for i in layer_indices:
            prefix = f"model.layers.{i}."

            # Input layernorm
            w_norm = self.mpk.attach_input(
                torch_tensor=state_dict[f"{prefix}input_layernorm.weight"],
                name=f"layer_{i}_input_layernorm",
            )
            self.mpk.rmsnorm_layer(
                input=self.x, weight=w_norm, output=self.rmsnorm_out,
                grid_dim=(self.max_num_batched_tokens, 1, 1),
                block_dim=(128, 1, 1),
            )

            # MLA attention
            self._build_mla_attention_layer(i, state_dict)

            # Residual connection
            self.x = self.attn_proj_out

            # AllReduce after attention
            if self.world_size > 1:
                self.mpk.allreduce_layer(
                    input=self.attn_proj_out, buffer=self.allreduce_buf,
                    output=self.allreduce_out,
                    grid_dim=(self.hidden_size // 64, 1, 1),
                    block_dim=(128, 1, 1),
                )
                self.x = self.allreduce_out

            # Post-attention layernorm
            w_post_norm = self.mpk.attach_input(
                torch_tensor=state_dict[f"{prefix}post_attention_layernorm.weight"],
                name=f"layer_{i}_post_attn_layernorm",
            )
            self.mpk.rmsnorm_layer(
                input=self.x, weight=w_post_norm, output=self.rmsnorm_out,
                grid_dim=(self.max_num_batched_tokens, 1, 1),
                block_dim=(128, 1, 1),
            )

            # MLP: dense (layers 0-2) or MoE (layers 3-60)
            if i < FIRST_MOE_LAYER:
                self._build_dense_mlp(i, state_dict)
            else:
                self._build_moe_mlp(i, state_dict)

            # Residual + optional AllReduce
            self.x = self.mlp_out
            if self.world_size > 1:
                self.mpk.allreduce_layer(
                    input=self.mlp_out, buffer=self.allreduce_buf,
                    output=self.allreduce_out,
                    grid_dim=(self.hidden_size // 64, 1, 1),
                    block_dim=(128, 1, 1),
                )
                self.x = self.allreduce_out

    def build_from_dict(self, state_dict: dict, with_lm_head: bool,
                        layer_indices: list = None):
        """Build the DeepSeek V3 computation graph.

        Args:
            layer_indices: If provided, only build these layers (for correctness testing).
        """
        padded_vocab_size = 129280  # DeepSeek V3 vocab size (already aligned)

        # Embed layer
        self.x = self.mpk.attach_input(
            torch_tensor=self.input_tokens, name="input_token"
        )
        self.w_embed = self.mpk.attach_input(
            torch_tensor=state_dict["model.embed_tokens.weight"],
            name="embed_tokens",
        )
        w_embed = self.w_embed
        self.y = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.hidden_size),
            dtype=bfloat16, name="embed_out", io_category="cuda_tensor",
        )
        self.mpk.embed_layer(
            input=self.x, weight=w_embed, output=self.y,
            grid_dim=(1, 1, 1), block_dim=(128, 1, 1), input_source=1,
        )
        self.x = self.y

        # Intermediate tensors
        self._new_intermediate_tensors()

        # Build all decoder layers
        self.build_layers(state_dict, layer_indices=layer_indices)

        # Final norm + LM head
        w_final_norm = self.mpk.attach_input(
            torch_tensor=state_dict["model.norm.weight"],
            name="model_norm_weight",
        )
        self.mpk.rmsnorm_layer(
            input=self.x, weight=w_final_norm, output=self.rmsnorm_out,
            grid_dim=(self.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )

        if with_lm_head:
            lm_head_weight = state_dict["lm_head.weight"]
            if lm_head_weight.shape[0] < padded_vocab_size:
                lm_head_weight = torch.cat([
                    lm_head_weight,
                    torch.zeros(padded_vocab_size - lm_head_weight.shape[0],
                                self.hidden_size, device="cuda"),
                ], dim=0)

            self.w_lm_head = self.mpk.attach_input(
                torch_tensor=lm_head_weight, name="lm_head",
            )
            w_lm_head = self.w_lm_head
            lm_head_out = self.mpk.new_tensor(
                dims=(self.max_num_batched_tokens, padded_vocab_size),
                dtype=bfloat16, name="lm_head_out", io_category="cuda_tensor",
            )
            self.mpk.linear_layer(
                input=self.rmsnorm_out, weight=w_lm_head, output=lm_head_out,
                grid_dim=(grid_for_rmsnorm_linear_layer(padded_vocab_size), 1, 1),
                block_dim=(128, 1, 1),
            )

            # Argmax
            self.argmax_out_dtensor = self.mpk.attach_input(
                torch_tensor=self.output_tokens, name="output_token",
            )
            argmax_out = self.argmax_out_dtensor
            self.mpk.argmax_partial_layer(
                input=lm_head_out, output=(self.argmax_part_value, self.argmax_part_index),
                grid_dim=(self.max_num_batched_tokens, 1, 1),
                block_dim=(128, 1, 1),
            )
            self.mpk.argmax_reduce_layer(
                input=(self.argmax_part_value, self.argmax_part_index),
                output=argmax_out,
                grid_dim=(self.max_num_batched_tokens, 1, 1),
                block_dim=(128, 1, 1),
            )

        # Optional MTP layer
        self._build_mtp_layer(state_dict)
