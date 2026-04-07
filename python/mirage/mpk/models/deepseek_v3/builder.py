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
from ....core import bfloat16, int64


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
INTERMEDIATE_SIZE = 18432  # MLP intermediate
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

        # MTP config
        self.mtp_config = getattr(mpk, 'spec_decode_config', None)

    def build_from_model(self, model_name: str, model_path: str = None):
        raise NotImplementedError(
            "DeepSeek V3 is too large for direct HuggingFace loading. "
            "Use build_from_config() with pre-converted weights."
        )

    def build_from_config(self, model_config: MirageModelConfig):
        """Build from pre-processed config with absorbed weights."""
        self.ckv_kpe_cache = model_config.k_cache  # [num_layers, num_pages, page_size, 576]
        self.position_embeddings = model_config.position_embeddings

        self.build_from_dict(
            model_config.state_dict,
            model_config.with_lm_head,
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
                dims=(mbt, self.hidden_size),
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

    def _build_mla_attention_layer(self, layer_idx: int, state_dict: dict):
        """Build MLA attention for one decoder layer.

        After weight absorption, the MLA attention uses:
        - q_a_proj: [hidden, q_lora_rank] → compress Q
        - q_a_layernorm: [q_lora_rank] → normalize
        - q_b_proj_absorbed: [q_lora_rank, num_q_heads * qk_head_dim] → Q with absorbed KV
        - kv_a_proj_with_mqa: [hidden, 576] → compressed KV + rope
        - kv_a_layernorm: [kv_lora_rank] → normalize c_latent part
        """
        prefix = f"model.layers.{layer_idx}."

        # Step 1: q_a_proj
        w_q_a = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}self_attn.q_a_proj.weight"],
            name=f"layer_{layer_idx}_q_a_proj",
        )
        self.mpk.linear_layer(
            input=self.rmsnorm_out,
            weight=w_q_a,
            output=self.q_a_out,
            grid_dim=(grid_for_rmsnorm_linear_layer(w_q_a.dim(0)), 1, 1),
            block_dim=(128, 1, 1),
        )

        # Step 2: q_a_layernorm
        w_q_a_ln = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}self_attn.q_a_layernorm.weight"],
            name=f"layer_{layer_idx}_q_a_layernorm",
        )
        self.mpk.rmsnorm_layer(
            input=self.q_a_out,
            weight=w_q_a_ln,
            output=self.q_a_out,  # in-place
            grid_dim=(self.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )

        # Step 3: q_b_proj (absorbed — includes kv_b_proj weight absorption)
        # Output: [batch, num_local_q_heads * qk_head_dim]
        w_q_b = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}self_attn.q_b_proj.weight"],
            name=f"layer_{layer_idx}_q_b_proj",
        )
        self.mpk.linear_layer(
            input=self.q_a_out,
            weight=w_q_b,
            output=self.q_nope_pe,
            grid_dim=(grid_for_rmsnorm_linear_layer(w_q_b.dim(0)), 1, 1),
            block_dim=(128, 1, 1),
        )

        # Step 4: kv_a_proj split into c_latent and k_pe
        # The HF weight kv_a_proj_with_mqa has shape [576, hidden_size].
        # We split it: first 512 rows → c_latent, last 64 rows → k_pe.
        # This allows applying kv_a_layernorm to c_latent only.
        kv_a_full_weight = state_dict[f"{prefix}self_attn.kv_a_proj_with_mqa.weight"]
        w_kv_a_latent = self.mpk.attach_input(
            torch_tensor=kv_a_full_weight[:self.kv_lora_rank].contiguous(),
            name=f"layer_{layer_idx}_kv_a_latent_proj",
        )
        w_kv_a_rope = self.mpk.attach_input(
            torch_tensor=kv_a_full_weight[self.kv_lora_rank:].contiguous(),
            name=f"layer_{layer_idx}_kv_a_rope_proj",
        )
        # c_latent = rmsnorm_out @ w_kv_a_latent^T → [batch, 512]
        self.mpk.linear_layer(
            input=self.rmsnorm_out,
            weight=w_kv_a_latent,
            output=self.c_latent_out,
            grid_dim=(grid_for_rmsnorm_linear_layer(w_kv_a_latent.dim(0)), 1, 1),
            block_dim=(128, 1, 1),
        )
        # k_pe = rmsnorm_out @ w_kv_a_rope^T → [batch, 64]
        self.mpk.linear_layer(
            input=self.rmsnorm_out,
            weight=w_kv_a_rope,
            output=self.k_pe_out,
            grid_dim=(grid_for_rmsnorm_linear_layer(w_kv_a_rope.dim(0)), 1, 1),
            block_dim=(128, 1, 1),
        )

        # Step 5: kv_a_layernorm on c_latent ONLY (512 dims)
        w_kv_a_ln = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}self_attn.kv_a_layernorm.weight"],
            name=f"layer_{layer_idx}_kv_a_layernorm",
        )
        self.mpk.rmsnorm_layer(
            input=self.c_latent_out,
            weight=w_kv_a_ln,
            output=self.c_latent_out,  # in-place
            grid_dim=(self.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )

        # Step 6: MLA paged attention
        # Pass c_latent and k_pe as separate inputs.
        # The kernel combines them into a 576-dim entry and writes to cache.
        cache = self.mpk.attach_input(
            torch_tensor=self.ckv_kpe_cache[layer_idx],
            name=f"layer_{layer_idx}_ckv_kpe_cache",
        )
        self.mpk.paged_mla_layer(
            q_nope_pe=self.q_nope_pe,
            ckv_kpe_cache=cache,
            c_latent_new=self.c_latent_out,
            k_pe_new=self.k_pe_out,
            output=self.attn_out,
            grid_dim=(self.mpk.max_num_batched_requests, 1, 1),
            block_dim=(128, 1, 1),
            num_q_heads=self.num_local_q_heads,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
        )

        # Step 7: O projection + residual
        w_o = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}self_attn.o_proj.weight"],
            name=f"layer_{layer_idx}_o_proj",
        )
        self.mpk.splitk_linear_layer(
            input=self.attn_out,
            weight=w_o,
            output=self.attn_proj_out,
            grid_dim=(self.hidden_size // 128, 128 * 128 // self.hidden_size, 1),
            block_dim=(256, 1, 1),
        )

    def _build_dense_mlp(self, layer_idx: int, state_dict: dict):
        """Build dense MLP for layers 0-2."""
        prefix = f"model.layers.{layer_idx}."

        w_gate_up = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}mlp.gate_up_proj.weight"],
            name=f"layer_{layer_idx}_gate_up_proj",
        )
        self.mpk.linear_layer(
            input=self.rmsnorm_out,
            weight=w_gate_up,
            output=self.mlp_mid,
            grid_dim=(grid_for_rmsnorm_linear_layer(w_gate_up.dim(0)), 1, 1),
            block_dim=(128, 1, 1),
        )
        self.mpk.silu_mul_layer(
            input=self.mlp_mid,
            output=self.silu_mul_out,
            grid_dim=(self.intermediate_size // 64, 1, 1),
            block_dim=(128, 1, 1),
        )
        w_down = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}mlp.down_proj.weight"],
            name=f"layer_{layer_idx}_down_proj",
        )
        self.mpk.splitk_linear_layer(
            input=self.silu_mul_out,
            weight=w_down,
            output=self.mlp_out,
            grid_dim=(self.hidden_size // 128, 128 * 128 // self.hidden_size, 1),
            block_dim=(256, 1, 1),
        )

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

        # TopK softmax routing
        self.mpk.moe_topk_softmax_routing_layer(
            input=router_logits,
            output=(moe_topk_weights, moe_routing_indices, moe_mask),
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )

        # Expert W1+W3 (gate + up projection)
        w_experts_w13 = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}experts.w13.weight"],
            name=f"layer_{layer_idx}_experts_w13",
        )
        moe_mid = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, NUM_EXPERTS_PER_TOK,
                  2 * self.intermediate_size),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_moe_mid",
            io_category="cuda_tensor",
        )
        self.mpk.moe_w13_linear_layer(
            input=self.rmsnorm_out,
            weight=w_experts_w13,
            moe_routing_indices=moe_routing_indices,
            moe_mask=moe_mask,
            output=moe_mid,
            grid_dim=(NUM_EXPERTS, 1, 1),
            block_dim=(128, 1, 1),
        )

        # SiLU activation
        moe_silu_out = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, NUM_EXPERTS_PER_TOK,
                  self.intermediate_size),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_moe_silu",
            io_category="cuda_tensor",
        )
        self.mpk.moe_silu_mul_layer(
            input=moe_mid,
            output=moe_silu_out,
            grid_dim=(self.max_num_batched_tokens * NUM_EXPERTS_PER_TOK, 1, 1),
            block_dim=(128, 1, 1),
        )

        # Expert W2 (down projection)
        w_experts_w2 = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}experts.w2.weight"],
            name=f"layer_{layer_idx}_experts_w2",
        )
        moe_down_out = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, NUM_EXPERTS_PER_TOK,
                  self.hidden_size),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_moe_down",
            io_category="cuda_tensor",
        )
        self.mpk.moe_w2_linear_layer(
            input=moe_silu_out,
            weight=w_experts_w2,
            moe_routing_indices=moe_routing_indices,
            moe_mask=moe_mask,
            output=moe_down_out,
            grid_dim=(NUM_EXPERTS, 1, 1),
            block_dim=(128, 1, 1),
        )

        # Weighted sum + residual
        self.mpk.moe_mul_sum_add_layer(
            input=moe_down_out,
            weight=moe_topk_weights,
            residual=self.x,
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
        """Build MLA attention using a custom weight prefix (for MTP reuse)."""
        attn_prefix = f"{prefix}self_attn."

        w_q_a = self.mpk.attach_input(
            torch_tensor=state_dict[f"{attn_prefix}q_a_proj.weight"],
            name=f"mtp_{attn_prefix}q_a_proj",
        )
        self.mpk.linear_layer(
            input=self.rmsnorm_out, weight=w_q_a, output=self.q_a_out,
            grid_dim=(grid_for_rmsnorm_linear_layer(w_q_a.dim(0)), 1, 1),
            block_dim=(128, 1, 1),
        )

        w_q_a_ln = self.mpk.attach_input(
            torch_tensor=state_dict[f"{attn_prefix}q_a_layernorm.weight"],
            name=f"mtp_{attn_prefix}q_a_layernorm",
        )
        self.mpk.rmsnorm_layer(
            input=self.q_a_out, weight=w_q_a_ln, output=self.q_a_out,
            grid_dim=(self.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )

        w_q_b = self.mpk.attach_input(
            torch_tensor=state_dict[f"{attn_prefix}q_b_proj.weight"],
            name=f"mtp_{attn_prefix}q_b_proj",
        )
        self.mpk.linear_layer(
            input=self.q_a_out, weight=w_q_b, output=self.q_nope_pe,
            grid_dim=(grid_for_rmsnorm_linear_layer(w_q_b.dim(0)), 1, 1),
            block_dim=(128, 1, 1),
        )

        kv_a_full = state_dict[f"{attn_prefix}kv_a_proj_with_mqa.weight"]
        w_kv_a_latent = self.mpk.attach_input(
            torch_tensor=kv_a_full[:self.kv_lora_rank].contiguous(),
            name=f"mtp_{attn_prefix}kv_a_latent",
        )
        w_kv_a_rope = self.mpk.attach_input(
            torch_tensor=kv_a_full[self.kv_lora_rank:].contiguous(),
            name=f"mtp_{attn_prefix}kv_a_rope",
        )
        self.mpk.linear_layer(
            input=self.rmsnorm_out, weight=w_kv_a_latent, output=self.c_latent_out,
            grid_dim=(grid_for_rmsnorm_linear_layer(w_kv_a_latent.dim(0)), 1, 1),
            block_dim=(128, 1, 1),
        )
        self.mpk.linear_layer(
            input=self.rmsnorm_out, weight=w_kv_a_rope, output=self.k_pe_out,
            grid_dim=(grid_for_rmsnorm_linear_layer(w_kv_a_rope.dim(0)), 1, 1),
            block_dim=(128, 1, 1),
        )

        w_kv_a_ln = self.mpk.attach_input(
            torch_tensor=state_dict[f"{attn_prefix}kv_a_layernorm.weight"],
            name=f"mtp_{attn_prefix}kv_a_layernorm",
        )
        self.mpk.rmsnorm_layer(
            input=self.c_latent_out, weight=w_kv_a_ln, output=self.c_latent_out,
            grid_dim=(self.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )

        # MTP attention uses its own KV cache
        self.mpk.paged_mla_layer(
            q_nope_pe=self.q_nope_pe,
            ckv_kpe_cache=self.mtp_ckv_kpe_cache_tensor,
            c_latent_new=self.c_latent_out,
            k_pe_new=self.k_pe_out,
            output=self.attn_out,
            grid_dim=(self.mpk.max_num_batched_requests, 1, 1),
            block_dim=(128, 1, 1),
            num_q_heads=self.num_local_q_heads,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
        )

        w_o = self.mpk.attach_input(
            torch_tensor=state_dict[f"{attn_prefix}o_proj.weight"],
            name=f"mtp_{attn_prefix}o_proj",
        )
        self.mpk.splitk_linear_layer(
            input=self.attn_out, weight=w_o, output=self.attn_proj_out,
            grid_dim=(self.hidden_size // 128, 128 * 128 // self.hidden_size, 1),
            block_dim=(256, 1, 1),
        )

    def _build_dense_mlp_with_prefix(self, prefix: str, state_dict: dict):
        """Build dense MLP using a custom weight prefix (for MTP reuse)."""
        mlp_prefix = f"{prefix}mlp."

        w_gate_up = self.mpk.attach_input(
            torch_tensor=state_dict[f"{mlp_prefix}gate_up_proj.weight"],
            name=f"mtp_{mlp_prefix}gate_up_proj",
        )
        self.mpk.linear_layer(
            input=self.rmsnorm_out, weight=w_gate_up, output=self.mlp_mid,
            grid_dim=(grid_for_rmsnorm_linear_layer(w_gate_up.dim(0)), 1, 1),
            block_dim=(128, 1, 1),
        )
        self.mpk.silu_mul_layer(
            input=self.mlp_mid, output=self.silu_mul_out,
            grid_dim=(self.intermediate_size // 64, 1, 1),
            block_dim=(128, 1, 1),
        )
        w_down = self.mpk.attach_input(
            torch_tensor=state_dict[f"{mlp_prefix}down_proj.weight"],
            name=f"mtp_{mlp_prefix}down_proj",
        )
        self.mpk.splitk_linear_layer(
            input=self.silu_mul_out, weight=w_down, output=self.mlp_out,
            grid_dim=(self.hidden_size // 128, 128 * 128 // self.hidden_size, 1),
            block_dim=(256, 1, 1),
        )

    def _build_moe_mlp_with_prefix(self, prefix: str, state_dict: dict):
        """Build MoE MLP using a custom weight prefix (for MTP reuse)."""
        mlp_prefix = f"{prefix}mlp."

        w_gate = self.mpk.attach_input(
            torch_tensor=state_dict[f"{mlp_prefix}gate.weight"],
            name=f"mtp_{mlp_prefix}gate",
        )
        moe_topk_weights = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, NUM_EXPERTS_PER_TOK),
            dtype=bfloat16, name="mtp_moe_topk_weights", io_category="cuda_tensor",
        )
        moe_routing_indices = self.mpk.new_tensor(
            dims=(NUM_EXPERTS, self.max_num_batched_tokens),
            dtype=bfloat16, name="mtp_moe_routing_indices", io_category="cuda_tensor",
        )
        moe_mask = self.mpk.new_tensor(
            dims=(NUM_EXPERTS + 1, 1),
            dtype=bfloat16, name="mtp_moe_mask", io_category="cuda_tensor",
        )
        router_logits = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, NUM_EXPERTS),
            dtype=bfloat16, name="mtp_router_logits", io_category="cuda_tensor",
        )
        self.mpk.linear_layer(
            input=self.rmsnorm_out, weight=w_gate, output=router_logits,
            grid_dim=(grid_for_rmsnorm_linear_layer(w_gate.dim(0)), 1, 1),
            block_dim=(128, 1, 1),
        )

        moe_output = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.hidden_size),
            dtype=bfloat16, name="mtp_moe_output", io_category="cuda_tensor",
        )
        self.mpk.tensor_init_layer(
            input=moe_output, dummy_input=self.rmsnorm_out,
            dummy_output=self.rmsnorm_out,
            grid_dim=(self.max_num_batched_tokens, 1, 1), block_dim=(128, 1, 1),
        )
        self.mpk.moe_topk_softmax_routing_layer(
            input=router_logits,
            output=(moe_topk_weights, moe_routing_indices, moe_mask),
            grid_dim=(1, 1, 1), block_dim=(128, 1, 1),
        )

        w_experts_w13 = self.mpk.attach_input(
            torch_tensor=state_dict[f"{mlp_prefix}experts.w13.weight"],
            name=f"mtp_{mlp_prefix}experts_w13",
        )
        moe_mid = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, NUM_EXPERTS_PER_TOK,
                  2 * self.intermediate_size),
            dtype=bfloat16, name="mtp_moe_mid", io_category="cuda_tensor",
        )
        self.mpk.moe_w13_linear_layer(
            input=self.rmsnorm_out, weight=w_experts_w13,
            moe_routing_indices=moe_routing_indices, moe_mask=moe_mask,
            output=moe_mid,
            grid_dim=(NUM_EXPERTS, 1, 1), block_dim=(128, 1, 1),
        )

        moe_silu_out = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, NUM_EXPERTS_PER_TOK,
                  self.intermediate_size),
            dtype=bfloat16, name="mtp_moe_silu", io_category="cuda_tensor",
        )
        self.mpk.moe_silu_mul_layer(
            input=moe_mid, output=moe_silu_out,
            grid_dim=(self.max_num_batched_tokens * NUM_EXPERTS_PER_TOK, 1, 1),
            block_dim=(128, 1, 1),
        )

        w_experts_w2 = self.mpk.attach_input(
            torch_tensor=state_dict[f"{mlp_prefix}experts.w2.weight"],
            name=f"mtp_{mlp_prefix}experts_w2",
        )
        moe_down_out = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, NUM_EXPERTS_PER_TOK,
                  self.hidden_size),
            dtype=bfloat16, name="mtp_moe_down", io_category="cuda_tensor",
        )
        self.mpk.moe_w2_linear_layer(
            input=moe_silu_out, weight=w_experts_w2,
            moe_routing_indices=moe_routing_indices, moe_mask=moe_mask,
            output=moe_down_out,
            grid_dim=(NUM_EXPERTS, 1, 1), block_dim=(128, 1, 1),
        )

        self.mpk.moe_mul_sum_add_layer(
            input=moe_down_out, weight=moe_topk_weights,
            residual=self.mtp_x, output=moe_output,
            grid_dim=(self.max_num_batched_tokens, 1, 1), block_dim=(128, 1, 1),
        )
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
            # For step 0, the main model's argmax output is already in output_tokens
            draft_input = self.output_tokens if step == 0 else draft_token_ids

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
        tokens_buffer = self.mpk.meta_tensors.get("tokens", None)
        step_tensor = self.mpk.meta_tensors.get("step", None)
        num_new_tokens_tensor = self.mpk.meta_tensors.get("num_new_tokens", None)
        main_model_output = self.output_tokens

        if tokens_buffer is not None and step_tensor is not None:
            self.mpk.mtp_prepare_verify_layer(
                main_token=main_model_output,
                draft_tokens=all_draft_ids,
                tokens_buffer=tokens_buffer,
                step=step_tensor,
                num_new_tokens=num_new_tokens_tensor,
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
        current_position = self.mpk.meta_tensors.get("step", None)
        if current_position is not None:
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

    def build_layers(self, state_dict: dict):
        """Build all 61 decoder layers."""
        for i in range(self.num_layers):
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

    def build_from_dict(self, state_dict: dict, with_lm_head: bool):
        """Build the full DeepSeek V3 computation graph."""
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
        self.build_layers(state_dict)

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
            argmax_out = self.mpk.attach_input(
                torch_tensor=self.output_tokens, name="output_token",
            )
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
