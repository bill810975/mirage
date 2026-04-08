"""Builder integration test: verify DeepSeek V3 task graph generation.

Uses a REDUCED model (2 layers instead of 61) to fit in GPU memory.
Validates: shape correctness, method existence, task registration, graph generation.
"""

import torch
import sys
import os

# TP=1 for single-GPU builder validation (no allreduce needed)
# Weight shapes still use LOCAL dimensions as if TP=8 for realistic testing
TP = 1
HIDDEN = 7168
NUM_Q_HEADS = 128
# Use TP=8 sharded sizes for weight shapes (simulates rank 0 of 8-GPU setup)
SIMULATED_TP = 8
LOCAL_Q_HEADS = NUM_Q_HEADS // SIMULATED_TP  # 16
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM_TOTAL = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
V_HEAD_DIM_TOTAL = KV_LORA_RANK  # 512
INTERMEDIATE = 18432
MOE_INTERMEDIATE = 2048
NUM_EXPERTS = 256
VOCAB = 129280

LOCAL_INTERMEDIATE = INTERMEDIATE // SIMULATED_TP
LOCAL_MOE_INTERMEDIATE = MOE_INTERMEDIATE // SIMULATED_TP

# Reduced for testing — 2 layers: 1 dense + 1 MoE
TEST_NUM_LAYERS = 2
TEST_FIRST_MOE_LAYER = 1  # layer 0 = dense, layer 1 = MoE


def fp8_weight(shape, device):
    """Create mock FP8 weight + packed UE8M0 scale_inv."""
    # Use bfloat16 as placeholder for FP8 (same size, builder only checks shape)
    w = torch.randn(shape, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    # Scale: one uint32 per 128 elements along reduction dim (last dim)
    scale_shape = list(shape)
    scale_shape[-1] = (shape[-1] + 127) // 128
    s = torch.ones(scale_shape, device=device, dtype=torch.float32)
    return w, s


def make_attn_weights(prefix, device):
    d = {}
    p = f"{prefix}self_attn."
    w, s = fp8_weight((Q_LORA_RANK, HIDDEN), device)
    d[f"{p}q_a_proj.weight"] = w
    d[f"{p}q_a_proj.weight_scale_inv"] = s
    d[f"{p}q_a_layernorm.weight"] = torch.randn(Q_LORA_RANK, device=device, dtype=torch.bfloat16)
    w, s = fp8_weight((LOCAL_Q_HEADS * QK_HEAD_DIM_TOTAL, Q_LORA_RANK), device)
    d[f"{p}q_b_proj.weight"] = w
    d[f"{p}q_b_proj.weight_scale_inv"] = s
    w, s = fp8_weight((QK_HEAD_DIM_TOTAL, HIDDEN), device)
    d[f"{p}kv_a_proj_with_mqa.weight"] = w
    d[f"{p}kv_a_proj_with_mqa.weight_scale_inv"] = s
    d[f"{p}kv_a_layernorm.weight"] = torch.randn(KV_LORA_RANK, device=device, dtype=torch.bfloat16)
    w, s = fp8_weight((HIDDEN, LOCAL_Q_HEADS * V_HEAD_DIM_TOTAL), device)
    d[f"{p}o_proj.weight"] = w
    d[f"{p}o_proj.weight_scale_inv"] = s
    return d


def make_moe_weights(prefix, device):
    d = {}
    p = f"{prefix}mlp."
    # Router: BF16
    d[f"{p}gate.weight"] = torch.randn(NUM_EXPERTS, HIDDEN, device=device, dtype=torch.bfloat16)
    d[f"{p}gate.e_score_correction_bias"] = torch.randn(NUM_EXPERTS, device=device, dtype=torch.float32)
    # Experts: FP8
    w, s = fp8_weight((NUM_EXPERTS, 2 * LOCAL_MOE_INTERMEDIATE, HIDDEN), device)
    d[f"{p}experts.w13.weight"] = w
    d[f"{p}experts.w13.weight_scale_inv"] = s
    w, s = fp8_weight((NUM_EXPERTS, HIDDEN, LOCAL_MOE_INTERMEDIATE), device)
    d[f"{p}experts.w2.weight"] = w
    d[f"{p}experts.w2.weight_scale_inv"] = s
    # Shared expert: FP8
    for proj in ["gate_proj", "up_proj"]:
        w, s = fp8_weight((LOCAL_MOE_INTERMEDIATE, HIDDEN), device)
        d[f"{p}shared_experts.{proj}.weight"] = w
        d[f"{p}shared_experts.{proj}.weight_scale_inv"] = s
    w, s = fp8_weight((HIDDEN, LOCAL_MOE_INTERMEDIATE), device)
    d[f"{p}shared_experts.down_proj.weight"] = w
    d[f"{p}shared_experts.down_proj.weight_scale_inv"] = s
    return d


def make_mock_state_dict(device):
    d = {}
    d["model.embed_tokens.weight"] = torch.randn(VOCAB, HIDDEN, device=device, dtype=torch.bfloat16)
    d["model.norm.weight"] = torch.randn(HIDDEN, device=device, dtype=torch.bfloat16)
    d["lm_head.weight"] = torch.randn(VOCAB, HIDDEN, device=device, dtype=torch.bfloat16)

    for i in range(TEST_NUM_LAYERS):
        prefix = f"model.layers.{i}."
        d[f"{prefix}input_layernorm.weight"] = torch.randn(HIDDEN, device=device, dtype=torch.bfloat16)
        d[f"{prefix}post_attention_layernorm.weight"] = torch.randn(HIDDEN, device=device, dtype=torch.bfloat16)
        d.update(make_attn_weights(prefix, device))
        if i < TEST_FIRST_MOE_LAYER:
            w, s = fp8_weight((2 * LOCAL_INTERMEDIATE, HIDDEN), device)
            d[f"{prefix}mlp.gate_up_proj.weight"] = w
            d[f"{prefix}mlp.gate_up_proj.weight_scale_inv"] = s
            w, s = fp8_weight((HIDDEN, LOCAL_INTERMEDIATE), device)
            d[f"{prefix}mlp.down_proj.weight"] = w
            d[f"{prefix}mlp.down_proj.weight_scale_inv"] = s
        else:
            d.update(make_moe_weights(prefix, device))

    # MTP layer
    mtp = f"model.layers.{TEST_NUM_LAYERS}."
    d[f"{mtp}enorm.weight"] = torch.randn(HIDDEN, device=device, dtype=torch.bfloat16)
    d[f"{mtp}hnorm.weight"] = torch.randn(HIDDEN, device=device, dtype=torch.bfloat16)
    d[f"{mtp}eh_proj.weight"] = torch.randn(HIDDEN, 2 * HIDDEN, device=device, dtype=torch.bfloat16)
    d[f"{mtp}shared_head.norm.weight"] = torch.randn(HIDDEN, device=device, dtype=torch.bfloat16)
    d[f"{mtp}input_layernorm.weight"] = torch.randn(HIDDEN, device=device, dtype=torch.bfloat16)
    d[f"{mtp}post_attention_layernorm.weight"] = torch.randn(HIDDEN, device=device, dtype=torch.bfloat16)
    d.update(make_attn_weights(mtp, device))
    d.update(make_moe_weights(mtp, device))

    return d


def test_builder_task_graph():
    import mirage as mi
    from mirage.mpk.models.deepseek_v3 import builder as ds_builder
    from mirage.mpk.models.deepseek_v3.builder import DeepSeekV3Builder
    from mirage.mpk.models.graph_builder import MirageModelConfig
    from mirage.mpk.speculative import MTPConfig

    # Monkey-patch constants for reduced model
    ds_builder.NUM_LAYERS = TEST_NUM_LAYERS
    ds_builder.FIRST_MOE_LAYER = TEST_FIRST_MOE_LAYER
    ds_builder.INTERMEDIATE_SIZE = INTERMEDIATE

    device = "cuda:0"
    torch.cuda.set_device(0)

    max_num_batched_tokens = 8
    max_num_batched_requests = 1
    max_num_pages = 32
    page_size = 16
    max_seq_length = 512

    tokens = torch.zeros(max_num_batched_requests, max_seq_length, dtype=torch.long, device=device)
    input_tokens = torch.zeros(max_num_batched_tokens, 1, dtype=torch.long, device=device)
    output_tokens = torch.zeros(max_num_batched_tokens, 1, dtype=torch.long, device=device)
    step = torch.zeros(max_num_batched_requests, dtype=torch.int32, device=device)
    num_new_tokens = torch.ones(max_num_batched_requests, dtype=torch.int32, device=device)
    prompt_lengths = torch.full((max_num_batched_requests,), 10, dtype=torch.int, device=device)
    qo_indptr = torch.zeros(max_num_batched_requests + 1, dtype=torch.int32, device=device)
    kv_indptr = torch.zeros(max_num_batched_requests + 1, dtype=torch.int32, device=device)
    kv_indices = torch.zeros(max_num_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.zeros(max_num_batched_requests, dtype=torch.int32, device=device)

    total_cache_layers = TEST_NUM_LAYERS + 1  # +1 for MTP
    ckv_kpe_cache = torch.zeros(
        total_cache_layers, max_num_pages, page_size, QK_HEAD_DIM_TOTAL,
        dtype=torch.bfloat16, device=device,
    )

    mtp_config = MTPConfig(num_speculative_tokens=1, rejection_sample_method="strict")

    num_workers, num_schedulers = mi.get_configurations_from_gpu(0)
    print(f"num_workers={num_workers}, num_schedulers={num_schedulers}")

    mpk = mi.PersistentKernel(
        mode="offline",
        world_size=TP,
        mpi_rank=0,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        num_remote_schedulers=0,
        max_seq_length=max_seq_length,
        max_num_batched_requests=max_num_batched_requests,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_pages=max_num_pages,
        page_size=page_size,
        eos_token_id=100257,
        meta_tensors={
            "step": step,
            "tokens": tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "num_new_tokens": num_new_tokens,
            "prompt_lengths": prompt_lengths,
            "qo_indptr_buffer": qo_indptr,
            "paged_kv_indptr_buffer": kv_indptr,
            "paged_kv_indices_buffer": kv_indices,
            "paged_kv_last_page_len_buffer": kv_last_page_len,
        },
        spec_decode_config=mtp_config,
        use_cutlass_kernel=True,
        profiler_tensor=None,
        trace_name="",
    )

    print("Creating mock state dict (2 layers)...")
    state_dict = make_mock_state_dict(device)
    print(f"  {len(state_dict)} keys")

    model_config = MirageModelConfig(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        vocab_size=VOCAB,
        local_num_q_heads=LOCAL_Q_HEADS,
        local_num_kv_heads=1,
        head_dim=QK_HEAD_DIM_TOTAL,
        num_layers=TEST_NUM_LAYERS,
        k_cache=[ckv_kpe_cache[i] for i in range(total_cache_layers)],
        v_cache=[ckv_kpe_cache[i] for i in range(total_cache_layers)],
        position_embeddings=None,
        state_dict=state_dict,
        with_lm_head=True,
    )

    print("Building computation graph...")
    builder = DeepSeekV3Builder(mpk)
    try:
        builder.build_from_config(model_config)
        print("  build_from_config() OK")
    except Exception as e:
        print(f"  build_from_config() FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print("  Main model + MTP builder completed successfully!")

    print("Generating task graph...")
    try:
        results = mpk.kn_graph.generate_task_graph(num_gpus=TP, my_gpu_id=0)
        print(f"  Task graph: {len(results['json_file'])} bytes")
        print(f"  CUDA code: {len(results['cuda_code'])} bytes")
        with open("/tmp/test_task_graph.json", "w") as f:
            f.write(results["json_file"])
        with open("/tmp/test_kernel.cu", "w") as f:
            f.write(results["cuda_code"])
        print("  Saved to /tmp/test_task_graph.json and /tmp/test_kernel.cu")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n=== Builder integration test PASSED ===")


if __name__ == "__main__":
    test_builder_task_graph()
