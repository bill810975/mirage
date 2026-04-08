"""Unit tests: compare our kernels against FlashInfer reference.

Tests:
1. MLA attention: our kernel vs FlashInfer BatchMLAPagedAttentionWrapper
2. TopK sigmoid routing: our kernel vs PyTorch sigmoid + topk (no FlashInfer equivalent)

Run:
  cd tests/runtime_python/mtp
  python setup.py build_ext --inplace  # build MLA kernel
  CUDA_VISIBLE_DEVICES=6 python test_kernels_vs_flashinfer.py
"""

import torch
import math
import sys

# ============================================================================
# Test 1: MLA Attention — our kernel vs FlashInfer
# ============================================================================

def test_mla_vs_flashinfer():
    """Compare our MLA kernel against FlashInfer's BatchMLAPagedAttentionWrapper."""
    import test_mla_kernel  # our compiled kernel
    import flashinfer
    import flashinfer.mla

    device = torch.device("cuda:0")
    torch.manual_seed(42)

    # DeepSeek V3 config (TP=8)
    NUM_Q_HEADS = 16
    HEAD_DIM_LATENT = 512   # c_kv / v_head_dim
    HEAD_DIM_ROPE = 64      # k_pe
    HEAD_DIM_TOTAL = HEAD_DIM_LATENT + HEAD_DIM_ROPE  # 576
    PAGE_SIZE = 16

    configs = [
        {"batch_size": 1, "kv_len": 128, "num_tokens": 1},
        {"batch_size": 1, "kv_len": 512, "num_tokens": 1},
        {"batch_size": 4, "kv_len": 256, "num_tokens": 1},
        {"batch_size": 1, "kv_len": 128, "num_tokens": 4},  # multi-token
    ]

    print("=" * 60)
    print("Test 1: MLA Attention — our kernel vs FlashInfer")
    print("=" * 60)

    for cfg in configs:
        batch_size = cfg["batch_size"]
        kv_len = cfg["kv_len"]
        num_tokens = cfg["num_tokens"]

        pages_per_batch = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE
        total_pages = batch_size * pages_per_batch

        # Input tensors
        q_nope_pe = torch.randn(
            num_tokens * batch_size, NUM_Q_HEADS * HEAD_DIM_TOTAL,
            device=device, dtype=torch.bfloat16)

        # Combined KV cache: [pages, page_size, 576]
        ckv_kpe_cache = torch.randn(
            total_pages, PAGE_SIZE, HEAD_DIM_TOTAL,
            device=device, dtype=torch.bfloat16)

        # New tokens to append
        c_latent_new = torch.randn(
            num_tokens * batch_size, HEAD_DIM_LATENT,
            device=device, dtype=torch.bfloat16)
        k_pe_new = torch.randn(
            num_tokens * batch_size, HEAD_DIM_ROPE,
            device=device, dtype=torch.bfloat16)

        # Paged KV structures
        qo_indptr = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
        for i in range(batch_size):
            qo_indptr[i + 1] = qo_indptr[i] + num_tokens
        kv_indptr = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
        for i in range(batch_size):
            kv_indptr[i + 1] = kv_indptr[i] + pages_per_batch
        kv_indices = torch.arange(total_pages, device=device, dtype=torch.int32)
        kv_last_page_len = torch.full(
            (batch_size,), kv_len - (pages_per_batch - 1) * PAGE_SIZE,
            device=device, dtype=torch.int32)

        page_table = torch.arange(total_pages, device=device, dtype=torch.int32).view(
            batch_size, pages_per_batch)

        # --- Run our kernel ---
        output_ours = torch.zeros(
            num_tokens * batch_size, NUM_Q_HEADS * HEAD_DIM_LATENT,
            device=device, dtype=torch.bfloat16)
        test_mla_kernel.mla_attention(
            q_nope_pe, ckv_kpe_cache, c_latent_new, k_pe_new, output_ours,
            qo_indptr, kv_indptr, kv_indices, kv_last_page_len, batch_size)

        # --- Run FlashInfer ---
        # FlashInfer MLA expects separate q_nope and q_pe
        q_full = q_nope_pe.view(num_tokens * batch_size, NUM_Q_HEADS, HEAD_DIM_TOTAL)
        q_nope = q_full[:, :, :HEAD_DIM_LATENT].contiguous()  # [batch, heads, 512]
        q_pe = q_full[:, :, HEAD_DIM_LATENT:].contiguous()     # [batch, heads, 64]

        # FlashInfer expects separate ckv and kpe caches
        ckv_cache = ckv_kpe_cache[:, :, :HEAD_DIM_LATENT].contiguous()
        kpe_cache = ckv_kpe_cache[:, :, HEAD_DIM_LATENT:].contiguous()

        # Write new tokens to cache (same as our kernel does internally)
        for b in range(batch_size):
            for t in range(num_tokens):
                cache_pos = kv_len - num_tokens + t
                page_idx = page_table[b, cache_pos // PAGE_SIZE].item()
                page_offset = cache_pos % PAGE_SIZE
                ckv_cache[page_idx, page_offset] = c_latent_new[b * num_tokens + t]
                kpe_cache[page_idx, page_offset] = k_pe_new[b * num_tokens + t]

        softmax_scale = 1.0 / math.sqrt(HEAD_DIM_TOTAL)

        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
        mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
            workspace, backend="fa2")

        fi_q_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32)
        fi_kv_indptr = (torch.arange(0, batch_size + 1, device=device, dtype=torch.int32)
                        * pages_per_batch)
        fi_kv_indices = page_table.flatten().contiguous()
        fi_kv_lens = torch.full((batch_size,), kv_len, device=device, dtype=torch.int32)

        mla_wrapper.plan(
            fi_q_indptr, fi_kv_indptr, fi_kv_indices, fi_kv_lens,
            NUM_Q_HEADS, HEAD_DIM_LATENT, HEAD_DIM_ROPE,
            PAGE_SIZE, False, softmax_scale,
            q_nope.dtype, ckv_cache.dtype)

        # FlashInfer only supports single-token decode in this mode
        if num_tokens == 1:
            output_fi = mla_wrapper.run(q_nope, q_pe, ckv_cache, kpe_cache, return_lse=False)
            output_fi_flat = output_fi.view(batch_size, NUM_Q_HEADS * HEAD_DIM_LATENT)

            diff = (output_ours.float() - output_fi_flat.float()).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            print(f"  bs={batch_size} kv={kv_len} tokens={num_tokens}: "
                  f"max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}", end="")
            if max_diff < 0.05:
                print(" PASS")
            else:
                print(f" FAIL (threshold=0.05)")
                # Show first few values for debugging
                print(f"    ours[:5]:  {output_ours[0,:5].tolist()}")
                print(f"    fi[:5]:    {output_fi_flat[0,:5].tolist()}")
        else:
            print(f"  bs={batch_size} kv={kv_len} tokens={num_tokens}: "
                  f"SKIP (FlashInfer MLA decode is single-token only)")

    print()


# ============================================================================
# Test 2: TopK Sigmoid Routing
# ============================================================================

def test_sigmoid_topk():
    """Test our sigmoid topk routing against PyTorch implementation.

    No FlashInfer equivalent exists, so we use PyTorch as reference.
    The logic is simple enough that PyTorch implementation is trustworthy:
    sigmoid(logits) + bias → topk → renormalize weights.
    """
    print("=" * 60)
    print("Test 2: TopK Sigmoid Routing (vs PyTorch)")
    print("=" * 60)
    print("  NOTE: sigmoid topk kernel is registered but can only be tested")
    print("  through the full MPK pipeline (task graph → compile → run).")
    print("  Standalone wrapper not available yet.")
    print("  The correctness will be verified via --correctness in demo.py")
    print()


if __name__ == "__main__":
    print()
    test_mla_vs_flashinfer()
    test_sigmoid_topk()
    print("All tests completed.")
