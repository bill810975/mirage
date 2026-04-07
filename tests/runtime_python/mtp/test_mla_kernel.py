"""
GPU test for MLA paged attention kernel.

Compares the CUDA kernel output against PyTorch reference implementation.
Requires: GPU with CUDA support. Build first with:
    cd tests/runtime_python/mtp && pip install -e .

Usage:
    python tests/runtime_python/mtp/test_mla_kernel.py
"""

import math
import torch

torch.set_printoptions(sci_mode=False, profile="full")

# Import the compiled CUDA extension
try:
    import test_mla_kernel
    HAS_KERNEL = True
except ImportError:
    HAS_KERNEL = False
    print("Warning: test_mla_kernel not built. Run: cd tests/runtime_python/mtp && pip install -e .")

# Test configuration (must match the CUDA wrapper)
NUM_Q_HEADS = 16
QK_HEAD_DIM = 576
V_HEAD_DIM = 512
MAX_SEQ_LEN = 4096
PAGE_SIZE = 16


def paged_mla_attention_ref(
    q_nope_pe,       # [num_tokens, NUM_Q_HEADS * QK_HEAD_DIM]
    ckv_kpe_cache,   # [num_pages, PAGE_SIZE, QK_HEAD_DIM]
    kv_new,          # [num_tokens, QK_HEAD_DIM]
    qo_indptr,       # [num_requests + 1]
    kv_indptr,       # [num_requests + 1]
    kv_indices,      # [total_pages]
    kv_last_page_len,  # [num_requests]
    num_q_heads=NUM_Q_HEADS,
    qk_head_dim=QK_HEAD_DIM,
    v_head_dim=V_HEAD_DIM,
):
    """PyTorch reference for paged MLA attention."""
    num_tokens_total = q_nope_pe.shape[0]
    page_size = ckv_kpe_cache.shape[1]
    num_requests = qo_indptr.shape[0] - 1
    softmax_scale = 1.0 / math.sqrt(qk_head_dim)

    output = torch.zeros(num_tokens_total, num_q_heads * v_head_dim,
                         dtype=q_nope_pe.dtype, device=q_nope_pe.device)

    for req in range(num_requests):
        first_tok = qo_indptr[req].item()
        last_tok = qo_indptr[req + 1].item()
        num_tok = last_tok - first_tok
        if num_tok == 0:
            continue

        first_pg = kv_indptr[req].item()
        last_pg = kv_indptr[req + 1].item()
        num_pg = last_pg - first_pg
        last_pg_len = kv_last_page_len[req].item()
        seq_len = (num_pg - 1) * page_size + last_pg_len

        # Gather KV cache
        kv_list = []
        for p in range(num_pg):
            pi = kv_indices[first_pg + p].item()
            if p == num_pg - 1:
                kv_list.append(ckv_kpe_cache[pi, :last_pg_len])
            else:
                kv_list.append(ckv_kpe_cache[pi])
        kv_all = torch.cat(kv_list, dim=0)  # [seq_len, qk_head_dim]

        # Write new KV entries
        kv_all[seq_len - num_tok:seq_len] = kv_new[first_tok:last_tok]
        # Write back to cache pages
        for t in range(num_tok):
            cp = seq_len - num_tok + t
            pi = kv_indices[first_pg + cp // page_size].item()
            po = cp % page_size
            ckv_kpe_cache[pi, po] = kv_new[first_tok + t]

        for t in range(num_tok):
            valid_len = seq_len - num_tok + t + 1
            k = kv_all[:valid_len].float()
            v = kv_all[:valid_len, :v_head_dim].float()
            q = q_nope_pe[first_tok + t].float().view(num_q_heads, qk_head_dim)

            scores = torch.matmul(q, k.T) * softmax_scale
            probs = torch.softmax(scores, dim=-1)
            out = torch.matmul(probs, v)
            output[first_tok + t] = out.reshape(-1).to(q_nope_pe.dtype)

    return output


def test_single_request_single_token():
    """1 request, 1 token, KV cache with 32 entries."""
    print("\n=== Test: single request, single token ===")
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    num_tokens = 1
    num_pages = 4  # 4 pages * 16 page_size = up to 64 positions
    kv_len = 32    # 32 KV entries (2 full pages)

    q = torch.randn(num_tokens, NUM_Q_HEADS * QK_HEAD_DIM, device=device, dtype=torch.bfloat16)
    cache = torch.randn(num_pages, PAGE_SIZE, QK_HEAD_DIM, device=device, dtype=torch.bfloat16)
    kv_new = torch.randn(num_tokens, QK_HEAD_DIM, device=device, dtype=torch.bfloat16)

    # seq_len = kv_len + num_tokens = 33
    # Pages needed: ceil(33/16) = 3
    total_seq = kv_len + num_tokens  # 33
    pages_needed = (total_seq + PAGE_SIZE - 1) // PAGE_SIZE  # 3
    last_page_len = total_seq - (pages_needed - 1) * PAGE_SIZE  # 33 - 32 = 1

    qo_indptr = torch.tensor([0, num_tokens], device=device, dtype=torch.int32)
    kv_indptr = torch.tensor([0, pages_needed], device=device, dtype=torch.int32)
    kv_indices = torch.arange(pages_needed, device=device, dtype=torch.int32)
    kv_last_page_len = torch.tensor([last_page_len], device=device, dtype=torch.int32)

    output_kernel = torch.zeros(num_tokens, NUM_Q_HEADS * V_HEAD_DIM, device=device, dtype=torch.bfloat16)

    # Run kernel
    # Make copies for reference (since cache is modified in-place)
    cache_ref = cache.clone()
    # Split kv_new into c_latent (512) and k_pe (64) for the kernel
    c_latent_new = kv_new[:, :V_HEAD_DIM].contiguous()
    k_pe_new = kv_new[:, V_HEAD_DIM:].contiguous()
    test_mla_kernel.mla_attention(
        q, cache, c_latent_new, k_pe_new, output_kernel,
        qo_indptr, kv_indptr, kv_indices, kv_last_page_len, 1
    )

    # Run reference (still uses combined kv_new)
    output_ref = paged_mla_attention_ref(
        q, cache_ref, kv_new,
        qo_indptr.cpu(), kv_indptr.cpu(), kv_indices.cpu(), kv_last_page_len.cpu()
    )

    # Compare
    diff = (output_kernel.float() - output_ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")

    try:
        torch.testing.assert_close(output_kernel, output_ref, rtol=5e-2, atol=5e-2)
        print("  PASSED!")
    except AssertionError as e:
        print(f"  FAILED: {e}")
        # Print first few elements for debugging
        print(f"  Kernel[0,:8]: {output_kernel[0,:8]}")
        print(f"  Ref[0,:8]:    {output_ref[0,:8]}")
        raise


def test_multi_token_causal():
    """1 request, 4 tokens (MTP verify scenario), causal masking."""
    print("\n=== Test: multi-token causal ===")
    device = torch.device("cuda:0")
    torch.manual_seed(123)

    num_tokens = 4
    kv_len = 48
    num_pages = 8

    q = torch.randn(num_tokens, NUM_Q_HEADS * QK_HEAD_DIM, device=device, dtype=torch.bfloat16)
    cache = torch.randn(num_pages, PAGE_SIZE, QK_HEAD_DIM, device=device, dtype=torch.bfloat16)
    kv_new = torch.randn(num_tokens, QK_HEAD_DIM, device=device, dtype=torch.bfloat16)

    total_seq = kv_len + num_tokens  # 52
    pages_needed = (total_seq + PAGE_SIZE - 1) // PAGE_SIZE  # 4
    last_page_len = total_seq - (pages_needed - 1) * PAGE_SIZE  # 52 - 48 = 4

    qo_indptr = torch.tensor([0, num_tokens], device=device, dtype=torch.int32)
    kv_indptr = torch.tensor([0, pages_needed], device=device, dtype=torch.int32)
    kv_indices = torch.arange(pages_needed, device=device, dtype=torch.int32)
    kv_last_page_len = torch.tensor([last_page_len], device=device, dtype=torch.int32)

    output_kernel = torch.zeros(num_tokens, NUM_Q_HEADS * V_HEAD_DIM, device=device, dtype=torch.bfloat16)

    cache_ref = cache.clone()
    c_latent_new = kv_new[:, :V_HEAD_DIM].contiguous()
    k_pe_new = kv_new[:, V_HEAD_DIM:].contiguous()
    test_mla_kernel.mla_attention(
        q, cache, c_latent_new, k_pe_new, output_kernel,
        qo_indptr, kv_indptr, kv_indices, kv_last_page_len, 1
    )

    output_ref = paged_mla_attention_ref(
        q, cache_ref, kv_new,
        qo_indptr.cpu(), kv_indptr.cpu(), kv_indices.cpu(), kv_last_page_len.cpu()
    )

    diff = (output_kernel.float() - output_ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")

    # Verify causal: each token's output should be different
    for t in range(num_tokens - 1):
        assert not torch.allclose(output_kernel[t], output_kernel[t + 1]), \
            f"Token {t} and {t+1} outputs should differ (causal masking)"

    try:
        torch.testing.assert_close(output_kernel, output_ref, rtol=5e-2, atol=5e-2)
        print("  PASSED!")
    except AssertionError as e:
        print(f"  FAILED: {e}")
        raise


def test_small_heads():
    """Smaller config for quick validation: 4 heads."""
    print("\n=== Test: small heads (4 heads) ===")
    # Note: This uses the same compiled kernel with NUM_Q_HEADS=16
    # We'll just use 16 heads but only check a subset
    device = torch.device("cuda:0")
    torch.manual_seed(7)

    num_tokens = 2
    num_pages = 4

    q = torch.randn(num_tokens, NUM_Q_HEADS * QK_HEAD_DIM, device=device, dtype=torch.bfloat16)
    cache = torch.randn(num_pages, PAGE_SIZE, QK_HEAD_DIM, device=device, dtype=torch.bfloat16)
    kv_new = torch.randn(num_tokens, QK_HEAD_DIM, device=device, dtype=torch.bfloat16)

    total_seq = 20 + num_tokens
    pages_needed = (total_seq + PAGE_SIZE - 1) // PAGE_SIZE
    last_page_len = total_seq - (pages_needed - 1) * PAGE_SIZE

    qo_indptr = torch.tensor([0, num_tokens], device=device, dtype=torch.int32)
    kv_indptr = torch.tensor([0, pages_needed], device=device, dtype=torch.int32)
    kv_indices = torch.arange(pages_needed, device=device, dtype=torch.int32)
    kv_last_page_len = torch.tensor([last_page_len], device=device, dtype=torch.int32)

    output_kernel = torch.zeros(num_tokens, NUM_Q_HEADS * V_HEAD_DIM, device=device, dtype=torch.bfloat16)

    cache_ref = cache.clone()
    c_latent_new = kv_new[:, :V_HEAD_DIM].contiguous()
    k_pe_new = kv_new[:, V_HEAD_DIM:].contiguous()
    test_mla_kernel.mla_attention(
        q, cache, c_latent_new, k_pe_new, output_kernel,
        qo_indptr, kv_indptr, kv_indices, kv_last_page_len, 1
    )

    output_ref = paged_mla_attention_ref(
        q, cache_ref, kv_new,
        qo_indptr.cpu(), kv_indptr.cpu(), kv_indices.cpu(), kv_last_page_len.cpu()
    )

    diff = (output_kernel.float() - output_ref.float()).abs()
    print(f"  Max diff: {diff.max().item():.6f}")
    print(f"  Mean diff: {diff.mean().item():.6f}")

    try:
        torch.testing.assert_close(output_kernel, output_ref, rtol=5e-2, atol=5e-2)
        print("  PASSED!")
    except AssertionError as e:
        print(f"  FAILED: {e}")
        raise


def benchmark_mla_kernel(
    batch_sizes=(1, 4, 16),
    kv_lens=(512, 2048, 8192),
    num_tokens=1,
    warmup=20,
    reps=200,
):
    """Benchmark our MLA kernel across batch sizes and KV lengths."""
    device = torch.device("cuda:0")
    print("\n" + "=" * 70)
    print("MLA Kernel Benchmark (DeepSeek V3 config: 16 heads/TP, QK=576, V=512)")
    print("=" * 70)
    print(f"{'Batch':>6} {'KV len':>8} {'Latency(us)':>12} {'TFLOPS':>10}")
    print("-" * 42)

    for bs in batch_sizes:
        for kv_len in kv_lens:
            # Allocate tensors
            q = torch.randn(num_tokens * bs, NUM_Q_HEADS * QK_HEAD_DIM,
                            device=device, dtype=torch.bfloat16)
            num_pages = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE
            cache = torch.randn(num_pages * bs, PAGE_SIZE, QK_HEAD_DIM,
                                device=device, dtype=torch.bfloat16)
            kv_new = torch.randn(num_tokens * bs, QK_HEAD_DIM,
                                 device=device, dtype=torch.bfloat16)
            c_new = kv_new[:, :V_HEAD_DIM].contiguous()
            k_pe_new = kv_new[:, V_HEAD_DIM:].contiguous()
            output = torch.zeros(num_tokens * bs, NUM_Q_HEADS * V_HEAD_DIM,
                                 device=device, dtype=torch.bfloat16)

            # Build paged KV structures (one request per batch entry)
            qo_indptr = torch.zeros(bs + 1, device=device, dtype=torch.int32)
            for i in range(bs):
                qo_indptr[i + 1] = qo_indptr[i] + num_tokens
            kv_indptr = torch.zeros(bs + 1, device=device, dtype=torch.int32)
            for i in range(bs):
                kv_indptr[i + 1] = kv_indptr[i] + num_pages
            kv_indices = torch.arange(num_pages * bs, device=device, dtype=torch.int32)
            last_pg_len = kv_len - (num_pages - 1) * PAGE_SIZE
            kv_last_page_len = torch.full((bs,), last_pg_len,
                                          device=device, dtype=torch.int32)

            # Warmup
            for _ in range(warmup):
                test_mla_kernel.mla_attention(
                    q, cache, c_new, k_pe_new, output,
                    qo_indptr, kv_indptr, kv_indices, kv_last_page_len, bs)
            torch.cuda.synchronize()

            # Time
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(reps):
                test_mla_kernel.mla_attention(
                    q, cache, c_new, k_pe_new, output,
                    qo_indptr, kv_indptr, kv_indices, kv_last_page_len, bs)
            end.record()
            torch.cuda.synchronize()

            ms = start.elapsed_time(end) / reps
            us = ms * 1000

            # FLOPS: per request: 2 * N_heads * (2*kv_len*QKD + kv_len*VD)
            # QK matmul: bs * N_heads * kv_len * QKD * 2
            # PV matmul: bs * N_heads * kv_len * VD * 2
            flops = bs * NUM_Q_HEADS * kv_len * (2 * QK_HEAD_DIM + 2 * V_HEAD_DIM) * 2
            tflops = flops / (ms * 1e-3) / 1e12

            print(f"{bs:>6} {kv_len:>8} {us:>12.1f} {tflops:>10.3f}")


if __name__ == "__main__":
    if not HAS_KERNEL:
        print("Cannot run GPU tests without compiled kernel. Exiting.")
        exit(1)

    print("=" * 60)
    print("MLA Paged Attention Kernel Tests")
    print("=" * 60)

    test_single_request_single_token()
    test_multi_token_causal()
    test_small_heads()

    print("\n" + "=" * 60)
    print("All MLA kernel tests PASSED!")
    print("=" * 60)

    # MAX_SEQ_LEN=4096 in compiled kernel, so kv_lens must stay under that
    benchmark_mla_kernel(batch_sizes=(1, 4, 16), kv_lens=(512, 1024, 2048, 4096))
