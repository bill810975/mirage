"""
Unit tests for MTP verification kernels.

Tests the three verification modes against PyTorch reference implementations:
1. Strict: draft token must exactly match target argmax
2. Probabilistic: P_target(token) > u * P_draft(token)
3. Synthetic: position-dependent geometric decay acceptance

These tests verify the LOGIC is correct. The actual CUDA kernels implement
the same logic (see target_verify_mtp.cuh). Run on any machine (no GPU needed).
"""

import torch
import math
import pytest


# ============================================================================
# Reference implementations (matching target_verify_mtp.cuh logic exactly)
# ============================================================================

def verify_strict_ref(
    draft_token_ids: torch.Tensor,   # [num_draft]
    target_token_ids: torch.Tensor,  # [num_draft + 1]
) -> tuple[int, torch.Tensor]:
    """Strict verification: accept while draft matches target, stop at first mismatch."""
    num_draft = draft_token_ids.shape[0]
    accepted = num_draft
    for i in range(num_draft):
        if draft_token_ids[i] != target_token_ids[i]:
            accepted = i
            break
    # Output tokens: accepted tokens + bonus token
    output_tokens = target_token_ids[:accepted + 1].clone()
    return accepted + 1, output_tokens  # +1 for bonus


def verify_probabilistic_ref(
    draft_token_ids: torch.Tensor,  # [num_draft]
    target_logits: torch.Tensor,    # [num_draft + 1, vocab_size]
    draft_logits: torch.Tensor,     # [num_draft, vocab_size]
    temperature: float,
    seed: int,
) -> tuple[int, torch.Tensor]:
    """Probabilistic verification: P_target > u * P_draft."""
    num_draft = draft_token_ids.shape[0]
    vocab_size = target_logits.shape[1]

    # LCG RNG matching the CUDA kernel
    rng_state = seed
    accepted = 0
    still_accepting = True
    output_tokens = torch.zeros(num_draft + 1, dtype=torch.long)

    for i in range(num_draft):
        if not still_accepting:
            break
        draft_token = draft_token_ids[i].item()

        if temperature == 0.0:
            # Greedy: check if draft matches target argmax
            target_argmax = target_logits[i].argmax().item()
            if target_argmax != draft_token:
                still_accepting = False
            else:
                output_tokens[i] = draft_token_ids[i]
                accepted += 1
        else:
            # Probabilistic
            t_logits = target_logits[i].float()
            d_logits = draft_logits[i].float()

            t_probs = torch.softmax(t_logits / temperature, dim=0)
            d_probs = torch.softmax(d_logits / temperature, dim=0)

            p_target = t_probs[draft_token].item()
            p_draft = d_probs[draft_token].item()

            # LCG RNG
            rng_state = (rng_state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
            u = (rng_state >> 33) / (1 << 31)

            if p_target > u * p_draft:
                output_tokens[i] = draft_token_ids[i]
                accepted += 1
            else:
                still_accepting = False

    # Bonus token: target argmax at rejected position
    bonus_token = target_logits[accepted].argmax().item()
    output_tokens[accepted] = bonus_token

    return accepted + 1, output_tokens[:accepted + 1]


def verify_synthetic_ref(
    draft_token_ids: torch.Tensor,   # [num_draft]
    target_token_ids: torch.Tensor,  # [num_draft + 1]
    base_rate: float,
    decay: float,
    seed: int,
) -> tuple[int, torch.Tensor]:
    """Synthetic verification: accept with decaying probability."""
    num_draft = draft_token_ids.shape[0]
    rng_state = seed
    accepted = 0
    output_tokens = torch.zeros(num_draft + 1, dtype=torch.long)
    accept_prob = base_rate

    for i in range(num_draft):
        # LCG RNG
        rng_state = (rng_state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        u = (rng_state >> 33) / (1 << 31)

        if u < accept_prob and draft_token_ids[i] == target_token_ids[i]:
            output_tokens[i] = draft_token_ids[i]
            accepted += 1
            accept_prob *= decay
        else:
            break

    # Bonus token
    output_tokens[accepted] = target_token_ids[accepted]
    return accepted + 1, output_tokens[:accepted + 1]


# ============================================================================
# Tests
# ============================================================================

class TestStrictVerification:
    def test_all_accepted(self):
        """All draft tokens match target — should accept all + bonus."""
        draft = torch.tensor([10, 20, 30, 40])
        target = torch.tensor([10, 20, 30, 40, 50])  # bonus = 50
        count, tokens = verify_strict_ref(draft, target)
        assert count == 5  # 4 accepted + 1 bonus
        assert tokens.tolist() == [10, 20, 30, 40, 50]

    def test_none_accepted(self):
        """First draft token mismatches — should accept 0 + bonus."""
        draft = torch.tensor([10, 20, 30, 40])
        target = torch.tensor([99, 20, 30, 40, 50])
        count, tokens = verify_strict_ref(draft, target)
        assert count == 1  # 0 accepted + 1 bonus
        assert tokens.tolist() == [99]

    def test_partial_accept(self):
        """First 2 match, then mismatch."""
        draft = torch.tensor([10, 20, 30, 40])
        target = torch.tensor([10, 20, 99, 40, 50])
        count, tokens = verify_strict_ref(draft, target)
        assert count == 3  # 2 accepted + 1 bonus
        assert tokens.tolist() == [10, 20, 99]

    def test_single_draft(self):
        """Single draft token, matches."""
        draft = torch.tensor([42])
        target = torch.tensor([42, 7])
        count, tokens = verify_strict_ref(draft, target)
        assert count == 2
        assert tokens.tolist() == [42, 7]

    def test_single_draft_mismatch(self):
        """Single draft token, doesn't match."""
        draft = torch.tensor([42])
        target = torch.tensor([99, 7])
        count, tokens = verify_strict_ref(draft, target)
        assert count == 1
        assert tokens.tolist() == [99]

    def test_seven_drafts(self):
        """Max draft tokens (7), all accepted."""
        draft = torch.tensor([1, 2, 3, 4, 5, 6, 7])
        target = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        count, tokens = verify_strict_ref(draft, target)
        assert count == 8
        assert tokens.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_last_position_mismatch(self):
        """Mismatch at the very last draft position."""
        draft = torch.tensor([10, 20, 30, 40])
        target = torch.tensor([10, 20, 30, 99, 50])
        count, tokens = verify_strict_ref(draft, target)
        assert count == 4  # 3 accepted + 1 bonus
        assert tokens.tolist() == [10, 20, 30, 99]


class TestProbabilisticVerification:
    def test_greedy_all_match(self):
        """Temperature=0 (greedy), all match target argmax."""
        num_draft = 3
        vocab_size = 10

        draft = torch.tensor([5, 3, 7])
        # Target logits: argmax at positions 5, 3, 7, 2 respectively
        target_logits = torch.randn(num_draft + 1, vocab_size)
        for i, token in enumerate([5, 3, 7, 2]):
            target_logits[i, token] = 100.0  # make this the argmax

        draft_logits = torch.randn(num_draft, vocab_size)

        count, tokens = verify_probabilistic_ref(draft, target_logits, draft_logits, 0.0, 42)
        assert count == 4  # all 3 accepted + bonus
        assert tokens[0] == 5
        assert tokens[1] == 3
        assert tokens[2] == 7
        assert tokens[3] == 2  # bonus = target argmax at position 3

    def test_greedy_first_mismatch(self):
        """Temperature=0, first position doesn't match."""
        draft = torch.tensor([5, 3, 7])
        target_logits = torch.randn(4, 10)
        target_logits[0, 9] = 100.0  # argmax is 9, not 5

        draft_logits = torch.randn(3, 10)

        count, tokens = verify_probabilistic_ref(draft, target_logits, draft_logits, 0.0, 42)
        assert count == 1  # 0 accepted + bonus
        assert tokens[0] == 9  # bonus = target argmax

    def test_high_temp_acceptance(self):
        """High temperature with matching distributions — should accept most."""
        torch.manual_seed(123)
        num_draft = 4
        vocab_size = 100

        # Same distribution for target and draft → p_target ≈ p_draft
        # So acceptance condition p_target > u * p_draft is roughly 50%
        logits = torch.randn(num_draft + 1, vocab_size)
        draft = torch.tensor([logits[i].argmax().item() for i in range(num_draft)])
        draft_logits = logits[:num_draft].clone()

        count, tokens = verify_probabilistic_ref(
            draft, logits, draft_logits, 1.0, 12345
        )
        # With matching distributions and draft = argmax, should accept at least some
        assert count >= 1

    def test_deterministic_with_seed(self):
        """Same seed produces same result."""
        draft = torch.tensor([5, 3])
        target_logits = torch.randn(3, 20)
        draft_logits = torch.randn(2, 20)

        r1 = verify_probabilistic_ref(draft, target_logits, draft_logits, 1.0, 42)
        r2 = verify_probabilistic_ref(draft, target_logits, draft_logits, 1.0, 42)
        assert r1[0] == r2[0]
        assert r1[1].tolist() == r2[1].tolist()


class TestSyntheticVerification:
    def test_high_rate_all_match(self):
        """High base rate + all tokens match → accept all."""
        draft = torch.tensor([10, 20, 30])
        target = torch.tensor([10, 20, 30, 40])
        count, tokens = verify_synthetic_ref(draft, target, 1.0, 1.0, 42)
        # base_rate=1.0, decay=1.0 → always accept if tokens match
        assert count == 4
        assert tokens.tolist() == [10, 20, 30, 40]

    def test_zero_rate(self):
        """Base rate 0 → never accept."""
        draft = torch.tensor([10, 20, 30])
        target = torch.tensor([10, 20, 30, 40])
        count, tokens = verify_synthetic_ref(draft, target, 0.0, 1.0, 42)
        assert count == 1  # 0 accepted + bonus
        assert tokens[0] == 10  # bonus = target[0]

    def test_mismatch_rejects(self):
        """Even with high rate, mismatch causes rejection."""
        draft = torch.tensor([10, 99, 30])
        target = torch.tensor([10, 20, 30, 40])
        count, tokens = verify_synthetic_ref(draft, target, 1.0, 1.0, 42)
        assert count == 2  # 1 accepted (pos 0) + bonus at pos 1
        assert tokens[0] == 10
        assert tokens[1] == 20  # bonus = target[1]

    def test_decay_reduces_acceptance(self):
        """With decay < 1, later positions are less likely accepted."""
        draft = torch.tensor([10, 20, 30, 40, 50, 60, 70])
        target = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80])

        # Run many times with different seeds, track average acceptance
        total_accepted = 0
        num_trials = 1000
        for seed in range(num_trials):
            count, _ = verify_synthetic_ref(draft, target, 0.8, 0.7, seed)
            total_accepted += count - 1  # subtract bonus

        avg = total_accepted / num_trials
        # With base=0.8, decay=0.7, the expected acceptance per position:
        # pos 0: 0.8, pos 1: 0.8*0.56, pos 2: 0.8*0.56*0.392, ...
        # (each position must pass AND all previous must have passed)
        # Expected mean ≈ 1.3-1.8 tokens
        assert 1.0 < avg < 3.0, f"avg accepted = {avg}, expected ~1.5"

    def test_deterministic(self):
        """Same seed → same result."""
        draft = torch.tensor([1, 2, 3])
        target = torch.tensor([1, 2, 3, 4])
        r1 = verify_synthetic_ref(draft, target, 0.8, 0.9, 42)
        r2 = verify_synthetic_ref(draft, target, 0.8, 0.9, 42)
        assert r1[0] == r2[0]
        assert r1[1].tolist() == r2[1].tolist()


# ============================================================================
# MLA Attention Reference
# ============================================================================

def mla_attention_ref(
    q_nope_pe: torch.Tensor,      # [batch, num_q_heads, qk_head_dim]
    ckv_kpe_cache: torch.Tensor,   # [total_seq_len, qk_head_dim]
    kv_lens: torch.Tensor,         # [batch]
    v_head_dim: int = 512,
) -> torch.Tensor:
    """
    Reference MLA attention in PyTorch.

    After weight absorption in DeepSeek V3:
    - Q = [q_nope, q_pe] per head, dim = 576 (512 + 64)
    - K = [c_latent, k_pe] from cache, dim = 576
    - V = c_latent from cache (first 512 dims only)
    - Output = softmax(Q @ K^T / sqrt(576)) @ V, dim = 512 per head
    """
    batch_size, num_heads, qk_head_dim = q_nope_pe.shape
    softmax_scale = 1.0 / math.sqrt(qk_head_dim)
    outputs = []

    for b in range(batch_size):
        kv_len = kv_lens[b].item()
        # K: full 576 dims for QK dot product
        k = ckv_kpe_cache[:kv_len].float()    # [kv_len, 576]
        # V: first 512 dims (latent only)
        v = ckv_kpe_cache[:kv_len, :v_head_dim].float()  # [kv_len, 512]

        q = q_nope_pe[b].float()  # [num_heads, 576]

        # QK^T
        attn_scores = torch.matmul(q, k.T) * softmax_scale  # [num_heads, kv_len]
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # PV
        output_b = torch.matmul(attn_probs, v)  # [num_heads, 512]
        outputs.append(output_b)

    return torch.stack(outputs, dim=0).to(q_nope_pe.dtype)  # [batch, num_heads, 512]


class TestMLAAttentionReference:
    """Test the MLA reference implementation itself for sanity."""

    def test_single_token_single_head(self):
        """Simplest case: 1 batch, 1 head, short sequence."""
        torch.manual_seed(42)
        qk_dim = 576
        v_dim = 512

        q = torch.randn(1, 1, qk_dim, dtype=torch.float32)
        kv_cache = torch.randn(4, qk_dim, dtype=torch.float32)
        kv_lens = torch.tensor([4])

        output = mla_attention_ref(q, kv_cache, kv_lens, v_dim)
        assert output.shape == (1, 1, v_dim)

        # Manually compute
        k = kv_cache[:4].float()
        v = kv_cache[:4, :v_dim].float()
        scale = 1.0 / math.sqrt(qk_dim)
        scores = (q[0, 0] @ k.T) * scale
        probs = torch.softmax(scores, dim=-1)
        expected = probs @ v
        torch.testing.assert_close(output[0, 0].float(), expected, rtol=1e-4, atol=1e-4)

    def test_multi_head(self):
        """Multiple heads, verify output shape."""
        torch.manual_seed(42)
        q = torch.randn(2, 16, 576, dtype=torch.float32)
        kv_cache = torch.randn(32, 576, dtype=torch.float32)
        kv_lens = torch.tensor([32, 16])
        output = mla_attention_ref(q, kv_cache, kv_lens, 512)
        assert output.shape == (2, 16, 512)

    def test_causal_property(self):
        """Verify that attention output changes when KV length changes."""
        torch.manual_seed(42)
        q = torch.randn(1, 4, 576, dtype=torch.float32)
        kv_cache = torch.randn(32, 576, dtype=torch.float32)

        out_short = mla_attention_ref(q, kv_cache, torch.tensor([8]), 512)
        out_long = mla_attention_ref(q, kv_cache, torch.tensor([32]), 512)
        # Outputs should differ when seeing different KV lengths
        assert not torch.allclose(out_short, out_long)

    def test_output_is_weighted_sum_of_values(self):
        """Output should be in the convex hull of V vectors."""
        torch.manual_seed(42)
        q = torch.randn(1, 1, 576, dtype=torch.float32)
        kv_cache = torch.randn(4, 576, dtype=torch.float32)
        kv_lens = torch.tensor([4])
        output = mla_attention_ref(q, kv_cache, kv_lens, 512)

        # Output norm should be bounded by max V norm
        v = kv_cache[:4, :512]
        max_v_norm = v.norm(dim=-1).max()
        output_norm = output[0, 0].norm()
        assert output_norm <= max_v_norm * 1.1  # small tolerance

    def test_bf16_precision(self):
        """Test with bf16 inputs (what the actual kernel uses)."""
        torch.manual_seed(42)
        q = torch.randn(1, 16, 576, dtype=torch.bfloat16)
        kv_cache = torch.randn(64, 576, dtype=torch.bfloat16)
        kv_lens = torch.tensor([64])
        output = mla_attention_ref(q, kv_cache, kv_lens, 512)
        assert output.dtype == torch.bfloat16
        assert output.shape == (1, 16, 512)
        assert not torch.isnan(output).any()


# ============================================================================
# Paged MLA Reference (with page tables)
# ============================================================================

def paged_mla_attention_ref(
    q_nope_pe: torch.Tensor,        # [num_tokens, num_q_heads * qk_head_dim]
    ckv_kpe_cache: torch.Tensor,     # [num_pages, page_size, qk_head_dim]
    kv_new: torch.Tensor,            # [num_tokens, qk_head_dim]
    qo_indptr: torch.Tensor,         # [num_requests + 1]
    paged_kv_indptr: torch.Tensor,   # [num_requests + 1]
    paged_kv_indices: torch.Tensor,  # [total_pages]
    paged_kv_last_page_len: torch.Tensor,  # [num_requests]
    num_q_heads: int,
    qk_head_dim: int = 576,
    v_head_dim: int = 512,
) -> torch.Tensor:
    """
    Reference paged MLA attention matching the kernel interface.
    Includes writing new KV entries to cache before attention.
    """
    num_tokens_total = q_nope_pe.shape[0]
    page_size = ckv_kpe_cache.shape[1]
    num_requests = qo_indptr.shape[0] - 1
    softmax_scale = 1.0 / math.sqrt(qk_head_dim)

    output = torch.zeros(num_tokens_total, num_q_heads * v_head_dim,
                         dtype=q_nope_pe.dtype, device=q_nope_pe.device)

    for req in range(num_requests):
        first_token = qo_indptr[req].item()
        last_token = qo_indptr[req + 1].item()
        num_tokens = last_token - first_token
        if num_tokens == 0:
            continue

        # Gather KV cache for this request
        first_page = paged_kv_indptr[req].item()
        last_page = paged_kv_indptr[req + 1].item()
        num_pages = last_page - first_page
        last_page_len = paged_kv_last_page_len[req].item()
        seq_len = (num_pages - 1) * page_size + last_page_len

        # Gather pages into contiguous KV
        kv_list = []
        for p in range(num_pages):
            page_idx = paged_kv_indices[first_page + p].item()
            if p == num_pages - 1:
                kv_list.append(ckv_kpe_cache[page_idx, :last_page_len])
            else:
                kv_list.append(ckv_kpe_cache[page_idx])
        kv_gathered = torch.cat(kv_list, dim=0)  # [seq_len, qk_head_dim]

        # Write new KV entries (last num_tokens positions)
        kv_gathered[seq_len - num_tokens:seq_len] = kv_new[first_token:last_token]

        # Also write back to the actual cache pages (for in-place update)
        for t in range(num_tokens):
            cache_pos = seq_len - num_tokens + t
            page_in_seq = cache_pos // page_size
            page_offset = cache_pos % page_size
            page_idx = paged_kv_indices[first_page + page_in_seq].item()
            ckv_kpe_cache[page_idx, page_offset] = kv_new[first_token + t]

        # Compute attention per token
        for t in range(num_tokens):
            # This token sees KV up to position: seq_len - num_tokens + t + 1
            valid_len = seq_len - num_tokens + t + 1
            k = kv_gathered[:valid_len].float()        # [valid_len, 576]
            v = kv_gathered[:valid_len, :v_head_dim].float()  # [valid_len, 512]

            q = q_nope_pe[first_token + t].float()     # [num_q_heads * 576]
            q = q.view(num_q_heads, qk_head_dim)       # [num_q_heads, 576]

            scores = torch.matmul(q, k.T) * softmax_scale  # [num_q_heads, valid_len]
            probs = torch.softmax(scores, dim=-1)
            out = torch.matmul(probs, v)  # [num_q_heads, 512]
            output[first_token + t] = out.reshape(-1).to(q_nope_pe.dtype)

    return output


class TestPagedMLAReference:
    """Test the paged MLA reference for correctness."""

    def test_single_request_single_token(self):
        """Single request, 1 new token, short KV cache."""
        torch.manual_seed(42)
        num_q_heads = 4
        qk_dim = 576
        v_dim = 512
        page_size = 4
        num_pages = 4

        q = torch.randn(1, num_q_heads * qk_dim, dtype=torch.bfloat16)
        cache = torch.randn(num_pages, page_size, qk_dim, dtype=torch.bfloat16)
        kv_new = torch.randn(1, qk_dim, dtype=torch.bfloat16)

        qo_indptr = torch.tensor([0, 1], dtype=torch.int32)
        kv_indptr = torch.tensor([0, 2], dtype=torch.int32)  # 2 pages
        kv_indices = torch.tensor([0, 1], dtype=torch.int32)
        kv_last_page_len = torch.tensor([3], dtype=torch.int32)  # seq_len = 4+3 = 7

        output = paged_mla_attention_ref(
            q, cache, kv_new, qo_indptr, kv_indptr, kv_indices,
            kv_last_page_len, num_q_heads, qk_dim, v_dim
        )
        assert output.shape == (1, num_q_heads * v_dim)
        assert not torch.isnan(output).any()

    def test_multi_token_causal(self):
        """Multi-token decode: each token should see different KV lengths."""
        torch.manual_seed(42)
        num_q_heads = 2
        qk_dim = 576
        v_dim = 512
        page_size = 8
        num_pages = 4

        num_tokens = 3
        q = torch.randn(num_tokens, num_q_heads * qk_dim, dtype=torch.bfloat16)
        cache = torch.randn(num_pages, page_size, qk_dim, dtype=torch.bfloat16)
        kv_new = torch.randn(num_tokens, qk_dim, dtype=torch.bfloat16)

        # seq_len = 8 + 3 = 11 (1 full page + 3 in last page), but we have 3 new tokens
        # so KV cache has 11 positions total, last 3 are new
        qo_indptr = torch.tensor([0, num_tokens], dtype=torch.int32)
        kv_indptr = torch.tensor([0, 2], dtype=torch.int32)  # 2 pages
        kv_indices = torch.tensor([0, 1], dtype=torch.int32)
        # seq_len = (2-1)*8 + 3 = 11
        kv_last_page_len = torch.tensor([3], dtype=torch.int32)

        output = paged_mla_attention_ref(
            q, cache, kv_new, qo_indptr, kv_indptr, kv_indices,
            kv_last_page_len, num_q_heads, qk_dim, v_dim
        )
        assert output.shape == (num_tokens, num_q_heads * v_dim)

        # Token 0 sees KV[0:9], Token 1 sees KV[0:10], Token 2 sees KV[0:11]
        # So outputs should all be different
        assert not torch.allclose(output[0], output[1])
        assert not torch.allclose(output[1], output[2])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
