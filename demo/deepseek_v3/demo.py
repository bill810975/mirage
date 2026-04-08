from transformers import AutoTokenizer, AutoConfig
import torch
import torch.distributed as dist
import argparse
import os
import json

from mirage.mpk.models.deepseek_v3.builder import DeepSeekV3Builder
from mirage.mpk.models.graph_builder import MirageModelConfig


DEFAULT_SAVE_DIR = os.path.join("outputs", "deepseek_v3")
MAX_SAVE_TOKENS = 100

# DeepSeek V3 architecture constants
# MLA: 128 heads, compressed KV dim 512, rope dim 64, total head dim 576
DEEPSEEK_V3_NUM_HEADS = 128
DEEPSEEK_V3_KV_LORA_RANK = 512
DEEPSEEK_V3_QK_ROPE_HEAD_DIM = 64
DEEPSEEK_V3_HEAD_DIM_TOTAL = DEEPSEEK_V3_KV_LORA_RANK + DEEPSEEK_V3_QK_ROPE_HEAD_DIM  # 576


def grid_for_rmsnorm_linear_layer(size: int):
    if size / 96 > 400:
        assert size % 256 == 0, f"FATAL: Linear layer size not supported, it's {size}."
        return size // 256
    if size % 96 == 0:
        return 96
    elif size % 64 == 0:
        return 64


def max_factor_leq_n(m: int, n: int) -> int:
    """Return the largest factor of m that is less than or equal to n."""
    max_factor = 1
    i = 1
    while i * i <= m:
        if m % i == 0:
            if i <= n:
                max_factor = max(max_factor, i)
            if m // i <= n:
                max_factor = max(max_factor, m // i)
        i += 1
    return max_factor


def run_correctness_test(args, state_dict, layer_indices, rank, world_size):
    """Run PyTorch reference on selected layers and compare against MPK.

    Both use the SAME real weights from the checkpoint.
    Layer indices specify which layers to include (e.g., [0, 3] = 1 dense + 1 MoE).
    The MTP layer (index 61) is included if --mtp is set.
    """
    import torch.nn.functional as F
    import math

    device = f"cuda:{rank}"
    layer_indices = sorted(layer_indices)
    num_layers = len(layer_indices)
    include_mtp = args.mtp

    # DeepSeek V3 constants (after weight absorption)
    KV_LORA_RANK = DEEPSEEK_V3_KV_LORA_RANK  # 512
    QK_ROPE_HEAD_DIM = DEEPSEEK_V3_QK_ROPE_HEAD_DIM  # 64
    QK_HEAD_DIM = DEEPSEEK_V3_HEAD_DIM_TOTAL  # 576
    V_HEAD_DIM = KV_LORA_RANK  # 512
    NUM_Q_HEADS = DEEPSEEK_V3_NUM_HEADS // world_size
    HIDDEN = 7168
    FIRST_MOE = 3
    NUM_EXPERTS = 256
    TOPK = 8

    def rms_norm(x, weight, eps=1e-6):
        orig = x.dtype
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (weight.float() * x.float() * torch.rsqrt(v + eps)).to(orig)

    def sigmoid_topk(logits, bias, k):
        scores = torch.sigmoid(logits.float())
        routing = scores + bias.float().unsqueeze(0)
        _, idx = torch.topk(routing, k, dim=-1)
        w = torch.gather(scores, 1, idx)
        return w / w.sum(dim=-1, keepdim=True), idx

    def mla_attention(hidden, prefix, sd, kv_cache, seq_pos, num_heads):
        bs = hidden.shape[0]
        p = prefix + "self_attn."
        # q path (BF16 linear for now since FP8 core not ready)
        q_a = F.linear(hidden.float(), sd[f"{p}q_a_proj.weight"].float()).to(hidden.dtype)
        q_a = rms_norm(q_a, sd[f"{p}q_a_layernorm.weight"])
        q = F.linear(q_a.float(), sd[f"{p}q_b_proj.weight"].float()).to(hidden.dtype)
        q = q.view(bs, num_heads, QK_HEAD_DIM)
        # kv path
        kv_w = sd[f"{p}kv_a_proj_with_mqa.weight"]
        kv_full = F.linear(hidden.float(), kv_w.float()).to(hidden.dtype)
        c_lat = kv_full[:, :KV_LORA_RANK]
        k_pe = kv_full[:, KV_LORA_RANK:]
        c_lat = rms_norm(c_lat, sd[f"{p}kv_a_layernorm.weight"])
        kv_new = torch.cat([c_lat, k_pe], dim=-1)
        for b in range(bs):
            kv_cache[seq_pos + b] = kv_new[b]
        kv_all = kv_cache[:seq_pos + bs]
        # attention
        q_n, q_p = q[:, :, :KV_LORA_RANK], q[:, :, KV_LORA_RANK:]
        k_n, k_p = kv_all[:, :KV_LORA_RANK], kv_all[:, KV_LORA_RANK:]
        s = (torch.einsum('bhd,sd->bhs', q_n.float(), k_n.float()) +
             torch.einsum('bhd,sd->bhs', q_p.float(), k_p.float()))
        s = s / math.sqrt(QK_HEAD_DIM)
        attn = F.softmax(s, dim=-1)
        v = kv_all[:, :V_HEAD_DIM]
        out = torch.einsum('bhs,sd->bhd', attn, v.float()).to(hidden.dtype)
        flat = out.reshape(bs, num_heads * V_HEAD_DIM)
        return F.linear(flat.float(), sd[f"{p}o_proj.weight"].float()).to(hidden.dtype)

    def dense_mlp(hidden, prefix, sd):
        gu = F.linear(hidden.float(), sd[f"{prefix}mlp.gate_up_proj.weight"].float()).to(hidden.dtype)
        mid = gu.shape[-1] // 2
        x = F.silu(gu[:, :mid].float()).to(hidden.dtype) * gu[:, mid:]
        return F.linear(x.float(), sd[f"{prefix}mlp.down_proj.weight"].float()).to(hidden.dtype)

    def moe_mlp(hidden, prefix, sd):
        bs = hidden.shape[0]
        p = prefix + "mlp."
        logits = F.linear(hidden.float(), sd[f"{p}gate.weight"].float()).to(hidden.dtype)
        weights, topk_idx = sigmoid_topk(logits, sd[f"{p}gate.e_score_correction_bias"], TOPK)
        out = torch.zeros(bs, HIDDEN, device=device, dtype=hidden.dtype)
        for b in range(bs):
            for ki in range(TOPK):
                eid = topk_idx[b, ki].item()
                w = weights[b, ki].item()
                w13 = sd[f"{p}experts.w13.weight"][eid]
                gu = F.linear(hidden[b:b+1].float(), w13.float()).to(hidden.dtype)
                mid = gu.shape[-1] // 2
                x = F.silu(gu[:, :mid].float()).to(hidden.dtype) * gu[:, mid:]
                w2 = sd[f"{p}experts.w2.weight"][eid]
                out[b] += w * F.linear(x.float(), w2.float()).squeeze(0).to(hidden.dtype)
        # Shared expert
        sp = p + "shared_experts."
        sg = F.silu(F.linear(hidden.float(), sd[f"{sp}gate_proj.weight"].float()).to(hidden.dtype))
        su = F.linear(hidden.float(), sd[f"{sp}up_proj.weight"].float()).to(hidden.dtype)
        sd_out = F.linear((sg * su).float(), sd[f"{sp}down_proj.weight"].float()).to(hidden.dtype)
        return out + sd_out

    print(f"\n{'='*60}")
    print(f"Correctness Test: layers={layer_indices}, mtp={include_mtp}")
    print(f"{'='*60}")

    # Run PyTorch reference
    token_ids = torch.tensor([1, 2, 3], device=device, dtype=torch.long)  # simple input
    max_seq = 64
    kv_caches = [torch.zeros(max_seq, QK_HEAD_DIM, device=device, dtype=torch.bfloat16)
                 for _ in range(num_layers + (1 if include_mtp else 0))]

    hidden = F.embedding(token_ids[:1], state_dict["model.embed_tokens.weight"])
    if hidden.dim() == 1:
        hidden = hidden.unsqueeze(0)

    for cache_idx, layer_idx in enumerate(layer_indices):
        prefix = f"model.layers.{layer_idx}."
        normed = rms_norm(hidden, state_dict[f"{prefix}input_layernorm.weight"])
        attn_out = mla_attention(normed, prefix, state_dict, kv_caches[cache_idx], 0, NUM_Q_HEADS)
        hidden = hidden + attn_out
        normed = rms_norm(hidden, state_dict[f"{prefix}post_attention_layernorm.weight"])
        if layer_idx < FIRST_MOE:
            mlp_out = dense_mlp(normed, prefix, state_dict)
        else:
            mlp_out = moe_mlp(normed, prefix, state_dict)
        hidden = hidden + mlp_out

    hidden = rms_norm(hidden, state_dict["model.norm.weight"])
    logits = F.linear(hidden.float(), state_dict["lm_head.weight"].float())
    ref_token = logits.argmax(dim=-1).item()
    print(f"PyTorch reference output token: {ref_token}")
    print(f"PyTorch logits[0,:5]: {logits[0,:5].tolist()}")

    # TODO: Run MPK with same layers and compare
    # This requires builder.build_layers to respect the layer_indices list
    # For now, just validate the PyTorch reference runs correctly
    print(f"\nPyTorch reference completed successfully.")
    print(f"MPK comparison pending: builder needs --layers support in build_layers().")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek V3 demo with Mirage megakernel")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to converted DeepSeek V3 weights")
    parser.add_argument("--use-mirage", action="store_true",
                        help="Use Mirage megakernel")
    parser.add_argument("--profiling", action="store_true",
                        help="Enable profiling to generate trace")
    parser.add_argument("--max-num-batched-tokens", default=8, type=int,
                        help="Max number of tokens in a batch")
    parser.add_argument("--max-num-batched-requests", default=1, type=int,
                        help="Max number of requests in a batch")
    parser.add_argument("--page-size", default=128, type=int,
                        help="Page size for KV cache")
    parser.add_argument("--max-num-pages", default=64, type=int,
                        help="Max number of pages")
    parser.add_argument("--max-seq-length", default=4096, type=int,
                        help="Max sequence length")
    parser.add_argument("--prompt", type=str,
                        default="Give me a short introduction to large language model.",
                        help="Input prompt text")
    parser.add_argument("--mtp", action="store_true",
                        help="Enable MTP speculative decoding")
    parser.add_argument("--num-speculative-tokens", default=1, type=int,
                        choices=range(1, 8),
                        help="Number of speculative tokens for MTP (1-7)")
    parser.add_argument("--rejection-sample-method", default="strict", type=str,
                        choices=["strict", "probabilistic", "synthetic"],
                        help="Rejection sampling method for speculative decoding")
    parser.add_argument("--output-dir", help="Output files directory")
    parser.add_argument("--trace-name", default="", help="Perfetto trace output name")
    parser.add_argument("--ignore-eos", action="store_true",
                        help="Ignore eos token during generation")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Decode cap for CI determinism")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--do-sample", dest="do_sample", action="store_true",
                        help="Enable sampling (default off)")
    parser.add_argument("--save-tokens", nargs="?", const="auto", default=None,
                        help=(
                            "Optionally dump first N generated token_ids, text, and latency to JSON. "
                            "If path omitted, saves to outputs/deepseek_v3/{torch_output.json|mpk_output.json}."
                        ))
    # Developer correctness testing
    parser.add_argument("--correctness", action="store_true",
                        help="Run correctness test: compare MPK output against PyTorch reference")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated list of layer indices to load (e.g. '0,3,60'). "
                             "Used with --correctness to test a reduced model.")

    args = parser.parse_args()

    # Multi-GPU setup via MPI
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        world_size = comm.Get_size()
        rank = comm.Get_rank()
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
    except ImportError:
        world_size = 1
        rank = 0

    if args.save_tokens:
        if args.save_tokens == "auto":
            filename = "mpk_output.json" if args.use_mirage else "torch_output.json"
            save_path = os.path.join(DEFAULT_SAVE_DIR, filename)
        else:
            save_path = args.save_tokens
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    else:
        save_path = None

    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    global print
    if rank != 0:
        print = lambda *_, **__: None

    print("Input arguments:", args)
    print(f"world_size({world_size}) rank({rank})")
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(rank)

    # Load model config and tokenizer from converted weights
    print(f"Loading model config from: {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Extract DeepSeek V3 architecture parameters
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    vocab_size = config.vocab_size
    # MLA parameters
    kv_lora_rank = getattr(config, "kv_lora_rank", DEEPSEEK_V3_KV_LORA_RANK)
    qk_rope_head_dim = getattr(config, "qk_rope_head_dim", DEEPSEEK_V3_QK_ROPE_HEAD_DIM)
    ckv_kpe_dim = kv_lora_rank + qk_rope_head_dim  # 576 for DeepSeek V3
    num_attention_heads = config.num_attention_heads

    print(f"Model config: hidden_size={hidden_size}, num_layers={num_layers}, "
          f"vocab_size={vocab_size}, num_heads={num_attention_heads}, "
          f"kv_lora_rank={kv_lora_rank}, qk_rope_head_dim={qk_rope_head_dim}, "
          f"ckv_kpe_dim={ckv_kpe_dim}")

    total_num_requests = 1 if not args.use_mirage else args.max_num_batched_requests

    # Allocate token buffers
    tokens = torch.full(
        (total_num_requests, args.max_seq_length), 0, dtype=torch.long, device="cuda"
    )
    input_tokens = torch.full(
        (args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda"
    )
    output_tokens = torch.full(
        (args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda"
    )

    # Tokenize prompt
    messages = [
        {"role": "user", "content": args.prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
    for r in range(total_num_requests):
        for i in range(model_inputs.input_ids.shape[-1]):
            tokens[r, i] = model_inputs.input_ids[0, i]
    prompt_lengths = torch.full(
        (total_num_requests,), model_inputs.input_ids.shape[-1],
        dtype=torch.int, device="cuda"
    )

    step = torch.full((total_num_requests,), 0, dtype=torch.int32, device="cuda")
    num_new_tokens = torch.full((total_num_requests,), 1, dtype=torch.int32, device="cuda")

    starter, ender = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )

    if args.use_mirage:
        import mirage as mi

        # Pad vocab_size for task graph creation
        padded_vocab_size = ((vocab_size + 255) // 256) * 256

        if args.profiling:
            profiler_tensor = torch.zeros(
                3000 * 128, dtype=torch.uint64, device="cuda"
            ).contiguous()
        else:
            profiler_tensor = None

        # MTP speculative decoding config
        if args.mtp:
            spec_decode_config = mi.mpk.spec_decode_class(
                "lookahead",
                ngram_size=3,
                spec_length=args.num_speculative_tokens,
            )
        else:
            spec_decode_config = None

        num_workers, num_schedulers = mi.get_configurations_from_gpu(rank)

        # Meta tensor buffers for paged attention
        qo_indptr_buffer = torch.empty(
            args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda"
        )
        paged_kv_indptr_buffer = torch.empty(
            args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda"
        )
        paged_kv_indices_buffer = torch.empty(
            args.max_num_pages, dtype=torch.int32, device="cuda"
        )
        paged_kv_last_page_len_buffer = torch.empty(
            args.max_num_batched_requests, dtype=torch.int32, device="cuda"
        )

        # MLA uses a single combined ckv_kpe cache per layer
        # Shape: (num_layers, max_num_pages, page_size, ckv_kpe_dim)
        # where ckv_kpe_dim = kv_lora_rank + qk_rope_head_dim = 576
        ckv_kpe_cache = torch.zeros(
            (num_layers, args.max_num_pages, args.page_size, ckv_kpe_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )

        eos_token_id = config.eos_token_id if not args.ignore_eos else -1
        # Handle eos_token_id being a list (common in DeepSeek V3)
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]

        mpk = mi.PersistentKernel(
            mode="offline",
            world_size=world_size,
            mpi_rank=rank,
            num_workers=num_workers,
            num_local_schedulers=num_schedulers,
            num_remote_schedulers=0,
            max_seq_length=args.max_seq_length,
            max_num_batched_requests=args.max_num_batched_requests,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_pages=args.max_num_pages,
            page_size=args.page_size,
            eos_token_id=eos_token_id,
            meta_tensors={
                "step": step,
                "tokens": tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "num_new_tokens": num_new_tokens,
                "prompt_lengths": prompt_lengths,
                "qo_indptr_buffer": qo_indptr_buffer,
                "paged_kv_indptr_buffer": paged_kv_indptr_buffer,
                "paged_kv_indices_buffer": paged_kv_indices_buffer,
                "paged_kv_last_page_len_buffer": paged_kv_last_page_len_buffer,
            },
            profiler_tensor=profiler_tensor,
            trace_name=args.trace_name,
            spec_decode_config=spec_decode_config,
            use_cutlass_kernel=True,
        )

        # Load state dict from converted weights
        print(f"Loading model weights from: {args.model_path}")
        from safetensors.torch import load_file
        weight_file = os.path.join(
            args.model_path, f"model{rank}-mp{world_size}.safetensors"
        )
        if os.path.exists(weight_file):
            state_dict = load_file(weight_file, device="cuda")
        else:
            # Try single-file format
            candidates = [
                os.path.join(args.model_path, "model.safetensors"),
            ]
            state_dict = None
            for candidate in candidates:
                if os.path.exists(candidate):
                    state_dict = load_file(candidate, device="cuda")
                    break
            if state_dict is None:
                # Try loading from multiple shard files
                import glob
                shard_files = sorted(glob.glob(
                    os.path.join(args.model_path, "model-*.safetensors")
                ))
                if shard_files:
                    state_dict = {}
                    for shard_file in shard_files:
                        state_dict.update(load_file(shard_file, device="cuda"))
                else:
                    raise FileNotFoundError(
                        f"Could not find model weights at {args.model_path}. "
                        f"Expected {weight_file} or model.safetensors or model-*.safetensors"
                    )

        # Parse layer indices for correctness mode
        layer_indices_arg = None
        if args.correctness and args.layers:
            layer_indices_arg = [int(x) for x in args.layers.split(',')]

        # Correctness test: run PyTorch reference first
        if args.correctness:
            test_layers = layer_indices_arg if layer_indices_arg else list(range(num_layers))
            run_correctness_test(args, state_dict, test_layers, rank, world_size)

        # Build MLA model config for the builder
        model_config = MirageModelConfig(
            hidden_size=hidden_size,
            intermediate_size=getattr(config, "intermediate_size", None) or getattr(config, "moe_intermediate_size", 18432),
            vocab_size=vocab_size,
            local_num_q_heads=num_attention_heads // world_size,
            local_num_kv_heads=1,  # MLA uses single KV head (shared latent)
            head_dim=ckv_kpe_dim,  # 576 for MLA
            num_layers=num_layers,
            k_cache=[ckv_kpe_cache[i] for i in range(num_layers)],
            v_cache=[ckv_kpe_cache[i] for i in range(num_layers)],
            position_embeddings=None,
            state_dict=state_dict,
            with_lm_head=True,
        )

        # Build the computation graph using the DeepSeek V3 builder
        builder = DeepSeekV3Builder(mpk)
        builder.build_from_config(model_config, layer_indices=layer_indices_arg)

        results = mpk.kn_graph.generate_task_graph(
            num_gpus=world_size, my_gpu_id=rank
        )
        with open(f"task_graph_{rank}.json", "w") as f:
            f.write(results["json_file"])
        with open(f"kernel_{rank}.cu", "w") as f:
            f.write(results["cuda_code"])

        mpk.compile(output_dir=args.output_dir)

        # Run inference
        print("Starting inference with Mirage megakernel...")
        starter.record()
        mpk()
        ender.record()
        torch.cuda.synchronize()
        run_time = starter.elapsed_time(ender)

        print("tokens.shape = ", tokens.shape)
        for r in range(total_num_requests):
            generated_ids = tokens[r, : step[r] + 1]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(response)

        if total_num_requests > 1:
            print(f"Output length of each batch is same: {(step.max() == step.min()).item()}")

        print("Prompt length {}, generate length {}, per-token latency (both prefill and decode): {:.3f} ms".format(
            prompt_lengths[0], step.max().item() + 1 - prompt_lengths[0],
            run_time / (step.max().item() + 1)
        ))

        # Dump outputs to json
        if save_path and rank == 0:
            end_idx = step[0].item() + 1
            prompt_len = prompt_lengths[0].item()
            tokens_generated = max(0, end_idx - prompt_len)
            per_tok_ms = run_time / max(tokens_generated, 1)
            slice_end = min(end_idx, prompt_len + MAX_SAVE_TOKENS)
            token_ids = tokens[0, prompt_len:slice_end].tolist()
            response_text = tokenizer.decode(
                tokens[0, :end_idx], skip_special_tokens=True
            )
            out = {
                "token_ids": token_ids,
                "text": response_text,
                "latency_ms_per_token": per_tok_ms,
                "prompt_length": prompt_len,
                "generate_length": tokens_generated,
                "mode": "mpk",
            }
            with open(save_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Saved tokens to {save_path}")

    else:
        # Native PyTorch path (without Mirage)
        # DeepSeek V3 requires the model implementation for non-Mirage inference
        try:
            from transformers import AutoModelForCausalLM
            print(f"Loading DeepSeek V3 model from: {args.model_path}")
            with torch.device("cuda"):
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                ).to("cuda")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load DeepSeek V3 model for native inference: {e}. "
                "For native PyTorch inference, ensure transformers supports the model "
                "or use --use-mirage for Mirage megakernel inference."
            )

        prompt_len = prompt_lengths[0].item()
        output_len = (
            args.max_new_tokens
            if args.max_new_tokens is not None
            else (tokens.size(1) - prompt_len)
        )
        output_len = max(0, min(output_len, tokens.size(1) - prompt_len))
        decode_limit = prompt_len + output_len
        prev_pos = 0
        stream = torch.cuda.Stream()

        for cur_pos in range(prompt_len, decode_limit):
            step.fill_(cur_pos - 1)
            input_ids = tokens[:1, prev_pos:cur_pos]
            with torch.no_grad():
                logits = model(input_ids=input_ids).logits
            next_token = logits[:, -1, :].argmax(dim=-1)
            tokens[0, cur_pos] = next_token[0]
            prev_pos = cur_pos
            eos_id = config.eos_token_id
            if isinstance(eos_id, list):
                if next_token[0].item() in eos_id:
                    break
            elif next_token[0].item() == eos_id:
                break
            if cur_pos == prompt_len:
                torch.cuda.synchronize()
                starter.record()

        ender.record()
        torch.cuda.synchronize()
        run_time = starter.elapsed_time(ender)

        end_idx = prev_pos + 1
        generated_ids = tokens[:1, :end_idx]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
        print(
            "Prompt length {}, generate length {}, per-token latency {} ms".format(
                prompt_len, cur_pos - prompt_len,
                run_time / max(cur_pos - prompt_len, 1)
            )
        )

        # Dump outputs to json
        if save_path and rank == 0:
            tokens_generated = max(0, end_idx - prompt_len)
            per_tok_ms = run_time / max(tokens_generated, 1)
            slice_end = min(end_idx, prompt_len + MAX_SAVE_TOKENS)
            token_ids = tokens[0, prompt_len:slice_end].tolist()
            out = {
                "token_ids": token_ids,
                "text": tokenizer.decode(
                    tokens[0, :end_idx], skip_special_tokens=True
                ),
                "latency_ms_per_token": per_tok_ms,
                "prompt_length": prompt_len,
                "generate_length": tokens_generated,
                "mode": "torch",
            }
            with open(save_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Saved tokens to {save_path}")

    if world_size > 1:
        dist.destroy_process_group()
