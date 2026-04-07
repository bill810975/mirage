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

        # Build MLA model config for the builder
        # For MLA, we pass the single ckv_kpe_cache as both k_cache and v_cache
        # The builder will handle the MLA-specific cache layout
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
            position_embeddings=None,  # MLA handles position embeddings differently
            state_dict=state_dict,
            with_lm_head=True,
        )

        # Build the computation graph using the DeepSeek V3 builder
        builder = DeepSeekV3Builder(mpk)
        builder.build_from_config(model_config)

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
