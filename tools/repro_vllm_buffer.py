"""Reproduce vLLM's DeepEP usage against UCCL, isolated from vLLM.

Builds deep_ep.Buffer with exactly the arguments
vllm/distributed/device_communicators/all2all.py::DeepEPHTAll2AllManager
uses, then runs one dispatch+combine at GLM-5.3's MoE shapes.

--num-sms is the knob under test: vLLM hardcodes 20 (DeepEPAll2AllManagerBase),
while DeepEP's own test tuned to 24 and passed.
"""
import argparse, os, sys
import torch
import torch.distributed as dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-sms", type=int, default=20)
    ap.add_argument("--num-tokens", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=6144)      # GLM-5.3
    ap.add_argument("--num-experts", type=int, default=256)  # GLM-5.3
    ap.add_argument("--num-topk", type=int, default=8)       # GLM-5.3
    ap.add_argument("--buffer-mb", type=int, default=1024)   # VLLM_DEEPEP_BUFFER_SIZE_MB
    ap.add_argument("--fp8", action="store_true",
                    help="dispatch FP8 + scales, as vLLM does for a block-quantized model")
    ap.add_argument("--starve-ranks", action="store_true",
                    help="route every token to the first 8 experts, so only EP rank 0 "
                         "receives tokens and every other rank gets zero. Reproduces the "
                         "inference-only condition vLLM notes in modular_kernel.py: "
                         "'none of the tokens from the all2all reach this EP rank', which "
                         "'is only relevant for CUDAGraph incompatible all2all kernels "
                         "like the DeepEP high-throughput kernels'. Training never hits "
                         "this -- Megatron pushes large uniform batches.")
    ap.add_argument("--iters", type=int, default=1,
                    help="repeat dispatch+combine; GLM-5.3 has 75 sparse layers sharing one Buffer")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    group = dist.new_group(list(range(world)))

    import deep_ep

    nbytes = args.buffer_mb * 1024 * 1024
    num_qps_per_rank = args.num_sms // 2      # exactly what vLLM computes

    if rank == 0:
        print(f"world={world} num_sms={args.num_sms} num_qps_per_rank={num_qps_per_rank} "
              f"nvl={nbytes} rdma={nbytes} hidden={args.hidden} "
              f"experts={args.num_experts} topk={args.num_topk}", flush=True)

    # vLLM's kwargs, verbatim
    buf = deep_ep.Buffer(
        group=group,
        num_nvl_bytes=nbytes,
        num_rdma_bytes=nbytes,
        low_latency_mode=False,
        num_qps_per_rank=num_qps_per_rank,
        explicitly_destroy=True,
    )
    deep_ep.Buffer.set_num_sms(args.num_sms)
    dist.barrier(group)
    if rank == 0:
        print("buffer constructed + set_num_sms OK", flush=True)

    x = torch.randn((args.num_tokens, args.hidden), dtype=torch.bfloat16, device="cuda")
    logits = torch.randn((args.num_tokens, args.num_experts), dtype=torch.float32, device="cuda")
    if args.starve_ranks:
        # Bias hard toward experts 0..num_topk-1 so all tokens land on EP rank 0.
        logits[:, : args.num_topk] += 1e4
    topk_w, topk_idx = torch.topk(logits, args.num_topk, dim=-1)
    topk_idx = topk_idx.to(torch.int64)
    if rank == 0:
        uniq = torch.unique(topk_idx).tolist()
        print(f"routing touches {len(uniq)} distinct experts (ids {uniq[:10]}...)", flush=True)

    layout = buf.get_dispatch_layout(topk_idx, args.num_experts)
    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_in_rank, _ = layout
    dist.barrier(group)
    if rank == 0:
        print("get_dispatch_layout OK", flush=True)

    if args.fp8:
        from deep_ep.utils import per_token_cast_to_fp8, per_token_cast_back
        x_q = per_token_cast_to_fp8(x)
        x_in = (x_q[0], x_q[1].T.contiguous().T)   # scale layout the DeepEP test uses
    else:
        x_in = x

    for it in range(args.iters):
        recv_x, recv_idx, recv_w, num_recv_per_expert, handle, _ = buf.dispatch(
            x_in,
            topk_idx=topk_idx,
            topk_weights=topk_w.float(),
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
        )
        torch.cuda.synchronize()
        if rank == 0 and it == 0:
            print("DISPATCH OK", flush=True)

        # combine only accepts bf16 (EP_HOST_ASSERT(type == CUDA_R_16BF)), so an FP8
        # dispatch must be cast back first -- which is what the expert compute would
        # produce in a real run.
        if args.fp8:
            from deep_ep.utils import per_token_cast_back
            expert_out = per_token_cast_back(recv_x[0], recv_x[1])
        else:
            expert_out = recv_x

        combined, _, _ = buf.combine(expert_out, handle, topk_weights=recv_w)
        torch.cuda.synchronize()
        if rank == 0 and it == 0:
            print(f"COMBINE OK  out={tuple(combined.shape)}", flush=True)
        if rank == 0 and (it + 1) % 10 == 0:
            print(f"  iter {it+1}/{args.iters} ok", flush=True)

    dist.barrier(group)
    if rank == 0:
        print(f"RESULT: PASS  (fp8={args.fp8} iters={args.iters})", flush=True)

    buf.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
