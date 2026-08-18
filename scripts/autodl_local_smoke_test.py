#!/usr/bin/env python3
"""AutoDL/local deployment adaptation smoke test for isolated dual GPUs.

GPU 0 loads a small Hugging Face policy model. GPU 1 starts the official
VLLMServer, then the existing NCCL weight-transfer path is exercised once.
No assignment algorithm is implemented here.
"""

from __future__ import annotations

import argparse
import os

# These defaults must be set before importing torch/vLLM.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.setdefault("CS336_POLICY_DEVICE", "cuda:0")
os.environ.setdefault("CS336_VLLM_GPU", "1")
os.environ.setdefault("CS336_ATTN_IMPLEMENTATION", "sdpa")

import torch  # noqa: E402 - CUDA visibility must be set before importing torch.

from cs336_alignment.checkpoint import get_model_and_tokenizer  # noqa: E402
from cs336_alignment.local_utils import (  # noqa: E402
    collect_local_environment,
    print_local_environment,
    validate_local_environment,
    visible_memory_used_mib,
)
from cs336_alignment.vllm_utils import VLLMServer  # noqa: E402


def _print_memory(label: str, memory_mib: tuple[float, ...]) -> None:
    formatted = ", ".join(
        f"GPU {index}={used:.0f} MiB" for index, used in enumerate(memory_mib)
    )
    print(f"{label}: {formatted}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.20)
    parser.add_argument(
        "--isolation-tolerance-mib",
        type=float,
        default=256.0,
        help="Maximum incidental memory growth allowed on the other GPU.",
    )
    args = parser.parse_args()

    environment = collect_local_environment()
    print_local_environment(environment)
    issues = validate_local_environment(environment)
    if issues:
        raise RuntimeError("Environment check failed: " + "; ".join(issues))

    policy_device = os.environ["CS336_POLICY_DEVICE"]
    vllm_gpu = int(os.environ["CS336_VLLM_GPU"])
    if policy_device != "cuda:0" or vllm_gpu != 1:
        raise RuntimeError("Smoke test requires policy=cuda:0 and vLLM GPU=1.")

    torch.cuda.empty_cache()
    baseline = visible_memory_used_mib()
    _print_memory("Baseline memory", baseline)

    print(f"Loading policy model {args.model!r} on {policy_device} ...")
    policy, _ = get_model_and_tokenizer(args.model, policy_device)
    policy.eval()
    with torch.inference_mode():
        input_ids = torch.tensor([[0, 2]], device=policy_device)
        output = policy(input_ids=input_ids)
    torch.cuda.synchronize(0)
    if output.logits.device.index != 0:
        raise RuntimeError(
            f"Policy output landed on {output.logits.device}, not cuda:0."
        )

    after_policy = visible_memory_used_mib()
    _print_memory("After policy load", after_policy)
    if after_policy[0] <= baseline[0]:
        raise RuntimeError("Policy model did not allocate memory on GPU 0.")
    if after_policy[1] - baseline[1] > args.isolation_tolerance_mib:
        raise RuntimeError("Policy load unexpectedly increased GPU 1 memory.")

    server = VLLMServer(
        model_id=args.model,
        port=args.port,
        gpu=vllm_gpu,
        gpu_memory_utilization=args.gpu_memory_utilization,
        startup_timeout=args.startup_timeout,
    )
    try:
        print(f"Starting vLLM on physical/logical GPU {vllm_gpu} ...")
        server.start()
        after_server = visible_memory_used_mib()
        _print_memory("After vLLM start", after_server)
        if after_server[1] <= after_policy[1]:
            raise RuntimeError("vLLM did not allocate memory on GPU 1.")
        if after_server[0] - after_policy[0] > args.isolation_tolerance_mib:
            raise RuntimeError("vLLM start unexpectedly increased GPU 0 memory.")

        print("Initializing the official vLLM NCCL weight-transfer group ...")
        server.init_weight_sync(policy_device)
        server.sync_policy_weights(policy)
        completion = server.generate_completions(
            prompts=["Hello"],
            sampling_params={
                "temperature": 0.0,
                "max_tokens": 1,
                "n": 1,
                "seed": 0,
            },
        )
        if len(completion) != 1:
            raise RuntimeError(
                "vLLM completion endpoint returned an unexpected result count."
            )
        print("NCCL policy weight synchronization: PASS")
        print("vLLM completion request: PASS")
        print("Dual-GPU isolation: PASS")
    finally:
        server.stop()

    print("AutoDL/local dual-GPU smoke test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
