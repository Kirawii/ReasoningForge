#!/usr/bin/env python3
"""AutoDL/local dual-GPU reasoning-RL training runner.

AutoDL/local deployment adaptation:
  * logical cuda:0 owns the Hugging Face policy and optimizer;
  * logical cuda:1 owns the vLLM rollout server;
  * policy weights are synchronized to vLLM through the existing NCCL path.

The runner intentionally performs one optimizer update per freshly generated
rollout batch. This keeps the default path on-policy and avoids silently using
stale rollout log-probabilities.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

# AutoDL/local deployment adaptation: direct ``python scripts/...`` execution
# puts only scripts/ on sys.path, while the completed adapters live at repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Device placement must be fixed before importing torch/vLLM-related modules.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.setdefault("CS336_POLICY_DEVICE", "cuda:0")
os.environ.setdefault("CS336_VLLM_GPU", "1")
os.environ.setdefault("CS336_ATTN_IMPLEMENTATION", "sdpa")

import torch  # noqa: E402

from cs336_alignment.checkpoint import get_model_and_tokenizer  # noqa: E402
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn  # noqa: E402
from cs336_alignment.vllm_utils import VLLMServer  # noqa: E402
from tests.adapters import run_grpo_train_step  # noqa: E402


ALGORITHM_SETTINGS: dict[str, dict[str, Any]] = {
    "grpo": {
        "baseline": "mean",
        "advantage_normalizer": "std",
        "loss_normalization": "sequence",
    },
    "grpo_constant": {
        "baseline": "mean",
        "advantage_normalizer": "std",
        "loss_normalization": "constant",
    },
    "dr_grpo": {
        "baseline": "mean",
        "advantage_normalizer": "none",
        "loss_normalization": "constant",
    },
    "maxrl": {
        "baseline": "mean",
        "advantage_normalizer": "mean",
        "loss_normalization": "constant",
    },
    "rft": {
        "baseline": "none",
        "advantage_normalizer": "none",
        "loss_normalization": "constant",
    },
}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a reasoning policy with local GPU0 + vLLM GPU1.",
    )
    parser.add_argument("--model", required=True, help="HF model id or local model directory.")
    parser.add_argument("--train-data", default="data/gsm8k/train.jsonl")
    parser.add_argument(
        "--prompt-template",
        default="cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from", default=None, help="A step-* checkpoint directory.")
    parser.add_argument("--algorithm", choices=sorted(ALGORITHM_SETTINGS), default="grpo")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--prompts-per-rollout", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--advantage-eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--vllm-startup-timeout", type=int, default=900)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--rollout-request-batch-size", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=50, help="0 disables periodic checkpoints.")
    parser.add_argument(
        "--save-final",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
        raise ValueError("CUDA_VISIBLE_DEVICES must be exactly '0,1'.")
    if os.environ.get("CS336_POLICY_DEVICE") != "cuda:0":
        raise ValueError("CS336_POLICY_DEVICE must be 'cuda:0'.")
    if int(os.environ.get("CS336_VLLM_GPU", "-1")) != 1:
        raise ValueError("CS336_VLLM_GPU must be '1'.")
    if args.steps <= 0 or args.prompts_per_rollout <= 0:
        raise ValueError("steps and prompts-per-rollout must be positive.")
    if args.group_size <= 1:
        raise ValueError("group-size must be at least 2 for grouped reward normalization.")
    rollout_batch_size = args.prompts_per_rollout * args.group_size
    if rollout_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError(
            "prompts-per-rollout * group-size must be divisible by "
            "gradient-accumulation-steps."
        )
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive.")
    if not 0.0 < args.vllm_gpu_memory_utilization < 1.0:
        raise ValueError("vllm-gpu-memory-utilization must be between 0 and 1.")
    if args.save_every < 0:
        raise ValueError("save-every cannot be negative.")


def load_gsm8k(path: Path) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            question = record.get("question")
            answer = record.get("answer")
            if not isinstance(question, str) or not isinstance(answer, str):
                raise ValueError(f"Invalid GSM8K record at {path}:{line_number}.")
            ground_truth = answer.rsplit("####", maxsplit=1)[-1].strip()
            examples.append({"question": question, "ground_truth": ground_truth})
    if not examples:
        raise ValueError(f"No training examples found in {path}.")
    return examples


def select_examples(
    examples: list[dict[str, str]],
    count: int,
    seed: int,
    step: int,
) -> list[dict[str, str]]:
    if count > len(examples):
        raise ValueError("prompts-per-rollout exceeds the number of training examples.")
    rng = random.Random(seed + step)
    return [examples[index] for index in rng.sample(range(len(examples)), count)]


def load_template(path: Path) -> str:
    template = path.read_text(encoding="utf-8")
    if "{question}" not in template:
        raise ValueError(f"Prompt template {path} does not contain '{{question}}'.")
    return template


def checkpoint_source(args: argparse.Namespace) -> tuple[str, int]:
    if args.resume_from is None:
        return args.model, 0
    checkpoint_dir = Path(args.resume_from)
    state_path = checkpoint_dir / "trainer_state.pt"
    policy_dir = checkpoint_dir / "policy"
    if not state_path.is_file() or not policy_dir.is_dir():
        raise FileNotFoundError(
            f"Resume checkpoint must contain policy/ and trainer_state.pt: {checkpoint_dir}"
        )
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    return str(policy_dir), int(state["step"])


def save_checkpoint(
    output_dir: Path,
    step: int,
    model: torch.nn.Module,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> Path:
    checkpoint_dir = output_dir / f"step-{step:06d}"
    if checkpoint_dir.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint {checkpoint_dir}.")
    policy_dir = checkpoint_dir / "policy"
    policy_dir.mkdir(parents=True)
    model.save_pretrained(policy_dir, safe_serialization=True)
    tokenizer.save_pretrained(policy_dir)
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
        },
        checkpoint_dir / "trainer_state.pt",
    )
    return checkpoint_dir


def restore_optimizer_and_rng(
    resume_from: str | None,
    optimizer: torch.optim.Optimizer,
) -> None:
    if resume_from is None:
        return
    state = torch.load(
        Path(resume_from) / "trainer_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng_state"])
    torch.cuda.set_rng_state_all(state["cuda_rng_state"])


def scalar_metrics(metadata: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in metadata.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                result[key] = float(value.detach().cpu())
        elif isinstance(value, (int, float)):
            result[key] = float(value)
    return result


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def init_wandb(args: argparse.Namespace):
    if args.wandb_project is None:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
        resume="allow",
    )


def main() -> int:
    args = make_parser().parse_args()
    validate_args(args)

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("This runner requires exactly two visible CUDA GPUs.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    examples = load_gsm8k(Path(args.train_data))
    prompt_template = load_template(Path(args.prompt_template))
    model_source, start_step = checkpoint_source(args)
    if start_step >= args.steps:
        raise ValueError(f"Checkpoint step {start_step} is already >= requested steps {args.steps}.")

    policy_device = os.environ["CS336_POLICY_DEVICE"]
    vllm_gpu = int(os.environ["CS336_VLLM_GPU"])
    print(f"Loading policy from {model_source!r} on {policy_device} ...", flush=True)
    policy, tokenizer = get_model_and_tokenizer(model_source, policy_device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy.config.use_cache = False
    if args.gradient_checkpointing:
        policy.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    policy.train()

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=True,
    )
    restore_optimizer_and_rng(args.resume_from, optimizer)

    algorithm = ALGORITHM_SETTINGS[args.algorithm]
    rollout_batch_size = args.prompts_per_rollout * args.group_size
    normalization_constant = None
    if algorithm["loss_normalization"] == "constant":
        normalization_constant = rollout_batch_size * args.max_new_tokens

    wandb_run = init_wandb(args)
    server = VLLMServer(
        model_id=model_source,
        port=args.vllm_port,
        gpu=vllm_gpu,
        seed=args.seed,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        startup_timeout=args.vllm_startup_timeout,
    )

    metrics_path = output_dir / "metrics.jsonl"
    rollouts_path = output_dir / "rollouts.jsonl"
    last_saved_step = -1
    try:
        print(f"Starting vLLM on cuda:{vllm_gpu} ...", flush=True)
        server.start()
        print("Initializing NCCL weight synchronization ...", flush=True)
        server.init_weight_sync(policy_device)
        server.sync_policy_weights(policy)

        for step_index in range(start_step, args.steps):
            step = step_index + 1
            step_started = time.monotonic()
            selected = select_examples(
                examples,
                args.prompts_per_rollout,
                args.seed,
                step_index,
            )
            unique_prompts = [
                prompt_template.format(question=example["question"])
                for example in selected
            ]
            completions = server.generate_completions(
                prompts=unique_prompts,
                sampling_params={
                    "temperature": args.temperature,
                    "max_tokens": args.max_new_tokens,
                    "n": args.group_size,
                    "seed": args.seed + step_index,
                    "stop": ["</answer>"],
                    "include_stop_str_in_output": True,
                },
                batch_size=args.rollout_request_batch_size,
            )
            if len(completions) != rollout_batch_size:
                raise RuntimeError(
                    f"vLLM returned {len(completions)} completions; "
                    f"expected {rollout_batch_size}."
                )

            repeated_prompts = [
                prompt
                for prompt in unique_prompts
                for _ in range(args.group_size)
            ]
            repeated_ground_truths = [
                example["ground_truth"]
                for example in selected
                for _ in range(args.group_size)
            ]
            responses = [completion.text for completion in completions]

            loss, metadata = run_grpo_train_step(
                model=policy,
                tokenizer=tokenizer,
                optimizer=optimizer,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_grad_norm=args.max_grad_norm,
                reward_fn=r1_zero_reward_fn,
                repeated_prompts=repeated_prompts,
                rollout_responses=responses,
                repeated_ground_truths=repeated_ground_truths,
                group_size=args.group_size,
                baseline=algorithm["baseline"],
                advantage_eps=args.advantage_eps,
                advantage_normalizer=algorithm["advantage_normalizer"],
                importance_reweighting_method="none",
                loss_normalization=algorithm["loss_normalization"],
                normalization_constant=normalization_constant,
            )
            server.sync_policy_weights(policy)

            metrics = scalar_metrics(metadata)
            metrics.update(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "step_seconds": time.monotonic() - step_started,
                    "rollout_batch_size": rollout_batch_size,
                }
            )
            append_jsonl(metrics_path, metrics)
            append_jsonl(
                rollouts_path,
                {
                    "step": step,
                    "question": selected[0]["question"],
                    "ground_truth": selected[0]["ground_truth"],
                    "response": responses[0],
                },
            )
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)
            print(json.dumps(metrics, ensure_ascii=False), flush=True)

            if args.save_every and step % args.save_every == 0:
                saved = save_checkpoint(
                    output_dir,
                    step,
                    policy,
                    tokenizer,
                    optimizer,
                    args,
                )
                last_saved_step = step
                print(f"Saved checkpoint: {saved}", flush=True)

        if args.save_final and last_saved_step != args.steps:
            saved = save_checkpoint(
                output_dir,
                args.steps,
                policy,
                tokenizer,
                optimizer,
                args,
            )
            print(f"Saved final checkpoint: {saved}", flush=True)
    finally:
        server.stop()
        if wandb_run is not None:
            wandb_run.finish()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; vLLM shutdown requested.", file=sys.stderr)
        raise SystemExit(130)
