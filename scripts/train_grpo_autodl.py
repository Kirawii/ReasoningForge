#!/usr/bin/env python3
"""AutoDL/local dual-GPU reasoning-RL training runner.

AutoDL/local deployment adaptation:
  * logical cuda:0 owns the Hugging Face policy and optimizer;
  * logical cuda:1 owns the vLLM rollout server;
  * policy weights are synchronized to vLLM through the existing NCCL path.

The existing GRPO-family path performs one optimizer update per freshly
generated rollout batch. The independent ``kimi_opmd`` path instead freezes one
outer-loop batch and reference policy for explicitly configured inner updates.
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
from cs336_alignment.kimi_opmd import (  # noqa: E402
    compute_group_mean_advantages,
    compute_kimi_opmd_loss,
)
from cs336_alignment.vllm_utils import VLLMServer  # noqa: E402
from tests.adapters import (  # noqa: E402
    run_compute_rollout_rewards,
    run_get_response_log_probs,
    run_grpo_train_step,
    run_tokenize_prompt_and_output,
)


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
    "kimi_opmd": {
        "baseline": "mean",
        "advantage_normalizer": "none",
        "loss_normalization": "sequence",
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
        default="cs336_alignment/prompts/r1_zero.prompt",
    )
    parser.add_argument("--validation-data", default="data/gsm8k/test.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from", default=None, help="A step-* checkpoint directory.")
    parser.add_argument(
        "--algorithm",
        "--method",
        dest="algorithm",
        choices=sorted(ALGORITHM_SETTINGS),
        default="grpo",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--prompts-per-rollout", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--advantage-eps", type=float, default=1e-6)
    parser.add_argument(
        "--kimi-tau",
        type=float,
        default=0.01,
        help="Smoke-only OPMD coefficient; choose explicitly for formal experiments.",
    )
    parser.add_argument(
        "--kimi-inner-epochs",
        type=int,
        default=2,
        help="Smoke-only fixed-rollout inner updates per outer iteration.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--vllm-startup-timeout", type=int, default=900)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--rollout-request-batch-size", type=int, default=None)
    parser.add_argument("--validation-every", type=int, default=10, help="0 disables validation.")
    parser.add_argument("--validation-examples", type=int, default=1024)
    parser.add_argument("--validation-temperature", type=float, default=0.0)
    parser.add_argument("--validation-request-batch-size", type=int, default=32)
    parser.add_argument("--validation-rollouts-to-log", type=int, default=8)
    parser.add_argument(
        "--eval-at-start",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--save-every", type=int, default=0, help="0 disables periodic checkpoints.")
    parser.add_argument(
        "--save-final",
        action=argparse.BooleanOptionalAction,
        default=False,
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
    if not 0.0 <= args.adam_beta1 < 1.0 or not 0.0 <= args.adam_beta2 < 1.0:
        raise ValueError("Adam betas must be in [0, 1).")
    if not 0.0 < args.vllm_gpu_memory_utilization < 1.0:
        raise ValueError("vllm-gpu-memory-utilization must be between 0 and 1.")
    if args.save_every < 0:
        raise ValueError("save-every cannot be negative.")
    if args.validation_every < 0:
        raise ValueError("validation-every cannot be negative.")
    if args.validation_examples <= 0:
        raise ValueError("validation-examples must be positive.")
    if args.validation_request_batch_size <= 0:
        raise ValueError("validation-request-batch-size must be positive.")
    if args.validation_rollouts_to_log < 0:
        raise ValueError("validation-rollouts-to-log cannot be negative.")
    if args.algorithm == "kimi_opmd" and args.kimi_tau <= 0:
        raise ValueError("kimi-tau must be positive for kimi_opmd training.")
    if args.kimi_inner_epochs <= 0:
        raise ValueError("kimi-inner-epochs must be positive.")


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


def make_optimizer(
    policy: torch.nn.Module,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        policy.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        fused=True,
    )


def compute_reference_log_probs(
    policy: torch.nn.Module,
    tokenized: dict[str, torch.Tensor],
    microbatch_size: int,
    device: str,
) -> torch.Tensor:
    """Cache frozen per-token log-probs under the current outer reference."""
    chunks = []
    with torch.no_grad():
        for start in range(0, tokenized["input_ids"].shape[0], microbatch_size):
            end = start + microbatch_size
            output = run_get_response_log_probs(
                policy,
                tokenized["input_ids"][start:end].to(device),
                tokenized["labels"][start:end].to(device),
                return_token_entropy=False,
            )
            chunks.append(output["log_probs"].detach().cpu())
    return torch.cat(chunks, dim=0)


def run_kimi_outer_update(
    policy: torch.nn.Module,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    repeated_prompts: list[str],
    responses: list[str],
    repeated_ground_truths: list[str],
    outer_step: int,
    metrics_path: Path,
) -> tuple[torch.optim.Optimizer, torch.Tensor, dict[str, Any]]:
    """Run one frozen-reference Kimi outer iteration and its inner updates."""
    batch_size = len(responses)
    microbatch_size = batch_size // args.gradient_accumulation_steps
    device = str(next(policy.parameters()).device)
    tokenized = run_tokenize_prompt_and_output(repeated_prompts, responses, tokenizer)
    response_lengths = tokenized["response_mask"].sum(dim=1).float()
    raw_rewards, reward_metadata = run_compute_rollout_rewards(
        r1_zero_reward_fn,
        responses,
        repeated_ground_truths,
    )
    advantages = compute_group_mean_advantages(raw_rewards, args.group_size).detach()

    grouped_advantage_sums = advantages.reshape(-1, args.group_size).sum(dim=1)
    if not torch.allclose(
        grouped_advantage_sums,
        torch.zeros_like(grouped_advantage_sums),
        atol=1e-6,
        rtol=0.0,
    ):
        raise RuntimeError("Kimi group-mean advantages do not sum to zero.")

    previous_training_mode = policy.training
    policy.eval()
    try:
        reference_log_probs = compute_reference_log_probs(
            policy,
            tokenized,
            microbatch_size,
            device,
        )
        if reference_log_probs.requires_grad:
            raise RuntimeError("Kimi reference log-probs must be detached.")

        if optimizer.state:
            raise RuntimeError("Fresh Kimi optimizer unexpectedly contains state.")
        outer_event = {
            "split": "kimi_outer",
            "step": outer_step,
            "kimi/tau": args.kimi_tau,
            "kimi/inner_epochs": args.kimi_inner_epochs,
            "kimi/reference_refreshed": 1,
            "kimi/optimizer_reset": 1,
            "kimi/optimizer_state_entries_before": 0,
        }
        append_jsonl(metrics_path, outer_event)
        print(json.dumps(outer_event), flush=True)

        final_loss = torch.zeros((), device=device)
        final_inner_metrics: dict[str, Any] = {}
        for inner_step in range(1, args.kimi_inner_epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            total_loss = torch.zeros((), device=device)
            diagnostic_totals: dict[str, float] = {}

            for start in range(0, batch_size, microbatch_size):
                end = start + microbatch_size
                input_ids = tokenized["input_ids"][start:end].to(device)
                labels = tokenized["labels"][start:end].to(device)
                response_mask = tokenized["response_mask"][start:end].to(device)
                policy_output = run_get_response_log_probs(
                    policy,
                    input_ids,
                    labels,
                    return_token_entropy=False,
                )
                microbatch_loss, diagnostics = compute_kimi_opmd_loss(
                    policy_output["log_probs"],
                    reference_log_probs[start:end].to(device),
                    response_mask,
                    advantages[start:end].to(device),
                    tau=args.kimi_tau,
                )
                microbatch_weight = (end - start) / batch_size
                scaled_loss = microbatch_loss * microbatch_weight
                scaled_loss.backward()
                total_loss += scaled_loss.detach()
                for key, value in diagnostics.items():
                    diagnostic_totals[key] = diagnostic_totals.get(key, 0.0) + (
                        float(value.cpu()) * microbatch_weight
                    )

            if not torch.isfinite(total_loss):
                raise FloatingPointError("Kimi loss became non-finite.")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                args.max_grad_norm,
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Kimi gradient norm became non-finite.")
            optimizer.step()

            inner_metrics: dict[str, Any] = {
                "split": "kimi_inner",
                "step": outer_step,
                "kimi/inner_step": inner_step,
                "kimi/tau": args.kimi_tau,
                "loss": float(total_loss.cpu()),
                "grad_norm": float(grad_norm.cpu()),
                "kimi/response_length_mean": float(response_lengths.mean()),
                "kimi/response_length_min": float(response_lengths.min()),
                "kimi/response_length_max": float(response_lengths.max()),
                "kimi/nonfinite": 0,
                "kimi/optimizer_state_entries_after": len(optimizer.state),
                **diagnostic_totals,
            }
            append_jsonl(metrics_path, inner_metrics)
            print(json.dumps(inner_metrics, ensure_ascii=False), flush=True)
            final_loss = total_loss
            final_inner_metrics = inner_metrics

        outer_metadata: dict[str, Any] = {
            **reward_metadata,
            "advantage_mean": float(advantages.mean()),
            "advantage_std": float(advantages.std()),
            "reward_std": float(raw_rewards.std()),
            "kimi/tau": args.kimi_tau,
            "kimi/inner_epochs": args.kimi_inner_epochs,
            "kimi/reference_requires_grad": int(reference_log_probs.requires_grad),
            "kimi/group_advantage_sum_abs_max": float(grouped_advantage_sums.abs().max()),
            "kimi/response_length_mean": float(response_lengths.mean()),
            "kimi/response_length_min": float(response_lengths.min()),
            "kimi/response_length_max": float(response_lengths.max()),
            "kimi/nonfinite": 0,
            **{
                key: value
                for key, value in final_inner_metrics.items()
                if key.startswith("kimi/") or key == "grad_norm"
            },
        }
        return optimizer, final_loss, outer_metadata
    finally:
        policy.train(previous_training_mode)


def run_validation(
    server: VLLMServer,
    examples: list[dict[str, str]],
    prompt_template: str,
    args: argparse.Namespace,
    step: int,
    metrics_path: Path,
    rollouts_path: Path,
    wandb_run: Any,
) -> dict[str, Any]:
    """Evaluate the synchronized vLLM policy on a fixed GSM8K subset."""
    count = min(args.validation_examples, len(examples))
    selected = examples[:count]
    prompts = [
        prompt_template.format(question=example["question"])
        for example in selected
    ]
    started = time.monotonic()
    completions = server.generate_completions(
        prompts=prompts,
        sampling_params={
            "temperature": args.validation_temperature,
            "max_tokens": args.max_new_tokens,
            "n": 1,
            "seed": args.seed,
            "stop": ["</answer>"],
            "include_stop_str_in_output": True,
        },
        batch_size=args.validation_request_batch_size,
    )
    if len(completions) != count:
        raise RuntimeError(
            f"Validation returned {len(completions)} completions; expected {count}."
        )

    scores = [
        r1_zero_reward_fn(completion.text, example["ground_truth"])
        for completion, example in zip(completions, selected)
    ]
    metrics: dict[str, Any] = {
        "split": "validation",
        "step": step,
        "reward_mean": sum(score["reward"] for score in scores) / count,
        "format_reward_mean": sum(score["format_reward"] for score in scores) / count,
        "answer_reward_mean": sum(score["answer_reward"] for score in scores) / count,
        "examples": count,
        "validation_seconds": time.monotonic() - started,
    }
    append_jsonl(metrics_path, metrics)

    for index in range(min(args.validation_rollouts_to_log, count)):
        append_jsonl(
            rollouts_path,
            {
                "split": "validation",
                "step": step,
                "question": selected[index]["question"],
                "ground_truth": selected[index]["ground_truth"],
                "response": completions[index].text,
                **scores[index],
            },
        )

    if wandb_run is not None:
        wandb_run.log(
            {
                f"validation/{key}": value
                for key, value in metrics.items()
                if key not in {"split", "step"}
            },
            step=step,
        )
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    return metrics


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
    validation_examples = None
    if args.validation_every or args.eval_at_start:
        validation_examples = load_gsm8k(Path(args.validation_data))
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

    algorithm = ALGORITHM_SETTINGS[args.algorithm]
    optimizer = make_optimizer(policy, args)
    if args.algorithm != "kimi_opmd":
        restore_optimizer_and_rng(args.resume_from, optimizer)
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
    validation_rollouts_path = output_dir / "validation_rollouts.jsonl"
    last_saved_step = -1
    try:
        print(f"Starting vLLM on cuda:{vllm_gpu} ...", flush=True)
        server.start()
        print("Initializing NCCL weight synchronization ...", flush=True)
        server.init_weight_sync(policy_device)
        server.sync_policy_weights(policy)

        if args.eval_at_start:
            assert validation_examples is not None
            run_validation(
                server,
                validation_examples,
                prompt_template,
                args,
                start_step,
                metrics_path,
                validation_rollouts_path,
                wandb_run,
            )

        for step_index in range(start_step, args.steps):
            step = step_index + 1
            step_started = time.monotonic()
            if args.algorithm == "kimi_opmd":
                # Explicit outer-loop boundary: reset optimizer, then theta_i
                # becomes the rollout/reference policy for this fixed batch.
                optimizer = make_optimizer(policy, args)
                server.sync_policy_weights(policy)
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

            if args.algorithm == "kimi_opmd":
                optimizer, loss, metadata = run_kimi_outer_update(
                    policy=policy,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    args=args,
                    repeated_prompts=repeated_prompts,
                    responses=responses,
                    repeated_ground_truths=repeated_ground_truths,
                    outer_step=step,
                    metrics_path=metrics_path,
                )
            else:
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
                    "split": "train",
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
                    "split": "train",
                    "step": step,
                    "question": selected[0]["question"],
                    "ground_truth": selected[0]["ground_truth"],
                    "response": responses[0],
                },
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        f"train/{key}": value
                        for key, value in metrics.items()
                        if key not in {"split", "step"}
                    },
                    step=step,
                )
            print(json.dumps(metrics, ensure_ascii=False), flush=True)

            if args.validation_every and step % args.validation_every == 0:
                assert validation_examples is not None
                run_validation(
                    server,
                    validation_examples,
                    prompt_template,
                    args,
                    step,
                    metrics_path,
                    validation_rollouts_path,
                    wandb_run,
                )

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
