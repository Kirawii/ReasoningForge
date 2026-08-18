from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase



def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask aligned with
    labels that is 1 for response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str]
            List of prompt strings.
        output_strs: list[str]
            List of output strings.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.

    Returns:
        dict[str, torch.Tensor].
            Let prompt_and_output_lens be a list containing the lengths of the
            concatenated tokenized prompt and output strings. Then the returned
            dictionary should have the following keys:

            input_ids
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): the tokenized
                prompt and output strings, with the final token sliced off.
            labels
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): shifted input
                ids, i.e., the input ids without the first token.
            response_mask
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): a mask aligned
                with labels, with value 1 where the corresponding label token
                is part of the response and 0 otherwise.
    """
    assert len(prompt_strs) == len(output_strs)

    prompt_token_ids = [
        tokenizer.encode(prompt,add_special_tokens = False)
        for prompt in prompt_strs
    ]

    output_token_ids = [
        tokenizer.encode(output,add_special_tokens = False)
        for output in output_strs
    ]
    
    sequence_lengths = [
        len(prompt_ids) + len(output_ids)
        for prompt_ids,output_ids
        in zip(prompt_token_ids,output_token_ids)
    ]
    max_seq_len = max(sequence_lengths)
    full_ids = []
    full_mask = []

    for prompt_ids, output_ids in zip(prompt_token_ids, output_token_ids):
        pad_len = max_seq_len - len(prompt_ids) - len(output_ids)

        ids = (
            prompt_ids
            + output_ids
            + [tokenizer.pad_token_id] * pad_len
        )

        mask = (
            [0] * len(prompt_ids)
            + [1] * len(output_ids)
            + [0] * pad_len
        )
        full_ids.append(ids)
        full_mask.append(mask)
    full_ids = torch.tensor(full_ids, dtype=torch.long)
    full_mask = torch.tensor(full_mask, dtype=torch.bool)
    input_ids = full_ids[:, :-1]
    labels = full_ids[:, 1:]
    response_mask = full_mask[:, 1:]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }
    raise NotImplementedError


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probabilities (given the previous tokens)
    from a causal language model, and optionally the entropy of the model's
    next-token distribution.

    Args:
        model: PreTrainedModel
            HuggingFace model used for scoring (placed on the correct device
            and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor
            shape (batch_size, sequence_length), concatenated prompt + response
            tokens as produced by your tokenization method.
        labels: torch.Tensor
            shape (batch_size, sequence_length), labels as produced by your
            tokenization method.
        return_token_entropy: bool
            If True, also return per-token entropy.

    Returns:
        dict[str, torch.Tensor].
            "log_probs"
                shape (batch_size, sequence_length), conditional
                log-probabilities log p_(theta)(x_t | x_(<t)).
            "token_entropy"
                optional, shape (batch_size, sequence_length), per-token
                entropy for each position (present only if
                return_token_entropy=True).
    """
    # 1 forward
    import torch.nn.functional as F
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    #2 log_softmax
    all_log_probs = F.log_softmax(logits,dim=-1)
    #3 gather labels
    log_probs = torch.gather(
        all_log_probs,
        dim=-1,
        index = labels.unsqueeze(-1)
    ).squeeze(-1)
    result = {
        "log_probs": log_probs
    }
    if return_token_entropy:
        probs=all_log_probs.exp()
        token_entropy = -(probs * all_log_probs).sum(dim=-1)
        result["token_entropy"] = token_entropy

    return result


def run_compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute rewards for a list of rollout responses, along with metadata for
    the reward components.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            raw_rewards
                shape (rollout_batch_size,). Unnormalized rewards for each
                rollout response.
            metadata
                Reward statistics to log. At minimum, include the mean total
                and format rewards over the rollout batch.
    """
    assert len(rollout_responses) == len(repeated_ground_truths)
    rewards = []
    format_rewards = []
    answer_rewards = []
    for response,ground_truth in zip(
        rollout_responses,
        repeated_ground_truths
    ):
        scores = reward_fn(response,ground_truth)
        rewards.append(scores["reward"])
        format_rewards.append(scores["format_reward"])
        answer_rewards.append(scores["answer_reward"])
    raw_rewards = torch.tensor(rewards,dtype=torch.float32)
    metadata = {
        "reward_mean": raw_rewards.mean().item(),
        "format_reward_mean": torch.tensor(format_rewards).mean().item(),
        "answer_reward_mean": torch.tensor(answer_rewards).mean().item()
    }
    return raw_rewards, metadata
    raise NotImplementedError


def run_compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute advantages by applying the requested baseline and normalization
    within each group.

    Args:
        raw_rewards: torch.Tensor
            shape (rollout_batch_size,). Unnormalized rewards for each rollout
            response, where rollout_batch_size = n_prompts_per_rollout_batch *
            group_size.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            For this problem, support mean, which subtracts the per-group mean
            reward. Later, none will mean no baseline subtraction.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            For this problem, support std, which divides by the per-group
            standard deviation. Later, none will mean no normalization and
            mean will mean divide by the per-group mean reward.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            advantages
                shape (rollout_batch_size,). Group-normalized rewards for each
                rollout response.
            metadata
                your choice of other statistics to log (e.g. mean, std, max/min
                of rewards).
    """
    assert raw_rewards.numel() % group_size == 0
    grouped_rewards = raw_rewards.view(-1, group_size)
    grouped_mean = grouped_rewards.mean(dim=1, keepdim=True)
    grouped_std = grouped_rewards.std(dim=1, keepdim=True,correction=0)

    if baseline == "mean":
        advantages = grouped_rewards - grouped_mean
    elif baseline == "none":
        advantages = grouped_rewards.clone()
    else:
        raise ValueError(f"Unsupported baseline: {baseline}")
    if advantage_normalizer == "std":
        advantages = advantages / (grouped_std + advantage_eps)
    elif advantage_normalizer == "none":
        pass
    elif advantage_normalizer == "mean":
        advantages = advantages / (grouped_mean + advantage_eps)
    else:
        raise ValueError(f"Unsupported advantage_normalizer: {advantage_normalizer}")
    advantage = advantages.reshape(-1)
    metadata = {
        "advantage_mean": advantage.mean().item(),
        "advantage_std": advantage.std().item(),
        "reward_mean": raw_rewards.mean().item(),
        "reward_std": raw_rewards.std().item()
    }
    return advantage, metadata
    raise NotImplementedError


def run_compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy-gradient loss at every token, where
    raw_rewards_or_advantages is either the raw reward or an
    already-normalized advantage.

    Args:
        raw_rewards_or_advantages: torch.Tensor
            Shape (batch_size,) or (batch_size, 1), scalar reward/advantage for
            each rollout response.
        policy_log_probs: torch.Tensor
            Shape (batch_size, sequence_length), logprobs for each token.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style
            token-level reweighting and clipping; "gspo": do GSPO-style
            sequence-level reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        response_mask: torch.Tensor | None = None
            Optional shape (batch_size, sequence_length) mask over response
            tokens. Required for GSPO implementations that average the
            sequence-level log-ratio over response tokens only.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            per_token_policy_gradient_loss
                Shape (batch_size, sequence_length), the per-token
                policy-gradient loss (to be aggregated across the batch and
                sequence dimensions in the training loop).
            metadata
                Statistics from the underlying loss call, such as
                clip-fraction components.
    """
    advantages = raw_rewards_or_advantages.view(-1, 1)
    if importance_reweighting_method == "none":
        per_token_policy_gradient_loss = -advantages * policy_log_probs
        metadata = {}
        return per_token_policy_gradient_loss, metadata
    elif importance_reweighting_method == "noclip":
        if old_log_probs is None:
            raise ValueError("old_log_probs must be provided for importance reweighting.")
        log_ratio = policy_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        per_token_policy_gradient_loss = -advantages * ratio
        metadata = {}
        return per_token_policy_gradient_loss, metadata
    elif importance_reweighting_method == "grpo":
        if old_log_probs is None or cliprange is None:
            raise ValueError("old_log_probs and cliprange must be provided for GRPO.")
        log_ratio = policy_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        unclipped = ratio * advantages
        clipped_ratio = torch.clamp(
            ratio,min = 1.0 - cliprange,max = 1.0 + cliprange
        )
        clipped = clipped_ratio * advantages
        per_token_policy_gradient_loss = -torch.minimum(
            unclipped,clipped
        )
        clipped_mask = (
            ((advantages > 0) & (ratio > 1.0 + cliprange))|((advantages < 0) & (ratio < 1.0 - cliprange))
        )
        metadata = {"clip_fraction": clipped_mask.float().mean().item()}
        return per_token_policy_gradient_loss, metadata
    elif importance_reweighting_method == "gspo":
        if old_log_probs is None or cliprange is None or response_mask is None:
            raise ValueError("old_log_probs, cliprange, and response_mask must be provided for GSPO.")
        log_ratio = policy_log_probs - old_log_probs
        masked_log_ratio = log_ratio * response_mask
        response_lengths = response_mask.sum(dim=1, keepdim=True)
        sequence_log_ratio = (
            (masked_log_ratio).sum(dim=1, keepdim=True) / response_lengths 
        )
        sequence_ratio = torch.exp(sequence_log_ratio)
        unclipped = sequence_ratio * advantages
        clipped_ratio = torch.clamp(
            sequence_ratio,min = 1.0 - cliprange,max = 1.0 + cliprange
        )
        clipped = clipped_ratio * advantages
        per_token_policy_gradient_loss = -torch.minimum(
            unclipped,clipped
        )
        per_token_policy_gradient_loss = per_token_policy_gradient_loss.expand_as(policy_log_probs)
        clipped_mask = (
            ((advantages > 0) & (sequence_ratio > 1.0 + cliprange))|((advantages < 0) & (sequence_ratio < 1.0 - cliprange))
        )
        metadata = {"clip_fraction": clipped_mask.float().mean().item()}
        return per_token_policy_gradient_loss, metadata
    raise NotImplementedError


def run_aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    """Aggregate the per-token policy-gradient loss according to the response
    mask and loss-normalization strategy.

    Args:
        per_token_policy_gradient_loss: torch.Tensor
            Shape (batch_size, sequence_length), the per-token policy-gradient
            loss (to be aggregated across the batch and sequence dimensions in
            the training loop).
        mask
            torch.Tensor of shape (batch_size, sequence_length) denoting which
            positions should be included in the loss.
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant.
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        loss: torch.Tensor
            A scalar containing the average loss. Make sure you can later call
            backward on this loss.
    """
    mask = mask.to(per_token_policy_gradient_loss.dtype)
    masked_loss = per_token_policy_gradient_loss * mask
    if loss_normalization == "sequence":
        token_counts = mask.sum(dim=1)
        if torch.any(token_counts == 0):
            raise ValueError("Some sequences have zero tokens in the mask.")
        sequence_losses = (masked_loss.sum(dim=1) / token_counts)
        loss = sequence_losses.mean()
    elif loss_normalization == "constant":
        if normalization_constant is None:
            raise ValueError("normalization_constant must be provided for constant loss normalization.")
        loss = masked_loss.sum() / normalization_constant
    else:
        raise ValueError(f"Unsupported loss_normalization: {loss_normalization}")
    return loss

    raise NotImplementedError


def run_grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Execute forward-and-backward passes, with gradient_accumulation_steps
    microbatches.

    Args:
        model: PreTrainedModel
            HuggingFace model to train.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.
        optimizer: Optimizer
            Optimizer for the model.
        gradient_accumulation_steps: int
            Number of microbatches per optimizer step.
        max_grad_norm: float | None
            If not None, clip the gradient norm to this value before calling
            optimizer.step().
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        repeated_prompts: list[str]
            The prompts for the examples. The length of this list is
            rollout_batch_size, because the prompt for each example is repeated
            group_size times.
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            If mean, subtract the per-group mean reward; if none, do nothing.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            If std, divide by the per-group standard deviation; if none, do
            nothing; if mean, divide by the per-group mean reward.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style token-level
            reweighting and clipping; "gspo": do GSPO-style sequence-level
            reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant (fixed
            for all of training).
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss
                scalar tensor. The batch loss, adjusted for gradient
                accumulation. We return this so we can log it.
            metadata
                Dict with metadata from the underlying loss call, gradient norm
                before clipping, and any other statistics you might want to log.
    """
    batch_size = len(rollout_responses)

    assert len(repeated_prompts) == batch_size
    assert len(repeated_ground_truths) == batch_size
    assert batch_size % gradient_accumulation_steps == 0

    # 整个 rollout batch 一次 tokenize
    tokenized = run_tokenize_prompt_and_output(
        repeated_prompts,
        rollout_responses,
        tokenizer,
    )

    # 整个 rollout batch 一次计算 reward / advantage
    raw_rewards, reward_metadata = run_compute_rollout_rewards(
        reward_fn,
        rollout_responses,
        repeated_ground_truths,
    )

    advantages, advantage_metadata = (
        run_compute_group_normalized_rewards(
            raw_rewards,
            group_size,
            baseline=baseline,
            advantage_eps=advantage_eps,
            advantage_normalizer=advantage_normalizer,
        )
    )

    device = next(model.parameters()).device

    microbatch_size = (
        batch_size // gradient_accumulation_steps
    )

    optimizer.zero_grad(set_to_none=True)

    total_loss = torch.zeros((), device=device)
    loss_metadata_list = []

    for micro_idx in range(
        gradient_accumulation_steps
    ):
        start = micro_idx * microbatch_size
        end = start + microbatch_size

        input_ids = (
            tokenized["input_ids"][start:end]
            .to(device)
        )
        labels = (
            tokenized["labels"][start:end]
            .to(device)
        )
        response_mask = (
            tokenized["response_mask"][start:end]
            .to(device)
        )

        mb_advantages = (
            advantages[start:end]
            .to(device)
        )

        if old_log_probs is not None:
            mb_old_log_probs = (
                old_log_probs[start:end]
                .to(device)
                .detach()
            )
        else:
            mb_old_log_probs = None

        outputs = run_get_response_log_probs(
            model,
            input_ids,
            labels,
            return_token_entropy=False,
        )

        policy_log_probs = outputs["log_probs"]

        per_token_loss, loss_metadata = (
            run_compute_policy_gradient_loss(
                raw_rewards_or_advantages=mb_advantages,
                policy_log_probs=policy_log_probs,
                importance_reweighting_method=(
                    importance_reweighting_method
                ),
                old_log_probs=mb_old_log_probs,
                cliprange=cliprange,
                response_mask=response_mask,
            )
        )

        micro_loss = (
            run_aggregate_loss_across_microbatch(
                per_token_loss,
                response_mask,
                loss_normalization=loss_normalization,
                normalization_constant=(
                    normalization_constant
                ),
            )
        )

        scaled_loss = (
            micro_loss
            / gradient_accumulation_steps
        )

        scaled_loss.backward()

        total_loss += scaled_loss.detach()
        loss_metadata_list.append(loss_metadata)

    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float("inf"),
        )

    optimizer.step()

    optimizer.zero_grad(set_to_none=True)

    metadata = {}
    metadata.update(reward_metadata)
    metadata.update(advantage_metadata)
    metadata["grad_norm"] = grad_norm.item()

    clip_fractions = [
        m["clip_fraction"]
        for m in loss_metadata_list
        if "clip_fraction" in m
    ]

    if clip_fractions:
        metadata["clip_fraction"] = (
            sum(clip_fractions)
            / len(clip_fractions)
        )

    return total_loss, metadata
    raise NotImplementedError


"""
The below adapters are used in the optional 
RLHF / safety part of the Alignment assignment.
"""


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    raise NotImplementedError


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    raise NotImplementedError


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    raise NotImplementedError


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    raise NotImplementedError


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    raise NotImplementedError
