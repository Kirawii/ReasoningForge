"""Kimi k1.5 online policy mirror descent primitives.

This module is an independent training path. It does not alter the existing
GRPO, Dr.GRPO, MaxRL, RFT, or GSPO loss implementations.
"""

from __future__ import annotations

import torch


def compute_group_mean_advantages(
    raw_rewards: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Subtract each response group's mean reward without std normalization."""
    if raw_rewards.ndim != 1:
        raise ValueError("raw_rewards must have shape [B].")
    if group_size <= 0 or raw_rewards.numel() % group_size != 0:
        raise ValueError("group_size must be positive and divide the batch size.")
    grouped_rewards = raw_rewards.reshape(-1, group_size)
    grouped_advantages = grouped_rewards - grouped_rewards.mean(dim=1, keepdim=True)
    return grouped_advantages.reshape(-1)


def compute_kimi_opmd_loss(
    policy_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    advantages: torch.Tensor,
    tau: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute Kimi OPMD's sequence-level policy and mirror-descent loss.

    The complete response log probability is a masked token sum, not a token
    mean. Reference log probabilities and advantages are detached internally.
    """
    if policy_log_probs.ndim != 2:
        raise ValueError("policy_log_probs must have shape [B, L].")
    if reference_log_probs.shape != policy_log_probs.shape:
        raise ValueError("reference_log_probs must match policy_log_probs.")
    if response_mask.shape != policy_log_probs.shape:
        raise ValueError("response_mask must match policy_log_probs.")
    if advantages.shape != (policy_log_probs.shape[0],):
        raise ValueError("advantages must have shape [B].")
    if tau < 0:
        raise ValueError("tau must be non-negative.")

    # Accumulate long sequence log-probabilities in FP32. The cast preserves
    # policy gradients while avoiding BF16 reduction error in the log-ratio.
    policy_token_logp = policy_log_probs.float()
    mask = response_mask.to(dtype=policy_token_logp.dtype)
    detached_reference = reference_log_probs.detach().float()
    detached_advantages = advantages.detach()

    policy_seq_logp = (policy_token_logp * mask).sum(dim=-1)
    reference_seq_logp = (detached_reference * mask).sum(dim=-1)
    seq_log_ratio = policy_seq_logp - reference_seq_logp

    pg_per_sequence = -(detached_advantages * policy_seq_logp)
    mirror_per_sequence = 0.5 * tau * seq_log_ratio.square()
    loss = (pg_per_sequence + mirror_per_sequence).mean()

    pg_loss = pg_per_sequence.mean()
    pg_magnitude = pg_per_sequence.abs().mean()
    mirror_loss = mirror_per_sequence.mean()
    pg_force_abs_mean = detached_advantages.abs().mean()
    mirror_force_abs_mean = (tau * seq_log_ratio).abs().mean()

    diagnostics = {
        "kimi/pg_loss": pg_loss.detach(),
        "kimi/pg_loss_abs_mean": pg_magnitude.detach(),
        "kimi/mirror_loss": mirror_loss.detach(),
        "kimi/mirror_to_abs_pg_loss_ratio": (
            mirror_loss / (pg_loss.abs() + 1e-12)
        ).detach(),
        "kimi/mirror_to_pg_magnitude_ratio": (
            mirror_loss / (pg_magnitude + 1e-12)
        ).detach(),
        "kimi/pg_force_abs_mean": pg_force_abs_mean.detach(),
        "kimi/mirror_force_abs_mean": mirror_force_abs_mean.detach(),
        "kimi/mirror_to_pg_force_ratio": (
            mirror_force_abs_mean / (pg_force_abs_mean + 1e-12)
        ).detach(),
        "kimi/seq_log_ratio_mean": seq_log_ratio.mean().detach(),
        "kimi/seq_log_ratio_abs_mean": seq_log_ratio.abs().mean().detach(),
        "kimi/seq_log_ratio_sq_mean": seq_log_ratio.square().mean().detach(),
        "kimi/policy_seq_logp_mean": policy_seq_logp.mean().detach(),
        "kimi/reference_seq_logp_mean": reference_seq_logp.mean().detach(),
        "kimi/advantage_mean": detached_advantages.mean().detach(),
    }
    return loss, diagnostics
