from __future__ import annotations

import pytest
import torch

from cs336_alignment.kimi_opmd import (
    compute_group_mean_advantages,
    compute_kimi_opmd_loss,
)


def test_equal_policy_and_reference_has_zero_mirror_loss():
    policy = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]], requires_grad=True)
    reference = policy.detach().clone()
    mask = torch.ones_like(policy, dtype=torch.bool)
    advantages = torch.tensor([1.0, -1.0])

    _, diagnostics = compute_kimi_opmd_loss(policy, reference, mask, advantages, tau=0.2)

    torch.testing.assert_close(diagnostics["kimi/mirror_loss"], torch.tensor(0.0))


def test_zero_tau_reduces_to_group_mean_reinforce():
    policy = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]], requires_grad=True)
    reference = torch.zeros_like(policy)
    mask = torch.ones_like(policy, dtype=torch.bool)
    advantages = torch.tensor([2.0, -1.0])

    loss, diagnostics = compute_kimi_opmd_loss(policy, reference, mask, advantages, tau=0.0)

    expected = -(advantages * policy.sum(dim=-1)).mean()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(diagnostics["kimi/mirror_loss"], torch.tensor(0.0))


@pytest.mark.parametrize(
    ("current", "expected_gradient_sign"),
    [(1.0, 1), (-1.0, -1)],
)
def test_mirror_gradient_pushes_policy_toward_reference(current, expected_gradient_sign):
    policy = torch.tensor([[current]], requires_grad=True)
    reference = torch.tensor([[0.0]])
    mask = torch.ones_like(policy, dtype=torch.bool)
    advantages = torch.zeros(1)

    loss, _ = compute_kimi_opmd_loss(policy, reference, mask, advantages, tau=1.0)
    loss.backward()

    assert policy.grad is not None
    assert torch.sign(policy.grad).item() == expected_gradient_sign


def test_response_mask_excludes_prompt_and_padding():
    policy = torch.tensor([[100.0, -2.0, -3.0, 200.0]], requires_grad=True)
    reference = torch.zeros_like(policy)
    mask = torch.tensor([[False, True, True, False]])
    advantages = torch.tensor([1.0])

    loss, diagnostics = compute_kimi_opmd_loss(policy, reference, mask, advantages, tau=0.0)
    loss.backward()

    torch.testing.assert_close(loss, torch.tensor(5.0))
    torch.testing.assert_close(diagnostics["kimi/policy_seq_logp_mean"], torch.tensor(-5.0))
    assert policy.grad is not None
    torch.testing.assert_close(policy.grad, torch.tensor([[0.0, -1.0, -1.0, 0.0]]))


def test_reference_is_detached_while_policy_keeps_gradient():
    policy = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    reference = torch.tensor([[-1.5, -2.5]], requires_grad=True)
    mask = torch.ones_like(policy, dtype=torch.bool)
    advantages = torch.tensor([0.5], requires_grad=True)

    loss, _ = compute_kimi_opmd_loss(policy, reference, mask, advantages, tau=0.1)
    loss.backward()

    assert policy.grad is not None
    assert reference.grad is None
    assert advantages.grad is None


def test_group_mean_advantages_sum_to_zero_per_group():
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0, 2.0, 5.0])
    advantages = compute_group_mean_advantages(rewards, group_size=3)

    grouped_sums = advantages.reshape(-1, 3).sum(dim=1)
    torch.testing.assert_close(grouped_sums, torch.zeros(2), atol=1e-6, rtol=0.0)


def test_sequence_log_probability_uses_sum_not_token_mean():
    policy = torch.tensor([[1.0, 9.0], [1.0, 1.0]], requires_grad=True)
    reference = torch.zeros_like(policy)
    mask = torch.tensor([[True, False], [True, True]])
    advantages = torch.ones(2)

    loss, diagnostics = compute_kimi_opmd_loss(policy, reference, mask, advantages, tau=0.0)

    torch.testing.assert_close(loss, torch.tensor(-1.5))
    torch.testing.assert_close(diagnostics["kimi/policy_seq_logp_mean"], torch.tensor(1.5))
