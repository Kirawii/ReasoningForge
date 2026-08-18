#!/usr/bin/env bash
# AutoDL/local deployment adaptation: pin policy to GPU 0 and vLLM to GPU 1.
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: bash scripts/run_autodl_local.sh <command> [args ...]" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CS336_DEPLOYMENT="autodl-local"
export CS336_POLICY_DEVICE="${CS336_POLICY_DEVICE:-cuda:0}"
export CS336_VLLM_GPU="${CS336_VLLM_GPU:-1}"
export CS336_ATTN_IMPLEMENTATION="${CS336_ATTN_IMPLEMENTATION:-sdpa}"

if [[ "${CUDA_VISIBLE_DEVICES}" != "0,1" ]]; then
  echo "Expected CUDA_VISIBLE_DEVICES=0,1; got ${CUDA_VISIBLE_DEVICES}." >&2
  exit 2
fi

exec "$@"
