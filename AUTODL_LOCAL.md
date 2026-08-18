# AutoDL/local dual-GPU deployment adaptation

This guide adapts only the runtime. It does not implement GRPO, GSPO,
Dr. GRPO, MaxRL, RFT, or any function in `tests/adapters.py`.

## Device layout

- Logical `cuda:0`: Hugging Face policy model and optimizer.
- Logical `cuda:1`: vLLM rollout server.
- The existing `cs336_alignment/vllm_utils.py` NCCL weight-transfer engine
  synchronizes policy weights between the two processes on the same host.
- `cs336_alignment/modal_utils.py` is unchanged and remains the official Modal
  entry point.

The local launcher requires `CUDA_VISIBLE_DEVICES=0,1`. Do not start the policy
or vLLM separately with a conflicting device mask.

## Compatibility profile

The pinned local profile is Python 3.12, PyTorch 2.10.0+cu129, CUDA runtime
12.9, and vLLM 0.19.1. RTX 5090 is Blackwell compute capability 12.0 and
supports BF16. CUDA 12.9 GA requires NVIDIA Linux driver 575.51.03 or newer;
CUDA 12.9 Update 1 requires 575.57.08 or newer.

The repository's `gpu` extra keeps the official third-party flash-attn wheel
for Modal/B200. The local installation intentionally excludes that one package
because flash-attn 2.8.3 does not publish official RTX 5090/SM 12.0 support.
The local launcher explicitly sets `CS336_ATTN_IMPLEMENTATION=sdpa`; there is
no GPU-model auto-detection in the checkpoint loader. vLLM keeps its own
supported CUDA attention kernels.

## Install from an empty AutoDL instance

Choose an Ubuntu 22.04 image with two RTX 5090 GPUs and a driver new enough for
CUDA 12.9, then run:

```bash
cd /root/autodl-tmp
nvidia-smi
git clone https://github.com/stanford-cs336/assignment5-alignment.git
cd assignment5-alignment

curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv python install 3.12

export UV_LINK_MODE=copy
uv sync --extra gpu --no-install-package flash-attn
```

If this adaptation is on another branch or remote, check out that branch after
cloning and before `uv sync`.

For persistent model caches on AutoDL's data disk:

```bash
mkdir -p /root/autodl-tmp/huggingface
export HF_HOME=/root/autodl-tmp/huggingface
```

## Check and smoke test

```bash
bash scripts/run_autodl_local.sh \
  uv run python scripts/autodl_local_env_check.py --strict --require-rtx-5090

bash scripts/run_autodl_local.sh \
  uv run python scripts/autodl_local_smoke_test.py
```

The smoke test downloads `facebook/opt-125m` by default. To use a pre-downloaded
model directory, pass `--model /absolute/path/to/model`.

## Launch future assignment scripts

After you implement your own assignment training script, launch it through the
device wrapper and read the device values from the environment:

```bash
bash scripts/run_autodl_local.sh uv run python YOUR_TRAINING_SCRIPT.py [arguments]
```

Use `CS336_POLICY_DEVICE` for the policy device and pass
`gpu=int(os.environ["CS336_VLLM_GPU"])` to `VLLMServer`. Its defaults remain
`cuda:0` and `1`, respectively.
