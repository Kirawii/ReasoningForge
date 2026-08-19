#!/usr/bin/env bash
# AutoDL/local deployment adaptation: serial experiment queue for one 2-GPU host.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/run_autodl_experiment_queue.sh <current-grpo-pid>" >&2
  exit 2
fi

GRPO_PID="$1"
ROOT="/root/ReasoningForge"
MODEL="/root/autodl-tmp/models/OLMo-2-0425-1B"
EXPERIMENTS="/root/autodl-tmp/experiments"
LOGS="/root/autodl-tmp/logs"

cd "$ROOT"
export PATH="$ROOT/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$ROOT/.venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib"
export HF_HOME="/root/autodl-tmp/cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"
mkdir -p "$EXPERIMENTS" "$LOGS"

echo "Waiting for GRPO pid=$GRPO_PID ..."
while kill -0 "$GRPO_PID" 2>/dev/null; do
  sleep 30
done

python - "$EXPERIMENTS/grpo-seed0-pilot/metrics.jsonl" <<'PY'
import json, sys
records = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
steps = [record["step"] for record in records if record.get("split") == "train"]
if not steps or max(steps) != 200:
    raise SystemExit(f"GRPO did not finish at step 200; last step={max(steps) if steps else None}")
PY

run_training() {
  local name="$1"
  shift
  local output="$EXPERIMENTS/$name"
  local log="$LOGS/$name.log"
  if [[ -e "$output" ]]; then
    echo "Refusing to mix results into existing directory: $output" >&2
    return 1
  fi
  echo "Starting $name at $(date --iso-8601=seconds)"
  bash scripts/run_autodl_local.sh \
    .venv/bin/python scripts/train_grpo_autodl.py \
    --model "$MODEL" \
    --output-dir "$output" \
    --save-every 0 \
    --no-save-final \
    "$@" \
    > "$log" 2>&1
  echo "Completed $name at $(date --iso-8601=seconds)"
}

# Full Dr.GRPO baseline: preserve 200/32/8/512/lr/betas/grad-acc settings.
run_training "dr-grpo-seed0-full" \
  --algorithm dr_grpo \
  --steps 200 \
  --prompts-per-rollout 32 \
  --group-size 8 \
  --gradient-accumulation-steps 128 \
  --max-new-tokens 512 \
  --learning-rate 1e-5 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --seed 0

# Diagnostic-only 20-outer-iteration tau sweep. Selection is based on OPMD
# dynamics, not short-run validation accuracy. Validation is disabled here.
for tau in 0.01 0.03 0.1; do
  suffix="${tau/./p}"
  run_training "kimi-opmd-tau-${suffix}" \
    --algorithm kimi_opmd \
    --steps 20 \
    --prompts-per-rollout 32 \
    --group-size 8 \
    --gradient-accumulation-steps 128 \
    --max-new-tokens 512 \
    --learning-rate 1e-5 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --kimi-inner-epochs 2 \
    --kimi-tau "$tau" \
    --validation-every 0 \
    --seed 0
done

.venv/bin/python scripts/summarize_kimi_tau.py \
  "$EXPERIMENTS/kimi-opmd-tau-0p01" \
  "$EXPERIMENTS/kimi-opmd-tau-0p03" \
  "$EXPERIMENTS/kimi-opmd-tau-0p1" \
  --output-prefix "$EXPERIMENTS/kimi-opmd-tau-summary"
echo "Experiment queue complete."
