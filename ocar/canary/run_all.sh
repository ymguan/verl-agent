#!/usr/bin/env bash
# Wait for active Phase-A eval (main_ppo / eval_extra_seeds.sh) to finish,
# then run canary NLL probe across Qwen2.5 / Qwen3 scale family.
set -u
cd "$(dirname "$0")/../.."   # -> repo root
REPO=$(pwd)
LOG_DIR="${REPO}/logs/canary"
mkdir -p "${LOG_DIR}"

PY="/local_nvme/guanyiming/env/verl-agent-06x-py312/bin/python"
RUN="${REPO}/ocar/canary/run_canary.py"

# (model_path, gpu_id). All single-GPU fits in <30GB bf16 up to 14B.
MODELS=(
  "/local_nvme/rs/models/Qwen2.5-0.5B-Instruct|0"
  "/local_nvme/rs/models/Qwen3-0.6B|0"
  "/local_nvme/guanyiming/models/Qwen/Qwen2.5-7B-Instruct|0"
  "Qwen/Qwen3-8B|0"
  "Qwen/Qwen3-14B|0"
)

echo "[canary-runner] $(date) waiting for running evals to finish..."
while pgrep -f "eval_extra_seeds.sh|main_ppo" > /dev/null; do
    sleep 60
done
echo "[canary-runner] $(date) evals done; starting canary."

for entry in "${MODELS[@]}"; do
    MODEL="${entry%%|*}"
    GPU="${entry##*|}"
    slug=$(echo "${MODEL}" | tr '/' '_')
    log="${LOG_DIR}/${slug}.log"
    echo "[canary-runner] $(date) === ${MODEL} (gpu=${GPU}) ==="
    "${PY}" "${RUN}" --model "${MODEL}" --gpu "${GPU}" > "${log}" 2>&1
    rc=$?
    echo "[canary-runner] $(date) ${MODEL} rc=${rc}  log=${log}"
done

echo "[canary-runner] $(date) done."
