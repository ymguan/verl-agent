#!/usr/bin/env bash
# E3.2 scale scan runner — waits for eval + canary, then scans Qwen family.
# Expected runtime per model: ~2-5 min (12 trajs × ~8 steps × 4 NLL computations)
set -u
cd "$(dirname "$0")/../.."   # -> repo root
REPO=$(pwd)
LOG_DIR="${REPO}/logs/scale_scan"
mkdir -p "${LOG_DIR}"

PY="/local_nvme/guanyiming/env/verl-agent-06x-py312/bin/python"
RUN="${REPO}/ocar/analysis/scale_scan.py"

# Models: include Qwen2.5 family (safe) + Qwen3 (contamination check)
MODELS=(
  "/local_nvme/rs/models/Qwen2.5-0.5B-Instruct|0"
  "/local_nvme/rs/models/Qwen3-0.6B|0"
  "/local_nvme/guanyiming/models/Qwen/Qwen2.5-7B-Instruct|0"
  "Qwen/Qwen3-8B|0"
  "Qwen/Qwen3-14B|0"
)

echo "[scale_scan-runner] $(date) waiting for evals + canary to finish..."
while pgrep -f "eval_extra_seeds.sh|main_ppo|run_canary.py|run_all.sh" > /dev/null; do
    sleep 60
done
echo "[scale_scan-runner] $(date) pipeline clear; starting scale_scan."

for entry in "${MODELS[@]}"; do
    MODEL="${entry%%|*}"
    GPU="${entry##*|}"
    slug=$(echo "${MODEL}" | tr '/' '_')
    log="${LOG_DIR}/${slug}.log"
    echo "[scale_scan-runner] $(date) === ${MODEL} (gpu=${GPU}) ==="
    "${PY}" "${RUN}" --model "${MODEL}" --gpu "${GPU}" > "${log}" 2>&1
    rc=$?
    echo "[scale_scan-runner] $(date) ${MODEL} rc=${rc} log=${log}"
done

echo "[scale_scan-runner] $(date) done. running summary:"
"${PY}" -c "
import json, glob
for p in sorted(glob.glob('${REPO}/ocar/analysis_results/scale_scan/*.json')):
    d = json.load(open(p))
    s = d['summary']
    o = s.get('obs_nll_last') or {}
    wa = s.get('wm_A') or {}
    wb = s.get('wm_B') or {}
    wg = s.get('wm_gap') or {}
    print(f\"{d['model']:60s} obs={o.get('mean',0):.3f} wm_A={wa.get('mean',0):.3f} wm_B={wb.get('mean',0):.3f} gap={wg.get('mean',0):+.3f}\")
"
