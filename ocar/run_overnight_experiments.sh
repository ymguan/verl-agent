#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Overnight experiment driver
#
# Phase 1: Cross-generation base scorer scan
#   - Download missing Qwen models (2.5-Instruct + Qwen3 post-trained)
#   - Run scale_scan.py on all 10 models
#   - Generate cross_generation_scale report
#
# Phase 2: GRPO + entropy bonus ablation (Experiment A)
#   - 3 runs: seed=0, beta in {0.005, 0.01, 0.02}
#   - Runs are sequential (each needs all 4 GPUs for TP=4)
#   - save_freq=9999 so only final ckpt is written
#
# Phase 3: Auto-report (entropy_bonus_report.py)
#
# Logs: ocar/logs/overnight_<timestamp>/
# Status: ocar/logs/overnight_<timestamp>/STATUS.md (updated after each step)
# ──────────────────────────────────────────────────────────────
set -u  # stop on unset vars; intentionally NOT set -e so one failure doesn't abort
export WANDB_API_KEY='07d67694ce977d4e8e96369367c00af9a0becb7c'

REPO="/local_nvme/guanyiming/project/verl-agent"
cd "$REPO"

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$REPO/ocar/logs/overnight_$TS"
mkdir -p "$LOG_DIR"
STATUS="$LOG_DIR/STATUS.md"

PY="/local_nvme/guanyiming/env/verl-agent-06x-py312/bin/python"
MODEL_ROOT="/local_nvme/guanyiming/models/Qwen"
LOCAL_05B="/local_nvme/rs/models/Qwen2.5-0.5B-Instruct"
LOCAL_06B="/local_nvme/rs/models/Qwen3-0.6B"
LOCAL_7B="$MODEL_ROOT/Qwen2.5-7B-Instruct"

log() {
    local msg="[$(date +%H:%M:%S)] $*"
    echo "$msg" | tee -a "$STATUS"
}

{
echo "# Overnight Experiment Status — $TS"
echo ""
echo "Started: $(date)"
echo "Log dir: \`$LOG_DIR\`"
echo ""
} > "$STATUS"

# ══════════════════════════════════════════════════════════════
# PHASE 1: Base scorer scan
# ══════════════════════════════════════════════════════════════
log "=== PHASE 1: Cross-generation base scorer scan ==="

# Model list: (hf_id | local_path | label)
# Local paths preferred when available; else download from HF to MODEL_ROOT.
declare -a MODELS=(
    "Qwen/Qwen2.5-0.5B-Instruct|$LOCAL_05B|Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct|$MODEL_ROOT/Qwen2.5-1.5B-Instruct|Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct|$MODEL_ROOT/Qwen2.5-3B-Instruct|Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct|$LOCAL_7B|Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct|$MODEL_ROOT/Qwen2.5-14B-Instruct|Qwen2.5-14B-Instruct"
    "Qwen/Qwen3-0.6B|$LOCAL_06B|Qwen3-0.6B"
    "Qwen/Qwen3-1.7B|$MODEL_ROOT/Qwen3-1.7B|Qwen3-1.7B"
    "Qwen/Qwen3-4B|$MODEL_ROOT/Qwen3-4B|Qwen3-4B"
    "Qwen/Qwen3-8B|$MODEL_ROOT/Qwen3-8B|Qwen3-8B"
    "Qwen/Qwen3-14B|$MODEL_ROOT/Qwen3-14B|Qwen3-14B"
)

download_model() {
    local hf_id="$1" local_path="$2"
    if [ -d "$local_path" ] && [ -n "$(ls -A "$local_path" 2>/dev/null)" ]; then
        log "  already local: $local_path"
        return 0
    fi
    log "  downloading $hf_id -> $local_path"
    mkdir -p "$(dirname "$local_path")"
    "$PY" -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$hf_id', local_dir='$local_path',
                  local_dir_use_symlinks=False,
                  allow_patterns=['*.json','*.safetensors','*.txt','*.py','tokenizer*','vocab*','merges*'])
" >> "$LOG_DIR/download_$(basename $local_path).log" 2>&1
    return $?
}

log "Phase 1a: downloading missing models (sequential)..."
for entry in "${MODELS[@]}"; do
    IFS='|' read -r hf_id local_path label <<< "$entry"
    download_model "$hf_id" "$local_path"
done
log "Phase 1a: downloads complete."

# Phase 1b: run scans. Assign each model a GPU (0-3) and run up to 4 in parallel.
log "Phase 1b: running scale_scan on each model..."
SCAN_PY="$REPO/ocar/analysis/scale_scan.py"
# Existing outputs to skip
SCAN_OUT="$REPO/ocar/analysis_results/scale_scan"
mkdir -p "$SCAN_OUT"

run_scan() {
    local local_path="$1" label="$2" gpu="$3"
    local slug=$(echo "$local_path" | tr '/' '_')
    local out_json="$SCAN_OUT/${slug}.json"
    if [ -f "$out_json" ]; then
        log "  skip (cached): $label"
        return 0
    fi
    log "  scan $label (gpu=$gpu)"
    "$PY" "$SCAN_PY" --model "$local_path" --gpu "$gpu" \
        > "$LOG_DIR/scan_${label}.log" 2>&1
    local rc=$?
    log "    $label rc=$rc"
    return $rc
}

# Run in groups of 4 (one per GPU). Simple round-robin then wait.
gpu_idx=0
pids=()
for entry in "${MODELS[@]}"; do
    IFS='|' read -r hf_id local_path label <<< "$entry"
    [ ! -d "$local_path" ] && { log "  skip (no local): $label"; continue; }
    run_scan "$local_path" "$label" "$gpu_idx" &
    pids+=($!)
    gpu_idx=$(( (gpu_idx + 1) % 4 ))
    # When all 4 GPUs are in use, wait for them to finish
    if [ ${#pids[@]} -ge 4 ]; then
        for pid in "${pids[@]}"; do wait "$pid" || true; done
        pids=()
    fi
done
# Wait remaining
for pid in "${pids[@]}"; do wait "$pid" || true; done
log "Phase 1b: all scans complete."

# Phase 1c: cross-generation analysis
log "Phase 1c: cross-generation analysis..."
"$PY" "$REPO/ocar/analysis/cross_generation_scale.py" > "$LOG_DIR/cross_gen_scale.log" 2>&1
log "  report: ocar/analysis_results/cross_gen_scale/report.md"

log "=== PHASE 1 DONE ==="

# ══════════════════════════════════════════════════════════════
# PHASE 2: GRPO + entropy bonus ablation
# ══════════════════════════════════════════════════════════════
log "=== PHASE 2: GRPO + entropy bonus ablation ==="

ENT_SCRIPT="$REPO/examples/ocar_trainer/run_alfworld_entropy_bonus.sh"
BETAS=(-0.005 -0.01 -0.02)
SEED=0

for BETA in "${BETAS[@]}"; do
    # TAG with 'n' prefix for negative to avoid '=' in filename
    SAFE_TAG=$(echo "$BETA" | sed 's/-/n/')
    TAG="b${SAFE_TAG}_s${SEED}"
    log "  run: BETA=$BETA SEED=$SEED tag=$TAG"
    BETA="$BETA" SEED="$SEED" TAG="$TAG" N_GPUS=4 \
        bash "$ENT_SCRIPT" > "$LOG_DIR/entropy_${TAG}.log" 2>&1
    log "    $TAG rc=$?"
done

log "=== PHASE 2 DONE ==="

# ══════════════════════════════════════════════════════════════
# PHASE 3: Auto-report
# ══════════════════════════════════════════════════════════════
log "=== PHASE 3: Auto-report ==="
"$PY" "$REPO/ocar/analysis/entropy_bonus_report.py" > "$LOG_DIR/entropy_bonus_report.log" 2>&1
log "  report: ocar/analysis_results/entropy_bonus/report.md"

log "=== ALL DONE ==="
echo "" >> "$STATUS"
echo "Finished: $(date)" >> "$STATUS"
