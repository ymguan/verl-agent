#!/usr/bin/env bash
# Phase 2 relaunch + Qwen3-0.6B rescan (after /tmp space fix)
set -u
export WANDB_API_KEY='07d67694ce977d4e8e96369367c00af9a0becb7c'
export TMPDIR="/local_nvme/guanyiming/tmp"
mkdir -p "$TMPDIR"

REPO="/local_nvme/guanyiming/project/verl-agent"
cd "$REPO"
PY="/local_nvme/guanyiming/env/verl-agent-06x-py312/bin/python"
LOG_DIR="$REPO/ocar/logs/relaunch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
STATUS="$LOG_DIR/STATUS.md"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }

echo "# Relaunch — $(date)" > "$STATUS"

# ── Fix 1: rescan Qwen3-0.6B ──
log "rescanning Qwen3-0.6B..."
"$PY" "$REPO/ocar/analysis/scale_scan.py" \
    --model /local_nvme/guanyiming/models/Qwen/Qwen3-0.6B --gpu 0 \
    > "$LOG_DIR/scan_Qwen3-0.6B.log" 2>&1
log "  Qwen3-0.6B rc=$?"

# Regenerate cross-gen report
log "regenerating cross-gen report..."
"$PY" "$REPO/ocar/analysis/cross_generation_scale.py" > "$LOG_DIR/cross_gen.log" 2>&1
log "  cross-gen rc=$?"

# ── Phase 2: entropy bonus training (3 runs) ──
log "=== PHASE 2 RELAUNCH ==="
ENT_SCRIPT="$REPO/examples/ocar_trainer/run_alfworld_entropy_bonus.sh"
BETAS=(-0.005 -0.01 -0.02)
SEED=0

for BETA in "${BETAS[@]}"; do
    SAFE_TAG=$(echo "$BETA" | sed 's/-/n/')
    TAG="b${SAFE_TAG}_s${SEED}"
    log "  run: BETA=$BETA tag=$TAG"
    BETA="$BETA" SEED="$SEED" TAG="$TAG" N_GPUS=4 \
        bash "$ENT_SCRIPT" > "$LOG_DIR/entropy_${TAG}.log" 2>&1
    log "    $TAG rc=$?"
done

log "=== PHASE 2 DONE ==="

# ── Phase 3: report ──
log "generating report..."
"$PY" "$REPO/ocar/analysis/entropy_bonus_report.py" > "$LOG_DIR/report.log" 2>&1
log "  report rc=$?"

log "=== ALL DONE ==="
echo "Finished: $(date)" >> "$STATUS"
