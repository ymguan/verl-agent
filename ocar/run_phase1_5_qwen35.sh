#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Phase 1.5 add-on: Qwen3.5 post-trained models (2B, 4B, 9B).
#
# These are VLMs but we feed pure text. Loading via AutoModelForCausalLM
# may need AutoModelForVision2Seq wrapper — we try CausalLM first and fall
# back to explicit HF auto-class if needed.
#
# This script WAITS for the main overnight driver's Phase 1 scans to
# complete (signaled by presence of cross_gen_scale/report.md), then
# downloads + scans + regenerates the cross-generation report with the
# 3 extra models included.
#
# Launch AFTER main driver: nohup bash ocar/run_phase1_5_qwen35.sh &
# ──────────────────────────────────────────────────────────────
set -u
export WANDB_API_KEY='07d67694ce977d4e8e96369367c00af9a0becb7c'

REPO="/local_nvme/guanyiming/project/verl-agent"
cd "$REPO"
PY="/local_nvme/guanyiming/env/verl-agent-06x-py312/bin/python"
MODEL_ROOT="/local_nvme/guanyiming/models/Qwen"
SCAN_OUT="$REPO/ocar/analysis_results/scale_scan"
LOG_DIR="$REPO/ocar/logs/phase1_5_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
STATUS="$LOG_DIR/STATUS.md"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }

# Models to add (post-trained, non-Base names)
declare -a MODELS=(
    "Qwen/Qwen3.5-2B|$MODEL_ROOT/Qwen3.5-2B|Qwen3.5-2B"
    "Qwen/Qwen3.5-4B|$MODEL_ROOT/Qwen3.5-4B|Qwen3.5-4B"
    "Qwen/Qwen3.5-9B|$MODEL_ROOT/Qwen3.5-9B|Qwen3.5-9B"
)

# Wait for main Phase 1 to complete
REPORT="$REPO/ocar/analysis_results/cross_gen_scale/report.md"
log "waiting for main Phase 1 report: $REPORT"
while [ ! -f "$REPORT" ]; do
    sleep 120
done
log "main Phase 1 done; starting Qwen3.5 add-on"

# Download
for entry in "${MODELS[@]}"; do
    IFS='|' read -r hf_id local_path label <<< "$entry"
    if [ -d "$local_path" ] && [ -n "$(ls -A "$local_path" 2>/dev/null)" ]; then
        log "  already local: $label"
        continue
    fi
    log "  downloading $hf_id"
    mkdir -p "$(dirname "$local_path")"
    "$PY" -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$hf_id', local_dir='$local_path',
                  local_dir_use_symlinks=False,
                  allow_patterns=['*.json','*.safetensors','*.txt','*.py','tokenizer*','vocab*','merges*'])
" > "$LOG_DIR/download_${label}.log" 2>&1
    log "    rc=$?"
done

# Scan (pick whichever GPU has most free mem at the moment, simple strategy: 0)
log "running scans (serial, gpu=0)..."
for entry in "${MODELS[@]}"; do
    IFS='|' read -r hf_id local_path label <<< "$entry"
    [ ! -d "$local_path" ] && { log "  skip (no local): $label"; continue; }
    slug=$(echo "$local_path" | tr '/' '_')
    out_json="$SCAN_OUT/${slug}.json"
    if [ -f "$out_json" ]; then
        log "  skip (cached): $label"
        continue
    fi
    log "  scan $label"
    "$PY" "$REPO/ocar/analysis/scale_scan.py" --model "$local_path" --gpu 0 \
        > "$LOG_DIR/scan_${label}.log" 2>&1
    rc=$?
    log "    rc=$rc"
    if [ $rc -ne 0 ]; then
        # VLM fallback: log the error for manual inspection
        log "    !! failure — likely AutoModelForCausalLM incompatible with VLM; tail of log:"
        tail -15 "$LOG_DIR/scan_${label}.log" | tee -a "$STATUS"
    fi
done

# Extend cross_generation_scale.py model list with Qwen3.5 entries inline,
# then rerun. We append to MODELS in a sed-free way via a wrapper script.
log "regenerating cross-generation report with Qwen3.5 included..."
"$PY" -c "
import sys
sys.path.insert(0, '$REPO/ocar/analysis')
import cross_generation_scale as m
m.MODELS = m.MODELS + [
    ('Qwen3.5-2B', 'Qwen3.5', 2.0),
    ('Qwen3.5-4B', 'Qwen3.5', 4.0),
    ('Qwen3.5-9B', 'Qwen3.5', 9.0),
]
m.main()
" > "$LOG_DIR/cross_gen_scale_with_q35.log" 2>&1
log "done — report overwritten at $REPORT"

log "=== PHASE 1.5 COMPLETE ==="
