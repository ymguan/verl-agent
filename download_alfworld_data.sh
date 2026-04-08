#!/bin/bash
# ──────────────────────────────────────────────────────────────
# Download ALFWorld dataset (official, v0.4.2)
#
# Source: https://github.com/alfworld/alfworld (PyPI: alfworld==0.4.2)
# Data:   json_2.1.1 (6374 train + 251 valid_seen + 477 valid_unseen)
#         logic/ (alfred.pddl + alfred.twl2)
#         detectors/ (MaskRCNN, for multi-modal mode only)
#
# Usage:
#   bash download_alfworld_data.sh
#
# After download, set ALFWORLD_DATA to point to the data directory:
#   export ALFWORLD_DATA=/local_nvme/guanyiming/project/verl-agent/alfworld_data
# ──────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${ALFWORLD_DATA:-$SCRIPT_DIR/alfworld_data}"

echo "=== ALFWorld Data Download ==="
echo "Target directory: $DATA_DIR"

if [ -d "$DATA_DIR/json_2.1.1" ]; then
    echo "Data already exists at $DATA_DIR"
    echo "  train:        $(find $DATA_DIR/json_2.1.1/train -name '*.json' -type f | wc -l) games"
    echo "  valid_seen:   $(find $DATA_DIR/json_2.1.1/valid_seen -name '*.json' -type f | wc -l) games"
    echo "  valid_unseen: $(find $DATA_DIR/json_2.1.1/valid_unseen -name '*.json' -type f | wc -l) games"
    echo ""
    echo "To re-download, delete $DATA_DIR and run again."
    exit 0
fi

# Method 1: Use alfworld-download (if alfworld is installed)
if command -v alfworld-download &>/dev/null; then
    echo "Using alfworld-download..."
    export ALFWORLD_DATA="$DATA_DIR"
    mkdir -p "$DATA_DIR"
    alfworld-download
elif python3 -c "import alfworld" 2>/dev/null; then
    echo "Using python3 alfworld-download..."
    export ALFWORLD_DATA="$DATA_DIR"
    mkdir -p "$DATA_DIR"
    python3 -m alfworld.scripts.alfworld_download
else
    # Method 2: Direct download from GitHub releases
    echo "alfworld not installed. Downloading directly from GitHub..."
    mkdir -p "$DATA_DIR"
    cd "$DATA_DIR"

    BASE_URL="https://github.com/alfworld/alfworld/releases/download/0.4.2"

    echo "Downloading json_2.1.1..."
    wget -q "${BASE_URL}/json_2.1.1_json.zip" -O json_2.1.1_json.zip
    unzip -q json_2.1.1_json.zip
    rm json_2.1.1_json.zip

    echo "Downloading PDDL files..."
    wget -q "${BASE_URL}/json_2.1.1_pddl.zip" -O json_2.1.1_pddl.zip
    unzip -q json_2.1.1_pddl.zip
    rm json_2.1.1_pddl.zip

    echo "Downloading tw-pddl game files..."
    wget -q "${BASE_URL}/json_2.1.2_tw-pddl.zip" -O json_2.1.2_tw-pddl.zip
    unzip -q json_2.1.2_tw-pddl.zip
    rm json_2.1.2_tw-pddl.zip

    # Create logic directory with symlinks
    mkdir -p logic
    cp json_2.1.1/train/pick_and_place_simple-Statue-Dresser-309/game.tw-pddl logic/ 2>/dev/null || true

    echo "Downloading MaskRCNN detector (optional, for multi-modal)..."
    mkdir -p detectors
    wget -q "${BASE_URL}/mrcnn_alfred_objects_sep13_004.pth" -O detectors/mrcnn.pth || echo "  (skipped, not needed for text-only)"
fi

echo ""
echo "=== Download Complete ==="
echo "Data directory: $DATA_DIR"
echo "  train:        $(find $DATA_DIR/json_2.1.1/train -name '*.json' -type f 2>/dev/null | wc -l) games"
echo "  valid_seen:   $(find $DATA_DIR/json_2.1.1/valid_seen -name '*.json' -type f 2>/dev/null | wc -l) games"
echo "  valid_unseen: $(find $DATA_DIR/json_2.1.1/valid_unseen -name '*.json' -type f 2>/dev/null | wc -l) games"
echo ""
echo "To use this data, set:"
echo "  export ALFWORLD_DATA=$DATA_DIR"
