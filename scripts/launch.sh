#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_ID="${DATASET_ID:-1}"
CONFIGURATION="${CONFIGURATION:-2d}"
FOLD="${FOLD:-0}"

cd "$PROJECT_DIR"

echo "Preparing SkinSeg dataset..."
#python ./src/skinseg_technical_test/nnunetv2/prepare/prepare_skinseg.py

echo "Planning and preprocessing nnU-Net dataset ${DATASET_ID}..."
nnUNetv2_plan_and_preprocess -d "$DATASET_ID" --verify_dataset_integrity

echo "Training nnU-Net configuration ${CONFIGURATION}, fold ${FOLD}..."
nnUNetv2_train "$DATASET_ID" "$CONFIGURATION" "$FOLD"
