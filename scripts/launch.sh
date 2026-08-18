#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ID="${DATASET_ID:-1}"
CONFIGURATION="${CONFIGURATION:-2d}"
FOLDS="${FOLDS:-0 1 2}"
NNUNET_NUM_SPLITS="${NNUNET_NUM_SPLITS:-3}"
TRAINER="${TRAINER:-nnUNetTrainer_50epochs}"
DEVICE="${DEVICE:-cuda}"

cd "$PROJECT_DIR"
export NNUNET_NUM_SPLITS

echo "Preparing SkinSeg dataset..."
python ./src/skinseg_technical_test/nnunetv2/prepare/prepare_skinseg.py

echo "Planning and preprocessing nnU-Net dataset ${DATASET_ID}..."
nnUNetv2_plan_and_preprocess -d "$DATASET_ID" --verify_dataset_integrity

for FOLD in ${FOLDS}; do
	echo "Training nnU-Net configuration ${CONFIGURATION}, fold ${FOLD}..."
	nnUNetv2_train "$DATASET_ID" "$CONFIGURATION" "$FOLD" -tr "$TRAINER" -device "$DEVICE"
done
