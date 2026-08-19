#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ID="${DATASET_ID:-1}"
CONFIGURATION="${CONFIGURATION:-2d}"
FOLDS="${FOLDS:-0 1 2 3}"
NNUNET_NUM_SPLITS="${NNUNET_NUM_SPLITS:-4}"
TRAINER="${TRAINER:-nnUNetTrainer_60epochs}"
DEVICE="${DEVICE:-cuda}"
DATASET_NAME="${DATASET_NAME:-Dataset$(printf '%03d' "$DATASET_ID")_SkinSegmentation}"
PLANS_NAME="${PLANS_NAME:-nnUNetPlans}"
TEST_INPUT_DIR="${TEST_INPUT_DIR:-${nnUNet_raw:-$PROJECT_DIR/nnunet_data/nnunet_raw}/$DATASET_NAME/imagesTs}"
PREDICTION_OUTPUT_DIR="${PREDICTION_OUTPUT_DIR:-$PROJECT_DIR/predictions/$DATASET_NAME/$CONFIGURATION}"

export nnUNet_raw="${nnUNet_raw:-/workspace/nnunet_data/nnunet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/workspace/nnunet_data/nnunet_preprocessed}"
export nnUNet_results="${nnUNet_results:-/workspace/nnunet_data/nnunet_results}"

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

echo "Predicting test set with folds ${FOLDS}..."
nnUNetv2_predict \
	-i "$TEST_INPUT_DIR" \
	-o "$PREDICTION_OUTPUT_DIR" \
	-d "$DATASET_ID" \
	-c "$CONFIGURATION" \
	-tr "$TRAINER" \
	-f ${FOLDS} \
	-device "$DEVICE"

MODEL_DIR="$nnUNet_results/$DATASET_NAME/${TRAINER}__${PLANS_NAME}__${CONFIGURATION}"
echo "Generating validation scorecard for ${MODEL_DIR}..."
python ./src/skinseg_technical_test/nnunetv2/evaluation/metrics.py \
		--model-dir "$MODEL_DIR" \
		--ground-truth-dir "$nnUNet_raw/$DATASET_NAME/labelsTr"