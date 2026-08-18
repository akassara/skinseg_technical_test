#!/bin/bash
# Conda activation script for nnUNet paths
export nnUNet_raw="${PROJECT_DIR}/nnUNet/nnunetv2/raw"
export nnUNet_preprocessed="${PROJECT_DIR}/nnUNet/nnunetv2/preprocessed"
export nnUNet_results="${PROJECT_DIR}/nnUNet/nnunetv2/results"
echo "nnUNet paths initialized:"
echo "  nnUNet_raw: $nnUNet_raw"
echo "  nnUNet_preprocessed: $nnUNet_preprocessed"
echo "  nnUNet_results: $nnUNet_results"
