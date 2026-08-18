from pathlib import Path
import os

nnUNet_raw = Path(os.environ.get("nnUNet_raw", ""))
nnUNet_preprocessed = Path(os.environ.get("nnUNet_preprocessed", ""))
nnUNet_results = Path(os.environ.get("nnUNet_results", ""))
