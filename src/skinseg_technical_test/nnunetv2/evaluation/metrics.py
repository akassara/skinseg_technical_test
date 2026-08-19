import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.metrics import f1_score

try:
	from .dashboard_utils import write_html_scorecard
except ImportError:
	from dashboard_utils import write_html_scorecard

LOGGER = logging.getLogger(__name__)
LABEL_NAMES = {
	0: "background",
	1: "surface",
	2: "jde",
	3: "corneous",
}
DICE_COLUMNS = [f"dice_{name}" for name in LABEL_NAMES.values()]


def _case_id(path):
	name = path.name
	return name[:-7] if name.endswith(".nii.gz") else path.stem


def _read_labels(path):
	return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.int64)


def _class_metrics(reference, prediction, label):
	reference_mask = (reference == label).astype(np.uint8).ravel()
	prediction_mask = (prediction == label).astype(np.uint8).ravel()
	if reference_mask.sum() == 0:
		return {"dice": np.nan}
	return {
		"dice": f1_score(reference_mask, prediction_mask, zero_division=0),
	}


def compute_metrics(predictions_dir, ground_truth_dir=None, output_path=None):
	"""Create a per-fold validation scorecard for an nnU-Net model.

	``predictions_dir`` is a model directory containing ``fold_*/validation``
	folders. Predictions and labels are matched by their ``.nii.gz`` filename.
	The returned DataFrame contains one row per case and class, followed by
	per-fold macro averages. Set ``output_path`` to save the scorecard as CSV.
	"""
	model_dir = Path(predictions_dir)
	LOGGER.info("Starting validation scoring for model: %s", model_dir)
	if ground_truth_dir is None:
		dataset_name = next((part for part in model_dir.parts if part.startswith("Dataset")), None)
		if dataset_name is None:
			raise ValueError("ground_truth_dir is required when the model path has no DatasetXXX component")
		ground_truth_dir = Path(
			__import__("os").environ.get("nnUNet_raw", "/workspace/nnunet_data/nnunet_raw")
		) / dataset_name / "labelsTr"

	ground_truth_dir = Path(ground_truth_dir)
	LOGGER.info("Using validation labels from: %s", ground_truth_dir)
	rows = []
	validation_dirs = sorted(model_dir.glob("fold_*/validation"))
	if not validation_dirs:
		raise FileNotFoundError(f"No fold_*/validation directories found under {model_dir}")
	LOGGER.info("Found %d validation folds", len(validation_dirs))

	for validation_dir in validation_dirs:
		fold = validation_dir.parent.name
		prediction_paths = sorted(validation_dir.glob("*.nii.gz"))
		LOGGER.info("Processing %s: %d prediction files", fold, len(prediction_paths))
		for prediction_path in prediction_paths:
			reference_path = ground_truth_dir / prediction_path.name
			if not reference_path.exists():
				LOGGER.error("Missing ground truth for %s: %s", prediction_path.name, reference_path)
				raise FileNotFoundError(f"Missing ground truth for {prediction_path.name}: {reference_path}")

			reference = _read_labels(reference_path)
			prediction = _read_labels(prediction_path)
			if reference.shape != prediction.shape:
				LOGGER.error("Shape mismatch for %s: %s != %s", prediction_path.name, reference.shape, prediction.shape)
				raise ValueError(f"Shape mismatch for {prediction_path.name}: {reference.shape} != {prediction.shape}")

			labels = [
				int(label)
				for label in sorted(set(np.unique(reference)) | set(np.unique(prediction)))
				if int(label) in LABEL_NAMES
			]
			for label in labels:
				rows.append({
					"fold": fold,
					"case_id": _case_id(prediction_path),
					"label": int(label),
					"label_name": LABEL_NAMES[int(label)],
					**_class_metrics(reference, prediction, int(label)),
				})

	case_scores = pd.DataFrame(rows)
	case_scores = (
		case_scores.pivot(index=["fold", "case_id"], columns="label_name", values="dice")
		.rename(columns=lambda name: f"dice_{name}")
		.reset_index()
	)
	for column in DICE_COLUMNS:
		if column not in case_scores:
			case_scores[column] = np.nan
	fold_scores = (
		case_scores.groupby("fold", as_index=False)[DICE_COLUMNS]
		.mean()
		.assign(case_id="FOLD_MACRO_AVERAGE")
	)
	scorecard = pd.concat([case_scores, fold_scores[case_scores.columns]], ignore_index=True)
	scorecard = scorecard[["fold", "case_id", *DICE_COLUMNS]]
	scorecard = scorecard.sort_values(["fold", "case_id"]).reset_index(drop=True)

	if output_path is None:
		output_path = model_dir / "validation_scorecard.csv"
	scorecard.to_csv(output_path, index=False)
	LOGGER.info("Wrote CSV scorecard: %s (%d rows)", output_path, len(scorecard))
	return scorecard


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
	parser = argparse.ArgumentParser(description="Generate an nnU-Net validation scorecard.")
	parser.add_argument(
		"--model-dir",
		type=Path,
		default=Path("/workspace/nnunet_data/nnunet_results/Dataset001_SkinSegmentation/nnUNetTrainer_50epochs__nnUNetPlans__2d"),
		help="nnU-Net model directory containing fold_*/validation.",
	)
	parser.add_argument(
		"--ground-truth-dir",
		type=Path,
		default=Path("/workspace/nnunet_data/nnunet_raw/Dataset001_SkinSegmentation/labelsTr"),
		help="Directory containing the validation ground-truth NIfTI files.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help="CSV output path; defaults to validation_scorecard.csv in the model directory.",
	)
	parser.add_argument(
		"--html-output",
		type=Path,
		default=None,
		help="HTML output path; defaults to validation_scorecard.html in the model directory.",
	)
	args = parser.parse_args()
	scorecard = compute_metrics(args.model_dir, args.ground_truth_dir, args.output)
	html_path = write_html_scorecard(scorecard, args.model_dir, args.html_output)
	LOGGER.info("Wrote HTML scorecard: %s", html_path)


