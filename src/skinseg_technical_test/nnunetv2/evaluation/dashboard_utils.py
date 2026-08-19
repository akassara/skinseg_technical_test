from html import escape
import base64
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk

LOGGER = logging.getLogger(__name__)
COLORS = {0: "white", 1: "#ff4d6d", 2: "#35a7ff", 3: "#ffd166"}


def _read_array(path):
	return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).squeeze()


def _write_overlay(image_path, prediction_path, output_path):
	image = _read_array(image_path)
	prediction = _read_array(prediction_path)
	fig, axis = plt.subplots(figsize=(6, 5), dpi=140)
	axis_image = axis.imshow(image, cmap="gray")
	axis_image.set_clim(float(np.nanmin(image)), float(np.nanmax(image)))
	for label, color in COLORS.items():
		if label == 0:
			continue
		predicted_mask = (prediction == label).astype(np.float32)
		axis.contour(predicted_mask, colors=[color], linewidths=1.4)
	axis.set_title("Predicted contours")
	axis.axis("off")
	fig.tight_layout(pad=0.5)
	fig.savefig(output_path, bbox_inches="tight")
	plt.close(fig)


def _find_case_file(directory, case):
	"""Find a test file whose nnU-Net case name starts with ``case``."""
	for path in Path(directory).glob(f"{case}*.nii.gz"):
		return path
	return None


def write_html_scorecard(scorecard, model_dir, output_path=None):
	"""Write a styled, browser-readable HTML version of a scorecard."""
	model_dir = Path(model_dir).resolve()
	if output_path is None:
		output_path = model_dir / "validation_scorecard.html"
	output_path = Path(output_path).resolve()

	metric_columns = [column for column in scorecard.columns if column.startswith("dice_")]
	display_columns = ["split", "fold", "case", *metric_columns]
	display = scorecard[display_columns].copy()
	overlay_paths = {}
	overlay_sources = {
		(split, fold, case): (Path(image_path), Path(prediction_path))
		for split, fold, case, image_path, prediction_path in scorecard.attrs.get("overlay_sources", [])
	}
	dataset_name = next((part for part in model_dir.parts if part.startswith("Dataset")), None)
	data_root = Path("/workspace/nnunet_data")
	raw_root = Path(__import__("os").environ.get("nnUNet_raw", data_root / "nnunet_raw"))
	for row in scorecard.itertuples():
		if row.split != "test" or row.case == "FOLD_MACRO_AVERAGE":
			continue
		key = (row.split, row.fold, row.case)
		if key in overlay_sources:
			continue
		fold_dir = model_dir / row.fold / "predictionsTs"
		prediction_path = _find_case_file(fold_dir, row.case)
		image_path = _find_case_file(raw_root / dataset_name / "imagesTs", row.case)
		if prediction_path and image_path:
			overlay_sources[key] = (image_path, prediction_path)
		else:
			LOGGER.warning("Cannot create test overlay for %s", row.case)
	for (split, fold, case_id), (image_path, prediction_path) in overlay_sources.items():
		overlay_dir = model_dir / "scorecard_overlays"
		overlay_dir.mkdir(parents=True, exist_ok=True)
		overlay_path = overlay_dir / f"{split}_{fold}_{case_id}.png"
		if not overlay_path.exists():
			_write_overlay(image_path, prediction_path, overlay_path)
		overlay_paths[(split, fold, case_id)] = overlay_path
	display["overlay"] = [
		f'<img src="data:image/png;base64,{base64.b64encode(overlay_paths[(row.split, row.fold, row.case)].read_bytes()).decode("ascii")}" width="180" loading="lazy">'
		if (row.split, row.fold, row.case) in overlay_paths else ""
		for row in scorecard.itertuples()
	]
	for column in metric_columns:
		display[column] = display[column].map(lambda value: "N/A" if pd.isna(value) else f"{value:.1%}")

	summary = display[display["case"] == "FOLD_MACRO_AVERAGE"]
	summary_table = summary.drop(columns="overlay").to_html(index=False, classes="score-table summary-table", border=0)
	detail_table = display[display["case"] != "FOLD_MACRO_AVERAGE"].to_html(
		index=False, classes="score-table", border=0, escape=False
	)
	title = escape(model_dir.name)
	html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Validation scorecard: {title}</title>
<style>
:root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
body {{ margin: 0; background: #f4f7fb; color: #172033; }}
main {{ max-width: 1200px; margin: 0 auto; padding: 40px 24px 64px; }}
h1 {{ margin: 0 0 8px; font-size: clamp(1.5rem, 3vw, 2.25rem); }}
h2 {{ margin: 36px 0 12px; font-size: 1.1rem; }}
.subtitle {{ color: #61708a; margin: 0; word-break: break-word; }}
.table-wrap {{ overflow-x: auto; background: white; border: 1px solid #dce3ee; border-radius: 10px; box-shadow: 0 8px 24px #17203312; }}
.score-table {{ width: 100%; border-collapse: collapse; min-width: 680px; }}
.score-table th {{ background: #172033; color: white; text-align: left; font-weight: 600; }}
.score-table th, .score-table td {{ padding: 11px 14px; border-bottom: 1px solid #e7ecf3; white-space: nowrap; }}
.score-table tbody tr:last-child td {{ border-bottom: 0; }}
.score-table tbody tr:hover {{ background: #f0f6ff; }}
.summary-table tbody tr {{ background: #e8f2ff; font-weight: 600; }}
.metric {{ text-align: right; font-variant-numeric: tabular-nums; }}
</style>
</head>
<body>
<main>
<h1>Validation scorecard</h1>
<p class="subtitle">{title}</p>
<h2>Fold averages</h2>
<div class="table-wrap">{summary_table}</div>
<h2>Per-case metrics</h2>
<div class="table-wrap">{detail_table}</div>
</main>
</body>
</html>
"""
	Path(output_path).write_text(html, encoding="utf-8")
	LOGGER.info("Rendered HTML scorecard with %d rows", len(scorecard))
	return Path(output_path)
