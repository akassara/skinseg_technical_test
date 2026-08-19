from html import escape
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def write_html_scorecard(scorecard, model_dir, output_path=None):
	"""Write a styled, browser-readable HTML version of a scorecard."""
	model_dir = Path(model_dir)
	if output_path is None:
		output_path = model_dir / "validation_scorecard.html"

	metric_columns = [column for column in scorecard.columns if column.startswith("dice_")]
	display_columns = ["fold", "case_id", *metric_columns]
	display = scorecard[display_columns].copy()
	for column in metric_columns:
		display[column] = display[column].map(lambda value: f"{value:.1%}")

	summary = display[display["case_id"] == "FOLD_MACRO_AVERAGE"]
	summary_table = summary.to_html(index=False, classes="score-table summary-table", border=0)
	detail_table = display[display["case_id"] != "FOLD_MACRO_AVERAGE"].to_html(
		index=False, classes="score-table", border=0
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
