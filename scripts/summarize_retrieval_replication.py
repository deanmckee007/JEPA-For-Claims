"""Render completed retrieval replication metrics without refitting or selection."""
import argparse
import json
from pathlib import Path


def render_report(payload):
    metrics = payload["aggregate"]["metrics"]
    runs = payload["runs"]
    labels = {"copy_only": "Copy only", "flat_copy_residual": "Flat residual",
        "flat_with_candidate_filter": "Flat + candidate filter", "candidate": "Candidate decoder"}
    lines = ["# Retrieval replication and calibration — September 5", "",
        f"Completed {payload['completed_runs']} of {payload['planned_runs']} planned encoder/decoder pairs.", "",
        "Three existing online LeVJEPA encoders (42–44) are crossed with decoder seeds 201–203.",
        "A fixed content-ID-grouped 80/20 split inside training reserves 20% for calibration.",
        "Only the 80% fitting subset supplies decoder labels and retrieved next claims.",
        "The frozen encoders were already pretrained on the original training cohort.",
        "Calibration selects copy/add logit thresholds and cardinality scaling by inner micro F1.",
        "The search grid was fixed before scoring validation; test was never accessed.", "",
        f"Fit / calibration / validation rows: {runs[0]['fit_samples']} / {runs[0]['calibration_samples']} / {runs[0]['validation_samples']}.",
        "The original pilot fit all training rows; these scores use less decoder supervision.", "",
        "## Ranking and decoded sets", "",
        "AP uses the original model scores. Calibrated F1 uses decoded sets and is reported separately.", "",
        "| Model | CPT AP | ICD AP | CPT F1 before → after calibration | ICD F1 before → after calibration |",
        "|---|---:|---:|---:|---:|"]
    for condition, label in labels.items():
        result = metrics[condition]
        raw = result["uncalibrated"]
        calibrated = result["calibrated_sets"]
        lines.append(f"| {label} | {raw['cpt']['micro_average_precision']['mean']:.4f} | "
            f"{raw['icd']['micro_average_precision']['mean']:.4f} | "
            f"{raw['cpt']['micro_f1']['mean']:.4f} → {calibrated['cpt']['micro_f1']['mean']:.4f} | "
            f"{raw['icd']['micro_f1']['mean']:.4f} → {calibrated['icd']['micro_f1']['mean']:.4f} |")
    lines += ["", "## Paired comparisons", "",
        "Each cell reports candidate mean delta and the number of positive deltas across matched runs.", "",
        "| Comparator | CPT AP delta / wins | ICD AP delta / wins | Calibrated CPT F1 delta / wins | Calibrated ICD F1 delta / wins |",
        "|---|---:|---:|---:|---:|"]
    for comparator, modalities in payload["aggregate"]["paired_candidate_deltas"].items():
        cells = []
        for metric in ["micro_average_precision", "micro_f1"]:
            for modality in ["cpt", "icd"]:
                delta = modalities[modality][metric]
                cells.append(f"{delta['mean_delta']:+.4f} / {delta['wins']}/{delta['num_runs']}")
        lines.append(f"| {labels[comparator]} | " + " | ".join(cells) + " |")
    lines += ["", "## New-code coverage", "",
        "Known tokens only (IDs > 1): PAD and UNK are excluded. Persistent targets already occur",
        "in the last claim; new targets do not. Coverage is independent of the decoder seed.", "",
        "| Encoder seed | CPT overall / persistent / new recall | ICD overall / persistent / new recall |",
        "|---|---:|---:|"]
    unique = {run['encoder_seed']: run for run in runs}
    for seed, run in sorted(unique.items()):
        cells = []
        for key in ["cpt", "icd"]:
            coverage = run["candidate_coverage"][key]
            cells.append(" / ".join(f"{coverage[part]['recall']:.4f}" for part in ["all_known", "persistent", "new"]))
        lines.append(f"| {seed} | " + " | ".join(cells) + " |")
    lines += ["", "Calibration maximizes total micro F1; it may prefer persistence over emitting new codes.", "",
        "| Model | Calibrated CPT new-code precision / recall | Calibrated ICD new-code precision / recall |",
        "|---|---:|---:|"]
    for condition, label in labels.items():
        cells = []
        for key in ["cpt", "icd"]:
            values = metrics[condition]["calibrated_sets"][key]
            cells.append(f"{values['new_code_precision']['mean']:.4f} / {values['new_code_recall']['mean']:.4f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "## Interpretation limits and artifacts", "",
        "These are validation results on shared patients, not nine independent population samples.",
        "Crossed-run SD is descriptive. Copy-only is deterministic and is repeated for pairing,",
        "not counted as independent replication. Candidate and flat heads share budgets, data and",
        "hidden width but are not parameter-count matched. Grouping uses frozen content IDs because",
        "stable member IDs are not present. No test-set or production promotion is implied.", "",
        "The run directory contains protocol.json, inner_split.json, summary.json and one directory",
        "per seed pair with the calibrated settings, metrics, loss histories, and local trained heads.",
        "Reproduce with scripts/run_retrieval_replication.py using the three LeVJEPA checkpoints,",
        "--decoder-seeds 201 202 203, the frozen data contract, and a new output directory.",
        "Render this report with scripts/summarize_retrieval_replication.py SUMMARY_JSON --output REPORT_MD."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    if payload["completed_runs"] != payload["planned_runs"]:
        raise ValueError("Refusing to render an incomplete sweep as the final report")
    Path(args.output).write_text(render_report(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
