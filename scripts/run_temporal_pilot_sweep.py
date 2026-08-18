import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


EXPERIMENTS = [
    {
        "name": "baseline",
        "recipe": "sigreg_dense_hybrid_dollar_masked_grounding_sigreg0",
        "overrides": [],
    },
    {
        "name": "cpc_w010",
        "recipe": "a10_temporal_contrastive",
        "overrides": [],
    },
    {
        "name": "cpc_w005",
        "recipe": "a10_temporal_contrastive",
        "overrides": ["temporal_loss_weight=0.05"],
    },
    {
        "name": "cpc_w020",
        "recipe": "a10_temporal_contrastive",
        "overrides": ["temporal_loss_weight=0.2"],
    },
    {
        "name": "ts2vec_w005",
        "recipe": "a10_temporal_contrastive",
        "overrides": ["temporal_ssl_mode=ts2vec", "temporal_loss_weight=0.05"],
    },
    {
        "name": "cpc_ctx8_future2_w005",
        "recipe": "a10_temporal_contrastive",
        "overrides": [
            "temporal_context_k=8",
            "temporal_future_steps=2",
            "temporal_loss_weight=0.05",
        ],
    },
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a fixed-horizon baseline vs A10 temporal pilot sweep."
    )
    parser.add_argument("--data-path", required=True, help="Parquet dataset path.")
    parser.add_argument("--seed", type=int, required=True, help="Shared seed for the sweep.")
    parser.add_argument(
        "--summary-dir",
        type=str,
        default=None,
        help="Directory for aggregate summary artifacts. Defaults to experiments/a10_seed<seed>_pilot.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="A10pilot",
        help="Run-name prefix under experiments/runs.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=24,
        help="Fixed Stage 1 horizon used for all runs.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=4,
        help="Checkpoint interval for the fixed-horizon protocol.",
    )
    return parser.parse_args(argv)


def run_command(cmd, cwd: Path, log_path: Path):
    with open(log_path, "w", encoding="utf-8") as fh:
        subprocess.run(
            cmd,
            cwd=str(cwd),
            check=True,
            stdout=fh,
            stderr=subprocess.STDOUT,
        )


def build_rows(summary_json: Path):
    report = json.loads(summary_json.read_text(encoding="utf-8"))
    best_overall = report["best_by_target_probe_rmse_dollars"]
    best_q4 = report["best_by_q4_high_cost_rmse_dollars"]
    return {
        "best_overall_checkpoint": Path(best_overall["checkpoint"]).name,
        "best_overall_mae": best_overall["target_probe_mae_dollars"],
        "best_overall_wape": best_overall["target_probe_wape_percent"],
        "best_overall_rmse": best_overall["target_probe_rmse_dollars"],
        "best_overall_q4_rmse": best_overall["target_cost_bucket"]["q4_high_cost"][
            "target_probe_rmse_dollars"
        ],
        "best_q4_checkpoint": Path(best_q4["checkpoint"]).name,
        "best_q4_mae": best_q4["target_probe_mae_dollars"],
        "best_q4_wape": best_q4["target_probe_wape_percent"],
        "best_q4_rmse": best_q4["target_probe_rmse_dollars"],
        "best_q4_q4_rmse": best_q4["target_cost_bucket"]["q4_high_cost"][
            "target_probe_rmse_dollars"
        ],
    }


def main(argv=None):
    args = parse_args(argv)
    summary_dir = (
        Path(args.summary_dir)
        if args.summary_dir
        else REPO_ROOT / "experiments" / f"a10_seed{args.seed}_pilot"
    )
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for exp in EXPERIMENTS:
        run_name = f"{args.prefix}{args.seed}_{exp['name']}"
        run_rel = Path("experiments") / "runs" / run_name
        run_dir = REPO_ROOT / run_rel
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"START {run_name}", flush=True)
        train_cmd = [
            sys.executable,
            "scripts/train.py",
            "--recipe",
            exp["recipe"],
            "--data-path",
            args.data_path,
            "--accelerator",
            "gpu",
            "--devices",
            "1",
            "--representation-pretrain-epochs",
            str(args.epochs),
            "--generator-train-epochs",
            "0",
            "--joint-train-epochs",
            "0",
            "--disable-generative-save",
            "--out-encoder-ckpt",
            str(run_rel / "encoder.ckpt"),
            "--seed",
            str(args.seed),
            "--set",
            "checkpoint_save_top_k=0",
            "--set",
            f"checkpoint_every_n_epochs={args.checkpoint_every}",
            "--set",
            f"checkpoint_dirpath={run_rel.as_posix()}/checkpoints",
        ]
        for override in exp["overrides"]:
            train_cmd.extend(["--set", override])
        run_command(train_cmd, REPO_ROOT, run_dir / "train.log")

        eval_cmd = [
            sys.executable,
            "scripts/evaluate_checkpoint_series.py",
            "--checkpoint-dir",
            str(run_rel / "checkpoints"),
            "--pattern",
            "stage1-epoch*.ckpt",
            "--recipe",
            exp["recipe"],
            "--data-path",
            args.data_path,
            "--accelerator",
            "gpu",
            "--seed",
            str(args.seed),
            "--representation-source",
            "patient_representation_pre_sae",
            "--output-json",
            str(run_rel / "checkpoint_series_cost_probe.json"),
        ]
        for override in exp["overrides"]:
            eval_cmd.extend(["--set", override])
        run_command(eval_cmd, REPO_ROOT, run_dir / "eval.log")

        row = {
            "name": run_name,
            "recipe": exp["recipe"],
            "overrides": exp["overrides"],
        }
        row.update(build_rows(run_dir / "checkpoint_series_cost_probe.json"))
        rows.append(row)
        print(
            f"DONE {run_name} overall_rmse={row['best_overall_rmse']:.2f} "
            f"q4_rmse={row['best_q4_q4_rmse']:.2f}",
            flush=True,
        )

    rows.sort(key=lambda row: row["best_overall_rmse"])
    (summary_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        f"## A10 Seed {args.seed} Pilot Sweep",
        "",
        "Protocol:",
        f"- fixed {args.epochs}-epoch Stage 1 run",
        f"- checkpoints every {args.checkpoint_every} epochs",
        "- post-hoc checkpoint-series cost probe",
        f"- seed {args.seed}",
        "",
        "| Experiment | Recipe | Overrides | Best overall ckpt | MAE | WAPE | RMSE | q4 RMSE at overall best | Best q4 ckpt | q4 RMSE |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        overrides = ", ".join(row["overrides"]) if row["overrides"] else "default"
        lines.append(
            f"| {row['name']} | {row['recipe']} | {overrides} | {row['best_overall_checkpoint']} | "
            f"{row['best_overall_mae']:.2f} | {row['best_overall_wape']:.4f} | "
            f"{row['best_overall_rmse']:.2f} | {row['best_overall_q4_rmse']:.2f} | "
            f"{row['best_q4_checkpoint']} | {row['best_q4_q4_rmse']:.2f} |"
        )
    (summary_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("SWEEP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
