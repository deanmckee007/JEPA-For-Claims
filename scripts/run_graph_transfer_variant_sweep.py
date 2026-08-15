import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


VARIANTS = [
    {
        "name": "cpt_icd_only_hard",
        "overrides": [
            "graph_transfer_ttnc=False",
            "graph_embedding_mix=1.0",
        ],
    },
    {
        "name": "cpt_icd_only_mix025",
        "overrides": [
            "graph_transfer_ttnc=False",
            "graph_embedding_mix=0.25",
        ],
    },
    {
        "name": "cpt_icd_only_mix050",
        "overrides": [
            "graph_transfer_ttnc=False",
            "graph_embedding_mix=0.5",
        ],
    },
    {
        "name": "all_streams_mix025",
        "overrides": [
            "graph_embedding_mix=0.25",
        ],
    },
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run A11 graph-transfer variants against a fixed pretrained graph embedding checkpoint."
    )
    parser.add_argument("--graph-embedding-path", required=True, help="Path to graph_embeddings.pt")
    parser.add_argument("--data-path", required=True, help="Parquet dataset path.")
    parser.add_argument("--seed", type=int, default=314, help="Shared Stage 1 seed.")
    parser.add_argument("--recipe", default="a11_graph_ssl_transfer", help="Transfer recipe.")
    parser.add_argument(
        "--summary-dir",
        default=None,
        help="Directory for sweep summary artifacts. Defaults to experiments/a11_transfer_variants_seed<seed>.",
    )
    parser.add_argument(
        "--prefix",
        default="A11variant",
        help="Run-name prefix under experiments/runs.",
    )
    parser.add_argument("--epochs", type=int, default=24, help="Fixed Stage 1 horizon.")
    parser.add_argument("--checkpoint-every", type=int, default=4, help="Checkpoint cadence.")
    parser.add_argument(
        "--compare-report",
        default=None,
        help="Optional baseline checkpoint-series JSON used in the markdown summary.",
    )
    return parser.parse_args(argv)


def run_logged(cmd, cwd: Path, log_path: Path):
    with open(log_path, "w", encoding="utf-8") as fh:
        subprocess.run(
            cmd,
            cwd=str(cwd),
            check=True,
            stdout=fh,
            stderr=subprocess.STDOUT,
        )


def summarize_report(report_path: Path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    best_overall = report["best_by_target_probe_rmse_dollars"]
    best_q4 = report["best_by_q4_high_cost_rmse_dollars"]
    return {
        "best_overall_checkpoint": Path(best_overall["checkpoint"]).name,
        "best_overall_mae": best_overall["target_probe_mae_dollars"],
        "best_overall_wape": best_overall["target_probe_wape_percent"],
        "best_overall_rmse": best_overall["target_probe_rmse_dollars"],
        "best_overall_q4_rmse": best_overall["target_cost_bucket"]["q4_high_cost"]["target_probe_rmse_dollars"],
        "best_q4_checkpoint": Path(best_q4["checkpoint"]).name,
        "best_q4_rmse": best_q4["target_cost_bucket"]["q4_high_cost"]["target_probe_rmse_dollars"],
    }


def main(argv=None):
    args = parse_args(argv)
    summary_dir = (
        Path(args.summary_dir)
        if args.summary_dir
        else REPO_ROOT / "experiments" / f"a11_transfer_variants_seed{args.seed}"
    )
    summary_dir.mkdir(parents=True, exist_ok=True)

    baseline_compare = None
    if args.compare_report:
        baseline_compare = summarize_report(Path(args.compare_report))

    rows = []
    for variant in VARIANTS:
        run_name = f"{args.prefix}{args.seed}_{variant['name']}"
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
            args.recipe,
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
            f"graph_embedding_path={Path(args.graph_embedding_path).as_posix()}",
            "--set",
            "checkpoint_save_top_k=0",
            "--set",
            f"checkpoint_every_n_epochs={args.checkpoint_every}",
            "--set",
            f"checkpoint_dirpath={run_rel.as_posix()}/checkpoints",
        ]
        for override in variant["overrides"]:
            train_cmd.extend(["--set", override])
        run_logged(train_cmd, REPO_ROOT, run_dir / "train.log")

        eval_cmd = [
            sys.executable,
            "scripts/evaluate_checkpoint_series.py",
            "--checkpoint-dir",
            str(run_rel / "checkpoints"),
            "--pattern",
            "stage1-epoch*.ckpt",
            "--recipe",
            args.recipe,
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
            "--set",
            f"graph_embedding_path={Path(args.graph_embedding_path).as_posix()}",
        ]
        for override in variant["overrides"]:
            eval_cmd.extend(["--set", override])
        run_logged(eval_cmd, REPO_ROOT, run_dir / "eval.log")

        row = {
            "name": run_name,
            "recipe": args.recipe,
            "overrides": variant["overrides"],
        }
        row.update(summarize_report(run_dir / "checkpoint_series_cost_probe.json"))
        rows.append(row)
        print(
            f"DONE {run_name} overall_rmse={row['best_overall_rmse']:.2f} "
            f"q4_rmse={row['best_q4_rmse']:.2f}",
            flush=True,
        )

    rows.sort(key=lambda row: row["best_overall_rmse"])
    (summary_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        f"## A11 Transfer Variants Seed {args.seed}",
        "",
        "Protocol:",
        f"- fixed {args.epochs}-epoch Stage 1 run",
        f"- checkpoints every {args.checkpoint_every} epochs",
        "- post-hoc checkpoint-series cost probe",
        f"- shared graph embeddings: `{Path(args.graph_embedding_path).as_posix()}`",
        "",
        "| Experiment | Overrides | Best overall ckpt | MAE | WAPE | RMSE | q4 RMSE at overall best | Best q4 ckpt | Best q4 RMSE |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        overrides = ", ".join(row["overrides"])
        lines.append(
            f"| {row['name']} | {overrides} | {row['best_overall_checkpoint']} | "
            f"{row['best_overall_mae']:.2f} | {row['best_overall_wape']:.4f} | "
            f"{row['best_overall_rmse']:.2f} | {row['best_overall_q4_rmse']:.2f} | "
            f"{row['best_q4_checkpoint']} | {row['best_q4_rmse']:.2f} |"
        )
    if baseline_compare is not None:
        lines.extend(
            [
                "",
                "Baseline compare:",
                f"- overall checkpoint `{baseline_compare['best_overall_checkpoint']}`",
                f"- overall RMSE `${baseline_compare['best_overall_rmse']:.2f}`",
                f"- best q4 checkpoint `{baseline_compare['best_q4_checkpoint']}`",
                f"- best q4 RMSE `${baseline_compare['best_q4_rmse']:.2f}`",
            ]
        )
    (summary_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("SWEEP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
