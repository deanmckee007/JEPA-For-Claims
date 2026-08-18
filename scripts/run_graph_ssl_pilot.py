import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jepa_models.graph_ssl import build_code_graph, train_graph_ssl_embeddings
from jepa_utils.config import (
    Config,
    apply_config_overrides,
    apply_runtime_config_overrides,
    apply_training_recipe,
)
from jepa_utils.data_prep import prepare_data


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the A11 graph-SSL transfer pilot against the fixed-horizon Stage 1 recipe."
    )
    parser.add_argument("--data-path", required=True, help="Parquet dataset path.")
    parser.add_argument("--seed", type=int, default=314, help="Shared seed for graph pretrain and Stage 1.")
    parser.add_argument(
        "--base-recipe",
        default="sigreg_dense_hybrid_dollar_masked_grounding_sigreg0",
        help="Recipe used to build vocab/data and as the conceptual control family.",
    )
    parser.add_argument(
        "--transfer-recipe",
        default="a11_graph_ssl_transfer",
        help="Recipe used for the transferred Stage 1 run.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run directory name under experiments/runs. Defaults to A11pilot_seed<seed>.",
    )
    parser.add_argument(
        "--representation-pretrain-epochs",
        type=int,
        default=24,
        help="Fixed Stage 1 horizon.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=4,
        help="Checkpoint cadence for fixed-horizon evaluation.",
    )
    parser.add_argument("--graph-epochs", type=int, default=120, help="Offline graph SSL epochs.")
    parser.add_argument(
        "--graph-hidden-dim",
        type=int,
        default=None,
        help="Hidden size for the graph SSL encoder. Defaults to max(hidden_dim, 2 * embedding_dim).",
    )
    parser.add_argument("--graph-lr", type=float, default=1e-2, help="Graph SSL optimizer LR.")
    parser.add_argument("--graph-weight-decay", type=float, default=1e-4, help="Graph SSL weight decay.")
    parser.add_argument("--graph-mask-ratio", type=float, default=0.2, help="Masked-node ratio for graph SSL.")
    parser.add_argument("--graph-mask-loss-weight", type=float, default=1.0, help="Masked-node reconstruction loss weight.")
    parser.add_argument("--graph-negative-ratio", type=int, default=2, help="Negative samples per positive edge.")
    parser.add_argument("--graph-max-edges-per-epoch", type=int, default=40000, help="Positive edge samples per graph SSL epoch.")
    parser.add_argument(
        "--graph-temporal-edge-weight",
        type=float,
        default=0.0,
        help="Optional cross-claim temporal edge weight when building the code graph.",
    )
    parser.add_argument(
        "--graph-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for graph SSL pretraining.",
    )
    parser.add_argument(
        "--compare-report",
        default=None,
        help="Optional baseline checkpoint-series JSON to compare against in the summary.",
    )
    parser.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        default=None,
        help="Additional config overrides forwarded to the transferred Stage 1 run.",
    )
    return parser.parse_args(argv)


def _run_logged(cmd, cwd: Path, log_path: Path):
    with open(log_path, "w", encoding="utf-8") as fh:
        subprocess.run(
            cmd,
            cwd=str(cwd),
            check=True,
            stdout=fh,
            stderr=subprocess.STDOUT,
        )


def _build_data_config(args):
    config = apply_training_recipe(Config(), args.base_recipe)
    config.data_path = args.data_path
    config.seed = args.seed
    config.use_plotting = False
    config.use_generative_save = False
    config.pretrain_diffusion = False
    return apply_runtime_config_overrides(config)


def _summarize_report(report_path: Path):
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
    run_name = args.run_name or f"A11pilot_seed{args.seed}"
    run_rel = Path("experiments") / "runs" / run_name
    run_dir = REPO_ROOT / run_rel
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing data and vocab for graph SSL...", flush=True)
    data_config = _build_data_config(args)
    train_dataset, _, _, _, data_config, dataset = prepare_data(data_config)
    graph = build_code_graph(
        dataset.processed_data,
        dataset.cpt_vocab,
        dataset.icd_vocab,
        dataset.ttnc_vocab,
        include_ttnc=True,
        temporal_edge_weight=args.graph_temporal_edge_weight,
    )

    graph_hidden_dim = args.graph_hidden_dim or max(data_config.hidden_dim, data_config.embedding_dim * 2)
    print(
        f"Graph stats: nodes={graph.num_nodes} edges={graph.num_edges} hidden_dim={graph_hidden_dim}",
        flush=True,
    )
    graph_state = train_graph_ssl_embeddings(
        graph,
        embedding_dim=data_config.embedding_dim,
        hidden_dim=graph_hidden_dim,
        epochs=args.graph_epochs,
        lr=args.graph_lr,
        weight_decay=args.graph_weight_decay,
        mask_ratio=args.graph_mask_ratio,
        mask_loss_weight=args.graph_mask_loss_weight,
        negative_ratio=args.graph_negative_ratio,
        max_edges_per_epoch=args.graph_max_edges_per_epoch,
        device=args.graph_device,
        seed=args.seed,
    )
    graph_embedding_path = run_dir / "graph_embeddings.pt"
    torch.save(graph_state, graph_embedding_path)

    train_cmd = [
        sys.executable,
        "scripts/train.py",
        "--recipe",
        args.transfer_recipe,
        "--data-path",
        args.data_path,
        "--accelerator",
        "gpu",
        "--devices",
        "1",
        "--representation-pretrain-epochs",
        str(args.representation_pretrain_epochs),
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
        f"graph_embedding_path={run_rel.as_posix()}/graph_embeddings.pt",
        "--set",
        "checkpoint_save_top_k=0",
        "--set",
        f"checkpoint_every_n_epochs={args.checkpoint_every}",
        "--set",
        f"checkpoint_dirpath={run_rel.as_posix()}/checkpoints",
    ]
    for override in args.config_overrides or []:
        train_cmd.extend(["--set", override])
    print("Running transferred Stage 1 training...", flush=True)
    _run_logged(train_cmd, REPO_ROOT, run_dir / "train.log")

    eval_cmd = [
        sys.executable,
        "scripts/evaluate_checkpoint_series.py",
        "--checkpoint-dir",
        str(run_rel / "checkpoints"),
        "--pattern",
        "stage1-epoch*.ckpt",
        "--recipe",
        args.transfer_recipe,
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
        f"graph_embedding_path={run_rel.as_posix()}/graph_embeddings.pt",
    ]
    for override in args.config_overrides or []:
        eval_cmd.extend(["--set", override])
    print("Running checkpoint-series cost probe...", flush=True)
    _run_logged(eval_cmd, REPO_ROOT, run_dir / "eval.log")

    result_summary = _summarize_report(run_dir / "checkpoint_series_cost_probe.json")
    result_summary.update(
        {
            "run_name": run_name,
            "base_recipe": args.base_recipe,
            "transfer_recipe": args.transfer_recipe,
            "graph_embedding_path": str(graph_embedding_path),
            "graph_stats": graph_state["graph_stats"],
            "graph_epochs": args.graph_epochs,
            "graph_history_tail": graph_state["history"][-5:],
        }
    )

    compare_summary = None
    if args.compare_report:
        compare_summary = _summarize_report(Path(args.compare_report))
        result_summary["compare_report"] = args.compare_report
        result_summary["baseline_compare"] = compare_summary

    (run_dir / "pilot_summary.json").write_text(
        json.dumps(result_summary, indent=2),
        encoding="utf-8",
    )

    lines = [
        "## A11 Graph SSL Transfer Pilot",
        "",
        f"Run: `{run_name}`",
        f"Base recipe: `{args.base_recipe}`",
        f"Transfer recipe: `{args.transfer_recipe}`",
        f"Graph embedding path: `{graph_embedding_path.as_posix()}`",
        "",
        "Graph pretrain:",
        f"- nodes: `{graph.num_nodes}`",
        f"- edges: `{graph.num_edges}`",
        f"- epochs: `{args.graph_epochs}`",
        f"- device: `{args.graph_device}`",
        "",
        "Transferred Stage 1 best overall checkpoint:",
        f"- `{result_summary['best_overall_checkpoint']}`",
        f"- MAE `${result_summary['best_overall_mae']:.2f}`",
        f"- WAPE `{result_summary['best_overall_wape']:.4f}%`",
        f"- RMSE `${result_summary['best_overall_rmse']:.2f}`",
        f"- q4 RMSE `${result_summary['best_overall_q4_rmse']:.2f}`",
        "",
        "Transferred Stage 1 best q4 checkpoint:",
        f"- `{result_summary['best_q4_checkpoint']}`",
        f"- q4 RMSE `${result_summary['best_q4_rmse']:.2f}`",
    ]
    if compare_summary is not None:
        lines.extend(
            [
                "",
                "Baseline compare:",
                f"- overall checkpoint `{compare_summary['best_overall_checkpoint']}`",
                f"- overall RMSE `${compare_summary['best_overall_rmse']:.2f}`",
                f"- best q4 checkpoint `{compare_summary['best_q4_checkpoint']}`",
                f"- best q4 RMSE `${compare_summary['best_q4_rmse']:.2f}`",
            ]
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("A11 pilot complete.", flush=True)


if __name__ == "__main__":
    main()
