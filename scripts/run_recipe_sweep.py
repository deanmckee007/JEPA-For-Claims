import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_CONTRACT = REPO_ROOT / "artifacts" / "data_contracts" / "claims_seed42_v1.json"
DEFAULT_RECIPES = ["vicreg_baseline", "sigreg_core", "sigreg_dense"]


def get_metric(result, *names):
    for name in names:
        if name in result:
            return result.get(name)
    return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run train+eval sweeps for named JEPA SSL recipes."
    )
    parser.add_argument("--data-path", type=str, required=True, help="Path to the parquet dataset.")
    parser.add_argument(
        "--data-contract",
        type=str,
        default=str(DEFAULT_DATA_CONTRACT),
        help="Frozen split/vocabulary contract shared by all recipes.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="recipe_sweep",
        help="Directory for checkpoints, eval reports, and sweep summaries.",
    )
    parser.add_argument(
        "--recipes",
        nargs="+",
        default=DEFAULT_RECIPES,
        help="Recipe names to run. Defaults to the VICReg/SIGReg comparison trio.",
    )
    parser.add_argument(
        "--accelerator",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Trainer/eval accelerator to use.",
    )
    parser.add_argument("--devices", type=int, default=1, help="Lightning device count.")
    parser.add_argument(
        "--representation-pretrain-epochs",
        type=int,
        default=1,
        help="Number of stage-1 epochs to run for each recipe.",
    )
    parser.add_argument("--max-samples", type=int, default=2000, help="Max samples for downstream eval.")
    parser.add_argument("--retrieval-k", type=int, default=5, help="k for retrieval hit rate.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse checkpoints and eval JSON files that already exist in the output dir.",
    )
    return parser.parse_args(argv)


def build_train_command(
    recipe: str,
    data_path: str,
    checkpoint_path: Path,
    accelerator: str,
    devices: int,
    representation_pretrain_epochs: int,
    data_contract: str | None = None,
):
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train.py"),
        "--recipe",
        recipe,
        "--data-path",
        data_path,
        "--accelerator",
        accelerator,
        "--devices",
        str(devices),
        "--representation-pretrain-epochs",
        str(representation_pretrain_epochs),
        "--generator-train-epochs",
        "0",
        "--joint-train-epochs",
        "0",
        "--disable-generative-save",
        "--out-encoder-ckpt",
        str(checkpoint_path),
    ]
    if data_contract:
        command.extend(["--data-contract", data_contract])
    return command


def build_eval_command(
    recipe: str,
    data_path: str,
    checkpoint_path: Path,
    output_json: Path,
    accelerator: str,
    max_samples: int,
    retrieval_k: int,
    representation_source: str = "patient_representation_pre_sae",
    data_contract: str | None = None,
    split: str = "val",
):
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "evaluate_representations.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--data-path",
        data_path,
        "--recipe",
        recipe,
        "--accelerator",
        accelerator,
        "--max-samples",
        str(max_samples),
        "--retrieval-k",
        str(retrieval_k),
        "--representation-source",
        representation_source,
        "--output-json",
        str(output_json),
    ]
    if data_contract:
        command.extend(["--data-contract", data_contract, "--split", split])
    return command


def run_command(command):
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def load_eval_report(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_summary_files(results, output_dir: Path):
    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    summary_md = output_dir / "summary.md"

    summary_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    fieldnames = [
        "recipe",
        "checkpoint",
        "target_probe_mae_dollars",
        "target_probe_wape_percent",
        "target_probe_rmse_dollars",
        "target_probe_rmse_log1p",
        "cluster_silhouette",
        "ttnc_proxy_retrieval_hit_rate_at_5",
        "ttnc_proxy_label_cluster_ari",
        "ttnc_proxy_probe_accuracy",
        "ttnc_proxy_probe_macro_f1",
        "num_samples",
        "embedding_dim",
        "device",
        "ttnc_proxy_label_source",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "recipe": result.get("recipe"),
                    "checkpoint": result.get("checkpoint"),
                    "target_probe_mae_dollars": result.get("target_probe_mae_dollars"),
                    "target_probe_wape_percent": result.get("target_probe_wape_percent"),
                    "target_probe_rmse_dollars": result.get("target_probe_rmse_dollars"),
                    "target_probe_rmse_log1p": result.get("target_probe_rmse_log1p"),
                    "cluster_silhouette": result.get("cluster_silhouette"),
                    "ttnc_proxy_retrieval_hit_rate_at_5": get_metric(
                        result,
                        "ttnc_proxy_retrieval_hit_rate_at_5",
                        "specialty_retrieval_hit_rate_at_5",
                    ),
                    "ttnc_proxy_label_cluster_ari": get_metric(
                        result,
                        "ttnc_proxy_label_cluster_ari",
                        "cluster_ari",
                    ),
                    "ttnc_proxy_probe_accuracy": get_metric(
                        result,
                        "ttnc_proxy_probe_accuracy",
                        "specialty_probe_accuracy",
                    ),
                    "ttnc_proxy_probe_macro_f1": get_metric(
                        result,
                        "ttnc_proxy_probe_macro_f1",
                        "specialty_probe_macro_f1",
                    ),
                    "num_samples": result.get("num_samples"),
                    "embedding_dim": result.get("embedding_dim"),
                    "device": result.get("device"),
                    "ttnc_proxy_label_source": result.get(
                        "ttnc_proxy_label_source",
                        "legacy_specialty_proxy",
                    ),
                }
            )

    lines = [
        "# Recipe Sweep Summary",
        "",
        "| Recipe | MAE ($) | WAPE (%) | RMSE ($) | Log RMSE | Silhouette | TTNC Proxy Retrieval@5 | TTNC Proxy ARI | TTNC Proxy Acc | TTNC Proxy Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            "| {recipe} | {mae:.2f} | {wape:.2f} | {rmse_dollars:.2f} | {rmse_log1p:.4f} | {silhouette:.4f} | {retrieval:.4f} | {cluster_ari:.4f} | {acc:.4f} | {f1:.4f} |".format(
                recipe=result["recipe"],
                mae=result.get("target_probe_mae_dollars", float("nan")),
                wape=result.get("target_probe_wape_percent", float("nan")),
                rmse_dollars=result.get("target_probe_rmse_dollars", float("nan")),
                rmse_log1p=result.get("target_probe_rmse_log1p", float("nan")),
                silhouette=result.get("cluster_silhouette", float("nan")),
                retrieval=get_metric(result, "ttnc_proxy_retrieval_hit_rate_at_5", "specialty_retrieval_hit_rate_at_5") or float("nan"),
                cluster_ari=get_metric(result, "ttnc_proxy_label_cluster_ari", "cluster_ari") or float("nan"),
                acc=get_metric(result, "ttnc_proxy_probe_accuracy", "specialty_probe_accuracy") or float("nan"),
                f1=get_metric(result, "ttnc_proxy_probe_macro_f1", "specialty_probe_macro_f1") or float("nan"),
            )
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return summary_json, summary_csv, summary_md


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for recipe in args.recipes:
        recipe_dir = output_dir / recipe
        recipe_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = recipe_dir / "encoder.ckpt"
        eval_json = recipe_dir / "representation_eval.json"

        if not (args.skip_existing and checkpoint_path.exists()):
            run_command(
                build_train_command(
                    recipe=recipe,
                    data_path=args.data_path,
                    checkpoint_path=checkpoint_path,
                    accelerator=args.accelerator,
                    devices=args.devices,
                    representation_pretrain_epochs=args.representation_pretrain_epochs,
                    data_contract=args.data_contract,
                )
            )

        if not (args.skip_existing and eval_json.exists()):
            run_command(
                build_eval_command(
                    recipe=recipe,
                    data_path=args.data_path,
                    checkpoint_path=checkpoint_path,
                    output_json=eval_json,
                    accelerator=args.accelerator,
                    max_samples=args.max_samples,
                    retrieval_k=args.retrieval_k,
                    data_contract=args.data_contract,
                    split="val",
                )
            )

        results.append(load_eval_report(eval_json))

    summary_json, summary_csv, summary_md = write_summary_files(results, output_dir)
    print(json.dumps(results, indent=2))
    print(f"Summary JSON: {summary_json.resolve()}")
    print(f"Summary CSV: {summary_csv.resolve()}")
    print(f"Summary Markdown: {summary_md.resolve()}")


if __name__ == "__main__":
    main()
