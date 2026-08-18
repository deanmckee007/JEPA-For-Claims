import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_CONTRACT = REPO_ROOT / "artifacts" / "data_contracts" / "claims_seed42_v1.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run one JEPA experiment and record its config and evaluation artifacts."
    )
    parser.add_argument("--name", type=str, required=True, help="Unique experiment id.")
    parser.add_argument("--recipe", type=str, default="custom", help="Base recipe preset.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the parquet dataset.")
    parser.add_argument(
        "--data-contract",
        type=str,
        default=str(DEFAULT_DATA_CONTRACT),
        help="Frozen split/vocabulary contract shared by train and evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/runs",
        help="Root directory for per-experiment artifacts.",
    )
    parser.add_argument(
        "--accelerator",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Trainer/eval accelerator to use.",
    )
    parser.add_argument("--devices", type=int, default=1, help="Lightning device count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to use for training and evaluation.")
    parser.add_argument(
        "--representation-pretrain-epochs",
        type=int,
        default=20,
        help="Number of stage-1 epochs to run.",
    )
    parser.add_argument("--max-samples", type=int, default=2000, help="Max samples for downstream eval.")
    parser.add_argument("--retrieval-k", type=int, default=5, help="k for retrieval hit rate.")
    parser.add_argument(
        "--max-claims",
        type=int,
        default=4096,
        help="Maximum complete claims for Level-1 diagnostics.",
    )
    parser.add_argument(
        "--representation-source",
        type=str,
        default="patient_representation_pre_sae",
        help="Representation source to use for downstream evaluation.",
    )
    parser.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        default=None,
        help="Config override in key=value form. Repeat to set multiple fields.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse checkpoint and eval JSON if they already exist.",
    )
    return parser.parse_args(argv)


def build_train_command(
    name: str,
    recipe: str,
    data_path: str,
    checkpoint_path: Path,
    accelerator: str,
    devices: int,
    seed: int,
    representation_pretrain_epochs: int,
    config_overrides: list[str] | None = None,
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
        "--seed",
        str(seed),
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
    for override in config_overrides or []:
        command.extend(["--set", override])
    return command


def build_eval_command(
    recipe: str,
    data_path: str,
    checkpoint_path: Path,
    output_json: Path,
    accelerator: str,
    seed: int,
    max_samples: int,
    retrieval_k: int,
    representation_source: str,
    config_overrides: list[str] | None = None,
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
        "--seed",
        str(seed),
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
    for override in config_overrides or []:
        command.extend(["--set", override])
    return command


def build_claim_eval_command(
    recipe: str,
    data_path: str,
    checkpoint_path: Path,
    output_json: Path,
    accelerator: str,
    seed: int,
    config_overrides: list[str] | None = None,
    data_contract: str | None = None,
    max_claims: int = 4096,
):
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "evaluate_claim_representations.py"),
        "--checkpoint", str(checkpoint_path),
        "--data-path", data_path,
        "--recipe", recipe,
        "--accelerator", accelerator,
        "--seed", str(seed),
        "--max-claims", str(max_claims),
        "--output-json", str(output_json),
    ]
    if data_contract:
        command.extend(["--data-contract", data_contract, "--split", "val"])
    for override in config_overrides or []:
        command.extend(["--set", override])
    return command


def run_command(command):
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    experiment_dir = Path(args.output_dir) / args.name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = experiment_dir / "encoder.ckpt"
    eval_json = experiment_dir / "representation_eval.json"
    claim_eval_json = experiment_dir / "claim_representation_eval.json"
    spec_json = experiment_dir / "experiment_spec.json"
    result_json = experiment_dir / "experiment_result.json"

    spec = {
        "name": args.name,
        "recipe": args.recipe,
        "data_path": str(Path(args.data_path).resolve()),
        "data_contract": str(Path(args.data_contract).resolve()),
        "accelerator": args.accelerator,
        "devices": args.devices,
        "seed": args.seed,
        "representation_pretrain_epochs": args.representation_pretrain_epochs,
        "max_samples": args.max_samples,
        "retrieval_k": args.retrieval_k,
        "max_claims": args.max_claims,
        "representation_source": args.representation_source,
        "config_overrides": args.config_overrides or [],
        "checkpoint_path": str(checkpoint_path.resolve()),
        "eval_json": str(eval_json.resolve()),
        "claim_eval_json": str(claim_eval_json.resolve()),
    }
    write_json(spec_json, spec)

    if not (args.skip_existing and checkpoint_path.exists()):
        run_command(
            build_train_command(
                name=args.name,
                recipe=args.recipe,
                data_path=args.data_path,
                checkpoint_path=checkpoint_path,
                accelerator=args.accelerator,
                devices=args.devices,
                seed=args.seed,
                representation_pretrain_epochs=args.representation_pretrain_epochs,
                config_overrides=args.config_overrides,
                data_contract=args.data_contract,
            )
        )

    if not (args.skip_existing and eval_json.exists()):
        run_command(
            build_eval_command(
                recipe=args.recipe,
                data_path=args.data_path,
                checkpoint_path=checkpoint_path,
                output_json=eval_json,
                accelerator=args.accelerator,
                seed=args.seed,
                max_samples=args.max_samples,
                retrieval_k=args.retrieval_k,
                representation_source=args.representation_source,
                config_overrides=args.config_overrides,
                data_contract=args.data_contract,
                split="val",
            )
        )

    if args.recipe.startswith("composable_level1") and not (
        args.skip_existing and claim_eval_json.exists()
    ):
        run_command(
            build_claim_eval_command(
                recipe=args.recipe,
                data_path=args.data_path,
                checkpoint_path=checkpoint_path,
                output_json=claim_eval_json,
                accelerator=args.accelerator,
                seed=args.seed,
                config_overrides=args.config_overrides,
                data_contract=args.data_contract,
                max_claims=args.max_claims,
            )
        )

    eval_report = json.loads(eval_json.read_text(encoding="utf-8"))
    result = {
        "name": args.name,
        "recipe": args.recipe,
        "config_overrides": args.config_overrides or [],
        "artifact_dir": str(experiment_dir.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "eval_json": str(eval_json.resolve()),
        "seed": args.seed,
        "representation_source": args.representation_source,
    }
    result.update(eval_report)
    if claim_eval_json.exists():
        result["claim_level"] = json.loads(
            claim_eval_json.read_text(encoding="utf-8")
        )
    write_json(result_json, result)
    print(json.dumps(result, indent=2))
    print(f"Experiment directory: {experiment_dir.resolve()}")


if __name__ == "__main__":
    main()
