import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "experiments" / "core_architecture_ablations_20260814"
DEFAULT_CONTRACT = REPO_ROOT / "artifacts" / "data_contracts" / "claims_seed42_v1.json"


COMMON_OVERRIDES = [
    "claim_inclusion_policy=any_code",
    "data_split_seed=42",
    "checkpoint_save_top_k=0",
    "checkpoint_save_last=false",
]


ANCHORS = [
    {"name": f"lejepa_anycode_l1w1_seed{seed}_e4", "seed": seed, "epochs": 4}
    for seed in (42, 43, 44)
] + [
    {
        "name": "lejepa_anycode_l1w0_seed42_e4",
        "seed": 42,
        "epochs": 4,
        "overrides": ["level1_predictive_weight=0.0"],
    },
    {
        "name": "lejepa_anycode_l1w01_seed42_e4",
        "seed": 42,
        "epochs": 4,
        "overrides": ["level1_predictive_weight=0.1"],
    },
]


SCREENS = [
    {"name": "screen_anchor_seed42_e2"},
    {
        "name": "screen_cap10_moments_seed42_e2",
        "overrides": ["max_cpt_tokens=10", "max_icd_tokens=10"],
    },
    {
        "name": "screen_cap10_query_seed42_e2",
        "overrides": [
            "max_cpt_tokens=10", "max_icd_tokens=10",
            "claim_pooling_type=query_attention",
        ],
    },
    {
        "name": "screen_cap10_query_rarity_seed42_e2",
        "overrides": [
            "max_cpt_tokens=10", "max_icd_tokens=10",
            "claim_pooling_type=query_attention",
            "claim_pooling_use_rarity=true",
        ],
    },
    {
        "name": "screen_cap10_self_attention_seed42_e2",
        "overrides": [
            "max_cpt_tokens=10", "max_icd_tokens=10",
            "claim_pooling_type=self_attention",
        ],
    },
    {
        "name": "screen_ttnc_shared_seed42_e2",
        "overrides": ["share_ttnc_embeddings=true"],
    },
    {
        "name": "screen_ttnc_composer_only_seed42_e2",
        "overrides": ["ttnc_in_sequence=false"],
    },
    {
        "name": "screen_ttnc_sequence_only_seed42_e2",
        "overrides": ["ttnc_in_composer=false"],
    },
    {
        "name": "screen_ttnc_ordinal_seed42_e2",
        "overrides": ["use_ttnc_ordinal_embedding=true"],
    },
    {
        "name": "screen_query_decoder_seed42_e2",
        "overrides": ["level2_decoder_type=cross_attention"],
    },
    {
        "name": "screen_query_masked_claim_seed42_e2",
        "overrides": [
            "level2_decoder_type=cross_attention",
            "use_masked_claim_jepa=true",
        ],
    },
    {
        "name": "screen_query_multihypothesis_seed42_e2",
        "overrides": [
            "level2_decoder_type=cross_attention",
            "use_multi_hypothesis_future=true",
            "multi_hypothesis_count=4",
        ],
    },
    {
        "name": "screen_query_masked_multihypothesis_seed42_e2",
        "overrides": [
            "level2_decoder_type=cross_attention",
            "use_masked_claim_jepa=true",
            "use_multi_hypothesis_future=true",
            "multi_hypothesis_count=4",
        ],
    },
    {
        "name": "screen_history100_seed42_e2",
        "overrides": ["max_claims_len=100"],
    },
]


PROMOTIONS = [
    {
        "name": f"promote_history100_l1w1_seed{seed}_e4",
        "seed": seed,
        "epochs": 4,
        "overrides": ["max_claims_len=100"],
    }
    for seed in (42, 43, 44)
] + [
    {
        "name": f"lejepa_anycode_l1w01_seed{seed}_e4",
        "seed": seed,
        "epochs": 4,
        "overrides": ["level1_predictive_weight=0.1"],
    }
    for seed in (43, 44)
] + [
    {
        "name": "promote_history100_l1w01_seed42_e4",
        "seed": 42,
        "epochs": 4,
        "overrides": ["max_claims_len=100", "level1_predictive_weight=0.1"],
    }
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["anchors", "screens", "promotions", "all"],
        default="all",
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--data-contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="gpu")
    return parser.parse_args(argv)


def get_path(payload, *path):
    current = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def summarize(output_dir, specs):
    rows = []
    for spec in specs:
        result_path = output_dir / spec["name"] / "experiment_result.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        row = {
            "name": spec["name"],
            "seed": spec.get("seed", 42),
            "epochs": spec.get("epochs", 2),
            "overrides": ";".join(spec.get("overrides", [])),
            "mae_dollars": result.get("target_probe_mae_dollars"),
            "rmse_dollars": result.get("target_probe_rmse_dollars"),
            "wape_percent": result.get("target_probe_wape_percent"),
            "retrieval_at_5": result.get("ttnc_proxy_retrieval_hit_rate_at_5"),
            "silhouette": result.get("cluster_silhouette"),
            "missing_cpt_rmse": get_path(
                result, "missing_modality", "missing_cpt", "target_probe_rmse_dollars"
            ),
            "missing_icd_rmse": get_path(
                result, "missing_modality", "missing_icd", "target_probe_rmse_dollars"
            ),
            "cpt_to_icd_hit5": get_path(
                result, "claim_level", "cpt_to_icd", "hit_rate_at_5"
            ),
            "icd_to_cpt_hit5": get_path(
                result, "claim_level", "icd_to_cpt", "hit_rate_at_5"
            ),
            "rare_cpt_to_icd_hit5": get_path(
                result, "claim_level", "strata", "rare_half", "cpt_to_icd", "hit_rate_at_5"
            ),
        }
        rows.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    if rows:
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def run_spec(args, output_dir, spec):
    epochs = spec.get("epochs", 2)
    max_samples = 3188 if epochs >= 4 else 2000
    max_claims = 4096 if epochs >= 4 else 2048
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_experiment.py"),
        "--name", spec["name"],
        "--recipe", "composable_level1_lejepa",
        "--data-path", args.data_path,
        "--data-contract", args.data_contract,
        "--output-dir", str(output_dir),
        "--accelerator", args.accelerator,
        "--seed", str(spec.get("seed", 42)),
        "--representation-pretrain-epochs", str(epochs),
        "--max-samples", str(max_samples),
        "--max-claims", str(max_claims),
        "--skip-existing",
    ]
    for override in COMMON_OVERRIDES + spec.get("overrides", []):
        command.extend(["--set", override])
    run_dir = output_dir / spec["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    print(f"START {spec['name']} ({epochs} epochs)", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        raise RuntimeError(f"{spec['name']} failed:\n" + "\n".join(tail))
    print(f"DONE  {spec['name']}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    specs = []
    if args.phase in {"anchors", "all"}:
        specs.extend(ANCHORS)
    if args.phase in {"screens", "all"}:
        specs.extend(SCREENS)
    if args.phase in {"promotions", "all"}:
        specs.extend(PROMOTIONS)
    failures = []
    for spec in specs:
        try:
            run_spec(args, output_dir, spec)
        except Exception as exc:
            failures.append({"name": spec["name"], "error": str(exc)})
            print(f"FAILED {spec['name']}: {exc}", flush=True)
        summarize(output_dir, ANCHORS + SCREENS + PROMOTIONS)
        (output_dir / "failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8"
        )
    rows = summarize(output_dir, ANCHORS + SCREENS + PROMOTIONS)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
