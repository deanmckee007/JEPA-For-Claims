"""Cross frozen encoders and decoder seeds with training-only set calibration."""
import argparse
import copy
import gc
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jepa_models.hierarchical_model import HierarchicalClaimsModel
from jepa_utils.checkpointing import read_checkpoint_config, load_claims_model_checkpoint
from jepa_utils.config import apply_runtime_config_overrides
from jepa_utils.data_prep import prepare_data
from jepa_utils.claim_set_calibration import (
    DEFAULT_THRESHOLDS, DEFAULT_COUNT_SCALES, shortlist_logits,
    fit_set_calibration, decode_calibrated_sets, score_decoded_sets, candidate_coverage,
)
from scripts.run_cost_attribution_ladder import parse_checkpoint_spec
from scripts.run_frozen_generation_probe import (
    collect_frozen_dataset, token_ids_to_multi_hot, resolve_device,
    fit_decoder, predict_decoder, persistence_prediction, score_generation, write_json,
)
from scripts.run_retrieval_generation_probe import (
    SOURCE, retrieve_neighbors, build_candidates, fit_candidate,
)


def grouped_inner_split(sample_ids, fraction, seed):
    unique = np.asarray(sorted(set(sample_ids)))
    if not 0 < fraction < 1 or len(unique) < 3:
        raise ValueError("Need a nonempty inner split and at least three sample groups")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    count = min(max(1, round(len(unique) * fraction)), len(unique) - 2)
    calibration_ids = set(unique[:count])
    mask = np.asarray([sample in calibration_ids for sample in sample_ids])
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def eligible_sample_ids(subset, loader):
    """Align frozen collector eligibility with the sequential subset rows."""
    result, offset = [], 0
    for _, _, ttnc, _ in loader:
        eligible = ttnc.ne(0).sum(1).ge(2).numpy()
        indices = subset.indices[offset:offset + len(eligible)]
        result.extend(subset.dataset.sample_ids[index] for index, keep in zip(indices, eligible) if keep)
        offset += len(eligible)
    return np.asarray(result)


def subset_data(data, positions):
    return {key: {source: values[positions] for source, values in value.items()}
        if key == "features" else value[positions] for key, value in data.items()}


def concatenate_data(left, right):
    return {key: {source: np.concatenate([value, right[key][source]])
        for source, value in left[key].items()} if key == "features"
        else np.concatenate([left[key], right[key]]) for key in left}


def filter_prediction(prediction, candidates):
    result = {key: value.copy() for key, value in prediction.items()}
    for key, ids in zip(["cpt", "icd"], candidates):
        logits = result[f"{key}_logits"]
        selected = np.take_along_axis(logits, ids, 1)
        logits[:] = -30.0
        np.put_along_axis(logits, ids, selected, 1)
        logits[:, 0] = -30.0
    return result


def evaluate_prediction(prediction, evaluation, cal_size, config, support, objective="micro_f1"):
    """Freeze calibration choices before accessing outer validation labels."""
    settings, decoded = {}, {}
    for column, key in enumerate(["cpt", "icd"]):
        maximum = getattr(config, f"max_{key}_tokens")
        shortlist = shortlist_logits(prediction[f"{key}_logits"][:cal_size],
            evaluation[f"previous_{key}_ids"][:cal_size], maximum)
        settings[key] = fit_set_calibration(shortlist,
            prediction["cardinality_fraction"][:cal_size, column] * maximum,
            evaluation[f"{key}_ids"][:cal_size], objective=objective,
            previous_ids=evaluation[f"previous_{key}_ids"][:cal_size])
        val_shortlist = shortlist_logits(prediction[f"{key}_logits"][cal_size:],
            evaluation[f"previous_{key}_ids"][cal_size:], maximum)
        decoded[key] = decode_calibrated_sets(val_shortlist,
            prediction["cardinality_fraction"][cal_size:, column] * maximum, settings[key])
    # No validation label has entered fitting, retrieval, or threshold selection.
    val = subset_data(evaluation, np.arange(cal_size, len(evaluation["ttnc"])))
    val_prediction = {key: values[cal_size:] for key, values in prediction.items()}
    targets = [token_ids_to_multi_hot(val[f"{key}_ids"], getattr(config, f"{key}_vocab_size"))
        for key in ["cpt", "icd"]]
    raw_metrics = score_generation(val_prediction, *targets, val["ttnc"],
        train_cpt_support=support[0], train_icd_support=support[1],
        max_cpt_tokens=config.max_cpt_tokens, max_icd_tokens=config.max_icd_tokens, min_class_support=10)
    calibrated = {key: score_decoded_sets(decoded[key], val[f"{key}_ids"], val[f"previous_{key}_ids"])
        for key in ["cpt", "icd"]}
    return {"uncalibrated": raw_metrics, "calibrated_sets": calibrated, "calibration": settings}


def aggregate_runs(runs):
    summary = {}
    for condition in runs[0]["conditions"]:
        summary[condition] = {}
        for stage in ["uncalibrated", "calibrated_sets"]:
            summary[condition][stage] = {}
            for key in ["cpt", "icd"]:
                metrics = runs[0]["conditions"][condition][stage][key]
                summary[condition][stage][key] = {}
                for metric in metrics:
                    values = [run["conditions"][condition][stage][key][metric] for run in runs]
                    summary[condition][stage][key][metric] = {"mean": float(np.mean(values)),
                        "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}
    paired = {}
    for comparator in ["copy_only", "flat_copy_residual", "flat_with_candidate_filter"]:
        paired[comparator] = {}
        for key in ["cpt", "icd"]:
            paired[comparator][key] = {}
            for stage, metric in [("uncalibrated", "micro_average_precision"), ("calibrated_sets", "micro_f1")]:
                deltas = [run["conditions"]["candidate"][stage][key][metric] -
                    run["conditions"][comparator][stage][key][metric] for run in runs]
                paired[comparator][key][metric] = {"mean_delta": float(np.mean(deltas)),
                    "wins": sum(delta > 0 for delta in deltas), "num_runs": len(deltas),
                    "deltas": deltas}
    return {"metrics": summary, "paired_candidate_deltas": paired,
        "note": "Crossed runs share patients and encoder states; SD is descriptive, not an independent-sample confidence interval."}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", dest="checkpoints", type=parse_checkpoint_spec, required=True)
    for flag in ["data-path", "data-contract", "output-dir"]:
        parser.add_argument(f"--{flag}", required=True)
    parser.add_argument("--decoder-seeds", nargs="+", type=int, default=[201, 202, 203])
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--calibration-seed", type=int, default=20260905)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--max-candidates", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if min(args.neighbors, args.max_candidates, args.epochs, args.batch_size, args.hidden_dim, args.threads) < 1:
        parser.error("Budgets, epochs, batch size, threads and hidden dimension must be positive")
    if not 0 < args.calibration_fraction < 1:
        parser.error("Calibration fraction must lie strictly between zero and one")
    if len({item['seed'] for item in args.checkpoints}) != len(args.checkpoints):
        parser.error("Provide one checkpoint per encoder seed")
    args.decoder_seeds = sorted(set(args.decoder_seeds))
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {**vars(args), "threshold_grid": list(DEFAULT_THRESHOLDS),
        "count_scales": list(DEFAULT_COUNT_SCALES), "calibration_metric": "per-modality micro F1",
        "test_accessed": False, "evaluation_weights": "online",
        "retrieval_bank": "inner fitting split only; no calibration or validation targets",
        "set_budget": "at most configured max tokens; separate copy/add logit thresholds plus optional count scaling",
        "difference_from_pilot": "80% of training groups fit the heads; 20% reserved for calibration",
        "ranking_metrics": "uncalibrated logits; calibrated sets reported separately",
        "code_coverage": "known IDs > 1; PAD and UNK excluded from new-code metrics"}
    protocol = json.loads(json.dumps(protocol))
    manifest = output / "protocol.json"
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise ValueError("Output contains a different protocol; use a new directory")
    write_json(manifest, protocol)  # Freeze the protocol before any outer scoring.
    device = resolve_device(args.accelerator)
    runs = []
    expected_ids = None
    for checkpoint in args.checkpoints:
        config = copy.deepcopy(read_checkpoint_config(checkpoint["path"]))
        config.data_path, config.data_contract_path = args.data_path, args.data_contract
        config.use_generative_save = config.use_plotting = config.pretrain_diffusion = False
        config = apply_runtime_config_overrides(config)
        train_subset, _, val_subset, _, config, dataset = prepare_data(config, requested_eval_split="val")
        loaders = [DataLoader(subset, batch_size=args.batch_size, shuffle=False,
            collate_fn=dataset.collate_eval_fn) for subset in [train_subset, val_subset]]
        sample_ids = eligible_sample_ids(train_subset, loaders[0])
        val_ids = eligible_sample_ids(val_subset, loaders[1])
        if expected_ids is not None and (not np.array_equal(expected_ids[0], sample_ids) or not np.array_equal(expected_ids[1], val_ids)):
            raise ValueError("Encoder checkpoints produced different evaluation cohorts")
        expected_ids = (sample_ids, val_ids)
        fit_positions, cal_positions = grouped_inner_split(sample_ids, args.calibration_fraction, args.calibration_seed)
        split_info = {"fit_positions": fit_positions.tolist(), "calibration_positions": cal_positions.tolist(),
            "ordered_sample_ids_sha256": hashlib.sha256('\n'.join(sample_ids).encode()).hexdigest(),
            "validation_ids_sha256": hashlib.sha256('\n'.join(val_ids).encode()).hexdigest(),
            "grouping": "frozen content-based sample IDs, not unavailable member IDs",
            "data_contract_hash": config.data_contract_hash, "vocab_hash": config.vocab_hash}
        write_json(output / "inner_split.json", split_info)
        print(f"Encoder {checkpoint['seed']}: extracting frozen histories", flush=True)
        pl.seed_everything(checkpoint["seed"], workers=True)
        model = load_claims_model_checkpoint(HierarchicalClaimsModel, checkpoint["path"], config=config, map_location=device)
        train_all, val = [collect_frozen_dataset(model, loader, [SOURCE], device) for loader in loaders]
        del model
        if len(train_all["ttnc"]) != len(sample_ids) or len(val["ttnc"]) != len(val_ids):
            raise ValueError("Feature/sample ID alignment failed")
        train = subset_data(train_all, fit_positions)
        calibration = subset_data(train_all, cal_positions)
        evaluation = concatenate_data(calibration, val)
        cal_size = len(calibration["ttnc"])
        del train_all, calibration, val
        nearest = [retrieve_neighbors(train["features"][SOURCE], data["features"][SOURCE], args.neighbors, training=is_train)
            for data, is_train in [(train, True), (evaluation, False)]]
        candidates = [[build_candidates(data[f"previous_{key}_ids"], train[f"{key}_ids"], neighbors, args.max_candidates)
            for key in ["cpt", "icd"]] for data, neighbors in zip([train, evaluation], nearest)]
        coverage = {key: candidate_coverage(ids[cal_size:], evaluation[f"{key}_ids"][cal_size:],
            evaluation[f"previous_{key}_ids"][cal_size:]) for key, ids in zip(["cpt", "icd"], candidates[1])}
        support = [token_ids_to_multi_hot(train[f"{key}_ids"], getattr(config, f"{key}_vocab_size")).sum(0) for key in ["cpt", "icd"]]
        # Copy-only is deterministic and therefore scored once per encoder.
        copy_result = evaluate_prediction(persistence_prediction(evaluation, config), evaluation, cal_size, config, support)
        for decoder_seed in args.decoder_seeds:
            run_dir = output / f"encoder{checkpoint['seed']}_decoder{decoder_seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"Encoder {checkpoint['seed']}, decoder {decoder_seed}: fit and calibrate", flush=True)
            run_args = SimpleNamespace(**vars(args), seed=decoder_seed, candidate_artifact_path=str(run_dir / 'candidate.pt'))
            candidate_prediction, candidate_history = fit_candidate(train, evaluation, *candidates, config, run_args, device)
            candidate_result = evaluate_prediction(candidate_prediction, evaluation, cal_size, config, support)
            del candidate_prediction
            flat, mean, std, flat_history = fit_decoder(train["features"][SOURCE], train["cpt_ids"], train["icd_ids"], train["ttnc"],
                train["previous_cpt_ids"], train["previous_icd_ids"], train["previous_ttnc"],
                cpt_vocab_size=config.cpt_vocab_size, icd_vocab_size=config.icd_vocab_size, ttnc_vocab_size=config.ttnc_vocab_size,
                max_cpt_tokens=config.max_cpt_tokens, max_icd_tokens=config.max_icd_tokens, hidden_dim=args.hidden_dim,
                epochs=args.epochs, batch_size=args.batch_size, lr=1e-3, weight_decay=1e-4, count_loss_weight=0.25,
                copy_residual=True, copy_logit_boost=4.0, copy_ttnc_logit_boost=2.0, seed=decoder_seed, device=device)
            torch.save({"state_dict": flat.state_dict(), "feature_mean": mean, "feature_std": std,
                "config": config, "args": vars(args)}, run_dir / "flat.pt")
            flat_prediction = predict_decoder(flat, evaluation["features"][SOURCE], mean, std, device,
                previous_cpt_ids=evaluation["previous_cpt_ids"], previous_icd_ids=evaluation["previous_icd_ids"],
                previous_ttnc=evaluation["previous_ttnc"], copy_residual=True)
            del flat
            flat_result = evaluate_prediction(flat_prediction, evaluation, cal_size, config, support)
            filtered = filter_prediction(flat_prediction, candidates[1])
            filtered_result = evaluate_prediction(filtered, evaluation, cal_size, config, support)
            del flat_prediction, filtered
            run = {"encoder_seed": checkpoint["seed"], "decoder_seed": decoder_seed,
                "fit_samples": len(train["ttnc"]), "calibration_samples": cal_size, "validation_samples": len(val_ids),
                "candidate_coverage": coverage, "conditions": {"candidate": candidate_result,
                    "flat_copy_residual": flat_result, "flat_with_candidate_filter": filtered_result, "copy_only": copy_result},
                "loss_history": {"candidate": candidate_history, "flat": flat_history}}
            write_json(run_dir / "summary.json", run)
            runs.append(run)
            write_json(output / "summary.json", {"protocol": protocol, "completed_runs": len(runs),
                "planned_runs": len(args.checkpoints) * len(args.decoder_seeds), "aggregate": aggregate_runs(runs), "runs": runs})
            print(f"Completed {len(runs)}: ICD candidate AP={candidate_result['uncalibrated']['icd']['micro_average_precision']:.4f}, calibrated F1={candidate_result['calibrated_sets']['icd']['micro_f1']:.4f}", flush=True)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del train, evaluation, candidates


if __name__ == "__main__":
    main()
