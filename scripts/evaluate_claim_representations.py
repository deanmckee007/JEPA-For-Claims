import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jepa_models.hierarchical_model import HierarchicalClaimsModel
from jepa_utils.checkpointing import load_claims_model_checkpoint
from jepa_utils.data_prep import prepare_data
from scripts.evaluate_representations import build_eval_config, resolve_device


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate Level-1 CPT/ICD claim representations."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--data-contract", required=True)
    parser.add_argument("--recipe", default="custom")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--allow-legacy-checkpoint", action="store_true")
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-claims", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--set", dest="config_overrides", action="append", default=None)
    return parser.parse_args(argv)


def _retrieval_metrics(query, target, ks=(1, 5), chunk_size=512):
    query = F.normalize(query, dim=-1)
    target = F.normalize(target, dim=-1)
    hits = {k: 0 for k in ks}
    ranks = []
    for start in range(0, query.size(0), chunk_size):
        stop = min(start + chunk_size, query.size(0))
        similarity = query[start:stop] @ target.T
        paired = similarity[
            torch.arange(stop - start, device=query.device),
            torch.arange(start, stop, device=query.device),
        ]
        rank = (similarity > paired.unsqueeze(1)).sum(dim=1) + 1
        ranks.append(rank.cpu())
        for k in ks:
            hits[k] += int((rank <= k).sum().item())
    ranks = torch.cat(ranks).float()
    result = {f"hit_rate_at_{k}": hits[k] / max(query.size(0), 1) for k in ks}
    result["mean_reciprocal_rank"] = float((1.0 / ranks).mean().item())
    result["median_rank"] = float(ranks.median().item())
    result["paired_cosine"] = float((query * target).sum(dim=1).mean().item())
    return result


def _subset_metrics(query, target, mask):
    if int(mask.sum()) < 32:
        return {"num_claims": int(mask.sum())}
    result = _retrieval_metrics(query[mask], target[mask])
    result["num_claims"] = int(mask.sum())
    return result


@torch.no_grad()
def collect_claim_states(model, dataloader, config, device, max_claims):
    collected = {
        "cpt_prediction": [],
        "icd_target": [],
        "icd_prediction": [],
        "cpt_target": [],
        "cpt_count": [],
        "icd_count": [],
        "rarity": [],
        "full_claim": [],
        "missing_cpt": [],
        "missing_icd": [],
    }
    model.eval().to(device)
    remaining = max_claims
    for cpt, icd, ttnc, _ in dataloader:
        if remaining <= 0:
            break
        cpt = cpt.to(device)
        icd = icd.to(device)
        ttnc = ttnc.to(device)
        cpt_state, cpt_mask = model.context_encoder_lvl1(cpt, "cpt")
        icd_state, icd_mask = model.context_encoder_lvl1(icd, "icd")
        complete = cpt_mask & icd_mask & ttnc.ne(0)
        indices = complete.nonzero(as_tuple=False)
        if indices.numel() == 0:
            continue
        indices = indices[:remaining]
        batch_index, claim_index = indices[:, 0], indices[:, 1]
        cpt_prediction = model.prediction_block_lvl1(cpt_state, cpt_mask)
        icd_prediction = model.prediction_block_lvl1(icd_state, icd_mask)
        full = model.encode_claims(cpt, icd, ttnc)
        missing_cpt = model.encode_claims(torch.zeros_like(cpt), icd, ttnc)
        missing_icd = model.encode_claims(cpt, torch.zeros_like(icd), ttnc)

        cpt_tokens = cpt[batch_index, claim_index]
        icd_tokens = icd[batch_index, claim_index]
        cpt_rarity = config.cpt_rarity_scores.to(device)[cpt_tokens]
        icd_rarity = config.icd_rarity_scores.to(device)[icd_tokens]
        cpt_valid = cpt_tokens.ne(0)
        icd_valid = icd_tokens.ne(0)
        rarity = (
            (cpt_rarity * cpt_valid).sum(1) + (icd_rarity * icd_valid).sum(1)
        ) / (cpt_valid.sum(1) + icd_valid.sum(1)).clamp(min=1)

        for key, value in (
            ("cpt_prediction", cpt_prediction[batch_index, claim_index]),
            ("icd_target", icd_state[batch_index, claim_index]),
            ("icd_prediction", icd_prediction[batch_index, claim_index]),
            ("cpt_target", cpt_state[batch_index, claim_index]),
            ("cpt_count", cpt_valid.sum(1)),
            ("icd_count", icd_valid.sum(1)),
            ("rarity", rarity),
            ("full_claim", full[batch_index, claim_index]),
            ("missing_cpt", missing_cpt[batch_index, claim_index]),
            ("missing_icd", missing_icd[batch_index, claim_index]),
        ):
            collected[key].append(value.detach().cpu())
        remaining -= indices.size(0)
    return {key: torch.cat(value) for key, value in collected.items()}


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.accelerator)
    config = build_eval_config(args)
    _, _, eval_dataset, _, config, dataset = prepare_data(
        config, requested_eval_split=args.split
    )
    dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_eval_fn,
        shuffle=False,
    )
    model = load_claims_model_checkpoint(
        HierarchicalClaimsModel,
        args.checkpoint,
        config=config,
        map_location=device,
        allow_legacy=config.allow_legacy_checkpoint_loading,
    )
    if not model.use_composable_level1:
        raise ValueError("Claim-level evaluation requires the composable Level-1 path.")
    states = collect_claim_states(model, dataloader, config, device, args.max_claims)
    cpt_prediction = states["cpt_prediction"]
    icd_target = states["icd_target"]
    icd_prediction = states["icd_prediction"]
    cpt_target = states["cpt_target"]
    rarity_median = states["rarity"].median()
    total_codes = states["cpt_count"] + states["icd_count"]

    report = {
        "num_claims": int(cpt_prediction.size(0)),
        "cpt_to_icd": _retrieval_metrics(cpt_prediction, icd_target),
        "icd_to_cpt": _retrieval_metrics(icd_prediction, cpt_target),
        "strata": {
            "singleton_or_pair": {
                "cpt_to_icd": _subset_metrics(cpt_prediction, icd_target, total_codes <= 2),
                "icd_to_cpt": _subset_metrics(icd_prediction, cpt_target, total_codes <= 2),
            },
            "multi_code": {
                "cpt_to_icd": _subset_metrics(cpt_prediction, icd_target, total_codes > 2),
                "icd_to_cpt": _subset_metrics(icd_prediction, cpt_target, total_codes > 2),
            },
            "common_half": {
                "cpt_to_icd": _subset_metrics(cpt_prediction, icd_target, states["rarity"] <= rarity_median),
                "icd_to_cpt": _subset_metrics(icd_prediction, cpt_target, states["rarity"] <= rarity_median),
            },
            "rare_half": {
                "cpt_to_icd": _subset_metrics(cpt_prediction, icd_target, states["rarity"] > rarity_median),
                "icd_to_cpt": _subset_metrics(icd_prediction, cpt_target, states["rarity"] > rarity_median),
            },
        },
        "missing_modality": {
            "full_vs_missing_cpt_cosine": float(F.cosine_similarity(states["full_claim"], states["missing_cpt"]).mean()),
            "full_vs_missing_icd_cosine": float(F.cosine_similarity(states["full_claim"], states["missing_icd"]).mean()),
        },
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "evaluation_split": args.split,
        "data_contract_hash": config.data_contract_hash,
        "vocab_hash": config.vocab_hash,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
