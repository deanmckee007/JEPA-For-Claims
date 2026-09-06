"""Isolate new-code retrieval coverage and additions-aware calibration."""
import argparse
import copy
import gc
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jepa_models.hierarchical_model import HierarchicalClaimsModel
from jepa_utils.checkpointing import load_claims_model_checkpoint, read_checkpoint_config
from jepa_utils.config import apply_runtime_config_overrides
from jepa_utils.data_prep import prepare_data
from jepa_utils.claim_set_calibration import candidate_coverage
from scripts.run_frozen_generation_probe import (
    collect_frozen_dataset, token_ids_to_multi_hot, resolve_device,
    FrozenClaimDecoder, predict_decoder, persistence_prediction, write_json,
)
from scripts.run_retrieval_generation_probe import (
    SOURCE, retrieve_neighbors, build_candidates, CandidateDecoder, predict_candidate, fit_candidate,
)
from scripts.run_retrieval_replication import (
    grouped_inner_split, eligible_sample_ids, subset_data, concatenate_data,
    evaluate_prediction, filter_prediction,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ["checkpoint", "data-path", "data-contract", "replication-dir", "output-dir"]:
        parser.add_argument(f"--{flag}", required=True)
    parser.add_argument("--encoder-seed", type=int, default=42)
    parser.add_argument("--decoder-seed", type=int, default=201)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    replication = Path(args.replication_dir)
    source_protocol = json.loads((replication / "protocol.json").read_text())
    if any(source_protocol[key] != expected for key, expected in
        [('epochs', 20), ('hidden_dim', 256), ('neighbors', 16), ('max_candidates', 128)]):
        raise ValueError("This matched probe requires the original 20-epoch 16/128 replication budget")
    if not any(item['seed'] == args.encoder_seed and Path(item['path']).resolve() == Path(args.checkpoint).resolve()
        for item in source_protocol['checkpoints']):
        raise ValueError("Checkpoint does not match the replication encoder")
    protocol = {**vars(args), "test_accessed": False,
        "calibration_objectives": ["micro_f1", "balanced_new_f1"],
        "balanced_objective": "0.5 * total micro F1 + 0.5 * known-new-code micro F1",
        "reference_bank": "same inner 80% fitting split as replication",
        "retrieval_options": [{"neighbors": 16, "max_candidates": 128}, {"neighbors": 64, "max_candidates": 256}],
        "expansion_gate": "Train expanded decoder only if mean CPT/ICD inner new-code recall increases by at least 0.05",
        "decoder_epochs": 20, "hidden_dim": 256, "validation_used_for_selection": False}
    manifest = output / "protocol.json"
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise ValueError("Different experiment exists; choose a new output directory")
    write_json(manifest, protocol)
    device = resolve_device(args.accelerator)
    config = copy.deepcopy(read_checkpoint_config(args.checkpoint))
    config.data_path, config.data_contract_path = args.data_path, args.data_contract
    config.use_generative_save = config.use_plotting = config.pretrain_diffusion = False
    config = apply_runtime_config_overrides(config)
    train_subset, _, val_subset, _, config, dataset = prepare_data(config, requested_eval_split="val")
    loaders = [DataLoader(subset, batch_size=256, shuffle=False, collate_fn=dataset.collate_eval_fn)
        for subset in [train_subset, val_subset]]
    ids = eligible_sample_ids(train_subset, loaders[0])
    fit_positions, cal_positions = grouped_inner_split(ids, source_protocol['calibration_fraction'], source_protocol['calibration_seed'])
    original_split = json.loads((replication / 'inner_split.json').read_text())
    if hashlib.sha256('\n'.join(ids).encode()).hexdigest() != original_split['ordered_sample_ids_sha256']:
        raise ValueError("Training row identities/order do not match the saved replication")
    if fit_positions.tolist() != original_split['fit_positions'] or cal_positions.tolist() != original_split['calibration_positions']:
        raise ValueError("Inner split does not match the saved replication")
    model = load_claims_model_checkpoint(HierarchicalClaimsModel, args.checkpoint, config=config, map_location=device)
    train_all, val = [collect_frozen_dataset(model, loader, [SOURCE], device) for loader in loaders]
    del model
    train = subset_data(train_all, fit_positions)
    calibration = subset_data(train_all, cal_positions)
    evaluation = concatenate_data(calibration, val)
    cal_size = len(calibration['ttnc'])
    del train_all, calibration, val
    candidates, inner_coverage = {}, {}
    for option in protocol['retrieval_options']:
        key = str(option['neighbors'])
        nearest = [retrieve_neighbors(train['features'][SOURCE], data['features'][SOURCE], option['neighbors'], training=is_train)
            for data, is_train in [(train, True), (evaluation, False)]]
        candidates[key] = [[build_candidates(data[f'previous_{modality}_ids'], train[f'{modality}_ids'], neighbor, option['max_candidates'])
            for modality in ['cpt', 'icd']] for data, neighbor in zip([train, evaluation], nearest)]
        inner_coverage[key] = {modality: candidate_coverage(candidate[:cal_size], evaluation[f'{modality}_ids'][:cal_size],
            evaluation[f'previous_{modality}_ids'][:cal_size]) for modality, candidate in zip(['cpt', 'icd'], candidates[key][1])}
    gain = float(np.mean([inner_coverage['64'][key]['new']['recall'] - inner_coverage['16'][key]['new']['recall'] for key in ['cpt', 'icd']]))
    expanded = gain >= 0.05
    write_json(output / 'inner_retrieval_gate.json', {'inner_coverage': inner_coverage,
        'new_recall_gain': gain, 'expanded_decoder_selected': expanded})
    print(f"Inner new-code recall gain={gain:.4f}; expanded decoder selected={expanded}", flush=True)
    # Selection is complete before any outer candidate-coverage scoring.
    outer_coverage = {name: {key: candidate_coverage(candidate[cal_size:], evaluation[f'{key}_ids'][cal_size:],
        evaluation[f'previous_{key}_ids'][cal_size:]) for key, candidate in zip(['cpt', 'icd'], value[1])}
        for name, value in candidates.items()}
    sizes = [config.cpt_vocab_size, config.icd_vocab_size, config.ttnc_vocab_size]
    support = [token_ids_to_multi_hot(train[f'{key}_ids'], size).sum(0) for key, size in zip(['cpt', 'icd'], sizes)]
    results = {}

    def score(name, prediction):
        results[name] = {objective: evaluate_prediction(prediction, evaluation, cal_size, config, support, objective)
            for objective in protocol['calibration_objectives']}
        write_json(output / 'summary.json', {'protocol': protocol, 'inner_coverage': inner_coverage,
            'new_recall_gain': gain, 'expanded_decoder_selected': expanded,
            'validation_coverage': outer_coverage, 'conditions': results, 'completed': False})
        print(f"Scored {name}", flush=True)

    saved = replication / f'encoder{args.encoder_seed}_decoder{args.decoder_seed}'
    candidate_state = torch.load(saved / 'candidate.pt', map_location=device, weights_only=False)
    head = CandidateDecoder(candidate_state['input_dim'], candidate_state['hidden_dim'], candidate_state['vocab_sizes']).to(device)
    head.load_state_dict(candidate_state['state_dict'])
    prediction = predict_candidate(head, evaluation, candidates['16'][1], candidate_state['feature_mean'], candidate_state['feature_std'], sizes, 256, device)
    score('candidate16', prediction)
    del prediction, head, candidate_state
    flat_state = torch.load(saved / 'flat.pt', map_location=device, weights_only=False)
    flat = FrozenClaimDecoder(train['features'][SOURCE].shape[1], 256, *sizes).to(device)
    flat.load_state_dict(flat_state['state_dict'])
    prediction = predict_decoder(flat, evaluation['features'][SOURCE], flat_state['feature_mean'], flat_state['feature_std'], device,
        previous_cpt_ids=evaluation['previous_cpt_ids'], previous_icd_ids=evaluation['previous_icd_ids'],
        previous_ttnc=evaluation['previous_ttnc'], copy_residual=True)
    score('flat', prediction)
    score('flat_filter16', filter_prediction(prediction, candidates['16'][1]))
    score('flat_filter64', filter_prediction(prediction, candidates['64'][1]))
    del prediction, flat, flat_state
    score('copy_only', persistence_prediction(evaluation, config))
    if expanded:
        run_args = SimpleNamespace(seed=args.decoder_seed, hidden_dim=256, epochs=20, batch_size=256,
            candidate_artifact_path=str(output / 'candidate64.pt'))
        prediction, history = fit_candidate(train, evaluation, *candidates['64'], config, run_args, device)
        score('candidate64', prediction)
        write_json(output / 'expanded_loss_history.json', history)
        del prediction
    gc.collect()
    write_json(output / 'summary.json', {'protocol': protocol, 'inner_coverage': inner_coverage,
        'new_recall_gain': gain, 'expanded_decoder_selected': expanded,
        'validation_coverage': outer_coverage, 'conditions': results, 'completed': True})


if __name__ == '__main__':
    main()
