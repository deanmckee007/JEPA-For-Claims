# data_prep.py
import numpy as np
import torch
from pandas import read_parquet
from torch.utils.data import DataLoader, Subset, random_split
from jepa_utils.preprocessing import (
    filter_rows_with_min_ttnc_tokens,
    transform_target,
    tokenize_input,
    remove_trailing_time_token,
    filter_out_na_target
)
from data.vocab import create_vocab, calculate_rarity
from jepa_utils.dataset import ClaimsDataset
from jepa_utils.config import Config
from jepa_utils.data_contract import (
    attach_sample_ids,
    contract_vocab_state,
    load_or_create_data_contract,
)

def prepare_data(config, requested_eval_split=None):
    generator = torch.Generator().manual_seed(config.seed)

    # Load data
    pd_training_df = read_parquet(config.data_path)

    print('data prep, num rows before filter', pd_training_df.shape[0])
    # Optionally filter out NA targets
    if not config.use_na_targets:
        pd_training_df = filter_out_na_target(pd_training_df)
        
    # Filter rows with minimum TTNC tokens
    filtered_pd_training_df = filter_rows_with_min_ttnc_tokens(pd_training_df, config.min_ttnc_tokens).copy()
    print('data prep, num rows after filter', filtered_pd_training_df.shape[0])

    # Tokenize inputs
    filtered_pd_training_df['input'] = filtered_pd_training_df['input'].apply(tokenize_input)

    # Remove trailing TTNC tokens
    filtered_pd_training_df['input'] = filtered_pd_training_df['input'].apply(remove_trailing_time_token)

    filtered_pd_training_df = attach_sample_ids(filtered_pd_training_df)
    data_contract = None
    if config.data_contract_path:
        data_contract = load_or_create_data_contract(filtered_pd_training_df, config)
        assignments = data_contract["split_manifest"]["assignments"]
        filtered_pd_training_df["_split"] = filtered_pd_training_df["_sample_id"].map(assignments)
        (
            cpt_vocab,
            icd_vocab,
            ttnc_vocab,
            cpt_counter,
            icd_counter,
            ttnc_counter,
        ) = contract_vocab_state(data_contract)
        config.data_contract_hash = data_contract["contract_hash"]
        config.vocab_hash = data_contract["vocab_hash"]
    else:
        # Legacy mode remains available for old experiments, but all new
        # canonical runs should provide a frozen data contract.
        cpt_vocab, icd_vocab, ttnc_vocab, _, cpt_counter, icd_counter, ttnc_counter = create_vocab(
            filtered_pd_training_df,
            min_freq=config.vocab_min_freq,
        )

    # Debugging: Print vocab sizes and <UNK> token_ids
    print(f"CPT Vocab Size (including <UNK>): {len(cpt_vocab)}")
    print(f"CPT <UNK> token_id: {cpt_vocab.get('<UNK>')}")
    print(f"ICD Vocab Size (including <UNK>): {len(icd_vocab)}")
    print(f"ICD <UNK> token_id: {icd_vocab.get('<UNK>')}")
    print(f"TTNC Vocab Size (including <UNK>): {len(ttnc_vocab)}")
    print(f"TTNC <UNK> token_id: {ttnc_vocab.get('<UNK>')}")

    # Calculate separate rarity scores
    cpt_rarity, icd_rarity, ttnc_rarity = calculate_rarity(
        cpt_vocab, icd_vocab, ttnc_vocab, cpt_counter, icd_counter, ttnc_counter
    )

    # Debugging: Print max token_id in each rarity
    if cpt_rarity:
        print(f"Max CPT token_id: {max(cpt_rarity.keys())}")
    if icd_rarity:
        print(f"Max ICD token_id: {max(icd_rarity.keys())}")
    if ttnc_rarity:
        print(f"Max TTNC token_id: {max(ttnc_rarity.keys())}")

    # Transform target variable
    filtered_pd_training_df = transform_target(filtered_pd_training_df)

    # Initialize Dataset and DataLoader
    dataset = ClaimsDataset(
        filtered_pd_training_df,
        cpt_vocab=cpt_vocab,
        icd_vocab=icd_vocab,
        ttnc_vocab=ttnc_vocab,
        config=config
    )
    config.mean_target_baseline_rmse = dataset.mean_target_baseline_rmse

    # Split dataset into training and evaluation sets if needed
    if data_contract is not None:
        split_indices = {split_name: [] for split_name in ("train", "val", "test")}
        for dataset_index, split_name in enumerate(dataset.split_labels):
            if not dataset.sample_matches_evaluation_policy(
                dataset_index,
                config.min_valid_claims,
            ):
                continue
            split_indices[split_name].append(dataset_index)
        train_dataset = Subset(dataset, split_indices["train"])
        eval_split = requested_eval_split or config.evaluation_split
        if eval_split not in {"val", "test"}:
            raise ValueError("requested_eval_split must be either 'val' or 'test'.")
        eval_dataset = Subset(dataset, split_indices[eval_split])
        config.evaluation_split = eval_split
        train_targets = np.asarray(
            [dataset.targets[index] for index in split_indices["train"]],
            dtype=np.float64,
        )
        if train_targets.size:
            train_mean = train_targets.mean()
            config.mean_target_baseline_rmse = float(
                np.sqrt(np.mean((np.expm1(train_targets) - np.expm1(train_mean)) ** 2))
            )
    elif config.use_generative_save:
        # Calculate split lengths
        total_length = len(dataset)
        eval_length = int(total_length * 0.2)
        train_length = total_length - eval_length

        # Split the dataset
        train_dataset, eval_dataset = random_split(
            dataset,
            [train_length, eval_length],
            generator=generator,
        )
    else:
        train_dataset = dataset
        eval_dataset = None

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=True,
        generator=generator,
    )

    eval_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=config.eval_batch_size,
            collate_fn=dataset.collate_eval_fn,
            shuffle=False
        )

    # Update config with vocab sizes and steps
    config.cpt_vocab_size = len(cpt_vocab)
    config.icd_vocab_size = len(icd_vocab)
    config.ttnc_vocab_size = len(ttnc_vocab)
    config.steps_per_epoch = len(train_dataloader)

    # Initialize rarity score tensors for CPT
    cpt_rarity_scores = torch.ones(len(cpt_vocab))
    for token_str, token_id in cpt_vocab.items():
        if token_id < len(cpt_rarity_scores):
            cpt_rarity_scores[token_id] = cpt_rarity.get(token_id, 1.0)
        else:
            print(f"Warning: cpt_token_id {token_id} exceeds cpt_vocab_size {len(cpt_vocab)}")

    # Initialize rarity score tensors for ICD
    icd_rarity_scores = torch.ones(len(icd_vocab))
    for token_str, token_id in icd_vocab.items():
        if token_id < len(icd_rarity_scores):
            icd_rarity_scores[token_id] = icd_rarity.get(token_id, 1.0)
        else:
            print(f"Warning: icd_token_id {token_id} exceeds icd_vocab_size {len(icd_vocab)}")

    # Initialize rarity score tensors for TTNC
    ttnc_rarity_scores = torch.ones(len(ttnc_vocab))
    for token_str, token_id in ttnc_vocab.items():
        if token_id < len(ttnc_rarity_scores):
            ttnc_rarity_scores[token_id] = ttnc_rarity.get(token_id, 1.0)
        else:
            print(f"Warning: ttnc_token_id {token_id} exceeds ttnc_vocab_size {len(ttnc_vocab)}")

    # Update config with token rarity scores
    config.cpt_rarity_scores = cpt_rarity_scores
    config.icd_rarity_scores = icd_rarity_scores
    config.ttnc_rarity_scores = ttnc_rarity_scores

    config.cpt_id_to_token = {idx: token for token, idx in cpt_vocab.items()}
    config.icd_id_to_token = {idx: token for token, idx in icd_vocab.items()}
    config.ttnc_id_to_token = {idx: token for token, idx in ttnc_vocab.items()}

    return train_dataset, train_dataloader, eval_dataset, eval_dataloader, config, dataset
