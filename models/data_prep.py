# data_prep.py
import torch
from pandas import read_parquet
from torch.utils.data import DataLoader
from utils.preprocessing import (
    filter_rows_with_min_ttnc_tokens,
    transform_target,
    tokenize_input,
    remove_trailing_time_token
)
from data.vocab import create_vocab, calculate_rarity
from models.dataset import ClaimsDataset
from utils.config import Config

def prepare_data(config):
    # Load data
    pd_training_df = read_parquet(config.data_path)

    print('data prep, num rows before filter', pd_training_df.shape[0])
    # Filter rows with minimum TTNC tokens
    filtered_pd_training_df = filter_rows_with_min_ttnc_tokens(pd_training_df, config.min_ttnc_tokens).copy()
    print('data prep, num rows after filter', filtered_pd_training_df.shape[0])

    # Tokenize inputs
    filtered_pd_training_df['input'] = filtered_pd_training_df['input'].apply(tokenize_input)

    # Remove trailing TTNC tokens
    filtered_pd_training_df['input'] = filtered_pd_training_df['input'].apply(remove_trailing_time_token)

    # Create vocabularies
    cpt_vocab, icd_vocab, ttnc_vocab, unified_vocab_reverse, cpt_counter, icd_counter, ttnc_counter = create_vocab(filtered_pd_training_df)

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
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        collate_fn=dataset.collate_fn,
        shuffle=True
    )

    # Update config with vocab sizes and steps
    config.cpt_vocab_size = len(cpt_vocab)
    config.icd_vocab_size = len(icd_vocab)
    config.ttnc_vocab_size = len(ttnc_vocab)
    config.steps_per_epoch = len(dataloader)

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

    return dataset, dataloader, config
