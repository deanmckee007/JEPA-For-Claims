# models/dataset.py
import torch
from torch.utils.data import Dataset
import random
import numpy as np

class ClaimsDataset(Dataset):
    def __init__(self, dataframe, cpt_vocab, icd_vocab, ttnc_vocab, config):
        self.dataframe = dataframe
        self.cpt_vocab = cpt_vocab
        self.icd_vocab = icd_vocab
        self.ttnc_vocab = ttnc_vocab

        # Store config parameters
        self.max_claims_len = config.max_claims_len
        self.max_cpt_tokens = config.max_cpt_tokens
        self.max_icd_tokens = config.max_icd_tokens
        self.claim_inclusion_policy = getattr(
            config,
            'claim_inclusion_policy',
            'complete_only',
        )
        self.evaluation_claim_inclusion_policy = getattr(
            config,
            'evaluation_claim_inclusion_policy',
            None,
        )

        # Process the sequences and filter patients with fewer than 2 valid claims
        self.processed_data = []
        self.targets = []
        self.raw_sequence_lengths = []
        self.sample_ids = []
        self.split_labels = []
        self.mean_target_baseline_rmse = None

        for idx, row in dataframe.iterrows():
            claims = self.process_patient_sequence(row['input'])
            if len(claims) < config.min_valid_claims:
                continue  # Skip patients with insufficient valid claims
            self.processed_data.append(claims)
            self.targets.append(row['target'])
            self.raw_sequence_lengths.append(len(claims))
            self.sample_ids.append(row.get('_sample_id', str(idx)))
            self.split_labels.append(row.get('_split', 'unspecified'))

        # Print a debug statement for dataset size
        print(f"Number of processed samples after filtering: {len(self.processed_data)}")
        # Calculate RMSE of the error predicting mean log1p(target)
        self.mean_target_baseline_rmse = self.calculate_rmse()

    def process_patient_sequence(self, sequence):
        claims = []
        current_claim = {'cpt': [], 'icd': [], 'ttnc': None}

        def claim_is_eligible(claim):
            if claim['ttnc'] is None:
                return False
            if self.claim_inclusion_policy == 'complete_only':
                return bool(claim['cpt'] and claim['icd'])
            if self.claim_inclusion_policy == 'any_code':
                return bool(claim['cpt'] or claim['icd'])
            raise ValueError(
                f"Unsupported claim_inclusion_policy={self.claim_inclusion_policy!r}."
            )

        for token in sequence:
            if token.startswith('ttnc_'):
                if claim_is_eligible(current_claim):
                    claims.append(current_claim)
                # Start a new claim
                current_claim = {'cpt': [], 'icd': [], 'ttnc': token}
            elif token.startswith('cpt_'):
                current_claim['cpt'].append(token)
            elif token.startswith('icd_'):
                current_claim['icd'].append(token)

        if claim_is_eligible(current_claim):
            claims.append(current_claim)
        return claims

    @staticmethod
    def claim_matches_policy(claim, policy):
        if policy == 'complete_only':
            return bool(claim['cpt'] and claim['icd'])
        if policy == 'any_code':
            return bool(claim['cpt'] or claim['icd'])
        raise ValueError(f"Unsupported claim inclusion policy={policy!r}.")

    def sample_matches_evaluation_policy(self, index, min_valid_claims):
        policy = self.evaluation_claim_inclusion_policy
        if policy is None:
            return True
        matching_claims = sum(
            self.claim_matches_policy(claim, policy)
            for claim in self.processed_data[index]
        )
        return matching_claims >= min_valid_claims
    
    def calculate_rmse(self):
        if not self.targets:
            print("No valid targets to calculate RMSE.")
            self.mean_target_baseline_rmse = None
            return
        
        # Convert targets to a numpy array
        targets = np.array(self.targets)
        
        # Compute the mean of log1p targets
        mean_target = np.mean(targets)
        
        # Compute the squared errors
        squared_errors = (np.exp(targets) - np.exp(mean_target)) ** 2
        
        # Calculate the RMSE
        rmse = np.sqrt(np.mean(squared_errors))
        rmse = float(rmse)
        self.mean_target_baseline_rmse = rmse
        
        # Print the RMSE
        print(f"RMSE of the error predicting mean target(ln): {rmse:.4f}")
        return rmse


    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        claims = self.processed_data[idx]
        target = self.targets[idx]  # Get the corresponding target
        return claims, target  # Return both the claims and target

    def collate_fn(self, batch):
        return self._collate(batch, deterministic=False)

    def collate_eval_fn(self, batch):
        return self._collate(batch, deterministic=True)

    @staticmethod
    def _truncate_claim_tokens(tokens, max_tokens, deterministic):
        if len(tokens) <= max_tokens:
            return list(tokens)
        if deterministic:
            return sorted(tokens)[:max_tokens]
        return random.sample(tokens, max_tokens)

    def _collate(self, batch, deterministic):
        max_claims_len = self.max_claims_len
        max_cpt_tokens = self.max_cpt_tokens
        max_icd_tokens = self.max_icd_tokens

        cpt_lists = []
        icd_lists = []
        ttnc_list = []
        targets = []  # List to store the targets

        for claims, target in batch:  # Unpack the claims and target from the batch
            targets.append(target)  # Add the target for this patient

            if deterministic and self.evaluation_claim_inclusion_policy is not None:
                claims = [
                    claim
                    for claim in claims
                    if self.claim_matches_policy(
                        claim,
                        self.evaluation_claim_inclusion_policy,
                    )
                ]

            cpt_tokens = []
            icd_tokens = []
            ttnc_tokens = []

            # Truncate the patient claims to max_claims_len
            for claim in claims[-max_claims_len:]:
                # Handle CPT tokens
                cpt_claim = claim.get('cpt', [])
                cpt_claim = self._truncate_claim_tokens(
                    cpt_claim,
                    max_cpt_tokens,
                    deterministic,
                )
                cpt_tokens.append([self.cpt_vocab.get(token, self.cpt_vocab.get('<UNK>', 0)) for token in cpt_claim] + [self.cpt_vocab.get('<PAD>', 0)] * (max_cpt_tokens - len(cpt_claim)))

                # Handle ICD tokens
                icd_claim = claim.get('icd', [])
                icd_claim = self._truncate_claim_tokens(
                    icd_claim,
                    max_icd_tokens,
                    deterministic,
                )
                icd_tokens.append([self.icd_vocab.get(token, self.icd_vocab.get('<UNK>', 0)) for token in icd_claim] + [self.icd_vocab.get('<PAD>', 0)] * (max_icd_tokens - len(icd_claim)))

                # Handle TTNC tokens
                ttnc_tokens.append(self.ttnc_vocab.get(claim['ttnc'], self.ttnc_vocab.get('<PAD>', 0)))

            # Pad claims to max_claims_len if they are shorter
            num_padding = max_claims_len - len(cpt_tokens)
            if num_padding > 0:
                pad_cpt = [[self.cpt_vocab.get('<PAD>', 0)] * max_cpt_tokens] * num_padding
                pad_icd = [[self.icd_vocab.get('<PAD>', 0)] * max_icd_tokens] * num_padding
                pad_ttnc = [self.ttnc_vocab.get('<PAD>', 0)] * num_padding

                # Pad at the beginning
                cpt_tokens = pad_cpt + cpt_tokens
                icd_tokens = pad_icd + icd_tokens
                ttnc_tokens = pad_ttnc + ttnc_tokens

            cpt_lists.append(cpt_tokens)
            icd_lists.append(icd_tokens)
            ttnc_list.append(ttnc_tokens)

        # Convert claims to tensors
        cpt_tensor = torch.tensor(cpt_lists, dtype=torch.long)
        icd_tensor = torch.tensor(icd_lists, dtype=torch.long)
        ttnc_tensor = torch.tensor(ttnc_list, dtype=torch.long)

        # Convert targets to tensor (no padding needed for targets)
        target_tensor = torch.tensor(targets, dtype=torch.float32)

        return cpt_tensor, icd_tensor, ttnc_tensor, target_tensor

