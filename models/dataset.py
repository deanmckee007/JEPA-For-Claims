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

        # Process the sequences and filter patients with fewer than 2 valid claims
        self.processed_data = []
        self.targets = []

        for idx, row in dataframe.iterrows():
            claims = self.process_patient_sequence(row['input'])
            if len(claims) == 0:
                continue  # Skip patients with no valid claims

            self.processed_data.append(claims)
            self.targets.append(row['target'])

        # Print a debug statement for dataset size
        print(f"Number of processed samples after filtering: {len(self.processed_data)}")
        # Calculate RMSE of the error predicting mean log1p(target)
        self.calculate_rmse()

    def process_patient_sequence(self, sequence):
        claims = []
        current_claim = {'cpt': [], 'icd': [], 'ttnc': None}

        for token in sequence:
            if token.startswith('ttnc_'):
                # Check if the current claim has at least one CPT and one ICD before adding
                if current_claim['ttnc'] is not None and current_claim['cpt'] and current_claim['icd']:
                    claims.append(current_claim)
                # Start a new claim
                current_claim = {'cpt': [], 'icd': [], 'ttnc': token}
            elif token.startswith('cpt_'):
                current_claim['cpt'].append(token)
            elif token.startswith('icd_'):
                current_claim['icd'].append(token)

        # Add the last claim if it has at least one CPT and one ICD
        if current_claim['ttnc'] is not None and current_claim['cpt'] and current_claim['icd']:
            claims.append(current_claim)

        # Return claims only if they have at least one CPT and one ICD token
        valid_claims = [claim for claim in claims if claim['cpt'] and claim['icd']]
        
        # If no valid claims, return an empty list to indicate no valid sequence
        return valid_claims
    
    def calculate_rmse(self):
        if not self.targets:
            print("No valid targets to calculate RMSE.")
            return
        
        # Convert targets to a numpy array
        targets = np.array(self.targets)
        
        # Compute the mean of log1p targets
        mean_target = np.mean(targets)
        
        # Compute the squared errors
        squared_errors = (targets - mean_target) ** 2
        
        # Calculate the RMSE
        rmse = np.sqrt(np.mean(squared_errors))
        
        # Print the RMSE
        print(f"RMSE of the error predicting mean log1p(target): {rmse:.4f}")


    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        claims = self.processed_data[idx]
        target = self.targets[idx]  # Get the corresponding target
        return claims, target  # Return both the claims and target

    def collate_fn(self, batch):
        max_claims_len = self.max_claims_len
        max_cpt_tokens = self.max_cpt_tokens
        max_icd_tokens = self.max_icd_tokens

        cpt_lists = []
        icd_lists = []
        ttnc_list = []
        targets = []  # List to store the targets

        for claims, target in batch:  # Unpack the claims and target from the batch
            targets.append(target)  # Add the target for this patient

            cpt_tokens = []
            icd_tokens = []
            ttnc_tokens = []

            # Truncate the patient claims to max_claims_len
            for claim in claims[-max_claims_len:]:
                # Handle CPT tokens
                cpt_claim = claim.get('cpt', [])
                if len(cpt_claim) > max_cpt_tokens:
                    cpt_claim = random.sample(cpt_claim, max_cpt_tokens)
                cpt_tokens.append([self.cpt_vocab.get(token, self.cpt_vocab.get('<UNK>', 0)) for token in cpt_claim] + [self.cpt_vocab.get('<PAD>', 0)] * (max_cpt_tokens - len(cpt_claim)))

                # Handle ICD tokens
                icd_claim = claim.get('icd', [])
                if len(icd_claim) > max_icd_tokens:
                    icd_claim = random.sample(icd_claim, max_icd_tokens)
                icd_tokens.append([self.icd_vocab.get(token, self.icd_vocab.get('<UNK>', 0)) for token in icd_claim] + [self.icd_vocab.get('<PAD>', 0)] * (max_icd_tokens - len(icd_claim)))

                # Handle TTNC tokens
                ttnc_tokens.append(self.ttnc_vocab.get(claim['ttnc'], self.ttnc_vocab.get('<PAD>', 0)))

            # Pad claims to max_claims_len if they are shorter
            while len(cpt_tokens) < max_claims_len:
                cpt_tokens.append([self.cpt_vocab.get('<PAD>', 0)] * max_cpt_tokens)
            while len(icd_tokens) < max_claims_len:
                icd_tokens.append([self.icd_vocab.get('<PAD>', 0)] * max_icd_tokens)
            while len(ttnc_tokens) < max_claims_len:
                ttnc_tokens.append(self.ttnc_vocab.get('<PAD>', 0))

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

