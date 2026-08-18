import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from jepa_utils.config import Config
from jepa_utils.data_contract import (
    attach_sample_ids,
    build_data_contract,
    load_or_create_data_contract,
    save_data_contract,
)
from jepa_utils.dataset import ClaimsDataset
from jepa_utils.representation_eval import extract_raw_sequence_lengths


def make_contract_dataframe(num_rows=30):
    return attach_sample_ids(
        pd.DataFrame(
            {
                "input": [
                    [
                        "ttnc_1",
                        f"cpt_{index}",
                        f"icd_{index}",
                        "ttnc_2",
                        f"cpt_{index}_b",
                        f"icd_{index}_b",
                    ]
                    for index in range(num_rows)
                ],
                "target": [float(index + 1) for index in range(num_rows)],
            }
        )
    )


class TestDataContract(unittest.TestCase):
    def test_contract_freezes_disjoint_splits_and_train_only_vocab(self):
        dataframe = make_contract_dataframe()
        contract = build_data_contract(
            dataframe,
            seed=42,
            train_fraction=0.6,
            val_fraction=0.2,
            test_fraction=0.2,
            vocab_min_freq=1,
        )

        assignments = contract["split_manifest"]["assignments"]
        train_ids = {sample_id for sample_id, split in assignments.items() if split == "train"}
        val_ids = {sample_id for sample_id, split in assignments.items() if split == "val"}
        test_ids = {sample_id for sample_id, split in assignments.items() if split == "test"}

        self.assertFalse(train_ids & val_ids)
        self.assertFalse(train_ids & test_ids)
        self.assertFalse(val_ids & test_ids)
        self.assertEqual(train_ids | val_ids | test_ids, set(dataframe["_sample_id"]))

        heldout_row = dataframe[dataframe["_sample_id"].isin(val_ids | test_ids)].iloc[0]
        heldout_cpt = next(token for token in heldout_row["input"] if token.startswith("cpt_"))
        self.assertNotIn(heldout_cpt, contract["vocabularies"]["cpt"])

    def test_contract_load_fails_closed_when_data_changes(self):
        dataframe = make_contract_dataframe()
        config = Config()
        config.seed = 7
        config.train_split_fraction = 0.6
        config.val_split_fraction = 0.2
        config.test_split_fraction = 0.2
        config.vocab_min_freq = 1

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "contract.json"
            contract = build_data_contract(
                dataframe,
                seed=config.seed,
                train_fraction=config.train_split_fraction,
                val_fraction=config.val_split_fraction,
                test_fraction=config.test_split_fraction,
                vocab_min_freq=config.vocab_min_freq,
            )
            save_data_contract(contract, path)
            config.data_contract_path = str(path)

            loaded = load_or_create_data_contract(dataframe, config)
            self.assertEqual(loaded["contract_hash"], contract["contract_hash"])

            changed = dataframe.copy()
            changed.loc[0, "target"] += 1.0
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_or_create_data_contract(changed, config)

    def test_training_seed_can_change_without_changing_frozen_split_seed(self):
        dataframe = make_contract_dataframe()
        contract = build_data_contract(
            dataframe,
            seed=42,
            train_fraction=0.6,
            val_fraction=0.2,
            test_fraction=0.2,
            vocab_min_freq=1,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "contract.json"
            save_data_contract(contract, path)
            config = Config(
                data_contract_path=str(path),
                seed=43,
                data_split_seed=42,
                train_split_fraction=0.6,
                val_split_fraction=0.2,
                test_split_fraction=0.2,
                vocab_min_freq=1,
            )

            loaded = load_or_create_data_contract(dataframe, config)

            self.assertEqual(loaded["split_seed"], 42)
            self.assertEqual(config.seed, 43)

    def test_explicit_wrong_split_seed_still_fails_closed(self):
        dataframe = make_contract_dataframe()
        contract = build_data_contract(
            dataframe,
            seed=42,
            train_fraction=0.6,
            val_fraction=0.2,
            test_fraction=0.2,
            vocab_min_freq=1,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "contract.json"
            save_data_contract(contract, path)
            config = Config(
                data_contract_path=str(path),
                seed=43,
                data_split_seed=41,
                train_split_fraction=0.6,
                val_split_fraction=0.2,
                test_split_fraction=0.2,
                vocab_min_freq=1,
            )

            with self.assertRaisesRegex(ValueError, "data_split_seed"):
                load_or_create_data_contract(dataframe, config)

    def test_eval_collation_is_deterministic(self):
        config = Config()
        config.min_valid_claims = 2
        config.max_claims_len = 2
        config.max_cpt_tokens = 2
        config.max_icd_tokens = 2
        dataframe = pd.DataFrame(
            {
                "input": [
                    [
                        "ttnc_1",
                        "cpt_3",
                        "cpt_1",
                        "cpt_2",
                        "icd_3",
                        "icd_1",
                        "icd_2",
                        "ttnc_2",
                        "cpt_6",
                        "cpt_4",
                        "cpt_5",
                        "icd_6",
                        "icd_4",
                        "icd_5",
                    ]
                ],
                "target": [1.0],
            }
        )
        cpt_vocab = {"<PAD>": 0, "<UNK>": 1, **{f"cpt_{i}": i + 1 for i in range(1, 7)}}
        icd_vocab = {"<PAD>": 0, "<UNK>": 1, **{f"icd_{i}": i + 1 for i in range(1, 7)}}
        ttnc_vocab = {"<PAD>": 0, "<UNK>": 1, "ttnc_1": 2, "ttnc_2": 3}
        dataset = ClaimsDataset(dataframe, cpt_vocab, icd_vocab, ttnc_vocab, config)

        first = dataset.collate_eval_fn([dataset[0]])
        second = dataset.collate_eval_fn([dataset[0]])

        for first_tensor, second_tensor in zip(first, second):
            self.assertTrue(torch.equal(first_tensor, second_tensor))
        self.assertEqual(first[0][0, 0].tolist(), [2, 3])
        self.assertEqual(first[1][0, 0].tolist(), [2, 3])

    def test_any_code_policy_retains_natural_single_modality_claims(self):
        dataframe = pd.DataFrame(
            {
                "input": [[
                    "ttnc_1", "icd_1",
                    "ttnc_2", "cpt_2", "icd_2",
                    "ttnc_3",
                ]],
                "target": [1.0],
            }
        )
        cpt_vocab = {"<PAD>": 0, "<UNK>": 1, "cpt_2": 2}
        icd_vocab = {"<PAD>": 0, "<UNK>": 1, "icd_1": 2, "icd_2": 3}
        ttnc_vocab = {
            "<PAD>": 0,
            "<UNK>": 1,
            "ttnc_1": 2,
            "ttnc_2": 3,
            "ttnc_3": 4,
        }

        complete_config = Config()
        complete_config.min_valid_claims = 1
        complete = ClaimsDataset(
            dataframe, cpt_vocab, icd_vocab, ttnc_vocab, complete_config
        )
        self.assertEqual(len(complete.processed_data[0]), 1)

        any_code_config = Config()
        any_code_config.min_valid_claims = 1
        any_code_config.claim_inclusion_policy = "any_code"
        any_code_config.evaluation_claim_inclusion_policy = "complete_only"
        any_code_config.max_claims_len = 2
        any_code = ClaimsDataset(
            dataframe, cpt_vocab, icd_vocab, ttnc_vocab, any_code_config
        )
        self.assertEqual(len(any_code.processed_data[0]), 2)
        self.assertEqual(any_code.processed_data[0][0]["cpt"], [])
        self.assertEqual(any_code.processed_data[0][0]["icd"], ["icd_1"])
        self.assertTrue(any_code.sample_matches_evaluation_policy(0, 1))
        self.assertFalse(any_code.sample_matches_evaluation_policy(0, 2))
        eval_batch = any_code.collate_eval_fn([any_code[0]])
        self.assertEqual(eval_batch[2][0].ne(0).sum().item(), 1)
        self.assertEqual(extract_raw_sequence_lengths(any_code).tolist(), [1])


if __name__ == "__main__":
    unittest.main()
