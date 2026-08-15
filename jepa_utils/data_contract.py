import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from data.vocab import create_vocab


DATA_CONTRACT_VERSION = 1
SPLIT_NAMES = ("train", "val", "test")


def configured_split_seed(config) -> int:
    split_seed = getattr(config, "data_split_seed", None)
    return int(config.seed if split_seed is None else split_seed)


def sample_id_from_tokens(tokens) -> str:
    canonical = " ".join(tokens)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def attach_sample_ids(dataframe):
    dataframe = dataframe.copy()
    dataframe["_sample_id"] = dataframe["input"].apply(sample_id_from_tokens)
    return dataframe


def _canonical_json(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_payload(payload) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def compute_data_fingerprint(dataframe) -> str:
    records = []
    for sample_id, target in zip(dataframe["_sample_id"], dataframe["target"]):
        target_text = "nan" if target != target else format(float(target), ".17g")
        records.append(f"{sample_id}:{target_text}")
    return hashlib.sha256("\n".join(sorted(records)).encode("utf-8")).hexdigest()


def _build_split_assignments(sample_ids, seed, train_fraction, val_fraction):
    unique_ids = sorted(set(sample_ids))
    random.Random(seed).shuffle(unique_ids)
    num_ids = len(unique_ids)
    num_train = int(round(num_ids * train_fraction))
    num_val = int(round(num_ids * val_fraction))
    num_train = min(max(num_train, 1), num_ids)
    num_val = min(max(num_val, 0), max(num_ids - num_train, 0))

    assignments = {}
    for index, sample_id in enumerate(unique_ids):
        if index < num_train:
            split_name = "train"
        elif index < num_train + num_val:
            split_name = "val"
        else:
            split_name = "test"
        assignments[sample_id] = split_name
    return assignments


def _serialize_counter(counter):
    return {token: int(count) for token, count in sorted(counter.items())}


def build_data_contract(
    dataframe,
    *,
    seed,
    train_fraction,
    val_fraction,
    test_fraction,
    vocab_min_freq,
):
    if "_sample_id" not in dataframe:
        raise ValueError("Dataframe must have _sample_id before building a data contract.")

    assignments = _build_split_assignments(
        dataframe["_sample_id"],
        seed,
        train_fraction,
        val_fraction,
    )
    split_series = dataframe["_sample_id"].map(assignments)
    train_dataframe = dataframe[split_series == "train"]
    (
        cpt_vocab,
        icd_vocab,
        ttnc_vocab,
        _,
        cpt_counter,
        icd_counter,
        ttnc_counter,
    ) = create_vocab(train_dataframe, min_freq=vocab_min_freq)

    vocabularies = {
        "cpt": cpt_vocab,
        "icd": icd_vocab,
        "ttnc": ttnc_vocab,
    }
    counters = {
        "cpt": _serialize_counter(cpt_counter),
        "icd": _serialize_counter(icd_counter),
        "ttnc": _serialize_counter(ttnc_counter),
    }
    vocab_hash = _sha256_payload({"vocabularies": vocabularies, "counters": counters})
    split_counts = {
        split_name: sum(value == split_name for value in assignments.values())
        for split_name in SPLIT_NAMES
    }

    contract = {
        "version": DATA_CONTRACT_VERSION,
        "data_fingerprint": compute_data_fingerprint(dataframe),
        "num_rows": int(len(dataframe)),
        "num_unique_samples": int(len(assignments)),
        "split_seed": int(seed),
        "split_fractions": {
            "train": float(train_fraction),
            "val": float(val_fraction),
            "test": float(test_fraction),
        },
        "split_manifest": {
            "assignments": assignments,
            "unique_sample_counts": split_counts,
        },
        "vocab_min_freq": int(vocab_min_freq),
        "vocabularies": vocabularies,
        "counters": counters,
        "vocab_hash": vocab_hash,
    }
    contract["contract_hash"] = _sha256_payload(contract)
    return contract


def validate_data_contract(contract, dataframe, config):
    if contract.get("version") != DATA_CONTRACT_VERSION:
        raise ValueError(
            f"Unsupported data contract version={contract.get('version')!r}; "
            f"expected {DATA_CONTRACT_VERSION}."
        )

    stored_hash = contract.get("contract_hash")
    hash_payload = dict(contract)
    hash_payload.pop("contract_hash", None)
    actual_hash = _sha256_payload(hash_payload)
    if stored_hash != actual_hash:
        raise ValueError("Data contract hash mismatch; artifact may be corrupted or edited.")

    actual_fingerprint = compute_data_fingerprint(dataframe)
    if contract.get("data_fingerprint") != actual_fingerprint:
        raise ValueError(
            "Data fingerprint does not match the frozen contract. "
            "Create a new explicitly named contract for the changed dataset."
        )

    expected_fractions = {
        "train": float(config.train_split_fraction),
        "val": float(config.val_split_fraction),
        "test": float(config.test_split_fraction),
    }
    if contract.get("split_fractions") != expected_fractions:
        raise ValueError("Configured split fractions do not match the frozen data contract.")
    expected_split_seed = configured_split_seed(config)
    if contract.get("split_seed") != expected_split_seed:
        raise ValueError(
            "Configured data_split_seed does not match the frozen data contract "
            "split seed. Training seed and split seed are independent when "
            "data_split_seed is explicitly set."
        )
    if contract.get("vocab_min_freq") != int(config.vocab_min_freq):
        raise ValueError("Configured vocab_min_freq does not match the frozen data contract.")

    assignments = contract.get("split_manifest", {}).get("assignments", {})
    sample_ids = set(dataframe["_sample_id"])
    if set(assignments) != sample_ids:
        raise ValueError("Frozen split manifest does not cover exactly the current sample IDs.")
    invalid_splits = set(assignments.values()).difference(SPLIT_NAMES)
    if invalid_splits:
        raise ValueError(f"Frozen split manifest contains invalid splits: {sorted(invalid_splits)}")

    for vocab_name, vocab in contract.get("vocabularies", {}).items():
        if vocab.get("<PAD>") != 0 or vocab.get("<UNK>") != 1:
            raise ValueError(f"{vocab_name} vocabulary has invalid reserved-token IDs.")
    return contract


def save_data_contract(contract, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def load_or_create_data_contract(dataframe, config):
    path = Path(config.data_contract_path)
    if path.exists():
        contract = json.loads(path.read_text(encoding="utf-8"))
    else:
        if not config.create_data_contract_if_missing:
            raise FileNotFoundError(
                f"Frozen data contract not found: {path}. "
                "Create it during an explicitly authorized training setup."
            )
        contract = build_data_contract(
            dataframe,
            seed=configured_split_seed(config),
            train_fraction=config.train_split_fraction,
            val_fraction=config.val_split_fraction,
            test_fraction=config.test_split_fraction,
            vocab_min_freq=config.vocab_min_freq,
        )
        save_data_contract(contract, path)

    return validate_data_contract(contract, dataframe, config)


def contract_vocab_state(contract):
    vocabularies = contract["vocabularies"]
    counters = contract["counters"]
    return (
        dict(vocabularies["cpt"]),
        dict(vocabularies["icd"]),
        dict(vocabularies["ttnc"]),
        Counter(counters["cpt"]),
        Counter(counters["icd"]),
        Counter(counters["ttnc"]),
    )
