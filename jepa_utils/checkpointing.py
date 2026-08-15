import torch


ARCHITECTURE_CONFIG_KEYS = (
    "embedding_dim",
    "output_dim",
    "hidden_dim",
    "max_claims_len",
    "max_cpt_tokens",
    "max_icd_tokens",
    "num_layers",
    "rnn_type",
    "use_context_pooled_patient_representation",
    "use_sparse_autoencoder",
    "use_gated_fusion",
    "use_level1",
    "use_composable_level1",
    "claim_pooling_type",
    "claim_pooling_num_heads",
    "claim_pooling_use_rarity",
    "sigreg_formulation",
    "level1_marginal_regularizer",
    "level1_marginal_max_samples",
    "claim_inclusion_policy",
    "context_cpt_dropout_prob",
    "context_icd_dropout_prob",
    "use_level2_dense_prediction",
    "observed_claim_k",
    "future_claim_k",
    "level2_decoder_type",
    "ttnc_in_composer",
    "ttnc_in_sequence",
    "share_ttnc_embeddings",
    "use_ttnc_ordinal_embedding",
    "use_masked_claim_jepa",
    "use_multi_hypothesis_future",
    "multi_hypothesis_count",
    "use_bifurcated_patient_state",
    "use_predictive_state_bottleneck",
    "predictive_state_bottleneck_dim",
    "use_dense_decoder_bottleneck",
    "dense_decoder_bottleneck_dim",
    "use_world_model_dynamics",
    "use_temporal_contrastive",
    "use_masked_next_claim_token_grounding",
    "use_claim_prototypes",
    "claim_prototype_count",
    "use_predictor_head",
    "use_token_prediction_head",
    "use_diffusion",
    "cpt_vocab_size",
    "icd_vocab_size",
    "ttnc_vocab_size",
)

STAGE_TRANSITION_ARCHITECTURE_KEYS = frozenset(
    {"use_token_prediction_head", "use_diffusion"}
)
STAGE_TRANSITION_NEW_STATE_PREFIXES = (
    "logits_generator.",
    "diffusion_model.",
)


def _config_value(config, key):
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


def read_checkpoint_config(checkpoint_path, map_location="cpu"):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    return hyper_parameters.get("config")


def validate_checkpoint_compatibility(
    checkpoint_path,
    config,
    *,
    allow_legacy=False,
    allowed_architecture_mismatches=(),
    map_location="cpu",
):
    saved_config = read_checkpoint_config(checkpoint_path, map_location=map_location)
    if saved_config is None:
        raise ValueError("Checkpoint does not contain a saved Config object.")

    saved_contract_hash = _config_value(saved_config, "data_contract_hash")
    current_contract_hash = _config_value(config, "data_contract_hash")
    saved_vocab_hash = _config_value(saved_config, "vocab_hash")
    current_vocab_hash = _config_value(config, "vocab_hash")

    if not saved_contract_hash or not saved_vocab_hash:
        if not allow_legacy:
            raise ValueError(
                "Checkpoint predates frozen data/vocabulary contracts. "
                "Legacy loading must be explicitly enabled and cannot certify token identity."
            )
    else:
        if not current_contract_hash or not current_vocab_hash:
            raise ValueError(
                "Current evaluation config has no loaded data contract. "
                "Prepare data with the checkpoint's frozen contract before loading."
            )
        if saved_contract_hash != current_contract_hash:
            raise ValueError("Checkpoint data_contract_hash does not match evaluation data.")
        if saved_vocab_hash != current_vocab_hash:
            raise ValueError("Checkpoint vocab_hash does not match evaluation vocabulary.")

    allowed_architecture_mismatches = set(allowed_architecture_mismatches)
    unsupported_allowed_keys = allowed_architecture_mismatches.difference(
        STAGE_TRANSITION_ARCHITECTURE_KEYS
    )
    if unsupported_allowed_keys:
        raise ValueError(
            "Unsupported checkpoint architecture mismatch exemptions: "
            f"{sorted(unsupported_allowed_keys)}"
        )

    mismatches = []
    for key in ARCHITECTURE_CONFIG_KEYS:
        saved_value = _config_value(saved_config, key)
        current_value = _config_value(config, key)
        if saved_value != current_value and key not in allowed_architecture_mismatches:
            mismatches.append((key, saved_value, current_value))
    if mismatches:
        mismatch_text = ", ".join(
            f"{key}: checkpoint={saved!r}, current={current!r}"
            for key, saved, current in mismatches
        )
        raise ValueError(f"Checkpoint architecture/config mismatch: {mismatch_text}")
    return saved_config


def load_claims_model_checkpoint(
    model_class,
    checkpoint_path,
    *,
    config,
    map_location="cpu",
    allow_legacy=False,
    stage_transition=False,
):
    allowed_mismatches = (
        STAGE_TRANSITION_ARCHITECTURE_KEYS if stage_transition else ()
    )
    validate_checkpoint_compatibility(
        checkpoint_path,
        config,
        allow_legacy=allow_legacy,
        allowed_architecture_mismatches=allowed_mismatches,
        map_location=map_location,
    )
    if stage_transition:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
        if "state_dict" not in checkpoint:
            raise ValueError("Checkpoint does not contain a model state_dict.")

        model = model_class(config)
        incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(
                "Stage-transition checkpoint has unexpected model state: "
                f"{sorted(incompatible.unexpected_keys)}"
            )
        disallowed_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(STAGE_TRANSITION_NEW_STATE_PREFIXES)
        ]
        if disallowed_missing:
            raise ValueError(
                "Stage-transition checkpoint is missing non-generator model state: "
                f"{sorted(disallowed_missing)}"
            )
        if hasattr(model, "synchronize_generator_embeddings_from_active_encoder"):
            model.synchronize_generator_embeddings_from_active_encoder()
        return model

    return model_class.load_from_checkpoint(
        checkpoint_path,
        config=config,
        strict=True,
        map_location=map_location,
        weights_only=False,
    )
