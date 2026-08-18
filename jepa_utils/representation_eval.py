import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Subset


TTNC_PROXY_LABEL_SOURCE = "last_valid_ttnc"
REPRESENTATION_SOURCES = (
    "patient_representation",
    "patient_representation_pre_sae",
    "context_mean_pool",
    "context_max_pool",
    "context_pooled",
    "next_claim_prediction",
    "dense_decoder_latent",
)


def extract_last_valid_ttnc(ttnc_tensor: torch.Tensor, padding_idx: int = 0) -> torch.Tensor:
    valid_mask = ttnc_tensor != padding_idx
    valid_counts = valid_mask.sum(dim=1)
    reversed_last = torch.flip(valid_mask, dims=[1]).float().argmax(dim=1)
    last_indices = ttnc_tensor.size(1) - 1 - reversed_last
    last_indices = torch.where(valid_counts > 0, last_indices, torch.zeros_like(last_indices))
    batch_indices = torch.arange(ttnc_tensor.size(0), device=ttnc_tensor.device)
    labels = ttnc_tensor[batch_indices, last_indices]
    labels = labels.masked_fill(valid_counts == 0, padding_idx)
    return labels


def extract_sequence_lengths(ttnc_tensor: torch.Tensor, padding_idx: int = 0) -> torch.Tensor:
    return (ttnc_tensor != padding_idx).sum(dim=1)


def extract_raw_sequence_lengths(dataset, max_samples=None):
    if isinstance(dataset, Subset):
        base_lengths = extract_raw_sequence_lengths(dataset.dataset)
        lengths = base_lengths[np.asarray(dataset.indices)]
    elif (
        hasattr(dataset, "processed_data")
        and getattr(dataset, "evaluation_claim_inclusion_policy", None) is not None
    ):
        policy = dataset.evaluation_claim_inclusion_policy
        lengths = np.asarray(
            [
                sum(dataset.claim_matches_policy(claim, policy) for claim in claims)
                for claims in dataset.processed_data
            ],
            dtype=np.int64,
        )
    elif hasattr(dataset, "raw_sequence_lengths"):
        lengths = np.asarray(dataset.raw_sequence_lengths, dtype=np.int64)
    elif hasattr(dataset, "processed_data"):
        lengths = np.asarray([len(claims) for claims in dataset.processed_data], dtype=np.int64)
    else:
        lengths = np.asarray([len(dataset[idx][0]) for idx in range(len(dataset))], dtype=np.int64)

    if max_samples is not None:
        lengths = lengths[:max_samples]
    return lengths


def get_representation_source_names():
    return list(REPRESENTATION_SOURCES)


def select_representation_tensor(outputs: dict, representation_source: str) -> torch.Tensor:
    if representation_source == "patient_representation":
        return outputs["patient_representation"]
    if representation_source == "patient_representation_pre_sae":
        return outputs.get("patient_representation_pre_sae", outputs["patient_representation"])

    sequence_aux = outputs.get("sequence_aux") or {}
    if representation_source == "context_mean_pool":
        return sequence_aux["context_mean_pool"]
    if representation_source == "context_max_pool":
        return sequence_aux["context_max_pool"]
    if representation_source == "context_pooled":
        return sequence_aux["context_pooled"]
    if representation_source == "next_claim_prediction":
        prediction = outputs["prediction_lvl2"]
        return prediction[:, -1] if prediction.dim() == 3 else prediction
    if representation_source == "dense_decoder_latent":
        dense_decoder_latent = sequence_aux.get("dense_decoder_latent")
        if dense_decoder_latent is None:
            raise ValueError(
                "dense_decoder_latent is not available for this checkpoint/config. "
                "Enable the dense decoder bottleneck before probing this source."
            )
        return dense_decoder_latent

    raise ValueError(
        f"Unsupported representation_source={representation_source!r}. "
        f"Expected one of: {', '.join(REPRESENTATION_SOURCES)}."
    )


def collect_patient_representations(
    model,
    dataloader,
    device=None,
    max_samples=None,
    representation_source: str = "patient_representation_pre_sae",
    missing_modality: str | None = None,
):
    if device is None:
        device = next(model.parameters()).device

    model = model.to(device)
    model.eval()

    embeddings = []
    specialty_labels = []
    regression_targets = []
    sequence_lengths = []
    prototype_assignments = []
    prototype_probabilities = []
    prototype_active_masks = []

    with torch.no_grad():
        for batch in dataloader:
            cpt_tensor, icd_tensor, ttnc_tensor, target = batch
            if missing_modality == "cpt":
                cpt_tensor = torch.zeros_like(cpt_tensor)
            elif missing_modality == "icd":
                icd_tensor = torch.zeros_like(icd_tensor)
            elif missing_modality is not None:
                raise ValueError("missing_modality must be None, 'cpt', or 'icd'.")
            specialty = extract_last_valid_ttnc(ttnc_tensor).cpu()
            lengths = extract_sequence_lengths(ttnc_tensor).cpu()

            outputs = model(
                cpt_tensor=cpt_tensor.to(device),
                icd_tensor=icd_tensor.to(device),
                ttnc_tensor=ttnc_tensor.to(device),
                target=target.to(device),
                teacher_forcing=True,
                generation=False,
            )

            embeddings.append(
                select_representation_tensor(outputs, representation_source).detach().cpu()
            )
            specialty_labels.append(specialty)
            regression_targets.append(target.detach().cpu())
            sequence_lengths.append(lengths)
            assignments = outputs.get("claim_prototype_assignments")
            probabilities = outputs.get("claim_prototype_probabilities")
            active_mask = outputs.get("claim_prototype_active_mask")
            if (
                assignments is not None
                and probabilities is not None
                and active_mask is not None
            ):
                prototype_assignments.append(
                    assignments.detach().cpu()
                )
                prototype_probabilities.append(
                    probabilities.detach().cpu()
                )
                prototype_active_masks.append(active_mask.detach().cpu())

            if max_samples is not None:
                current_size = sum(chunk.size(0) for chunk in embeddings)
                if current_size >= max_samples:
                    break

    if not embeddings:
        raise ValueError("No embeddings were collected from the dataloader.")

    embeddings_np = torch.cat(embeddings, dim=0).numpy()
    specialty_np = torch.cat(specialty_labels, dim=0).numpy()
    targets_np = torch.cat(regression_targets, dim=0).numpy()
    sequence_lengths_np = torch.cat(sequence_lengths, dim=0).numpy()

    if max_samples is not None:
        embeddings_np = embeddings_np[:max_samples]
        specialty_np = specialty_np[:max_samples]
        targets_np = targets_np[:max_samples]
        sequence_lengths_np = sequence_lengths_np[:max_samples]

    metadata = {
        "sequence_lengths": sequence_lengths_np,
        "effective_sequence_lengths": sequence_lengths_np,
        "representation_source": representation_source,
    }
    if prototype_assignments:
        stacked_assignments = torch.cat(
            prototype_assignments,
            dim=0,
        )
        stacked_probabilities = torch.cat(
            prototype_probabilities,
            dim=0,
        )
        stacked_active_masks = torch.cat(prototype_active_masks, dim=0)
        if max_samples is not None:
            stacked_assignments = stacked_assignments[:max_samples]
            stacked_probabilities = stacked_probabilities[:max_samples]
            stacked_active_masks = stacked_active_masks[:max_samples]
        metadata["claim_prototype_assignments"] = stacked_assignments[
            stacked_active_masks
        ].numpy()
        metadata["claim_prototype_probabilities"] = stacked_probabilities[
            stacked_active_masks
        ].numpy()

    return embeddings_np, specialty_np, targets_np, metadata


def compute_claim_prototype_metrics(assignments, probabilities):
    assignments = np.asarray(assignments)
    probabilities = np.asarray(probabilities)
    if assignments.shape != probabilities.shape or assignments.ndim != 2:
        raise ValueError(
            "Prototype assignments and probabilities must have matching "
            "[samples, prototypes] shapes."
        )
    if assignments.shape[0] == 0:
        return {}

    target_labels = assignments.argmax(axis=1)
    prediction_labels = probabilities.argmax(axis=1)
    topk = min(5, assignments.shape[1])
    topk_indices = np.argpartition(probabilities, -topk, axis=1)[:, -topk:]
    marginal = assignments.mean(axis=0)
    effective_count = np.exp(
        -np.sum(marginal * np.log(np.clip(marginal, 1e-12, None)))
    )
    target_entropy = -np.sum(
        assignments * np.log(np.clip(assignments, 1e-12, None)),
        axis=1,
    )
    soft_nll = -np.sum(
        assignments * np.log(np.clip(probabilities, 1e-12, None)),
        axis=1,
    )
    return {
        "num_assignments": int(assignments.shape[0]),
        "prototype_count": int(assignments.shape[1]),
        "prediction_top1_accuracy": float(np.mean(prediction_labels == target_labels)),
        "prediction_top5_accuracy": float(
            np.mean((topk_indices == target_labels[:, None]).any(axis=1))
        ),
        "prediction_soft_nll": float(np.mean(soft_nll)),
        "target_entropy": float(np.mean(target_entropy)),
        "effective_prototype_count": float(effective_count),
        "effective_prototype_fraction": float(
            effective_count / assignments.shape[1]
        ),
        "hard_utilization_fraction": float(
            np.unique(target_labels).size / assignments.shape[1]
        ),
    }


def compute_claim_prototype_stability(full_probabilities, ablated_probabilities):
    full_probabilities = np.asarray(full_probabilities)
    ablated_probabilities = np.asarray(ablated_probabilities)
    if full_probabilities.shape != ablated_probabilities.shape:
        raise ValueError("Full and ablated prototype probabilities must have matching shapes.")
    if full_probabilities.ndim != 2 or full_probabilities.shape[0] == 0:
        return {}

    midpoint = 0.5 * (full_probabilities + ablated_probabilities)
    full_kl = np.sum(
        full_probabilities
        * (
            np.log(np.clip(full_probabilities, 1e-12, None))
            - np.log(np.clip(midpoint, 1e-12, None))
        ),
        axis=1,
    )
    ablated_kl = np.sum(
        ablated_probabilities
        * (
            np.log(np.clip(ablated_probabilities, 1e-12, None))
            - np.log(np.clip(midpoint, 1e-12, None))
        ),
        axis=1,
    )
    return {
        "top1_agreement": float(
            np.mean(
                full_probabilities.argmax(axis=1)
                == ablated_probabilities.argmax(axis=1)
            )
        ),
        "jensen_shannon_divergence": float(np.mean(0.5 * (full_kl + ablated_kl))),
    }


def cosine_retrieval_hit_rate_at_k(embeddings, labels, k=5):
    num_samples = embeddings.shape[0]
    if num_samples < 2:
        return None

    k = min(k, num_samples - 1)
    normalized = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), a_min=1e-8, a_max=None)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)
    topk_indices = np.argpartition(similarity, -k, axis=1)[:, -k:]
    topk_labels = labels[topk_indices]
    hits = (topk_labels == labels[:, None]).any(axis=1)
    return float(hits.mean())


def compute_representation_geometry_metrics(
    embeddings,
    cosine_sample_size=2048,
):
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")
    if embeddings.shape[0] < 2:
        raise ValueError("Need at least two samples to compute geometry metrics.")

    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    std = centered.std(axis=0)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)
    eigenvalue_sum = eigenvalues.sum()
    participation_ratio = (
        float((eigenvalue_sum**2) / np.square(eigenvalues).sum())
        if eigenvalue_sum > 0
        else 0.0
    )

    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    top1_share = float(sorted_eigenvalues[:1].sum() / eigenvalue_sum) if eigenvalue_sum > 0 else 0.0
    top5_share = float(sorted_eigenvalues[:5].sum() / eigenvalue_sum) if eigenvalue_sum > 0 else 0.0
    top10_share = float(sorted_eigenvalues[:10].sum() / eigenvalue_sum) if eigenvalue_sum > 0 else 0.0

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.clip(norms, a_min=1e-8, a_max=None)
    sample_size = min(cosine_sample_size, embeddings.shape[0])
    sampled = normalized[:sample_size]
    cosine_similarity = sampled @ sampled.T
    off_diagonal_mask = ~np.eye(sample_size, dtype=bool)
    pairwise_cosine = cosine_similarity[off_diagonal_mask]

    return {
        "num_samples": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "std_mean": float(std.mean()),
        "std_min": float(std.min()),
        "std_median": float(np.median(std)),
        "low_var_frac_lt_1e-3": float((std < 1e-3).mean()),
        "low_var_frac_lt_1e-2": float((std < 1e-2).mean()),
        "participation_ratio": participation_ratio,
        "participation_ratio_fraction": float(
            participation_ratio / embeddings.shape[1]
        ),
        "explained_variance_top1_share": top1_share,
        "explained_variance_top5_share": top5_share,
        "explained_variance_top10_share": top10_share,
        "mean_pairwise_cosine": float(pairwise_cosine.mean()),
        "median_pairwise_cosine": float(np.median(pairwise_cosine)),
        "cosine_sample_size": int(sample_size),
    }


def compute_ttnc_proxy_clustering_metrics(embeddings, labels, random_state=42):
    unique_labels = np.unique(labels)
    if unique_labels.size < 2 or embeddings.shape[0] <= unique_labels.size:
        return {}

    kmeans = KMeans(n_clusters=unique_labels.size, n_init=10, random_state=random_state)
    predicted_clusters = kmeans.fit_predict(embeddings)

    metrics = {
        "ttnc_proxy_label_cluster_ari": float(adjusted_rand_score(labels, predicted_clusters)),
    }

    if 1 < np.unique(predicted_clusters).size < embeddings.shape[0]:
        metrics["cluster_silhouette"] = float(silhouette_score(embeddings, predicted_clusters))

    return metrics


def compute_ttnc_proxy_probe_metrics(embeddings, labels, test_size=0.25, random_state=42):
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    if unique_labels.size < 2 or np.min(label_counts) < 2:
        return {}

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )
    except ValueError:
        return {}

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=random_state),
    )
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    return {
        "ttnc_proxy_probe_accuracy": float(accuracy_score(y_test, predictions)),
        "ttnc_proxy_probe_macro_f1": float(f1_score(y_test, predictions, average="macro")),
    }


def compute_regression_probe_metrics(embeddings, targets, test_size=0.25, random_state=42):
    if embeddings.shape[0] < 4:
        return {}

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        targets,
        test_size=test_size,
        random_state=random_state,
    )

    return compute_heldout_regression_probe_metrics(
        X_train,
        y_train,
        X_test,
        y_test,
    )


def fit_regression_probe(train_embeddings, train_targets):
    scaler_y = StandardScaler()
    train_targets_scaled = scaler_y.fit_transform(
        train_targets.reshape(-1, 1)
    ).flatten()
    regressor = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    regressor.fit(train_embeddings, train_targets_scaled)
    return regressor, scaler_y


def predict_regression_probe(regressor, scaler_y, embeddings):
    predictions_scaled = regressor.predict(embeddings)
    return scaler_y.inverse_transform(
        predictions_scaled.reshape(-1, 1)
    ).flatten()


def score_regression_predictions(predictions, targets):
    rmse_log1p = np.sqrt(np.mean((predictions - targets) ** 2))
    predictions_dollars = np.expm1(np.clip(predictions, a_min=0, a_max=10))
    targets_dollars = np.expm1(targets)
    mae_dollars = np.mean(np.abs(predictions_dollars - targets_dollars))
    rmse_dollars = np.sqrt(np.mean((predictions_dollars - targets_dollars) ** 2))
    target_volume = np.sum(np.abs(targets_dollars))
    wape_percent = (
        100.0 * np.sum(np.abs(predictions_dollars - targets_dollars)) / target_volume
        if target_volume > 0
        else float("nan")
    )

    return {
        "target_probe_rmse_log1p": float(rmse_log1p),
        "target_probe_mae_dollars": float(mae_dollars),
        "target_probe_rmse_dollars": float(rmse_dollars),
        "target_probe_wape_percent": float(wape_percent),
    }


def compute_heldout_regression_probe_metrics(
    train_embeddings,
    train_targets,
    eval_embeddings,
    eval_targets,
    *,
    return_predictions=False,
):
    if train_embeddings.shape[0] < 2 or eval_embeddings.shape[0] < 1:
        return ({}, np.asarray([])) if return_predictions else {}
    regressor, scaler_y = fit_regression_probe(train_embeddings, train_targets)
    predictions = predict_regression_probe(regressor, scaler_y, eval_embeddings)
    metrics = score_regression_predictions(predictions, eval_targets)
    if return_predictions:
        return metrics, predictions
    return metrics


def compute_missing_modality_metrics(
    train_embeddings,
    train_targets,
    full_embeddings,
    missing_embeddings,
    eval_targets,
):
    """Score ablated embeddings with the probe fit on full training inputs."""
    if full_embeddings.shape != missing_embeddings.shape:
        raise ValueError("Full and missing-modality embeddings must have the same shape.")
    regressor, scaler_y = fit_regression_probe(train_embeddings, train_targets)
    predictions = predict_regression_probe(regressor, scaler_y, missing_embeddings)
    metrics = score_regression_predictions(predictions, eval_targets)
    full_norm = np.linalg.norm(full_embeddings, axis=1)
    missing_norm = np.linalg.norm(missing_embeddings, axis=1)
    cosine = np.sum(full_embeddings * missing_embeddings, axis=1) / np.clip(
        full_norm * missing_norm,
        a_min=1e-8,
        a_max=None,
    )
    relative_l2 = np.linalg.norm(full_embeddings - missing_embeddings, axis=1) / np.clip(
        full_norm,
        a_min=1e-8,
        a_max=None,
    )
    metrics.update(
        {
            "full_to_missing_cosine_mean": float(np.mean(cosine)),
            "full_to_missing_cosine_median": float(np.median(cosine)),
            "full_to_missing_relative_l2_mean": float(np.mean(relative_l2)),
            "probe_training_protocol": "full_train_fit_applied_without_refit",
        }
    )
    return metrics


def _build_rank_buckets(values, bucket_names, stat_prefix):
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("Slice values must be a 1D array.")
    if values.size == 0:
        return {}

    sorted_indices = np.argsort(values, kind="mergesort")
    splits = np.array_split(sorted_indices, len(bucket_names))
    buckets = {}

    for bucket_name, bucket_indices in zip(bucket_names, splits):
        if bucket_indices.size == 0:
            continue
        bucket_values = values[bucket_indices]
        buckets[bucket_name] = {
            "indices": np.sort(bucket_indices),
            "summary": {
                f"{stat_prefix}_min": float(np.min(bucket_values)),
                f"{stat_prefix}_median": float(np.median(bucket_values)),
                f"{stat_prefix}_max": float(np.max(bucket_values)),
            },
        }

    return buckets


def _evaluate_slice_group(
    embeddings,
    specialty_labels,
    regression_targets,
    slice_buckets,
    retrieval_k,
    random_state,
):
    results = {}
    for bucket_name, bucket in slice_buckets.items():
        bucket_indices = bucket["indices"]
        slice_results = evaluate_representation_quality(
            embeddings[bucket_indices],
            specialty_labels[bucket_indices],
            regression_targets[bucket_indices],
            retrieval_k=retrieval_k,
            random_state=random_state,
            include_slices=False,
        )
        slice_results.update(bucket["summary"])
        results[bucket_name] = slice_results
    return results


def build_slice_reports(
    embeddings,
    specialty_labels,
    regression_targets,
    sequence_lengths,
    effective_sequence_lengths=None,
    retrieval_k=5,
    random_state=42,
):
    regression_targets = np.asarray(regression_targets)
    specialty_labels = np.asarray(specialty_labels)
    sequence_lengths = np.asarray(sequence_lengths) if sequence_lengths is not None else None
    effective_sequence_lengths = np.asarray(effective_sequence_lengths) if effective_sequence_lengths is not None else None

    if sequence_lengths is None and effective_sequence_lengths is None:
        raise ValueError("At least one sequence length array must be provided for slice reporting.")

    primary_sequence_lengths = sequence_lengths if sequence_lengths is not None else effective_sequence_lengths
    sequence_length_bucket_basis = "raw_claims" if sequence_lengths is not None else "effective_claims"
    effective_sequence_lengths = (
        effective_sequence_lengths
        if effective_sequence_lengths is not None
        else primary_sequence_lengths
    )

    target_dollars = np.exp(regression_targets)
    specialty_counts = np.vectorize(dict(zip(*np.unique(specialty_labels, return_counts=True))).get)(specialty_labels)

    slice_reports = {
        "target_cost_bucket": _evaluate_slice_group(
            embeddings,
            specialty_labels,
            regression_targets,
            _build_rank_buckets(
                target_dollars,
                ["q1_low_cost", "q2_mid_low_cost", "q3_mid_high_cost", "q4_high_cost"],
                "target_dollars",
            ),
            retrieval_k=retrieval_k,
            random_state=random_state,
        ),
        "sequence_length_bucket_basis": sequence_length_bucket_basis,
        "sequence_length_bucket": _evaluate_slice_group(
            embeddings,
            specialty_labels,
            regression_targets,
            _build_rank_buckets(
                primary_sequence_lengths,
                [
                    "q1_shortest_sequences",
                    "q2_short_sequences",
                    "q3_long_sequences",
                    "q4_longest_sequences",
                ],
                "sequence_length_claims",
            ),
            retrieval_k=retrieval_k,
            random_state=random_state,
        ),
        "ttnc_proxy_frequency_bucket": _evaluate_slice_group(
            embeddings,
            specialty_labels,
            regression_targets,
            _build_rank_buckets(
                specialty_counts,
                [
                    "q1_rarest_specialties",
                    "q2_less_common_specialties",
                    "q3_more_common_specialties",
                    "q4_most_common_specialties",
                ],
                "ttnc_proxy_frequency",
            ),
            retrieval_k=retrieval_k,
            random_state=random_state,
        ),
    }

    if effective_sequence_lengths is not None:
        slice_reports["effective_sequence_length_bucket"] = _evaluate_slice_group(
            embeddings,
            specialty_labels,
            regression_targets,
            _build_rank_buckets(
                effective_sequence_lengths,
                [
                    "q1_shortest_effective_sequences",
                    "q2_short_effective_sequences",
                    "q3_long_effective_sequences",
                    "q4_longest_effective_sequences",
                ],
                "effective_sequence_length_claims",
            ),
            retrieval_k=retrieval_k,
            random_state=random_state,
        )

    return slice_reports


def evaluate_representation_quality(
    embeddings,
    specialty_labels,
    regression_targets,
    sequence_lengths=None,
    effective_sequence_lengths=None,
    retrieval_k=5,
    random_state=42,
    include_slices=True,
):
    specialty_labels = np.asarray(specialty_labels)
    regression_targets = np.asarray(regression_targets)
    embeddings = np.asarray(embeddings)

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")

    results = {
        "num_samples": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "num_ttnc_proxy_labels": int(np.unique(specialty_labels).size),
        "ttnc_proxy_label_source": TTNC_PROXY_LABEL_SOURCE,
        "selection_metric_priority": {
            "primary": [
                "target_probe_mae_dollars",
                "target_probe_wape_percent",
                "target_probe_rmse_dollars",
                "val_rmse_improvement_vs_mean_baseline",
                "val_rmse_improvement_pct_vs_mean_baseline",
                "cluster_silhouette",
            ],
            "secondary": [
                f"ttnc_proxy_retrieval_hit_rate_at_{retrieval_k}",
                "ttnc_proxy_label_cluster_ari",
                "ttnc_proxy_probe_accuracy",
                "ttnc_proxy_probe_macro_f1",
            ],
        },
    }

    retrieval = cosine_retrieval_hit_rate_at_k(embeddings, specialty_labels, k=retrieval_k)
    if retrieval is not None:
        results[f"ttnc_proxy_retrieval_hit_rate_at_{retrieval_k}"] = retrieval

    results.update(
        compute_ttnc_proxy_clustering_metrics(
            embeddings,
            specialty_labels,
            random_state=random_state,
        )
    )
    results.update(
        compute_ttnc_proxy_probe_metrics(
            embeddings,
            specialty_labels,
            random_state=random_state,
        )
    )
    results.update(
        compute_regression_probe_metrics(
            embeddings,
            regression_targets,
            random_state=random_state,
        )
    )

    if include_slices and sequence_lengths is not None:
        results["slices"] = build_slice_reports(
            embeddings,
            specialty_labels,
            regression_targets,
            sequence_lengths=sequence_lengths,
            effective_sequence_lengths=effective_sequence_lengths,
            retrieval_k=retrieval_k,
            random_state=random_state,
        )

    return results


def save_representation_eval(results: dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
