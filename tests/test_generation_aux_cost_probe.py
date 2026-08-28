import torch

from jepa_models.ssl_objectives import sigreg_gaussian_distance
from scripts.run_generation_aux_cost_probe import (
    ResidualFeatureAdapter,
    SharedCostGenerationHead,
    auxiliary_loss,
    paired_deltas,
)
from scripts.run_generation_aux_label_efficiency import aggregate as aggregate_efficiency
from scripts.run_future_claim_cost_mediation import (
    CostProbe,
    aggregate as aggregate_mediation,
    oracle_claim_features,
    parse_args as parse_mediation_args,
)


def test_shared_cost_generation_head_shapes():
    model = SharedCostGenerationHead(8, 16, 11, 13, 5)
    output = model(torch.randn(4, 8))
    assert output["cost"].shape == (4,)
    assert output["cpt_logits"].shape == (4, 11)
    assert output["icd_logits"].shape == (4, 13)
    assert output["ttnc_logits"].shape == (4, 5)


def test_residual_adapter_starts_as_identity():
    adapter = ResidualFeatureAdapter(8, 3)
    features = torch.randn(4, 8)
    assert torch.equal(adapter(features), features)


def test_linear_heads_accept_adapted_features():
    model = SharedCostGenerationHead(8, 16, 11, 13, 5, adapter_dim=3, linear_heads=True)
    output = model(torch.randn(4, 8))
    assert output["cost"].shape == (4,)
    assert output["cpt_logits"].shape == (4, 11)


def test_paired_deltas_compare_auxiliary_to_same_seed_cost_only():
    runs = [
        {"condition": "pretrained", "aux_weight": 0.0, "seed": 42,
         "cost": {"target_probe_mae_dollars": 100.0}},
        {"condition": "pretrained", "aux_weight": 0.1, "seed": 42,
         "cost": {"target_probe_mae_dollars": 90.0}},
        {"condition": "pretrained", "aux_weight": 0.0, "seed": 43,
         "cost": {"target_probe_mae_dollars": 120.0}},
        {"condition": "pretrained", "aux_weight": 0.1, "seed": 43,
         "cost": {"target_probe_mae_dollars": 115.0}},
    ]
    result = paired_deltas(runs)["pretrained|0.1"]
    assert result["paired_seeds"] == 2
    assert result["mae_dollars_aux_minus_cost_only_mean"] == -7.5


def test_auxiliary_loss_supports_each_modality_subset():
    model = SharedCostGenerationHead(8, 16, 6, 7, 5)
    outputs = model(torch.randn(4, 8))
    cpt = torch.zeros(4, 6)
    icd = torch.zeros(4, 7)
    cpt[:, 2] = 1
    icd[:, 3] = 1
    ttnc = torch.tensor([1, 2, 3, 4])
    for modalities in (("cpt",), ("icd",), ("ttnc",), ("cpt", "icd")):
        assert torch.isfinite(auxiliary_loss(outputs, cpt, icd, ttnc, modalities))


def test_lejepa_sigreg_distance_is_finite_and_backpropagates():
    embeddings = torch.randn(32, 8, requires_grad=True)
    loss = sigreg_gaussian_distance(
        embeddings,
        num_slices=16,
        num_points=9,
        formulation="lejepa_convex",
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(embeddings.grad).all()


def test_efficiency_aggregate_computes_paired_delta_and_wins():
    runs = []
    for seed, baseline, improved in ((42, 100.0, 90.0), (43, 110.0, 105.0)):
        for ablation, mae in (("cost_only", baseline), ("ttnc_only", improved)):
            runs.append({
                "cost_label_fraction": 0.1,
                "seed": seed,
                "ablation": ablation,
                "cost": {
                    "target_probe_mae_dollars": mae,
                    "target_probe_rmse_dollars": mae + 50.0,
                },
            })
    result = aggregate_efficiency(runs)["0.1|ttnc_only"]
    assert result["paired_mae_delta_vs_cost_only"]["mean"] == -7.5
    assert result["paired_mae_delta_vs_cost_only"]["wins"] == 2


def test_future_claim_mediation_probe_shapes_and_oracle_features():
    assert CostProbe(8, hidden_dim=16, linear=True)(torch.randn(4, 8)).shape == (4,)
    assert CostProbe(8, hidden_dim=16, linear=False)(torch.randn(4, 8)).shape == (4,)
    targets = {
        "cpt_ids": torch.tensor([[1, 2], [3, 0]]).numpy(),
        "icd_ids": torch.tensor([[2, 0], [1, 3]]).numpy(),
        "ttnc": torch.tensor([2, 3]).numpy(),
    }
    features = oracle_claim_features(targets, 5, 5, 4)
    assert features.shape == (2, 11)
    assert features[0].sum() == 4.0


def test_mediation_aggregate_uses_direct_mlp_as_paired_reference():
    runs = []
    for pathway, mae in (("direct_embedding_mlp", 100.0), ("predicted_claims_linear", 90.0)):
        runs.append({
            "cost_label_fraction": 0.05,
            "seed": 42,
            "pathway": pathway,
            "cost": {
                "target_probe_mae_dollars": mae,
                "target_probe_rmse_dollars": mae + 50.0,
            },
        })
    result = aggregate_mediation(runs)["0.05|predicted_claims_linear"]
    assert result["paired_mae_delta_vs_direct_mlp"]["mean"] == -10.0
    assert result["paired_mae_delta_vs_direct_mlp"]["wins"] == 1


def test_mediation_args_normalize_claim_mediator_dims():
    args = parse_mediation_args([
        "--checkpoint", "encoder.ckpt",
        "--data-path", "claims.parquet",
        "--data-contract", "contract.json",
        "--output-dir", "results",
        "--claim-mediator-dims", "64", "16", "16", "32",
    ])
    assert args.claim_mediator_dims == [16, 32, 64]
