import math
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _distributed_sigreg_active() -> bool:
    return (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )


def _synchronize_sigreg_directions(directions: torch.Tensor) -> torch.Tensor:
    """Use identical random projections on every distributed worker."""
    if _distributed_sigreg_active():
        dist.broadcast(directions, src=0)
    return directions


def _global_empirical_characteristic_function(
    projected: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Average cosine/sine statistics over the full distributed batch.

    Only the two characteristic-function sufficient statistics are reduced;
    embeddings never leave their worker. The autograd-aware reductions retain
    gradients to each local embedding shard.
    """
    real_sum = torch.cos(projected).sum(dim=0)
    imag_sum = torch.sin(projected).sum(dim=0)
    count = projected.new_tensor(float(projected.size(0)))
    if _distributed_sigreg_active():
        from torch.distributed.nn.functional import all_reduce

        real_sum = all_reduce(real_sum, op=dist.ReduceOp.SUM)
        imag_sum = all_reduce(imag_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    count = count.clamp_min(1.0)
    return real_sum / count, imag_sum / count, count


def rep_relu(x: torch.Tensor) -> torch.Tensor:
    """Exact ReLU forward values with the smoother GELU surrogate gradient."""
    gelu = F.gelu(x)
    return F.relu(x).detach() + gelu - gelu.detach()


def apply_representation_link(x: torch.Tensor, link: str) -> torch.Tensor:
    """Apply a configured canonical representation link."""
    normalized = link.lower()
    if normalized == "identity":
        return x
    if normalized == "relu":
        return F.relu(x)
    if normalized == "reprelu":
        return rep_relu(x)
    raise ValueError(f"Unsupported representation link: {link!r}")


class BaseSSLObjective(nn.Module):
    """Common utilities for JEPA representation objectives."""

    def __init__(self, config):
        super().__init__()
        self.epsilon = config.epsilon
        self.loss_fn = nn.MSELoss()

    def _flatten_valid(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        prediction_flat = prediction.reshape(-1, prediction.size(-1))
        target_flat = target.reshape(-1, target.size(-1))

        if mask is None:
            valid_mask = torch.ones(
                prediction_flat.size(0),
                dtype=torch.bool,
                device=prediction.device,
            )
        else:
            valid_mask = mask.reshape(-1).bool()

        prediction_valid = prediction_flat[valid_mask]
        target_valid = target_flat[valid_mask]

        weights_valid = None
        if sample_weights is not None:
            weights_valid = sample_weights.reshape(-1)[valid_mask].to(prediction_valid.dtype)

        return prediction_valid, target_valid, weights_valid

    def _weighted_mse(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if prediction.numel() == 0:
            return prediction.new_zeros(())

        per_example = F.mse_loss(prediction, target, reduction="none").mean(dim=-1)
        if sample_weights is None:
            return per_example.mean()

        weight_sum = sample_weights.sum().clamp(min=self.epsilon)
        return (per_example * sample_weights).sum() / weight_sum

    def _zero_metrics(self, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        zero = torch.zeros((), device=device, dtype=dtype)
        return {
            "total": zero,
            "predictive": zero,
            "regularizer": zero,
            "diagnostics": {},
        }


class VICRegObjective(BaseSSLObjective):
    """Compatibility wrapper for the existing VICReg-style JEPA loss."""

    def __init__(self, config):
        super().__init__(config)
        self.target_var_lvl1 = config.target_var_lvl1
        self.target_var_lvl2 = config.target_var_lvl2
        self.var_penalty_scale_lvl1 = config.var_penalty_scale_lvl1
        self.var_penalty_scale_lvl2 = config.var_penalty_scale_lvl2
        self.cov_penalty_scale_lvl1 = config.cov_penalty_scale_lvl1
        self.cov_penalty_scale_lvl2 = config.cov_penalty_scale_lvl2
        self.amplification_power = config.amplification_power

    def _centered_predictive(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if prediction.numel() == 0:
            return prediction.new_zeros(())

        prediction_centered = prediction - prediction.mean(dim=0, keepdim=True)
        target_centered = target - target.mean(dim=0, keepdim=True)
        return self._weighted_mse(prediction_centered, target_centered, sample_weights)

    def compute(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        level: str = "2",
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        prediction_valid, target_valid, weights_valid = self._flatten_valid(
            prediction, target, mask=mask, sample_weights=sample_weights
        )

        if prediction_valid.numel() == 0:
            return self._zero_metrics(prediction.device, prediction.dtype)

        predictive = self._centered_predictive(prediction_valid, target_valid, weights_valid)

        if prediction_valid.size(0) < 2:
            zero = prediction_valid.new_zeros(())
            return {
                "total": predictive,
                "predictive": predictive,
                "regularizer": zero,
                "diagnostics": {
                    "variance": zero,
                    "covariance": zero,
                },
            }

        target_var = self.target_var_lvl1 if level == "1" else self.target_var_lvl2
        var_penalty = (
            self.var_penalty_scale_lvl1 if level == "1" else self.var_penalty_scale_lvl2
        )
        cov_penalty = (
            self.cov_penalty_scale_lvl1 if level == "1" else self.cov_penalty_scale_lvl2
        )

        prediction_centered = prediction_valid - prediction_valid.mean(dim=0, keepdim=True)
        target_centered = target_valid - target_valid.mean(dim=0, keepdim=True)

        std_prediction = torch.sqrt(prediction_centered.var(dim=0) + self.epsilon)
        std_target = torch.sqrt(target_centered.var(dim=0) + self.epsilon)
        variance = (
            torch.mean(F.relu(target_var - std_prediction)) / 2
            + torch.mean(F.relu(target_var - std_target)) / 2
        ) ** self.amplification_power

        covariance_prediction = (
            prediction_centered.T @ prediction_centered
        ) / max(prediction_centered.size(0) - 1, 1)
        covariance_target = (
            target_centered.T @ target_centered
        ) / max(target_centered.size(0) - 1, 1)
        identity = torch.eye(covariance_prediction.size(0), device=prediction.device)
        covariance = ((covariance_prediction - identity) ** 2).sum() / 2 + (
            (covariance_target - identity) ** 2
        ).sum() / 2

        regularizer = variance * var_penalty + covariance * cov_penalty
        return {
            "total": predictive + regularizer,
            "predictive": predictive,
            "regularizer": regularizer,
            "diagnostics": {
                "variance": variance,
                "covariance": covariance,
            },
        }


class SIGRegObjective(BaseSSLObjective):
    """Claims-native SIGReg using random projections and characteristic-function matching."""

    def __init__(self, config):
        super().__init__(config)
        self.sigreg_weight_lvl1 = config.sigreg_weight_lvl1
        self.sigreg_weight_lvl2 = config.sigreg_weight_lvl2
        self.num_slices = config.sigreg_num_slices
        self.num_points = config.sigreg_num_points
        self.formulation = getattr(config, "sigreg_formulation", "legacy_additive")

    def _characteristic_function_distance(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.numel() == 0:
            return embeddings.new_zeros(())

        directions = torch.randn(
            embeddings.size(-1),
            self.num_slices,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        directions = F.normalize(directions, dim=0, eps=self.epsilon)
        directions = _synchronize_sigreg_directions(directions)

        projections = embeddings @ directions
        if self.formulation in {"lejepa_convex", "levjepa_additive"}:
            t_values = torch.linspace(
                0.0,
                3.0,
                steps=self.num_points,
                device=embeddings.device,
                dtype=embeddings.dtype,
            )
        else:
            t_values = torch.linspace(
                0.25,
                2.25,
                steps=self.num_points,
                device=embeddings.device,
                dtype=embeddings.dtype,
            )

        projected = projections.unsqueeze(-1) * t_values.view(1, 1, -1)
        empirical_real, empirical_imag, global_count = (
            _global_empirical_characteristic_function(projected)
        )
        gaussian_real = torch.exp(-0.5 * (t_values**2)).view(1, -1)

        error = (empirical_real - gaussian_real) ** 2 + empirical_imag**2
        if self.formulation in {"lejepa_convex", "levjepa_additive"}:
            step = 3.0 / max(self.num_points - 1, 1)
            quadrature_weights = torch.full_like(t_values, 2.0 * step)
            if self.num_points > 1:
                quadrature_weights[[0, -1]] = step
            quadrature_weights = quadrature_weights * torch.exp(
                -0.5 * t_values.square()
            )
            return (
                (error @ quadrature_weights) * global_count
            ).mean()
        return error.mean()

    def compute(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        level: str = "2",
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        prediction_valid, target_valid, weights_valid = self._flatten_valid(
            prediction, target, mask=mask, sample_weights=sample_weights
        )

        if prediction_valid.numel() == 0:
            return self._zero_metrics(prediction.device, prediction.dtype)

        predictive = self._weighted_mse(prediction_valid, target_valid, weights_valid)
        sigreg_weight = (
            self.sigreg_weight_lvl1 if level == "1" else self.sigreg_weight_lvl2
        )
        sigreg_raw = (
            self._characteristic_function_distance(prediction_valid)
            + self._characteristic_function_distance(target_valid)
        ) / 2
        regularizer = sigreg_raw * sigreg_weight
        total = predictive + regularizer
        if self.formulation == "lejepa_convex":
            total = predictive * (1.0 - sigreg_weight) + regularizer
        return {
            "total": total,
            "predictive": predictive,
            "regularizer": regularizer,
            "diagnostics": {
                "sigreg_raw": sigreg_raw,
            },
        }


class RDMRegObjective(BaseSSLObjective):
    """Reference-distribution matching via sliced Wasserstein distance.

    The reference is a unit-variance generalized Gaussian before the configured
    link is applied. ``p=2`` is Gaussian and ``p=1`` is Laplace. By default only
    encoder targets are matched, mirroring LpWM; matching predictor outputs is
    retained as an explicit ablation.
    """

    def __init__(self, config):
        super().__init__(config)
        self.weight_lvl1 = float(config.rdmreg_weight_lvl1)
        self.weight_lvl2 = float(config.rdmreg_weight_lvl2)
        self.num_projections = int(config.rdmreg_num_projections)
        self.target_p = float(config.rdmreg_target_p)
        self.target_mu = float(config.rdmreg_target_mu)
        self.link_lvl1 = getattr(config, "representation_link_lvl1", "identity")
        self.link_lvl2 = getattr(config, "representation_link_lvl2", "identity")
        self.regularize_prediction = bool(
            getattr(config, "rdmreg_regularize_prediction", False)
        )
        self.support_weight_lvl1 = float(
            getattr(config, "rdmreg_support_alignment_weight_lvl1", 0.0)
        )
        self.support_weight_lvl2 = float(
            getattr(config, "rdmreg_support_alignment_weight_lvl2", 0.0)
        )
        self.support_temperature = float(
            getattr(config, "rdmreg_support_temperature", 0.5)
        )

    def _sample_reference(self, like: torch.Tensor, link: str) -> torch.Tensor:
        shape = like.shape
        if self.target_p == 2.0:
            reference = torch.randn(shape, device=like.device, dtype=like.dtype)
        elif self.target_p == 1.0:
            # Laplace(0, 1/sqrt(2)) has unit variance.
            scale = like.new_tensor(1.0 / math.sqrt(2.0))
            uniform = torch.rand(shape, device=like.device, dtype=like.dtype)
            uniform = uniform.clamp(
                min=torch.finfo(like.dtype).eps,
                max=1.0 - torch.finfo(like.dtype).eps,
            )
            reference = torch.where(
                uniform < 0.5,
                scale * torch.log(2.0 * uniform),
                -scale * torch.log(2.0 * (1.0 - uniform)),
            )
        else:
            # If |X / scale|^p ~ Gamma(1/p, 1), this scale gives Var[X]=1.
            concentration = 1.0 / self.target_p
            gamma_sample = torch._standard_gamma(
                torch.full(shape, concentration, device=like.device, dtype=like.dtype)
            )
            signs = torch.where(
                torch.rand(shape, device=like.device) < 0.5,
                like.new_tensor(-1.0),
                like.new_tensor(1.0),
            )
            scale = math.sqrt(
                math.gamma(1.0 / self.target_p)
                / math.gamma(3.0 / self.target_p)
            )
            reference = signs * scale * gamma_sample.pow(1.0 / self.target_p)
        return apply_representation_link(reference + self.target_mu, link).detach()

    def _sliced_wasserstein(self, embeddings: torch.Tensor, link: str) -> torch.Tensor:
        if embeddings.numel() == 0 or embeddings.size(0) < 2:
            return embeddings.new_zeros(())
        directions = torch.randn(
            embeddings.size(-1),
            self.num_projections,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        directions = F.normalize(directions, dim=0, eps=self.epsilon)
        reference = self._sample_reference(embeddings, link)
        projected_embeddings = (embeddings @ directions).sort(dim=0).values
        projected_reference = (reference @ directions).sort(dim=0).values
        return F.mse_loss(projected_embeddings, projected_reference)

    def _support_alignment(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Match exact target support with a smooth predictor activity proxy."""
        threshold = 10.0 * torch.finfo(target.dtype).eps
        target_support = (target.abs() > threshold).to(prediction.dtype).detach()
        positive_prediction = prediction.clamp_min(0.0)
        prediction_activity = -torch.expm1(
            -positive_prediction / self.support_temperature
        )
        per_example = (prediction_activity - target_support).abs().mean(dim=-1)
        if sample_weights is None:
            return per_example.mean()
        weight_sum = sample_weights.sum().clamp(min=self.epsilon)
        return (per_example * sample_weights).sum() / weight_sum

    def compute(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        level: str = "2",
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        prediction_valid, target_valid, weights_valid = self._flatten_valid(
            prediction, target, mask=mask, sample_weights=sample_weights
        )
        if prediction_valid.numel() == 0:
            return self._zero_metrics(prediction.device, prediction.dtype)

        predictive = self._weighted_mse(prediction_valid, target_valid, weights_valid)
        link = self.link_lvl1 if level == "1" else self.link_lvl2
        rdmreg_raw = self._sliced_wasserstein(target_valid, link)
        if self.regularize_prediction:
            rdmreg_raw = 0.5 * (
                rdmreg_raw + self._sliced_wasserstein(prediction_valid, link)
            )
        weight = self.weight_lvl1 if level == "1" else self.weight_lvl2
        support_weight = (
            self.support_weight_lvl1 if level == "1" else self.support_weight_lvl2
        )
        support_alignment_raw = self._support_alignment(
            prediction_valid, target_valid, weights_valid
        )
        regularizer = weight * rdmreg_raw + support_weight * support_alignment_raw
        threshold = 10.0 * torch.finfo(target_valid.dtype).eps
        return {
            "total": predictive + regularizer,
            "predictive": predictive,
            "regularizer": regularizer,
            "diagnostics": {
                "rdmreg_raw": rdmreg_raw,
                "support_alignment_raw": support_alignment_raw,
                "target_active_fraction": (
                    target_valid.abs() > threshold
                ).float().mean(),
                "prediction_active_fraction": (
                    prediction_valid.abs() > threshold
                ).float().mean(),
            },
        }


def sigreg_gaussian_distance(
    embeddings: torch.Tensor,
    *,
    num_slices: int = 256,
    num_points: int = 17,
    epsilon: float = 1e-4,
    formulation: str = "legacy_additive",
) -> torch.Tensor:
    """Characteristic-function distance used for typed marginal SIGReg."""
    if embeddings.numel() == 0 or embeddings.size(0) < 2:
        return embeddings.new_zeros(())
    directions = torch.randn(
        embeddings.size(-1),
        num_slices,
        device=embeddings.device,
        dtype=embeddings.dtype,
    )
    directions = F.normalize(directions, dim=0, eps=epsilon)
    directions = _synchronize_sigreg_directions(directions)
    projections = embeddings @ directions
    if formulation in {"lejepa_convex", "levjepa_additive"}:
        t_values = torch.linspace(
            0.0, 3.0, steps=num_points,
            device=embeddings.device, dtype=embeddings.dtype,
        )
    elif formulation == "legacy_additive":
        t_values = torch.linspace(
            0.25, 2.25, steps=num_points,
            device=embeddings.device, dtype=embeddings.dtype,
        )
    else:
        raise ValueError(f"Unsupported SIGReg formulation: {formulation!r}")
    projected = projections.unsqueeze(-1) * t_values.view(1, 1, -1)
    empirical_real, empirical_imag, global_count = _global_empirical_characteristic_function(
        projected
    )
    gaussian_real = torch.exp(-0.5 * t_values.square()).view(1, -1)
    error = (empirical_real - gaussian_real).square() + empirical_imag.square()
    if formulation in {"lejepa_convex", "levjepa_additive"}:
        step = 3.0 / max(num_points - 1, 1)
        quadrature_weights = torch.full_like(t_values, 2.0 * step)
        if num_points > 1:
            quadrature_weights[[0, -1]] = step
        quadrature_weights = quadrature_weights * torch.exp(-0.5 * t_values.square())
        return ((error @ quadrature_weights) * global_count).mean()
    return error.mean()


class WristbandGaussianRegularizer(nn.Module):
    """Calibrated Wristband Gaussian loss for a fixed-size embedding sample.

    This is a focused PyTorch implementation of ``C_WristbandGaussianLoss``
    from mvparakhin/ml-tidbits.  It retains the reflected joint repulsion,
    radial OT, W2 moment term, and null calibration used by the reference loss.
    The pairwise path is intentionally capped to a deterministic sample so it
    remains practical for the many claim states in a patient batch.
    """

    def __init__(
        self,
        embedding_dim: int,
        sample_size: int = 128,
        calibration_reps: int = 16,
        beta: float = 8.0,
        lambda_rad: float = 0.1,
        lambda_mom: float = 1.0,
        calibration_seed: int = 1729,
    ):
        super().__init__()
        if sample_size < 2 or calibration_reps < 2:
            raise ValueError("Wristband calibration needs at least two samples and repetitions.")
        self.embedding_dim = int(embedding_dim)
        self.sample_size = int(sample_size)
        self.beta = float(beta)
        self.alpha = math.sqrt(1.0 / 12.0)
        self.lambda_rad = float(lambda_rad)
        self.lambda_mom = float(lambda_mom)
        self.epsilon = 1e-12

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(calibration_seed))
        calibration = []
        with torch.no_grad():
            for _ in range(int(calibration_reps)):
                null = torch.randn(
                    self.sample_size,
                    self.embedding_dim,
                    generator=generator,
                )
                calibration.append(torch.stack(self._raw_components(null)))
        calibration_tensor = torch.stack(calibration)
        means = calibration_tensor.mean(dim=0)
        stds = calibration_tensor.std(dim=0, unbiased=True).clamp_min(1e-6)
        normalized = (calibration_tensor - means) / stds
        total = (
            normalized[:, 0]
            + self.lambda_rad * normalized[:, 1]
            + self.lambda_mom * normalized[:, 2]
        )
        self.register_buffer("component_means", means)
        self.register_buffer("component_stds", stds)
        self.register_buffer("total_std", total.std(unbiased=True).clamp_min(1e-6))

    def _w2_moment(self, x: torch.Tensor) -> torch.Tensor:
        n, d = x.shape
        mean = x.mean(dim=0, keepdim=True)
        centered = x - mean
        if d <= n:
            covariance = centered.T @ centered / float(n - 1)
        else:
            covariance = centered @ centered.T / float(n - 1)
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        moment = mean.square().sum() + (torch.sqrt(eigenvalues + 1e-7) - 1.0).square().sum()
        if d > n:
            moment = moment + float(d - n)
        return moment / float(d)

    def _radial_ot(self, t: torch.Tensor) -> torch.Tensor:
        sorted_t = t.sort().values
        quantiles = (
            torch.arange(t.numel(), device=t.device, dtype=t.dtype) + 0.5
        ) / float(t.numel())
        eps = max(self.epsilon, float(torch.finfo(t.dtype).eps))
        sorted_t = sorted_t.clamp(eps, 1.0 - eps)
        gaussian_t = math.sqrt(2.0) * torch.erfinv(2.0 * sorted_t - 1.0)
        gaussian_q = math.sqrt(2.0) * torch.erfinv(2.0 * quantiles - 1.0)
        uniform_loss = 12.0 * (sorted_t - quantiles).square().mean()
        gaussian_loss = (gaussian_t - gaussian_q).square().mean()
        return 0.5 * (uniform_loss + gaussian_loss)

    def _raw_components(self, x: torch.Tensor):
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        radius_sq = work.square().sum(dim=-1)
        directions = work * torch.rsqrt(radius_sq.clamp_min(self.epsilon)).unsqueeze(-1)
        degree = radius_sq.new_tensor(0.5 * float(work.size(-1)))
        tiny = float(torch.finfo(radius_sq.dtype).tiny)
        radial_cdf = torch.special.gammainc(degree, 0.5 * (radius_sq + tiny))

        gram = (directions @ directions.T).clamp(-1.0, 1.0)
        angular_exponent = (2.0 * self.beta * self.alpha**2) * (gram - 1.0)
        angular_exponent.diagonal().zero_()
        column = radial_cdf[:, None]
        row = radial_cdf[None, :]
        reflected_differences = (column - row, column + row, column + row - 2.0)
        kernel_sum = sum(
            torch.exp(angular_exponent - self.beta * difference.square()).sum()
            for difference in reflected_differences
        ) - float(work.size(0))
        denominator = 3.0 * float(work.size(0) ** 2) - float(work.size(0))
        repulsion = torch.log(kernel_sum / denominator + self.epsilon) / self.beta
        return repulsion, self._radial_ot(radial_cdf), self._w2_moment(work)

    def forward(self, embeddings: torch.Tensor):
        if embeddings.ndim != 2:
            embeddings = embeddings.reshape(-1, embeddings.size(-1))
        if embeddings.size(0) < 2:
            zero = embeddings.new_zeros(())
            return zero, {"repulsion": zero, "radial": zero, "moment": zero}
        if embeddings.size(0) > self.sample_size:
            indices = torch.linspace(
                0,
                embeddings.size(0) - 1,
                steps=self.sample_size,
                device=embeddings.device,
            ).round().long()
            embeddings = embeddings.index_select(0, indices)
        elif embeddings.size(0) < self.sample_size:
            # Calibration is fixed-size; repeat deterministically rather than
            # introduce a batch-dependent null scale.
            repeats = math.ceil(self.sample_size / embeddings.size(0))
            embeddings = embeddings.repeat(repeats, 1)[: self.sample_size]

        raw = torch.stack(self._raw_components(embeddings))
        normalized = (raw - self.component_means.to(raw)) / self.component_stds.to(raw)
        total = (
            normalized[0]
            + self.lambda_rad * normalized[1]
            + self.lambda_mom * normalized[2]
        ) / self.total_std.to(raw)
        return total, {
            "repulsion": normalized[0],
            "radial": normalized[1],
            "moment": normalized[2],
        }


def build_ssl_objective(config) -> BaseSSLObjective:
    objective_type = getattr(config, "ssl_objective_type", "vicreg").lower()
    if objective_type == "vicreg":
        return VICRegObjective(config)
    if objective_type == "sigreg":
        return SIGRegObjective(config)
    if objective_type == "rdmreg":
        return RDMRegObjective(config)
    raise ValueError(f"Unsupported ssl_objective_type={objective_type!r}")
