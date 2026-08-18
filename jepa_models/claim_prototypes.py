import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def sinkhorn_assignments(logits: torch.Tensor, iterations: int = 3) -> torch.Tensor:
    """Return balanced assignments for one claim-position stratum.

    Rows are claims and columns are prototypes.  Normalizing columns and then
    rows mirrors CAPI's position-wise Sinkhorn-Knopp update: prototype usage is
    balanced within a prediction slot, so relative claim position cannot be a
    shortcut for the pseudo-label.
    """

    if logits.dim() != 2:
        raise ValueError("sinkhorn_assignments expects a [samples, prototypes] tensor.")
    if logits.size(0) == 0:
        return logits

    assignments = torch.exp(logits - logits.max())
    for _ in range(iterations):
        assignments = assignments / assignments.sum(dim=0, keepdim=True).clamp_min(1e-12)
        assignments = assignments / assignments.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return assignments


class ClaimPrototypeObjective(nn.Module):
    """CAPI-style prediction of online composed-claim cluster assignments.

    Target claim representations are always detached.  The clustering head is
    trained against its own balanced assignments, while an independent student
    head maps Level-2 predictions to those assignments.  This preserves the
    shared, no-EMA claim encoder without allowing the prototype auxiliary to
    move its target representation directly.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_prototypes: int,
        *,
        assignment_temperature: float = 0.06,
        clustering_temperature: float = 0.12,
        student_temperature: float = 0.12,
        sinkhorn_iterations: int = 3,
        clustering_weight: float = 1.0,
    ):
        super().__init__()
        self.num_prototypes = int(num_prototypes)
        self.assignment_temperature = float(assignment_temperature)
        self.clustering_temperature = float(clustering_temperature)
        self.student_temperature = float(student_temperature)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.clustering_weight = float(clustering_weight)

        self.clustering_head = nn.Linear(
            embedding_dim,
            self.num_prototypes,
            bias=False,
        )
        self.student_head = nn.Linear(
            embedding_dim,
            self.num_prototypes,
            bias=False,
        )
        nn.init.normal_(self.clustering_head.weight, std=1.0)
        nn.init.normal_(self.student_head.weight, std=0.02)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        sample_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        squeeze_slot = prediction.dim() == 2
        if squeeze_slot:
            prediction = prediction.unsqueeze(1)
            target = target.unsqueeze(1)
        if prediction.dim() != 3 or target.shape != prediction.shape:
            raise ValueError("Prototype inputs must have matching [batch, slots, dim] shapes.")

        batch_size, slot_count, _ = prediction.shape
        if mask is None:
            active = torch.ones(
                batch_size,
                slot_count,
                dtype=torch.bool,
                device=prediction.device,
            )
        else:
            active = mask.bool().reshape(batch_size, slot_count)
        if sample_weights is not None:
            active = active & (sample_weights.reshape(batch_size, slot_count) > 0)

        normalized_prediction = F.normalize(prediction, dim=-1, eps=1e-7)
        normalized_target = F.normalize(target.detach(), dim=-1, eps=1e-7)
        student_logits = self.student_head(normalized_prediction)
        target_logits = self.clustering_head(normalized_target)

        assignments = target_logits.new_zeros(target_logits.shape)
        for slot_index in range(slot_count):
            slot_active = active[:, slot_index]
            if not slot_active.any():
                continue
            assignments[slot_active, slot_index] = sinkhorn_assignments(
                target_logits[slot_active, slot_index].detach()
                / self.assignment_temperature,
                iterations=self.sinkhorn_iterations,
            )

        flat_active = active.reshape(-1)
        flat_assignments = assignments.reshape(-1, self.num_prototypes)[flat_active]
        flat_student_logits = student_logits.reshape(-1, self.num_prototypes)[flat_active]
        flat_target_logits = target_logits.reshape(-1, self.num_prototypes)[flat_active]
        zero = prediction.new_zeros(())
        if flat_assignments.numel() == 0:
            return {
                "total": zero,
                "prediction": zero,
                "clustering": zero,
                "top1_accuracy": zero,
                "top5_accuracy": zero,
                "target_entropy": zero,
                "effective_prototypes": zero,
                "hard_utilization": zero,
                "assignments": assignments.detach(),
                "student_probabilities": F.softmax(
                    student_logits.detach() / self.student_temperature,
                    dim=-1,
                ),
                "active_mask": active,
            }

        prediction_loss = -(
            flat_assignments
            * F.log_softmax(flat_student_logits / self.student_temperature, dim=-1)
        ).sum(dim=-1).mean()
        clustering_loss = -(
            flat_assignments
            * F.log_softmax(flat_target_logits / self.clustering_temperature, dim=-1)
        ).sum(dim=-1).mean()
        total = prediction_loss + self.clustering_weight * clustering_loss

        target_labels = flat_assignments.argmax(dim=-1)
        top1 = (flat_student_logits.argmax(dim=-1) == target_labels).float().mean()
        topk = min(5, self.num_prototypes)
        top5 = (
            flat_student_logits.topk(topk, dim=-1).indices
            == target_labels.unsqueeze(-1)
        ).any(dim=-1).float().mean()
        target_entropy = -torch.xlogy(flat_assignments, flat_assignments).sum(dim=-1).mean()
        marginal = flat_assignments.mean(dim=0)
        effective_prototypes = torch.exp(-torch.xlogy(marginal, marginal).sum())
        hard_utilization = (
            torch.unique(target_labels).numel() / float(self.num_prototypes)
        )

        return {
            "total": total,
            "prediction": prediction_loss,
            "clustering": clustering_loss,
            "top1_accuracy": top1,
            "top5_accuracy": top5,
            "target_entropy": target_entropy,
            "effective_prototypes": effective_prototypes,
            "hard_utilization": prediction.new_tensor(hard_utilization),
            "assignments": assignments.detach(),
            "student_probabilities": F.softmax(
                student_logits.detach() / self.student_temperature,
                dim=-1,
            ),
            "active_mask": active,
        }
