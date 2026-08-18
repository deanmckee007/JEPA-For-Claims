from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CodeGraphData:
    adjacency: torch.Tensor
    adjacency_norm: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    node_type_ids: torch.Tensor
    cpt_slice: slice
    icd_slice: slice
    ttnc_slice: slice
    cpt_vocab_size: int
    icd_vocab_size: int
    ttnc_vocab_size: int
    num_nodes: int
    num_edges: int


def _pair_key(src: int, dst: int) -> tuple[int, int]:
    return (src, dst) if src < dst else (dst, src)


def build_code_graph(
    processed_data: Iterable[list[dict]],
    cpt_vocab: dict[str, int],
    icd_vocab: dict[str, int],
    ttnc_vocab: dict[str, int],
    *,
    include_ttnc: bool = True,
    min_token_id: int = 2,
    temporal_edge_weight: float = 0.0,
) -> CodeGraphData:
    cpt_vocab_size = len(cpt_vocab)
    icd_vocab_size = len(icd_vocab)
    ttnc_vocab_size = len(ttnc_vocab)

    cpt_offset = 0
    icd_offset = cpt_vocab_size
    ttnc_offset = cpt_vocab_size + icd_vocab_size
    num_nodes = cpt_vocab_size + icd_vocab_size + ttnc_vocab_size

    edge_weights: dict[tuple[int, int], float] = defaultdict(float)

    for patient_claims in processed_data:
        previous_nodes: list[int] | None = None
        for claim in patient_claims:
            claim_nodes: list[int] = []

            for token in claim.get("cpt", []):
                token_id = cpt_vocab.get(token, cpt_vocab.get("<UNK>", 1))
                if token_id >= min_token_id:
                    claim_nodes.append(cpt_offset + token_id)

            for token in claim.get("icd", []):
                token_id = icd_vocab.get(token, icd_vocab.get("<UNK>", 1))
                if token_id >= min_token_id:
                    claim_nodes.append(icd_offset + token_id)

            if include_ttnc:
                token_id = ttnc_vocab.get(
                    claim.get("ttnc", ""),
                    ttnc_vocab.get("<UNK>", 1),
                )
                if token_id >= min_token_id:
                    claim_nodes.append(ttnc_offset + token_id)

            claim_nodes = sorted(set(claim_nodes))
            if len(claim_nodes) < 2:
                previous_nodes = claim_nodes or previous_nodes
                continue

            for idx, src in enumerate(claim_nodes):
                for dst in claim_nodes[idx + 1 :]:
                    edge_weights[_pair_key(src, dst)] += 1.0

            if temporal_edge_weight > 0.0 and previous_nodes:
                for src in previous_nodes:
                    for dst in claim_nodes:
                        if src == dst:
                            continue
                        edge_weights[_pair_key(src, dst)] += temporal_edge_weight

            previous_nodes = claim_nodes

    if not edge_weights:
        raise ValueError("Graph builder found no usable code edges.")

    edge_pairs = sorted(edge_weights.keys())
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(
        [edge_weights[pair] for pair in edge_pairs],
        dtype=torch.float32,
    )

    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adjacency[edge_index[0], edge_index[1]] = edge_weight
    adjacency[edge_index[1], edge_index[0]] = edge_weight
    adjacency.fill_diagonal_(1.0)
    degree = adjacency.sum(dim=1).clamp_min(1.0)
    deg_inv_sqrt = degree.pow(-0.5)
    adjacency_norm = deg_inv_sqrt.unsqueeze(1) * adjacency * deg_inv_sqrt.unsqueeze(0)

    node_type_ids = torch.empty(num_nodes, dtype=torch.long)
    node_type_ids[cpt_offset:icd_offset] = 0
    node_type_ids[icd_offset:ttnc_offset] = 1
    node_type_ids[ttnc_offset:] = 2

    return CodeGraphData(
        adjacency=adjacency,
        adjacency_norm=adjacency_norm,
        edge_index=edge_index,
        edge_weight=edge_weight,
        node_type_ids=node_type_ids,
        cpt_slice=slice(cpt_offset, icd_offset),
        icd_slice=slice(icd_offset, ttnc_offset),
        ttnc_slice=slice(ttnc_offset, num_nodes),
        cpt_vocab_size=cpt_vocab_size,
        icd_vocab_size=icd_vocab_size,
        ttnc_vocab_size=ttnc_vocab_size,
        num_nodes=num_nodes,
        num_edges=edge_index.shape[1],
    )


class CodeGraphSSLModel(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        hidden_dim: int,
        num_node_types: int = 3,
    ):
        super().__init__()
        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)
        self.node_type_embeddings = nn.Embedding(num_node_types, embedding_dim)
        self.encoder = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def encode(
        self,
        adjacency_norm: torch.Tensor,
        node_type_ids: torch.Tensor,
        masked_node_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.node_embeddings.weight + self.node_type_embeddings(node_type_ids)
        target_base = base.detach()

        if masked_node_ids is not None and masked_node_ids.numel() > 0:
            base = base.clone()
            base[masked_node_ids] = 0.0

        propagated = adjacency_norm @ base
        latent = self.encoder(propagated + base)
        latent = F.normalize(latent, dim=-1)
        return latent, target_base

    def masked_reconstruction_loss(
        self,
        latent: torch.Tensor,
        target_base: torch.Tensor,
        masked_node_ids: torch.Tensor,
    ) -> torch.Tensor:
        if masked_node_ids.numel() == 0:
            return latent.new_tensor(0.0)
        reconstructed = self.reconstruction_head(latent[masked_node_ids])
        return F.mse_loss(reconstructed, target_base[masked_node_ids])


def train_graph_ssl_embeddings(
    graph: CodeGraphData,
    *,
    embedding_dim: int,
    hidden_dim: int,
    epochs: int = 120,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    mask_ratio: float = 0.2,
    mask_loss_weight: float = 1.0,
    negative_ratio: int = 2,
    max_edges_per_epoch: int = 40000,
    device: str | torch.device = "cpu",
    seed: int = 42,
) -> dict:
    device = torch.device(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    adjacency_norm = graph.adjacency_norm.to(device)
    node_type_ids = graph.node_type_ids.to(device)
    edge_index = graph.edge_index.to(device)
    edge_weight = graph.edge_weight.to(device)

    model = CodeGraphSSLModel(
        num_nodes=graph.num_nodes,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    bce = nn.BCEWithLogitsLoss()

    edge_prob = edge_weight / edge_weight.sum().clamp_min(1e-8)
    num_sample_edges = min(max_edges_per_epoch, graph.num_edges)
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        sampled_edge_ids = torch.multinomial(
            edge_prob.cpu(),
            num_samples=num_sample_edges,
            replacement=graph.num_edges < num_sample_edges,
            generator=generator,
        ).to(device)

        sampled_edge_index = edge_index[:, sampled_edge_ids]
        num_masked_nodes = max(1, int(graph.num_nodes * mask_ratio))
        masked_node_ids = torch.randperm(
            graph.num_nodes,
            generator=generator,
        )[:num_masked_nodes].to(device)

        latent, target_base = model.encode(
            adjacency_norm=adjacency_norm,
            node_type_ids=node_type_ids,
            masked_node_ids=masked_node_ids,
        )

        pos_src = sampled_edge_index[0]
        pos_dst = sampled_edge_index[1]
        pos_scores = (latent[pos_src] * latent[pos_dst]).sum(dim=-1)
        pos_loss = bce(pos_scores, torch.ones_like(pos_scores))

        neg_src = pos_src.repeat_interleave(negative_ratio)
        neg_dst = torch.randint(
            low=0,
            high=graph.num_nodes,
            size=(neg_src.numel(),),
            generator=generator,
        ).to(device)
        same_node_mask = neg_src == neg_dst
        if same_node_mask.any():
            neg_dst[same_node_mask] = (neg_dst[same_node_mask] + 1) % graph.num_nodes
        neg_scores = (latent[neg_src] * latent[neg_dst]).sum(dim=-1)
        neg_loss = bce(neg_scores, torch.zeros_like(neg_scores))

        reconstruction_loss = model.masked_reconstruction_loss(
            latent=latent,
            target_base=target_base,
            masked_node_ids=masked_node_ids,
        )
        loss = pos_loss + neg_loss + mask_loss_weight * reconstruction_loss
        loss.backward()
        optimizer.step()

        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(loss.detach().cpu()),
                "pos_loss": float(pos_loss.detach().cpu()),
                "neg_loss": float(neg_loss.detach().cpu()),
                "reconstruction_loss": float(reconstruction_loss.detach().cpu()),
            }
        )

    model.eval()
    with torch.no_grad():
        latent, _ = model.encode(
            adjacency_norm=adjacency_norm,
            node_type_ids=node_type_ids,
            masked_node_ids=None,
        )
        latent = latent.detach().cpu()

    cpt_embeddings = latent[graph.cpt_slice].clone()
    icd_embeddings = latent[graph.icd_slice].clone()
    ttnc_embeddings = latent[graph.ttnc_slice].clone()

    for embedding_tensor in (cpt_embeddings, icd_embeddings, ttnc_embeddings):
        if embedding_tensor.shape[0] > 0:
            embedding_tensor[0].zero_()
        if embedding_tensor.shape[0] > 2:
            embedding_tensor[1] = embedding_tensor[2:].mean(dim=0)

    return {
        "embedding_dim": embedding_dim,
        "cpt_embeddings": cpt_embeddings,
        "icd_embeddings": icd_embeddings,
        "ttnc_embeddings": ttnc_embeddings,
        "history": history,
        "graph_stats": {
            "num_nodes": graph.num_nodes,
            "num_edges": graph.num_edges,
            "cpt_vocab_size": graph.cpt_vocab_size,
            "icd_vocab_size": graph.icd_vocab_size,
            "ttnc_vocab_size": graph.ttnc_vocab_size,
        },
    }
