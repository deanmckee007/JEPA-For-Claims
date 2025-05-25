import torch
from jepa_utils.config import Config
from jepa_models.discrete_diffusion import DiscreteDiffusionModel


def main():
    cfg = Config()
    cfg.cpt_vocab_size = 10
    cfg.icd_vocab_size = 10
    cfg.ttnc_vocab_size = 5
    cfg.embedding_dim = 8
    cfg.max_cpt_tokens = 2
    cfg.max_icd_tokens = 2
    cfg.diffusion_steps = 5
    model = DiscreteDiffusionModel(cfg)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(5):
        cpt = torch.randint(0, cfg.cpt_vocab_size, (4, cfg.max_cpt_tokens))
        icd = torch.randint(0, cfg.icd_vocab_size, (4, cfg.max_icd_tokens))
        ttnc = torch.randint(0, cfg.ttnc_vocab_size, (4,))
        loss = model.p_losses(cpt, icd, ttnc, torch.zeros(4, dtype=torch.long))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"step {step} loss {loss.item():.4f}")


if __name__ == "__main__":
    main()
