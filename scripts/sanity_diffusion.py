import torch
from jepa_utils.config import Config
from models.diffusion import ClaimD3PM


def main():
    cfg = Config()
    cfg.cpt_vocab_size = 10
    cfg.icd_vocab_size = 10
    cfg.ttnc_vocab_size = 5
    cfg.embedding_dim = 8
    cfg.max_cpt_tokens = 2
    cfg.max_icd_tokens = 2
    cfg.diffusion_steps = 5
    vocab_size = cfg.cpt_vocab_size + cfg.icd_vocab_size + cfg.ttnc_vocab_size + 3
    seq_len = cfg.max_cpt_tokens + cfg.max_icd_tokens + 1
    model = ClaimD3PM(cfg, vocab_size, cfg.embedding_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(5):
        tokens = torch.randint(0, vocab_size, (4, seq_len))
        t = torch.zeros(4, dtype=torch.long)
        loss = model(tokens)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"step {step} loss {loss.item():.4f}")


if __name__ == "__main__":
    main()
