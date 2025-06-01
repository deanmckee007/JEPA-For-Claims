## 🧠 Project Overview

This repository implements a **Joint Embedding Predictive Architecture (JEPA)** for healthcare claims **plus a discrete‑diffusion generator** for high‑fidelity synthetic claims.

* **Context/Target Encoders (JEPA)** — learn patient & claim representations with VICReg, sparse auto‑encoding, and next‑claim prediction.
* **Claim‑level D3PM Generator** — per‑claim discrete diffusion (100 steps, classifier‑free guidance) that conditions on the patient embedding and jointly trains with JEPA.
* **Multi‑loss Training Schedule**

  1. **Phase 1**: Diffusion‑only pre‑train until convergence.
  2. **Phase 2**: Joint fine‑tune with JEPA (uncertainty‑weighted losses).

All components are toggled via `Config` flags so agents can compose pipelines without hard‑coding.

---

## 🤖 Agent Responsibilities

| Agent                   | Core Files                       | Duties                                                                                        |
| ----------------------- | -------------------------------- | --------------------------------------------------------------------------------------------- |
| **Project Coordinator** | *repo root*                      | Maintain folder structure, config propagation, import hygiene.                                |
| **Data Agent**          | `preprocessing.py`, `dataset.py` | Parse raw claims → token sequences; build vocab; rarity scores; dataset splits.               |
| **Encoder Agent**       | `encoders.py`                    | Level‑1 (within‑claim) & Level‑2 (across‑claims) encoders; apply token weighting & attention. |
| **Diffusion Agent**     | `models/diffusion/claim_d3pm.py` | Implement noise scheduler, denoiser, conditioning; expose `generate_claim()`.                 |
| **Predictor Agent**     | `prediction_blocks.py`           | Next‑claim / token heads; integrates JEPA outputs & diffusion samples.                        |
| **SAE Agent**           | `sparse_autoencoder.py`          | Top‑K sparse auto‑encoder for representation compression.                                     |
| **Training Agent**      | `scripts/train.py`, `trainers/`  | Stage‑aware training loop, checkpoint orchestration, metric logging.                          |

---

## 🏋️ Training Phases

1. **Phase 1 – Diffusion Pre‑train**
   Flag: `pretrain_diffusion=True`.  Train `ClaimD3PM` with cross‑entropy; save `diffusion_only.ckpt`.
2. **Phase 2 – Joint JEPA + Diffusion**
   Load encoder & denoiser; enable VICReg, SAE, diffusion; uncertainty‑weighted log‑vars.

Optional **Phase 3** — hierarchical diffusion (sequence‑level) once claim‑level is stable.

---

## 🔑 Key Config Flags

```yaml
use_diffusion: true            # enable Claim‑level D3PM
pretrain_diffusion: true       # run Phase 1
freeze_denoiser_core: false    # allow gradients into denoiser after Phase 1
Guidance_scale: 4.0            # classifier‑free guidance γ
Diffusion_steps: 100           # discrete timesteps T
```

---

## ✅ Agent Check‑list

* **Data Agent** builds vocab with `<PAD>`, `<CLS_CLAIM>`, `<SEQ_END>`.
* **Diffusion Agent** can round‑trip a claim (noise → denoise) with <5% token error on validation.
* **Training Agent** logs decreasing diffusion CE in Phase 1 and balanced multi‑loss in Phase 2.
* **Predictor Agent** uses synthetic claims during training when `use_diffusion` is true.

---

For historical design notes and phase details see `implementation_plans/diffusion_implementation.py`.
