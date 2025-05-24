# agents.md

## 🧠 Overview

This project implements a **modular JEPA-style architecture** for healthcare claims modeling. It supports prediction, regression, and token generation across CPT, ICD, and TTNC code types, using either **adversarial** or **diffusion-based** generation paths.

The model is configurable via `Config`, supports multi-loss optimization (VICReg, BCE, MSE, diffusion), and includes attention-based aggregation, token rarity weighting, and sparse autoencoding.

---

## 🤖 Agent Roles

### 🗂️ Project Coordinator Agent
- Maintains repo organization and Python import consistency.
- Ensures separation of:
  - `jepa_models/` → model definitions (encoders, diffusion, hierarchical)
  - `jepa_utils/` → preprocessing, tensor ops, config
  - `scripts/` → training, generation, and evaluation logic

---

### 🔍 Data Agent
- Responsible for:
  - Preprocessing (`preprocessing.py`)
  - Vocabulary construction and rarity scoring
  - Dataset creation (`dataset.py`)
- Key constraints:
  - Token types: `cpt_`, `icd_`, `ttnc_`
  - Minimum claims per patient: `min_ttnc_tokens`
  - Target transformation via `log1p` and optional capping

---

### 🧬 Encoder Agent
- Owns:
  - `Level1Encoder`: Aggregates individual code types (mean, max, min)
  - `Level2Encoder`: Aggregates full claims with attention across token-level, component-level, and statistical aggregates
- Must honor config flags for:
  - `use_code_attention`
  - `use_token_rarity`
  - `use_component_attention`, `use_aggregate_attention`
  - `use_variance_embeddings`

---

### 🧠 Predictor Agent
- Defines:
  - `Level1PredictionBlock`: Predicts CPT ↔ ICD claim components
  - `Level2PredictionBlock`: Predicts patient-level claim representation
- Supports:
  - GRU, LSTM, Transformer via `rnn_type`
  - Choice of patient representation: mean-only or mean+max (`use_context_pooled_patient_representation`)
  - Outputs fed into task heads and generator(s)

---

### 🎯 Generator Agent

#### 🔵 GAN Generator
- Class: `LogitsGenerator`
- Predicts logits over vocabularies
- Optionally supervised via `BCEWithLogitsLoss` and discriminator adversarial training

#### 🟢 Diffusion Generator
- Class: `DiffusionModel`
- Implements denoising diffusion probabilistic model
- Supports optional conditioning via `condition_proj`
- Pretraining triggered by `pretrain_diffusion=True`, joint training scaled by `diffusion_weight`

Codex agents must respect `use_diffusion` config toggle. Never invoke both generators without checking.

---

### 🧪 SAE Agent
- File: `sparse_autoencoder.py`
- Provides a Top-K sparse autoencoder for compressing claim or patient-level embeddings
- Controlled via:
  - `use_sparse_autoencoder`
  - `sae_hidden_dim`
  - `sae_k` (number of top-k activations)

---

### 🔧 Training Agent
- File: `scripts/train.py`
- Must handle:
  - Conditional training phases: diffusion-only pretraining or joint JEPA training
  - Logging of all relevant losses (VICReg, BCE, adversarial, diffusion)
  - Final evaluation: `.csv` output of generated vs actual claims
- Lightning callbacks include:
  - `ModelCheckpoint`
  - `RichProgressBar`
  - `RichModelSummary`

Additional config flags:
  - `pretrain_diffusion_epochs`: number of epochs to pretrain diffusion
  - `freeze_transferred_embeddings`: freeze copied embeddings for initial epochs
  - `encoder_unfreeze_layers`: layers left trainable when encoders are frozen
  - `debug_low_threshold`: sets `threshold` to `0.05` and disables entropy when true

### 🏋️ Training Schedule

Stage 1 – Representation Pre‑Train

Goal: learn patient/claim embeddings with VICReg‑L2 (next‑claim prediction) and Sparse Auto‑Encoder (top‑K reconstruction).

Encoder: unfrozen.

Heads active: VICReg‑L2, SAE.

Config ON: use_level2_vicreg = True, level_2_weight = 1.0, use_sparse_autoencoder = True.

Config OFF: use_token_prediction_head = False, use_diffusion = False.

Stage 2 – Diffusion Generator

Goal: keep encoder fixed; train Diffusion + Token heads to output CPT / ICD / TTNC codes.

Checkpoint: load checkpoints/encoder_only.ckpt (or path from --pretrained_encoder_ckpt).

Encoder freeze: call freeze_encoder(except_last_n_layers = 1); optional adapter layers LR ≈ 1 e‑4.

Heads active: token prediction, diffusion (diffusion_weight ≈ 0.1).

Config overrides: use_token_prediction_head = True, use_diffusion = True, level_2_weight = 0.0.

Stage 3 – Joint Fine‑Tune (optional)

Goal: light co‑adaptation once generator has learned.

Encoder: unfreeze; tiny encoder LR (≈ 2 e‑5) while keeping generator LR higher.

Heads: all losses active.

Checkpoint flow

Stage 1 must save encoder_only.ckpt (path overridable via --out_encoder_ckpt).

Stage 2 refuses to start unless that file—or the path in --pretrained_encoder_ckpt—is found.

⚙️ Freezing & Optimisers

Helper freeze_encoder(except_last_n_layers=1) is only invoked when current_stage == "stage2".

Stage 2 builds two optimiser groups:

adapter_or_last_layers – LR ≈ 1e‑4

generator_params – LR ≈ 5e‑4

📏 Loss Weights & Log‑Vars

Initialise all learnable log‑variances at 0.0 and clamp to [‑5, 5] each forward pass.

Stage‑specific weights:

Stage 1 → level_2_weight = 1.0

Stage 2 → level_2_weight = 0.0

📊 Metrics

Validation RMSE is computed only when use_predictor_head = True; otherwise the metric is skipped to avoid NaNs.

Log both raw and precision‑weighted VICReg‑L2 once per epoch: vicreg_lvl2_raw, vicreg_lvl2_wgt.

✅ Acceptance Checklist for Agents

Stage 1 finishes with non‑zero vicreg_lvl2_wgt & SAE losses and saves encoder_only.ckpt.

Stage 2 loads that checkpoint, freezes encoder, and logs vicreg_lvl2_wgt ≈ 0 (or very small) while diffusion/token losses decrease.

No console spam from one‑off debug prints; logs remain clean.