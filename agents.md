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

1. **Stage 1 – Representation Pretrain**
   - `use_token_prediction_head = False`
   - `use_diffusion = False`
   - Stop when VICReg or SAE loss plateaus.

2. **Stage 2 – Generator Training**
   - Load `encoder_only.ckpt` and freeze encoders using `freeze_encoder(except_last_n_layers=1)`.
   - Keep only adapter layers trainable at **LR ≈ 1e-4** while token head, TTNC classifier and diffusion use **LR ≈ 5e-4**.
   - Perform a one-time gradient check to verify that frozen layers report `grad == None`.
   - Enable `use_token_prediction_head` and `use_diffusion` with `diffusion_weight ≈ 0.1-0.2`.
   - Apply an LR scheduler (e.g. `StepLR(gamma=0.5, step_size=2)`) during this stage.

3. **Stage 3 – Joint Fine‑Tune (optional)**
   - `unfreeze_encoders()` and train all losses with a lower encoder LR.

