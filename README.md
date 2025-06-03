# JEPA-For-Claims
Joint Embedding Predictive Architecture for healthcare claims.  A hierarchical approach using within-claims representations and across-claims representation.  Original JEPA paper - https://arxiv.org/abs/2301.08243

Conceptually, the goal here is to generate high quality embeddings for a variety of downstream tasks.  I have a simple prediction head attached that's toggled on/off in config and that should be fine for specializing to a task or expanding to multi-task.  The claims components here are limited to procedures and diagnoses, but anyone implementing this should introduce all of the components relevant for their inference/prediction tasks.

## Setup

Install the dependencies with:
```bash
pip install -r requirements.txt
```

## Quick-start debug

If CPT or ICD predictions look empty, toggle a helper flag:

```python
config.debug_low_threshold = True
```

This sets the model's generation threshold to `0.05` and disables the
entropy term so logits fire during sanity checks. Remember to set it
back to `False` before real training.

# What can I do with this thing?
Self-supervised models shine where you have a ton of unlabeled data and you want to maximize value from the labeled data you do have.
Claims often have missing/incorrect entries.  Commercial directories and downloads from CMS are often wrong or simply outdated.
Where we have labels we can trust, we can do better.
JEPA for claims allows us to extract representations at a variety of abstractions -
So, want to infer a provider's specialty?  Extract a level 1 provider embedding (within claims) because that captures the procedures and diagnoses providers do.  Optionally also include level 2 provider embeddings since the claims up and downstream from a provider introduce temporal contextual information.  Inferring referring provider is a more obvious use case for level 2 representations.  These features are now the input to the supervised learning model of your choice (or prediction head on this model).

*New*

I've extended this to do claims generation that can be inspected via csv.  It actually works surprisingly well given the smallish training data (12k obs).  The old GAN implementation was removed to keep things simple.

Another interesting point is that although I'm doing claims here, this approach can be conceptually applied to pretty much any sequence-of-composite-entity problems.

## Diffusion-based Generation

The generator now uses a diffusion model instead of the old GAN approach.
Enable it by setting `use_diffusion = True` in your `Config`.
Leaving it `False` will disable diffusion-based claim synthesis.

# Notes for use
I have options to toggle all of the level 2 attentional transformations on/off.
For the level 2 prediction block you can choose between GRU, LSTM, and Transformer.  GRU should work well with sequences up to ~50, consider using LSTM beyond that, and transformer if you're feeling spicy.
I'm using my personal laptop for training (yes, yes, I know) - with a GeForce GTX 1660.  It's only got 6GB of memory and I'm able to run with 128 batch size with up to about 125 claims as a context window.

## Sparse Autoencoder with Gated Fusion

JEPA can optionally leverage a Sparse Autoencoder (SAE) to pretrain claim level
representations. 

To fuse the SAE representation with the JEPA encoders set a gating option in the
configuration.  Set `use_gated_fusion = True` in your `Config` dataclass and the
model will instantiate a multi-layer gating network that blends the SAE output with
the contextual embedding from JEPA.  The hidden size is controlled by
`gating_hidden_dim`.  The average gating weight is logged each epoch so you can
monitor how much the model relies on the SAE representation.  An additional
metric, `gating_sae_fraction`, reports the L2 norm of the SAE contribution
relative to the combined representation, giving a clearer picture of how much
the SAE output influences the dense claim embedding. This mechanism improves
training stability by letting the pretrained SAE guide the hierarchical
encoders during early epochs.

## Diffusion Generator

Set `use_diffusion = True` in `Config` to train and sample from a denoising diffusion model.
The diffusion generator produces CPT, ICD and TTNC tokens and can replace the
heuristic generator during inference.
The training script performs a short pretraining phase for the diffusion
generator by default.  Disable this by setting `pretrain_diffusion = False` in
`Config`.  If you are sharing embeddings across modalities, skip this phase to
avoid interfering with the shared weights.

Diffusion support has fully replaced the old GAN implementation. Set `use_diffusion = False` if you want to disable claim synthesis.

### Command‑line Quick Start

Run the diffusion pretrain and joint phases sequentially:

```bash
python scripts/train.py --phase pretrain
python scripts/train.py --phase joint --resume diffusion_only.ckpt
```

Once training completes you can sample synthetic claims with:

```bash
python scripts/generate_claims.py joint.ckpt --num 10
```
During `--phase diffusion_only_finetune`, if the `--resume` checkpoint is a joint
file the loader automatically freezes JEPA parameters and fine-tunes the
diffusion module in place.
## Multi-Stage Training

### Stage 1 – Self-Supervised Representation Pre-Train
Learn rich patient embeddings using VICReg-L2 and a sparse auto-encoder.

- **Compute**
  - **VICReg-L2**: predict the next-claim Level-2 embedding from the current context.
  - **Sparse Auto-Encoder (SAE)**: reconstruct the same Level-2 embedding via a top-K bottleneck.
- **Trainable modules**
  - Context & target Level-2 encoders
  - SAE encoder/decoder & gating network
- **Frozen**
  - Token-prediction head
  - Diffusion model
- **Key config**
  ```yaml
  use_level2_vicreg: true
  level_2_weight: 1.0
  use_sparse_autoencoder: true
  sae_weight: 1.0
  use_token_prediction_head: false
  use_diffusion: false
  current_stage: "stage1"
  ```
- **Checkpoint**: save `encoder_only.ckpt` at epoch end.

### Stage 2 – Diffusion-Based Code Generation
Keep the learned representations fixed and train the generator to emit CPT/ICD/TTNC codes.

- **Compute**
  - Encoders & SAE perform a forward pass only to supply representations.
  - `ClaimD3PM.generate_claim()` denoises from uniform tokens to discrete codes.
  - Optionally a token head can directly output logits.
- **Trainable modules**
  - Diffusion model & its projection layers
  - Token-prediction head (if enabled)
  - Gating network adapters
- **Frozen modules & losses**
  - Context/target encoders and SAE (`requires_grad=False`)
  - VICReg-L2 and SAE losses disabled (`level_2_weight=0.0`, `sae_weight=0.0`)
- **Key config**
  ```yaml
  use_sparse_autoencoder: true      # compute but don’t train SAE
  sae_weight: 0.0
  use_level2_vicreg: false
  level_2_weight: 0.0
  use_token_prediction_head: [true|false]
  use_diffusion: true
  diffusion_weight: 0.1
  current_stage: "stage2"
  freeze_encoder_at_stage2: true
  ```
- **Inference**
  1. Load `encoder_only.ckpt`.
  2. Call `freeze_encoder(...)`.
  3. Run `model.predict()` to produce `predictions.csv`.

### Investigation: Why No CPT/ICD Codes in predictions.csv
If `predictions.csv` is empty, verify the following:

1. **Stage 2 config** – confirm `use_diffusion=true` and `use_token_prediction_head` is set as intended.
2. **Predict step logic** – dump raw CPT/ICD and TTNC logits during `predict_step` to ensure they exceed your thresholds.
3. **Threshold & decoding** – check the multi-label threshold or top-K logic, printing selected indices per sample.
4. **Diffusion sampling** – instrument `ClaimD3PM.generate_claim()` to log denoising steps and final token IDs.
5. **Gating network** – log `gating_sae_fraction` each prediction. Force it to zero temporarily to test diffusion only.
6. **CSV writer** – trace token IDs to file rows and ensure no default empty token is inserted when none are found.
7. **Quick experiments** – run with token-only (`use_token_prediction_head=true`, `use_diffusion=false`) or diffusion-only to isolate issues.

