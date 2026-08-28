# JEPA-For-Claims

Joint Embedding Predictive Architecture experiments for healthcare claims, with
within-claim and across-claim representations. The conceptual starting point is
the [original JEPA paper](https://arxiv.org/abs/2301.08243).

This repository is best treated as a research reference, not a drop-in claims
library. Its most reusable output is the set of experimentally tested design
patterns, controls, and failure modes. An agent adapting the work to another
project should translate those findings to the local data model, endpoint,
operating point, and production constraints rather than preserve these classes
or configurations verbatim.

## General findings

These conclusions were observed on frozen validation splits and, where noted in
the linked reports, replicated across encoder and downstream seeds. They are
hypotheses with unusually good local evidence—not universal constants.

| Finding | General lesson | Implementation consequence |
|---|---|---|
| Pretraining helps most when labels are scarce | The value of a pretrained sequence representation can disappear when a strong supervised learner has abundant labels | Measure a label-efficiency curve instead of reporting only a full-label score |
| Strong raw-history baselines are mandatory | Boosted trees over carefully engineered sparse history can beat much more sophisticated encoders on ordinary cost regression | Compare against exact sparse features and nonlinear trees, not only a mean predictor or linear probe |
| Endpoint choice can reverse the conclusion | Average cost error and concentration in the highest-risk 1--2% reward different information | Predeclare the selection budget and report precision, lift, ranking quality, and captured outcome at that budget |
| Learned and engineered features are complementary | Representations often add useful nonlinear or rank signal without winning as standalone features | Test feature fusion and percentile-rank blending, but select them on broad ranking quality rather than one lucky cutoff |
| Patient-view invariance improves missing-history robustness | Training related partial-history views to agree can make the patient state degrade more gracefully when old events are absent | Construct views that preserve the prediction boundary and at least the latest context event; never corrupt the held-out target |
| Robustness and decodability can trade off against low-label performance | The patient-view recipe improved robustness and full-label CPT/ICD decoding but was not uniformly best for low-label CPT, TTNC, or cost | Keep multiple representation recipes when endpoints differ; avoid declaring a single universal encoder winner |
| Sequence attention is not automatically a better downstream readout | A learned attentive probe underperformed simple pooled embeddings in the tail-ranking experiments | Treat attention as an ablation and require gains across seeds and ranking metrics |
| Online and averaged weights may serve different endpoints | Polyak weights were better for some global representation metrics, while online weights ranked the cost tail better | Evaluate the exact checkpoint state that will serve each downstream task |
| Sparse and dense states can contain orthogonal signal | Sparse+dense fusion sometimes improved embedding-only cutoff retrieval, but degraded broader ranking once strong raw features were present | Retain sparsity only when it adds stable incremental value; sparsity itself is not evidence of interpretability or utility |
| Generation should exploit persistence | The last observed claim was a strong baseline; copy-plus-residual decoding was more useful than independently synthesizing a whole claim | Copy persistent tokens and predict residual changes, cardinality, and timing; compare against copy-only |
| Flat add/remove prediction is often too sparse | Explicit vocabulary-wide change events produced weak and inconsistent gains | Retrieve a small candidate set before classification or generation rather than scoring every possible token equally |
| Auxiliary targets can help cost without fully mediating it | Next-claim supervision improved low-label cost prediction, but much of the gain survived shuffled labels and label-free regularization | Include shuffled-target, random-feature, and label-free controls before attributing improvement to semantics |
| Smoothness needs anti-collapse protection | Consistency regularization improved cost but collapsed hidden geometry until paired with the existing SigReg objective | Track feature variance and norm; use an anti-collapse objective instead of trusting downstream error alone |

Three practical results illustrate the pattern:

- With roughly 1% of cost-tail labels, raw history plus the patient-view
  representation more than doubled top-1.5% precision over raw history alone.
- With a nominal 70% of older context claims removed, the patient-view
  representation preserved its top-tail precision while the comparison encoder
  degraded.
- With full decoder supervision, the patient-view representation improved both
  CPT and ICD average precision across all three encoder seeds, while another
  representation remained better for TTNC and slightly better for cost MAE.

The detailed evidence and caveats are in
[the frozen-generation report](docs/frozen_generation_probe_20260818.md),
[the latent-prediction study](docs/lpwm_claims_pilot_20260826.md), and
[the patient-view study](docs/levjepa_patient_views_20260828.md).

## How to transfer the work

For a new domain, preserve the experimental logic before preserving the model:

1. Define a composite event, its temporal context, and a strictly future target.
2. Freeze entity IDs, dates, vocabularies, transforms, and splits in a data
   contract before comparing objectives.
3. Establish exact sparse, boosted-tree, persistence, shuffled-label, and random
   encoder controls.
4. Pretrain representations without downstream labels, then evaluate frozen
   probes at several label budgets.
5. Measure the operational endpoint directly. Aggregate RMSE is not a substitute
   for top-budget retrieval, calibration, or event-set quality.
6. Repeat promising results across encoder seeds, not only downstream-head seeds.
7. Promote the smallest mechanism that survives those controls.

The claims components here are procedures, diagnoses, and time-to-next-claim.
An adaptation should introduce the event components that are actually relevant
to its inference task.

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

The architecture is not specific to claims. The same experiments apply to a
sequence of composite events—for example, transactions containing items and
actors, visits containing observations and interventions, or sessions containing
heterogeneous actions.

## Diffusion-based Generation

Diffusion remains an experimental generator path, not the currently recommended
proof of concept. Frozen copy-plus-residual decoding provided a cleaner test of
whether pretrained embeddings contain downstream next-event information. Revisit
discrete diffusion only after a sparse candidate decoder has a calibrated,
reproducible advantage.

Set `use_diffusion = True` in `Config` to enable the implementation. Leaving it
`False` disables diffusion-based claim synthesis.

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
the SAE output influences the dense claim embedding. The mechanism is designed
to improve early training stability by letting the pretrained SAE guide the
hierarchical encoders. Whether it actually helps must be established with a
dense control; the experiments in this repository found cutoff-specific
complementary signal, not a general sparse-representation win.

## Diffusion Generator

Set `use_diffusion = True` in `Config` to train and sample from a denoising diffusion model.
The diffusion generator produces CPT, ICD and TTNC tokens and can replace the
heuristic generator during inference.
The training script performs a short pretraining phase for the diffusion
generator by default.  Disable this by setting `pretrain_diffusion = False` in
`Config`.  If you are sharing embeddings across modalities, skip this phase to
avoid interfering with the shared weights.

Diffusion support replaced the old GAN implementation, but is not enabled by
default. Set `use_diffusion = False` to disable it.
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
  - `DiffusionModel.sample()` denoises from Gaussian noise to discrete code embeddings.
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
  4. Alternatively run `python scripts/generate_claims.py <ckpt>` to export predictions with actual codes.

### Investigation: Why No CPT/ICD Codes in predictions.csv
If `predictions.csv` is empty, verify the following:

1. **Stage 2 config** – confirm `use_diffusion=true` and `use_token_prediction_head` is set as intended.
2. **Predict step logic** – dump raw CPT/ICD and TTNC logits during `predict_step` to ensure they exceed your thresholds.
3. **Threshold & decoding** – check the multi-label threshold or top-K logic, printing selected indices per sample.
4. **Diffusion sampling** – instrument `DiscreteDiffusionModel.sample()` to log denoising steps and final token IDs.
5. **Gating network** – log `gating_sae_fraction` each prediction. Force it to zero temporarily to test diffusion only.
6. **CSV writer** – trace token IDs to file rows and ensure no default empty token is inserted when none are found.
7. **Quick experiments** – run with token-only (`use_token_prediction_head=true`, `use_diffusion=false`) or diffusion-only to isolate issues.

