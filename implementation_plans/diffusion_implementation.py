"""diffusion_implementation.py ─ High‑level blueprint for claim‑level D3PM **and repo cleanup**
================================================================================
This file is NOT executable; Codex agents use it as the single source of truth
for implementing, refactoring, and unifying discrete‑diffusion with JEPA.

───────────────────────────────────────────────────────────────────────────────
1.  File Layout (post‑cleanup)
    • models/diffusion/claim_d3pm.py       ← core noise scheduler + denoiser (✓)
    • trainers/diffusion_trainer.py        ← Phase‑1 pre‑train loop (✓)
    • scripts/generate_claims.py           ← sampling CLI util  (✓)
    • eval/diffusion_fidelity.py           ← KS / JSD report   (✓)
    • scripts/train.py                     ← unified Phase‑1 / Phase‑2 runner

2.  Discrete Noise Process (D3PM)
    • Vocab  = CPT | ICD | TTNC | <PAD> | <CLS_CLAIM> | <SEQ_END>
    • Steps  = 100; learnable log‑channel matrix Q_t  (see ClaimD3PM)

3.  Denoiser Architecture
    • Transformer Encoder 4× layers, width = embedding_dim
    • Timestep embedding + FiLM conditioning from patient embedding z
    • Output proj → vocab logits; CE loss.

4.  Training Phases
    Phase‑1  (diffusion_pretrain) → trainer.fit(ClaimD3PM, dataloader)
    Phase‑2  (joint)              → JEPA + diffusion CE with log‑var weighting

5.  Config Additions (YAML)
    diffusion:
        steps: 100
        guidance_scale: 4.0
        pretrain_epochs: 20
        freeze_denoiser_core: false
        diffusion_weight_init: 0.3   # relative to vicreg weight

6.  Sampling Pipeline (generate_claims.py)
    1. Encode patient → z
    2. x_T ← Uniform[0, vocab)
    3. for t=T…1: logits = uncond + γ*(cond‑uncond); x_{t‑1} ← Cat(logits)
    4. Detokenise codes.

7.  Evaluation Metrics (diffusion_fidelity.py)
    • KS statistic and Jensen‑Shannon divergence on code distributions.
    • Callable: diffusion_fidelity(real_codes, synth_codes) → {ks, jsd}

8.  Acceptance Criteria
    ✓ Phase‑1 val CE ≤ 0.4 by epoch 20.
    ✓ JSD ≤ 0.05 between synthetic and real claim code freq.
    ✓ Downstream AUROC drop ≤ 2 pts vs real‑trained baseline.

9.  **Migration & Clean‑up Tasks**  (Codex TODO list)
    ──────────────────────────────────────────────────────────────
    A. Deprecate legacy diffusion
       • Delete: diffusion.py, discrete_diffusion.py, DiscreteDiffusionModel class.
       • Remove old flags (`use_diffusion_legacy`, etc.) from config.
    B. Consolidate training workflow
       • scripts/train.py: implement --phase {pretrain,joint}.  Auto‑detect
         if ClaimD3PM checkpoint exists; skip Phase‑1 when resuming.
       • Hook diffusion loss into JEPA’s uncertainty‑weighted loss stack.
    C. Update imports & namespace
       • All modules should import ClaimD3PM from
         `models.diffusion.claim_d3pm`.
       • Provide one‑version shim (from jepa_models.diffusion_models import
         ClaimD3PM) that raises DeprecationWarning.
    D. Documentation refresh
       • agents.md → reflect single diffusion path (already partly done).
       • README + docs/diffusion.md → quick‑start commands:
           ```bash
           python scripts/train.py --phase pretrain
           python scripts/train.py --phase joint --resume diffusion_only.ckpt
           python scripts/generate_claims.py checkpoints/…
           ```
    E. Unit tests
       • tests/test_diffusion_roundtrip.py: ensure <5% token error over 64
         random claims.
       • tests/test_sampling_speed.py: generate 128 claims < 2 s on CPU.
    F. CI integration
       • Add GitHub Action `diffusion-ci.yml` running unit tests on PRs.
    G. Version bump
       • Update `__version__` to `0.4.0` (minor feature + cleanup).

"""
