# Stage-1 Representation Pretraining Notes

This phase trains the encoders using VICReg and the optional sparse autoencoder.

## Valid Claim Mask
A claim counts as valid if **any** component contains a real token:
- at least one CPT code,
- or at least one ICD code,
- or a TTNC token that isn't `<PAD>`.
If an entire patient has no valid claims after masking, the final row is forced
to valid so the embedding never collapses to all zeros.

## Embedding Health Check
After the patient representation is computed (and optionally fused with the SAE),
the mean absolute magnitude is checked. Values below `1e-6` trigger a runtime
error, catching masking bugs early. The average magnitude is logged once per
epoch as `embedding_mag`.

## SAE Settings
`sae_k` is reduced to roughly one quarter of the embedding dimension to maintain
sparsity pressure. SAE precision is initialized at zero log-variance so its loss
carries normal weight.
