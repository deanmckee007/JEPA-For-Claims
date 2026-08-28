# Invalidated run

Do not use the metrics in this directory.

The first full-cohort run included two single-claim patients for which the probe
extractor emitted padding as the supervised TTNC target. Because padding is
masked out of the TTNC output distribution, this produced a constant loss of
approximately `4.5e34`.

The extractor was repaired to require at least one observed historical claim
and one held-out target claim. Valid replacement results are in:

`experiments/frozen_generation_probe_20260818_context_seed42/`
