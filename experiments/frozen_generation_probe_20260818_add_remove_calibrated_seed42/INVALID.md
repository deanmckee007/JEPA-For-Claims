# Partially invalid experiment

The calibrated event loss is valid, but reconstructed-set F1 and cardinality
metrics in this directory used expected probability mass across the full
vocabulary. Thousands of small add probabilities inflated the selected set.
Use the later `frozen_generation_probe_20260818_add_remove_events_seed42`
artifact instead.
