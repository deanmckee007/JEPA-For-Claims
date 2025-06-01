"""Evaluate fidelity of synthetic claims using KS/JSD metrics."""

import numpy as np
from scipy.stats import ks_2samp, entropy


def diffusion_fidelity(real, synthetic):
    real = np.asarray(real)
    synth = np.asarray(synthetic)
    ks = ks_2samp(real.ravel(), synth.ravel()).statistic
    p_real = np.bincount(real.ravel()) + 1
    p_synth = np.bincount(synth.ravel()) + 1
    p_real = p_real / p_real.sum()
    p_synth = p_synth / p_synth.sum()
    jsd = entropy((p_real + p_synth) / 2, qk=p_real) / 2 + entropy((p_real + p_synth) / 2, qk=p_synth) / 2
    return {"ks": ks, "jsd": jsd}
