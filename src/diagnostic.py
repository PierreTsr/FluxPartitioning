import numpy as np
from scipy.stats import norm


def geweke(chain, frac_init=0.1, frac_final=0.5):
    n = chain.shape[0]
    sample_1 = chain[:int(frac_init * n)]
    sample_2 = chain[int(frac_final * n):]
    m_1 = sample_1.shape[0]
    m_2 = sample_2.shape[0]
    z = (np.mean(sample_1, axis=0) - np.mean(sample_2, axis=0)) / \
        np.sqrt(np.var(sample_1, axis=0) / m_1 + np.var(sample_2, axis=0) / m_2)
    return z, 2 * (1 - norm.cdf(np.abs(z)))


def gelman_rubin(chains):
    n = chains.shape[0]
    m = chains.shape[-1]
    b = n / (m - 1) * np.sum(
        (np.mean(chains, axis=0, keepdims=True) - np.mean(chains, axis=(0, -1), keepdims=True)) ** 2, axis=-1)
    b = np.squeeze(b)
    w = 1 / m / (n - 1) * np.sum(np.sum((chains - np.mean(chains, axis=0, keepdims=True)) ** 2, axis=0), axis=-1)
    w = np.squeeze(w)
    var = (n - 1) / n * w + b / n
    return np.sqrt(var / w)
