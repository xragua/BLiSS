"""Generate shuffled synthetic spectra for empirical false-positive estimation."""
import numpy as np

class SyntheticSpectrumGenerator:
    """Generate shuffled synthetic spectra with the configured simulation settings.

    Attributes
    ----------
    num_simulations : int
        Number of synthetic spectra concatenated into the output arrays.
    seed : int or None
        Seed passed to NumPy's random generator for reproducibility.
    z_score_th : float
        Absolute z-score threshold above which residual outliers are replaced
        before shuffling.
    """

    def __init__(self, num_simulations=2, seed=None, z_score_th=4):
        """Create a synthetic-spectrum generator.

        Parameters
        ----------
        num_simulations : int, default: 2
            Number of shuffled synthetic spectra to generate.
        seed : int or None, default: None
            Random seed used by ``numpy.random.default_rng``.
        z_score_th : float, default: 4
            Absolute z-score threshold used to replace strong residual outliers before
            shuffling.
        """
        self.num_simulations = num_simulations
        self.seed = seed
        self.z_score_th = z_score_th

    def generate(self, t, c, sc):
        """Generate synthetic spectra from one observed residual spectrum.

        Parameters
        ----------
        t : array-like
            Original coordinate grid.
        c : array-like
            Residual or line-excess values to shuffle.
        sc : array-like
            One-sigma uncertainties associated with ``c``.

        Returns
        -------
        tuple of numpy.ndarray
            Synthetic coordinate, value, and uncertainty arrays concatenating all
            simulations.
        """
        return calculate_synthetic_lines_spectra(t, c, sc, self.num_simulations, self.seed, self.z_score_th)

def calculate_synthetic_lines_spectra(t, c, sc, num_simulations, seed=None, z_score_th=4):
    """Create shuffled synthetic residual spectra for false-positive estimation.

    Parameters
    ----------
    t : array-like
        Original coordinate grid.
    c : array-like
        Residual values to shuffle after strong outliers are replaced.
    sc : array-like
        One-sigma uncertainties paired with ``c``.
    num_simulations : int
        Number of shuffled spectra to concatenate.
    seed : int or None, default: None
        Random seed for reproducible permutations and outlier replacements.
    z_score_th : float, default: 4
        Absolute z-score threshold defining residual outliers.

    Returns
    -------
    tuple of numpy.ndarray
        ``(tsim, simc, ssimc)`` containing concatenated synthetic coordinates,
        shuffled residual values, and shuffled uncertainties.
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(t, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    sc = np.asarray(sc, dtype=np.float64)
    resid = c.copy()
    z_resid = (resid - np.mean(resid)) / np.std(resid)
    outlier_mask = np.abs(z_resid) > z_score_th
    safe_mask = np.abs(z_resid) < z_score_th
    if np.any(outlier_mask) and np.any(safe_mask):
        safe_vals = resid[safe_mask]
        replace_vals = rng.choice(safe_vals, size=outlier_mask.sum(), replace=True)
        resid[outlier_mask] = replace_vals
    n = len(t)
    total = n * num_simulations
    tsim = np.zeros(total)
    simc = np.zeros(total)
    ssimc = np.zeros(total)
    mean_diff_t = np.mean(np.diff(t))
    mixed_diffs = np.concatenate([[mean_diff_t], np.diff(t)])
    for k in range(num_simulations):
        perm = rng.permutation(n)
        start = k * n
        end = start + n
        shuffled_diffs = mixed_diffs[perm]
        if np.max(tsim) > 1:
            tsim[start] = np.max(tsim) + np.mean(np.diff(t))
        if np.max(tsim) < 1:
            tsim[start] = np.min(t)
        tsim[start:end] = tsim[start] + np.cumsum(shuffled_diffs)
        simc[start:end] = resid[perm]
        ssimc[start:end] = sc[perm]
    return (tsim, simc, ssimc)
