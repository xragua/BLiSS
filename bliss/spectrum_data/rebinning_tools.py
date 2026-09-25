"""Count-conserving rebinning of detector spectra."""
import numpy as np


# ---------------------------------------------------------------------------
# Count-conserving rebinning (for spectra in counts, i.e. loader output)
# ---------------------------------------------------------------------------

def rebin_counts(energy, counts, uncertainties, bin_width, method='none',
                 scale=None, min_bins=1, remainder='merge'):
    """Rebin a count spectrum by summing consecutive bins.

    Counts are **summed** (uncertainties in quadrature, widths added), so
    Poisson statistics are preserved and the result can be converted to a
    density afterwards.

    Parameters
    ----------
    energy, counts, uncertainties, bin_width : array-like
        Bin centres, counts (net counts are allowed), one-sigma uncertainties
        and bin widths. Must be sorted by energy.
    method : {'none', 'bins', 'snr', 'resolution'}
        * ``'none'``: return the input unchanged.
        * ``'bins'``: group ``scale`` consecutive bins.
        * ``'snr'``: accumulate bins until ``counts / sigma >= scale``.
        * ``'resolution'``: group bins into fixed-width intervals of ``scale``
          energy units; each input bin is assigned to the interval containing
          its centre, so bins are never split.
    scale : int or float
        Number of bins, S/N threshold, or interval width, depending on ``method``.
    min_bins : int, default 1
        Minimum number of input bins per output bin (``'snr'`` only).
    remainder : {'merge', 'drop'}, default 'merge'
        What to do with trailing bins that do not complete a group
        (``'bins'`` and ``'snr'``): merge them into the last output bin or
        drop them.

    Returns
    -------
    energy_new, counts_new, uncertainties_new, bin_width_new : numpy.ndarray
        Output bin centres (midpoint of the grouped edges), summed counts,
        uncertainties added in quadrature, and total widths.
    group : numpy.ndarray of int
        Index of the output bin each input bin was assigned to (-1 if dropped).
    """
    energy = np.asarray(energy, dtype=float)
    counts = np.asarray(counts, dtype=float)
    uncertainties = np.asarray(uncertainties, dtype=float)
    bin_width = np.asarray(bin_width, dtype=float)
    n = len(energy)
    if not (len(counts) == len(uncertainties) == len(bin_width) == n):
        raise ValueError('energy, counts, uncertainties and bin_width must have the same length.')
    if n == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty, np.array([], dtype=int)
    if np.any(np.diff(energy) < 0):
        raise ValueError('energy must be sorted in increasing order.')
    if remainder not in ('merge', 'drop'):
        raise ValueError("remainder must be 'merge' or 'drop'.")

    method = str(method).lower()
    if method == 'none':
        return energy.copy(), counts.copy(), uncertainties.copy(), bin_width.copy(), np.arange(n)

    if scale is None:
        raise ValueError(f"method='{method}' requires a scale.")

    # ---- build the group index for every input bin -------------------------
    group = np.full(n, -1, dtype=int)

    if method == 'bins':
        nb = int(scale)
        if nb < 1:
            raise ValueError('scale must be >= 1 for method="bins".')
        group[:] = np.arange(n) // nb
        n_full = (n // nb) * nb
        if n_full < n:                       # incomplete trailing group
            if remainder == 'merge' and n_full > 0:
                group[n_full:] = group[n_full - 1]
            elif remainder == 'drop':
                group[n_full:] = -1

    elif method == 'snr':
        threshold = float(scale)
        if threshold <= 0:
            raise ValueError('scale must be positive for method="snr".')
        g = 0
        acc_counts = 0.0
        acc_var = 0.0
        n_acc = 0
        start = 0
        for i in range(n):
            acc_counts += counts[i]
            acc_var += uncertainties[i] ** 2
            n_acc += 1
            snr = acc_counts / np.sqrt(acc_var) if acc_var > 0 else 0.0
            if snr >= threshold and n_acc >= min_bins:
                group[start:i + 1] = g
                g += 1
                acc_counts = acc_var = 0.0
                n_acc = 0
                start = i + 1
        if start < n:                        # trailing bins below threshold
            if remainder == 'merge' and g > 0:
                group[start:] = g - 1
            elif remainder == 'merge':       # nothing reached threshold: one bin
                group[start:] = 0
            else:
                group[start:] = -1

    elif method == 'resolution':
        width = float(scale)
        if not np.isfinite(width) or width <= 0:
            raise ValueError('scale must be a positive finite width for method="resolution".')
        x0 = energy[0] - 0.5 * bin_width[0]
        raw = np.floor((energy - x0) / width).astype(int)
        # renumber consecutively (skip empty intervals)
        _, group[:] = np.unique(raw, return_inverse=True)

    else:
        raise ValueError("method must be 'none', 'bins', 'snr', or 'resolution'.")

    return apply_groups(energy, counts, uncertainties, bin_width, group) + (group,)


def apply_groups(energy, counts, uncertainties, bin_width, group):
    """Sum native bins according to a precomputed group index.

    This is the second half of ``rebin_counts``; it is exposed so that a
    grouping derived from the data can be applied unchanged to synthetic
    realizations.

    Parameters
    ----------
    energy, counts, uncertainties, bin_width : array-like
        Native bins (sorted by energy).
    group : array-like of int
        Output-bin index of each input bin; ``-1`` means dropped.

    Returns
    -------
    energy_new, counts_new, uncertainties_new, bin_width_new : numpy.ndarray
    """
    energy = np.asarray(energy, dtype=float)
    counts = np.asarray(counts, dtype=float)
    uncertainties = np.asarray(uncertainties, dtype=float)
    bin_width = np.asarray(bin_width, dtype=float)
    group = np.asarray(group, dtype=int)

    keep = group >= 0
    ids = np.unique(group[keep])
    e_lo = energy - 0.5 * bin_width
    e_hi = energy + 0.5 * bin_width

    energy_new = np.empty(len(ids))
    counts_new = np.empty(len(ids))
    unc_new = np.empty(len(ids))
    width_new = np.empty(len(ids))
    for j, gid in enumerate(ids):
        m = group == gid
        lo, hi = e_lo[m].min(), e_hi[m].max()
        energy_new[j] = 0.5 * (lo + hi)
        width_new[j] = bin_width[m].sum()
        counts_new[j] = counts[m].sum()
        unc_new[j] = np.sqrt(np.sum(uncertainties[m] ** 2))

    return energy_new, counts_new, unc_new, width_new
