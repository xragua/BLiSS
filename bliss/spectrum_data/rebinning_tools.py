"""Rebin one-dimensional spectra by bin count, S/N criterion, or resolution."""
import numpy as np

def _clean_arrays(x, y, sy):
    """Convert spectrum arrays to numpy arrays and remove invalid bins.

    Parameters
    ----------
    x : array-like
        Spectral coordinate values.
    y : array-like
        Spectral counts, rates, or flux values.
    sy : array-like
        One-sigma uncertainties on ``y``.

    Returns
    -------
    tuple of numpy.ndarray
        Filtered ``x``, ``y``, and ``sy`` arrays containing only finite values with
        strictly positive uncertainties.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    sy = np.asarray(sy)
    valid_mask = (sy > 0) & np.isfinite(x) & np.isfinite(y) & np.isfinite(sy)
    return (x[valid_mask], y[valid_mask], sy[valid_mask])

def rebin_bins(x, y, sy, nbin):
    """Rebin a spectrum by grouping a fixed number of consecutive bins.

    Parameters
    ----------
    x : array-like
        Spectral coordinate values.
    y : array-like
        Spectral values to average within each group.
    sy : array-like
        One-sigma uncertainties used as inverse-variance weights.
    nbin : int
        Number of consecutive input bins combined into each output bin.

    Returns
    -------
    tuple of list
        Weighted coordinate centers, weighted spectral values, and propagated
        uncertainties for each rebinned group.
    """
    x, y, sy = _clean_arrays(x, y, sy)
    x_new, y_new, sy_new = [], [], []
    for start in range(0, len(y), nbin):
        stop = start + nbin
        x_bin = x[start:stop]
        y_bin = y[start:stop]
        sy_bin = sy[start:stop]
        if len(y_bin) == 0:
            continue
        w = 1.0 / sy_bin**2
        if np.sum(w) <= 0:
            continue
        x_new.append(np.sum(x_bin * w) / np.sum(w))
        y_new.append(np.sum(y_bin * w) / np.sum(w))
        sy_new.append(np.sqrt(1.0 / np.sum(w)))

    return np.array(x_new), np.array(y_new), np.array(sy_new)

def rebin_snr(x, y, sy, min_snr=5,min_bins=1):
    """Accumulate adjacent bins until a target uncertainty-to-signal criterion is met.

    Parameters
    ----------
    x : array-like
        Spectral coordinate values.
    y : array-like
        Spectral values to combine.
    sy : array-like
        One-sigma uncertainties used for inverse-variance weighting.
    snr_threshold : float
        Threshold applied to the current combined uncertainty divided by the
        weighted signal. A bin is emitted once this value is less than or equal to
        the threshold.

    Returns
    -------
    tuple of list
        Weighted coordinates, weighted values, and propagated uncertainties for the
        adaptive bins.
    """
    x, y, sy = _clean_arrays(x, y, sy)
    w, y_bin, x_bin, sy_bin = ([], [], [], [])
    y_new, x_new, sy_new = ([], [], [])
    for xi, yi, syi in zip(x, y, sy):
        x_bin.append(xi)
        y_bin.append(yi)
        sy_bin.append(syi)
        x_arr = np.array(x_bin)
        y_arr = np.array(y_bin)
        sy_arr = np.array(sy_bin)
        w = 1.0 / sy_arr**2
        y_weighted = np.sum(y_arr * w) / np.sum(w)
        x_weighted = np.sum(x_arr * w) / np.sum(w)
        sy_weighted = np.sqrt(1.0 / np.sum(w))
        snr_now = np.abs(y_weighted) / sy_weighted
        if (snr_now >= min_snr) and (len(y_bin) >= min_bins):
            x_new.append(x_weighted)
            y_new.append(y_weighted)
            sy_new.append(sy_weighted)
            x_bin, y_bin, sy_bin = [], [], []
    return np.array(x_new), np.array(y_new), np.array(sy_new)

def rebin_resolution(x, y, sy, resolution):
    """Rebin a spectrum onto fixed-width coordinate intervals.

    Parameters
    ----------
    x : array-like
        Spectral coordinate values.
    y : array-like
        Spectral values summed inside each output interval.
    sy : array-like
        One-sigma uncertainties propagated in quadrature inside each interval.
    resolution : float
        Width of each output interval in the same units as ``x``.

    Returns
    -------
    tuple of numpy.ndarray
        Mean coordinate, summed value, and propagated uncertainty for each populated
        output interval.
    """
    x, y, sy = _clean_arrays(x, y, sy)
    x_start, x_end = (x.min(), x.max())
    edges = np.arange(x_start, x_end + resolution, resolution)
    x_new, y_new, sy_new = ([], [], [])
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (x >= left) & (x < right)
        if not np.any(mask):
            continue
        x_bin = x[mask]
        y_bin = y[mask]
        sy_bin = sy[mask]
        x_new.append(np.mean(x_bin))
        y_new.append(np.sum(y_bin))
        sy_new.append(np.sqrt(np.sum(sy_bin ** 2)))
    return (np.array(x_new), np.array(y_new), np.array(sy_new))
