"""Rebin one-dimensional spectra by bin count, S/N criterion, or resolution."""
import numpy as np

def _estimate_bin_width_from_centers(x):
    """Estimate bin widths from bin centers."""

    x = np.asarray(x, dtype=float)

    if len(x) == 0:
        return np.array([])

    if len(x) == 1:
        return np.ones_like(x)

    edges = np.empty(len(x) + 1, dtype=float)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])

    return np.diff(edges)

def _clean_arrays(x, y, sy, *extra_arrays):
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
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sy = np.asarray(sy, dtype=float)
    extras = [np.asarray(arr, dtype=float) for arr in extra_arrays]
    valid_mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(sy)
        & (sy > 0))
    for arr in extras:
        valid_mask &= np.isfinite(arr)
    cleaned = [x[valid_mask], y[valid_mask], sy[valid_mask]]
    for arr in extras:
        cleaned.append(arr[valid_mask])
    return tuple(cleaned)

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

def rebin_resolution(x, y, sy, resolution, bin_width=None):
    """
    Rebin a spectrum in density units onto fixed-width coordinate intervals.

    Use this for spectra in units such as:

        photons cm^-2 s^-1 keV^-1
        counts s^-1 keV^-1

    The integrated quantity over energy is preserved.

    Returns
    -------
    x_new, y_new, sy_new, bin_width_new : numpy.ndarray
        Rebinned coordinate, density values, uncertainties, and bin widths.
    """

    if bin_width is None:
        x, y, sy = _clean_arrays(x, y, sy)
    else:
        x, y, sy, bin_width = _clean_arrays(x, y, sy, bin_width)

    if len(x) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    sy = sy[order]

    if bin_width is None:
        bin_width = _estimate_bin_width_from_centers(x)
    else:
        bin_width = bin_width[order]

    valid_width = np.isfinite(bin_width) & (bin_width > 0)

    x = x[valid_width]
    y = y[valid_width]
    sy = sy[valid_width]
    bin_width = bin_width[valid_width]

    if len(x) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    resolution = float(resolution)

    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be a positive finite number.")

    input_left = x - 0.5 * bin_width
    input_right = x + 0.5 * bin_width

    x_start = np.nanmin(input_left)
    x_end = np.nanmax(input_right)

    edges = np.arange(x_start, x_end + resolution, resolution)

    if edges[-1] < x_end:
        edges = np.append(edges, x_end)

    x_new = []
    y_new = []
    sy_new = []
    bin_width_new = []

    for left, right in zip(edges[:-1], edges[1:]):

        overlap = np.minimum(input_right, right) - np.maximum(input_left, left)
        overlap = np.maximum(overlap, 0.0)

        mask = overlap > 0

        if not np.any(mask):
            continue

        dE = overlap[mask]
        total_width = np.sum(dE)

        if total_width <= 0:
            continue

        integrated_y = np.sum(y[mask] * dE)
        integrated_sy = np.sqrt(np.sum((sy[mask] * dE) ** 2))

        y_out = integrated_y / total_width
        sy_out = integrated_sy / total_width

        x_out = np.sum(x[mask] * dE) / total_width

        x_new.append(x_out)
        y_new.append(y_out)
        sy_new.append(sy_out)
        bin_width_new.append(total_width)

    return np.asarray(x_new),np.asarray(y_new),np.asarray(sy_new)