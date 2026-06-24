"""Empirical baseline estimation for one-dimensional spectra."""
import numpy as np

def moving_average(data, window_size):
    """Compute an edge-padded moving average with the same length as the input.

    Parameters
    ----------
    data : array-like
        One-dimensional values to smooth, usually observed spectral counts or
        fluxes.
    window_size : int
        Number of samples included in the averaging window.

    Returns
    -------
    numpy.ndarray
        Smoothed array with the same length as ``data``.
    """
    pad_width = window_size // 2
    padded_data = np.pad(data, pad_width, mode='edge')
    weights = np.ones(window_size) / window_size
    moving_avg = np.convolve(padded_data, weights, mode='valid')
    if len(moving_avg) < len(data):
        moving_avg = np.append(moving_avg, moving_avg[-1])
    if len(moving_avg) > len(data):
        moving_avg = moving_avg[0:-1]
    return moving_avg

def base_calculator_old(y):
    """Estimate the empirical continuum baseline below candidate emission features.

    Parameters
    ----------
    y : array-like
        Observed spectral values.

    Returns
    -------
    numpy.ndarray
        Baseline estimate obtained from the minimum of several moving-average
        smoothings, clipped so it never exceeds the observed spectrum.
    """
    n = len(y)
    max_window = max(int(n / 3), 3)
    values_ = np.linspace(3, max_window, 100)
    values = np.unique([int(c) for c in values_])
    moving_averages = [moving_average(y, i) for i in values if len(moving_average(y, i)) == n]
    if not moving_averages:
        return np.array([])
    s = np.array(moving_averages)
    base = np.min(s, axis=0)
    return np.minimum(base, y)



def base_calculator(
    y,
    min_window=5,
    max_window=None,
    n_windows=50,
    q_smooth=0.25,
    cap_quantile=None,
):
    """Estimate a robust empirical baseline for emission-line searches.

    Parameters
    ----------
    y : array-like
        Observed spectral values.
    min_window : int
        Smallest smoothing window, in bins.
    max_window : int or None
        Largest smoothing window, in bins. If None, uses len(y) // 20.
    n_windows : int
        Number of smoothing windows to test.
    q_smooth : float
        Quantile of the family of smoothed curves. Values around 0.2--0.35
        give a lower-envelope baseline without collapsing to the absolute
        minimum.
    cap_quantile : float or None
        Optional local upper cap. For example 0.6 caps the baseline using a
        local 60th percentile of the data. Usually I would start with None.

    Returns
    -------
    numpy.ndarray
        Robust empirical baseline.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    if n == 0:
        return np.array([])

    if max_window is None:
        max_window = max(min_window, n // 20)

    min_window = int(min_window)
    max_window = int(max_window)

    if min_window < 3:
        min_window = 3

    if max_window < min_window:
        max_window = min_window

    # Use odd window sizes only
    windows = np.linspace(min_window, max_window, n_windows)
    windows = np.unique([int(w) + (int(w) + 1) % 2 for w in windows])
    windows = windows[windows >= 3]

    smooths = np.array([moving_average(y, w) for w in windows])

    # Instead of the absolute minimum, use a low quantile
    base = np.quantile(smooths, q_smooth, axis=0)

    # Optional: avoid very high baselines if desired, but do not force base <= y point by point
    if cap_quantile is not None:
        cap_window = max_window
        if cap_window % 2 == 0:
            cap_window += 1

        pad = cap_window // 2
        padded = np.pad(y, pad, mode="edge")

        local_caps = []
        for i in range(n):
            local_caps.append(np.quantile(padded[i:i + cap_window], cap_quantile))

        local_caps = np.array(local_caps)
        base = np.minimum(base, local_caps)

    return base
