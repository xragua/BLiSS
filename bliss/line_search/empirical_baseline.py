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

def base_calculator(y):
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
