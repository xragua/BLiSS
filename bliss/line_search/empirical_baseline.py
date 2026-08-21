"""Shape-based empirical baseline for BLiSS."""

import numpy as np


def choose_baseline_window(
    energy,
    baseline_window=0.4,
    max_range_fraction=0.2,
):
    """Choose the physical width used for baseline estimation.

    Parameters
    ----------
    energy : array-like
        Spectral energy grid.

    baseline_window : float or array-like, default=0.4
        Preferred running-median width in the same units as energy.

    max_range_fraction : float, default=0.2
        Maximum allowed window as a fraction of the total
        spectral energy range.

    Returns
    -------
    numpy.ndarray
        Baseline window associated with each energy point.
    """

    energy = np.asarray(energy, dtype=float)

    finite = np.isfinite(energy)

    if finite.sum() < 2:
        raise ValueError(
            "energy must contain at least two finite values."
        )

    energy_range = (
        np.nanmax(energy[finite])
        - np.nanmin(energy[finite])
    )

    baseline_window = np.asarray(
        baseline_window,
        dtype=float,
    )

    # Same baseline scale everywhere
    if baseline_window.ndim == 0:
        baseline_window = np.full_like(
            energy,
            float(baseline_window),
        )

    # Or optionally one scale per energy point
    if baseline_window.shape != energy.shape:
        raise ValueError(
            "baseline_window must be a scalar or have "
            "the same shape as energy."
        )

    if np.any(baseline_window <= 0):
        raise ValueError(
            "baseline_window must be positive."
        )

    # Avoid a window that is too large compared with
    # the available spectral range
    max_window = (
        max_range_fraction
        * energy_range
    )

    window = np.minimum(
        baseline_window,
        max_window,
    )

    return window


def _interp_invalid(x, y):
    """Interpolate occasional invalid baseline values."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() == 0:
        return np.zeros_like(y)

    if valid.sum() == 1:
        return np.full_like(
            y,
            y[valid][0],
        )

    return np.interp(
        x,
        x[valid],
        y[valid],
    )


def moving_median_x(
    x,
    y,
    width,
    min_points=3,
):
    """Running median in x units using reflection padding at the edges."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError(
            "x and y must have the same shape."
        )

    n = len(y)

    if n == 0:
        return np.array([])

    # --------------------------------------------------
    # Width can be scalar or one value per energy point
    # --------------------------------------------------

    width = np.asarray(
        width,
        dtype=float,
    )

    if width.ndim == 0:
        width = np.full(
            n,
            float(width),
        )

    if width.shape != x.shape:
        raise ValueError(
            "width must be a scalar or have "
            "the same shape as x."
        )

    if np.any(width <= 0):
        raise ValueError(
            "All window widths must be positive."
        )

    # --------------------------------------------------
    # Valid spectrum
    # --------------------------------------------------

    finite = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x_valid = x[finite]
    y_valid = y[finite]

    if len(x_valid) == 0:
        return np.zeros_like(y)

    xmin = np.min(x_valid)
    xmax = np.max(x_valid)

    # Padding only needs to cover the largest half-window
    pad_width = 0.5 * np.max(width)

    # --------------------------------------------------
    # Reflect left edge
    # --------------------------------------------------

    left_source = (
        (x_valid > xmin)
        & (x_valid <= xmin + pad_width)
    )

    x_left = (
        2.0 * xmin
        - x_valid[left_source]
    )

    y_left = y_valid[left_source]

    # --------------------------------------------------
    # Reflect right edge
    # --------------------------------------------------

    right_source = (
        (x_valid < xmax)
        & (x_valid >= xmax - pad_width)
    )

    x_right = (
        2.0 * xmax
        - x_valid[right_source]
    )

    y_right = y_valid[right_source]

    # --------------------------------------------------
    # Extended spectrum
    # --------------------------------------------------

    x_extended = np.concatenate(
        [
            x_left,
            x_valid,
            x_right,
        ]
    )

    y_extended = np.concatenate(
        [
            y_left,
            y_valid,
            y_right,
        ]
    )

    order = np.argsort(
        x_extended
    )

    x_extended = x_extended[order]
    y_extended = y_extended[order]

    # --------------------------------------------------
    # Calculate baseline on original energy grid
    # --------------------------------------------------

    baseline = np.full(
        n,
        np.nan,
    )

    for i in range(n):

        if not finite[i]:
            continue

        half_width = (
            width[i] / 2.0
        )

        local = (
            (x_extended >= x[i] - half_width)
            & (x_extended <= x[i] + half_width)
        )

        if local.sum() >= min_points:
            baseline[i] = np.median(
                y_extended[local]
            )

    return _interp_invalid(
        x,
        baseline,
    )


def base_calculator(
    x,
    y,
    baseline_window=0.4,
    max_range_fraction=0.2,
    min_points=3,
    return_info=False,
):
    """Estimate the BLiSS empirical baseline from spectral shape.

    The baseline is obtained with a running median whose width is
    defined directly in physical spectral units.

    The baseline scale is independent of the maximum Gaussian width
    allowed during line fitting.

    For spectra covering a narrow energy interval, the baseline window
    is limited to a fraction of the available range.

    Reflection padding is used at both boundaries.

    No statistical errors or sigma clipping are used.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    window_width = choose_baseline_window(
        x,
        baseline_window=baseline_window,
        max_range_fraction=max_range_fraction,
    )

    baseline = moving_median_x(
        x,
        y,
        width=window_width,
        min_points=min_points,
    )

    if return_info:

        info = {
            "window_width": window_width,
            "baseline_window": baseline_window,
            "max_range_fraction": max_range_fraction,
            "energy_range": (
                np.nanmax(x)
                - np.nanmin(x)
            ),
            "min_points": min_points,
        }

        return baseline, info

    return baseline