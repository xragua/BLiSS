"""Empirical baseline estimation for one-dimensional spectra.

This module implements the default BLiSS empirical baseline used for blind
emission-line searches. The baseline is not a physical continuum model. It is
an algorithmic lower-envelope estimate built from the point-wise minimum of a
family of sigma-clipped moving averages, followed by a one-sided removal of
narrow downward baseline spikes.
"""

from __future__ import annotations

import numpy as np


# ============================================================
# Basic helpers
# ============================================================

def _interp_invalid(x, y):
    """Fill NaNs or invalid values by linear interpolation.

    Parameters
    ----------
    x : array-like
        Coordinate grid.
    y : array-like
        Values to be interpolated.

    Returns
    -------
    numpy.ndarray
        Array with invalid values replaced by linear interpolation.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() == 0:
        return np.zeros_like(y, dtype=float)

    if valid.sum() == 1:
        return np.full_like(y, y[valid][0], dtype=float)

    return np.interp(x, x[valid], y[valid])


def moving_average(y, window):
    """Simple moving average with same output length as input."""

    y = np.asarray(y, dtype=float)
    n = len(y)
    window = int(window)

    if n == 0:
        return np.array([])

    if window <= 1:
        return y.copy()

    window = min(window, n)

    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y, kernel, mode="same")


def moving_average_masked(y, window, mask=None):
    """Moving average with same output length, ignoring masked points.

    Parameters
    ----------
    y : array-like
        Input spectrum.
    window : int
        Moving-average window in bins.
    mask : array-like of bool or None
        Accepted points. If None, all finite points are accepted.

    Returns
    -------
    numpy.ndarray
        Masked moving-average estimate.
    """

    y = np.asarray(y, dtype=float)
    n = len(y)
    window = int(window)

    if n == 0:
        return np.array([])

    if window <= 1:
        return y.copy()

    window = min(window, n)

    if mask is None:
        mask = np.isfinite(y)
    else:
        mask = np.asarray(mask, dtype=bool) & np.isfinite(y)

    kernel = np.ones(window, dtype=float)

    numerator = np.convolve(
        np.where(mask, y, 0.0),
        kernel,
        mode="same",
    )

    denominator = np.convolve(
        mask.astype(float),
        kernel,
        mode="same",
    )

    out = np.full(n, np.nan, dtype=float)
    good = denominator > 0
    out[good] = numerator[good] / denominator[good]

    return out


def running_median(y, window):
    """Running median with same output length as input."""

    y = np.asarray(y, dtype=float)
    n = len(y)
    window = int(window)

    if n == 0:
        return np.array([])

    if window <= 1:
        return y.copy()

    window = min(window, n)

    if window % 2 == 0:
        window += 1

    half = window // 2
    out = np.empty(n, dtype=float)

    for i in range(n):
        i1 = max(0, i - half)
        i2 = min(n, i + half + 1)
        out[i] = np.nanmedian(y[i1:i2])

    return out


# ============================================================
# One sigma-clipped moving average
# ============================================================

def sigma_clipped_moving_average(
    values,
    window,
    n_iter=1,
    clip_sigma=1.0,
    reject="both",
    sigma_mode="global",
    min_valid_fraction=0.25,
    return_info=False,
):
    """Compute one sigma-clipped moving-average baseline.

    Parameters
    ----------
    values : array-like
        Input spectrum.
    window : int
        Moving-average window in bins.
    n_iter : int, default=1
        Number of sigma-clipping iterations.
    clip_sigma : float, default=1.0
        Sigma threshold for clipping.
    reject : {"both", "positive", "negative"}, default="both"
        Which residuals to reject.
    sigma_mode : {"global", "local"}, default="global"
        Whether to estimate a single global dispersion or a local one.
    min_valid_fraction : float, default=0.25
        Stop if fewer than this fraction of bins remain accepted.
    return_info : bool, default=False
        If True, also return mask and iteration diagnostics.

    Returns
    -------
    baseline : numpy.ndarray
        Sigma-clipped moving-average baseline for the requested window.
    info : dict, optional
        Returned only if ``return_info=True``.
    """

    values = np.asarray(values, dtype=float)
    n = len(values)

    if n == 0:
        if return_info:
            return np.array([]), {}
        return np.array([])

    x = np.arange(n, dtype=float)
    finite = np.isfinite(values)
    mask = finite.copy()
    min_valid = max(3, int(min_valid_fraction * n))
    history = []

    window = max(1, min(int(window), n))

    for i in range(int(n_iter)):

        baseline = moving_average_masked(values, window=window, mask=mask)
        baseline = _interp_invalid(x, baseline)

        residual = values - baseline

        if sigma_mode == "global":
            sigma = np.nanstd(residual[mask])

            if not np.isfinite(sigma) or sigma <= 0:
                break

            sigma_arr = np.full_like(values, sigma, dtype=float)

        elif sigma_mode == "local":
            mean_r = moving_average_masked(
                residual,
                window=window,
                mask=mask,
            )
            mean_r2 = moving_average_masked(
                residual**2,
                window=window,
                mask=mask,
            )

            mean_r = _interp_invalid(x, mean_r)
            mean_r2 = _interp_invalid(x, mean_r2)

            var = mean_r2 - mean_r**2
            var[var < 0] = 0.0
            sigma_arr = np.sqrt(var)

            fallback = np.nanstd(residual[mask])
            bad = ~np.isfinite(sigma_arr) | (sigma_arr <= 0)

            if np.isfinite(fallback) and fallback > 0:
                sigma_arr[bad] = fallback
            else:
                break

        else:
            raise ValueError("sigma_mode must be 'global' or 'local'.")

        if reject == "both":
            new_mask = finite & (np.abs(residual) <= clip_sigma * sigma_arr)
        elif reject == "positive":
            new_mask = finite & (residual <= clip_sigma * sigma_arr)
        elif reject == "negative":
            new_mask = finite & (residual >= -clip_sigma * sigma_arr)
        else:
            raise ValueError("reject must be 'both', 'positive', or 'negative'.")

        n_kept = int(new_mask.sum())
        history.append(
            {
                "iteration": i + 1,
                "window": int(window),
                "n_kept": n_kept,
                "kept_fraction": n_kept / n,
            }
        )

        if n_kept < min_valid:
            break

        if np.array_equal(new_mask, mask):
            mask = new_mask
            break

        mask = new_mask

    baseline = moving_average_masked(values, window=window, mask=mask)
    baseline = _interp_invalid(x, baseline)

    if return_info:
        info = {
            "window": int(window),
            "mask": mask,
            "rejected": finite & ~mask,
            "history": history,
        }
        return baseline, info

    return baseline


# ============================================================
# Minimum of many sigma-clipped moving averages
# ============================================================

def base_calculator_min_sigma_clipped_moving_averages(
    values,
    min_window=3,
    max_window=50,
    n_windows=30,
    windows=None,
    n_iter=1,
    clip_sigma=1.0,
    reject="both",
    sigma_mode="global",
    clip_to_data=False,
    return_info=False,
):
    """Lower envelope from the minimum of sigma-clipped moving averages.

    The function computes one sigma-clipped moving-average baseline per window
    size and then returns the point-wise minimum across all accepted windows.
    """

    values = np.asarray(values, dtype=float)
    n = len(values)

    if n == 0:
        if return_info:
            return np.array([]), {}
        return np.array([])

    if windows is None:
        max_window = int(max_window) if max_window is not None else max(int(n / 3), 3)
        max_window = max(1, min(max_window, n))
        min_window = max(1, min(int(min_window), max_window))

        raw_windows = np.linspace(
            min_window,
            max_window,
            int(n_windows),
        )
        windows = np.unique(raw_windows.astype(int))
    else:
        windows = np.unique(np.asarray(windows, dtype=int))
        windows = windows[(windows >= 1) & (windows <= n)]

    all_baselines = []
    per_window_info = {}

    for window in windows:
        base_w, info_w = sigma_clipped_moving_average(
            values,
            window=window,
            n_iter=n_iter,
            clip_sigma=clip_sigma,
            reject=reject,
            sigma_mode=sigma_mode,
            return_info=True,
        )

        if len(base_w) == n and np.all(np.isfinite(base_w)):
            all_baselines.append(base_w)
            per_window_info[int(window)] = info_w

    if len(all_baselines) == 0:
        if return_info:
            return np.array([]), {
                "windows": windows,
                "all_baselines": np.array([]),
                "per_window_info": per_window_info,
            }
        return np.array([])

    all_baselines = np.asarray(all_baselines)
    baseline = np.nanmin(all_baselines, axis=0)

    if clip_to_data:
        baseline = np.minimum(baseline, values)

    if return_info:
        info = {
            "windows": windows,
            "all_baselines": all_baselines,
            "per_window_info": per_window_info,
        }
        return baseline, info

    return baseline


# ============================================================
# One-sided downward-hair removal
# ============================================================

def remove_downward_hairs(
    baseline,
    window=9,
    clip_sigma=2.0,
    n_iter=10,
    strength=1.0,
):
    """Remove narrow downward spikes from a baseline.

    This correction is one-sided: only significant negative deviations of the
    baseline relative to a local running median are lifted. Positive spectral
    excesses in the observed data are not suppressed by this step.
    """

    cleaned = np.asarray(baseline, dtype=float).copy()

    if len(cleaned) == 0:
        return np.array([])

    strength = float(strength)

    for _ in range(int(n_iter)):
        local = running_median(cleaned, window=window)
        diff = cleaned - local

        negative_diff = diff[np.isfinite(diff) & (diff < 0)]

        if len(negative_diff) == 0:
            break

        med = np.nanmedian(negative_diff)
        sigma = 1.4826 * np.nanmedian(np.abs(negative_diff - med))

        if not np.isfinite(sigma) or sigma <= 0:
            sigma = np.nanstd(negative_diff)

        if not np.isfinite(sigma) or sigma <= 0:
            break

        hairs = diff < -float(clip_sigma) * sigma

        if not np.any(hairs):
            break

        cleaned[hairs] = (
            (1.0 - strength) * cleaned[hairs]
            + strength * local[hairs]
        )

    return cleaned


# ============================================================
# Default BLiSS baseline
# ============================================================

def base_calculator(
    y,
    min_window=None,
    max_window=None,
    n_windows=30,
    n_iter=1,
    clip_sigma=1.0,
    reject="both",
    sigma_mode="global",
    clip_to_data=False,
    hair_window=9,
    hair_clip_sigma=2.0,
    hair_n_iter=10,
    hair_strength=1.0,
    return_info=False,
):
    """Estimate the default BLiSS empirical baseline.

    This is the public baseline entry point used by BLiSS. It applies:

    1. a family of sigma-clipped moving averages,
    2. the point-wise minimum across window sizes,
    3. a one-sided removal of narrow downward baseline spikes.

    By default, the moving-average windows are chosen adaptively from the
    spectrum length:

    - min_window = max(len(y) / 100, 5)
    - max_window = max(len(y) / 10, 50)

    These values can be overridden by passing explicit min_window and
    max_window values.
    """

    y = np.asarray(y, dtype=float)
    n = len(y)

    if n == 0:
        if return_info:
            return np.array([]), {}
        return np.array([])

    # --------------------------------------------------------
    # Adaptive default windows
    # --------------------------------------------------------

    if min_window is None:
        min_window = int(max(n / 100, 5))
    else:
        min_window = int(min_window)

    if max_window is None:
        max_window = int(max(n / 10, 50))
    else:
        max_window = int(max_window)

    # Keep windows inside valid limits
    min_window = max(1, min(min_window, n))
    max_window = max(1, min(max_window, n))

    # Avoid inverted ranges for short spectra
    if max_window < min_window:
        max_window = min_window

    baseline_raw, info = base_calculator_min_sigma_clipped_moving_averages(
        y,
        min_window=min_window,
        max_window=max_window,
        n_windows=n_windows,
        n_iter=n_iter,
        clip_sigma=clip_sigma,
        reject=reject,
        sigma_mode=sigma_mode,
        clip_to_data=False,
        return_info=True,
    )

    baseline_clean = remove_downward_hairs(
        baseline_raw,
        window=hair_window,
        clip_sigma=hair_clip_sigma,
        n_iter=hair_n_iter,
        strength=hair_strength,
    )

    if clip_to_data:
        baseline_clean = np.minimum(baseline_clean, y)

    if return_info:
        info["baseline_raw"] = baseline_raw
        info["baseline_clean"] = baseline_clean
        info["min_window"] = min_window
        info["max_window"] = max_window
        info["n_windows"] = n_windows
        info["n_iter"] = n_iter
        info["clip_sigma"] = clip_sigma
        info["reject"] = reject
        info["sigma_mode"] = sigma_mode
        info["clip_to_data"] = clip_to_data
        info["hair_window"] = hair_window
        info["hair_clip_sigma"] = hair_clip_sigma
        info["hair_n_iter"] = hair_n_iter
        info["hair_strength"] = hair_strength
        return baseline_clean, info

    return baseline_clean