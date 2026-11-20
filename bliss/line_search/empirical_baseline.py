"""Shape-based empirical baseline for BLiSS."""

from numbers import Integral

import numpy as np
from scipy.stats import binom, chi2, poisson


def _baseline_window_limit(energy, max_range_fraction):
    """Resolve the physical cap once on the full observed energy grid."""
    energy = np.asarray(energy, dtype=float)
    finite = energy[np.isfinite(energy)]
    if finite.size < 2 or np.ptp(finite) <= 0:
        raise ValueError(
            "Cannot estimate the baseline with this rebinning: at least two "
            "distinct finite energy bins are required. Use finer rebinning."
        )
    fraction = np.asarray(max_range_fraction, dtype=float)
    if fraction.ndim != 0 or not np.isfinite(fraction) or fraction <= 0:
        raise ValueError("max_range_fraction must be a finite, positive number.")
    limit = float(fraction * np.ptp(finite))
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("The maximum baseline window must be finite and positive.")
    return limit


def _resolve_baseline_window_limit(energy, max_range_fraction, max_window_width):
    """Use a supplied physical cap without deriving it from sampled centers."""
    if max_window_width is None:
        return _baseline_window_limit(energy, max_range_fraction)
    try:
        limit = np.asarray(max_window_width, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('max_window_width must be a finite, positive scalar.') from exc
    if (isinstance(max_window_width, (bool, np.bool_)) or limit.ndim != 0
            or not np.isfinite(limit) or limit <= 0):
        raise ValueError('max_window_width must be a finite, positive scalar.')
    return float(limit)


def choose_baseline_window(
    energy,
    baseline_window=0.4,
    max_range_fraction=0.2,
    *,
    max_window_width=None,
):
    """Choose the physical width used for baseline estimation.

    Parameters
    ----------
    energy : array-like
        Spectral energy grid.

    baseline_window : float, array-like, or callable, default=0.4
        Preferred running-median width in the same units as energy.
        A callable ``f(energy)`` is evaluated on the input grid. A static
        array must have the same shape as ``energy``. The search pipeline
        resolves initial widths on the observed working grid and interpolates
        them onto the synthetic null grids. Each grid can then enlarge its
        local windows to reach min_points, up to the same physical cap.

    max_range_fraction : float, default=0.2
        Maximum allowed window as a fraction of the total
        spectral energy range.

    max_window_width : float or None, keyword-only
        Explicit physical window cap, for example one resolved from a
        requested energy interval before rebinning. When provided, it
        replaces the cap derived from ``max_range_fraction`` and the span
        of the sampled centers. Must be finite and positive.

    Returns
    -------
    numpy.ndarray
        Initial baseline window associated with each energy point.
    """

    energy = np.asarray(energy, dtype=float)

    max_window = _resolve_baseline_window_limit(
        energy, max_range_fraction, max_window_width)

    # Optionally an energy-dependent window given as a callable:
    # evaluate it on the grid the baseline will be computed on.
    if callable(baseline_window):
        baseline_window = baseline_window(energy)

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

    if np.any(~np.isfinite(baseline_window) | (baseline_window <= 0)):
        raise ValueError(
            "baseline_window must be finite and positive."
        )

    # Avoid a window that is too large compared with
    # the available spectral range
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
        raise ValueError("Cannot estimate the baseline: no finite baseline values are available.")

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
    *,
    max_width=None,
    return_width=False,
):
    """Running median with reflected edges and locally adaptive widths.

    Enlarge each window only as far as needed to include ``min_points``
    finite samples, counting reflected edge samples as before. ``max_width``
    is the physical upper limit; if omitted, the supplied widths are the
    limits and cannot grow. Failure at any finite point raises ValueError.
    ``return_width=True`` also returns the widths actually used.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError(
            "x and y must have the same shape."
        )

    if isinstance(min_points, (bool, np.bool_)) or not isinstance(min_points, Integral) or min_points < 1:
        raise ValueError("min_points must be a positive integer.")

    n = len(y)

    if n == 0:
        empty = np.array([])
        return (empty, empty.copy()) if return_width else empty

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

    if np.any(~np.isfinite(width) | (width <= 0)):
        raise ValueError(
            "All window widths must be finite and positive."
        )

    limit = width.copy() if max_width is None else np.asarray(max_width, dtype=float)
    if limit.ndim == 0:
        limit = np.full(n, float(limit))
    if limit.shape != x.shape or np.any(~np.isfinite(limit) | (limit <= 0)):
        raise ValueError("max_width must contain finite, positive limits matching x.")
    effective_width = np.minimum(width, limit).copy()

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
        raise ValueError("Cannot estimate the baseline: the spectrum has no finite samples.")

    xmin = np.min(x_valid)
    xmax = np.max(x_valid)

    # Reflection must cover windows that may grow up to the physical cap.
    pad_width = 0.5 * np.max(limit)
    tolerance = 8 * np.finfo(float).eps * max(1., np.max(np.abs(x_valid)), pad_width)

    # --------------------------------------------------
    # Reflect left edge
    # --------------------------------------------------

    left_source = (
        (x_valid > xmin)
        & (x_valid <= xmin + pad_width + tolerance)
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
        & (x_valid >= xmax - pad_width - tolerance)
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

        half_width = effective_width[i] / 2.0
        distances = np.abs(x_extended - x[i])
        local = distances <= half_width + tolerance

        if local.sum() < min_points:
            available = distances <= limit[i] / 2.0 + tolerance
            if available.sum() < min_points:
                raise ValueError(
                    "Cannot estimate the baseline with this rebinning: "
                    f"min_points={min_points} cannot be reached at energy={x[i]:.8g} "
                    f"within the maximum window width ({limit[i]:.8g}). "
                    "Use finer rebinning or increase max_range_fraction."
                )
            needed = np.partition(distances[available], min_points - 1)[min_points - 1]
            half_width = min(needed, limit[i] / 2.0)
            effective_width[i] = 2.0 * half_width
            local = distances <= half_width + tolerance

        baseline[i] = np.median(y_extended[local])

    baseline = _interp_invalid(x, baseline)
    if return_width:
        return baseline, effective_width
    return baseline


def base_calculator(
    x,
    y,
    baseline_window=0.4,
    max_range_fraction=0.2,
    min_points=3,
    return_info=False,
    *,
    max_window_width=None,
):
    """Estimate the BLiSS empirical baseline from spectral shape.

    The baseline is obtained with a running median whose width is
    defined directly in physical spectral units. The width may be a
    scalar, an array with one value per energy point, or a callable
    ``f(energy)`` evaluated on the input grid (energy-dependent window).

    The baseline scale is independent of the maximum Gaussian width
    allowed during line fitting.

    Each window grows locally if needed to include min_points, up to
    max_range_fraction times the full energy range. If any finite point
    cannot meet the minimum within that cap, raise ValueError with guidance
    to use finer rebinning or increase max_range_fraction.

    ``max_window_width`` optionally supplies a finite, positive physical
    cap instead, without recomputing it from this grid's sampled centers.
    This does not relax ``min_points`` in the supplied context or at edges.

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

    max_window_width = _resolve_baseline_window_limit(
        x, max_range_fraction, max_window_width)
    window_width = choose_baseline_window(
        x,
        baseline_window=baseline_window,
        max_range_fraction=max_range_fraction,
        max_window_width=max_window_width,
    )

    baseline, effective_width = moving_median_x(
        x,
        y,
        width=window_width,
        min_points=min_points,
        max_width=max_window_width,
        return_width=True,
    )

    if return_info:

        info = {
            "window_width": effective_width,
            "initial_window_width": window_width,
            "max_window_width": max_window_width,
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


def _nearest_radius(sorted_x, center, first, stop, minimum):
    """Distance to the minimum-th nearest sample in a sorted interval.

    Distances to the left and right of ``center`` are individually sorted.
    Partitioning these two lists avoids building a distance array for every
    window, including when ``minimum`` is large.
    """
    split = int(np.searchsorted(sorted_x, center, side="left"))
    split = min(max(split, first), stop)
    n_left, n_right = split - first, stop - split
    low, high = max(0, minimum - n_right), min(minimum, n_left)
    while low <= high:
        n_take_left = (low + high) // 2
        n_take_right = minimum - n_take_left
        left_previous = (center - sorted_x[split - n_take_left]
                         if n_take_left else -np.inf)
        left_next = (center - sorted_x[split - n_take_left - 1]
                     if n_take_left < n_left else np.inf)
        right_previous = (sorted_x[split + n_take_right - 1] - center
                          if n_take_right else -np.inf)
        right_next = (sorted_x[split + n_take_right] - center
                      if n_take_right < n_right else np.inf)
        if left_previous <= right_next and right_previous <= left_next:
            return max(left_previous, right_previous)
        if left_previous > right_next:
            high = n_take_left - 1
        else:
            low = n_take_left + 1
    raise RuntimeError("Cannot locate the requested baseline window samples.")


def moving_poisson_mean_x(
    x,
    source_counts,
    background_counts,
    background_scale,
    exposure,
    bin_width,
    width,
    min_points=3,
    *,
    max_width=None,
    return_width=False,
    exclude_mask=None,
):
    """Smooth count density using exposure-weighted local Poisson means.

    Each window returns ``max(sum(S) - alpha * sum(B), 0) / (T * sum(dE))``.
    Empty bins contribute their exposure and width, rather than an invented
    count or error. A truly empty window remains zero. With background, this
    is a nonnegative local net-density estimate; the line likelihood still
    treats source and background observations separately.

    Windows use the same reflected edges, minimum number of samples, and
    minimal expansion up to ``max_width`` as ``moving_median_x``. Reflection
    is only a boundary convention for smoothing, not extra likelihood data.
    Prefix sums and sorted searches keep the calculation O(n log n), without
    allocating a full distance array per output bin.

    ``exclude_mask`` omits marked bins from the count sums, exposure-width
    sums and ``min_points`` support, including their reflected copies. The
    output remains aligned with every original bin. Reflection keeps the
    original spectrum boundaries rather than moving them to unmasked bins.
    """
    x = np.asarray(x, dtype=float)
    source_counts = np.asarray(source_counts, dtype=float)
    background_counts = np.asarray(background_counts, dtype=float)
    bin_width = np.asarray(bin_width, dtype=float)
    arrays = (x, source_counts, background_counts, bin_width)
    if x.ndim != 1 or any(array.shape != x.shape for array in arrays):
        raise ValueError("Poisson baseline arrays must be one-dimensional and have the same shape.")
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("Poisson baseline arrays must contain finite values.")
    if np.any(source_counts < 0) or np.any(background_counts < 0):
        raise ValueError("Poisson baseline counts must be non-negative.")
    if np.any(bin_width <= 0):
        raise ValueError("Poisson baseline bin widths must be positive.")
    exposure = np.asarray(exposure, dtype=float)
    scale = np.asarray(background_scale, dtype=float)
    if exposure.ndim != 0 or not np.isfinite(exposure) or exposure <= 0:
        raise ValueError("Poisson baseline exposure must be a finite, positive scalar.")
    if scale.ndim != 0 or not np.isfinite(scale) or scale < 0:
        raise ValueError("Poisson baseline background_scale must be a finite, non-negative scalar.")
    if isinstance(min_points, (bool, np.bool_)) or not isinstance(min_points, Integral) or min_points < 1:
        raise ValueError("min_points must be a positive integer.")
    if exclude_mask is None:
        excluded = np.zeros(x.shape, dtype=bool)
    else:
        excluded = np.asarray(exclude_mask)
        if excluded.shape != x.shape or excluded.dtype.kind != 'b':
            raise ValueError('exclude_mask must be a boolean array matching the energy grid.')
    if len(x) == 0:
        empty = np.array([], dtype=float)
        return (empty, empty.copy()) if return_width else empty

    requested = np.asarray(width, dtype=float)
    if requested.ndim == 0:
        requested = np.full(x.shape, float(requested))
    if requested.shape != x.shape:
        raise ValueError("width must be a scalar or have the same shape as x.")
    if np.any(~np.isfinite(requested) | (requested <= 0)):
        raise ValueError("All window widths must be finite and positive.")
    limit = requested.copy() if max_width is None else np.asarray(max_width, dtype=float)
    if limit.ndim == 0:
        limit = np.full(x.shape, float(limit))
    if limit.shape != x.shape or np.any(~np.isfinite(limit) | (limit <= 0)):
        raise ValueError("max_width must contain finite, positive limits matching x.")
    effective_width = np.minimum(requested, limit).copy()

    xmin, xmax = np.min(x), np.max(x)
    pad_width = 0.5 * np.max(limit)
    tolerance = 8 * np.finfo(float).eps * max(1., np.max(np.abs(x)), pad_width)
    available = ~excluded
    left = (x > xmin) & (x <= xmin + pad_width + tolerance) & available
    right = (x < xmax) & (x >= xmax - pad_width - tolerance) & available
    x_extended = np.concatenate((2 * xmin - x[left], x[available], 2 * xmax - x[right]))
    net_counts = source_counts - float(scale) * background_counts
    extended_counts = np.concatenate((net_counts[left], net_counts[available], net_counts[right]))
    extended_width = np.concatenate((bin_width[left], bin_width[available], bin_width[right]))
    order = np.argsort(x_extended)
    x_extended = x_extended[order]
    count_sum = np.concatenate(([0.], np.cumsum(extended_counts[order])))
    width_sum = np.concatenate(([0.], np.cumsum(extended_width[order])))

    half_width = effective_width / 2
    first = np.searchsorted(x_extended, x - half_width - tolerance, side="left")
    stop = np.searchsorted(x_extended, x + half_width + tolerance, side="right")
    expand = np.flatnonzero(stop - first < min_points)
    for i in expand:
        available_first = int(np.searchsorted(
            x_extended, x[i] - limit[i] / 2 - tolerance, side="left"))
        available_stop = int(np.searchsorted(
            x_extended, x[i] + limit[i] / 2 + tolerance, side="right"))
        if available_stop - available_first < min_points:
            raise ValueError(
                "Cannot estimate the baseline with this rebinning: "
                f"min_points={min_points} cannot be reached at energy={x[i]:.8g} "
                f"within the maximum window width ({limit[i]:.8g}). "
                "Use finer rebinning or increase max_range_fraction."
            )
        radius = min(_nearest_radius(
            x_extended, x[i], available_first, available_stop, min_points), limit[i] / 2)
        effective_width[i] = 2 * radius
        first[i] = np.searchsorted(x_extended, x[i] - radius - tolerance, side="left")
        stop[i] = np.searchsorted(x_extended, x[i] + radius + tolerance, side="right")

    numerator = np.maximum(count_sum[stop] - count_sum[first], 0.)
    denominator = float(exposure) * (width_sum[stop] - width_sum[first])
    baseline = numerator / denominator
    if return_width:
        return baseline, effective_width
    return baseline


def _poisson_continuum_seeds(x, source, background, scale, exposure, bin_width,
                             baseline, effective_width, excluded):
    """Conservative positive-excess seeds using independent real-bin counts.

    With no background, condition on the source-window count sum and compare
    the tested bin with its physical-width fraction. With background, bound
    the gross window intensity and that bin's background separately before
    taking an upper Poisson tail. Neither test uses reflected duplicate events.
    """
    n = len(x)
    order = np.argsort(x, kind='stable')
    real = order[~excluded[order]]
    real_x = x[real]
    tolerance = 8 * np.finfo(float).eps * max(
        1., np.max(np.abs(x)), np.max(effective_width) / 2)
    first = np.searchsorted(real_x, x - effective_width / 2 - tolerance, side='left')
    stop = np.searchsorted(real_x, x + effective_width / 2 + tolerance, side='right')
    source_sum = np.concatenate(([0.], np.cumsum(source[real])))
    width_sum = np.concatenate(([0.], np.cumsum(bin_width[real])))
    window_source = source_sum[stop] - source_sum[first]
    window_width = width_sum[stop] - width_sum[first]
    fraction = np.divide(bin_width, window_width, out=np.ones_like(bin_width), where=window_width > 0)
    fraction = np.clip(fraction, 0., 1.)
    if scale == 0:
        threshold = .01 / (n * 5)
        probability = binom.sf(source - 1, np.rint(window_source), fraction)
    else:
        threshold = .01 / (n * 5 * 3)
        gross_upper = .5 * chi2.isf(threshold, 2 * (window_source + 1))
        background_upper = .5 * chi2.isf(threshold, 2 * (background + 1))
        upper_mean = gross_upper * fraction + scale * background_upper
        probability = poisson.sf(source - 1, upper_mean)
    positive_excess = source - scale * background > baseline * (exposure * bin_width)
    return (~excluded & positive_excess & (window_width > 0)
            & np.isfinite(probability) & (probability < threshold))


def _contiguous_seed_halo(x, bin_width, seeds):
    """Add one real adjacent bin per side without crossing channel gaps."""
    order = np.argsort(x, kind='stable')
    centers, widths = x[order], bin_width[order]
    marked = seeds[order]
    expanded = marked.copy()
    right_edge = centers[:-1] + widths[:-1] / 2
    left_edge = centers[1:] - widths[1:] / 2
    scale = np.maximum.reduce((np.abs(right_edge), np.abs(left_edge), np.ones_like(left_edge)))
    contiguous = np.abs(right_edge - left_edge) <= 64 * np.finfo(float).eps * scale
    expanded[:-1] |= marked[1:] & contiguous
    expanded[1:] |= marked[:-1] & contiguous
    result = np.zeros_like(seeds)
    result[order] = expanded
    return result


def moving_poisson_continuum_x(
    x,
    source_counts,
    background_counts,
    background_scale,
    exposure,
    bin_width,
    width,
    min_points=3,
    *,
    max_width=None,
    return_width=False,
    exclude_mask=None,
    return_mask=False,
):
    """Estimate a count continuum while conservatively excluding narrow excesses.

    Start from the ordinary count/exposure-width mean. At most five passes
    identify strong positive excesses with real-bin count tests and omit each
    seed plus one contiguous neighbor on each side from continuum estimation.
    Masking never removes spectral data or likelihood bins. Counts, widths,
    reflected copies and sample support of masked bins are all omitted from
    the mean; ``min_points`` and the physical cap remain strict everywhere.
    The mean is recomputed after every new mask, including the fifth pass.

    The no-background conditional binomial threshold is .01/(5*n). With
    background, conservative upper count limits and a Poisson tail use
    .01/(15*n). These assume a locally constant count density. The repeated,
    data-dependent masking and adaptive grouping are an empirical continuum
    procedure, not exact statistical calibration. Broad or weak features,
    particularly with strong background, may remain in the continuum.
    Truly empty windows remain zero; no count/error floor is introduced.

    Return the baseline by default, (baseline, widths) with ``return_width``,
    (baseline, mask) with ``return_mask``, or (baseline, widths, mask) with both.
    Arrays always follow the original energy order and inputs are not changed.
    """
    baseline, effective_width = moving_poisson_mean_x(
        x, source_counts, background_counts, background_scale, exposure,
        bin_width, width, min_points=min_points, max_width=max_width,
        return_width=True, exclude_mask=exclude_mask)
    x = np.asarray(x, dtype=float)
    source = np.asarray(source_counts, dtype=float)
    background = np.asarray(background_counts, dtype=float)
    widths = np.asarray(bin_width, dtype=float)
    scale, exposure = float(background_scale), float(exposure)
    excluded = (np.zeros(x.shape, dtype=bool) if exclude_mask is None
                else np.asarray(exclude_mask, dtype=bool).copy())
    if (not np.allclose(source, np.rint(source), rtol=0, atol=1e-6)
            or not np.allclose(background, np.rint(background), rtol=0, atol=1e-6)):
        raise ValueError('Line-resistant Poisson continuum requires raw integer source and background counts.')
    # Remove only representational roundoff from the conditional count law.
    source, background = np.rint(source), np.rint(background)
    for _ in range(5 if len(x) else 0):
        seeds = _poisson_continuum_seeds(
            x, source, background, scale, exposure, widths,
            baseline, effective_width, excluded)
        if not np.any(seeds):
            break
        excluded |= _contiguous_seed_halo(x, widths, seeds)
        baseline, effective_width = moving_poisson_mean_x(
            x, source_counts, background_counts, scale, exposure,
            widths, width, min_points=min_points, max_width=max_width,
            return_width=True, exclude_mask=excluded)
    if return_width and return_mask:
        return baseline, effective_width, excluded
    if return_width:
        return baseline, effective_width
    if return_mask:
        return baseline, excluded
    return baseline


def poisson_base_calculator(
    x,
    source_counts,
    background_counts,
    background_scale,
    exposure,
    bin_width,
    baseline_window=0.4,
    max_range_fraction=0.2,
    min_points=3,
    return_info=False,
    *,
    max_window_width=None,
):
    """Estimate a sparse-count continuum without changing the fitting grid.

    The estimate is a running count/exposure-width ratio, not the median of
    individual bin densities. Thus sparse nonzero counts contribute even
    when most bins are empty. Strong positive excesses and their contiguous
    neighbors are conservatively masked only from continuum estimation using
    ``moving_poisson_continuum_x``. Widths, reflected edges and adaptive limits
    follow ``base_calculator``; no positive floor is introduced. Observed and
    synthetic spectra must use the same rule, recomputing their own masks.

    ``max_window_width`` optionally supplies a finite, positive physical
    cap instead of deriving one from ``max_range_fraction`` and this grid's
    centers. ``min_points`` remains strict throughout the supplied context.

    This is a smoothed empirical continuum, not a physical continuum model.
    Weak/broad features and features under strong background can still be
    absorbed. Its uncertainty is not propagated by the line fit, which treats
    the estimated continuum as fixed. The heuristic masking thresholds do not
    provide an exact false-exclusion guarantee for adaptive or repeated tests.
    """
    x = np.asarray(x, dtype=float)
    max_window_width = _resolve_baseline_window_limit(
        x, max_range_fraction, max_window_width)
    window_width = choose_baseline_window(
        x, baseline_window=baseline_window, max_range_fraction=max_range_fraction,
        max_window_width=max_window_width)
    baseline, effective_width, excluded = moving_poisson_continuum_x(
        x, source_counts, background_counts, background_scale, exposure,
        bin_width, width=window_width, min_points=min_points,
        max_width=max_window_width, return_width=True, return_mask=True,
    )
    if return_info:
        return baseline, {
            "window_width": effective_width,
            "initial_window_width": window_width,
            "max_window_width": max_window_width,
            "baseline_window": baseline_window,
            "max_range_fraction": max_range_fraction,
            "energy_range": np.nanmax(x) - np.nanmin(x),
            "min_points": min_points,
            "excluded_mask": excluded,
        }
    return baseline
