"""Gaussian models and initial-parameter generators used by BLiSS fits."""
import numpy as np
from scipy.optimize import curve_fit

WIDTH_FIT_COLUMNS = ['sigma_fixed', 'sigma_reference_center', 'fit_uncertainty']
LOCAL_MODEL_COLUMNS = ['local_model_selection', 'local_n_components_initial',
                       'local_n_components_selected', 'local_bic_single',
                       'local_bic_multiple', 'local_bic_delta',
                       'local_model_selection_message']

MIN_INSTRUMENTAL_SIGMA_FRACTION = 0.1

def gaussian(x, amplitude, center, sigma):
    """Evaluate a single Gaussian profile.

    Parameters
    ----------
    x : array-like
        Coordinates where the profile is evaluated.
    amplitude : float
        Peak height of the Gaussian above zero.
    center : float
        Coordinate of the Gaussian centroid.
    sigma : float
        Standard deviation of the Gaussian in the same units as ``x``.

    Returns
    -------
    numpy.ndarray
        Gaussian profile evaluated at ``x``.
    """
    return amplitude * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))

def n_gaussian(x, *params):
    """Evaluate a sum of Gaussian components.

    Parameters
    ----------
    x : array-like
        Coordinates where the combined profile is evaluated.
    *params : float
        Flat parameter sequence grouped as ``amplitude, center, sigma`` for each
        Gaussian component.

    Returns
    -------
    numpy.ndarray
        Sum of all Gaussian components evaluated at ``x``.
    """
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        amplitude, center, sigma = params[i:i + 3]
        y += gaussian(x, amplitude, center, sigma)
    return y


def fit_gaussian_components(x, y, p0, bounds, uncertainties, *,
                            response_sigma=None, fixed_widths=None,
                            reference_centers=None, maxfev=100000,
                            optimizer=curve_fit):
    """Fit a Gaussian sum, then fix sub-instrumental widths in one refit.

    Only free parameters are passed to the optimizer. Inherited fixed widths
    remain fixed, including in the global fit. After the first fit, each free
    sigma below the valid instrumental sigma at its fitted center is replaced
    by that instrumental value; the block is refitted at most once. The
    lookup center/value are frozen for that refit, not updated iteratively.
    Missing/invalid response values or centers outside the response grid do
    not trigger fixation. Other free
    widths are not reclassified after the second fit.

    Returns full parameters/covariance, the per-component fixed mask and the
    resolution lookup centers. Covariance rows/columns for fixed sigmas are
    zero *conditionally*, not measured zero-error widths. Callers export
    their esigma as NaN with sigma_fixed=True. All parameter uncertainties in
    a block with any fixed width are conditional on those imposed widths.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    initial = np.asarray(p0, dtype=float).copy()
    lower, upper = (np.asarray(b, dtype=float) for b in bounds)
    count = len(initial) // 3
    widths = (np.full(count, np.nan) if fixed_widths is None
              else np.asarray(fixed_widths, dtype=float).copy())
    centers = (np.full(count, np.nan) if reference_centers is None
               else np.asarray(reference_centers, dtype=float).copy())
    if len(initial) != 3 * count or widths.shape != (count,) or centers.shape != (count,):
        raise ValueError('Gaussian parameter/width metadata shapes do not match.')
    if np.any(~np.isnan(widths) & (~np.isfinite(widths) | (widths <= 0))):
        raise ValueError('Fixed Gaussian widths must be finite and positive.')
    fixed = np.isfinite(widths)
    initial[2::3] = np.where(fixed, widths, initial[2::3])

    def solve(start):
        free = np.ones(len(start), dtype=bool)
        free[2::3] = ~fixed
        template = start.copy()

        def model(coordinates, *parameters):
            full = template.copy()
            full[free] = parameters
            return n_gaussian(coordinates, *full)

        optimal, covariance = optimizer(
            model, x, y, p0=start[free], bounds=(lower[free], upper[free]),
            sigma=uncertainties, absolute_sigma=True, maxfev=maxfev)
        template[free] = optimal
        full_covariance = np.zeros((len(start), len(start)), dtype=float)
        full_covariance[np.ix_(free, free)] = covariance
        return template, full_covariance

    optimal, covariance = solve(initial)
    if response_sigma is not None:
        response = np.asarray(response_sigma, dtype=float)
        if response.shape != x.shape:
            raise ValueError('Instrumental resolution must align with the fit grid.')
        instrumental = np.interp(optimal[1::3], x, response)
        newly_fixed = (~fixed & np.isfinite(instrumental) & (instrumental > 0)
                       & (optimal[1::3] >= x[0]) & (optimal[1::3] <= x[-1])
                       & (optimal[2::3] < instrumental))
        if np.any(newly_fixed):
            centers[newly_fixed] = optimal[1::3][newly_fixed]
            fixed |= newly_fixed
            optimal[2::3][newly_fixed] = instrumental[newly_fixed]
            try:
                optimal, covariance = solve(optimal)
            except (RuntimeError, ValueError) as exc:
                raise RuntimeError(f'Instrumental-width refit failed: {exc}') from exc
    return optimal, covariance, fixed, centers


def gaussian_area_error(amplitude, sigma, eamplitude, esigma,
                        cov_amplitude_sigma, sigma_fixed=False):
    """Propagate the full amplitude/width covariance to Gaussian area.

    For area = sqrt(2*pi) * amplitude * sigma, the first-order variance is
    2*pi * (sigma**2 * eamplitude**2 + amplitude**2 * esigma**2
            + 2 * amplitude * sigma * cov_amplitude_sigma).
    Inputs broadcast as NumPy arrays. Missing/nonfinite covariance, invalid
    parameter errors, or an inconsistent covariance block return NaN; missing
    covariance is never interpreted as zero. This is a local linear error
    estimate, not a calibrated detection significance. For explicitly fixed
    widths, esigma may be NaN or zero and cov_amplitude_sigma must be zero;
    the returned error is conditional: sqrt(2*pi) * abs(sigma) * eamplitude.
    """
    amp, width, ea, ew, cov, fixed = np.broadcast_arrays(*[
        np.asarray(v, dtype=float) for v in
        (amplitude, sigma, eamplitude, esigma, cov_amplitude_sigma, sigma_fixed)])
    fixed = fixed == 1
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        rho = (cov / ea) / ew
        valid = (np.isfinite(amp) & np.isfinite(width)
                 & np.isfinite(ea) & (ea > 0)
                 & np.isfinite(ew) & (ew > 0) & np.isfinite(cov)
                 & (np.abs(rho) <= 1 + 32 * np.finfo(float).eps))
        # Only round-off beyond |rho|=1 is clipped; invalid blocks stay NaN.
        rho = np.clip(rho, -1, 1)
        a, b = width * ea, amp * ew
        scale = np.maximum(np.abs(a), np.abs(b))
        u = np.divide(a, scale, out=np.zeros_like(scale), where=scale > 0)
        v = np.divide(b, scale, out=np.zeros_like(scale), where=scale > 0)
        # A scaled quadratic form avoids subtracting two large variances
        # when amplitude and width are strongly anticorrelated.
        error = (np.sqrt(2 * np.pi) * scale
                 * np.hypot(u + rho * v, np.sqrt((1 - rho) * (1 + rho)) * v))
    free_error = np.where(valid & np.isfinite(error), error, np.nan)
    with np.errstate(over='ignore', invalid='ignore'):
        conditional_error = np.sqrt(2 * np.pi) * np.abs(width) * ea
    fixed_valid = (np.isfinite(amp) & np.isfinite(width) & (width > 0)
                   & np.isfinite(ea) & (ea > 0)
                   & (np.isnan(ew) | (ew == 0)) & (cov == 0)
                   & np.isfinite(conditional_error))
    return np.where(fixed, np.where(fixed_valid, conditional_error, np.nan), free_error)

def p0_generator(x, y, good_peaks_dataframe, response_sigma=None):
    """Build initial parameters and bounds for fitting local candidate peaks.

    Parameters
    ----------
    x : array-like
        Energy or coordinate grid of the candidate block.
    y : array-like
        Observed values in the candidate block.
    good_peaks_dataframe : pandas.DataFrame
        Peak table returned by ``find_peaks_new`` after candidate filtering. The
        function uses ``position``, ``energy``, and ``twidth``.
    response_sigma : array-like or None, default: None
        Instrumental Gaussian-equivalent sigma evaluated on ``x``. When supplied,
        it is used as the initial width at each candidate energy instead of the
        generic 0.05 coordinate-unit fallback. A fixed lower bound of 0.1 times
        the valid instrumental sigma at the initial peak is imposed; without
        valid resolution the previous zero lower bound is retained.

    Returns
    -------
    tuple
        ``(p0, bounds)`` where ``p0`` is a flat list of Gaussian initial guesses and
        ``bounds`` is the pair of lower and upper bounds expected by
        ``scipy.optimize.curve_fit``.
    """
    p0, bound_low, bound_high = ([], [], [])
    for i in range(len(good_peaks_dataframe)):
        p0.append(y[good_peaks_dataframe.position.loc[i]])
        p0.append(good_peaks_dataframe.energy.loc[i])
        position = int(good_peaks_dataframe.position.loc[i])
        if response_sigma is not None:
            response_sigma = np.asarray(response_sigma, dtype=float)
            sigma_guess = response_sigma[position]
        else:
            sigma_guess = np.nan
        # Freeze the numerical width floor at the initial candidate energy.
        sigma_floor = (MIN_INSTRUMENTAL_SIGMA_FRACTION * sigma_guess
                       if np.isfinite(sigma_guess) and sigma_guess > 0 else 0.0)
        if not np.isfinite(sigma_guess) or sigma_guess <= 0:
            if (good_peaks_dataframe.twidth.loc[i] < 0.05) & (good_peaks_dataframe.twidth.loc[i] > 0):
                sigma_guess = good_peaks_dataframe.twidth.loc[i]
            else:
                sigma_guess = 0.05
        p0.append(sigma_guess)
        bound_low.append(y[good_peaks_dataframe.position.loc[i]] * 0)
        bound_low.append(good_peaks_dataframe.energy.loc[i] * 0.99)
        bound_low.append(sigma_floor)
        bound_high.append(y[good_peaks_dataframe.position.loc[i]] * 100)
        bound_high.append(good_peaks_dataframe.energy.loc[i] * 1.01)
        bound_high.append(max(0.25, sigma_guess * 2.0))
    return (p0, (bound_low, bound_high))

def p0_generator_final(x, y, clean_lines, response_sigma=None):
    """Build initial parameters and bounds for the final multi-line fit.

    Parameters
    ----------
    x : array-like
        Full spectral coordinate grid. Present for API consistency; the current
        implementation uses the candidate table directly.
    y : array-like
        Full spectral values. Present for API consistency; the current
        implementation uses the candidate table directly.
    clean_lines : pandas.DataFrame
        Candidate table containing ``amplitude``, ``center``, and ``sigma`` columns
        from the preliminary line search.
    response_sigma : array-like or None, default: None
        Instrumental Gaussian-equivalent sigma evaluated on ``x``. When supplied,
        the response width at each candidate center is used to initialize the final
        fit. The lower width bound is fixed at 0.1 times that instrumental
        sigma; without valid resolution the previous zero bound is retained.

    Returns
    -------
    tuple
        ``(p0, bounds)`` for the final call to ``curve_fit``.
    """
    p0, bound_low, bound_high = ([], [], [])
    for i in range(len(clean_lines)):
        p0.append(clean_lines.amplitude.loc[i])
        p0.append(clean_lines.center.loc[i])
        if response_sigma is not None:
            response_sigma_array = np.asarray(response_sigma, dtype=float)
            sigma_guess = np.interp(clean_lines.center.loc[i], np.asarray(x, dtype=float), response_sigma_array)
        else:
            sigma_guess = np.nan
        # Freeze the numerical width floor at the initial candidate energy.
        sigma_floor = (MIN_INSTRUMENTAL_SIGMA_FRACTION * sigma_guess
                       if np.isfinite(sigma_guess) and sigma_guess > 0 else 0.0)
        if not np.isfinite(sigma_guess) or sigma_guess <= 0:
            if (clean_lines.sigma.loc[i] < 0.05) & (clean_lines.sigma.loc[i] > 0):
                sigma_guess = clean_lines.sigma.loc[i]
            else:
                sigma_guess = 0.05
        p0.append(sigma_guess)
        bound_low.append(0)
        bound_low.append(clean_lines.center.loc[i] - 0.1)
        bound_low.append(sigma_floor)
        bound_high.append(clean_lines.amplitude.loc[i] * 10)
        bound_high.append(clean_lines.center.loc[i] + 0.1)
        bound_high.append(max(clean_lines.sigma.loc[i] + 0.01, sigma_guess * 2.0))
    return (p0, (bound_low, bound_high))
