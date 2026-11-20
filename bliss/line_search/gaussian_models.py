"""Gaussian models and initial-parameter generators used by BLiSS fits."""
import numpy as np


MIN_INSTRUMENTAL_SIGMA_FRACTION = 0.1

def validate_sigma_bounds(min_sigma_line, max_sigma_line):
    """Reject invalid configured bounds before starting any fit or simulation."""
    for name, value in (("min_sigma_line", min_sigma_line), ("max_sigma_line", max_sigma_line)):
        if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive.")
    if min_sigma_line >= max_sigma_line:
        raise ValueError("max_sigma_line must exceed min_sigma_line.")


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


def gaussian_area_error(amplitude, sigma, eamplitude, esigma,
                        cov_amplitude_sigma, *, width_fixed_instrumental=False):
    """Propagate the full amplitude/width covariance to Gaussian area.
    For area = sqrt(2*pi) * amplitude * sigma, the first-order variance is
    2*pi * (sigma**2 * eamplitude**2 + amplitude**2 * esigma**2
            + 2 * amplitude * sigma * cov_amplitude_sigma).
    Inputs broadcast as NumPy arrays. Missing/nonfinite covariance, invalid
    parameter errors, or an inconsistent covariance block return NaN; missing
    covariance is never interpreted as zero. This is a local linear error
    estimate, not a calibrated detection significance.

    With an explicitly true ``width_fixed_instrumental`` flag, sigma is a
    fixed model value, not an estimated parameter. Its conditional area error
    is sqrt(2*pi) * sigma * eamplitude; esigma and its covariance are unused.
    This does not include uncertainty in the instrumental response.
    """
    flags = np.asarray(width_fixed_instrumental, dtype=object)
    fixed = np.fromiter(
        (isinstance(flag, (bool, np.bool_)) and bool(flag) for flag in flags.flat),
        dtype=bool, count=flags.size,
    ).reshape(flags.shape)
    amp, width, ea, ew, cov, fixed = np.broadcast_arrays(*[
        np.asarray(v, dtype=float) for v in
        (amplitude, sigma, eamplitude, esigma, cov_amplitude_sigma)], fixed)
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
        fixed_error = np.sqrt(2 * np.pi) * width * ea
        valid_fixed = (np.isfinite(amp) & (amp > 0)
                       & np.isfinite(width) & (width > 0)
                       & np.isfinite(ea) & (ea > 0) & np.isfinite(fixed_error))
    return np.where(fixed,
                    np.where(valid_fixed, fixed_error, np.nan),
                    np.where(valid & np.isfinite(error), error, np.nan))

def _amplitude_upper_bound(y, uncertainties=None):
    """Use the highest observed bin plus the error of that same bin."""
    values = np.asarray(y, dtype=float)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError('Amplitude bounds require a nonempty array of finite bin values.')
    peak = int(np.argmax(values))
    peak_error = 0.0
    if uncertainties is not None:
        errors = np.asarray(uncertainties, dtype=float)
        if errors.shape != values.shape:
            raise ValueError('Amplitude-bound uncertainties must match the bin values.')
        peak_error = errors[peak]
        if not np.isfinite(peak_error) or peak_error < 0:
            raise ValueError('The maximum bin must have a finite, nonnegative uncertainty.')
    upper = float(values[peak] + peak_error)
    if not np.isfinite(upper) or upper <= 0:
        raise ValueError('The maximum bin plus its uncertainty must be finite and positive.')
    return upper


def p0_generator(x, y, good_peaks_dataframe, response_sigma=None,
                 max_sigma_line=0.1, min_sigma_line=1e-6, *, uncertainties=None):
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
        valid resolution min_sigma_line is used. The effective lower bound is
        the larger of min_sigma_line and the instrumental bound.
    min_sigma_line : float, default: 1e-6
        Positive absolute lower sigma bound in keV.
    max_sigma_line : float, default: 0.1
        Configured upper sigma bound in keV; local fits also cap it at max(x)-min(x).
        Must be finite, positive and greater than the instrumental lower bound.
    uncertainties : array-like or None, default: None
        Errors on y. All component amplitudes are bounded by the maximum
        observed bin plus the error of that same bin (first maximum if tied).
        Initial amplitudes are clipped to [0, upper]. BLiSS fits always supply
        these errors. For legacy direct calls without errors, upper is max(y).

    Returns
    -------
    tuple
        ``(p0, bounds)`` where ``p0`` is a flat list of Gaussian initial guesses and
        ``bounds`` is the pair of lower and upper bounds expected by
        ``scipy.optimize.curve_fit``.
    """
    validate_sigma_bounds(min_sigma_line, max_sigma_line)
    p0, bound_low, bound_high = ([], [], [])
    if len(good_peaks_dataframe) == 0:
        return (p0, (bound_low, bound_high))
    amplitude_max = _amplitude_upper_bound(y, uncertainties)
    max_sigma_line = min(max_sigma_line, float(np.ptp(np.asarray(x, dtype=float))))
    for i in range(len(good_peaks_dataframe)):
        p0.append(float(np.clip(y[good_peaks_dataframe.position.loc[i]], 0, amplitude_max)))
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
        sigma_floor = max(min_sigma_line, sigma_floor)
        if not np.isfinite(sigma_guess) or sigma_guess <= 0:
            if (good_peaks_dataframe.twidth.loc[i] < 0.05) & (good_peaks_dataframe.twidth.loc[i] > 0):
                sigma_guess = good_peaks_dataframe.twidth.loc[i]
            else:
                sigma_guess = 0.001
        if max_sigma_line <= sigma_floor:
            raise ValueError('max_sigma_line must exceed the effective sigma lower bound.')
        p0.append(float(np.clip(sigma_guess, sigma_floor, max_sigma_line)))
        bound_low.append(0.0)
        bound_low.append(good_peaks_dataframe.energy.loc[i] * 0.99)
        bound_low.append(sigma_floor)
        bound_high.append(amplitude_max)
        bound_high.append(good_peaks_dataframe.energy.loc[i] * 1.01)
        bound_high.append(max_sigma_line)
    return (p0, (bound_low, bound_high))

def p0_generator_final(x, y, clean_lines, response_sigma=None,
                       max_sigma_line=0.1, min_sigma_line=1e-6, *, uncertainties=None):
    """Build initial parameters and bounds for the final multi-line fit.

    Parameters
    ----------
    x : array-like
        Full spectral coordinate grid of the global fit.
    y : array-like
        Observed spectral values over the global fit interval, including
        the continuum. This full interval defines the amplitude-bound block.
    clean_lines : pandas.DataFrame
        Candidate table containing ``amplitude``, ``center``, and ``sigma`` columns
        from the preliminary line search.
    response_sigma : array-like or None, default: None
        Instrumental Gaussian-equivalent sigma evaluated on ``x``. When supplied,
        the response width at each candidate center is used to initialize the final
        fit. The lower width bound is fixed at 0.1 times that instrumental
        sigma or min_sigma_line, whichever is larger. Without valid resolution,
        min_sigma_line is used.
    min_sigma_line : float, default: 1e-6
        Positive absolute lower sigma bound in keV.
    max_sigma_line : float, default: 0.1
        Configured upper sigma bound in keV, capped by each saved fit_block_width.
        Legacy tables without that width retain the configured cap.
        Must be finite, positive and greater than the instrumental lower bound.
    uncertainties : array-like or None, default: None
        Errors on y. All component amplitudes are bounded by the maximum
        observed bin plus the error of that same bin (first maximum if tied).
        Initial amplitudes are clipped to [0, upper]. BLiSS fits always supply
        these errors. For legacy direct calls without errors, upper is max(y).

    Returns
    -------
    tuple
        ``(p0, bounds)`` for the final call to ``curve_fit``.
    """
    validate_sigma_bounds(min_sigma_line, max_sigma_line)
    p0, bound_low, bound_high = ([], [], [])
    if len(clean_lines) == 0:
        return (p0, (bound_low, bound_high))
    amplitude_max = _amplitude_upper_bound(y, uncertainties)
    for i in range(len(clean_lines)):
        block_width = clean_lines.iloc[i].get('fit_block_width', np.nan)
        if np.isnan(block_width):
            sigma_upper = max_sigma_line  # Legacy table without local context.
        elif not np.isfinite(block_width) or block_width <= 0:
            raise ValueError('fit_block_width must be finite and positive when supplied.')
        else:
            sigma_upper = min(max_sigma_line, float(block_width))
        p0.append(float(np.clip(clean_lines.amplitude.loc[i], 0, amplitude_max)))
        p0.append(clean_lines.center.loc[i])
        if response_sigma is not None:
            response_sigma_array = np.asarray(response_sigma, dtype=float)
            sigma_guess = np.interp(clean_lines.center.loc[i], np.asarray(x, dtype=float), response_sigma_array)
        else:
            sigma_guess = np.nan
        # Freeze the numerical width floor at the initial candidate energy.
        sigma_floor = (MIN_INSTRUMENTAL_SIGMA_FRACTION * sigma_guess
                       if np.isfinite(sigma_guess) and sigma_guess > 0 else 0.0)
        sigma_floor = max(min_sigma_line, sigma_floor)
        if not np.isfinite(sigma_guess) or sigma_guess <= 0:
            if (clean_lines.sigma.loc[i] < 0.05) & (clean_lines.sigma.loc[i] > 0):
                sigma_guess = clean_lines.sigma.loc[i]
            else:
                sigma_guess = 0.05
        if sigma_upper <= sigma_floor:
            raise ValueError('max_sigma_line must exceed the effective sigma lower bound.')
        p0.append(float(np.clip(sigma_guess, sigma_floor, sigma_upper)))
        bound_low.append(0)
        bound_low.append(clean_lines.center.loc[i] - 0.1)
        bound_low.append(sigma_floor)
        bound_high.append(amplitude_max)
        bound_high.append(clean_lines.center.loc[i] + 0.1)
        bound_high.append(sigma_upper)
    return (p0, (bound_low, bound_high))
