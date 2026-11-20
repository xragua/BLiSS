"""Poisson Gaussian fits on raw source/background counts.

The likelihood uses bin-integrated models, including empty bins. Supplied
observational error bars are not likelihood weights. With background, its
unknown mean is profiled separately in every bin using the joint source and
background Poisson likelihood (the W-statistic construction).

Returned covariances are local expected-information approximations with a
fixed continuum. They are not exact Poisson confidence intervals, especially
near parameter bounds or for weak/unresolved lines.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr


@dataclass(frozen=True)
class PoissonFitData:
    """Raw counts and conversion factors aligned with a fitting grid."""

    source_counts: np.ndarray
    background_counts: np.ndarray
    background_scale: float
    exposure: float
    bin_width: np.ndarray

    def __post_init__(self):
        arrays = {}
        for name in ('source_counts', 'background_counts', 'bin_width'):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.ndim != 1 or not np.all(np.isfinite(value)):
                raise ValueError(f'Poisson {name} must be a finite one-dimensional array.')
            arrays[name] = value
        if len({value.shape for value in arrays.values()}) != 1:
            raise ValueError('Poisson source/background counts and bin widths must have matching shapes.')
        for name in ('source_counts', 'background_counts'):
            value = arrays[name]
            if np.any(value < 0) or not np.allclose(value, np.rint(value), rtol=0, atol=1e-6):
                raise ValueError(f'Poisson {name} must contain nonnegative raw integer counts.')
            # Counts reconstructed from rates or net counts can have roundoff.
            arrays[name] = np.rint(value)
        if np.any(arrays['bin_width'] <= 0):
            raise ValueError('Poisson bin widths must be positive.')
        for name, positive in (('exposure', True), ('background_scale', False)):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
                    or not np.isfinite(value) or (value <= 0 if positive else value < 0)):
                qualifier = 'positive' if positive else 'nonnegative'
                raise ValueError(f'Poisson {name} must be finite and {qualifier}.')
            object.__setattr__(self, name, float(value))
        for name, value in arrays.items():
            object.__setattr__(self, name, value)

    def sliced(self, selection):
        """Select bins while preserving exposure and background scaling."""
        return PoissonFitData(
            self.source_counts[selection], self.background_counts[selection],
            self.background_scale, self.exposure, self.bin_width[selection],
        )


def validate_poisson_groups(energy, bin_width, group):
    """Reject grouped bins that cannot be represented by one physical interval.

    The integrated line model uses a center and total width for each output
    bin. Combining channels across a gap would invent a different integration
    interval. Dropped channels and nonconsecutive group ids are supported;
    the grouping is never changed by this validation.
    """
    energy = np.asarray(energy, dtype=float)
    width = np.asarray(bin_width, dtype=float)
    group = np.asarray(group)
    if (energy.ndim != 1 or width.shape != energy.shape or group.shape != energy.shape
            or not np.all(np.isfinite(energy))
            or not np.all(np.isfinite(width) & (width > 0))
            or not np.all(np.isfinite(group)) or np.any(group != np.floor(group))):
        raise ValueError('Poisson grouping requires finite energies, positive widths and matching integer group ids.')
    keep = group >= 0
    if not np.any(keep):
        return
    _, inverse = np.unique(group[keep], return_inverse=True)
    left = energy[keep] - width[keep] / 2
    right = energy[keep] + width[keep] / 2
    total_width = np.bincount(inverse, weights=width[keep])
    members = np.bincount(inverse)
    first = np.full(total_width.shape, np.inf)
    last = np.full(total_width.shape, -np.inf)
    np.minimum.at(first, inverse, left)
    np.maximum.at(last, inverse, right)
    # Allow accumulated floating-point roundoff, not a physical missing span.
    coordinate_scale = np.maximum.reduce((np.abs(first), np.abs(last), total_width,
                                         np.ones_like(total_width)))
    tolerance = 64 * np.finfo(float).eps * coordinate_scale * members
    if np.any(np.abs((last - first) - total_width) > tolerance):
        raise ValueError(
            'Poisson fitting requires contiguous channels within each rebinned bin. '
            'This grouping crosses gaps or overlapping channels. '
            "Use rebin_method='optimal' or 'counts', or use finer binning.")


def _validate_grid(energy, baseline, data):
    energy = np.asarray(energy, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if energy.shape != data.source_counts.shape or baseline.shape != energy.shape:
        raise ValueError('Poisson energy, baseline and count arrays must have matching shapes.')
    if not np.all(np.isfinite(energy)):
        raise ValueError('Poisson energies must be finite.')
    if not np.all(np.isfinite(baseline) & (baseline >= 0)):
        raise ValueError('Poisson baseline must be finite and nonnegative.')
    return energy, baseline


def _integrated_lines(energy, width, params, *, derivative=False):
    """Gaussian integrals and their exact parameter derivatives."""
    params = np.asarray(params, dtype=float)
    if params.ndim != 1 or params.size % 3 or not np.all(np.isfinite(params)):
        raise ValueError('Gaussian parameters must be finite amplitude/center/sigma triples.')
    if np.any(params[::3] < 0) or np.any(params[2::3] <= 0):
        raise ValueError('Poisson Gaussian amplitudes must be nonnegative and sigmas positive.')
    result = np.zeros_like(energy)
    jac = np.empty((energy.size, params.size)) if derivative else None
    root_two_pi = np.sqrt(2 * np.pi)
    for i in range(0, params.size, 3):
        amplitude, center, sigma = params[i:i + 3]
        lo = (energy - .5 * width - center) / sigma
        hi = (energy + .5 * width - center) / sigma
        # The survival-function difference retains precision in positive tails.
        integral_probability = np.where(
            lo >= 0, ndtr(-lo) - ndtr(-hi), ndtr(hi) - ndtr(lo),
        )
        unit_integral = sigma * root_two_pi * integral_probability
        result += amplitude * unit_integral
        if derivative:
            exp_lo, exp_hi = np.exp(-.5 * lo**2), np.exp(-.5 * hi**2)
            jac[:, i] = unit_integral
            jac[:, i + 1] = amplitude * (exp_lo - exp_hi)
            jac[:, i + 2] = amplitude * (
                root_two_pi * integral_probability + lo * exp_lo - hi * exp_hi)
    return (result, jac) if derivative else result


def gaussian_count_model(energy, baseline, data, params=None):
    """Expected net source counts from a fixed continuum and integrated lines."""
    energy, baseline = _validate_grid(energy, baseline, data)
    expected = baseline * data.exposure * data.bin_width
    if params is not None:
        expected = expected + data.exposure * _integrated_lines(energy, data.bin_width, params)
    if not np.all(np.isfinite(expected) & (expected >= 0)):
        raise ValueError('Poisson expected counts must be finite and nonnegative.')
    return expected


def profile_background_counts(source_counts, background_counts, background_scale, source_model):
    """Maximize the joint Poisson likelihood over each background-bin mean.

    ``source_model`` is the expected source-only count contribution. The
    source-region mean is ``source_model + background_scale * background``.
    The nonnegative quadratic root is evaluated without subtracting nearly
    equal positive numbers.
    """
    source = np.asarray(source_counts, dtype=float)
    background = np.asarray(background_counts, dtype=float)
    model = np.asarray(source_model, dtype=float)
    scale = float(background_scale)
    if scale == 0:
        return background.copy()
    a = scale * (1 + scale)
    b = (1 + scale) * model - scale * (source + background)
    root = np.hypot(b, 2 * np.sqrt(a) * np.sqrt(background) * np.sqrt(model))
    profiled = np.empty_like(model)
    positive_b = b > 0
    profiled[~positive_b] = (root[~positive_b] - b[~positive_b]) / (2 * a)
    # Rationalized root: 2 * background * model / (root + b).
    profiled[positive_b] = ((2 * background[positive_b] / (root[positive_b] + b[positive_b]))
                            * model[positive_b])
    return profiled


def poisson_deviance_residuals(observed, expected):
    """Signed square-root Poisson deviances, including exact zero-count bins.

    The smallest positive float only regularizes log(0) at an impossible
    positive observation with exactly zero model; it is not a count floor
    or an observational uncertainty. Expected zero with observed zero gives
    exactly zero residual.
    """
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if observed.shape != expected.shape:
        raise ValueError('Poisson observations and expectations must have matching shapes.')
    if (not np.all(np.isfinite(observed) & (observed >= 0))
            or not np.all(np.isfinite(expected) & (expected >= 0))):
        raise ValueError('Poisson observations and expectations must be finite and nonnegative.')
    deviance = 2 * expected.copy()
    positive = observed > 0
    n = observed[positive]
    mu = np.maximum(expected[positive], np.finfo(float).tiny)
    delta = (mu - n) / n
    close = np.abs(delta) < 1e-3
    term = np.empty_like(n)
    d = delta[close]
    # Series of delta - log1p(delta), preserving tiny residuals near equality.
    term[close] = n[close] * d**2 * (
        .5 + d * (-1/3 + d * (.25 + d * (-.2 + d * (1/6 + d * (-1/7 + d/8))))))
    term[~close] = mu[~close] - n[~close] + n[~close] * (np.log(n[~close]) - np.log(mu[~close]))
    deviance[positive] = 2 * np.maximum(term, 0)
    return np.sign(expected - observed) * np.sqrt(deviance)


def _profiled_means(data, source_model):
    background = profile_background_counts(
        data.source_counts, data.background_counts, data.background_scale, source_model)
    return source_model + data.background_scale * background, background


def poisson_model_uncertainty(energy, baseline, data, params=None):
    """Model-based net-density noise for diagnostics, never likelihood weights."""
    source_model = gaussian_count_model(energy, baseline, data, params)
    gross, background = _profiled_means(data, source_model)
    return (np.sqrt(gross + data.background_scale**2 * background)
            / (data.exposure * data.bin_width))


def _information_covariance(energy, data, params, source_model, free_parameters=None):
    """Expected Fisher covariance for free parameters after profiling background.

    Fixed parameters are removed from the information matrix *before* inversion.
    Their rows and columns in the returned full-size covariance are NaN.
    """
    _, derivative = _integrated_lines(energy, data.bin_width, params, derivative=True)
    free = (np.ones(len(params), dtype=bool) if free_parameters is None
            else np.asarray(free_parameters, dtype=bool))
    if free.shape != (len(params),):
        raise ValueError('Free-parameter mask must match the Gaussian parameter vector.')
    derivative = derivative[:, free]
    derivative *= data.exposure
    gross, background = _profiled_means(data, source_model)
    variance = gross + data.background_scale**2 * background
    invalid = np.full((len(params), len(params)), np.nan)
    if not np.any(free):
        return invalid
    # Gaussian tails can underflow to exactly zero. They add no numerically
    # resolvable information. Genuine zero-amplitude components still have
    # zero center/width columns and fail the rank check below.
    weighted = np.divide(
        derivative, np.sqrt(variance)[:, None], out=np.zeros_like(derivative),
        where=variance[:, None] > 0,
    )
    scales = np.linalg.norm(weighted, axis=0)
    if (weighted.shape[0] < weighted.shape[1]
            or not np.all(np.isfinite(scales) & (scales > 0))):
        return invalid
    try:
        _, singular, vt = np.linalg.svd(weighted / scales, full_matrices=False)
    except np.linalg.LinAlgError:
        return invalid
    tolerance = np.finfo(float).eps * max(weighted.shape) * singular[0]
    if np.any(singular <= tolerance):
        return invalid
    covariance = ((vt.T / singular**2) @ vt) / scales[:, None] / scales[None, :]
    if np.all(np.isfinite(covariance)):
        invalid[np.ix_(free, free)] = covariance
    return invalid


def fit_poisson_gaussians(energy, baseline, data, p0, bounds, max_nfev=100000,
                         *, fixed_parameters=None):
    """Fit bounded Gaussian lines with Cash or profiled Poisson background.

    Empty bins remain in the likelihood. The continuum is fixed, in density
    units; it is not subtracted from the observed Poisson counts. Gaussian
    amplitudes remain peak densities and each Gaussian is integrated over
    the physical bin edges. Covariance is not scaled by residual scatter.
    Rank-deficient information returns NaNs, keeping numerical invalidity
    visible to the common candidate-quality policy.

    ``fixed_parameters``, when supplied, is a full parameter vector: finite
    values are held fixed and NaNs denote free parameters. Only free
    parameters enter the optimizer and the Fisher information inversion.
    The returned covariance retains full size with NaN fixed rows/columns.
    """
    energy, baseline = _validate_grid(energy, baseline, data)
    p0 = np.asarray(p0, dtype=float)
    if energy.size == 0 or p0.size == 0:
        raise ValueError('A Poisson Gaussian fit needs bins and at least one component.')
    p0 = p0.copy()
    fixed = (np.full(p0.shape, np.nan) if fixed_parameters is None
             else np.asarray(fixed_parameters, dtype=float))
    if fixed.shape != p0.shape or np.any(np.isinf(fixed)):
        raise ValueError('Fixed parameters must match p0 and contain finite values or NaN.')
    free = np.isnan(fixed)
    lower = np.broadcast_to(np.asarray(bounds[0], dtype=float), p0.shape)
    upper = np.broadcast_to(np.asarray(bounds[1], dtype=float), p0.shape)
    if np.any((fixed[~free] < lower[~free]) | (fixed[~free] > upper[~free])):
        raise ValueError('Fixed Gaussian parameters must lie within their configured bounds.')
    p0[~free] = fixed[~free]
    gaussian_count_model(energy, baseline, data, p0)  # Validate parameter layout.

    def expand(free_values):
        parameters = p0.copy()
        parameters[free] = free_values
        return parameters

    def residual(free_values):
        parameters = expand(free_values)
        model = gaussian_count_model(energy, baseline, data, parameters)
        gross, background = _profiled_means(data, model)
        result = poisson_deviance_residuals(data.source_counts, gross)
        if data.background_scale > 0:
            result = np.concatenate((result, poisson_deviance_residuals(data.background_counts, background)))
        return result

    if not np.any(free):
        return p0, np.full((p0.size, p0.size), np.nan)
    result = least_squares(
        residual, p0[free], bounds=(lower[free], upper[free]), max_nfev=max_nfev,
        x_scale='jac', ftol=1e-10, xtol=1e-10, gtol=1e-10,
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f'Poisson Gaussian fit did not converge: {result.message}')
    parameters = expand(result.x)
    fitted_model = gaussian_count_model(energy, baseline, data, parameters)
    return parameters, _information_covariance(
        energy, data, parameters, fitted_model, free_parameters=free)
