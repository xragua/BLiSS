"""Refit sub-instrumental Gaussian widths as unresolved line components."""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from .gaussian_models import n_gaussian, _instrumental_sigma_upper_bound
from .poisson_fitting import fit_poisson_gaussians


def fit_with_instrumental_width(
    energy, target, uncertainties, p0, bounds, *, response_sigma=None,
    poisson_data=None, baseline=None, maxfev=100000, fixed_sigmas=None,
    gaussian_fit=curve_fit, poisson_fit=fit_poisson_gaussians,
):
    """Fit free parameters, then fix widths narrower than instrumental sigma.

    ``fixed_sigmas`` optionally supplies one width per component: NaN leaves
    that width free, and a finite positive value fixes it before the first
    fit. Supplied widths are never reopened or changed after a center moves.
    They must lie within the effective width bounds; no clipping is applied.

    A fixed width uses the instrumental sigma interpolated at the center of
    the fit that first triggered the rule. That numerical value stays fixed
    while amplitudes, centers, and other widths are jointly refitted. If a
    previously free width becomes sub-instrumental after refitting, the rule
    is applied to that component too; already fixed widths are not reopened.

    Returns full Gaussian parameters, full covariance, and one boolean per
    component indicating a width automatically fixed to the instrumental
    resolution during this call (not a width supplied in ``fixed_sigmas``).
    Fixed-width covariance
    rows/columns are NaN: uncertainties on remaining parameters are conditional
    on the adopted width. Without supplied widths or a resolution estimate,
    the ordinary free fit is unchanged. An upper width bound below the valid
    instrumental sigma is raised to three times that sigma before fitting.
    If a fitted center
    requires a larger cap, the free fit is repeated with that relaxed cap
    before deciding which widths to fix. Lower bounds are unchanged.
    """
    energy = np.asarray(energy, dtype=float)
    parameters = np.asarray(p0, dtype=float).copy()
    if parameters.ndim != 1 or parameters.size == 0 or parameters.size % 3:
        raise ValueError('Gaussian fits require nonempty amplitude/center/sigma triples.')
    lower = np.broadcast_to(np.asarray(bounds[0], dtype=float), parameters.shape)
    upper = np.broadcast_to(np.asarray(bounds[1], dtype=float), parameters.shape).copy()
    response = None if response_sigma is None else np.asarray(response_sigma, dtype=float)
    if response is not None and response.shape != energy.shape:
        raise ValueError('Instrumental sigmas must match the fitting energy grid.')
    width_fixed = np.zeros(parameters.size // 3, dtype=bool)
    fixed = np.full(parameters.shape, np.nan)
    if response is not None:
        upper[2::3] = _instrumental_sigma_upper_bound(
            upper[2::3], np.interp(parameters[1::3], energy, response))
    if fixed_sigmas is not None:
        supplied = np.asarray(fixed_sigmas, dtype=float)
        if (supplied.shape != width_fixed.shape or np.any(np.isinf(supplied))
                or np.any(np.isfinite(supplied) & (supplied <= 0))):
            raise ValueError(
                'fixed_sigmas must contain one positive finite width or NaN '
                'per Gaussian component.')
        supplied_indices = 3 * np.flatnonzero(np.isfinite(supplied)) + 2
        values = supplied[np.isfinite(supplied)]
        if np.any((values < lower[supplied_indices]) | (values > upper[supplied_indices])):
            raise ValueError(
                'Supplied fixed sigma lies outside the configured width bounds; '
                'cannot apply the requested fixed-width model.')
        fixed[supplied_indices] = values
        parameters[supplied_indices] = values

    def fit(current):
        free = np.isnan(fixed)
        if poisson_data is not None:
            options = {'max_nfev': maxfev}
            if not np.all(free):
                options['fixed_parameters'] = fixed
            return poisson_fit(energy, baseline, poisson_data, current,
                               (lower, upper), **options)
        if np.all(free):
            return gaussian_fit(
                n_gaussian, energy, target, p0=current, bounds=(lower, upper),
                sigma=uncertainties, absolute_sigma=True, maxfev=maxfev)

        template = current.copy()
        template[~free] = fixed[~free]

        def reduced_model(x, *values):
            full = template.copy()
            full[free] = values
            return n_gaussian(x, *full)

        optimal, covariance = gaussian_fit(
            reduced_model, energy, target, p0=template[free],
            bounds=(lower[free], upper[free]), sigma=uncertainties,
            absolute_sigma=True, maxfev=maxfev)
        template[free] = optimal
        full_covariance = np.full((template.size, template.size), np.nan)
        full_covariance[np.ix_(free, free)] = covariance
        return template, full_covariance

    while True:
        parameters, covariance = fit(parameters)
        parameters = np.asarray(parameters, dtype=float)
        if response is None:
            return parameters, covariance, width_fixed
        instrumental = np.interp(parameters[1::3], energy, response)
        relaxed = _instrumental_sigma_upper_bound(upper[2::3], instrumental)
        free_widths = np.isnan(fixed[2::3])
        expand = free_widths & (relaxed > upper[2::3])
        if np.any(expand):
            upper[3 * np.flatnonzero(expand) + 2] = relaxed[expand]
            continue
        newly_fixed = (free_widths & np.isfinite(instrumental) & (instrumental > 0)
                       & (parameters[2::3] < instrumental))
        if not np.any(newly_fixed):
            return parameters, covariance, width_fixed
        width_indices = 3 * np.flatnonzero(newly_fixed) + 2
        values = instrumental[newly_fixed]
        if np.any((values < lower[width_indices]) | (values > upper[width_indices])):
            raise ValueError(
                'Instrumental sigma lies outside the configured width bounds; '
                'cannot fix this component to the instrumental width.')
        fixed[width_indices] = values
        parameters[width_indices] = values
        width_fixed[newly_fixed] = True
