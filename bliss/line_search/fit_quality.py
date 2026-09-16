"""Keep numerical fit quality separate from candidate detection scores."""
import numpy as np
import pandas as pd
from .gaussian_models import WIDTH_FIT_COLUMNS

FIT_QUALITY_COLUMNS = ['fit_converged', 'fit_status', 'fit_evaluable',
                       'fit_reasons', 'fit_message', 'fit_block_id',
                       'fit_center_initial', *WIDTH_FIT_COLUMNS]
# Relative machine-precision guard, not an astrophysical significance cut.
RELATIVE_ERROR_FLOOR = np.sqrt(np.finfo(float).eps)


def annotate_fit_quality(lines):
    """Retain all rows; flag invalid errors and review large uncertainties.

    No intensity or area cut is used. Errors <= sqrt(machine epsilon) times
    the parameter scale are numerically suspect. Centroid errors use sigma
    as their scale so the rule does not depend on the energy origin.
    Explicitly fixed widths may have NaN (or conditional zero) esigma with
    zero amplitude/width covariance; their other errors are still checked.
    Large relative errors and the existing width-bound flag only request
    review. Missing convergence metadata is unknown, not a recorded success.
    """
    result = lines.copy()
    if 'sigma_fixed' not in result:
        result['sigma_fixed'] = False
    fixed = result['sigma_fixed'].eq(True).fillna(False).to_numpy(dtype=bool)
    result['sigma_fixed'] = fixed
    if 'sigma_reference_center' not in result:
        result['sigma_reference_center'] = np.nan
    if 'fit_uncertainty' not in result:
        result['fit_uncertainty'] = np.where(fixed, 'conditional_on_fixed_widths', 'free_widths')
    for col in ['fit_converged', 'fit_block_id', 'fit_center_initial']:
        if col not in result:
            result[col] = np.nan
    if 'fit_message' not in result:
        result['fit_message'] = ''
    reasons = [[] for _ in range(len(result))]
    invalid = np.zeros(len(result), dtype=bool)
    review = np.zeros(len(result), dtype=bool)

    def mark(mask, reason, fatal=True):
        mask = np.asarray(mask, dtype=bool)
        if fatal:
            invalid[mask] = True
        else:
            review[mask] = True
        for i in np.flatnonzero(mask):
            reasons[i].append(reason)

    def values(col):
        if col not in result:
            return np.full(len(result), np.nan)
        return pd.to_numeric(result[col], errors='coerce').to_numpy(dtype=float)

    mark(result['fit_converged'].eq(False).fillna(False), 'fit_failed')
    amp, center, sigma = (values(c) for c in ['amplitude', 'center', 'sigma'])
    mark(~np.isfinite(amp) | ~np.isfinite(center) | ~np.isfinite(sigma),
         'nonfinite_parameters')
    mark((amp <= 0) | (sigma <= 0), 'nonpositive_parameters')
    mark(fixed & ~(np.isnan(values('esigma')) | (values('esigma') == 0)),
         'invalid_fixed_sigma_error')
    mark(fixed & (values('cov_amplitude_sigma') != 0), 'invalid_fixed_sigma_covariance')
    for col, scale in [('eamplitude', np.abs(amp)), ('ecenter', np.abs(sigma)),
                       ('esigma', np.abs(sigma))]:
        error = values(col)
        active = ~fixed if col == 'esigma' else np.ones(len(result), dtype=bool)
        mark(active & (~np.isfinite(error) | (error <= 0)), 'invalid_' + col)
        mark(active & np.isfinite(error) & (error > 0) & np.isfinite(scale)
             & (error <= RELATIVE_ERROR_FLOOR * scale),
             'numerically_suspect_' + col)
        mark(active & np.isfinite(error) & np.isfinite(scale) & (error > scale),
             'large_' + col, fatal=False)
    if 'sigma_at_lower_bound' in result:
        mark(~fixed & result['sigma_at_lower_bound'].eq(True).fillna(False),
             'sigma_at_lower_bound', fatal=False)
    if 'fit_status' in result:
        # Recompute rather than retaining stale annotations after a refit.
        result = result.drop(columns=['fit_status'])
    result['fit_status'] = np.where(invalid, 'not_evaluable',
                                    np.where(review, 'review', 'ok'))
    result['fit_evaluable'] = ~invalid
    result['fit_reasons'] = [';'.join(r) for r in reasons]
    return result
