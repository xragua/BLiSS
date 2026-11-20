"""Record numerical fit quality for candidate evaluation."""
import numpy as np
import pandas as pd

FIT_QUALITY_COLUMNS = ['fit_converged', 'fit_evaluable',
                       'fit_reasons', 'fit_message', 'fit_block_id',
                       'fit_center_initial']
# Relative machine-precision guard, not an astrophysical significance cut.
RELATIVE_ERROR_FLOOR = np.sqrt(np.finfo(float).eps)
MAX_CENTER_ERROR_KEV = 1.0


def center_error_exceeds_limit(errors):
    """Identify errors strictly above 1 keV; NaNs remain numerical diagnostics."""
    numeric = pd.to_numeric(errors, errors='coerce')
    values = (numeric.to_numpy(dtype=float, na_value=np.nan)
              if isinstance(numeric, pd.Series) else np.asarray(numeric, dtype=float))
    return values > MAX_CENTER_ERROR_KEV


def discard_large_center_errors(lines):
    """Remove ecenter > 1 keV, preserving row labels and other diagnostics."""
    if 'ecenter' not in lines:
        return lines.copy()
    return lines.loc[~center_error_exceeds_limit(lines['ecenter'])].copy()


def annotate_fit_quality(lines):
    """Retain all rows and record whether their numerical fit is evaluable.

    No intensity or area cut is used. Errors <= sqrt(machine epsilon) times
    the parameter scale are numerically suspect. Centroid errors use sigma
    as their scale so the rule does not depend on the energy origin.
    Centroid errors > 1 keV are ineligible, including when cached quality
    flags are stale. Amplitude and width errors have no upper cut. Missing
    convergence metadata is unknown, not a recorded success.
    Width errors are inapplicable only when width_fixed_instrumental or
    width_fixed_three_bin is explicitly boolean True. Other parameter
    checks still apply, including when both width flags are true.
    """
    result = lines.copy()
    for col in ['fit_converged', 'fit_block_id', 'fit_center_initial']:
        if col not in result:
            result[col] = np.nan
    if 'fit_message' not in result:
        result['fit_message'] = ''
    for column in ('width_fixed_instrumental', 'width_fixed_three_bin'):
        if column not in result:
            result[column] = False
        else:
            result[column] = result[column].map(
                lambda flag: isinstance(flag, (bool, np.bool_)) and bool(flag)).astype(bool)
    width_fixed = (result['width_fixed_instrumental'].to_numpy(dtype=bool)
                   | result['width_fixed_three_bin'].to_numpy(dtype=bool))
    reasons = [[] for _ in range(len(result))]
    invalid = np.zeros(len(result), dtype=bool)

    def mark(mask, reason):
        mask = np.asarray(mask, dtype=bool)
        invalid[mask] = True
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
    mark(center_error_exceeds_limit(values('ecenter')), 'ecenter_exceeds_1_keV')
    for col, scale in [('eamplitude', np.abs(amp)), ('ecenter', np.abs(sigma)),
                       ('esigma', np.abs(sigma))]:
        error = values(col)
        required = ~width_fixed if col == 'esigma' else np.ones(len(result), dtype=bool)
        mark(required & (~np.isfinite(error) | (error <= 0)), 'invalid_' + col)
        mark(required & np.isfinite(error) & (error > 0) & np.isfinite(scale)
             & (error <= RELATIVE_ERROR_FLOOR * scale),
             'numerically_suspect_' + col)
    result['fit_evaluable'] = ~invalid
    result['fit_reasons'] = [';'.join(r) for r in reasons]
    return result
