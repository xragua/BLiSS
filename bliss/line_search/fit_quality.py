"""Keep numerical fit quality separate from candidate detection scores."""
import numpy as np
import pandas as pd

FIT_QUALITY_COLUMNS = ['fit_converged', 'fit_evaluable',
                       'fit_reasons', 'fit_message', 'fit_block_id',
                       'fit_center_initial']
# Relative machine-precision guard, not an astrophysical significance cut.
RELATIVE_ERROR_FLOOR = np.sqrt(np.finfo(float).eps)


def annotate_fit_quality(lines):
    """Retain all rows and record whether their numerical fit is evaluable.

    No intensity or area cut is used. Errors <= sqrt(machine epsilon) times
    the parameter scale are numerically suspect. Centroid errors use sigma
    as their scale so the rule does not depend on the energy origin.
    Missing convergence metadata is unknown, not a recorded success.
    """
    result = lines.copy()
    for col in ['fit_converged', 'fit_block_id', 'fit_center_initial']:
        if col not in result:
            result[col] = np.nan
    if 'fit_message' not in result:
        result['fit_message'] = ''
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
    for col, scale in [('eamplitude', np.abs(amp)), ('ecenter', np.abs(sigma)),
                       ('esigma', np.abs(sigma))]:
        error = values(col)
        mark(~np.isfinite(error) | (error <= 0), 'invalid_' + col)
        mark(np.isfinite(error) & (error > 0) & np.isfinite(scale)
             & (error <= RELATIVE_ERROR_FLOOR * scale),
             'numerically_suspect_' + col)
    result['fit_evaluable'] = ~invalid
    result['fit_reasons'] = [';'.join(r) for r in reasons]
    return result
