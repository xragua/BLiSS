"""Compare local line statistics with maxima from the existing null searches.

No spectra are generated or fitted here. Each p-value covers one statistic
and one specified energy interval, conditional on the simulated noise model.
It does not correct for choosing among the five statistics afterwards.
"""
from numbers import Integral

import numpy as np
import pandas as pd

from ..line_search.fit_quality import (
    annotate_fit_quality, center_error_exceeds_limit, discard_large_center_errors,
)


LOOK_ELSEWHERE_STATISTICS = ('area', 'snr_area', 'snr_peak', 'ew', 'relative_power')
P_VALUE_COLUMNS = [f'p_global_{name}' for name in LOOK_ELSEWHERE_STATISTICS]


def simulation_maxima(candidates, n_sim, columns):
    """Return one row per generated simulation, including empty searches.

    Input statistics must already be masked for fit quality and energy range.
    NaN marks an unavailable statistic or a simulation with no eligible line;
    such a simulation cannot exceed an observed value but stays in N.
    """
    if isinstance(n_sim, (bool, np.bool_)) or not isinstance(n_sim, Integral) or n_sim < 0:
        raise ValueError('n_sim must be a nonnegative integer.')
    if candidates is None:
        raise ValueError('Missing synthetic catalogue is not an empty null search.')
    index = pd.RangeIndex(n_sim, name='sim')
    if candidates.empty:
        return pd.DataFrame(np.nan, index=index, columns=list(columns))
    if 'sim' not in candidates:
        raise ValueError("synthetic_candidates must have a 'sim' column.")
    sim = pd.to_numeric(candidates['sim'], errors='coerce').to_numpy(dtype=float)
    if (not np.isfinite(sim).all() or np.any(sim != np.floor(sim))
            or np.any(sim < 0) or np.any(sim >= n_sim)):
        raise ValueError('sim identifiers must be integers in [0, n_sim).')
    statistics = candidates[list(columns)].apply(pd.to_numeric, errors='coerce')
    statistics = statistics.where(np.isfinite(statistics))
    if 'ecenter' in candidates:
        statistics.loc[center_error_exceeds_limit(candidates['ecenter']), :] = np.nan
    statistics['sim'] = sim.astype(int)
    return statistics.groupby('sim')[list(columns)].max().reindex(index)


def monte_carlo_pvalues(values, maxima):
    """Count maxima >= each observed value and return k and (k+1)/(N+1).

    Sorting once and using binary search avoids a candidate-by-simulation
    comparison matrix. ``side='left'`` includes ties in the exceedance count.
    Invalid observed values and N=0 give NaN, not a detection significance.
    """
    values = np.asarray(values, dtype=float)
    maxima = np.asarray(maxima, dtype=float)
    counts = np.full(values.shape, np.nan)
    pvalues = np.full(values.shape, np.nan)
    n_sim = len(maxima)
    if n_sim == 0:
        return counts, pvalues
    ordered = np.sort(maxima[np.isfinite(maxima)])
    valid = np.isfinite(values)
    counts[valid] = len(ordered) - np.searchsorted(ordered, values[valid], side='left')
    pvalues[valid] = (counts[valid] + 1) / (n_sim + 1)
    return counts, pvalues


def _eligible_statistics(table, *, en1, en2):
    """Mask local statistics using the same numerical and band rules everywhere."""
    if not np.isfinite([en1, en2]).all() or en2 <= en1:
        raise ValueError('Require finite energy limits with en2 > en1.')
    statistics = table.reindex(columns=LOOK_ELSEWHERE_STATISTICS).apply(
        pd.to_numeric, errors='coerce').astype(float)
    if table.empty:
        return statistics
    quality = annotate_fit_quality(table)
    eligible = (quality['fit_converged'].eq(True) & quality['fit_evaluable']
                & pd.to_numeric(table['center'], errors='coerce').between(en1, en2))
    return statistics.where(np.isfinite(statistics)).where(eligible, axis=0)


def single_simulation_maxima(candidates, *, en1, en2):
    """Reduce one completed null search to five eligible maxima.

    Metrics must already have been calculated from the local fits. A missing
    statistic, empty search or entirely non-evaluable search gives NaN for the
    corresponding maximum. The caller must retain this row in the total N.
    Finite negative statistics remain comparable and are not replaced by zero.
    """
    if candidates is None:
        raise ValueError('Missing synthetic catalogue is not an empty null search.')
    statistics = _eligible_statistics(candidates, en1=en1, en2=en2)
    return statistics.max(axis=0, skipna=True).astype(float)


def calculate_look_elsewhere_pvalues_from_maxima(lines, maxima, *, en1, en2):
    """Return local candidates with p-values from one maximum row per null.

    Maxima must already use the same eligibility rules and energy interval as
    the observed candidates. All rows count toward N, including all-NaN rows;
    omitting an empty or non-evaluable simulation would change the p-values.
    Values must be finite or NaN. Neither input table is modified.
    """
    lines = discard_large_center_errors(lines)
    observed = _eligible_statistics(lines, en1=en1, en2=en2)
    if not isinstance(maxima, pd.DataFrame):
        raise ValueError('Null maxima must be a DataFrame with one row per simulation.')
    missing = sorted(set(LOOK_ELSEWHERE_STATISTICS) - set(maxima.columns))
    if missing:
        raise ValueError(f'Null maxima are missing statistic columns: {missing}')
    if not maxima.columns.is_unique:
        raise ValueError('Null maxima must have unique statistic columns.')
    try:
        numeric_maxima = maxima[list(LOOK_ELSEWHERE_STATISTICS)].apply(
            pd.to_numeric, errors='raise').astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError('Null maxima must contain only finite numeric values or NaN.') from exc
    if np.isinf(numeric_maxima.to_numpy()).any():
        raise ValueError('Null maxima must contain only finite numeric values or NaN.')

    result = lines.copy()
    for column in LOOK_ELSEWHERE_STATISTICS:
        _, result[f'p_global_{column}'] = monte_carlo_pvalues(
            observed[column].to_numpy(), numeric_maxima[column].to_numpy())
    return result


def calculate_look_elsewhere_pvalues(lines, synthetic_candidates, n_sim, *, en1, en2):
    """Return the local catalogue with five p-values and the N-by-5 null maxima.

    Both inputs contain metrics from local fits. Eligibility requires a
    recorded converged, evaluable fit and a centroid inside [en1, en2].
    Each statistic must be finite; no S/N or area threshold is applied.
    A missing statistic only prevents evaluation of its own p-value.

    Fit quality is recomputed for the eligibility masks.
    Rows with ecenter > 1 keV are discarded; other measurements and flags are kept.
    This full-catalogue entry point uses the same eligibility and comparison
    as incremental reduction.
    """
    if not np.isfinite([en1, en2]).all() or en2 <= en1:
        raise ValueError('Require finite energy limits with en2 > en1.')
    if synthetic_candidates is None:
        raise ValueError('Missing synthetic catalogue is not an empty null search.')

    synthetic = _eligible_statistics(synthetic_candidates, en1=en1, en2=en2)
    if 'sim' in synthetic_candidates:
        synthetic['sim'] = synthetic_candidates['sim']
    maxima = simulation_maxima(synthetic, n_sim, LOOK_ELSEWHERE_STATISTICS)
    result = calculate_look_elsewhere_pvalues_from_maxima(
        lines, maxima, en1=en1, en2=en2,
    )
    return result, maxima
