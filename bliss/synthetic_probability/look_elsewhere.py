"""Compare local line statistics with maxima from the existing null searches.

No spectra are generated or fitted here. Each p-value covers one statistic
and one specified energy interval, conditional on the simulated noise model.
It does not correct for choosing among the five statistics afterwards.
"""
from numbers import Integral

import numpy as np
import pandas as pd

from ..line_search.fit_quality import annotate_fit_quality


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


def calculate_look_elsewhere_pvalues(lines, synthetic_candidates, n_sim, *, en1, en2):
    """Return the local catalogue with five p-values and the N-by-5 null maxima.

    Both inputs contain metrics from local fits. Eligibility requires a
    recorded converged, evaluable fit and a centroid inside [en1, en2].
    Each statistic must be finite; no S/N, area, or BLiSS-score threshold
    is applied. A missing statistic only
    prevents evaluation of its own p-value.

    Fit quality is recomputed for the masks alone: extra GMM feature checks
    must not select this comparison. Original rows, flags and scores are kept.
    """
    if not np.isfinite([en1, en2]).all() or en2 <= en1:
        raise ValueError('Require finite energy limits with en2 > en1.')
    if synthetic_candidates is None:
        raise ValueError('Missing synthetic catalogue is not an empty null search.')

    # 1. Apply the same fit and band rules to observed and synthetic candidates.
    def eligible_statistics(table):
        quality = annotate_fit_quality(table)
        eligible = (quality['fit_converged'].eq(True) & quality['fit_evaluable']
                    & pd.to_numeric(table['center'], errors='coerce').between(en1, en2))
        statistics = table.reindex(columns=LOOK_ELSEWHERE_STATISTICS).apply(
            pd.to_numeric, errors='coerce').astype(float)
        return statistics.where(np.isfinite(statistics)).where(eligible, axis=0)

    observed = eligible_statistics(lines)
    synthetic = eligible_statistics(synthetic_candidates)
    if 'sim' in synthetic_candidates:
        synthetic['sim'] = synthetic_candidates['sim']

    # 2. Reduce all five statistics in a single grouping, retaining all N searches.
    maxima = simulation_maxima(synthetic, n_sim, LOOK_ELSEWHERE_STATISTICS)

    # 3. Compare each observed statistic with its corresponding null maxima.
    result = lines.copy()
    for column in LOOK_ELSEWHERE_STATISTICS:
        _, result[f'p_global_{column}'] = monte_carlo_pvalues(
            observed[column].to_numpy(), maxima[column].to_numpy())
    return result, maxima
