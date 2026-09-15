"""Detect contiguous excess regions and fit local Gaussian candidate lines."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from .fit_quality import annotate_fit_quality
from .peak_selection import find_peaks_new
from .gaussian_models import (gaussian, n_gaussian, gaussian_area_error,
                              p0_generator, fit_gaussian_components,
                              WIDTH_FIT_COLUMNS, LOCAL_MODEL_COLUMNS)

DEFAULT_MIN_PEAK_SEPARATION = 0.001
"""Floor of the minimum separation between retained peaks, in spectral-axis
units (keV by default). When the instrumental ``response_sigma`` is
available, the effective local separation is
``max(DEFAULT_MIN_PEAK_SEPARATION, sigma_inst(E))``."""

DEFAULT_SINGLE_COMPONENT_BIC_MARGIN = 6.0


def _validate_bic_margin(margin):
    if margin is not None and (not np.isfinite(margin) or margin < 0):
        raise ValueError('single_component_bic_margin must be finite and nonnegative, or None.')


def _single_component_alternative(block, target, fit_sigma, popt, width_fixed,
                                  bic_margin):
    """Try one resolved Gaussian only when at least two widths were fixed.

    Compare chi-square + k*log(n) on the same signed residuals, absolute
    errors and fixed baseline. k counts only free parameters in each model.
    This adds at most one optimizer call, with a frozen instrumental width
    floor at the moment-based initial center. A failed, non-evaluable or
    width-bound alternative leaves the existing multiple fit untouched.
    This is a conservative model-selection heuristic, not a line probability.
    """
    count = len(popt) // 3
    info = dict(local_model_selection='not_tested',
                local_n_components_initial=count, local_n_components_selected=count,
                local_bic_single=np.nan, local_bic_multiple=np.nan,
                local_bic_delta=np.nan, local_model_selection_message='')
    if bic_margin is None:
        info['local_model_selection'] = 'disabled'
        return None, info
    if count < 2 or np.count_nonzero(width_fixed) < 2:
        return None, info

    x = np.asarray(block.energy, dtype=float)
    weights = np.maximum(target, 0.)
    if block.response_sigma is None or not np.isfinite(weights.sum()) or weights.sum() <= 0:
        return None, info
    center = float(np.sum(weights * x) / weights.sum())
    floor = float(np.interp(center, x, block.response_sigma))
    if not np.isfinite(floor) or floor <= 0:
        info['local_model_selection_message'] = 'No valid resolution at the single-component initial center.'
        return None, info

    spread = np.sqrt(np.sum(weights * (x-center)**2) / weights.sum())
    upper_width = max(float(np.ptp(x)), 2*floor)
    initial = np.array([float(np.max(target)), center,
                        np.clip(spread, 1.05*floor, .95*upper_width)])
    bounds = ([0., float(x[0]), floor], [np.inf, float(x[-1]), upper_width])
    n = len(x)
    chi2_multiple = float(np.sum(((target-n_gaussian(x, *popt))/fit_sigma)**2))
    info['local_bic_multiple'] = chi2_multiple + (3*count-np.count_nonzero(width_fixed))*np.log(n)
    try:
        single, covariance = curve_fit(
            gaussian, x, target, p0=initial, bounds=bounds,
            sigma=fit_sigma, absolute_sigma=True, maxfev=10000)
    except (RuntimeError, ValueError) as exc:
        info.update(local_model_selection='single_fit_failed',
                    local_model_selection_message=str(exc))
        return None, info

    chi2_single = float(np.sum(((target-gaussian(x, *single))/fit_sigma)**2))
    info['local_bic_single'] = chi2_single + 3*np.log(n)
    info['local_bic_delta'] = info['local_bic_multiple'] - info['local_bic_single']
    with np.errstate(invalid='ignore'):
        errors = np.sqrt(np.diag(covariance))
    quality = annotate_fit_quality(pd.DataFrame([dict(
        amplitude=single[0], center=single[1], sigma=single[2],
        eamplitude=errors[0], ecenter=errors[1], esigma=errors[2],
        cov_amplitude_sigma=covariance[0, 2], fit_converged=True)]))
    area_error = gaussian_area_error(single[0], single[2], errors[0], errors[2],
                                     covariance[0, 2])
    if (not quality.fit_evaluable.iloc[0] or not np.isfinite(info['local_bic_delta'])
            or not np.isfinite(area_error) or area_error <= 0):
        info.update(local_model_selection='single_not_evaluable',
                    local_model_selection_message=str(quality.fit_reasons.iloc[0]))
        return None, info
    # A free width at a bound is not evidence for a resolved broad profile.
    fitted_resolution = float(np.interp(single[1], x, block.response_sigma))
    if (not np.isfinite(fitted_resolution) or fitted_resolution <= 0
            or single[2] <= 1.01*max(floor, fitted_resolution)
            or single[2] >= .99*upper_width):
        info['local_model_selection'] = 'single_at_width_bound'
        return None, info
    if info['local_bic_delta'] < bic_margin:
        info['local_model_selection'] = 'multiple_gaussians'
        return None, info

    info.update(local_model_selection='single_gaussian', local_n_components_selected=1)
    return (single, covariance, np.array([False]), np.array([np.nan]), initial, bounds), info


@dataclass
class CandidateBlock:
    """Contiguous positive-excess region prepared for local Gaussian fitting.

    Attributes
    ----------
    excess : numpy.ndarray
        Residual values above the empirical baseline within the candidate block.
    energy : numpy.ndarray
        Coordinate values corresponding to the block.
    values : numpy.ndarray
        Original observed spectral values in the block.
    uncertainties : numpy.ndarray
        One-sigma uncertainties associated with ``values``.
    baseline : numpy.ndarray
        Empirical baseline values over the same block.
    response_sigma : numpy.ndarray or None
        Instrumental Gaussian-equivalent sigma over the block, when available.
    """
    excess: np.ndarray
    energy: np.ndarray
    values: np.ndarray
    uncertainties: np.ndarray
    baseline: np.ndarray
    response_sigma: np.ndarray | None = None


class CandidateRegionDetector:
    """Small wrapper object for detecting and fitting raw line candidates."""

    def detect(self, x, y, sy, ylines, base, response_sigma=None,
               min_peak_separation=None,
               single_component_bic_margin=DEFAULT_SINGLE_COMPONENT_BIC_MARGIN):
        """Detect raw candidate lines from baseline-subtracted spectral excesses.

        Parameters
        ----------
        x : array-like
            Spectral coordinate grid.
        y : array-like
            Observed spectral values.
        sy : array-like
            One-sigma uncertainties on ``y``.
        ylines : array-like
            Baseline-subtracted line excess array. Non-zero contiguous regions
            are treated as candidate blocks.
        base : array-like
            Empirical baseline evaluated on ``x``.
        response_sigma : array-like or None, default: None
            Instrumental Gaussian-equivalent sigma aligned with ``x``.
        min_peak_separation : float or None, default: None
            Floor of the minimum separation between retained peaks, in
            spectral-axis units. ``None`` uses
            ``DEFAULT_MIN_PEAK_SEPARATION``.
        single_component_bic_margin : float or None, default: 6
            Required BIC improvement for a single resolved Gaussian to replace
            a block with at least two fixed widths. None disables this check.

        Returns
        -------
        pandas.DataFrame
            Preliminary Gaussian candidate table with fitted parameters and
            local context columns.
        """
        return return_raw_lines(
            x, y, sy, ylines, base,
            response_sigma=response_sigma,
            min_peak_separation=min_peak_separation,
            single_component_bic_margin=single_component_bic_margin,
        )


def _build_candidate_blocks(x, y, sy, ylines, base, response_sigma=None):
    """Split non-zero line excesses into contiguous candidate blocks.

    Blocks are maximal runs of non-zero bins in ``ylines``, extended by one
    zero-valued padding bin on each side (where available) to anchor the
    Gaussian wings. The final run of the interval is included, and runs
    touching either edge of the grid are padded only where possible.

    Parameters
    ----------
    x, y, sy : array-like
        Coordinate grid, observed values, and one-sigma uncertainties.
    ylines : array-like
        Baseline-subtracted excess array. Non-zero runs define candidate
        regions.
    base : array-like
        Baseline values corresponding to ``x``.
    response_sigma : array-like or None, default: None
        Instrumental Gaussian-equivalent sigma aligned with ``x``.

    Returns
    -------
    list of CandidateBlock
        Candidate blocks containing local arrays for fitting.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    sy = np.asarray(sy)
    ylines = np.asarray(ylines)
    base = np.asarray(base)
    resp = None if response_sigma is None else np.asarray(response_sigma,
                                                          dtype=float)

    nonzero = np.nonzero(ylines)[0]
    if len(nonzero) == 0:
        return []

    # Maximal contiguous runs of non-zero indices.
    breaks = np.where(np.diff(nonzero) > 1)[0]
    run_starts = nonzero[np.concatenate(([0], breaks + 1))]
    run_ends = nonzero[np.concatenate((breaks, [len(nonzero) - 1]))]

    blocks = []
    last = len(ylines) - 1
    for lo_nz, hi_nz in zip(run_starts, run_ends):
        lo = max(int(lo_nz) - 1, 0)      # one zero-padding bin on the left
        hi = min(int(hi_nz) + 1, last)   # one zero-padding bin on the right
        sl = slice(lo, hi + 1)
        blocks.append(
            CandidateBlock(
                excess=np.array(ylines[sl]),
                energy=np.array(x[sl]),
                values=np.array(y[sl]),
                uncertainties=np.array(sy[sl]),
                baseline=np.array(base[sl]),
                response_sigma=None if resp is None else np.interp(
                    np.asarray(x[sl], dtype=float),
                    np.asarray(x, dtype=float),
                    resp,
                ),
            )
        )
    return blocks


def _fit_candidate_block(block, block_index, min_peak_separation=None,
                         single_component_bic_margin=DEFAULT_SINGLE_COMPONENT_BIC_MARGIN):
    """Fit Gaussian components inside one candidate block.

    Peak retention follows two rules: (i) minimum separation -- among peaks
    closer than the local separation scale, only the most prominent is kept;
    (ii) relative prominence -- peaks below 10 per cent of the largest
    prominence in the block are discarded. The local separation scale is
    ``max(min_peak_separation, sigma_inst(E))`` when the instrumental
    ``response_sigma`` is available, and ``min_peak_separation`` otherwise.
    The number of Gaussians is capped at ``floor(n_bins / 4)``, keeping the
    most prominent peaks. Local fits are performed on the signed
    baseline-subtracted residual of the block, weighted by the supplied
    one-sigma observational uncertainties. Their absolute scale is retained
    in the covariance; the empirical baseline is treated as fixed. A free
    width below the instrumental sigma is fixed to that value in one joint
    refit. Such widths have sigma_fixed=True and esigma=NaN; the other errors
    are conditional on the imposed width. No response means no fixation.
    Blocks with at least two fixed widths also try one resolved Gaussian.
    It replaces the multiple model only if BIC improves by the configured
    margin; the local decision and both BIC values are exported.

    Parameters
    ----------
    block : CandidateBlock
        Local region containing spectral values, uncertainties, and baseline.
    block_index : int
        Index of the block in the candidate-block list, used only for
        diagnostic error messages.
    min_peak_separation : float or None, default: None
        Floor of the minimum separation between retained peaks, in
        spectral-axis units. ``None`` uses ``DEFAULT_MIN_PEAK_SEPARATION``.

    Returns
    -------
    list of dict
        One dictionary per fitted local Gaussian, containing amplitude,
        center, sigma, formal errors, block-level R-squared, and mean block
        uncertainty. ``cov_amplitude_sigma`` retains the amplitude/width
        covariance from this joint fit. ``sigma_lower_bound`` records the width floor;
        ``sigma_at_lower_bound`` marks widths within 1% of a positive floor.
    """
    _validate_bic_margin(single_component_bic_margin)
    rows = []
    noise_on_block = np.mean(block.uncertainties)
    sep_floor = (DEFAULT_MIN_PEAK_SEPARATION if min_peak_separation is None
                 else float(min_peak_separation))
    if len(block.values) > 3 and max(block.values) > 0:
        peaks = find_peaks_new(block.energy, block.values)
        if len(peaks) > 0:
            good_list = []
            for i in range(len(peaks)):
                if block.response_sigma is not None:
                    local_sep = max(
                        sep_floor,
                        float(np.interp(float(peaks.energy[i]),
                                        block.energy,
                                        block.response_sigma)),
                    )
                else:
                    local_sep = sep_floor
                prominence_ratio = peaks.prominences[i] / max(peaks.prominences)
                idx1 = abs(peaks.energy[i] - peaks.energy) < local_sep
                if all(peaks.prominences[i] - peaks.prominences[idx1] >= 0) & (prominence_ratio > 0.1):
                    good_list.append(i)
            good_peaks = (
                peaks.loc[good_list]
                .sort_values(by='prominences', ascending=False)
                .reset_index(drop=True)
            )
            max_peaks = int(np.floor(len(block.energy) / 4))
            good_peaks = good_peaks[0:max(1, max_peaks)]
            if len(good_peaks) > 0:
                p0, bounds = p0_generator(block.energy, block.values,
                                          good_peaks,
                                          response_sigma=block.response_sigma)
                try:
                    target = block.values - block.baseline
                    fit_sigma = np.asarray(block.uncertainties, dtype=float)
                    if (fit_sigma.shape != target.shape
                            or not np.all(np.isfinite(fit_sigma) & (fit_sigma > 0))):
                        raise ValueError("Local fit requires finite positive uncertainties "
                                         "for every fitted bin.")
                    popt, pcov, width_fixed, reference_centers = fit_gaussian_components(
                        block.energy, target, p0, bounds, fit_sigma,
                        response_sigma=block.response_sigma, optimizer=curve_fit)
                    alternative, model_info = _single_component_alternative(
                        block, target, fit_sigma, popt, width_fixed,
                        single_component_bic_margin)
                    if alternative is not None:
                        popt, pcov, width_fixed, reference_centers, p0, bounds = alternative
                    errors = np.sqrt(np.diag(pcov))
                    errors[2::3][width_fixed] = np.nan
                    yfit = n_gaussian(block.energy, *popt)
                    ss_tot = np.sum((target - np.mean(target)) ** 2)
                    if ss_tot > 0:
                        rsq = 1 - np.sum((target - yfit) ** 2) / ss_tot
                    else:
                        rsq = np.nan
                    popt_ = np.reshape(popt, (-1, 3))
                    errors_ = np.reshape(errors, (-1, 3))
                    for k in range(len(popt_)):
                        rows.append({'fit_converged': True,
                                     **model_info,
                                     'fit_message': '',
                                     'fit_block_id': block_index,
                                     'fit_center_initial': p0[3*k + 1],
                                     'amplitude': popt_[k][0],
                                     'center': popt_[k][1],
                                     'sigma': popt_[k][2],
                                     'sigma_fixed': bool(width_fixed[k]),
                                     'sigma_reference_center': reference_centers[k],
                                     'fit_uncertainty': ('conditional_on_fixed_widths'
                                                         if width_fixed.any() else 'free_widths'),
                                     'sigma_lower_bound': bounds[0][3*k + 2],
                                     'sigma_at_lower_bound': bool(
                                         not width_fixed[k] and bounds[0][3*k + 2] > 0 and
                                         popt_[k][2] <= 1.01 * bounds[0][3*k + 2]),
                                     'eamplitude': errors_[k][0],
                                     'ecenter': errors_[k][1],
                                     'esigma': errors_[k][2],
                                     'cov_amplitude_sigma': pcov[3*k, 3*k + 2],
                                     'rsq': rsq,
                                     'noise_on_block': noise_on_block})
                except (RuntimeError, ValueError) as exc:
                    print(f'Error fitting block {block_index}: {exc}')
                    # Preserve a diagnostic row per attempted component. The
                    # center locates the attempted fit, not a measured centroid.
                    for k in range(len(good_peaks)):
                        rows.append(dict(
                            center=p0[3*k + 1], fit_center_initial=p0[3*k + 1],
                            amplitude=np.nan, sigma=np.nan, eamplitude=np.nan,
                            ecenter=np.nan, esigma=np.nan,
                            cov_amplitude_sigma=np.nan, rsq=np.nan,
                            noise_on_block=noise_on_block,
                            sigma_lower_bound=bounds[0][3*k + 2],
                            sigma_at_lower_bound=False, sigma_fixed=False,
                            sigma_reference_center=np.nan, fit_uncertainty='unavailable',
                            fit_converged=False,
                            fit_message=str(exc), fit_block_id=block_index))
    return rows


def _add_line_context(fitted, x, y, base):
    """Attach nearest-bin continuum and signal values to fitted candidates.

    Parameters
    ----------
    fitted : pandas.DataFrame
        Preliminary fitted Gaussian table with a ``center`` column.
    x : array-like
        Full coordinate grid used to locate the nearest bin to each center.
    y : array-like
        Observed spectral values.
    base : array-like
        Empirical baseline values.

    Returns
    -------
    pandas.DataFrame
        Candidate table with ``base_on_line``, ``value_on_line``, and
        ``relative_power`` columns added.
    """
    if len(fitted) == 0:
        fitted['base_on_line'] = []
        fitted['value_on_line'] = []
        fitted['relative_power'] = []
        return fitted
    min_diff_positions = []
    for i in range(len(fitted)):
        center = fitted.center.iloc[i]
        min_diff_positions.append(int(np.argmin(np.abs(x - center)))
                                  if np.isfinite(center) else None)
    fitted['base_on_line'] = [base[pos] if pos is not None else np.nan
                              for pos in min_diff_positions]
    fitted['value_on_line'] = [y[pos] if pos is not None else np.nan
                               for pos in min_diff_positions]
    fitted['relative_power'] = (fitted.value_on_line - fitted.base_on_line) / (fitted.value_on_line + fitted.base_on_line)
    return fitted


def return_raw_lines(x, y, sy, ylines, base, response_sigma=None,
                     min_peak_separation=None,
                     single_component_bic_margin=DEFAULT_SINGLE_COMPONENT_BIC_MARGIN):
    """Detect contiguous excesses and fit preliminary Gaussian line candidates.

    Parameters
    ----------
    x : array-like
        Spectral coordinate grid.
    y : array-like
        Observed spectral values.
    sy : array-like
        One-sigma uncertainties on ``y``.
    ylines : array-like
        Baseline-subtracted line-excess array.
    base : array-like
        Empirical baseline evaluated over the full spectrum.
    response_sigma : array-like or None, default: None
        Instrumental Gaussian-equivalent sigma aligned with ``x``.
    min_peak_separation : float or None, default: None
        Floor of the minimum separation between retained peaks, in
        spectral-axis units. ``None`` uses ``DEFAULT_MIN_PEAK_SEPARATION``;
        when ``response_sigma`` is provided, the effective local separation
        is ``max(min_peak_separation, sigma_inst(E))``.
    single_component_bic_margin : float or None, default: 6
        Required improvement in BIC to replace a multiple-component block
        having at least two fixed widths by one resolved Gaussian. At most
        one extra fit per eligible block; None disables this comparison.

    Returns
    -------
    pandas.DataFrame
        Raw candidate-line table with Gaussian parameters, parameter errors,
        goodness-of-fit information, local continuum context, and instrumental
        ``response_sigma`` interpolated at each fitted centroid (NaN if absent).
        Quality metadata retains failed attempts: their center is the initial
        guess, their fitted amplitude/width/errors are NaN, and the failure
        message is stored. Non-evaluable attempts are not detection claims.
    """
    _validate_bic_margin(single_component_bic_margin)
    blocks = _build_candidate_blocks(x, y, sy, ylines, base,
                                     response_sigma=response_sigma)
    rows = []
    for block_index, block in enumerate(blocks):
        rows.extend(_fit_candidate_block(
            block, block_index, min_peak_separation=min_peak_separation,
            single_component_bic_margin=single_component_bic_margin))
    fitted = pd.DataFrame(rows, columns=['amplitude', 'center', 'sigma',
                                         'eamplitude', 'ecenter', 'esigma',
                                         'cov_amplitude_sigma',
                                         'rsq', 'noise_on_block',
                                         'sigma_lower_bound', 'sigma_at_lower_bound',
                                         'fit_converged', 'fit_message',
                                         'fit_block_id', 'fit_center_initial',
                                         *WIDTH_FIT_COLUMNS, *LOCAL_MODEL_COLUMNS])
    fitted = _add_line_context(fitted, x, y, base)
    fitted['response_sigma'] = np.nan
    if response_sigma is not None and len(fitted):
        fitted['response_sigma'] = np.interp(fitted['center'], x, response_sigma)
    return annotate_fit_quality(fitted).reset_index(drop=True)
