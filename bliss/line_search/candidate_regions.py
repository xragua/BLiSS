"""Detect contiguous excess regions and fit local Gaussian candidate lines."""
from __future__ import annotations
from dataclasses import dataclass
from numbers import Real
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from .fit_quality import annotate_fit_quality, center_error_exceeds_limit, discard_large_center_errors
from .poisson_fitting import PoissonFitData, fit_poisson_gaussians, poisson_model_uncertainty
from .peak_selection import find_peaks_new
from .gaussian_models import n_gaussian, p0_generator, validate_sigma_bounds
from .instrumental_width import fit_with_instrumental_width

DEFAULT_MIN_PEAK_SEPARATION = 0.001
"""Floor of the minimum separation between retained peaks, in spectral-axis
units (keV by default). When the instrumental ``response_sigma`` is
available, the effective local separation is
``max(DEFAULT_MIN_PEAK_SEPARATION, sigma_inst(E))``."""

DEFAULT_MIN_RELATIVE_PROMINENCE = 0.05
"""Strict lower bound on peak prominence divided by the block maximum."""


def validate_min_relative_prominence(value):
    """Require a finite relative candidate-selection threshold in [0, 1)."""
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or not 0 <= value < 1):
        raise ValueError("min_relative_prominence must be a finite number in [0, 1).")


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
    poisson_data: PoissonFitData | None = None


class CandidateRegionDetector:
    """Small wrapper object for detecting and fitting raw line candidates."""

    def detect(self, x, y, sy, ylines, base, response_sigma=None,
               min_peak_separation=None, max_sigma_line=0.1,
               min_relative_prominence=DEFAULT_MIN_RELATIVE_PROMINENCE, min_sigma_line=1e-6,
               *, poisson_data=None):
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
        min_sigma_line : float, default: 1e-6
            Absolute lower sigma bound; combined with the instrumental floor.
        max_sigma_line : float, default: 0.1
            Configured upper sigma bound in keV, capped by the fitted block energy span.
        min_relative_prominence : float, default: 0.05
            Retain peaks strictly above this fraction of the largest
            prominence in their block. Must be finite and in [0, 1).

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
            max_sigma_line=max_sigma_line, min_sigma_line=min_sigma_line,
            min_relative_prominence=min_relative_prominence,
            poisson_data=poisson_data,
        )


def _build_candidate_blocks(x, y, sy, ylines, base, response_sigma=None, *, poisson_data=None):
    """Split non-zero line excesses into contiguous candidate blocks.

    Blocks are maximal runs of non-zero bins in ``ylines``, extended by up to
    two observed bins on each side to anchor the Gaussian wings. Padding stops
    before another non-zero run or the edge of the grid. Neighboring blocks
    may share zero-excess padding bins, but never include each other's excess.
    The final run of the interval is included.

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
    for i, (lo_nz, hi_nz) in enumerate(zip(run_starts, run_ends)):
        # Keep the added context outside neighboring candidate excesses.
        left_limit = int(run_ends[i - 1]) + 1 if i > 0 else 0
        right_limit = int(run_starts[i + 1]) - 1 if i + 1 < len(run_starts) else last
        lo = max(int(lo_nz) - 2, left_limit)
        hi = min(int(hi_nz) + 2, right_limit)
        sl = slice(lo, hi + 1)
        blocks.append(
            CandidateBlock(
                excess=np.array(ylines[sl]),
                energy=np.array(x[sl]),
                values=np.array(y[sl]),
                uncertainties=np.array(sy[sl]),
                baseline=np.array(base[sl]),
                poisson_data=None if poisson_data is None else poisson_data.sliced(sl),
                response_sigma=None if resp is None else np.interp(
                    np.asarray(x[sl], dtype=float),
                    np.asarray(x, dtype=float),
                    resp,
                ),
            )
        )
    return blocks


def _fit_candidate_block(block, block_index, min_peak_separation=None, max_sigma_line=0.1,
                         min_relative_prominence=DEFAULT_MIN_RELATIVE_PROMINENCE, min_sigma_line=1e-6):
    """Fit Gaussian components inside one candidate block.

    Peak retention follows two rules: (i) minimum separation -- among peaks
    closer than the local separation scale, only the most prominent is kept;
    (ii) relative prominence -- peaks must exceed ``min_relative_prominence``
    times the largest prominence in the block (5 per cent by default).
    The local separation scale is
    ``max(min_peak_separation, sigma_inst(E))`` when the instrumental
    ``response_sigma`` is available, and ``min_peak_separation`` otherwise.
    The number of Gaussians is capped at ``floor(n_bins / 4)``, keeping the
    most prominent peaks. Local Gaussian-likelihood fits use the signed
    baseline-subtracted residual of the block, weighted by the supplied
    one-sigma observational uncertainties. Their absolute scale is retained
    in the covariance; the empirical baseline is treated as fixed. Amplitude,
    center and sigma initially remain free within their bounds. Widths below
    the instrumental sigma at the fitted center are then fixed to that value
    and the remaining parameters are jointly refitted. Fixed-width errors
    are NaN; amplitude/center errors are conditional on that assumed width.
    Every amplitude is bounded above by the maximum observed bin in this block
    plus the uncertainty of that same bin, including the continuum in the bin
    value. The same rule is used for observed and simulated blocks.
    With poisson_data, fit original source/background counts using Poisson
    likelihood and the bin-integrated continuum + line model. Raw zero
    errors are allowed and are never used as weights. Covariances are local
    Fisher estimates; noise_on_block uses the fitted expected count variance.
    Components with ecenter > 1 keV are discarded. There is no upper cut on
    esigma or eamplitude. Retained components keep their final joint-fit
    parameters and covariance.
    return_raw_lines records numerical validity; non-evaluable fits are
    excluded from significance comparisons, equally for data and simulations.

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
    min_sigma_line : float, default: 1e-6
        Absolute lower sigma bound; combined with the instrumental floor.
    max_sigma_line : float, default: 0.1
        Configured upper sigma bound in keV, capped by the fitted block energy span.
    min_relative_prominence : float, default: 0.05
        Strict lower bound on peak prominence divided by the block maximum.
        Must be finite and in [0, 1); this is not a significance threshold.

    Returns
    -------
    list of dict
        One dictionary per retained local Gaussian, containing amplitude,
        center, sigma, formal errors, and mean block
        uncertainty. ``cov_amplitude_sigma`` retains the amplitude/width
        covariance from this joint fit.
    """
    validate_min_relative_prominence(min_relative_prominence)
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
                if (all(peaks.prominences[i] - peaks.prominences[idx1] >= 0)
                        and prominence_ratio > min_relative_prominence):
                    good_list.append(i)
            good_peaks = (
                peaks.loc[good_list]
                .sort_values(by='prominences', ascending=False)
                .reset_index(drop=True)
            )
            max_peaks = int(np.floor(len(block.energy) / 4))
            good_peaks = good_peaks[0:max(1, max_peaks)]
            if len(good_peaks) > 0:
                fit_sigma = np.asarray(block.uncertainties, dtype=float)
                valid_sigma = (fit_sigma.shape == np.shape(block.values)
                               and np.all(np.isfinite(fit_sigma) &
                                              ((fit_sigma >= 0) if block.poisson_data is not None
                                               else (fit_sigma > 0))))
                if valid_sigma:
                    # Keep impossible configured bounds as explicit errors;
                    # invalid data errors still produce failed-attempt rows below.
                    p0, bounds = p0_generator(
                        block.energy, block.values, good_peaks,
                        response_sigma=block.response_sigma,
                        max_sigma_line=max_sigma_line, min_sigma_line=min_sigma_line,
                        uncertainties=fit_sigma,
                    )
                try:
                    # block.excess locates candidates; fitting must also retain
                    # fluctuations below the baseline in observed and null data.
                    target = block.values - block.baseline
                    if not valid_sigma or fit_sigma.shape != target.shape:
                        raise ValueError("Local fit requires finite positive uncertainties "
                                         "for every fitted bin.")
                    popt, pcov, width_fixed = fit_with_instrumental_width(
                        block.energy, target, fit_sigma, p0, bounds,
                        response_sigma=block.response_sigma,
                        poisson_data=block.poisson_data, baseline=block.baseline,
                        gaussian_fit=curve_fit, poisson_fit=fit_poisson_gaussians,
                    )
                    if block.poisson_data is not None:
                        noise_on_block = float(np.mean(poisson_model_uncertainty(
                            block.energy, block.baseline, block.poisson_data, popt)))
                    errors = np.sqrt(np.diag(pcov))
                    popt_ = np.reshape(popt, (-1, 3))
                    errors_ = np.reshape(errors, (-1, 3))
                    for k in range(len(popt_)):
                        eamplitude, ecenter, esigma = errors_[k]
                        if center_error_exceeds_limit(ecenter):
                            continue
                        rows.append({'fit_converged': True,
                                     'fit_message': '',
                                     'fit_block_id': block_index,
                                     'fit_block_width': float(np.ptp(block.energy)),
                                     'fit_center_initial': p0[3*k + 1],
                                     'amplitude': popt_[k][0],
                                     'center': popt_[k][1],
                                     'sigma': popt_[k][2],
                                     'eamplitude': eamplitude,
                                     'ecenter': ecenter,
                                     'esigma': esigma,
                                     'width_fixed_instrumental': bool(width_fixed[k]),
                                     'cov_amplitude_sigma': pcov[3*k, 3*k + 2],
                                     'noise_on_block': noise_on_block})
                except (RuntimeError, ValueError) as exc:
                    print(f'Error fitting block {block_index}: {exc}')
                    # Preserve a diagnostic row per attempted component. The
                    # center locates the attempted fit, not a measured centroid.
                    for k in range(len(good_peaks)):
                        rows.append(dict(
                            center=good_peaks.energy.iloc[k],
                            fit_center_initial=good_peaks.energy.iloc[k],
                            amplitude=np.nan, sigma=np.nan, eamplitude=np.nan,
                            ecenter=np.nan, esigma=np.nan,
                            width_fixed_instrumental=False,
                            cov_amplitude_sigma=np.nan,
                            noise_on_block=noise_on_block,
                            fit_converged=False,
                            fit_message=str(exc), fit_block_id=block_index,
                            fit_block_width=float(np.ptp(block.energy))))
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
        Candidate table with ``base_on_line`` and
        ``relative_power`` columns added.
    """
    if len(fitted) == 0:
        fitted['base_on_line'] = []
        fitted['relative_power'] = []
        return fitted
    min_diff_positions = []
    for i in range(len(fitted)):
        center = fitted.center.iloc[i]
        min_diff_positions.append(int(np.argmin(np.abs(x - center)))
                                  if np.isfinite(center) else None)
    fitted['base_on_line'] = [base[pos] if pos is not None else np.nan
                              for pos in min_diff_positions]
    values_on_line = np.array([y[pos] if pos is not None else np.nan
                              for pos in min_diff_positions])
    fitted['relative_power'] = (values_on_line - fitted.base_on_line) / (values_on_line + fitted.base_on_line)
    return fitted


def return_raw_lines(x, y, sy, ylines, base, response_sigma=None,
                     min_peak_separation=None, max_sigma_line=0.1,
                     min_relative_prominence=DEFAULT_MIN_RELATIVE_PROMINENCE, min_sigma_line=1e-6,
                     *, poisson_data=None):
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
    min_sigma_line : float, default: 1e-6
        Absolute lower sigma bound; combined with the instrumental floor.
    max_sigma_line : float, default: 0.1
        Configured upper sigma bound in keV, capped by the fitted block energy span.
    min_relative_prominence : float, default: 0.05
        Retain peaks strictly above this fraction of the largest prominence
        in their block. Must be finite and in [0, 1). Use the same value for
        observed and null spectra; this is not a significance threshold.

    poisson_data : PoissonFitData or None
        Aligned raw source/background counts and exposure-width conversion.
        When provided, use a Poisson likelihood on counts instead of weighted
        residuals. Zero raw errors are valid. The same input type and continuum
        estimator must be used in observed and simulated searches.

    Returns
    -------
    pandas.DataFrame
        Raw candidate-line table with Gaussian parameters, parameter errors,
        goodness-of-fit information, local continuum context, and instrumental
        ``response_sigma`` interpolated at each fitted centroid (NaN if absent).
        Centroid errors > 1 keV are discarded. Amplitude and width errors have
        no upper cut. Numerical
        validity is recorded by annotate_fit_quality and checked identically
        for observed and simulated candidates before significance evaluation.
        Quality metadata retains failed attempts: their center is the initial
        guess, their fitted amplitude/width/errors are NaN, and the failure
        message is stored. Non-evaluable attempts are not detection claims.
    """
    validate_sigma_bounds(min_sigma_line, max_sigma_line)
    validate_min_relative_prominence(min_relative_prominence)
    blocks = _build_candidate_blocks(x, y, sy, ylines, base,
                                     response_sigma=response_sigma, poisson_data=poisson_data)
    rows = []
    for block_index, block in enumerate(blocks):
        rows.extend(_fit_candidate_block(
            block, block_index, min_peak_separation=min_peak_separation,
            max_sigma_line=max_sigma_line, min_sigma_line=min_sigma_line,
            min_relative_prominence=min_relative_prominence))
    fitted = pd.DataFrame(rows, columns=['amplitude', 'center', 'sigma',
                                         'eamplitude', 'ecenter', 'esigma',
                                         'width_fixed_instrumental',
                                         'cov_amplitude_sigma',
                                         'noise_on_block',
                                         'fit_converged', 'fit_message',
                                         'fit_block_id', 'fit_center_initial', 'fit_block_width'])
    fitted = _add_line_context(fitted, x, y, base)
    fitted['response_sigma'] = np.nan
    if response_sigma is not None and len(fitted):
        fitted['response_sigma'] = np.interp(fitted['center'], x, response_sigma)
    return annotate_fit_quality(discard_large_center_errors(fitted)).reset_index(drop=True)
