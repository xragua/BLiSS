"""Detect contiguous excess regions and fit local Gaussian candidate lines."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from .peak_selection import find_peaks_new
from .gaussian_models import n_gaussian, p0_generator

DEFAULT_MIN_PEAK_SEPARATION = 0.001
"""Floor of the minimum separation between retained peaks, in spectral-axis
units (keV by default). When the instrumental ``response_sigma`` is
available, the effective local separation is
``max(DEFAULT_MIN_PEAK_SEPARATION, sigma_inst(E))``."""


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
               min_peak_separation=None):
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


def _fit_candidate_block(block, block_index, min_peak_separation=None):
    """Fit Gaussian components inside one candidate block.

    Peak retention follows two rules: (i) minimum separation -- among peaks
    closer than the local separation scale, only the most prominent is kept;
    (ii) relative prominence -- peaks below 10 per cent of the largest
    prominence in the block are discarded. The local separation scale is
    ``max(min_peak_separation, sigma_inst(E))`` when the instrumental
    ``response_sigma`` is available, and ``min_peak_separation`` otherwise.
    The number of Gaussians is capped at ``floor(n_bins / 4)``, keeping the
    most prominent peaks. Local fits are performed on the signed
    baseline-subtracted residual of the block.

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
        uncertainty.
    """
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
                    popt, pcov = curve_fit(n_gaussian, block.energy, target,
                                           p0=p0, bounds=bounds,
                                           maxfev=100000)
                    errors = np.sqrt(np.diag(pcov))
                    yfit = n_gaussian(block.energy, *popt)
                    ss_tot = np.sum((target - np.mean(target)) ** 2)
                    if ss_tot > 0:
                        rsq = 1 - np.sum((target - yfit) ** 2) / ss_tot
                    else:
                        rsq = np.nan
                    popt_ = np.reshape(popt, (-1, 3))
                    errors_ = np.reshape(errors, (-1, 3))
                    for k in range(len(good_peaks)):
                        rows.append({'amplitude': popt_[k][0],
                                     'center': popt_[k][1],
                                     'sigma': popt_[k][2],
                                     'eamplitude': errors_[k][0],
                                     'ecenter': errors_[k][1],
                                     'esigma': errors_[k][2],
                                     'rsq': rsq,
                                     'noise_on_block': noise_on_block})
                except RuntimeError as exc:
                    print(f'Error fitting block {block_index}: {exc}')
                except ValueError as exc:
                    print(f'Error fitting block {block_index}: {exc}')
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
        min_diff_index = np.argmin(np.abs(x - fitted.center.iloc[i]))
        min_diff_positions.append(min_diff_index)
    fitted['base_on_line'] = [base[pos] for pos in min_diff_positions]
    fitted['value_on_line'] = [y[pos] for pos in min_diff_positions]
    fitted['relative_power'] = (fitted.value_on_line - fitted.base_on_line) / (fitted.value_on_line + fitted.base_on_line)
    return fitted


def return_raw_lines(x, y, sy, ylines, base, response_sigma=None,
                     min_peak_separation=None):
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

    Returns
    -------
    pandas.DataFrame
        Raw candidate-line table with Gaussian parameters, parameter errors,
        goodness-of-fit information, and local continuum context.
    """
    blocks = _build_candidate_blocks(x, y, sy, ylines, base,
                                     response_sigma=response_sigma)
    rows = []
    for block_index, block in enumerate(blocks):
        rows.extend(_fit_candidate_block(
            block, block_index, min_peak_separation=min_peak_separation))
    fitted = pd.DataFrame(rows, columns=['amplitude', 'center', 'sigma',
                                         'eamplitude', 'ecenter', 'esigma',
                                         'rsq', 'noise_on_block'])
    fitted = _add_line_context(fitted, x, y, base)
    return fitted.reset_index(drop=True)
