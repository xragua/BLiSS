"""Detect contiguous excess regions and fit local Gaussian candidate lines."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from .peak_selection import find_peaks_new
from .gaussian_models import n_gaussian, p0_generator

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

    def detect(self, x, y, sy, ylines, base):
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
            Baseline-subtracted line excess array. Non-zero contiguous regions are
            treated as candidate blocks.
        base : array-like
            Empirical baseline evaluated on ``x``.

        Returns
        -------
        pandas.DataFrame
            Preliminary Gaussian candidate table with fitted parameters and local
            context columns.
        """
        return return_raw_lines(x, y, sy, ylines, base)

def _build_candidate_blocks(x, y, sy, ylines, base, response_sigma=None):
    """Split non-zero line excesses into contiguous candidate blocks.

    Parameters
    ----------
    x, y, sy : array-like
        Coordinate grid, observed values, and one-sigma uncertainties.
    ylines : array-like
        Baseline-subtracted excess array. Non-zero runs define candidate regions.
    base : array-like
        Baseline values corresponding to ``x``.
    response_sigma : array-like or None, default: None
        Instrumental Gaussian-equivalent sigma aligned with ``x``.

    Returns
    -------
    list of CandidateBlock
        Candidate blocks containing local arrays for fitting.
    """
    nonzero_indices = np.nonzero(ylines)[0]
    if len(nonzero_indices) == 0:
        return []
    blocks = []
    start_index = nonzero_indices[0]
    n = 0
    block_arrays = {}
    xblocks = {}
    yblocks = {}
    syblocks = {}
    contblocks = {}
    for i in range(len(nonzero_indices)):
        if nonzero_indices[i] - nonzero_indices[i - 1] == 1:
            end_index = nonzero_indices[i]
            continue
        end_index = nonzero_indices[i]
        block_arrays[n] = np.array(ylines[start_index - 1:end_index])
        xblocks[n] = np.array(x[start_index - 1:end_index])
        yblocks[n] = np.array(y[start_index - 1:end_index])
        syblocks[n] = np.array(sy[start_index - 1:end_index])
        contblocks[n] = np.array(base[start_index - 1:end_index])
        start_index = nonzero_indices[i]
        n += 1
    for i in range(len(block_arrays)):
        for j in range(1, len(block_arrays[i])):
            if block_arrays[i][j] == 0 and block_arrays[i][j - 1] == 0 and (len(block_arrays[i]) > 2):
                block_arrays[i] = block_arrays[i][:j]
                xblocks[i] = xblocks[i][:j]
                yblocks[i] = yblocks[i][:j]
                syblocks[i] = syblocks[i][:j]
                contblocks[i] = contblocks[i][:j]
                break
    for i in range(len(block_arrays)):
        blocks.append(CandidateBlock(excess=block_arrays[i], energy=xblocks[i], values=yblocks[i], uncertainties=syblocks[i], baseline=contblocks[i], response_sigma=None if response_sigma is None else np.interp(xblocks[i], np.asarray(x, dtype=float), np.asarray(response_sigma, dtype=float))))
    return blocks

def _fit_candidate_block(block, block_index):
    """Fit Gaussian components inside one candidate block.

    Parameters
    ----------
    block : CandidateBlock
        Local region containing spectral values, uncertainties, and baseline.
    block_index : int
        Index of the block in the candidate-block list, used only for diagnostic
        error messages.

    Returns
    -------
    list of dict
        One dictionary per fitted local Gaussian, containing amplitude, center,
        sigma, formal errors, block-level R-squared, and mean block uncertainty.
    """
    rows = []
    noise_on_block = np.mean(block.uncertainties)
    if len(block.values) > 3 and max(block.values) > 0:
        res_dif = 0.001
        peaks = find_peaks_new(block.energy, block.values)
        if len(peaks) > 0:
            good_list = []
            for i in range(len(peaks)):
                prominence_ratio = peaks.prominences[i] / max(peaks.prominences)
                idx1 = abs(peaks.energy[i] - peaks.energy) < res_dif
                if all(peaks.prominences[i] - peaks.prominences[idx1] >= 0) & (prominence_ratio > 0.1):
                    good_list.append(i)
            good_peaks = peaks.loc[good_list].sort_values(by='prominences').reset_index(drop=True)
            max_peaks = int(np.floor(len(block.energy) / 4))
            good_peaks = good_peaks[0:max(1, max_peaks)]
            if len(good_peaks) > 0:
                p0, bounds = p0_generator(block.energy, block.values, good_peaks, response_sigma=block.response_sigma)
                try:
                    popt, pcov = curve_fit(n_gaussian, block.energy, block.values - block.baseline, p0=p0, bounds=bounds, maxfev=100000)
                    errors = np.sqrt(np.diag(pcov))
                    yfit = n_gaussian(block.energy, *popt)
                    rsq = 1 - np.sum((block.values - yfit) ** 2) / np.sum((block.values - np.mean(block.values)) ** 2)
                    popt_ = np.reshape(popt, (-1, 3))
                    errors_ = np.reshape(errors, (-1, 3))
                    for k in range(len(good_peaks)):
                        rows.append({'amplitude': popt_[k][0], 'center': popt_[k][1], 'sigma': popt_[k][2], 'eamplitude': errors_[k][0], 'ecenter': errors_[k][1], 'esigma': errors_[k][2], 'rsq': rsq, 'noise_on_block': noise_on_block})
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

def return_raw_lines(x, y, sy, ylines, base, response_sigma=None):
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

    Returns
    -------
    pandas.DataFrame
        Raw candidate-line table with Gaussian parameters, parameter errors,
        goodness-of-fit information, and local continuum context.
    """
    blocks = _build_candidate_blocks(x, y, sy, ylines, base, response_sigma=response_sigma)
    rows = []
    for block_index, block in enumerate(blocks):
        rows.extend(_fit_candidate_block(block, block_index))
    fitted = pd.DataFrame(rows, columns=['amplitude', 'center', 'sigma', 'eamplitude', 'ecenter', 'esigma', 'rsq', 'noise_on_block'])
    fitted = _add_line_context(fitted, x, y, base)
    return fitted.reset_index(drop=True)
