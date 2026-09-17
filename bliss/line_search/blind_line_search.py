"""High-level BLiSS emission-line search pipeline."""
from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from .empirical_baseline import base_calculator
from .fit_quality import annotate_fit_quality, FIT_QUALITY_COLUMNS
from .candidate_regions import return_raw_lines
from .gaussian_models import n_gaussian, p0_generator_final, gaussian_area_error
from ..synthetic_probability.synthetic_spectra import generate_null_realizations
from ..synthetic_probability.gmm_score import eval_bliss_score_gmm
from ..synthetic_probability.look_elsewhere import (
    P_VALUE_COLUMNS, calculate_look_elsewhere_pvalues,
)
from ..plotting.run_output_manager import ensure_output_folder
from ..spectrum_data.fits_spectrum_loader import (
    load_fits_spectrum,
    read_pha_metadata,
    align_to_spectrum,
)
from ..spectrum_data.rebinning_tools import rebin_counts

# Full working schema: retain covariance, fit context and quality diagnostics
# until all fitting and energy-window selection have finished.
LINE_OUTPUT_COLUMNS = [
    'center', 'ecenter', 'sigma', 'esigma', 'amplitude', 'eamplitude',
    'cov_amplitude_sigma',
    'relative_power', 'noise_on_block',
    'base_on_line', 'snr_peak', 'snr_amplitude', 'area', 'earea',
    'snr_area', 'ew', 'bliss_score',
    'response_feature',
    'bliss_score_status', *FIT_QUALITY_COLUMNS, *P_VALUE_COLUMNS,]

CANDIDATE_COLUMNS = LINE_OUTPUT_COLUMNS.copy()
FINAL_OUTPUT_COLUMNS = [
    'center', 'ecenter',
    'sigma', 'esigma',
    'amplitude', 'eamplitude',
    'area', 'earea',
    'ew',
    'relative_power',
    'snr_peak',
    'snr_area',
    'bliss_score',
    'response_feature',
    *P_VALUE_COLUMNS,
]


@dataclass
class BlindLineSearchConfig:
    """Configuration values controlling the default BLiSS pipeline.

    Attributes
    ----------
    en1, en2 : float
        Lower and upper energy limits used when selecting candidates.
    energy_pad : float
        Extra energy range added around the requested interval during
        candidate detection and fitting.

    num_synthetic_simulations : int
        Nonnegative number of synthetic spectra for bliss_score and the five
        look-elsewhere p-values. Zero disables simulation-based significance.
    synthetic_seed : int or None
        Random seed used for synthetic-spectrum generation.

    final_fit_maxfev : int
        Maximum number of function evaluations allowed in the final
        ``curve_fit``.

    response_feature_threshold : float
        Robust score above which unusually sharp ARF effective-area
        structure is flagged.

    min_area_snr : float or None
        Optional GMM preselection on covariance-aware area S/N, applied to
        observed and null candidates. None (default) disables this cut.

    max_sigma_line : float
        Maximum Gaussian sigma allowed for an individual line candidate,
        in keV. Applied to local data/null fits and the optional global fit.

    noise_model : {'poisson', 'gaussian'}
        Statistical model of the synthetic null spectra. ``'poisson'`` draws
        counts at channel resolution (default); ``'gaussian'`` is kept only
        for validation/comparison and should not be used for science results.

    rebin_method, rebin_scale, rebin_min_bins
        Count-conserving rebinning applied to the native spectrum before the
        search and to every synthetic null realization (see ``rebin_counts``).

    baseline_window : float, array-like, or callable
        Preferred physical width, in keV, of the running-median window
        used to estimate the empirical baseline. A callable ``f(energy)``
        is evaluated on every grid where a baseline is computed (data and
        each null realization), enabling energy-dependent windows.

    max_range_fraction : float
        Maximum baseline-window width as a fraction of the total energy
        range of the input spectrum.

    min_points : int
        Minimum number of spectral points required inside a local
        running-median window.
    """

    en1: float = 0.2
    en2: float = 10.0
    energy_pad: float = 0.1

    num_synthetic_simulations: int = 10
    synthetic_seed: Optional[int] = None

    final_fit_maxfev: int = 100000
    response_feature_threshold: float = 5.0

    # Line fitting
    max_sigma_line: float = 0.1

    # Empirical baseline
    baseline_window: float = 0.4
    max_range_fraction: float = 0.2
    min_points: int = 3

    # Synthetic noise
    noise_model: str = "poisson"       # 'poisson' (default) | 'gaussian' (validation only)

    # Rebinning applied to count spectra (and to every null realization)
    rebin_method: str = "none"          # 'none' | 'bins' | 'snr' | 'resolution'
    rebin_scale: Optional[float] = None
    rebin_min_bins: int = 1

    # No area-significance cut before the GMM by default.
    min_area_snr: Optional[float] = None



@dataclass
class PreparedSpectrum:
    """Sorted spectral arrays used internally by the BLiSS pipeline.

    Attributes
    ----------
    energy : numpy.ndarray
        Sorted spectral coordinate grid.
    values : numpy.ndarray
        Observed spectral values sorted by energy.
    uncertainties : numpy.ndarray
        One-sigma uncertainties sorted with the spectrum.
    bin_width : numpy.ndarray
        Width of each spectral bin, used when estimating equivalent width.
    response_sigma : numpy.ndarray or None
        Instrumental Gaussian-equivalent sigma aligned with the spectral grid.
    arf_energy : numpy.ndarray or None
        Energy grid of the associated ARF response.
    effective_area : numpy.ndarray or None
        Effective area of the associated ARF response.
    """
    energy: np.ndarray
    values: np.ndarray
    uncertainties: np.ndarray
    bin_width: np.ndarray
    response_sigma: Optional[np.ndarray] = None
    arf_energy: Optional[np.ndarray] = None
    effective_area: Optional[np.ndarray] = None
    native: Optional["NativeCounts"] = None

@dataclass
class NativeCounts:
    """Channel-resolution count information kept alongside a density spectrum.
    Attributes
    ----------
    energy, bin_width : numpy.ndarray
        Native channel centres and widths (keV).
    counts, uncertainties : numpy.ndarray
        Native (net) counts and their uncertainties, as loaded.
    bkg_counts : numpy.ndarray
        Raw background counts per native channel (zeros if no background).
    bkg_scale : float
        Factor applied to background counts before subtraction.
    exposure : float
        Exposure time in seconds.
    response_sigma : numpy.ndarray or None
        Instrumental sigma on the native grid.
    group : numpy.ndarray of int
        Output-bin index of each native channel after rebinning (-1 = dropped).
    """
    energy: np.ndarray
    bin_width: np.ndarray
    counts: np.ndarray
    uncertainties: np.ndarray
    bkg_counts: np.ndarray
    bkg_scale: float
    exposure: float
    response_sigma: Optional[np.ndarray]
    group: np.ndarray


def prepare_spectrum(
    pha_path,
    rmf_path,
    arf_path=None,
    background_path=None,
    *,
    rebin_method: str = "none",
    rebin_scale: Optional[float] = None,
    rebin_min_bins: int = 1,
) -> PreparedSpectrum:
    """Load a PHA spectrum, rebin it conserving counts, and convert to density.

    The returned ``PreparedSpectrum`` is in counts s^-1 keV^-1 on the rebinned
    grid. Its ``native`` attribute keeps the channel-resolution counts,
    background and grouping needed to generate Poisson null realizations that
    follow exactly the same rebinning.
    """
    spectrum = load_fits_spectrum(
        pha_path=pha_path,
        background_path=background_path,
        rmf_path=rmf_path,
        arf_path=arf_path,
    )
    meta = align_to_spectrum(
        read_pha_metadata(pha_path, rmf_path, background_path),
        spectrum.energy,
    )

    exposure = float(meta["exposure"])
    if not np.isfinite(exposure) or exposure <= 0:
        raise ValueError("PHA file has no valid EXPOSURE keyword.")

    # Native (net) counts. The loader returns the column as stored: counts,
    # or counts/s for RATE columns.
    if meta["values_unit"] == "rate":
        counts = spectrum.values * exposure
        counts_unc = spectrum.uncertainties * exposure
    else:
        counts = spectrum.values.copy()
        counts_unc = spectrum.uncertainties.copy()

    # Count-conserving rebinning
    e_new, n_new, s_new, de_new, group = rebin_counts(
        spectrum.energy, counts, counts_unc, spectrum.bin_width,
        method=rebin_method, scale=rebin_scale, min_bins=rebin_min_bins,
    )

    # Density: counts s^-1 keV^-1
    conv = exposure * de_new
    values = n_new / conv
    uncertainties = s_new / conv

    response_sigma = None
    if spectrum.response_sigma is not None:
        response_sigma = np.interp(
            e_new, spectrum.energy, spectrum.response_sigma,
            left=spectrum.response_sigma[0], right=spectrum.response_sigma[-1],
        )

    native = NativeCounts(
        energy=spectrum.energy,
        bin_width=spectrum.bin_width,
        counts=counts,
        uncertainties=counts_unc,
        bkg_counts=np.asarray(meta["bkg_counts"], dtype=float),
        bkg_scale=float(meta["bkg_scale"]),
        exposure=exposure,
        response_sigma=spectrum.response_sigma,
        group=group,
    )

    return PreparedSpectrum(
        energy=e_new,
        values=values,
        uncertainties=uncertainties,
        bin_width=de_new,
        response_sigma=response_sigma,
        arf_energy=spectrum.arf_energy,
        effective_area=spectrum.effective_area,
        native=native,
    )

def _baseline_and_line_excess(
    spectrum: PreparedSpectrum,
    base: Optional[np.ndarray] = None,
    ylines: Optional[np.ndarray] = None,
    baseline_window: float | np.ndarray = 0.4,
    max_range_fraction: float = 0.2,
    min_points: int = 3,
):
    """Return empirical baseline and positive line excess.

    Parameters
    ----------
    spectrum : PreparedSpectrum
        Prepared input spectrum.

    base : array-like or None, default=None
        User-provided baseline. If omitted, the empirical BLiSS
        baseline is calculated from the spectrum.

    ylines : array-like or None, default=None
        User-provided positive-excess spectrum. If omitted, it is
        calculated as ``max(values - baseline, 0)``.

    baseline_window : float or array-like, default=0.4
        Running-median width in physical energy units.

    max_range_fraction : float, default=0.2
        Maximum baseline-window width as a fraction of the
        available energy range.

    min_points : int, default=3
        Minimum number of points required inside the local
        running-median window.

    Returns
    -------
    base, ylines : numpy.ndarray
        Empirical baseline and positive-excess spectrum.
    """

    if base is None:

        base = base_calculator(
            spectrum.energy,
            spectrum.values,
            baseline_window=baseline_window,
            max_range_fraction=max_range_fraction,
            min_points=min_points,
        )

    else:

        base = np.asarray(
            base,
            dtype=float,
        )

    if len(base) != len(spectrum.energy):
        raise ValueError(
            "base must have the same length as the spectrum."
        )

    if ylines is None:

        ylines = np.maximum(
            spectrum.values - base,
            0.0,
        )

    else:

        ylines = np.asarray(
            ylines,
            dtype=float,
        )

    if len(ylines) != len(spectrum.energy):
        raise ValueError(
            "ylines must have the same length as the spectrum."
        )

    return base, ylines

def _slice_prepared_spectrum(
    spectrum: PreparedSpectrum,
    mask: np.ndarray,
) -> PreparedSpectrum:
    """Return a PreparedSpectrum restricted to a boolean mask."""

    mask = np.asarray(mask, dtype=bool)

    if len(mask) != len(spectrum.energy):
        raise ValueError("mask must have the same length as the spectrum.")

    return PreparedSpectrum(
        energy=spectrum.energy[mask],
        values=spectrum.values[mask],
        uncertainties=spectrum.uncertainties[mask],
        bin_width=spectrum.bin_width[mask],
        response_sigma=None if spectrum.response_sigma is None else spectrum.response_sigma[mask],
        arf_energy=spectrum.arf_energy,
        effective_area=spectrum.effective_area,
    )

def _arf_feature_profile(arf_energy, effective_area):
    """Return a robust sharpness score for structure in an ARF effective-area curve.

    The score is based on the absolute gradient of log effective area and is
    normalized with the median absolute deviation. It is used only to flag
    candidates near unusually sharp response structure; candidates are never
    rejected automatically.
    """
    if arf_energy is None or effective_area is None:
        return None, None

    energy = np.asarray(arf_energy, dtype=float)
    area = np.asarray(effective_area, dtype=float)
    good = np.isfinite(energy) & np.isfinite(area) & (area > 0)
    energy = energy[good]
    area = area[good]
    if len(energy) < 3:
        return None, None

    order = np.argsort(energy)
    energy = energy[order]
    area = area[order]
    sharpness = np.abs(np.gradient(np.log(area), energy))
    median = np.nanmedian(sharpness)
    mad = np.nanmedian(np.abs(sharpness - median))
    if not np.isfinite(mad) or mad <= 0:
        score = np.zeros_like(sharpness)
        score[sharpness > median] = np.inf
        return energy, score
    score = 0.67448975 * (sharpness - median) / mad
    return energy, np.maximum(score, 0.0)


def _flag_response_features(lines, spectrum, threshold):
    """Annotate candidates that lie near unusually sharp ARF structure."""
    lines = lines.copy()
    lines['response_feature'] = False
    if len(lines) == 0:
        return lines

    arf_energy, score = _arf_feature_profile(spectrum.arf_energy, spectrum.effective_area)
    if arf_energy is None:
        return lines

    arf_step = np.nanmedian(np.diff(arf_energy)) if len(arf_energy) > 1 else 0.0
    for idx, center in lines['center'].items():
        center = float(center)
        if not np.isfinite(center):
            continue
        if spectrum.response_sigma is not None and len(spectrum.response_sigma):
            local_sigma = float(np.interp(center, spectrum.energy, spectrum.response_sigma))
        else:
            local_sigma = 0.0
        radius = max(local_sigma, float(arf_step) if np.isfinite(arf_step) else 0.0)
        nearby = np.abs(arf_energy - center) <= radius
        if not np.any(nearby):
            nearest = int(np.argmin(np.abs(arf_energy - center)))
            local_score = float(score[nearest])
        else:
            local_score = float(np.nanmax(score[nearby]))
        lines.at[idx, 'response_feature'] = bool(local_score >= threshold)
    return lines


def _safe_divide(numerator, denominator):
    """Return numerator / denominator, using NaN where the ratio is undefined."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.full_like(numerator, np.nan, dtype=float)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
    np.divide(numerator, denominator, out=out, where=valid)
    return out


def _add_candidate_metrics(lines: pd.DataFrame) -> pd.DataFrame:
    """Add pre-global-fit diagnostics to candidate lines.

    The equivalent width reported here is an approximate candidate EW, computed
    from the local Gaussian area divided by the empirical baseline at the line
    centroid. If the spectral coordinate is in keV, the returned EW is in eV.
    The global-fit EW is recomputed later from the final multi-Gaussian model.
    Area errors include amplitude/width covariance; legacy tables without that
    covariance have unavailable area errors and area S/N, not diagonal estimates.
    """
    lines = lines.copy()

    if len(lines) == 0:
        for col in ['snr_peak', 'snr_amplitude', 'area', 'earea', 'snr_area', 'ew']:
            if col not in lines.columns:
                lines[col] = []
        return lines

    numeric_cols = [
        'amplitude', 'eamplitude', 'sigma', 'esigma', 'cov_amplitude_sigma',
        'noise_on_block', 'base_on_line'
    ]
    for col in numeric_cols:
        if col in lines.columns:
            lines[col] = pd.to_numeric(lines[col], errors='coerce')
        else:
            lines[col] = np.nan

    k = np.sqrt(2.0 * np.pi)

    lines['snr_peak'] = _safe_divide(lines['amplitude'], lines['noise_on_block'])
    lines['snr_amplitude'] = _safe_divide(lines['amplitude'], lines['eamplitude'])

    lines['area'] = lines['amplitude'] * lines['sigma'] * k
    lines['earea'] = gaussian_area_error(
        lines['amplitude'], lines['sigma'], lines['eamplitude'], lines['esigma'],
        lines['cov_amplitude_sigma'],
    )
    lines['snr_area'] = _safe_divide(lines['area'], lines['earea'])

    # Approximate candidate equivalent width. Assumes energy in keV, hence x1000 -> eV.
    lines['ew'] = _safe_divide(lines['area'], lines['base_on_line']) * 1000.0

    return lines.replace([np.inf, -np.inf], np.nan)


def final_fit_and_metrics(
    spectrum: PreparedSpectrum,
    clean_lines: pd.DataFrame,
    base: Optional[np.ndarray] = None,
    ylines: Optional[np.ndarray] = None,
    final_fit_maxfev: int = 100000,
    baseline_window=0.4,
    max_range_fraction: float = 0.2,
    min_points: int = 3,
    response_feature_threshold: float = 5.0,
    max_sigma_line: float = 0.1,
):
    """Refit selected candidates and compute final line diagnostics.

    Area errors use the final joint fit covariance, replacing local covariance.
    All amplitudes, centers and widths are free within their bounds.
    Every width is bounded above by ``max_sigma_line`` (default 0.1 keV).
    Input ``bliss_score`` is preserved, including when the global fit fails.
    It describes the earlier observed/null comparison. The returned table
    contains exactly ``FINAL_OUTPUT_COLUMNS``; full diagnostics stay internal.

    The baseline parameters are used only when ``base``/``ylines`` are not
    provided; they must then match the values used for the candidate search.
    ``baseline_window`` may be a float, an array aligned with
    ``spectrum.energy``, or a callable ``f(energy)``.
    ARF feature flags and scores are recomputed at the fitted centroids using
    ``response_feature_threshold``. Without usable ARF data, the flag is False
    (not flagged) and the score is NaN (unavailable).
    """

    result, yfit = _fit_global_with_diagnostics(
        spectrum=spectrum,
        clean_lines=clean_lines,
        base=base,
        ylines=ylines,
        final_fit_maxfev=final_fit_maxfev,
        baseline_window=baseline_window,
        max_range_fraction=max_range_fraction,
        min_points=min_points,
        response_feature_threshold=response_feature_threshold,
        max_sigma_line=max_sigma_line,
    )
    return result.reindex(columns=FINAL_OUTPUT_COLUMNS).copy(), yfit


def _fit_global_with_diagnostics(
    spectrum: PreparedSpectrum,
    clean_lines: pd.DataFrame,
    base: Optional[np.ndarray] = None,
    ylines: Optional[np.ndarray] = None,
    final_fit_maxfev: int = 100000,
    baseline_window=0.4,
    max_range_fraction: float = 0.2,
    min_points: int = 3,
    response_feature_threshold: float = 5.0,
    max_sigma_line: float = 0.1,
):
    """Compute the global fit, retaining context until final output selection."""

    base, ylines = _baseline_and_line_excess(
        spectrum,
        base=base,
        ylines=ylines,
        baseline_window=baseline_window,
        max_range_fraction=max_range_fraction,
        min_points=min_points,
    )

    clean_lines = clean_lines.copy().reset_index(drop=True)
    missing_for_fit = sorted({'amplitude', 'center', 'sigma'} - set(clean_lines))
    if len(clean_lines) and missing_for_fit:
        raise ValueError(
            'pd_lines must contain amplitude, center, and sigma columns. '
            f'Missing: {missing_for_fit}')
    missing_metrics = {'area', 'earea', 'snr_peak', 'snr_area', 'ew'} - set(clean_lines)
    missing_baseline = 'base_on_line' not in clean_lines
    if len(clean_lines):
        if 'noise_on_block' not in clean_lines and 'snr_peak' in clean_lines:
            # Recover the same block noise from a compact candidate catalogue.
            clean_lines['noise_on_block'] = _safe_divide(
                clean_lines['amplitude'], clean_lines['snr_peak'])
        if 'noise_on_block' not in clean_lines or 'relative_power' not in clean_lines:
            nearest = [int(np.argmin(np.abs(spectrum.energy - center)))
                       for center in clean_lines['center']]
            if 'noise_on_block' not in clean_lines:
                clean_lines['noise_on_block'] = spectrum.uncertainties[nearest]
            if 'relative_power' not in clean_lines:
                values, continuum = spectrum.values[nearest], base[nearest]
                clean_lines['relative_power'] = _safe_divide(
                    values - continuum, values + continuum)
    # Compact catalogues have no internal flags. Recheck their parameter
    # errors before refitting; parameter-only legacy inputs remain supported.
    if ('fit_evaluable' not in clean_lines
            and {'ecenter', 'esigma', 'eamplitude'}.issubset(clean_lines)):
        clean_lines = annotate_fit_quality(clean_lines)
    for col in LINE_OUTPUT_COLUMNS:
        if col not in clean_lines:
            clean_lines[col] = np.nan
    invalid = clean_lines['fit_evaluable'].eq(False)
    rejected = clean_lines[invalid].copy()
    clean_lines = clean_lines[~invalid].reset_index(drop=True)
    # Rejected rows keep their local measurements. Only fill missing metrics
    # for legacy inputs; never recompute compact-table errors without covariance.
    if len(rejected) and missing_metrics:
        if missing_baseline and 'ew' in missing_metrics:
            rejected['base_on_line'] = [
                base[int(np.argmin(np.abs(spectrum.energy - center)))]
                for center in rejected['center']]
        metrics = _add_candidate_metrics(rejected)
        for col in missing_metrics:
            rejected[col] = metrics[col]
    global_converged = False
    global_message = ''
    fitted_final = pd.DataFrame(
        columns=[
            "amplitude",
            "center",
            "sigma",
            "eamplitude",
            "ecenter",
            "esigma",
            "cov_amplitude_sigma",
        ]
    )

    yfit = np.zeros_like(spectrum.energy, dtype=float)

    if len(clean_lines) > 0:
        p0, bounds = p0_generator_final(
            spectrum.energy,
            spectrum.values,
            clean_lines,
            response_sigma=spectrum.response_sigma,
            max_sigma_line=max_sigma_line,
        )

        # Use the spectral uncertainties as weights in the global fit.
        # curve_fit bounds constrain fitted parameters, not their errors.
        fit_sigma = np.asarray(spectrum.uncertainties, dtype=float)

        valid_sigma = (
            np.isfinite(fit_sigma)
            & (fit_sigma > 0)
        )

        if np.any(valid_sigma):
            fallback_sigma = np.nanmedian(fit_sigma[valid_sigma])
        else:
            fallback_sigma = 1.0

        fit_sigma = np.where(
            valid_sigma,
            fit_sigma,
            fallback_sigma,
        )

        try:
            popt, pcov = curve_fit(
                n_gaussian, spectrum.energy, ylines, p0=p0, bounds=bounds,
                sigma=fit_sigma, absolute_sigma=True, maxfev=final_fit_maxfev,
            )

            global_converged = True
            errors = np.sqrt(np.diag(pcov))

            yfit = n_gaussian(
                spectrum.energy,
                *popt,
            )

            popt_ = np.reshape(popt, (-1, 3))
            errors_ = np.reshape(errors, (-1, 3))

            for k in range(len(clean_lines)):
                new_line = np.concatenate(
                    [
                        popt_[k],
                        errors_[k],
                        [pcov[3*k, 3*k + 2]],
                    ]
                )
                fitted_final.loc[k] = new_line

        except (RuntimeError, ValueError) as exc:
            global_message = str(exc)
            print(f"Error final fitting: {exc}")

    # Detection p-values refer to the local search and remain unchanged by refitting.
    clean_select = clean_lines[["bliss_score", "noise_on_block", "relative_power", *P_VALUE_COLUMNS]]

    result = pd.concat(
        [
            fitted_final,
            clean_select,
        ],
        axis=1,
    )

    if len(result) == 0:
        return rejected.reindex(columns=LINE_OUTPUT_COLUMNS).reset_index(drop=True), yfit

    cols = [
        "amplitude",
        "center",
        "sigma",
        "eamplitude",
        "ecenter",
        "esigma",
        "cov_amplitude_sigma",
    ]

    result[cols] = result[cols].apply(
        pd.to_numeric,
        errors="coerce",
    )

    result['snr_peak'] = _safe_divide(result['amplitude'], result['noise_on_block'])

    k = np.sqrt(2.0 * np.pi)

    result["area"] = (
        result["amplitude"]
        * result["sigma"]
        * k
    )

    result["earea"] = gaussian_area_error(
        result["amplitude"], result["sigma"], result["eamplitude"], result["esigma"],
        result["cov_amplitude_sigma"],
    )

    result["snr_area"] = _safe_divide(
        result["area"],
        result["earea"],
    )

    ew_vals = []

    for row in result.itertuples(index=False):

        center = float(row.center)
        sigma = float(row.sigma)

        mask = (
            (spectrum.energy >= center - 2.0 * sigma)
            & (spectrum.energy <= center + 2.0 * sigma)
        )

        valid_mask = (
            mask
            & np.isfinite(base)
            & (base != 0)
        )

        if np.any(valid_mask):
            ew = (
                np.sum(
                    # Integrate this component only, excluding neighbouring lines.
                    n_gaussian(
                        spectrum.energy[valid_mask],
                        float(row.amplitude), center, sigma,
                    )
                    / base[valid_mask]
                    * spectrum.bin_width[valid_mask]
                )
                * 1000.0
            )
        else:
            ew = np.nan

        ew_vals.append(ew)

    result["ew"] = ew_vals

    result['fit_converged'] = global_converged
    result['fit_message'] = global_message
    result['fit_center_initial'] = clean_lines['center'].to_numpy()
    result['fit_block_id'] = np.nan  # the global fit is not a local block
    if not global_converged:
        result['center'] = result['fit_center_initial']
    result = annotate_fit_quality(result)
    result['bliss_score_status'] = (clean_lines['bliss_score_status'].to_numpy()
                              if 'bliss_score_status' in clean_lines else 'unavailable')
    result = _flag_response_features(result, spectrum, response_feature_threshold)
    result = result.reindex(columns=LINE_OUTPUT_COLUMNS)

    # Keep the observed/null score and its status as provenance. Global fit
    # failures are recorded in the separate fit-quality columns above.
    result = pd.concat([result, rejected.reindex(columns=LINE_OUTPUT_COLUMNS)],
                       ignore_index=True)
    return result, yfit


def plot_global_fit(
    spectrum: PreparedSpectrum,
    base: np.ndarray,
    yfit: np.ndarray,
    output_path: str | Path | None = None,
    *,
    show_plot: bool = True,
    energy_min: float | None = None,
    energy_max: float | None = None,
    size_fig_input: tuple[float, float] | None = None,
) -> None:
    """Plot the spectrum, empirical baseline, and global line model.

    Parameters
    ----------
    spectrum : PreparedSpectrum
        Prepared spectrum containing energy, values, and uncertainties.
    base : np.ndarray
        Empirical baseline.
    yfit : np.ndarray
        Global fitted line model.
    output_path : str, Path, or None, default=None
        If given, save the figure to this path.
    show_plot : bool, default=True
        Whether to display the figure.
    energy_min : float or None, default=None
        Minimum energy shown in the plot.
    energy_max : float or None, default=None
        Maximum energy shown in the plot.
    size_fig_input : tuple or None, default=None
        Figure size, e.g. ``(10, 5)``.
    """

    if size_fig_input is None:
        size_fig = (10, 5)
    else:
        size_fig = size_fig_input

    plt.figure(figsize=size_fig)

    plt.errorbar(
        spectrum.energy,
        spectrum.values,
        yerr=spectrum.uncertainties,
        label="Data",
        alpha=0.2,
    )

    plt.plot(spectrum.energy, base, "k:", label="base")
    plt.plot(spectrum.energy, yfit, "g:", label="Lines")
    plt.plot(spectrum.energy, yfit + base, "r", label="Line+base")

    if energy_min is not None or energy_max is not None:
        plt.xlim(energy_min, energy_max)

    plt.xlabel("Energy (keV)")
    plt.ylabel("Spectra")
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()


def fit_global(
    pd_lines: pd.DataFrame,
    spectrum: PreparedSpectrum,
    *,
    base: Optional[np.ndarray] = None,
    ylines: Optional[np.ndarray] = None,
    show_plot: bool = True,
    output_dir=None,
    plot_name: str = "bliss_global_fit.png",
    save_csv: bool = True,
    final_fit_maxfev: int = 100000,
    baseline_window=0.4,
    max_range_fraction: float = 0.2,
    min_points: int = 3,
    return_yfit: bool = False,
    energy_min: float | None = None,
    energy_max: float | None = None,
    size_fig_input: tuple[float, float] | None = None,
    response_feature_threshold: float = 5.0,
    max_sigma_line: float = 0.1,
):
    """Run only the expensive global fit on user-filtered candidate lines.

    Parameters
    ----------
    pd_lines : pandas.DataFrame
        Candidate-line table after any user-defined filtering.
    spectrum : PreparedSpectrum
        Spectrum returned by ``prepare_spectrum`` with the same rebinning used
        for the candidate search.
    base, ylines : numpy.ndarray or None, default: None
        Optional precomputed baseline and baseline-subtracted excess spectrum.
    baseline_window, max_range_fraction, min_points
        Empirical-baseline parameters, used only when ``base``/``ylines``
        are not provided; they must then match the values used for the
        candidate search (e.g. pass ``config.baseline_window``).
        ``baseline_window`` may be a float, an array aligned with
        ``spectrum.energy``, or a callable ``f(energy)``.
    show_plot : bool, default: True
        Display the final diagnostic plot.
    output_dir : str, pathlib.Path, or None, default: None
        If provided, save ``global_fit_lines.csv`` and the diagnostic plot there.
    plot_name : str, default: "bliss_global_fit.png"
        Diagnostic plot filename inside ``output_dir``.
    save_csv : bool, default: True
        Save the fitted line table when ``output_dir`` is provided.
    final_fit_maxfev : int, default: 100000
        Maximum number of function evaluations in ``curve_fit``.
    response_feature_threshold : float, default: 5.0
        ARF sharpness-score threshold for flagging fitted lines; pass the same
        value as in the candidate-search config. The returned table and CSV
        include ``response_feature``. With no usable ARF, it is False
        (not flagged); the numeric feature score stays internal.
    max_sigma_line : float, default: 0.1
        Upper sigma bound for every line, in keV. Pass the candidate-search
        value (``config.max_sigma_line``) to use the same limit.
    return_yfit : bool, default: False
        If true, return ``(result, yfit)`` instead of only ``result``.
    energy_min, energy_max : float or None, default=None
        Minimum and maximum energy shown in the diagnostic plot.
        This only affects the plot, not the fitted energy range.
    size_fig_input : tuple or None, default=None
        Figure size passed to ``plot_global_fit``, e.g. ``(10, 5)``.

    Returns
    -------
    pandas.DataFrame or tuple
        Final fitted line table with exactly ``FINAL_OUTPUT_COLUMNS``,
        optionally with the fitted line-only model.
    """

    base, ylines = _baseline_and_line_excess(
        spectrum,
        base=base,
        ylines=ylines,
        baseline_window=baseline_window,
        max_range_fraction=max_range_fraction,
        min_points=min_points,
    )

    result, yfit = final_fit_and_metrics(
        spectrum=spectrum,
        clean_lines=pd_lines,
        base=base,
        ylines=ylines,
        final_fit_maxfev=final_fit_maxfev,
        response_feature_threshold=response_feature_threshold,
        max_sigma_line=max_sigma_line,
    )

    output_path = None

    if output_dir is not None:
        output_dir = ensure_output_folder(output_dir)
        output_path = output_dir / plot_name

        if save_csv:
            result.to_csv(
                output_dir / "global_fit_lines.csv",
                index=False,
            )

    if show_plot or output_path is not None:
        plot_global_fit(
            spectrum=spectrum,
            base=base,
            yfit=yfit,
            output_path=output_path,
            show_plot=show_plot,
            energy_min=energy_min,
            energy_max=energy_max,
            size_fig_input=size_fig_input,
        )

    if return_yfit:
        return result, yfit

    return result


class BlindLineSearchPipeline:
    """Run the blind emission-line search on one spectrum.

    By default, the pipeline now stops after candidate detection and bliss_score
    estimation. The expensive global multi-Gaussian fit can be run later with
    ``fit_global`` after the user filters the candidate table.

    Each run also compares five local statistics with maxima from the same
    null searches. ``n_sim`` records all generated realizations, including
    empty searches, and ``null_maxima`` retains their five maxima for inspection.
    """

    def __init__(self, config: Optional[BlindLineSearchConfig] = None):
        """Create a pipeline instance."""
        self.config = config or BlindLineSearchConfig()

    def run(
        self,
        pha,
        rmf,
        arf=None,
        bkg=None,
        *,
        en1: Optional[float] = None,
        en2: Optional[float] = None,
        energy_pad: Optional[float] = None,
        final_fit: bool = False,
        show_plot: bool = False,
        output_dir=None,
        plot_name: str = 'bliss_fit.png',
    ) -> pd.DataFrame:
        """Execute the BLiSS workflow on a PHA spectrum.

        ``pha`` and ``rmf`` are required; ``arf`` and ``bkg`` are optional.
        The spectrum is rebinned conserving counts according to the config and
        converted to counts s^-1 keV^-1. The empirical baseline is computed from the full input spectrum.
        Candidate detection, local Gaussian fitting, bliss_score estimation,
        and the optional global fit are performed inside the selected energy
        interval enlarged by ``energy_pad``. The returned catalogue is finally
        restricted to the nominal ``en1``--``en2`` interval. Both modes return
        and export exactly ``FINAL_OUTPUT_COLUMNS``.
        The five ``p_global_*`` columns describe the local detection search
        over the nominal interval and are preserved by an optional global fit.
        """

        output_dir = ensure_output_folder(output_dir)

        # ------------------------------------------------------------
        # Load full spectrum
        # ------------------------------------------------------------
        spectrum_full = prepare_spectrum(
            pha, rmf, arf, bkg,
            rebin_method=self.config.rebin_method,
            rebin_scale=self.config.rebin_scale,
            rebin_min_bins=self.config.rebin_min_bins,
        )
        self._active_spectrum = spectrum_full

        # ------------------------------------------------------------
        # Resolve nominal science window
        # ------------------------------------------------------------
        en1_use = self.config.en1 if en1 is None else en1
        en2_use = self.config.en2 if en2 is None else en2

        if en2_use <= en1_use:
            raise ValueError(
                f"Invalid energy range: en1={en1_use}, en2={en2_use}. "
                "Require en2 > en1."
            )

        # ------------------------------------------------------------
        # Resolve internal padding
        # ------------------------------------------------------------
        pad = self.config.energy_pad if energy_pad is None else energy_pad
        pad = max(float(pad), 0.0)

        fit_en1 = en1_use - pad
        fit_en2 = en2_use + pad

        fit_mask = (
            (spectrum_full.energy >= fit_en1)
            & (spectrum_full.energy <= fit_en2)
        )

        if not np.any(fit_mask):
            raise ValueError(
                f"No spectral bins found in the padded energy range: "
                f"{fit_en1}--{fit_en2}."
            )

        # ------------------------------------------------------------
        # Baseline from the full spectrum
        # ------------------------------------------------------------
        base_full, baseline_info = base_calculator(
            spectrum_full.energy,
            spectrum_full.values,
            baseline_window=self.config.baseline_window,
            max_range_fraction=self.config.max_range_fraction,
            min_points=self.config.min_points,
            return_info=True,
        )

        ylines_full = np.maximum(spectrum_full.values - base_full,0)
        # ------------------------------------------------------------
        # Restrict search/fit arrays to padded interval
        # ------------------------------------------------------------
        spectrum = _slice_prepared_spectrum(spectrum_full, fit_mask)
        base = base_full[fit_mask]
        ylines = ylines_full[fit_mask]

        # ------------------------------------------------------------
        # Raw candidate detection and first local Gaussian fits
        # inside the padded interval
        # ------------------------------------------------------------
        raw_candidates = return_raw_lines(
            spectrum.energy,
            spectrum.values,
            spectrum.uncertainties,
            ylines,
            base,
            response_sigma=spectrum.response_sigma,
            max_sigma_line=self.config.max_sigma_line,
        )

        # ------------------------------------------------------------
        # Null realizations: same rebinning, baseline and detection
        # chain as the data, one candidate table per realization
        # ------------------------------------------------------------
        null_realizations = generate_null_realizations(
            spectrum_full,
            base_full,
            fit_en1,
            fit_en2,
            self.config,
            window_width=baseline_info["window_width"],
        )

        synthetic_tables = []
        for k, null in enumerate(null_realizations):
            cand = return_raw_lines(
                null.energy,
                null.values,
                null.uncertainties,
                null.ylines,
                null.baseline,
                response_sigma=null.response_sigma,
                max_sigma_line=self.config.max_sigma_line,
            )
            cand["sim"] = k
            synthetic_tables.append(cand)

        synthetic_candidates = (
            pd.concat(synthetic_tables, ignore_index=True)
            if synthetic_tables
            else pd.DataFrame(columns=raw_candidates.columns)
        )


        self.synthetic_candidates = _add_candidate_metrics(synthetic_candidates)
        self.n_sim = len(null_realizations)

        candidates = eval_bliss_score_gmm(
            raw_candidates,
            synthetic_candidates,
            simx=spectrum.energy,
            x=spectrum.energy,
            n_sim=self.n_sim,
            min_area_snr=self.config.min_area_snr,
        )
        # ------------------------------------------------------------
        # Final catalogue restricted to the nominal science window
        # ------------------------------------------------------------
        selected = self._select_candidates(
            candidates,
            en1=en1_use,
            en2=en2_use,
        )
        selected, self.null_maxima = calculate_look_elsewhere_pvalues(
            selected, self.synthetic_candidates, self.n_sim,
            en1=en1_use, en2=en2_use,
        )

        if not final_fit:
            result = selected.reindex(columns=FINAL_OUTPUT_COLUMNS).copy()
            self._write_candidate_outputs(result, output_dir)
            return result

        # ------------------------------------------------------------
        # Optional global fit over the padded interval,
        # but only for candidates whose centroids are inside en1--en2
        # ------------------------------------------------------------
        result, yfit = _fit_global_with_diagnostics(
            spectrum=spectrum,
            base=base,
            ylines=ylines,
            clean_lines=selected,
            final_fit_maxfev=self.config.final_fit_maxfev,
            response_feature_threshold=self.config.response_feature_threshold,
            max_sigma_line=self.config.max_sigma_line,
        )

        # Safety: keep only nominal-window lines after final fitting too
        result = self._select_energy_window(
            result,
            en1=en1_use,
            en2=en2_use,
        )

        if show_plot:
            plot_global_fit(
                spectrum=spectrum,
                base=base,
                yfit=yfit,
                output_path=output_dir / plot_name,
                show_plot=True,
            )

        result = result.reindex(columns=FINAL_OUTPUT_COLUMNS).copy()
        self._write_outputs(result, output_dir)
        return result

    @staticmethod
    def _select_energy_window(candidates: pd.DataFrame, *, en1: float, en2: float) -> pd.DataFrame:
        """Restrict rows without recomputing fitted metrics or dropping context."""
        clean_lines = candidates.copy()
        if clean_lines.empty:
            return clean_lines.reset_index(drop=True)
        selection_center = clean_lines.center.copy()
        if 'fit_center_initial' in clean_lines:
            selection_center = selection_center.where(
                np.isfinite(selection_center), clean_lines['fit_center_initial'])
        # Keep unlocalizable failed rows as diagnostics rather than losing them.
        return clean_lines[
            selection_center.between(en1, en2) | ~np.isfinite(selection_center)
        ].reset_index(drop=True)

    def _select_candidates(self, candidates: pd.DataFrame, *, en1: float, en2: float) -> pd.DataFrame:
        """Select local candidates and compute the metrics needed for refitting."""
        clean_lines = self._select_energy_window(candidates, en1=en1, en2=en2)
        clean_lines = _add_candidate_metrics(clean_lines)
        clean_lines = _flag_response_features(
            clean_lines,
            self._active_spectrum,
            self.config.response_feature_threshold,
        )

        # Preserve the score assigned during the observed/null comparison.
        if 'bliss_score' not in clean_lines.columns:
            clean_lines['bliss_score'] = np.nan

        for col in CANDIDATE_COLUMNS:
            if col not in clean_lines.columns:
                clean_lines[col] = np.nan

        clean_lines = clean_lines[CANDIDATE_COLUMNS]
        return clean_lines

    def _final_fit_and_metrics(
        self,
        spectrum: PreparedSpectrum,
        base: np.ndarray,
        ylines: np.ndarray,
        clean_lines: pd.DataFrame,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Backward-compatible wrapper around the standalone final-fit function."""
        return final_fit_and_metrics(
            spectrum=spectrum,
            clean_lines=clean_lines,
            base=base,
            ylines=ylines,
            final_fit_maxfev=self.config.final_fit_maxfev,
            response_feature_threshold=self.config.response_feature_threshold,
            max_sigma_line=self.config.max_sigma_line,
        )

    def _plot_final_fit(
        self,
        spectrum: PreparedSpectrum,
        base: np.ndarray,
        yfit: np.ndarray,
        output_path: str | Path,
    ) -> None:
        """Backward-compatible wrapper around the standalone plotting function."""
        plot_global_fit(
            spectrum=spectrum,
            base=base,
            yfit=yfit,
            output_path=output_path,
            show_plot=True,
        )

    def _write_candidate_outputs(self, candidates: pd.DataFrame, output_dir: Path) -> None:
        """Write the pre-global-fit candidate table and run summary."""
        candidates.to_csv(output_dir / 'candidate_lines.csv', index=False)
        with open(output_dir / 'run_summary.txt', 'w') as handle:
            handle.write('BLiSS candidate search completed\n')
            handle.write(f'Results folder: {output_dir}\n')
            handle.write(f'Number of candidate lines before global fit: {len(candidates)}\n')
            handle.write(f'Null simulations: {self.n_sim}; p_global_* use local-search maxima.\n')
            handle.write('Run fit_global(candidate_lines, spectrum) after user filtering to perform the global fit.\n')

    def _write_outputs(self, result: pd.DataFrame, output_dir: Path) -> None:
        """Write the final candidate table and run summary."""
        result.to_csv(output_dir / 'candidate_lines_global_fit.csv', index=False)
        with open(output_dir / 'run_summary.txt', 'w') as handle:
            handle.write('BLiSS run completed with global fit\n')
            handle.write(f'Results folder: {output_dir}\n')
            handle.write(f'Number of fitted candidates: {len(result)}\n')
            handle.write(f'Null simulations: {self.n_sim}; p_global_* retain local-search p-values.\n')


def _build_config(
    config, en1, en2, energy_pad, rebin_method, rebin_scale, rebin_min_bins,
    num_synthetic_simulations, synthetic_seed,
):
    if config is None:
        config = BlindLineSearchConfig()
    config = replace(
        config,
        en1=en1, en2=en2, energy_pad=energy_pad,
        rebin_method=rebin_method, rebin_scale=rebin_scale,
        rebin_min_bins=rebin_min_bins,
    )
    if num_synthetic_simulations is not None:
        config = replace(config, num_synthetic_simulations=num_synthetic_simulations)
    if synthetic_seed is not None:
        config = replace(config, synthetic_seed=synthetic_seed)
    return config


def find_candidate_lines(
    pha,
    rmf,
    arf=None,
    bkg=None,
    *,
    en1=0,
    en2=10,
    energy_pad=0.0,
    output_dir=None,
    rebin_method="none",
    rebin_scale=None,
    rebin_min_bins=1,
    num_synthetic_simulations=None,
    synthetic_seed=None,
    config: Optional[BlindLineSearchConfig] = None,
):
    """Run BLiSS up to candidate detection and bliss_score estimation.

    Parameters
    ----------
    pha, rmf : path-like
        Source spectrum and its redistribution matrix (required).
    arf, bkg : path-like or None
        Optional effective-area file and background spectrum.
    en1, en2, energy_pad : float
        Search interval and internal padding.
    rebin_method, rebin_scale, rebin_min_bins
        Count-conserving rebinning applied to the data and to every synthetic
        null realization (see ``rebin_counts``).
    num_synthetic_simulations, synthetic_seed
        Number of null realizations and random seed.
    config : BlindLineSearchConfig or None
        Full configuration; the keywords above override its values.

    Returns
    -------
    pandas.DataFrame
        Local candidate catalogue with exactly ``FINAL_OUTPUT_COLUMNS``.
        The same columns are saved to ``candidate_lines.csv``.
    """
    config = _build_config(
        config, en1, en2, energy_pad, rebin_method, rebin_scale, rebin_min_bins,
        num_synthetic_simulations, synthetic_seed,
    )
    pipeline = BlindLineSearchPipeline(config=config)
    return pipeline.run(pha, rmf, arf, bkg, final_fit=False, output_dir=output_dir)


def find_emission_lines(
    pha,
    rmf,
    arf=None,
    bkg=None,
    *,
    en1=0,
    en2=10,
    energy_pad=0.0,
    show_plot=False,
    output_dir=None,
    plot_name='bliss_fit.png',
    final_fit: bool = False,
    rebin_method="none",
    rebin_scale=None,
    rebin_min_bins=1,
    num_synthetic_simulations=None,
    synthetic_seed=None,
    config: Optional[BlindLineSearchConfig] = None,
):
    """Run the default BLiSS emission-line search (see ``find_candidate_lines``
    for the input options)."""
    config = _build_config(
        config, en1, en2, energy_pad, rebin_method, rebin_scale, rebin_min_bins,
        num_synthetic_simulations, synthetic_seed,
    )
    pipeline = BlindLineSearchPipeline(config=config)
    return pipeline.run(
        pha, rmf, arf, bkg,
        final_fit=final_fit,
        show_plot=show_plot,
        output_dir=output_dir,
        plot_name=plot_name,
    )
