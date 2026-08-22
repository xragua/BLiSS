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
from .candidate_regions import return_raw_lines
from .gaussian_models import n_gaussian, p0_generator_final
from ..synthetic_probability.synthetic_spectra import generate_null_realizations
from ..synthetic_probability.gmm_probability import eval_line_probability_gmm
from ..plotting.run_output_manager import ensure_output_folder
from ..spectrum_data.fits_spectrum_loader import (
    load_fits_spectrum,
    read_pha_metadata,
    align_to_spectrum,
)
from ..spectrum_data.rebinning_tools import rebin_counts

LINE_OUTPUT_COLUMNS = [
    'center', 'ecenter', 'sigma', 'esigma', 'amplitude', 'eamplitude',
    'relative_power', 'noise_on_block', 'value_on_line',
    'base_on_line', 'snr_peak', 'snr_amplitude', 'area', 'earea',
    'snr_area', 'ew', 'cluster_probability',] #'response_feature', 'response_feature_score']

CANDIDATE_COLUMNS = LINE_OUTPUT_COLUMNS.copy()
FINAL_OUTPUT_COLUMNS = [
    'center', 'ecenter', 'sigma', 'esigma', 'amplitude', 'eamplitude',
    'relative_power', 'noise_on_block', 'value_on_line',
    'base_on_line', 'snr_peak', 'snr_amplitude', 'area', 'earea',
    'snr_area', 'ew', 'cluster_probability',] #'response_feature', 'response_feature_score', 'fit_error_flag']


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
        Number of synthetic spectra generated for probability estimation.
    synthetic_seed : int or None
        Random seed used for synthetic-spectrum generation.

    final_fit_maxfev : int
        Maximum number of function evaluations allowed in the final
        ``curve_fit``.

    snr_confidence_threshold : float
        Any available S/N diagnostic above which a candidate is assigned
        probability 1.

    response_feature_threshold : float
        Robust score above which unusually sharp ARF effective-area
        structure is flagged.

    max_sigma_line : float
        Maximum Gaussian sigma allowed for an individual line candidate,
        in keV.

    noise_model : {'poisson', 'gaussian'}
        Statistical model of the synthetic null spectra. ``'poisson'`` draws
        counts at channel resolution (default); ``'gaussian'`` is kept only
        for validation/comparison and should not be used for science results.

    rebin_method, rebin_scale, rebin_min_bins
        Count-conserving rebinning applied to the native spectrum before the
        search and to every synthetic null realization (see ``rebin_counts``).

    baseline_window : float
        Preferred physical width, in keV, of the running-median window
        used to estimate the empirical baseline.

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
    snr_confidence_threshold: float = 1e100
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
    lines['response_feature_score'] = np.nan
    if len(lines) == 0:
        return lines

    arf_energy, score = _arf_feature_profile(spectrum.arf_energy, spectrum.effective_area)
    if arf_energy is None:
        return lines

    arf_step = np.nanmedian(np.diff(arf_energy)) if len(arf_energy) > 1 else 0.0
    for idx, center in lines['center'].items():
        center = float(center)
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
        lines.at[idx, 'response_feature_score'] = local_score
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


def _snr_confidence_mask(
    lines: pd.DataFrame,
    threshold: float,
    *,
    columns=('snr_peak', 'snr_area', 'snr_amplitude'),
) -> pd.Series:
    """Return rows where any available S/N diagnostic exceeds ``threshold``."""
    if len(lines) == 0:
        return pd.Series([], index=lines.index, dtype=bool)

    high_snr = pd.Series(False, index=lines.index, dtype=bool)
    for col in columns:
        if col in lines.columns:
            high_snr |= pd.to_numeric(lines[col], errors='coerce') >= threshold
    return high_snr


def _add_candidate_metrics(lines: pd.DataFrame) -> pd.DataFrame:
    """Add pre-global-fit diagnostics to candidate lines.

    The equivalent width reported here is an approximate candidate EW, computed
    from the local Gaussian area divided by the empirical baseline at the line
    centroid. If the spectral coordinate is in keV, the returned EW is in eV.
    The global-fit EW is recomputed later from the final multi-Gaussian model.
    """
    lines = lines.copy()

    if len(lines) == 0:
        for col in ['snr_peak', 'snr_amplitude', 'area', 'earea', 'snr_area', 'ew']:
            if col not in lines.columns:
                lines[col] = []
        return lines

    numeric_cols = [
        'amplitude', 'eamplitude', 'sigma', 'esigma',
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
    lines['earea'] = np.sqrt(
        (lines['sigma'] * k * lines['eamplitude']) ** 2
        +
        (lines['amplitude'] * k * lines['esigma']) ** 2
    )
    lines['snr_area'] = _safe_divide(lines['area'], lines['earea'])

    # Approximate candidate equivalent width. Assumes energy in keV, hence x1000 -> eV.
    lines['ew'] = _safe_divide(lines['area'], lines['base_on_line']) * 1000.0

    return lines.replace([np.inf, -np.inf], np.nan)


def _ensure_line_context(
    clean_lines: pd.DataFrame,
    spectrum: PreparedSpectrum,
    base: np.ndarray,
    *,
    snr_confidence_threshold: float = 4.0,
) -> pd.DataFrame:
    """Ensure that user-filtered candidate tables contain final-fit context columns."""
    clean_lines = clean_lines.copy().reset_index(drop=True)

    if len(clean_lines) == 0:
        for col in CANDIDATE_COLUMNS:
            if col not in clean_lines.columns:
                clean_lines[col] = []
        return clean_lines

    required_for_fit = {'amplitude', 'center', 'sigma'}
    missing_for_fit = sorted(required_for_fit - set(clean_lines.columns))
    if missing_for_fit:
        raise ValueError(
            'pd_lines must contain amplitude, center, and sigma columns. '
            f'Missing: {missing_for_fit}'
        )

    nearest_positions = [
        int(np.argmin(np.abs(spectrum.energy - center)))
        for center in clean_lines['center'].to_numpy(dtype=float)
    ]

    if 'base_on_line' not in clean_lines.columns:
        clean_lines['base_on_line'] = [base[pos] for pos in nearest_positions]
    if 'value_on_line' not in clean_lines.columns:
        clean_lines['value_on_line'] = [spectrum.values[pos] for pos in nearest_positions]
    if 'noise_on_block' not in clean_lines.columns:
        clean_lines['noise_on_block'] = [spectrum.uncertainties[pos] for pos in nearest_positions]
    if 'relative_power' not in clean_lines.columns:
        denom = clean_lines['value_on_line'] + clean_lines['base_on_line']
        clean_lines['relative_power'] = np.where(
            denom != 0,
            (clean_lines['value_on_line'] - clean_lines['base_on_line']) / denom,
            np.nan,
        )
    # Recompute local candidate diagnostics if the user passed a minimal
    # table without the convenience S/N columns. The global fit will later
    # overwrite the fit-dependent quantities using the final covariance.
    if (
        'snr_peak' not in clean_lines.columns
        or 'snr_area' not in clean_lines.columns
        or 'snr_amplitude' not in clean_lines.columns
    ):
        clean_lines = _add_candidate_metrics(clean_lines)

    if 'cluster_probability' not in clean_lines.columns:
        clean_lines['cluster_probability'] = np.nan

    high_snr = _snr_confidence_mask(clean_lines, snr_confidence_threshold)
    clean_lines.loc[high_snr, 'cluster_probability'] = 1.0

    for col in CANDIDATE_COLUMNS:
        if col not in clean_lines.columns:
            clean_lines[col] = np.nan

    return clean_lines


def final_fit_and_metrics(
    spectrum: PreparedSpectrum,
    clean_lines: pd.DataFrame,
    base: Optional[np.ndarray] = None,
    ylines: Optional[np.ndarray] = None,
    final_fit_maxfev: int = 100000,
    snr_confidence_threshold: float = 4.0,
):
    """Refit selected candidates and compute final line diagnostics."""

    base, ylines = _baseline_and_line_excess(
        spectrum,
        base=base,
        ylines=ylines,
    )

    clean_lines = _ensure_line_context(
        clean_lines,
        spectrum=spectrum,
        base=base,
        snr_confidence_threshold=snr_confidence_threshold,
    )

    fitted_final = pd.DataFrame(
        columns=[
            "amplitude",
            "center",
            "sigma",
            "eamplitude",
            "ecenter",
            "esigma",
        ]
    )

    yfit = np.zeros_like(spectrum.energy, dtype=float)

    if len(clean_lines) > 0:
        p0, bounds = p0_generator_final(
            spectrum.energy,
            spectrum.values,
            clean_lines,
            response_sigma=spectrum.response_sigma,
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
                n_gaussian,
                spectrum.energy,
                ylines,
                p0=p0,
                bounds=bounds,
                sigma=fit_sigma,
                absolute_sigma=True,
                maxfev=final_fit_maxfev,
            )

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
                    ]
                )
                fitted_final.loc[k] = new_line

        except RuntimeError as exc:
            print(f"Error final fitting: {exc}")

        except ValueError as exc:
            print(f"Error final fitting: {exc}")

    clean_select = clean_lines[
        [
            "relative_power",
            "noise_on_block",
            "value_on_line",
            "base_on_line",
            "cluster_probability",
        ]
    ]

    result = pd.concat(
        [
            fitted_final,
            clean_select,
        ],
        axis=1,
    )

    if len(result) == 0:
        return pd.DataFrame(columns=FINAL_OUTPUT_COLUMNS), yfit

    cols = [
        "amplitude",
        "center",
        "sigma",
        "eamplitude",
        "ecenter",
        "esigma",
    ]

    result[cols] = result[cols].apply(
        pd.to_numeric,
        errors="coerce",
    )

    result["noise_on_block"] = pd.to_numeric(
        result["noise_on_block"],
        errors="coerce",
    )

    result["snr_peak"] = _safe_divide(
        result["amplitude"],
        result["noise_on_block"],
    )

    result["snr_amplitude"] = _safe_divide(
        result["amplitude"],
        result["eamplitude"],
    )

    k = np.sqrt(2.0 * np.pi)

    result["area"] = (
        result["amplitude"]
        * result["sigma"]
        * k
    )

    result["earea"] = np.sqrt(
        (result["sigma"] * k * result["eamplitude"]) ** 2
        +
        (result["amplitude"] * k * result["esigma"]) ** 2
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
                    yfit[valid_mask]
                    / base[valid_mask]
                    * spectrum.bin_width[valid_mask]
                )
                * 1000.0
            )
        else:
            ew = np.nan

        ew_vals.append(ew)

    result["ew"] = ew_vals

    # --------------------------------------------------------
    # Reliability flags for covariance-derived uncertainties
    # --------------------------------------------------------

    result["fit_error_flag"] = ""

    center_bound_width = 0.2
    sigma_bound_width = clean_lines["sigma"].to_numpy(dtype=float) + 0.01

    bad_center_error = (
        ~np.isfinite(result["ecenter"])
        | (result["ecenter"] > center_bound_width)
    )

    bad_sigma_error = (
        ~np.isfinite(result["esigma"])
        | (result["esigma"] > sigma_bound_width)
    )

    bad_amplitude_error = (
        ~np.isfinite(result["eamplitude"])
        | (result["eamplitude"] > np.abs(result["amplitude"]))
    )

    bad_error = (
        bad_center_error
        | bad_sigma_error
        | bad_amplitude_error
    )

    result.loc[bad_error, "fit_error_flag"] = "unconstrained"

    result = result[FINAL_OUTPUT_COLUMNS]

    high_snr = _snr_confidence_mask(
        result,
        snr_confidence_threshold,
    )

    result.loc[high_snr, "cluster_probability"] = 1.0

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


def add_look_elsewhere_p(
    lines: pd.DataFrame,
    synthetic_candidates: pd.DataFrame,
    n_sim: int,
    column: str = "amplitude",
) -> pd.DataFrame:
    """Add a Monte Carlo global (look-elsewhere) p-value for each line.

    For every line, count the null realizations in which at least one
    synthetic candidate anywhere in the search interval has ``column``
    greater than or equal to the line's value. Realizations with no
    candidate at all count as non-exceeding. The p-value uses the standard
    Monte Carlo estimator ``(k + 1) / (n_sim + 1)``, so it is never exactly
    zero; its resolution is set by ``n_sim``.

    Parameters
    ----------
    lines : pandas.DataFrame
        Candidate or fitted line table containing ``column``.
    synthetic_candidates : pandas.DataFrame
        Synthetic candidate table with a ``sim`` column
        (``BlindLineSearchPipeline.synthetic_candidates``).
    n_sim : int
        Number of null realizations generated
        (``BlindLineSearchPipeline.n_sim``).
    column : str, default "amplitude"
        Statistic compared, e.g. ``"amplitude"``, ``"area"``, ``"snr_peak"``,
        ``"snr_area"``.

    Returns
    -------
    pandas.DataFrame
        Copy of ``lines`` with ``n_exceed_<column>`` and ``p_global_<column>``.
    """
    lines = lines.copy()
    if column not in lines.columns:
        raise ValueError(f"'{column}' is not a column of the line table.")

    synth = synthetic_candidates
    if synth is None or len(synth) == 0:
        per_sim_max = np.array([], dtype=float)
    else:
        if column not in synth.columns:
            synth = _add_candidate_metrics(synth)
        if "sim" not in synth.columns:
            raise ValueError("synthetic_candidates must have a 'sim' column.")
        per_sim_max = synth.groupby("sim")[column].max().to_numpy(dtype=float)
        per_sim_max = per_sim_max[np.isfinite(per_sim_max)]

    values = pd.to_numeric(lines[column], errors="coerce").to_numpy(dtype=float)
    k = np.array([np.sum(per_sim_max >= v) if np.isfinite(v) else n_sim for v in values])

    lines[f"n_exceed_{column}"] = k
    lines[f"p_global_{column}"] = (k + 1) / (n_sim + 1)
    return lines



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
    snr_confidence_threshold: float = 4.0,
    return_yfit: bool = False,
    energy_min: float | None = None,
    energy_max: float | None = None,
    size_fig_input: tuple[float, float] | None = None,
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
    snr_confidence_threshold : float, default: 4.0
        S/N threshold above which ``cluster_probability`` is set to 1 if any of
        ``snr_peak``, ``snr_area`` or ``snr_amplitude`` exceeds it.
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
        Final fitted line table, optionally with the fitted line-only model.
    """

    base, ylines = _baseline_and_line_excess(
        spectrum,
        base=base,
        ylines=ylines,
    )

    result, yfit = final_fit_and_metrics(
        spectrum=spectrum,
        clean_lines=pd_lines,
        base=base,
        ylines=ylines,
        final_fit_maxfev=final_fit_maxfev,
        snr_confidence_threshold=snr_confidence_threshold,
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

    By default, the pipeline now stops after candidate detection and probability
    estimation. The expensive global multi-Gaussian fit can be run later with
    ``fit_global`` after the user filters the candidate table.
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
        Candidate detection, local Gaussian fitting, probability estimation,
        and the optional global fit are performed inside the selected energy
        interval enlarged by ``energy_pad``. The returned catalogue is finally
        restricted to the nominal ``en1``--``en2`` interval.
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
        base_full = base_calculator(
            spectrum_full.energy,
            spectrum_full.values,
            baseline_window=self.config.baseline_window,
            max_range_fraction=self.config.max_range_fraction,
            min_points=self.config.min_points,
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
            )
            cand["sim"] = k
            synthetic_tables.append(cand)

        synthetic_candidates = (
            pd.concat(synthetic_tables, ignore_index=True)
            if synthetic_tables
            else pd.DataFrame(columns=raw_candidates.columns)
        )


        self.synthetic_candidates = _add_candidate_metrics(synthetic_candidates)
        self.n_sim = max(1, len(null_realizations))  

        candidates = eval_line_probability_gmm(
            raw_candidates,
            synthetic_candidates,
            simx=spectrum.energy,
            x=spectrum.energy,
            n_sim=max(1, len(null_realizations)),
        )
        # ------------------------------------------------------------
        # Final catalogue restricted to the nominal science window
        # ------------------------------------------------------------
        selected = self._select_candidates(
            candidates,
            en1=en1_use,
            en2=en2_use,
        )

        if not final_fit:
            self._write_candidate_outputs(selected, output_dir)
            return selected

        # ------------------------------------------------------------
        # Optional global fit over the padded interval,
        # but only for candidates whose centroids are inside en1--en2
        # ------------------------------------------------------------
        result, yfit = final_fit_and_metrics(
            spectrum=spectrum,
            base=base,
            ylines=ylines,
            clean_lines=selected,
            final_fit_maxfev=self.config.final_fit_maxfev,
            snr_confidence_threshold=self.config.snr_confidence_threshold,
        )

        # Safety: keep only nominal-window lines after final fitting too
        result = self._select_candidates(
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

        self._write_outputs(result, output_dir)
        return result
    def _select_candidates(self, candidates: pd.DataFrame, *, en1: float, en2: float) -> pd.DataFrame:
        """Restrict candidate lines to the requested energy interval."""
        clean_lines = candidates.copy()
        clean_lines = clean_lines[
            (clean_lines.center >= en1) & (clean_lines.center <= en2)
        ].reset_index(drop=True)
        clean_lines = _add_candidate_metrics(clean_lines)
        clean_lines = _flag_response_features(
            clean_lines,
            self._active_spectrum,
            self.config.response_feature_threshold,
        )

        # Before the optional expensive global fit, force very significant
        # candidates to probability 1 if any S/N diagnostic is high. This
        # makes the returned clean_lines table consistent with the final-fit output.
        if 'cluster_probability' not in clean_lines.columns:
            clean_lines['cluster_probability'] = np.nan
        high_snr = _snr_confidence_mask(clean_lines, self.config.snr_confidence_threshold)
        clean_lines.loc[high_snr, 'cluster_probability'] = 1.0

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
            snr_confidence_threshold=self.config.snr_confidence_threshold,
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
            handle.write('Run fit_global(candidate_lines, spectrum) after user filtering to perform the global fit.\n')

    def _write_outputs(self, result: pd.DataFrame, output_dir: Path) -> None:
        """Write the final candidate table and run summary."""
        result.to_csv(output_dir / 'candidate_lines_global_fit.csv', index=False)
        with open(output_dir / 'run_summary.txt', 'w') as handle:
            handle.write('BLiSS run completed with global fit\n')
            handle.write(f'Results folder: {output_dir}\n')
            handle.write(f'Number of fitted candidates: {len(result)}\n')


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
    """Run BLiSS up to candidate detection and probability estimation.

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
