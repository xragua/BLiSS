"""High-level BLiSS emission-line search pipeline."""
from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from .empirical_baseline import base_calculator, poisson_base_calculator
from .poisson_fitting import (
    PoissonFitData, fit_poisson_gaussians, gaussian_count_model, validate_poisson_groups,
)
from .fit_quality import (
    annotate_fit_quality, FIT_QUALITY_COLUMNS, center_error_exceeds_limit,
    discard_large_center_errors,
)
from .candidate_regions import (
    return_raw_lines, DEFAULT_MIN_RELATIVE_PROMINENCE,
    validate_min_relative_prominence,
)
from .gaussian_models import n_gaussian, p0_generator_final, gaussian_area_error, validate_sigma_bounds
from .instrumental_width import fit_with_instrumental_width
from ..synthetic_probability.synthetic_spectra import generate_null_realizations
from ..synthetic_probability.parallel_search import (
    iter_parallel_null_candidates, validate_parallel_options,
)
from ..synthetic_probability.look_elsewhere import (
    LOOK_ELSEWHERE_STATISTICS, P_VALUE_COLUMNS, single_simulation_maxima,
    calculate_look_elsewhere_pvalues_from_maxima,
)
from ..plotting.run_output_manager import (
    ensure_output_folder, _check_output_paths, _reserve_output_paths, _plot_output_path,
)
from ..spectrum_data.fits_spectrum_loader import (
    load_fits_spectrum,
    read_pha_metadata,
    align_to_spectrum,
)
from ..spectrum_data.rebinning_tools import (
    rebin_counts, OPTIMAL_METHODS, COUNT_METHODS, OptimalBinningResponse, optimal_response_from_rmf,
    validate_oversampling,
)

# Full working schema: retain covariance, fit context and quality diagnostics
# until all fitting and energy-window selection have finished.
LINE_OUTPUT_COLUMNS = [
    'center', 'ecenter', 'sigma', 'esigma', 'width_fixed_instrumental', 'amplitude', 'eamplitude',
    'cov_amplitude_sigma', 'fit_block_width',
    'relative_power', 'noise_on_block',
    'base_on_line', 'snr_peak', 'snr_amplitude', 'area', 'earea',
    'snr_area', 'ew',
    'response_feature',
    *FIT_QUALITY_COLUMNS, *P_VALUE_COLUMNS,]

CANDIDATE_COLUMNS = LINE_OUTPUT_COLUMNS.copy()
FINAL_OUTPUT_COLUMNS = [
    'center', 'ecenter',
    'sigma', 'esigma',
    'width_fixed_instrumental',
    'amplitude', 'eamplitude',
    'area', 'earea',
    'ew',
    'relative_power',
    'snr_peak',
    'snr_area',
    'response_feature',
    *P_VALUE_COLUMNS,
]


def _validate_fit_statistic(value):
    if value not in ('auto', 'poisson', 'gaussian'):
        raise ValueError("fit_statistic must be 'auto', 'poisson' or 'gaussian'.")


@dataclass
class BlindLineSearchConfig:
    """Configuration values controlling the default BLiSS pipeline.

    Attributes
    ----------
    en1, en2 : float
        Lower and upper search limits in keV (defaults 1.7 and 12.0).
        Both are editable; no instrument-specific hard limit is imposed.
    energy_pad : float
        Extra energy range added around the requested interval during
        candidate detection and fitting.

    num_synthetic_simulations : int
        Nonnegative number of synthetic spectra for the five
        look-elsewhere p-values. Zero disables simulation-based significance.
    synthetic_seed : int or None
        Random seed used for synthetic-spectrum generation.
    n_jobs : int
        Maximum worker processes for null searches. Default 1 runs serially.
        Spawned workers each use one BLAS/OpenMP thread.
    simulation_batch_size : int
        Simulations per worker task, default 5. Both parallel options must
        be positive integers. Changing them preserves the random draws.
    keep_synthetic_candidates : bool, default: False
        Retain the complete null candidate catalogue for inspection. By
        default, each table is reduced to five maxima and then released.
        Both modes use the same maxima for p-values and count empty searches.

    fit_statistic : {'auto', 'poisson', 'gaussian'}, default 'auto'
        Auto uses Poisson likelihood for raw integer counts without supplied
        error models; other inputs retain Gaussian weighted fits. Explicit
        poisson requires compatible source and background counts. Gaussian
        validation noise selects Gaussian fits in auto mode. Poisson fits
        integrate lines over bins, profile any Poisson background, and use
        a count-based continuum that conservatively masks strong emission
        only while estimating the mean. Its uncertainty is not included in
        the formal line covariance. Gaussian fits retain the median baseline.

    final_fit_maxfev : int
        Maximum number of function evaluations allowed in the final fit.

    response_feature_threshold : float
        Robust score above which unusually sharp ARF effective-area
        structure is flagged.

    min_sigma_line : float, default: 1e-6
        Absolute lower sigma bound in keV. The effective lower bound is
        max(min_sigma_line, 0.1 * instrumental_sigma) where available.
    max_sigma_line : float
        Maximum Gaussian sigma allowed for an individual line candidate,
        in keV. Applied to local data/null fits and the optional global fit.

    min_relative_prominence : float
        Peaks must exceed this fraction of the largest prominence in their
        candidate block. Default 0.05; finite values in [0, 1) are allowed.
        Applied identically to observed and null searches, including workers.
        This selects fit seeds, not statistical significance.

    noise_model : {'poisson', 'gaussian'}
        Statistical model of the synthetic null spectra. ``'poisson'`` draws
        counts at channel resolution (default); ``'gaussian'`` is kept only
        for validation/comparison and should not be used for science results.

    assume_poisson_rate_errors : bool, default: False
        Allow RATE/COUNT_RATE without explicit errors only when True. Missing
        errors are estimated as sqrt(rate * exposure) / exposure, assuming raw
        Poisson counts divided by exposure. Applies to source and background
        loading, including native information used by null simulations.
        Supplied STAT_ERR, ERROR or ERR values are always preserved.

    rebin_method, rebin_scale, rebin_min_bins
        Count-conserving rebinning applied to the native spectrum before the
        search and to every synthetic null realization (see ``rebin_counts``).

    oversampling : float or None
        Optional maximum sampling factor for optimal grouping. The effective
        size is max(optimal size, FWHM / oversampling). None uses pure optimal
        grouping, which is the default rebinning method.

    baseline_window : float, array-like, or callable
        Preferred physical width, in keV, of the continuum-estimation window
        used to estimate the empirical baseline. A callable ``f(energy)``
        is evaluated on the rebinned working grid (science band plus context).
        A static array must match that grid. Initial capped widths are
        interpolated onto each null grid. Each grid locally expands windows
        as needed to reach min_points, respecting the same physical cap.

    max_range_fraction : float
        Maximum baseline-window width as a fraction of the requested search
        span including energy_pad. Resolved before grouping and kept fixed
        in all simulations. If min_points cannot be reached within this limit
        in the working band, the analysis raises ValueError.

    baseline_context : (float, float) or None
        Working energy limits in keV. None expands the padded search interval
        by half the maximum baseline-window width on each side. Explicit
        limits may extend this automatic context and must contain it. Native
        channels are selected before rebinning, identically for data and nulls.
        Channels outside this domain do not determine groups or the continuum.

    min_points : int
        Minimum number of finite samples required inside a local
        continuum-estimation window, including reflected edge samples. Windows
        expand locally to reach this minimum, up to max_range_fraction.
    """

    en1: float = 1.7
    en2: float = 12.0
    energy_pad: float = 0.1

    num_synthetic_simulations: int = 10
    synthetic_seed: Optional[int] = None

    final_fit_maxfev: int = 100000
    response_feature_threshold: float = 5.0

    # Line fitting
    max_sigma_line: float = 0.1
    min_sigma_line: float = 1e-6

    # Empirical baseline
    baseline_window: float = 0.4
    max_range_fraction: float = 0.2
    min_points: int = 3

    # Synthetic noise
    noise_model: str = "poisson"       # 'poisson' (default) | 'gaussian' (validation only)

    # Rebinning applied to count spectra (and to every null realization)
    rebin_method: str = "optimal"          # 'none' | 'bins' | 'snr' | 'counts' | 'resolution' | 'optimal'
    rebin_scale: Optional[float] = None
    rebin_min_bins: int = 1

    n_jobs: int = 1
    simulation_batch_size: int = 5
    min_relative_prominence: float = DEFAULT_MIN_RELATIVE_PROMINENCE
    oversampling: Optional[float] = None
    assume_poisson_rate_errors: bool = False
    keep_synthetic_candidates: bool = False
    fit_statistic: str = "auto"
    baseline_context: Optional[tuple[float, float]] = None

    def __post_init__(self):
        _validate_fit_statistic(self.fit_statistic)
        validate_oversampling(self.oversampling, self.rebin_method)
        validate_sigma_bounds(self.min_sigma_line, self.max_sigma_line)
        validate_parallel_options(self.n_jobs, self.simulation_batch_size)
        validate_min_relative_prominence(self.min_relative_prominence)
        if not isinstance(self.keep_synthetic_candidates, (bool, np.bool_)):
            raise ValueError('keep_synthetic_candidates must be a boolean.')




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
        Observational error estimates sorted with the spectrum. In Poisson
        mode these are raw sqrt(count variance) estimates, possibly zero;
        they are neither likelihood weights nor confidence intervals.
    bin_width : numpy.ndarray
        Width of each spectral bin, used when estimating equivalent width.
    response_sigma : numpy.ndarray or None
        Instrumental Gaussian-equivalent sigma aligned with the spectral grid.
    arf_energy : numpy.ndarray or None
        Energy grid of the associated ARF response.
    effective_area : numpy.ndarray or None
        Effective area of the associated ARF response.
    baseline_max_window_width : float or None
        Physical continuum-window cap set by the pipeline before grouping.
        Nulls inherit it along with the complete native working domain.
        None retains the standalone convention based on the input-grid span.
    """
    energy: np.ndarray
    values: np.ndarray
    uncertainties: np.ndarray
    bin_width: np.ndarray
    response_sigma: Optional[np.ndarray] = None
    arf_energy: Optional[np.ndarray] = None
    effective_area: Optional[np.ndarray] = None
    native: Optional["NativeCounts"] = None
    poisson_data: Optional[PoissonFitData] = None
    baseline_max_window_width: Optional[float] = None

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
    optimal_response : OptimalBinningResponse or None
        RMF quantities reused to repeat optimal grouping for each null spectrum.
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
    optimal_response: Optional[OptimalBinningResponse] = None


def prepare_spectrum(
    pha_path,
    rmf_path,
    arf_path=None,
    background_path=None,
    *,
    rebin_method: str = "optimal",
    rebin_scale: Optional[float] = None,
    rebin_min_bins: int = 1,
    oversampling: Optional[float] = None,
    assume_poisson_rate_errors: bool = False,
    fit_statistic: str = "auto",
    energy_range: Optional[tuple[float, float]] = None,
) -> PreparedSpectrum:
    """Load a PHA spectrum, rebin it conserving counts, and convert to density.

    With ``energy_range=(lo, hi)``, select native channels by centre in this
    interval before any grouping; retain complete channels and their counts.
    None retains the complete loaded grid. The RMF itself is not cropped, so
    instrumental widths are still measured from its complete response profiles.

    The returned ``PreparedSpectrum`` is in counts s^-1 keV^-1 on the rebinned
    grid. Its ``native`` attribute keeps the channel-resolution counts,
    background and grouping needed to generate Poisson null realizations that
    follow exactly the same rebinning.

    RATE/COUNT_RATE inputs require STAT_ERR, ERROR or ERR by default. Setting
    ``assume_poisson_rate_errors=True`` estimates missing errors from each
    file's rate and EXPOSURE assuming raw Poisson counts divided by exposure.
    This applies consistently to source, background and native count metadata;
    explicit errors always take precedence.

    All methods retain empty native channels with their original variances.
    In Poisson fitting, zero raw variances stay zero and are not fit weights;
    plotted Poisson intervals are computed separately. Gaussian fitting keeps
    the legacy one-count fallback only on zero-variance output bins.
    fit_statistic='auto' selects Poisson for compatible raw counts. Explicit
    'poisson' rejects incompatible count/error models; 'gaussian' requests the
    legacy weighted fit. Source and background must both be compatible.

    ``rebin_method="optimal"`` (alias ``"kaastra"``) uses Kaastra & Bleeker
    (2016), equations 36--37 and Appendix B.1. Set ``rebin_scale=None`` and
    ``rebin_min_bins=1``. The grid is chosen from the RMF and gross source-region
    counts; optional background subtraction affects the values and propagated
    uncertainties, not the counts used to choose groups. Empty native channels
    are retained with their original variances. Only a zero-variance output
    bin receives a one-count variance floor only in Gaussian fitting mode.

    Optimal grouping is the default. ``oversampling=None`` uses the original
    optimal rule; a positive factor additionally imposes FWHM / oversampling
    as the minimum group size (see ``rebin_counts`` for channel boundaries).

    ``rebin_method="counts"`` (alias ``"mincounts"``) requires a positive
    integer ``rebin_scale`` such as 5. Empty channels remain on the native
    grid and contribute their widths, without adding artificial variance to
    every empty native channel. Grouping is repeated for every Poisson null.
    """
    _validate_fit_statistic(fit_statistic)
    validate_oversampling(oversampling, rebin_method)
    optimal = str(rebin_method).lower() in OPTIMAL_METHODS
    minimum_counts = str(rebin_method).lower() in COUNT_METHODS
    spectrum = load_fits_spectrum(
        pha_path=pha_path,
        background_path=background_path,
        rmf_path=rmf_path,
        arf_path=arf_path,
        keep_zero_uncertainty=True,
        assume_poisson_rate_errors=assume_poisson_rate_errors,
    )
    if energy_range is not None:
        lo, hi = _validate_energy_interval(energy_range, "energy_range")
        keep = (spectrum.energy >= lo) & (spectrum.energy <= hi)
        if not np.any(keep):
            raise ValueError(f"No native channels in the working energy range: {lo}--{hi}.")
        # Select whole channels before forming any groups. In particular, a
        # sparse trailing group cannot extend into the unused response tail.
        spectrum = replace(
            spectrum, energy=spectrum.energy[keep], values=spectrum.values[keep],
            uncertainties=spectrum.uncertainties[keep], bin_width=spectrum.bin_width[keep],
            response_sigma=(None if spectrum.response_sigma is None
                            else spectrum.response_sigma[keep]),
        )
    meta = align_to_spectrum(
        read_pha_metadata(
            pha_path, rmf_path, background_path,
            assume_poisson_rate_errors=assume_poisson_rate_errors,
        ),
        spectrum.energy,
        count_validation_range=energy_range,
    )

    exposure = float(meta["exposure"])
    if not np.isfinite(exposure) or exposure <= 0:
        raise ValueError("PHA file has no valid EXPOSURE keyword.")

    compatible = bool(meta.get('poisson_compatible', False))
    if fit_statistic == 'poisson' and not compatible:
        raise ValueError(
            "Poisson fitting requires raw nonnegative integer COUNTS without "
            "a supplied Gaussian/systematic error model, for both source and "
            "background. RATE requires assume_poisson_rate_errors=True and "
            "recoverable raw counts. Use fit_statistic='gaussian' for other inputs.")
    use_poisson = compatible and fit_statistic != 'gaussian'

    # Native (net) counts. The loader returns the column as stored: counts,
    # or counts/s for RATE columns.
    if meta["values_unit"] == "rate":
        counts = spectrum.values * exposure
        counts_unc = spectrum.uncertainties * exposure
    else:
        counts = spectrum.values.copy()
        counts_unc = spectrum.uncertainties.copy()

    optimal_response = None
    rebin_options = dict(source_counts=meta['src_counts']) if minimum_counts else {}
    if optimal:
        # Preserve native variances; the zero-variance fallback is applied after grouping.
        optimal_response = optimal_response_from_rmf(rmf_path, spectrum.energy)
        rebin_options = dict(source_counts=meta['src_counts'],
                             optimal_response=optimal_response)

    # Count-conserving rebinning
    e_new, n_new, s_new, de_new, group = rebin_counts(
        spectrum.energy, counts, counts_unc, spectrum.bin_width,
        method=rebin_method, scale=rebin_scale, min_bins=rebin_min_bins,
        oversampling=oversampling,
        zero_variance_floor=not use_poisson,
        **rebin_options,
    )

    # Density: counts s^-1 keV^-1
    conv = exposure * de_new
    values = n_new / conv
    uncertainties = s_new / conv
    poisson_data = None
    if use_poisson:
        validate_poisson_groups(spectrum.energy, spectrum.bin_width, group)
        keep = group >= 0
        _, inverse = np.unique(group[keep], return_inverse=True)
        poisson_data = PoissonFitData(
            np.bincount(inverse, weights=meta['src_counts'][keep], minlength=len(e_new)),
            np.bincount(inverse, weights=meta['bkg_counts'][keep], minlength=len(e_new)),
            float(meta['bkg_scale']), exposure, de_new,
        )

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
        optimal_response=optimal_response,
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
        poisson_data=poisson_data,
    )


def _validate_energy_interval(value, name):
    bounds = np.asarray(value, dtype=float)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[1] <= bounds[0]:
        raise ValueError(f"Invalid energy range: {name} must contain two finite, increasing energy limits.")
    return float(bounds[0]), float(bounds[1])


def _resolve_working_band(en1, en2, energy_pad, max_range_fraction, baseline_context):
    """Resolve the domain and continuum cap before any count-dependent grouping."""
    en1, en2 = _validate_energy_interval((en1, en2), "en1/en2")
    pad = np.asarray(energy_pad, dtype=float)
    if pad.ndim != 0 or not np.isfinite(pad):
        raise ValueError("energy_pad must be a finite number.")
    pad = max(float(pad), 0.0)
    fit_lo, fit_hi = en1 - pad, en2 + pad
    fraction = np.asarray(max_range_fraction, dtype=float)
    if fraction.ndim != 0 or not np.isfinite(fraction) or fraction <= 0:
        raise ValueError("max_range_fraction must be a finite, positive number.")
    cap = float(fraction * (fit_hi - fit_lo))
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("The maximum baseline window must be finite and positive.")
    automatic = (fit_lo - cap / 2, fit_hi + cap / 2)
    if not np.all(np.isfinite(automatic)):
        raise ValueError("The working energy range must be finite.")
    if baseline_context is None:
        context = automatic
    else:
        context = _validate_energy_interval(baseline_context, "baseline_context")
        tolerance = 8 * np.finfo(float).eps * max(1., *np.abs(automatic), *np.abs(context))
        if context[0] > automatic[0] + tolerance or context[1] < automatic[1] - tolerance:
            raise ValueError(
                "baseline_context must include the automatic continuum context "
                f"({automatic[0]:.8g}, {automatic[1]:.8g}) keV. "
                "It may extend that interval; use en1/en2 to change the search band.")
        # Decimal input equal to an automatic boundary may differ by one ULP.
        context = (min(context[0], automatic[0]), max(context[1], automatic[1]))
    return (fit_lo, fit_hi), context, cap


def _spectrum_baseline(spectrum, **options):
    """Use the continuum estimator associated with this spectrum's likelihood."""
    cap = getattr(spectrum, "baseline_max_window_width", None)
    if cap is not None:
        options.setdefault("max_window_width", cap)
    data = getattr(spectrum, "poisson_data", None)
    if data is None:
        return base_calculator(spectrum.energy, spectrum.values, **options)
    return poisson_base_calculator(
        spectrum.energy, data.source_counts, data.background_counts,
        data.background_scale, data.exposure, data.bin_width, **options)


def _poisson_fit_options(spectrum):
    data = getattr(spectrum, "poisson_data", None)
    return {} if data is None else {"poisson_data": data}


def _baseline_and_residuals(
    spectrum: PreparedSpectrum,
    base: Optional[np.ndarray] = None,
    ylines: Optional[np.ndarray] = None,
    baseline_window: float | np.ndarray = 0.4,
    max_range_fraction: float = 0.2,
    min_points: int = 3,
):
    """Return empirical baseline and signed baseline-subtracted residuals.

    Parameters
    ----------
    spectrum : PreparedSpectrum
        Prepared input spectrum.

    base : array-like or None, default=None
        User-provided baseline. If omitted, the empirical BLiSS
        baseline is calculated from the spectrum.

    ylines : array-like or None, default=None
        User-provided signed residuals on the spectrum grid. If omitted,
        calculated as ``values - baseline``. Unlike the positive excess
        used to locate candidate blocks, this fitting target retains
        fluctuations below the baseline. Supplied residuals are used as-is.

    baseline_window : float or array-like, default=0.4
        Continuum-estimation window width in physical energy units.

    max_range_fraction : float, default=0.2
        Maximum baseline-window width as a fraction of the
        available energy range.

    min_points : int, default=3
        Minimum number of points required inside the local
        continuum-estimation window.

    Returns
    -------
    base, ylines : numpy.ndarray
        Empirical baseline and signed residual spectrum.
    """

    if base is None:

        base = _spectrum_baseline(
            spectrum,
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

        ylines = np.asarray(spectrum.values, dtype=float) - base

    else:

        ylines = np.asarray(
            ylines,
            dtype=float,
        )

    if len(ylines) != len(spectrum.energy):
        raise ValueError(
            "ylines must have the same length as the spectrum."
        )

    if getattr(spectrum, 'poisson_data', None) is not None and not np.allclose(
            ylines, np.asarray(spectrum.values) - base, rtol=1e-10, atol=1e-12):
        raise ValueError("Poisson fitting uses raw counts: supplied ylines must equal values - base.")
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
        poisson_data=(None if getattr(spectrum, 'poisson_data', None) is None
                      else spectrum.poisson_data.sliced(mask)),
        baseline_max_window_width=getattr(spectrum, 'baseline_max_window_width', None),
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
    if 'width_fixed_instrumental' not in lines:
        lines['width_fixed_instrumental'] = False
    else:
        lines['width_fixed_instrumental'] = lines['width_fixed_instrumental'].map(
            lambda flag: isinstance(flag, (bool, np.bool_)) and bool(flag)).astype(bool)
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
    Free-width area errors include amplitude/width covariance; legacy tables
    without that covariance have unavailable area errors and area S/N.
    For explicitly fixed widths, area errors use amplitude uncertainty alone.
    """
    # Filter before infinities are normalized to NaN diagnostics below.
    lines = discard_large_center_errors(lines)

    if 'width_fixed_instrumental' not in lines:
        lines['width_fixed_instrumental'] = False
    else:
        lines['width_fixed_instrumental'] = lines['width_fixed_instrumental'].map(
            lambda flag: isinstance(flag, (bool, np.bool_)) and bool(flag)).astype(bool)

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
        width_fixed_instrumental=lines['width_fixed_instrumental'],
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
    min_sigma_line: float = 1e-6,
):
    """Refit selected candidates and compute final line diagnostics.

    Area errors use the final joint fit covariance, replacing local covariance.
    Gaussian fitting uses signed ``values - base`` residuals, as in local
    observed and null fits. Poisson fitting instead uses original counts and
    the bin-integrated continuum + lines model. Optional ``ylines`` must
    contain precomputed signed residuals,
    not the positive excess used to locate candidate blocks.
    All amplitudes, centers and widths are initially free within their bounds.
    Widths below the instrumental sigma at the fitted center are fixed to that
    value and remaining parameters are jointly refitted. The output marks
    these widths with ``width_fixed_instrumental=True`` and ``esigma=NaN``.
    Their amplitude/center/area errors are conditional on the fixed width.
    Every amplitude is bounded by the maximum observed bin over the global
    fit interval plus that bin's effective fitting uncertainty. The bin value
    includes the continuum; initial amplitudes are clipped to this interval.
    The lower width bound is max(min_sigma_line, 0.1 * instrumental_sigma),
    using min_sigma_line alone when no valid instrumental width is available.
    Every width is bounded above by the smaller of ``max_sigma_line`` (default
    0.1 keV) and its saved local ``fit_block_width``, when available.
    Components with ecenter > 1 keV are discarded before and after refitting.
    The returned line model includes only retained components, preserving
    their joint-fit parameters and covariance without another refit.
    Amplitude and width errors have no upper cut.
    Input detection p-values are preserved, including when the global fit fails.
    They describe the earlier local search. The returned table
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
        min_sigma_line=min_sigma_line,
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
    min_sigma_line: float = 1e-6,
):
    """Compute the global fit, retaining context until final output selection."""
    validate_sigma_bounds(min_sigma_line, max_sigma_line)

    base, ylines = _baseline_and_residuals(
        spectrum,
        base=base,
        ylines=ylines,
        baseline_window=baseline_window,
        max_range_fraction=max_range_fraction,
        min_points=min_points,
    )

    # This explicit filter cannot be bypassed by cached fit-quality flags.
    clean_lines = discard_large_center_errors(clean_lines).reset_index(drop=True)
    if 'width_fixed_instrumental' not in clean_lines:
        clean_lines['width_fixed_instrumental'] = False
    else:
        clean_lines['width_fixed_instrumental'] = clean_lines['width_fixed_instrumental'].map(
            lambda flag: isinstance(flag, (bool, np.bool_)) and bool(flag)).astype(bool)
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
    # Compact catalogues need numerical quality checks before refitting.
    # For cached flags, recheck invalid centroid errors;
    # preserve the existing treatment of other exclusions and legacy inputs.
    error_columns = {'ecenter', 'esigma', 'eamplitude'}
    if len(clean_lines) and 'ecenter' in clean_lines:
        quality = annotate_fit_quality(clean_lines)
        # Missing optional error columns remain unknown in legacy inputs;
        # nonfinite errors explicitly supplied by the caller are still invalid.
        missing_reasons = {'invalid_' + name for name in error_columns - set(clean_lines)}
        if missing_reasons:
            quality['fit_reasons'] = quality['fit_reasons'].map(
                lambda reasons: ';'.join(reason for reason in reasons.split(';')
                                         if reason not in missing_reasons))
            quality['fit_evaluable'] = quality['fit_reasons'].eq('')
        if 'fit_evaluable' not in clean_lines and error_columns.issubset(clean_lines):
            clean_lines = quality
        else:
            recheck = quality['fit_reasons'].str.contains(
                r'(?:^|;)(?:invalid_ecenter|numerically_suspect_ecenter)(?:;|$)')
            if recheck.any():
                if 'fit_evaluable' not in clean_lines:
                    clean_lines['fit_evaluable'] = pd.Series(np.nan, index=clean_lines.index, dtype=object)
                if 'fit_reasons' not in clean_lines:
                    clean_lines['fit_reasons'] = ''
                clean_lines.loc[recheck, ['fit_evaluable', 'fit_reasons']] = quality.loc[
                    recheck, ['fit_evaluable', 'fit_reasons']]
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

        if getattr(spectrum, 'poisson_data', None) is None:
            fit_sigma = np.where(valid_sigma, fit_sigma, fallback_sigma)
        p0, bounds = p0_generator_final(
            spectrum.energy,
            spectrum.values,
            clean_lines,
            response_sigma=spectrum.response_sigma,
            max_sigma_line=max_sigma_line,
            min_sigma_line=min_sigma_line,
            uncertainties=fit_sigma,
        )

        try:
            popt, pcov, width_fixed = fit_with_instrumental_width(
                spectrum.energy, ylines, fit_sigma, p0, bounds,
                response_sigma=spectrum.response_sigma,
                poisson_data=getattr(spectrum, 'poisson_data', None), baseline=base,
                maxfev=final_fit_maxfev,
                gaussian_fit=curve_fit, poisson_fit=fit_poisson_gaussians,
            )

            global_converged = True
            errors = np.sqrt(np.diag(pcov))

            popt_ = np.reshape(popt, (-1, 3))
            errors_ = np.reshape(errors, (-1, 3))
            # Preserve joint-fit row positions for covariance and provenance;
            # rebuild the model using only components surviving the error cut.
            keep_components = ~center_error_exceeds_limit(errors_[:, 1])
            kept_parameters = popt_[keep_components].ravel()
            if getattr(spectrum, 'poisson_data', None) is None:
                yfit = n_gaussian(spectrum.energy, *kept_parameters)
            else:
                data = spectrum.poisson_data
                yfit = gaussian_count_model(
                    spectrum.energy, np.zeros_like(base), data, kept_parameters,
                ) / (data.exposure * data.bin_width)

            for k in range(len(clean_lines)):
                new_line = np.concatenate(
                    [
                        popt_[k],
                        errors_[k],
                        [pcov[3*k, 3*k + 2]],
                    ]
                )
                fitted_final.loc[k] = new_line
            fitted_final['width_fixed_instrumental'] = width_fixed

        except (RuntimeError, ValueError) as exc:
            global_message = str(exc)
            print(f"Error final fitting: {exc}")

    # Detection p-values refer to the local search and remain unchanged by refitting.
    clean_select = clean_lines[["noise_on_block", "relative_power", *P_VALUE_COLUMNS]].copy()
    clean_select['fit_block_width'] = clean_lines.get('fit_block_width', np.nan)
    clean_select['fit_center_initial'] = clean_lines['center'].to_numpy()

    result = pd.concat(
        [
            fitted_final,
            clean_select,
        ],
        axis=1,
    )
    result = discard_large_center_errors(result)

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
        width_fixed_instrumental=result.get('width_fixed_instrumental', False),
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
            # Integrate this component only, excluding neighbouring lines.
            params = [float(row.amplitude), center, sigma]
            if getattr(spectrum, 'poisson_data', None) is None:
                line_density = n_gaussian(spectrum.energy[valid_mask], *params)
            else:
                data = spectrum.poisson_data.sliced(valid_mask)
                line_density = gaussian_count_model(
                    spectrum.energy[valid_mask], np.zeros(valid_mask.sum()), data, params,
                ) / (data.exposure * data.bin_width)
            ew = float(np.sum(line_density / base[valid_mask]
                              * spectrum.bin_width[valid_mask]) * 1000.0)
        else:
            ew = np.nan

        ew_vals.append(ew)

    result["ew"] = ew_vals

    result['fit_converged'] = global_converged
    result['fit_message'] = global_message
    result['fit_block_id'] = np.nan  # the global fit is not a local block
    if not global_converged:
        result['center'] = result['fit_center_initial']
    result = annotate_fit_quality(result)
    result = _flag_response_features(result, spectrum, response_feature_threshold)
    result = result.reindex(columns=LINE_OUTPUT_COLUMNS)

    # Keep local detection p-values as provenance. Global fit failures are
    # recorded in the separate fit-quality columns above.
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
    overwrite: bool = False,
) -> None:
    """Plot the spectrum, empirical baseline, and global line model.

    Y-axis limits are autoscaled from the displayed energy interval only.

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
    overwrite : bool, default: False
        Allow replacement of an existing output image. Otherwise raise
        FileExistsError. Names without an extension use savefig's default format.
    """
    _plot_spectrum_diagnostic(
        spectrum, base, yfit=yfit, output_path=output_path,
        show_plot=show_plot, energy_min=energy_min, energy_max=energy_max,
        size_fig_input=size_fig_input, overwrite=overwrite,
    )


def _poisson_plot_errors(data):
    """Central 68.27% Garwood count intervals, converted to density.

    For background-subtracted data the independent marginal interval errors
    are combined in quadrature; these net bars are an approximation, not an
    exact joint confidence interval. Neither kind of bar weights the fit.
    """
    from scipy.stats import chi2

    tail = (1.0 - 0.682689492137086) / 2.0

    def interval_errors(counts):
        lower = np.zeros_like(counts)
        positive = counts > 0
        lower[positive] = .5 * chi2.ppf(tail, 2 * counts[positive])
        upper = .5 * chi2.ppf(1 - tail, 2 * (counts + 1))
        return counts - lower, upper - counts

    source_lo, source_hi = interval_errors(data.source_counts)
    background_lo, background_hi = interval_errors(data.background_counts)
    scale = data.background_scale
    # Subtraction reverses which side of the background interval contributes.
    low = np.hypot(source_lo, scale * background_hi)
    high = np.hypot(source_hi, scale * background_lo)
    return np.stack((low, high)) / (data.exposure * data.bin_width)


def _plot_spectrum_diagnostic(
    spectrum: PreparedSpectrum,
    base: np.ndarray,
    yfit: np.ndarray | None = None,
    *,
    candidate_centers=None,
    output_path: str | Path | None = None,
    show_plot: bool = True,
    energy_min: float | None = None,
    energy_max: float | None = None,
    size_fig_input: tuple[float, float] | None = None,
    overwrite: bool = False,
) -> None:
    """Draw data and baseline with either a line model or candidate markers."""

    if output_path is not None:
        output_path = _plot_output_path(output_path)
        _check_output_paths([output_path], overwrite=overwrite)

    if size_fig_input is None:
        size_fig = (10, 5)
    else:
        size_fig = size_fig_input

    # Restrict every plotted array so y autoscaling uses only the visible
    # energy interval, including the data uncertainties and all model curves.
    energy = np.asarray(spectrum.energy)
    visible = np.isfinite(energy)
    if energy_min is not None:
        visible &= energy >= energy_min
    if energy_max is not None:
        visible &= energy <= energy_max

    plot_energy = energy[visible]
    plot_base = np.asarray(base)[visible]
    plot_yfit = None if yfit is None else np.asarray(yfit)[visible]

    plt.figure(figsize=size_fig)

    poisson_data = getattr(spectrum, 'poisson_data', None)
    if poisson_data is None:
        plot_errors = np.asarray(spectrum.uncertainties)[visible]
        label = "Data"
    else:
        plot_errors = _poisson_plot_errors(poisson_data.sliced(visible))
        label = ("Data (Poisson intervals)" if poisson_data.background_scale == 0
                 else "Data (approx. Poisson intervals)")
    plt.errorbar(
        plot_energy,
        np.asarray(spectrum.values)[visible],
        yerr=plot_errors,
        label=label,
        fmt='.-',
        markersize=3,
        color=(0.1216, 0.4667, 0.7059, 0.7),
        ecolor=(0.1216, 0.4667, 0.7059, 0.2),
    )

    plt.plot(plot_energy, plot_base, "k:", label="base")
    if plot_yfit is not None and np.any(np.isfinite(plot_yfit) & (plot_yfit != 0)):
        plt.plot(plot_energy, plot_yfit, "g:", label="Lines")
        plt.plot(plot_energy, plot_yfit + plot_base, "r", label="Line+base")
    if candidate_centers is not None and plot_energy.size:
        centers = np.asarray(candidate_centers, dtype=float)
        centers = centers[np.isfinite(centers)]
        if energy_min is not None:
            centers = centers[centers >= energy_min]
        if energy_max is not None:
            centers = centers[centers <= energy_max]
        for i, center in enumerate(np.unique(centers)):
            # Axis-height markers do not change the data's vertical scale.
            plt.axvline(center, color="red", linestyle="--", linewidth=1,
                        alpha=0.8, label="Candidates" if i == 0 else "_nolegend_")

    if energy_min is not None or energy_max is not None:
        plt.xlim(energy_min, energy_max)

    plt.xlabel("Energy (keV)")
    plt.ylabel("Spectra")
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        try:
            with _reserve_output_paths([output_path], overwrite=overwrite):
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
        except BaseException:
            plt.close()
            raise

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
    min_sigma_line: float = 1e-6,
    overwrite: bool = False,
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
        Optional precomputed baseline and signed ``values - base`` residuals.
        If omitted, residuals are calculated without clipping negative bins.
        Supplied ``ylines`` are used as-is; do not pass the positive excess
        used for candidate detection.
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
    overwrite : bool, default: False
        Allow replacement of the files saved by this call. Otherwise reject
        existing destinations before fitting or writing. Other files in the
        directory are allowed, including a previous candidate catalogue.
    final_fit_maxfev : int, default: 100000
        Maximum number of function evaluations in ``curve_fit``.
    response_feature_threshold : float, default: 5.0
        ARF sharpness-score threshold for flagging fitted lines; pass the same
        value as in the candidate-search config. The returned table and CSV
        include ``response_feature``. With no usable ARF, it is False
        (not flagged); the numeric feature score stays internal.
    min_sigma_line : float, default: 1e-6
        Absolute lower sigma bound in keV. The effective lower bound is
        max(min_sigma_line, 0.1 * instrumental_sigma) where available.
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

    output_path = None
    output_paths = []
    if output_dir is not None:
        output_dir = ensure_output_folder(output_dir)
        output_path = _plot_output_path(output_dir / plot_name)
        output_paths = [output_path]
        if save_csv:
            output_paths.append(output_dir / 'global_fit_lines.csv')
        _check_output_paths(output_paths, overwrite=overwrite)

    base, ylines = _baseline_and_residuals(
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
        min_sigma_line=min_sigma_line,
    )

    with _reserve_output_paths(output_paths, overwrite=overwrite):
        if output_dir is not None and save_csv:
            result.to_csv(
                output_dir / "global_fit_lines.csv",
                index=False,
            )

        if show_plot or output_path is not None:
            # This call owns every reserved destination, including the figure.
            plot_global_fit(
                spectrum=spectrum,
                base=base,
                yfit=yfit,
                output_path=output_path,
                show_plot=show_plot,
                energy_min=energy_min,
                energy_max=energy_max,
                size_fig_input=size_fig_input,
                overwrite=True,
            )

    if return_yfit:
        return result, yfit

    return result


class BlindLineSearchPipeline:
    """Run the blind emission-line search on one spectrum.

    By default, the pipeline now stops after candidate detection and p-value
    estimation. The expensive global multi-Gaussian fit can be run later with
    ``fit_global`` after the user filters the candidate table.

    Each run also compares five local statistics with maxima from the same
    null searches. ``n_sim`` records all generated realizations, including
    empty searches, and ``null_maxima`` retains their five maxima for inspection.
    ``synthetic_candidates`` is None unless config.keep_synthetic_candidates
    is True, in which case it contains all null candidate rows and metrics.
    """

    def __init__(self, config: Optional[BlindLineSearchConfig] = None):
        """Create a pipeline instance."""
        self.config = config or BlindLineSearchConfig()
        self.synthetic_candidates = None
        self.null_maxima = None
        self.n_sim = 0

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
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """Execute the BLiSS workflow on a PHA spectrum.

        ``pha`` and ``rmf`` are required; ``arf`` and ``bkg`` are optional.
        The spectrum is rebinned conserving counts according to the config and
        converted to counts s^-1 keV^-1. Native channels are first restricted
        to the padded search interval plus continuum context. The empirical
        baseline is computed over this complete working band, and nulls use
        exactly the same native domain and physical window cap.
        Candidate detection, local Gaussian fitting, p-value estimation,
        and the optional global fit are performed inside the selected energy
        interval enlarged by ``energy_pad``. The returned catalogue is finally
        restricted to the nominal ``en1``--``en2`` interval. Both modes return
        and export exactly ``FINAL_OUTPUT_COLUMNS``.
        The five ``p_global_*`` columns describe the local detection search
        over the nominal interval and are preserved by an optional global fit.
        With show_plot=True, save and display the spectrum and baseline. Without
        a global fit, red vertical markers locate valid local candidates, with
        no p-value threshold. With final_fit=True, show the global model instead.
        Automatic output folders are exclusive. With an explicit output_dir,
        existing destinations raise FileExistsError unless overwrite=True;
        files not written by this call are left alone.
        """

        # Do not retain diagnostic catalogues from a previous run or expose
        # partial null maxima after a failed simulation/reduction.
        self.synthetic_candidates = None
        self.null_maxima = None
        self.n_sim = 0
        _validate_fit_statistic(self.config.fit_statistic)
        if not isinstance(self.config.keep_synthetic_candidates, (bool, np.bool_)):
            raise ValueError('keep_synthetic_candidates must be a boolean.')
        validate_oversampling(self.config.oversampling, self.config.rebin_method)
        validate_sigma_bounds(self.config.min_sigma_line, self.config.max_sigma_line)
        validate_parallel_options(self.config.n_jobs, self.config.simulation_batch_size)
        validate_min_relative_prominence(self.config.min_relative_prominence)
        output_dir = ensure_output_folder(output_dir)
        table_name = 'candidate_lines_global_fit.csv' if final_fit else 'candidate_lines.csv'
        output_paths = [output_dir / table_name, output_dir / 'run_summary.txt']
        plot_path = None
        if show_plot:
            plot_path = _plot_output_path(output_dir / plot_name)
            output_paths.append(plot_path)
        _check_output_paths(output_paths, overwrite=overwrite)

        # Resolve a count-independent working band before grouping. The same
        # native channels and physical continuum cap will be used by the nulls.
        en1_use = self.config.en1 if en1 is None else en1
        en2_use = self.config.en2 if en2 is None else en2
        pad = self.config.energy_pad if energy_pad is None else energy_pad
        (fit_en1, fit_en2), context, max_window_width = _resolve_working_band(
            en1_use, en2_use, pad, self.config.max_range_fraction,
            self.config.baseline_context,
        )
        self.search_energy_range = (float(en1_use), float(en2_use))
        self.baseline_context = context
        self.baseline_max_window_width = max_window_width

        # ------------------------------------------------------------
        # Load and group the complete working band (science + context)
        # ------------------------------------------------------------
        spectrum_full = prepare_spectrum(
            pha, rmf, arf, bkg,
            energy_range=context,
            rebin_method=self.config.rebin_method,
            rebin_scale=self.config.rebin_scale,
            rebin_min_bins=self.config.rebin_min_bins,
            oversampling=self.config.oversampling,
            assume_poisson_rate_errors=self.config.assume_poisson_rate_errors,
            fit_statistic=('gaussian' if self.config.fit_statistic == 'auto'
                           and self.config.noise_model == 'gaussian'
                           else self.config.fit_statistic),
        )
        spectrum_full = replace(spectrum_full, baseline_max_window_width=max_window_width)
        self._active_spectrum = spectrum_full
        self.fit_statistic = ('poisson' if getattr(spectrum_full, 'poisson_data', None) is not None
                              else 'gaussian')
        if self.fit_statistic == 'poisson' and self.config.noise_model != 'poisson':
            raise ValueError("Poisson fitting requires noise_model='poisson'.")

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
        # Baseline from the complete bounded working spectrum
        # ------------------------------------------------------------
        base_full, baseline_info = _spectrum_baseline(
            spectrum_full,
            baseline_window=self.config.baseline_window,
            max_range_fraction=self.config.max_range_fraction,
            min_points=self.config.min_points,
            return_info=True,
        )

        # Nulls must adapt the same initial scale to their own rebinned grids.
        initial_window_width = baseline_info.get(
            "initial_window_width", baseline_info["window_width"],
        )

        # Positive excess only locates candidate blocks. Fits retain all bins:
        # raw source/background counts for Poisson, signed residuals for Gaussian.
        ylines_full = np.maximum(spectrum_full.values - base_full, 0.0)
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
            min_sigma_line=self.config.min_sigma_line,
            min_relative_prominence=self.config.min_relative_prominence,
            **_poisson_fit_options(spectrum),
        )

        # ------------------------------------------------------------
        # Null realizations: same rebinning, baseline and detection
        # chain as the data, one candidate table per realization
        # ------------------------------------------------------------
        def serial_candidates():
            null_realizations = generate_null_realizations(
                spectrum_full, base_full, fit_en1, fit_en2, self.config,
                window_width=initial_window_width, stream=True,
            )
            try:
                for k, null in enumerate(null_realizations):
                    cand = return_raw_lines(
                        null.energy, null.values, null.uncertainties,
                        null.ylines, null.baseline,
                        response_sigma=null.response_sigma,
                        max_sigma_line=self.config.max_sigma_line,
                        min_sigma_line=self.config.min_sigma_line,
                        min_relative_prominence=self.config.min_relative_prominence,
                        **_poisson_fit_options(null),
                    )
                    cand["sim"] = k
                    yield cand
            finally:
                close = getattr(null_realizations, 'close', None)
                if close is not None:
                    close()

        null_tables = (
            iter_parallel_null_candidates(
                spectrum_full, base_full, fit_en1, fit_en2, self.config,
                window_width=initial_window_width,
            ) if self.config.n_jobs > 1 else serial_candidates()
        )
        retained_tables = [] if self.config.keep_synthetic_candidates else None
        maxima_rows = []
        try:
            for sim, cand in enumerate(null_tables):
                if len(cand):
                    if 'sim' not in cand:
                        raise ValueError("Synthetic candidates must have a 'sim' column.")
                    identifiers = pd.to_numeric(cand['sim'], errors='coerce').to_numpy(dtype=float)
                    if not np.all(identifiers == sim):
                        raise ValueError('Each streamed null table must use its consecutive sim identifier.')
                metrics = _add_candidate_metrics(cand)
                maxima_rows.append(single_simulation_maxima(
                    metrics, en1=en1_use, en2=en2_use,
                ).to_numpy(dtype=float))
                if retained_tables is not None:
                    retained_tables.append(metrics)
                del cand, metrics
        finally:
            # Closing a failed parallel stream also closes its worker pool.
            close = getattr(null_tables, 'close', None)
            if close is not None:
                close()

        n_sim = len(maxima_rows)
        null_maxima = pd.DataFrame(
            maxima_rows, columns=LOOK_ELSEWHERE_STATISTICS,
            index=pd.RangeIndex(n_sim, name='sim'), dtype=float,
        )

        # ------------------------------------------------------------
        # Final catalogue restricted to the nominal science window
        # ------------------------------------------------------------
        selected = self._select_candidates(
            raw_candidates,
            en1=en1_use,
            en2=en2_use,
        )
        selected = calculate_look_elsewhere_pvalues_from_maxima(
            selected, null_maxima,
            en1=en1_use, en2=en2_use,
        )
        self.null_maxima = null_maxima
        self.n_sim = n_sim
        if retained_tables is not None:
            self.synthetic_candidates = (
                pd.concat(retained_tables, ignore_index=True) if retained_tables
                else _add_candidate_metrics(pd.DataFrame(columns=raw_candidates.columns))
            )

        if not final_fit:
            result = selected.reindex(columns=FINAL_OUTPUT_COLUMNS).copy()
            with _reserve_output_paths(output_paths, overwrite=overwrite):
                if show_plot:
                    quality = annotate_fit_quality(selected)
                    centers = pd.to_numeric(selected['center'], errors='coerce')
                    valid = (quality['fit_converged'].eq(True)
                             & quality['fit_evaluable'] & np.isfinite(centers))
                    _plot_spectrum_diagnostic(
                        spectrum=spectrum, base=base,
                        candidate_centers=centers[valid].to_numpy(),
                        output_path=plot_path, show_plot=True, overwrite=True,
                    )
                # This run owns the reserved table, summary and optional figure.
                self._write_candidate_outputs(result, output_dir, overwrite=True)
            return result

        # ------------------------------------------------------------
        # Optional global fit over the padded interval,
        # but only for candidates whose centroids are inside en1--en2
        # ------------------------------------------------------------
        result, yfit = _fit_global_with_diagnostics(
            spectrum=spectrum,
            base=base,
            # Compute signed residuals from this spectrum and baseline;
            # ylines above is only the positive detection mask.
            clean_lines=selected,
            final_fit_maxfev=self.config.final_fit_maxfev,
            response_feature_threshold=self.config.response_feature_threshold,
            max_sigma_line=self.config.max_sigma_line,
            min_sigma_line=self.config.min_sigma_line,
        )

        # Safety: keep only nominal-window lines after final fitting too
        result = self._select_energy_window(
            result,
            en1=en1_use,
            en2=en2_use,
        )

        result = result.reindex(columns=FINAL_OUTPUT_COLUMNS).copy()
        with _reserve_output_paths(output_paths, overwrite=overwrite):
            # The nested writers may fill the files reserved by this run.
            if show_plot:
                plot_global_fit(
                    spectrum=spectrum,
                    base=base,
                    yfit=yfit,
                    output_path=plot_path,
                    show_plot=True,
                    overwrite=True,
                )
            self._write_outputs(result, output_dir, overwrite=True)
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

        for col in CANDIDATE_COLUMNS:
            if col not in clean_lines.columns:
                clean_lines[col] = np.nan

        clean_lines = clean_lines[CANDIDATE_COLUMNS]
        return clean_lines

    def _write_candidate_outputs(self, candidates: pd.DataFrame, output_dir: Path, *, overwrite=False) -> None:
        """Write the pre-global-fit candidate table and run summary."""
        with _reserve_output_paths(
            [output_dir / 'candidate_lines.csv', output_dir / 'run_summary.txt'],
            overwrite=overwrite,
        ):
            candidates.to_csv(output_dir / 'candidate_lines.csv', index=False)
            with open(output_dir / 'run_summary.txt', 'w') as handle:
                handle.write('BLiSS candidate search completed\n')
                handle.write(f'Results folder: {output_dir}\n')
                handle.write(f'Fit statistic: {getattr(self, "fit_statistic", self.config.fit_statistic)}\n')
                handle.write(f'Search energy range (keV): {self.search_energy_range}\n')
                handle.write(f'Continuum context requested (keV): {self.baseline_context}\n')
                handle.write(f'Maximum continuum window (keV): {self.baseline_max_window_width}\n')
                handle.write(f'Number of candidate lines before global fit: {len(candidates)}\n')
                handle.write(f'Null simulations: {self.n_sim}; p_global_* use local-search maxima.\n')
                handle.write('Run fit_global(candidate_lines, spectrum) after user filtering to perform the global fit.\n')

    def _write_outputs(self, result: pd.DataFrame, output_dir: Path, *, overwrite=False) -> None:
        """Write the final candidate table and run summary."""
        with _reserve_output_paths(
            [output_dir / 'candidate_lines_global_fit.csv', output_dir / 'run_summary.txt'],
            overwrite=overwrite,
        ):
            result.to_csv(output_dir / 'candidate_lines_global_fit.csv', index=False)
            with open(output_dir / 'run_summary.txt', 'w') as handle:
                handle.write('BLiSS run completed with global fit\n')
                handle.write(f'Results folder: {output_dir}\n')
                handle.write(f'Fit statistic: {getattr(self, "fit_statistic", self.config.fit_statistic)}\n')
                handle.write(f'Search energy range (keV): {self.search_energy_range}\n')
                handle.write(f'Continuum context requested (keV): {self.baseline_context}\n')
                handle.write(f'Maximum continuum window (keV): {self.baseline_max_window_width}\n')
                handle.write(f'Number of fitted candidates: {len(result)}\n')
                handle.write(f'Null simulations: {self.n_sim}; p_global_* retain local-search p-values.\n')


# Distinguish an omitted nullable option from an explicit None that clears it.
_UNSET = object()


def _build_config(
    config, en1, en2, energy_pad, rebin_method, rebin_scale, rebin_min_bins,
    num_synthetic_simulations, synthetic_seed, oversampling=_UNSET,
    assume_poisson_rate_errors=None, fit_statistic=None, baseline_context=_UNSET,
):
    if config is None:
        config = BlindLineSearchConfig()
    overrides = dict(
        en1=en1,
        en2=en2,
        energy_pad=energy_pad,
        rebin_method=rebin_method,
        rebin_min_bins=rebin_min_bins,
        num_synthetic_simulations=num_synthetic_simulations,
        assume_poisson_rate_errors=assume_poisson_rate_errors,
        fit_statistic=fit_statistic,
    )
    overrides = {key: value for key, value in overrides.items() if value is not None}
    for key, value in dict(
        rebin_scale=rebin_scale, synthetic_seed=synthetic_seed, oversampling=oversampling,
        baseline_context=baseline_context,
    ).items():
        if value is not _UNSET:
            overrides[key] = value
    return replace(config, **overrides)


def find_candidate_lines(
    pha,
    rmf,
    arf=None,
    bkg=None,
    *,
    en1=None,
    en2=None,
    energy_pad=None,
    show_plot=False,
    output_dir=None,
    plot_name='bliss_fit.png',
    final_fit: bool = False,
    rebin_method=None,
    rebin_scale=_UNSET,
    rebin_min_bins=None,
    num_synthetic_simulations=None,
    synthetic_seed=_UNSET,
    config: Optional[BlindLineSearchConfig] = None,
    oversampling=_UNSET,
    assume_poisson_rate_errors=None,
    fit_statistic=None,
    baseline_context=_UNSET,
    overwrite: bool = False,
):
    """Run BLiSS candidate detection, p-value estimation and an optional global fit.
    
    Parameters
    ----------
    pha, rmf : path-like
        Source spectrum and its redistribution matrix (required).
    arf, bkg : path-like or None
        Optional effective-area file and background spectrum.
    en1, en2, energy_pad : float or None
        Search interval and internal padding. None inherits the config value.
        Without a config, defaults are 1.7, 12.0 and 0.1 keV respectively.
        These are editable starting values, not instrument validity limits.
    rebin_method, rebin_scale, rebin_min_bins
        Count-conserving rebinning applied to the data and to every synthetic
        null realization (see ``rebin_counts``). Omitted options inherit the
        config. None also inherits rebin_method and rebin_min_bins, but an
        explicit rebin_scale=None clears the configured scale.
    num_synthetic_simulations, synthetic_seed
        Number of null realizations and random seed. Omitted options inherit
        the config. num_synthetic_simulations=None also inherits its value;
        synthetic_seed=None explicitly clears a configured seed.
    config : BlindLineSearchConfig or None
        Full configuration. Only explicitly supplied values override its
        fields; the original object is not modified. Without a config,
        omitted options use BlindLineSearchConfig defaults.
    oversampling : float or None
        Optional maximum sampling factor for optimal grouping. Omission
        inherits the config value; an explicit None disables the limit.
    assume_poisson_rate_errors : bool or None
        Allow Poisson estimation of missing RATE/COUNT_RATE errors for source
        and background. None inherits the config (False by default). Explicit
        True or False overrides it; supplied error columns are preserved.
    fit_statistic : {'auto', 'poisson', 'gaussian'} or None
        Fitting likelihood. None inherits the config (auto by default).
        Poisson uses raw counts, a line-resistant count-mean continuum, and
        no error floor.
        Supplied non-Poisson error models keep the Gaussian path in auto mode.
    baseline_context : (float, float) or None
        Working energy limits for continuum estimation and native grouping.
        Omission inherits the config; None uses the automatic context: the
        padded search interval plus half the maximum window on each side.
        Explicit limits must contain that context and can extend it. The
        maximum window is max_range_fraction times the padded search span,
        shared unchanged by data and nulls. Extra context does not enlarge
        the search interval. Changing context can change adaptive groups.
    output_dir : path-like or None
        Results directory. None creates a unique timestamped directory.
    final_fit : bool, default: False
        If True, jointly fit eligible local candidates and return the global-fit
        catalogue. Otherwise return local candidates. To select candidates first,
        use final_fit=False, filter the table, then call fit_global.
    show_plot : bool, default: False
        Save and display local candidate markers when final_fit=False, or the
        fitted multi-Gaussian model when final_fit=True.
    plot_name : str, default: 'bliss_fit.png'
        Diagnostic image name inside output_dir when show_plot=True.
    overwrite : bool, default: False
        Allow replacement of this call's existing output files in output_dir.
        Otherwise a collision raises FileExistsError before the search.
    
    Returns
    -------
    pandas.DataFrame
        Candidate catalogue with exactly ``FINAL_OUTPUT_COLUMNS``. Local results
        are saved to ``candidate_lines.csv``; with final_fit=True the global-fit
        results are saved to ``candidate_lines_global_fit.csv``.
    """
    config = _build_config(
        config, en1, en2, energy_pad, rebin_method, rebin_scale, rebin_min_bins,
        num_synthetic_simulations, synthetic_seed, oversampling,
        assume_poisson_rate_errors, fit_statistic, baseline_context,
    )
    pipeline = BlindLineSearchPipeline(config=config)
    return pipeline.run(
        pha, rmf, arf, bkg,
        final_fit=final_fit,
        show_plot=show_plot,
        output_dir=output_dir,
        plot_name=plot_name,
        overwrite=overwrite,
    )
