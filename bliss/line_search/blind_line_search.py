"""High-level BLiSS emission-line search pipeline."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from .empirical_baseline import base_calculator
from .candidate_regions import return_raw_lines
from .gaussian_models import n_gaussian, p0_generator_final
from ..synthetic_probability.synthetic_spectra import calculate_synthetic_noise_spectra
from ..synthetic_probability.gmm_probability import eval_line_probability_gmm
from ..plotting.run_output_manager import ensure_output_folder

LINE_OUTPUT_COLUMNS = [
    'center', 'ecenter', 'sigma', 'esigma', 'amplitude', 'eamplitude',
    'relative_power', 'noise_on_block', 'value_on_line',
    'base_on_line', 'snr_peak', 'snr_amplitude', 'area', 'earea',
    'snr_area', 'ew', 'cluster_probability'
]

CANDIDATE_COLUMNS = LINE_OUTPUT_COLUMNS.copy()
FINAL_OUTPUT_COLUMNS = LINE_OUTPUT_COLUMNS.copy()


@dataclass
class BlindLineSearchConfig:
    """Configuration values controlling the default BLiSS pipeline.

    Attributes
    ----------
    en1, en2 : float
        Lower and upper energy limits used when selecting candidates.
    num_synthetic_simulations : int
        Number of shuffled synthetic spectra generated for probability estimation.
    final_fit_maxfev : int
        Maximum number of function evaluations allowed in the final ``curve_fit``.
    snr_confidence_threshold : float
        Any available S/N diagnostic above which a candidate is assigned probability 1.
    """
    en1: float = 0.2
    en2: float = 10.0
    energy_pad: float = 0.1
    num_synthetic_simulations: int = 10
    synthetic_seed: Optional[int] = None
    final_fit_maxfev: int = 100000
    snr_confidence_threshold: float = 4.0


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
    """
    energy: np.ndarray
    values: np.ndarray
    uncertainties: np.ndarray
    bin_width: np.ndarray


def prepare_spectrum(spectra_or_energy, y=None, sy=None, bin_width=None) -> PreparedSpectrum:
    """Load and sort spectrum data from a file, arrays, or a Spectrum-like object.

    Parameters
    ----------
    spectra_or_energy : str, pathlib.Path, Spectrum-like object, or array-like
        Four-column text spectrum file, an object with ``energy``, ``values`` and
        ``uncertainties`` attributes, or the coordinate array for direct array input.
    y : array-like or None, default: None
        Spectral values for direct array input.
    sy : array-like or None, default: None
        One-sigma uncertainties for direct array input.
    bin_width : array-like or None, default: None
        Optional bin widths for direct array input. If omitted, they are estimated
        from adjacent coordinate spacing.

    Returns
    -------
    PreparedSpectrum
        Spectrum sorted by increasing coordinate value.
    """
    if isinstance(spectra_or_energy, (str, Path)):
        spectra = pd.read_csv(
            spectra_or_energy,
            sep='\\s+',
            comment='#',
            header=None,
            engine='python',
        )
        if spectra.shape[1] != 4:
            raise ValueError('File must have 4 columns: E_low, E_high, counts, error.')
        x = np.asarray((spectra[0] + spectra[1]) / 2.0)
        dE = np.asarray(spectra[1] - spectra[0])
        y = np.asarray(spectra[2])
        sy = np.asarray(spectra[3])
    elif all(hasattr(spectra_or_energy, attr) for attr in ('energy', 'values', 'uncertainties')):
        x = np.asarray(spectra_or_energy.energy)
        y = np.asarray(spectra_or_energy.values)
        sy = np.asarray(spectra_or_energy.uncertainties)
        dE_obj = getattr(spectra_or_energy, 'bin_width', None)
        if dE_obj is None:
            dE = _estimate_bin_width(x)
        else:
            dE = np.asarray(dE_obj)
    else:
        x = np.asarray(spectra_or_energy)
        if y is None or sy is None:
            raise ValueError('If using direct arrays, y and sy must be provided.')
        y = np.asarray(y)
        sy = np.asarray(sy)
        if bin_width is None:
            dE = _estimate_bin_width(x)
        else:
            dE = np.asarray(bin_width)

    if not (len(x) == len(y) == len(sy) == len(dE)):
        raise ValueError('energy, values, uncertainties, and bin_width must have the same length.')

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(sy) & np.isfinite(dE) & (sy > 0)
    x = x[valid]
    y = y[valid]
    sy = sy[valid]
    dE = dE[valid]

    order = np.argsort(x)
    return PreparedSpectrum(
        energy=x[order],
        values=y[order],
        uncertainties=sy[order],
        bin_width=dE[order],
    )


def _estimate_bin_width(x: np.ndarray) -> np.ndarray:
    """Estimate bin widths from a one-dimensional coordinate grid."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.array([], dtype=float)
    if len(x) == 1:
        return np.ones_like(x, dtype=float)
    dE = np.diff(x)
    return np.append(dE, dE[-1])


def _baseline_and_line_excess(
    spectrum: PreparedSpectrum,
    base: Optional[np.ndarray] = None,
    ylines: Optional[np.ndarray] = None,
):
    """Return baseline and positive line excess for a prepared spectrum."""
    if base is None:
        base = base_calculator(spectrum.values)
    else:
        base = np.asarray(base, dtype=float)

    if len(base) != len(spectrum.energy):
        raise ValueError('base must have the same length as the spectrum.')

    if ylines is None:
        ylines = np.maximum(spectrum.values - base, 0)
    else:
        ylines = np.asarray(ylines, dtype=float)

    if len(ylines) != len(spectrum.energy):
        raise ValueError('ylines must have the same length as the spectrum.')

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
    )

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
    *,
    final_fit_maxfev: int = 100000,
    snr_confidence_threshold: float = 4.0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Refit selected candidates and compute final line diagnostics.

    This is the expensive global step. It is intentionally independent from the
    candidate-search step, so users can filter ``clean_lines`` before running the
    multi-Gaussian fit.

    Parameters
    ----------
    spectrum : PreparedSpectrum
        Sorted spectrum used for the final fit.
    clean_lines : pandas.DataFrame
        User-filtered candidate lines. At minimum it must contain ``amplitude``,
        ``center``, and ``sigma``. If local context columns are missing, they are
        estimated from the nearest spectral bin.
    base : numpy.ndarray or None, default: None
        Empirical baseline evaluated on ``spectrum.energy``. If omitted, it is
        recomputed from ``spectrum.values``.
    ylines : numpy.ndarray or None, default: None
        Baseline-subtracted spectrum fitted with the multi-Gaussian model. If
        omitted, it is recomputed as ``max(values - base, 0)``.
    final_fit_maxfev : int, default: 100000
        Maximum number of function evaluations allowed in the final ``curve_fit``.
    snr_confidence_threshold : float, default: 4.0
        S/N threshold above which ``cluster_probability`` is set to 1 if any of ``snr_peak``, ``snr_area`` or ``snr_amplitude`` exceeds it.

    Returns
    -------
    tuple
        ``(result, yfit)`` where ``result`` is the final candidate DataFrame and
        ``yfit`` is the fitted multi-Gaussian line model on the full spectrum grid.
    """
    base, ylines = _baseline_and_line_excess(spectrum, base=base, ylines=ylines)
    clean_lines = _ensure_line_context(
        clean_lines,
        spectrum=spectrum,
        base=base,
        snr_confidence_threshold=snr_confidence_threshold,
    )

    fitted_final = pd.DataFrame(
        columns=['amplitude', 'center', 'sigma', 'eamplitude', 'ecenter', 'esigma']
    )
    yfit = np.zeros_like(spectrum.energy, dtype=float)

    if len(clean_lines) > 0:
        p0, bounds = p0_generator_final(spectrum.energy, spectrum.values, clean_lines)
        try:
            popt, pcov = curve_fit(
                n_gaussian,
                spectrum.energy,
                ylines,
                p0=p0,
                bounds=bounds,
                maxfev=final_fit_maxfev,
            )
            errors = np.sqrt(np.diag(pcov))
            yfit = n_gaussian(spectrum.energy, *popt)
            popt_ = np.reshape(popt, (-1, 3))
            errors_ = np.reshape(errors, (-1, 3))
            for k in range(len(clean_lines)):
                new_line = np.concatenate([popt_[k], errors_[k]])
                fitted_final.loc[k] = new_line
        except RuntimeError as exc:
            print(f'Error final fitting: {exc}')
        except ValueError as exc:
            print(f'Error final fitting: {exc}')

    clean_select = clean_lines[
        ['relative_power', 'noise_on_block', 'value_on_line', 'base_on_line', 'cluster_probability']
    ]
    result = pd.concat([fitted_final, clean_select], axis=1)

    if len(result) == 0:
        return pd.DataFrame(columns=FINAL_OUTPUT_COLUMNS), yfit

    cols = ['amplitude', 'sigma', 'eamplitude', 'esigma']
    result[cols] = result[cols].apply(pd.to_numeric, errors='coerce')
    result['noise_on_block'] = pd.to_numeric(result['noise_on_block'], errors='coerce')
    result['snr_peak'] = _safe_divide(result['amplitude'], result['noise_on_block'])
    result['snr_amplitude'] = _safe_divide(result['amplitude'], result['eamplitude'])

    k = np.sqrt(2.0 * np.pi)
    result['area'] = result['amplitude'] * result['sigma'] * k
    result['earea'] = np.sqrt(
        (result['sigma'] * k * result['eamplitude']) ** 2
        +
        (result['amplitude'] * k * result['esigma']) ** 2
    )
    result['snr_area'] = _safe_divide(result['area'], result['earea'])

    ew_vals = []
    for row in result.itertuples(index=False):
        center = float(row.center)
        sigma = float(row.sigma)
        mask = (spectrum.energy >= center - 2.0 * sigma) & (spectrum.energy <= center + 2.0 * sigma)
        valid_mask = mask & np.isfinite(base) & (base != 0)
        if np.any(valid_mask):
            ew = np.sum(yfit[valid_mask] / base[valid_mask] * spectrum.bin_width[valid_mask]) * 1000.0
        else:
            ew = np.nan
        ew_vals.append(ew)

    result['ew'] = ew_vals
    result = result[FINAL_OUTPUT_COLUMNS]
    high_snr = _snr_confidence_mask(result, snr_confidence_threshold)
    result.loc[high_snr, 'cluster_probability'] = 1.0
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
    spectra_or_energy,
    y=None,
    sy=None,
    *,
    bin_width=None,
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
    spectra_or_energy : str, pathlib.Path, Spectrum-like object, or array-like
        Same spectrum input accepted by the main BLiSS pipeline.
    y, sy : array-like or None, default: None
        Spectral values and uncertainties for direct array input.
    bin_width : array-like or None, default: None
        Optional bin widths for direct array input.
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

    spectrum = prepare_spectrum(
        spectra_or_energy,
        y=y,
        sy=sy,
        bin_width=bin_width,
    )

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
        spectra_or_energy,
        y=None,
        sy=None,
        *,
        en1: Optional[float] = None,
        en2: Optional[float] = None,
        energy_pad: Optional[float] = None,
        final_fit: bool = False,
        show_plot: bool = False,
        output_dir=None,
        plot_name: str = 'bliss_fit.png',
    ) -> pd.DataFrame:
        """Execute the BLiSS workflow.The empirical baseline is computed from the full input spectrum.
        Candidate detection, local Gaussian fitting, probability estimation,
        and the optional global fit are performed inside the selected energy
        interval enlarged by ``energy_pad``. The returned catalogue is finally
        restricted to the nominal ``en1``--``en2`` interval.
        """

        output_dir = ensure_output_folder(output_dir)

        # ------------------------------------------------------------
        # Load full spectrum
        # ------------------------------------------------------------
        spectrum_full = self._load_input(spectra_or_energy, y, sy)

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
        base_full = base_calculator(spectrum_full.values)
        ylines_full = np.maximum(spectrum_full.values - base_full, 0)

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
        )

        # ------------------------------------------------------------
        # Synthetic residual spectra in the same padded interval
        # ------------------------------------------------------------
        simx, simy, simsy, simbase, simylines = calculate_synthetic_noise_spectra(
            spectrum.energy,
            spectrum.values,
            spectrum.uncertainties,
            base,
            self.config.num_synthetic_simulations,
            seed=self.config.synthetic_seed,
        )

        

        synthetic_candidates = return_raw_lines(
            simx,
            simy,
            simsy,
            simylines,
            simbase,
        )

        candidates = eval_line_probability_gmm(
            raw_candidates,
            synthetic_candidates,
            simx=simx,
            x=spectrum.energy,
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
    def _load_input(self, spectra_or_energy, y=None, sy=None) -> PreparedSpectrum:
        """Load and sort spectrum data from a file or direct arrays."""
        return prepare_spectrum(spectra_or_energy, y=y, sy=sy)

    def _select_candidates(self, candidates: pd.DataFrame, *, en1: float, en2: float) -> pd.DataFrame:
        """Restrict candidate lines to the requested energy interval."""
        clean_lines = candidates.copy()
        clean_lines = clean_lines[
            (clean_lines.center >= en1) & (clean_lines.center <= en2)
        ].reset_index(drop=True)
        clean_lines = _add_candidate_metrics(clean_lines)

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


def find_candidate_lines(
    spectra_or_energy,
    y=None,
    sy=None,
    en1=0,
    en2=10,
    energy_pad=0.0,
    output_dir=None,
):
    """Run BLiSS only up to candidate detection/probability estimation."""
    config = BlindLineSearchConfig(
        en1=en1,
        en2=en2,
        energy_pad=energy_pad,
    )
    pipeline = BlindLineSearchPipeline(config=config)
    return pipeline.run(
        spectra_or_energy,
        y=y,
        sy=sy,
        final_fit=False,
        output_dir=output_dir,
    )


def find_emission_lines(
    spectra_or_energy,
    y=None,
    sy=None,
    en1=0,
    en2=10,
    energy_pad=0.0,
    show_plot=False,
    output_dir=None,
    plot_name='bliss_fit.png',
    *,
    final_fit: bool = False,
):
    """Run the default BLiSS emission-line search."""
    config = BlindLineSearchConfig(
        en1=en1,
        en2=en2,
        energy_pad=energy_pad,
    )
    pipeline = BlindLineSearchPipeline(config=config)
    return pipeline.run(
        spectra_or_energy,
        y=y,
        sy=sy,
        final_fit=final_fit,
        show_plot=show_plot,
        output_dir=output_dir,
        plot_name=plot_name,
    )
