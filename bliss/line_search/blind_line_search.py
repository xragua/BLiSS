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
from ..synthetic_probability.synthetic_spectra import calculate_synthetic_lines_spectra
from ..synthetic_probability.gmm_probability import eval_line_probability_gmm
from ..plotting.run_output_manager import ensure_output_folder
CANDIDATE_COLUMNS = ['center', 'ecenter', 'sigma', 'esigma', 'amplitude', 'eamplitude', 'relative_power', 'rsq', 'noise_on_block', 'value_on_line', 'base_on_line', 'cluster_probability']
FINAL_OUTPUT_COLUMNS = ['center', 'sigma', 'amplitude', 'ecenter', 'esigma', 'eamplitude', 'base_on_line', 'value_on_line', 'noise_on_block', 'snr_peak','snr_area', 'relative_power', 'area', 'earea', 'ew', 'cluster_probability']

@dataclass
class BlindLineSearchConfig:
    """Configuration values controlling the default BLiSS pipeline.

    Attributes
    ----------
    en1, en2 : float
        Lower and upper energy limits used when selecting final candidates.
    num_synthetic_simulations : int
        Number of shuffled synthetic spectra generated for probability estimation.
    final_fit_maxfev : int
        Maximum number of function evaluations allowed in the final ``curve_fit``.
    snr_confidence_threshold : float
        Signal-to-noise ratio above which a candidate is assigned probability 1.
    """
    en1: float = 0.0
    en2: float = 10.0
    num_synthetic_simulations: int = 2
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

class BlindLineSearchPipeline:
    """Run the full blind emission-line search on one spectrum.

    The pipeline estimates an empirical baseline, searches positive residuals for
    Gaussian-like line candidates, estimates false-positive probability using
    synthetic spectra, refits the selected candidates together, and writes compact
    run outputs.
    """

    def __init__(self, config: Optional[BlindLineSearchConfig]=None):
        """Create a pipeline instance.

        Parameters
        ----------
        config : BlindLineSearchConfig or None, default: None
            Pipeline configuration. When omitted, the default BLiSS search limits and
            fitting settings are used.
        """
        self.config = config or BlindLineSearchConfig()

    def run(self, spectra_or_energy, y=None, sy=None, *, en1: Optional[float]=None, en2: Optional[float]=None, show_plot: bool=False, output_dir=None, plot_name: str='bliss_fit.png') -> pd.DataFrame:
        """Execute the BLiSS line-search workflow.

        Parameters
        ----------
        spectra_or_energy : str, pathlib.Path, or array-like
            Either a four-column text spectrum file or the coordinate array for direct
            array input.
        y : array-like or None, default: None
            Observed spectral values. Required when ``spectra_or_energy`` is an array.
        sy : array-like or None, default: None
            One-sigma uncertainties on ``y``. Required when ``spectra_or_energy`` is an
            array.
        en1 : float or None, default: None
            Lower energy bound for accepted candidates. If omitted, the value from the
            pipeline configuration is used.
        en2 : float or None, default: None
            Upper energy bound for accepted candidates. If omitted, the value from the
            pipeline configuration is used.
        show_plot : bool, default: False
            Save and display the final diagnostic plot when true.
        output_dir : str, pathlib.Path, or None, default: None
            Folder used for ``candidate_lines.csv`` and ``run_summary.txt``. If omitted,
            a timestamped results folder is created.
        plot_name : str, default: "bliss_fit.png"
            Filename for the diagnostic plot inside ``output_dir``.

        Returns
        -------
        pandas.DataFrame
            Final BLiSS candidate table with fitted parameters, uncertainties,
            signal-to-noise ratio, equivalent width, and cluster probability.
        """
        output_dir = ensure_output_folder(output_dir)
        spectrum = self._load_input(spectra_or_energy, y, sy)
        base = base_calculator(spectrum.values)
        ylines = np.maximum(spectrum.values - base, 0)
        raw_candidates = return_raw_lines(spectrum.energy, spectrum.values, spectrum.uncertainties, ylines, base)
        simx, simy, simsy = calculate_synthetic_lines_spectra(spectrum.energy, ylines, spectrum.uncertainties, self.config.num_synthetic_simulations)
        synthetic_candidates = return_raw_lines(simx, simy, simsy, simy, np.zeros(len(simx)))
        candidates = eval_line_probability_gmm(raw_candidates, synthetic_candidates, simx=simx, x=spectrum.energy)
        selected = self._select_candidates(candidates, en1=self.config.en1 if en1 is None else en1, en2=self.config.en2 if en2 is None else en2)
        result, yfit = self._final_fit_and_metrics(spectrum=spectrum, base=base, ylines=ylines, clean_lines=selected)
        if show_plot:
            self._plot_final_fit(spectrum=spectrum, base=base, yfit=yfit, output_path=output_dir / plot_name)
        self._write_outputs(result, output_dir)
        return result

    def _load_input(self, spectra_or_energy, y=None, sy=None) -> PreparedSpectrum:
        """Load and sort spectrum data from a file or direct arrays.

        Parameters
        ----------
        spectra_or_energy : str, pathlib.Path, or array-like
            Four-column file with ``E_low, E_high, value, uncertainty`` columns, or an
            array of coordinate values.
        y : array-like or None, default: None
            Spectral values for direct array input.
        sy : array-like or None, default: None
            One-sigma uncertainties for direct array input.

        Returns
        -------
        PreparedSpectrum
            Spectrum sorted by increasing coordinate value.
        """
        if isinstance(spectra_or_energy, (str, Path)):
            spectra = pd.read_csv(spectra_or_energy, sep='\\s+', comment='#', header=None, engine='python')
            if spectra.shape[1] != 4:
                raise ValueError('File must have 4 columns: E_low, E_high, counts, error.')
            x = np.asarray((spectra[0] + spectra[1]) / 2.0)
            dE = np.asarray(spectra[1] - spectra[0])
            y = np.asarray(spectra[2])
            sy = np.asarray(spectra[3])
        else:
            x = np.asarray(spectra_or_energy)
            if y is None or sy is None:
                raise ValueError('If using direct arrays, y and sy must be provided.')
            y = np.asarray(y)
            sy = np.asarray(sy)
            dE = np.diff(x)
            dE = np.append(dE, dE[-1])
        order = np.argsort(x)
        return PreparedSpectrum(energy=x[order], values=y[order], uncertainties=sy[order], bin_width=np.asarray(dE)[order])

    def _select_candidates(self, candidates: pd.DataFrame, *, en1: float, en2: float) -> pd.DataFrame:
        """Restrict candidate lines to the requested energy interval.

        Parameters
        ----------
        candidates : pandas.DataFrame
            Candidate table produced by probability evaluation.
        en1, en2 : float
            Inclusive lower and upper energy bounds applied to the ``center`` column.

        Returns
        -------
        pandas.DataFrame
            Candidate table restricted to the expected BLiSS output columns.
        """
        clean_lines = candidates
        clean_lines = clean_lines[(clean_lines.center >= en1) & (clean_lines.center <= en2)].reset_index(drop=True)
        clean_lines = clean_lines[CANDIDATE_COLUMNS]
        return clean_lines

    def _final_fit_and_metrics(self, spectrum: PreparedSpectrum, base: np.ndarray, ylines: np.ndarray, clean_lines: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Refit selected candidates and compute final line diagnostics.

        Parameters
        ----------
        spectrum : PreparedSpectrum
            Sorted spectrum used for the final fit.
        base : numpy.ndarray
            Empirical baseline evaluated on ``spectrum.energy``.
        ylines : numpy.ndarray
            Baseline-subtracted spectrum fitted with the multi-Gaussian model.
        clean_lines : pandas.DataFrame
            Candidate lines selected after probability filtering and energy clipping.

        Returns
        -------
        tuple
            ``(result, yfit)`` where ``result`` is the final candidate DataFrame and
            ``yfit`` is the fitted multi-Gaussian line model on the full spectrum grid.
        """
        fitted_final = pd.DataFrame(columns=['amplitude', 'center', 'sigma', 'eamplitude', 'ecenter', 'esigma'])
        yfit = np.zeros_like(spectrum.energy, dtype=float)
        if len(clean_lines) > 0:
            p0, bounds = p0_generator_final(spectrum.energy, spectrum.values, clean_lines)
            try:
                popt, pcov = curve_fit(n_gaussian, spectrum.energy, ylines, p0=p0, bounds=bounds, maxfev=self.config.final_fit_maxfev)
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
        clean_select = clean_lines[['relative_power', 'noise_on_block', 'value_on_line', 'base_on_line', 'cluster_probability']]
        result = pd.concat([fitted_final, clean_select], axis=1)
        if len(result) == 0:
            return (pd.DataFrame(columns=FINAL_OUTPUT_COLUMNS), yfit)
        cols = ['amplitude', 'sigma', 'eamplitude', 'esigma']
        result[cols] = result[cols].apply(pd.to_numeric, errors='coerce')
        result['snr_peak'] = result.amplitude / result.noise_on_block
        k = np.sqrt(2.0 * np.pi)
        result['area'] = result['amplitude'] * result['sigma'] * k
        result['earea'] = np.sqrt((result['sigma'] * k * result['eamplitude']) ** 2 + (result['amplitude'] * k * result['esigma']) ** 2)
        result['snr_area'] = result.area / result.earea
        ew_vals = []
        for row in result.itertuples(index=False):
            center = float(row.center)
            sigma = float(row.sigma)
            mask = (spectrum.energy >= center - 2.0 * sigma) & (spectrum.energy <= center + 2.0 * sigma)
            if np.any(mask):
                ew = np.sum(yfit[mask] / base[mask] * spectrum.bin_width[mask]) * 1000.0
            else:
                ew = np.nan
            ew_vals.append(ew)
        result['ew'] = ew_vals
        result = result[FINAL_OUTPUT_COLUMNS]
        result.loc[result.snr_peak >= self.config.snr_confidence_threshold, 'cluster_probability'] = 1
        return (result, yfit)

    def _plot_final_fit(self, spectrum: PreparedSpectrum, base: np.ndarray, yfit: np.ndarray, output_path: str | Path) -> None:
        """Save the final spectrum, baseline, and line-model diagnostic plot.

        Parameters
        ----------
        spectrum : PreparedSpectrum
            Spectrum plotted with uncertainty bars.
        base : numpy.ndarray
            Empirical baseline curve.
        yfit : numpy.ndarray
            Fitted line-only model.
        output_path : str or pathlib.Path
            Destination image path.
        """
        plt.figure()
        plt.errorbar(spectrum.energy, spectrum.values, yerr=spectrum.uncertainties, label='Data', alpha=0.2)
        plt.plot(spectrum.energy, base, 'k:', label='base')
        plt.plot(spectrum.energy, yfit, 'g:', label='Lines')
        plt.plot(spectrum.energy, yfit + base, 'r', label='Line+base')
        plt.xlabel('Energy (keV)')
        plt.ylabel('Spectra')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()

    def _write_outputs(self, result: pd.DataFrame, output_dir: Path) -> None:
        """Write the final candidate table and run summary.

        Parameters
        ----------
        result : pandas.DataFrame
            Final BLiSS candidate table.
        output_dir : pathlib.Path
            Directory where ``candidate_lines.csv`` and ``run_summary.txt`` are saved.
        """
        result.to_csv(output_dir / 'candidate_lines.csv', index=False)
        with open(output_dir / 'run_summary.txt', 'w') as handle:
            handle.write('BLiSS run completed\n')
            handle.write(f'Results folder: {output_dir}\n')
            handle.write(f'Number of detected candidates: {len(result)}\n')

def find_emission_lines(spectra_or_energy, y=None, sy=None, en1=0, en2=10, show_plot=False, output_dir=None, plot_name='bliss_fit.png'):
    """Run the default BLiSS emission-line search.

    Parameters
    ----------
    spectra_or_energy : str, pathlib.Path, or array-like
        Four-column spectrum file or coordinate array.
    y : array-like or None, default: None
        Spectral values for direct array input.
    sy : array-like or None, default: None
        One-sigma uncertainties for direct array input.
    en1, en2 : float, default: 0, 10
        Inclusive energy interval used to keep final candidates.
    show_plot : bool, default: False
        Save and display the final diagnostic plot when true.
    output_dir : str, pathlib.Path, or None, default: None
        Output folder for BLiSS result files.
    plot_name : str, default: "bliss_fit.png"
        Diagnostic plot filename.

    Returns
    -------
    pandas.DataFrame
        Final candidate-line table returned by ``BlindLineSearchPipeline.run``.
    """
    config = BlindLineSearchConfig(en1=en1, en2=en2)
    pipeline = BlindLineSearchPipeline(config=config)
    return pipeline.run(spectra_or_energy, y=y, sy=sy, show_plot=show_plot, output_dir=output_dir, plot_name=plot_name)
