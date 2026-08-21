"""Public API for BLiSS spectral line-search tools."""

from .spectrum_data.rebinning_tools import rebin_bins, rebin_snr, rebin_resolution
from .spectrum_data.text_spectrum_loader import load_text_spectrum
from .spectrum_data.fits_spectrum_loader import load_fits_spectrum
from .spectrum_data.spectrum_container import Spectrum

from .line_search.empirical_baseline import base_calculator
from .line_search.peak_selection import find_peaks_new
from .line_search.candidate_regions import CandidateRegionDetector, return_raw_lines
from .line_search.gaussian_models import gaussian, n_gaussian, p0_generator, p0_generator_final
from .line_search.blind_line_search import (
    BlindLineSearchPipeline,
    PreparedSpectrum,
    prepare_spectrum,
    find_candidate_lines,
    find_emission_lines,
    fit_global,
    final_fit_and_metrics,
    plot_global_fit,
)

from .synthetic_probability.synthetic_spectra import (
    SyntheticSpectrumGenerator,
    calculate_synthetic_lines_spectra,
)
from .synthetic_probability.gmm_probability import (
    GMMLineProbabilityEvaluator,
    real_probability,
    eval_line_probability_gmm,
)

from .line_identification.line_identifier import (
    LineIdentifier,
    identify_line,
    add_most_probable_ion,
    get_all_compatible_lines,
)

from .plotting.line_probability_plotter import plot_line_prob
from .plotting.run_output_manager import create_bliss_results_folder, ensure_output_folder
from .isis_interface import (
    write_isis_line_model_files,
    write_isis_files_from_bliss_results,
    clean_zero_area_egauss_model,
    run_bliss_for_isis,
)

__all__ = [
    "Spectrum",
    "PreparedSpectrum",
    "load_text_spectrum",
    "load_fits_spectrum",
    "rebin_bins",
    "rebin_snr",
    "rebin_resolution",
    "base_calculator",
    "find_peaks_new",
    "CandidateRegionDetector",
    "return_raw_lines",
    "gaussian",
    "n_gaussian",
    "p0_generator",
    "p0_generator_final",
    "BlindLineSearchPipeline",
    "prepare_spectrum",
    "find_candidate_lines",
    "find_emission_lines",
    "fit_global",
    "final_fit_and_metrics",
    "plot_global_fit",
    "SyntheticSpectrumGenerator",
    "calculate_synthetic_lines_spectra",
    "GMMLineProbabilityEvaluator",
    "real_probability",
    "eval_line_probability_gmm",
    "LineIdentifier",
    "identify_line",
    "add_most_probable_ion",
    "get_all_compatible_lines",
    "plot_line_prob",
    "create_bliss_results_folder",
    "ensure_output_folder",
    "write_isis_line_model_files",
    "write_isis_files_from_bliss_results",
    "clean_zero_area_egauss_model",
    "run_bliss_for_isis",
]
