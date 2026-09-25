"""Public API for BLiSS spectral line-search tools."""

from .spectrum_data.rebinning_tools import rebin_counts, apply_groups
from .spectrum_data.fits_spectrum_loader import load_fits_spectrum, read_pha_metadata, align_to_spectrum
from .spectrum_data.spectrum_container import Spectrum

from .line_search.empirical_baseline import base_calculator
from .line_search.peak_selection import find_peaks_new
from .line_search.candidate_regions import CandidateRegionDetector, return_raw_lines
from .line_search.gaussian_models import gaussian, n_gaussian, p0_generator, p0_generator_final
from .line_search.blind_line_search import (
    BlindLineSearchConfig,
    BlindLineSearchPipeline,
    NativeCounts,
    PreparedSpectrum,
    prepare_spectrum,
    find_candidate_lines,
    find_emission_lines,
    fit_global,
    final_fit_and_metrics,
    plot_global_fit,
)

from .synthetic_probability.synthetic_spectra import (
    NullRealization,
    generate_null_realizations,
)
from .synthetic_probability.gmm_score import (
    GMMBlissScoreEvaluator,
    calculate_bliss_score,
    eval_bliss_score_gmm,
)

from .line_identification.line_identifier import (
    LineIdentifier,
    identify_line,
    add_most_probable_ion,
    get_all_compatible_lines,
)

from .plotting.line_score_plotter import plot_bliss_score
from .plotting.run_output_manager import create_bliss_results_folder, ensure_output_folder


__all__ = [
    "Spectrum",
    "PreparedSpectrum",
    "NativeCounts",
    "load_fits_spectrum",
    "read_pha_metadata",
    "align_to_spectrum",
    "rebin_counts",
    "apply_groups",
    "base_calculator",
    "find_peaks_new",
    "CandidateRegionDetector",
    "return_raw_lines",
    "gaussian",
    "n_gaussian",
    "p0_generator",
    "p0_generator_final",
    "BlindLineSearchConfig",
    "BlindLineSearchPipeline",
    "prepare_spectrum",
    "find_candidate_lines",
    "find_emission_lines",
    "fit_global",
    "final_fit_and_metrics",
    "plot_global_fit",
    "NullRealization",
    "generate_null_realizations",
    "GMMBlissScoreEvaluator",
    "calculate_bliss_score",
    "eval_bliss_score_gmm",
    "LineIdentifier",
    "identify_line",
    "add_most_probable_ion",
    "get_all_compatible_lines",
    "plot_bliss_score",
    "create_bliss_results_folder",
    "ensure_output_folder",
    #"write_isis_line_model_files",
    #"write_isis_files_from_bliss_results",
    #"clean_zero_area_egauss_model",
    #"run_bliss_for_isis",
]

from .score_columns import normalize_score_columns, read_bliss_csv
__all__ += ["normalize_score_columns", "read_bliss_csv"]
