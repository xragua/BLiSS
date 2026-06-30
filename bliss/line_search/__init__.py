"""Core algorithms for baseline estimation, peak selection, and line fitting."""

from .blind_line_search import (
    BlindLineSearchConfig,
    BlindLineSearchPipeline,
    PreparedSpectrum,
    final_fit_and_metrics,
    find_candidate_lines,
    find_emission_lines,
    fit_global,
    plot_global_fit,
    prepare_spectrum,
)

__all__ = [
    'BlindLineSearchConfig',
    'BlindLineSearchPipeline',
    'PreparedSpectrum',
    'final_fit_and_metrics',
    'find_candidate_lines',
    'find_emission_lines',
    'fit_global',
    'plot_global_fit',
    'prepare_spectrum',
]
