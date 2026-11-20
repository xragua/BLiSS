"""Diagnostic plots comparing data, baseline, and fitted line models."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from .run_output_manager import _check_output_paths, _reserve_output_paths, _plot_output_path

def plot_final_bliss_fit(energy: np.ndarray, values: np.ndarray, uncertainties: np.ndarray, baseline: np.ndarray, line_model: np.ndarray, output_path: str | Path, *, overwrite: bool=False):
    """Save a diagnostic plot of the data, baseline, and fitted line model.

    Parameters
    ----------
    energy : numpy.ndarray
        Spectral coordinate grid.
    values : numpy.ndarray
        Observed spectral values.
    uncertainties : numpy.ndarray
        One-sigma uncertainties plotted as error bars.
    baseline : numpy.ndarray
        Empirical continuum baseline.
    line_model : numpy.ndarray
        Fitted line-only model evaluated on ``energy``.
    output_path : str or pathlib.Path
        Destination image path.
    overwrite : bool, default: False
        Allow replacement of an existing image. Otherwise raise FileExistsError.
    """
    output_path = _plot_output_path(output_path)
    _check_output_paths([output_path], overwrite=overwrite)
    plt.figure(figsize=(10, 5))
    plt.errorbar(energy, values, yerr=uncertainties, fmt='-', alpha=0.2, label='Data')
    plt.plot(energy, baseline, 'k:', linewidth=2, label='Base')
    plt.plot(energy, line_model, 'g:', linewidth=2, label='Lines')
    plt.plot(energy, line_model + baseline, 'r', linewidth=2, label='Line + base')
    plt.xlabel('Energy (keV)')
    plt.ylabel('Spectral value')
    plt.legend()
    plt.tight_layout()
    try:
        with _reserve_output_paths([output_path], overwrite=overwrite):
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
    finally:
        plt.close()
