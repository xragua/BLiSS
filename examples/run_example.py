"""
Run from the project root:
    python3 examples/run_example.py
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bliss import rebin_bins, rebin_resolution, rebin_snr, find_emission_lines, add_most_probable_ion, plot_line_prob, create_bliss_results_folder
HERE = Path(__file__).resolve().parent
OUT = create_bliss_results_folder(ROOT / 'results', 'bliss')

def save_current_figure(name):
    """Save the active Matplotlib figure into the example output folder.

    Parameters
    ----------
    name : str
        Filename used inside the timestamped BLiSS results directory.

    Returns
    -------
    None
        The figure is written to disk and then closed.
    """
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=150, bbox_inches='tight')
    plt.close()

def load_four_column_spectrum(path):
    """Load a four-column example spectrum and keep valid positive bins.

    Parameters
    ----------
    path : str or pathlib.Path
        Text file with lower energy edge, upper energy edge, spectral value,
        and one-sigma uncertainty.

    Returns
    -------
    tuple of numpy.ndarray
        Energy-bin centers, spectral values, and uncertainties after removing
        bins with non-positive values or uncertainties.
    """
    data = pd.read_csv(path, sep='\\s+', comment='#', header=None, engine='python')
    x = np.asarray((data[0] + data[1]) / 2)
    y = np.asarray(data[2])
    sy = np.asarray(data[3])
    good = (y > 0) & (sy > 0)
    return (x[good], y[good], sy[good])
xmmx, xmmy, xmmsy = load_four_column_spectrum(HERE / 'xmm.dat')
xifux, xifuy, xifusy = load_four_column_spectrum(HERE / 'xifu.dat')
rxmmx_bins, rxmmy_bins, rxmmsy_bins = rebin_bins(xmmx, xmmy, xmmsy, 1)
plt.figure()
plt.errorbar(rxmmx_bins, rxmmy_bins, yerr=rxmmsy_bins)
plt.xlabel('Energy (keV)')
plt.ylabel('Spectra')
plt.title('Rebined spectra')
save_current_figure('01_xmm_rebin_bins.png')
rxifux, rxifuy, rxifusy = rebin_resolution(xifux, xifuy, xifusy, 0.02)
plt.figure()
plt.errorbar(rxifux, rxifuy, yerr=rxifusy)
plt.xlabel('Energy (keV)')
plt.ylabel('Spectra')
plt.title('Rebined spectra')
save_current_figure('02_xifu_rebin_resolution.png')
rxmmx, rxmmy, rxmmsy = rebin_snr(xmmx, xmmy, xmmsy, 0.8)
plt.figure()
plt.errorbar(rxmmx, rxmmy, yerr=rxmmsy)
plt.xlabel('Energy (keV)')
plt.ylabel('Spectra')
plt.title('Rebined spectra')
save_current_figure('03_xmm_rebin_snr.png')
xmm_lines = find_emission_lines(rxmmx, rxmmy, rxmmsy, show_plot=True, output_dir=OUT, plot_name='04_xmm_bliss_fit.png')
xifu_lines = find_emission_lines(rxifux, rxifuy, rxifusy, show_plot=True, output_dir=OUT, plot_name='05_xifu_bliss_fit.png')
xmm_lines_ident = add_most_probable_ion(xmm_lines, 1200)
xifu_lines_ident = add_most_probable_ion(xifu_lines, 1200)
plot_line_prob(xmm_lines_ident)
save_current_figure('06_xmm_line_probability.png')
plot_line_prob(xifu_lines_ident)
save_current_figure('07_xifu_line_probability.png')
plt.figure()
for i in range(len(xmm_lines_ident)):
    plt.plot(xmm_lines_ident.doppler_kms[i], xmm_lines_ident.best_chi[i], 'bo', alpha=xmm_lines_ident.cluster_probability[i])
    plt.text(xmm_lines_ident.doppler_kms[i] + 10, xmm_lines_ident.best_chi[i] + 0.1, xmm_lines_ident.ion[i], alpha=xmm_lines_ident.cluster_probability[i])
plt.xlim(-1300, 1400)
plt.xlabel('Doppler (km/s)')
plt.ylabel('Best $\\zeta$')
save_current_figure('08_xmm_doppler_best_chi.png')
xmm_lines.to_csv(OUT / 'xmm_lines.csv', index=False)
xifu_lines.to_csv(OUT / 'xifu_lines.csv', index=False)
xmm_lines_ident.to_csv(OUT / 'xmm_lines_identified.csv', index=False)
xifu_lines_ident.to_csv(OUT / 'xifu_lines_identified.csv', index=False)
print('BLiSS example completed.')
print(f'Output folder: {OUT}')
print(f'XMM lines:  {len(xmm_lines)}')
print(f'XIFU lines: {len(xifu_lines)}')