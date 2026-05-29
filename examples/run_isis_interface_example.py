"""Example: run BLiSS and export ISIS-compatible line model files.

Run from the project root:
    python3 examples/run_isis_interface_example.py
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from bliss import load_text_spectrum, rebin_snr, find_emission_lines, create_bliss_results_folder, write_isis_files_from_bliss_results
HERE = Path(__file__).resolve().parent
OUT = create_bliss_results_folder(ROOT / 'results', 'bliss')
energy, values, errors, _ = load_text_spectrum(HERE / 'xmm.dat')
energy, values, errors = rebin_snr(energy, values, errors, 0.8)
candidates = find_emission_lines(energy, values, errors, show_plot=True, output_dir=OUT, plot_name='isis_input_bliss_fit.png')
isis_files = write_isis_files_from_bliss_results(candidates, OUT, model_name='xmm_bliss', text_spectrum_path=HERE / 'xmm.dat')
pd.Series({k: str(v) for k, v in isis_files.items()}).to_csv(OUT / 'isis_exported_files.txt', header=False)
print('ISIS interface example completed.')
print(f'Output folder: {OUT}')
for key, path in isis_files.items():
    print(f'{key}: {path}')
