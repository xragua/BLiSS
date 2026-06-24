"""Example: load a PHA/background/RMF spectrum with astropy and run BLiSS.

Run from the project root:
    python3 examples/run_fits_loader_example.py
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bliss import load_fits_spectrum, find_emission_lines, create_bliss_results_folder
HERE = Path(__file__).resolve().parent
DATA = HERE / 'fits_spectrum_data'
OUT = create_bliss_results_folder(ROOT / 'results', 'bliss')
spectrum = load_fits_spectrum(DATA / 'med4.ds', background_path=DATA / 'bkglp.fits', rmf_path=DATA / 'r_plateau.rmf')
candidates = find_emission_lines(spectrum.energy, spectrum.values, spectrum.uncertainties, show_plot=True, output_dir=OUT, plot_name='fits_loader_bliss_fit.png')
candidates.to_csv(OUT / 'fits_loader_candidates.csv', index=False)
print('FITS loader example completed.')
print(f'Output folder: {OUT}')
print(f'Candidates: {len(candidates)}')
