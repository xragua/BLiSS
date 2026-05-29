"""Command-line entry points for running BLiSS on ISIS-exported spectra."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from bliss import find_emission_lines
from bliss.spectrum_data.text_spectrum_loader import load_text_spectrum
from bliss.isis_interface.isis_script_writer import write_isis_line_model_files, spectrum_axis_is_keV

def run_bliss_for_isis(spectrum_path: str | Path='spec_0.dat', output_dir: str | Path='.', model_name: str='', *, show_plot: bool=False):
    """Run BLiSS on an ISIS text spectrum and create ISIS line-model files.

    Parameters
    ----------
    spectrum_path : str or pathlib.Path, default: "spec_0.dat"
        Four-column spectrum exported by ISIS, with lower bin edge, upper bin edge,
        value, and uncertainty.
    output_dir : str or pathlib.Path, default: "."
        Folder where the candidate CSV and ISIS S-Lang files are written.
    model_name : str, default: ""
        Suffix used in generated filenames.
    show_plot : bool, default: False
        If true, request the BLiSS diagnostic plot from the underlying pipeline.

    Returns
    -------
    dict of str to pathlib.Path
        Paths to the generated ISIS model, parameter, and candidate files.
    """
    spectrum_path = Path(spectrum_path)
    output_dir = Path(output_dir)
    x, y, sy, _bin_width = load_text_spectrum(spectrum_path)
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(sy) & (y > 0) & (sy > 0)
    candidates = find_emission_lines(x[good], y[good], sy[good], show_plot=show_plot)
    use_egauss = spectrum_axis_is_keV(spectrum_path)
    files = write_isis_line_model_files(candidates, output_dir=output_dir, model_name=model_name, use_egauss=use_egauss)
    candidates.to_csv(output_dir / f"clean_lines_{model_name or 'default'}.csv", index=False)
    return files

def main() -> None:
    """Parse command-line arguments and run the ISIS export workflow."""
    parser = argparse.ArgumentParser(description='Run modular BLiSS and write ISIS S-Lang line-model files.')
    parser.add_argument('model_name', nargs='?', default='', help='Suffix for set_line_model_<name>.sl and set_line_parameters_<name>.sl')
    parser.add_argument('--spectrum', default='spec_0.dat', help='ISIS write_plot spectrum file, default: spec_0.dat')
    parser.add_argument('--output-dir', default='.', help='Folder where ISIS helper files are written')
    parser.add_argument('--plot', action='store_true', help='Save/display BLiSS diagnostic plot if supported')
    args = parser.parse_args()
    run_bliss_for_isis(args.spectrum, args.output_dir, args.model_name, show_plot=args.plot)
if __name__ == '__main__':
    main()
