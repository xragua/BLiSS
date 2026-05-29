"""Write ISIS/S-Lang model and parameter files from BLiSS candidate tables."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

def spectrum_axis_is_keV(text_spectrum_path: str | Path) -> bool:
    """Check whether a text spectrum appears to use keV as its energy unit.

    Parameters
    ----------
    text_spectrum_path : str or pathlib.Path
        Spectrum file whose first header lines are inspected for the string
        ``keV``.

    Returns
    -------
    bool
        ``True`` when the header mentions keV, or when the file is unavailable;
        ``False`` otherwise. The default ``True`` preserves the expected behaviour
        for energy-space ISIS spectra.
    """
    path = Path(text_spectrum_path)
    try:
        return any(('(keV)' in line or 'keV' in line for line in path.read_text(errors='replace').splitlines()[:20]))
    except FileNotFoundError:
        return True

def _component_name(use_egauss: bool) -> str:
    """Return the ISIS Gaussian component name to write.

    Parameters
    ----------
    use_egauss : bool
        Whether candidate centers are in energy units and should therefore use
        ISIS ``egauss`` components instead of wavelength-space ``gauss`` components.

    Returns
    -------
    str
        ``"egauss"`` when ``use_egauss`` is true; otherwise ``"gauss"``.
    """
    return 'egauss' if use_egauss else 'gauss'

def _prepare_candidates(candidates: pd.DataFrame, add_sentinel: bool=True) -> pd.DataFrame:
    """Validate and optionally extend the candidate-line table for ISIS output.

    Parameters
    ----------
    candidates : pandas.DataFrame
        BLiSS candidate table. It must contain ``center``, ``ecenter``, ``sigma``,
        and ``esigma`` columns because these values are written into ISIS
        parameter bounds.
    add_sentinel : bool, default: True
        If true, append a dummy high-energy line with zero width and uncertainty.
        This reproduces the extra terminal line expected by the ISIS helper
        scripts.

    Returns
    -------
    pandas.DataFrame
        Reset-index copy of the candidate table, with the optional sentinel row
        appended.

    Raises
    ------
    ValueError
        If one or more required candidate columns are missing.
    """
    required = {'center', 'ecenter', 'sigma', 'esigma'}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f'Candidate table is missing required columns: {sorted(missing)}')
    clean = candidates.copy().reset_index(drop=True)
    if add_sentinel:
        sentinel = {col: 0 for col in clean.columns}
        sentinel.update({'center': 100000000.0, 'ecenter': 0.0, 'sigma': 0.0, 'esigma': 0.0})
        clean.loc[len(clean)] = sentinel
    return clean

def _safe_model_suffix(model_name: str | None) -> str:
    """Normalize the suffix used in generated ISIS filenames.

    Parameters
    ----------
    model_name : str or None
        User-supplied suffix placed after ``set_line_model_`` and
        ``set_line_parameters_``.

    Returns
    -------
    str
        Empty string when ``model_name`` is ``None``; otherwise the string version
        of ``model_name``.
    """
    return '' if model_name is None else str(model_name)

def write_isis_line_model_files(candidates: pd.DataFrame, output_dir: str | Path, *, model_name: str | None='', use_egauss: bool=True, add_sentinel_line: bool=True, area_initial_value: float=0.0, area_min: float=0.0, area_max: float=100000000.0, sigma_initial_value: float=0.0001, sigma_min: float=0.0, sigma_max: float=0.01) -> dict[str, Path]:
    """Write ISIS/S-Lang files describing BLiSS Gaussian candidates.

    Parameters
    ----------
    candidates : pandas.DataFrame
        Candidate-line table returned by BLiSS. The writer uses ``center`` and
        ``ecenter`` for the line-center parameter and validates that ``sigma`` and
        ``esigma`` are also present.
    output_dir : str or pathlib.Path
        Directory where the S-Lang model file, parameter file, and candidate CSV
        are written.
    model_name : str or None, default: ""
        Filename suffix used for ``set_line_model_<model_name>.sl`` and
        ``set_line_parameters_<model_name>.sl``.
    use_egauss : bool, default: True
        Selects ISIS ``egauss`` components when true and ``gauss`` components when
        false.
    add_sentinel_line : bool, default: True
        Append a dummy high-energy line to the candidate table before writing the
        files.
    area_initial_value : float, default: 0.0
        Initial value assigned to each Gaussian area parameter.
    area_min : float, default: 0.0
        Lower bound assigned to each Gaussian area parameter.
    area_max : float, default: 100000000.0
        Upper bound assigned to each Gaussian area parameter.
    sigma_initial_value : float, default: 0.0001
        Initial value assigned to each Gaussian width parameter.
    sigma_min : float, default: 0.0
        Lower bound assigned to each Gaussian width parameter.
    sigma_max : float, default: 0.01
        Upper bound assigned to each Gaussian width parameter.

    Returns
    -------
    dict of str to pathlib.Path
        Paths to the generated model file, parameter file, and cleaned candidate
        CSV, with keys ``model_file``, ``parameter_file``, and ``candidate_csv``.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    clean = _prepare_candidates(candidates, add_sentinel=add_sentinel_line)
    component = _component_name(use_egauss)
    suffix = _safe_model_suffix(model_name)
    model_file = output / f'set_line_model_{suffix}.sl'
    parameter_file = output / f'set_line_parameters_{suffix}.sl'
    clean_csv = output / f"isis_candidates_{suffix or 'default'}.csv"
    terms = '+'.join([f'{component}({i}) \n' for i in range(1, len(clean) + 1)])
    model_file.write_text(f'public define linemodel(){{\n\t{terms};\n}}\n\n', encoding='utf-8')
    with parameter_file.open('w', encoding='utf-8') as handle:
        for i, row in clean.iterrows():
            center = float(row['center'])
            ecenter = float(row.get('ecenter', 0.0))
            handle.write(f'set_par("{component}({i + 1}).center", {center}, 1, {max(center - 2 * ecenter, 0)}, {center + 2 * ecenter});\n')
        handle.write('\n')
        for i in range(len(clean)):
            handle.write(f'set_par("{component}({i + 1}).area", {area_initial_value}, 1, {area_min}, {area_max});\n')
        handle.write('\n')
        for i in range(len(clean)):
            handle.write(f'set_par("{component}({i + 1}).sigma", {sigma_initial_value}, 1, {sigma_min}, {sigma_max});\n')
    clean.to_csv(clean_csv, index=False)
    return {'model_file': model_file, 'parameter_file': parameter_file, 'candidate_csv': clean_csv}

def write_isis_files_from_bliss_results(candidates: pd.DataFrame, output_dir: str | Path, *, model_name: str | None='', text_spectrum_path: str | Path | None=None) -> dict[str, Path]:
    """Write ISIS helper files using the component type implied by the spectrum.

    Parameters
    ----------
    candidates : pandas.DataFrame
        BLiSS candidate table to convert into ISIS Gaussian components.
    output_dir : str or pathlib.Path
        Directory where the generated ISIS files are saved.
    model_name : str or None, default: ""
        Suffix included in generated filenames.
    text_spectrum_path : str, pathlib.Path, or None, default: None
        Optional spectrum file used to decide whether the x-axis is in keV. If no
        file is supplied, ``egauss`` components are used.

    Returns
    -------
    dict of str to pathlib.Path
        Paths returned by :func:`write_isis_line_model_files`.
    """
    use_egauss = True if text_spectrum_path is None else spectrum_axis_is_keV(text_spectrum_path)
    return write_isis_line_model_files(candidates, output_dir, model_name=model_name, use_egauss=use_egauss)
