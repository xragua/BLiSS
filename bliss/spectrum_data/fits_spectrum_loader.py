"""Read PHA/RMF/ARF-style FITS spectra into BLiSS spectrum objects."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
from .spectrum_container import Spectrum
POSSIBLE_SPECTRUM_COLUMNS = ['COUNTS', 'RATE', 'COUNT_RATE', 'PHA', 'SRC_COUNTS']
POSSIBLE_ERROR_COLUMNS = ['STAT_ERR', 'ERROR', 'ERR']
POSSIBLE_CHANNEL_COLUMNS = ['CHANNEL', 'PI']

def _import_astropy_fits():
    """Import ``astropy.io.fits`` only when FITS loading is requested.

    Returns
    -------
    module
        The ``astropy.io.fits`` module.

    Raises
    ------
    ImportError
        If Astropy is not installed.
    """
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise ImportError('The FITS spectrum loader requires astropy. Install it with: pip install astropy') from exc
    return fits

def _find_existing_column(table, candidates):
    """Return the first available column from a list of supported names.

    Parameters
    ----------
    table : object
        FITS table HDU whose column names are inspected.
    candidates : sequence of str
        Accepted column names, checked case-insensitively and in order.

    Returns
    -------
    str or None
        Matching candidate name, or ``None`` when none of the requested columns is
        present.
    """
    if table is None:
        return None
    names = [name.upper() for name in table.columns.names or []]
    for candidate in candidates:
        if candidate.upper() in names:
            return candidate
    return None

def _find_spectrum_hdu(hdul):
    """Find the FITS table HDU containing spectrum and channel columns.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        Open FITS file to search.

    Returns
    -------
    astropy.io.fits.BinTableHDU
        First table HDU containing a supported channel column and a supported
        spectrum-value column.

    Raises
    ------
    ValueError
        If no suitable spectrum table is found.
    """
    for hdu in hdul:
        if getattr(hdu, 'data', None) is None:
            continue
        if not hasattr(hdu, 'columns'):
            continue
        names = [name.upper() for name in hdu.columns.names or []]
        has_channel = any((col in names for col in POSSIBLE_CHANNEL_COLUMNS))
        has_spectrum = any((col in names for col in POSSIBLE_SPECTRUM_COLUMNS))
        if has_channel and has_spectrum:
            return hdu
    raise ValueError('Could not find a valid spectrum table in FITS file.')

def _read_ebounds_from_rmf(path: str | Path):
    """Read channel energy bounds from an RMF EBOUNDS extension.

    Parameters
    ----------
    path : str or pathlib.Path
        RMF file containing ``CHANNEL``, ``E_MIN``, and ``E_MAX`` columns.

    Returns
    -------
    tuple of numpy.ndarray
        Channel numbers, lower energy bounds, and upper energy bounds.

    Raises
    ------
    ValueError
        If the RMF file does not contain an EBOUNDS-style table.
    """
    fits = _import_astropy_fits()
    with fits.open(path) as hdul:
        for hdu in hdul:
            if getattr(hdu, 'data', None) is None:
                continue
            if not hasattr(hdu, 'columns'):
                continue
            names = set(hdu.columns.names or [])
            if {'CHANNEL', 'E_MIN', 'E_MAX'}.issubset(names):
                channel = np.asarray(hdu.data['CHANNEL'])
                e_min = np.asarray(hdu.data['E_MIN'], dtype=float)
                e_max = np.asarray(hdu.data['E_MAX'], dtype=float)
                return (channel, e_min, e_max)
    raise ValueError(f'Could not find EBOUNDS extension in RMF file: {path}')

def _read_rmf_resolution(path: str | Path):
    """Estimate the detector line-spread width as a function of energy from an RMF.

    Parameters
    ----------
    path : str or pathlib.Path
        RMF file containing a redistribution MATRIX extension and EBOUNDS table.

    Returns
    -------
    tuple of numpy.ndarray
        Incident-energy bin centers and Gaussian-equivalent instrumental sigma.
        The sigma is calculated as the response-weighted standard deviation of
        detector-channel energies for each incident-energy row.

    Raises
    ------
    ValueError
        If the RMF does not contain the required redistribution information.
    """
    fits = _import_astropy_fits()
    rmf_channel, e_min, e_max = _read_ebounds_from_rmf(path)
    channel_energy = (e_min + e_max) / 2.0
    channel_lookup = {int(ch): i for i, ch in enumerate(rmf_channel)}

    with fits.open(path) as hdul:
        for hdu in hdul:
            if getattr(hdu, 'data', None) is None or not hasattr(hdu, 'columns'):
                continue
            names = set(hdu.columns.names or [])
            required = {'ENERG_LO', 'ENERG_HI', 'N_GRP', 'F_CHAN', 'N_CHAN', 'MATRIX'}
            if not required.issubset(names):
                continue

            incident_energy = (
                np.asarray(hdu.data['ENERG_LO'], dtype=float)
                + np.asarray(hdu.data['ENERG_HI'], dtype=float)
            ) / 2.0
            response_sigma = np.full(len(incident_energy), np.nan, dtype=float)

            for row_index, row in enumerate(hdu.data):
                n_grp = int(row['N_GRP'])
                f_chan = np.atleast_1d(row['F_CHAN'])[:n_grp]
                n_chan = np.atleast_1d(row['N_CHAN'])[:n_grp]
                matrix = np.asarray(row['MATRIX'], dtype=float).ravel()

                energies = []
                weights = []
                offset = 0
                for first, count in zip(f_chan, n_chan):
                    count = int(count)
                    group_weights = matrix[offset:offset + count]
                    offset += count
                    for channel, weight in zip(range(int(first), int(first) + count), group_weights):
                        idx = channel_lookup.get(int(channel))
                        if idx is not None and np.isfinite(weight) and weight > 0:
                            energies.append(channel_energy[idx])
                            weights.append(weight)

                if weights:
                    energies = np.asarray(energies, dtype=float)
                    weights = np.asarray(weights, dtype=float)
                    weight_sum = np.sum(weights)
                    mean_energy = np.sum(weights * energies) / weight_sum
                    variance = np.sum(weights * (energies - mean_energy) ** 2) / weight_sum
                    response_sigma[row_index] = np.sqrt(max(variance, 0.0))

            valid = np.isfinite(response_sigma) & (response_sigma > 0)
            if np.any(valid):
                return incident_energy[valid], response_sigma[valid]

    raise ValueError(f'Could not derive instrumental resolution from RMF file: {path}')

def _read_arf(path: str | Path):
    """Read energy bins and effective area from an ARF response file.

    Parameters
    ----------
    path : str or pathlib.Path
        ARF file containing ``ENERG_LO``, ``ENERG_HI``, and ``SPECRESP`` columns.

    Returns
    -------
    tuple of numpy.ndarray
        Energy-bin centers and effective area.

    Raises
    ------
    ValueError
        If the ARF file does not contain a SPECRESP-style table.
    """
    fits = _import_astropy_fits()
    with fits.open(path) as hdul:
        for hdu in hdul:
            if getattr(hdu, 'data', None) is None or not hasattr(hdu, 'columns'):
                continue
            names = set(hdu.columns.names or [])
            if {'ENERG_LO', 'ENERG_HI', 'SPECRESP'}.issubset(names):
                e_lo = np.asarray(hdu.data['ENERG_LO'], dtype=float)
                e_hi = np.asarray(hdu.data['ENERG_HI'], dtype=float)
                effective_area = np.asarray(hdu.data['SPECRESP'], dtype=float)
                return (e_lo + e_hi) / 2.0, effective_area
    raise ValueError(f'Could not find SPECRESP extension in ARF file: {path}')

def _header_float(header, key: str, default: float):
    """Read a numeric FITS header keyword with a fallback value.

    Parameters
    ----------
    header : mapping
        FITS header object.
    key : str
        Header keyword to read.
    default : float
        Value returned when the keyword is missing or cannot be converted to
        ``float``.

    Returns
    -------
    float
        Header value converted to float, or ``default``.
    """
    try:
        return float(header.get(key, default))
    except Exception:
        return float(default)

def _background_scale(source_header, background_header):
    """Compute the scale factor applied to a background spectrum.

    Parameters
    ----------
    source_header : mapping
        FITS header for the source spectrum, read for ``BACKSCAL`` and
        ``EXPOSURE``.
    background_header : mapping
        FITS header for the background spectrum, read for ``BACKSCAL`` and
        ``EXPOSURE``.

    Returns
    -------
    float
        Multiplicative factor used before subtracting the background counts from
        the source counts.
    """
    src_backscal = _header_float(source_header, 'BACKSCAL', 1.0)
    bkg_backscal = _header_float(background_header, 'BACKSCAL', 1.0)
    src_exposure = _header_float(source_header, 'EXPOSURE', 1.0)
    bkg_exposure = _header_float(background_header, 'EXPOSURE', src_exposure)
    if bkg_backscal == 0:
        return 1.0
    if bkg_exposure == 0:
        return 1.0
    return src_backscal / bkg_backscal * (src_exposure / bkg_exposure)

def _extract_spectrum_arrays(hdu):
    """Extract channel, value, and uncertainty arrays from a spectrum HDU.

    Parameters
    ----------
    hdu : astropy.io.fits.BinTableHDU
        FITS table containing a supported spectrum column and, ideally, channel and
        error columns.

    Returns
    -------
    tuple of numpy.ndarray
        Channel coordinates, spectral values, and one-sigma uncertainties. If no
        error column is present, Poisson errors are estimated from the values.

    Raises
    ------
    ValueError
        If no supported spectrum-value column is available.
    """
    data = hdu.data
    spectrum_column = _find_existing_column(hdu, POSSIBLE_SPECTRUM_COLUMNS)
    if spectrum_column is None:
        raise ValueError(f'No supported spectrum column found.\nAvailable columns: {hdu.columns.names}')
    values = np.asarray(data[spectrum_column], dtype=float)
    error_column = _find_existing_column(hdu, POSSIBLE_ERROR_COLUMNS)
    if error_column is not None:
        uncertainties = np.asarray(data[error_column], dtype=float)
    else:
        uncertainties = np.sqrt(np.clip(values, 0, None))
    channel_column = _find_existing_column(hdu, POSSIBLE_CHANNEL_COLUMNS)
    if channel_column is not None:
        channel = np.asarray(data[channel_column], dtype=float)
    else:
        channel = np.arange(len(values), dtype=float)
    return (channel, values, uncertainties)

def load_fits_spectrum(pha_path: str | Path, background_path: Optional[str | Path]=None, rmf_path: Optional[str | Path]=None, arf_path: Optional[str | Path]=None, *, subtract_background: bool=True, as_arrays: bool=False) -> Spectrum | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a folded FITS PHA spectrum with optional RMF and ARF information.

    Parameters
    ----------
    pha_path : str or pathlib.Path
        Source PHA/FITS spectrum file. Observed detector-space counts or rates are
        retained directly; the spectrum is not unfolded.
    background_path : str, pathlib.Path, or None, default: None
        Optional background spectrum. It is scaled by BACKSCAL and EXPOSURE before
        subtraction.
    rmf_path : str, pathlib.Path, or None, default: None
        Optional RMF file used to convert channel numbers into energy-bin centers
        and widths from EBOUNDS and to estimate the instrumental line-spread width
        from the redistribution matrix.
    arf_path : str, pathlib.Path, or None, default: None
        Optional ARF file. Its energy grid and effective area are retained in the
        returned ``Spectrum`` object for response-aware downstream analysis.
    subtract_background : bool, default: True
        If true and ``background_path`` is supplied, subtract the scaled background
        and propagate uncertainties in quadrature.
    as_arrays : bool, default: False
        If true, return raw arrays instead of a ``Spectrum`` object.

    Returns
    -------
    Spectrum or tuple of numpy.ndarray
        Cleaned folded spectrum sorted by energy. When ``as_arrays`` is true,
        returns ``(energy, values, uncertainties, bin_width)``.
    """
    fits = _import_astropy_fits()
    with fits.open(pha_path) as hdul:
        source_hdu = _find_spectrum_hdu(hdul)
        source_header = source_hdu.header
        channel, values, uncertainties = _extract_spectrum_arrays(source_hdu)
    if background_path is not None and subtract_background:
        with fits.open(background_path) as hdul:
            background_hdu = _find_spectrum_hdu(hdul)
            background_header = background_hdu.header
            bkg_channel, bkg_values, bkg_uncertainties = _extract_spectrum_arrays(background_hdu)
        scale = _background_scale(source_header, background_header)
        min_size = min(len(values), len(bkg_values))
        values = values[:min_size] - scale * bkg_values[:min_size]
        uncertainties = np.sqrt(uncertainties[:min_size] ** 2 + (scale * bkg_uncertainties[:min_size]) ** 2)
        channel = channel[:min_size]
    response_sigma = None
    if rmf_path is not None:
        rmf_channel, e_min, e_max = _read_ebounds_from_rmf(rmf_path)
        lookup = {int(ch): i for i, ch in enumerate(rmf_channel)}
        idx = np.array([lookup.get(int(ch), -1) for ch in channel])
        valid = idx >= 0
        if not np.any(valid):
            raise ValueError('No PHA channels matched RMF EBOUNDS.')
        idx = idx[valid]
        energy = (e_min[idx] + e_max[idx]) / 2.0
        bin_width = e_max[idx] - e_min[idx]
        values = values[valid]
        uncertainties = uncertainties[valid]
        resolution_energy, resolution_sigma = _read_rmf_resolution(rmf_path)
        response_sigma = np.interp(
            energy,
            resolution_energy,
            resolution_sigma,
            left=resolution_sigma[0],
            right=resolution_sigma[-1],
        )
    else:
        energy = channel.astype(float)
        bin_width = np.ones_like(energy, dtype=float)
    arf_energy = None
    effective_area = None
    if arf_path is not None:
        arf_energy, effective_area = _read_arf(arf_path)
    good = np.isfinite(energy) & np.isfinite(values) & np.isfinite(uncertainties) & (uncertainties > 0)
    spectrum = Spectrum(
        energy[good],
        values[good],
        uncertainties[good],
        bin_width[good],
        rmf_path=rmf_path,
        arf_path=arf_path,
        arf_energy=arf_energy,
        effective_area=effective_area,
        response_sigma=None if response_sigma is None else response_sigma[good],
    ).sorted()
    if as_arrays:
        return (spectrum.energy, spectrum.values, spectrum.uncertainties, spectrum.bin_width)
    return spectrum
