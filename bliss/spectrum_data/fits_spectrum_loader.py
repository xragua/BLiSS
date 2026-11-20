"""Read PHA/RMF/ARF-style FITS spectra into BLiSS spectrum objects."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
from .spectrum_container import Spectrum
POSSIBLE_SPECTRUM_COLUMNS = ['COUNTS', 'RATE', 'COUNT_RATE', 'PHA', 'SRC_COUNTS']
POSSIBLE_ERROR_COLUMNS = ['STAT_ERR', 'ERROR', 'ERR']
POSSIBLE_CHANNEL_COLUMNS = ['CHANNEL', 'PI']
RATE_COLUMNS = {'RATE', 'COUNT_RATE'}

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

def _response_core_sigma(energy, weights):
    """Gaussian-equivalent core width from interpolated half-height crossings.

    Requires both crossings and at least two samples in the half-height core.
    The input must include zero-weight channels, including gaps in the RMF.
    Invalid or undersampled cores return NaN rather than a tail-based width.
    """
    energy = np.asarray(energy, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(energy) < 3 or len(weights) != len(energy):
        return np.nan
    peak = int(np.argmax(weights))
    if not np.isfinite(weights[peak]) or weights[peak] <= 0:
        return np.nan
    half = weights[peak] / 2.0
    lo = hi = peak
    while lo > 0 and weights[lo - 1] >= half:
        lo -= 1
    while hi < len(weights) - 1 and weights[hi + 1] >= half:
        hi += 1
    if lo == 0 or hi == len(weights) - 1 or hi == lo:
        return np.nan
    left = np.interp(half, [weights[lo - 1], weights[lo]],
                     [energy[lo - 1], energy[lo]])
    right = np.interp(half, [weights[hi + 1], weights[hi]],
                      [energy[hi + 1], energy[hi]])
    width = right - left
    return width / np.sqrt(8.0 * np.log(2.0)) if width > 0 else np.nan


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
        The sigma is the FWHM of the principal response core divided by
        sqrt(8 ln 2), excluding the contribution of distant response tails.
        Rows without a resolved core are retained as NaN.

    Raises
    ------
    ValueError
        If the RMF does not contain the required redistribution information.
    """
    fits = _import_astropy_fits()
    rmf_channel, e_min, e_max = _read_ebounds_from_rmf(path)
    channel_energy = (e_min + e_max) / 2.0
    channel_order = np.argsort(channel_energy)
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

                channel_indices = []
                weights = []
                offset = 0
                for first, count in zip(f_chan, n_chan):
                    count = int(count)
                    group_weights = matrix[offset:offset + count]
                    offset += count
                    for channel, weight in zip(range(int(first), int(first) + count), group_weights):
                        idx = channel_lookup.get(int(channel))
                        if idx is not None and np.isfinite(weight) and weight > 0:
                            channel_indices.append(idx)
                            weights.append(weight)

                if weights:
                    # Restore zero-probability channels so gaps cannot broaden
                    # the interpolated half-height crossings.
                    profile = np.zeros(len(channel_energy), dtype=float)
                    np.add.at(profile, channel_indices, weights)
                    response_sigma[row_index] = _response_core_sigma(
                        channel_energy[channel_order], profile[channel_order])

            incident_order = np.argsort(incident_energy)
            return incident_energy[incident_order], response_sigma[incident_order]

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

def _extract_spectrum_arrays(hdu, *, assume_poisson_rate_errors: bool=False):
    """Extract channel, value, and uncertainty arrays from a spectrum HDU.

    Parameters
    ----------
    hdu : astropy.io.fits.BinTableHDU
        FITS table containing a supported spectrum column and, ideally, channel and
        error columns.
    assume_poisson_rate_errors : bool, default: False
        Allow missing rate errors to be estimated as sqrt(rate * exposure) /
        exposure, assuming raw Poisson counts divided by the exposure time.
        Explicit STAT_ERR, ERROR or ERR values always take precedence.

    Returns
    -------
    tuple of numpy.ndarray
        Channel coordinates, spectral values, and one-sigma uncertainties. If no
        error column is present, count errors are estimated as sqrt(counts).
        RATE and COUNT_RATE require the explicit Poisson opt-in above.

    Raises
    ------
    ValueError
        If no supported spectrum-value column is available, rate errors are
        missing without opt-in, or Poisson inference has invalid rates/exposure.
    """
    data = hdu.data
    spectrum_column = _find_existing_column(hdu, POSSIBLE_SPECTRUM_COLUMNS)
    if spectrum_column is None:
        raise ValueError(f'No supported spectrum column found.\nAvailable columns: {hdu.columns.names}')
    values = np.asarray(data[spectrum_column], dtype=float)
    error_column = _find_existing_column(hdu, POSSIBLE_ERROR_COLUMNS)
    if error_column is not None:
        uncertainties = np.asarray(data[error_column], dtype=float)
    elif spectrum_column.upper() in RATE_COLUMNS:
        if not assume_poisson_rate_errors:
            raise ValueError(
                f'{spectrum_column} spectrum has no STAT_ERR, ERROR or ERR column. '
                'Provide uncertainties or set assume_poisson_rate_errors=True '
                'to estimate them from the rate and EXPOSURE assuming Poisson statistics.'
            )
        exposure = _header_float(hdu.header, 'EXPOSURE', np.nan)
        if not np.isfinite(exposure) or exposure <= 0:
            raise ValueError(
                f'{spectrum_column} Poisson error estimation requires a finite, '
                'positive EXPOSURE keyword.'
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(
                f'{spectrum_column} Poisson error estimation requires finite, '
                'non-negative rates.'
            )
        uncertainties = np.sqrt(values * exposure) / exposure
    else:
        uncertainties = np.sqrt(np.clip(values, 0, None))
    channel_column = _find_existing_column(hdu, POSSIBLE_CHANNEL_COLUMNS)
    if channel_column is not None:
        channel = np.asarray(data[channel_column], dtype=float)
    else:
        channel = np.arange(len(values), dtype=float)
    return (channel, values, uncertainties)


def _validate_channel_ids(channel, label):
    """Reject ambiguous channel identifiers before matching source and background."""
    channel = np.asarray(channel, dtype=float)
    if (channel.ndim != 1 or not np.all(np.isfinite(channel))
            or np.any(channel != np.floor(channel))):
        raise ValueError(f'{label} CHANNEL/PI values must be finite integers.')
    unique, counts = np.unique(channel, return_counts=True)
    duplicates = unique[counts > 1]
    if duplicates.size:
        shown = ', '.join(str(int(c)) for c in duplicates[:8])
        suffix = ', ...' if duplicates.size > 8 else ''
        raise ValueError(f'{label} contains duplicate CHANNEL/PI values: {shown}{suffix}.')
    return channel


def _background_indices(channel, bkg_channel):
    """Find background rows for every retained source channel, in source order."""
    channel = _validate_channel_ids(channel, 'Source spectrum')
    bkg_channel = _validate_channel_ids(bkg_channel, 'Background spectrum')
    lookup = {int(c): i for i, c in enumerate(bkg_channel)}
    indices = np.array([lookup.get(int(c), -1) for c in channel], dtype=int)
    missing = channel[indices < 0]
    if missing.size:
        shown = ', '.join(str(int(c)) for c in missing[:8])
        suffix = ', ...' if missing.size > 8 else ''
        raise ValueError(
            f'Background spectrum is missing required source channels: {shown}{suffix}. '
            'Provide a background entry for every source channel used in the analysis.'
        )
    return indices


def load_fits_spectrum(pha_path: str | Path, background_path: Optional[str | Path]=None, rmf_path: Optional[str | Path]=None, arf_path: Optional[str | Path]=None, *, subtract_background: bool=True, as_arrays: bool=False, keep_zero_uncertainty: bool=False, assume_poisson_rate_errors: bool=False) -> Spectrum | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a folded FITS PHA spectrum with optional RMF and ARF information.

    Parameters
    ----------
    pha_path : str or pathlib.Path
        Source PHA/FITS spectrum file. Observed detector-space counts or rates are
        retained directly; the spectrum is not unfolded.
    background_path : str, pathlib.Path, or None, default: None
        Optional background spectrum. Values and errors are matched by CHANNEL
        (or PI), then converted to the source column's units using BACKSCAL and
        EXPOSURE. Extra background channels are ignored. Missing retained source
        channels, duplicate IDs or noninteger/nonfinite IDs raise ValueError.
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
    keep_zero_uncertainty : bool, default: False
        Retain finite empty channels with zero uncertainty for subsequent count-space
        grouping. ``prepare_spectrum`` enables this for every rebinning method
        to preserve the native grid. Native variances are retained;
        ``rebin_counts`` applies variance one only to zero-variance output bins
        after grouping, leaving every positive variance unchanged.
    assume_poisson_rate_errors : bool, default: False
        Reject RATE/COUNT_RATE without STAT_ERR, ERROR or ERR by default. True
        estimates missing errors as sqrt(rate * exposure) / exposure using each
        file's own positive, finite EXPOSURE. This assumes non-negative raw
        Poisson counts divided by exposure. Applies to source and background;
        explicit errors are always preserved.

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
        channel, values, uncertainties = _extract_spectrum_arrays(
            source_hdu, assume_poisson_rate_errors=assume_poisson_rate_errors,
        )
        source_column = _find_existing_column(source_hdu, POSSIBLE_SPECTRUM_COLUMNS)
    if background_path is not None and subtract_background:
        channel = _validate_channel_ids(channel, 'Source spectrum')
    # Only channels retained by the RMF require a matching background entry.
    # Keep source order here; Spectrum.sorted() orders all arrays together later.
    if rmf_path is not None:
        rmf_channel, e_min, e_max = _read_ebounds_from_rmf(rmf_path)
        lookup = {int(ch): i for i, ch in enumerate(rmf_channel)}
        idx = np.array([lookup.get(int(ch), -1) for ch in channel])
        valid = idx >= 0
        if not np.any(valid):
            raise ValueError('No PHA channels matched RMF EBOUNDS.')
        idx = idx[valid]
        channel = channel[valid]
        values = values[valid]
        uncertainties = uncertainties[valid]
    if background_path is not None and subtract_background:
        with fits.open(background_path) as hdul:
            background_hdu = _find_spectrum_hdu(hdul)
            background_header = background_hdu.header
            bkg_channel, bkg_values, bkg_uncertainties = _extract_spectrum_arrays(
                background_hdu, assume_poisson_rate_errors=assume_poisson_rate_errors,
            )
            background_column = _find_existing_column(background_hdu, POSSIBLE_SPECTRUM_COLUMNS)
        bkg_indices = _background_indices(channel, bkg_channel)
        bkg_values = bkg_values[bkg_indices]
        bkg_uncertainties = bkg_uncertainties[bkg_indices]
        scale = _background_scale(source_header, background_header)
        # The base scale is for counts; adapt it to the stored column units.
        if background_column.upper() in RATE_COLUMNS:
            exposure = _header_float(background_header, 'EXPOSURE', np.nan)
            if not np.isfinite(exposure) or exposure <= 0:
                raise ValueError(f'{background_path}: RATE column but no valid EXPOSURE keyword.')
            scale *= exposure
        if source_column.upper() in RATE_COLUMNS:
            exposure = _header_float(source_header, 'EXPOSURE', np.nan)
            if not np.isfinite(exposure) or exposure <= 0:
                raise ValueError(f'{pha_path}: RATE column but no valid EXPOSURE keyword.')
            scale /= exposure
        values = values - scale * bkg_values
        uncertainties = np.sqrt(uncertainties ** 2 + (scale * bkg_uncertainties) ** 2)
    response_sigma = None
    if rmf_path is not None:
        energy = (e_min[idx] + e_max[idx]) / 2.0
        bin_width = e_max[idx] - e_min[idx]
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
    valid_uncertainty = uncertainties > 0
    if keep_zero_uncertainty:
        valid_uncertainty |= (uncertainties == 0) & (values == 0)
    good = np.isfinite(energy) & np.isfinite(values) & np.isfinite(uncertainties) & valid_uncertainty
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




############################
############################
############################
"""Read the count-space information needed for Poisson null simulations.

This module does not modify ``fits_spectrum_loader``; it reuses its private
helpers so that channel selection, column detection and background scaling
are identical to the ones used when the Spectrum object was built.
"""


def _read_pha_counts(path, *, assume_poisson_rate_errors: bool=False):
    """Return (channel, counts, header, column) for one PHA file.

    If the spectrum column is a rate, it is multiplied back by EXPOSURE so
    that the returned array is always in counts. Missing rate errors require
    the same explicit Poisson opt-in as load_fits_spectrum.
    """
    fits = _import_astropy_fits()
    with fits.open(path) as hdul:
        hdu = _find_spectrum_hdu(hdul)
        header = hdu.header
        channel, values, _ = _extract_spectrum_arrays(
            hdu, assume_poisson_rate_errors=assume_poisson_rate_errors,
        )
        column = _find_existing_column(hdu, POSSIBLE_SPECTRUM_COLUMNS)
    exposure = _header_float(header, 'EXPOSURE', np.nan)
    if column.upper() in RATE_COLUMNS:
        if not np.isfinite(exposure) or exposure <= 0:
            raise ValueError(f'{path}: RATE column but no valid EXPOSURE keyword.')
        values = values * exposure
    return channel, values, header, column


def _has_poisson_count_law(path, *, assume_poisson_rate_errors=False):
    """Identify raw Poisson inputs without overriding supplied error models."""
    fits = _import_astropy_fits()
    with fits.open(path) as hdul:
        hdu = _find_spectrum_hdu(hdul)
        column = _find_existing_column(hdu, POSSIBLE_SPECTRUM_COLUMNS)
        if _find_existing_column(hdu, POSSIBLE_ERROR_COLUMNS) is not None:
            return False
        if hdu.header.get('POISSERR', True) is False:
            return False
        if 'SYS_ERR' in hdu.columns.names and np.any(hdu.data['SYS_ERR'] != 0):
            return False
        if float(hdu.header.get('SYS_ERR', 0.0)) != 0:
            return False
        return column is not None and (
            column.upper() not in RATE_COLUMNS or assume_poisson_rate_errors)


def _are_raw_poisson_counts(*arrays):
    """Check count values on the selected grid, separately from file error models."""
    return all(np.all(np.isfinite(a) & (a >= 0) &
                         (np.abs(a - np.rint(a)) <= 1e-6)) for a in arrays)


def read_pha_metadata(pha_path, rmf_path, background_path=None, *, assume_poisson_rate_errors: bool=False) -> dict:
    """Collect exposure, scaling factors and raw counts on the RMF energy grid.

    Background counts are matched by channel with the same validation as
    load_fits_spectrum. Every source channel retained by the RMF must have
    exactly one background entry; an absent entry is not treated as zero.
    RATE/COUNT_RATE without explicit errors are rejected unless
    assume_poisson_rate_errors=True, with the same validation as the loader
    for both source and background. Rate values are converted to counts using
    each file's own EXPOSURE.

    Returns
    -------
    dict with keys
        exposure, areascal, backscal : float  (source header)
        values_unit : {'counts', 'rate'}
            Unit of the values returned by ``load_fits_spectrum`` for this file.
        bkg_scale : float
            Factor applied to background counts before subtraction (same as
            the loader). 0.0 when no background is given.
        energy, bin_width : numpy.ndarray
            Channel energy centres and widths from EBOUNDS.
        src_counts, bkg_counts : numpy.ndarray
            Raw counts per channel on that grid. ``bkg_counts`` is zeros when
            no background is given.
        poisson_error_model : bool
            Whether both files permit the Poisson path under their declared
            error models and RATE opt-in. This file-level decision is retained
            when selecting a working domain.
        poisson_compatible : bool
            The error-model decision combined with nonnegative integer counts
            on the returned grid. With an explicit validation range,
            ``align_to_spectrum`` reevaluates the count-value condition on
            every original channel in that domain, including channels that
            the spectrum loader subsequently discarded.
    """
    channel, src_counts, src_header, column = _read_pha_counts(
        pha_path, assume_poisson_rate_errors=assume_poisson_rate_errors,
    )

    if background_path is not None:
        channel = _validate_channel_ids(channel, 'Source spectrum')
    rmf_channel, e_min, e_max = _read_ebounds_from_rmf(rmf_path)
    lookup = {int(c): i for i, c in enumerate(rmf_channel)}
    idx = np.array([lookup.get(int(c), -1) for c in channel])
    valid = idx >= 0
    if not np.any(valid):
        raise ValueError('No PHA channels matched RMF EBOUNDS.')
    idx = idx[valid]
    channel = channel[valid]
    src_counts = src_counts[valid]

    bkg_scale = 0.0
    bkg_counts = np.zeros_like(src_counts)
    if background_path is not None:
        bkg_channel, bkg_raw, bkg_header, _ = _read_pha_counts(
            background_path, assume_poisson_rate_errors=assume_poisson_rate_errors,
        )
        bkg_scale = _background_scale(src_header, bkg_header)
        bkg_counts = bkg_raw[_background_indices(channel, bkg_channel)]

    poisson_error_model = (
        _has_poisson_count_law(pha_path, assume_poisson_rate_errors=assume_poisson_rate_errors)
        and (background_path is None or _has_poisson_count_law(
            background_path, assume_poisson_rate_errors=assume_poisson_rate_errors))
    )
    return {
        'exposure': _header_float(src_header, 'EXPOSURE', np.nan),
        'areascal': _header_float(src_header, 'AREASCAL', 1.0),
        'backscal': _header_float(src_header, 'BACKSCAL', 1.0),
        'values_unit': 'rate' if column.upper() in RATE_COLUMNS else 'counts',
        'poisson_error_model': poisson_error_model,
        'poisson_compatible': (poisson_error_model
                               and _are_raw_poisson_counts(src_counts, bkg_counts)),
        'bkg_scale': bkg_scale,
        'energy': (e_min[idx] + e_max[idx]) / 2.0,
        'bin_width': e_max[idx] - e_min[idx],
        'src_counts': src_counts,
        'bkg_counts': bkg_counts,
    }

def align_to_spectrum(meta: dict, spectrum_energy, *, count_validation_range=None) -> dict:
    """Restrict metadata arrays to the bins present in ``spectrum_energy``.

    The loader removes bins with non-finite or non-positive uncertainties and
    sorts by energy; this selects the matching channels by exact energy match.
    An explicit ``count_validation_range=(lo, hi)`` reevaluates count-law
    compatibility on all original metadata channels in that energy interval,
    before selecting loader-retained channels. Invalid counts inside the
    requested domain must not be hidden by the loader's quality selection.
    File-level error models are preserved. With no validation range, or legacy
    metadata lacking an error-model flag, retain the original compatibility
    decision unchanged.
    """
    spectrum_energy = np.asarray(spectrum_energy, dtype=float)
    order = np.argsort(meta['energy'])
    sorted_energy = meta['energy'][order]
    pos = np.searchsorted(sorted_energy, spectrum_energy)
    pos = np.clip(pos, 0, len(sorted_energy) - 1)
    if not np.allclose(sorted_energy[pos], spectrum_energy, rtol=0, atol=1e-9):
        raise ValueError('Spectrum energies do not match RMF EBOUNDS grid.')
    sel = order[pos]
    out = dict(meta)
    for key in ('energy', 'bin_width', 'src_counts', 'bkg_counts'):
        out[key] = meta[key][sel]
    if count_validation_range is not None:
        bounds = np.asarray(count_validation_range, dtype=float)
        if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[1] <= bounds[0]:
            raise ValueError('count_validation_range must contain two finite, increasing energy limits.')
        if 'poisson_error_model' in meta:
            in_domain = (meta['energy'] >= bounds[0]) & (meta['energy'] <= bounds[1])
            out['poisson_compatible'] = (bool(meta['poisson_error_model'])
                                         and _are_raw_poisson_counts(
                                             meta['src_counts'][in_domain],
                                             meta['bkg_counts'][in_domain]))
    return out
