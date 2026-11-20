"""Match fitted centroids against user-supplied transition tables."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class _AtomicTable:
    frame: pd.DataFrame
    ions: np.ndarray
    energy_keV: np.ndarray
    weights: np.ndarray | None


def _require_atomic_database(table):
    if table is None:
        raise ValueError(
            'No atomic database selected. Supply your own pandas DataFrame '
            'via atomic_table / pd_data; BLiSS does not include a default catalogue.')


def _numeric_column(table, column, *, positive):
    if column not in table.columns:
        raise ValueError(f"Atomic database is missing column {column!r}.")
    try:
        values = pd.to_numeric(table[column], errors='raise').to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Column {column!r} must contain numeric values.") from exc
    valid = np.isfinite(values) & (values > 0 if positive else values >= 0)
    if not valid.all():
        requirement = 'positive' if positive else 'nonnegative'
        raise ValueError(f"Column {column!r} must contain finite, {requirement} values.")
    return values


def _to_keV(values, unit, *, wavelength):
    if unit is None:
        option = 'wavelength_unit' if wavelength else 'energy_unit'
        raise ValueError(f"Specify {option} for the selected atomic database column.")
    # The existing keV-only interface does not need the optional FITS/unit import.
    if not wavelength and unit == 'keV':
        return values
    from astropy import units as u

    try:
        source_unit = u.Unit(unit)
        expected = u.m if wavelength else u.keV
        if not source_unit.is_equivalent(expected):
            raise ValueError('wrong physical dimension')
        energy = (values * source_unit).to_value(u.keV, equivalencies=u.spectral())
    except (TypeError, ValueError) as exc:
        kind = 'length' if wavelength else 'energy'
        raise ValueError(f"Invalid {kind} unit: {unit!r}.") from exc
    if not np.all(np.isfinite(energy) & (energy > 0)):
        raise ValueError('Converted atomic energies must be finite and positive.')
    return energy


def _prepare_atomic_table(pd_data, *, ion_col='ion', energy_col=None,
                          wavelength_col=None, energy_unit=None, wavelength_unit=None,
                          weight_col=None, consistency_rtol=1e-3):
    _require_atomic_database(pd_data)
    if not isinstance(pd_data, pd.DataFrame):
        raise TypeError('The atomic database must be a pandas DataFrame.')
    if not pd_data.columns.is_unique:
        raise ValueError('Atomic database column names must be unique.')
    if ion_col not in pd_data.columns:
        raise ValueError(f"Atomic database is missing ion column {ion_col!r}.")
    ions = pd_data[ion_col]
    if ions.isna().any() or ions.astype(str).str.strip().eq('').any():
        raise ValueError('Ion identifiers must not be missing or empty.')
    if (not np.isscalar(consistency_rtol) or not np.isfinite(consistency_rtol)
            or consistency_rtol < 0):
        raise ValueError('consistency_rtol must be finite and nonnegative.')

    # Preserve the legacy energy_keV interface when no coordinate is selected.
    if energy_col is None and wavelength_col is None:
        if 'energy_keV' not in pd_data.columns:
            raise ValueError('Specify energy_col or wavelength_col (or both).')
        energy_col = 'energy_keV'
    energy = None
    if energy_col is not None:
        unit = 'keV' if energy_unit is None and energy_col == 'energy_keV' else energy_unit
        energy = _to_keV(_numeric_column(pd_data, energy_col, positive=True),
                         unit, wavelength=False)
    if wavelength_col is not None:
        from_wavelength = _to_keV(
            _numeric_column(pd_data, wavelength_col, positive=True),
            wavelength_unit, wavelength=True)
        if energy is not None:
            consistent = np.isclose(energy, from_wavelength, rtol=consistency_rtol, atol=0)
            if not consistent.all():
                positions = np.flatnonzero(~consistent)[:5].tolist()
                raise ValueError('Energy and wavelength disagree at row positions '
                                 f'{positions} (consistency_rtol={consistency_rtol}).')
        else:
            energy = from_wavelength

    weights = (None if weight_col is None else
               _numeric_column(pd_data, weight_col, positive=False))
    return _AtomicTable(pd_data, ions.to_numpy(), energy, weights)


def _set_output_column(candidates, name, values):
    """Keep colliding database metadata under a unique database_* name."""
    if name in candidates.columns:
        if np.array_equal(candidates[name].to_numpy(), np.asarray(values)):
            return
        original_name = f'database_{name}'
        while original_name in candidates.columns:
            original_name = f'database_{original_name}'
        candidates.rename(columns={name: original_name}, inplace=True)
    candidates[name] = values


def _identify_line(center_energy_keV, v_doppler_kms, table):
    c = 299792.458
    delta_E = 0 if v_doppler_kms is None else center_energy_keV * abs(v_doppler_kms) / c
    indices = np.flatnonzero((table.energy_keV >= center_energy_keV - delta_E)
                             & (table.energy_keV <= center_energy_keV + delta_E))
    distances = np.abs(table.energy_keV[indices] - center_energy_keV)
    if table.weights is None:
        order = np.argsort(distances, kind='stable')
    else:
        # Weight first; proximity breaks equal weights, then original row order.
        order = np.lexsort((indices, distances, -table.weights[indices]))
    indices = indices[order]
    candidates = table.frame.iloc[indices].copy().reset_index(drop=True)
    energies = table.energy_keV[indices]
    _set_output_column(candidates, 'ion', table.ions[indices])
    _set_output_column(candidates, 'energy_keV', energies)
    _set_output_column(candidates, 'doppler_kms', c * (center_energy_keV - energies) / energies)
    first = ['ion', 'energy_keV', 'doppler_kms']
    return candidates[first + [col for col in candidates.columns if col not in first]]


class LineIdentifier:
    """Identify lines with an explicitly supplied atomic-table DataFrame.

    BLiSS does not ship, download or automatically load an atomic catalogue.
    Read a catalogue of your choice into a pandas DataFrame and supply it as
    ``atomic_table``. Omitting it raises a clear error.

    ``ion_col`` selects the ion/transition label. Select ``energy_col`` and/or
    ``wavelength_col`` with their physical units, e.g. ``energy_unit='eV'`` or
    ``wavelength_unit='Angstrom'``. With neither coordinate selected, the legacy
    ``energy_keV`` column is used in keV. Both coordinates must agree within
    ``consistency_rtol`` (default 1e-3, i.e. 0.1%); energy is then used for matching.

    ``weight_col`` optionally ranks compatible transitions in descending order;
    ties and unweighted custom tables are ordered by energy distance. Weights
    must be finite and nonnegative and are not probabilities. Only an explicit
    ``weight_col`` selects weighted ranking; otherwise energy proximity is used.

    The matching half-window uses ``abs(v_doppler_kms)``: either sign gives
    the same matches. Zero or None requires an exact energy match. This
    tolerance is separate from the inferred ``doppler_kms``, which retains
    its sign: c * (measured_energy - reference_energy) / reference_energy.
    Input tables are never modified. All-match results retain database columns
    and add canonical ``ion``, ``energy_keV`` and ``doppler_kms`` columns. Conflicting
    original names are preserved with a ``database_`` prefix (repeated if needed).
    """

    def __init__(self, v_doppler_kms=1200, atomic_table=None, *,
                 ion_col='ion', energy_col=None, wavelength_col=None,
                 energy_unit=None, wavelength_unit=None, weight_col=None,
                 consistency_rtol=1e-3):
        _require_atomic_database(atomic_table)
        self.v_doppler_kms = v_doppler_kms
        self.atomic_table = atomic_table
        self._column_options = dict(
            ion_col=ion_col, energy_col=energy_col, wavelength_col=wavelength_col,
            energy_unit=energy_unit, wavelength_unit=wavelength_unit,
            weight_col=weight_col, consistency_rtol=consistency_rtol)

    def add_most_probable(self, lines):
        """Attach the highest-ranked ion; the legacy name implies no probability."""
        return add_most_probable_ion(
            lines, self.v_doppler_kms, pd_data=self.atomic_table, **self._column_options)

    def all_compatible(self, lines):
        """Return all matches and database metadata; fitted-row indices must be unique."""
        return get_all_compatible_lines(
            lines, self.v_doppler_kms, pd_data=self.atomic_table, **self._column_options)


def identify_line(center_energy_keV, center_sigma_keV=None, v_doppler_kms=None,
                  pd_data=None, *, ion_col='ion', energy_col=None,
                  wavelength_col=None, energy_unit=None, wavelength_unit=None,
                  weight_col=None, consistency_rtol=1e-3):
    """Find compatible transitions for a centroid in keV.

    See ``LineIdentifier`` for custom column, unit and ranking options. The
    energy half-window is ``center_energy_keV * abs(v_doppler_kms) / c``;
    zero or None requires an exact energy match. ``center_sigma_keV`` is retained
    for compatibility and does not widen it. Returns all database columns alongside
    canonical ion, energy_keV and inferred doppler_kms, even for an empty result.
    """
    table = _prepare_atomic_table(
        pd_data, ion_col=ion_col, energy_col=energy_col, wavelength_col=wavelength_col,
        energy_unit=energy_unit, wavelength_unit=wavelength_unit,
        weight_col=weight_col, consistency_rtol=consistency_rtol)
    return _identify_line(center_energy_keV, v_doppler_kms, table)


def add_most_probable_ion(pd_fit, v_doppler_kms, pd_data=None, *,
                          ion_col='ion', energy_col=None, wavelength_col=None,
                          energy_unit=None, wavelength_unit=None, weight_col=None,
                          consistency_rtol=1e-3):
    """Add the highest-ranked compatible ion and inferred Doppler velocity.

    All fitted-line columns are preserved. See ``LineIdentifier`` for custom
    database options. Despite the legacy name, ranking weights are not
    probabilities. Use ``get_all_compatible_lines`` for full database metadata.
    The matching tolerance uses abs(v_doppler_kms); inferred velocities keep
    their sign.
    """
    _require_atomic_database(pd_data)
    if pd_fit is None:
        return pd_fit
    out = pd_fit.copy().reset_index(drop=True)
    if len(out) == 0:
        for col in ['doppler_kms', 'ion']:
            if col not in out.columns:
                out.insert(0, col, [])
        return out
    if 'center' not in out.columns:
        raise ValueError("pd_fit must contain a 'center' column.")
    table = _prepare_atomic_table(
        pd_data, ion_col=ion_col, energy_col=energy_col, wavelength_col=wavelength_col,
        energy_unit=energy_unit, wavelength_unit=wavelength_unit,
        weight_col=weight_col, consistency_rtol=consistency_rtol)
    ions, vdoppler = [], []
    for center in out['center']:
        candidates = _identify_line(center, v_doppler_kms, table)
        if not candidates.empty:
            best = candidates.iloc[0]
            ions.append(best['ion'])
            vdoppler.append(best['doppler_kms'])
        else:
            ions.append('--')
            vdoppler.append(np.nan)
    out = out.drop(columns=['ion', 'doppler_kms'], errors='ignore')
    out.insert(0, 'doppler_kms', vdoppler)
    out.insert(0, 'ion', ions)
    return out


def get_all_compatible_lines(pd_fit, v_doppler_kms, pd_data=None, *,
                             ion_col='ion', energy_col=None, wavelength_col=None,
                             energy_unit=None, wavelength_unit=None, weight_col=None,
                             consistency_rtol=1e-3):
    """Map fitted-row indices to all matches, retaining database metadata.

    See ``LineIdentifier`` for custom column, unit and ranking options.
    Fitted-row indices must be unique; otherwise raise ValueError before
    matching to prevent dictionary entries from overwriting each other.
    Concatenate fitted tables with pd.concat([...], ignore_index=True), or
    use pd_fit.reset_index(drop=True). Unique input indices are preserved.
    Duplicate row indices in the atomic database are allowed.
    The matching tolerance uses abs(v_doppler_kms), equally for either sign.
    """
    if not pd_fit.index.is_unique:
        raise ValueError(
            'pd_fit contains duplicate row indices. Concatenate fitted-line '
            'tables with pd.concat([...], ignore_index=True), or use '
            'pd_fit = pd_fit.reset_index(drop=True) before identification.'
        )
    table = _prepare_atomic_table(
        pd_data, ion_col=ion_col, energy_col=energy_col, wavelength_col=wavelength_col,
        energy_unit=energy_unit, wavelength_unit=wavelength_unit,
        weight_col=weight_col, consistency_rtol=consistency_rtol)
    return {idx: _identify_line(row['center'], v_doppler_kms, table)
            for idx, row in pd_fit.iterrows()}
