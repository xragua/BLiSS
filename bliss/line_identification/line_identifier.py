"""Match fitted line centroids against the BLiSS atomic transition table."""
import numpy as np
import pandas as pd
from .atomic_line_table import st_reduced

class LineIdentifier:
    """Identify fitted BLiSS lines using a configurable Doppler window.

    Attributes
    ----------
    v_doppler_kms : float
        Velocity half-width, in km/s, used to decide whether an atomic transition
        is compatible with a fitted line centroid.
    atomic_table : pandas.DataFrame
        Atomic transition table searched by the identifier.
    """

    def __init__(self, v_doppler_kms=1200, atomic_table=st_reduced):
        """Create a line identifier.

        Parameters
        ----------
        v_doppler_kms : float, default: 1200
            Doppler velocity half-width used to define the allowed energy window around
            each fitted line centroid.
        atomic_table : pandas.DataFrame, default: bundled table
            Atomic transition table with at least ``energy_keV`` and ranking columns.
        """
        self.v_doppler_kms = v_doppler_kms
        self.atomic_table = atomic_table

    def add_most_probable(self, lines):
        """Attach the highest-ranked compatible ion to each fitted line.

        Parameters
        ----------
        lines : pandas.DataFrame
            BLiSS line table containing at least a ``center`` column and, when
            available, fitted widths and amplitudes.

        Returns
        -------
        pandas.DataFrame
            Copy of the input table augmented with the best atomic-line match for each
            candidate.
        """
        return add_most_probable_ion(lines, self.v_doppler_kms)

    def all_compatible(self, lines):
        """Return all atomic transitions compatible with each fitted line.

        Parameters
        ----------
        lines : pandas.DataFrame
            BLiSS candidate table containing fitted line centers.

        Returns
        -------
        dict
            Mapping from candidate-row index to a DataFrame of all atomic transitions
            inside the configured Doppler window.
        """
        return get_all_compatible_lines(lines, self.v_doppler_kms)

def identify_line(center_energy_keV, center_sigma_keV=None, v_doppler_kms=None, pd_data=st_reduced):
    """Find atomic transitions compatible with one measured line center.

    Parameters
    ----------
    center_energy_keV : float
        Fitted line centroid in keV.
    center_sigma_keV : float or None, default: None
        Fitted Gaussian width in keV. It is accepted for API compatibility but the
        current energy window is set by ``v_doppler_kms``.
    v_doppler_kms : float or None, default: None
        Velocity half-width used to compute the allowed energy interval around
        ``center_energy_keV``. If omitted or zero, only exact-energy matches can be
        returned.
    pd_data : pandas.DataFrame, default: bundled atomic table
        Atomic transition table to search. It must include ``energy_keV`` and the
        columns used for ranking, such as ``scaled_prob``.

    Returns
    -------
    pandas.DataFrame
        Compatible transitions sorted by decreasing ``scaled_prob`` and including
        the implied Doppler shift in km/s.
    """
    c = 299792.458
    elines = np.array(pd_data.energy_keV)
    delta_E_sigma = 0
    delta_E_doppler = 0
    if center_sigma_keV:
        delta_E_sigma = center_sigma_keV
    if v_doppler_kms:
        delta_E_doppler = center_energy_keV * v_doppler_kms / c
    delta_E = delta_E_doppler
    energy_min = center_energy_keV - delta_E
    energy_max = center_energy_keV + delta_E
    idx = np.where((elines >= energy_min) & (elines <= energy_max))[0]
    candidates = st_reduced.iloc[idx].copy().reset_index(drop=True)
    candidates['doppler_kms'] = c * (center_energy_keV - candidates['energy_keV']) / candidates['energy_keV']
    candidates = candidates.sort_values(by='scaled_prob', ascending=False).reset_index(drop=True)
    desired_order = ['ion',  'energy_keV', 'doppler_kms', 'center', 'sigma', 'amplitude', 'ecenter', 'esigma', 'eamplitude',
       'base_on_line', 'value_on_line', 'noise_on_block', 'snr_peak',
       'snr_area', 'relative_power', 'area', 'earea', 'ew',
       'cluster_probability']
    return candidates[[col for col in desired_order if col in candidates.columns]]

def add_most_probable_ion(pd_fit, v_doppler_kms):
    """Add the strongest compatible atomic identification to each BLiSS line.

    Parameters
    ----------
    pd_fit : pandas.DataFrame
        Fitted BLiSS line table. The function reads ``center`` and optionally
        ``sigma`` from each row.
    v_doppler_kms : float
        Velocity half-width used when matching fitted centers to atomic rest
        energies.

    Returns
    -------
    pandas.DataFrame
        Input line table combined with the top-ranked atomic identification for
        each row when a compatible transition is available.
    """
    top_candidates = []
    for idx, row in pd_fit.iterrows():
        center_energy = row['center']
        sigma_center_energy = row.get('sigma', None)
        candidates = identify_line(center_energy, sigma_center_energy, v_doppler_kms)
        if not candidates.empty:
            top_candidates.append(candidates.iloc[0])
        else:
            top_candidates.append(pd.Series(dtype='object'))
    ion_info_df = pd.DataFrame(top_candidates).reset_index(drop=True)
    result = pd.concat([pd_fit.reset_index(drop=True), ion_info_df], axis=1)
    desired_order = ['ion',  'energy_keV', 'doppler_kms', 'center', 'sigma', 'amplitude', 'ecenter', 'esigma', 'eamplitude',
       'base_on_line', 'value_on_line', 'noise_on_block', 'snr_peak',
       'snr_area', 'relative_power', 'area', 'earea', 'ew',
       'cluster_probability']
    return result[[col for col in desired_order if col in result.columns]]

def get_all_compatible_lines(pd_fit, v_doppler_kms):
    """Collect every compatible atomic transition for each fitted line.

    Parameters
    ----------
    pd_fit : pandas.DataFrame
        Fitted BLiSS line table with a ``center`` column.
    v_doppler_kms : float
        Doppler velocity half-width used for the compatibility search.

    Returns
    -------
    dict
        Dictionary keyed by the input row index. Each value is a DataFrame of
        compatible transitions, or an empty DataFrame when no transition matches.
    """
    compatible_lines_dict = {}
    for idx, row in pd_fit.iterrows():
        center_energy = row['center']
        sigma_center_energy = row.get('sigma', None)
        candidates = identify_line(center_energy, sigma_center_energy, v_doppler_kms)
        compatible_lines_dict[idx] = candidates if not candidates.empty else pd.DataFrame()
    return compatible_lines_dict
