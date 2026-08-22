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
        """Attach the highest-ranked compatible ion to each fitted line."""
        return add_most_probable_ion(
            lines,
            self.v_doppler_kms,
            pd_data=self.atomic_table,
        )

    def all_compatible(self, lines):
        """Return all atomic transitions compatible with each fitted line."""
        return get_all_compatible_lines(
            lines,
            self.v_doppler_kms,
            pd_data=self.atomic_table,
        )


def identify_line(center_energy_keV, center_sigma_keV=None, v_doppler_kms=None, pd_data=st_reduced):
    """Find atomic transitions compatible with one measured line center."""

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

    candidates = pd_data.iloc[idx].copy().reset_index(drop=True)

    if len(candidates) == 0:
        return candidates

    candidates["doppler_kms"] = (
        c * (center_energy_keV - candidates["energy_keV"]) / candidates["energy_keV"]
    )

    candidates = candidates.sort_values(
        by="scaled_prob",
        ascending=False,
    ).reset_index(drop=True)

    desired_order = [
        "ion",
        "energy_keV",
        "doppler_kms",
        "center",
        "sigma",
        "amplitude",
        "ecenter",
        "esigma",
        "eamplitude",
        "base_on_line",
        "value_on_line",
        "noise_on_block",
        "snr_peak",
        "snr_area",
        "relative_power",
        "area",
        "earea",
        "ew",
        "cluster_probability",
    ]

    return candidates[[col for col in desired_order if col in candidates.columns]]

def add_most_probable_ion(pd_fit, v_doppler_kms, pd_data=st_reduced):
    """Add the most probable ion and its Doppler velocity.

    All original columns in ``pd_fit`` are preserved.
    """

    if pd_fit is None:
        return pd_fit

    out = pd_fit.copy().reset_index(drop=True)

    if len(out) == 0:
        for col in ["doppler_kms", "ion"]:
            if col not in out.columns:
                out.insert(0, col, [])
        return out

    if "center" not in out.columns:
        raise ValueError("pd_fit must contain a 'center' column.")

    ions = []
    vdoppler = []

    for _, row in out.iterrows():

        center_energy = row["center"]
        sigma_center_energy = row.get("sigma", None)

        candidates = identify_line(
            center_energy_keV=center_energy,
            center_sigma_keV=sigma_center_energy,
            v_doppler_kms=v_doppler_kms,
            pd_data=pd_data,
        )

        if not candidates.empty and "ion" in candidates.columns:
            best = candidates.iloc[0]

            ions.append(best["ion"])
            vdoppler.append(best["doppler_kms"])

        else:
            ions.append("--")
            vdoppler.append(np.nan)

    for col in ["ion", "doppler_kms"]:
        if col in out.columns:
            out = out.drop(columns=[col])

    out.insert(0, "doppler_kms", vdoppler)
    out.insert(0, "ion", ions)

    return out


def get_all_compatible_lines(pd_fit, v_doppler_kms, pd_data=st_reduced):
    """Collect every compatible atomic transition for each fitted line."""

def get_all_compatible_lines(pd_fit, v_doppler_kms, pd_data=st_reduced):
    """Collect every compatible atomic transition for each fitted line."""

    compatible_lines_dict = {}

    for idx, row in pd_fit.iterrows():

        center_energy = row["center"]
        sigma_center_energy = row.get("sigma", None)

        candidates = identify_line(
            center_energy_keV=center_energy,
            center_sigma_keV=sigma_center_energy,
            v_doppler_kms=v_doppler_kms,
            pd_data=pd_data,
        )

        compatible_lines_dict[idx] = (
            candidates if not candidates.empty else pd.DataFrame()
        )

    return compatible_lines_dict
