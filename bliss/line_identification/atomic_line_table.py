"""Load and prepare the atomic transition table used for line identification."""
from pathlib import Path
import pandas as pd

def load_atomic_database(path=None):
    """Load the atomic transition table used by BLiSS line identification.

    Parameters
    ----------
    path : str, pathlib.Path, or None, default: None
        Tab-separated atomic-line table. When omitted, the bundled
        ``st_reduced.dat`` file next to this module is loaded.

    Returns
    -------
    pandas.DataFrame
        Atomic database containing transition information such as ion name,
        rest-frame energy, Einstein coefficient, and the columns used to rank
        candidate identifications.
    """
    if path is None:
        path = Path(__file__).resolve().parent / 'st_reduced.dat'
    return pd.read_csv(path, sep='\t')

st_reduced = load_atomic_database()

# ------------------------------------------------------------
# Atomic identification score
# ------------------------------------------------------------
# xdef is treated as an elemental abundance term.
# Aul is the Einstein coefficient of the transition.
#
# This score is only a heuristic ranking quantity used when several
# transitions are compatible with the same observed BLiSS centroid.
# It is not a physical line flux or a statistical probability.
# ------------------------------------------------------------

import numpy as np

abundance = st_reduced["xdef"].astype(float)
aul = st_reduced["Aul"].astype(float)

# Normalize abundance safely
if np.nanmax(abundance) > 0:
    abundance_norm = abundance / np.nanmax(abundance)
else:
    abundance_norm = np.ones_like(abundance, dtype=float)

# Aul spans many orders of magnitude, so use log10(Aul)
log_aul = np.log10(np.clip(aul, 1e-99, None))

if np.nanmax(log_aul) > np.nanmin(log_aul):
    aul_norm = (
        (log_aul - np.nanmin(log_aul))
        / (np.nanmax(log_aul) - np.nanmin(log_aul))
    )
else:
    aul_norm = np.ones_like(log_aul, dtype=float)

st_reduced["abundance_norm"] = abundance_norm
st_reduced["Aul_norm"] = aul_norm

# Keep the old column name for compatibility with line_identifier.py
st_reduced["scaled_prob"] = abundance_norm * aul_norm
