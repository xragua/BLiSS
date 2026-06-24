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
st_reduced['scaled_prob'] = st_reduced.xdef
