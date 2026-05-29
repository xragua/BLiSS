"""Load four-column text spectra containing lower/upper energy bounds."""
import numpy as np
import pandas as pd

def load_text_spectrum(path):
    """Load a four-column text spectrum and sort it by energy.

    Parameters
    ----------
    path : str or pathlib.Path
        Text file with columns ``E_low``, ``E_high``, spectral value, and
        uncertainty. Comment lines beginning with ``#`` are ignored.

    Returns
    -------
    tuple of numpy.ndarray
        ``(energy, values, uncertainties, bin_width)`` sorted by increasing energy,
        where ``energy`` is the midpoint of the lower and upper bin edges.

    Raises
    ------
    ValueError
        If the input file does not contain exactly four columns.
    """
    spectra = pd.read_csv(path, sep='\\s+', comment='#', header=None, engine='python')
    if spectra.shape[1] != 4:
        raise ValueError('File must have 4 columns: E_low, E_high, counts, error.')
    energy = ((spectra[0] + spectra[1]) / 2).to_numpy()
    bin_width = (spectra[1] - spectra[0]).to_numpy()
    values = spectra[2].to_numpy()
    uncertainties = spectra[3].to_numpy()
    order = np.argsort(energy)
    return (energy[order], values[order], uncertainties[order], bin_width[order])
