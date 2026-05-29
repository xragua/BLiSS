"""Dataclass used to pass sorted spectral arrays through BLiSS."""
from dataclasses import dataclass
import numpy as np

@dataclass
class Spectrum:
    """One-dimensional spectrum passed through BLiSS loaders and pipelines.

    Attributes
    ----------
    energy : numpy.ndarray
        Spectral coordinate values, usually bin centers in keV.
    values : numpy.ndarray
        Counts, rates, or flux values at each coordinate.
    uncertainties : numpy.ndarray
        One-sigma uncertainties associated with ``values``.
    bin_width : numpy.ndarray or None, default: None
        Width of each spectral bin. When available, it is used for equivalent-width
        estimates.
    """
    energy: np.ndarray
    values: np.ndarray
    uncertainties: np.ndarray
    bin_width: np.ndarray | None = None

    def sorted(self):
        """Return a copy of the spectrum ordered by increasing energy.

        Returns
        -------
        Spectrum
            New spectrum with ``energy``, ``values``, ``uncertainties``, and optional
            ``bin_width`` sorted consistently.
        """
        order = np.argsort(self.energy)
        bin_width = None if self.bin_width is None else self.bin_width[order]
        return Spectrum(self.energy[order], self.values[order], self.uncertainties[order], bin_width)
