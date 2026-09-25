"""Dataclass used to pass sorted spectral arrays through BLiSS."""
from dataclasses import dataclass
from pathlib import Path
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
    rmf_path : str, pathlib.Path, or None, default: None
        Path to the RMF response file associated with the spectrum.
    arf_path : str, pathlib.Path, or None, default: None
        Path to the ARF effective-area file associated with the spectrum.
    arf_energy : numpy.ndarray or None, default: None
        Energy-bin centers of the associated ARF response.
    effective_area : numpy.ndarray or None, default: None
        Effective area from the associated ARF response.
    response_sigma : numpy.ndarray or None, default: None
        Instrumental Gaussian-equivalent sigma, in the same energy units as
        ``energy``, evaluated at each spectral bin from the RMF redistribution.
    """
    energy: np.ndarray
    values: np.ndarray
    uncertainties: np.ndarray
    bin_width: np.ndarray | None = None
    rmf_path: str | Path | None = None
    arf_path: str | Path | None = None
    arf_energy: np.ndarray | None = None
    effective_area: np.ndarray | None = None
    response_sigma: np.ndarray | None = None

    def sorted(self):
        """Return a copy of the spectrum ordered by increasing energy.

        Returns
        -------
        Spectrum
            New spectrum with ``energy``, ``values``, ``uncertainties``, optional
            ``bin_width``, and energy-aligned response information sorted
            consistently.
        """
        order = np.argsort(self.energy)
        bin_width = None if self.bin_width is None else self.bin_width[order]
        response_sigma = None if self.response_sigma is None else self.response_sigma[order]
        return Spectrum(
            self.energy[order],
            self.values[order],
            self.uncertainties[order],
            bin_width,
            rmf_path=self.rmf_path,
            arf_path=self.arf_path,
            arf_energy=self.arf_energy,
            effective_area=self.effective_area,
            response_sigma=response_sigma,
        )
