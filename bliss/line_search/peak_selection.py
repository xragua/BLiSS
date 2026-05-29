"""Peak-selection helpers wrapping scipy.signal.find_peaks."""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def find_peaks_new(t, x):
    """Find local peaks and convert their widths to physical coordinate units.

    Parameters
    ----------
    t : array-like
        Coordinate array associated with the spectrum, usually energy in keV.
    x : array-like
        Spectral values on which local maxima are searched.

    Returns
    -------
    pandas.DataFrame
        Peak table with sample positions, prominences, scipy widths, interpolated
        lower and upper coordinate limits, centroid coordinate, and ``twidth`` in
        the same units as ``t``.
    """
    t = np.asarray(t)
    x = np.asarray(x)
    peak_positions, properties = find_peaks(x, prominence=0, width=0)
    if len(peak_positions) == 0:
        return pd.DataFrame(columns=['position', 'prominences', 'widths', 'width_heights', 'left_ips', 'right_ips', 'energy', 'ienergy', 'eenergy', 'twidth'])
    prominences = properties['prominences']
    widths = properties['widths']
    width_heights = -properties['width_heights']
    left_ips = properties['left_ips']
    right_ips = properties['right_ips']
    left_indices = np.floor(left_ips).astype(int)
    right_indices = np.floor(right_ips).astype(int)
    left_fraction = left_ips % 1
    right_fraction = right_ips % 1
    coordinate_step = np.insert(np.diff(t), 0, 0, axis=0)
    left_energy = t[left_indices] + coordinate_step[left_indices] * left_fraction
    right_energy = t[right_indices] + coordinate_step[right_indices] * right_fraction
    coordinate_width = right_energy - left_energy
    peak_energy = t[peak_positions]
    sorted_indices = np.argsort(peak_positions)
    data = {'position': peak_positions[sorted_indices], 'prominences': prominences[sorted_indices], 'widths': widths[sorted_indices], 'width_heights': width_heights[sorted_indices], 'left_ips': left_ips[sorted_indices], 'right_ips': right_ips[sorted_indices], 'energy': peak_energy[sorted_indices], 'ienergy': left_energy[sorted_indices], 'eenergy': right_energy[sorted_indices], 'twidth': coordinate_width[sorted_indices]}
    return pd.DataFrame(data).reset_index(drop=True)
