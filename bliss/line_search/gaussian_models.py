"""Gaussian models and initial-parameter generators used by BLiSS fits."""
import numpy as np

def gaussian(x, amplitude, center, sigma):
    """Evaluate a single Gaussian profile.

    Parameters
    ----------
    x : array-like
        Coordinates where the profile is evaluated.
    amplitude : float
        Peak height of the Gaussian above zero.
    center : float
        Coordinate of the Gaussian centroid.
    sigma : float
        Standard deviation of the Gaussian in the same units as ``x``.

    Returns
    -------
    numpy.ndarray
        Gaussian profile evaluated at ``x``.
    """
    return amplitude * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))

def n_gaussian(x, *params):
    """Evaluate a sum of Gaussian components.

    Parameters
    ----------
    x : array-like
        Coordinates where the combined profile is evaluated.
    *params : float
        Flat parameter sequence grouped as ``amplitude, center, sigma`` for each
        Gaussian component.

    Returns
    -------
    numpy.ndarray
        Sum of all Gaussian components evaluated at ``x``.
    """
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        amplitude, center, sigma = params[i:i + 3]
        y += gaussian(x, amplitude, center, sigma)
    return y

def p0_generator(x, y, good_peaks_dataframe):
    """Build initial parameters and bounds for fitting local candidate peaks.

    Parameters
    ----------
    x : array-like
        Energy or coordinate grid of the candidate block.
    y : array-like
        Observed values in the candidate block.
    good_peaks_dataframe : pandas.DataFrame
        Peak table returned by ``find_peaks_new`` after candidate filtering. The
        function uses ``position``, ``energy``, and ``twidth``.

    Returns
    -------
    tuple
        ``(p0, bounds)`` where ``p0`` is a flat list of Gaussian initial guesses and
        ``bounds`` is the pair of lower and upper bounds expected by
        ``scipy.optimize.curve_fit``.
    """
    p0, bound_low, bound_high = ([], [], [])
    for i in range(len(good_peaks_dataframe)):
        p0.append(y[good_peaks_dataframe.position.loc[i]])
        p0.append(good_peaks_dataframe.energy.loc[i])
        if (good_peaks_dataframe.twidth.loc[i] < 0.05) & (good_peaks_dataframe.twidth.loc[i] > 0):
            p0.append(good_peaks_dataframe.twidth.loc[i])
        else:
            p0.append(0.05)
        bound_low.append(y[good_peaks_dataframe.position.loc[i]] * 0)
        bound_low.append(good_peaks_dataframe.energy.loc[i] * 0.99)
        bound_low.append(0)
        bound_high.append(y[good_peaks_dataframe.position.loc[i]] * 100)
        bound_high.append(good_peaks_dataframe.energy.loc[i] * 1.01)
        bound_high.append(0.25)
    return (p0, (bound_low, bound_high))

def p0_generator_final(x, y, clean_lines):
    """Build initial parameters and bounds for the final multi-line fit.

    Parameters
    ----------
    x : array-like
        Full spectral coordinate grid. Present for API consistency; the current
        implementation uses the candidate table directly.
    y : array-like
        Full spectral values. Present for API consistency; the current
        implementation uses the candidate table directly.
    clean_lines : pandas.DataFrame
        Candidate table containing ``amplitude``, ``center``, and ``sigma`` columns
        from the preliminary line search.

    Returns
    -------
    tuple
        ``(p0, bounds)`` for the final call to ``curve_fit``.
    """
    p0, bound_low, bound_high = ([], [], [])
    for i in range(len(clean_lines)):
        p0.append(clean_lines.amplitude.loc[i])
        p0.append(clean_lines.center.loc[i])
        if (clean_lines.sigma.loc[i] < 0.05) & (clean_lines.sigma.loc[i] > 0):
            p0.append(clean_lines.sigma.loc[i])
        else:
            p0.append(0.05)
        bound_low.append(0)
        bound_low.append(clean_lines.center.loc[i] - 0.1)
        bound_low.append(0)
        bound_high.append(clean_lines.amplitude.loc[i] * 10)
        bound_high.append(clean_lines.center.loc[i] + 0.1)
        bound_high.append(clean_lines.sigma.loc[i] + 0.1)
    return (p0, (bound_low, bound_high))
