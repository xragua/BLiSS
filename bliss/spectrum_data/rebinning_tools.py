"""Count-conserving rebinning of detector spectra."""
from dataclasses import dataclass
import numpy as np


OPTIMAL_METHODS = ('optimal', 'kaastra')
COUNT_METHODS = ('counts', 'mincounts')


def validate_oversampling(oversampling, method='optimal'):
    """Oversampling is an optional positive factor for optimal grouping only."""
    if oversampling is None:
        return
    if (isinstance(oversampling, (bool, np.bool_))
            or not isinstance(oversampling, (int, float, np.integer, np.floating))
            or not np.isfinite(oversampling) or oversampling <= 0):
        raise ValueError('oversampling must be a finite positive number or None.')
    if str(method).lower() not in OPTIMAL_METHODS:
        raise ValueError('oversampling requires optimal or kaastra rebinning.')


@dataclass(frozen=True)
class OptimalBinningResponse:
    """RMF quantities for Kaastra & Bleeker (2016), Appendix B.1.

    All arrays correspond to the native spectrum, sorted by energy.
    ``fwhm_channels`` is measured on the full RMF channel grid; NaN means
    the core is unresolved or truncated and the channel must stay ungrouped.
    ``counts_start``/``counts_stop`` select the half-height core plus one
    neighbouring channel on either side (stop is exclusive).
    ``counts_correction`` is the full response divided by its sum over that
    same window, including response wings rather than assuming a Gaussian.
    """
    fwhm_channels: np.ndarray
    counts_start: np.ndarray
    counts_stop: np.ndarray
    counts_correction: np.ndarray


def optimal_response_from_rmf(rmf_path, energy):
    """Read the instrumental quantities needed for optimal *data* binning.

     Half-height crossings are linearly
    interpolated in channel coordinates. 

    Channels with no response or with a core truncated at an RMF boundary
    are left ungrouped. 
    """
    from .fits_spectrum_loader import _import_astropy_fits, _read_ebounds_from_rmf

    channels, low, high = _read_ebounds_from_rmf(rmf_path)
    order = np.argsort((low + high) / 2.0)
    channel_energy = ((low + high) / 2.0)[order]
    energy = np.asarray(energy, dtype=float)
    if (energy.ndim != 1 or not np.all(np.isfinite(energy))
            or np.any(np.diff(energy) <= 0)
            or not np.all(np.isfinite(channel_energy))
            or np.any(np.diff(channel_energy) <= 0)):
        raise ValueError('Optimal binning requires unique, increasing finite channel energies.')
    indices = np.searchsorted(channel_energy, energy)
    if (np.any(indices >= len(channel_energy))
            or not np.allclose(channel_energy[indices], energy, rtol=0, atol=1e-9)):
        raise ValueError('Optimal binning energies must match native RMF EBOUNDS channels.')
    lookup = {int(channels[i]): k for k, i in enumerate(order)}
    n_channels = len(channels)

    def profile(row):
        result = np.zeros(n_channels, dtype=float)
        groups = int(row['N_GRP'])
        firsts = np.atleast_1d(row['F_CHAN'])[:groups]
        sizes = np.atleast_1d(row['N_CHAN'])[:groups]
        weights = np.asarray(row['MATRIX'], dtype=float).ravel()
        offset = 0
        for first, size in zip(firsts, sizes):
            size = int(size)
            values = weights[offset:offset + size]
            offset += size
            positions = np.array([lookup.get(ch, -1)
                                  for ch in range(int(first), int(first) + size)])
            if len(values) != size:
                raise ValueError('RMF MATRIX is shorter than its declared channel groups.')
            valid = (positions >= 0) & np.isfinite(values) & (values > 0)
            np.add.at(result, positions[valid], values[valid])
        return result

    n = len(energy)
    fwhm = np.full(n, np.nan)
    starts = np.arange(n, dtype=int)
    stops = starts + 1
    correction = np.ones(n)
    fits = _import_astropy_fits()
    with fits.open(rmf_path) as hdul:
        required = {'ENERG_LO', 'ENERG_HI', 'N_GRP', 'F_CHAN', 'N_CHAN', 'MATRIX'}
        matrix = next((hdu.data for hdu in hdul
                       if hasattr(hdu, 'columns') and hdu.data is not None
                       and required.issubset(hdu.columns.names or [])), None)
        if matrix is None:
            raise ValueError('Optimal binning requires an RMF redistribution MATRIX, not just EBOUNDS.')
        best_value = np.zeros(n)
        best_row = np.full(n, -1, dtype=int)
        # Find the incident energy with maximum response at each data channel.
        for j, row in enumerate(matrix):
            values = profile(row)[indices]
            better = values > best_value
            best_value[better] = values[better]
            best_row[better] = j
        # Reconstruct only the selected response rows; never allocate a dense RMF.
        selected = np.argsort(best_row, kind='stable')
        boundaries = np.flatnonzero(np.diff(best_row[selected])) + 1
        for batch in np.split(selected, boundaries):
            if not len(batch) or best_row[batch[0]] < 0:
                continue
            weights = profile(matrix[best_row[batch[0]]])
            total = weights.sum()
            for pos in batch:
                channel = indices[pos]
                half = 0.5 * weights[channel]
                left = right = channel
                while left > 0 and weights[left - 1] >= half:
                    left -= 1
                while right + 1 < n_channels and weights[right + 1] >= half:
                    right += 1
                if left == 0 or right == n_channels - 1:
                    continue  # cannot measure both half-height crossings
                c1 = left - 1 + (half - weights[left - 1]) / (weights[left] - weights[left - 1])
                c2 = right + (weights[right] - half) / (weights[right] - weights[right + 1])
                fwhm[pos] = max(c2 - c1, 1.0)
                starts[pos] = np.searchsorted(indices, left - 1, side='left')
                stops[pos] = np.searchsorted(indices, right + 1, side='right')
                correction[pos] = max(1.0, total / weights[left - 1:right + 2].sum())
    if n and not np.any(np.isfinite(fwhm)):
        raise ValueError('Optimal binning could not measure any complete response core in the RMF.')
    return OptimalBinningResponse(fwhm, starts, stops, correction)


def _optimal_groups(energy, bin_width, source_counts, response, oversampling=None):
    """
    Reference: https://doi.org/10.1051/0004-6361/201527395
    The counts determining the grid are gross observed counts, before any
    background subtraction. Net counts and their uncertainties are summed
    separately by ``apply_groups``. Empty spectra use the low-count limit.
    """
    validate_oversampling(oversampling)
    n = len(energy)
    source_counts = np.asarray(source_counts, dtype=float)
    if (source_counts.shape != energy.shape
            or not np.all(np.isfinite(source_counts) & (source_counts >= 0))):
        raise ValueError('Optimal binning needs finite, nonnegative source_counts before background subtraction.')
    if (np.any(np.diff(energy) <= 0) or not np.all(np.isfinite(energy))
            or not np.all(np.isfinite(bin_width) & (bin_width > 0))):
        raise ValueError('Optimal binning needs increasing finite energies and positive finite bin widths.')
    c = np.asarray(response.fwhm_channels, dtype=float)
    first = np.asarray(response.counts_start, dtype=int)
    stop = np.asarray(response.counts_stop, dtype=int)
    h = np.asarray(response.counts_correction, dtype=float)
    if (any(a.shape != energy.shape for a in (c, first, stop, h))
            or np.any(first < 0) or np.any(stop > n) or np.any(first >= stop)
            or not np.all(np.isfinite(h) & (h >= 1))):
        raise ValueError('Optimal response arrays do not match the native spectrum.')
    valid = np.isfinite(c) & (c > 0)
    if not np.any(valid):
        raise ValueError('Optimal binning requires a measurable instrumental response.')
    c = np.where(valid, np.maximum(c, 1.0), 1.0)
    resolution_elements = max(1.0, np.sum(1.0 / c))
    cumulative = np.r_[0.0, np.cumsum(source_counts)]
    events = (cumulative[stop] - cumulative[first]) * h
    x = np.full(n, -np.inf)
    positive = events > 0
    x[positive] = np.log(events[positive]) + np.log1p(0.20 * np.log(resolution_elements))
    fraction = np.ones(n)
    high_counts = x > 2.119
    z = x[high_counts]
    fraction[high_counts] = (0.08 + 7.0 / z + 1.8 / z**2) / (1.0 + 5.9 / z)
    sizes = np.maximum(1, np.floor(np.minimum(fraction * c, n))).astype(int)
    sizes[~valid] = 1
    minimum_sizes = None
    if oversampling is not None:
        # FWHM is measured in native channel coordinates, as in the optimal
        # rule. Round the optional lower bound UP to avoid undershooting it.
        with np.errstate(over='ignore', divide='ignore'):
            minimum_sizes = np.maximum(1, np.ceil(np.minimum(c / oversampling, n))).astype(int)
        minimum_sizes[~valid] = 1
        sizes = np.maximum(sizes, minimum_sizes)

    # Missing channels and unmeasurable response cores are grouping boundaries.
    lo, hi = energy - bin_width / 2, energy + bin_width / 2
    adjacent = np.isclose(lo[1:], hi[:-1], rtol=1e-7, atol=1e-12)
    breaks = np.flatnonzero(~adjacent | ~valid[1:] | ~valid[:-1]) + 1
    group = np.empty(n, dtype=int)
    start = gid = 0
    for segment_end in np.r_[breaks, n]:
        segment_starts = []
        while start < segment_end:
            proposed_end = min(start + sizes[start], segment_end)
            positions = np.arange(start, proposed_end)
            end = min(proposed_end, int(np.min(positions + sizes[positions])))
            if minimum_sizes is not None:
                # A varying response must not shorten the group below the
                # oversampling floor of any channel it contains.
                minimum = int(np.max(minimum_sizes[start:end]))
                while end < min(start + minimum, segment_end):
                    previous_end = end
                    end = min(start + minimum, segment_end)
                    minimum = max(minimum, int(np.max(minimum_sizes[previous_end:end])))
                # A short tail may need more than one preceding group when
                # the instrumental FWHM rises sharply near the boundary.
                while end - start < minimum and segment_starts:
                    previous_start = segment_starts.pop()
                    minimum = max(minimum, int(np.max(minimum_sizes[previous_start:start])))
                    start = previous_start
                    gid -= 1
            group[start:end] = gid
            if minimum_sizes is not None:
                segment_starts.append(start)
            start = end
            gid += 1
    return group


# ---------------------------------------------------------------------------
# Count-conserving rebinning (for spectra in counts, i.e. loader output)
# ---------------------------------------------------------------------------

def rebin_counts(energy, counts, uncertainties, bin_width, method='optimal',
                 scale=None, min_bins=1, remainder='merge', *, rmf_path=None,
                 source_counts=None, optimal_response=None, oversampling=None,
                 zero_variance_floor=True):
    """Rebin a count spectrum by summing consecutive bins.

    Counts are **summed** (uncertainties in quadrature, widths added), so
    Poisson statistics are preserved and the result can be converted to a
    density afterwards. Every method preserves positive variances, including
    values below one. Only a zero-variance output bin receives variance one
    after grouping; no floor is added to individual native channels.

    Parameters
    ----------
    energy, counts, uncertainties, bin_width : array-like
        Bin centres, counts (net counts are allowed), one-sigma uncertainties
        and bin widths. Must be sorted by energy.
    method : {'none', 'bins', 'snr', 'counts', 'mincounts', 'resolution', 'optimal', 'kaastra'}, default 'optimal'
        * ``'none'``: retain the native grid, applying the same zero-variance
          fallback to each final bin.
        * ``'bins'``: group ``scale`` consecutive bins.
        * ``'snr'``: accumulate bins until ``counts / sigma >= scale``.
        * ``'counts'`` (alias ``'mincounts'``): accumulate gross source counts
          until at least ``scale`` counts and ``min_bins`` native channels
          are present. Preserve empty native channels and stop at energy gaps.
          With ``remainder='merge'``, append a short remainder to the previous
          bin in the same segment; if the entire segment is below threshold,
          retain that segment as one bin. Counts never depend on error floors.
        * ``'resolution'``: group bins into fixed-width intervals of ``scale``
          energy units, anchored at the start of each contiguous segment;
          each input bin is assigned to the interval containing
          its centre, so bins are never split.
        * ``'optimal'`` (alias ``'kaastra'``): Kaastra & Bleeker (2016) data
          binning using the RMF and observed counts. Preserves every channel,
          including a short last bin, and never groups across energy gaps.
          Native variances are summed without a floor; only zero-variance
          output bins receive a one-count variance floor after grouping.
    scale : int or float
        Number of bins, S/N threshold, minimum counts, or interval width,
        depending on ``method``.
        Leave as None for optimal binning, which derives its own scale.
    min_bins : int, default 1
        Minimum number of input bins per output bin (``'snr'`` or ``'counts'``).
    remainder : {'merge', 'drop'}, default 'merge'
        What to do with trailing bins that do not complete a group
        (``'bins'``, ``'snr'``, and ``'counts'``), separately within each
        contiguous energy segment: merge them into the last output bin or
        drop them. A whole segment below threshold is retained with 'merge'.
        For S/N grouping, merging a remainder can lower the final S/N below
        scale, especially with background-subtracted counts. The threshold
        is a grouping target, not a guarantee for every final bin.
    rmf_path : path-like or None
        RMF with MATRIX and EBOUNDS, required for optimal binning unless
        ``optimal_response`` is already supplied.
    source_counts : array-like or None
        Gross counts in the source extraction region, before subtraction of
        the background, on the same grid as ``counts``. Optimal binning uses
        these counts to choose groups. Defaults to ``counts`` for spectra
        without background; explicitly supply it for background-subtracted
        spectra, including those with negative net counts.
    optimal_response : OptimalBinningResponse or None
        Precomputed RMF quantities from ``optimal_response_from_rmf``, reused
        when grouping independent Poisson realizations on the same native grid.

    oversampling : float or None, default None
        Optional maximum sampling factor for optimal/kaastra grouping. Use
        max(optimal size, FWHM / oversampling), in native channel units, with
        the oversampling lower bound rounded up to whole channels. None keeps
        pure optimal grouping. A short tail is merged with the previous group
        in the same segment. A whole segment shorter than the bound is retained;
        gaps and channels without measurable response are never crossed.

    zero_variance_floor : bool, default True
        Legacy Gaussian-weight safeguard: replace final zero errors by one
        count. Poisson fitting passes False and preserves zero raw variance;
        these raw errors are not Poisson confidence intervals or fit weights.

    Returns
    -------
    energy_new, counts_new, uncertainties_new, bin_width_new : numpy.ndarray
        Output bin centres (midpoint of the grouped edges), summed counts,
        uncertainties added in quadrature, and total widths.
    group : numpy.ndarray of int
        Index of the output bin each input bin was assigned to (-1 if dropped).
    """
    if not isinstance(zero_variance_floor, (bool, np.bool_)):
        raise ValueError("zero_variance_floor must be a boolean.")
    validate_oversampling(oversampling, method)
    energy = np.asarray(energy, dtype=float)
    counts = np.asarray(counts, dtype=float)
    uncertainties = np.asarray(uncertainties, dtype=float)
    bin_width = np.asarray(bin_width, dtype=float)
    method = str(method).lower()
    if method not in ('none', 'bins', 'snr', 'counts', 'mincounts', 'resolution', 'optimal', 'kaastra'):
        raise ValueError('Unknown rebinning method: ' + method)
    if any(a.ndim != 1 for a in (energy, counts, uncertainties, bin_width)):
        raise ValueError('Spectrum arrays must be one-dimensional.')
    if (not all(np.all(np.isfinite(a)) for a in (energy, counts, uncertainties, bin_width))
            or np.any(uncertainties < 0) or np.any(bin_width <= 0)):
        raise ValueError('Spectrum arrays must be finite, with nonnegative errors and positive widths.')
    if method in ('bins', 'snr', 'resolution'):
        if (isinstance(scale, (bool, np.bool_))
                or not isinstance(scale, (int, float, np.integer, np.floating))
                or not np.isfinite(scale) or scale <= 0):
            raise ValueError('scale must be a finite positive number.')
        if method == 'bins' and not float(scale).is_integer():
            raise ValueError('scale must be a positive integer for bins grouping.')
    if method == 'snr':
        if (isinstance(min_bins, (bool, np.bool_))
                or not isinstance(min_bins, (int, float, np.integer, np.floating))
                or not np.isfinite(min_bins) or min_bins < 1 or not float(min_bins).is_integer()):
            raise ValueError('min_bins must be a positive integer.')
    n = len(energy)
    if not (len(counts) == len(uncertainties) == len(bin_width) == n):
        raise ValueError('energy, counts, uncertainties and bin_width must have the same length.')
    if n == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty, np.array([], dtype=int)
    if np.any(np.diff(energy) <= 0):
        raise ValueError('energy must be sorted in increasing order.')
    if remainder not in ('merge', 'drop'):
        raise ValueError("remainder must be 'merge' or 'drop'.")

    method = str(method).lower()
    if method == 'none':
        final_uncertainties = (np.where(uncertainties == 0, 1.0, uncertainties)
                               if zero_variance_floor else uncertainties.copy())
        return energy.copy(), counts.copy(), final_uncertainties, bin_width.copy(), np.arange(n)

    if method in COUNT_METHODS:
        if (isinstance(scale, (bool, np.bool_)) or scale is None
                or not np.isscalar(scale) or not np.isfinite(float(scale))
                or float(scale) < 1 or not float(scale).is_integer()):
            raise ValueError('Minimum-count grouping requires a positive integer scale.')
        if (isinstance(min_bins, (bool, np.bool_)) or not np.isscalar(min_bins)
                or not np.isfinite(float(min_bins)) or float(min_bins) < 1
                or not float(min_bins).is_integer()):
            raise ValueError('min_bins must be a positive integer.')
        gross = np.asarray(counts if source_counts is None else source_counts, dtype=float)
        if (gross.shape != counts.shape or not np.all(np.isfinite(gross) & (gross >= 0))
                or not np.all(np.isfinite(counts))
                or not np.all(np.isfinite(uncertainties) & (uncertainties >= 0))
                or not np.all(np.isfinite(energy)) or np.any(np.diff(energy) <= 0)
                or not np.all(np.isfinite(bin_width) & (bin_width > 0))):
            raise ValueError('Minimum-count grouping requires finite arrays, nonnegative gross counts '
                             'and errors, increasing energies and positive widths.')
        lo, hi = energy - bin_width / 2, energy + bin_width / 2
        breaks = np.flatnonzero(~np.isclose(lo[1:], hi[:-1], rtol=1e-7, atol=1e-12)) + 1
        group = np.full(n, -1, dtype=int)
        segment_start = gid = 0
        for segment_end in np.r_[breaks, n]:
            start, accumulated, segment_gid = segment_start, 0.0, gid
            for i in range(segment_start, segment_end):
                accumulated += gross[i]
                if accumulated >= float(scale) and i + 1 - start >= int(min_bins):
                    group[start:i + 1] = gid
                    gid += 1
                    start, accumulated = i + 1, 0.0
            if start < segment_end and remainder == 'merge':
                if gid > segment_gid:
                    group[start:segment_end] = gid - 1
                else:
                    group[start:segment_end] = gid
                    gid += 1
            segment_start = segment_end
        return apply_groups(energy, counts, uncertainties, bin_width, group,
                            zero_variance_floor=zero_variance_floor) + (group,)

    if method in OPTIMAL_METHODS:
        if scale is not None or min_bins != 1:
            raise ValueError('Optimal binning requires scale=None and min_bins=1; its scale is derived from the RMF and counts.')
        if optimal_response is None:
            if rmf_path is None:
                raise ValueError('Optimal binning requires rmf_path or a precomputed optimal_response.')
            optimal_response = optimal_response_from_rmf(rmf_path, energy)
        group = _optimal_groups(energy, bin_width,
                                counts if source_counts is None else source_counts,
                                optimal_response, oversampling=oversampling)
        return apply_groups(energy, counts, uncertainties, bin_width, group,
                            zero_variance_floor=zero_variance_floor) + (group,)

    if scale is None:
        raise ValueError(f"method='{method}' requires a scale.")

    # Process disconnected energy intervals independently. Otherwise the
    # summed widths and midpoint would describe a fictitious interval in a gap.
    lo, hi = energy - bin_width / 2, energy + bin_width / 2
    breaks = np.flatnonzero(~np.isclose(lo[1:], hi[:-1], rtol=1e-7, atol=1e-12)) + 1
    if len(breaks):
        group = np.full(n, -1, dtype=int)
        offset = 0
        for segment in np.split(np.arange(n), breaks):
            local = rebin_counts(
                energy[segment], counts[segment], uncertainties[segment], bin_width[segment],
                method=method, scale=scale, min_bins=min_bins, remainder=remainder,
                zero_variance_floor=zero_variance_floor,
            )[-1]
            keep = local >= 0
            group[segment[keep]] = local[keep] + offset
            if np.any(keep):
                offset += int(local[keep].max()) + 1
        return apply_groups(energy, counts, uncertainties, bin_width, group,
                            zero_variance_floor=zero_variance_floor) + (group,)

    # ---- build the group index for every input bin -------------------------
    group = np.full(n, -1, dtype=int)

    if method == 'bins':
        nb = int(scale)
        if nb < 1:
            raise ValueError('scale must be >= 1 for method="bins".')
        group[:] = np.arange(n) // nb
        n_full = (n // nb) * nb
        if n_full < n:                       # incomplete trailing group
            if remainder == 'merge' and n_full > 0:
                group[n_full:] = group[n_full - 1]
            elif remainder == 'drop':
                group[n_full:] = -1

    elif method == 'snr':
        threshold = float(scale)
        if threshold <= 0:
            raise ValueError('scale must be positive for method="snr".')
        g = 0
        acc_counts = 0.0
        acc_var = 0.0
        n_acc = 0
        start = 0
        for i in range(n):
            acc_counts += counts[i]
            acc_var += uncertainties[i] ** 2
            n_acc += 1
            snr = acc_counts / np.sqrt(acc_var) if acc_var > 0 else 0.0
            if snr >= threshold and n_acc >= min_bins:
                group[start:i + 1] = g
                g += 1
                acc_counts = acc_var = 0.0
                n_acc = 0
                start = i + 1
        if start < n:                        # trailing bins below threshold
            if remainder == 'merge' and g > 0:
                group[start:] = g - 1
            elif remainder == 'merge':       # nothing reached threshold: one bin
                group[start:] = 0
            else:
                group[start:] = -1

    elif method == 'resolution':
        width = float(scale)
        if not np.isfinite(width) or width <= 0:
            raise ValueError('scale must be a positive finite width for method="resolution".')
        x0 = energy[0] - 0.5 * bin_width[0]
        raw = np.floor((energy - x0) / width).astype(int)
        # renumber consecutively (skip empty intervals)
        _, group[:] = np.unique(raw, return_inverse=True)

    else:
        raise ValueError("method must be 'none', 'bins', 'snr', 'counts', 'mincounts', 'resolution', 'optimal', or 'kaastra'.")

    return apply_groups(energy, counts, uncertainties, bin_width, group,
                            zero_variance_floor=zero_variance_floor) + (group,)


def apply_groups(energy, counts, uncertainties, bin_width, group, *, zero_variance_floor=True):
    """Sum native bins according to a precomputed group index.

    This is the second half of ``rebin_counts``; it is exposed so that a
    grouping derived from the data can be applied unchanged to synthetic
    realizations. Native variances are summed unchanged. Only a zero-variance
    final bin receives variance one, before any conversion from counts to
    density; positive variances below one are preserved.

    Parameters
    ----------
    energy, counts, uncertainties, bin_width : array-like
        Native bins (sorted by energy).
    group : array-like of int
        Output-bin index of each input bin; ``-1`` means dropped.

    zero_variance_floor : bool, default True
        Legacy Gaussian-weight safeguard: replace final zero errors by one
        count. Poisson fitting passes False and preserves zero raw variance;
        these raw errors are not Poisson confidence intervals or fit weights.

    Returns
    -------
    energy_new, counts_new, uncertainties_new, bin_width_new : numpy.ndarray
    """
    energy = np.asarray(energy, dtype=float)
    counts = np.asarray(counts, dtype=float)
    uncertainties = np.asarray(uncertainties, dtype=float)
    bin_width = np.asarray(bin_width, dtype=float)
    group = np.asarray(group, dtype=int)

    keep = group >= 0
    ids = np.unique(group[keep])
    e_lo = energy - 0.5 * bin_width
    e_hi = energy + 0.5 * bin_width

    energy_new = np.empty(len(ids))
    counts_new = np.empty(len(ids))
    unc_new = np.empty(len(ids))
    width_new = np.empty(len(ids))
    for j, gid in enumerate(ids):
        m = group == gid
        lo, hi = e_lo[m].min(), e_hi[m].max()
        energy_new[j] = 0.5 * (lo + hi)
        width_new[j] = bin_width[m].sum()
        counts_new[j] = counts[m].sum()
        unc_new[j] = np.sqrt(np.sum(uncertainties[m] ** 2))

    if zero_variance_floor:
        unc_new = np.where(unc_new == 0, 1.0, unc_new)
    return energy_new, counts_new, unc_new, width_new
