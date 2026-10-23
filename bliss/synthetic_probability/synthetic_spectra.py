"""Generate null (line-free) synthetic spectra for BLiSS significance estimates.

The null is generated in detector space: the empirical baseline measured on
the rebinned spectrum is interpolated to the native channel grid, converted to
expected counts (times exposure and channel width, plus the scaled background
that was subtracted), and source and background counts are drawn from
independent Poisson distributions. Each realization is then rebinned using
the configured rule, converted to counts s^-1 keV^-1, and its empirical
baseline is recomputed, so the non-linearity of the baseline estimator is
propagated empirically into the null distribution. The physical median-window
widths are inherited from the full observed spectrum. This is the detector-space
analogue of ``fakeit`` with the empirical baseline as the model.

A Gaussian model (fluctuations N(0, sigma_i) around the baseline on the
rebinned grid) is retained only for validation and comparison.
"""
from __future__ import annotations
from dataclasses import dataclass
from numbers import Integral
import numpy as np

from ..line_search.empirical_baseline import choose_baseline_window, moving_median_x
from ..spectrum_data.rebinning_tools import rebin_counts, apply_groups, OPTIMAL_METHODS


# --------------------------------------------------------------------------
# Single-realization generators
# --------------------------------------------------------------------------

def _gaussian_realization(rng, baseline, errors):
    y_sim = baseline + rng.normal(loc=0.0, scale=errors)
    return y_sim, errors.copy()


def _poisson_realization(rng, mu_src, mu_bkg, bkg_scale, conv, *, return_source=False):
    """Draw source and background counts and combine them like the loader.

    Net value  : (S - s B) / conv
    Uncertainty: sqrt(S + s^2 B) / conv   (quadrature, as in the loader)

    A floor of one count is applied to the variance so that bins with zero
    simulated counts keep a finite, positive uncertainty. This is the only
    deliberate departure from the loader, which drops such bins.

    ``return_source=True`` also returns the gross source-region draw, used to
    repeat optimal binning before background subtraction.
    """
    s_sim = rng.poisson(mu_src).astype(float)
    b_sim = rng.poisson(mu_bkg).astype(float) if bkg_scale > 0 else np.zeros_like(s_sim)
    y_sim = (s_sim - bkg_scale * b_sim) / conv
    var = np.maximum(s_sim + bkg_scale ** 2 * b_sim, 1.0)
    sy_sim = np.sqrt(var) / conv
    if return_source:
        return y_sim, sy_sim, s_sim
    return y_sim, sy_sim


# --------------------------------------------------------------------------
# Null realizations that follow the full data chain (native -> rebin -> baseline)
# --------------------------------------------------------------------------

@dataclass
class NullRealization:
    """One line-free realization on its own (possibly adaptive) grid,
    restricted to the fit interval."""
    energy: np.ndarray
    values: np.ndarray
    uncertainties: np.ndarray
    baseline: np.ndarray
    ylines: np.ndarray
    response_sigma: np.ndarray | None


def generate_null_realizations(
    spectrum_full,
    base_full,
    fit_en1,
    fit_en2,
    config,
    *,
    noise_model=None,
    window_width=None,
    stream=False,
):
    """Generate null realizations that reproduce the data-processing chain.

    The null is Poisson at channel resolution: source and background counts
    are drawn independently, rebinned, converted to density, and the empirical
    baseline is recomputed on each realization. Fixed groups are inherited
    from the data; adaptive S/N and optimal grouping are repeated for each
    realization. Optimal grouping uses the full native grid so that its count
    windows, resolution-element count and grouping origin match the observed
    procedure. With
    ``noise_model="gaussian"`` (validation only) fluctuations are instead drawn
    as N(0, sigma_i) around the baseline on the rebinned grid.

    Median-window widths are resolved on the full observed grid, including
    the ``max_range_fraction`` cap, and interpolated onto each simulation grid.
    The cap is not reapplied to the shorter simulated interval. The region
    simulated is the fit interval enlarged by the largest effective window
    on each side, to provide baseline context. Each realization is then
    restricted to ``[fit_en1, fit_en2]``.

    Parameters
    ----------
    spectrum_full : PreparedSpectrum
        Full rebinned spectrum in counts s^-1 keV^-1 as returned by
        ``prepare_spectrum``, including its ``native`` count information.
    base_full : numpy.ndarray
        Empirical baseline of ``spectrum_full`` on its full grid.
    fit_en1, fit_en2 : float
        Padded search interval.
    config : BlindLineSearchConfig
        Supplies ``num_synthetic_simulations``, ``synthetic_seed``,
        ``noise_model``, rebinning and baseline parameters. The simulation
        count must be a nonnegative integer; zero returns an empty list.
    noise_model : {'poisson', 'gaussian'} or None
        Overrides ``config.noise_model``.
    window_width : numpy.ndarray or None
        Effective median-window widths on ``spectrum_full.energy``, already
        capped on that full grid. The pipeline passes the widths used for the
        observed baseline. If omitted, they are resolved from ``config`` on
        the full grid once, before generating any realizations.

    stream : bool, default False
        Yield one realization at a time when True, avoiding storage of all
        spectra. The default retains the original list-returning API.

    ``config.n_jobs`` and ``config.simulation_batch_size`` also apply to this
    direct generator. Workers rebin and recompute baselines without searching
    for lines. Random draws retain their serial order and each realization
    retains its own adaptive energy grid.

    Returns
    -------
    list or iterator of NullRealization
        One entry per realization. Grids may differ between entries when the
        rebinning is adaptive (``'snr'`` or ``'optimal'``/``'kaastra'``).
    """
    n_sim = _validate_simulation_count(config.num_synthetic_simulations)
    if n_sim == 0:
        return iter(()) if stream else []
    if getattr(config, 'n_jobs', 1) > 1:
        # Import here because the shared worker implementation uses the
        # numerical helpers defined below in this module.
        from .parallel_search import iter_parallel_null_realizations
        realizations = iter_parallel_null_realizations(
            spectrum_full, base_full, fit_en1, fit_en2, config,
            noise_model=noise_model, window_width=window_width,
        )
        return realizations if stream else list(realizations)
    context = _prepare_null_context(
        spectrum_full, base_full, fit_en1, fit_en2, config,
        noise_model=noise_model, window_width=window_width,
    )
    rng = np.random.default_rng(config.synthetic_seed)
    realizations = (_finish_null_sample(context, _draw_null_sample(context, rng))
                    for _ in range(n_sim))
    return realizations if stream else list(realizations)


def _validate_simulation_count(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError('num_synthetic_simulations must be a nonnegative integer.')
    return int(value)


@dataclass(frozen=True)
class _NullContext:
    """Numeric inputs shared by serial execution and spawned workers.

    Callable baseline settings are resolved in the parent and are never sent
    to a worker, so notebook-defined callables remain supported.
    """
    model: str
    energy: np.ndarray
    response_energy: np.ndarray
    response_sigma: np.ndarray | None
    window_energy: np.ndarray
    window_width: np.ndarray
    min_points: int
    fit_en1: float
    fit_en2: float
    sampling: dict


def _prepare_null_context(spectrum_full, base_full, fit_en1, fit_en2, config,
                          *, noise_model=None, window_width=None):
    native = getattr(spectrum_full, "native", None)
    if native is None:
        raise ValueError("spectrum_full has no native count information; "
                         "build it with prepare_spectrum(pha, rmf, ...).")
    model = str(config.noise_model if noise_model is None else noise_model).lower()
    if model not in ("poisson", "gaussian"):
        raise ValueError("noise_model must be 'poisson' or 'gaussian'.")
    energy_full = np.asarray(spectrum_full.energy, dtype=float)
    if window_width is None:
        window_width = choose_baseline_window(
            energy_full, baseline_window=config.baseline_window,
            max_range_fraction=config.max_range_fraction,
        )
    window_width = np.asarray(window_width, dtype=float)
    if (window_width.shape != energy_full.shape
            or not np.all(np.isfinite(window_width) & (window_width > 0))):
        raise ValueError("window_width must contain one finite, positive width "
                         "per energy bin in spectrum_full.")
    margin = float(np.max(window_width))
    base_full = np.asarray(base_full, dtype=float)

    if model == "poisson":
        method = str(config.rebin_method).lower()
        region = (native.energy >= fit_en1 - margin) & (native.energy <= fit_en2 + margin)
        if method in OPTIMAL_METHODS:
            if getattr(native, 'optimal_response', None) is None:
                raise ValueError('Optimal null grouping requires a spectrum prepared with rebin_method="optimal".')
            region = np.ones(len(native.energy), dtype=bool)
        if not np.any(region):
            raise ValueError("No native channels in the simulation region.")
        energy = native.energy[region]
        widths = native.bin_width[region]
        bkg = np.clip(native.bkg_counts[region], 0.0, None)
        scale = float(native.bkg_scale)
        exposure = float(native.exposure)
        density = np.interp(energy, spectrum_full.energy, base_full)
        sampling = dict(
            mu_src=np.clip(density * exposure * widths + scale * bkg, 0.0, None),
            mu_bkg=bkg, bkg_scale=scale, exposure=exposure,
            bin_width=widths, group=native.group[region], rebin_method=method,
            rebin_scale=config.rebin_scale, rebin_min_bins=config.rebin_min_bins,
            optimal_response=getattr(native, 'optimal_response', None),
        )
        response_energy, response_sigma = native.energy, native.response_sigma
    else:
        region = (energy_full >= fit_en1 - margin) & (energy_full <= fit_en2 + margin)
        energy = energy_full[region]
        sampling = dict(baseline=base_full[region],
                        errors=np.asarray(spectrum_full.uncertainties, dtype=float)[region])
        response_energy = energy_full
        response_sigma = getattr(spectrum_full, 'response_sigma', None)
    return _NullContext(model, energy, response_energy, response_sigma,
                        energy_full, window_width, config.min_points,
                        fit_en1, fit_en2, sampling)


def _draw_null_sample(context, rng):
    """Draw in simulation order in the parent, preserving the original RNG stream."""
    inputs = context.sampling
    if context.model == 'gaussian':
        values, errors = _gaussian_realization(rng, inputs['baseline'], inputs['errors'])
        return values, errors, None
    args = (rng, inputs['mu_src'], inputs['mu_bkg'], inputs['bkg_scale'],
            np.ones_like(inputs['mu_src']))
    if inputs['rebin_method'] in OPTIMAL_METHODS:
        return _poisson_realization(*args, return_source=True)
    values, errors = _poisson_realization(*args)
    return values, errors, None


def _finish_null_sample(context, sample):
    """Rebin a draw, compute its baseline and restrict it to the fit interval."""
    values, errors, source = sample
    energy = context.energy
    inputs = context.sampling
    if context.model == 'poisson':
        method = inputs['rebin_method']
        if method == 'snr' or method in OPTIMAL_METHODS:
            extra = (dict(source_counts=source, optimal_response=inputs['optimal_response'])
                     if method in OPTIMAL_METHODS else {})
            energy, values, errors, widths, _ = rebin_counts(
                energy, values, errors, inputs['bin_width'], method=method,
                scale=inputs['rebin_scale'], min_bins=inputs['rebin_min_bins'], **extra,
            )
        else:
            energy, values, errors, widths = apply_groups(
                energy, values, errors, inputs['bin_width'], inputs['group'])
        conversion = inputs['exposure'] * widths
        values, errors = values / conversion, errors / conversion

    # Resolve the same physical window on each adaptive grid, without recapping.
    widths = np.interp(energy, context.window_energy, context.window_width)
    baseline = moving_median_x(energy, values, width=widths, min_points=context.min_points)
    excess = np.maximum(values - baseline, 0.0)
    response = None
    if context.response_sigma is not None:
        response = np.interp(energy, context.response_energy, context.response_sigma,
                             left=context.response_sigma[0], right=context.response_sigma[-1])
    inside = (energy >= context.fit_en1) & (energy <= context.fit_en2)
    return NullRealization(
        energy=energy[inside], values=values[inside], uncertainties=errors[inside],
        baseline=baseline[inside], ylines=excess[inside],
        response_sigma=None if response is None else response[inside],
    )
