"""Generate null (line-free) synthetic spectra for BLiSS significance estimates.

The null is generated in detector space: the empirical baseline measured on
the rebinned spectrum is interpolated to the native channel grid, converted to
expected counts (times exposure and channel width, plus the scaled background
that was subtracted), and source and background counts are drawn from
independent Poisson distributions. Each realization is then rebinned using
the configured rule, converted to counts s^-1 keV^-1, and its empirical
baseline is recomputed, so the non-linearity of the baseline estimator is
propagated empirically into the null distribution. Poisson fits retain the
separate source and background draws, omit artificial variance floors, and
estimate the continuum from locally accumulated counts after excluding strong
line bins from that estimate. Gaussian fits retain
the legacy uncertainty weighting and median continuum. The initial window
widths and maximum allowed width are inherited from the observed working band;
each realization enlarges windows as needed to reach min_points. This is the
detector-space analogue of ``fakeit`` with the empirical baseline as the model.

A Gaussian model (fluctuations N(0, sigma_i) around the baseline on the
rebinned grid) is retained only for validation and comparison.
"""
from __future__ import annotations
from dataclasses import dataclass
from numbers import Integral
import numpy as np

from ..line_search.empirical_baseline import (
    choose_baseline_window, moving_median_x, moving_poisson_continuum_x,
    _resolve_baseline_window_limit,
)
from ..line_search.poisson_fitting import PoissonFitData, validate_poisson_groups
from ..spectrum_data.rebinning_tools import rebin_counts, apply_groups, OPTIMAL_METHODS, COUNT_METHODS, validate_oversampling


# --------------------------------------------------------------------------
# Single-realization generators
# --------------------------------------------------------------------------

def _gaussian_realization(rng, baseline, errors):
    y_sim = baseline + rng.normal(loc=0.0, scale=errors)
    return y_sim, errors.copy()


def _poisson_realization(rng, mu_src, mu_bkg, bkg_scale, conv, *,
                         return_source=False, return_background=False):
    """Draw source and background counts and combine them like the loader.

    Net value  : (S - s B) / conv
    Uncertainty: sqrt(S + s^2 B) / conv   (quadrature, as in the loader)

    Native variances, including zero and positive values below one, are left
    unchanged. Poisson fits preserve zero variance after grouping as well;
    the Gaussian-fit compatibility branch applies its fallback after grouping.

    ``return_source=True`` also returns the gross source-region draw, used to
    repeat optimal binning before background subtraction.
    ``return_background=True`` returns both raw draws, without consuming any
    additional random numbers, for the source/background Poisson likelihood.
    """
    s_sim = rng.poisson(mu_src).astype(float)
    b_sim = rng.poisson(mu_bkg).astype(float) if bkg_scale > 0 else np.zeros_like(s_sim)
    y_sim = (s_sim - bkg_scale * b_sim) / conv
    var = s_sim + bkg_scale ** 2 * b_sim
    sy_sim = np.sqrt(var) / conv
    if return_background:
        return y_sim, sy_sim, s_sim, b_sim
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
    poisson_data: PoissonFitData | None = None


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
    from the data; adaptive S/N, minimum-count and optimal grouping are repeated
    for each realization. All three adaptive methods use the full native grid
    so that their grouping origin and full-band context match the observed
    procedure. The fit interval is selected only after grouping and baseline
    estimation. With
    ``noise_model="gaussian"`` (validation only) fluctuations are instead drawn
    as N(0, sigma_i) around the baseline on the rebinned grid.

    Initial window widths and their physical cap are inherited from the
    observed spectrum and interpolated onto each simulation grid. When the
    pipeline has selected a baseline working domain and stored its cap, every
    simulation uses that entire already-restricted domain, including fixed
    grouping and Gaussian draws. It is not cropped or regrouped partially a
    second time. For low-level spectra without an explicit domain cap, the
    previous full-spectrum cap and simulation-context rules are retained.
    Impossible windows raise ValueError and abort both serial and parallel
    execution. Each realization is then
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
        Initial median-window widths on ``spectrum_full.energy``, already
        capped on that full grid. The pipeline passes the initial widths,
        before any expansion to reach min_points. If omitted, they are resolved
        from ``config`` on the full grid once, before generating any realizations.

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
    max_window_width: float
    min_points: int
    fit_en1: float
    fit_en2: float
    sampling: dict


def _prepare_null_context(spectrum_full, base_full, fit_en1, fit_en2, config,
                          *, noise_model=None, window_width=None):
    validate_oversampling(config.oversampling, config.rebin_method)
    native = getattr(spectrum_full, "native", None)
    if native is None:
        raise ValueError("spectrum_full has no native count information; "
                         "build it with prepare_spectrum(pha, rmf, ...).")
    model = str(config.noise_model if noise_model is None else noise_model).lower()
    if model not in ("poisson", "gaussian"):
        raise ValueError("noise_model must be 'poisson' or 'gaussian'.")
    poisson_fitting = getattr(spectrum_full, 'poisson_data', None) is not None
    if poisson_fitting and model != 'poisson':
        raise ValueError("Poisson fits require noise_model='poisson'; Gaussian "
                         "null draws are not Poisson count observations.")
    energy_full = np.asarray(spectrum_full.energy, dtype=float)
    domain_cap = getattr(spectrum_full, 'baseline_max_window_width', None)
    has_working_domain = domain_cap is not None
    max_window_width = _resolve_baseline_window_limit(
        energy_full, config.max_range_fraction, domain_cap,
    )
    if window_width is None:
        cap_option = dict(max_window_width=max_window_width) if has_working_domain else {}
        window_width = choose_baseline_window(
            energy_full, baseline_window=config.baseline_window,
            max_range_fraction=config.max_range_fraction, **cap_option,
        )
    window_width = np.asarray(window_width, dtype=float)
    if (window_width.shape != energy_full.shape
            or not np.all(np.isfinite(window_width) & (window_width > 0))):
        raise ValueError("window_width must contain one finite, positive width "
                         "per energy bin in spectrum_full.")
    # The pipeline already chose the native working domain before grouping.
    # A second crop could split fixed groups or change the continuum context.
    # Low-level spectra without this metadata retain the earlier context rule.
    margin = max_window_width
    base_full = np.asarray(base_full, dtype=float)

    if model == "poisson":
        method = str(config.rebin_method).lower()
        region = (np.ones(len(native.energy), dtype=bool) if has_working_domain else
                  (native.energy >= fit_en1 - margin) & (native.energy <= fit_en2 + margin))
        if method in OPTIMAL_METHODS:
            if getattr(native, 'optimal_response', None) is None:
                raise ValueError('Optimal null grouping requires a spectrum prepared with rebin_method="optimal".')
            region = np.ones(len(native.energy), dtype=bool)
        elif method == 'snr' or method in COUNT_METHODS:
            # The observed groups start at the first native channel. Cropping
            # before grouping would change the origin of adaptive SNR/count bins.
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
            oversampling=config.oversampling,
            optimal_response=getattr(native, 'optimal_response', None),
            poisson_fitting=poisson_fitting,
        )
        response_energy, response_sigma = native.energy, native.response_sigma
    else:
        region = (np.ones(len(energy_full), dtype=bool) if has_working_domain else
                  (energy_full >= fit_en1 - margin) & (energy_full <= fit_en2 + margin))
        energy = energy_full[region]
        sampling = dict(baseline=base_full[region],
                        errors=np.asarray(spectrum_full.uncertainties, dtype=float)[region])
        response_energy = energy_full
        response_sigma = getattr(spectrum_full, 'response_sigma', None)
    return _NullContext(model, energy, response_energy, response_sigma,
                        energy_full, window_width, max_window_width, config.min_points,
                        fit_en1, fit_en2, sampling)


def _draw_null_sample(context, rng):
    """Draw in simulation order in the parent, preserving the original RNG stream."""
    inputs = context.sampling
    if context.model == 'gaussian':
        values, errors = _gaussian_realization(rng, inputs['baseline'], inputs['errors'])
        return values, errors, None
    args = (rng, inputs['mu_src'], inputs['mu_bkg'], inputs['bkg_scale'],
            np.ones_like(inputs['mu_src']))
    if inputs.get('poisson_fitting', False):
        return _poisson_realization(*args, return_background=True)
    if inputs['rebin_method'] in COUNT_METHODS or inputs['rebin_method'] in OPTIMAL_METHODS:
        return _poisson_realization(*args, return_source=True)
    values, errors = _poisson_realization(*args)
    return values, errors, None


def _finish_null_sample(context, sample):
    """Rebin a draw, compute its baseline and restrict it to the fit interval."""
    values, errors, source = sample[:3]
    background = sample[3] if len(sample) == 4 else None
    energy = context.energy
    inputs = context.sampling
    poisson_data = None
    if context.model == 'poisson':
        poisson_fitting = inputs.get('poisson_fitting', False)
        method = inputs['rebin_method']
        if method == 'snr' or method in OPTIMAL_METHODS or method in COUNT_METHODS:
            extra = (dict(source_counts=source, optimal_response=inputs['optimal_response'],
                          oversampling=inputs['oversampling'])
                     if method in OPTIMAL_METHODS else {})
            if method in COUNT_METHODS:
                extra = dict(source_counts=source)
            if poisson_fitting:
                extra['zero_variance_floor'] = False
            energy, values, errors, widths, group = rebin_counts(
                energy, values, errors, inputs['bin_width'], method=method,
                scale=inputs['rebin_scale'], min_bins=inputs['rebin_min_bins'], **extra,
            )
        else:
            group = inputs['group']
            extra = dict(zero_variance_floor=False) if poisson_fitting else {}
            energy, values, errors, widths = apply_groups(
                energy, values, errors, inputs['bin_width'], group, **extra)
        if poisson_fitting:
            validate_poisson_groups(context.energy, inputs['bin_width'], group)
            if source is None or background is None:
                raise ValueError('Poisson fitting requires separate source and background draws.')
            # Group ids can be nonconsecutive after a context crop. Use the
            # same sorted retained ids as apply_groups, including dropped bins.
            keep = np.asarray(group) >= 0
            _, inverse = np.unique(np.asarray(group)[keep], return_inverse=True)
            poisson_data = PoissonFitData(
                source_counts=np.bincount(inverse, weights=np.asarray(source)[keep]),
                background_counts=np.bincount(inverse, weights=np.asarray(background)[keep]),
                background_scale=inputs['bkg_scale'], exposure=inputs['exposure'],
                bin_width=widths,
            )
        conversion = inputs['exposure'] * widths
        values, errors = values / conversion, errors / conversion

    # Start from the same physical scale, then expand on this realization's grid.
    # The physical cap remains the one resolved for the observed working domain.
    window_widths = np.interp(energy, context.window_energy, context.window_width)
    if poisson_data is None:
        baseline = moving_median_x(
            energy, values, width=window_widths, min_points=context.min_points,
            max_width=context.max_window_width,
        )
    else:
        baseline = moving_poisson_continuum_x(
            energy, poisson_data.source_counts, poisson_data.background_counts,
            poisson_data.background_scale, poisson_data.exposure,
            poisson_data.bin_width, width=window_widths, min_points=context.min_points,
            max_width=context.max_window_width,
        )
    # Positive excess only locates blocks. Poisson fits use the raw counts;
    # Gaussian fits retain the signed values - baseline within each block.
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
        poisson_data=None if poisson_data is None else poisson_data.sliced(inside),
    )
