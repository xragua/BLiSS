"""Generate null (line-free) synthetic spectra for BLiSS significance estimates.

The null is generated in detector space: the empirical baseline measured on
the rebinned spectrum is interpolated to the native channel grid, converted to
expected counts (times exposure and channel width, plus the scaled background
that was subtracted), and source and background counts are drawn from
independent Poisson distributions. Each realization is then summed into the
same bins as the data, converted to counts s^-1 keV^-1, and its empirical
baseline is recomputed, so the non-linearity of the baseline estimator is
propagated empirically into the null distribution. This is the detector-space
analogue of ``fakeit`` with the empirical baseline as the model.

A Gaussian model (fluctuations N(0, sigma_i) around the baseline on the
rebinned grid) is retained only for validation and comparison.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..line_search.empirical_baseline import base_calculator
from ..spectrum_data.rebinning_tools import rebin_counts, apply_groups


# --------------------------------------------------------------------------
# Single-realization generators
# --------------------------------------------------------------------------

def _gaussian_realization(rng, baseline, errors):
    y_sim = baseline + rng.normal(loc=0.0, scale=errors)
    return y_sim, errors.copy()


def _poisson_realization(rng, mu_src, mu_bkg, bkg_scale, conv):
    """Draw source and background counts and combine them like the loader.

    Net value  : (S - s B) / conv
    Uncertainty: sqrt(S + s^2 B) / conv   (quadrature, as in the loader)

    A floor of one count is applied to the variance so that bins with zero
    simulated counts keep a finite, positive uncertainty. This is the only
    deliberate departure from the loader, which drops such bins.
    """
    s_sim = rng.poisson(mu_src).astype(float)
    b_sim = rng.poisson(mu_bkg).astype(float) if bkg_scale > 0 else np.zeros_like(s_sim)
    y_sim = (s_sim - bkg_scale * b_sim) / conv
    var = np.maximum(s_sim + bkg_scale ** 2 * b_sim, 1.0)
    sy_sim = np.sqrt(var) / conv
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
):
    """Generate null realizations that reproduce the data-processing chain.

    The null is Poisson at channel resolution: source and background counts
    are drawn independently, summed into the *same* bins as the data
    (``rebin_counts`` with the configured method), converted to density, and
    the empirical baseline is recomputed on each realization. With
    ``noise_model="gaussian"`` (validation only) fluctuations are instead drawn
    as N(0, sigma_i) around the baseline on the rebinned grid.

    The region simulated is the fit interval enlarged by one baseline window
    on each side, so the recomputed baseline has the same edge behaviour as
    the data baseline; each realization is then restricted to
    ``[fit_en1, fit_en2]``.

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
        ``noise_model``, rebinning and baseline parameters.
    noise_model : {'poisson', 'gaussian'} or None
        Overrides ``config.noise_model``.

    Returns
    -------
    list of NullRealization
        One entry per realization. Grids may differ between entries when the
        rebinning is adaptive (``'snr'``).
    """
    native = getattr(spectrum_full, "native", None)
    if native is None:
        raise ValueError("spectrum_full has no native count information; "
                         "build it with prepare_spectrum(pha, rmf, ...).")
    model = str(config.noise_model if noise_model is None else noise_model).lower()
    if model not in ("poisson", "gaussian"):
        raise ValueError("noise_model must be 'poisson' or 'gaussian'.")
    rng = np.random.default_rng(config.synthetic_seed)
    w = config.baseline_window
    if callable(w):
        w = w(np.asarray(spectrum_full.energy, dtype=float))
    margin = float(np.max(w))
    n_sim = int(config.num_synthetic_simulations)
    base_full = np.asarray(base_full, dtype=float)

    def _finish(e, y, sy, resp_src_e, resp_src):
        """Baseline, positive excess, response sigma, and cut to the interval."""
        base = base_calculator(
            e, y,
            baseline_window=config.baseline_window,
            max_range_fraction=config.max_range_fraction,
            min_points=config.min_points,
        ) 
        ylines = np.maximum(y - base, 0.0)
        resp = None
        if resp_src is not None:
            resp = np.interp(e, resp_src_e, resp_src,
                             left=resp_src[0], right=resp_src[-1])
        inside = (e >= fit_en1) & (e <= fit_en2)
        return NullRealization(
            energy=e[inside], values=y[inside], uncertainties=sy[inside],
            baseline=base[inside], ylines=ylines[inside],
            response_sigma=None if resp is None else resp[inside],
        )

    realizations = []

    # ---------------------------------------------------------------- Poisson
    if model == "poisson":
        region = (native.energy >= fit_en1 - margin) & (native.energy <= fit_en2 + margin)
        if not np.any(region):
            raise ValueError("No native channels in the simulation region.")
        e_nat = native.energy[region]
        de_nat = native.bin_width[region]
        bkg = np.clip(native.bkg_counts[region], 0.0, None)
        group_nat = native.group[region]
        s = float(native.bkg_scale)
        T = float(native.exposure)

        # Line-free expectation per native channel: rebinned baseline (density)
        # interpolated to the channel grid, times T*dE, plus the scaled
        # background that the loader subtracted.
        mu_density = np.interp(e_nat, spectrum_full.energy, base_full)
        mu_src = np.clip(mu_density * T * de_nat + s * bkg, 0.0, None)

        for _ in range(n_sim):
            net, unc = _poisson_realization(rng, mu_src, bkg, s, np.ones_like(mu_src))
            if config.rebin_method == "snr":
                e, n, sn, de, _g = rebin_counts(
                    e_nat, net, unc, de_nat, method="snr",
                    scale=config.rebin_scale, min_bins=config.rebin_min_bins)
            else:
                e, n, sn, de = apply_groups(e_nat, net, unc, de_nat, group_nat)
            conv = T * de
            realizations.append(_finish(e, n / conv, sn / conv,
                                        native.energy, native.response_sigma))

    # --------------------------------------------------------------- Gaussian
    else:
        e_all = np.asarray(spectrum_full.energy, dtype=float)
        region = (e_all >= fit_en1 - margin) & (e_all <= fit_en2 + margin)
        e = e_all[region]
        base_r = base_full[region]
        err_r = np.asarray(spectrum_full.uncertainties, dtype=float)[region]
        resp_src = getattr(spectrum_full, "response_sigma", None)
        for _ in range(n_sim):
            y, sy = _gaussian_realization(rng, base_r, err_r)
            realizations.append(_finish(e, y, sy, e_all, resp_src))

    return realizations
