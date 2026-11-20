# Functions

This page documents the main public functions used in the current BLiSS workflow.

BLiSS works directly with folded PHA spectra and their RMF response files. Rebinning is performed in count space, the working spectrum is converted to counts s⁻¹ keV⁻¹, and the same data-processing chain is applied to the synthetic null spectra.

A standard workflow is:

1. Prepare and inspect the spectrum with `prepare_spectrum`.
2. Inspect the empirical continuum using the estimator appropriate to the fit statistic (see below).
3. Run the blind local search with `find_candidate_lines`.
4. Filter the returned catalogue using the global Monte Carlo p-values, and any analysis-specific criteria.
5. Run the optional simultaneous multi-Gaussian fit with `fit_global`.
6. Optionally identify transitions and plot the results.

BLiSS can also be used as a scriptable candidate generator within a broader spectral-analysis pipeline. If no simulation-based assessment is required, the null simulations can be disabled:

```python
candidate_lines = find_candidate_lines(
    pha=pha,
    rmf=rmf,
    arf=arf,
    bkg=bkg,
    en1=0.5,
    en2=8.0,
    num_synthetic_simulations=0,
    output_dir="results/bliss_candidates",
    oversampling=None,
)
```

This faster mode:

1. prepares and rebins the spectrum internally;
2. detects and locally fits the candidate emission lines;
3. returns the candidates as a `pandas.DataFrame`;
4. automatically saves the same catalogue as `candidate_lines.csv`.

The returned table or CSV file can then be passed programmatically to another spectral-fitting package or incorporated into an automated batch-analysis workflow. With `num_synthetic_simulations=0`, the `p_global_*` columns remain unavailable (`NaN`).

The main functions are available from the top-level package:

```python
import numpy as np

from bliss import (
    BlindLineSearchConfig,
    BlindLineSearchPipeline,
    LineIdentifier,
    load_fits_spectrum,
    rebin_counts,
    prepare_spectrum,
    base_calculator,
    find_candidate_lines,
    fit_global,
    plot_global_fit,
    identify_line,
    add_most_probable_ion,
    get_all_compatible_lines,
)
```

---

## Spectrum preparation

### `prepare_spectrum`

```python
prepare_spectrum(
    pha_path,
    rmf_path,
    arf_path=None,
    background_path=None,
    *,
    rebin_method="optimal",
    rebin_scale=None,
    rebin_min_bins=1,
    oversampling=None,
    assume_poisson_rate_errors=False,
    fit_statistic="auto",
    energy_range=None,
)
```

Load a PHA spectrum, optionally subtract its background, rebin it while conserving counts, and convert it to a density spectrum in counts s⁻¹ keV⁻¹.

The returned `PreparedSpectrum` also retains the native count-space information required to generate Poisson null realizations with the same rebinning as the observed spectrum.

`energy_range=(lo, hi)` selects native channels by their centres before grouping,
retaining whole channels and their counts. Source, background and response
metadata remain aligned on that working grid. A direct
call with `energy_range=None` retains the full available input grid. The search
pipeline supplies its resolved continuum context instead; see **Search and
continuum domains** below. File integrity, declared error models and
source/background channel alignment are still validated at loading. Poisson
count-value compatibility is checked on all source/background channels in the
requested native domain, including any channels later excluded by the loader.

Every rebinning method retains finite empty native channels and their original
zero variances. Variances are summed without adding a minimum to each channel.
All positive variances are preserved, including values below one. For Poisson
fits, zero-variance output bins also remain zero: these count-based error
estimates are not likelihood weights or confidence intervals. Empty bins still
contribute to the Poisson likelihood. The Gaussian-fit compatibility branch
retains its variance-`1` fallback only for zero-variance output bins, before
conversion to density. The selected policy is shared by data and simulations.
For `rebin_method="none"`, each native channel is already an output bin.

`fit_statistic="auto"` selects Poisson fitting for recoverable nonnegative integer
`COUNTS` without explicit statistical errors, nonzero systematic errors, or a
`POISSERR=False` declaration. `RATE`/`COUNT_RATE` additionally require
`assume_poisson_rate_errors=True`, no supplied error column, and rates that
recover integer counts using the exposure. These requirements apply to both
source and background. Other supported inputs retain Gaussian uncertainty
weighting. Forcing `fit_statistic="poisson"` on incompatible inputs raises an
error; `"gaussian"` explicitly retains the previous fitting method.

**Parameters**

- `pha_path` *(path-like)* – Source PHA spectrum.
- `rmf_path` *(path-like)* – Redistribution matrix. It supplies the channel energy bounds and the instrumental line-spread width.
- `arf_path` *(path-like or None, default=None)* – Optional ancillary response. When provided, BLiSS can flag candidates close to sharp effective-area structure.
- `background_path` *(path-like or None, default=None)* – Optional background PHA spectrum. It is scaled and subtracted before rebinning.
- `assume_poisson_rate_errors` *(bool, default=False)* – Explicitly allow Poisson estimation of missing `RATE`/`COUNT_RATE` errors, for both source and background. Supplied errors always take precedence; see `load_fits_spectrum` below.
- `fit_statistic` *{"auto", "poisson", "gaussian"}, default="auto"* – Select the likelihood and its compatible continuum estimator. Use matching settings in the search and a later global fit.
- `energy_range` *(pair of floats or None, default=None)* – Native-channel energy interval in keV to retain before rebinning. `None` keeps the full input grid for standalone preparation; it does not apply the search defaults.
- `rebin_method` *{"none", "bins", "snr", "counts", "mincounts", "resolution", "optimal", "kaastra"}, default="optimal"* – Count-space grouping rule; `"kaastra"` is an alias for `"optimal"`.
- `rebin_scale` *(int, float, or None)* – Number of native bins, target S/N, minimum gross counts, or fixed energy width, depending on `rebin_method`.
- `rebin_min_bins` *(int, default=1)* – Minimum number of native bins per output bin for S/N or minimum-count rebinning.

All grouping methods stop at energy gaps. For `bins`, `snr`, and `counts`,
`remainder="merge"` keeps a short trailing group by merging it within its own
contiguous segment (or retaining the whole segment if it is too short).
For S/N grouping this can lower the final S/N below the target, particularly
with background-subtracted counts. `resolution` uses intervals anchored at
the start of each contiguous segment; actual widths follow whole native
channels and need not equal the requested interval width.

The rebinning methods are:

| Method | Meaning of `rebin_scale` | Behaviour |
| --- | --- | --- |
| `"none"` | `None` | Keep the native channel grid. |
| `"bins"` | Number of channels | Sum consecutive native bins. |
| `"snr"` | Target S/N | Accumulate bins until the target S/N is reached. |
| `"counts"` / `"mincounts"` | Minimum gross source counts | Accumulate counts before background subtraction. |
| `"resolution"` | Width in keV | Group channels into fixed-width energy intervals without splitting input bins. |
| `"optimal"` / `"kaastra"` | `None` | Kaastra–Bleeker data binning from the RMF and gross observed counts. Keep `rebin_min_bins=1`. |

All null simulations retain the full working domain of the observed analysis,
after its initial native-channel selection. Poisson draws use its native grid;
Gaussian validation draws use its rebinned grid. Adaptive `"snr"`, `"counts"` and
`"optimal"` grouping starts at the native working grid's first channel and is repeated for
each Poisson draw. Fixed grouping retains the observed groups without a further
context crop. The requested fit interval is selected after grouping and baseline
estimation. Serial and parallel runs use the same procedure.

Optimal binning implements equations 36–37 and Appendix B.1 of
[Kaastra & Bleeker (2016)](https://arxiv.org/html/1601.05309#A2.SS1).
It measures the FWHM and the core-to-total count correction directly from
the RMF, including response wings. It groups whole native channels, preserves
the last bin and separates gaps. Channels without a measurable response core
remain ungrouped; an entirely unusable RMF raises an error. This groups the
observed data; it does not rewrite the RMF or construct the paper's model grid.


Optimal binning is now the default for the configuration, spectrum preparation,
search wrappers, and `rebin_counts`. Set `rebin_method="none"` (or `method="none"`
in `rebin_counts`) explicitly to retain the native grid.

The optional positive, finite `oversampling` factor sets a lower bound on group
size: `max(optimal size, FWHM / oversampling)`. Both sizes use the native channel
coordinates measured from the RMF. The oversampling bound is rounded **up** to
whole channels and enforced for every measurable response core in the group.
Thus this option can produce wider bins than pure optimal grouping. For example,
FWHM = 8 native channels and `oversampling=2` require at least 4 channels per group.
`oversampling=None` retains the original optimal rule.

With this limit enabled, a short final group is merged with its predecessor in
the same contiguous segment. Gaps and channels without a measurable response core
remain boundaries; a whole segment that is too short is retained as one group.
Channels without measurable response remain ungrouped. These boundary cases may
fall below the requested minimum size. No counts are discarded or channels split.
The option is supported only with `optimal`/`kaastra`; combining it with another
method raises an error. Poisson simulations recompute the combined grouping for
each draw, including when run in parallel.

```python
config = BlindLineSearchConfig(oversampling=3)
# Or supply the same optional factor directly:
spectrum = prepare_spectrum("source.pha", "response.rmf", oversampling=3)
lines = find_candidate_lines("source.pha", "response.rmf", oversampling=3)
```

In `find_candidate_lines`, omitted options inherit
those fields from `config`; without a config they use `BlindLineSearchConfig`
defaults, including optimal grouping with no additional limit. An explicit
`oversampling` overrides the config value: pass `oversampling=None` to remove a
configured limit. `rebin_method=None` continues to inherit the method. Use
matching settings when preparing a spectrum for a later global fit.

No background file is required. With a background, the groups are chosen from
the gross source-region counts; scaled background subtraction and uncertainty
propagation still determine the returned spectrum. Native variances are summed
without adding a variance floor to each empty channel. Poisson fitting also
omits a floor after grouping; the Gaussian branch retains its final-bin fallback.
Each Poisson realization repeats optimal grouping on its own gross
counts over the full native working grid before selecting the search interval.

```python
spectrum = prepare_spectrum(
    "source.pha", "response.rmf",
    rebin_method="optimal",
)

# The same option is available in find_candidate_lines and BlindLineSearchConfig.
candidates = find_candidate_lines(
    "source.pha", "response.rmf",
    rebin_method="optimal",
    num_synthetic_simulations=100,
    synthetic_seed=123,
)
```

**Returns**

- `PreparedSpectrum` – A container with the following main attributes:

| Attribute | Contents |
| --- | --- |
| `energy` | Rebinned energy-bin centres in keV. |
| `values` | Rebinned spectrum in counts s⁻¹ keV⁻¹,. |
| `uncertainties` | Propagated count-based error estimates in density units; not Poisson likelihood weights or exact intervals. |
| `bin_width` | Rebinned energy-bin widths in keV. |
| `response_sigma` | RMF-derived Gaussian-equivalent instrumental sigma, when available. |
| `arf_energy`, `effective_area` | ARF grid and effective area, when supplied. |
| `native` | Native counts, exposure, background information, and grouping used by the null simulations. |
| `poisson_data` | Separate grouped source/background counts, scale and conversion factors for Poisson fits; `None` for Gaussian fits. |
| `baseline_max_window_width` | Fixed physical continuum-window cap resolved by the pipeline; `None` for standalone preparation unless set explicitly. |

**Example**

```python
spectrum = prepare_spectrum(
    pha_path=pha,
    rmf_path=rmf,
    arf_path=arf,
    background_path=bkg,
    rebin_method="bins",
    rebin_scale=2,
)

print(
    f"{len(spectrum.energy)} bins, "
    f"median width {1e3 * np.median(spectrum.bin_width):.2f} eV, "
    f"exposure {spectrum.native.exposure:.0f} s"
)
```

### `load_fits_spectrum`

```python
load_fits_spectrum(
    pha_path,
    background_path=None,
    rmf_path=None,
    arf_path=None,
    *,
    subtract_background=True,
    as_arrays=False,
    assume_poisson_rate_errors=False,
)
```

Load a folded PHA spectrum without the count-to-density conversion performed by `prepare_spectrum`.

`RATE` and `COUNT_RATE` spectra must provide `STAT_ERR`, `ERROR` or `ERR` by
default. Without an error column, loading raises `ValueError` with instructions
to provide errors or explicitly set `assume_poisson_rate_errors=True`. A
`POISSERR=True` header alone does not enable this option.

With the option enabled, missing rate errors are estimated as
`sqrt(rate * exposure) / exposure`, using each file's own `EXPOSURE` before
background scaling and subtraction. This requires finite, non-negative rates
and a finite, positive exposure, and assumes the rate is raw Poisson counts
divided by exposure. Supplied error columns are preserved regardless of the
option. Count-column error handling is unchanged.

The same option applies to the loader, `read_pha_metadata`, `prepare_spectrum`,
`BlindLineSearchConfig`, `find_candidate_lines`.
Source and background follow the same policy, including the native count
information used by null simulations. For example:

```python
config = BlindLineSearchConfig(assume_poisson_rate_errors=True)
candidates = find_candidate_lines("source.pha", "response.rmf", config=config)
# Equivalently: find_candidate_lines(..., assume_poisson_rate_errors=True)
```

Source and background values and errors are matched by `CHANNEL` (or `PI`),
regardless of their row order. Scaling by `BACKSCAL` and `EXPOSURE` is applied
after matching. The native count information used for Poisson simulations
uses the same matching rule.

The background must contain every source channel retained by the RMF; without
an RMF, it must contain every source channel. Extra background channels are
ignored. Missing required channels, duplicate channel IDs in either spectrum,
or IDs that are not finite integers raise `ValueError`. An explicit background
entry with zero counts is valid; an absent entry is not assumed to be zero.
For example: `Background spectrum is missing required source channels: 42.`
With `subtract_background=False`, the background file is not read or validated.

By default, this returns a sorted `Spectrum` object. With `as_arrays=True`, it returns `(energy, values, uncertainties, bin_width)`. For a normal BLiSS search, prefer `prepare_spectrum`, because it also stores the native information required by the Poisson simulations.

### `rebin_counts`

```python
rebin_counts(
    energy,
    counts,
    uncertainties,
    bin_width,
    method="optimal",
    scale=None,
    min_bins=1,
    remainder="merge",
    *,
    rmf_path=None,
    source_counts=None,
    optimal_response=None,
    oversampling=None,
    zero_variance_floor=True,
)
```

Low-level count-conserving rebinning. Counts are summed, uncertainties are added in quadrature, and bin widths are added. With the compatibility default `zero_variance_floor=True`, a final uncertainty of exactly zero is replaced by one in count space; positive uncertainties, including those below one, are unchanged. `False` preserves zero output variance. Poisson spectrum preparation and nulls pass `False` for every grouping method. SNR grouping always uses the original native variances.

It returns:

```text
energy_new, counts_new, uncertainties_new, bin_width_new, group
```

`group` gives the output-bin index assigned to each native bin; `-1` marks a dropped bin. `apply_groups(energy, counts, uncertainties, bin_width, group, zero_variance_floor=True)` applies an existing grouping with the same optional zero-variance rule. This is used when fixed groups must be retained for the data and its null realizations.

---

## Empirical baseline

Poisson searches use `poisson_base_calculator` from
`bliss.line_search.empirical_baseline`. It estimates a running count-density
mean while iteratively excluding strong emission from the continuum estimate.
The mean is `max(sum(source - scale * background), 0) /
(exposure * sum(bin_width))`, using only unmasked bins in both numerator and
denominator. This prevents a strong narrow line from being spread into a broad
shoulder by the moving window, while avoiding the zero median common in sparse
spectra. The fitting grid and observed counts are unchanged: masked bins still
participate in candidate detection and line fitting.

The fixed procedure performs at most five masking iterations. Without a
background, it checks each positive excess against a conditional binomial
upper tail: given the total counts in its real, unmasked window, the bin's
expected fraction is its width divided by the window's retained width. The
threshold is `0.01 / (n_bins * 5)`. Reflected samples are used only for smoothing,
not as independent observations in this test. With a background, conservative
Garwood upper bounds for source-window and background counts form an upper
gross-count predictor; a Poisson upper-tail check uses one third of that
threshold. Net counts are never treated as Poisson observations for this check.
Seeds must lie above the current net continuum. Each seed also masks one
adjacent bin on each side where available, without crossing energy gaps.

Windows expand to retain `min_points` unmasked samples, subject to the same
fixed cap and reflected-edge convention. If the requirement becomes impossible,
the run stops rather than silently relaxing it. Unequal bin widths are handled
through summed counts and widths. Empty windows are not assigned invented
counts. The entire procedure is repeated independently on every null spectrum,
including parallel searches. The Gaussian branch retains its running median.

This is an empirical masking heuristic, not a physical continuum fit or a
calibrated line-significance test: `0.01` is **not** a line p-value. The binomial
construction assumes a locally constant count density; adaptive grouping and
iterative selection require validation of the complete procedure. Faint wings,
broad or crowded emission, and strong backgrounds can still contaminate the
continuum. The nonnegative net-mean constraint can also bias a weak continuum
upwards under strong background; this change does not correct that limitation.
Inspect the continuum and rerun the candidate search and null simulations
before interpreting p-values from this estimator.

The lower-level `moving_poisson_mean_x(..., exclude_mask=None)` remains a pure
count-density mean, with optional explicit exclusions. The automatic estimator
uses `moving_poisson_continuum_x`; `poisson_base_calculator(..., return_info=True)`
also returns `info["excluded_mask"]`, aligned with its input grid. These fixed
masking rules add no new configuration parameters.

`fit_global` selects the estimator from `spectrum.poisson_data` when `base` is
omitted. The direct `base_calculator` API below remains the running median for
the Gaussian branch; it does not automatically inspect a `PreparedSpectrum`.

### Search and continuum domains

The default nominal search band is **1.7–12.0 keV**. These are editable defaults,
not instrument-dependent limits: explicit `en1` and `en2` take precedence. With
`pad = max(energy_pad, 0)`, the pipeline resolves the following physical bounds
before loading and regrouping the working channels:

```text
fit_lo = en1 - pad
fit_hi = en2 + pad
max_window_width = max_range_fraction * (fit_hi - fit_lo)
automatic_context = (fit_lo - max_window_width / 2,
                     fit_hi + max_window_width / 2)
```

`baseline_context=None` selects this automatic context. An explicit `(lo, hi)`
must contain the whole automatic context and may only enlarge it. The context
is applied to native channels **before** rebinning, so a remote empty tail
outside it cannot create an enormous final group or block continuum estimation.
Only available channels are retained; the response remains the original RMF.

The continuum is estimated over the entire selected working context. The window
cap above stays fixed: it is not recomputed from rebinned centres, from an
explicitly enlarged context, or from a simulated grid. `min_points` remains
strict throughout that context. An impossible window still stops the run,
including when the user explicitly enlarges the context into a sparse tail.

Initial `baseline_window` values are resolved on the observed working grid.
A supplied array must match that rebinned grid; a callable is evaluated there
once. Null grids receive interpolated initial widths and the same fixed cap,
then perform their own local expansion to reach `min_points`.

Data and nulls start from exactly the same native working domain. Changing this
domain can change adaptive group boundaries, continuum estimates, candidate
measurements and random draws, even with the same seed. Recompute the local
search and its simulations before interpreting p-values under the new settings.

### `base_calculator`

```python
base_calculator(
    x,
    y,
    baseline_window=0.4,
    max_range_fraction=0.2,
    min_points=3,
    return_info=False,
    *,
    max_window_width=None,
)
```

Estimate the BLiSS empirical baseline with a running median defined in physical spectral units.

The baseline is an empirical description of the observed spectral shape, not a physical continuum model. Its window should be broad enough not to absorb narrow emission lines, but narrow enough to follow genuine changes in the underlying spectrum. 

**Parameters**

- `x` *(array-like)* – Spectral coordinate, normally energy in keV.
- `y` *(array-like)* – Spectral values on the same grid.
- `baseline_window` *(float, array-like, or callable, default=0.4)* – Preferred running-median width in the same units as `x`. A scalar uses one width everywhere; an array supplies one width per point; a callable `f(x)` allows the width to vary with energy.
- `max_range_fraction` *(float, default=0.2)* – For standalone calls without an explicit cap, maximum window as a fraction of the input energy span.
- `max_window_width` *(float or None, default=None)* – Optional fixed positive physical cap. The pipeline supplies its resolved cap so that regrouping and simulations cannot change it. `None` retains the standalone input-span rule.
- `min_points` *(int, default=3)* – Positive minimum number of finite samples in each local median window, including reflected edge samples. Windows expand locally to reach this minimum without exceeding the resolved cap.
- `return_info` *(bool, default=False)* – If `True`, also return the effective window and other baseline metadata.

**Returns**

- `baseline` *(numpy.ndarray)* – Empirical baseline on the input grid.
- `(baseline, info)` *(tuple, optional)* – Returned when `return_info=True`. `info["window_width"]` contains the width actually used at every point. `info["initial_window_width"]` contains the initial capped widths, and `info["max_window_width"]` is the physical cap.

`poisson_base_calculator` and `choose_baseline_window` accept the same keyword-only
`max_window_width=None` option. Their standalone defaults continue to derive the
cap from their input energy span.

The initial window is the smaller of `baseline_window` and the maximum allowed
width. At each finite point, the window expands only as far as needed to include
`min_points`. If any point cannot reach the minimum within the cap, the analysis
stops with `ValueError`, for example:

```text
Cannot estimate the baseline with this rebinning: min_points=3 cannot be reached at energy=1 within the maximum window width (0.2). Use finer rebinning or increase max_range_fraction.
```

Nonfinite input samples are excluded; occasional gaps are interpolated after
estimating valid points. An entirely invalid spectrum raises an error. A valid
zero-valued spectrum can still have a zero baseline.

Observed and simulated spectra use the same initial physical window widths and
the same cap, resolved from the padded nominal search interval. Each simulated grid
adapts its own windows to reach `min_points`; it does not inherit any enlargement
caused by the observed binning. A failure in a simulation also stops the run,
including parallel execution. A change in working context can change draws
relative to earlier versions even with the same seed; serial and parallel runs
still use identical draws for a given seed and configuration.

**Example**

```python
baseline, baseline_info = base_calculator(
    spectrum.energy,
    spectrum.values,
    baseline_window=0.4,
    return_info=True,
)

residuals = spectrum.values - baseline
line_excess = np.maximum(residuals, 0.0)  # Locate candidate blocks only.
```

Fits with Gaussian error weighting use the signed `residuals`, including bins below the baseline.
The positive `line_excess` only identifies the blocks to search; it is not the
fitting target. Poisson fits use the raw source/background counts against a
continuum-plus-lines count model instead of residuals. This distinction applies
to observed and simulated spectra.

---

## Blind candidate search

### `find_candidate_lines`

```python
find_candidate_lines(
    pha,
    rmf,
    arf=None,
    bkg=None,
    *,
    en1=None,
    en2=None,
    energy_pad=None,
    show_plot=False,
    output_dir=None,
    plot_name="bliss_fit.png",
    final_fit=False,
    rebin_method=None,
    rebin_scale=_UNSET,
    rebin_min_bins=None,
    num_synthetic_simulations=None,
    synthetic_seed=_UNSET,
    config=None,
    oversampling=_UNSET,
    assume_poisson_rate_errors=None,
    fit_statistic=None,
    baseline_context=_UNSET,
    overwrite=False,
)
```

Run BLiSS local candidate detection and simulation-based assessment. By default,
`final_fit=False` returns the local candidate catalogue. With `final_fit=True`,
it also fits all eligible candidates jointly and writes
`candidate_lines_global_fit.csv`. `show_plot=True` displays local candidate
markers or the global-fit model according to `final_fit`.

For a manually selected subset, use `find_candidate_lines(final_fit=False)`,
filter the candidates, and call `fit_global` separately.

`_UNSET` in the signatures denotes an internal marker for an omitted argument;
callers simply leave that keyword out. Omitted configuration options inherit
`config`, or use `BlindLineSearchConfig()` defaults when no config is provided.
Only explicit overrides are applied, and the supplied config is not modified.
For `en1`, `en2`, `energy_pad`, `rebin_method`, `rebin_min_bins`,
`num_synthetic_simulations`, `assume_poisson_rate_errors` and `fit_statistic`, `None` also means
inherit. For `rebin_scale`, `synthetic_seed` and `oversampling`, an explicit
`None` clears the configured value. For `baseline_context`, omission inherits
the config, while explicit `None` restores the automatic context.

**Compatibility:** without a config, the default interval is now 1.7–12.0 keV
with 0.1 keV padding, matching `BlindLineSearchConfig`. These values can be
changed for any instrument or scientific band. To inherit a configured seed,
oversampling or continuum context, omit the keyword; explicit `None` clears
that override. Searches now select the native working context before grouping,
so earlier full-input results and p-values must be recomputed.

This is the recommended first search step. BLiSS selects the native working
context, rebins the folded PHA spectrum, estimates the empirical baseline on
that context, searches the requested interval, fits the local Gaussian
candidates, and computes five global Monte Carlo p-values from the same null
searches.

With Poisson fitting, local and global fits compare the original source counts
with the fixed continuum plus Gaussian lines integrated over each bin. Empty
bins remain in the likelihood without artificial error weights. With a
background, BLiSS profiles the background expectation in the joint Poisson
likelihood (W-statistic construction); it does not apply Cash statistics to
background-subtracted counts. See the [XSPEC discussion of low-count
spectra](https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/node395.html)
and [likelihoods with background](https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/node393.html).

The Gaussian branch fits signed `values - baseline`, including negative bins,
weighted by the supplied uncertainties. Positive excess only locates candidate
blocks in either branch. Data and nulls use the same selected fitter and
continuum estimator in serial and parallel execution.

Poisson parameter errors use the local expected Fisher-information covariance
with the continuum fixed, including the profiled-background variance. They are
approximate curvature errors, not exact confidence intervals; weak lines and
parameter bounds can make them unreliable. Rank-deficient covariance remains
non-evaluable under the existing numerical checks. `snr_peak` uses the fitted
model's expected source-plus-scaled-background noise in this branch.

Every Gaussian amplitude is constrained during fitting to
`0 <= amplitude <= values[i_max] + uncertainties[i_max]`, where
`i_max = argmax(values)` in that fitting block (the first maximum if tied).
The observed bin values include the continuum, and the added error belongs
to that same bin. Every component in a local block shares its cap. Each null
spectrum calculates its own cap from its own rebinned values and errors,
identically in serial and parallel searches. Initial amplitudes are clipped
to the allowed interval before fitting.

Local and global fitting first leave every Gaussian width free within its bounds.
After that fit, each width below the instrumental `response_sigma`, interpolated
at its fitted center, is fixed to that instrumental value. All remaining free
parameters are then refitted jointly, including the amplitudes and centers of
fixed-width components and every parameter of the other components. The fixed
value remains constant during this reduced fit. If another free component
then falls below its instrumental width, the rule is applied to it as well;
already fixed widths are not reopened. This is an unresolved-line
approximation using the RMF core width; `sigma` describes the observed Gaussian
profile, not an intrinsic width convolved through the full response matrix.
No usable instrumental width means no instrumental fixing. A required fixed
width outside the configured bounds is an explicit fit failure, not a reason
to relax those bounds.

`width_fixed_instrumental=True` identifies components treated this way. Their
`sigma` is an assumption and `esigma=NaN`, rather than a measured zero error.
The remaining errors are conditional on that width. In particular,
`earea = sqrt(2*pi) * sigma * eamplitude` for a fixed width; free widths retain
the full amplitude-width covariance formula. `False` does not establish that
a line is resolved. Fixed-width rows remain evaluable if their other parameters
and errors pass the numerical checks. This rule also applies to observed and
synthetic local searches, including parallel workers. Rerun the search and null
simulations to calibrate p-values under the new fitting rule.

Candidates with `ecenter > 1` keV are automatically removed from the catalogue,
including `ecenter=+inf`. The comparison is strict: exactly 1 keV is retained
if the other checks pass. This cut also applies to components with fixed
instrumental widths. There is no corresponding upper cut on `esigma` or
`eamplitude`; large finite values of those errors alone do not remove a row.
The same centroid-error filter applies to local observed and null fits,
serial and parallel searches, global-fit inputs and outputs, plots, and
significance utilities supplied with external candidate tables.

Rows with `ecenter=NaN` remain diagnostic, non-evaluable entries rather than
being removed by this comparison. Other numerical checks still mark failed
fits, nonfinite parameters, nonpositive amplitude or width, invalid errors,
and errors suspiciously small relative to numerical precision as non-evaluable.
The intentional `esigma=NaN` of an instrumentally fixed width is exempt from
the width-error check, but its amplitude and centroid checks still apply.
Remaining non-evaluable rows have unavailable p-values and are excluded from
significance comparisons and from global refitting. The 1 keV cut is a
candidate-selection rule, not a confidence bound or a detection threshold.
Rerun the local search and null simulations before using p-values under this
policy; p-values computed without this centroid-error filter must not be reused.

With the default `final_fit=False`, filter the returned table and pass the selected rows to `fit_global` for a simultaneous multi-Gaussian fit. Alternatively, set `final_fit=True` to perform that fit directly on all eligible candidates.

**Parameters**

- `pha`, `rmf` *(path-like)* – Source PHA spectrum and RMF response; both are required.
- `arf`, `bkg` *(path-like or None)* – Optional ARF and background PHA spectrum.
- `en1`, `en2` *(float or None, effective defaults 1.7 and 12.0)* – Nominal energy interval in keV. Explicit values override the config; `None` inherits it. The returned catalogue is restricted to this interval.
- `energy_pad` *(float or None, effective default=0.1)* – Extra energy range fitted on both sides of the nominal interval. Padding reduces boundary effects, but the final catalogue is still restricted to `en1`–`en2`.
- `baseline_context` *(pair of floats or None)* – Optional larger native working interval in keV; it must contain the automatic context defined above. Omission inherits the config; explicit `None` restores automatic bounds. It does not change the nominal search band or the continuum-window cap.
- `output_dir` *(path-like or None)* – Results directory. If omitted, BLiSS creates a unique timestamped folder below `results/`, adding `_001`, `_002`, etc. when needed, including concurrent runs.
- `overwrite` *(bool, default=False)* – Explicitly allow replacement of the files written by this call. Otherwise an existing destination raises `FileExistsError` before the search starts.
- `final_fit` *(bool, default=False)* – Return local candidates by default; if `True`, jointly refit eligible candidates and return the global-fit catalogue.
- `show_plot` *(bool, default=False)* – Save and display local candidate markers when `final_fit=False`, or the global-fit model when `final_fit=True`. No p-value threshold is applied to local markers.
- `plot_name` *(str, default="bliss_fit.png")* – Name of that diagnostic image inside `output_dir`, written only when `show_plot=True`.
- `rebin_method`, `rebin_scale`, `rebin_min_bins` – Count-conserving rebinning passed to `prepare_spectrum` and reproduced in the null simulations.
- `num_synthetic_simulations` *(int or None)* – Number of null realizations. `None` retains the value in `config`, or the default value of 10 when no config is supplied. Zero disables simulation-based significance.
- `synthetic_seed` *(int or None)* – Random seed for reproducible null simulations. Omission inherits; explicit `None` clears the seed.
- `oversampling` *(float or None)* – Optional maximum sampling factor for optimal grouping. Omission inherits; explicit `None` removes the extra limit.
- `assume_poisson_rate_errors` *(bool or None)* – `None` inherits the config (default `False`). Explicit `True` allows estimation of missing rate errors assuming Poisson counts; explicit `False` restores strict rejection even when the config enables it. Applies to source and background.
- `fit_statistic` *(str or None)* – `None` inherits the config (`"auto"` by default). Choose `"poisson"` or `"gaussian"` explicitly to require that fitting method; see the input-compatibility rules under `prepare_spectrum`.
- `config` *(`BlindLineSearchConfig` or None)* – Optional advanced configuration. The explicit wrapper arguments above override their corresponding configuration fields.

**Returns and files**

- `pandas.DataFrame` – Compact local-candidate catalogue, or the global-fit catalogue when `final_fit=True`.
- `candidate_lines.csv` – Local catalogue written when `final_fit=False`.
- `candidate_lines_global_fit.csv` – Global-fit catalogue written when `final_fit=True`.
- `run_summary.txt` – A brief run summary, including the fit statistic actually selected.
- `bliss_fit.png` (or `plot_name`) – Local candidate diagnostic or global-fit figure, when `show_plot=True`.

With `final_fit=False`, the candidate figure uses the same spectrum, uncertainties and continuum as
the global-fit figure. Dashed red vertical markers identify local fits that
converged and pass the numerical validity checks, with centers in the nominal
search band. Candidates with `ecenter > 1` keV have already been removed.
Large finite `esigma` or `eamplitude` alone do not exclude a marker.
No significance threshold is imposed.
Markers are labelled `Candidates`; they also appear when no simulations were
requested and p-values are unavailable. Failed or
non-evaluable attempts remain in the table but are not drawn as measured lines.
With no valid candidates, the figure still shows the data and baseline.
Plotting does not perform a global fit or change the returned catalogue.

For Poisson inputs, plotted source-count bars are asymmetric central 68.27%
Garwood intervals, converted to density. With a measured background, the
source/background marginal errors are combined in quadrature for an
approximate net-spectrum display; these are not exact net-count intervals.
The bars are not fit weights. In particular, zero observed counts do not imply
zero uncertainty, and adopting Poisson fitting does not guarantee smaller bars.

The returned table and CSV contain exactly:

```text
center, ecenter, sigma, esigma, width_fixed_instrumental, amplitude, eamplitude,
area, earea, ew, relative_power, snr_peak, snr_area,
response_feature,
p_global_area, p_global_snr_area, p_global_snr_peak,
p_global_ew, p_global_relative_power
```

| Column | Meaning |
| --- | --- |
| `center`, `ecenter` | Local fitted centroid and its one-sigma error, in keV. |
| `sigma`, `esigma` | Observed Gaussian width and its one-sigma curvature error, in keV. `esigma` is `NaN` when width is fixed. |
| `width_fixed_instrumental` | Boolean: the fit fixed this width to the instrumental sigma and refitted the remaining parameters. `False` is not a claim that the line is resolved. |
| `amplitude`, `eamplitude` | Local Gaussian peak amplitude and error. |
| `area`, `earea` | Integrated Gaussian area and covariance-aware error, conditional on the instrumental width when fixed. |
| `ew` | Approximate equivalent width in eV. |
| `relative_power` | Local line-excess strength relative to the observed level and baseline. |
| `snr_peak` | Peak amplitude divided by the local block noise. |
| `snr_area` | Gaussian area divided by its covariance-aware error. |
| `response_feature` | `True` when the candidate lies close to unusually sharp ARF structure. Without a usable ARF, it is `False`. |
| `p_global_*` | Global Monte Carlo p-value for the named local-fit statistic. |

For each `p_global_*`, BLiSS compares the observed statistic with the maximum value recovered in every eligible null search over the nominal `en1`–`en2` interval. If `k` of `N` null maxima are at least as large as the observed value, the reported value is `(k + 1) / (N + 1)`. Consequently, the smallest resolvable p-value is `1 / (N + 1)`. The five columns test five different statistics; they do not apply an additional correction for choosing among those statistics afterwards.

The parameter `num_synthetic_simulations=N` controls the five p-values. With `N=0`, all `p_global_*` values are unavailable (`NaN`).

**Example**

```python
candidates = find_candidate_lines(
    pha=pha,
    rmf=rmf,
    arf=arf,
    bkg=bkg,
    en1=0.8,
    en2=8.0,
    energy_pad=0.1,
    rebin_method="bins",
    rebin_scale=2,
    num_synthetic_simulations=100,
    synthetic_seed=32002,
    output_dir="results/vela_x1_candidates",
)
```

A possible selection before the global fit is:

```python
p_columns = [
    "p_global_area",
    "p_global_snr_area",
    "p_global_snr_peak",
    "p_global_ew",
    "p_global_relative_power",
]

pvalue_mask = candidates[p_columns].le(0.01).any(axis=1)

selected = candidates[pvalue_mask].reset_index(drop=True)
```

These thresholds are analysis choices, not fixed BLiSS defaults. Candidates flagged by `response_feature` should be inspected rather than automatically interpreted as astrophysical lines.

---

## Search configuration and pipeline interface

### `BlindLineSearchConfig`

```python
BlindLineSearchConfig(
    en1=1.7,
    en2=12.0,
    energy_pad=0.1,
    baseline_context=None,
    num_synthetic_simulations=10,
    synthetic_seed=None,
    final_fit_maxfev=100000,
    response_feature_threshold=5.0,
    max_sigma_line=0.1,
    baseline_window=0.4,
    max_range_fraction=0.2,
    min_points=3,
    noise_model="poisson",
    rebin_method="optimal",
    rebin_scale=None,
    rebin_min_bins=1,
    n_jobs=1,
    simulation_batch_size=5,
    min_relative_prominence=0.05,
    oversampling=None,
    assume_poisson_rate_errors=False,
    keep_synthetic_candidates=False,
    fit_statistic="auto",
)
```

Use this dataclass when the search needs more control than the convenience wrapper exposes.

| Field | Purpose |
| --- | --- |
| `en1`, `en2`, `energy_pad` | Nominal search band and internal padding, in keV. Defaults: 1.7, 12.0 and 0.1; not instrument-dependent limits. |
| `baseline_context` | Default `None`: automatic padded-fit context plus half the maximum continuum-window width per side. An explicit pair may enlarge, but cannot shrink, this context. |
| `num_synthetic_simulations`, `synthetic_seed` | Number and random seed of null realizations. |
| `n_jobs`, `simulation_batch_size` | Maximum worker processes and simulations per task; both must be positive integers. Defaults: 1 and 5. |
| `keep_synthetic_candidates` | Default `False`: retain only the five null maxima per simulation. Set `True` to also keep the full null candidate catalogue for inspection; p-values are identical. |
| `noise_model` | `"poisson"` for science analyses; `"gaussian"` is retained for validation and comparison only. |
| `fit_statistic` | `"auto"` selects Poisson for compatible raw counts, otherwise Gaussian weighting. `"poisson"` requires compatible inputs; `"gaussian"` retains the previous fits. Gaussian validation noise selects Gaussian fits in auto mode; explicit Poisson fits require Poisson null draws. |
| `assume_poisson_rate_errors` | Default `False`: require explicit errors for `RATE`/`COUNT_RATE`. `True` estimates missing errors from rate and exposure assuming Poisson counts, for source and background. Supplied errors are preserved. |
| `rebin_method`, `rebin_scale`, `rebin_min_bins` | Count-space rebinning applied consistently to data and null spectra; default method: `optimal`. |
| `oversampling` | Optional positive maximum sampling factor for optimal grouping; default `None`. Uses the larger group size from the optimal and FWHM/factor rules. |
| `baseline_window`, `max_range_fraction`, `min_points` | Physical continuum-window settings: a count-density mean with iterative emission masking for Poisson fits, a running median for Gaussian fits. The cap is `max_range_fraction` times the padded nominal-band width, fixed for data and nulls. |
| `max_sigma_line` | Configured maximum Gaussian sigma in keV; capped by each local block energy span, also retained in global fits. |
| `min_relative_prominence` | Retain peaks strictly above this fraction of the largest prominence in their block. Default: 0.05; must be finite and in [0, 1). |
| `response_feature_threshold` | Robust ARF-sharpness threshold used for `response_feature`. |
| `final_fit_maxfev` | Maximum number of function evaluations for the optional global fit. |

`min_relative_prominence` selects the starting peaks for local Gaussian fits;
it is not a statistical significance threshold. Lower values admit weaker
peaks relative to their block maximum. Observed and null searches use the
same value, including parallel workers. Changing it requires rerunning the
null simulations before interpreting the resulting p-values.

For 100 simulations, `n_jobs=2, simulation_batch_size=5` uses at most two
persistent worker processes with five spectra per task. Each worker limits its
BLAS/OpenMP pools to one thread, including Apple Accelerate through its startup
environment setting. Rebinning, baseline estimation and local line
searches run in the workers; the parent draws the random samples in their
original order. The same seed therefore produces the same random draws when
the process count or batch size changes. Numerical fits can still have small
floating-point differences between platforms or numerical-library versions.

Only a bounded queue of batches is kept in flight. Each completed candidate
table is reduced immediately to its five eligible maxima, then released by
default. Every simulation contributes one row, including an all-NaN row when
it finds no eligible candidates. This keeps all simulations in the p-value
denominator and preserves the same fit-quality, nominal-band and tie rules.
Memory grows with five maxima per simulation plus the bounded in-flight
work, instead of the entire null catalogue. Worker or reduction errors stop
the run; failed simulations are never silently excluded. The default
`n_jobs=1` also processes spectra one at a time. A single batch is handled
locally to avoid starting workers for too little work.

In a notebook, restart the kernel after updating BLiSS and rerun the imports.
In a standalone Python script, put the pipeline call inside
`if __name__ == "__main__":` because worker processes use `spawn`.
Process startup has a cost, so short runs may be faster with `n_jobs=1`.

### `BlindLineSearchPipeline.run`

```python
pipeline.run(
    pha,
    rmf,
    arf=None,
    bkg=None,
    *,
    en1=None,
    en2=None,
    energy_pad=None,
    final_fit=False,
    show_plot=False,
    output_dir=None,
    plot_name="bliss_fit.png",
    overwrite=False,
)
```

The explicit pipeline interface is useful when the complete configuration or the null-search diagnostics are needed. After a run, the instance exposes:

- `pipeline.synthetic_candidates` – `None` by default; the concatenated null candidate catalogue when `config.keep_synthetic_candidates=True`;
- `pipeline.null_maxima` – One row per null realization and one column per global statistic;
- `pipeline.n_sim` – Number of generated null realizations.

For this interface, set `baseline_context` through `BlindLineSearchConfig`.
The optional `en1`, `en2` and `energy_pad` arguments to `run` override the
corresponding config fields before the context and its window cap are resolved.

`show_plot=True` saves and displays a diagnostic figure in either mode. With
`final_fit=False`, it shows data, baseline and red vertical candidate markers.
With `final_fit=True`, it shows the global Gaussian model and its sum with the
baseline. Both modes use `plot_name`; `show_plot=False` creates no pipeline
figure. All candidate-table and p-value calculations are independent of this
display option.

**Compatibility:** code that inspects `pipeline.synthetic_candidates` must now
opt in with `BlindLineSearchConfig(keep_synthetic_candidates=True)`. This
retains all rows that survive the `ecenter > 1` keV filter, including remaining
diagnostic failures, and uses more memory. The
default and opt-in modes calculate p-values from the same maxima without
rerunning any simulations. Both convenience search functions inherit this
setting through their `config` argument. Reusing a pipeline resets these
diagnostics; an interrupted null search does not publish partial p-values.

Automatic folders are always new. An explicit `output_dir` may already exist,
but only the files this call will write are checked for collisions. By default,
an existing destination raises `FileExistsError` with a message explaining how
to choose another directory/name or explicitly use `overwrite=True`. Files are
also reserved exclusively before writing, so concurrent calls cannot silently
replace one another's outputs. Other files in the directory are preserved.

The same policy applies to `find_candidate_lines`,
`fit_global` and saved figures. For example:

```python
# Explicitly replace this search's previous outputs in the chosen directory.
candidates = find_candidate_lines(
    pha, rmf, output_dir="results/my_run", overwrite=True,
)
```

`fit_global` can add its differently named table and figure to a candidate
folder without `overwrite=True`. With `save_csv=False`, its CSV is not checked
or replaced. The pipeline checks its figure whenever `show_plot=True`, in
both local and global modes; `fit_global` saves its figure whenever `output_dir` is
provided, even with `show_plot=False`.

**Example**

```python
config = BlindLineSearchConfig(
    en1=0.8,
    en2=8.0,
    energy_pad=0.1,
    baseline_window=0.4,
    rebin_method="bins",
    rebin_scale=2,
    num_synthetic_simulations=100,
    synthetic_seed=32002,
    noise_model="poisson",
    n_jobs=2,
    simulation_batch_size=5,
)

pipeline = BlindLineSearchPipeline(config)
candidates = pipeline.run(
    pha=pha,
    rmf=rmf,
    arf=arf,
    bkg=bkg,
    output_dir="results/vela_x1_candidates",
)
```

---

## Global multi-Gaussian fit

### `fit_global`

```python
fit_global(
    pd_lines,
    spectrum,
    *,
    base=None,
    ylines=None,
    show_plot=True,
    output_dir=None,
    plot_name="bliss_global_fit.png",
    save_csv=True,
    final_fit_maxfev=100000,
    baseline_window=0.4,
    max_range_fraction=0.2,
    min_points=3,
    return_yfit=False,
    energy_min=None,
    energy_max=None,
    size_fig_input=None,
    response_feature_threshold=5.0,
    max_sigma_line=0.1,
    overwrite=False,
)
```

Run the simultaneous multi-Gaussian fit on a user-selected candidate table.

This function should normally be called after `find_candidate_lines`. Unlike older BLiSS versions, its second argument must be a `PreparedSpectrum`, not separate energy, value, and uncertainty arrays.

**Parameters**

- `pd_lines` *(pandas.DataFrame)* – Selected candidate table. It must contain at least `amplitude`, `center`, and `sigma`. Compact tables returned by `find_candidate_lines` are accepted directly.
- `spectrum` *(`PreparedSpectrum`)* – Spectrum returned by `prepare_spectrum`, using the same files, native working domain, rebinning and `fit_statistic` as the candidate search. Standalone preparation defaults to the full input, so pass the matching `energy_range` when reconstructing a search spectrum.
- `base`, `ylines` *(numpy.ndarray or None)* – Optional precomputed baseline and signed residuals (`spectrum.values - base`). Omit them to use the statistic-appropriate continuum automatically. Gaussian fits use supplied residuals as-is. Poisson fits use raw counts; an explicitly supplied `ylines` must equal `values - base` or is rejected. Never pass the positive detection mask as residuals.
- `show_plot` *(bool, default=True)* – Display the global-fit diagnostic plot.
- `output_dir` *(path-like or None)* – If supplied, save the table and plot in this directory.
- `plot_name` *(str, default="bliss_global_fit.png")* – Plot filename inside `output_dir`.
- `save_csv` *(bool, default=True)* – Save `global_fit_lines.csv` when `output_dir` is supplied.
- `overwrite` *(bool, default=False)* – Allow replacement of this call's table and figure. Otherwise collisions are rejected before fitting or writing; existing candidate files with other names are allowed.
- `final_fit_maxfev` *(int, default=100000)* – Maximum number of optimizer function evaluations.
- `baseline_window`, `max_range_fraction`, `min_points` – Baseline settings used only when `base` and `ylines` are not supplied. They should match the candidate-search settings. A stored `spectrum.baseline_max_window_width` preserves the pipeline's fixed cap; otherwise standalone fitting derives its cap from the supplied energy span.
- `return_yfit` *(bool, default=False)* – Return the line-only model together with the table.
- `energy_min`, `energy_max` *(float or None)* – X-axis limits for the plot only; they do not restrict the fitted data.
- `size_fig_input` *(tuple or None)* – Optional figure size, for example `(10, 5)`.
- `response_feature_threshold` *(float, default=5.0)* – ARF-sharpness threshold applied again at the globally fitted centroids.
- `max_sigma_line` *(float, default=0.1)* – Configured upper bound on Gaussian sigma, in keV, combined with each internally retained `fit_block_width` using the smaller value. Use the same value as in the local search.

**Returns**

- `pandas.DataFrame` – Global-fit catalogue with the same compact columns as the local candidate table.
- `(pandas.DataFrame, numpy.ndarray)` – Returned when `return_yfit=True`; the second item is the line-only model on `spectrum.energy`.

The global fit recomputes the centroid, width, amplitude, area, equivalent width, and their relevant uncertainties. It starts with widths free, even for candidates whose local `width_fixed_instrumental` flag was true, and applies the instrumental-width rule using its own fitted centers and widths. Compact fixed-width candidates with `esigma=NaN` are valid inputs when their remaining measurements pass the checks. The area uncertainty includes amplitude-width covariance for free widths and is conditional on sigma for fixed widths. All five input `p_global_*` values are preserved because they describe the earlier local detection search; they are not recalculated from the global-fit measurements.

The global fit removes input candidates with `ecenter > 1` keV before fitting
and removes components whose newly fitted `ecenter` exceeds 1 keV afterward.
This includes `+inf` and applies equally to fixed-width components. Exactly
1 keV is not removed by this cut. Removed components are also omitted from
the returned line model and its plot. Large finite width or amplitude errors
alone do not remove components. Input p-values on retained rows remain those
of the local search: applying a global fit to an older catalogue does not
recalibrate p-values from a search performed without the centroid-error filter.
Repeat the local search and its simulations under the same selection rule.

Poisson global fits use the same integrated source/background count likelihood
as local fits. The returned line-only model is the bin-averaged Gaussian density.
Gaussian global fits retain signed residuals, including bins below the
continuum. Changing the likelihood or continuum estimator requires repeating
the local search and null simulations to obtain corresponding p-values;
refitting only the final catalogue does not recalibrate them.

The amplitude rule is the same in the global fit, with the entire supplied
fit interval as its block: all components share the maximum observed bin
plus that bin's propagated count-based error (including any scaled-background
contribution). Poisson display intervals do not alter this cap. In the Gaussian
branch, the existing fallback for invalid errors is applied before the cap.

Direct calls to the parameter generators `p0_generator` and
`p0_generator_final` accept `uncertainties=...`. All BLiSS fits supply this
argument. For compatibility, a direct generator call without uncertainties
uses `max(y)` as the cap, with no added error.

**Example**

```python
# Reproduce the automatic context and fixed cap of the search example above.
fit_lo, fit_hi = 0.8 - 0.1, 8.0 + 0.1
max_window_width = 0.2 * (fit_hi - fit_lo)
context = (fit_lo - max_window_width / 2, fit_hi + max_window_width / 2)

spectrum = prepare_spectrum(
    pha_path=pha,
    rmf_path=rmf,
    arf_path=arf,
    background_path=bkg,
    rebin_method="bins",
    rebin_scale=2,
    energy_range=context,
)
spectrum.baseline_max_window_width = max_window_width

global_lines, line_model = fit_global(
    selected,
    spectrum,
    baseline_window=0.4,  # Match the search; select the estimator automatically.
    show_plot=True,
    output_dir="results/vela_x1_global_fit",
    energy_min=0.8,
    energy_max=8.0,
    size_fig_input=(10, 5),
    return_yfit=True,
)
```


---

## Plotting

### `plot_global_fit`

```python
plot_global_fit(
    spectrum,
    base,
    yfit,
    output_path=None,
    *,
    show_plot=True,
    energy_min=None,
    energy_max=None,
    size_fig_input=None,
    overwrite=False,
)
```

Plot the prepared spectrum, empirical baseline, line-only model, and total `baseline + line model`. This function is called internally by `fit_global`, but can also be used directly when `return_yfit=True` was requested.

It returns `None`. Supply `output_path` to save the figure.

Existing images raise `FileExistsError` unless `overwrite=True`. A filename
without an extension uses Matplotlib's configured default format (normally
`.png`); collision checks include that extension. The legacy
`plot_final_bliss_fit(..., overwrite=False)` uses the same policy. A plot name
that coincides with a table or summary produced by the same call is rejected,
even with `overwrite=True`.

---

## Atomic-line identification

BLiSS does not ship or select an atomic-transition database. Obtain the
catalogue you want to use independently, read it into a pandas DataFrame,
and pass it explicitly to the identification API. Importing BLiSS, searching
for lines, and fitting candidates do not load an atomic database. Omitting
the database during identification raises an error.

For a file containing `ion` labels and `energy_keV` values:

```python
import pandas as pd
from bliss import LineIdentifier

catalog = pd.read_csv("my_transitions.csv")
identifier = LineIdentifier(
    atomic_table=catalog,
    ion_col="ion",
    energy_col="energy_keV",
    energy_unit="keV",
    v_doppler_kms=1000,
)
```

Use the appropriate reader for your file; for example,
`pd.read_csv("my_transitions.tsv", sep="\t")` for tab-separated data.
Configure the label and coordinate columns, their units, and any optional
ranking weight as described below. BLiSS adds neither automatic scores nor
catalogue provenance metadata, and does not use such metadata to choose a ranking.

**Migration:** old imports of `st_reduced` or `atomic_line_table` are no
longer supported. Replace them with your own DataFrame and pass it as
`atomic_table=catalog` or `pd_data=catalog`. There is no bundled replacement
catalogue or `load_atomic_database` API.

### `add_most_probable_ion`

```python
add_most_probable_ion(pd_fit, v_doppler_kms, pd_data=None)
```

For each fitted centroid, find compatible transitions within the requested Doppler-velocity window and prepend the highest-ranked `ion` and its inferred `doppler_kms`. All original catalogue columns are preserved.

```python
identified = add_most_probable_ion(
    global_lines,
    v_doppler_kms=800,
    pd_data=catalog,
)
```

### `identify_line`

```python
identify_line(
    center_energy_keV,
    center_sigma_keV=None,
    v_doppler_kms=None,
    pd_data=None,
)
```

Return the atomic transitions compatible with one measured centroid, retaining their database metadata. Transitions are ordered by the explicitly selected weight column, or by energy proximity when no weight is selected. In the current implementation, the matching interval is defined by `v_doppler_kms`; `center_sigma_keV` is retained in the interface but does not widen that interval.

The matching tolerance uses the magnitude `abs(v_doppler_kms)`. Thus `1000`
and `-1000` km/s select the same transitions, with energy half-width
`center_energy_keV * abs(v_doppler_kms) / 299792.458`. Both interval endpoints
are included. Zero or `None` requires an exact energy match. This applies to
`identify_line`, `add_most_probable_ion`, `get_all_compatible_lines` and both
`LineIdentifier` methods for every user-supplied database.

The inferred output `doppler_kms` remains signed:
`299792.458 * (measured_energy - reference_energy) / reference_energy`.
Only the matching tolerance takes an absolute value. Previously, a negative
tolerance inverted the interval bounds and returned no matches.

### `get_all_compatible_lines`

```python
get_all_compatible_lines(pd_fit, v_doppler_kms, pd_data=None)
```

Return a dictionary that maps each input row index to a DataFrame containing all compatible atomic transitions.

The fitted-line table must have unique row indices. Duplicate indices raise
`ValueError` before matching, so no line's identifications silently replace
another's. Unique indices are preserved, including nonconsecutive numbers or
text labels. The same check applies to `LineIdentifier.all_compatible`.
Duplicate row indices in the atomic database are allowed.

When combining fitted-line tables, create unique indices explicitly:

```python
combined_lines = pd.concat([lines_a, lines_b], ignore_index=True)
all_matches = get_all_compatible_lines(
    combined_lines, v_doppler_kms=800, pd_data=catalog,
)
```

For an already combined table, use
`combined_lines = combined_lines.reset_index(drop=True)` before identification.
BLiSS already uses `ignore_index=True` when concatenating fitted or simulated
catalogue rows internally.

The object-oriented equivalent is:

```python
identifier = LineIdentifier(atomic_table=catalog, v_doppler_kms=800)
identified = identifier.add_most_probable(global_lines)
all_matches = identifier.all_compatible(global_lines)
```


### Supplying a transition database

All three functions require an explicit `pd_data` DataFrame. `LineIdentifier`
requires it as `atomic_table`. Their `None` default signals a missing database;
it does not trigger loading. Each row represents a transition, so ion labels
may repeat.
Select the ion/transition label column and at least one coordinate column:

```python
my_database = pd.read_csv("my_transitions.tsv", sep="\t")
identifier = LineIdentifier(
    atomic_table=my_database,
    ion_col="Ion_name",
    energy_col="Energy",
    energy_unit="eV",
    weight_col="Emissivity",  # Optional: larger values rank first.
    v_doppler_kms=800,
)
identified = identifier.add_most_probable(global_lines)
all_matches = identifier.all_compatible(global_lines)
```

For a wavelength-only database, replace `energy_col` and `energy_unit` with:

```python
wavelength_col="Wavelength",
wavelength_unit="Angstrom",
```

The same options can be passed directly to the functional interface:

```python
matches = identify_line(
    center_energy_keV=6.7,
    v_doppler_kms=800,
    pd_data=my_database,
    ion_col="Ion_name",
    energy_col="Energy",
    energy_unit="eV",
    weight_col="Emissivity",
)
```

Energy and wavelength units are required for custom column names and must have
the corresponding physical dimension. The legacy `energy_keV` column defaults
to keV. Internally, comparisons use keV. If both coordinate columns are selected,
they must agree within `consistency_rtol=1e-3` (0.1%, configurable); the supplied
energy is then used for matching. Missing labels, nonpositive or nonfinite
coordinates, and inconsistent coordinates raise an error.

An optional weight column must contain finite, nonnegative numeric values.
Weights order only the transitions already inside the matching window, with
energy proximity breaking ties. Without an explicit `weight_col`, every table
is ordered by energy proximity. Column names such as `scaled_prob` and any
DataFrame provenance metadata do not activate a ranking automatically.
BLiSS does not calculate identification scores from the table. Weights are
relative priorities, not calibrated probabilities; the function names containing
`most_probable` remain for compatibility.

`identify_line` and `all_compatible` retain every original database column and
provide canonical `ion`, `energy_keV`, and inferred `doppler_kms` columns.
Conflicting original column names are preserved with a `database_` prefix,
repeated if needed to avoid another collision. `add_most_probable` keeps the
existing compact fitted catalogue, adding the selected ion and velocity.
Neither input DataFrame is modified. Doppler velocity remains a separate
matching-window option and is not required in the database.

---

## Advanced simulation and significance utilities

The high-level search functions call these utilities automatically. They are exposed mainly for validation and method-development work.

| API | Purpose |
| --- | --- |
| `generate_null_realizations(spectrum_full, base_full, fit_en1, fit_en2, config, *, noise_model=None, window_width=None)` | Generate line-free null spectra while reproducing the observed processing chain. Returns a list of `NullRealization` objects. |
| `final_fit_and_metrics(spectrum, clean_lines, ...)` | Lower-level global-fit function that always returns `(result, yfit)` and omits plotting and file output. |
| `create_bliss_results_folder(base_dir="results", suffix="bliss")` | Create an exclusive timestamped results directory; collisions add `_001`, `_002`, etc. |
| `ensure_output_folder(output_dir=None)` | Create or validate an explicit directory, or create a unique timestamped one when omitted. Writers check actual file destinations separately. |

### Block-dependent sigma ceiling

Local fits use `min(max_sigma_line, max(block.energy) - min(block.energy))`
as the upper Gaussian sigma bound, including the fitted block context bins.
The configured default remains 0.1 keV. This applies identically to Gaussian
and Poisson fits, observed data and serial/parallel null searches.
`fit_block_width` stores this span only in internal fitting tables; public
candidate/final tables and CSV exports omit it. A global fit in the same pipeline
retains each candidate's original block ceiling and combines it with the requested
maximum. A later standalone fit of an exported catalogue has no original block
width and uses the configured maximum, unless that context is supplied explicitly. Initial widths are clipped
to the effective interval; an upper bound at or below the effective minimum
raises an explicit error rather than relaxing either bound.

A conditional instrumental-width refit must also respect this ceiling. If the
instrumental value required for a narrow component exceeds its permitted sigma
range, the fit is marked failed explicitly; the width is not silently clipped.
