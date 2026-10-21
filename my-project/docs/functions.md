# Functions

This page documents the main public functions used in the current BLiSS workflow.

BLiSS works directly with folded PHA spectra and their RMF response files. Rebinning is performed in count space, the working spectrum is converted to counts s⁻¹ keV⁻¹, and the same data-processing chain is applied to the synthetic null spectra.

A standard workflow is:

1. Prepare and inspect the spectrum with `prepare_spectrum`.
2. Optionally inspect the empirical baseline with `base_calculator`.
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
    find_emission_lines,
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
    rebin_method="none",
    rebin_scale=None,
    rebin_min_bins=1,
)
```

Load a PHA spectrum, optionally subtract its background, rebin it while conserving counts, and convert it to a density spectrum in counts s⁻¹ keV⁻¹.

The returned `PreparedSpectrum` also retains the native count-space information required to generate Poisson null realizations with the same rebinning as the observed spectrum.

**Parameters**

- `pha_path` *(path-like)* – Source PHA spectrum.
- `rmf_path` *(path-like)* – Redistribution matrix. It supplies the channel energy bounds and the instrumental line-spread width.
- `arf_path` *(path-like or None, default=None)* – Optional ancillary response. When provided, BLiSS can flag candidates close to sharp effective-area structure.
- `background_path` *(path-like or None, default=None)* – Optional background PHA spectrum. It is scaled and subtracted before rebinning.
- `rebin_method` *{"none", "bins", "snr", "resolution"}, default="none"* – Count-space grouping rule.
- `rebin_scale` *(int, float, or None)* – Number of native bins, target S/N, or fixed energy width, depending on `rebin_method`.
- `rebin_min_bins` *(int, default=1)* – Minimum number of native bins per output bin for S/N rebinning.

The rebinning methods are:

| Method | Meaning of `rebin_scale` | Behaviour |
| --- | --- | --- |
| `"none"` | `None` | Keep the native channel grid. |
| `"bins"` | Number of channels | Sum consecutive native bins. |
| `"snr"` | Target S/N | Accumulate bins until the target S/N is reached. |
| `"resolution"` | Width in keV | Group channels into fixed-width energy intervals without splitting input bins. |

**Returns**

- `PreparedSpectrum` – A container with the following main attributes:

| Attribute | Contents |
| --- | --- |
| `energy` | Rebinned energy-bin centres in keV. |
| `values` | Rebinned spectrum in counts s⁻¹ keV⁻¹,. |
| `uncertainties` | One-sigma uncertainties in the same units as `values`. |
| `bin_width` | Rebinned energy-bin widths in keV. |
| `response_sigma` | RMF-derived Gaussian-equivalent instrumental sigma, when available. |
| `arf_energy`, `effective_area` | ARF grid and effective area, when supplied. |
| `native` | Native counts, exposure, background information, and grouping used by the null simulations. |

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
)
```

Load a folded PHA spectrum without the count-to-density conversion performed by `prepare_spectrum`.

By default, this returns a sorted `Spectrum` object. With `as_arrays=True`, it returns `(energy, values, uncertainties, bin_width)`. For a normal BLiSS search, prefer `prepare_spectrum`, because it also stores the native information required by the Poisson simulations.

### `rebin_counts`

```python
rebin_counts(
    energy,
    counts,
    uncertainties,
    bin_width,
    method="none",
    scale=None,
    min_bins=1,
    remainder="merge",
)
```

Low-level count-conserving rebinning. Counts are summed, uncertainties are added in quadrature, and bin widths are added.

It returns:

```text
energy_new, counts_new, uncertainties_new, bin_width_new, group
```

`group` gives the output-bin index assigned to each native bin; `-1` marks a dropped bin. `apply_groups(energy, counts, uncertainties, bin_width, group)` applies an existing grouping to another count spectrum, which is useful when the same fixed grouping must be used for the data and its null realizations.

---

## Empirical baseline

### `base_calculator`

```python
base_calculator(
    x,
    y,
    baseline_window=0.4,
    max_range_fraction=0.2,
    min_points=3,
    return_info=False,
)
```

Estimate the BLiSS empirical baseline with a running median defined in physical spectral units.

The baseline is an empirical description of the observed spectral shape, not a physical continuum model. Its window should be broad enough not to absorb narrow emission lines, but narrow enough to follow genuine changes in the underlying spectrum. 

**Parameters**

- `x` *(array-like)* – Spectral coordinate, normally energy in keV.
- `y` *(array-like)* – Spectral values on the same grid.
- `baseline_window` *(float, array-like, or callable, default=0.4)* – Preferred running-median width in the same units as `x`. A scalar uses one width everywhere; an array supplies one width per point; a callable `f(x)` allows the width to vary with energy.
- `max_range_fraction` *(float, default=0.2)* – Maximum allowed window as a fraction of the full spectral range.
- `min_points` *(int, default=3)* – Minimum number of points required in a local median window.
- `return_info` *(bool, default=False)* – If `True`, also return the effective window and other baseline metadata.

**Returns**

- `baseline` *(numpy.ndarray)* – Empirical baseline on the input grid.
- `(baseline, info)` *(tuple, optional)* – Returned when `return_info=True`. `info["window_width"]` contains the effective width at every point.

**Example**

```python
baseline, baseline_info = base_calculator(
    spectrum.energy,
    spectrum.values,
    baseline_window=0.4,
    return_info=True,
)

line_excess = np.maximum(spectrum.values - baseline, 0.0)
```

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
    en1=0,
    en2=10,
    energy_pad=0.0,
    output_dir=None,
    rebin_method="none",
    rebin_scale=None,
    rebin_min_bins=1,
    num_synthetic_simulations=None,
    synthetic_seed=None,
    config=None,
)
```

Run BLiSS up to local candidate detection and simulation-based assessment.

This is the recommended first search step. BLiSS loads and rebins the folded PHA spectrum, estimates the empirical baseline on the full input band, searches the requested interval, fits the local Gaussian candidates, and computes five global Monte Carlo p-values from the same null searches.

The function does **not** perform the final simultaneous multi-Gaussian fit. Filter the returned table first and pass the selected rows to `fit_global`.

**Parameters**

- `pha`, `rmf` *(path-like)* – Source PHA spectrum and RMF response; both are required.
- `arf`, `bkg` *(path-like or None)* – Optional ARF and background PHA spectrum.
- `en1`, `en2` *(float, defaults 0 and 10)* – Nominal energy interval in keV. The returned catalogue is restricted to this interval.
- `energy_pad` *(float, default=0.0)* – Extra energy range fitted on both sides of the nominal interval. Padding reduces boundary effects, but the final catalogue is still restricted to `en1`–`en2`.
- `output_dir` *(path-like or None)* – Results directory. If omitted, BLiSS creates a timestamped folder below `results/`.
- `rebin_method`, `rebin_scale`, `rebin_min_bins` – Count-conserving rebinning passed to `prepare_spectrum` and reproduced in the null simulations.
- `num_synthetic_simulations` *(int or None)* – Number of null realizations. `None` retains the value in `config`, or the default value of 10 when no config is supplied. Zero disables simulation-based significance.
- `synthetic_seed` *(int or None)* – Random seed for reproducible null simulations.
- `config` *(`BlindLineSearchConfig` or None)* – Optional advanced configuration. The explicit wrapper arguments above override their corresponding configuration fields.

**Returns and files**

- `pandas.DataFrame` – Compact local-candidate catalogue.
- `candidate_lines.csv` – The same catalogue written to the results directory.
- `run_summary.txt` – A brief run summary.

The returned table and CSV contain exactly:

```text
center, ecenter, sigma, esigma, amplitude, eamplitude,
area, earea, ew, relative_power, snr_peak, snr_area,
response_feature,
p_global_area, p_global_snr_area, p_global_snr_peak,
p_global_ew, p_global_relative_power
```

| Column | Meaning |
| --- | --- |
| `center`, `ecenter` | Local fitted centroid and its one-sigma error, in keV. |
| `sigma`, `esigma` | Local Gaussian width and its one-sigma error, in keV. |
| `amplitude`, `eamplitude` | Local Gaussian peak amplitude and error. |
| `area`, `earea` | Integrated Gaussian area and covariance-aware error. |
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
    en1=0.2,
    en2=10.0,
    energy_pad=0.1,
    num_synthetic_simulations=10,
    synthetic_seed=None,
    final_fit_maxfev=100000,
    response_feature_threshold=5.0,
    max_sigma_line=0.1,
    baseline_window=0.4,
    max_range_fraction=0.2,
    min_points=3,
    noise_model="poisson",
    rebin_method="none",
    rebin_scale=None,
    rebin_min_bins=1,
)
```

Use this dataclass when the search needs more control than the convenience wrapper exposes.

| Field | Purpose |
| --- | --- |
| `en1`, `en2`, `energy_pad` | Nominal search band and internal padding, in keV. |
| `num_synthetic_simulations`, `synthetic_seed` | Number and random seed of null realizations. |
| `noise_model` | `"poisson"` for science analyses; `"gaussian"` is retained for validation and comparison only. |
| `rebin_method`, `rebin_scale`, `rebin_min_bins` | Count-space rebinning applied consistently to data and null spectra. |
| `baseline_window`, `max_range_fraction`, `min_points` | Empirical running-median baseline settings. |
| `max_sigma_line` | Maximum allowed Gaussian sigma in keV for local and global fits. |
| `response_feature_threshold` | Robust ARF-sharpness threshold used for `response_feature`. |
| `final_fit_maxfev` | Maximum number of function evaluations for the optional global fit. |

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
)
```

The explicit pipeline interface is useful when the complete configuration or the null-search diagnostics are needed. After a run, the instance exposes:

- `pipeline.synthetic_candidates` – Concatenated candidates from all null realizations;
- `pipeline.null_maxima` – One row per null realization and one column per global statistic;
- `pipeline.n_sim` – Number of generated null realizations.

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
)
```

Run the simultaneous multi-Gaussian fit on a user-selected candidate table.

This function should normally be called after `find_candidate_lines`. Unlike older BLiSS versions, its second argument must be a `PreparedSpectrum`, not separate energy, value, and uncertainty arrays.

**Parameters**

- `pd_lines` *(pandas.DataFrame)* – Selected candidate table. It must contain at least `amplitude`, `center`, and `sigma`. Compact tables returned by `find_candidate_lines` are accepted directly.
- `spectrum` *(`PreparedSpectrum`)* – Spectrum returned by `prepare_spectrum`, using the same files and rebinning as the candidate search.
- `base`, `ylines` *(numpy.ndarray or None)* – Optional precomputed baseline and positive line-excess arrays on the `spectrum` grid.
- `show_plot` *(bool, default=True)* – Display the global-fit diagnostic plot.
- `output_dir` *(path-like or None)* – If supplied, save the table and plot in this directory.
- `plot_name` *(str, default="bliss_global_fit.png")* – Plot filename inside `output_dir`.
- `save_csv` *(bool, default=True)* – Save `global_fit_lines.csv` when `output_dir` is supplied.
- `final_fit_maxfev` *(int, default=100000)* – Maximum number of `curve_fit` evaluations.
- `baseline_window`, `max_range_fraction`, `min_points` – Baseline settings used only when `base` and `ylines` are not supplied. They should match the candidate-search settings.
- `return_yfit` *(bool, default=False)* – Return the line-only model together with the table.
- `energy_min`, `energy_max` *(float or None)* – X-axis limits for the plot only; they do not restrict the fitted data.
- `size_fig_input` *(tuple or None)* – Optional figure size, for example `(10, 5)`.
- `response_feature_threshold` *(float, default=5.0)* – ARF-sharpness threshold applied again at the globally fitted centroids.
- `max_sigma_line` *(float, default=0.1)* – Upper bound on every fitted Gaussian sigma, in keV. Use the same value as in the local search.

**Returns**

- `pandas.DataFrame` – Global-fit catalogue with the same compact columns as the local candidate table.
- `(pandas.DataFrame, numpy.ndarray)` – Returned when `return_yfit=True`; the second item is the line-only model on `spectrum.energy`.

The global fit recomputes the centroid, width, amplitude, area, equivalent width, and their relevant uncertainties. The area uncertainty includes amplitude-width covariance from the joint fit. All five input `p_global_*` values are preserved because they describe the earlier local detection search; they are not recalculated from the global-fit measurements.

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

baseline = base_calculator(
    spectrum.energy,
    spectrum.values,
    baseline_window=0.4,
)

global_lines, line_model = fit_global(
    selected,
    spectrum,
    base=baseline,
    show_plot=True,
    output_dir="results/vela_x1_global_fit",
    energy_min=0.8,
    energy_max=8.0,
    size_fig_input=(10, 5),
    return_yfit=True,
)
```

### `find_emission_lines`

```python
find_emission_lines(
    pha,
    rmf,
    arf=None,
    bkg=None,
    *,
    en1=0,
    en2=10,
    energy_pad=0.0,
    show_plot=False,
    output_dir=None,
    plot_name="bliss_fit.png",
    final_fit=False,
    rebin_method="none",
    rebin_scale=None,
    rebin_min_bins=1,
    num_synthetic_simulations=None,
    synthetic_seed=None,
    config=None,
)
```

Convenience wrapper around the same pipeline. With the default `final_fit=False`, it behaves like `find_candidate_lines`. With `final_fit=True`, it immediately fits all eligible local candidates simultaneously and writes `candidate_lines_global_fit.csv`.

The staged `find_candidate_lines` → user filtering → `fit_global` workflow is preferable when the global fit should use only a scientifically selected subset.

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
)
```

Plot the prepared spectrum, empirical baseline, line-only model, and total `baseline + line model`. This function is called internally by `fit_global`, but can also be used directly when `return_yfit=True` was requested.

It returns `None`. Supply `output_path` to save the figure.

---

## Atomic-line identification

### `add_most_probable_ion`

```python
add_most_probable_ion(pd_fit, v_doppler_kms, pd_data=st_reduced)
```

For each fitted centroid, find compatible transitions within the requested Doppler-velocity window and prepend the highest-ranked `ion` and its inferred `doppler_kms`. All original catalogue columns are preserved.

```python
identified = add_most_probable_ion(
    global_lines,
    v_doppler_kms=800,
)
```

### `identify_line`

```python
identify_line(
    center_energy_keV,
    center_sigma_keV=None,
    v_doppler_kms=None,
    pd_data=st_reduced,
)
```

Return the atomic transitions compatible with one measured centroid, sorted by the bundled ranking quantity `scaled_prob`. In the current implementation, the matching interval is defined by `v_doppler_kms`; `center_sigma_keV` is retained in the interface but does not widen that interval.

### `get_all_compatible_lines`

```python
get_all_compatible_lines(pd_fit, v_doppler_kms, pd_data=st_reduced)
```

Return a dictionary that maps each input row index to a DataFrame containing all compatible atomic transitions.

The object-oriented equivalent is:

```python
identifier = LineIdentifier(v_doppler_kms=800)
identified = identifier.add_most_probable(global_lines)
all_matches = identifier.all_compatible(global_lines)
```

---

## Advanced simulation and significance utilities

The high-level search functions call these utilities automatically. They are exposed mainly for validation and method-development work.

| API | Purpose |
| --- | --- |
| `generate_null_realizations(spectrum_full, base_full, fit_en1, fit_en2, config, *, noise_model=None, window_width=None)` | Generate line-free null spectra while reproducing the observed processing chain. Returns a list of `NullRealization` objects. |
| `final_fit_and_metrics(spectrum, clean_lines, ...)` | Lower-level global-fit function that always returns `(result, yfit)` and omits plotting and file output. |
| `create_bliss_results_folder(base_dir="results", suffix="bliss")` | Create a timestamped results directory. |
| `ensure_output_folder(output_dir=None)` | Create or validate an explicit output directory, or create a timestamped one when omitted. |
