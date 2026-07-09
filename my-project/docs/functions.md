# Functions

This page summarizes the main public functions used in a standard BLiSS workflow.

The recommended workflow is:

1. optionally rebin the spectrum;
2. estimate or inspect the empirical baseline;
3. run the blind candidate search with `find_candidate_lines`;
4. filter the returned candidate table;
5. run the optional global multi-Gaussian fit with `fit_global`;
6. inspect the result with the plotting utilities.

```python
from bliss.spectrum_data.rebinning_tools import rebin_bins, rebin_snr, rebin_resolution
from bliss.line_search.empirical_baseline import base_calculator
from bliss.line_search.blind_line_search import find_candidate_lines, fit_global
from bliss.plotting.line_probability_plotter import plot_line_prob
```

---

## Baseline estimation

### `base_calculator(y, min_window=None, max_window=None, n_windows=30, n_iter=1, clip_sigma=1.0, reject="both", sigma_mode="global", clip_to_data=False, hair_window=9, hair_clip_sigma=2.0, hair_n_iter=10, hair_strength=1.0, return_info=False)`

Estimate the default BLiSS empirical baseline for a one-dimensional spectrum.

The baseline is an algorithmic lower-envelope estimate. It is intended for blind line detection and should not be interpreted as a physical continuum model.

The default method applies:

1. a family of sigma-clipped moving averages;
2. the point-wise minimum across window sizes;
3. a one-sided correction that removes narrow downward baseline spikes.

By default, the moving-average windows are chosen adaptively from the spectrum length:

- `min_window = max(len(y) / 100, 5)`
- `max_window = max(len(y) / 10, 50)`

**Parameters**

- `y` *(array-like)* – Input spectral values.
- `min_window`, `max_window` *(int or None)* – Minimum and maximum moving-average window sizes in bins. If omitted, adaptive values are used.
- `n_windows` *(int, default=30)* – Number of window sizes sampled between `min_window` and `max_window`.
- `n_iter` *(int, default=1)* – Number of sigma-clipping iterations.
- `clip_sigma` *(float, default=1.0)* – Sigma threshold used during clipping.
- `reject` *{"both", "positive", "negative"}, default="both"* – Which residuals are rejected during sigma clipping.
- `sigma_mode` *{"global", "local"}, default="global"* – Whether the clipping dispersion is estimated globally or locally.
- `clip_to_data` *(bool, default=False)* – If `True`, force the final baseline not to exceed the data.
- `hair_window` *(int, default=9)* – Running-median window for the downward-spike correction.
- `hair_clip_sigma` *(float, default=2.0)* – Clipping threshold for downward-spike detection.
- `hair_n_iter` *(int, default=10)* – Maximum number of downward-spike correction iterations.
- `hair_strength` *(float, default=1.0)* – Strength of the downward-spike correction.
- `return_info` *(bool, default=False)* – If `True`, also return diagnostic information.

**Returns**

- `baseline` *(numpy.ndarray)* – Empirical baseline with the same length as `y`.
- `info` *(dict, optional)* – Returned only if `return_info=True`.

**Example**

```python
base = base_calculator(values)
ylines = np.maximum(values - base, 0)
```

---

## Rebinning functions

### `rebin_bins(x, y, sy, nbin)`

Rebin a spectrum by grouping a fixed number of original bins into each output bin.

Values are combined using inverse-variance weighting, with weights `1 / sy**2`.

**Use case**

Use this when you want a simple reduction of spectral resolution by grouping every `nbin` adjacent points.

**Parameters**

- `x` *(array-like)* – Energy, wavelength, or spectral coordinate array.
- `y` *(array-like)* – Spectral values.
- `sy` *(array-like)* – One-sigma uncertainties associated with `y`.
- `nbin` *(int)* – Number of input points per output bin.

**Returns**

- `x_rebinned` *(numpy.ndarray)* – Weighted mean coordinate values.
- `y_rebinned` *(numpy.ndarray)* – Weighted mean spectral values.
- `sy_rebinned` *(numpy.ndarray)* – Combined one-sigma uncertainties.

---

### `rebin_snr(x, y, sy, snr_threshold)`

Adaptively rebin a spectrum until each output bin reaches a target signal-to-noise ratio.

Adjacent points are accumulated until the combined bin reaches `snr_threshold`.

**Use case**

Use this for faint spectra or spectra with strongly variable statistical quality across the band.

**Parameters**

- `x` *(array-like)* – Energy, wavelength, or spectral coordinate array.
- `y` *(array-like)* – Spectral values.
- `sy` *(array-like)* – One-sigma uncertainties associated with `y`.
- `snr_threshold` *(float)* – Minimum target S/N per output bin.

**Returns**

- `x_rebinned` *(numpy.ndarray)* – Weighted mean coordinate values.
- `y_rebinned` *(numpy.ndarray)* – Weighted mean spectral values.
- `sy_rebinned` *(numpy.ndarray)* – Combined one-sigma uncertainties.

---

### `rebin_resolution(x, y, sy, resolution)`

Rebin a spectrum into bins of approximately fixed width in the spectral coordinate.

**Use case**

Use this when you want a uniform spectral spacing, for example to match an instrumental resolution or to prepare spectra for repeated BLiSS searches over the same grid.

**Parameters**

- `x` *(array-like)* – Energy, wavelength, or spectral coordinate array.
- `y` *(array-like)* – Spectral values.
- `sy` *(array-like)* – One-sigma uncertainties associated with `y`.
- `resolution` *(float)* – Width of each output bin in the same units as `x`.

**Returns**

- `x_rebinned` *(numpy.ndarray)* – Weighted mean coordinate values.
- `y_rebinned` *(numpy.ndarray)* – Weighted mean spectral values.
- `sy_rebinned` *(numpy.ndarray)* – Combined one-sigma uncertainties.

**Example**

```python
energy_reb, values_reb, errors_reb = rebin_resolution(
    energy,
    values,
    errors,
    resolution=0.01,
)
```

---

## Blind candidate search

### `find_candidate_lines(spectra_or_energy, y=None, sy=None, en1=0, en2=10, energy_pad=0.0, output_dir=None)`

Run BLiSS up to candidate detection and probability estimation.

This is the recommended first step of the BLiSS line-search workflow. It estimates the empirical baseline, isolates positive excess regions, fits local Gaussian candidates, compares the detections with synthetic residual spectra, and returns a candidate-line catalogue.

This function does **not** perform the final global multi-Gaussian fit. To do that, filter the returned candidate table and pass it to `fit_global`.

**Parameters**

- `spectra_or_energy` *(str, pathlib.Path, Spectrum-like object, or array-like)* – Input spectrum. This can be:
  - a four-column ASCII file with `E_low, E_high, values, error`;
  - an object with `energy`, `values`, and `uncertainties` attributes;
  - an array of spectral coordinates, in which case `y` and `sy` must also be provided.
- `y` *(array-like or None, default=None)* – Spectral values for direct array input.
- `sy` *(array-like or None, default=None)* – One-sigma uncertainties for direct array input.
- `en1` *(float, default=0)* – Lower limit of the nominal search interval.
- `en2` *(float, default=10)* – Upper limit of the nominal search interval.
- `energy_pad` *(float, default=0.0)* – Extra padding added internally around `en1` and `en2` during candidate detection. The final returned table is still restricted to `en1`–`en2`.
- `output_dir` *(str, pathlib.Path, or None, default=None)* – If provided, save `candidate_lines.csv` and `run_summary.txt` in this directory. If omitted, BLiSS creates a timestamped results folder.

**Returns**

- `pandas.DataFrame` – Candidate-line table before the optional global fit.

The returned table contains:

```text
center, ecenter, sigma, esigma, amplitude, eamplitude,
relative_power, noise_on_block, value_on_line, base_on_line,
snr_peak, snr_amplitude, area, earea, snr_area, ew,
cluster_probability
```

**Example**

```python
candidate_lines = find_candidate_lines(
    energy,
    values,
    errors,
    en1=6.0,
    en2=7.2,
    energy_pad=0.1,
    output_dir="results/vela_x1_fe_candidates",
)
```

A typical filtering step before the global fit could be:

```python
clean_mask = (
    (candidate_lines["cluster_probability"] >= 0.90)
    & (candidate_lines["relative_power"] >= 0.10)
    & (
        (candidate_lines["snr_peak"] >= 5)
        | (candidate_lines["snr_area"] >= 5)
        | (candidate_lines["snr_amplitude"] >= 5)
    )
)

clean_lines = candidate_lines[clean_mask].reset_index(drop=True)
```

---

## Global multi-Gaussian fit

### `fit_global(pd_lines, spectra_or_energy, y=None, sy=None, *, bin_width=None, base=None, ylines=None, show_plot=True, output_dir=None, plot_name="bliss_global_fit.png", save_csv=True, final_fit_maxfev=100000, snr_confidence_threshold=4.0, return_yfit=False, energy_min=None, energy_max=None, size_fig_input=None)`

Run the final simultaneous multi-Gaussian fit on a user-selected candidate table.

This function is intended to be called after `find_candidate_lines`, once the user has applied their own filters to the candidate catalogue.

The fit is performed on the line-excess spectrum, using the empirical baseline either computed internally or supplied through `base` and `ylines`.

**Parameters**

- `pd_lines` *(pandas.DataFrame)* – Candidate-line table after user filtering. It must contain at least `amplitude`, `center`, and `sigma`.
- `spectra_or_energy` *(str, pathlib.Path, Spectrum-like object, or array-like)* – Same input types accepted by `find_candidate_lines`.
- `y` *(array-like or None, default=None)* – Spectral values for direct array input.
- `sy` *(array-like or None, default=None)* – One-sigma uncertainties for direct array input.
- `bin_width` *(array-like or None, default=None)* – Optional bin widths for direct array input. If omitted, they are estimated from adjacent coordinate spacing.
- `base` *(numpy.ndarray or None, default=None)* – Optional precomputed empirical baseline.
- `ylines` *(numpy.ndarray or None, default=None)* – Optional precomputed positive line-excess array.
- `show_plot` *(bool, default=True)* – If `True`, display the final diagnostic plot.
- `output_dir` *(str, pathlib.Path, or None, default=None)* – If provided, save the fitted line table and diagnostic plot there.
- `plot_name` *(str, default="bliss_global_fit.png")* – Diagnostic plot filename inside `output_dir`.
- `save_csv` *(bool, default=True)* – If `True`, save `global_fit_lines.csv` when `output_dir` is provided.
- `final_fit_maxfev` *(int, default=100000)* – Maximum number of function evaluations passed to `scipy.optimize.curve_fit`.
- `snr_confidence_threshold` *(float, default=4.0)* – If any available S/N diagnostic exceeds this value, `cluster_probability` is set to 1.
- `return_yfit` *(bool, default=False)* – If `True`, return `(result, yfit)` instead of only `result`.
- `energy_min`, `energy_max` *(float or None, default=None)* – Minimum and maximum energy shown in the diagnostic plot. These values affect only the plot, not the fitted range.
- `size_fig_input` *(tuple or None, default=None)* – Figure size passed to the diagnostic plot, for example `(10, 5)`.

**Returns**

- `pandas.DataFrame` – Final fitted line table.
- `(pandas.DataFrame, numpy.ndarray)` – Returned only if `return_yfit=True`; the second element is the fitted line-only model.

The final table contains:

```text
center, ecenter, sigma, esigma, amplitude, eamplitude,
relative_power, noise_on_block, value_on_line, base_on_line,
snr_peak, snr_amplitude, area, earea, snr_area, ew,
cluster_probability, fit_error_flag
```

`fit_error_flag` is set to `unconstrained` when the covariance-derived uncertainties are not informative.

**Example**

```python
global_lines = fit_global(
    clean_lines,
    energy,
    values,
    errors,
    show_plot=True,
    output_dir="results/vela_x1_fe_global_fit",
    energy_min=6.0,
    energy_max=7.2,
    size_fig_input=(10, 5),
)
```

If the line-only model is also needed:

```python
global_lines, yfit = fit_global(
    clean_lines,
    energy,
    values,
    errors,
    return_yfit=True,
)
```

---

## Plotting functions

### `plot_line_prob(df, show=True, size_fig_input=None)`

Plot Gaussian candidate components colored by `cluster_probability`.

If the input table contains an `ion` column, ion labels are added automatically above the corresponding Gaussian components.

**Parameters**

- `df` *(pandas.DataFrame)* – Candidate or fitted-line table. It must contain finite values in `center`, `sigma`, `amplitude`, and `cluster_probability`.
- `show` *(bool, default=True)* – If `True`, display the figure and return `None`. If `False`, return the matplotlib `(fig, ax)` objects.
- `size_fig_input` *(tuple or None, default=None)* – Optional figure size, for example `(10, 5)`.

**Returns**

- `None` – If `show=True`.
- `(fig, ax)` – If `show=False`.

**Example**

```python
plot_line_prob(global_lines)
```

To save the figure manually:

```python
fig, ax = plot_line_prob(global_lines, show=False)
fig.savefig("line_probabilities.png", dpi=150, bbox_inches="tight")
```

---

### `plot_global_fit(spectrum, base, yfit, output_path=None, *, show_plot=True, energy_min=None, energy_max=None, size_fig_input=None)`

Plot the spectrum, empirical baseline, fitted line-only model, and total `baseline + line model`.

This function is used internally by `fit_global`, but it can also be used directly when working with a prepared BLiSS spectrum.

**Parameters**

- `spectrum` *(`PreparedSpectrum`)* – Prepared spectrum containing `energy`, `values`, `uncertainties`, and `bin_width`.
- `base` *(numpy.ndarray)* – Empirical baseline evaluated on the same grid.
- `yfit` *(numpy.ndarray)* – Fitted line-only model evaluated on the same grid.
- `output_path` *(str, pathlib.Path, or None, default=None)* – If provided, save the plot to this path.
- `show_plot` *(bool, default=True)* – If `True`, display the figure.
- `energy_min`, `energy_max` *(float or None, default=None)* – Optional x-axis limits.
- `size_fig_input` *(tuple or None, default=None)* – Optional figure size.

**Returns**

- `None`

---

### `plot_final_bliss_fit(energy, values, uncertainties, baseline, line_model, output_path)`

Save a diagnostic plot of the data, empirical baseline, line-only model, and total model.

**Parameters**

- `energy` *(numpy.ndarray)* – Spectral coordinate grid.
- `values` *(numpy.ndarray)* – Observed spectral values.
- `uncertainties` *(numpy.ndarray)* – One-sigma uncertainties plotted as error bars.
- `baseline` *(numpy.ndarray)* – Empirical baseline.
- `line_model` *(numpy.ndarray)* – Fitted line-only model evaluated on `energy`.
- `output_path` *(str or pathlib.Path)* – Destination image path.

**Returns**

- `None`

**Example**

```python
plot_final_bliss_fit(
    energy,
    values,
    errors,
    base,
    yfit,
    "final_bliss_fit.png",
)
```
