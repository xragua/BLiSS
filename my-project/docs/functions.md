
# FUNCTIONS

---

## Rebin functions

---

### `rebin_bins(x, y, sy, nbin)`

Rebins spectra by grouping data points into fixed-size bins.  
Each bin contains a specified number of data points (`nbin`), and values are combined  
using weighted averages (`weights = 1 / σ²`).  

**Use case:**  
Reduce data resolution or smooth spectra without biasing results toward low-uncertainty points.

**Parameters:**  
- `x` *(array-like)* – Energy or wavelength array.  
- `y` *(array-like)* – Flux or counts corresponding to `x`.  
- `sy` *(array-like)* – 1σ uncertainties associated with `y`.  
- `nbin` *(int)* – Number of points per new bin.

**Returns:**  
- `x_rebinned` *(np.ndarray)* – Weighted mean x-values per bin.  
- `y_rebinned` *(np.ndarray)* – Weighted mean y-values per bin.  
- `sy_rebinned` *(np.ndarray)* – Combined 1σ errors per bin.

---

### `rebin_snr(x, y, sy, snr_threshold)`

Performs adaptive rebinning based on a target signal-to-noise ratio (SNR).  
Data points are accumulated until the combined bin reaches the desired SNR,  
ensuring consistent statistical quality across the spectrum. Ideal for faint sources or data with variable noise levels.

**Parameters:**  
- `x` *(array-like)* – Energy or wavelength array.  
- `y` *(array-like)* – Flux or counts corresponding to `x`.  
- `sy` *(array-like)* – 1σ uncertainties associated with `y`.  
- `snr_threshold` *(float)* – Minimum SNR per output bin.

**Returns:**  
- `x_rebinned` *(np.ndarray)* – Weighted mean x-values per bin.  
- `y_rebinned` *(np.ndarray)* – Weighted mean y-values per bin.  
- `sy_rebinned` *(np.ndarray)* – Combined 1σ errors per bin.

---

### `rebin_resolution(x, y, sy, resolution)`

Rebins spectra according to a specified resolution along the x-axis (e.g., energy or wavelength).  
The data are grouped within bins of fixed width (`resolution`), and combined through weighted averages. The aim of this method is to match instrumental resolution or create uniformly spaced spectral data.

**Parameters:**  
- `x` *(array-like)* – Energy or wavelength array.  
- `y` *(array-like)* – Flux or counts corresponding to `x`.  
- `sy` *(array-like)* – 1σ uncertainties associated with `y`.  
- `resolution` *(float)* – Desired bin width for the rebinned spectrum.

**Returns:**  
- `x_rebinned` *(np.ndarray)* – Weighted mean x-values per bin.  
- `y_rebinned` *(np.ndarray)* – Weighted mean y-values per bin.  
- `sy_rebinned` *(np.ndarray)* – Combined 1σ errors per bin.

---

## Find emission lines

---

### `find_emission_lines(spectra_or_energy, y=None, sy=None, en1=0, en2=10, show_plot=False)`

Performs end-to-end emission-line detection and characterization.

1. **Input handling:** accepts a file path to a 4-column ASCII spectrum (`E_low, E_high, counts, error`) or arrays `x, y, sy`.  
2. **Continuum estimation:** computes `base` and `ylines = y - base`.  
3. **Detection:** isolates candidate line regions and fits Gaussian profiles.  
4. **Simulation and scoring:** builds synthetic lines, evaluates probabilities with a Gaussian Mixture Model (GMM).  
5. **Final fitting:** re-fits selected lines globally to refine parameters.  
6. **Metrics:** computes SNR, integrated area, errors, and equivalent width.

**Parameters:**  
- `spectra_or_energy` *(str or array-like)* – Path to ASCII file or energy array.  
- `y` *(array-like, optional)* – Flux or counts (required if `spectra_or_energy` is an array).  
- `sy` *(array-like, optional)* – 1σ uncertainties (required if `spectra_or_energy` is an array).  
- `en1` *(float, optional)* – Lower limit of energy range for line selection (default = 0).  
- `en2` *(float, optional)* – Upper limit of energy range for line selection (default = 10).  
- `show_plot` *(bool, optional)* – If `True`, plots fitted lines and continuum.

**Returns:**  
- `pd.DataFrame` – Table of detected emission lines with columns:  
  `center, sigma, amplitude, ecenter, esigma, eamplitude, base_on_line, value_on_line, noise_on_block, snr, relative_power, area, earea, ew, cluster_probability`.

> **Notes:**  
> - Input files must contain four whitespace-separated columns: `E_low, E_high, counts, error`.  
> - SNR = `amplitude / noise_on_block`.  
> - Area = `amplitude * sigma * sqrt(2π)`.  
> - EW ≈ ∑((line_flux / continuum) × dE) around ±2σ.


---

## Identify lines
Functions for matching detected emission features with catalogued atomic transitions.

---

### `identify_line(center_energy_keV, center_sigma_keV=None, v_doppler_kms=None, pd_data=st_reduced)`

Finds and ranks candidate emission lines near an observed energy.  
The search window is defined by the Doppler velocity tolerance (and optionally by the line’s σ).  
Candidates are ranked by their scaled flux (Aul strength).

**Parameters:**  
- `center_energy_keV` *(float)* – Observed line energy in keV.  
- `center_sigma_keV` *(float, optional)* – Uncertainty (σ) in the observed energy in keV.  
- `v_doppler_kms` *(float, optional)* – Doppler velocity tolerance in km/s.  
- `pd_data` *(pd.DataFrame, optional)* – Reference line catalog (default = `st_reduced`, from the XSTAR database).

**Returns:**  
- `pd.DataFrame` – Candidate emission lines within the energy window, sorted by `scaled_flux` (descending).  
  Includes computed Doppler shift values (`doppler_kms`) and other relevant line fields.

> **Note:** Uses the **XSTAR** line database as the default catalog for matching.

---

### `add_most_probable_ion(pd_fit, v_doppler_kms)`

Annotates each fitted emission line with the **most probable ion** from the reference catalog,  
based on proximity within a Doppler velocity tolerance.

**Parameters:**  
- `pd_fit` *(pd.DataFrame)* – DataFrame of detected emission lines containing at least a `center` column  
  (and optionally `sigma`).  
- `v_doppler_kms` *(float)* – Doppler velocity tolerance in km/s used for matching.

**Returns:**  
- `pd.DataFrame` – Combined DataFrame containing the original fitted lines plus the most probable ion  
  and its catalog properties (e.g., `ion`, `energy_keV`, `doppler_kms`, etc.).

---

### `get_all_compatible_lines(pd_fit, v_doppler_kms)`

For each detected emission line, retrieves **all possible catalog lines** that are compatible  
with the specified Doppler velocity window.

**Parameters:**  
- `pd_fit` *(pd.DataFrame)* – DataFrame of detected emission lines with at least a `center` column  
  (and optionally `sigma`).  
- `v_doppler_kms` *(float)* – Doppler velocity tolerance in km/s used for line matching.

**Returns:**  
- `dict[int, pd.DataFrame]` – Dictionary mapping each index of `pd_fit` to a DataFrame  
  containing all matching catalog lines (empty if no matches are found).

---

## Plot functions
---

### `plot_line_prob(df)`
Plots Gaussian components (e.g., detected emission lines) from a fitted dataset.  
Each Gaussian is colored according to its cluster probability,  
with optional ion labels if the `ion` column is present.  

**Use case:**  
Visualize fitted emission lines and their likelihoods in an informative, publication-ready plot.

**Parameters:**  
- `df` *(pd.DataFrame)* – DataFrame containing Gaussian fit results with columns such as  
  `center`, `sigma`, `amplitude`, and `cluster_probability`.  
  If an `ion` column is present, labels will be added automatically.

**Returns:**  
- *None* – Displays a matplotlib figure showing all fitted components with color-coded probabilities.
