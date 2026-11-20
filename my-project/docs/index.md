# BLiSS — Blind Line Search System

**BLiSS** is an open-source Python package for finding emission lines in spectra without telling the code where to look first. In other words: a **real blind line-search algorithm** — you only need your spectrum.

**BLiSS** was designed to simplify and speed up spectral analysis in X-ray astronomy. Instead of manually inspecting spectra, guessing candidate energies, and testing lines one by one, **BLiSS** performs an automatic search for statistically significant emission features and can optionally cross-match them with a user-supplied table of atomic transitions.


The package combines Gaussian line-profile fitting, Monte Carlo significance estimation, and optional identification using a transition catalogue selected and supplied by the user. No atomic database is shipped with BLiSS. It is especially useful when the continuum is complex, uncertain, or simply not the main focus of the analysis.


Although **BLiSS** was developed for X-ray spectra, the core idea is more general: if you have spectral data with uncertainties, BLiSS can help you search for emission features in a systematic and reproducible way.


---

## Features

- **Blind emission-line search**
  - Automatically searches spectra for statistically significant emission features.
  - No prior knowledge of line positions is required.

- **Automatic Gaussian fitting**
  - Fits candidate features with Gaussian profiles.
  - Provides statistical ranking of all detected lines.

- **Continuum-independent workflow**
  - Line searches can be performed without defining a global continuum model beforehand.

- **Atomic line identification**
  - Cross-matches detected features with a transition table supplied as a pandas DataFrame.
  - Supports Doppler velocity constraints and explicit label, energy/wavelength, and unit settings.
  - Ranks matches by energy proximity unless an optional `weight_col` is specified; no database or scoring scheme is selected automatically.

- **Flexible rebinning**
  - Rebin spectra by instrumental resolution or target signal-to-noise ratio.
  
- **Null-spectrum evaluation**
  - Reports five global Monte Carlo p-values based on synthetic null spectra.

- **Reproducible analyses**
  - Applies identical search criteria across large spectral samples.

---

# Installation

Install BLiSS directly from PyPI:

```bash
pip install bliss-lib
```

or install the development version from GitHub:

```bash
git clone https://github.com/xragua/bliss.git
cd bliss
pip install -e .
```

---

# Quick start

```python
import bliss

# Import your spectrum

# Run the blind line search

# Inspect the detected candidates
```

For optional identification, obtain a transition catalogue independently and
read it with a suitable reader such as `pd.read_csv`. Pass the DataFrame as
`LineIdentifier(atomic_table=catalog, ...)` or `pd_data=catalog` in the functional
API. See [Atomic-line identification](functions.md#atomic-line-identification)
for the required columns and units.

Example notebooks are available in the `notebooks/` directory:

https://github.com/xragua/BLiSS/tree/main/notebooks

---

# Documentation

GitHub repository:

https://github.com/xragua/bliss

Latest releases:

https://github.com/xragua/bliss/releases

---

# Requirements

- Python 3.10 or newer
- NumPy
- SciPy
- Pandas
- Astropy
- Matplotlib

These dependencies are installed automatically with BLiSS.

---

# Citation

If you use BLiSS in your research, please cite:

> @misc{abalo2026blindlinesearchsystem,
      title={Blind Line Search System: BLiSS}, 
      author={Luis Abalo and Graciela Sanjurjo-Ferrín and Jessica Planelles-Villalva and José Joaquín Rodes-Roca and José Miguel Torrejón},
      year={2026},
      eprint={2607.07783},
      archivePrefix={arXiv},
      primaryClass={astro-ph.IM},
      url={https://arxiv.org/abs/2607.07783}, 
}



---

# Support

Bug reports, feature requests and suggestions are welcome through GitHub Issues:

https://github.com/xragua/bliss/issues


---

# License

This project is distributed under the **MIT License**.

