# BLiSS — Blind Line Search System

[![arXiv](https://img.shields.io/badge/arXiv-2607.07783-b31b1b.svg)](https://arxiv.org/abs/2607.07783)
[![Documentation](https://img.shields.io/badge/docs-BLiSS-blue.svg)](https://xragua.github.io/BLiSS/)

**BLiSS** (Blind Line Search System) is an open-source Python package for the automatic detection and identification of emission lines in astronomical spectra.

Rather than requiring users to manually inspect spectra or provide a list of candidate line energies, BLiSS performs a genuine blind line search directly on the data, ranking statistically significant emission-line candidates and optionally identifying them using atomic databases.

Originally developed for X-ray spectroscopy, BLiSS is designed around a general workflow that can be applied to any spectral dataset with associated uncertainties.

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
  - Cross-matches detected features with XSTAR atomic transitions.
  - Supports Doppler velocity constraints.

- **Flexible rebinning**
  - Rebin spectra by instrumental resolution or target signal-to-noise ratio.

- **ISIS integration**
  - Automatically generates Gaussian components for ISIS spectral fitting.

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

> Abalo, L., Sanjurjo-Ferrín, G., et al. (2026), *BLiSS: Blind Line Search System*, Astronomy & Computing.

(The citation will be updated once the paper is published.)

---

# Support

Bug reports, feature requests and suggestions are welcome through GitHub Issues:

https://github.com/xragua/bliss/issues

For scientific questions, you can also contact:

luisabalo.com 

---

# License

This project is distributed under the **MIT License**.
## bliss_score

`bliss_score` is an empirical score in [0, 1] based on the excess of candidate
rates over synthetic null spectra. It is not a calibrated posterior probability
or a detection p-value. Its definition and numerical calculation are unchanged
by this naming update. New candidate tables use `bliss_score` and
`bliss_score_status`; plots and notebook thresholds use the same terminology.

Use `from bliss import read_bliss_csv` to read historical candidate CSVs with
the new column names without modifying the original experiment files. The
public scoring API is `calculate_bliss_score`, `eval_bliss_score_gmm`,
`GMMBlissScoreEvaluator`, and `plot_bliss_score`. Restart notebook kernels after
updating the package. Historical figures and experiment manifests retain the
labels and code hashes of the run that produced them.
