
%[![status](https://joss.theoj.org/papers/aaea14e694ee06c3d7258dcf71487da7/status.svg)]%(https://joss.theoj.org/papers/aaea14e694ee06c3d7258dcf71487da7)

# BLiSS — Blind Line Search System

**BLiSS** (Blind Line Search System) is an open-source Python package for finding emission lines in spectra without telling the code where to look first.


In other words: a **real blind line-search algorithm** — you only need your spectrum.

**BLiSS** was designed to simplify and speed up spectral analysis in X-ray astronomy. Instead of manually inspecting spectra, guessing candidate energies, and testing lines one by one, **BLiSS** performs an automatic search for statistically significant emission features and then cross-matches them with atomic transitions.


The package combines Gaussian line-profile fitting, statistical scoring, and database-based identification using XSTAR atomic data. It is especially useful when the continuum is complex, uncertain, or simply not the main focus of the analysis.


Although **BLiSS** was developed for X-ray spectra, the core idea is more general: if you have spectral data with uncertainties, BLiSS can help you search for emission features in a systematic and reproducible way.


---

## Main features

BLiSS is designed to make emission-line searches fast, systematic, and reproducible. Its main features include:

- **Real blind emission-line search**  
  No prior list of candidate energies is required. BLiSS searches the spectrum directly and identifies statistically relevant emission-like features.

- **Automatic Gaussian fitting**  
  Candidate features are modeled with Gaussian profiles and ranked using statistical scores, helping users decide which lines deserve further inspection.

- **Continuum-independent workflow**  
  BLiSS does not require a predefined global continuum model to start searching for lines, making it especially useful for complex or uncertain spectral shapes.

- **Atomic line identification with XSTAR**  
  Detected candidates can be cross-matched with atomic transitions from XSTAR, including Doppler-velocity constraints to account for shifted lines.

- **Flexible rebinning utilities**  
  BLiSS includes tools to rebin spectra according to instrumental resolution or to reach a target signal-to-noise ratio.

- **ISIS integration**  
  Detected emission lines can be automatically added to ISIS spectral models, simplifying follow-up fitting and interpretation.

- **Reproducible analysis**  
  The same search criteria can be applied consistently across many spectra, reducing subjective choices and making BLiSS useful for large datasets.


If you have any questions or need assistance, please feel free to reach out: graciela.sanjurjo@ua.es.



## Getting started

### Installation
---------------------------------------------------------


You can install the package directly from PyPI using pip: **pip install bliss**.

Or download the code from [here](https://github.com/xragua/bliss/releases/tag/0.2.9).

Some examples of their usage are presented [here](https://github.com/xragua/bliss/tree/main/example).

And here [here](https://github.com/xragua/bliss) is the page of this package 

---

## License

MIT license
