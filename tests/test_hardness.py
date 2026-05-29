import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

from bliss.line_search.empirical_baseline import moving_average, base_calculator
from bliss.line_search.peak_selection import find_peaks_new
from bliss.line_search.gaussian_models import gaussian, n_gaussian, p0_generator, p0_generator_final
from bliss.line_search.candidate_regions import (
    CandidateBlock,
    CandidateRegionDetector,
    _add_line_context,
    _build_candidate_blocks,
    _fit_candidate_block,
    return_raw_lines,
)
from bliss.spectrum_data.rebinning_tools import _clean_arrays, rebin_bins, rebin_snr, rebin_resolution
from bliss.spectrum_data.text_spectrum_loader import load_text_spectrum
from bliss.spectrum_data.spectrum_container import Spectrum
from bliss.synthetic_probability.synthetic_spectra import SyntheticSpectrumGenerator, calculate_synthetic_lines_spectra
from bliss.synthetic_probability.gmm_probability import GMMLineProbabilityEvaluator, eval_line_probability_gmm, real_probability
from bliss.line_identification.atomic_line_table import load_atomic_database
from bliss.line_identification.line_identifier import (
    LineIdentifier,
    add_most_probable_ion,
    get_all_compatible_lines,
    identify_line,
)
from bliss.plotting.line_probability_plotter import plot_line_prob
from bliss.plotting.run_output_manager import create_bliss_results_folder, ensure_output_folder
from bliss.plotting.spectrum_diagnostic_plotter import plot_final_bliss_fit
from bliss.isis_interface.isis_script_writer import (
    _component_name,
    _prepare_candidates,
    _safe_model_suffix,
    spectrum_axis_is_keV,
    write_isis_files_from_bliss_results,
    write_isis_line_model_files,
)
from bliss.isis_interface.isis_model_cleaner import (
    _area_value,
    _parse_float,
    _renumber_parameter_rows,
    clean_zero_area_egauss_model,
)


def test_baseline_peak_and_gaussian_helpers():
    x = np.linspace(1, 10, 120)
    y = 4 + gaussian(x, 12, 5, 0.2) + gaussian(x, 6, 7, 0.15)

    avg = moving_average(y, 5)
    base = base_calculator(y)
    peaks = find_peaks_new(x, y)

    assert len(avg) == len(y)
    assert np.all(base <= y)
    assert len(peaks) >= 2
    assert find_peaks_new(x, np.ones_like(x)).empty
    assert np.allclose(n_gaussian(x, 1, 5, 0.2, 2, 6, 0.1), gaussian(x, 1, 5, 0.2) + gaussian(x, 2, 6, 0.1))

    good_peaks = peaks.iloc[:1].reset_index(drop=True)
    p0, bounds = p0_generator(x, y, good_peaks)
    assert len(p0) == 3
    clean_lines = pd.DataFrame({"amplitude": [10.0], "center": [5.0], "sigma": [0.03]})
    p0_final, bounds_final = p0_generator_final(x, y, clean_lines)
    assert p0_final == [10.0, 5.0, 0.03]
    assert bounds[0][0] == 0
    assert bounds_final[1][1] == pytest.approx(5.1)


def test_candidate_region_detection_and_fitting(monkeypatch):
    x = np.linspace(1, 4, 80)
    y = 10 + 30 * np.exp(-0.5 * ((x - 2.5) / 0.08) ** 2)
    sy = np.ones_like(x)
    base = np.full_like(x, 10.0)
    ylines = y - base

    detector = CandidateRegionDetector()
    detected = detector.detect(x, y, sy, ylines, base)
    assert isinstance(detected, pd.DataFrame)
    blocks = _build_candidate_blocks(x, y, sy, ylines, base)
    assert all(isinstance(block, CandidateBlock) for block in blocks)
    assert len(blocks) >= 1

    fitted = return_raw_lines(x, y, sy, ylines, base)
    assert {"center", "relative_power", "base_on_line"}.issubset(fitted.columns)

    empty = _add_line_context(pd.DataFrame(), x, y, base)
    assert empty.empty and "relative_power" in empty.columns
    assert _fit_candidate_block(CandidateBlock(np.array([1.0]), x[:1], y[:1], sy[:1], base[:1]), 0) == []

    # Exercise the exception branch without requiring scipy to fail naturally.
    from bliss.line_search import candidate_regions as cr
    monkeypatch.setattr(cr, "curve_fit", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("forced")))
    assert _fit_candidate_block(blocks[0], 0) == []


def test_spectrum_loading_rebinning_and_container(tmp_path):
    text = tmp_path / "spec.txt"
    text.write_text("2.0 2.2 20 2\n1.0 1.2 10 1\n# ignored\n")
    energy, values, errors, widths = load_text_spectrum(text)
    assert energy.tolist() == [1.1, 2.1]
    assert widths.tolist() == pytest.approx([0.2, 0.2])

    with pytest.raises(ValueError):
        bad = tmp_path / "bad.txt"; bad.write_text("1 2 3\n"); load_text_spectrum(bad)

    spec = Spectrum(np.array([2.0, 1.0]), np.array([20.0, 10.0]), np.array([2.0, 1.0]), np.array([0.2, 0.1])).sorted()
    assert spec.energy.tolist() == [1.0, 2.0]
    assert Spectrum(np.array([2.0, 1.0]), np.array([20.0, 10.0]), np.array([2.0, 1.0])).sorted().bin_width is None

    x = np.arange(1, 10, dtype=float)
    y = np.arange(10, 19, dtype=float)
    sy = np.ones_like(x)
    clean_x, clean_y, clean_sy = _clean_arrays([1, 2, np.nan], [2, np.inf, 4], [1, 1, -1])
    assert clean_x.tolist() == [1]
    assert len(rebin_bins(x, y, sy, 2)[0]) > 0
    assert len(rebin_snr(x, y, sy, 0.2)[0]) > 0
    xr, yr, syr = rebin_resolution(x, y, sy, 2.0)
    assert len(xr) == len(yr) == len(syr)


def test_synthetic_spectra_and_gmm_probability():
    t = np.linspace(1, 2, 30)
    c = np.ones(30)
    c[3] = 100.0
    sc = np.full(30, 0.2)
    tsim, simc, ssimc = calculate_synthetic_lines_spectra(t, c, sc, num_simulations=3, seed=1, z_score_th=2)
    assert len(tsim) == len(simc) == len(ssimc) == 90
    assert SyntheticSpectrumGenerator(num_simulations=1, seed=2).generate(t, c, sc)[0].size == 30
    assert real_probability(0, 10) == 0
    assert real_probability(10, 2) == pytest.approx(0.8)

    lines = pd.DataFrame(
        {
            "amplitude": [10.0, 12.0],
            "sigma": [0.1, 0.12],
            "value_on_line": [20.0, 22.0],
            "noise_on_block": [2.0, 2.0],
        }
    )
    simlines = pd.DataFrame(
        {
            "amplitude": [2.0, 3.0, 2.5],
            "sigma": [0.2, 0.22, 0.21],
            "value_on_line": [4.0, 5.0, 4.5],
            "noise_on_block": [2.0, 2.0, 2.0],
        }
    )
    out = eval_line_probability_gmm(lines.copy(), simlines.copy(), np.linspace(1, 3, 20), np.linspace(1, 3, 10), k_min=1, k_max=2, covariance_types=("full", "diag"), show_plot=True)
    assert "cluster_probability" in out.columns
    assert GMMLineProbabilityEvaluator(k_min=1, k_max=1).evaluate(lines.copy(), simlines.copy(), np.arange(5), np.arange(5)).shape[0] == 2


def test_line_identification_and_atomic_table():
    table = load_atomic_database()
    assert "energy_keV" in table.columns
    target = float(table.energy_keV.iloc[0])
    identified = identify_line(target, v_doppler_kms=3000)
    assert not identified.empty

    lines = pd.DataFrame({"center": [target, 999.0], "sigma": [0.01, 0.01], "amplitude": [1.0, 1.0]})
    top = add_most_probable_ion(lines, 3000)
    compatible = get_all_compatible_lines(lines, 3000)
    helper = LineIdentifier(v_doppler_kms=3000)
    assert "ion" in top.columns
    assert 0 in compatible and 1 in compatible
    assert not helper.add_most_probable(lines.iloc[:1]).empty
    assert not helper.all_compatible(lines.iloc[:1])[0].empty


def test_plotting_and_output_helpers(tmp_path):
    df = pd.DataFrame({"center": [6.4, 6.7], "sigma": [0.1, 0.1], "amplitude": [10, 15], "cluster_probability": [0.8, 0.9], "ion": ["Fe I", "Fe XXV"]})
    fig, ax = plot_line_prob(df, show=True)
    assert fig is not None and ax is not None

    out = tmp_path / "diagnostic.png"
    energy = np.linspace(1, 2, 10)
    plot_final_bliss_fit(energy, energy * 0 + 2, energy * 0 + 0.1, energy * 0 + 1.5, energy * 0 + 0.2, out)
    assert out.exists()

    made = create_bliss_results_folder(tmp_path, suffix="unit")
    ensured = ensure_output_folder(tmp_path / "manual")
    assert made.exists() and ensured.exists()


def _isis_candidates():
    return pd.DataFrame({"center": [6.4], "ecenter": [0.01], "sigma": [0.05], "esigma": [0.002], "extra": [7]})


def test_isis_writers_and_cleaner(tmp_path):
    assert _component_name(True) == "egauss"
    assert _component_name(False) == "gauss"
    assert _safe_model_suffix(None) == ""
    assert _safe_model_suffix("m1") == "m1"

    with pytest.raises(ValueError):
        _prepare_candidates(pd.DataFrame({"center": [1]}))
    clean = _prepare_candidates(_isis_candidates(), add_sentinel=True)
    assert clean.center.iloc[-1] == 100000000.0

    header = tmp_path / "keV_spec.txt"
    header.write_text("# Energy (keV) counts\n")
    assert spectrum_axis_is_keV(header) is True
    assert spectrum_axis_is_keV(tmp_path / "missing.txt") is True
    no_keV = tmp_path / "angstrom.txt"
    no_keV.write_text("# Angstrom spectrum\n")
    assert spectrum_axis_is_keV(no_keV) is False

    written = write_isis_line_model_files(_isis_candidates(), tmp_path / "isis", model_name="test", use_egauss=False, add_sentinel_line=False)
    assert written["model_file"].read_text().count("gauss(1)") >= 1
    assert "set_par" in written["parameter_file"].read_text()
    written2 = write_isis_files_from_bliss_results(_isis_candidates(), tmp_path / "isis2", text_spectrum_path=no_keV)
    assert written2["model_file"].exists()

    assert _parse_float("1.2") == 1.2
    assert _parse_float("bad") is None
    assert _area_value("1  a b c 0.0 d e") == 0.0
    assert _area_value("too short") is None
    assert _renumber_parameter_rows(["  9  foo", "bar"])[0].startswith("   1")

    par = tmp_path / "in.par"
    par.write_text(
        "oldmodel\nheader\n"
        "  1  egauss(1).area     x x 0.0 x x\n"
        "  2  egauss(1).center   x x 6.4 x x\n"
        "  3  egauss(1).sigma    x x 0.1 x x\n"
        "  4  egauss(2).area     x x 5.0 x x\n"
        "  5  egauss(2).center   x x 6.7 x x\n"
        "  6  egauss(2).sigma    x x 0.1 x x\n"
        "not a parameter\n"
    )
    out = clean_zero_area_egauss_model(par, tmp_path / "out.par")
    text = out.read_text()
    assert "egauss(1)" in text and "egauss(2)" not in text

    short = tmp_path / "short.par"
    short.write_text("only one line\n")
    assert clean_zero_area_egauss_model(short, tmp_path / "short_out.par").exists()
