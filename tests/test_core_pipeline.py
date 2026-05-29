import numpy as np
import pandas as pd
import pytest

from bliss.line_search.blind_line_search import (
    BlindLineSearchConfig,
    BlindLineSearchPipeline,
    FINAL_OUTPUT_COLUMNS,
    find_emission_lines,
)


def _candidate_table(probability=0.4):
    return pd.DataFrame(
        {
            "center": [6.4],
            "ecenter": [0.01],
            "sigma": [0.04],
            "esigma": [0.005],
            "amplitude": [35.0],
            "eamplitude": [2.0],
            "relative_power": [0.2],
            "rsq": [0.95],
            "noise_on_block": [2.0],
            "value_on_line": [135.0],
            "base_on_line": [100.0],
            "cluster_probability": [probability],
        }
    )


def test_pipeline_direct_arrays_run_with_monkeypatched_candidate_steps(monkeypatch, tmp_path):
    energy = np.linspace(1.0, 10.0, 200)
    counts = 100.0 + 35.0 * np.exp(-0.5 * ((energy - 6.4) / 0.04) ** 2)
    errors = np.full_like(energy, 2.0)

    from bliss.line_search import blind_line_search as bls

    monkeypatch.setattr(bls, "return_raw_lines", lambda *args, **kwargs: _candidate_table())
    monkeypatch.setattr(bls, "calculate_synthetic_lines_spectra", lambda x, y, sy, n: (x, y * 0, sy))
    monkeypatch.setattr(bls, "eval_line_probability_gmm", lambda lines, *args, **kwargs: lines)

    result = find_emission_lines(energy, y=counts, sy=errors, output_dir=tmp_path, show_plot=False)

    assert list(result.columns) == FINAL_OUTPUT_COLUMNS
    assert len(result) == 1
    assert result.loc[0, "center"] == pytest.approx(6.4, abs=0.02)
    assert result.loc[0, "snr"] > 4
    assert result.loc[0, "cluster_probability"] == 1
    assert (tmp_path / "candidate_lines.csv").exists()
    assert (tmp_path / "run_summary.txt").read_text().startswith("BLiSS run completed")


def test_pipeline_loads_file_and_handles_empty_candidates(tmp_path):
    path = tmp_path / "spectrum.dat"
    path.write_text("1.0 1.1 10 1\n0.5 0.6 8 1\n")
    pipeline = BlindLineSearchPipeline(BlindLineSearchConfig())

    spectrum = pipeline._load_input(path)
    assert np.all(np.diff(spectrum.energy) > 0)
    assert spectrum.bin_width.tolist() == pytest.approx([0.1, 0.1])

    result, yfit = pipeline._final_fit_and_metrics(
        spectrum=spectrum,
        base=np.array([8.0, 9.0]),
        ylines=np.array([0.0, 1.0]),
        clean_lines=_candidate_table().iloc[0:0],
    )
    assert result.empty
    assert np.allclose(yfit, 0)


def test_pipeline_rejects_bad_inputs(tmp_path):
    pipeline = BlindLineSearchPipeline()
    with np.testing.assert_raises(ValueError):
        pipeline._load_input([1, 2, 3])
    bad = tmp_path / "bad.dat"
    bad.write_text("1 2 3\n")
    with np.testing.assert_raises(ValueError):
        pipeline._load_input(bad)
