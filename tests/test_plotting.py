
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from bliss.plotting.line_score_plotter import plot_bliss_score

def test_plotting_executes():
    df = pd.DataFrame({
        "center": [6.4, 6.7],
        "sigma": [0.1, 0.1],
        "amplitude": [10, 15],
        "bliss_score": [0.8, 0.9],
        "ion": ["Fe I", "Fe XXV"]
    })

    fig, ax = plot_bliss_score(df, show=False)

    assert fig is not None
    assert ax is not None
