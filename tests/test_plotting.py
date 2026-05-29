
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from bliss.plotting.line_probability_plotter import plot_line_prob

def test_plotting_executes():
    df = pd.DataFrame({
        "center": [6.4, 6.7],
        "sigma": [0.1, 0.1],
        "amplitude": [10, 15],
        "cluster_probability": [0.8, 0.9],
        "ion": ["Fe I", "Fe XXV"]
    })

    fig, ax = plot_line_prob(df, show=False)

    assert fig is not None
    assert ax is not None
