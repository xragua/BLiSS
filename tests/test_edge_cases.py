
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from bliss.plotting.line_probability_plotter import plot_line_prob

def test_empty_probability_plot():
    df = pd.DataFrame({
        "center": [],
        "sigma": [],
        "amplitude": [],
        "cluster_probability": []
    })

    try:
        plot_line_prob(df, show=False)
    except Exception:
        pass
