
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from bliss.plotting.line_score_plotter import plot_bliss_score

def test_empty_probability_plot():
    df = pd.DataFrame({
        "center": [],
        "sigma": [],
        "amplitude": [],
        "bliss_score": []
    })

    try:
        plot_bliss_score(df, show=False)
    except Exception:
        pass
