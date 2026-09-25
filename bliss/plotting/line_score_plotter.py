
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_bliss_score(df, show=True, size_fig_input=None):
    """Plot candidate bliss_score as a function of fitted line energy."""
    from ..score_columns import normalize_score_columns
    df = normalize_score_columns(df)
    df = df[
        np.isfinite(df["center"])
        & np.isfinite(df["sigma"])
        & np.isfinite(df["amplitude"])
        & np.isfinite(df["bliss_score"])
        & (df["sigma"] > 0)
    ].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No valid candidate lines to plot.")
    size_fig = size_fig_input if size_fig_input is not None else (10, 5)
    x = np.linspace(
        df["center"].min() - 3 * df["sigma"].max(),
        df["center"].max() + 3 * df["sigma"].max(),
        600,
    )
    unique_scores = np.sort(df["bliss_score"].unique())[::-1]
    cmap = plt.cm.rainbow
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    fig, ax = plt.subplots(figsize=size_fig)
    has_ion = "ion" in df.columns

    max_y = 0.0
    offset = 0.02 * df["amplitude"].max()
    for _, row in df.iterrows():
        p = float(row["bliss_score"])
        color = cmap(norm(p))

        mu = float(row["center"])
        sigma = float(row["sigma"])
        amp = float(row["amplitude"])

        y = amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        ax.plot(x, y, color=color, alpha=0.8)

        peak_height = amp + (offset if has_ion else 0.0)
        max_y = max(max_y, peak_height)

        if has_ion:
            ion = row["ion"]
            if isinstance(ion, str) and ion.strip():
                ax.text(
                    mu,
                    amp + offset,
                    str(ion),
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=color,
                )

    legend_elements = [
        Line2D(
            [0],
            [0],
            color=cmap(norm(float(p))),
            lw=2,
            label=f"bliss_score = {p:.3f}",
        )
        for p in unique_scores
    ]

    ax.legend(
        handles=legend_elements,
        title="bliss_score",
        loc="upper right",
        framealpha=0.8,
        fontsize=9,
    )

    ax.set_ylim(0, max_y * 1.25)
    ax.set_title(
        "Gaussian components colored by bliss_score"
        + (" with ion labels" if has_ion else "")
    )
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Amplitude")

    fig.tight_layout()

    if show:
        plt.show()
        return None

    return fig, ax