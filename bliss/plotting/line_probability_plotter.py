"""Plot BLiSS candidate probability against line energy."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_line_prob(df, show=True):
    """Plot candidate cluster probability as a function of fitted line energy.

    Parameters
    ----------
    df : pandas.DataFrame
        Candidate table containing ``center`` and ``cluster_probability`` columns.
    show : bool, default: True
        Whether to display the figure immediately with ``plt.show``.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the probability scatter plot.
    """
    x = np.linspace(df['center'].min() - 3 * df['sigma'].max(), df['center'].max() + 3 * df['sigma'].max(), 600)
    unique_probs = np.sort(df['cluster_probability'].unique())[::-1]
    cmap = plt.cm.rainbow
    norm = plt.Normalize(vmin=unique_probs.min(), vmax=unique_probs.max())
    color_map = {p: cmap(norm(p)) for p in unique_probs}
    plt.figure(figsize=(max(df.center) - min(df.center), 5))
    has_ion = 'ion' in df.columns
    max_y = 0
    offset = 0.02 * df['amplitude'].max()
    for _, row in df.iterrows():
        p = row['cluster_probability']
        color = color_map[p]
        mu = row['center']
        sigma = row['sigma']
        A = row['amplitude']
        y = A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        plt.plot(x, y, color=color, alpha=0.8)
        peak_height = A + (offset if has_ion else 0)
        if peak_height > max_y:
            max_y = peak_height
        if has_ion:
            plt.text(mu, A + offset, str(row['ion']), rotation=90, ha='center', va='bottom', fontsize=8, color=color)
    legend_elements = [Line2D([0], [0], color=color_map[p], lw=2, label=f'p = {p:.3f}') for p in unique_probs]
    plt.legend(handles=legend_elements, title='Cluster Probability', loc='upper right', framealpha=0.8, fontsize=9)
    plt.ylim(0, max_y * 1.25)
    plt.title('Gaussian components colored by cluster_probability' + (' with ion labels' if has_ion else ''))
    plt.xlabel('Energy (keV)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    if show:
        plt.show()
    return (plt.gcf(), plt.gca())
