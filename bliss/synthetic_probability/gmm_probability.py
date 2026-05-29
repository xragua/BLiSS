"""Estimate candidate-line reliability by comparing real and synthetic detections."""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='divide by zero encountered in divide')

class GMMLineProbabilityEvaluator:
    """Evaluate candidate reliability with Gaussian-mixture clustering.

    Attributes
    ----------
    k_min, k_max : int
        Minimum and maximum number of Gaussian-mixture components tested.
    covariance_types : tuple of str
        Scikit-learn covariance types considered during BIC model selection.
    """

    def __init__(self, k_min=1, k_max=20, covariance_types=('full',)):
        """Create a Gaussian-mixture probability evaluator.

        Parameters
        ----------
        k_min : int, default: 1
            Minimum number of mixture components tested.
        k_max : int, default: 20
            Maximum number of mixture components tested, limited internally by the
            number of samples.
        covariance_types : tuple of str, default: ("full",)
            Covariance structures passed to ``sklearn.mixture.GaussianMixture``.
        """
        self.k_min = k_min
        self.k_max = k_max
        self.covariance_types = covariance_types

    def evaluate(self, lines, simlines, simx, x, show_plot=False):
        """Evaluate real candidates against synthetic detections.

        Parameters
        ----------
        lines : pandas.DataFrame
            Candidate lines detected in the observed spectrum.
        simlines : pandas.DataFrame
            Candidate lines detected in synthetic spectra.
        simx : array-like
            Coordinate array of the synthetic spectra, used to normalize synthetic line
            rates.
        x : array-like
            Coordinate array of the real spectrum, used to normalize real line rates.
        show_plot : bool, default: False
            Whether to show the BIC curve used for selecting the mixture model.

        Returns
        -------
        pandas.DataFrame
            Real candidate lines with added GMM labels and cluster-probability values.
        """
        return eval_line_probability_gmm(lines, simlines, simx, x, self.k_min, self.k_max, self.covariance_types, show_plot)

def real_probability(real_rate, sim_rate):
    """Convert real and synthetic detection rates into a clipped reliability score.

    Parameters
    ----------
    real_rate : float
        Number or rate of candidate detections in real data.
    sim_rate : float
        Number or rate of candidate detections in synthetic data.

    Returns
    -------
    float
        Probability-like score ``(real_rate - sim_rate) / real_rate`` clipped to
        the interval [0, 1]. Returns 0 when ``real_rate`` is zero.
    """
    if real_rate == 0:
        return 0.0
    return max(0.0, min(1.0, (real_rate - sim_rate) / real_rate))

def eval_line_probability_gmm(lines, simlines, simx, x, k_min=1, k_max=20, covariance_types=('full',), show_plot=False):
    """Assign cluster-based reliability scores to observed candidate lines.

    Parameters
    ----------
    lines : pandas.DataFrame
        Candidate table from the observed spectrum. It must contain ``amplitude``,
        ``sigma``, ``value_on_line``, and ``noise_on_block``.
    simlines : pandas.DataFrame
        Candidate table from synthetic spectra with the same feature columns as
        ``lines``.
    simx : array-like
        Synthetic-spectrum coordinate grid used to compute synthetic candidate
        density.
    x : array-like
        Observed-spectrum coordinate grid used to compute real candidate density.
    k_min : int, default: 1
        Minimum number of GMM components tested.
    k_max : int, default: 20
        Maximum number of GMM components tested before limiting by sample count.
    covariance_types : tuple of str, default: ("full",)
        Covariance structures considered during BIC model selection.
    show_plot : bool, default: False
        Whether to display the BIC diagnostic plot.

    Returns
    -------
    pandas.DataFrame
        Rows corresponding to real candidates only, with ``gmm_label`` and
        ``cluster_probability`` columns added.
    """
    lines['real'] = 1
    simlines['real'] = 0
    lines_sim_real = pd.concat([lines, simlines])
    lines_sim_real['ratio'] = lines_sim_real.sigma / lines_sim_real.amplitude
    lines_sim_real['snr'] = lines_sim_real.value_on_line / lines_sim_real.noise_on_block
    data = lines_sim_real[['amplitude', 'sigma', 'snr', 'ratio']]
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    n_samples = X.shape[0]
    k_max_eff = max(k_min, min(k_max, n_samples - 1))
    if k_max_eff < k_min:
        k_max_eff = k_min
    best_bic = np.inf
    best_model = None
    bics_to_plot, ks_to_plot, covs_to_plot = ([], [], [])
    for cov in covariance_types:
        for k in range(k_min, k_max_eff + 1):
            gmm = GaussianMixture(n_components=k, covariance_type=cov, random_state=0)
            gmm.fit(X)
            bic_val = gmm.bic(X)
            bics_to_plot.append(bic_val)
            ks_to_plot.append(k)
            covs_to_plot.append(cov)
            if bic_val < best_bic:
                best_bic = bic_val
                best_model = gmm
    if show_plot:
        plt.figure(figsize=(6, 4))
        if len(covariance_types) == 1:
            plt.plot(range(k_min, k_max_eff + 1), [b for b, c in zip(bics_to_plot, covs_to_plot) if c == covariance_types[0]], 'o-')
            plt.xlabel('Number of components (k)')
            plt.ylabel('BIC (lower is better)')
            plt.title(f"BIC for covariance_type='{covariance_types[0]}'")
        else:
            for cov in covariance_types:
                ks = [k for k, c in zip(ks_to_plot, covs_to_plot) if c == cov]
                bs = [b for b, c in zip(bics_to_plot, covs_to_plot) if c == cov]
                plt.plot(ks, bs, 'o-', label=cov)
            plt.xlabel('Number of components (k)')
            plt.ylabel('BIC (lower is better)')
            plt.title('BIC across covariance types')
            plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    cluster_labels = best_model.predict(X)
    lines_sim_real['gmm_label'] = cluster_labels
    lines_sim_real = lines_sim_real.reset_index(drop=True)
    cluster_labels = best_model.predict(X)
    lines_sim_real['gmm_label'] = cluster_labels
    lines_sim_real = lines_sim_real.reset_index(drop=True)
    cluster_contains_sim, cluster_contains_real = ([], [])
    for i in range(max(cluster_labels)):
        simnumber = len(simlines)
        realnumber = len(lines)
        sim_group = lines_sim_real[(lines_sim_real.gmm_label == i) & (lines_sim_real.real == 0)]
        real_group = lines_sim_real[(lines_sim_real.gmm_label == i) & (lines_sim_real.real == 1)]
        cluster_contains_sim.append(len(sim_group) / simnumber)
        cluster_contains_real.append(len(real_group) / realnumber)
    idx_desc = np.argsort(cluster_contains_sim)[::-1]
    idx_desc_loop = idx_desc
    filtered_lines_sim_real = lines_sim_real
    real_group = filtered_lines_sim_real[filtered_lines_sim_real.real == 1]
    sim_group = filtered_lines_sim_real[filtered_lines_sim_real.real == 0]
    real_rate = len(real_group)
    sim_rate = len(sim_group) / 10
    cluster_probability = np.round(real_probability(real_rate, sim_rate), 2)
    lines_sim_real['cluster_probability'] = cluster_probability
    lines_real = lines_sim_real[lines_sim_real.real == 1]
    lines_real = lines_real.dropna(axis=1, how='all')
    for j in idx_desc_loop:
        filtered_lines_sim_real = filtered_lines_sim_real[filtered_lines_sim_real.gmm_label != j].reset_index(drop=True)
        real_group = filtered_lines_sim_real[(filtered_lines_sim_real.gmm_label != j) & (filtered_lines_sim_real.real == 1)]
        sim_group = filtered_lines_sim_real[(filtered_lines_sim_real.gmm_label != j) & (filtered_lines_sim_real.real == 0)]
        real_rate = len(real_group) / (max(x) - min(x))
        sim_rate = len(sim_group) / (max(simx) - min(simx))
        lines_sim_real.loc[lines_sim_real.gmm_label == j, 'cluster_probability'] = cluster_probability
        cluster_probability = np.round(real_probability(real_rate, sim_rate), 2)
    return lines_sim_real[lines_sim_real.real == 1].reset_index(drop=True)
