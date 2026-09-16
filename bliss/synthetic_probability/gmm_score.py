"""Estimate candidate-line reliability by comparing real and synthetic detections."""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from ..line_search.fit_quality import annotate_fit_quality
from ..line_search.gaussian_models import gaussian_area_error
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='divide by zero encountered in divide')

class GMMBlissScoreEvaluator:
    """Evaluate candidate reliability with Gaussian-mixture clustering.

    Attributes
    ----------
    k_min, k_max : int
        Minimum and maximum number of Gaussian-mixture components tested.
    covariance_types : tuple of str
        Scikit-learn covariance types considered during BIC model selection.
    """

    def __init__(self, k_min=1, k_max=20, covariance_types=('full',),
                 min_area_snr=None):
        """Create a Gaussian-mixture score evaluator.

        Parameters
        ----------
        k_min : int, default: 1
            Minimum number of mixture components tested.
        k_max : int, default: 20
            Maximum number of mixture components tested, limited internally by the
            number of samples.
        covariance_types : tuple of str, default: ("full",)
            Covariance structures passed to ``sklearn.mixture.GaussianMixture``.
        min_area_snr : float or None, default: None
            Require covariance-aware area S/N strictly above this value in both
            populations when specified. None disables this optional cut.
        """
        self.k_min = k_min
        self.k_max = k_max
        self.covariance_types = covariance_types
        self.min_area_snr = min_area_snr

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
            Real candidate lines with added GMM labels and bliss_score values.
        """
        return eval_bliss_score_gmm(
            lines, simlines, simx, x, self.k_min, self.k_max,
            self.covariance_types, show_plot, min_area_snr=self.min_area_snr)

def calculate_bliss_score(real_rate, sim_rate):
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
        Empirical score ``(real_rate - sim_rate) / real_rate`` clipped to
        the interval [0, 1]. Returns 0 when ``real_rate`` is zero.
    """
    if real_rate == 0:
        return 0.0
    return max(0.0, min(1.0, (real_rate - sim_rate) / real_rate))

def _eval_bliss_score_gmm_valid(lines, simlines, simx, x, k_min=1, k_max=20, covariance_types=('full',), show_plot=False, n_sim=1):
    """Assign cluster-based reliability scores to observed candidate lines.

    Parameters
    ----------
    lines : pandas.DataFrame
        Candidate table from the observed spectrum. It must contain ``amplitude``,
        ``sigma``, ``eamplitude``, ``esigma``, ``relative_power``, and
        ``noise_on_block``. Optional ``response_sigma`` gives the instrumental
        sigma at the centroid, in the same units as the fitted ``sigma``.
        The feature space uses peak S/N, Gaussian area, relative power, and
        ``sigma / response_sigma`` when instrumental sigmas are finite and
        positive for every observed and synthetic candidate. Otherwise the
        width feature is omitted for the entire comparison. Area-error diagnostics
        use ``cov_amplitude_sigma``; if missing, those diagnostics remain NaN.
        Area errors and area S/N are not GMM features. The public wrapper
        applies its optional area S/N preselection before calling this function.
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
    n_sim : int, default: 1
        Number of synthetic realizations concatenated in ``simlines``; the
        synthetic detection rate is normalised per realization.

    Returns
    -------
    pandas.DataFrame
        Rows corresponding to real candidates only, with ``gmm_label`` and
        ``bliss_score`` columns added.
    """

    lines = lines.copy()
    simlines = simlines.copy()

    if len(lines) == 0:
        lines["gmm_label"] = []
        lines["bliss_score"] = []
        return lines.reset_index(drop=True)

    if len(simlines) == 0:
        lines["gmm_label"] = np.nan
        lines["bliss_score"] = 1.0
        return lines.reset_index(drop=True)

    lines['real'] = 1
    simlines['real'] = 0
    lines_sim_real = pd.concat([lines, simlines])
    eps = 1e-12
    k = np.sqrt(2.0 * np.pi)

    lines_sim_real['peak_snr'] = (
        lines_sim_real['amplitude'] /
        (lines_sim_real['noise_on_block'] + eps)
    )
    feature_columns = ['peak_snr', 'area', 'relative_power']
    if 'response_sigma' in lines_sim_real.columns:
        instrumental_sigma = pd.to_numeric(
            lines_sim_real['response_sigma'], errors='coerce'
        ).to_numpy(dtype=float)
        if np.all(np.isfinite(instrumental_sigma) & (instrumental_sigma > 0)):
            lines_sim_real['width_ratio'] = lines_sim_real['sigma'] / instrumental_sigma
            feature_columns.insert(1, 'width_ratio')
    lines_sim_real['area'] = (
        lines_sim_real['amplitude'] *
        lines_sim_real['sigma'] *
        k
    )
    lines_sim_real['earea'] = gaussian_area_error(
        lines_sim_real['amplitude'], lines_sim_real['sigma'],
        lines_sim_real['eamplitude'], lines_sim_real['esigma'],
        lines_sim_real.get('cov_amplitude_sigma', np.nan),
        sigma_fixed=lines_sim_real.get('sigma_fixed', False),
    )
    lines_sim_real['area_snr'] = np.where(
        lines_sim_real['earea'] > 0,
        lines_sim_real['area'] / lines_sim_real['earea'], np.nan,
    )
    # Area uncertainty is diagnostic, not a GMM feature. Keep unavailable
    # errors/S/N as NaN instead of converting them to artificial exact zeros.
    for col in ['peak_snr', 'area'] + (
        ['width_ratio'] if 'width_ratio' in feature_columns else []
    ):
        lines_sim_real[col] = np.nan_to_num(
            lines_sim_real[col],
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )
    data = lines_sim_real[feature_columns]
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
    lines_sim_real["gmm_label"] = cluster_labels
    lines_sim_real = lines_sim_real.reset_index(drop=True)
    cluster_ids = np.unique(cluster_labels)
    cluster_contains_sim, cluster_contains_real = [], []
    simnumber = len(simlines)
    realnumber = len(lines)
    for i in cluster_ids:
        sim_group = lines_sim_real[
            (lines_sim_real.gmm_label == i) & (lines_sim_real.real == 0)
        ]
        real_group = lines_sim_real[
            (lines_sim_real.gmm_label == i) & (lines_sim_real.real == 1)
        ]
        cluster_contains_sim.append(len(sim_group) / simnumber if simnumber > 0 else 0.0)
        cluster_contains_real.append(len(real_group) / realnumber if realnumber > 0 else 0.0)
    idx_desc_loop = cluster_ids[np.argsort(cluster_contains_sim)[::-1]]
    filtered_lines_sim_real = lines_sim_real
    real_group = filtered_lines_sim_real[filtered_lines_sim_real.real == 1]
    sim_group = filtered_lines_sim_real[filtered_lines_sim_real.real == 0]
    real_rate = len(real_group) / (max(x) - min(x))
    sim_rate = len(sim_group) / ((max(simx) - min(simx)) * n_sim)
    bliss_score = np.round(calculate_bliss_score(real_rate, sim_rate), 2)
    lines_sim_real['bliss_score'] = bliss_score
    lines_real = lines_sim_real[lines_sim_real.real == 1]
    lines_real = lines_real.dropna(axis=1, how='all')
    for j in idx_desc_loop:
        filtered_lines_sim_real = filtered_lines_sim_real[filtered_lines_sim_real.gmm_label != j].reset_index(drop=True)
        real_group = filtered_lines_sim_real[(filtered_lines_sim_real.gmm_label != j) & (filtered_lines_sim_real.real == 1)]
        sim_group = filtered_lines_sim_real[(filtered_lines_sim_real.gmm_label != j) & (filtered_lines_sim_real.real == 0)]
        real_rate = len(real_group) / (max(x) - min(x))
        sim_rate = len(sim_group) / ((max(simx) - min(simx)) * n_sim)
        lines_sim_real.loc[lines_sim_real.gmm_label == j, 'bliss_score'] = bliss_score
        bliss_score = np.round(calculate_bliss_score(real_rate, sim_rate), 2)
    return lines_sim_real[lines_sim_real.real == 1].reset_index(drop=True)


def eval_bliss_score_gmm(lines, simlines, simx, x, k_min=1, k_max=20,
                              covariance_types=('full',), show_plot=False, n_sim=1,
                              min_area_snr=None):
    """Score evaluable observed/null fits, preserving every observed row.

    The GMM variables and score formula are unchanged. Failed fits, invalid
    parameters and unusable formal errors are excluded symmetrically before
    scaling, clustering and candidate-rate calculations. By default no area
    S/N cut is applied (``min_area_snr=None``). An explicit nonnegative finite
    threshold enables optional preselection; missing/nonpositive area errors
    cannot pass an enabled cut.

    Excluded rows retain their fit-quality metadata and NaN scores. Their
    ``bliss_score_status`` distinguishes ``low_area_snr`` from ``invalid_area_error``
    and ``invalid_fit``. No eligible null candidates leaves scores unavailable
    rather than 1. The null rate remains normalized by the original ``n_sim``.
    See ``_eval_bliss_score_gmm_valid`` for the feature/parameter details.
    """
    if min_area_snr is not None:
        min_area_snr = float(min_area_snr)
        if not np.isfinite(min_area_snr) or min_area_snr < 0:
            raise ValueError('min_area_snr must be nonnegative and finite, or None.')
    observed = annotate_fit_quality(lines).reset_index(drop=True)
    synthetic = annotate_fit_quality(simlines).reset_index(drop=True)
    # Features that cannot be computed must not become artificial zero values.
    for table in [observed, synthetic]:
        for col in ['noise_on_block', 'relative_power']:
            v = (pd.to_numeric(table[col], errors='coerce') if col in table
                 else pd.Series(np.nan, index=table.index))
            bad = ~np.isfinite(v) | ((v <= 0) if col == 'noise_on_block' else False)
            table.loc[bad, 'fit_evaluable'] = False
            table.loc[bad, 'fit_status'] = 'not_evaluable'
            table.loc[bad, 'fit_reasons'] = table.loc[bad, 'fit_reasons'].map(
                lambda reason: ';'.join(filter(None, [reason, 'invalid_' + col])))
        table['bliss_score_status'] = np.where(table['fit_evaluable'], 'eligible', 'invalid_fit')
        if min_area_snr is not None:
            # Recompute from fitted parameters/covariance; never trust stale
            # area diagnostics or silently assume missing covariance is zero.
            params = table.reindex(columns=[
                'amplitude', 'sigma', 'eamplitude', 'esigma', 'cov_amplitude_sigma',
            ]).apply(pd.to_numeric, errors='coerce')
            area = (np.sqrt(2 * np.pi) * params.amplitude * params.sigma).to_numpy()
            error = gaussian_area_error(
                *[params[col] for col in params.columns],
                sigma_fixed=table.get('sigma_fixed', False))
            usable = np.isfinite(area) & np.isfinite(error) & (error > 0)
            snr = np.divide(area, error, out=np.full(len(table), np.nan), where=usable)
            table['area'], table['earea'], table['area_snr'] = area, error, snr
            eligible_fit = table['fit_evaluable'].to_numpy()
            table.loc[eligible_fit & ~usable, 'bliss_score_status'] = 'invalid_area_error'
            table.loc[eligible_fit & usable & (snr <= min_area_snr),
                      'bliss_score_status'] = 'low_area_snr'
    observed['gmm_label'] = np.nan
    observed['bliss_score'] = np.nan
    good = observed['bliss_score_status'].eq('eligible')
    valid_null = synthetic[synthetic['bliss_score_status'].eq('eligible')].copy()
    if not good.any():
        return observed
    if valid_null.empty:
        observed.loc[good, 'bliss_score_status'] = 'no_valid_null_candidates'
        return observed
    scored = _eval_bliss_score_gmm_valid(
        observed[good].copy(), valid_null, simx, x,
        k_min=k_min, k_max=k_max, covariance_types=covariance_types,
        show_plot=show_plot, n_sim=n_sim)
    scored.index = observed.index[good]
    scored['bliss_score_status'] = 'evaluated'
    return pd.concat([scored, observed[~good]]).sort_index().reset_index(drop=True)
