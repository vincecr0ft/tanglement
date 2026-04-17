"""
Bayesian inference models for Bell-CHSH experiments.

Four models, all returning PosteriorResult:

1. BinaryConjugate     — Dirichlet-Multinomial, exact (no MCMC)
2. BayesianBootstrap    — Nonparametric, any outcome type
3. Tomographic          — PyMC NUTS, 9 or 15 Fano coefficients
4. BalkePearl           — PyMC NUTS over 16 response-type probabilities
"""

import numpy as np
import pymc as pm
import arviz as az
from dataclasses import dataclass
from typing import Dict, Optional
import warnings

from .quantum import ExperimentData, bloch_vector, rho_from_fano
from .dag import BellDAG


# ─── Common output ────────────────────────────────────────────────────

@dataclass
class PosteriorResult:
    """Unified output from all inference models."""
    correlations: Dict[int, Dict]       # k → {mean, std, ci, true}
    chsh: Dict                          # {mean, std, ci, P_violates}
    extra: Dict                         # model-specific extras
    idata: Optional[object] = None      # ArviZ InferenceData (PyMC models)


def _corr_summary(samples, true=None):
    d = {'mean': float(np.mean(samples)), 'std': float(np.std(samples)),
         'ci': (float(np.percentile(samples, 2.5)),
                float(np.percentile(samples, 97.5)))}
    if true is not None:
        d['true'] = float(true)
        d['covered'] = d['ci'][0] <= true <= d['ci'][1]
    return d


def _chsh_from_4corr(c0, c1, c2, c3):
    """Compute CHSH posterior samples from 4 correlation sample arrays."""
    combs = {'S1': c0+c1+c2-c3, 'S2': c0+c1-c2+c3,
             'S3': c0-c1+c2+c3, 'S4': -c0+c1+c2+c3}
    all_abs = np.stack([np.abs(v) for v in combs.values()])
    mx = np.max(all_abs, axis=0)
    return mx, combs


def _chsh_horodecki(C_samples):
    """Horodecki S_max from (n, 3, 3) correlation tensor samples."""
    n = C_samples.shape[0]
    s = np.zeros(n)
    for i in range(n):
        eigs = np.sort(np.linalg.eigvalsh(C_samples[i].T @ C_samples[i]))[::-1]
        s[i] = 2 * np.sqrt(max(eigs[0], 0) + max(eigs[1], 0))
    return s


# ─── Model 1: Binary conjugate ───────────────────────────────────────

class BinaryConjugate:
    """
    Exact Dirichlet-Multinomial posterior for binary ±1 outcomes.
    No MCMC — posterior samples drawn directly from Dirichlet.
    """

    def __init__(self, alpha=None):
        self.alpha = np.ones(4) if alpha is None else np.asarray(alpha)

    def fit(self, data: ExperimentData, n_samples=20000, rng=None):
        if rng is None: rng = np.random.default_rng(123)
        omap = {(1,1):0, (1,-1):1, (-1,1):2, (-1,-1):3}
        n_set = len(data.setting_pairs)
        csamp = {}

        for k in range(n_set):
            mask = data.setting_indices == k
            counts = np.zeros(4)
            for x, y in zip(data.outcomes_x[mask], data.outcomes_y[mask]):
                counts[omap[(int(x), int(y))]] += 1
            p = rng.dirichlet(self.alpha + counts, size=n_samples)
            csamp[k] = p[:,0] + p[:,3] - p[:,1] - p[:,2]

        corrs = {k: _corr_summary(csamp[k], data.true_correlations.get(k))
                 for k in csamp}

        if n_set >= 4:
            mx, combs = _chsh_from_4corr(csamp[0], csamp[1], csamp[2], csamp[3])
        else:
            mx = np.zeros(n_samples)

        return PosteriorResult(
            correlations=corrs,
            chsh=_corr_summary(mx) | {'P_violates': float(np.mean(mx > 2))},
            extra={'chsh_samples': mx},
        )


# ─── Model 2: Bayesian bootstrap ─────────────────────────────────────

class BayesianBootstrap:
    """
    Nonparametric posterior via Dirichlet-weighted means.
    Correct for ratio estimators (avoids Fieller problem).
    Works for any outcome type in [-1, +1].
    """

    def fit(self, data: ExperimentData, n_samples=15000, rng=None):
        if rng is None: rng = np.random.default_rng(123)
        n_set = len(data.setting_pairs)
        csamp = {}

        for k in range(n_set):
            mask = data.setting_indices == k
            products = data.outcomes_x[mask] * data.outcomes_y[mask]
            n = len(products)
            # Chunked to limit memory
            chunk = min(2000, n_samples)
            samples = []
            for start in range(0, n_samples, chunk):
                batch = min(chunk, n_samples - start)
                w = rng.dirichlet(np.ones(n), size=batch)
                samples.append(w @ products)
            csamp[k] = np.concatenate(samples)

        corrs = {k: _corr_summary(csamp[k], data.true_correlations.get(k))
                 for k in csamp}

        if n_set >= 4:
            mx, _ = _chsh_from_4corr(csamp[0], csamp[1], csamp[2], csamp[3])
        else:
            mx = np.zeros(n_samples)

        return PosteriorResult(
            correlations=corrs,
            chsh=_corr_summary(mx) | {'P_violates': float(np.mean(mx > 2))},
            extra={},
        )


# ─── Model 3: Tomographic (PyMC NUTS) ────────────────────────────────

class Tomographic:
    """
    PyMC NUTS over Fano coefficients from tomographic data.
    
    9 settings → 9 correlation tensor elements (+ Bloch from marginals)
    Purity constraint as soft potential.
    """

    def __init__(self, sigma_prior=2.0, purity_beta=50.0, fit_bloch=True):
        self.sigma_prior = sigma_prior
        self.beta = purity_beta
        self.fit_bloch = fit_bloch

    def _design_matrix(self, settings):
        A = np.zeros((len(settings), 9))
        for k, (θa, φa, θb, φb) in enumerate(settings):
            A[k] = np.outer(bloch_vector(θa, φa), bloch_vector(θb, φb)).ravel()
        return A

    def fit(self, data: ExperimentData,
            n_draws=1500, n_tune=1000, n_chains=2, seed=42):

        settings = data.setting_pairs
        n_set = len(settings)
        A = self._design_matrix(settings)

        # Observed correlations and SEs
        obs_c, obs_se = np.zeros(n_set), np.zeros(n_set)
        obs_ax, obs_bx = {}, {}

        for k in range(n_set):
            mask = data.setting_indices == k
            xs, ys = data.outcomes_x[mask], data.outcomes_y[mask]
            prods = xs * ys
            obs_c[k] = np.mean(prods)
            obs_se[k] = np.std(prods, ddof=1) / np.sqrt(len(prods))

        se_floor = 1.0 / np.sqrt(data.n_per_setting)
        obs_se = np.maximum(obs_se, se_floor)

        # Bloch vectors from marginals (if 9 settings in 3×3 grid)
        obs_a, se_a, obs_b, se_b = None, None, None, None
        if self.fit_bloch and n_set == 9:
            obs_a, se_a = np.zeros(3), np.zeros(3)
            obs_b, se_b = np.zeros(3), np.zeros(3)
            for i in range(3):
                ax = np.concatenate([data.outcomes_x[data.setting_indices == i*3+j]
                                     for j in range(3)])
                obs_a[i] = np.mean(ax)
                se_a[i] = max(np.std(ax, ddof=1)/np.sqrt(len(ax)), se_floor/np.sqrt(3))
                bx = np.concatenate([data.outcomes_y[data.setting_indices == j*3+i]
                                     for j in range(3)])
                obs_b[i] = np.mean(bx)
                se_b[i] = max(np.std(bx, ddof=1)/np.sqrt(len(bx)), se_floor/np.sqrt(3))

        # Init from least squares
        t_init = np.linalg.pinv(A) @ obs_c

        with pm.Model():
            tc = pm.Normal('t_corr', 0, self.sigma_prior, shape=9)
            pm.Normal('obs_c', mu=pm.math.dot(A, tc), sigma=obs_se,
                      observed=obs_c, shape=n_set)

            if obs_a is not None:
                ta = pm.Normal('t_a', 0, self.sigma_prior, shape=3)
                tb = pm.Normal('t_b', 0, self.sigma_prior, shape=3)
                pm.Normal('obs_a', mu=ta, sigma=se_a, observed=obs_a, shape=3)
                pm.Normal('obs_b', mu=tb, sigma=se_b, observed=obs_b, shape=3)
                purity = (1 + pm.math.sum(ta**2) + pm.math.sum(tb**2) +
                          pm.math.sum(tc**2)) / 4
            else:
                purity = (1 + pm.math.sum(tc**2)) / 4

            if self.beta > 0:
                pm.Potential('purity', pm.math.switch(
                    purity > 1, -self.beta * (purity - 1)**2, 0.0))

            initvals = {'t_corr': t_init}
            if obs_a is not None:
                initvals['t_a'] = obs_a
                initvals['t_b'] = obs_b

            idata = pm.sample(draws=n_draws, tune=n_tune, chains=n_chains,
                              random_seed=seed, progressbar=False,
                              return_inferencedata=True,
                              initvals=initvals, init='adapt_diag')

        # Extract
        tc_post = idata.posterior['t_corr'].values.reshape(-1, 9)
        C_samp = tc_post.reshape(-1, 3, 3)
        ns = tc_post.shape[0]

        # Predicted correlations
        c_pred = tc_post @ A.T
        corrs = {}
        for k in range(n_set):
            corrs[k] = _corr_summary(c_pred[:, k], data.true_correlations.get(k))

        # CHSH via Horodecki
        chsh_s = _chsh_horodecki(C_samp)

        # Build Fano matrix from posterior mean
        T_mean = np.zeros((4, 4))
        T_mean[0, 0] = 1.0
        T_mean[1:, 1:] = np.mean(C_samp, axis=0)
        if obs_a is not None:
            ta_post = idata.posterior['t_a'].values.reshape(-1, 3)
            tb_post = idata.posterior['t_b'].values.reshape(-1, 3)
            T_mean[1:, 0] = np.mean(ta_post, axis=0)
            T_mean[0, 1:] = np.mean(tb_post, axis=0)

        rho_mean = rho_from_fano(T_mean)
        eigvals = np.linalg.eigvalsh(rho_mean)

        return PosteriorResult(
            correlations=corrs,
            chsh=_corr_summary(chsh_s) | {'P_violates': float(np.mean(chsh_s > 2))},
            extra={
                'T_mean': T_mean,
                'T_std': np.zeros_like(T_mean),  # filled below
                'C_mean': np.mean(C_samp, axis=0),
                'C_std': np.std(C_samp, axis=0),
                'rho_eigenvalues': eigvals,
                'purity': float(np.real(np.trace(rho_mean @ rho_mean))),
                'chsh_samples': chsh_s,
            },
            idata=idata,
        )


# ─── Model 4: Balke-Pearl (PyMC NUTS) ────────────────────────────────

class BalkePearl:
    """
    Bayesian inference over the 16-vertex response-function polytope.
    
    Following Polson et al. (2603.28973) Section 8.3:
    q ~ Dirichlet(α) on 16 response types, likelihood from observed counts.
    
    Uses PyMC with a Dirichlet prior and custom potential for the
    multinomial likelihood, avoiding the IS degeneracy of the v2 code.
    """

    def __init__(self, alpha=None):
        self.alpha = np.ones(16) if alpha is None else np.asarray(alpha)

    def fit(self, data: ExperimentData,
            n_draws=2000, n_tune=1000, n_chains=2, seed=42):

        dag = BellDAG()
        V = dag.vertex_matrix()            # (16, 4) correlations
        R = dag.vertex_outcome_indicators() # (16, 4, 4) outcome indicators

        # Observed counts
        omap = {(1,1):0, (1,-1):1, (-1,1):2, (-1,-1):3}
        obs_counts = np.zeros((4, 4))
        for sp in range(4):
            mask = data.setting_indices == sp
            for x, y in zip(data.outcomes_x[mask], data.outcomes_y[mask]):
                obs_counts[sp, omap[(int(x), int(y))]] += 1

        with pm.Model():
            # Dirichlet prior on response-type probabilities
            q = pm.Dirichlet('q', a=self.alpha, shape=16)

            # For each setting pair, outcome probabilities = R^T @ q
            for sp in range(4):
                p_sp = pm.math.dot(R[:, sp, :].T, q)  # (4,)
                pm.Multinomial(f'counts_{sp}', n=int(obs_counts[sp].sum()),
                               p=p_sp, observed=obs_counts[sp].astype(int))

            idata = pm.sample(draws=n_draws, tune=n_tune, chains=n_chains,
                              random_seed=seed, progressbar=False,
                              return_inferencedata=True, init='adapt_diag')

        q_post = idata.posterior['q'].values.reshape(-1, 16)
        ns = q_post.shape[0]

        # Correlations from q
        corr_post = q_post @ V  # (ns, 4)
        corrs = {}
        for k in range(4):
            corrs[k] = _corr_summary(corr_post[:, k], data.true_correlations.get(k))

        # CHSH
        mx, combs = _chsh_from_4corr(
            corr_post[:,0], corr_post[:,1], corr_post[:,2], corr_post[:,3])

        # Sparsity: how many types carry weight
        q_mean = np.mean(q_post, axis=0)
        active = int(np.sum(q_mean > 0.01))

        # CHSH per vertex (for interpretation)
        chsh_per_v = V[:,0] + V[:,1] + V[:,2] - V[:,3]

        return PosteriorResult(
            correlations=corrs,
            chsh=_corr_summary(mx) | {'P_violates': float(np.mean(mx > 2))},
            extra={
                'q_mean': q_mean,
                'active_types': active,
                'chsh_per_vertex': chsh_per_v,
                'chsh_samples': mx,
            },
            idata=idata,
        )
