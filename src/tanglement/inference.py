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
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .quantum import ExperimentData, bloch_vector, rho_from_fano
from .dag import BellDAG

__all__ = [
    "PosteriorResult",
    "HypothesisTestResult",
    "BinaryConjugate",
    "BayesianBootstrap",
    "Tomographic",
    "BalkePearl",
    "PhysicalTomographic",
    "FrequentistBellTest",
]


# ─── Common output ────────────────────────────────────────────────────

@dataclass
class PosteriorResult:
    """Unified output from all inference models.

    Attributes
    ----------
    correlations : dict mapping setting index to posterior summary
        Each value has keys: mean, std, ci, and optionally true, covered.
    chsh : dict with keys mean, std, ci, P_violates
        Posterior summary of max|S| across the four CHSH combinations.
    extra : dict of model-specific outputs (always includes chsh_samples)
    idata : ArviZ InferenceData, present only for PyMC-based models
    """
    correlations: Dict[int, Dict]
    chsh: Dict
    extra: Dict
    idata: Optional[object] = None


def _corr_summary(samples: np.ndarray,
                   true: Optional[float] = None) -> Dict:
    """Compute posterior summary statistics from an array of samples."""
    d = {'mean': float(np.mean(samples)), 'std': float(np.std(samples)),
         'ci': (float(np.percentile(samples, 2.5)),
                float(np.percentile(samples, 97.5)))}
    if true is not None:
        d['true'] = float(true)
        d['covered'] = d['ci'][0] <= true <= d['ci'][1]
    return d


def _chsh_from_4corr(c0: np.ndarray, c1: np.ndarray,
                      c2: np.ndarray, c3: np.ndarray
                      ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute CHSH posterior samples from 4 correlation sample arrays."""
    combs = {'S1': c0+c1+c2-c3, 'S2': c0+c1-c2+c3,
             'S3': c0-c1+c2+c3, 'S4': -c0+c1+c2+c3}
    all_abs = np.stack([np.abs(v) for v in combs.values()])
    mx = np.max(all_abs, axis=0)
    return mx, combs


def _chsh_horodecki(C_samples: np.ndarray) -> np.ndarray:
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
        if rng is None:
            rng = np.random.default_rng(123)
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
        if rng is None:
            rng = np.random.default_rng(123)
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
            extra={'chsh_samples': mx},
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

        # Predicted correlations
        c_pred = tc_post @ A.T
        corrs = {}
        for k in range(n_set):
            corrs[k] = _corr_summary(c_pred[:, k], data.true_correlations.get(k))

        # CHSH via Horodecki
        chsh_s = _chsh_horodecki(C_samp)

        # Build Fano matrix statistics from posterior
        T_mean = np.zeros((4, 4))
        T_std = np.zeros((4, 4))
        T_mean[0, 0] = 1.0
        T_mean[1:, 1:] = np.mean(C_samp, axis=0)
        T_std[1:, 1:] = np.std(C_samp, axis=0)
        if obs_a is not None:
            ta_post = idata.posterior['t_a'].values.reshape(-1, 3)
            tb_post = idata.posterior['t_b'].values.reshape(-1, 3)
            T_mean[1:, 0] = np.mean(ta_post, axis=0)
            T_mean[0, 1:] = np.mean(tb_post, axis=0)
            T_std[1:, 0] = np.std(ta_post, axis=0)
            T_std[0, 1:] = np.std(tb_post, axis=0)

        rho_mean = rho_from_fano(T_mean)
        eigvals = np.linalg.eigvalsh(rho_mean)

        return PosteriorResult(
            correlations=corrs,
            chsh=_corr_summary(chsh_s) | {'P_violates': float(np.mean(chsh_s > 2))},
            extra={
                'T_mean': T_mean,
                'T_std': T_std,
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


def _cholesky_to_rho(L_real: np.ndarray, L_imag: np.ndarray) -> np.ndarray:
    """Reconstruct a physical density matrix from Cholesky parameters.

    L is a 4x4 lower-triangular matrix with 10 real and 6 imaginary
    free parameters. rho = L @ L.conj().T / Tr[L @ L.conj().T].
    """
    L = np.zeros((4, 4), dtype=complex)
    # Diagonal (real only): indices 0..3
    L[0, 0] = L_real[0]
    L[1, 1] = L_real[1]
    L[2, 2] = L_real[2]
    L[3, 3] = L_real[3]
    # Below-diagonal: real indices 4..9, imag indices 0..5
    L[1, 0] = L_real[4] + 1j * L_imag[0]
    L[2, 0] = L_real[5] + 1j * L_imag[1]
    L[2, 1] = L_real[6] + 1j * L_imag[2]
    L[3, 0] = L_real[7] + 1j * L_imag[3]
    L[3, 1] = L_real[8] + 1j * L_imag[4]
    L[3, 2] = L_real[9] + 1j * L_imag[5]

    rho = L @ L.conj().T
    tr = np.real(np.trace(rho))
    return rho / tr if tr > 1e-15 else np.eye(4, dtype=complex) / 4


def _rho_to_cholesky_init(rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract Cholesky parameters from a density matrix for initialisation."""
    from .quantum import project_to_physical
    rho_phys = project_to_physical(rho)
    try:
        L = np.linalg.cholesky(rho_phys)
    except np.linalg.LinAlgError:
        L = np.eye(4, dtype=complex) * 0.5

    L_real = np.array([
        np.real(L[0, 0]), np.real(L[1, 1]),
        np.real(L[2, 2]), np.real(L[3, 3]),
        np.real(L[1, 0]), np.real(L[2, 0]),
        np.real(L[2, 1]), np.real(L[3, 0]),
        np.real(L[3, 1]), np.real(L[3, 2]),
    ])
    L_imag = np.array([
        np.imag(L[1, 0]), np.imag(L[2, 0]),
        np.imag(L[2, 1]), np.imag(L[3, 0]),
        np.imag(L[3, 1]), np.imag(L[3, 2]),
    ])
    return L_real, L_imag


# ─── Model 5: Physical tomographic (Fano + physical projection) ─────

class PhysicalTomographic:
    """Full quantum state tomography with physical density matrix reconstruction.

    Delegates Fano-coefficient inference to Tomographic (PyMC NUTS),
    then reconstructs a physical density matrix for each posterior sample
    via eigenvalue projection (clip negatives, renormalise). Computes
    posterior distributions over concurrence, negativity, entanglement
    of formation, purity, and eigenvalues.

    Requires >= 9 tomographic settings ({X,Y,Z} x {X,Y,Z}).

    Parameters
    ----------
    sigma_prior : prior std on Fano coefficients (passed to Tomographic)
    purity_beta : strength of purity soft constraint (passed to Tomographic)
    fit_bloch : whether to fit Bloch vectors from marginals
    """

    def __init__(self, sigma_prior: float = 2.0, purity_beta: float = 50.0,
                 fit_bloch: bool = True):
        self.sigma_prior = sigma_prior
        self.beta = purity_beta
        self.fit_bloch = fit_bloch

    def fit(self, data: ExperimentData,
            n_draws: int = 1500, n_tune: int = 1000,
            n_chains: int = 2, seed: int = 42) -> PosteriorResult:

        from .quantum import (
            fano_decomposition, concurrence, negativity,
            entanglement_of_formation, project_to_physical,
        )

        settings = data.setting_pairs
        n_set = len(settings)
        if n_set < 9:
            raise ValueError(
                f"PhysicalTomographic requires >= 9 settings, got {n_set}")

        # Delegate Fano inference to Tomographic
        tomo = Tomographic(sigma_prior=self.sigma_prior,
                           purity_beta=self.beta,
                           fit_bloch=self.fit_bloch)
        tomo_result = tomo.fit(data, n_draws=n_draws, n_tune=n_tune,
                               n_chains=n_chains, seed=seed)

        # Reconstruct physical ρ for each posterior Fano sample
        idata = tomo_result.idata
        tc_post = idata.posterior['t_corr'].values.reshape(-1, 9)
        ns = tc_post.shape[0]

        has_bloch = 't_a' in idata.posterior
        if has_bloch:
            ta_post = idata.posterior['t_a'].values.reshape(-1, 3)
            tb_post = idata.posterior['t_b'].values.reshape(-1, 3)

        rho_samples = np.zeros((ns, 4, 4), dtype=complex)
        eig_samples = np.zeros((ns, 4))
        conc_samples = np.zeros(ns)
        neg_samples = np.zeros(ns)
        eof_samples = np.zeros(ns)
        purity_samples = np.zeros(ns)

        for s in range(ns):
            T = np.zeros((4, 4))
            T[0, 0] = 1.0
            T[1:, 1:] = tc_post[s].reshape(3, 3)
            if has_bloch:
                T[1:, 0] = ta_post[s]
                T[0, 1:] = tb_post[s]

            rho_s = project_to_physical(rho_from_fano(T))
            rho_samples[s] = rho_s
            eig_samples[s] = np.sort(np.real(np.linalg.eigvalsh(rho_s)))[::-1]
            conc_samples[s] = concurrence(rho_s)
            neg_samples[s] = negativity(rho_s)
            eof_samples[s] = entanglement_of_formation(rho_s)
            purity_samples[s] = float(np.real(np.trace(rho_s @ rho_s)))

        rho_mean = np.mean(rho_samples, axis=0)
        T_mean = np.real(fano_decomposition(rho_mean))

        # CHSH via Horodecki on the raw correlation tensor
        C_samp = tc_post.reshape(-1, 3, 3)
        chsh_s = _chsh_horodecki(C_samp)

        return PosteriorResult(
            correlations=tomo_result.correlations,
            chsh=_corr_summary(chsh_s) | {'P_violates': float(np.mean(chsh_s > 2))},
            extra={
                'rho_mean': rho_mean,
                'rho_samples': rho_samples,
                'T_mean': T_mean,
                'eigenvalues': _corr_summary(eig_samples[:, 0]),
                'eigenvalue_samples': eig_samples,
                'concurrence': _corr_summary(conc_samples),
                'concurrence_samples': conc_samples,
                'negativity': _corr_summary(neg_samples),
                'negativity_samples': neg_samples,
                'eof': _corr_summary(eof_samples),
                'eof_samples': eof_samples,
                'purity': _corr_summary(purity_samples),
                'purity_samples': purity_samples,
                'chsh_samples': chsh_s,
            },
            idata=idata,
        )


# ─── Model 6: Frequentist Bell test ─────────────────────────────────

@dataclass
class HypothesisTestResult:
    """Output from FrequentistBellTest.

    Attributes
    ----------
    test_statistic : observed max|S| across the four CHSH combinations
    p_value : P(T >= t_obs | H0), where H0 is local realism
    reject_h0 : whether p_value < alpha
    alpha : significance level used
    method : 'asymptotic' or 'bootstrap'
    observed_S : dict of all four CHSH combination values
    se_S : standard errors for each combination
    ci : confidence interval for max|S|
    nearest_local : closest point in the local polytope to the observed correlations
    """
    test_statistic: float
    p_value: float
    reject_h0: bool
    alpha: float
    method: str
    observed_S: Dict[str, float]
    se_S: Dict[str, float]
    ci: Tuple[float, float]
    nearest_local: Optional[np.ndarray]


class FrequentistBellTest:
    """Classical hypothesis test for Bell-CHSH violation.

    H₀: The correlation vector lies in the local-realist polytope (|S| ≤ 2).
    H₁: The correlation vector lies outside the local polytope.

    Two methods:
      - 'asymptotic': Gaussian approximation for the CHSH statistic
      - 'bootstrap': Parametric bootstrap under the nearest H₀ point
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def test(self, data: ExperimentData, method: str = 'asymptotic',
             n_bootstrap: int = 10000,
             rng: Optional[np.random.Generator] = None) -> HypothesisTestResult:
        """Run the hypothesis test.

        Parameters
        ----------
        data : Bell experiment data (>= 4 settings required)
        method : 'asymptotic' (Gaussian) or 'bootstrap' (parametric)
        n_bootstrap : number of bootstrap replicates (only used if method='bootstrap')
        rng : numpy random generator

        Returns
        -------
        HypothesisTestResult
        """
        if rng is None:
            rng = np.random.default_rng(42)

        n_set = len(data.setting_pairs)
        if n_set < 4:
            raise ValueError("Need >= 4 settings for CHSH test")

        # Compute sample correlations and variances per setting
        E_hat = np.zeros(4)
        Var_hat = np.zeros(4)
        n_counts = np.zeros(4, dtype=int)
        for k in range(4):
            mask = data.setting_indices == k
            prods = data.outcomes_x[mask] * data.outcomes_y[mask]
            n_k = len(prods)
            n_counts[k] = n_k
            E_hat[k] = np.mean(prods)
            Var_hat[k] = np.var(prods, ddof=1) / n_k

        # All 4 CHSH combinations
        coeffs = np.array([
            [1,  1,  1, -1],   # S1
            [1,  1, -1,  1],   # S2
            [1, -1,  1,  1],   # S3
            [-1, 1,  1,  1],   # S4
        ], dtype=float)
        S_hat = coeffs @ E_hat
        S_var = coeffs**2 @ Var_hat
        S_se = np.sqrt(S_var)

        observed_S = {f'S{i+1}': float(S_hat[i]) for i in range(4)}
        se_S = {f'S{i+1}': float(S_se[i]) for i in range(4)}

        # Test statistic: max|Ŝᵢ|
        best_idx = np.argmax(np.abs(S_hat))
        t_obs = float(np.abs(S_hat[best_idx]))

        # Nearest point in local polytope
        from .polytope import nearest_local_point
        c_local = nearest_local_point(E_hat)

        if method == 'asymptotic':
            p_value, ci = self._asymptotic(S_hat, S_se, t_obs, best_idx)
        elif method == 'bootstrap':
            p_value, ci = self._bootstrap(data, c_local, n_bootstrap, t_obs, rng)
        else:
            raise ValueError(f"method must be 'asymptotic' or 'bootstrap', "
                             f"got '{method}'")

        return HypothesisTestResult(
            test_statistic=t_obs,
            p_value=p_value,
            reject_h0=p_value < self.alpha,
            alpha=self.alpha,
            method=method,
            observed_S=observed_S,
            se_S=se_S,
            ci=ci,
            nearest_local=c_local,
        )

    def _asymptotic(self, S_hat, S_se, t_obs, best_idx):
        """Asymptotic Gaussian p-value.

        Under H₀, the true |S| ≤ 2. The least-favorable null is |S| = 2.
        Z = (|Ŝ| - 2) / σ_S, one-sided p = 1 - Φ(Z).
        """
        from scipy.stats import norm
        se = S_se[best_idx]
        if se < 1e-15:
            p_value = 0.0 if t_obs > 2 else 1.0
        else:
            z = (t_obs - 2.0) / se
            p_value = float(norm.sf(z))
        ci = (float(t_obs - 1.96 * se), float(t_obs + 1.96 * se))
        return p_value, ci

    def _bootstrap(self, data, c_local, n_bootstrap, t_obs, rng):
        """Parametric bootstrap under the nearest local-realist point.

        Simulates data from H₀ (using c_local as the generating
        correlation vector), computes the null distribution of max|S|,
        returns the bootstrap p-value.
        """
        n_per = data.n_per_setting

        boot_stats = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            E_boot = np.zeros(4)
            for k in range(4):
                p_same = (1 + c_local[k]) / 2  # P(outcomes agree)
                n_same = rng.binomial(n_per, np.clip(p_same, 0, 1))
                E_boot[k] = (2 * n_same - n_per) / n_per

            S_boot = np.array([
                E_boot[0] + E_boot[1] + E_boot[2] - E_boot[3],
                E_boot[0] + E_boot[1] - E_boot[2] + E_boot[3],
                E_boot[0] - E_boot[1] + E_boot[2] + E_boot[3],
                -E_boot[0] + E_boot[1] + E_boot[2] + E_boot[3],
            ])
            boot_stats[b] = np.max(np.abs(S_boot))

        p_value = float(np.mean(boot_stats >= t_obs))
        ci = (float(np.percentile(boot_stats, 2.5)),
              float(np.percentile(boot_stats, 97.5)))
        return p_value, ci
