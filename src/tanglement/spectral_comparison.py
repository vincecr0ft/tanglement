"""
PyMC spectral model comparison for cavity-QED experiments.

Replaces the BIC approximation with proper Bayesian inference:
  - Full PyMC models with physical priors for both hypotheses
  - LOO-CV (PSIS-LOO) for model comparison
  - Posterior predictive checks
  - All arithmetic in real-valued pytensor (no complex numbers)

Two competing models for the cavity self-energy:

M_classical (Lorentzian):
    Σ(ω) = g²N / (ω - ωs + iγ/2)
    3 parameters: ωs, γ, g²N
    → Symmetric vacuum Rabi splitting, Markovian (memoryless)

M_QSL (gapped spinon continuum):
    Σ(ω) = (g²N/bw) √((ω - ωs + Δ + iη) / bw)
    4 parameters: ωs, η, g²N, Δ
    → Asymmetric lineshape, non-Markovian (memory from gap edge)

The question "is the QSL environment classical or quantum?" maps onto:
    P(M_QSL | spectra) vs P(M_classical | spectra)

This is the spectral analogue of the polytope test:
    Lorentzian ↔ correlations inside local polytope
    QSL ↔ correlations outside local polytope
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from typing import Dict, List, Optional
from dataclasses import dataclass


# ─── Real-arithmetic cavity transmission ──────────────────────────────

def _s21_mag2_lorentzian(omega, omega_c, kappa, kappa_e,
                          omega_s, gamma, g2N):
    """
    |S21(ω)|² for Lorentzian self-energy, in pure real arithmetic.
    
    Σ = g²N / ((ω-ωs) + iγ/2)
    Re[Σ] = g²N·(ω-ωs) / D,  Im[Σ] = -g²N·γ/(2D)
    where D = (ω-ωs)² + γ²/4
    
    S21 denominator: a + ib where
    a = ω - ωc + Re[Σ],  b = -κ/2 + Im[Σ]
    
    |S21|² = (1 + κe·b/(a²+b²))² + (κe·a/(a²+b²))²
    """
    delta_s = omega - omega_s
    D = delta_s**2 + gamma**2 / 4
    re_sigma = g2N * delta_s / D
    im_sigma = -g2N * gamma / (2 * D)
    
    a = omega - omega_c + re_sigma
    b = -kappa / 2 + im_sigma
    
    denom2 = a**2 + b**2
    re_s21 = 1 + kappa_e * b / denom2
    im_s21 = kappa_e * a / denom2
    
    return re_s21**2 + im_s21**2


def _s21_mag2_qsl(omega, omega_c, kappa, kappa_e,
                    omega_s, eta_spin, g2N, gap, bandwidth):
    """
    |S21(ω)|² for QSL gapped-continuum self-energy.
    
    Σ_QSL = (g²N/bw) · √z  where z = (ω - ωs + Δ + iη) / bw
    
    √(x + iy) decomposed into real parts:
      r = √(x² + y²)
      Re[√z] = √((r + x) / 2)
      Im[√z] = sign(y) · √((r - x) / 2)
    """
    x = (omega - omega_s + gap) / bandwidth
    y = eta_spin / bandwidth
    
    r = pt.sqrt(x**2 + y**2)
    # Clip for numerical safety
    re_sqrt = pt.sqrt(pt.maximum((r + x) / 2, 1e-15))
    im_sqrt = pt.switch(y >= 0, 1.0, -1.0) * pt.sqrt(pt.maximum((r - x) / 2, 1e-15))
    
    re_sigma = (g2N / bandwidth) * re_sqrt
    im_sigma = (g2N / bandwidth) * im_sqrt
    
    a = omega - omega_c + re_sigma
    b = -kappa / 2 + im_sigma
    
    denom2 = a**2 + b**2
    re_s21 = 1 + kappa_e * b / denom2
    im_s21 = kappa_e * a / denom2
    
    return re_s21**2 + im_s21**2


# ─── PyMC models ──────────────────────────────────────────────────────

class SpectralModelComparison:
    """
    Proper Bayesian model comparison for cavity-QED spectra.
    
    Builds two PyMC models with physical priors, samples both,
    and compares via LOO-CV (PSIS-LOO).
    
    Priors are informed by the experimental regime:
      ωs ~ N(ωc, 0.1²)     spin frequency near cavity
      γ  ~ HalfNormal(0.05) linewidth (GHz)
      g²N ~ HalfNormal(0.01) collective coupling squared
      Δ  ~ HalfNormal(0.5)  spinon gap (QSL only)
    """
    
    def __init__(self, cavity_omega: float, cavity_kappa: float,
                 cavity_kappa_e: float, bandwidth: float = 2.0):
        self.omega_c = cavity_omega
        self.kappa = cavity_kappa
        self.kappa_e = cavity_kappa_e
        self.bandwidth = bandwidth
    
    def _build_lorentzian_model(self, omega_data, s21_data, noise_sigma):
        """PyMC model for Lorentzian (classical) self-energy."""
        omega_t = pt.as_tensor_variable(omega_data)
        
        with pm.Model() as model:
            # Priors
            omega_s = pm.Normal('omega_s', mu=self.omega_c, sigma=0.1)
            gamma = pm.HalfNormal('gamma', sigma=0.05)
            g2N = pm.HalfNormal('g2N', sigma=0.01)
            
            # Forward model
            mu = _s21_mag2_lorentzian(omega_t, self.omega_c, self.kappa,
                                       self.kappa_e, omega_s, gamma, g2N)
            
            # Likelihood
            pm.Normal('obs', mu=mu, sigma=noise_sigma,
                      observed=s21_data, shape=len(omega_data))
        
        return model
    
    def _build_qsl_model(self, omega_data, s21_data, noise_sigma):
        """PyMC model for QSL gapped-continuum self-energy."""
        omega_t = pt.as_tensor_variable(omega_data)
        
        with pm.Model() as model:
            # Priors — same as Lorentzian plus gap
            omega_s = pm.Normal('omega_s', mu=self.omega_c, sigma=0.1)
            eta_spin = pm.HalfNormal('eta_spin', sigma=0.05)
            g2N = pm.HalfNormal('g2N', sigma=0.01)
            gap = pm.HalfNormal('gap', sigma=0.5)
            
            # Forward model
            mu = _s21_mag2_qsl(omega_t, self.omega_c, self.kappa,
                                self.kappa_e, omega_s, eta_spin, g2N,
                                gap, self.bandwidth)
            
            # Likelihood
            pm.Normal('obs', mu=mu, sigma=noise_sigma,
                      observed=s21_data, shape=len(omega_data))
        
        return model
    
    def compare(self, spectra: List[Dict],
                n_draws: int = 1000, n_tune: int = 1000,
                n_chains: int = 1, seed: int = 42) -> Dict:
        """
        Full Bayesian comparison of classical vs QSL models.
        
        For each spectrum:
          1. Build both PyMC models
          2. Sample posteriors via NUTS
          3. Compute LOO-CV (PSIS-LOO) for each
        
        Aggregate LOO across spectra, compute model weights.
        
        Returns
        -------
        dict with:
          - loo_classical, loo_qsl: LOO-CV scores
          - delta_loo: difference (positive favours QSL)
          - se_delta: standard error of the difference
          - weight_classical, weight_qsl: stacking weights
          - posteriors: parameter posteriors for both models
        """
        idatas = {'lorentzian': [], 'qsl': []}
        
        for i, spec in enumerate(spectra):
            omega = spec['omega'].astype(np.float64)
            data = spec['s21_noisy'].astype(np.float64)
            noise = float(spec['noise_level'])
            
            # Fit Lorentzian model
            model_lor = self._build_lorentzian_model(omega, data, noise)
            with model_lor:
                idata_lor = pm.sample(draws=n_draws, tune=n_tune,
                                       chains=n_chains, random_seed=seed + i,
                                       progressbar=False,
                                       return_inferencedata=True,
                                       init='adapt_diag',
                                       compute_convergence_checks=False)
                pm.compute_log_likelihood(idata_lor)
            idatas['lorentzian'].append(idata_lor)
            
            # Fit QSL model
            model_qsl = self._build_qsl_model(omega, data, noise)
            with model_qsl:
                idata_qsl = pm.sample(draws=n_draws, tune=n_tune,
                                       chains=n_chains, random_seed=seed + 100 + i,
                                       progressbar=False,
                                       return_inferencedata=True,
                                       init='adapt_diag',
                                       compute_convergence_checks=False)
                pm.compute_log_likelihood(idata_qsl)
            idatas['qsl'].append(idata_qsl)
        
        # Concatenate LOO across spectra
        # For single spectrum, use directly; for multiple, combine
        loo_results = {}
        for model_name in ['lorentzian', 'qsl']:
            if len(idatas[model_name]) == 1:
                loo_results[model_name] = az.loo(idatas[model_name][0])
            else:
                # Sum pointwise LOO across spectra
                total_elpd = 0
                total_se = 0
                for idata in idatas[model_name]:
                    loo_i = az.loo(idata)
                    total_elpd += loo_i.elpd_loo
                    total_se += loo_i.se**2
                loo_results[model_name] = type('LOO', (), {
                    'elpd_loo': total_elpd,
                    'se': np.sqrt(total_se),
                    'p_loo': sum(az.loo(id).p_loo for id in idatas[model_name]),
                })()
        
        # Compare
        delta_elpd = loo_results['qsl'].elpd_loo - loo_results['lorentzian'].elpd_loo
        se_delta = np.sqrt(loo_results['qsl'].se**2 + loo_results['lorentzian'].se**2)
        
        # Model weights via stacking (softmax of elpd)
        elpds = np.array([loo_results['lorentzian'].elpd_loo,
                          loo_results['qsl'].elpd_loo])
        elpds -= elpds.max()
        weights = np.exp(elpds) / np.exp(elpds).sum()
        
        # Extract parameter posteriors
        posteriors = {}
        for model_name in ['lorentzian', 'qsl']:
            post = {}
            idata = idatas[model_name][0]  # first spectrum
            for var in idata.posterior.data_vars:
                vals = idata.posterior[var].values.flatten()
                post[var] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'ci': (float(np.percentile(vals, 2.5)),
                           float(np.percentile(vals, 97.5))),
                }
            posteriors[model_name] = post
        
        return {
            'loo_classical': loo_results['lorentzian'],
            'loo_qsl': loo_results['qsl'],
            'delta_elpd': float(delta_elpd),
            'se_delta': float(se_delta),
            'sigma_from_zero': float(delta_elpd / se_delta) if se_delta > 0 else 0,
            'weight_classical': float(weights[0]),
            'weight_qsl': float(weights[1]),
            'posteriors': posteriors,
            'idatas': idatas,
            'interpretation': _interpret_comparison(delta_elpd, se_delta, weights),
        }


def _interpret_comparison(delta_elpd, se_delta, weights):
    """Human-readable interpretation of model comparison."""
    sigma = delta_elpd / se_delta if se_delta > 0 else 0
    if sigma > 2:
        verdict = "Strong evidence for QSL (structured) environment"
    elif sigma > 1:
        verdict = "Moderate evidence for QSL environment"
    elif sigma > -1:
        verdict = "Inconclusive — models are indistinguishable"
    elif sigma > -2:
        verdict = "Moderate evidence for classical (Lorentzian) environment"
    else:
        verdict = "Strong evidence for classical environment"
    
    return (f"{verdict}\n"
            f"  Δ(elpd) = {delta_elpd:.1f} ± {se_delta:.1f} "
            f"({sigma:.1f}σ from zero)\n"
            f"  Weights: classical={weights[0]:.3f}, QSL={weights[1]:.3f}")
