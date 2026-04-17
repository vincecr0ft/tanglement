"""
Collider angular distribution → density matrix pipeline.

Simulates spin-1/2 ⊗ spin-1/2 pair production (e.g. top quarks)
and reconstructs the Fano coefficients from decay lepton angles.

Full 4-angle parameterisation: (cosθ₁, φ₁, cosθ₂, φ₂).

Angular distribution:
    p ∝ Σᵢⱼ Tᵢⱼ fᵢ(θ₁,φ₁) gⱼ(θ₂,φ₂)

Extraction (orthogonality):
    Tᵢⱼ = 9 <fᵢ gⱼ>  for i,j > 0   (factor 9 = 3×3)
    Tᵢ₀ = 3 <fᵢ>      for i > 0     (Bloch vector)

For weighted events (detector effects): Bayesian bootstrap
correctly propagates ratio-estimator uncertainty.
"""

import numpy as np
from typing import Dict
from .quantum import fano_decomposition
from .inference import _corr_summary, _chsh_horodecki, PosteriorResult

__all__ = ["ColliderSimulator", "ColliderBayesian"]


class ColliderSimulator:
    """Generate and analyse angular distributions from a density matrix."""

    @staticmethod
    def _density_basis(ct1, p1, ct2, p2):
        """Raw basis (no normalisation) for accept-reject density."""
        n = len(ct1)
        st1 = np.sqrt(np.maximum(1 - ct1**2, 0))
        st2 = np.sqrt(np.maximum(1 - ct2**2, 0))
        f = np.column_stack([np.ones(n), ct1, st1*np.cos(p1), st1*np.sin(p1)])
        g = np.column_stack([np.ones(n), ct2, st2*np.cos(p2), st2*np.sin(p2)])
        return f[:,:,None] * g[:,None,:]

    @staticmethod
    def _extraction_basis(ct1, p1, ct2, p2):
        """Normalised basis: <Bᵢⱼ> = Tᵢⱼ under the true density."""
        n = len(ct1)
        st1 = np.sqrt(np.maximum(1 - ct1**2, 0))
        st2 = np.sqrt(np.maximum(1 - ct2**2, 0))
        f = np.column_stack([np.ones(n), 3*ct1,
                             3*st1*np.cos(p1), 3*st1*np.sin(p1)])
        g = np.column_stack([np.ones(n), 3*ct2,
                             3*st2*np.cos(p2), 3*st2*np.sin(p2)])
        return f[:,:,None] * g[:,None,:]

    def generate(self, rho, n_events, rng=None):
        """Accept-reject sampling. Returns (cos_t1, phi1, cos_t2, phi2)."""
        if rng is None:
            rng = np.random.default_rng(42)
        T = fano_decomposition(rho)
        max_p = (1 + np.sum(np.abs(T))) / (16 * np.pi**2)

        out = [[] for _ in range(4)]
        accepted = 0
        batch = max(n_events * 2, 10000)

        while accepted < n_events:
            ct1 = rng.uniform(-1, 1, batch)
            p1 = rng.uniform(0, 2*np.pi, batch)
            ct2 = rng.uniform(-1, 1, batch)
            p2 = rng.uniform(0, 2*np.pi, batch)

            B = self._density_basis(ct1, p1, ct2, p2)
            density = np.maximum(np.sum(T * B, axis=(1,2)) / (16*np.pi**2), 0)
            mask = rng.uniform(0, max_p, batch) < density

            for arr, vals in zip(out, [ct1, p1, ct2, p2]):
                arr.append(vals[mask])
            accepted += mask.sum()

        return tuple(np.concatenate(a)[:n_events] for a in out)

    def extract(self, ct1, p1, ct2, p2, weights=None):
        """
        Extract Fano coefficients. Returns dict with T_hat, T_se, basis_data.
        
        With weights: ratio estimator T̂ᵢⱼ = Σ wₙ Bᵢⱼⁿ / Σ wₙ (Fieller problem).
        Without weights: simple mean (no ratio issues).
        """
        B = self._extraction_basis(ct1, p1, ct2, p2)
        n = len(ct1)
        if weights is None:
            weights = np.ones(n)

        W = np.sum(weights)
        T_hat = np.einsum('n,nij->ij', weights, B) / W
        residuals = B - T_hat[None,:,:]
        T_se = np.sqrt(np.einsum('n,nij,nij->ij', weights**2,
                                  residuals, residuals)) / W

        return {'T_hat': T_hat, 'T_se': T_se, 'weights': weights,
                'basis_data': B, 'n_events': n}


class ColliderBayesian:
    """
    Bayesian bootstrap inference on Fano coefficients from collider data.
    Correctly handles ratio estimators (weighted events).
    """

    def fit(self, collider_data: Dict, n_posterior=8000, rng=None):
        if rng is None:
            rng = np.random.default_rng(123)

        B = collider_data['basis_data']
        w = collider_data['weights']
        n = collider_data['n_events']

        # Chunked bootstrap (memory-safe)
        chunk = min(500, n_posterior)
        T_post = np.zeros((n_posterior, 4, 4))

        for start in range(0, n_posterior, chunk):
            end = min(start + chunk, n_posterior)
            batch = end - start
            dw = rng.dirichlet(np.ones(n), size=batch)
            cw = dw * w[None, :]
            Ws = cw.sum(axis=1)
            T_post[start:end] = np.einsum('sn,nij->sij', cw, B) / Ws[:,None,None]
            del dw, cw

        C_post = T_post[:, 1:, 1:]
        chsh_s = _chsh_horodecki(C_post)

        T_mean = np.mean(T_post, axis=0)
        T_std = np.std(T_post, axis=0)

        corrs = {}
        for i in range(3):
            for j in range(3):
                k = i * 3 + j
                corrs[k] = _corr_summary(C_post[:, i, j])

        return PosteriorResult(
            correlations=corrs,
            chsh=_corr_summary(chsh_s) | {'P_violates': float(np.mean(chsh_s > 2))},
            extra={
                'T_mean': T_mean, 'T_std': T_std,
                'T_hat_freq': collider_data['T_hat'],
                'T_se_freq': collider_data['T_se'],
                'chsh_samples': chsh_s,
            },
        )
