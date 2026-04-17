"""
Cavity-QED module for quantum spin liquid experiments.

Maps the bell_chsh framework onto two experimental scenarios:

1. Q-Sense (WP4): Two superconducting qubits coupled to a QSL.
   Qubit frequencies are "settings", readout is "outcomes".
   QSL ground state mediates correlations → Bell test.

2. EsQuL: Witness spins coupled through a QSL via spinons.
   ESR frequency/field settings probe spin-spin correlations
   mediated by the spin-liquid environment.

Physics:
  Cavity-spin coupling: H = ωc a†a + Σ ωk σk†σk + g Σ(a†σk + a σk†)
  Input-output: S21(ω) = 1 + iκe / (ω - ωc - iκ/2 + Σ_spin(ω))
  Spin self-energy: Σ_spin(ω) encodes the QSL spectral function
  Strong coupling: g√N > (κ + γ)/2 → normal-mode splitting

The spectral function of the QSL environment determines whether
correlations are classical (Lorentzian bath → local polytope)
or quantum (structured/gapped → can violate CHSH).
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

from .quantum import (ExperimentData, fano_decomposition, rho_from_fano,
                      bell_state, werner_state, PAULI, I2, σ_x, σ_y, σ_z)


# ─── Cavity-spin Hamiltonian ──────────────────────────────────────────

@dataclass
class CavityParams:
    """Microwave cavity parameters."""
    omega_c: float      # cavity frequency (GHz)
    kappa: float        # total cavity linewidth (GHz)
    kappa_e: float      # external coupling rate (GHz)


@dataclass
class SpinParams:
    """Spin ensemble parameters."""
    omega_s: float      # spin transition frequency (GHz)
    gamma: float        # spin linewidth (GHz)
    g: float            # single spin-cavity coupling (GHz)
    n_spins: int        # number of spins
    
    @property
    def g_collective(self) -> float:
        """Collective coupling g√N."""
        return self.g * np.sqrt(self.n_spins)
    
    @property
    def cooperativity(self) -> float:
        """Cooperativity C = 4g²N / (κγ) — must be >> 1 for strong coupling."""
        return 4 * self.g**2 * self.n_spins


def cavity_transmission(omega: np.ndarray, cavity: CavityParams,
                         spin_self_energy: np.ndarray) -> np.ndarray:
    """
    Cavity transmission S21(ω) via input-output theory.
    
    S21(ω) = 1 + i κe / (ω - ωc - iκ/2 + Σ_spin(ω))
    
    where Σ_spin(ω) is the spin self-energy encoding the environment.
    """
    denom = omega - cavity.omega_c - 1j * cavity.kappa / 2 + spin_self_energy
    return 1.0 + 1j * cavity.kappa_e / denom


def lorentzian_self_energy(omega: np.ndarray, spin: SpinParams) -> np.ndarray:
    """
    Self-energy for a simple Lorentzian spin bath (classical/incoherent):
    
        Σ(ω) = g²N / (ω - ωs + iγ/2)
    
    This is the standard Jaynes-Cummings result for identical spins.
    Produces vacuum Rabi splitting when g√N > (κ+γ)/4.
    """
    return spin.g**2 * spin.n_spins / (omega - spin.omega_s + 1j * spin.gamma / 2)


def qsl_self_energy(omega: np.ndarray, spin: SpinParams,
                     delta_gap: float = 0.1,
                     spectral_type: str = 'gapped_continuum') -> np.ndarray:
    """
    Self-energy for a QSL-mediated spin environment.
    
    Unlike a Lorentzian bath, a QSL has structured spectral weight:
    - Gapped continuum: onset at Δ_gap, power-law rise (spinon pair creation)
    - Memory effects: non-Markovian (frequency-dependent damping)
    
    For a gapped spinon continuum with density of states ρ(ε) ∝ (ε-Δ)^α θ(ε-Δ):
    
        Σ_QSL(ω) = g²N ∫ ρ(ε) / (ω - ε + iη) dε
    
    This gives a self-energy with:
    - Real part: dispersive shift (level repulsion from spinon continuum)
    - Imaginary part: damping that turns on above the gap
    
    Parameters
    ----------
    delta_gap : spinon gap energy (GHz)
    spectral_type : 'gapped_continuum' or 'power_law'
    """
    coupling_strength = spin.g**2 * spin.n_spins
    eta = spin.gamma / 2   # broadening
    
    if spectral_type == 'gapped_continuum':
        # Spinon pair continuum: ρ(ε) ∝ √(ε - Δ) θ(ε - Δ)
        # Integrate analytically for the model self-energy
        # Use a simplified form: Σ(ω) = g²N · √(ω - Δ + iη) / bandwidth
        bandwidth = 2.0  # GHz, spinon bandwidth
        z = (omega - spin.omega_s + delta_gap + 1j * eta) / bandwidth
        sigma = coupling_strength * np.sqrt(z) / bandwidth
        return sigma
    
    elif spectral_type == 'power_law':
        # Sub-ohmic: Σ(ω) ∝ ω^s with s < 1 (memory effects)
        s = 0.5
        sigma = coupling_strength * ((omega - spin.omega_s + 1j * eta) / 1.0)**s
        return sigma
    
    else:
        return lorentzian_self_energy(omega, spin)


# ─── Two-qubit Bell experiment via QSL (Q-Sense WP4) ─────────────────

@dataclass 
class QubitProbeParams:
    """Parameters for a superconducting qubit probe."""
    omega_q: float        # qubit frequency (GHz), tunable
    T1: float             # relaxation time (μs)
    T2: float             # dephasing time (μs)
    g_coupling: float     # qubit-QSL coupling (MHz)


def qsl_mediated_correlation(rho_qsl: np.ndarray,
                              omega1: float, omega2: float,
                              phi1: float, phi2: float) -> float:
    """
    Correlation E[X₁X₂] between two qubits coupled to a QSL.
    
    The QSL ground state mediates an effective interaction between
    qubits via virtual spinon exchange. The correlation depends on:
    - Qubit frequencies ω₁, ω₂ (the "measurement settings")
    - Measurement basis angles φ₁, φ₂
    - QSL density matrix ρ_QSL (the "hidden variable" or quantum state)
    
    For the Bell test:
      Setting A = (ω₁, φ₁) → mapped to Bloch sphere direction
      Setting B = (ω₂, φ₂) → mapped to Bloch sphere direction
      Outcome X = qubit 1 readout ∈ {-1, +1}
      Outcome Y = qubit 2 readout ∈ {-1, +1}
    
    If ρ_QSL is entangled, E[XY] can violate CHSH.
    If ρ_QSL is separable, E[XY] stays in local polytope.
    
    We model this by treating the qubit measurement axes as spin
    measurement directions on an effective 2-qubit state ρ_eff that
    the QSL mediates between the probe qubits.
    """
    # The QSL mediates an effective 2-qubit state between the probes.
    # The probe frequencies/phases select the measurement basis.
    # This is exactly our existing quantum.py framework.
    from .quantum import measurement_operator, quantum_expectation
    
    # Map (omega, phi) to Bloch sphere angles
    # Frequency detuning → polar angle (how strongly qubit couples to QSL)
    # Phase → azimuthal angle (measurement basis rotation)
    theta1 = omega1  # already in radians for the model
    theta2 = omega2
    
    return quantum_expectation(rho_qsl, theta1, phi1, theta2, phi2)


def generate_qubit_bell_data(rho_eff: np.ndarray,
                              settings: List[Tuple[float, float, float, float]],
                              n_per_setting: int,
                              readout_fidelity: float = 0.95,
                              rng: Optional[np.random.Generator] = None
                              ) -> ExperimentData:
    """
    Generate Bell experiment data from two qubits coupled to a QSL.
    
    Adds realistic readout errors: with probability (1-F), the outcome
    is flipped. This attenuates correlations by F² and must be
    corrected in the analysis (or the bound adjusted).
    
    Parameters
    ----------
    rho_eff : effective 2-qubit state mediated by QSL
    settings : list of (θ₁, φ₁, θ₂, φ₂) qubit measurement settings
    n_per_setting : measurements per setting pair
    readout_fidelity : probability of correct single-shot readout
    """
    from .quantum import outcome_probabilities, generate_data
    
    # Generate ideal data
    data = generate_data(rho_eff, settings, n_per_setting, rng=rng)
    
    if readout_fidelity < 1.0:
        if rng is None:
            rng = np.random.default_rng(42)
        # Flip each outcome independently with probability 1-F
        n = len(data.outcomes_x)
        flip_x = rng.random(n) > readout_fidelity
        flip_y = rng.random(n) > readout_fidelity
        data.outcomes_x[flip_x] *= -1
        data.outcomes_y[flip_y] *= -1
        
        # Update true correlations to account for attenuation
        F2 = readout_fidelity**2 + (1 - readout_fidelity)**2  # net fidelity
        # E_observed = (2F-1)² E_true for binary readout
        eta = (2 * readout_fidelity - 1)**2
        data.true_correlations = {k: v * eta 
                                   for k, v in data.true_correlations.items()}
    
    return data


# ─── Witness-spin spectral extraction (EsQuL) ─────────────────────────

def simulate_esr_spectrum(omega: np.ndarray,
                           cavity: CavityParams,
                           spin: SpinParams,
                           environment: str = 'lorentzian',
                           qsl_gap: float = 0.1,
                           noise_level: float = 0.01,
                           rng: Optional[np.random.Generator] = None
                           ) -> Dict:
    """
    Simulate ESR cavity transmission spectrum.
    
    Returns |S21(ω)|² with noise, plus the complex S21 for phase-sensitive analysis.
    
    Parameters
    ----------
    omega : frequency array (GHz)
    cavity : cavity parameters
    spin : spin ensemble parameters  
    environment : 'lorentzian' (classical bath) or 'qsl' (structured)
    qsl_gap : spinon gap for QSL environment (GHz)
    noise_level : Gaussian noise on |S21|²
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    if environment == 'lorentzian':
        sigma = lorentzian_self_energy(omega, spin)
    elif environment == 'qsl':
        sigma = qsl_self_energy(omega, spin, delta_gap=qsl_gap)
    else:
        sigma = np.zeros_like(omega, dtype=complex)
    
    s21 = cavity_transmission(omega, cavity, sigma)
    s21_mag2 = np.abs(s21)**2
    
    # Add measurement noise
    s21_noisy = s21_mag2 + noise_level * rng.standard_normal(len(omega))
    
    return {
        'omega': omega,
        's21_complex': s21,
        's21_mag2': s21_mag2,
        's21_noisy': s21_noisy,
        'noise_level': noise_level,
        'self_energy': sigma,
        'environment': environment,
    }


class SpectralFanoExtractor:
    """
    Extract effective spin correlation parameters from cavity spectra.
    
    The cavity transmission encodes spin-spin correlations via the
    self-energy. By fitting spectra at multiple magnetic field settings
    (which tune ωs), we extract the frequency-dependent self-energy
    and decompose it into contributions from different correlation channels.
    
    This is the spectral analogue of the collider angular extraction:
    - Collider: angular basis functions fᵢ(θ) gⱼ(φ) → Fano coefficients
    - Cavity: spectral basis functions hᵢ(ω, B) → self-energy components
    
    The self-energy at different field settings B probes different
    spin correlations, just as different measurement angles probe
    different density matrix elements.
    """
    
    def __init__(self, cavity: CavityParams):
        self.cavity = cavity
    
    def fit_spectrum(self, spectrum: Dict,
                      model: str = 'lorentzian') -> Dict:
        """
        Fit a single spectrum to extract spin parameters.
        
        For Lorentzian model: fit ωs, γ, g²N (3 parameters)
        For QSL model: fit ωs, γ, g²N, Δ_gap, spectral_index (5 parameters)
        """
        from scipy.optimize import minimize
        
        omega = spectrum['omega']
        data = spectrum['s21_noisy']
        noise = spectrum['noise_level']
        
        def residual_lorentzian(params):
            omega_s, gamma, g2N = params
            sp = SpinParams(omega_s, gamma, np.sqrt(g2N / max(1, 1)), 1)
            # Override for direct g²N parameterisation
            sigma = g2N / (omega - omega_s + 1j * gamma / 2)
            s21 = cavity_transmission(omega, self.cavity, sigma)
            model_data = np.abs(s21)**2
            return np.sum(((model_data - data) / noise)**2)
        
        def residual_qsl(params):
            omega_s, gamma, g2N, gap = params
            bandwidth = 2.0
            z = (omega - omega_s + gap + 1j * gamma/2) / bandwidth
            sigma = g2N * np.sqrt(z + 0j) / bandwidth
            s21 = cavity_transmission(omega, self.cavity, sigma)
            model_data = np.abs(s21)**2
            return np.sum(((model_data - data) / noise)**2)
        
        if model == 'lorentzian':
            x0 = [self.cavity.omega_c, 0.01, 0.001]
            bounds = [(omega.min(), omega.max()), (1e-4, 1.0), (1e-6, 1.0)]
            res = minimize(residual_lorentzian, x0, bounds=bounds, method='L-BFGS-B')
            omega_s, gamma, g2N = res.x
            return {'omega_s': omega_s, 'gamma': gamma, 'g2N': g2N,
                    'chi2': res.fun, 'model': model, 'ndf': len(omega) - 3}
        
        elif model == 'qsl':
            x0 = [self.cavity.omega_c, 0.01, 0.001, 0.1]
            bounds = [(omega.min(), omega.max()), (1e-4, 1.0),
                      (1e-6, 1.0), (0, 2.0)]
            res = minimize(residual_qsl, x0, bounds=bounds, method='L-BFGS-B')
            omega_s, gamma, g2N, gap = res.x
            return {'omega_s': omega_s, 'gamma': gamma, 'g2N': g2N,
                    'gap': gap, 'chi2': res.fun, 'model': model,
                    'ndf': len(omega) - 4}
    
    def bayesian_model_comparison(self, spectra: List[Dict],
                                    n_draws: int = 800,
                                    n_tune: int = 800,
                                    seed: int = 42) -> Dict:
        """
        Bayesian comparison: Lorentzian (classical) vs QSL (quantum) environment.
        
        Uses full PyMC posterior inference with LOO-CV (PSIS-LOO) for
        model comparison. See spectral_comparison.py for details.
        
        This is the spectral analogue of asking whether correlations
        lie inside the local polytope (Lorentzian) or outside it (QSL).
        """
        from .spectral_comparison import SpectralModelComparison
        comp = SpectralModelComparison(
            self.cavity.omega_c, self.cavity.kappa, self.cavity.kappa_e)
        return comp.compare(spectra, n_draws=n_draws, n_tune=n_tune,
                           n_chains=1, seed=seed)


# ─── Entanglement witness from spectral data ─────────────────────────

def entanglement_witness_from_correlations(correlations: Dict[int, float],
                                            readout_fidelity: float = 1.0
                                            ) -> Dict:
    """
    Given measured correlations from two qubits coupled to a QSL,
    compute the entanglement witness value.
    
    For CHSH: W = |S| - 2. If W > 0, the state is entangled.
    
    With imperfect readout (fidelity F), measured correlations are
    attenuated: E_meas = η · E_true where η = (2F-1)².
    The corrected CHSH bound is |S|/η ≤ 2 for local realism,
    equivalently |S_meas| ≤ 2η.
    
    This correction is critical for millikelvin qubit experiments
    where F ~ 0.95-0.99.
    """
    from .polytope import chsh_values
    
    keys = sorted(correlations.keys())
    c = np.array([correlations[k] for k in keys[:4]])
    
    chsh = chsh_values(c)
    eta = (2 * readout_fidelity - 1)**2
    
    # Corrected CHSH: divide measured |S| by η to get true |S|
    S_corrected = chsh['max_abs_S'] / eta if eta > 0 else float('inf')
    
    return {
        'S_measured': chsh['max_abs_S'],
        'S_corrected': S_corrected,
        'readout_eta': eta,
        'violates_corrected': S_corrected > 2.0,
        'layer': chsh['layer'],
        'entanglement_witness': S_corrected - 2.0,
    }
