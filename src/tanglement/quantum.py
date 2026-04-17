"""
Quantum simulation primitives for Bell-CHSH experiments.
Provides ground-truth data generation for validating inference.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field

# Pauli matrices
I2 = np.eye(2, dtype=complex)
σ_x = np.array([[0, 1], [1, 0]], dtype=complex)
σ_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
σ_z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = [I2, σ_x, σ_y, σ_z]
PAULI_LABELS = ['I', 'X', 'Y', 'Z']

# Bloch sphere direction labels → (θ, φ)
DIRECTION_ANGLES = {
    'X': (np.pi / 2, 0.0),
    'Y': (np.pi / 2, np.pi / 2),
    'Z': (0.0, 0.0),
}


def bell_state(name: str = "phi_plus") -> np.ndarray:
    """4×4 density matrix for standard Bell states."""
    e0, e1 = np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)
    vecs = {
        "phi_plus":  (np.kron(e0, e0) + np.kron(e1, e1)) / np.sqrt(2),
        "phi_minus": (np.kron(e0, e0) - np.kron(e1, e1)) / np.sqrt(2),
        "psi_plus":  (np.kron(e0, e1) + np.kron(e1, e0)) / np.sqrt(2),
        "psi_minus": (np.kron(e0, e1) - np.kron(e1, e0)) / np.sqrt(2),
    }
    ψ = vecs[name]
    return np.outer(ψ, ψ.conj())


def werner_state(p: float) -> np.ndarray:
    """ρ = p|Ψ⁻⟩⟨Ψ⁻| + (1-p)I/4. Violates CHSH for p > 1/√2."""
    return p * bell_state("psi_minus") + (1 - p) * np.eye(4) / 4


def bloch_vector(theta: float, phi: float = 0.0) -> np.ndarray:
    """Unit vector n̂ = (sinθ cosφ, sinθ sinφ, cosθ)."""
    return np.array([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)])


def measurement_operator(theta: float, phi: float = 0.0) -> np.ndarray:
    """Spin-1/2 observable M = n̂·σ̄, eigenvalues ±1."""
    n = bloch_vector(theta, phi)
    return n[0] * σ_x + n[1] * σ_y + n[2] * σ_z


def fano_decomposition(ρ: np.ndarray) -> np.ndarray:
    """
    Fano coefficients: ρ = (1/4) Σᵢⱼ Tᵢⱼ (σᵢ⊗σⱼ),  Tᵢⱼ = Tr[ρ(σᵢ⊗σⱼ)].
    T₀₀=1. Free parameters: 3 (Alice Bloch) + 3 (Bob Bloch) + 9 (correlation) = 15.
    """
    T = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            T[i, j] = np.real(np.trace(ρ @ np.kron(PAULI[i], PAULI[j])))
    return T


def rho_from_fano(T: np.ndarray) -> np.ndarray:
    """Reconstruct ρ from Fano matrix."""
    ρ = np.zeros((4, 4), dtype=complex)
    for i in range(4):
        for j in range(4):
            ρ += T[i, j] * np.kron(PAULI[i], PAULI[j])
    return ρ / 4


def quantum_expectation(ρ: np.ndarray, θ_a: float, φ_a: float,
                         θ_b: float, φ_b: float) -> float:
    """E[XY | settings] = Tr[ρ (Mₐ⊗M_b)].  Args match setting tuple (θ_a, φ_a, θ_b, φ_b)."""
    return np.real(np.trace(ρ @ np.kron(
        measurement_operator(θ_a, φ_a), measurement_operator(θ_b, φ_b))))


def outcome_probabilities(ρ: np.ndarray, θ_a: float, φ_a: float,
                           θ_b: float, φ_b: float
                           ) -> Dict[Tuple[int, int], float]:
    """P(X=x, Y=y | settings) via projectors Pₓ = (I + x·Mₐ)/2.  Args: (θ_a, φ_a, θ_b, φ_b)."""
    Ma = measurement_operator(θ_a, φ_a)
    Mb = measurement_operator(θ_b, φ_b)
    probs = {}
    for x in [-1, 1]:
        for y in [-1, 1]:
            proj = np.kron((I2 + x * Ma) / 2, (I2 + y * Mb) / 2)
            probs[(x, y)] = max(0.0, np.real(np.trace(ρ @ proj)))
    total = sum(probs.values())
    return {k: v / total for k, v in probs.items()}


def horodecki_smax(ρ: np.ndarray) -> float:
    """
    Horodecki criterion: S_max = 2√(m₁+m₂) where m₁≥m₂ are the two
    largest eigenvalues of CᵀC, C = T[1:,1:] the correlation tensor.
    """
    C = fano_decomposition(ρ)[1:, 1:]
    eigs = np.sort(np.linalg.eigvalsh(C.T @ C))[::-1]
    return 2 * np.sqrt(max(eigs[0], 0) + max(eigs[1], 0))


# ─── Data generation ─────────────────────────────────────────────────

@dataclass
class ExperimentData:
    """Container for Bell experiment observations."""
    setting_indices: np.ndarray    # (N,) index into setting_pairs
    outcomes_x: np.ndarray         # (N,)
    outcomes_y: np.ndarray         # (N,)
    setting_pairs: list            # [(θ_a, φ_a, θ_b, φ_b), ...]
    n_per_setting: int
    true_correlations: Dict[int, float]
    true_rho: np.ndarray


def generate_data(ρ: np.ndarray,
                  settings: List[Tuple[float, float, float, float]],
                  n_per_setting: int,
                  rng: Optional[np.random.Generator] = None) -> ExperimentData:
    """
    Sample binary ±1 outcomes from quantum state across multiple settings.
    
    settings: list of (θ_a, φ_a, θ_b, φ_b)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    all_idx, all_x, all_y = [], [], []
    true_corr = {}
    outcome_vals = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for k, (θa, φa, θb, φb) in enumerate(settings):
        probs = outcome_probabilities(ρ, θa, φa, θb, φb)
        p_vec = np.array([probs[o] for o in outcome_vals])
        p_vec /= p_vec.sum()

        idx = rng.choice(4, size=n_per_setting, p=p_vec)
        for i in idx:
            x, y = outcome_vals[i]
            all_idx.append(k)
            all_x.append(x)
            all_y.append(y)

        true_corr[k] = quantum_expectation(ρ, θa, φa, θb, φb)

    return ExperimentData(
        setting_indices=np.array(all_idx),
        outcomes_x=np.array(all_x, dtype=float),
        outcomes_y=np.array(all_y, dtype=float),
        setting_pairs=settings,
        n_per_setting=n_per_setting,
        true_correlations=true_corr,
        true_rho=ρ.copy(),
    )


# Standard setting configurations

def chsh_settings() -> List[Tuple[float, float, float, float]]:
    """CHSH-optimal angles for |Φ+⟩: (θ_a, φ_a, θ_b, φ_b)."""
    return [
        (0.0,      0.0, np.pi/4,  0.0),   # A=1, B=1
        (0.0,      0.0, -np.pi/4, 0.0),   # A=1, B=2
        (np.pi/2,  0.0, np.pi/4,  0.0),   # A=2, B=1
        (np.pi/2,  0.0, -np.pi/4, 0.0),   # A=2, B=2
    ]


def tomographic_settings() -> List[Tuple[float, float, float, float]]:
    """9 settings for full correlation tensor: {X,Y,Z}⊗{X,Y,Z}."""
    settings = []
    for la in ['X', 'Y', 'Z']:
        for lb in ['X', 'Y', 'Z']:
            θa, φa = DIRECTION_ANGLES[la]
            θb, φb = DIRECTION_ANGLES[lb]
            settings.append((θa, φa, θb, φb))
    return settings
