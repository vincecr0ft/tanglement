"""
Polytope geometry: CHSH facets, Balke-Pearl LP bounds, NPA SDP.

The local-realist polytope is conv(16 deterministic strategies).
Its 8 nontrivial facets are the CHSH inequalities |S| ≤ 2.

Three nested sets (Polson et al. Figure 1):
    Local polytope  ⊂  Quantum set  ⊂  No-signaling polytope
       |S| ≤ 2          |S| ≤ 2√2          |S| ≤ 4
"""

import numpy as np
from scipy.optimize import linprog
from typing import Dict

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False


def chsh_values(c: np.ndarray) -> Dict:
    """
    Evaluate all CHSH combinations for correlation vector
    c = [E(XY|1,1), E(XY|1,2), E(XY|2,1), E(XY|2,2)].
    """
    S = {
        'S1': c[0] + c[1] + c[2] - c[3],
        'S2': c[0] + c[1] - c[2] + c[3],
        'S3': c[0] - c[1] + c[2] + c[3],
        'S4': -c[0] + c[1] + c[2] + c[3],
    }
    best = max(S, key=lambda k: abs(S[k]))
    val = abs(S[best])
    return {
        'S': S,
        'max_abs_S': val,
        'best': best,
        'layer': ('local' if val <= 2 else
                  'quantum' if val <= 2 * np.sqrt(2) else
                  'no-signaling' if val <= 4 else 'invalid'),
    }


def in_local_polytope(c: np.ndarray) -> bool:
    """LP feasibility test: ∃ q ≥ 0, 1ᵀq = 1, Vᵀq = c?"""
    from .dag import BellDAG
    V = BellDAG().vertex_matrix()
    n_v = V.shape[0]
    A_eq = np.vstack([V.T, np.ones((1, n_v))])
    b_eq = np.concatenate([c, [1.0]])
    res = linprog(np.zeros(n_v), A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0, None)] * n_v, method='highs')
    return res.success


def npa_level1() -> float:
    """Level-1 NPA SDP for CHSH → returns Tsirelson bound 2√2."""
    if not HAS_CVXPY:
        return float('nan')
    Γ = cp.Variable((5, 5), symmetric=True)
    constraints = [Γ >> 0] + [Γ[i, i] == 1 for i in range(5)]
    S = Γ[1, 3] + Γ[1, 4] + Γ[2, 3] - Γ[2, 4]
    prob = cp.Problem(cp.Maximize(S), constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    return prob.value if prob.status.startswith('optimal') else float('nan')
