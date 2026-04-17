"""
Causal graph definitions for Bell and IV experiments.

Bell DAG (Gill 2211.05569, Figure 2):
    Experimenter → A, B;  Λ → X, Y;  A → X;  B → Y

IV DAG (Polson et al. 2603.28973, Section 3.1):
    Z → X → Y;  U → X;  U → Y;  Z ⊥ U
"""

import numpy as np
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from itertools import product
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class ResponseType:
    """Deterministic strategy (vertex of the local-realist polytope)."""
    alice_responses: Dict[int, int]   # setting → outcome
    bob_responses: Dict[int, int]

    def correlation(self, a: int, b: int) -> float:
        return float(self.alice_responses[a] * self.bob_responses[b])


class BellDAG:
    """
    Bell experiment DAG.  d-separation encodes locality:
        X ⊥ B | {A, Λ},   Y ⊥ A | {B, Λ},   (A,B) ⊥ Λ
    """

    def __init__(self, n_settings_a: int = 2, n_settings_b: int = 2,
                 outcomes: List[int] = [-1, 1]):
        self.n_a, self.n_b, self.outcomes = n_settings_a, n_settings_b, outcomes
        self.graph = BayesianNetwork([
            ('Experimenter', 'A'), ('Experimenter', 'B'),
            ('Lambda', 'X'), ('Lambda', 'Y'),
            ('A', 'X'), ('B', 'Y'),
        ])

    def verify_locality(self) -> Dict[str, bool]:
        g = self.graph
        return {
            'X ⊥ B | {A,Λ}': not g.is_dconnected('X', 'B', {'A', 'Lambda'}),
            'Y ⊥ A | {B,Λ}': not g.is_dconnected('Y', 'A', {'B', 'Lambda'}),
            'A ⊥ Λ':         not g.is_dconnected('A', 'Lambda', set()),
            'B ⊥ Λ':         not g.is_dconnected('B', 'Lambda', set()),
        }

    def response_types(self) -> List[ResponseType]:
        """All |outcomes|^(n_a+n_b) deterministic strategies."""
        types = []
        for a_resp in product(self.outcomes, repeat=self.n_a):
            for b_resp in product(self.outcomes, repeat=self.n_b):
                types.append(ResponseType(
                    alice_responses={s: a_resp[s] for s in range(self.n_a)},
                    bob_responses={s: b_resp[s] for s in range(self.n_b)},
                ))
        return types

    def vertex_matrix(self) -> np.ndarray:
        """(n_types, n_a*n_b) matrix of correlations per vertex."""
        types = self.response_types()
        V = np.zeros((len(types), self.n_a * self.n_b))
        for k, rt in enumerate(types):
            for a in range(self.n_a):
                for b in range(self.n_b):
                    V[k, a * self.n_b + b] = rt.correlation(a, b)
        return V

    def vertex_outcome_indicators(self) -> np.ndarray:
        """(n_types, n_pairs, 4) indicator: type k gives outcome oi at pair sp."""
        ov = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        types = self.response_types()
        R = np.zeros((len(types), self.n_a * self.n_b, 4))
        for k, rt in enumerate(types):
            for a in range(self.n_a):
                for b in range(self.n_b):
                    sp = a * self.n_b + b
                    xv, yv = rt.alice_responses[a], rt.bob_responses[b]
                    for oi, (xo, yo) in enumerate(ov):
                        if xv == xo and yv == yo:
                            R[k, sp, oi] = 1.0
        return R

    @staticmethod
    def factorisation() -> str:
        return ("P(X,Y|A,B) = ∫ P(X|A,λ) P(Y|B,λ) dP(λ)\n"
                "≡ Bell local hidden variable model (Fine Thm conditions 1–3)")


class NonlocalDAG:
    """Non-local model: X depends on B, Y depends on A → locality fails."""
    def __init__(self):
        self.graph = BayesianNetwork([
            ('Experimenter', 'A'), ('Experimenter', 'B'),
            ('Lambda', 'X'), ('Lambda', 'Y'),
            ('A', 'X'), ('A', 'Y'), ('B', 'X'), ('B', 'Y'),
        ])

    def verify_locality(self) -> Dict[str, bool]:
        g = self.graph
        return {
            'X ⊥ B | {A,Λ}': not g.is_dconnected('X', 'B', {'A', 'Lambda'}),
            'Y ⊥ A | {B,Λ}': not g.is_dconnected('Y', 'A', {'B', 'Lambda'}),
        }


class IVDAG:
    """Instrumental variable DAG: Z→X→Y, U→X, U→Y, Z⊥U."""
    def __init__(self):
        self.graph = BayesianNetwork([
            ('Z', 'X'), ('X', 'Y'), ('U', 'X'), ('U', 'Y'),
        ])

    def verify_assumptions(self) -> Dict[str, bool]:
        g = self.graph
        return {
            'Z ⊥ U':          not g.is_dconnected('Z', 'U', set()),
            'Z ⊥ Y | {X,U}':  not g.is_dconnected('Z', 'Y', {'X', 'U'}),
        }

    @staticmethod
    def bell_mapping() -> Dict[str, str]:
        return {
            'Z (instrument)':  'measurement setting',
            'X (treatment)':   "Alice's outcome",
            'Y (outcome)':     "Bob's outcome",
            'U (confounder)':  'hidden variable λ',
        }
