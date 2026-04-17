"""
tanglement: Bayesian DAG inference for quantum entanglement detection.

Three-layer architecture:
  Layer 1 (dag):       Causal graph structure via pgmpy
  Layer 2 (polytope):  Response-function polytope, LP/SDP bounds
  Layer 3 (inference): Bayesian posterior via PyMC and conjugate models

Applications: Bell-CHSH tests, collider spin tomography, quantum spin liquids.

References:
  Gill (arXiv:2211.05569) — Bell-CHSH as DAG exercise
  Polson, Sokolov, Zantedeschi (arXiv:2603.28973) — Unified polytope framework
"""

__version__ = "0.1.0"

from tanglement.quantum import (
    bell_state,
    werner_state,
    generate_data,
    chsh_settings,
    tomographic_settings,
    horodecki_smax,
    fano_decomposition,
)
from tanglement.dag import BellDAG, NonlocalDAG, IVDAG
from tanglement.polytope import chsh_values, in_local_polytope
from tanglement.inference import (
    BinaryConjugate,
    BayesianBootstrap,
    Tomographic,
    BalkePearl,
    PosteriorResult,
)

__all__ = [
    "__version__",
    # quantum
    "bell_state",
    "werner_state",
    "generate_data",
    "chsh_settings",
    "tomographic_settings",
    "horodecki_smax",
    "fano_decomposition",
    # dag
    "BellDAG",
    "NonlocalDAG",
    "IVDAG",
    # polytope
    "chsh_values",
    "in_local_polytope",
    # inference
    "BinaryConjugate",
    "BayesianBootstrap",
    "Tomographic",
    "BalkePearl",
    "PosteriorResult",
]
