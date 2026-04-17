# tanglement

**Bayesian DAG inference for quantum entanglement detection.**

[![CI](https://github.com/vincecr0ft/tanglement/actions/workflows/ci.yml/badge.svg)](https://github.com/vincecr0ft/tanglement/actions)
[![PyPI](https://img.shields.io/pypi/v/tanglement)](https://pypi.org/project/tanglement/)

---

`tanglement` implements the connection between Bell's theorem and causal inference as a practical Bayesian inference engine, following [Gill (2211.05569)](https://arxiv.org/abs/2211.05569) and [Polson, Sokolov & Zantedeschi (2603.28973)](https://arxiv.org/abs/2603.28973).

## Installation

```bash
pip install tanglement          # core
pip install tanglement[all]     # with cvxpy (NPA SDP) and matplotlib
pip install tanglement[dev]     # with pytest
```

## Quick start

```python
from tanglement.quantum import bell_state, generate_data, chsh_settings
from tanglement.inference import BinaryConjugate

# Generate data from |Φ+⟩
rho = bell_state('phi_plus')
data = generate_data(rho, chsh_settings(), n_per_setting=2000)

# Bayesian CHSH test
result = BinaryConjugate().fit(data, n_samples=20000)
print(f"P(CHSH violation | data) = {result.chsh['P_violates']:.4f}")
# → P(CHSH violation | data) = 1.0000
```

## Architecture

```
src/tanglement/
├── quantum.py              Pauli algebra, states, Fano decomposition
├── dag.py                  pgmpy DAGs: Bell, IV — d-separation, response types
├── polytope.py             CHSH facets, LP feasibility, NPA SDP
├── inference.py            4 Bayesian models (conjugate, bootstrap, PyMC, Balke-Pearl)
├── collider.py             Collider angular → density matrix pipeline
├── cavity_qed.py           Cavity-QED for quantum spin liquid experiments
└── spectral_comparison.py  PyMC LOO-CV: classical bath vs QSL
```

## Three layers

| Layer | Module | What it does |
|-------|--------|-------------|
| **DAG** | `dag.py` | Encodes Bell locality as *d*-separation. Enumerates 16 polytope vertices. |
| **Polytope** | `polytope.py` | CHSH facets, LP membership test, NPA SDP (Tsirelson bound). |
| **Inference** | `inference.py` | Four models returning calibrated posteriors on CHSH. |

## Inference models

| Model | Method | Best for |
|-------|--------|----------|
| `BinaryConjugate` | Exact Dirichlet | Binary ±1, qubit readout |
| `BayesianBootstrap` | Dirichlet weights | Ratio estimators, continuous data |
| `Tomographic` | PyMC NUTS | Full ρ reconstruction (15 Fano params) |
| `BalkePearl` | PyMC NUTS | Polytope-native partial identification |

## Applications

- **Collider spin tomography**: angular distributions → Fano coefficients → CHSH
- **Cavity-QED / QSL**: two-qubit Bell test with readout correction; LOO-CV spectral model comparison
- **Causal inference**: Bell ↔ IV structural equivalence; Balke-Pearl bounds

## Testing

```bash
pytest                     # all 40 tests
pytest -m "not slow"       # fast tests only (32 tests, ~10s)
pytest -m slow             # PyMC MCMC tests only (8 tests, ~2min)
```

## Development

```bash
git clone https://github.com/tanglement/tanglement.git
cd tanglement
pip install -e ".[dev]"
pytest
```

## References

- Gill, R.D. (2023). Bell's theorem is an exercise in the statistical theory of causality. [arXiv:2211.05569](https://arxiv.org/abs/2211.05569)
- Polson, N.G., Sokolov, V., Zantedeschi, D. (2026). Bell's inequality, causal bounds, and quantum Bayesian computation. [arXiv:2603.28973](https://arxiv.org/abs/2603.28973)
