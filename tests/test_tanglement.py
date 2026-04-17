"""Complete pytest suite for tanglement."""
import numpy as np
import pytest
from numpy.testing import assert_allclose


class TestDAG:
    def test_bell_locality(self):
        from tanglement.dag import BellDAG
        assert all(BellDAG().verify_locality().values())

    def test_nonlocal_breaks_locality(self):
        from tanglement.dag import NonlocalDAG
        assert not all(NonlocalDAG().verify_locality().values())

    def test_iv_assumptions(self):
        from tanglement.dag import IVDAG
        assert all(IVDAG().verify_assumptions().values())

    def test_16_response_types(self):
        from tanglement.dag import BellDAG
        assert len(BellDAG().response_types()) == 16

    def test_vertex_chsh_bound(self):
        from tanglement.dag import BellDAG
        V = BellDAG().vertex_matrix()
        for k in range(16):
            assert abs(V[k,0]+V[k,1]+V[k,2]-V[k,3]) <= 2.0 + 1e-10


class TestPolytope:
    def test_phi_plus_outside_local(self):
        from tanglement.quantum import bell_state, quantum_expectation, chsh_settings
        from tanglement.polytope import in_local_polytope
        rho = bell_state('phi_plus')
        c = np.array([quantum_expectation(rho, *s) for s in chsh_settings()])
        assert not in_local_polytope(c)

    def test_product_inside_local(self):
        from tanglement.quantum import quantum_expectation, chsh_settings
        from tanglement.polytope import in_local_polytope
        e0 = np.array([1,0], dtype=complex)
        rho = np.outer(np.kron(e0,e0), np.kron(e0,e0).conj())
        c = np.array([quantum_expectation(rho, *s) for s in chsh_settings()])
        assert in_local_polytope(c)

    def test_chsh_tsirelson(self):
        from tanglement.quantum import bell_state, quantum_expectation, chsh_settings
        from tanglement.polytope import chsh_values
        rho = bell_state('phi_plus')
        c = np.array([quantum_expectation(rho, *s) for s in chsh_settings()])
        r = chsh_values(c)
        assert r['layer'] == 'quantum'
        assert_allclose(r['max_abs_S'], 2*np.sqrt(2), atol=1e-6)

    @pytest.mark.skipif(not pytest.importorskip("cvxpy", reason="cvxpy"), reason="no cvxpy")
    def test_npa_tsirelson(self):
        from tanglement.polytope import npa_level1
        assert_allclose(npa_level1(), 2*np.sqrt(2), atol=0.05)


class TestQuantum:
    def test_fano_roundtrip(self):
        from tanglement.quantum import bell_state, fano_decomposition, rho_from_fano
        rho = bell_state('phi_plus')
        assert_allclose(rho, rho_from_fano(fano_decomposition(rho)), atol=1e-12)

    def test_horodecki_phi_plus(self):
        from tanglement.quantum import bell_state, horodecki_smax
        assert_allclose(horodecki_smax(bell_state('phi_plus')), 2*np.sqrt(2), atol=1e-10)

    def test_horodecki_singlet(self):
        from tanglement.quantum import bell_state, horodecki_smax
        assert_allclose(horodecki_smax(bell_state('psi_minus')), 2*np.sqrt(2), atol=1e-10)

    def test_werner_violation_threshold(self):
        from tanglement.quantum import werner_state, horodecki_smax
        assert horodecki_smax(werner_state(0.85)) > 2.0
        assert horodecki_smax(werner_state(0.75)) > 2.0
        assert horodecki_smax(werner_state(0.5)) < 2.0
        assert horodecki_smax(werner_state(0.3)) < 2.0

    def test_werner_smax_linear(self):
        from tanglement.quantum import werner_state, horodecki_smax
        for p in [0.5, 0.7, 0.85, 1.0]:
            assert_allclose(horodecki_smax(werner_state(p)), p*2*np.sqrt(2), atol=1e-6)


class TestBinaryInference:
    def test_detects_phi_plus(self):
        from tanglement.quantum import bell_state, generate_data, chsh_settings
        from tanglement.inference import BinaryConjugate
        data = generate_data(bell_state('phi_plus'), chsh_settings(), 2000, rng=np.random.default_rng(42))
        assert BinaryConjugate().fit(data, n_samples=10000).chsh['P_violates'] > 0.99

    def test_no_false_violation(self):
        from tanglement.quantum import generate_data, chsh_settings
        from tanglement.inference import BinaryConjugate
        e0 = np.array([1,0], dtype=complex)
        rho = np.outer(np.kron(e0,e0), np.kron(e0,e0).conj())
        data = generate_data(rho, chsh_settings(), 2000, rng=np.random.default_rng(42))
        assert BinaryConjugate().fit(data, n_samples=10000).chsh['P_violates'] < 0.05

    def test_werner_085_violation(self):
        from tanglement.quantum import werner_state, generate_data, chsh_settings
        from tanglement.inference import BinaryConjugate
        data = generate_data(werner_state(0.85), chsh_settings(), 5000, rng=np.random.default_rng(42))
        assert BinaryConjugate().fit(data, n_samples=15000).chsh['P_violates'] > 0.99

    def test_werner_050_no_violation(self):
        from tanglement.quantum import werner_state, generate_data, chsh_settings
        from tanglement.inference import BinaryConjugate
        data = generate_data(werner_state(0.5), chsh_settings(), 5000, rng=np.random.default_rng(42))
        assert BinaryConjugate().fit(data, n_samples=15000).chsh['P_violates'] < 0.05

    def test_bootstrap_matches_conjugate(self):
        from tanglement.quantum import bell_state, generate_data, chsh_settings
        from tanglement.inference import BinaryConjugate, BayesianBootstrap
        data = generate_data(bell_state('phi_plus'), chsh_settings(), 3000, rng=np.random.default_rng(42))
        r1 = BinaryConjugate().fit(data, n_samples=10000)
        r2 = BayesianBootstrap().fit(data, n_samples=10000)
        assert_allclose(r1.chsh['mean'], r2.chsh['mean'], atol=0.05)

    def test_coverage_calibration(self):
        from tanglement.quantum import bell_state, generate_data, chsh_settings
        from tanglement.inference import BinaryConjugate
        rho = bell_state('phi_plus')
        n_cov = sum(1 for s in range(50)
                    if BinaryConjugate().fit(
                        generate_data(rho, chsh_settings(), 500, rng=np.random.default_rng(s)),
                        n_samples=5000, rng=np.random.default_rng(123)
                    ).correlations[0]['covered'])
        assert n_cov >= 40, f"Coverage {n_cov}/50"

    def test_posterior_narrows_with_N(self):
        from tanglement.quantum import bell_state, generate_data, chsh_settings
        from tanglement.inference import BinaryConjugate
        rho = bell_state('phi_plus')
        stds = [BinaryConjugate().fit(
            generate_data(rho, chsh_settings(), N, rng=np.random.default_rng(42)),
            n_samples=15000).chsh['std'] for N in [500, 2000, 8000]]
        assert stds[1] < stds[0] * 0.7
        assert stds[2] < stds[1] * 0.7


class TestTomographic:
    @pytest.mark.slow
    @pytest.mark.pymc
    def test_phi_plus(self):
        from tanglement.quantum import bell_state, fano_decomposition, generate_data, tomographic_settings
        from tanglement.inference import Tomographic
        rho = bell_state('phi_plus')
        data = generate_data(rho, tomographic_settings(), 3000, rng=np.random.default_rng(42))
        r = Tomographic(fit_bloch=False).fit(data, n_draws=500, n_tune=500, n_chains=1)
        assert r.chsh['P_violates'] > 0.99
        C_true = fano_decomposition(rho)[1:,1:]
        n_cov = sum(1 for i in range(3) for j in range(3)
                    if abs(r.extra['C_mean'][i,j]-C_true[i,j]) < 2.5*r.extra['C_std'][i,j])
        assert n_cov >= 7

    @pytest.mark.slow
    @pytest.mark.pymc
    def test_15_param_bloch(self):
        from tanglement.quantum import bell_state, generate_data, tomographic_settings
        from tanglement.inference import Tomographic
        e0 = np.array([1,0], dtype=complex)
        ep = (e0+np.array([0,1], dtype=complex))/np.sqrt(2)
        rho = 0.7*bell_state('phi_plus') + 0.3*np.outer(np.kron(e0,ep), np.kron(e0,ep).conj())
        data = generate_data(rho, tomographic_settings(), 5000, rng=np.random.default_rng(42))
        r = Tomographic(fit_bloch=True).fit(data, n_draws=800, n_tune=800, n_chains=1)
        assert abs(r.extra['purity'] - np.real(np.trace(rho@rho))) < 0.02

    @pytest.mark.slow
    @pytest.mark.pymc
    def test_werner(self):
        from tanglement.quantum import werner_state, generate_data, tomographic_settings
        from tanglement.inference import Tomographic
        data = generate_data(werner_state(0.85), tomographic_settings(), 3000, rng=np.random.default_rng(42))
        assert Tomographic(fit_bloch=False).fit(data, n_draws=500, n_tune=500, n_chains=1).chsh['P_violates'] > 0.99


class TestBalkePearl:
    @pytest.mark.slow
    @pytest.mark.pymc
    def test_product_no_violation(self):
        from tanglement.quantum import generate_data, chsh_settings
        from tanglement.inference import BalkePearl
        e0 = np.array([1,0], dtype=complex)
        rho = np.outer(np.kron(e0,e0), np.kron(e0,e0).conj())
        data = generate_data(rho, chsh_settings(), 1000, rng=np.random.default_rng(42))
        assert BalkePearl().fit(data, n_draws=300, n_tune=300, n_chains=1).chsh['P_violates'] < 0.1

    @pytest.mark.slow
    @pytest.mark.pymc
    def test_quantum_saturates_boundary(self):
        from tanglement.quantum import bell_state, generate_data, chsh_settings
        from tanglement.inference import BalkePearl
        data = generate_data(bell_state('phi_plus'), chsh_settings(), 2000, rng=np.random.default_rng(42))
        r = BalkePearl().fit(data, n_draws=300, n_tune=300, n_chains=1)
        assert 1.8 < r.chsh['mean'] < 2.05

    @pytest.mark.slow
    @pytest.mark.pymc
    def test_sparsity(self):
        from tanglement.quantum import bell_state, generate_data, chsh_settings
        from tanglement.inference import BalkePearl
        data = generate_data(bell_state('phi_plus'), chsh_settings(), 1000, rng=np.random.default_rng(42))
        assert BalkePearl().fit(data, n_draws=300, n_tune=300, n_chains=1).extra['active_types'] < 16


class TestCollider:
    def test_singlet_accuracy(self):
        from tanglement.quantum import bell_state, fano_decomposition
        from tanglement.collider import ColliderSimulator
        rho = bell_state('psi_minus')
        ct1, p1, ct2, p2 = ColliderSimulator().generate(rho, 20000, rng=np.random.default_rng(42))
        extr = ColliderSimulator().extract(ct1, p1, ct2, p2)
        assert_allclose(np.diag(extr['T_hat'][1:,1:]), np.diag(fano_decomposition(rho)[1:,1:]), atol=0.15)

    def test_bayesian_chsh(self):
        from tanglement.quantum import bell_state
        from tanglement.collider import ColliderSimulator, ColliderBayesian
        ct1, p1, ct2, p2 = ColliderSimulator().generate(bell_state('psi_minus'), 20000, rng=np.random.default_rng(42))
        r = ColliderBayesian().fit(ColliderSimulator().extract(ct1, p1, ct2, p2), n_posterior=3000, rng=np.random.default_rng(123))
        assert r.chsh['P_violates'] > 0.99

    def test_weighted_events(self):
        from tanglement.quantum import bell_state
        from tanglement.collider import ColliderSimulator, ColliderBayesian
        ct1, p1, ct2, p2 = ColliderSimulator().generate(bell_state('psi_minus'), 20000, rng=np.random.default_rng(42))
        weights = 0.5 + 0.5*np.abs(ct1)
        r = ColliderBayesian().fit(ColliderSimulator().extract(ct1, p1, ct2, p2, weights=weights), n_posterior=3000, rng=np.random.default_rng(123))
        assert r.chsh['P_violates'] > 0.90

    def test_tensor_coverage(self):
        from tanglement.quantum import bell_state, fano_decomposition
        from tanglement.collider import ColliderSimulator, ColliderBayesian
        rho = bell_state('psi_minus')
        ct1, p1, ct2, p2 = ColliderSimulator().generate(rho, 30000, rng=np.random.default_rng(42))
        r = ColliderBayesian().fit(ColliderSimulator().extract(ct1, p1, ct2, p2), n_posterior=5000, rng=np.random.default_rng(123))
        C_true = fano_decomposition(rho)[1:,1:]
        C_m, C_s = r.extra['T_mean'][1:,1:], r.extra['T_std'][1:,1:]
        n_cov = sum(1 for i in range(3) for j in range(3) if abs(C_m[i,j]-C_true[i,j]) < 2.5*C_s[i,j])
        assert n_cov >= 6


class TestCavityQED:
    def test_entangled_qsl(self):
        from tanglement.quantum import bell_state, chsh_settings
        from tanglement.cavity_qed import generate_qubit_bell_data
        from tanglement.inference import BinaryConjugate
        data = generate_qubit_bell_data(bell_state('psi_minus'), chsh_settings(), 2000, readout_fidelity=1.0, rng=np.random.default_rng(42))
        assert BinaryConjugate().fit(data, n_samples=10000).chsh['P_violates'] > 0.99

    def test_classical_mediator(self):
        from tanglement.quantum import werner_state, chsh_settings
        from tanglement.cavity_qed import generate_qubit_bell_data
        from tanglement.inference import BinaryConjugate
        data = generate_qubit_bell_data(werner_state(0.5), chsh_settings(), 3000, readout_fidelity=1.0, rng=np.random.default_rng(42))
        assert BinaryConjugate().fit(data, n_samples=10000).chsh['P_violates'] < 0.05

    def test_readout_correction(self):
        from tanglement.quantum import bell_state, chsh_settings
        from tanglement.cavity_qed import generate_qubit_bell_data, entanglement_witness_from_correlations
        from tanglement.inference import BinaryConjugate
        data = generate_qubit_bell_data(bell_state('phi_plus'), chsh_settings(), 3000, readout_fidelity=0.93, rng=np.random.default_rng(42))
        r = BinaryConjugate().fit(data, n_samples=10000)
        w = entanglement_witness_from_correlations({k: r.correlations[k]['mean'] for k in range(4)}, readout_fidelity=0.93)
        assert w['violates_corrected']

    def test_readout_sweep_consistency(self):
        from tanglement.quantum import bell_state, chsh_settings
        from tanglement.cavity_qed import generate_qubit_bell_data, entanglement_witness_from_correlations
        from tanglement.inference import BinaryConjugate
        rho = bell_state('phi_plus')
        for F in [0.90, 0.95, 1.0]:
            data = generate_qubit_bell_data(rho, chsh_settings(), 3000, readout_fidelity=F, rng=np.random.default_rng(42))
            r = BinaryConjugate().fit(data, n_samples=10000)
            w = entanglement_witness_from_correlations({k: r.correlations[k]['mean'] for k in range(4)}, readout_fidelity=F)
            assert_allclose(w['S_corrected'], 2*np.sqrt(2), atol=0.15)

    def test_cavity_spectrum_finite(self):
        from tanglement.cavity_qed import CavityParams, SpinParams, simulate_esr_spectrum
        cavity = CavityParams(omega_c=4.0, kappa=0.01, kappa_e=0.006)
        spin = SpinParams(omega_s=4.0, gamma=0.005, g=5e-9, n_spins=int(1e15))
        omega = np.linspace(3.96, 4.04, 100)
        for env in ['lorentzian', 'qsl']:
            spec = simulate_esr_spectrum(omega, cavity, spin, environment=env, noise_level=0.003, rng=np.random.default_rng(42))
            assert np.all(np.isfinite(spec['s21_mag2']))


class TestSpectralComparison:
    @pytest.mark.slow
    @pytest.mark.pymc
    def test_lorentzian_favours_classical(self):
        from tanglement.cavity_qed import CavityParams, SpinParams, simulate_esr_spectrum
        from tanglement.spectral_comparison import SpectralModelComparison
        cav = CavityParams(omega_c=4.0, kappa=0.01, kappa_e=0.006)
        sp = SpinParams(omega_s=4.0, gamma=0.005, g=5e-9, n_spins=int(1e15))
        omega = np.linspace(3.96, 4.04, 100)
        spec = simulate_esr_spectrum(omega, cav, sp, environment='lorentzian', noise_level=0.003, rng=np.random.default_rng(42))
        r = SpectralModelComparison(cav.omega_c, cav.kappa, cav.kappa_e).compare([spec], n_draws=400, n_tune=400, n_chains=1)
        assert r['weight_classical'] > 0.5

    @pytest.mark.slow
    @pytest.mark.pymc
    def test_qsl_favours_qsl(self):
        from tanglement.cavity_qed import CavityParams, SpinParams, simulate_esr_spectrum
        from tanglement.spectral_comparison import SpectralModelComparison
        cav = CavityParams(omega_c=4.0, kappa=0.01, kappa_e=0.006)
        sp = SpinParams(omega_s=4.0, gamma=0.005, g=5e-9, n_spins=int(1e15))
        omega = np.linspace(3.96, 4.04, 100)
        spec = simulate_esr_spectrum(omega, cav, sp, environment='qsl', qsl_gap=0.01, noise_level=0.003, rng=np.random.default_rng(42))
        r = SpectralModelComparison(cav.omega_c, cav.kappa, cav.kappa_e).compare([spec], n_draws=400, n_tune=400, n_chains=1)
        assert r['weight_qsl'] > 0.5


class TestIntegration:
    def test_three_layers_agree(self):
        from tanglement.quantum import bell_state, generate_data, chsh_settings, quantum_expectation
        from tanglement.dag import BellDAG
        from tanglement.polytope import chsh_values, in_local_polytope
        from tanglement.inference import BinaryConjugate
        rho = bell_state('phi_plus')
        settings = chsh_settings()
        assert all(BellDAG().verify_locality().values())
        c = np.array([quantum_expectation(rho, *s) for s in settings])
        assert not in_local_polytope(c)
        assert chsh_values(c)['layer'] == 'quantum'
        data = generate_data(rho, settings, 3000, rng=np.random.default_rng(42))
        assert BinaryConjugate().fit(data, n_samples=15000).chsh['P_violates'] > 0.99

    def test_qsl_full_pipeline(self):
        from tanglement.quantum import werner_state, chsh_settings
        from tanglement.cavity_qed import generate_qubit_bell_data, entanglement_witness_from_correlations
        from tanglement.inference import BinaryConjugate
        from tanglement.polytope import in_local_polytope
        rho, F = werner_state(0.85), 0.97
        data = generate_qubit_bell_data(rho, chsh_settings(), 3000, readout_fidelity=F, rng=np.random.default_rng(42))
        r = BinaryConjugate().fit(data, n_samples=15000)
        c = {k: r.correlations[k]['mean'] for k in range(4)}
        w = entanglement_witness_from_correlations(c, readout_fidelity=F)
        assert w['violates_corrected']
        eta = (2*F-1)**2
        assert not in_local_polytope(np.array([c[k]/eta for k in range(4)]))


class TestInputValidation:
    def test_bell_state_invalid_name(self):
        from tanglement.quantum import bell_state
        with pytest.raises(ValueError, match="Unknown Bell state"):
            bell_state("not_a_state")

    def test_werner_state_out_of_range(self):
        from tanglement.quantum import werner_state
        with pytest.raises(ValueError):
            werner_state(-0.1)
        with pytest.raises(ValueError):
            werner_state(1.5)

    def test_generate_data_empty_settings(self):
        from tanglement.quantum import bell_state, generate_data
        with pytest.raises(ValueError, match="non-empty"):
            generate_data(bell_state('phi_plus'), [], 100)

    def test_generate_data_zero_samples(self):
        from tanglement.quantum import bell_state, generate_data, chsh_settings
        with pytest.raises(ValueError, match="n_per_setting"):
            generate_data(bell_state('phi_plus'), chsh_settings(), 0)


class TestQuantumPrimitives:
    def test_bloch_vector_z(self):
        from tanglement.quantum import bloch_vector
        b = bloch_vector(0.0, 0.0)
        assert_allclose(b, [0, 0, 1], atol=1e-12)

    def test_bloch_vector_x(self):
        from tanglement.quantum import bloch_vector
        b = bloch_vector(np.pi / 2, 0.0)
        assert_allclose(b, [1, 0, 0], atol=1e-12)

    def test_measurement_eigenvalues(self):
        from tanglement.quantum import measurement_operator
        M = measurement_operator(0.0, 0.0)
        eigs = np.sort(np.linalg.eigvalsh(M))
        assert_allclose(eigs, [-1, 1], atol=1e-12)

    def test_outcome_probabilities_sum_to_one(self):
        from tanglement.quantum import bell_state, outcome_probabilities
        rho = bell_state('phi_plus')
        probs = outcome_probabilities(rho, 0.0, 0.0, np.pi/4, 0.0)
        assert_allclose(sum(probs.values()), 1.0, atol=1e-12)

    def test_outcome_probabilities_consistent(self):
        from tanglement.quantum import bell_state, outcome_probabilities, quantum_expectation
        rho = bell_state('phi_plus')
        θa, φa, θb, φb = 0.0, 0.0, np.pi/4, 0.0
        probs = outcome_probabilities(rho, θa, φa, θb, φb)
        E_from_probs = sum(x * y * p for (x, y), p in probs.items())
        E_direct = quantum_expectation(rho, θa, φa, θb, φb)
        assert_allclose(E_from_probs, E_direct, atol=1e-10)


class TestTomographicStd:
    @pytest.mark.slow
    @pytest.mark.pymc
    def test_t_std_nonzero(self):
        from tanglement.quantum import bell_state, generate_data, tomographic_settings
        from tanglement.inference import Tomographic
        rho = bell_state('phi_plus')
        data = generate_data(rho, tomographic_settings(), 3000,
                             rng=np.random.default_rng(42))
        r = Tomographic(fit_bloch=True).fit(
            data, n_draws=500, n_tune=500, n_chains=1)
        T_std = r.extra['T_std']
        # Correlation block should have nonzero std
        assert np.all(T_std[1:, 1:] > 0)
        # Bloch vector stds should be nonzero when fit_bloch=True
        assert np.all(T_std[1:, 0] > 0)
        assert np.all(T_std[0, 1:] > 0)


class TestBootstrapExtra:
    def test_chsh_samples_present(self):
        from tanglement.quantum import bell_state, generate_data, chsh_settings
        from tanglement.inference import BayesianBootstrap
        data = generate_data(bell_state('phi_plus'), chsh_settings(), 1000,
                             rng=np.random.default_rng(42))
        r = BayesianBootstrap().fit(data, n_samples=5000)
        assert 'chsh_samples' in r.extra
        assert len(r.extra['chsh_samples']) == 5000
