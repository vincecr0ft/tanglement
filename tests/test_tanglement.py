"""
Complete pytest suite for tanglement.
"""
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
        with pytest.raises(ValueError, match="p must be in"):
            werner_state(1.5)
        with pytest.raises(ValueError, match="p must be in"):
            werner_state(-0.1)

    def test_generate_data_empty_settings(self):
        from tanglement.quantum import generate_data, bell_state
        with pytest.raises(ValueError, match="non-empty"):
            generate_data(bell_state('phi_plus'), [], 100)

    def test_generate_data_zero_samples(self):
        from tanglement.quantum import generate_data, bell_state, chsh_settings
        with pytest.raises(ValueError, match="n_per_setting"):
            generate_data(bell_state('phi_plus'), chsh_settings(), 0)


class TestEntanglementMeasures:
    def test_negativity_bell(self):
        from tanglement.quantum import bell_state, negativity
        assert negativity(bell_state('phi_plus')) > 0.49

    def test_negativity_product(self):
        from tanglement.quantum import negativity
        e0 = np.array([1, 0], dtype=complex)
        rho = np.outer(np.kron(e0, e0), np.kron(e0, e0).conj())
        assert_allclose(negativity(rho), 0.0, atol=1e-10)

    def test_concurrence_bell(self):
        from tanglement.quantum import bell_state, concurrence
        assert_allclose(concurrence(bell_state('phi_plus')), 1.0, atol=1e-10)

    def test_concurrence_product(self):
        from tanglement.quantum import concurrence
        e0 = np.array([1, 0], dtype=complex)
        rho = np.outer(np.kron(e0, e0), np.kron(e0, e0).conj())
        assert_allclose(concurrence(rho), 0.0, atol=1e-10)

    def test_concurrence_werner(self):
        from tanglement.quantum import werner_state, concurrence
        # Werner(p) concurrence = max(0, (3p-1)/2)
        assert_allclose(concurrence(werner_state(1.0)), 1.0, atol=1e-6)
        assert_allclose(concurrence(werner_state(0.5)), 0.25, atol=1e-6)
        assert_allclose(concurrence(werner_state(0.3)), 0.0, atol=1e-6)

    def test_eof_monotone(self):
        from tanglement.quantum import werner_state, entanglement_of_formation
        eof_high = entanglement_of_formation(werner_state(0.9))
        eof_low = entanglement_of_formation(werner_state(0.5))
        eof_sep = entanglement_of_formation(werner_state(0.3))
        assert eof_high > eof_low > 0
        assert_allclose(eof_sep, 0.0, atol=1e-10)

    def test_partial_transpose_ppt(self):
        from tanglement.quantum import bell_state, partial_transpose
        rho_pt = partial_transpose(bell_state('phi_plus'))
        eigs = np.linalg.eigvalsh(rho_pt)
        assert np.min(eigs) < -0.4  # entangled state has negative eigenvalue

    def test_project_to_physical(self):
        from tanglement.quantum import project_to_physical
        # A non-physical matrix
        bad = np.diag([0.5, 0.3, 0.1, -0.1]).astype(complex)
        rho = project_to_physical(bad)
        eigs = np.linalg.eigvalsh(rho)
        assert np.all(eigs >= -1e-10)
        assert_allclose(np.trace(rho), 1.0, atol=1e-10)


class TestPolytopeMisc:
    def test_nearest_local_point_inside(self):
        from tanglement.polytope import nearest_local_point
        c = np.array([0.5, 0.5, 0.5, 0.5])
        proj = nearest_local_point(c)
        assert_allclose(proj, c, atol=0.05)

    def test_nearest_local_point_outside(self):
        from tanglement.polytope import nearest_local_point, in_local_polytope
        c = np.array([1/np.sqrt(2)] * 3 + [-1/np.sqrt(2)])
        proj = nearest_local_point(c)
        assert in_local_polytope(proj)


class TestQuantumPrimitives:
    def test_bloch_vector_z(self):
        from tanglement.quantum import bloch_vector
        assert_allclose(bloch_vector(0.0), [0, 0, 1], atol=1e-10)

    def test_bloch_vector_x(self):
        from tanglement.quantum import bloch_vector
        assert_allclose(bloch_vector(np.pi/2, 0.0), [1, 0, 0], atol=1e-10)

    def test_measurement_eigenvalues(self):
        from tanglement.quantum import measurement_operator
        M = measurement_operator(np.pi/4, 0)
        eigs = np.sort(np.linalg.eigvalsh(M))
        assert_allclose(eigs, [-1, 1], atol=1e-10)


class TestPhysicalTomographic:
    """Tests for PhysicalTomographic (Cholesky-parametrized density matrix inference)."""

    @pytest.mark.slow
    @pytest.mark.pymc
    def test_phi_plus_entanglement(self):
        """Maximally entangled state should yield concurrence ~ 1, negativity ~ 0.5."""
        from tanglement.quantum import bell_state, generate_data, tomographic_settings
        from tanglement.inference import PhysicalTomographic
        rho = bell_state('phi_plus')
        data = generate_data(rho, tomographic_settings(), 3000, rng=np.random.default_rng(42))
        r = PhysicalTomographic().fit(data, n_draws=500, n_tune=500, n_chains=1)
        assert r.extra['concurrence']['mean'] > 0.7
        assert r.extra['negativity']['mean'] > 0.3
        assert r.extra['purity']['mean'] > 0.8
        assert r.chsh['P_violates'] > 0.90

    @pytest.mark.slow
    @pytest.mark.pymc
    def test_product_state_separable(self):
        """Product state should yield concurrence ~ 0, negativity ~ 0."""
        from tanglement.quantum import generate_data, tomographic_settings
        from tanglement.inference import PhysicalTomographic
        e0 = np.array([1, 0], dtype=complex)
        rho = np.outer(np.kron(e0, e0), np.kron(e0, e0).conj())
        data = generate_data(rho, tomographic_settings(), 3000, rng=np.random.default_rng(42))
        r = PhysicalTomographic().fit(data, n_draws=500, n_tune=500, n_chains=1)
        assert r.extra['concurrence']['mean'] < 0.15
        assert r.extra['negativity']['mean'] < 0.1

    @pytest.mark.slow
    @pytest.mark.pymc
    def test_rho_samples_physical(self):
        """Every posterior ρ sample must be positive semidefinite with trace 1."""
        from tanglement.quantum import bell_state, generate_data, tomographic_settings
        from tanglement.inference import PhysicalTomographic
        data = generate_data(bell_state('psi_minus'), tomographic_settings(), 2000,
                             rng=np.random.default_rng(42))
        r = PhysicalTomographic().fit(data, n_draws=300, n_tune=300, n_chains=1)
        for s in range(min(50, r.extra['rho_samples'].shape[0])):
            rho_s = r.extra['rho_samples'][s]
            eigs = np.linalg.eigvalsh(rho_s)
            assert np.all(eigs > -1e-8), f"Negative eigenvalue at sample {s}: {eigs}"
            assert_allclose(np.trace(rho_s), 1.0, atol=1e-6)

    @pytest.mark.slow
    @pytest.mark.pymc
    def test_werner_concurrence_accuracy(self):
        """Werner(0.85) concurrence should be near (3*0.85-1)/2 = 0.775."""
        from tanglement.quantum import werner_state, generate_data, tomographic_settings
        from tanglement.inference import PhysicalTomographic
        data = generate_data(werner_state(0.85), tomographic_settings(), 5000,
                             rng=np.random.default_rng(42))
        r = PhysicalTomographic().fit(data, n_draws=600, n_tune=600, n_chains=1)
        true_conc = (3 * 0.85 - 1) / 2  # = 0.775
        assert abs(r.extra['concurrence']['mean'] - true_conc) < 0.15

    @pytest.mark.slow
    @pytest.mark.pymc
    def test_eof_positive_for_entangled(self):
        """Entangled state should have E_F > 0."""
        from tanglement.quantum import bell_state, generate_data, tomographic_settings
        from tanglement.inference import PhysicalTomographic
        data = generate_data(bell_state('phi_plus'), tomographic_settings(), 3000,
                             rng=np.random.default_rng(42))
        r = PhysicalTomographic().fit(data, n_draws=400, n_tune=400, n_chains=1)
        assert r.extra['eof']['mean'] > 0.3

    @pytest.mark.slow
    @pytest.mark.pymc
    def test_requires_9_settings(self):
        """Should raise ValueError with fewer than 9 settings."""
        from tanglement.quantum import bell_state, generate_data, chsh_settings
        from tanglement.inference import PhysicalTomographic
        data = generate_data(bell_state('phi_plus'), chsh_settings(), 1000,
                             rng=np.random.default_rng(42))
        with pytest.raises(ValueError, match="9 settings"):
            PhysicalTomographic().fit(data)


class TestFrequentistBellTest:
    """Tests for FrequentistBellTest (classical hypothesis testing)."""

    def test_rejects_bell_state_asymptotic(self):
        """Asymptotic test should reject H₀ for |Φ+⟩ with enough data."""
        from tanglement.quantum import bell_state, generate_data, chsh_settings
        from tanglement.inference import FrequentistBellTest
        data = generate_data(bell_state('phi_plus'), chsh_settings(), 5000,
                             rng=np.random.default_rng(42))
        r = FrequentistBellTest(alpha=0.05).test(data, method='asymptotic')
        assert r.reject_h0
        assert r.p_value < 0.001
        assert r.test_statistic > 2.0

    def test_no_false_rejection_product(self):
        """Product state should not reject H₀."""
        from tanglement.quantum import generate_data, chsh_settings
        from tanglement.inference import FrequentistBellTest
        e0 = np.array([1, 0], dtype=complex)
        rho = np.outer(np.kron(e0, e0), np.kron(e0, e0).conj())
        data = generate_data(rho, chsh_settings(), 5000, rng=np.random.default_rng(42))
        r = FrequentistBellTest(alpha=0.05).test(data, method='asymptotic')
        assert not r.reject_h0
        assert r.p_value > 0.05

    def test_rejects_bell_state_bootstrap(self):
        """Bootstrap test should reject H₀ for |Φ+⟩."""
        from tanglement.quantum import bell_state, generate_data, chsh_settings
        from tanglement.inference import FrequentistBellTest
        data = generate_data(bell_state('phi_plus'), chsh_settings(), 5000,
                             rng=np.random.default_rng(42))
        r = FrequentistBellTest(alpha=0.05).test(data, method='bootstrap',
                                                  n_bootstrap=5000,
                                                  rng=np.random.default_rng(123))
        assert r.reject_h0
        assert r.p_value < 0.01

    def test_no_false_rejection_bootstrap(self):
        """Product state bootstrap test should not reject H₀."""
        from tanglement.quantum import generate_data, chsh_settings
        from tanglement.inference import FrequentistBellTest
        e0 = np.array([1, 0], dtype=complex)
        rho = np.outer(np.kron(e0, e0), np.kron(e0, e0).conj())
        data = generate_data(rho, chsh_settings(), 5000, rng=np.random.default_rng(42))
        r = FrequentistBellTest(alpha=0.05).test(data, method='bootstrap',
                                                  n_bootstrap=5000,
                                                  rng=np.random.default_rng(123))
        assert not r.reject_h0

    def test_werner_threshold_detection(self):
        """Werner(0.85) should be detectable; Werner(0.5) should not."""
        from tanglement.quantum import werner_state, generate_data, chsh_settings
        from tanglement.inference import FrequentistBellTest
        # Above threshold
        data_high = generate_data(werner_state(0.85), chsh_settings(), 10000,
                                  rng=np.random.default_rng(42))
        r_high = FrequentistBellTest(alpha=0.05).test(data_high)
        assert r_high.reject_h0
        # Below threshold
        data_low = generate_data(werner_state(0.5), chsh_settings(), 10000,
                                 rng=np.random.default_rng(42))
        r_low = FrequentistBellTest(alpha=0.05).test(data_low)
        assert not r_low.reject_h0

    def test_observed_S_structure(self):
        """Result should contain all 4 CHSH values and standard errors."""
        from tanglement.quantum import bell_state, generate_data, chsh_settings
        from tanglement.inference import FrequentistBellTest
        data = generate_data(bell_state('phi_plus'), chsh_settings(), 1000,
                             rng=np.random.default_rng(42))
        r = FrequentistBellTest().test(data)
        assert set(r.observed_S.keys()) == {'S1', 'S2', 'S3', 'S4'}
        assert set(r.se_S.keys()) == {'S1', 'S2', 'S3', 'S4'}
        assert all(v > 0 for v in r.se_S.values())

    def test_nearest_local_is_local(self):
        """The nearest local point should lie inside the local polytope."""
        from tanglement.quantum import bell_state, generate_data, chsh_settings
        from tanglement.inference import FrequentistBellTest
        from tanglement.polytope import in_local_polytope
        data = generate_data(bell_state('phi_plus'), chsh_settings(), 1000,
                             rng=np.random.default_rng(42))
        r = FrequentistBellTest().test(data)
        assert in_local_polytope(r.nearest_local)

    def test_confidence_interval_contains_statistic(self):
        """The CI should bracket the test statistic for asymptotic method."""
        from tanglement.quantum import bell_state, generate_data, chsh_settings
        from tanglement.inference import FrequentistBellTest
        data = generate_data(bell_state('phi_plus'), chsh_settings(), 2000,
                             rng=np.random.default_rng(42))
        r = FrequentistBellTest().test(data, method='asymptotic')
        assert r.ci[0] <= r.test_statistic <= r.ci[1]

    def test_power_increases_with_N(self):
        """p-value should decrease as sample size grows."""
        from tanglement.quantum import bell_state, generate_data, chsh_settings
        from tanglement.inference import FrequentistBellTest
        rho = bell_state('phi_plus')
        p_vals = []
        for N in [500, 2000, 8000]:
            data = generate_data(rho, chsh_settings(), N, rng=np.random.default_rng(42))
            r = FrequentistBellTest().test(data)
            p_vals.append(r.p_value)
        assert p_vals[1] < p_vals[0]
        assert p_vals[2] < p_vals[1]

    def test_invalid_method_raises(self):
        """Should raise ValueError for unknown method."""
        from tanglement.quantum import bell_state, generate_data, chsh_settings
        from tanglement.inference import FrequentistBellTest
        data = generate_data(bell_state('phi_plus'), chsh_settings(), 100,
                             rng=np.random.default_rng(42))
        with pytest.raises(ValueError, match="method"):
            FrequentistBellTest().test(data, method='invalid')

    def test_bayesian_frequentist_consistency(self):
        """Bayesian P(S>2) and frequentist p-value should agree directionally."""
        from tanglement.quantum import bell_state, generate_data, chsh_settings
        from tanglement.inference import BinaryConjugate, FrequentistBellTest
        rho = bell_state('phi_plus')
        data = generate_data(rho, chsh_settings(), 3000, rng=np.random.default_rng(42))
        bayes = BinaryConjugate().fit(data, n_samples=10000)
        freq = FrequentistBellTest().test(data)
        # Both should strongly detect violation
        assert bayes.chsh['P_violates'] > 0.99
        assert freq.reject_h0
        assert freq.p_value < 0.001

    def test_outcome_probabilities_sum_to_one(self):
        from tanglement.quantum import bell_state, outcome_probabilities
        probs = outcome_probabilities(bell_state('phi_plus'), 0, 0, np.pi/4, 0)
        assert_allclose(sum(probs.values()), 1.0, atol=1e-10)

    def test_outcome_probabilities_consistent(self):
        from tanglement.quantum import bell_state, outcome_probabilities, quantum_expectation
        rho = bell_state('psi_minus')
        args = (np.pi/3, 0.0, np.pi/6, 0.0)
        probs = outcome_probabilities(rho, *args)
        E = sum(x * y * p for (x, y), p in probs.items())
        assert_allclose(E, quantum_expectation(rho, *args), atol=1e-10)


class TestTomographicStd:
    """Verify T_std bug fix: Tomographic model must return nonzero T_std."""
    @pytest.mark.slow
    @pytest.mark.pymc
    def test_t_std_nonzero(self):
        from tanglement.quantum import bell_state, generate_data, tomographic_settings
        from tanglement.inference import Tomographic
        rho = bell_state('phi_plus')
        data = generate_data(rho, tomographic_settings(), 3000, rng=np.random.default_rng(42))
        r = Tomographic(fit_bloch=True).fit(data, n_draws=500, n_tune=500, n_chains=1)
        # Correlation tensor std should be nonzero
        assert np.any(r.extra['T_std'][1:, 1:] > 0)
        # Bloch vector std should be nonzero
        assert np.any(r.extra['T_std'][1:, 0] > 0)
        assert np.any(r.extra['T_std'][0, 1:] > 0)


class TestBootstrapExtra:
    def test_chsh_samples_present(self):
        from tanglement.quantum import bell_state, generate_data, chsh_settings
        from tanglement.inference import BayesianBootstrap
        data = generate_data(bell_state('phi_plus'), chsh_settings(), 500, rng=np.random.default_rng(42))
        r = BayesianBootstrap().fit(data, n_samples=1000)
        assert 'chsh_samples' in r.extra
        assert len(r.extra['chsh_samples']) == 1000


# ═══════════════════════════════════════════════════════════════════
#  Tau-pair analysis pipeline tests
# ═══════════════════════════════════════════════════════════════════

class TestTauClassification:
    def test_pinu(self):
        from tanglement.tautau import classify_decay
        name, alpha, _ = classify_decay([-211, -16])
        assert name == 'pi_nu'
        assert alpha == 1.0

    def test_rhonu(self):
        from tanglement.tautau import classify_decay
        name, alpha, _ = classify_decay([211, 111, 16])
        assert name == 'rho_nu'
        assert_allclose(alpha, 0.46)

    def test_leptonic(self):
        from tanglement.tautau import classify_decay
        name, alpha, _ = classify_decay([11, -12, 16])
        assert name == 'e_nu_nu'
        assert_allclose(alpha, 0.333)

    def test_unknown(self):
        from tanglement.tautau import classify_decay
        name, alpha, _ = classify_decay([321, 311, 16])
        assert name == 'other'
        assert alpha == 0.0


class TestBoostAndAngles:
    def test_boost_preserves_mass(self):
        from tanglement.tautau import boost_to_rest_frame
        # Tau along z with known momentum; pion should keep its mass
        p4_tau = (22.8, 0, 0, 22.73)
        p4_pi = (15.0, 0, 0.5, 14.99)
        E, px, py, pz = boost_to_rest_frame(p4_tau, p4_pi)
        m2_orig = p4_pi[0]**2 - p4_pi[1]**2 - p4_pi[2]**2 - p4_pi[3]**2
        m2_boost = E**2 - px**2 - py**2 - pz**2
        assert_allclose(m2_boost, m2_orig, atol=1e-6)

    def test_boost_at_rest_is_identity(self):
        from tanglement.tautau import boost_to_rest_frame
        p4_tau = (1.777, 0, 0, 0)
        p4_pi = (0.896, 0.3, -0.5, 0.7)
        result = boost_to_rest_frame(p4_tau, p4_pi)
        assert_allclose(result, p4_pi, atol=1e-10)

    def test_rest_frame_angles_forward(self):
        from tanglement.tautau import compute_rest_frame_angles
        # Tau along z; hadron along z in lab => forward in RF
        p4_tau = (22.8, 0, 0, 22.73)
        p4_had = (20.0, 0, 0, 20.0)
        p4_beam = (45.6, 0, 0, 45.6)
        cos_t, phi = compute_rest_frame_angles(p4_tau, p4_had, p4_beam)
        assert cos_t > 0.5


class TestTauFanoExtraction:
    def test_singlet_extraction(self):
        """Synthetic singlet data -> correct Fano coefficients."""
        from tanglement.collider import ColliderSimulator
        from tanglement.quantum import bell_state
        from tanglement.tautau import bayesian_fano_with_alpha

        sim = ColliderSimulator()
        rho = bell_state('psi_minus')
        ct1, p1, ct2, p2 = sim.generate(rho, 5000, rng=np.random.default_rng(42))
        raw = sim.extract(ct1, p1, ct2, p2)

        result = bayesian_fano_with_alpha(raw, 1.0, 1.0, n_posterior=3000,
                                           rng=np.random.default_rng(123))
        T = result.extra['T_mean']
        assert T[1, 1] < -0.8
        assert T[2, 2] < -0.8
        assert T[3, 3] < -0.8

    def test_alpha_correction_widens_posterior(self):
        """Dividing by small alpha should widen the posterior."""
        from tanglement.collider import ColliderSimulator
        from tanglement.quantum import bell_state
        from tanglement.tautau import bayesian_fano_with_alpha

        sim = ColliderSimulator()
        rho = bell_state('psi_minus')
        ct1, p1, ct2, p2 = sim.generate(rho, 2000, rng=np.random.default_rng(42))
        raw = sim.extract(ct1, p1, ct2, p2)

        r1 = bayesian_fano_with_alpha(raw, 1.0, 1.0, n_posterior=2000,
                                       rng=np.random.default_rng(1))
        r2 = bayesian_fano_with_alpha(raw, 0.46, 0.46, n_posterior=2000,
                                       rng=np.random.default_rng(1))
        std1 = r1.extra['T_std'][1, 1]
        std2 = r2.extra['T_std'][1, 1]
        assert std2 > std1 * 3

    def test_alpha_correction_preserves_mean(self):
        """Alpha correction should scale the mean by 1/alpha^2."""
        from tanglement.collider import ColliderSimulator
        from tanglement.quantum import bell_state
        from tanglement.tautau import bayesian_fano_with_alpha

        sim = ColliderSimulator()
        rho = bell_state('psi_minus')
        ct1, p1, ct2, p2 = sim.generate(rho, 5000, rng=np.random.default_rng(42))
        raw = sim.extract(ct1, p1, ct2, p2)

        r1 = bayesian_fano_with_alpha(raw, 1.0, 1.0, n_posterior=5000,
                                       rng=np.random.default_rng(1))
        r2 = bayesian_fano_with_alpha(raw, 0.46, 0.46, n_posterior=5000,
                                       rng=np.random.default_rng(1))

        ratio = r2.extra['T_mean'][1, 1] / r1.extra['T_mean'][1, 1]
        expected_ratio = 1.0 / (0.46**2)
        assert_allclose(ratio, expected_ratio, rtol=0.15)

    def test_zpole_entanglement_detected(self):
        """Z-pole density matrix should show entanglement."""
        from tanglement.collider import ColliderSimulator
        from tanglement.quantum import rho_from_fano, project_to_physical
        from tanglement.tautau import bayesian_fano_with_alpha

        T_Z = np.zeros((4, 4))
        T_Z[0, 0] = 1.0
        T_Z[1, 1] = -0.99
        T_Z[2, 2] = -0.33
        T_Z[3, 3] = -0.33
        rho = project_to_physical(rho_from_fano(T_Z))

        sim = ColliderSimulator()
        ct1, p1, ct2, p2 = sim.generate(rho, 8000, rng=np.random.default_rng(42))
        raw = sim.extract(ct1, p1, ct2, p2)

        result = bayesian_fano_with_alpha(raw, 1.0, 1.0, n_posterior=5000,
                                           rng=np.random.default_rng(123))
        assert result.chsh['P_violates'] > 0.95


class TestCauchyDiagnostic:
    def test_small_n_wide_ci(self):
        """With N=30 and alpha=0.12, the CI should be very wide."""
        from tanglement.collider import ColliderSimulator
        from tanglement.quantum import bell_state
        from tanglement.tautau import bayesian_fano_with_alpha

        sim = ColliderSimulator()
        rho = bell_state('psi_minus')
        ct1, p1, ct2, p2 = sim.generate(rho, 30, rng=np.random.default_rng(42))
        raw = sim.extract(ct1, p1, ct2, p2)

        result = bayesian_fano_with_alpha(raw, 0.12, 0.12, n_posterior=3000,
                                           rng=np.random.default_rng(1))
        ci = result.correlations[0]['ci']
        ci_width = ci[1] - ci[0]
        assert ci_width > 30

    def test_bayesian_ci_covers_truth(self):
        """Bayesian 95% CI should cover the true C_zz."""
        from tanglement.collider import ColliderSimulator
        from tanglement.quantum import bell_state, fano_decomposition
        from tanglement.tautau import bayesian_fano_with_alpha

        rho = bell_state('psi_minus')
        T_true = fano_decomposition(rho)
        true_C_zz = T_true[1, 1]

        sim = ColliderSimulator()
        ct1, p1, ct2, p2 = sim.generate(rho, 500, rng=np.random.default_rng(42))
        raw = sim.extract(ct1, p1, ct2, p2)

        result = bayesian_fano_with_alpha(raw, 1.0, 1.0, n_posterior=5000,
                                           rng=np.random.default_rng(1))
        ci = result.correlations[0]['ci']
        assert ci[0] <= true_C_zz <= ci[1]
