"""Reference values: energies from PySCF 2.15 (RHF/UHF/MP2/FCI, STO-3G, Cartesian functions,
SCF converged to 1e-13), and the water numbers also published with the Crawford group's
programming projects (SCF −74.942079928192, MP2 −0.049149636120, Eₙᵤ꜀ 8.002367061810450)."""

import numpy as np
import pytest

from hf import ammonia, dipole, fci_two_electron, h2, heh_plus, methane, mp2, mulliken, rhf, uhf, water
from hf.basis import STO3G, STO3G_1S_ZETA1, sto3g
from hf.integrals import boys, overlap_contracted
from hf.molecule import Molecule

RHF_REF = {
    "h2": (h2, -1.116714325063, -0.013157870053),
    "heh_plus": (heh_plus, -2.841836499287, -0.007238266770),
    "water": (water, -74.942079928192, -0.049149636120),
    "ammonia": (ammonia, -55.454038527135, -0.047172453998),
    "methane": (methane, -39.726743332498, -0.056459728963),
}


@pytest.fixture(scope="module")
def water_rhf():
    return rhf(water())


@pytest.mark.parametrize("name", list(RHF_REF))
def test_rhf_and_mp2_energies(name):
    make, e_ref, mp2_ref = RHF_REF[name]
    res = rhf(make())
    assert res.energy == pytest.approx(e_ref, abs=1e-9)
    assert mp2(res) == pytest.approx(mp2_ref, abs=1e-9)


def test_water_against_the_published_reference(water_rhf):
    assert water().nuclear_repulsion() == pytest.approx(8.002367061810450, abs=1e-12)
    assert water_rhf.energy == pytest.approx(-74.942079928192, abs=1e-10)
    assert mp2(water_rhf) == pytest.approx(-0.049149636120, abs=1e-10)


def test_water_dipole_and_charges(water_rhf):
    mu = dipole(water_rhf, water())
    assert np.linalg.norm(mu) == pytest.approx(0.6035212965, abs=1e-8)
    assert abs(mu[0]) < 1e-10 and abs(mu[2]) < 1e-10  # along the C2 axis
    q = mulliken(water_rhf, water())
    assert q.sum() == pytest.approx(0.0, abs=1e-10)
    assert q[0] == pytest.approx(-0.253146052405, abs=1e-8) and q[1] == pytest.approx(q[2])


def test_converged_density_is_self_consistent(water_rhf):
    S, D, F = water_rhf.ints.S, water_rhf.D, water_rhf.F
    assert np.allclose(D @ S @ D, D, atol=1e-10)                       # idempotent (pure state)
    assert 2 * np.trace(D @ S) == pytest.approx(10.0)                    # ten electrons
    assert np.abs(F @ D @ S - S @ D @ F).max() < 1e-8                    # [F, D] = 0


def test_diis_converges_faster_than_plain_iteration():
    with_diis = rhf(water(), use_diis=True)
    without = rhf(water(), use_diis=False)
    assert with_diis.energy == pytest.approx(without.energy, abs=1e-9)
    assert with_diis.iterations < without.iterations


def test_h2_full_ci_and_variational_ordering():
    res = rhf(h2(1.4))
    e_fci = fci_two_electron(res)
    assert e_fci == pytest.approx(-1.137275943617, abs=1e-9)
    assert e_fci < res.energy + mp2(res) < res.energy  # FCI below MP2 below HF, at equilibrium


def test_uhf_breaks_symmetry_only_when_stretched():
    near = uhf(h2(1.4), guess_mix=0.3)
    assert near.energy == pytest.approx(-1.116714325063, abs=1e-9) and near.s2 < 1e-8
    far = uhf(h2(5.0), guess_mix=0.3)
    assert far.energy == pytest.approx(-0.933587542756, abs=1e-9)
    assert far.s2 == pytest.approx(0.99459, abs=1e-4)  # spin-contaminated: half singlet, half triplet
    h_atom = uhf(Molecule.from_list([(1, (0, 0, 0))]), n_alpha=1).energy
    assert h_atom == pytest.approx(-0.466581849557, abs=1e-9)


def test_rhf_rejects_open_shells():
    with pytest.raises(ValueError):
        rhf(Molecule.from_list([(1, (0, 0, 0))]))


def test_boys_function():
    assert boys(0, 0.0) == pytest.approx(1.0)
    T = 3.7
    assert boys(0, T) == pytest.approx(0.5 * np.sqrt(np.pi / T) * __import__("math").erf(np.sqrt(T)))
    for n in range(4):  # downward recurrence F_n = (2T F_{n+1} + e^{-T}) / (2n + 1)
        assert boys(n, T) == pytest.approx((2 * T * boys(n + 1, T) + np.exp(-T)) / (2 * n + 1))


def test_s_overlap_matches_closed_form():
    """Two normalised s Gaussians: S = (2√(ab)/(a+b))^{3/2} exp(-ab R²/(a+b))."""
    from hf.basis import BasisFunction

    a, b, R = 0.8, 0.3, 1.7
    f = BasisFunction.contracted((0, 0, 0), (0, 0, 0), [a], [1.0])
    g = BasisFunction.contracted((0, 0, R), (0, 0, 0), [b], [1.0])
    expected = (2 * np.sqrt(a * b) / (a + b)) ** 1.5 * np.exp(-a * b * R**2 / (a + b))
    assert overlap_contracted(f, g) == pytest.approx(expected, rel=1e-12)


def test_basis_functions_are_normalised_and_zeta_scaling_is_sto3g():
    for bf in sto3g(water().atoms):
        assert overlap_contracted(bf, bf) == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(np.array(STO3G_1S_ZETA1) * 1.24**2, STO3G[1]["1s"], rtol=1e-8)


def test_integrals_match_pyscf():
    gto = pytest.importorskip("pyscf.gto")
    from hf.scf import compute_integrals

    for m in (water(), ammonia(), heh_plus(1.1)):
        ints = compute_integrals(m)
        mol = gto.M(atom=[(Z, tuple(xyz)) for Z, xyz in m.atoms], unit="bohr", basis="sto-3g", charge=m.charge,
                    cart=True, verbose=0)
        assert np.allclose(ints.S, mol.intor("int1e_ovlp"), atol=1e-12)
        assert np.allclose(ints.T, mol.intor("int1e_kin"), atol=1e-12)
        assert np.allclose(ints.V, mol.intor("int1e_nuc"), atol=1e-11)
        assert np.allclose(ints.eri, mol.intor("int2e"), atol=1e-12)
