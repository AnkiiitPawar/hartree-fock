"""Restricted and unrestricted Hartree–Fock with DIIS, plus MP2 and two-electron full CI.

RHF (closed shell, n/2 doubly occupied orbitals). With density D = C_occ C_occᵀ:
    F = H + 2J(D) - K(D),   J_μν = Σ (μν|λσ) D_λσ,   K_μν = Σ (μλ|νσ) D_λσ
    F C = S C ε  (Roothaan–Hall), solved by symmetric orthogonalisation X = S^{-1/2}
    E = Σ D_μν (H_μν + F_μν) + E_nuc
DIIS (Pulay, 1980) extrapolates F from the last few iterations using the commutator
error e = F D S - S D F, which vanishes exactly at self-consistency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh

from .basis import sto3g
from .integrals import dipole_matrices, one_electron, two_electron
from .molecule import Molecule


@dataclass
class Integrals:
    S: np.ndarray
    T: np.ndarray
    V: np.ndarray
    eri: np.ndarray
    basis: list
    e_nuc: float

    @property
    def H(self) -> np.ndarray:
        return self.T + self.V


def compute_integrals(mol: Molecule, zetas: dict | None = None) -> Integrals:
    basis = sto3g(mol.atoms, zetas)
    S, T, V = one_electron(basis, mol.atoms)
    return Integrals(S, T, V, two_electron(basis), basis, mol.nuclear_repulsion())


def orthogonaliser(S: np.ndarray) -> np.ndarray:
    w, U = np.linalg.eigh(S)
    return U @ np.diag(w ** -0.5) @ U.T


def coulomb(eri, D):
    return np.einsum("pqrs,rs->pq", eri, D)


def exchange(eri, D):
    return np.einsum("prqs,rs->pq", eri, D)


class DIIS:
    def __init__(self, size: int = 6) -> None:
        self.size, self.focks, self.errors = size, [], []

    def extrapolate(self, F: np.ndarray, err: np.ndarray) -> np.ndarray:
        self.focks.append(F.copy())
        self.errors.append(err.ravel().copy())
        self.focks, self.errors = self.focks[-self.size:], self.errors[-self.size:]
        m = len(self.focks)
        if m < 2:
            return F
        B = -np.ones((m + 1, m + 1))
        B[m, m] = 0.0
        B[:m, :m] = np.array([[ei @ ej for ej in self.errors] for ei in self.errors])
        rhs = np.zeros(m + 1)
        rhs[m] = -1.0
        try:
            c = np.linalg.solve(B, rhs)[:m]
        except np.linalg.LinAlgError:
            return F
        return sum(ci * Fi for ci, Fi in zip(c, self.focks))


@dataclass
class RHFResult:
    energy: float
    electronic: float
    orbital_energies: np.ndarray
    C: np.ndarray
    D: np.ndarray
    F: np.ndarray
    n_occ: int
    iterations: int
    history: list = field(default_factory=list)  # (energy, |FDS - SDF|) per iteration
    ints: Integrals | None = None


def rhf(mol: Molecule, ints: Integrals | None = None, use_diis: bool = True, max_iter: int = 200,
        e_tol: float = 1e-12, d_tol: float = 1e-10, zetas: dict | None = None) -> RHFResult:
    if mol.n_electrons % 2:
        raise ValueError("RHF needs an even number of electrons; use uhf()")
    ints = ints or compute_integrals(mol, zetas)
    S, H, eri = ints.S, ints.H, ints.eri
    n_occ = mol.n_electrons // 2
    X = orthogonaliser(S)
    eps, C = eigh(X.T @ H @ X)
    C = X @ C
    D = C[:, :n_occ] @ C[:, :n_occ].T          # core-Hamiltonian guess
    diis = DIIS() if use_diis else None
    e_old, history = 0.0, []
    for it in range(1, max_iter + 1):
        F = H + 2 * coulomb(eri, D) - exchange(eri, D)
        e_el = float(np.sum(D * (H + F)))
        err = X.T @ (F @ D @ S - S @ D @ F) @ X
        history.append((e_el + ints.e_nuc, float(np.abs(err).max())))
        if diis is not None:
            F = diis.extrapolate(F, err)
        eps, Cp = eigh(X.T @ F @ X)
        C = X @ Cp
        D_new = C[:, :n_occ] @ C[:, :n_occ].T
        if abs(e_el - e_old) < e_tol and np.abs(D_new - D).max() < d_tol:
            D = D_new
            break
        D, e_old = D_new, e_el
    F = H + 2 * coulomb(eri, D) - exchange(eri, D)
    e_el = float(np.sum(D * (H + F)))
    return RHFResult(e_el + ints.e_nuc, e_el, eps, C, D, F, n_occ, it, history, ints)


@dataclass
class UHFResult:
    energy: float
    Ca: np.ndarray
    Cb: np.ndarray
    Da: np.ndarray
    Db: np.ndarray
    s2: float
    iterations: int


def uhf(mol: Molecule, n_alpha: int | None = None, ints: Integrals | None = None, guess_mix: float = 0.0,
        max_iter: int = 500, e_tol: float = 1e-12) -> UHFResult:
    """Unrestricted HF: separate α and β orbitals,
        Fᵅ = H + J(Dᵅ + Dᵝ) - K(Dᵅ),  Fᵝ = H + J(Dᵅ + Dᵝ) - K(Dᵝ).
    ``guess_mix`` rotates the α HOMO toward the LUMO (and β the other way) to break the
    spin symmetry, which lets UHF find the lower, spin-polarised solution where one exists.
    """
    ints = ints or compute_integrals(mol)
    S, H, eri = ints.S, ints.H, ints.eri
    ne = mol.n_electrons
    na = n_alpha if n_alpha is not None else (ne + 1) // 2
    nb = ne - na
    X = orthogonaliser(S)
    _, C = eigh(X.T @ H @ X)
    C = X @ C
    Ca, Cb = C.copy(), C.copy()
    if guess_mix and C.shape[1] > na:
        h, l = na - 1, na
        c, s = np.cos(guess_mix), np.sin(guess_mix)
        Ca[:, h], Ca[:, l] = c * C[:, h] + s * C[:, l], -s * C[:, h] + c * C[:, l]
        Cb[:, h], Cb[:, l] = c * C[:, h] - s * C[:, l], s * C[:, h] + c * C[:, l]
    Da, Db = Ca[:, :na] @ Ca[:, :na].T, Cb[:, :nb] @ Cb[:, :nb].T
    e_old = 0.0
    for it in range(1, max_iter + 1):
        J = coulomb(eri, Da + Db)
        Fa, Fb = H + J - exchange(eri, Da), H + J - exchange(eri, Db)
        e = 0.5 * float(np.sum((Da + Db) * H) + np.sum(Da * Fa) + np.sum(Db * Fb))
        _, Ca = eigh(X.T @ Fa @ X)
        _, Cb = eigh(X.T @ Fb @ X)
        Ca, Cb = X @ Ca, X @ Cb
        Da_new, Db_new = Ca[:, :na] @ Ca[:, :na].T, Cb[:, :nb] @ Cb[:, :nb].T
        converged = abs(e - e_old) < e_tol and max(np.abs(Da_new - Da).max(), np.abs(Db_new - Db).max()) < 1e-9
        Da, Db, e_old = 0.5 * (Da + Da_new), 0.5 * (Db + Db_new), e  # damping keeps the iteration stable
        if converged:
            Da, Db = Da_new, Db_new
            break
    # ⟨S²⟩ = Sz(Sz+1) + N_β - Σ |⟨iα|jβ⟩|²
    sz = 0.5 * (na - nb)
    ov = Ca[:, :na].T @ S @ Cb[:, :nb]
    s2 = sz * (sz + 1) + nb - float(np.sum(ov**2))
    return UHFResult(e_old + ints.e_nuc, Ca, Cb, Da, Db, s2, it)


def mo_eri(eri: np.ndarray, C: np.ndarray) -> np.ndarray:
    """(pq|rs) in the MO basis by four quarter-transformations, O(N⁵)."""
    return np.einsum("pi,qj,pqrs,rk,sl->ijkl", C, C, eri, C, C, optimize=True)


def mp2(res: RHFResult) -> float:
    """Second-order Møller–Plesset correlation energy for a closed shell:
        E⁽²⁾ = Σ_{ijab} (ia|jb) [2(ia|jb) - (ib|ja)] / (εᵢ + εⱼ - εₐ - ε_b)."""
    g = mo_eri(res.ints.eri, res.C)
    o = res.n_occ
    eps = res.orbital_energies
    iajb = g[:o, o:, :o, o:]
    denom = eps[:o, None, None, None] - eps[None, o:, None, None] + eps[None, None, :o, None] - eps[None, None, None, o:]
    return float(np.sum(iajb * (2 * iajb - iajb.transpose(0, 3, 2, 1)) / denom))


def fci_two_electron(res: RHFResult) -> float:
    """Exact (full CI) ground-state energy for a two-electron system in the given basis.

    One α and one β electron: the determinants |i_α j_β⟩ span the whole M_S = 0 space, and
    since the two spins are different there is no exchange term:
        ⟨i_α j_β| H |k_α l_β⟩ = h_ik δ_jl + δ_ik h_jl + (ik|jl)
    The lowest eigenvalue of this n² × n² matrix is the singlet ground state.
    """
    g = mo_eri(res.ints.eri, res.C)
    h = res.C.T @ res.ints.H @ res.C
    n = h.shape[0]
    eye = np.eye(n)
    M = (np.einsum("ik,jl->ijkl", h, eye) + np.einsum("ik,jl->ijkl", eye, h) + g.transpose(0, 2, 1, 3))
    return float(np.linalg.eigvalsh(M.reshape(n * n, n * n))[0]) + res.ints.e_nuc


def dipole(res: RHFResult, mol: Molecule) -> np.ndarray:
    """Dipole moment (atomic units) about the origin: Σ Z_A R_A - 2 Σ D_μν ⟨μ|r|ν⟩."""
    M = dipole_matrices(res.ints.basis)
    elec = -2 * np.einsum("kij,ij->k", M, res.D)
    nuc = sum(Z * xyz for Z, xyz in mol.atoms)
    return elec + nuc


def mulliken(res: RHFResult, mol: Molecule) -> np.ndarray:
    pop = 2 * np.diag(res.D @ res.ints.S)
    charges = np.array([Z for Z, _ in mol.atoms], float)
    for mu, bf in enumerate(res.ints.basis):
        charges[bf.atom] -= pop[mu]
    return charges
