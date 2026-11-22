"""Contracted Cartesian Gaussian basis functions and the STO-3G basis set.

A primitive Cartesian Gaussian centred at A is
    g(r) = (x - Aₓ)ˡ (y - A_y)ᵐ (z - A_z)ⁿ exp(-α |r - A|²),
with normalisation
    N = (2α/π)^{3/4} (4α)^{L/2} / √((2l-1)!! (2m-1)!! (2n-1)!!),   L = l + m + n.
A basis function is a fixed linear combination ("contraction") of primitives sharing the
same centre and angular part, renormalised so that ⟨φ|φ⟩ = 1.

STO-3G exponents and coefficients are the standard published values (Hehre, Stewart &
Pople, 1969), identical to those distributed with common quantum chemistry programs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Exponents per shell; the coefficients are shared across elements (that is what STO-3G is:
# a least-squares fit of three Gaussians to one Slater function, rescaled by ζ²).
S_COEF = (0.15432897, 0.53532814, 0.44463454)
SP_S_COEF = (-0.09996723, 0.39951283, 0.70011547)
SP_P_COEF = (0.15591627, 0.60768372, 0.39195739)

STO3G = {
    1: {"1s": (3.42525091, 0.62391373, 0.16885540)},
    2: {"1s": (6.36242139, 1.15892300, 0.31364979)},
    6: {"1s": (71.6168370, 13.0450960, 3.5305122), "2sp": (2.9412494, 0.6834831, 0.2222899)},
    7: {"1s": (99.1061690, 18.0523120, 4.8856602), "2sp": (3.7804559, 0.8784966, 0.2857144)},
    8: {"1s": (130.7093200, 23.8088610, 6.4436083), "2sp": (5.0331513, 1.1695961, 0.3803890)},
}
# Exponents of the ζ = 1 STO-3G 1s fit; scaled by ζ² for Szabo & Ostlund's examples.
STO3G_1S_ZETA1 = (2.227660584, 0.405771156, 0.1098175104)


def double_factorial(n: int) -> int:
    return 1 if n <= 0 else n * double_factorial(n - 2)


def primitive_norm(alpha: float, lmn: tuple[int, int, int]) -> float:
    l, m, n = lmn
    L = l + m + n
    num = (2 * alpha / np.pi) ** 0.75 * (4 * alpha) ** (L / 2)
    den = np.sqrt(double_factorial(2 * l - 1) * double_factorial(2 * m - 1) * double_factorial(2 * n - 1))
    return num / den


@dataclass
class BasisFunction:
    center: np.ndarray          # bohr
    lmn: tuple[int, int, int]
    exps: np.ndarray
    coefs: np.ndarray           # include primitive normalisation and overall renormalisation
    atom: int = 0
    label: str = ""

    @classmethod
    def contracted(cls, center, lmn, exps, coefs, atom=0, label="") -> "BasisFunction":
        exps = np.asarray(exps, float)
        c = np.asarray(coefs, float) * np.array([primitive_norm(a, lmn) for a in exps])
        bf = cls(np.asarray(center, float), tuple(lmn), exps, c, atom, label)
        from .integrals import overlap_contracted  # local import: integrals imports this module

        bf.coefs = bf.coefs / np.sqrt(overlap_contracted(bf, bf))
        return bf

    def value(self, pts: np.ndarray) -> np.ndarray:
        """Evaluate φ at points (n, 3), for plotting orbitals."""
        d = pts - self.center
        r2 = (d * d).sum(1)
        ang = d[:, 0] ** self.lmn[0] * d[:, 1] ** self.lmn[1] * d[:, 2] ** self.lmn[2]
        return ang * (self.coefs[None, :] * np.exp(-np.outer(r2, self.exps))).sum(1)


P_LMN = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
SYMBOL = {1: "H", 2: "He", 6: "C", 7: "N", 8: "O"}


def sto3g(atoms, zetas: dict | None = None) -> list[BasisFunction]:
    """STO-3G basis for a list of (Z, xyz). ``zetas`` optionally maps an atom index to a
    custom 1s Slater exponent ζ (only for H and He, as in Szabo & Ostlund's HeH⁺ example)."""
    basis = []
    for i, (Z, xyz) in enumerate(atoms):
        sym = SYMBOL[Z]
        if zetas and i in zetas:
            exps = np.array(STO3G_1S_ZETA1) * zetas[i] ** 2
            basis.append(BasisFunction.contracted(xyz, (0, 0, 0), exps, S_COEF, i, f"{sym}{i} 1s"))
            continue
        shells = STO3G[Z]
        basis.append(BasisFunction.contracted(xyz, (0, 0, 0), shells["1s"], S_COEF, i, f"{sym}{i} 1s"))
        if "2sp" in shells:
            basis.append(BasisFunction.contracted(xyz, (0, 0, 0), shells["2sp"], SP_S_COEF, i, f"{sym}{i} 2s"))
            for lmn, ax in zip(P_LMN, "xyz"):
                basis.append(BasisFunction.contracted(xyz, lmn, shells["2sp"], SP_P_COEF, i, f"{sym}{i} 2p{ax}"))
    return basis
