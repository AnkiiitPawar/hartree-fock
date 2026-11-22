"""Molecules: nuclear charges and positions (bohr), total charge, and nuclear repulsion."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ANGSTROM = 1 / 0.52917721092  # bohr per ångström (CODATA 2010, as used by common programs)


@dataclass
class Molecule:
    atoms: list = field(default_factory=list)  # [(Z, np.array([x, y, z]) in bohr)]
    charge: int = 0

    @classmethod
    def from_list(cls, spec, charge: int = 0, unit: str = "bohr") -> "Molecule":
        scale = ANGSTROM if unit == "angstrom" else 1.0
        return cls([(int(Z), np.asarray(xyz, float) * scale) for Z, xyz in spec], charge)

    @property
    def n_electrons(self) -> int:
        return sum(Z for Z, _ in self.atoms) - self.charge

    def nuclear_repulsion(self) -> float:
        e = 0.0
        for i, (Zi, Ri) in enumerate(self.atoms):
            for Zj, Rj in self.atoms[:i]:
                e += Zi * Zj / np.linalg.norm(Ri - Rj)
        return e


def h2(R: float = 1.4) -> Molecule:
    return Molecule.from_list([(1, (0, 0, 0)), (1, (0, 0, R))])


def heh_plus(R: float = 1.4632) -> Molecule:
    return Molecule.from_list([(2, (0, 0, 0)), (1, (0, 0, R))], charge=1)


def water() -> Molecule:
    """The STO-3G water geometry used in the Crawford group's programming projects (bohr)."""
    return Molecule.from_list([(8, (0.000000000000, -0.143225816552, 0.000000000000)),
                               (1, (1.638036840407, 1.136548822547, -0.000000000000)),
                               (1, (-1.638036840407, 1.136548822547, -0.000000000000))])


def methane() -> Molecule:
    """Tetrahedral CH₄ with C–H = 1.089 Å."""
    d = 1.089 / np.sqrt(3)
    return Molecule.from_list([(6, (0, 0, 0)), (1, (d, d, d)), (1, (-d, -d, d)), (1, (-d, d, -d)),
                               (1, (d, -d, -d))], unit="angstrom")


def ammonia() -> Molecule:
    """NH₃ with N–H = 1.012 Å and H–N–H = 106.7°."""
    r, ang = 1.012, np.radians(106.7)
    # Place the three H on a cone around z with the requested H–N–H angle.
    cos_t = np.sqrt(1 - (2 / 3) * (1 - np.cos(ang)))
    sin_t = np.sqrt(1 - cos_t**2)
    hs = [(1, (r * sin_t * np.cos(phi), r * sin_t * np.sin(phi), -r * cos_t))
          for phi in (0, 2 * np.pi / 3, 4 * np.pi / 3)]
    return Molecule.from_list([(7, (0, 0, 0))] + hs, unit="angstrom")
