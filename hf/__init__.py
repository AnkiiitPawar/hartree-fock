"""Hartree–Fock from first principles: Gaussian integrals, SCF, MP2 and two-electron full CI."""

from .molecule import Molecule, ammonia, h2, heh_plus, methane, water
from .scf import compute_integrals, dipole, fci_two_electron, mp2, mulliken, rhf, uhf

__all__ = ["Molecule", "h2", "heh_plus", "water", "methane", "ammonia", "compute_integrals", "rhf", "uhf", "mp2",
           "fci_two_electron", "dipole", "mulliken"]
