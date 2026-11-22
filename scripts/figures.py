"""README figures: H₂ dissociation by four methods, SCF convergence with and without DIIS,
and the valence orbitals of water.

usage: python scripts/figures.py     (needs matplotlib; about a minute)
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hf import Molecule, fci_two_electron, h2, mp2, rhf, uhf, water  # noqa: E402

DOCS = Path("docs")


def dissociation():
    Rs = np.round(np.arange(0.6, 6.01, 0.1), 2)
    rows = []
    for R in Rs:
        r = rhf(h2(R))
        rows.append((r.energy, uhf(h2(R), guess_mix=0.3).energy, r.energy + mp2(r), fci_two_electron(r)))
    E = np.array(rows)
    e_h = uhf(Molecule.from_list([(1, (0, 0, 0))]), n_alpha=1).energy
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    styles = [("RHF", "#c0504d", "-"), ("UHF", "#e6a23c", "--"), ("MP2", "#8064a2", "-."), ("full CI", "#1f5f99", "-")]
    for k, (label, color, ls) in enumerate(styles):
        ax.plot(Rs, E[:, k], ls, color=color, lw=2 if k in (0, 3) else 1.6, label=label)
    ax.axhline(2 * e_h, color="#888", lw=0.8, ls=":")
    ax.text(5.95, 2 * e_h + 0.01, "2 × E(H atom), the correct limit", ha="right", fontsize=8, color="#555")
    split = Rs[np.argmax(E[:, 0] - E[:, 1] > 1e-6)]
    ax.axvline(split, color="#e6a23c", lw=0.7, alpha=0.7)
    ax.text(split + 0.05, -1.13, f"UHF departs from RHF\nnear R = {split:.1f} bohr", fontsize=8, color="#b07a18")
    ax.set(xlabel="H–H distance R (bohr)", ylabel="total energy (hartree)", ylim=(-1.16, -0.55),
           title="H₂ in STO-3G: restricted HF fails to dissociate")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(DOCS / "h2_dissociation.png", dpi=130)
    i_min = [int(np.argmin(E[:, k])) for k in range(4)]
    return {label: (Rs[i], E[i, k]) for k, ((label, _, _), i) in enumerate(zip(styles, i_min))}, e_h


def convergence():
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    final = rhf(water()).energy
    for use, label, color in ((False, "plain Roothaan iteration", "#c0504d"), (True, "with DIIS", "#1f5f99")):
        res = rhf(water(), use_diis=use)
        err = [max(abs(e - final), 1e-16) for e, _ in res.history]
        ax.semilogy(range(1, len(err) + 1), err, "o-", ms=3, color=color, label=f"{label} ({res.iterations} iterations)")
    ax.set(xlabel="iteration", ylabel="|E − E_converged| (hartree)", title="SCF convergence, water / STO-3G")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(DOCS / "scf_convergence.png", dpi=130)


def water_orbitals():
    mol = water()
    res = rhf(mol)
    basis = res.ints.basis
    names = ["1a₁", "2a₁", "1b₂", "3a₁", "1b₁ (HOMO)", "4a₁ (LUMO)", "2b₂"]
    g = np.linspace(-3.2, 3.2, 161)
    X, Y = np.meshgrid(g, g)
    plane_xy = np.stack([X.ravel(), Y.ravel() + 0.4, np.zeros(X.size)], 1)   # molecular plane
    plane_zy = np.stack([np.zeros(X.size), Y.ravel() + 0.4, X.ravel()], 1)   # perpendicular, through the C2 axis
    show = [1, 2, 3, 4, 5]
    fig = plt.figure(figsize=(14, 3.6))
    for k, mo in enumerate(show):
        ax = fig.add_subplot(1, len(show) + 1, k + 1)
        pts = plane_zy if mo == 4 else plane_xy
        val = sum(res.C[mu, mo] * bf.value(pts) for mu, bf in enumerate(basis)).reshape(X.shape)
        lim = np.abs(val).max()
        ax.contourf(X, Y + 0.4, val, levels=np.linspace(-lim, lim, 21), cmap="RdBu_r")
        ax.contour(X, Y + 0.4, val, levels=[0], colors="k", linewidths=0.4)
        for Z, xyz in mol.atoms:
            ax.plot(xyz[2] if mo == 4 else xyz[0], xyz[1], "o", color="k" if Z == 8 else "w", mec="k", ms=6 if Z == 8 else 4)
        ax.set(aspect="equal", xticks=[], yticks=[],
               title=f"{names[mo]}\nε = {res.orbital_energies[mo]:+.3f} Eₕ", xlabel="z (out of plane)" if mo == 4 else "")
    ax = fig.add_subplot(1, len(show) + 1, len(show) + 1)
    eps = res.orbital_energies
    label_y = []
    for i, e in enumerate(eps):
        occ = i < res.n_occ
        y = e if e > -2 else -1.6
        ax.hlines(y, 0.2, 0.8, color="#1f5f99" if occ else "#c0504d", lw=2)
        ty = y if not label_y or y - label_y[-1] > 0.07 else label_y[-1] + 0.07  # keep labels apart
        label_y.append(ty)
        ax.text(0.85, ty, names[i].split()[0] + (" ↑↓" if occ else ""), va="center", fontsize=8)
    ax.set(xlim=(0, 1.6), ylim=(-1.75, 0.8), xticks=[], ylabel="orbital energy (hartree)",
           title=f"levels (1a₁ at {eps[0]:.2f},\ndrawn at −1.6)")
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(DOCS / "water_orbitals.png", dpi=120)
    return eps


if __name__ == "__main__":
    minima, e_h = dissociation()
    print("minima (R, E):", {k: (float(r), round(float(e), 6)) for k, (r, e) in minima.items()}, "E(H) =", e_h)
    convergence()
    print("orbital energies:", np.round(water_orbitals(), 4))
    print("figures written to docs/")
