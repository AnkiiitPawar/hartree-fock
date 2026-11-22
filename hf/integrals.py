"""One- and two-electron integrals over Cartesian Gaussians by the McMurchie–Davidson scheme.

The product of two Gaussians is expanded in Hermite Gaussians Λₜ centred at the Gaussian
product centre P = (aA + bB)/p, p = a + b:
    Gᵢ(x; a, A) Gⱼ(x; b, B) = Σₜ Eₜ^{ij} Λₜ(x; p, P)
with E coefficients from the recurrences (Helgaker, Jørgensen & Olsen, ch. 9.5)
    E₀⁰⁰ = exp(-q Q²),  q = ab/p,  Q = Aₓ - Bₓ
    Eₜ^{i+1,j} = Eₜ₋₁^{ij}/(2p) - (q Q / a) Eₜ^{ij} + (t+1) Eₜ₊₁^{ij}
    Eₜ^{i,j+1} = Eₜ₋₁^{ij}/(2p) + (q Q / b) Eₜ^{ij} + (t+1) Eₜ₊₁^{ij}
Coulomb-type integrals over Hermite Gaussians reduce to the Boys function
    Fₙ(T) = ∫₀¹ t²ⁿ exp(-T t²) dt
through the auxiliary integrals R_{tuv}^{n}.
"""

from __future__ import annotations

from functools import lru_cache
from math import exp, pi, sqrt

import numpy as np
from scipy.special import hyp1f1


def boys(n: int, T: float) -> float:
    """Fₙ(T) = ₁F₁(n + ½; n + 3/2; -T) / (2n + 1)."""
    return float(hyp1f1(n + 0.5, n + 1.5, -T)) / (2 * n + 1)


@lru_cache(maxsize=None)
def E(i: int, j: int, t: int, Q: float, a: float, b: float) -> float:
    """Hermite expansion coefficient Eₜ^{ij} for one Cartesian direction."""
    p = a + b
    q = a * b / p
    if t < 0 or t > i + j:
        return 0.0
    if i == j == t == 0:
        return exp(-q * Q * Q)
    if j == 0:
        return (E(i - 1, j, t - 1, Q, a, b) / (2 * p) - (q * Q / a) * E(i - 1, j, t, Q, a, b)
                + (t + 1) * E(i - 1, j, t + 1, Q, a, b))
    return (E(i, j - 1, t - 1, Q, a, b) / (2 * p) + (q * Q / b) * E(i, j - 1, t, Q, a, b)
            + (t + 1) * E(i, j - 1, t + 1, Q, a, b))


def overlap_prim(a, lmn1, A, b, lmn2, B) -> float:
    """⟨G_a|G_b⟩ = E₀^{l₁l₂} E₀^{m₁m₂} E₀^{n₁n₂} (π/p)^{3/2} (primitives, unnormalised)."""
    s = 1.0
    for k in range(3):
        s *= E(lmn1[k], lmn2[k], 0, A[k] - B[k], a, b)
    return s * (pi / (a + b)) ** 1.5


def kinetic_prim(a, lmn1, A, b, lmn2, B) -> float:
    """⟨G_a| -½∇² |G_b⟩, by differentiating G_b: each ∂²/∂x² gives three overlap terms."""
    l2, m2, n2 = lmn2
    term0 = b * (2 * (l2 + m2 + n2) + 3) * overlap_prim(a, lmn1, A, b, lmn2, B)
    term1 = -2 * b * b * sum(overlap_prim(a, lmn1, A, b, tuple(v + 2 * (k == d) for k, v in enumerate(lmn2)), B)
                             for d in range(3))
    term2 = -0.5 * sum(lmn2[d] * (lmn2[d] - 1) * overlap_prim(a, lmn1, A, b,
                                                               tuple(v - 2 * (k == d) for k, v in enumerate(lmn2)), B)
                       for d in range(3) if lmn2[d] >= 2)
    return term0 + term1 + term2


def R(t: int, u: int, v: int, n: int, p: float, PC: np.ndarray, RPC2: float, cache: dict) -> float:
    """Hermite Coulomb integral R_{tuv}^{n}(p, P - C) by downward recursion in t, u, v."""
    key = (t, u, v, n)
    if key in cache:
        return cache[key]
    if t == u == v == 0:
        val = (-2 * p) ** n * boys(n, p * RPC2)
    elif t > 0:
        val = (t - 1) * R(t - 2, u, v, n + 1, p, PC, RPC2, cache) if t > 1 else 0.0
        val += PC[0] * R(t - 1, u, v, n + 1, p, PC, RPC2, cache)
    elif u > 0:
        val = (u - 1) * R(t, u - 2, v, n + 1, p, PC, RPC2, cache) if u > 1 else 0.0
        val += PC[1] * R(t, u - 1, v, n + 1, p, PC, RPC2, cache)
    else:
        val = (v - 1) * R(t, u, v - 2, n + 1, p, PC, RPC2, cache) if v > 1 else 0.0
        val += PC[2] * R(t, u, v - 1, n + 1, p, PC, RPC2, cache)
    cache[key] = val
    return val


def nuclear_prim(a, lmn1, A, b, lmn2, B, C) -> float:
    """⟨G_a| 1/|r - C| |G_b⟩ = (2π/p) Σ_{tuv} Eₜ Eᵤ Eᵥ R_{tuv}(p, P - C)."""
    p = a + b
    P = (a * A + b * B) / p
    PC = P - C
    cache: dict = {}
    total = 0.0
    for t in range(lmn1[0] + lmn2[0] + 1):
        et = E(lmn1[0], lmn2[0], t, A[0] - B[0], a, b)
        for u in range(lmn1[1] + lmn2[1] + 1):
            eu = E(lmn1[1], lmn2[1], u, A[1] - B[1], a, b)
            for v in range(lmn1[2] + lmn2[2] + 1):
                ev = E(lmn1[2], lmn2[2], v, A[2] - B[2], a, b)
                total += et * eu * ev * R(t, u, v, 0, p, PC, float(PC @ PC), cache)
    return 2 * pi / p * total


def eri_prim(a, l1, A, b, l2, B, c, l3, C, d, l4, D) -> float:
    """(ab|cd) = 2π^{5/2} / (pq√(p+q)) Σ Eᵃᵇ Eᶜᵈ (-1)^{τ+ν+φ} R_{t+τ,u+ν,v+φ}(α, P - Q)."""
    p, q = a + b, c + d
    alpha = p * q / (p + q)
    P = (a * A + b * B) / p
    Q = (c * C + d * D) / q
    PQ = P - Q
    RPQ2 = float(PQ @ PQ)
    cache: dict = {}
    eab = [[E(l1[k], l2[k], t, A[k] - B[k], a, b) for t in range(l1[k] + l2[k] + 1)] for k in range(3)]
    ecd = [[E(l3[k], l4[k], t, C[k] - D[k], c, d) for t in range(l3[k] + l4[k] + 1)] for k in range(3)]
    total = 0.0
    for t, et in enumerate(eab[0]):
        for u, eu in enumerate(eab[1]):
            for v, ev in enumerate(eab[2]):
                left = et * eu * ev
                if left == 0.0:
                    continue
                for tau, etau in enumerate(ecd[0]):
                    for nu, enu in enumerate(ecd[1]):
                        for phi, ephi in enumerate(ecd[2]):
                            sign = -1.0 if (tau + nu + phi) % 2 else 1.0
                            total += left * sign * etau * enu * ephi * R(t + tau, u + nu, v + phi, 0, alpha, PQ,
                                                                         RPQ2, cache)
    return 2 * pi**2.5 / (p * q * sqrt(p + q)) * total


def dipole_prim(a, lmn1, A, b, lmn2, B, origin, k: int) -> float:
    """⟨G_a| (r - O)ₖ |G_b⟩: in direction k, ∫Λₜ (x - Oₓ) dx = √(π/p) (δₜ₁ + (Pₓ - Oₓ) δₜ₀)."""
    p = a + b
    P = (a * A + b * B) / p
    val = (pi / p) ** 1.5
    for d in range(3):
        if d == k:
            val *= E(lmn1[d], lmn2[d], 1, A[d] - B[d], a, b) + (P[d] - origin[d]) * E(lmn1[d], lmn2[d], 0, A[d] - B[d], a, b)
        else:
            val *= E(lmn1[d], lmn2[d], 0, A[d] - B[d], a, b)
    return val


def _contract2(f, bf1, bf2, *extra) -> float:
    return sum(c1 * c2 * f(a, bf1.lmn, bf1.center, b, bf2.lmn, bf2.center, *extra)
               for a, c1 in zip(bf1.exps, bf1.coefs) for b, c2 in zip(bf2.exps, bf2.coefs))


def overlap_contracted(bf1, bf2) -> float:
    return _contract2(overlap_prim, bf1, bf2)


def one_electron(basis, atoms):
    """Overlap S, kinetic T and nuclear attraction V = -Σ_C Z_C ⟨μ|1/r_C|ν⟩ matrices."""
    n = len(basis)
    S, T, V = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            S[i, j] = S[j, i] = _contract2(overlap_prim, basis[i], basis[j])
            T[i, j] = T[j, i] = _contract2(kinetic_prim, basis[i], basis[j])
            V[i, j] = V[j, i] = -sum(Z * _contract2(nuclear_prim, basis[i], basis[j], np.asarray(xyz, float))
                                     for Z, xyz in atoms)
    return S, T, V


def dipole_matrices(basis, origin=np.zeros(3)):
    n = len(basis)
    M = np.zeros((3, n, n))
    for k in range(3):
        for i in range(n):
            for j in range(i + 1):
                M[k, i, j] = M[k, j, i] = _contract2(dipole_prim, basis[i], basis[j], origin, k)
    return M


def two_electron(basis) -> np.ndarray:
    """All (μν|λσ) in chemists' notation, computing only the 8-fold unique quartets."""
    n = len(basis)
    eri = np.zeros((n, n, n, n))
    for i in range(n):
        for j in range(i + 1):
            ij = i * (i + 1) // 2 + j
            for k in range(n):
                for l in range(k + 1):
                    kl = k * (k + 1) // 2 + l
                    if ij < kl:
                        continue
                    bi, bj, bk, bl = basis[i], basis[j], basis[k], basis[l]
                    val = 0.0
                    for a, ca in zip(bi.exps, bi.coefs):
                        for b, cb in zip(bj.exps, bj.coefs):
                            for c, cc in zip(bk.exps, bk.coefs):
                                for d, cd in zip(bl.exps, bl.coefs):
                                    val += ca * cb * cc * cd * eri_prim(a, bi.lmn, bi.center, b, bj.lmn, bj.center,
                                                                        c, bk.lmn, bk.center, d, bl.lmn, bl.center)
                    for (p_, q_, r_, s_) in ((i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k),
                                             (k, l, i, j), (l, k, i, j), (k, l, j, i), (l, k, j, i)):
                        eri[p_, q_, r_, s_] = val
    return eri
