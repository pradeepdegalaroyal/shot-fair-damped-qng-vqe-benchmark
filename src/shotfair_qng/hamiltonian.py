"""Transverse-field Ising model on a ring and finite-shot energy / Pauli-moment estimators."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_SDG = np.diag([1, -1j])


@lru_cache(maxsize=None)
def basis_change(n: int, basis: str) -> np.ndarray | None:
    """Dense unitary mapping the ``basis`` Pauli eigenbasis to the computational basis (``None`` for Z)."""
    if basis == "Z":
        return None
    one = _H if basis == "X" else _H @ _SDG
    u = np.array([[1.0 + 0j]])
    for _ in range(n):
        u = np.kron(u, one)
    return u


@lru_cache(maxsize=None)
def z_table(n: int) -> np.ndarray:
    """``(2**n, n)`` table of +-1 eigenvalues of Z_q on each computational basis state."""
    k = np.arange(2**n)
    bits = (k[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1
    return (1 - 2 * bits).astype(float)


@dataclass
class TFIM:
    """H = -J sum_i Z_i Z_{i+1} - h sum_i X_i with periodic boundaries.

    The energy is measured in two settings (Z basis for the ZZ terms, X basis for the X terms), so one energy
    evaluation costs ``2 * shots`` circuit executions.
    """

    n: int
    J: float = 1.0
    h: float = 1.0
    w_z: np.ndarray = field(init=False, repr=False)
    w_x: np.ndarray = field(init=False, repr=False)
    n_settings: int = 2

    def __post_init__(self) -> None:
        z = z_table(self.n)
        zz = sum(z[:, q] * z[:, (q + 1) % self.n] for q in range(self.n))
        self.w_z = -self.J * zz
        self.w_x = -self.h * z.sum(axis=1)

    def matrix(self) -> np.ndarray:
        """Dense Hamiltonian (for exact diagonalisation and tests)."""
        n = self.n
        x = np.array([[0, 1], [1, 0]], dtype=float)
        zm = np.diag([1.0, -1.0])

        def op(o: np.ndarray, q: int) -> np.ndarray:
            m = np.array([[1.0]])
            for i in range(n):
                m = np.kron(m, o if i == q else np.eye(2))
            return m

        hm = np.zeros((2**n, 2**n))
        for q in range(n):
            hm -= self.J * op(zm, q) @ op(zm, (q + 1) % n)
            hm -= self.h * op(x, q)
        return hm

    def ground_energy(self) -> float:
        """Exact ground-state energy."""
        return float(np.linalg.eigvalsh(self.matrix())[0])

    def product_state_energy(self, restarts: int = 8, seed: int = 0) -> float:
        """Lowest energy of a product state with per-site angles in the x-z plane (classical mean-field baseline)."""
        from scipy.optimize import minimize

        n = self.n

        def e(phi: np.ndarray) -> float:
            zs, xs = np.cos(phi), np.sin(phi)
            return float(-self.J * np.sum(zs * np.roll(zs, -1)) - self.h * np.sum(xs))

        rng = np.random.default_rng(seed)
        best = np.inf
        for r in range(restarts):
            x0 = np.full(n, np.arcsin(min(1.0, self.h / (2 * self.J)))) if r == 0 else rng.uniform(0, np.pi, n)
            best = min(best, minimize(e, x0, method="BFGS").fun)
        return float(best)

    def exact_energy(self, states: np.ndarray) -> np.ndarray:
        """Noise-free energies of a batch of states, shape ``(B,)``."""
        pz = np.abs(states) ** 2
        px = np.abs(states @ basis_change(self.n, "X").T) ** 2
        return pz @ self.w_z + px @ self.w_x

    def sampled_energy(
        self, states: np.ndarray, shots: int, rng: np.random.Generator, fidelity: float = 1.0
    ) -> np.ndarray:
        """Finite-shot energy estimates of a batch of states.

        Each state and each measurement setting gets ``shots`` independent shots. ``fidelity`` < 1 applies a global
        depolarising channel, which mixes every outcome distribution with the uniform one.
        """
        pz = np.abs(states) ** 2
        px = np.abs(states @ basis_change(self.n, "X").T) ** 2
        n_basis = pz.shape[1]
        out = 0.0
        for p, w in ((pz, self.w_z), (px, self.w_x)):
            p = fidelity * p + (1 - fidelity) / n_basis
            p = p / p.sum(axis=1, keepdims=True)
            out = out + rng.multinomial(shots, p) @ w
        return out / shots


def sampled_pauli_metric_block(
    state: np.ndarray, basis: str, shots: int, rng: np.random.Generator, fidelity: float = 1.0
) -> np.ndarray:
    """Shot-estimated block of the Fubini-Study metric for one layer of single-qubit rotations.

    For a layer exp(-i theta_q P_q / 2) the block is ``g_ij = (<P_i P_j> - <P_i><P_j>) / 4`` with the diagonal
    ``(1 - <P_i>^2) / 4``; all P_i commute, so one measurement setting (``basis``) on the state *before* the layer
    gives every entry.
    """
    n = int(np.log2(state.size))
    u = basis_change(n, basis)
    amp = state if u is None else u @ state
    p = fidelity * np.abs(amp) ** 2 + (1 - fidelity) / state.size
    p = p / p.sum()
    counts = rng.multinomial(shots, p)
    z = z_table(n)
    m = counts @ z / shots
    c = (z * counts[:, None]).T @ z / shots
    g = (c - np.outer(m, m)) / 4
    np.fill_diagonal(g, (1 - m**2) / 4)
    return g
