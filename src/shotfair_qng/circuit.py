"""Batched statevector simulation of a layered hardware-efficient ansatz.

Qubit ``q`` is tensor axis ``q`` of the state, so basis index ``k`` has bit ``q`` equal to
``(k >> (n - 1 - q)) & 1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KINDS = ("ry", "ryrz", "ryrz_dup")


@dataclass(frozen=True)
class Ansatz:
    """Layer list of an ansatz.

    Attributes:
        n: number of qubits.
        depth: number of entangling blocks.
        kind: ``ry`` (real, minimal), ``ryrz`` (generic) or ``ryrz_dup`` (a second RZ layer is stacked on
            every RZ layer, so every RZ angle has an exactly redundant partner).
        layers: tuple of ``(gate, first_param_index)``; ``gate`` is ``RY``, ``RZ`` or ``CZ``
            (``first_param_index`` is -1 for ``CZ``).
        n_params: number of rotation angles.
        cz_sign: diagonal of the full CZ ring, shape ``(2**n,)``.
    """

    n: int
    depth: int
    kind: str
    layers: tuple[tuple[str, int], ...]
    n_params: int
    cz_sign: np.ndarray

    @property
    def rotation_layers(self) -> list[tuple[str, int]]:
        """Rotation layers in circuit order."""
        return [(g, o) for g, o in self.layers if g != "CZ"]


def build_ansatz(n: int, depth: int, kind: str = "ryrz") -> Ansatz:
    """Build ``depth`` blocks of rotations plus a CZ ring, followed by a final rotation block."""
    if kind not in KINDS:
        raise ValueError(f"unknown ansatz kind {kind!r}")
    rot = {"ry": ["RY"], "ryrz": ["RY", "RZ"], "ryrz_dup": ["RY", "RZ", "RZ"]}[kind]
    layers: list[tuple[str, int]] = []
    p = 0
    for block in range(depth + 1):
        for g in rot:
            layers.append((g, p))
            p += n
        if block < depth:
            layers.append(("CZ", -1))
    k = np.arange(2**n)
    bits = (k[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1
    sign = np.ones(2**n)
    for q in range(n):
        r = (q + 1) % n
        if r != q and not (n == 2 and q == 1):
            sign *= 1 - 2 * (bits[:, q] & bits[:, r])
    return Ansatz(n, depth, kind, tuple(layers), p, sign)


def _rot(psi: np.ndarray, gate: str, theta: np.ndarray, q: int) -> np.ndarray:
    """Apply a single-qubit rotation with per-batch-element angle ``theta`` to qubit ``q``."""
    b = psi.shape[0]
    v = psi.reshape(b, 2**q, 2, -1)
    a0, a1 = v[:, :, 0, :], v[:, :, 1, :]
    if gate == "RY":
        c = np.cos(theta / 2)[:, None, None]
        s = np.sin(theta / 2)[:, None, None]
        n0, n1 = c * a0 - s * a1, s * a0 + c * a1
    else:  # RZ
        e = np.exp(-0.5j * theta)[:, None, None]
        n0, n1 = e * a0, np.conj(e) * a1
    return np.stack([n0, n1], axis=2).reshape(b, -1)


def simulate(
    ans: Ansatz, params: np.ndarray, record: bool = False
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Simulate the ansatz for a batch of parameter vectors.

    Args:
        ans: ansatz description.
        params: array of shape ``(B, P)`` (or ``(P,)``, treated as ``B = 1``).
        record: if True also return the state before every rotation layer (batch element 0 only).

    Returns:
        Final states of shape ``(B, 2**n)`` and the list of recorded pre-layer states (empty unless ``record``).
    """
    params = np.atleast_2d(params)
    psi = np.zeros((params.shape[0], 2**ans.n), dtype=complex)
    psi[:, 0] = 1.0
    recorded: list[np.ndarray] = []
    for gate, off in ans.layers:
        if gate == "CZ":
            psi = psi * ans.cz_sign
            continue
        if record:
            recorded.append(psi[0].copy())
        for q in range(ans.n):
            psi = _rot(psi, gate, params[:, off + q], q)
    return psi, recorded
