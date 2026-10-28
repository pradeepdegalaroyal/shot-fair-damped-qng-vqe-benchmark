"""Fast correctness tests of the simulator, estimators and optimiser ledger."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shotfair_qng import METHODS, TFIM, RunConfig, build_ansatz, run_vqe, simulate  # noqa: E402
from shotfair_qng.circuit import _rot  # noqa: E402
from shotfair_qng.hamiltonian import sampled_pauli_metric_block  # noqa: E402
from shotfair_qng.vqe import _natural_step  # noqa: E402


def test_exact_energy_matches_dense_matrix() -> None:
    h, ans = TFIM(4), build_ansatz(4, 2, "ryrz")
    th = np.random.default_rng(0).normal(size=ans.n_params)
    psi = simulate(ans, th)[0][0]
    assert h.exact_energy(psi[None])[0] == pytest.approx(np.real(psi.conj() @ h.matrix() @ psi), abs=1e-10)


def test_states_are_normalised_and_ground_energy_is_lower_bound() -> None:
    h, ans = TFIM(5), build_ansatz(5, 2, "ryrz_dup")
    th = np.random.default_rng(1).normal(size=(3, ans.n_params))
    psi = simulate(ans, th)[0]
    assert np.allclose(np.linalg.norm(psi, axis=1), 1.0)
    assert np.all(h.exact_energy(psi) >= h.ground_energy() - 1e-9)


def test_parameter_shift_is_exact_for_rotation_gates() -> None:
    h, ans = TFIM(3), build_ansatz(3, 1, "ryrz")
    th = np.random.default_rng(2).normal(size=ans.n_params)
    k, eps = 4, 1e-6
    shift = np.zeros_like(th)
    shift[k] = np.pi / 2
    ps = (h.exact_energy(simulate(ans, th + shift)[0]) - h.exact_energy(simulate(ans, th - shift)[0]))[0] / 2
    d = np.zeros_like(th)
    d[k] = eps
    fd = (h.exact_energy(simulate(ans, th + d)[0]) - h.exact_energy(simulate(ans, th - d)[0]))[0] / (2 * eps)
    assert ps == pytest.approx(fd, abs=1e-6)


def test_sampled_energy_is_unbiased_and_depolarising_scales_it() -> None:
    h, ans = TFIM(4), build_ansatz(4, 2, "ryrz")
    th = np.random.default_rng(3).normal(size=(1, ans.n_params))
    psi = simulate(ans, th)[0]
    exact = h.exact_energy(psi)[0]
    rng = np.random.default_rng(0)
    est = h.sampled_energy(np.repeat(psi, 400, axis=0), 200, rng)
    assert est.mean() == pytest.approx(exact, abs=4 * est.std() / np.sqrt(400))
    noisy = h.sampled_energy(np.repeat(psi, 400, axis=0), 200, rng, fidelity=0.5)
    assert noisy.mean() == pytest.approx(0.5 * exact, abs=4 * noisy.std() / np.sqrt(400))


def test_metric_block_matches_fubini_study_of_the_layer() -> None:
    n = 3
    rng = np.random.default_rng(4)
    psi0 = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    psi0 /= np.linalg.norm(psi0)
    g_hat = sampled_pauli_metric_block(psi0, "Z", 400_000, rng)

    def after(theta: np.ndarray) -> np.ndarray:
        s = psi0[None].copy()
        for q in range(n):
            s = _rot(s, "RZ", theta[q : q + 1], q)
        return s[0]

    # Fubini-Study metric g_ij = -1/2 d_i d_j |<psi(0)|psi(theta)>|^2 at theta = 0 by central differences
    eps, g = 1e-3, np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            def f(a: float, b: float) -> float:
                t = np.zeros(n)
                t[i] += a
                t[j] += b
                return abs(np.vdot(psi0, after(t))) ** 2

            g[i, j] = -0.5 * (f(eps, eps) - f(eps, -eps) - f(-eps, eps) + f(-eps, -eps)) / (4 * eps**2)
    assert np.allclose(g_hat, g, atol=3e-3)


def test_natural_step_interpolates_to_scaled_gradient_descent() -> None:
    rng = np.random.default_rng(5)
    a = rng.normal(size=(4, 4))
    g_metric, grad = a @ a.T / 16, rng.normal(size=4)
    step, pred = _natural_step(g_metric, grad, 0.1, 1e8)
    assert np.allclose(step, -4 * 0.1 * grad, rtol=1e-5)
    assert pred > 0


@pytest.mark.parametrize("method", METHODS)
def test_budget_is_respected_and_ledger_is_monotone(method: str) -> None:
    cfg = RunConfig(method, n=3, depth=1, shots=16, budget_per_param=3000.0, lr=0.05, max_iters=50)
    tr = run_vqe(cfg)
    assert tr.shots[-1] <= 3000.0 * build_ansatz(3, 1, "ryrz").n_params
    assert np.all(np.diff(tr.shots) > 0)
    assert len(tr.energy) == len(tr.shots)
    assert np.all(np.isfinite(tr.energy))


def test_same_seed_is_reproducible_and_adam_descends() -> None:
    cfg = RunConfig("adam_ps", n=3, depth=1, shots=64, budget_per_param=20_000.0, lr=0.05, seed=7)
    a, b = run_vqe(cfg), run_vqe(cfg)
    assert np.allclose(a.energy, b.energy)
    assert a.energy[-1] < a.energy[0]
