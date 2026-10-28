"""VQE optimisers under an explicit shot ledger.

Every circuit execution is charged: parameter-shift gradients (2P energy evaluations), metric blocks (one setting per
rotation layer), the two energy evaluations of the trust-region gain ratio, and the two evaluations of SPSA. A run stops
when the next iteration would exceed the shot budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from .circuit import Ansatz, build_ansatz, simulate
from .hamiltonian import TFIM, sampled_pauli_metric_block

METHODS = ("adam_ps", "spsa", "qng_std", "qng_damped", "qng_tr")
LAM_STD = 1e-3  # standard QNG: minimal Tikhonov term, as in common libraries
LAM_DAMPED = 0.1  # fixed damping
LAM0_TR, LAM_MIN, LAM_MAX = 0.1, 1e-4, 10.0  # trust-region damping start and clip range
SPSA_MAX_ITERS = 800  # SPSA raises its shots per evaluation so the whole budget is used within this many iterations
SPSA_C, SPSA_STAB = 0.3, 20.0  # SPSA perturbation size and stability constant; the gain a is RunConfig.lr


@dataclass(frozen=True)
class RunConfig:
    """One VQE run.

    Attributes:
        method: one of :data:`METHODS`.
        n: number of qubits of the transverse-field Ising ring.
        depth: entangling blocks of the ansatz.
        kind: ansatz kind (see :func:`shotfair_qng.circuit.build_ansatz`).
        shots: shots per measurement setting (an energy evaluation uses two settings).
        budget_per_param: total shot budget divided by the number of parameters.
        lr: Adam learning rate / natural-gradient step size / SPSA gain ``a``.
        lam: damping of ``qng_damped`` (ignored by the other methods).
        fidelity: global depolarising fidelity of every circuit (1 = noiseless).
        seed: seed of the initial point and of the shot noise.
        max_iters: hard cap on optimiser iterations.
        init_scale: standard deviation of the Gaussian initial angles.
    """

    method: str
    n: int = 6
    depth: int = 3
    kind: str = "ryrz"
    shots: int = 128
    budget_per_param: float = 200_000.0
    lr: float = 0.1
    lam: float = LAM_DAMPED
    fidelity: float = 1.0
    seed: int = 0
    max_iters: int = 1500
    # (SPSA is additionally limited by SPSA_MAX_ITERS through a larger shot count per evaluation)
    init_scale: float = 0.3


@dataclass
class Trace:
    """Result of one run: exact (noise-free) energy after every iteration versus cumulative shots."""

    cfg: RunConfig
    shots: np.ndarray
    energy: np.ndarray
    theta: np.ndarray
    lam_final: float = float("nan")
    extras: dict = field(default_factory=dict)

    @property
    def final_energy(self) -> float:
        return float(self.energy[-1])


def _method_rng(cfg: RunConfig) -> tuple[np.random.Generator, np.random.Generator]:
    init = np.random.default_rng(cfg.seed + 10_000)  # same start for every method at a given seed
    shot = np.random.default_rng(np.random.SeedSequence([cfg.seed, METHODS.index(cfg.method)]))
    return init, shot


def _sampled(problem: TFIM, ans: Ansatz, thetas: np.ndarray, cfg: RunConfig, rng: np.random.Generator) -> np.ndarray:
    states, _ = simulate(ans, thetas)
    return problem.sampled_energy(states, cfg.shots, rng, cfg.fidelity)


def _param_shift_grad(problem: TFIM, ans: Ansatz, theta: np.ndarray, cfg: RunConfig, rng: np.random.Generator) -> np.ndarray:
    p = ans.n_params
    batch = np.tile(theta, (2 * p, 1))
    idx = np.arange(p)
    batch[idx, idx] += np.pi / 2
    batch[p + idx, idx] -= np.pi / 2
    e = _sampled(problem, ans, batch, cfg, rng)
    return (e[:p] - e[p:]) / 2


def _metric(ans: Ansatz, theta: np.ndarray, cfg: RunConfig, rng: np.random.Generator) -> np.ndarray:
    """Block-diagonal Fubini-Study metric, one shot-estimated block per rotation layer."""
    _, recorded = simulate(ans, theta, record=True)
    g = np.zeros((ans.n_params, ans.n_params))
    for (gate, off), state in zip(ans.rotation_layers, recorded):
        basis = "Y" if gate == "RY" else "Z"
        g[off : off + ans.n, off : off + ans.n] = sampled_pauli_metric_block(state, basis, cfg.shots, rng, cfg.fidelity)
    return g


def _natural_step(g_metric: np.ndarray, grad: np.ndarray, lr: float, lam: float) -> tuple[np.ndarray, float]:
    """Scale-normalised damped natural-gradient step and its predicted energy decrease.

    The preconditioner is ``A = (G + lam I) / (1 + 4 lam)`` so that a block with ``G = I / 4`` is mapped to the same
    step length for every ``lam``; ``lam -> infinity`` recovers plain gradient descent with rate ``4 lr``.
    """
    a = (g_metric + lam * np.eye(len(grad))) / (1 + 4 * lam)
    sol = np.linalg.solve(a, grad)
    return -lr * sol, 0.5 * lr * float(grad @ sol)


def run_vqe(cfg: RunConfig, problem: TFIM | None = None, ans: Ansatz | None = None) -> Trace:
    """Run one VQE optimisation under the shot budget and return the exact-energy trace."""
    if cfg.method not in METHODS:
        raise ValueError(f"unknown method {cfg.method!r}")
    problem = problem or TFIM(cfg.n)
    ans = ans or build_ansatz(cfg.n, cfg.depth, cfg.kind)
    init_rng, rng = _method_rng(cfg)
    p = ans.n_params
    theta = init_rng.normal(0.0, cfg.init_scale, p)
    budget = cfg.budget_per_param * p
    cost_eval = problem.n_settings * cfg.shots
    n_layers = len(ans.rotation_layers)

    if cfg.method == "spsa":
        shots_eff = max(cfg.shots, math.ceil(budget / (SPSA_MAX_ITERS * 2 * problem.n_settings)))
        cfg_eval = replace(cfg, shots=shots_eff)
        cost_iter = 2 * problem.n_settings * shots_eff
    else:
        cost_iter = 2 * p * cost_eval
        if cfg.method != "adam_ps":
            cost_iter += n_layers * cfg.shots
        if cfg.method == "qng_tr":
            cost_iter += 2 * cost_eval

    def exact(th: np.ndarray) -> float:
        return float(problem.exact_energy(simulate(ans, th)[0])[0])

    used, it = 0, 0
    shots_log, energy_log = [0.0], [exact(theta)]
    lam = LAM0_TR
    m, v = np.zeros(p), np.zeros(p)

    while used + cost_iter <= budget and it < cfg.max_iters:
        it += 1
        if cfg.method == "spsa":
            a_k = cfg.lr / (it + SPSA_STAB) ** 0.602
            c_k = SPSA_C / it**0.101
            delta = rng.choice([-1.0, 1.0], size=p)
            e = _sampled(problem, ans, np.stack([theta + c_k * delta, theta - c_k * delta]), cfg_eval, rng)
            theta = theta - a_k * (e[0] - e[1]) / (2 * c_k) * delta
        elif cfg.method == "adam_ps":
            grad = _param_shift_grad(problem, ans, theta, cfg, rng)
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad**2
            theta = theta - cfg.lr * (m / (1 - 0.9**it)) / (np.sqrt(v / (1 - 0.999**it)) + 1e-8)
        else:
            grad = _param_shift_grad(problem, ans, theta, cfg, rng)
            g_metric = _metric(ans, theta, cfg, rng)
            use = {"qng_std": LAM_STD, "qng_damped": cfg.lam, "qng_tr": lam}[cfg.method]
            step, pred = _natural_step(g_metric, grad, cfg.lr, use)
            if cfg.method == "qng_tr":
                e = _sampled(problem, ans, np.stack([theta, theta + step]), cfg, rng)
                rho = (e[0] - e[1]) / pred if pred > 0 else -np.inf
                if rho > 0.75:
                    lam = max(lam / 2, LAM_MIN)
                elif rho < 0.25:
                    lam = min(lam * 2, LAM_MAX)
            theta = theta + step
        used += cost_iter
        if cfg.method != "spsa" or it % 5 == 0:  # SPSA logs less often: its iterations are cheap, so exact logging would dominate
            shots_log.append(float(used))
            energy_log.append(exact(theta))
    if shots_log[-1] != used:
        shots_log.append(float(used))
        energy_log.append(exact(theta))

    return Trace(cfg, np.array(shots_log), np.array(energy_log), theta, lam_final=lam if cfg.method == "qng_tr" else float("nan"))
