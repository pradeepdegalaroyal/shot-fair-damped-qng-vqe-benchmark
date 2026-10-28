"""Reproduce every number and figure: ``python experiments/run_all.py`` (add ``--quick`` for a smoke run)."""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per worker; parallelism comes from the process pool

import argparse
import json
import multiprocessing as mp
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shotfair_qng import METHODS, TFIM, RunConfig, build_ansatz, run_vqe  # noqa: E402
from shotfair_qng.plotting import make_all_figures  # noqa: E402

MAIN_SEEDS = (0, 1, 2, 3, 4)
SWEEP_SEEDS = (0, 1, 2)
TUNE_SEEDS = (100, 101)  # disjoint from the evaluation seeds
LR_GRID = {
    "adam_ps": (0.003, 0.01, 0.03),
    "spsa": (0.1, 0.3, 1.0),
    "qng_std": (0.003, 0.01, 0.03),
    "qng_damped": (0.003, 0.01, 0.03),
    "qng_tr": (0.003, 0.01, 0.03),
}
LAMBDA_GRID = (1e-4, 1e-2, 1e-1, 1.0, 10.0)
ERROR_TARGETS = (0.05, 0.03)  # relative energy errors; 3 % is the product-state (mean-field) level of the 6-site ring


def reference(n: int) -> dict[str, float]:
    """Exact ground energy and best product-state (mean-field) energy of the n-site ring."""
    h = TFIM(n)
    return {"E0": h.ground_energy(), "E_MF": h.product_state_energy()}


def ansatz_floor(n: int, depth: int, kind: str, starts: int = 3, maxiter: int = 250) -> float:
    """Best exact (noise-free) energy reachable by the ansatz, from L-BFGS-B with exact parameter-shift gradients."""
    from scipy.optimize import minimize

    from shotfair_qng.circuit import simulate

    h, ans = TFIM(n), build_ansatz(n, depth, kind)
    p = ans.n_params
    idx = np.arange(p)

    def fun(th: np.ndarray) -> tuple[float, np.ndarray]:
        b = np.tile(th, (2 * p + 1, 1))
        b[idx, idx] += np.pi / 2
        b[p + idx, idx] -= np.pi / 2
        e = h.exact_energy(simulate(ans, b)[0])
        return float(e[-1]), (e[:p] - e[p : 2 * p]) / 2

    rng = np.random.default_rng(0)
    return min(minimize(fun, rng.normal(0, 1, p), jac=True, method="L-BFGS-B", options={"maxiter": maxiter}).fun for _ in range(starts))


def run_job(cfg: RunConfig) -> dict:
    """Run one configuration and return a flat summary row plus a down-sampled trace."""
    tr = run_vqe(cfg)
    ref = REFS[cfg.n]
    e0, emf = ref["E0"], ref["E_MF"]
    rec = (emf - tr.energy) / (emf - e0)  # correlation energy recovered; 0 = mean field, 1 = exact
    row = asdict(cfg)
    row.update(
        final_energy=tr.final_energy,
        rel_err=(tr.final_energy - e0) / abs(e0),
        recovered=float(rec[-1]),
        best_recovered=float(rec.max()),
        n_iters=len(tr.shots) - 1,
        shots_used=float(tr.shots[-1]),
        lam_final=tr.lam_final,
    )
    rel = (tr.energy - e0) / abs(e0)
    for t in ERROR_TARGETS:
        hit = np.nonzero(rel <= t)[0]
        row[f"shots_to_{int(t * 100)}"] = float(tr.shots[hit[0]]) if len(hit) else float("nan")
    k = np.unique(np.linspace(0, len(tr.shots) - 1, 120).astype(int))
    row["trace"] = json.dumps({"shots": tr.shots[k].round(0).tolist(), "recovered": rec[k].round(5).tolist()})
    return row


REFS: dict[int, dict[str, float]] = {}


def _init(refs: dict[int, dict[str, float]]) -> None:
    REFS.update(refs)


def pool_map(jobs: list[RunConfig], workers: int, refs: dict) -> list[dict]:
    """Run jobs in a process pool (the main process computes nothing heavy)."""
    if workers <= 1:
        _init(refs)
        return [run_job(j) for j in jobs]
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(refs,)) as pool:
        return list(pool.imap_unordered(run_job, jobs, chunksize=1))


def dedupe(jobs: list[RunConfig]) -> list[RunConfig]:
    seen, out = set(), []
    for j in jobs:
        if j not in seen:
            seen.add(j)
            out.append(j)
    return out


def build_jobs(base: RunConfig, lrs: dict[str, float], quick: bool) -> list[RunConfig]:
    """All evaluation runs of the study (main, shot sweep, fidelity sweep, damping grid, ablation, scaling)."""

    def mk(method: str, **kw) -> RunConfig:
        return replace(base, method=method, lr=lrs[method], **kw)

    main_seeds = MAIN_SEEDS[:2] if quick else MAIN_SEEDS
    sweep_seeds = SWEEP_SEEDS[:1] if quick else SWEEP_SEEDS
    shots_grid = (32, 128) if quick else (32, 128, 512)
    fid_grid = (1.0, 0.6) if quick else (1.0, 0.8, 0.5)
    jobs: list[RunConfig] = []
    for m in METHODS:  # main comparison
        jobs += [mk(m, seed=s) for s in main_seeds]
    for m in METHODS:  # shot-noise sweep and depolarising sweep
        jobs += [mk(m, seed=s, shots=sh) for sh in shots_grid for s in sweep_seeds]
        jobs += [mk(m, seed=s, fidelity=f) for f in fid_grid for s in sweep_seeds]
    for lam in LAMBDA_GRID[:2] if quick else LAMBDA_GRID:  # fixed-damping grid
        jobs += [mk("qng_damped", seed=s, shots=sh, lam=lam) for sh in shots_grid for s in sweep_seeds]
    kinds = ("ry", "ryrz") if quick else ("ry", "ryrz", "ryrz_dup")
    for kind in kinds:  # redundancy ablation
        for m in ("adam_ps", "spsa", "qng_std", "qng_tr"):
            jobs += [mk(m, seed=s, kind=kind) for s in sweep_seeds]
    for n in (4,) if quick else (4, 6, 8):  # scaling in qubits
        for m in ("adam_ps", "spsa", "qng_std", "qng_tr"):
            jobs += [mk(m, seed=s, n=n) for s in sweep_seeds]
    return dedupe(jobs)


def tune(base: RunConfig, workers: int, refs: dict, quick: bool) -> tuple[dict[str, float], pd.DataFrame]:
    """Pick each method's step size on tuning seeds that are disjoint from the evaluation seeds."""
    seeds = TUNE_SEEDS[:1] if quick else TUNE_SEEDS
    grid = {m: g[:2] if quick else g for m, g in LR_GRID.items()}
    jobs = [replace(base, method=m, lr=lr, seed=s) for m, g in grid.items() for lr in g for s in seeds]
    df = pd.DataFrame(pool_map(jobs, workers, refs)).drop(columns="trace")
    med = df.groupby(["method", "lr"])["recovered"].median().reset_index()
    best = {m: float(med[med.method == m].sort_values("recovered").iloc[-1]["lr"]) for m in grid}
    return best, med


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="tiny smoke run (< 1 minute)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=ROOT)
    args = ap.parse_args()
    t0 = time.time()
    res, figs = args.out / ("results_quick" if args.quick else "results"), args.out / ("figures_quick" if args.quick else "figures")
    res.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    if args.quick:
        base = RunConfig("adam_ps", n=4, depth=2, shots=64, budget_per_param=12_000.0, max_iters=60, init_scale=0.3)
        ns = (4,)
    else:
        base = RunConfig("adam_ps", init_scale=0.3, budget_per_param=200_000.0)
        ns = (4, 6, 8)
    refs = {n: reference(n) for n in ns}
    floors = {f"n{n}_{k}": ansatz_floor(n, base.depth, k, starts=1 if args.quick else 3) for n in ns for k in (("ryrz",) if n != base.n else ("ry", "ryrz", "ryrz_dup"))}
    json.dump({"reference": {str(n): r for n, r in refs.items()}, "ansatz_floor_energy": floors}, open(res / "reference.json", "w"), indent=2)
    print(f"[{time.time() - t0:5.0f}s] references done", flush=True)

    lrs, tune_df = tune(base, args.workers, refs, args.quick)
    tune_df.to_csv(res / "tuning.csv", index=False)
    json.dump(lrs, open(res / "tuned_lr.json", "w"), indent=2)
    print(f"[{time.time() - t0:5.0f}s] tuned step sizes: {lrs}", flush=True)

    jobs = build_jobs(base, lrs, args.quick)
    print(f"[{time.time() - t0:5.0f}s] running {len(jobs)} evaluation runs", flush=True)
    df = pd.DataFrame(pool_map(jobs, args.workers, refs))
    df = df.sort_values(["n", "kind", "fidelity", "shots", "method", "lam", "seed"]).reset_index(drop=True)
    df.drop(columns="trace").to_csv(res / "runs.csv", index=False)
    df[["method", "n", "kind", "shots", "fidelity", "lam", "seed", "trace"]].to_json(res / "traces.json", orient="records")
    print(f"[{time.time() - t0:5.0f}s] runs saved", flush=True)

    make_all_figures(df, refs, figs, res)
    print(f"[{time.time() - t0:5.0f}s] figures and summary tables saved to {figs.name}/ and {res.name}/", flush=True)


if __name__ == "__main__":
    main()
