"""Summary tables, hypothesis tests and figures built from the run table."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

from .vqe import LAM_STD  # noqa: E402

LABEL = {
    "adam_ps": "Adam + parameter shift",
    "spsa": "SPSA",
    "qng_std": "QNG, lambda = 1e-3",
    "qng_damped": "QNG, fixed lambda = 0.1",
    "qng_tr": "QNG, trust-region lambda",
}
COLOR = {  # muted Tableau-10
    "adam_ps": "#4e79a7",
    "spsa": "#59a14f",
    "qng_std": "#e15759",
    "qng_damped": "#edc948",
    "qng_tr": "#b07aa1",
}
ORDER = list(LABEL)
GRAY = "#7f7f7f"
DPI = 200


def _plain_log_axis(ax: plt.Axes, axis: str = "y") -> None:
    """Log scale with plain-number tick labels and no minor labels."""
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_formatter(ScalarFormatter())
    a.set_minor_formatter(NullFormatter())


def _style() -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})


def _main_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of the main setting: most common n and shots, generic ansatz, no gate noise, default damping."""
    n, shots = df["n"].mode()[0], df["shots"].mode()[0]
    return df[(df.n == n) & (df.shots == shots) & (df.kind == "ryrz") & (df.fidelity == 1.0) & (df.lam == 0.1)]


def _mean_std(x: pd.Series) -> str:
    return f"{x.mean():.2f} +- {x.std(ddof=1):.2f}"


def summary_tables(df: pd.DataFrame, refs: dict, res: Path) -> dict:
    """Write the CSV summaries and the hypothesis tests; return the hypothesis dictionary."""
    main = _main_subset(df)
    n = int(main.n.iloc[0])
    mf_gap = (refs[n]["E_MF"] - refs[n]["E0"]) / abs(refs[n]["E0"]) * 100
    rows = []
    for m in ORDER:
        g = main[main.method == m]
        rows.append(
            {
                "method": m,
                "n_seeds": len(g),
                "rel_err_pct_mean_std": _mean_std(g.rel_err * 100),
                "rel_err_pct_median": round(float((g.rel_err * 100).median()), 3),
                "recovered_pct_mean_std": _mean_std(g.recovered * 100),
                "shots_to_5pct_error_median": g.shots_to_5.median(),
                "runs_reaching_5pct_error": f"{g.shots_to_5.notna().sum()}/{len(g)}",
                "runs_reaching_3pct_error": f"{g.shots_to_3.notna().sum()}/{len(g)}",
                "iterations_mean": round(float(g.n_iters.mean()) * (5 if m == "spsa" else 1), 1),  # SPSA logs every 5th iteration
                "shots_used_mean": round(float(g.shots_used.mean())),
                "final_lambda_mean": g.lam_final.mean(),
            }
        )
    pd.DataFrame(rows).to_csv(res / "summary_main.csv", index=False)
    pd.DataFrame([{"n": n, "mean_field_rel_err_pct": mf_gap}]).to_csv(res / "mean_field_baseline.csv", index=False)

    cond = df[(df.kind == "ryrz") & (df.n == n) & (df.lam == 0.1) & (df.seed.isin(sorted(df.seed.unique())[:3]))]
    sweeps = (
        cond[((cond.fidelity == 1.0)) | (cond.shots == df["shots"].mode()[0])]
        .groupby(["fidelity", "shots", "method"])
        .agg(rel_err_pct_median=("rel_err", lambda x: 100 * x.median()), rel_err_pct_mean=("rel_err", lambda x: 100 * x.mean()), rel_err_pct_std=("rel_err", lambda x: 100 * x.std(ddof=1)), n_seeds=("seed", "count"))
        .reset_index()
    )
    sweeps.to_csv(res / "summary_noise_sweeps.csv", index=False)

    abl = (
        df[(df.n == n) & (df.shots == df["shots"].mode()[0]) & (df.fidelity == 1.0) & (df.lam == 0.1) & (df.seed.isin(sorted(df.seed.unique())[:3]))]
        .groupby(["kind", "method"])
        .agg(rel_err_pct_mean=("rel_err", lambda x: 100 * x.mean()), rel_err_pct_std=("rel_err", lambda x: 100 * x.std(ddof=1)), rel_err_pct_median=("rel_err", lambda x: 100 * x.median()), n_params=("n_iters", "size"))
        .reset_index()
        .drop(columns="n_params")
    )
    abl.to_csv(res / "summary_ablation.csv", index=False)

    sc = (
        df[(df.kind == "ryrz") & (df.shots == df["shots"].mode()[0]) & (df.fidelity == 1.0) & (df.lam == 0.1) & (df.seed.isin(sorted(df.seed.unique())[:3]))]
        .groupby(["n", "method"])
        .agg(rel_err_pct_mean=("rel_err", lambda x: 100 * x.mean()), rel_err_pct_std=("rel_err", lambda x: 100 * x.std(ddof=1)), rel_err_pct_median=("rel_err", lambda x: 100 * x.median()), recovered_mean=("recovered", "mean"))
        .reset_index()
    )
    sc.to_csv(res / "summary_scaling.csv", index=False)

    # Hypotheses: H1 stability (standard / trust-region error ratio), H2 shot-fair advantage over the best classical-optimiser baseline
    def ratio_ci(a: np.ndarray, b: np.ndarray, reps: int = 4000) -> list[float]:
        rng = np.random.default_rng(0)
        r = [np.median(rng.choice(a, len(a))) / np.median(rng.choice(b, len(b))) for _ in range(reps)]
        return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]

    def test(a_df: pd.DataFrame, b_df: pd.DataFrame, thr: float) -> dict:
        a, b = a_df.rel_err.to_numpy(), b_df.rel_err.to_numpy()
        ratio = float(np.median(a) / np.median(b))
        ci = ratio_ci(a, b)
        return {"median_ratio": ratio, "bootstrap_95ci": ci, "mannwhitney_p_two_sided": float(mannwhitneyu(a, b).pvalue), "threshold": thr, "supported": bool(ratio >= thr and ci[0] > 1.0), "n_a": len(a), "n_b": len(b)}

    out: dict = {}
    g = {m: main[main.method == m] for m in ORDER}
    out["H1_main_setting"] = test(g["qng_std"], g["qng_tr"], 2.0) | {"claim": "median error of standard QNG / median error of trust-region QNG >= 2"}
    best_cl = min(("adam_ps", "spsa"), key=lambda m: g[m].rel_err.median())
    out["H2_main_setting"] = test(g[best_cl], g["qng_tr"], 1.25) | {"best_baseline": best_cl, "claim": "median error of best of {Adam, SPSA} / median error of trust-region QNG >= 1.25"}
    pool = df[(df.n == n) & (df.kind == "ryrz") & (df.lam == 0.1) & (df.seed.isin([0, 1, 2]))]
    pg = {m: pool[pool.method == m] for m in ORDER}
    out["H1_pooled_sweeps"] = test(pg["qng_std"], pg["qng_tr"], 2.0) | {"claim": "same as H1, pooled over shot and fidelity sweeps (seeds 0-2)"}
    best_p = min(("adam_ps", "spsa"), key=lambda m: pg[m].rel_err.median())
    out["H2_pooled_sweeps"] = test(pg[best_p], pg["qng_tr"], 1.25) | {"best_baseline": best_p, "claim": "same as H2, pooled over shot and fidelity sweeps (seeds 0-2)"}
    abl_red = df[(df.n == n) & (df.kind == "ryrz_dup") & (df.shots == df["shots"].mode()[0]) & (df.fidelity == 1.0)] if (df.kind == "ryrz_dup").any() else None
    if abl_red is not None:
        ar = {m: abl_red[abl_red.method == m] for m in ORDER}
        out["H1_redundant_ansatz"] = test(ar["qng_std"], ar["qng_tr"], 2.0) | {"claim": "H1 on the exactly redundant ansatz (seeds 0-2)"}
    out["mean_field_rel_err_pct"] = mf_gap
    json.dump(out, open(res / "hypothesis_tests.json", "w"), indent=2)
    return out


def fig_main(df: pd.DataFrame, refs: dict, path: Path) -> None:
    """Violin plus box plot of final relative energy error, seeds overlaid."""
    main = _main_subset(df)
    n = int(main.n.iloc[0])
    mf = (refs[n]["E_MF"] - refs[n]["E0"]) / abs(refs[n]["E0"]) * 100
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    rng = np.random.default_rng(1)
    data = [np.clip(main[main.method == m].rel_err.to_numpy() * 100, 1e-3, None) for m in ORDER]
    pos = np.arange(len(ORDER))
    # violins are drawn on log10 values so that the log-scale ticks below stay readable
    ldata = [np.log10(d) for d in data]
    vp = ax.violinplot(ldata, positions=pos, widths=0.8, showextrema=False)
    for body, m in zip(vp["bodies"], ORDER):
        body.set_facecolor(COLOR[m]); body.set_alpha(0.35); body.set_edgecolor(COLOR[m])
    ax.boxplot(ldata, positions=pos, widths=0.18, showfliers=False, medianprops={"color": "black"}, boxprops={"color": "#444"}, whiskerprops={"color": "#444"}, capprops={"color": "#444"})
    for i, d in enumerate(ldata):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=26, color=COLOR[ORDER[i]], edgecolor="black", linewidth=0.6, zorder=3)
    ax.axhline(np.log10(mf), color=GRAY, ls="--", lw=1.4)
    ax.text(len(ORDER) - 0.5, np.log10(mf) * 1.0 + 0.04, f"best product state: {mf:.2f} %", ha="right", va="bottom", color="#444", fontsize=9)
    ticks = [0.1, 0.3, 1, 3, 10, 30]
    ax.set_yticks(np.log10(ticks)); ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.set_xticks(pos); ax.set_xticklabels([LABEL[m].replace(", ", "\n").replace(" + ", "\n+ ") for m in ORDER], fontsize=8.5)
    ax.set_ylabel("final relative energy error (%), log scale")
    shots = int(main.shots.iloc[0])
    ax.set_title(f"{n}-qubit Ising ring, at most {int(main.shots_used.max()):,} shots per run\n({shots} shots per setting, {len(main) // len(ORDER)} seeds; dots = individual seeds)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=DPI); plt.close(fig)


def fig_heatmap(df: pd.DataFrame, path: Path, res: Path) -> None:
    """Annotated heatmap: median relative error over (damping, shots per setting)."""
    n = int(_main_subset(df).n.iloc[0])
    base = df[(df.n == n) & (df.kind == "ryrz") & (df.fidelity == 1.0) & (df.seed.isin([0, 1, 2]))]
    shots = sorted(base.shots.unique())
    lams = sorted(base[(base.method == "qng_damped")].lam.unique())
    rows, labels = [], []
    for lam in lams:
        sub = base[(base.method == "qng_damped") & (base.lam == lam)]
        rows.append([100 * sub[sub.shots == s].rel_err.median() for s in shots]); labels.append(f"fixed {lam:g}")
    for m, lab in (("qng_tr", "trust-region (adaptive)"), ("qng_std", f"standard ({LAM_STD:g})"), ("adam_ps", "Adam (no metric)"), ("spsa", "SPSA (no metric)")):
        sub = base[base.method == m]
        rows.append([100 * sub[sub.shots == s].rel_err.median() for s in shots]); labels.append(lab)
    z = np.array(rows)
    pd.DataFrame(z, index=labels, columns=[f"shots_{s}" for s in shots]).to_csv(res / "heatmap_damping_vs_shots.csv")
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    im = ax.imshow(np.log10(np.clip(z, 1e-3, None)), cmap="cividis_r", aspect="auto")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{z[i, j]:.2f}", ha="center", va="center", fontsize=9, color="white" if np.log10(max(z[i, j], 1e-3)) > np.nanmean(np.log10(np.clip(z, 1e-3, None))) else "black")
    ax.axhline(len(lams) - 0.5, color="white", lw=3)
    ax.set_xticks(range(len(shots))); ax.set_xticklabels(shots)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("shots per measurement setting"); ax.set_ylabel("damping lambda of the natural-gradient step")
    ax.set_title("Median relative energy error (%) at equal shot budget\n(3 seeds per cell)", fontsize=10)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax); cb.set_label("log10 of relative error (%)")
    fig.tight_layout(); fig.savefig(path, dpi=DPI); plt.close(fig)


def fig_ecdf(df: pd.DataFrame, path: Path) -> None:
    """Empirical CDF of the cumulative shots needed to first reach 5 % and 3 % relative energy error."""
    n = int(_main_subset(df).n.iloc[0])
    pool = df[(df.n == n) & (df.kind == "ryrz") & (df.fidelity == 1.0) & (df.lam == 0.1)]
    # SPSA sets its own shots per evaluation from the budget, so its runs at other `shots` values are duplicates
    pool = pool[(pool.method != "spsa") | (pool.shots == df["shots"].mode()[0])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, key, tgt in zip(axes, ("shots_to_5", "shots_to_3"), (5, 3)):
        for m in ORDER:
            x = pool[pool.method == m][key].to_numpy()
            total = len(x)
            xs = np.sort(x[~np.isnan(x)])
            if len(xs):
                ax.step(np.r_[xs, pool.shots_used.max()], np.r_[np.arange(1, len(xs) + 1), len(xs)] / total, where="post", color=COLOR[m], lw=2, label=f"{LABEL[m]} ({len(xs)}/{total})")
            else:
                ax.plot([], [], color=COLOR[m], lw=2, label=f"{LABEL[m]} (0/{total})")
        ax.set_xscale("log"); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("cumulative shots (log scale)")
        ax.set_title(f"reach a relative energy error of {tgt} %", fontsize=10)
        ax.legend(fontsize=7.5, loc="upper left", title="(runs reaching target)", title_fontsize=7.5)
    axes[0].set_ylabel("fraction of runs that reached the target")
    fig.suptitle("Shots-to-target, pooled over shot-noise sweep and seeds (exact energy used as oracle for reading the trace)", fontsize=9.5)
    fig.tight_layout(); fig.savefig(path, dpi=DPI); plt.close(fig)


def fig_noise(df: pd.DataFrame, path: Path) -> None:
    """Median error versus shots per setting and versus depolarising fidelity, individual seeds as dots."""
    n = int(_main_subset(df).n.iloc[0])
    base = df[(df.n == n) & (df.kind == "ryrz") & (df.lam == 0.1) & (df.seed.isin([0, 1, 2]))]
    main_shots = df["shots"].mode()[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    panels = (
        (axes[0], base[base.fidelity == 1.0], "shots", "shots per measurement setting", True),
        (axes[1], base[base.shots == main_shots], "fidelity", "global depolarising fidelity of each circuit", False),
    )
    for ax, sub, key, xl, logx in panels:
        for m in ORDER:
            g = sub[sub.method == m]
            xs = sorted(g[key].unique())
            med = [100 * g[g[key] == x].rel_err.median() for x in xs]
            ax.plot(xs, med, "-o", color=COLOR[m], lw=2, label=LABEL[m])
            for x in xs:
                ax.scatter([x] * len(g[g[key] == x]), 100 * g[g[key] == x].rel_err, s=12, color=COLOR[m], alpha=0.45)
        ax.set_yscale("log"); ax.set_xlabel(xl); _plain_log_axis(ax)
        if logx:
            ax.set_xscale("log"); ax.set_xticks(sorted(sub[key].unique())); ax.set_xticklabels(sorted(sub[key].unique())); ax.minorticks_off()
        else:
            ax.invert_xaxis()
    axes[0].set_yticks([2, 3, 5, 10, 20])
    axes[0].set_ylabel("final relative energy error (%)")
    axes[0].set_title("finite-shot noise (fixed total budget)", fontsize=10); axes[1].set_title("gate noise (fixed shots)", fontsize=10)
    axes[0].legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(path, dpi=DPI); plt.close(fig)


def fig_ablation_scaling(df: pd.DataFrame, path: Path) -> None:
    """Redundancy ablation (grouped box plot) and qubit-count scaling (median with seed dots)."""
    n = int(_main_subset(df).n.iloc[0])
    shots = df["shots"].mode()[0]
    base = df[(df.shots == shots) & (df.fidelity == 1.0) & (df.lam == 0.1) & (df.seed.isin([0, 1, 2]))]
    methods = ["adam_ps", "spsa", "qng_std", "qng_tr"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    kinds = [k for k in ("ry", "ryrz", "ryrz_dup") if k in set(base.kind)]
    w = 0.8 / len(methods)
    for j, m in enumerate(methods):
        for i, k in enumerate(kinds):
            v = base[(base.n == n) & (base.kind == k) & (base.method == m)].rel_err.to_numpy() * 100
            x = i + (j - (len(methods) - 1) / 2) * w
            ax.boxplot([v], positions=[x], widths=w * 0.85, showfliers=False, patch_artist=True, boxprops={"facecolor": COLOR[m], "alpha": 0.5}, medianprops={"color": "black"})
            ax.scatter(np.full(len(v), x), v, s=12, color="black", zorder=3)
    ax.set_xticks(range(len(kinds))); ax.set_xticklabels({"ry": "RY only\n(minimal)", "ryrz": "RY + RZ\n(generic)", "ryrz_dup": "RY + RZ + RZ\n(exactly redundant)"}[k] for k in kinds)
    ax.set_yscale("log"); _plain_log_axis(ax); ax.set_yticks([1.5, 2, 3, 5, 8]); ax.set_ylabel("final relative energy error (%)"); ax.set_title("Ablation: ansatz redundancy", fontsize=10)
    for m in methods:
        ax.plot([], [], "s", color=COLOR[m], alpha=0.6, label=LABEL[m])
    ax.legend(fontsize=7.5)
    ax = axes[1]
    sc = base[base.kind == "ryrz"]
    for m in methods:
        g = sc[sc.method == m]
        ns = sorted(g.n.unique())
        ax.plot(ns, [100 * g[g.n == x].rel_err.median() for x in ns], "-o", color=COLOR[m], lw=2, label=LABEL[m])
        for x in ns:
            ax.scatter([x] * len(g[g.n == x]), 100 * g[g.n == x].rel_err, s=12, color=COLOR[m], alpha=0.45)
    ax.set_xticks(sorted(sc.n.unique())); ax.set_yscale("log"); _plain_log_axis(ax); ax.set_yticks([1.5, 2, 3, 5])
    ax.set_xlabel("number of qubits (budget scales with parameters)"); ax.set_ylabel("final relative energy error (%)"); ax.set_title("Scaling in qubit count", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=DPI); plt.close(fig)


def make_all_figures(df: pd.DataFrame, refs: dict, figs: Path, res: Path) -> None:
    """Write all summary tables and figures."""
    _style()
    summary_tables(df, refs, res)
    fig_main(df, refs, figs / "fig1_final_error_violin.png")
    fig_heatmap(df, figs / "fig2_damping_vs_shots_heatmap.png", res)
    fig_ecdf(df, figs / "fig3_shots_to_target_ecdf.png")
    fig_noise(df, figs / "fig4_noise_sweeps.png")
    fig_ablation_scaling(df, figs / "fig5_ablation_and_scaling.png")
