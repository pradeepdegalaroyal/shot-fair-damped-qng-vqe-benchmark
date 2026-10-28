# Research gap: does a damped quantum natural gradient still pay off when every shot is charged?

All results in this repository are classical simulations (exact statevectors with sampled measurement outcomes, up to 8 qubits). Nothing here is a hardware result and nothing here claims a quantum advantage.

## State of the art (from the supplied literature)

Papers are cited as author, year and digital object identifier (doi).

* **Variational eigensolvers and their optimisers.** McClean et al., 2016 (doi:10.1088/1367-2630/18/2/023023) set up the variational hybrid quantum-classical loop and report that derivative-free optimisers can save up to three orders of magnitude of cost over earlier ones. McClean et al., 2018 (doi:10.1038/s41467-018-07090-4) show that for random circuits the probability of a non-zero gradient shrinks exponentially with qubit number, which is why small-angle (near-identity) starting points are used in practice. Yuan et al., 2019 (doi:10.22331/q-2019-10-07-191) review the variational principles (Rayleigh-Ritz, Dirac-Frenkel, McLachlan, time-dependent) that give rise to metric-based updates.
* **Geometry-aware updates.** Alwan et al., 2027 (doi:10.1007/978-3-032-30497-1_18) point out that the quantum natural gradient (QNG) update is ill-posed when the Fubini-Study metric is singular, for example under redundant parameterisation. They propose a Tikhonov-damped trust-region version with a gain-ratio damping rule. They report that it is stable under finite-shot noise and reaches target energies in 1-2 iterations in 2-12 qubit variational quantum eigensolver (VQE) experiments.
* **Noise and hardware.** Temme et al., 2017 (doi:10.1103/PhysRevLett.119.180509) and Kjaergaard et al., 2020 (doi:10.1146/annurev-conmatphys-031119-050605) fix the noisy intermediate-scale quantum (NISQ) setting in which short circuits and limited shot counts are the binding resources. Wang et al., 2026 (doi:10.1088/2058-9565/aea01f) report a hardware run in which circuit depth is "the present hardware bottleneck".
* **Simulation and statistics practice.** Nguyen et al., 2027 (doi:10.1007/978-981-92-2587-3_22) speed up classical simulation of variational quantum classifiers on graphics processing units (GPUs); Pehlivanoglu et al., 2026 (doi:10.1088/2058-9565/aeaf7a) ask how much interference work can be skipped in approximate simulation; Pinheiro et al., 2027 (doi:10.1007/978-3-032-39898-7_12) compare models over several independent initialisations with significance tests.

## Related work table

| Paper | Year | Role | What it does | What it does NOT cover |
|---|---|---|---|---|
| McClean et al., doi:10.1088/1367-2630/18/2/023023 | 2016 | landmark | Theory of the variational hybrid loop; derivative-free optimisation savings | Metric-based updates; a shot ledger that includes the extra circuits an optimiser needs |
| McClean et al., doi:10.1038/s41467-018-07090-4 | 2018 | landmark | Exponentially vanishing gradients for random circuits | Which optimiser to use once the start point is chosen well; finite-shot noise in the optimiser |
| Temme et al., doi:10.1103/PhysRevLett.119.180509 | 2017 | landmark | Zero-noise extrapolation and quasi-probability error mitigation for short circuits | Optimiser design; the shot cost of mitigation inside a training loop |
| Kjaergaard et al., doi:10.1146/annurev-conmatphys-031119-050605 | 2020 | landmark | Review of superconducting-qubit hardware and early NISQ algorithms | Any optimiser comparison |
| Yuan et al., doi:10.22331/q-2019-10-07-191 | 2019 | review | Variational principles for real and imaginary time, mixed states, ansatz design | Measured shot cost of estimating the metric, and its noise |
| Koch et al., doi:10.1140/epjqt/s40507-022-00138-x | 2022 | review | Status and roadmap of quantum optimal control, including open systems | Benchmarks of variational-circuit optimisers |
| Alwan et al., doi:10.1007/978-3-032-30497-1_18 | 2027 | adjacent | Regularised QNG trust region; stable under singular metrics and finite shots; 2-12 qubit VQE | Total-shot accounting (the abstract reports iterations), a shot-frugal classical baseline, an explicit gate-noise sweep |
| Nguyen et al., doi:10.1007/978-981-92-2587-3_22 | 2027 | recent | Faster GPU simulation of variational classifiers | Sample complexity of training on a device |
| Pehlivanoglu et al., doi:10.1088/2058-9565/aeaf7a | 2026 | recent | Approximate path-sum simulation by pruning interference | Variational training, optimiser stability |
| Wang et al., doi:10.1088/2058-9565/aea01f | 2026 | recent | Open-system simulation primitive with a hardware depth bottleneck | Variational optimisation under a shot budget |
| Dour, doi:10.1007/s10751-026-02799-x | 2026 | recent | Exact solvers and VQE on small molecular fragments, with noise and sampling sensitivity | Comparison of optimiser geometry at equal shots |
| Pinheiro et al., doi:10.1007/978-3-032-39898-7_12 | 2027 | recent | Quantum generative models compared over independent initialisations with significance tests | Eigensolvers; natural-gradient methods |
| Adnan and Fallon, doi:10.1007/978-3-032-32035-3_3 | 2027 | adjacent | Comparative study of quantum and classical machine learning in simulation | Controlled, shot-accounted optimiser comparison |

## The gap

Recent work argues that regularised geometry-aware optimisers are *robust* to finite-shot noise and redundant parameters, but robustness is judged per iteration. On a device the currency is circuit executions: a natural-gradient step also needs metric circuits, and a trust-region rule needs extra energy evaluations. A shot-frugal optimiser such as simultaneous-perturbation stochastic approximation (SPSA) needs only two energy evaluations per step regardless of parameter count. Whether damped QNG beats the standard parameter-shift gradient method and SPSA *at equal total shots*, and how that changes with shots per setting, gate noise and redundant parameters, is not measured in the supplied papers.

**Novelty in one sentence:** a ledger-based, equal-shot benchmark of standard, fixed-damping and trust-region QNG against parameter-shift Adam and SPSA, over a shot-noise sweep, a depolarising-noise sweep, an exactly redundant ansatz and 4-8 qubits.

## Testable hypotheses and success metric

Problem: transverse-field Ising ring at the critical point, 6 qubits (also 4 and 8), hardware-efficient ansatz started near the identity. Metric: exact (noise-free) relative energy error of the final iterate, `(E - E0) / |E0|`, in percent, median over seeds. The best product-state (mean-field) energy is the classical reference; it sits at 2.94 % relative error for 6 qubits.

* **H1 (stability).** The median final error of standard QNG (damping 1e-3) is at least 2x that of trust-region QNG, with a bootstrap 95 % interval of the ratio above 1.
* **H2 (shot-fair advantage).** The median final error of the better of {Adam with parameter shift, SPSA} is at least 1.25x that of trust-region QNG, with a bootstrap 95 % interval above 1.

Either hypothesis is falsified if its ratio or its interval fails. Both are tested on the main setting (5 seeds) and pooled over the noise sweeps (3 seeds per cell). With so few seeds the tests have limited power; intervals are reported rather than only point estimates.
