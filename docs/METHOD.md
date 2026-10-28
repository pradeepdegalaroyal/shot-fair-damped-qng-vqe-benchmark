# Method

## Problem and ansatz

* **Hamiltonian.** Transverse-field Ising model (TFIM) on a ring, `H = -J sum Z_i Z_{i+1} - h sum X_i`, `J = h = 1` (critical point), `n = 6` qubits for the main study and `n = 4, 8` for the scaling study. The exact ground energy `E0` comes from dense diagonalisation; the classical reference is the best product-state energy (mean-field baseline).
* **Ansatz.** Hardware-efficient layers: `depth = 3` blocks of single-qubit rotations followed by a ring of controlled-Z (CZ) gates, then a final rotation block. Three kinds: `ry` (RY only, minimal), `ryrz` (RY then RZ, generic, 48 parameters for 6 qubits) and `ryrz_dup` (RY, RZ, RZ: every RZ angle has an exactly redundant partner). Initial angles are Gaussian with standard deviation 0.3 (near the identity, in the spirit of the barren-plateau remedy). Near the identity the RZ layers act on almost-Z-eigenstates, so the metric blocks of those layers are close to singular, which is the regime where damping is supposed to matter.
* **Simulation.** Exact batched statevector in NumPy. Measurements are sampled: the energy needs two settings (Z basis for the ZZ terms, X basis for the X terms); each setting uses `s` shots drawn from the exact outcome distribution. Gate noise is a global depolarising channel with fidelity `f`, which mixes every outcome distribution with the uniform one (a deliberately crude model that rescales all expectation values by `f`).

## Optimisers (all charged in shots)

QNG below means quantum natural gradient; RY and RZ are single-qubit rotations about the Y and Z axes.

| Name | Update | Shots per iteration |
|---|---|---|
| `adam_ps` (standard quantum baseline) | Adam on the parameter-shift gradient | `2P` energy evaluations |
| `spsa` (shot-frugal baseline) | simultaneous-perturbation stochastic approximation, 2 evaluations per step | 2 energy evaluations, with the shots per evaluation raised so that the whole budget is used in at most 800 iterations |
| `qng_std` (standard QNG) | `-eta * (G + lam I)^-1 grad`, `lam = 1e-3` | gradient + one metric setting per rotation layer |
| `qng_damped` | same with fixed `lam` (swept) | same |
| `qng_tr` (trust-region damping) | same, `lam` adapted by the gain ratio `rho = actual / predicted` energy decrease (halve if `rho > 0.75`, double if `rho < 0.25`, clipped to `[1e-4, 10]`) | gradient + metric + 2 extra energy evaluations |

* `P` is the parameter count; one energy evaluation costs `2s` shots. The run stops when the next iteration would exceed the budget `200 000 * P` shots. The metric is the block-diagonal Fubini-Study metric: each rotation layer is one commuting block, estimated from `s` sampled bit strings of the state before the layer, so its noise is real finite-shot noise.
* The damping is scale-normalised, `A = (G + lam I) / (1 + 4 lam)`, so a block with `G = I/4` gets the same step length for every `lam` and `lam -> infinity` is gradient descent with rate `4 eta`. Without this the damping sweep would silently be a learning-rate sweep.
* Step sizes `eta` (Adam learning rate, SPSA gain) are tuned per method from a three-value grid on two tuning seeds (100, 101) that are never used for evaluation; the tuned values are saved in `results/tuned_lr.json`. The shots per setting are not retuned across the sweeps.
* Disclosure: before the final configuration was fixed, a smaller exploratory run (also on the tuning seeds) showed that a Gaussian start with standard deviation 1.0 leaves all methods near the product-state level, which is why the near-identity start and a larger budget were adopted. Hypothesis thresholds were written after that exploration.

## Metrics

* Final relative energy error `(E - E0) / |E0|` of the final iterate, evaluated with the exact state (a simulation-only luxury; it is not used by any optimiser).
* Shots until the exact error first falls below 5 % and 3 % (oracle reading of the trace, for the empirical cumulative distribution figure).
* Correlation energy recovered, `(E_MF - E) / (E_MF - E0)`, where `E_MF` is the best product-state energy: 0 means mean-field level, 1 means exact.
* Hypotheses H1 and H2 (see `RESEARCH_GAP.md`): ratio of medians with a bootstrap 95 % interval (4000 resamples) and a Mann-Whitney U test.

## Experiments

1. **Main comparison**: `n = 6`, generic ansatz, 128 shots per setting, 5 seeds (0-4), all five optimisers.
2. **Noise robustness sweep**: shots per setting 32 / 128 / 512 at fixed total budget per parameter, and depolarising fidelity 1.0 / 0.8 / 0.5 at 128 shots, 3 seeds (0-2). The iteration cap (1500) and the SPSA cap (800 iterations) can make a run use less than its budget; the actual shots used are in `results/runs.csv`.
3. **Damping grid**: fixed `lam` in {1e-4, 1e-2, 0.1, 1, 10} against shots per setting (heatmap), with the adaptive and standard variants as extra rows.
4. **Ablation**: ansatz redundancy (`ry`, `ryrz`, `ryrz_dup`) for Adam, SPSA, standard QNG and trust-region QNG.
5. **Scaling**: `n = 4, 6, 8` with the budget proportional to the parameter count.

## Seeds, determinism and cost

Every run is deterministic given its seed (initial point and shot noise are separate random streams; the initial point is shared by all optimisers at a given seed). Expected wall time for `python experiments/run_all.py` is stated in the README; `--quick` is a one-minute smoke run on 4 qubits.
