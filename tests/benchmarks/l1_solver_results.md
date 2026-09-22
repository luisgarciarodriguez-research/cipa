# L1 solver configurations for D7 (2.0.0rc3, P2)

Reference matrix: the first IR-preserving 50,000-row subsample of
`ieee_cis_fraud` after standardization (50,000 x 432, minority 1,750,
IR 27.6:1) — the draw the pipeline itself uses for D2 and D7. Seed 42,
`class_weight="balanced"`, `max_iter=10_000`, one thread per fit, six fits
running concurrently on a 32-core machine.

Raw results in `l1_solver_results.jsonl`, produced by `l1_solver_probe.py`.

## `dual="auto"` already resolves to the primal solver

Verified by comparing `coef_` after a single iteration: for this shape
(50,000 samples, 432 features) `"auto"` selects `dual=False`. The rc1/rc2
baseline that fails to converge **is** the primal solver, so "try `dual=False`
explicitly", one of the configurations the guide asked for, is the baseline.

## Results

| configuration | dual | tol | converged | n_iter | L1 | seconds |
|---|---|---|---|---|---|---|
| baseline (rc1/rc2) | auto → False | 1e-4 | **no** | 10,000 (cap) | 0.16044 | 2,820.7 |
| primal | False | 1e-3 | **yes** | 3,091 | 0.16060 | 1,394.8 |
| primal | False | 1e-2 | yes | 352 | 0.16224 | 213.7 |
| dual | True | 1e-4 | no | 10,000 (cap) | 0.13770 | 615.1 |
| dual | True | 1e-3 | no | 10,000 (cap) | 0.13770 | 613.2 |
| dual | True | 1e-2 | no | 10,000 (cap) | 0.13770 | 615.6 |

The baseline reproduces the published L1 of 0.16044 exactly. Wall times are not
comparable with the 4,288.7 s of the rc1 pilot, which ran uncontended; within
this table they are comparable to each other, and the iteration counts are
comparable to anything.

## The dual solver is not an option

It never converges, and the tolerance makes no difference at all: all three
values are identical because the stopping criterion is never what ends the
fit — the cap always is. Its L1 sits 0.023 below the converged answer, so it is
not arriving at the same place faster, it is not arriving.

## Accepting `tol=1e-3`, rejecting `tol=1e-2`

Against the guide's criteria, in order:

1. **Converges in the hardest case.** `tol=1e-3` converges in 3,091 iterations,
   under the cap. `tol=1e-2` converges in 352.
2. **Stability where rc2 converged.** Across ten small study datasets
   (`breast_cancer_wisconsin`, `pima_diabetes`, `ecoli_imu`, `yeast_me3`,
   `nasa_pc1`, `spambase`, `ozone_level`, `abalone_19`, `mammography`,
   `secom`), moving from 1e-4 to 1e-3 shifts L1 by at most **9.02e-04**, under
   the declared 1e-3; seven of the ten are bit-identical. On the hard subsample
   the shift is 1.6e-04.
3. **Time.** 1,394.8 s under contention, roughly 15 min uncontended at the
   baseline's per-iteration rate. Above the 10-minute target, which the guide
   states as an ideal rather than a bound, and it buys a converged fit.

`tol=1e-2` fails criterion 2, and not marginally. On the `heterogeneous_scales`
regression case, whose features are deliberately on very different scales:

| tol | L1 | n_iter | converged |
|---|---|---|---|
| 1e-4 | 0.13467 | 35 | yes |
| 1e-3 | 0.13867 | 28 | yes |
| 1e-2 | **0.46600** | 12 | yes |

At 1e-2 the fit **reports convergence while landing somewhere else entirely**,
a 3.5x change in L1. That is the worst failure mode available: a wrong number
with no warning attached. Ill-conditioned data is where a loose tolerance lies,
and it is exactly where D7 is asked to work.

## Why the default stays at 1e-4

`tol=1e-3` changes `heterogeneous_scales` by 0.004 in L1 and 0.002 in D7, which
breaks `test_d7_matches_v1_2_1_where_l1_converged` for that case. The guide
requires the v1.2.1 regression to stay exact under the new defaults, so the
package ships `DEFAULT_SVC_TOL = 1e-4` and exposes `svc_tol` as a parameter
instead of changing what every consumer computes.

**The recommendation for the study is to run with `svc_tol=1e-3`**, set in
cipa-extended's protocol. That is where the decision belongs: it moves reported
values, and how the extension treats an incomplete fit is a validity question,
not a packaging one. With the default, `n_iter` and `converged` now say plainly
which datasets are reporting the error of a fit that never finished.
