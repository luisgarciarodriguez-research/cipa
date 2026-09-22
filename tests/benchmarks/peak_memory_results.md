# Peak memory of the brute-force neighbour search (2.0.0rc3, P1.4)

Measured with `peak_memory_probe.py`, one configuration per process, on a
32-core machine with 123 GB of RAM, scikit-learn 1.7.2, numpy 1.26.4.
Peak is the process high-water mark (`VmHWM`). Matrices are synthetic: memory
depends on shape, not on values.

## What the guide expected, and what is actually true

`cipa_2.0.0rc3_handoff.md` §2.4 assumed brute force materialises a
`chunk_size x n_fit` distance block that `chunk_size` has to bound. It does
not. Since scikit-learn 1.1, `kneighbors(algorithm="brute")` on float64
Euclidean data dispatches to the Cython `ArgKmin` reduction, which streams
with a per-thread heap. A 65,536 x 50,000 block would be 26 GB; the measured
addition is 33 MiB.

`chunk_size` still bounds the output arrays `kneighbors_excluding_self`
allocates per block, `chunk x (k+1) x 8 x 2` bytes, which is the ~8 MiB
difference between the two columns below. It does not bound the search.

## Case A — the D2/D7 subsample shape: 50,000 x 432, 50,000 queries, k=5

Data: 164.8 MB. Figures are MiB **added** over the data.

| `working_memory` | `chunk_size` = 1,024 | `chunk_size` = 65,536 |
|---|---|---|
| 128 MiB | 24.5 | 32.5 |
| 1,024 MiB (default) | 25.7 | 33.1 |
| 4,096 MiB | 25.1 | 33.0 |

Neither control moves the peak. Results are identical across all six runs.

## Case B — the D3 shape: 425,741 queries against 2,520,798 x 70, k=5

| quantity | value |
|---|---|
| data | 1,346.3 MB |
| peak before the search | 1,750.3 MiB |
| **peak during the search** | **1,843.7 MiB** |
| added by the search | 93.4 MiB |
| wall time | 541.2 s |

**Maximum observed across every configuration: 1,843.7 MiB**, of which the
brute-force search accounts for 93.4 MiB and the dataset itself for the rest.
Switching the default above 15 features to brute force therefore costs no
meaningful memory, which was the risk §2.4 asked about.

## Caveat on the wall time

541 s is brute force on synthetic data. Brute force costs essentially the same
on any matrix of that shape, but the 4,094 s that `timing_pilot.json` records
for this step under rc2 was `ball_tree` on the real `cic_ids_2017`, and tree
performance does depend on how the data is distributed. Read the comparison as
an estimate; the measurement that counts is the rerun on the real datasets.

## Version dependency

If a future scikit-learn stops dispatching to `ArgKmin`, the fallback is
`pairwise_distances_chunked`, whose footprint is governed by
`sklearn.get_config()["working_memory"]` and would have to be measured again.
`test_brute_force_euclidean_uses_the_argkmin_reduction` fails first if that
happens.
