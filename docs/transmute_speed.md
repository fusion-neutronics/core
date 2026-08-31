# Speeding up `Material.transmute()`

The result of working through [issue #576](https://github.com/fusion-neutronics/core/issues/576),
one finding per pull request, each measured against the one before it and each
proved to leave the inventories bit-identical.

## What was measured

`tools/bench_transmute.py`, on an SS316-like steel (six elements, ~430
reachable nuclides on ENDF/B-8.1) at CCFE-709 under a 1/E-plus-14-MeV spectrum,
over a one-hour pulse and a one-hour cooldown. Four shapes:

| case | what it is |
|---|---|
| **once** | the first `transmute` in a fresh interpreter, so it carries the Arrow decode of every reachable chain nuclide |
| **repeat** | the median of the calls after it on the same material, with the previous results object dropped each time |
| **once, uncertainty** | the same first call with `data_uncertainty=`, at a fixed 128 samples |
| **repeat, uncertainty** | the median of the uncertainty calls after it |

Every row below was re-measured back to back on one machine (32 cores, AMD
Ryzen AI Max+ 395) against the same harness, so the columns are comparable
with each other and not only with their neighbours. Run-to-run spread is
about 3%.

Regenerate with:

```bash
python tools/bench_transmute.py --label "<name>"   # appends a row
python tools/plot_transmute_bench.py               # this table and the graph
```

## The running table

| step | once | repeat | once, uncertainty | repeat, uncertainty |
|---|---|---|---|---|
| vanilla | 5.10 s (1.00x) | 4.89 s (1.00x) | 16.81 s (1.00x) | 16.79 s (1.00x) |
| 9 release profile | 5.18 s (0.98x) | 4.92 s (0.99x) | 16.82 s (1.00x) | 16.66 s (1.01x) |
| 1a interior slice | 2.93 s (1.74x) | 2.73 s (1.79x) | 12.52 s (1.34x) | 12.41 s (1.35x) |
| 8 per-call overheads | 2.45 s (2.08x) | 2.26 s (2.16x) | 12.24 s (1.37x) | 12.11 s (1.39x) |
| 2 one walk | 2.58 s (1.97x) | 2.35 s (2.08x) | 12.32 s (1.36x) | 12.23 s (1.37x) |
| 6 two allocations | 2.51 s (2.03x) | 2.36 s (2.07x) | 12.44 s (1.35x) | 12.19 s (1.38x) |
| 3 keep the data | 2.48 s (2.06x) | 0.40 s (12.32x) | 12.26 s (1.37x) | 10.08 s (1.67x) |
| 7 release the GIL | 2.66 s (1.92x) | 0.40 s (12.31x) | 12.19 s (1.38x) | 10.10 s (1.66x) |
| 5a parallel load | 0.76 s (6.73x) | 0.40 s (12.32x) | 10.45 s (1.61x) | 10.09 s (1.66x) |
| 5b parallel collapse | 0.51 s (10.04x) | 0.14 s (35.40x) | 10.24 s (1.64x) | 9.84 s (1.71x) |
| 4 parallel replicas | 0.49 s (10.40x) | 0.14 s (35.50x) | 1.65 s (10.19x) | 1.26 s (13.30x) |
| 5c parallel setup | 0.51 s (9.99x) | 0.14 s (35.25x) | 1.38 s (12.19x) | 1.01 s (16.69x) |
| 1b indexed reads | 0.48 s (10.52x) | 0.13 s (37.96x) | 1.35 s (12.50x) | 0.98 s (17.19x) |

![wall time and cumulative speedup across the twelve findings](transmute_speed.png)

## Reading it

Four things carry almost all of it, and they are not the four the issue
expected:

- **finding 1a** (bisect for a group's evaluation points instead of rescanning
  the grid) is the single largest sequential win: 1.8x on the plain cases and
  1.35x on the uncertainty ones, from the collapse alone going 2.6 s -> 0.27 s.
- **finding 3** (load the cross sections into the material and keep them) is
  what makes a *repeated* call cheap: 2.37 s -> 0.40 s, because the Arrow decode
  stopped being redone on every call.
- **finding 5a and 5b** (parallel decode, parallel collapse) take the first call
  from 2.66 s to 0.51 s.
- **finding 4** (run a block of uncertainty replicas at once) takes the
  uncertainty cases from ~10 s to ~1.4 s.

Three items measured as **nothing** on this path, and are worth recording as
such rather than quietly claimed:

- **finding 9** (a release profile with thin LTO) was predicted at 5-15%. It
  measures as noise, because the cross-crate call it was supposed to inline --
  `Reaction::cross_section_at` -- already carries `#[inline]`.
- **finding 8**'s four items together are worth about 5 ms of a 2.5 s call. The
  issue put `reaction_type_to_mt` alone at 40-60 ms; the whole of
  `activation_mts` measures 0.2 ms. What *did* pay in that PR was the same
  bisection applied to the branching fold's own integrator, which #576 does not
  mention: the isomeric-branching overlay went from 0.35 s to 0.08 s per call.
- **finding 2** (one walk per group, skip empty groups) is a wash on a spectrum
  that fills the structure. It is 64x on the collapse for a monoenergetic 14 MeV
  source in CCFE-709, which is the case it exists for.

And two are worth something the benchmark cannot show:

- **finding 7** (release the GIL) is zero on one call. It is what lets two
  solves in two Python threads take 0.25 s instead of 0.49 s.
- **finding 6** (two allocations) is inside the spread on its own. It is a
  prerequisite for finding 4's number being attributable rather than partly a
  measurement of refcount contention.

## The accuracy constraint

Every step is **bit-identical** to `vanilla`:

- `tools/bench_transmute.py --compare vanilla` compares the final inventory as
  hex floats, and the sigmas beside it;
- `cargo run --release -p yani-transmute --example collapse_golden` prints the
  collapse itself as bit patterns -- per-group terms, one-group rates and
  fission-yield weights -- and its output is byte-identical throughout;
- `crates/yani-transmute/tests/thread_count_determinism.rs` runs the same case
  in a 1-thread and a 7-thread pool and compares the inventories, the
  **per-replica** ensemble, the sigmas, the sample count and the truncation
  counters to the last bit.

One thing had to be fixed before any of that could be checked at all:
`Material::nuclides` is a `HashMap`, and `get_atoms_per_barn_cm` summed the
fractions straight out of it, so 103 of 200 final densities disagreed at ~1e-15
between one process and the next. That is the same defect as #502, one layer up.

## Build flags, measured

Three build-level levers were checked after the twelve findings, on the same
`cargo run --release -p yani --example cram_perf` (the 3820-nuclide ENDF/B-VIII.1
SFR chain), best of three or more runs on one machine (AMD Ryzen AI Max+ 395).
Recorded here rather than adopted, because two of the three are worth nothing
and that is only obvious once someone has measured it.

| build | best | against baseline |
|---|---|---|
| baseline `--release` | 313.0 ms | |
| faer without the `rayon` feature | 314.1 ms | **nothing** |
| `RUSTFLAGS="-C target-cpu=x86-64-v2"` | 317.3 ms | **nothing**, if anything slower |
| `RUSTFLAGS="-C target-cpu=native"` | **302.6 ms** | **~3.3%** |

- **faer's `rayon` feature buys nothing here**, because `cram.rs` passes
  `Par::Seq` explicitly and that is the only `Par` in the crate. The comment in
  `crates/yani/Cargo.toml` used to credit it with 5-6%; it now records the
  measurement instead. The feature stays only because the target gate around it
  keeps `spindle -> atomic-wait` off wasm32.
- **`x86-64-v2` is not worth adding to the wheel jobs.** It was the obvious
  candidate for a published-wheel ISA baseline, and on this workload it does not
  pay for itself. The finding-9 row above measures LTO only and does not speak to
  the ISA baseline, so this is the row that does.
- **`native` is worth about 3%**, and must never go into a published wheel: it
  emits instructions the build machine has and the user's may not, which is a
  SIGILL on older hardware and under Rosetta 2. It is documented in
  `developer_info.md` as a from-source escape hatch for people building on the
  machine they will run on.

## Explicitly out of scope: checked, and wrong to do

Carried over from issue #576 so the negative results survive the issue being
closed. The first one is a wrong-answer trap rather than a missed optimisation.

- **Parallelising the 24 CRAM poles.** `crates/yani/src/cram.rs` is the
  *incomplete partial fraction* form: `y` is seeded from `n0`, each pole's
  right-hand side is filled from the *current* `y`, and each pole accumulates
  back into it. Pole k+1's input is pole k's output. It is **not** 24 independent
  shifted solves, and anyone reading it as the classical partial-fraction CRAM
  will introduce a wrong answer.
- **Switching `cram.rs` off `Par::Seq`.** faer partitions work by thread count,
  so the same matrix can round differently on a 22-core box than an 8-core one.
  `crates/yani/tests/matrix_reproducibility.rs` exists to prevent exactly that.
- **Symbolic-LU caching in `cram_solve_sparse`.** The 24 numeric factorizations
  dominate the symbolic phase by an order of magnitude, and a pattern-keyed cache
  needs an `O(nnz)` hash (most of what it saves) plus invalidation when the BFS
  set changes between steps, which it legitimately does as nuclides cross the
  density floor. A few percent for real complexity; parallelise replicas instead.
- **SIMD or `fast-math` on the collapse.** It is a serial float reduction, and
  reassociating it breaks the bit-identity everything else rests on.
- **Parallelising the nominal step loop, or the step loop inside a replica.**
  The current material is the state, and each step is the next step's input.
- **Parallelising `Ensemble::push`.** Welford is a float recurrence and the
  sample index is the replica index.
- **Parallelising the inner group loop.** The trapezoid sums are ordered float
  accumulations over 709 groups; a parallel reduction changes the rounding of a
  loop about 200x shorter than the one above it.
