# Task: finish issue #404 (wire the fluence bound into the transmutation driver)

Work in `/home/jon/yamc-org/yamc` on a new branch off `main`. Read
https://github.com/fusion-neutronics/yamc/issues/404 first, including all three of
its comments: they carry measurements you should not have to redo.

## Background: what already landed

**#403** removed the `reduce_transport_chain` hop count from
`Model.simulate_transmutation`. Products are now the material's whole reachable
closure (`yani::reachable_nuclides`, `crates/yani/src/chain.rs:366`), loaded at
`LoadScope::activation` rather than transport scope
(`crates/yamc/src/transmute.rs:227`). A hop count could not express the right answer
because how far a product chain runs depends on fluence, not graph distance.

**#406** added `yani::populated_nuclides` (`crates/yani/src/chain.rs:420`) with six
tests and a measurement example (`crates/yani/examples/bound_prune.rs`). **Nothing
calls it.** Wiring it in is this task.

## What the bound does

`reachable_nuclides` answers a graph question. From almost any seed the walk
saturates on one large strongly connected component, so Fe56, H2O, SS316 and
Zircaloy all reach the same 1060 nuclides, 461 of them reactive. A coupled solve of
an Fe56 sphere populates 49.

`populated_nuclides` gives each node an upper bound on the density it could attain
over the irradiation and drops any whose bound stays under the floor the stepper
already applies (`1e-30`, `crates/yani-transmute/src/transmutation_stepper.rs:88`).
Depth then tracks fluence. There is no new user setting and there must not be one:
removing settings is the entire point of #401.

Signature:

```rust
pub fn populated_nuclides<F>(
    chain: &HashMap<String, ChainNuclide>,
    seeds: &HashMap<String, f64>,   // nuclide -> initial atom density
    total_time: f64,                // seconds
    floor: f64,                     // 1e-30, absolute density
    reaction_rate: F,               // (parent, reaction_kind) -> per-atom rate [1/s]
) -> HashSet<String>
```

Run `cargo run --release -p yani --example bound_prune` to see it prune 461 reactive
nuclides to 82-95 while keeping all 49 that a real solve populates.

## Why it is worth doing

The cost of carrying products is **not** loading, it is per-collision tally scoring.
`TransmutationTallies` registers every nuclide in `material.nuclide_data` that the
chain knows (`crates/yani-transmute/src/transmutation_tallies.rs:348`) and scores
sigma x track-length for each on every segment, so the work is
`n_nuclides x n_MTs x n_segments`.

Measured, coupled Fe56 sphere, one year at 1e14 n/s, against pre-exported chains:

| particles | pruned (50 reactive) | full (461 reactive) | saving |
|---|---|---|---|
| 2 000 | 2.90 s | 5.26 s | 2.4 s |
| 20 000 | 2.98 s | 6.50 s | 3.5 s |
| 100 000 | 3.29 s | 12.11 s | 8.8 s (73%) |

The pruned run is nearly flat, the full one scales steeply, so the saving grows
without bound in particle count. **RAM is not the prize**: pruning saves under 5%,
because #403's activation scope already made products cheap to hold. Do not justify
this work on memory.

## The structural obstacle

A product's reaction rate comes from CE scoring **during** transport. There is no
stored group flux to fold against afterwards, so a nuclide must be in the tally
before the solve to get a rate. You cannot compute the bound from rates that do not
exist yet.

The intended shape is therefore:

1. A cheap scouting solve with seeds only (few nuclides in the tally, so it is fast)
   yields real rates for the seeds.
2. Run the bound with those rates to pick the product set.
3. Load only that set at activation scope, rebuild the tallies, and continue as now.

In `coupled` mode compute the bound once and reuse it across steps so the extra
solve amortises. In `independent` mode there is a single solve today
(`crates/yamc/src/transmute.rs:357`), so this adds a second one; check the net is
still a win at realistic particle counts before keeping it.

## The unresolved design question (decide this first)

A driver cannot rate a nuclide it has not loaded. What `reaction_rate` returns for
those decides whether the result is a bound at all.

- **Returning zero is unsound.** Each unloaded nuclide is individually under the
  floor, but a node fed by enough of them clears it, and zeroed edges hide that. The
  test `many_sub_floor_parents_can_clear_the_floor_together` in
  `crates/yani/src/chain.rs` builds exactly this: 200 parents at 1e-32 sum to 2e-30
  and clear a 1e-30 floor, and it asserts both that a ceiling keeps the sink and that
  zeroing loses it.
- **A one-shot cross-section ceiling is sound but useless.** At plausible sigma the
  bound degenerates to the whole closure, buying nothing. `bound_prune` shows this:
  the `sigma = 100 b` rows return 1060 nodes / 461 reactive.

Two candidate resolutions, both unvalidated:

- **Iterate.** Grow the loaded set in rounds and re-run the bound as more parents
  become rated, terminating when a ceiling sweep adds nothing. Rigorous, but each
  round that needs new rates needs another solve unless you fold offline.
- **In-degree margin.** The chain's in-degree is mean 4.36, max 16, so the
  contribution missed by zeroing one generation is bounded by roughly
  `in_degree x floor`. Testing against `floor / margin` would cover it. Cheap and
  needs no extra solve, but the compounding across generations must be bounded
  properly, not hand-waved.

Pick one deliberately and justify it in the PR. Do not ship a guess.

## Validation required before opening the PR

1. `cargo test --workspace --features mesh --exclude yamc-gpu`
2. `pytest packages/yamc-core/tests`
3. `cargo fmt --all --check`, `cargo clippy --workspace --features mesh --exclude yamc-gpu --all-targets`, `ruff check .`
4. Coupled and independent Fe56 probes against `main`, at 2 000 and 100 000
   particles, confirming the populated inventory is unchanged and the time saving is
   real.
5. The 72 FNS foils in `~/openmc_activator/fns` and a spread of PNNL materials, vs
   `main`. #403 got 0 disagreements with a median density error of exactly 0.0; hold
   that bar.

## Practical notes, learned the hard way

- Build with `maturin develop --release` from `packages/yamc-core`, since maturin's
  `-m` takes a Cargo manifest rather than a pyproject. The venv is `.venv` at the
  repo root.
- `cargo run --bin stub_gen` and any pyo3 test binary need
  `LD_LIBRARY_PATH=/home/jon/.pyenv/versions/3.13.13/lib`. Regenerate stubs with
  `python scripts/build_stubs.py`, never the raw binary, and commit the result.
- **Cap memory on every test run.** A transmutation change earlier took the whole
  machine down at 27.4 GB. Use
  `systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 -- <cmd>`, which
  confines a kill to the test tree. Do **not** use `ulimit -v`: the Arrow data is
  memory-mapped, so it aborts on virtual reservations rather than real usage.
- Validate on **both** transmutation paths. `Material.transmute` (transport-free,
  `crates/yani-transmute/src/material_transmute.rs`) and
  `Model.simulate_transmutation` (transport-coupled, `crates/yamc/src/transmute.rs`)
  respond in opposite directions to the same change. #403's OOM happened precisely
  because only the first was measured.
- Measure in the transport-dominated regime. At 2 000 particles the pruning looks
  worthless; at 100 000 it is 73% of runtime. A cheap benchmark will mislead you.
- No deprecated aliases or compatibility shims, and no new user-facing knob.
- No em dashes anywhere. No AI attribution in commits, PR bodies or comments.
- GitHub Actions was in a major outage on 2026-08-06, so nothing merged that day was
  CI-verified. If Actions is back, a run on `main` is worth having for macOS,
  Windows, WASM and MPI, which cannot be tested locally.

## Deliverable

A PR closing #404: the bound wired into both transmutation modes, the design
question resolved and explained, the validation above in the PR body, and the
`bound_prune` example updated or removed if it no longer reflects how the bound is
driven.
