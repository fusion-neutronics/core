//! Layout constants: MT slot tables, per-distribution dimension caps,
//! and energy-out / URR column indices shared across the `neutron::xs` split.

/// MT identifiers we care about. yamc's `Reaction` uses the ENDF
/// numbering directly, so these match the standard.
pub const MT_ELASTIC: i32 = 2;
/// Radiative capture -- required as a presence check (every neutron
/// evaluation has MT 102) but `xs_absorption` is computed as
/// `σ_t − σ_e − σ_inelastic_modeled` so it auto-includes every
/// non-modeled channel (capture, (n,p), (n,α), the inelastic-with-
/// charged-particle MTs, etc.) the same way the CPU's
/// `FastXSGrid` populates the absorption slot.
pub const MT_CAPTURE: i32 = 102;

/// MT numbers in the order they appear in per-MT GPU buffers. Slot
/// `k` carries data for `MT_SLOTS[k]`. Slots 0..=40 are the discrete-
/// level + continuum inelastic series MT 51..=91 (single-neutron-out,
/// `MT_YIELDS[k] == 1`). Slot 41 is MT 16 (n,2n) and slot 42 is MT 17
/// (n,3n) -- multi-neutron-out reactions whose `MT_YIELDS[k] > 1`. The
/// kernel multiplies the surviving particle's weight by the slot's
/// yield when sampling that MT, which gives the same expected tally as
/// emitting `yield` independent neutrons (yamc's CPU side clones the
/// primary particle `yield` times; the weight multiplier is the
/// statistically-equivalent variance-reduction form).
///
/// Slots 43..=47 are the slice-F charged-particle-out + neutron MTs:
/// MT 22 (n,n'α), 28 (n,n'p), 32 (n,n'd), 33 (n,n't), 34 (n,n'³He).
/// Each emits a single neutron (`MT_YIELDS[k] == 1`); the charged
/// particle is dropped on the floor (yamc's CPU side does the same --
/// only the first neutron product is sampled in `scatter_with_awr`).
/// CPU classifies these as scattering MTs in
/// `SCATTERING_MTS_NON_INELASTIC`; pulling them into the slot table
/// removes them from the absorption derivation so they're sampled as
/// inelastic-branch channels rather than silently dropped.
///
/// Slots 48..=55 close the remaining neutron-emitting-MT coverage gap.
/// These channels emit one or more neutrons but were missing from the
/// slot table, so the kernel never sampled them and their neutrons were
/// silently counted as absorption (over-absorption -> wrong flux). All
/// are in the CPU's `SCATTERING_MTS_NON_INELASTIC`, so the CPU emits
/// their neutrons; this brings the GPU into line. They reuse the same
/// File-5/6 distribution laws already supported (Kalbach-Mann, n-body
/// phase space, continuous-tabular, level-inelastic), so adding them is
/// pure slot-table wiring -- the extraction loop populates each new
/// slot's xs / Q / yield / distribution automatically.
///   MT 5  (n,misc / catch-all neutron-emitting; multiplicity from the
///         yield curve, often 2 on light/heavy structurals)
///   MT 23 (n,n'3α)        MT 24 (n,2nα)   MT 25 (n,3nα)
///   MT 37 (n,4n)          MT 41 (n,2np)   MT 44 (n,n'2p)   MT 45 (n,n'pα)
///
/// Slots 56..=61 close the same gap for the breakup channels (issue #106):
///   MT 11 (n,2nd)   MT 29 (n,n'3α)   MT 30 (n,2n2α)
///   MT 35 (n,n'd2α) MT 36 (n,n't2α)  MT 42 (n,3np)
/// Nearly absent from ENDF/B-VIII.1 but near-universal in TENDL (MT 11 in
/// 541 of 558 TENDL-2017 nuclides and 1521 of 1649 in TENDL-2025; MT 42 in
/// 525 and 1465), so before this the GPU over-absorbed on almost every
/// TENDL nuclide while ENDF/B-VIII.1 models looked fine.
/// The neutron multiplicity for every slot comes from the first neutron
/// product's yield curve (`yield_per_mt`, evaluated per energy on the
/// GPU); `MT_YIELDS` below is the constant-yield reference (its
/// `round()` value), used only for documentation / test fixtures.
///
/// Owned by yamc-nuclide (issue #111): this table is not just a buffer
/// layout, it is the order the per-collision reaction-type cumulative
/// walk visits candidates in, so the production CPU's
/// `FastXSGrid::sample_inelastic_scatter_reaction` walks it too and a
/// shared `xi_mt` selects the same MT on both backends. yamc-nuclide is
/// the lowest crate that both sides depend on, so it holds the single
/// definition; re-exported here under the `MT_SLOTS` name the kernel,
/// extraction, and tests already use.
pub use yamc_nuclide::nuclide::INELASTIC_MT_SLOTS as MT_SLOTS;
/// Per-slot neutron multiplicity, indexed parallel to `MT_SLOTS`.
/// 1 for MT 51..=91 (single-neutron-out inelastic); 2 for MT 16
/// ((n,2n)); 3 for MT 17 ((n,3n)); 1 for the slice-F charged-particle-
/// out + neutron MTs (slots 43..=47). Slots 48..=55 carry the
/// closed-coverage MTs' nominal multiplicity, and slots 56..=61 the
/// breakup channels' (issue #106). The kernel does NOT read
/// this table at runtime -- it multiplies the surviving neutron's
/// weight by the energy-dependent `yield_per_mt` value (the actual
/// product yield curve). `MT_YIELDS` is kept as the constant-yield
/// reference for documentation and the slot-table sanity tests.
pub const MT_YIELDS: [u32; 62] = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, // MT 51..=91
    2, // MT 16
    3, // MT 17
    1, // MT 22 (n,n'α)
    1, // MT 28 (n,n'p)
    1, // MT 32 (n,n'd)
    1, // MT 33 (n,n't)
    1, // MT 34 (n,n'³He)
    2, // MT 5  (n,misc) -- nominal; actual yield from yield_per_mt
    1, // MT 23 (n,n'3α)
    2, // MT 24 (n,2nα)
    3, // MT 25 (n,3nα)
    4, // MT 37 (n,4n)
    2, // MT 41 (n,2np)
    1, // MT 44 (n,n'2p)
    1, // MT 45 (n,n'pα)
    2, // MT 11 (n,2nd)
    1, // MT 29 (n,n'3α)
    2, // MT 30 (n,2n2α)
    1, // MT 35 (n,n'd2α)
    1, // MT 36 (n,n't2α)
    3, // MT 42 (n,3np)
];
/// Number of MT slots reserved per material in the GPU's flat per-MT
/// buffers. Materials whose nuclides don't have a particular MT have
/// zero xs in that slot -- the kernel just never selects it.
///
/// Re-exported from yamc-physics (issue #111): the shared inelastic
/// dispatcher bakes the slot count into its
/// `slab * MT_INELASTIC_COUNT + slot` addressing, and yamc-physics
/// cannot depend on yamc-gpu. It derives the count from the same
/// yamc-nuclide slot table `MT_SLOTS` re-exports; the assert below
/// keeps the two locked together.
pub use yamc_physics::gpu::flat::inelastic_dispatch::MT_INELASTIC_COUNT;
const _: () = assert!(
    MT_INELASTIC_COUNT == MT_SLOTS.len(),
    "yamc-physics MT_INELASTIC_COUNT must equal MT_SLOTS.len()"
);
/// Bounds of the contiguous discrete-level inelastic MT range (MT 51..=91),
/// used when iterating those levels directly. Code that needs the full set
/// of inelastic slots (including the non-contiguous MT 16 / 17 in slots
/// 41–42) iterates `MT_SLOTS` instead.
pub const MT_INELASTIC_FIRST: i32 = 51;
pub const MT_INELASTIC_LAST: i32 = 91;

/// Histogram (0) / linear-linear (1) interpolation between adjacent `mu`
/// points within a tabulated angular distribution, matching yamc-nuclide's
/// `TabulatedInterp` discriminants. The kernel reads this per (MT slot ×
/// incident-energy index) value to pick the CDF-inversion branch.
///
/// Owned by yamc-physics (issue #111): the shared flat samplers branch on
/// them and the shared per-law extraction
/// (`yamc_physics::gpu::flat::eout_extract`) writes them, and yamc-physics
/// cannot depend on yamc-gpu. Re-exported here so the kernel, extraction,
/// and test use sites are unchanged.
pub use yamc_physics::gpu::flat::elastic_mu_cm::{ANGLE_INTERP_HISTOGRAM, ANGLE_INTERP_LINLIN};

/// Outgoing-energy distribution kind discriminants (`EOUT_KIND_*`).
/// Owned by yamc-physics (issue #111): the shared inelastic dispatcher
/// branches on them, and yamc-physics cannot depend on yamc-gpu. See
/// `yamc_physics::gpu::flat::inelastic_dispatch` for the per-kind
/// documentation; re-exported here so the kernel, extraction, and test
/// use sites are unchanged.
pub use yamc_physics::gpu::flat::inelastic_dispatch::{
    EOUT_KIND_CONTINUOUS_TABULAR, EOUT_KIND_CORRELATED, EOUT_KIND_EVAPORATION,
    EOUT_KIND_KALBACH_MANN, EOUT_KIND_LEVEL_INELASTIC, EOUT_KIND_MAXWELL,
    EOUT_KIND_NBODY_PHASE_SPACE, EOUT_KIND_TABULATED, EOUT_KIND_WATT,
};

/// Per-slot multi-component caps for the Evaporation / correlated
/// angle-energy mixtures (`MAX_EVAP_COMPONENTS` / `MAX_CORR_COMPONENTS`,
/// both 4). Owned by yamc-physics (issue #111): the shared per-law
/// extraction (`EvapSlot::from_evaps` / `CorrSlot::from_components` in
/// `yamc_physics::gpu::flat::eout_extract`) is what enforces them, and
/// yamc-physics cannot depend on yamc-gpu. See that module for the
/// per-constant rationale; re-exported here so the buffer-layout doc
/// comments and use sites are unchanged.
pub use yamc_physics::gpu::flat::eout_extract::{MAX_CORR_COMPONENTS, MAX_EVAP_COMPONENTS};
/// Column count for `urr_xs` -- packed `[total URR cells × URR_XS_COLS]`
/// buffer holding the four URR cross-section values at each (energy,
/// cdf-band) cell: total, elastic, fission, n_gamma. Heating is dropped
/// (the GPU kernel has no heating-from-URR path yet; the smooth value
/// carries on).
pub const URR_XS_COLS: usize = 4;
pub const URR_XS_TOTAL: usize = 0;
pub const URR_XS_ELASTIC: usize = 1;
pub const URR_XS_FISSION: usize = 2;
pub const URR_XS_NGAMMA: usize = 3;
/// Column count for `urr_meta` -- packed `[n_slab × URR_META_COLS]`
/// u32 buffer of per-(material, nuclide) URR flags (issue #210: URR is
/// applied to EVERY in-range URR nuclide, so the buffer is keyed on the
/// global slab index, one row per (material, nuclide), including non-URR
/// slabs with `PRESENT = 0` and the void slab). Layout:
///   0 = URR_META_PRESENT          : 0 = no URR data on this nuclide, 1 = present
///   1 = URR_META_N_ENERGIES       : number of energy grid points
///   2 = URR_META_N_CDF            : number of CDF bands per energy
///   3 = URR_META_INTERP           : 0 = LinLin, 1 = LogLog (matches `UrrInterpolation`)
///   4 = URR_META_INELASTIC_FLAG   : 0 or 1 -- whether smooth inelastic is included in the URR sum
///   5 = URR_META_ABSORPTION_FLAG  : 0 or 1 -- whether URR capture column is full absorption (1) or n_gamma only (0)
///   6 = URR_META_MULTIPLY_SMOOTH  : 0 = table values are absolute XS, 1 = factors to multiply smooth XS
///   7 = URR_META_ZA               : the nuclide's `Z*1000+A` stream key (`Nuclide::urr_stream_key`),
///                                   mixed into the per-collision base seed to give each isotope an
///                                   independent probability-table band (issue #204)
pub const URR_META_COLS: usize = 8;
pub const URR_META_PRESENT: usize = 0;
pub const URR_META_N_ENERGIES: usize = 1;
pub const URR_META_N_CDF: usize = 2;
pub const URR_META_INTERP: usize = 3;
pub const URR_META_INELASTIC_FLAG: usize = 4;
pub const URR_META_ABSORPTION_FLAG: usize = 5;
pub const URR_META_MULTIPLY_SMOOTH: usize = 6;
pub const URR_META_ZA: usize = 7;

/// Column count for `permt_meta` -- the packed per-(slab, MT slot) descriptor
/// for the SPARSE per-MT inelastic storage (issue #212 follow-up). Lives here
/// (not in `transport`, which consumes it) because it is pure buffer-layout
/// metadata shared with the translate layer, which must also build on the
/// macOS stub build where `transport` is `#[cfg]`-gated out.
///
/// The per-MT inelastic XS / yield were stored DENSE: every (slab, MT slot) kept
/// the full coarse grid even though each isotope populates only a few of the
/// `MT_INELASTIC_COUNT` slots, and each only above its reaction threshold
/// (measured 90-99.9% zeros: SS316 10.5x, tungsten 1737x). That dense buffer was
/// both the RAM hog and the buffer that overran the GPU's ~4 GB
/// `maxStorageBufferRange` on a multi-material union grid. `permt_meta` instead
/// records, per (slab, MT slot), the tight range of coarse-grid indices where
/// the slot's XS is nonzero plus the base into the tight value buffers
/// `xs_inelastic_per_mt_sparse` / `yield_per_mt_sparse` (both share the same
/// offset / range). The kernel returns 0 for XS (and 1.0 for yield, the dense
/// default) outside the stored range, which matches the dense buffers exactly,
/// so single-material / homogeneous problems stay bit-identical.
///
/// Layout is row-major `[n_slab × MT_INELASTIC_COUNT × PERMT_META_COLS]`, indexed
/// as `permt_meta[(slab * MT_INELASTIC_COUNT + slot) * PERMT_META_COLS + COL_*]`
/// -- the SAME (slab, slot) ordering as the per-(slab, slot) scalar
/// `q_inelastic_per_mt`.
pub const PERMT_META_COLS: u32 = 3;
/// Base into `xs_inelastic_per_mt_sparse` / `yield_per_mt_sparse` for this
/// (slab, MT slot)'s first stored point.
pub const COL_PERMT_VALUE_OFFSET: u32 = 0;
/// First coarse-grid energy index (relative to this material's own coarse grid)
/// where the slot's XS is nonzero.
pub const COL_PERMT_I_START: u32 = 1;
/// Number of stored (contiguous) coarse-grid points for this slot; `0` means the
/// slot is absent / all-zero (nothing stored, kernel returns 0 XS / 1.0 yield
/// everywhere for it).
pub const COL_PERMT_N_STORED: u32 = 2;
