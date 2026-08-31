//! Multi-tally accumulator pack for the multi-cell transport kernel.
//!
//! The kernel writes one atomic-add per tally per step into a flat
//! `tally_out` u64 buffer; each tally's slot range, cell-binning,
//! energy-binning, and score-kind discriminant are packed into the
//! six side buffers below. Building the pack is one-shot host-side
//! work in `yamc::gpu::dispatch::run_on_gpu`; the kernel just
//! consumes it.
//!
//! # Layout
//!
//! For tally `t`:
//! - `score_kinds[t]` -- `SCORE_FLUX` / `SCORE_TOTAL` / `SCORE_ABSORPTION`
//! - `n_cells_per_tally[t]` -- number of cell bins
//! - `cell_to_bin[t * n_geom_cells + c]` -- bin index for geometry
//!   cell `c`, or `NOT_IN_TALLY` (`u32::MAX`) if the tally's
//!   `CellFilter` doesn't include `c`
//! - `n_bins_per_tally[t]` -- number of energy bins
//! - `log_edges[edges_offsets[t]..edges_offsets[t] + n_bins_per_tally[t] + 1]`
//!   -- log-space bin edges (inclusive at the lower edge,
//!   exclusive at the upper)
//! - `n_mesh_per_tally[t]` -- number of spatial mesh (voxel) bins, `1` for a
//!   non-mesh tally (issue #234). Mesh is the innermost flat dimension.
//! - `tally_out[out_offsets[t] + (bin_cell * n_bins_per_tally[t] + bin_e) *
//!   n_mesh_per_tally[t] + voxel_bin]` -- fixed-point accumulator (1 / 2^30
//!   scale, same as slice A). For a non-mesh tally `n_mesh == 1` and
//!   `voxel_bin == 0`, collapsing to `out_offsets[t] + bin_cell * n_bins + bin_e`.
//!
//! Per kernel step the inner loop walks `0..n_tallies`, computes
//! `track_length × score_xs × weight` per tally, and atomic-adds
//! the fixed-point bits into the indexed slot.

/// Score kind: track length only (raw flux estimator). Multiplied
/// by `1.0`.
pub const SCORE_FLUX: u32 = 0;
/// Score kind: track length × σ_t at the particle's current
/// energy. Equivalent to the CPU's `Score::Total`.
pub const SCORE_TOTAL: u32 = 1;
/// Score kind: track length × σ_a at the particle's current
/// energy. Equivalent to the CPU's `Score::Absorption`.
pub const SCORE_ABSORPTION: u32 = 2;
/// Score kind: track length × σ_MT at the particle's current
/// energy, where MT is selected per tally via `score_data[t]`,
/// indexing into the `xs_score_per_mt` buffer the dispatch
/// builds. Used for arbitrary reaction-rate scores (`elastic`,
/// `inelastic`, `fission`, per-MT integers, …) and as the
/// sub-routine for KERMA-shape scores.
pub const SCORE_PER_MT: u32 = 3;

/// Mesh-tally kind discriminant for [`TalliesPack::mesh_kind`] (issue #234).
/// A tally with `MESH_NONE` has no spatial mesh dimension (`n_mesh_per_tally
/// == 1`, `voxel_bin` forced to 0); the flat index collapses to the pre-mesh
/// `(cell_bin * n_bins + energy_bin)` form, byte-identical to a run without any
/// mesh support. The other kinds run a per-step voxel walk (track-length) or a
/// single `get_bin` lookup (collision estimator) to fan a step's contribution
/// across the voxels it crosses.
pub const MESH_NONE: u32 = 0;
/// Axis-aligned rectangular mesh, row-major voxel index `(iz*ny+iy)*nx+ix`.
/// (Discriminant 2 was a Morton/Z-order variant, removed in issue #337 after
/// it measured slower than row-major on both backends.)
pub const MESH_RECT_ROWMAJOR: u32 = 1;
/// Cylindrical `(r, φ, z)` mesh, voxel index `(iz*nphi+iphi)*nr+ir`.
pub const MESH_CYLINDRICAL: u32 = 3;

/// Number of `f64` words in the rectangular-mesh descriptor packed into
/// [`TalliesPack::mesh_params`]: `lower_left[3]`, `upper_right[3]`,
/// `inv_width[3]`, `width[3]`, `shape[3]` (shape stored as `f64`, cast back to
/// `u32` in the kernel). `upper_right` is packed explicitly (rather than
/// recomputed as `ll + w*n`) so the in/out-of-mesh bounds test is bit-exact
/// against the CPU `RegularRectangularMesh`.
pub const MESH_RECT_PARAMS: usize = 15;
/// Number of leading `f64` header words in the cylindrical-mesh descriptor:
/// `origin[3]`, `nr`, `nphi`, `nz`, `full_phi` (0/1). The three grids
/// (`r_grid` of `nr+1`, `r_grid_sq` of `nr+1`, `phi_grid` of `nphi+1`,
/// `z_grid` of `nz+1`) follow the header in that order.
pub const MESH_CYL_HEADER: usize = 7;

/// Number of `f64` words an energy-function table occupies in
/// [`TalliesPack::efunc_params`] for `n_points` tabulated points: one header
/// word (`n_points`), the `n_points` energies, then the natural-cubic-spline
/// coefficients, four (`a`, `b`, `c`, `d`) per interval (issue #271).
///
/// `EnergyFunctionFilter::new` requires `n_points >= 4`, so a packed table is
/// never degenerate; an absent filter is encoded as an EMPTY range rather than
/// a zero-point table.
#[inline]
pub const fn efunc_table_len(n_points: usize) -> usize {
    1 + n_points + 4 * (n_points - 1)
}

/// Default per-step contribution scale for the fixed-point
/// atomic accumulator. Per-step contributions are
/// `track_length × score_xs × weight`, which for flux /
/// reaction-rate scoring is order-of-unity in cm·barn-cm units;
/// `2^30 ≈ 1.07e9` keeps ~9 digits of precision while staying
/// well below the `i64` range across millions of source
/// particles.
pub const DEFAULT_FIXED_POINT_SCALE: f64 = 1_073_741_824.0;
/// Fixed-point scale used for KERMA-shape MTs (heating 301,
/// heating-local 901, damage-energy 444). Their per-step
/// contributions are `eV`-scale (~1e6–1e7 per step in
/// actinides), so the default scale would overflow the i64
/// accumulator after only a few thousand source particles.
/// Scale 1.0 truncates to whole eV -- well below the eV-scale
/// statistical noise.
pub const KERMA_FIXED_POINT_SCALE: f64 = 1.0;

/// Upper bound for the signed fixed-point accumulator. The atomic
/// accumulator is a `u64` reinterpreted as two's-complement `i64`,
/// so the magnitude of the running sum must stay below `2^63`. We
/// size per-MT scales so the worst-case accumulated total stays
/// below `2^62`, leaving a full bit of headroom for the
/// summed-across-fission-generations realisation. Used by the
/// dispatch's per-MT scale sizing (issue #150).
pub const FIXED_POINT_ACC_CEILING: f64 = 4_611_686_018_427_387_904.0; // 2^62

/// Anchor for the per-history sum-of-squares fixed-point scale
/// ([`sum_sq_fixed_point_scale`]). `2^40`. Chosen so a DEFAULT-scaled
/// flux tally (`sum_scale = 2^30`) gets `sum_sq` scale `2^20`, the ratio
/// the hardware-validated spike proved (per-history flush reproduces the
/// per-step mean to `1.4e-10`).
pub const SUMSQ_SCALE_ANCHOR: f64 = 1_099_511_627_776.0; // 2^40

/// Dedicated sum-of-squares fixed-point scale for KERMA-shape tallies
/// (heating 301, heating-local 901, damage-energy 444), whose linear scale
/// is [`KERMA_FIXED_POINT_SCALE`] = 1.0. The generic `sum_scale^2 / 2^40`
/// form would give `2^-40` here, a quantum of `~2^40 eV^2 ~ (1 MeV)^2`, so
/// any per-history deposit below ~740 keV would round its square to zero and
/// collapse the variance. `2^-10` instead resolves the squared deposit down
/// to `sqrt(0.5 * 2^10) ~ 23 eV` (far below realistic per-history heating),
/// while a single launch (<= 100k histories, each depositing at most the
/// ~14 MeV source energy plus reaction Q) accumulates
/// `sum_h (x_h eV)^2 * 2^-10` to well under the `2^62` i64 envelope
/// (~50x margin).
pub const KERMA_SUMSQ_SCALE: f64 = 0.000_976_562_5; // 2^-10

/// Fixed-point scale for the per-history sum-of-squares atomic of a tally
/// whose linear (sum) scale is `sum_scale` (batch-free variance, issue
/// #233). Sized `min(sum_scale^2 / SUMSQ_SCALE_ANCHOR, FIXED_POINT_ACC_CEILING)`.
///
/// # Why this cannot overflow the i64 accumulator
///
/// Writing `S = sum_scale` and `S_sq = S^2 / A` (before the ceiling
/// clamp), a single history's packed square is
/// `round(x_h^2 * S_sq) = round( x_h * (x_h * S / A) * S )`. As long as a
/// single history's contribution to a bin stays below `X_max = A / S`
/// (order `1e3` cm for a flux tally, comfortably above any physical
/// per-history track length), the factor `x_h * S / A <= 1`, so each
/// squared quantum is `<= round(x_h * S)` (the linear quantum). The sum
/// accumulator is already sized (issue #150) so `Σ_h round(x_h * S)` stays
/// below `FIXED_POINT_ACC_CEILING` for one launch, so the sum-of-squares
/// accumulator lives inside the same validated envelope. This mirrors the
/// #150 per-MT argument with `X_max` playing the role `peak` plays there.
/// The i64 accumulator only has to survive ONE launch (the buffer is
/// zeroed per launch and folded into an f64 host-side accumulator across
/// chunks), so no history-count-across-the-whole-run term is needed.
///
/// Both the kernel (inline, at history-flush) and the host unpack use this
/// identical formula, so the round trip
/// `x -> round(x^2 * S_sq) -> (bits as i64 as f64) / S_sq` is consistent.
///
/// KERMA-shape tallies (`sum_scale <= KERMA_FIXED_POINT_SCALE`, i.e. 1.0) take
/// the dedicated [`KERMA_SUMSQ_SCALE`] instead: the generic form underflows
/// their eV-magnitude squares (see that constant's docs). Every other tally
/// has `sum_scale >= DEFAULT_FIXED_POINT_SCALE` (2^30), so the generic form
/// resolves its squares fine.
#[inline]
pub fn sum_sq_fixed_point_scale(sum_scale: f64) -> f64 {
    if sum_scale <= KERMA_FIXED_POINT_SCALE {
        return KERMA_SUMSQ_SCALE;
    }
    (sum_scale * sum_scale / SUMSQ_SCALE_ANCHOR).min(FIXED_POINT_ACC_CEILING)
}

/// Per-slot fixed-point scale for the `SCORE_PER_MT` reaction-rate
/// tallies, sized from each MT's expected magnitude (issues #150, #307).
///
/// The problem being solved (#150): the default scale `2^30` rounds a
/// per-collision contribution of `Σ_MT / Σ_t ~ 2e-11` (tiny high-threshold
/// channels like Fe56 MT 111 `(n,2p)`) to exactly 0, so the tally reads
/// back as an exact zero. Such a channel needs its contributions amplified
/// before they hit the integer accumulator.
///
/// The anchor is each channel's largest share of the macroscopic total,
/// `r_max = max over (material, energy) of Σ_MT(E) / Σ_t(E)`, and the scale
/// is `scale_s = DEFAULT / r_max`.
///
/// # Why this cannot overflow the i64 accumulator
///
/// The per-step contribution scored into the integer accumulator is
/// `d · Σ_MT(E) · w · scale_s`. Substituting the anchor:
///
/// `d · Σ_MT(E) · w · scale_s = d · Σ_t(E) · w · DEFAULT · [Σ_MT(E)/Σ_t(E)] / r_max`
/// `                          ≤ d · Σ_t(E) · w · DEFAULT`
///
/// because `Σ_MT(E)/Σ_t(E) ≤ r_max` by construction of `r_max`. So *every*
/// per-step contribution is bounded, step for step on the same track, by
/// what a **DEFAULT-scaled total tally** would accumulate over the
/// identical history. The DEFAULT-scaled flux/total tallies are the
/// existing, validated, overflow-safe baseline
/// ([`DEFAULT_FIXED_POINT_SCALE`] docs), so a per-MT tally at this anchor
/// lives inside that same proven envelope. The bound is per-step and
/// composes additively, so it carries to the whole accumulated sum with no
/// history-count term. The collision estimator
/// (`w · Σ_MT / Σ_t · scale_s ≤ w · DEFAULT`) and the per-history flush (a
/// sum of bounded per-step terms) inherit the same bound.
///
/// Anchoring on `Σ_t(E)` at the SAME energy is what makes the argument
/// hold; the earlier form `DEFAULT · (Σ_t_max / peak_MT)` did not
/// (issue #307). It compared a channel's peak against the maximum of `Σ_t`
/// over the whole grid, and on a strong 1/v absorber those two live at
/// opposite ends of the energy range: a B10 sphere at 1 g/cc has
/// `Σ_t_max ≈ 1.16e4 /cm` (the `(n,α)` 1/v tail at the bottom of the grid)
/// but `peak Σ_MT51 ≈ 5.0e-3 /cm` around 1 MeV, giving `scale ≈ 2.5e15`.
/// The transport never sees anything close to `Σ_t_max` -- fast neutrons in
/// B10 see `Σ_t ~ 0.1 /cm` -- so the "envelope matches a DEFAULT-scaled
/// total tally" claim was false by five orders of magnitude and the
/// accumulator wrapped `2^64` exactly once per 100k-history launch, flipping
/// B10 MT 51/52/56..61 (and Gd157 MT 51/52/53/107, and Xe135) negative.
///
/// Clamps:
/// - never below [`DEFAULT_FIXED_POINT_SCALE`]: a channel that is at some
///   energy the whole of `Σ_t` (`r_max = 1`) anchors to exactly the default,
///   matching the total tally it shares the envelope with. So large channels
///   like MT 2 stay at the default -- they do NOT get a huge scale. The clamp
///   also absorbs `r_max` slightly exceeding 1 from interpolation drift
///   between the per-MT curves and `Σ_t`.
/// - never above [`FIXED_POINT_ACC_CEILING`] (`2^62`): a defensive cap so a
///   pathologically tiny `r_max` (a channel essentially closed over the whole
///   grid) can't produce an absurd scale that would risk the `i64`
///   accumulator if that channel were somehow exercised.
/// - KERMA-shape MTs (301/901/444) keep their dedicated
///   [`KERMA_FIXED_POINT_SCALE`] and are NOT rescaled here (their eV per-step
///   magnitude is the opposite problem).
///
/// # Buffers
///
/// - `xs_score_per_mt` is the macroscopic per-MT score buffer the kernel
///   indexes, laid out `[material][slot][energy]` with `slot` running over
///   `score_mts`.
/// - `sigma_t_score_grid` is the macroscopic total on the SAME grid and in
///   the same material order, laid out `[material][energy]`. Points where it
///   is zero (below a material's data range, or a synthetic void slot) are
///   skipped: no transport happens there, so they cannot contribute to the
///   accumulator.
///
/// Returns one scale per entry of `score_mts` (same slot order). An empty
/// `score_mts` returns an empty vec (no per-MT tallies).
pub fn per_mt_fixed_point_scales(
    score_mts: &[i32],
    xs_score_per_mt: &[f64],
    sigma_t_score_grid: &[f64],
    n_material_slots: usize,
    n_grid: usize,
) -> Vec<f64> {
    if score_mts.is_empty() {
        return Vec::new();
    }
    let n_mts = score_mts.len();
    score_mts
        .iter()
        .enumerate()
        .map(|(slot, &mt)| {
            // KERMA-shape MTs keep their own (small) scale.
            if matches!(mt, 301 | 901 | 444) {
                return KERMA_FIXED_POINT_SCALE;
            }
            // Largest share of the macroscopic total this channel ever
            // reaches, across every material and energy point.
            let mut r_max = 0.0f64;
            for mat in 0..n_material_slots {
                let base = mat * n_mts * n_grid + slot * n_grid;
                let t_base = mat * n_grid;
                for ie in 0..n_grid {
                    let sigma_t = sigma_t_score_grid[t_base + ie];
                    if sigma_t <= 0.0 {
                        continue;
                    }
                    let r = xs_score_per_mt[base + ie] / sigma_t;
                    if r > r_max {
                        r_max = r;
                    }
                }
            }
            // No data (all-zero channel) -> nothing to capture; keep default.
            if r_max <= 0.0 {
                return DEFAULT_FIXED_POINT_SCALE;
            }
            // Anchor: the channel's strongest step maps to the same integer
            // magnitude a DEFAULT-scaled total tally reaches on that step.
            let scale = DEFAULT_FIXED_POINT_SCALE / r_max;
            // Never scale a per-MT tally below the default, never above the
            // defensive i64 ceiling.
            scale.clamp(DEFAULT_FIXED_POINT_SCALE, FIXED_POINT_ACC_CEILING)
        })
        .collect()
}

/// How a transport kernel (neutron OR photon) accumulates tally variance
/// (issue #233). Shared by both kernels' hosts so the per-history / per-source
/// machinery is identical across paths.
#[derive(Clone, Copy)]
pub enum TallyVarianceMode<'a> {
    /// Per-step atomic accumulation into `tally_out` (batch-means / no
    /// per-history state). Byte-identical to the pre-#233 kernel.
    PerStep,
    /// Each source particle IS a complete history; it flushes its per-bin
    /// `sum` + `sum_sq` into the doubled `tally_out` at history end (batch-free
    /// per-history variance). Photon-source and non-fissile neutron paths.
    PerHistory,
    /// Each history flushes its per-bin `sum` into a per-source accumulator
    /// keyed by `source_idx`, so descendants transported in later launches
    /// (fission progeny, secondary/decay photons) fold into their originating
    /// source particle's variance sample. `src_acc` is sized
    /// `chunk_sources * total_bins`; `sum_sq` is computed host-side at finalize.
    /// `source_idx` is `None` for a source launch (identity `[0..n)`), `Some`
    /// for a descendant launch (the drained descendants' banked indices).
    PerSource {
        chunk_sources: u32,
        total_bins: u32,
        source_idx: Option<&'a [u32]>,
    },
    /// Mesh-tally per-source variance (issue #234). Like [`PerSource`] on the
    /// host (same `src_acc` sizing, `source_idx`, and host-side `sum_sq`
    /// finalize), but the kernel accumulates each contribution DIRECTLY into
    /// `src_acc` (bypassing the touched-list/spill) so a track-length mesh tally
    /// crossing many voxels per step stays O(1) per crossing instead of the
    /// touched-list's O(distinct^2) dedup. Used for any model carrying a mesh
    /// tally. No per-history spill is allocated (`spill_cap == 0`).
    PerSourceDirect {
        chunk_sources: u32,
        total_bins: u32,
        source_idx: Option<&'a [u32]>,
    },
}

impl TallyVarianceMode<'_> {
    pub fn per_history(&self) -> bool {
        matches!(
            self,
            TallyVarianceMode::PerHistory | TallyVarianceMode::PerSource { .. }
        )
    }
    pub fn per_source(&self) -> bool {
        matches!(self, TallyVarianceMode::PerSource { .. })
    }
    /// True for the issue-#234 mesh path (direct-to-`src_acc` scoring, no
    /// touched-list). Sets the kernel `mesh_direct` comptime flag.
    pub fn mesh_direct(&self) -> bool {
        matches!(self, TallyVarianceMode::PerSourceDirect { .. })
    }
    /// True when the mode uses the `src_acc` per-source accumulator + host-side
    /// `sum_sq` finalize (`PerSource` or `PerSourceDirect`).
    pub fn uses_src_acc(&self) -> bool {
        matches!(
            self,
            TallyVarianceMode::PerSource { .. } | TallyVarianceMode::PerSourceDirect { .. }
        )
    }
    /// `src_acc` row stride (`total_out_len`) for the per-source modes, else 0.
    pub fn total_bins(&self) -> u32 {
        match self {
            TallyVarianceMode::PerSource { total_bins, .. }
            | TallyVarianceMode::PerSourceDirect { total_bins, .. } => *total_bins,
            _ => 0,
        }
    }
}

/// Round a (possibly negative) scaled fixed-point value to the
/// nearest integer with symmetric handling of the two signs, then
/// reinterpret the `i64` as the `u64` bits the atomic accumulator
/// adds. This is the plain-Rust twin of the `#[cube]` kernel's inline
/// rounding; both must stay bit-identical for the matched-stream
/// equivalence. Using the signed-symmetric form (instead of a bare
/// `(x + 0.5) as i64`) removes the round-half-up bias that truncated
/// tiny positive contributions inconsistently (issue #150).
#[inline]
pub fn round_fixed_point_bits(contrib: f64, scale: f64) -> u64 {
    let scaled = contrib * scale;
    let rounded = if scaled >= 0.0 {
        (scaled + 0.5) as i64
    } else {
        -((-scaled + 0.5) as i64)
    };
    rounded as u64
}

/// Sentinel for `cell_to_bin[…]` when a geometry cell isn't part of
/// the tally's `CellFilter`. The kernel skips accumulation for
/// that (tally, cell) pair.
pub const NOT_IN_TALLY: u32 = u32::MAX;

/// Packed multi-tally side-data for a single kernel launch.
///
/// All `Vec`s are flat; indexing patterns are documented on the
/// module docstring above. The struct owns its data so the dispatch
/// layer can build it once and hand it to the launcher by reference.
#[derive(Debug, Clone)]
pub struct TalliesPack {
    /// Score kind discriminant per tally. Length `n_tallies`.
    pub score_kinds: Vec<u32>,
    /// Cell-bin map per tally, flat `[n_tallies × n_geom_cells]`.
    /// `NOT_IN_TALLY` for cells not in the tally's `CellFilter`.
    pub cell_to_bin: Vec<u32>,
    /// Number of cell bins per tally. Length `n_tallies`.
    pub n_cells_per_tally: Vec<u32>,
    /// Offsets into `log_edges` per tally. Length `n_tallies`.
    pub edges_offsets: Vec<u32>,
    /// Number of energy bins per tally. Length `n_tallies`.
    pub n_bins_per_tally: Vec<u32>,
    /// Concatenated log-space bin edges. Tally `t` occupies
    /// `log_edges[edges_offsets[t]..edges_offsets[t] +
    /// n_bins_per_tally[t] + 1]`.
    pub log_edges: Vec<f64>,
    /// Offsets into `tally_out` per tally. Length `n_tallies + 1`
    /// (last entry is total length).
    pub out_offsets: Vec<u32>,
    /// Per-tally auxiliary integer. Currently only used by
    /// `SCORE_PER_MT` (slot index into the dispatch's
    /// `xs_score_per_mt` buffer) and by KERMA-shape scores. Ignored
    /// for `SCORE_FLUX` / `SCORE_TOTAL` / `SCORE_ABSORPTION`. Length
    /// `n_tallies`.
    pub score_data: Vec<u32>,
    /// Per-tally ENDF MT number for `SCORE_PER_MT` tallies (`0` for
    /// every other score kind). Length `n_tallies`. The GPU kernel
    /// reads this so that, inside the unresolved-resonance-region
    /// (URR) window, a capture (MT 102) or absorption (MT 27)
    /// reaction-rate tally is scored with the URR-perturbed
    /// macroscopic cross-section the transport step actually used,
    /// instead of the smooth `xs_score_per_mt` value. Without the
    /// correlation between the per-collision URR draw and the scored
    /// capture XS, GPU capture in the URR biases high (resonance
    /// self-shielding lost). The CPU twin has no URR block, so this
    /// is a no-op there and the matched-stream equivalence holds.
    pub score_mt: Vec<u32>,
    /// Per-tally fixed-point scale used by the atomic-add
    /// accumulator. Most tallies use `DEFAULT_FIXED_POINT_SCALE`
    /// (2^30); KERMA-shape tallies (heating 301, heating-local
    /// 901, damage-energy 444) use `KERMA_FIXED_POINT_SCALE`
    /// (1.0) so eV-scale per-step contributions don't overflow
    /// the u64 atomic accumulator across many source particles.
    /// Length `n_tallies`.
    pub fixed_point_scales: Vec<f64>,
    /// Per-tally estimator flag: `1` = collision estimator,
    /// `0` = track-length (default). A collision tally scores
    /// `weight × score_xs / Σ_t` once per real collision (the
    /// collision-density flux estimator) instead of the
    /// per-step `track_length × score_xs × weight`. The kernel
    /// gates the per-step block on `is_collision[t] == 0` and
    /// runs a parallel collision-site scoring loop for
    /// `is_collision[t] == 1`, so a tally never gets both
    /// contributions. Length `n_tallies`.
    pub is_collision: Vec<u32>,
    /// Number of `parent_nuclides` bins per tally (D1S). `1` for any tally
    /// without a `ParentNuclideFilter` -- the parent dimension collapses and
    /// the flat index `(cell_bin * 1 + 0) * n_bins + e == cell_bin * n_bins + e`
    /// stays byte-identical to a run without the parent dimension. Length
    /// `n_tallies`. The output block for tally `t` is sized
    /// `n_cells * n_parent * n_bins`, with stride order `cell -> parent ->
    /// energy` (matching the CPU 7D layout: parent outside energy, inside cell).
    pub n_parent_per_tally: Vec<u32>,
    /// Offsets into `parent_ids` per tally. Length `n_tallies + 1`.
    pub parent_offsets: Vec<u32>,
    /// Concatenated resolved parent-nuclide ids (`NuclideId.get()` widened to
    /// u32), in filter order, per tally. Tally `t` occupies `parent_ids[
    /// parent_offsets[t]..parent_offsets[t+1]]`. Empty range for a tally without
    /// a parent filter (n_parent == 1). The kernel maps a photon's banked
    /// parent id to a bin by linear scan over this range (matching the CPU
    /// `ParentNuclideFilter::get_bin` `position` scan); a photon whose parent id
    /// is not in the range is NOT scored into that tally.
    pub parent_ids: Vec<u32>,
    /// Number of spatial mesh (voxel) bins per tally (issue #234). `1` for any
    /// tally without a `MeshFilter` -- the mesh dimension collapses and the flat
    /// index `((cell_bin * n_parent + parent) * n_bins + e) * 1 + 0` stays
    /// byte-identical to a run without mesh support. For a mesh tally it is the
    /// mesh's storage size (`MeshFilter::num_bins()`, i.e. `nx*ny*nz` for a
    /// rectangular mesh; every slot is reachable, none are padding).
    /// Mesh is the INNERMOST flat dimension
    /// (stride 1), matching the CPU 7D layout `... -> energy -> mesh`. Length
    /// `n_tallies`. The output block for tally `t` is sized
    /// `n_cells * n_parent * n_bins * n_mesh`.
    pub n_mesh_per_tally: Vec<u32>,
    /// Mesh-kind discriminant per tally: `MESH_NONE` / `MESH_RECT_ROWMAJOR` /
    /// `MESH_CYLINDRICAL`. Length `n_tallies`. Selects how
    /// the kernel derives voxel bins from the step segment. `MESH_NONE` tallies
    /// never touch the mesh machinery.
    pub mesh_kind: Vec<u32>,
    /// Offsets into `mesh_params` per tally. Length `n_tallies + 1` (prefix
    /// sum; last entry is the total length). A `MESH_NONE` tally has an empty
    /// range.
    pub mesh_params_offsets: Vec<u32>,
    /// Concatenated per-tally mesh geometry descriptors (issue #234), all as
    /// `f64` so a single buffer carries both rectangular and cylindrical meshes.
    /// Tally `t` occupies `mesh_params[mesh_params_offsets[t]..mesh_params_offsets[t+1]]`.
    /// Rectangular: `[lower_left[3], upper_right[3], inv_width[3], width[3],
    /// shape[3]]` (`MESH_RECT_PARAMS` words, shape cast to `u32` in the kernel).
    /// Cylindrical: `[origin[3], nr, nphi, nz, full_phi]` (`MESH_CYL_HEADER`)
    /// then `r_grid[nr+1]`, `r_grid_sq[nr+1]`, `phi_grid[nphi+1]`, `z_grid[nz+1]`.
    pub mesh_params: Vec<f64>,
    /// Offsets into `efunc_params` per tally. Length `n_tallies + 1` (prefix
    /// sum; last entry is the total length). A tally with no
    /// `EnergyFunctionFilter` has an EMPTY range -- that emptiness is the
    /// presence flag, so no separate per-tally count buffer is needed
    /// (issue #271).
    pub efunc_offsets: Vec<u32>,
    /// Concatenated per-tally energy-function tables (`EnergyFunctionFilter`,
    /// i.e. `energy_function=` / `dose_coefficients=`). Tally `t` occupies
    /// `efunc_params[efunc_offsets[t]..efunc_offsets[t+1]]`, laid out
    /// `[n_points, energy[n], coeffs[4*(n-1)]]` -- the point count is a
    /// leading header word, matching the cylindrical mesh descriptor's
    /// header-then-grids convention above.
    ///
    /// Unlike every other per-tally payload here this is NOT a bin dimension:
    /// the filter contributes `num_bins() == 1`, so it never widens the output
    /// block. It is a multiplicative weight on the score plus a hard gate --
    /// a particle whose energy falls outside `[energy[0], energy[n-1]]` scores
    /// NOTHING into that tally, matching the CPU's `return` on
    /// `get_weight() == None`.
    ///
    /// The coefficients are the natural cubic spline the CPU solved at filter
    /// construction; the kernel only evaluates the polynomial, so both
    /// backends interpolate identically.
    pub efunc_params: Vec<f64>,
}

impl TalliesPack {
    pub fn n_tallies(&self) -> u32 {
        self.score_kinds.len() as u32
    }

    /// Total length of the kernel's output buffer (sum of
    /// `n_cells × n_bins` across tallies).
    pub fn total_out_len(&self) -> u32 {
        self.out_offsets.last().copied().unwrap_or(0)
    }

    /// A degenerate single-Flux-on-one-cell-bin pack with no energy
    /// bins. Used when the model has no tallies -- the kernel still
    /// needs at least one slot so the buffer length doesn't go to
    /// zero (cubecl rejects zero-length buffers).
    pub fn dummy_single_bin(n_geom_cells: u32) -> Self {
        // One tally with a single cell bin (every geom cell maps to
        // bin 0) and a single energy bin spanning [-∞, +∞].
        let cell_to_bin = vec![0u32; n_geom_cells.max(1) as usize];
        Self {
            score_kinds: vec![SCORE_FLUX],
            cell_to_bin,
            n_cells_per_tally: vec![1],
            edges_offsets: vec![0],
            n_bins_per_tally: vec![1],
            log_edges: vec![f64::NEG_INFINITY, f64::INFINITY],
            out_offsets: vec![0, 1],
            score_data: vec![0],
            score_mt: vec![0],
            fixed_point_scales: vec![DEFAULT_FIXED_POINT_SCALE],
            is_collision: vec![0],
            n_parent_per_tally: vec![1],
            parent_offsets: vec![0, 0],
            parent_ids: Vec::new(),
            n_mesh_per_tally: vec![1],
            mesh_kind: vec![MESH_NONE],
            mesh_params_offsets: vec![0, 0],
            mesh_params: Vec::new(),
            efunc_offsets: vec![0, 0],
            efunc_params: Vec::new(),
        }
    }

    /// Two-tally fixture used by kernel/CPU-mirror tests and benches:
    /// tally 0 is per-cell `Flux`, tally 1 is per-cell `Absorption`,
    /// both sharing the same energy bin edges. Cell-bin map is
    /// identity (one cell → one bin), so `tally_outputs[t][c*B + e]`
    /// is the flux/absorption tally for cell `c`, energy bin `e`.
    pub fn flux_abs_pack(n_geom_cells: u32, log_edges: &[f64]) -> Self {
        assert!(
            log_edges.len() >= 2,
            "log_edges must have at least one bin (>= 2 entries)"
        );
        let n_cells = n_geom_cells as usize;
        let n_bins = (log_edges.len() - 1) as u32;
        let mut cell_to_bin = Vec::with_capacity(2 * n_cells);
        for _ in 0..2 {
            for c in 0..n_geom_cells {
                cell_to_bin.push(c);
            }
        }
        let mut log_edges_concat = Vec::with_capacity(2 * log_edges.len());
        log_edges_concat.extend_from_slice(log_edges);
        log_edges_concat.extend_from_slice(log_edges);
        let cells_x_bins = n_geom_cells * n_bins;
        Self {
            score_kinds: vec![SCORE_FLUX, SCORE_ABSORPTION],
            cell_to_bin,
            n_cells_per_tally: vec![n_geom_cells, n_geom_cells],
            edges_offsets: vec![0, log_edges.len() as u32],
            n_bins_per_tally: vec![n_bins, n_bins],
            log_edges: log_edges_concat,
            out_offsets: vec![0, cells_x_bins, 2 * cells_x_bins],
            score_data: vec![0, 0],
            // Tally 1 is SCORE_ABSORPTION (kind), not SCORE_PER_MT, so
            // its MT slot stays 0 -- the URR substitution only fires
            // for SCORE_PER_MT MT 102 / 27.
            score_mt: vec![0, 0],
            fixed_point_scales: vec![DEFAULT_FIXED_POINT_SCALE, DEFAULT_FIXED_POINT_SCALE],
            is_collision: vec![0, 0],
            n_parent_per_tally: vec![1, 1],
            parent_offsets: vec![0, 0, 0],
            parent_ids: Vec::new(),
            n_mesh_per_tally: vec![1, 1],
            mesh_kind: vec![MESH_NONE, MESH_NONE],
            mesh_params_offsets: vec![0, 0, 0],
            mesh_params: Vec::new(),
            efunc_offsets: vec![0, 0, 0],
            efunc_params: Vec::new(),
        }
    }
}

/// Row-major flat voxel index for a rectangular mesh, `(iz*ny+iy)*nx+ix`
/// (matches `RegularRectangularMesh::get_bin_from_indices` under RowMajor).
#[inline]
pub fn rect_rowmajor_bin(ix: u32, iy: u32, iz: u32, nx: u32, ny: u32) -> u32 {
    (iz * ny + iy) * nx + ix
}

/// Row-major voxel bin for a point in a rectangular mesh, or `None` if the
/// point is outside the mesh. Operates on the packed descriptor
/// `[lower_left[3], upper_right[3], inv_width[3], width[3], shape[3]]`. Faithful
/// twin of `RegularRectangularMesh::get_bin` (floor, clamp, half-open bounds),
/// used by the collision estimator's mesh binning (issue #234).
#[inline]
pub fn rect_mesh_bin_at(params: &[f64], pos: [f64; 3]) -> Option<u32> {
    let lower_left = [params[0], params[1], params[2]];
    let upper_right = [params[3], params[4], params[5]];
    let inv_width = [params[6], params[7], params[8]];
    let shape = [
        params[12] as usize,
        params[13] as usize,
        params[14] as usize,
    ];
    let mut idx = [0u32; 3];
    for i in 0..3 {
        if pos[i] < lower_left[i] || pos[i] >= upper_right[i] {
            return None;
        }
        let f = (pos[i] - lower_left[i]) * inv_width[i];
        let mut k = f.floor() as usize;
        if k >= shape[i] {
            k = shape[i] - 1;
        }
        idx[i] = k as u32;
    }
    Some(rect_rowmajor_bin(
        idx[0],
        idx[1],
        idx[2],
        shape[0] as u32,
        shape[1] as u32,
    ))
}

/// Amanatides-Woo voxel walk of the segment `r0 -> r1` over a rectangular
/// row-major mesh, operating on the packed descriptor
/// `[lower_left[3], upper_right[3], inv_width[3], width[3], shape[3]]`
/// ([`MESH_RECT_PARAMS`] words). Invokes `emit(voxel_bin, length_fraction)`
/// once per voxel crossed, where `length_fraction` is the fraction of the
/// segment's total length spent in that voxel.
///
/// This is the plain-Rust twin of the `#[cube]` neutron/photon kernels' inline
/// mesh DDA and a faithful port of `RegularRectangularMesh::bins_crossed_iter`
/// (issue #234): same `TINY_BIT`/`FP_PRECISION` thresholds, same ray/AABB slab
/// entry test for tracks starting outside the mesh, same min-distance axis
/// selection, and the same AW `distance += t_delta` incremental update. Keeping
/// the two implementations byte-for-byte equivalent is what lets the
/// matched-stream CPU/GPU equivalence hold for mesh tallies.
// Indexed loops over the 3 axes mirror the `#[cube]` kernel's inline DDA (and
// `RegularRectangularMesh::bins_crossed_iter`, which carries the same allow);
// enumerate()/iterator forms obscure the axis bookkeeping and would diverge
// from the kernel twin.
#[allow(clippy::needless_range_loop)]
pub fn rect_mesh_crossings(
    params: &[f64],
    r0: [f64; 3],
    r1: [f64; 3],
    direction: [f64; 3],
    mut emit: impl FnMut(u32, f64),
) {
    const TINY_BIT: f64 = 1e-8;
    const FP_PRECISION: f64 = 1e-14;

    let lower_left = [params[0], params[1], params[2]];
    let upper_right = [params[3], params[4], params[5]];
    let inv_width = [params[6], params[7], params[8]];
    let width = [params[9], params[10], params[11]];
    let shape = [
        params[12] as usize,
        params[13] as usize,
        params[14] as usize,
    ];
    let nx = shape[0] as u32;
    let ny = shape[1] as u32;

    let total_length =
        ((r1[0] - r0[0]).powi(2) + (r1[1] - r0[1]).powi(2) + (r1[2] - r0[2]).powi(2)).sqrt();
    if total_length < 2.0 * TINY_BIT {
        return;
    }

    // Reciprocal direction; 0.0 for (near-)axis-parallel components (never
    // consumed, since t_delta is INFINITY there).
    let inv_dir = [
        if direction[0].abs() < FP_PRECISION {
            0.0
        } else {
            1.0 / direction[0]
        },
        if direction[1].abs() < FP_PRECISION {
            0.0
        } else {
            1.0 / direction[1]
        },
        if direction[2].abs() < FP_PRECISION {
            0.0
        } else {
            1.0 / direction[2]
        },
    ];
    let mut step = [0i32; 3];
    let mut t_delta = [f64::INFINITY; 3];
    for i in 0..3 {
        if direction[i].abs() < FP_PRECISION {
            continue;
        }
        step[i] = if direction[i] > 0.0 { 1 } else { -1 };
        t_delta[i] = width[i] * inv_dir[i].abs();
    }

    // Locate the entry voxel: direct floor if r0 is inside the mesh, else a
    // ray/AABB slab test to find where the segment enters (nudged in by
    // TINY_BIT), mirroring bins_crossed_iter.
    let mut indices = [0usize; 3];
    let mut in_mesh = true;
    for i in 0..3 {
        if r0[i] < lower_left[i] || r0[i] >= upper_right[i] {
            in_mesh = false;
            break;
        }
        let idx_float = (r0[i] - lower_left[i]) * inv_width[i];
        indices[i] = idx_float.floor() as usize;
        if indices[i] >= shape[i] {
            indices[i] = shape[i] - 1;
        }
    }

    let mut traveled = 0.0;
    let mut current_pos = r0;

    if !in_mesh {
        let mut t_enter = f64::NEG_INFINITY;
        let mut t_exit = f64::INFINITY;
        for i in 0..3 {
            if direction[i].abs() < 1e-14 {
                if r0[i] < lower_left[i] || r0[i] >= upper_right[i] {
                    return;
                }
            } else {
                let inv_d = 1.0 / direction[i];
                let t1 = (lower_left[i] - r0[i]) * inv_d;
                let t2 = (upper_right[i] - r0[i]) * inv_d;
                let (t_near, t_far) = if inv_d > 0.0 { (t1, t2) } else { (t2, t1) };
                t_enter = t_enter.max(t_near);
                t_exit = t_exit.min(t_far);
            }
        }
        if t_enter >= t_exit || t_exit <= 0.0 || t_enter >= total_length {
            return;
        }
        let t_start = t_enter.max(0.0) + TINY_BIT;
        traveled = t_start;
        for i in 0..3 {
            current_pos[i] = r0[i] + direction[i] * t_start;
        }
        in_mesh = true;
        for i in 0..3 {
            if current_pos[i] < lower_left[i] || current_pos[i] >= upper_right[i] {
                in_mesh = false;
                break;
            }
            let idx_float = (current_pos[i] - lower_left[i]) * inv_width[i];
            indices[i] = idx_float.floor() as usize;
            if indices[i] >= shape[i] {
                indices[i] = shape[i] - 1;
            }
        }
        if !in_mesh {
            return;
        }
    }

    // Per-axis distance to the next voxel boundary (Amanatides-Woo seed).
    let mut dist = [f64::INFINITY; 3];
    let mut next_index = indices;
    for dim in 0..3 {
        if direction[dim].abs() < FP_PRECISION {
            continue;
        }
        if direction[dim] > 0.0 && indices[dim] < shape[dim] {
            next_index[dim] = indices[dim] + 1;
            let boundary = lower_left[dim] + (indices[dim] + 1) as f64 * width[dim];
            dist[dim] = traveled + (boundary - current_pos[dim]) * inv_dir[dim];
        } else if direction[dim] <= 0.0 {
            let boundary = lower_left[dim] + indices[dim] as f64 * width[dim];
            dist[dim] = traveled + (boundary - current_pos[dim]) * inv_dir[dim];
            next_index[dim] = if indices[dim] > 0 {
                indices[dim] - 1
            } else {
                shape[dim]
            };
        }
    }

    loop {
        // Axis whose boundary is crossed first.
        let mut min_dim = 0usize;
        let mut min_dist = dist[0];
        for i in 1..3 {
            if dist[i] < min_dist {
                min_dist = dist[i];
                min_dim = i;
            }
        }

        let length_in_voxel = if min_dist >= total_length {
            total_length - traveled
        } else {
            min_dist - traveled
        };
        let emit_this = length_in_voxel > TINY_BIT;
        if emit_this {
            let bin = rect_rowmajor_bin(
                indices[0] as u32,
                indices[1] as u32,
                indices[2] as u32,
                nx,
                ny,
            );
            emit(bin, length_in_voxel / total_length);
        }

        if min_dist >= total_length {
            return;
        }

        traveled = min_dist;
        indices[min_dim] = next_index[min_dim];
        if indices[min_dim] >= shape[min_dim] {
            return;
        }

        dist[min_dim] += t_delta[min_dim];
        next_index[min_dim] = if step[min_dim] > 0 {
            indices[min_dim] + 1
        } else if indices[min_dim] > 0 {
            indices[min_dim] - 1
        } else {
            shape[min_dim]
        };
    }
}

// =====================================================================
// Cylindrical mesh DDA twin (issue #279)
// =====================================================================

/// Distances below this along a track are treated as zero (matches
/// `yamc_tallies::mesh`'s `CYL_TINY`).
const CYL_TINY: f64 = 1e-8;
/// Direction/denominator components below this are treated as zero.
const CYL_FP: f64 = 1e-14;
/// A candidate boundary crossing only counts if it lies more than this far
/// ahead of the current cursor.
const CYL_COINCIDENT: f64 = 1e-10;
/// Two pi (azimuthal wrap point).
const CYL_TWO_PI: f64 = std::f64::consts::TAU;

/// Bracket `v` into a cell of a sorted, strictly-increasing grid: index `i`
/// with `grid[i] <= v < grid[i+1]`, or `None` when `v` is outside
/// `[grid[0], grid[last])`. Twin of `yamc_tallies::mesh::bracket`.
#[inline]
fn cyl_bracket(grid: &[f64], v: f64) -> Option<usize> {
    if v < grid[0] || v >= grid[grid.len() - 1] {
        return None;
    }
    Some(grid.partition_point(|&g| g <= v) - 1)
}

/// Parsed view over a packed cylindrical-mesh descriptor. The descriptor is the
/// header `[origin[3], nr, nphi, nz, full_phi]` ([`MESH_CYL_HEADER`] words) then
/// the four grids `r_grid[nr+1]`, `r_grid_sq[nr+1]`, `phi_grid[nphi+1]`,
/// `z_grid[nz+1]` in that order (built by `yamc::gpu::dispatch::build_mesh_descriptor`).
struct CylMeshView<'a> {
    origin: [f64; 3],
    r_grid: &'a [f64],
    r_grid_sq: &'a [f64],
    phi_grid: &'a [f64],
    z_grid: &'a [f64],
    nr: usize,
    nphi: usize,
    nz: usize,
    full_phi: bool,
}

impl<'a> CylMeshView<'a> {
    /// Slice the packed descriptor into grid views.
    fn parse(params: &'a [f64]) -> Self {
        let origin = [params[0], params[1], params[2]];
        let nr = params[3] as usize;
        let nphi = params[4] as usize;
        let nz = params[5] as usize;
        let full_phi = params[6] != 0.0;
        let mut off = MESH_CYL_HEADER;
        let r_grid = &params[off..off + nr + 1];
        off += nr + 1;
        let r_grid_sq = &params[off..off + nr + 1];
        off += nr + 1;
        let phi_grid = &params[off..off + nphi + 1];
        off += nphi + 1;
        let z_grid = &params[off..off + nz + 1];
        Self {
            origin,
            r_grid,
            r_grid_sq,
            phi_grid,
            z_grid,
            nr,
            nphi,
            nz,
            full_phi,
        }
    }

    /// Flat bin index, `ir` innermost: `(iz*nphi + iphi)*nr + ir`.
    #[inline]
    fn bin_from_indices(&self, ir: usize, iphi: usize, iz: usize) -> u32 {
        ((iz * self.nphi + iphi) * self.nr + ir) as u32
    }

    /// Per-axis bracketed cell of a local-frame point, or `None` if outside.
    /// Twin of `CylindricalMesh::indices_local`.
    fn indices_local(&self, x: f64, y: f64, zc: f64) -> Option<[usize; 3]> {
        let rho = x.hypot(y);
        let ir = cyl_bracket(self.r_grid, rho)?;
        let iz = cyl_bracket(self.z_grid, zc)?;
        let iphi = if rho < CYL_FP {
            0
        } else {
            let mut phi = y.atan2(x);
            if phi < 0.0 {
                phi += CYL_TWO_PI;
            }
            if self.full_phi && phi >= self.phi_grid[self.nphi] {
                phi = self.phi_grid[self.nphi] - CYL_FP;
            }
            cyl_bracket(self.phi_grid, phi)?
        };
        Some([ir, iphi, iz])
    }

    /// Locate the flat bin of an absolute position, or `None` when outside.
    /// Twin of `CylindricalMesh::get_bin`.
    fn bin_at(&self, position: [f64; 3]) -> Option<u32> {
        let x = position[0] - self.origin[0];
        let y = position[1] - self.origin[1];
        let zc = position[2] - self.origin[2];
        self.indices_local(x, y, zc)
            .map(|[ir, iphi, iz]| self.bin_from_indices(ir, iphi, iz))
    }

    /// First root `> l` of `a·t² + 2b·t + (c - R²) = 0`. `∞` when none.
    /// Twin of `CylindricalMesh::shell_crossing`.
    fn shell_crossing(&self, a: f64, b: f64, c: f64, r_sq: f64, r: f64, l: f64) -> f64 {
        if r <= 0.0 || a < CYL_FP {
            return f64::INFINITY;
        }
        let pn = b / a;
        let disc = pn * pn - (c - r_sq) / a;
        if disc < 0.0 {
            return f64::INFINITY;
        }
        let sq = disc.sqrt();
        let t1 = -pn - sq;
        let t2 = -pn + sq;
        let thresh = l + CYL_COINCIDENT;
        if t1 > thresh {
            t1
        } else if t2 > thresh {
            t2
        } else {
            f64::INFINITY
        }
    }

    /// Smallest radial-shell crossing `> l` over both walls of cell `ir`.
    fn radial_next(&self, a: f64, b: f64, c: f64, ir: usize, l: f64) -> f64 {
        let inner = self.shell_crossing(a, b, c, self.r_grid_sq[ir], self.r_grid[ir], l);
        let outer = self.shell_crossing(a, b, c, self.r_grid_sq[ir + 1], self.r_grid[ir + 1], l);
        inner.min(outer)
    }

    /// Distance `> l` at which the track crosses the half-plane at angle `phi`.
    /// `∞` if it does not, or if on the antipodal (`phi + π`) half-plane.
    /// Twin of `CylindricalMesh::phi_crossing`.
    fn phi_crossing(&self, p: [f64; 3], d: [f64; 3], phi: f64, l: f64) -> f64 {
        let (s, co) = phi.sin_cos();
        let denom = d[0] * s - d[1] * co;
        if denom.abs() < CYL_FP {
            return f64::INFINITY;
        }
        let t = -(p[0] * s - p[1] * co) / denom;
        if t <= l + CYL_COINCIDENT {
            return f64::INFINITY;
        }
        let x = p[0] + t * d[0];
        let y = p[1] + t * d[1];
        if co * x + s * y > 0.0 {
            t
        } else {
            f64::INFINITY
        }
    }

    /// Smallest azimuthal-wall crossing `> l` over both walls of cell `iphi`.
    fn phi_next(&self, p: [f64; 3], d: [f64; 3], iphi: usize, l: f64) -> f64 {
        let lo = self.phi_crossing(p, d, self.phi_grid[iphi], l);
        let hi = self.phi_crossing(p, d, self.phi_grid[iphi + 1], l);
        lo.min(hi)
    }

    /// Distance `> l` to the next axial plane bounding cell `iz`.
    /// Twin of `CylindricalMesh::z_next`.
    fn z_next(&self, pz: f64, dz: f64, iz: usize, l: f64) -> f64 {
        if dz.abs() < CYL_FP {
            return f64::INFINITY;
        }
        let plane = if dz > 0.0 {
            self.z_grid[iz + 1]
        } else {
            self.z_grid[iz]
        };
        let t = (plane - pz) / dz;
        if t > l + CYL_COINCIDENT {
            t
        } else {
            f64::INFINITY
        }
    }

    /// Smallest distance in `(l, total]` at which the track first lies inside
    /// the mesh. Handles initial entry and re-entry through the central hole /
    /// a φ wedge. Twin of `CylindricalMesh::first_entry`.
    #[allow(clippy::too_many_arguments)]
    fn first_entry(
        &self,
        p: [f64; 3],
        d: [f64; 3],
        a: f64,
        b: f64,
        c: f64,
        l: f64,
        total: f64,
    ) -> Option<f64> {
        let mut best = f64::INFINITY;
        let mut consider = |t: f64, this: &Self| {
            if t > l + CYL_COINCIDENT && t < best && t <= total {
                let probe = [
                    p[0] + (t + CYL_TINY) * d[0],
                    p[1] + (t + CYL_TINY) * d[1],
                    p[2] + (t + CYL_TINY) * d[2],
                ];
                if this.indices_local(probe[0], probe[1], probe[2]).is_some() {
                    best = t;
                }
            }
        };

        for &(r_sq, r) in &[
            (self.r_grid_sq[0], self.r_grid[0]),
            (self.r_grid_sq[self.nr], self.r_grid[self.nr]),
        ] {
            if r > 0.0 && a >= CYL_FP {
                let pn = b / a;
                let disc = pn * pn - (c - r_sq) / a;
                if disc >= 0.0 {
                    let sq = disc.sqrt();
                    consider(-pn - sq, self);
                    consider(-pn + sq, self);
                }
            }
        }

        if !self.full_phi {
            for &phi in &[self.phi_grid[0], self.phi_grid[self.nphi]] {
                let t = self.phi_crossing(p, d, phi, l);
                if t.is_finite() {
                    consider(t, self);
                }
            }
        }

        if d[2].abs() >= CYL_FP {
            for &zp in &[self.z_grid[0], self.z_grid[self.nz]] {
                consider((zp - p[2]) / d[2], self);
            }
        }

        if best.is_finite() {
            Some(best)
        } else {
            None
        }
    }
}

/// Energy-function weight for the table packed at `off` in
/// [`TalliesPack::efunc_params`], or `None` when `energy` falls outside the
/// tabulated range (issue #271).
///
/// Plain-Rust twin of the `#[cube]` `energy_function_weight_kernel`, and a
/// faithful port of `EnergyFunctionFilter::get_weight`. `None` means the CPU
/// would have dropped the whole scoring event, not that the weight is zero --
/// callers must skip the score, not multiply by 0.
///
/// Faithfulness matters bit for bit here, so note the two things it does NOT
/// do: it does not interpolate (the spline coefficients arrive precomputed), and
/// it works in LINEAR energy, not the log energy the surrounding cross-section
/// lookups use. The bracket search reproduces the CPU's `binary_search_by`
/// tie-breaking exactly: the grid is strictly increasing, so an exact knot hit
/// gives `dx == 0` on both sides, and `energy == e[n-1]` lands on the final
/// interval evaluated at its right end on both.
pub fn energy_function_weight(params: &[f64], off: usize, energy: f64) -> Option<f64> {
    let n = params[off] as usize;
    let e0 = off + 1;
    let c0 = e0 + n;
    if energy < params[e0] || energy > params[e0 + n - 1] {
        return None;
    }
    // First index with `e[idx] > energy`; the bracket is the one below it.
    // Phrased as a lower bound rather than an upper one so the midpoint is a
    // plain `(lo + hi) / 2`, matching the other binary searches in the kernel.
    // The range test above guarantees `e[0] <= energy`, so `lo >= 1` here and
    // the decrement cannot underflow.
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if params[e0 + mid] <= energy {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let idx = (lo - 1).min(n - 2);
    let dx = energy - params[e0 + idx];
    let a = params[c0 + 4 * idx];
    let b = params[c0 + 4 * idx + 1];
    let c = params[c0 + 4 * idx + 2];
    let d = params[c0 + 4 * idx + 3];
    Some(a + dx * (b + dx * (c + dx * d)))
}

/// Voxel bin for a point in a cylindrical mesh, or `None` if outside. Plain-Rust
/// twin of the `#[cube]` `cyl_mesh_bin_at_kernel` and a faithful port of
/// `CylindricalMesh::get_bin`; drives the collision estimator's mesh binning
/// (issue #279). Operates on the packed descriptor (see [`CylMeshView`]).
#[inline]
pub fn cyl_mesh_bin_at(params: &[f64], pos: [f64; 3]) -> Option<u32> {
    CylMeshView::parse(params).bin_at(pos)
}

/// Analytic `(r, φ, z)` voxel walk of the segment `r0 -> r1` over a cylindrical
/// mesh, invoking `emit(voxel_bin, length_fraction)` once per voxel crossed.
/// Plain-Rust twin of the `#[cube]` `cyl_mesh_score_src_acc` and a faithful port
/// of `CylindricalMesh::bins_crossed` (issue #279): the same radial quadratic
/// `ρ²(t)=a·t²+2b·t+c`, the through-axis perigee forced event `t* = -b/a`, the
/// central `r < r_min` hole re-entry via `first_entry`, and the `full_phi` seam
/// clamp. A track through the central hole produces two crossing groups with an
/// untallied gap. Keeping this byte-faithful to `CylindricalMesh` is what the
/// `cyl_mesh_dda_twin` gate test checks before the GPU kernel is trusted.
pub fn cyl_mesh_crossings(
    params: &[f64],
    r0: [f64; 3],
    r1: [f64; 3],
    direction: [f64; 3],
    mut emit: impl FnMut(u32, f64),
) {
    let v = CylMeshView::parse(params);
    let total =
        ((r1[0] - r0[0]).powi(2) + (r1[1] - r0[1]).powi(2) + (r1[2] - r0[2]).powi(2)).sqrt();

    if total < 2.0 * CYL_TINY {
        let mid = [
            0.5 * (r0[0] + r1[0]),
            0.5 * (r0[1] + r1[1]),
            0.5 * (r0[2] + r1[2]),
        ];
        if let Some(bin) = v.bin_at(mid) {
            emit(bin, 1.0);
        }
        return;
    }

    let p = [
        r0[0] - v.origin[0],
        r0[1] - v.origin[1],
        r0[2] - v.origin[2],
    ];
    let d = direction;
    let a = d[0] * d[0] + d[1] * d[1];
    let b = p[0] * d[0] + p[1] * d[1];
    let c = p[0] * p[0] + p[1] * p[1];

    let point = |l: f64| [p[0] + l * d[0], p[1] + l * d[1], p[2] + l * d[2]];

    let mut l = 0.0_f64;
    let max_iter = 4 * (v.nr + v.nphi + v.nz) + 32;
    let mut iter = 0;

    while l < total {
        iter += 1;
        if iter > max_iter {
            break;
        }
        let probe = point((l + CYL_TINY).min(total));
        match v.indices_local(probe[0], probe[1], probe[2]) {
            Some([ir, iphi, iz]) => {
                let dr = v.radial_next(a, b, c, ir, l);
                let dphi = v.phi_next(p, d, iphi, l);
                let dz = v.z_next(p[2], d[2], iz, l);
                let d_peri = if a >= CYL_FP {
                    let t_star = -b / a;
                    if t_star > l + CYL_COINCIDENT {
                        t_star
                    } else {
                        f64::INFINITY
                    }
                } else {
                    f64::INFINITY
                };
                let dmin = dr.min(dphi).min(dz).min(d_peri).min(total);
                let seg = dmin - l;
                if seg > CYL_TINY {
                    emit(v.bin_from_indices(ir, iphi, iz), seg / total);
                }
                if dmin >= total {
                    break;
                }
                l = dmin;
            }
            None => match v.first_entry(p, d, a, b, c, l, total) {
                Some(te) if te < total => l = te,
                _ => break,
            },
        }
    }
}

#[cfg(test)]
mod per_mt_scale_tests {
    //! Regression tests for the `SCORE_PER_MT` fixed-point scale policy
    //! (issues #150 and #307).
    //!
    //! These are pure host-side arithmetic on the scale sizing, so they run
    //! under a plain `cargo test --workspace` on every platform -- no GPU
    //! feature, no adapter, no nuclear data. The on-hardware counterpart is
    //! `crates/yamc/tests/gpu_per_mt_reaction_rate_overflow.rs`, which runs
    //! the real B10 sphere through the kernel.

    use super::*;

    /// One material, `N_GRID` energy points, laid out as
    /// `xs_score_per_mt[slot * N_GRID + ie]` / `sigma_t[ie]`.
    const N_GRID: usize = 512;

    /// A B10-shaped pair of curves, in macroscopic units (1/cm) for a
    /// 1 g/cc B10 sphere as measured on the real ENDF/B-8.1 data (issue #307):
    ///
    /// - `sigma_t` is dominated by the `(n,α)` 1/v tail at the bottom of the
    ///   grid, reaching `1.163e4 /cm` at the first grid point, and falls to
    ///   `~0.1 /cm` across the fast range where a 14 MeV source actually
    ///   deposits its track length.
    /// - the MT 51 discrete-level channel is zero below its ~0.79 MeV
    ///   threshold and peaks at `4.98e-3 /cm` in the fast range.
    ///
    /// Returns `(xs_score_per_mt, sigma_t, peak_index)`.
    fn b10_shaped_curves() -> (Vec<f64>, Vec<f64>, usize) {
        // Log grid from 1e-5 eV to 2e7 eV.
        let e_lo = 1e-5_f64;
        let e_hi = 2e7_f64;
        let step = (e_hi / e_lo).ln() / (N_GRID - 1) as f64;
        let energy: Vec<f64> = (0..N_GRID)
            .map(|i| e_lo * (i as f64 * step).exp())
            .collect();
        // Σ_t: 1/v absorption (anchored so the first point is 1.163e4 /cm)
        // plus a flat 0.1 /cm elastic floor.
        let sigma_t: Vec<f64> = energy
            .iter()
            .map(|&e| 1.163e4 * (e_lo / e).sqrt() + 0.1)
            .collect();
        // MT 51: zero below 0.79 MeV, a triangular bump peaking at 4.98e-3
        // /cm at 2 MeV and closing again by 20 MeV.
        let threshold = 7.9e5_f64;
        let peak_e = 2.0e6_f64;
        let mut xs = vec![0.0_f64; N_GRID];
        let mut peak_index = 0usize;
        let mut peak_value = 0.0_f64;
        for (i, &e) in energy.iter().enumerate() {
            if e <= threshold {
                continue;
            }
            let v = if e <= peak_e {
                4.98e-3 * (e - threshold) / (peak_e - threshold)
            } else {
                4.98e-3 * (e_hi - e) / (e_hi - peak_e)
            };
            xs[i] = v.max(0.0);
            if xs[i] > peak_value {
                peak_value = xs[i];
                peak_index = i;
            }
        }
        (xs, sigma_t, peak_index)
    }

    /// Every per-step contribution a `SCORE_PER_MT` tally can make must be
    /// bounded by what a DEFAULT-scaled TOTAL tally would make on the same
    /// step: `Σ_MT(E) · scale_s ≤ Σ_t(E) · DEFAULT` at every energy point.
    /// This is the invariant the whole no-overflow argument rests on.
    ///
    /// Pre-#307 the scale was `DEFAULT · Σ_t_max / peak_MT`, which on a
    /// strong 1/v absorber compares the channel's fast-range peak against the
    /// thermal end of Σ_t and breaks this bound by ~5 orders of magnitude.
    #[test]
    fn per_mt_scale_bounds_every_step_by_a_default_scaled_total_tally() {
        let (xs, sigma_t, _) = b10_shaped_curves();
        let scales = per_mt_fixed_point_scales(&[51], &xs, &sigma_t, 1, N_GRID);
        let scale = scales[0];
        for ie in 0..N_GRID {
            assert!(
                xs[ie] * scale <= sigma_t[ie] * DEFAULT_FIXED_POINT_SCALE * (1.0 + 1e-12),
                "per-MT step contribution exceeds the DEFAULT-scaled total tally at \
                 grid point {ie}: sigma_mt={:e} * scale={scale:e} = {:e} > sigma_t={:e} * \
                 DEFAULT = {:e}",
                xs[ie],
                xs[ie] * scale,
                sigma_t[ie],
                sigma_t[ie] * DEFAULT_FIXED_POINT_SCALE
            );
        }
    }

    /// The B10 launch that issue #307 reported, reproduced as arithmetic.
    ///
    /// The GPU accumulates one launch (`FIXED_LAUNCH_CHUNK` = 100k source
    /// histories) into a `u64` atomic reinterpreted as two's-complement
    /// `i64`. The measured CPU answer for the B10 sphere is `4.176e-2` MT 51
    /// reactions per source history, so the launch accumulates
    /// `rate · 1e5 · scale` integer quanta. With the pre-#307 scale of
    /// `2.5085e15` that is `1.048e19`, past `2^63 = 9.223e18`: it wrapped
    /// `2^64` exactly once and the tally read back `-3.176e-2`.
    #[test]
    fn per_mt_scale_keeps_a_b10_launch_inside_the_i64_accumulator() {
        const LAUNCH_HISTORIES: f64 = 100_000.0;
        // Measured on the real B10 sphere (35 cm, 1 g/cc, 14.06 MeV point
        // source) by the production CPU transport.
        const MT51_RATE_PER_HISTORY: f64 = 4.176e-2;

        let (xs, sigma_t, _) = b10_shaped_curves();
        let scales = per_mt_fixed_point_scales(&[51], &xs, &sigma_t, 1, N_GRID);
        let accumulated = MT51_RATE_PER_HISTORY * LAUNCH_HISTORIES * scales[0];
        assert!(
            accumulated < FIXED_POINT_ACC_CEILING,
            "one 100k-history launch accumulates {accumulated:e} integer quanta at \
             scale {:e}, past the 2^62 envelope (i64 wraps at 2^63 and the tally \
             reads back negative)",
            scales[0]
        );
        // The sum-of-squares accumulator rides on the same scale and must
        // survive the same launch.
        let sq_scale = sum_sq_fixed_point_scale(scales[0]);
        let accumulated_sq =
            MT51_RATE_PER_HISTORY * MT51_RATE_PER_HISTORY * LAUNCH_HISTORIES * sq_scale;
        assert!(
            accumulated_sq < FIXED_POINT_ACC_CEILING,
            "the per-history sum-of-squares accumulator overflows: {accumulated_sq:e}"
        );
    }

    /// Issue #150's property, which the #307 fix must preserve: a tiny
    /// high-threshold channel (Fe56 MT 111 `(n,2p)`) is amplified enough that
    /// its contributions survive integer rounding instead of truncating the
    /// whole tally to an exact zero.
    #[test]
    fn per_mt_scale_still_amplifies_a_tiny_high_threshold_channel() {
        // Fe56-shaped: Σ_t ~ 0.4 /cm across the fast range, MT 111 peaking at
        // 1e-8 /cm near 20 MeV. One quantum of a DEFAULT-scaled tally is
        // 1/2^30 = 9.3e-10, so a ~2e-11 per-step contribution rounds to 0.
        let sigma_t = vec![0.4_f64; N_GRID];
        let mut xs = vec![0.0_f64; N_GRID];
        for (i, v) in xs.iter_mut().enumerate().skip(N_GRID - 8) {
            *v = 1e-8 * (i - (N_GRID - 9)) as f64 / 8.0;
        }
        let scales = per_mt_fixed_point_scales(&[111], &xs, &sigma_t, 1, N_GRID);
        let peak = xs.iter().cloned().fold(0.0_f64, f64::max);
        // A single step of 1 cm at the channel's peak must land far above one
        // integer quantum.
        let quanta = 1.0 * peak * scales[0];
        assert!(
            quanta > 1e6,
            "tiny channel under-amplified: a peak 1 cm step packs to {quanta:e} quanta \
             at scale {:e}",
            scales[0]
        );
    }

    /// A channel that IS the total at some energy anchors to exactly the
    /// default (no amplification), and KERMA-shape MTs keep their own scale.
    #[test]
    fn per_mt_scale_leaves_dominant_and_kerma_channels_alone() {
        let sigma_t = vec![1.0_f64; N_GRID];
        // Slot 0 = MT 2 (all of Σ_t somewhere), slot 1 = MT 301 (heating).
        let mut xs = vec![0.0_f64; 2 * N_GRID];
        xs[..N_GRID].copy_from_slice(&sigma_t);
        xs[N_GRID..].fill(5.0e6); // eV-magnitude KERMA curve
        let scales = per_mt_fixed_point_scales(&[2, 301], &xs, &sigma_t, 1, N_GRID);
        assert_eq!(scales[0], DEFAULT_FIXED_POINT_SCALE);
        assert_eq!(scales[1], KERMA_FIXED_POINT_SCALE);
    }

    /// Grid points where a material has no cross-section data (Σ_t == 0) and
    /// synthetic void material slots are skipped rather than producing a
    /// divide-by-zero or an infinite scale.
    #[test]
    fn per_mt_scale_skips_zero_total_points_and_void_slots() {
        let mut sigma_t = vec![0.0_f64; 2 * N_GRID];
        let mut xs = vec![0.0_f64; 2 * N_GRID];
        // Material 0: real data over the upper half of the grid only.
        for ie in N_GRID / 2..N_GRID {
            sigma_t[ie] = 0.5;
            xs[ie] = 0.05;
        }
        // Material 1: the synthetic void slot -- all zero in both buffers.
        let scales = per_mt_fixed_point_scales(&[51], &xs, &sigma_t, 2, N_GRID);
        assert!(scales[0].is_finite());
        assert_eq!(scales[0], DEFAULT_FIXED_POINT_SCALE * 10.0);
    }
}
