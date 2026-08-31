//! Host-only extractor for the per-material secondary photon-production
//! product-selection table (GPU coupled neutron->photon, slice S1).
//!
//! Mirrors the per-material, density-weighted, interpolate-onto-the-shared
//! grid pattern of [`super::extract`]. The job of this slice is to lay out the
//! data the GPU walk needs to reproduce the CPU's per-(reaction, product)
//! photon selection in
//! `yamc_physics::photon::photon_production::sample_photon_product`.
//!
//! # Why per-PRODUCT, not per-reaction
//!
//! The CPU walk does NOT select a reaction and then fold that reaction's photon
//! products together. It walks every `(reaction, photon-product-index)` pair and
//! selects ONE specific product with weight
//!
//! ```text
//! weight(rxn, product, E) = scaling_factor(mt) * rxn_xs(E) * product_yield(E)
//! ```
//!
//! and each selected product carries its OWN outgoing-energy distribution
//! (a discrete gamma line, a continuous tabular spectrum, ...). Folding the
//! products of a reaction into a single weight (the original S1 cut) would make
//! it impossible for the GPU to know which product's distribution to sample in
//! S3. A single Fe56 reaction can carry many photon products (e.g. MT 102 has a
//! whole cascade of discrete lines), so this is the common case, not an edge.
//!
//! S1 therefore packs PER-PRODUCT metadata: which reaction-xs row each product
//! reads, its scaling constant, its yield curve on the grid, and a tag for its
//! outgoing-energy law (with the discrete-line params inline; S3 fills the
//! continuous-distribution buffers). A later slice walks these per-product
//! weights cumulatively to reproduce the CPU selection exactly.
//!
//! No GPU / kernel code lives here; this is pure CPU data prep, always built.

use super::distributions::{EoutSlot, PhotonAngleSlot};
use super::NuclideXsError;
use yamc_nuclide::nuclide::{is_fission_mt, Nuclide};
use yamc_nuclide::particle_type::ParticleType;
use yamc_nuclide::reaction_product::{AngleEnergyDistribution, EnergyDistribution};

/// Outgoing-energy-distribution tag for a packed photon product, recorded by S1
/// so S3 knows which sampler to wire up. Discrete lines (`DiscretePhoton`) carry
/// their parameters inline in the per-product arrays; the continuous variants
/// only get a slot-index placeholder here (S3 fills the real distribution
/// buffers). Anything S1 can't classify (empty / `energy: None` distribution) is
/// tagged [`PHOTON_EOUT_KIND_NONE`]; such a product can never be sampled, but it
/// still occupies a slot so the per-product weight walk and the aggregate stay
/// in lock-step with the CPU's two passes.
pub const PHOTON_EOUT_KIND_NONE: u32 = 0;
/// Discrete photon line: `prod_line_energy`, `prod_primary_flag`, `prod_awr`
/// hold the parameters; `E_out = energy` (flag != 2) or
/// `energy + awr/(awr+1) * E_in` (flag == 2), matching
/// `EnergyDistribution::DiscretePhoton::sample`.
pub const PHOTON_EOUT_KIND_DISCRETE: u32 = 1;
/// Continuous tabular photon spectrum (ENDF File 5/6 Law 1). S1 records only
/// `prod_dist_slot` (a placeholder index); S3 extracts the tabular CDF buffers.
pub const PHOTON_EOUT_KIND_CONTINUOUS_TABULAR: u32 = 2;

/// `prod_dist_slot` value for a product whose outgoing-energy distribution does
/// not need a continuous-distribution buffer (discrete or none). Real
/// continuous-tabular products get a dense `0..n_continuous` index assigned by
/// S1 so S3 can size its buffers; this sentinel marks "no continuous slot".
pub const PHOTON_DIST_SLOT_NONE: u32 = u32::MAX;

/// Per-material secondary photon-production product-selection table, laid out
/// for a future GPU walk.
///
/// All energy-indexed buffers are sampled on the caller-supplied shared
/// `log_energy_grid` (length `n_grid`). The GPU computes product `i`'s selection
/// weight at grid index `g` as
///
/// ```text
/// prod_scaling[i] * rxn_xs[prod_rxn_idx[i] * n_grid + g] * prod_yield_grid[i * n_grid + g]
/// ```
///
/// and walks those weights cumulatively against
/// `Sum_i weight(i, g) == photon_prod[g]` to pick a product, reproducing the
/// CPU's per-(reaction, product) `sample_photon_product` selection.
///
/// `photon_prod` reproduces the CPU's per-material `FastXSGrid::photon_prod`
/// (density-weighted across nuclides), used for the photon-COUNT
/// `y_t = photon_prod / total`.
#[derive(Debug, Clone)]
pub struct GpuPhotonProductionXs {
    /// Number of points on the shared energy grid (columns per row).
    pub n_grid: usize,

    /// Aggregate photon-production xs per energy, density-weighted across the
    /// material's nuclides. Length `n_grid`. Equals the CPU
    /// `FastXSGrid::photon_prod` summed over nuclides, and equals
    /// `Sum_products weight(product, g)` at every grid point `g`. Drives the
    /// photon-count yield `y_t = photon_prod / total`.
    pub photon_prod: Vec<f64>,

    /// Number of packed photon-producing reaction rows. Each row is one
    /// (nuclide, photon-producing MT) instance: keying per-nuclide rather than
    /// deduplicating by MT keeps the per-product weight formula exact for
    /// multi-nuclide materials (a product reads exactly its own nuclide's
    /// density-weighted macroscopic xs). For a single-nuclide material this is
    /// just one row per photon MT. A material with no photon production carries
    /// `n_photon_rxn == 1` (a single padded zero row).
    pub n_photon_rxn: usize,
    /// Per-(nuclide, MT) macroscopic photon-production cross section on the
    /// grid, flat and **row-major**: entry `rxn_idx * n_grid + g` is
    /// `density * micro_rxn_xs(E_g)` for that reaction row (the
    /// `extract_score_xs_per_mt` pattern, but kept per-nuclide). Length
    /// `n_photon_rxn * n_grid`. Read by every product whose `prod_rxn_idx`
    /// points at the row.
    pub rxn_xs: Vec<f64>,
    /// MT number per packed reaction row. Length `n_photon_rxn`; a
    /// no-photon-production material carries a single `0` entry.
    pub rxn_mt: Vec<i32>,

    /// Number of packed photon products. Equals the total number of
    /// (reaction, photon-product) pairs the CPU walk would visit across the
    /// material (e.g. ~532 for a single-nuclide Fe56 material). A material with
    /// no photon production carries `n_product == 1` (a single padded zero
    /// slot) so downstream buffers are never empty.
    pub n_product: usize,
    /// Reaction-row index each product reads its xs from. Length `n_product`;
    /// values index `rxn_xs` rows in `0..n_photon_rxn`.
    pub prod_rxn_idx: Vec<u32>,
    /// Per-product scaling constant (the CPU `scaling_factor(mt)`): the
    /// delayed-photon scaling for fission MTs, else `1.0`. Energy-independent on
    /// the nuclide's own grid (the fission factor varies with energy, but on a
    /// non-fission photon row -- the common case -- it is exactly `1.0`; fission
    /// photon rows fold the per-grid-point factor into `prod_yield_grid`
    /// instead, leaving this `1.0`, so the GPU formula stays a single multiply).
    ///
    /// See [`Self::prod_yield_grid`] for where the fission factor actually goes.
    pub prod_scaling: Vec<f64>,
    /// Per-product yield curve on the shared grid, flat and **product-major**:
    /// entry `i * n_grid + g` is `scaling_factor(mt, E_g) * product_yield(E_g)`
    /// for product `i`. Length `n_product * n_grid`.
    ///
    /// Yields are stored on the GRID, not as scalars: the audit found Fe56's 532
    /// photon products are all `Yield::Tabulated1D`, and 6 of them are genuinely
    /// energy-dependent (the continuous-tabular continuum photons on the
    /// inelastic MTs). A scalar-per-product would silently flatten those. The
    /// energy-dependent delayed-photon scaling for fission MTs is folded in here
    /// too (so `prod_scaling` stays a constant `1.0` and the GPU multiply is
    /// `prod_scaling * rxn_xs * prod_yield_grid` with no per-grid-point branch).
    pub prod_yield_grid: Vec<f64>,
    /// Outgoing-energy-distribution tag per product. Length `n_product`. One of
    /// [`PHOTON_EOUT_KIND_NONE`] / [`PHOTON_EOUT_KIND_DISCRETE`] /
    /// [`PHOTON_EOUT_KIND_CONTINUOUS_TABULAR`].
    pub prod_eout_kind: Vec<u32>,
    /// Discrete-line outgoing energy (eV) per product. Meaningful only when
    /// `prod_eout_kind[i] == PHOTON_EOUT_KIND_DISCRETE`; `0.0` otherwise.
    /// Length `n_product`.
    pub prod_line_energy: Vec<f64>,
    /// Discrete-line `primary_flag` per product (2 => energy depends on E_in;
    /// otherwise the line is at the fixed `prod_line_energy`). Meaningful only
    /// for discrete products; `0` otherwise. Length `n_product`.
    pub prod_primary_flag: Vec<i32>,
    /// Discrete-line target atomic-weight ratio per product (used by the
    /// `primary_flag == 2` energy formula). Meaningful only for discrete
    /// products; `0.0` otherwise. Length `n_product`.
    pub prod_awr: Vec<f64>,
    /// Continuous-distribution slot index per product: a dense
    /// `0..n_continuous` index for products tagged
    /// [`PHOTON_EOUT_KIND_CONTINUOUS_TABULAR`] (so S3 can size and address its
    /// distribution buffers), else [`PHOTON_DIST_SLOT_NONE`]. Length
    /// `n_product`.
    pub prod_dist_slot: Vec<u32>,

    // --------------------------- slice S3 ---------------------------
    // The kinematics (`E_out`, `mu`) sampler reads the buffers below. The
    // angle buffers are per-PRODUCT (every product carries a tabulated
    // angle); the continuous-tabular eout buffers are per-CONTINUOUS-SLOT
    // (only the `n_continuous` products tagged
    // `PHOTON_EOUT_KIND_CONTINUOUS_TABULAR`, addressed by `prod_dist_slot`).
    /// Number of continuous-tabular outgoing-energy slots. Equals the
    /// count of products with `prod_eout_kind == CONTINUOUS_TABULAR`
    /// (6 for single-nuclide Fe56). The eout buffers below are sized
    /// `n_continuous` slots wide. A material with no continuous-tabular
    /// photon products carries `n_continuous == 1` (a single zero slot)
    /// so the buffers are never empty.
    pub n_continuous: usize,
    /// Per-continuous-slot incident-energy count, length `n_continuous`.
    /// Indexed by `prod_dist_slot[product]`. Feeds
    /// `sample_continuous_tabular_eout`'s `n_eout`.
    pub ct_n_eout: Vec<u32>,
    /// Per-continuous-slot outer `histogram_interp` flag (1 = histogram,
    /// suppress the stochastic E_in bracket pick + bracket-bound stretch),
    /// length `n_continuous`. Feeds `sample_continuous_tabular_eout`'s
    /// `hist_outer`.
    pub ct_hist: Vec<u32>,
    /// Tight CSR base (issue #104): the global ae-row in the tight
    /// `ct_energy_grid` / `ct_n_x` / `ct_interp` / `ct_n_discrete` arrays
    /// where each continuous slot's incident-energy rows begin. Length
    /// `n_continuous` (one per continuous slot). The sampler reads
    /// `eg_off_e = ct_ae_offset[prod_dist_slot[product]]`.
    pub ct_ae_offset: Vec<u32>,
    /// Tight CSR base (issue #104): the index into the tight `ct_x` /
    /// `ct_cdf` / `ct_p` arrays where each ae-row's outgoing points begin.
    /// Length = total ae-rows across all continuous slots (== `ct_n_x.len()`).
    /// The sampler reads row `eg_off_e + bin` at `ct_x_offset[eg_off_e + bin]`.
    pub ct_x_offset: Vec<u32>,
    /// Continuous-tabular incident-energy grid, tight CSR: slot `slot`'s
    /// incident energies are the `ct_n_eout[slot]` entries starting at
    /// `ct_ae_offset[slot]`. Length = sum of `ct_n_eout` over all slots
    /// (== `ct_n_x.len()`).
    pub ct_energy_grid: Vec<f64>,
    /// Per-ae-row outgoing-point count, tight CSR (one entry per ae-row,
    /// addressed by `ct_ae_offset[slot] + a`). Length = total ae-rows.
    pub ct_n_x: Vec<u32>,
    /// Outgoing-energy values, tight CSR: ae-row `r`'s `ct_n_x[r]` points
    /// start at `ct_x_offset[r]`. Length = sum of `ct_n_x`.
    pub ct_x: Vec<f64>,
    /// Outgoing-energy CDF (renormalised to end at 1.0), same tight layout
    /// as `ct_x`. Length = sum of `ct_n_x`.
    pub ct_cdf: Vec<f64>,
    /// Outgoing-energy per-point PDF (scaled to match the renormalised
    /// CDF), same tight layout as `ct_x`. Length = sum of `ct_n_x`.
    pub ct_p: Vec<f64>,
    /// Per-ae-row inner interpolation flag (`ANGLE_INTERP_HISTOGRAM` /
    /// `ANGLE_INTERP_LINLIN`), tight CSR (addressed by
    /// `ct_ae_offset[slot] + a`). Length = total ae-rows.
    pub ct_interp: Vec<u32>,
    /// Per-ae-row discrete-line prefix count, tight CSR (addressed by
    /// `ct_ae_offset[slot] + a`). Length = total ae-rows.
    pub ct_n_discrete: Vec<u32>,

    /// Per-product incident-energy count for the angular table, length
    /// `n_product`. `0` => isotropic (kernel draws `2*xi - 1`); our data
    /// never hits that (every photon product carries a 2-point angle).
    pub pa_n_energies: Vec<u32>,
    /// Tight CSR base (issue #104): the global ae-row in the tight
    /// `pa_energy_grid` / `pa_n_mu` / `pa_interp` arrays where each product's
    /// incident-energy rows begin. Length `n_product` (one per product). The
    /// sampler reads `ae_off = pa_ae_offset[product]`.
    pub pa_ae_offset: Vec<u32>,
    /// Tight CSR base (issue #104): the index into the tight `pa_mu` / `pa_cdf`
    /// / `pa_pdf` arrays where each ae-row's mu points begin. Length = total
    /// ae-rows across all products (== `pa_n_mu.len()`). The sampler reads
    /// row `ae_off + bin` at `pa_mu_offset[ae_off + bin]`.
    pub pa_mu_offset: Vec<u32>,
    /// Per-product angular incident-energy grid, tight CSR (issue #104):
    /// product `p`'s incident energies are the `pa_n_energies[p]` entries
    /// starting at `pa_ae_offset[p]`. Length = sum of `pa_n_energies`
    /// (== total ae-rows). The sampler finds the E_in bin in this row.
    pub pa_energy_grid: Vec<f64>,
    /// Per-ae-row mu-point count, tight CSR (one entry per ae-row, addressed
    /// by `pa_ae_offset[p] + a`). Length = total ae-rows. Feeds
    /// `invert_angle_cdf`'s `n_mu`.
    pub pa_n_mu: Vec<u32>,
    /// Scattering-cosine values, tight CSR: ae-row `r`'s `pa_n_mu[r]` points
    /// start at `pa_mu_offset[r]`. Length = sum of `pa_n_mu`. The sampler
    /// reads at `mu_off = pa_mu_offset[ae_off + bin]`.
    pub pa_mu: Vec<f64>,
    /// Scattering-cosine CDF (renormalised to end at 1.0), same tight layout
    /// as `pa_mu`. Length = sum of `pa_n_mu`.
    pub pa_cdf: Vec<f64>,
    /// Scattering-cosine per-point PDF (scaled to match the renormalised
    /// CDF), same tight layout as `pa_mu`. Length = sum of `pa_n_mu`.
    pub pa_pdf: Vec<f64>,
    /// Per-ae-row angular interpolation flag, tight CSR (addressed by
    /// `pa_ae_offset[p] + a`). Length = total ae-rows.
    pub pa_interp: Vec<u32>,
}

impl GpuPhotonProductionXs {
    /// Build a synthetic non-emitting "void material" table for a
    /// material-less (void) cell on the coupled neutron->photon path.
    ///
    /// Identical in shape to the no-photon-production fallback an empty
    /// real material would yield (a single zero row + zero product /
    /// continuous / angle slot, so the S3 buffers are never empty). A
    /// void cell maps to this slot via `cell_to_material`; because the
    /// neutron kernel's photon-emission block is gated behind
    /// `collide_first` (never true in a `sigma_t = 0` void cell), the
    /// slot is never sampled -- it exists only to keep the per-material
    /// offset tables (`*_per_material`) in-bounds for the void index.
    pub fn void(n_grid: usize) -> Self {
        let empty_angle = PhotonAngleSlot::empty();
        let empty_ct = EoutSlot::empty();
        Self {
            n_grid,
            photon_prod: vec![0.0; n_grid],
            n_photon_rxn: 1,
            rxn_xs: vec![0.0; n_grid],
            rxn_mt: vec![0],
            n_product: 1,
            prod_rxn_idx: vec![0],
            prod_scaling: vec![0.0],
            prod_yield_grid: vec![0.0; n_grid],
            prod_eout_kind: vec![PHOTON_EOUT_KIND_NONE],
            prod_line_energy: vec![0.0],
            prod_primary_flag: vec![0],
            prod_awr: vec![0.0],
            prod_dist_slot: vec![PHOTON_DIST_SLOT_NONE],
            n_continuous: 1,
            ct_n_eout: vec![empty_ct.n_energies()],
            ct_hist: vec![empty_ct.histogram_interp()],
            // Tight CSR (issue #104): one ae-row base (0) for the single empty
            // continuous slot; the empty slot has no rows so `ct_x_offset` is
            // empty.
            ct_ae_offset: vec![0],
            ct_x_offset: Vec::new(),
            ct_energy_grid: empty_ct.energy_grid().to_vec(),
            ct_n_x: empty_ct.n_x().to_vec(),
            ct_x: empty_ct.x().to_vec(),
            ct_cdf: empty_ct.cdf().to_vec(),
            ct_p: empty_ct.p().to_vec(),
            ct_interp: empty_ct.interp().to_vec(),
            ct_n_discrete: empty_ct.n_discrete().to_vec(),
            pa_n_energies: vec![empty_angle.n_energies],
            // Tight CSR (issue #104): one ae-row base (0) for the single empty
            // angle product; the empty slot has no rows so `pa_mu_offset` and
            // the per-row / per-point arrays are empty.
            pa_ae_offset: vec![0],
            pa_mu_offset: Vec::new(),
            pa_energy_grid: empty_angle.energy_grid,
            pa_n_mu: empty_angle.n_mu,
            pa_mu: empty_angle.mu,
            pa_cdf: empty_angle.cdf,
            pa_pdf: empty_angle.pdf,
            pa_interp: empty_angle.interp,
        }
    }
}

/// Delayed-photon scaling for a photon-producing MT at a grid index, matching
/// `sample_photon_product`'s `scaling_factor` closure: fission MTs are scaled by
/// `delayed_photon_scaling` (interpolated with `interp_factor`), everything else
/// by `1.0`. Here the grid *is* the nuclide's own grid, so `interp_factor` is
/// `0.0` (no sub-bin interpolation) -- the value at `i_grid` is exact.
#[inline]
fn delayed_scaling_at(delayed_photon_scaling: &[f64], mt: i32, i_grid: usize) -> f64 {
    if !is_fission_mt(mt) || delayed_photon_scaling.is_empty() {
        return 1.0;
    }
    delayed_photon_scaling[i_grid.min(delayed_photon_scaling.len() - 1)]
}

/// Classify a photon product's single outgoing-energy distribution into a
/// GPU eout-kind tag plus the discrete-line parameters (when discrete).
///
/// Returns `(kind, line_energy, primary_flag, awr)`. Mirrors the CPU's
/// `has_valid_distribution` check in `sample_secondary_photons`: a product whose
/// (only) distribution is empty or an `UncorrelatedAngleEnergy { energy: None }`
/// is unsamplable and tagged [`PHOTON_EOUT_KIND_NONE`]. Photon products in our
/// data carry exactly one distribution (verified by audit), so S1 reads the
/// first; if a future product carried several, S1 still classifies the first and
/// S3 would refine -- but the selection weight (this slice's job) is unaffected.
fn classify_photon_eout(distributions: &[AngleEnergyDistribution]) -> (u32, f64, i32, f64) {
    let Some(dist) = distributions.first() else {
        return (PHOTON_EOUT_KIND_NONE, 0.0, 0, 0.0);
    };
    match dist {
        AngleEnergyDistribution::UncorrelatedAngleEnergy { energy, .. } => match energy {
            Some(EnergyDistribution::DiscretePhoton {
                primary_flag,
                energy,
                atomic_weight_ratio,
            }) => (
                PHOTON_EOUT_KIND_DISCRETE,
                *energy,
                *primary_flag,
                *atomic_weight_ratio,
            ),
            Some(EnergyDistribution::ContinuousTabular { .. }) => {
                (PHOTON_EOUT_KIND_CONTINUOUS_TABULAR, 0.0, 0, 0.0)
            }
            // Other energy laws on a photon product are not expected in our
            // data; tag NONE so the slot is inert until S3 (if ever) handles it.
            _ => (PHOTON_EOUT_KIND_NONE, 0.0, 0, 0.0),
        },
        // Correlated / Kalbach-Mann / NBody / Evaporation angle-energy laws are
        // not used by photon products in our libraries; mark inert.
        _ => (PHOTON_EOUT_KIND_NONE, 0.0, 0, 0.0),
    }
}

/// Extract a photon product's angular distribution into a
/// [`PhotonAngleSlot`]. Mirrors the CPU `sample_uncorrelated` /
/// `AngleDistribution::sample` path: the first
/// `UncorrelatedAngleEnergy`'s `angle` is packed; anything else (no
/// distribution, or a non-uncorrelated variant) yields the empty slot,
/// which the kernel samples isotropically. Our data always carries a
/// 2-point tabulated angle, so the empty path is only a safety net.
fn extract_photon_angle(distributions: &[AngleEnergyDistribution]) -> PhotonAngleSlot {
    let Some(dist) = distributions.first() else {
        return PhotonAngleSlot::empty();
    };
    match dist {
        AngleEnergyDistribution::UncorrelatedAngleEnergy { angle, .. } => {
            PhotonAngleSlot::from_angle(angle)
        }
        _ => PhotonAngleSlot::empty(),
    }
}

/// Extract a photon product's `ContinuousTabular` outgoing-energy
/// distribution into an [`EoutSlot`], reusing the neutron eout machinery
/// so the buffers feed `sample_continuous_tabular_eout` byte-compatibly.
/// Returns `None` for any non-continuous-tabular product (discrete / none
/// occupy no continuous slot).
fn extract_photon_continuous_tabular(
    distributions: &[AngleEnergyDistribution],
) -> Option<EoutSlot> {
    let dist = distributions.first()?;
    match dist {
        AngleEnergyDistribution::UncorrelatedAngleEnergy {
            energy:
                Some(EnergyDistribution::ContinuousTabular {
                    energy,
                    energy_out,
                    histogram_interp,
                }),
            ..
        } => Some(EoutSlot::from_photon_continuous_tabular(
            energy,
            energy_out,
            *histogram_interp,
        )),
        _ => None,
    }
}

/// Extract the per-material photon-production product-selection table.
///
/// `nuclides` is `&[(&Nuclide, atom_density)]`, aggregated density-weighted
/// (same convention as [`super::extract::extract_material_xs`]). For every
/// (nuclide, photon-producing MT) the macroscopic production xs is interpolated
/// onto the shared `log_energy_grid` and stored as a [`GpuPhotonProductionXs::rxn_xs`]
/// row; for every photon product of that reaction a per-product entry is packed
/// carrying which row it reads, its scaling/yield curve, and its outgoing-energy
/// tag, exactly reproducing the CPU `sample_photon_product` per-(reaction,
/// product) selection.
///
/// Reaction rows and products are enumerated nuclide-by-nuclide, ascending MT
/// within a nuclide, product order within a reaction (matching the CPU walk's
/// `photon_product_idx` ordering), so the packed layout is deterministic.
///
/// A material with no photon production returns `n_photon_rxn == 1` /
/// `n_product == 1` with a single padded zero row + zero product slot and an
/// all-zero `photon_prod`, mirroring `extract_score_xs_per_mt`'s no-data padding
/// so downstream buffers are never empty.
pub fn extract_photon_production_xs(
    nuclides: &[(&Nuclide, f64)],
    temperature: &str,
    log_energy_grid: &[f64],
) -> Result<GpuPhotonProductionXs, NuclideXsError> {
    if nuclides.is_empty() {
        return Err(NuclideXsError::EmptyMaterial);
    }
    let n_grid = log_energy_grid.len();
    // The caller passes the shared *log* energy grid; the nuclide reactions
    // index by linear energy, so undo the log once up front.
    let energy_grid: Vec<f64> = log_energy_grid.iter().map(|&le| le.exp()).collect();

    // Packed per-reaction-row and per-product buffers, filled as we walk
    // nuclides -> photon MTs (ascending) -> photon products (in order).
    let mut rxn_xs: Vec<f64> = Vec::new();
    let mut rxn_mt: Vec<i32> = Vec::new();
    let mut prod_rxn_idx: Vec<u32> = Vec::new();
    let mut prod_scaling: Vec<f64> = Vec::new();
    let mut prod_yield_grid: Vec<f64> = Vec::new();
    let mut prod_eout_kind: Vec<u32> = Vec::new();
    let mut prod_line_energy: Vec<f64> = Vec::new();
    let mut prod_primary_flag: Vec<i32> = Vec::new();
    let mut prod_awr: Vec<f64> = Vec::new();
    let mut prod_dist_slot: Vec<u32> = Vec::new();
    let mut n_continuous: u32 = 0;

    // Slice S3 per-product angle buffers (product-major) and per-continuous-
    // slot eout buffers (slot-major), appended in lock-step with the product
    // walk so addressing matches `prod_dist_slot` / product index.
    let mut pa_n_energies: Vec<u32> = Vec::new();
    // Tight CSR bases (issue #104): `pa_ae_offset` (one per product) is the
    // global ae-row where that product's incident-energy rows begin in the
    // tight `pa_energy_grid` / `pa_n_mu` / `pa_interp` arrays; `pa_mu_offset`
    // (one per ae-row) is the index into the tight `pa_mu` / `pa_cdf` /
    // `pa_pdf` arrays where that row's mu points begin.
    let mut pa_ae_offset: Vec<u32> = Vec::new();
    let mut pa_mu_offset: Vec<u32> = Vec::new();
    let mut pa_energy_grid: Vec<f64> = Vec::new();
    let mut pa_n_mu: Vec<u32> = Vec::new();
    let mut pa_mu: Vec<f64> = Vec::new();
    let mut pa_cdf: Vec<f64> = Vec::new();
    let mut pa_pdf: Vec<f64> = Vec::new();
    let mut pa_interp: Vec<u32> = Vec::new();

    let mut ct_n_eout: Vec<u32> = Vec::new();
    let mut ct_hist: Vec<u32> = Vec::new();
    // Tight CSR bases (issue #104): `ct_ae_offset` (one per continuous slot) is
    // the global ae-row where that slot's incident-energy rows begin in the
    // tight `ct_energy_grid` / `ct_n_x` / `ct_interp` / `ct_n_discrete` arrays;
    // `ct_x_offset` (one per ae-row) is the index into the tight `ct_x` /
    // `ct_cdf` / `ct_p` arrays where that row's outgoing points begin.
    let mut ct_ae_offset: Vec<u32> = Vec::new();
    let mut ct_x_offset: Vec<u32> = Vec::new();
    let mut ct_energy_grid: Vec<f64> = Vec::new();
    let mut ct_n_x: Vec<u32> = Vec::new();
    let mut ct_x: Vec<f64> = Vec::new();
    let mut ct_cdf: Vec<f64> = Vec::new();
    let mut ct_p: Vec<f64> = Vec::new();
    let mut ct_interp: Vec<u32> = Vec::new();
    let mut ct_n_discrete: Vec<u32> = Vec::new();

    let mut photon_prod = vec![0.0_f64; n_grid];

    for (nuclide, density) in nuclides {
        let temp_idx = nuclide
            .get_temp_idx(temperature)
            .ok_or_else(|| NuclideXsError::TemperatureNotLoaded(temperature.to_string()))?;
        let reactions = &nuclide.reactions[temp_idx];
        // Delayed-photon scaling is per-nuclide (folded from fission energy
        // release). Pull it from the matching fast_xs grid when present.
        let delayed_photon_scaling: &[f64] = nuclide
            .fast_xs
            .get(temp_idx)
            .map(|g| g.delayed_photon_scaling.as_slice())
            .unwrap_or(&[]);

        // Photon-producing MTs for THIS nuclide, ascending (deterministic).
        let mut photon_mts: Vec<i32> = reactions
            .iter()
            .filter_map(|(&mt, rxn)| {
                rxn.products
                    .iter()
                    .any(|p| p.is_particle_type(&ParticleType::Photon))
                    .then_some(mt)
            })
            .collect();
        photon_mts.sort_unstable();

        for &mt in &photon_mts {
            let Some(rxn) = reactions.get(&mt) else {
                continue;
            };

            // Macroscopic per-(nuclide, MT) production xs row on the grid.
            let rxn_idx = rxn_mt.len() as u32;
            rxn_mt.push(mt);
            let row_start = rxn_xs.len();
            rxn_xs.resize(row_start + n_grid, 0.0);
            for (i, &e) in energy_grid.iter().enumerate() {
                rxn_xs[row_start + i] = density * rxn.cross_section_at(e).unwrap_or(0.0);
            }

            // One packed product per photon product of this reaction, in
            // product order (matches the CPU `photon_product_idx` ordering).
            for product in &rxn.products {
                if !product.is_particle_type(&ParticleType::Photon) {
                    continue;
                }

                // Per-grid yield curve, with the fission delayed-photon scaling
                // (energy-dependent) folded in so prod_scaling stays a constant.
                let prod_idx = prod_eout_kind.len();
                let yslot = prod_idx * n_grid;
                prod_yield_grid.resize(yslot + n_grid, 0.0);
                for (i, &e) in energy_grid.iter().enumerate() {
                    let scaling = delayed_scaling_at(delayed_photon_scaling, mt, i);
                    let y = product
                        .product_yield
                        .as_ref()
                        .map(|yld| yld.evaluate(e))
                        .unwrap_or(1.0);
                    prod_yield_grid[yslot + i] = scaling * y;
                }

                prod_rxn_idx.push(rxn_idx);
                // Scaling is folded into the yield grid above; keep this 1.0 so
                // the GPU weight is a single rxn_xs * yield_grid multiply.
                prod_scaling.push(1.0);

                let (kind, line_e, pflag, awr) = classify_photon_eout(&product.distribution);
                prod_eout_kind.push(kind);
                prod_line_energy.push(line_e);
                prod_primary_flag.push(pflag);
                prod_awr.push(awr);
                if kind == PHOTON_EOUT_KIND_CONTINUOUS_TABULAR {
                    prod_dist_slot.push(n_continuous);
                    n_continuous += 1;
                    // Pack this product's continuous-tabular eout slot (slot-
                    // major, in dense `prod_dist_slot` order). `None` here
                    // would be a classify/extract disagreement -- pad an empty
                    // CT slot so addressing stays in lock-step rather than
                    // silently shifting later slots.
                    let slot = extract_photon_continuous_tabular(&product.distribution)
                        .unwrap_or_else(EoutSlot::empty);
                    ct_n_eout.push(slot.n_energies());
                    ct_hist.push(slot.histogram_interp());
                    // Tight CSR bases (issue #104): record this slot's ae-row
                    // base BEFORE extending the per-row arrays, then a per-row
                    // x-point base accumulated from the current `ct_x` length.
                    ct_ae_offset.push(ct_n_x.len() as u32);
                    let mut x_base = ct_x.len() as u32;
                    for &n_x in slot.n_x() {
                        ct_x_offset.push(x_base);
                        x_base += n_x;
                    }
                    ct_energy_grid.extend_from_slice(slot.energy_grid());
                    ct_n_x.extend_from_slice(slot.n_x());
                    ct_x.extend_from_slice(slot.x());
                    ct_cdf.extend_from_slice(slot.cdf());
                    ct_p.extend_from_slice(slot.p());
                    ct_interp.extend_from_slice(slot.interp());
                    ct_n_discrete.extend_from_slice(slot.n_discrete());
                } else {
                    prod_dist_slot.push(PHOTON_DIST_SLOT_NONE);
                }

                // Per-product angular table (every photon product carries
                // one; isotropic-empty is a safety net our data never hits).
                // Tight CSR bases (issue #104): record this product's ae-row
                // base BEFORE extending the per-row arrays, then a per-row
                // mu-point base accumulated from the current `pa_mu` length.
                let ang = extract_photon_angle(&product.distribution);
                pa_n_energies.push(ang.n_energies);
                pa_ae_offset.push(pa_n_mu.len() as u32);
                let mut mu_base = pa_mu.len() as u32;
                for &n_mu in &ang.n_mu {
                    pa_mu_offset.push(mu_base);
                    mu_base += n_mu;
                }
                pa_energy_grid.extend_from_slice(&ang.energy_grid);
                pa_n_mu.extend_from_slice(&ang.n_mu);
                pa_mu.extend_from_slice(&ang.mu);
                pa_cdf.extend_from_slice(&ang.cdf);
                pa_pdf.extend_from_slice(&ang.pdf);
                pa_interp.extend_from_slice(&ang.interp);

                // Fold this product's contribution into the aggregate.
                let off = rxn_idx as usize * n_grid;
                for (i, pp) in photon_prod.iter_mut().enumerate() {
                    *pp += rxn_xs[off + i] * prod_yield_grid[yslot + i];
                }
            }
        }
    }

    // No photon production anywhere: pad to a single zero row + product slot,
    // a single isotropic-empty angle slot, and a single zero CT slot so the
    // S3 buffers are never empty. Identical to the void-material slot.
    if prod_eout_kind.is_empty() {
        return Ok(GpuPhotonProductionXs::void(n_grid));
    }

    // No continuous-tabular products: pad a single zero CT slot so the eout
    // buffers are never empty (the angle buffers always carry n_product
    // entries, so they are already non-empty here).
    if n_continuous == 0 {
        let empty_ct = EoutSlot::empty();
        ct_n_eout.push(empty_ct.n_energies());
        ct_hist.push(empty_ct.histogram_interp());
        // One ae-row base for the single empty slot; the empty slot has no rows
        // so `ct_x_offset` gains nothing.
        ct_ae_offset.push(ct_n_x.len() as u32);
        let mut x_base = ct_x.len() as u32;
        for &n_x in empty_ct.n_x() {
            ct_x_offset.push(x_base);
            x_base += n_x;
        }
        ct_energy_grid.extend_from_slice(empty_ct.energy_grid());
        ct_n_x.extend_from_slice(empty_ct.n_x());
        ct_x.extend_from_slice(empty_ct.x());
        ct_cdf.extend_from_slice(empty_ct.cdf());
        ct_p.extend_from_slice(empty_ct.p());
        ct_interp.extend_from_slice(empty_ct.interp());
        ct_n_discrete.extend_from_slice(empty_ct.n_discrete());
    }
    let n_continuous = ct_n_eout.len();

    Ok(GpuPhotonProductionXs {
        n_grid,
        photon_prod,
        n_photon_rxn: rxn_mt.len(),
        rxn_xs,
        rxn_mt,
        n_product: prod_eout_kind.len(),
        prod_rxn_idx,
        prod_scaling,
        prod_yield_grid,
        prod_eout_kind,
        prod_line_energy,
        prod_primary_flag,
        prod_awr,
        prod_dist_slot,
        n_continuous,
        ct_n_eout,
        ct_hist,
        ct_ae_offset,
        ct_x_offset,
        ct_energy_grid,
        ct_n_x,
        ct_x,
        ct_cdf,
        ct_p,
        ct_interp,
        ct_n_discrete,
        pa_n_energies,
        pa_ae_offset,
        pa_mu_offset,
        pa_energy_grid,
        pa_n_mu,
        pa_mu,
        pa_cdf,
        pa_pdf,
        pa_interp,
    })
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use yamc_nuclide::reaction_product::EnergyDistribution;

    /// Test arrow datasets live in `crates/yamc/tests/`. Resolve relative to
    /// this crate's manifest dir (mirrors the physics-crate photon test).
    fn td(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("yamc")
            .join("tests")
            .join(name)
    }

    fn load(name: &str) -> Option<Nuclide> {
        if !td(name).exists() {
            eprintln!("Skipping: {name} not found");
            return None;
        }
        Some(
            yamc_nuclide::nuclide_loader::load_nuclide(td(name), &yamc_nuclide::LoadScope::full())
                .unwrap_or_else(|e| panic!("Failed to load {name}: {e}")),
        )
    }

    /// The GPU per-product selection weight at grid index `i_grid`, exactly as
    /// the kernel will compute it:
    /// `prod_scaling[p] * rxn_xs[prod_rxn_idx[p] * n_grid + i] * prod_yield_grid[p * n_grid + i]`.
    fn gpu_product_weight(pp: &GpuPhotonProductionXs, p: usize, i_grid: usize) -> f64 {
        let row = pp.prod_rxn_idx[p] as usize * pp.n_grid + i_grid;
        pp.prod_scaling[p] * pp.rxn_xs[row] * pp.prod_yield_grid[p * pp.n_grid + i_grid]
    }

    /// Replicate the CPU Pass-1 per-(reaction, photon-product) terms from
    /// `yamc_physics::photon::photon_production::sample_photon_product`,
    /// evaluated at the exact `energy` the extractor sees (the reconstructed
    /// grid energy `exp(ln(E))`). Returns the ordered list of per-product
    /// weights `scaling_factor(mt) * rxn_xs(E) * product_yield(E)` in the SAME
    /// order the CPU walk visits them (ascending photon MT, product order),
    /// plus the total. Both sides therefore consume identical per-reaction
    /// cross sections and yields via `cross_section_at` / `product_yield`, so
    /// the parity check isolates the selection layout -- not the lossy log
    /// round-trip.
    fn cpu_product_weights_at(
        grid: &yamc_nuclide::nuclide::FastXSGrid,
        i_grid: usize,
        energy: f64,
    ) -> (Vec<f64>, f64) {
        // Walk in ascending photon MT order to match the extractor's per-nuclide
        // ordering (the FastXSGrid table is not necessarily MT-sorted).
        let mut order: Vec<usize> = (0..grid.photon_rxn_mt_numbers.len()).collect();
        order.sort_by_key(|&j| grid.photon_rxn_mt_numbers[j]);

        let mut weights = Vec::new();
        let mut total = 0.0;
        for &j in &order {
            let mt = grid.photon_rxn_mt_numbers[j];
            let reaction = &grid.photon_rxn_reactions[j];
            let rxn_xs = reaction.cross_section_at(energy).unwrap_or(0.0);
            let f = if is_fission_mt(mt) && !grid.delayed_photon_scaling.is_empty() {
                grid.delayed_photon_scaling[i_grid.min(grid.delayed_photon_scaling.len() - 1)]
            } else {
                1.0
            };
            for product in &reaction.products {
                if product.is_particle_type(&ParticleType::Photon) {
                    let y = product
                        .product_yield
                        .as_ref()
                        .map(|yld| yld.evaluate(energy))
                        .unwrap_or(1.0);
                    let w = f * rxn_xs * y;
                    weights.push(w);
                    total += w;
                }
            }
        }
        (weights, total)
    }

    /// Per-product parity: for a single-nuclide Fe56 material at unit density,
    /// reconstruct each packed product's GPU weight and assert it matches the
    /// CPU's per-(reaction, product) `scaling * rxn_xs * yield` term to ~1e-9
    /// relative, AND that the packed product count matches the CPU walk's, at
    /// several energies spanning thermal/resonance/fast. This pins that a
    /// cumulative walk over the packed weights reproduces the CPU selection
    /// (S2) and that each slot's distribution metadata (S3) lines up with the
    /// right product.
    #[test]
    fn fe56_per_product_weights_match_cpu() {
        let Some(nuclide) = load("Fe56.arrow") else {
            return;
        };
        // Same temperature the extractor resolves ("294"); Fe56 carries
        // multiple temperatures and their grids differ.
        let temp_idx = nuclide
            .get_temp_idx("294")
            .expect("Fe56 must have 294 K loaded");
        let grid = &nuclide.fast_xs[temp_idx];
        assert!(
            !grid.photon_rxn_xs.is_empty(),
            "Fe56 must carry photon-producing reactions for this test"
        );

        // Build the shared log grid from the nuclide's own grid so the
        // extractor's interpolation is an identity (exact parity target).
        let log_energy_grid: Vec<f64> = grid.energy.iter().map(|e| e.ln()).collect();

        let pp = extract_photon_production_xs(&[(&nuclide, 1.0)], "294", &log_energy_grid)
            .expect("extract Fe56 photon production");

        // Structural invariants.
        assert_eq!(pp.n_grid, grid.energy.len());
        assert_eq!(pp.rxn_xs.len(), pp.n_grid * pp.n_photon_rxn);
        assert_eq!(pp.rxn_mt.len(), pp.n_photon_rxn);
        assert_eq!(pp.prod_rxn_idx.len(), pp.n_product);
        assert_eq!(pp.prod_scaling.len(), pp.n_product);
        assert_eq!(pp.prod_yield_grid.len(), pp.n_product * pp.n_grid);
        assert_eq!(pp.prod_eout_kind.len(), pp.n_product);
        assert_eq!(pp.prod_line_energy.len(), pp.n_product);
        assert_eq!(pp.prod_primary_flag.len(), pp.n_product);
        assert_eq!(pp.prod_awr.len(), pp.n_product);
        assert_eq!(pp.prod_dist_slot.len(), pp.n_product);

        // Per-product count must match the CPU walk's number of (reaction,
        // photon-product) pairs. ~532 for Fe56.
        let cpu_n_products: usize = {
            let mut n = 0;
            for j in 0..grid.photon_rxn_mt_numbers.len() {
                for product in &grid.photon_rxn_reactions[j].products {
                    if product.is_particle_type(&ParticleType::Photon) {
                        n += 1;
                    }
                }
            }
            n
        };
        assert_eq!(
            pp.n_product, cpu_n_products,
            "packed product count must equal the CPU's (reaction, photon-product) pair count"
        );
        eprintln!(
            "Fe56: n_photon_rxn={}, n_product={}",
            pp.n_photon_rxn, pp.n_product
        );

        // Probe energies: thermal, resonance, fast, and the 14 MeV DT peak.
        let probes = [0.0253_f64, 1.0, 1.0e3, 1.0e5, 1.0e6, 14.0e6];
        for &e_probe in &probes {
            let (i_grid, _f) = grid.lookup_grid_index(e_probe);
            // Reconstruct the energy the extractor sees from the log grid.
            let e = log_energy_grid[i_grid].exp();

            let (cpu_weights, cpu_total) = cpu_product_weights_at(grid, i_grid, e);
            assert_eq!(
                cpu_weights.len(),
                pp.n_product,
                "CPU per-product weight count must match packed n_product"
            );

            // (a) Per-product weight parity, slot by slot.
            let mut packed_total = 0.0_f64;
            for (p, &cpu_w) in cpu_weights.iter().enumerate() {
                let gpu_w = gpu_product_weight(&pp, p, i_grid);
                packed_total += gpu_w;
                let denom = cpu_w.abs().max(1e-300);
                let rel = (gpu_w - cpu_w).abs() / denom;
                assert!(
                    rel <= 1e-9 || (gpu_w == 0.0 && cpu_w == 0.0),
                    "product {p} @ E~{e:.4e} (i_grid={i_grid}): GPU {gpu_w} vs CPU {cpu_w}, rel={rel:.3e}"
                );
            }

            // (b) Sum of per-product weights == CPU Pass-1 total == aggregate.
            let agg = pp.photon_prod[i_grid];
            assert!(
                (packed_total - agg).abs() <= 1e-9 * agg.abs().max(1.0),
                "packed product-weight sum {packed_total} != aggregate photon_prod {agg} @ i_grid={i_grid}"
            );
            let denom = cpu_total.abs().max(1e-300);
            let rel = (packed_total - cpu_total).abs() / denom;
            eprintln!(
                "E~{e:.4e} (i_grid={i_grid}): packed_total={packed_total} cpu_total={cpu_total} rel={rel:.3e}"
            );
            assert!(
                rel <= 1e-9,
                "E~{e:.4e}: packed total {packed_total} vs CPU total_prob {cpu_total}, rel={rel:.3e}"
            );
        }
    }

    /// The per-product outgoing-energy tags and discrete-line parameters must
    /// match the source reactions: every Fe56 photon product has exactly one
    /// distribution; the discrete ones carry the correct energy / primary_flag /
    /// awr, and the continuous ones get a dense `prod_dist_slot`.
    #[test]
    fn fe56_per_product_eout_metadata() {
        let Some(nuclide) = load("Fe56.arrow") else {
            return;
        };
        let temp_idx = nuclide.get_temp_idx("294").expect("294");
        let grid = &nuclide.fast_xs[temp_idx];
        let log_energy_grid: Vec<f64> = grid.energy.iter().map(|e| e.ln()).collect();
        let pp = extract_photon_production_xs(&[(&nuclide, 1.0)], "294", &log_energy_grid).unwrap();

        // Rebuild the CPU-order list of (kind, line_e, pflag, awr) and check
        // it matches the packed metadata slot for slot.
        let mut order: Vec<usize> = (0..grid.photon_rxn_mt_numbers.len()).collect();
        order.sort_by_key(|&j| grid.photon_rxn_mt_numbers[j]);

        let mut p = 0usize;
        let mut n_discrete = 0usize;
        let mut n_continuous = 0usize;
        let mut max_slot: i64 = -1;
        for &j in &order {
            for product in &grid.photon_rxn_reactions[j].products {
                if !product.is_particle_type(&ParticleType::Photon) {
                    continue;
                }
                assert_eq!(
                    product.distribution.len(),
                    1,
                    "Fe56 photon products are single-distribution"
                );
                match &product.distribution[0] {
                    AngleEnergyDistribution::UncorrelatedAngleEnergy {
                        energy:
                            Some(EnergyDistribution::DiscretePhoton {
                                primary_flag,
                                energy,
                                atomic_weight_ratio,
                            }),
                        ..
                    } => {
                        assert_eq!(pp.prod_eout_kind[p], PHOTON_EOUT_KIND_DISCRETE);
                        assert_eq!(pp.prod_line_energy[p], *energy);
                        assert_eq!(pp.prod_primary_flag[p], *primary_flag);
                        assert_eq!(pp.prod_awr[p], *atomic_weight_ratio);
                        assert_eq!(pp.prod_dist_slot[p], PHOTON_DIST_SLOT_NONE);
                        n_discrete += 1;
                    }
                    AngleEnergyDistribution::UncorrelatedAngleEnergy {
                        energy: Some(EnergyDistribution::ContinuousTabular { .. }),
                        ..
                    } => {
                        assert_eq!(pp.prod_eout_kind[p], PHOTON_EOUT_KIND_CONTINUOUS_TABULAR);
                        assert_ne!(pp.prod_dist_slot[p], PHOTON_DIST_SLOT_NONE);
                        max_slot = max_slot.max(pp.prod_dist_slot[p] as i64);
                        n_continuous += 1;
                    }
                    _ => {
                        assert_eq!(pp.prod_eout_kind[p], PHOTON_EOUT_KIND_NONE);
                    }
                }
                p += 1;
            }
        }
        assert_eq!(p, pp.n_product);
        // Dense continuous slots: max index is exactly count-1.
        if n_continuous > 0 {
            assert_eq!(max_slot, n_continuous as i64 - 1);
        }
        eprintln!(
            "Fe56 eout: discrete={n_discrete}, continuous={n_continuous}, total={}",
            pp.n_product
        );
    }

    /// A material with no photon-producing reactions pads to a single zero row
    /// and a single zero product slot.
    #[test]
    fn no_photon_production_pads_single_zero_slot() {
        use std::collections::HashMap;
        use std::sync::Arc;
        use yamc_nuclide::Reaction;

        // Minimal nuclide: one elastic reaction, no photon products.
        let temp = "294".to_string();
        let energy_grid: Vec<f64> = (0..8)
            .map(|i| {
                let frac = i as f64 / 7.0;
                (1e-3_f64.ln() + frac * (1e6_f64.ln() - 1e-3_f64.ln())).exp()
            })
            .collect();
        let elastic = Reaction {
            cross_section: vec![3.0; energy_grid.len()].into(),
            threshold_idx: 0,
            energy: energy_grid.clone().into(),
            mt_number: 2,
            q_value: 0.0,
            products: vec![],
            scatter_in_cm: false,
            redundant: false,
        };
        let mut reactions_for_temp: HashMap<i32, Arc<Reaction>> = HashMap::new();
        reactions_for_temp.insert(2, Arc::new(elastic));
        let mut energy_map = HashMap::new();
        energy_map.insert(temp.clone(), energy_grid.clone().into());

        let nuclide = Nuclide {
            name: Some("NoGamma".to_string()),
            element: None,
            atomic_symbol: Some("Ng".to_string()),
            atomic_number: Some(1),
            neutron_number: Some(0),
            mass_number: Some(1),
            atomic_weight_ratio: Some(1.0),
            library: None,
            energy: Some(energy_map),
            reactions: vec![reactions_for_temp],
            fissionable: false,
            available_temperatures: vec![temp.clone()],
            loaded_temperatures: vec![temp],
            data_path: None,
            fission_nu: None,
            fast_xs: vec![],
            urr_data: vec![],
            urr_present: false,
            fission_photon_release: None,
            elastic_flat_cache: Default::default(),
            fission_chi_flat_cache: Default::default(),
            delayed_neutron_cache: Default::default(),
            inelastic_angle_flat_cache: Default::default(),
            covariance: None,
            load_scope: Default::default(),
        };

        let log_energy_grid: Vec<f64> = energy_grid.iter().map(|e| e.ln()).collect();
        let pp = extract_photon_production_xs(&[(&nuclide, 1.0)], "294", &log_energy_grid).unwrap();
        assert_eq!(pp.n_photon_rxn, 1);
        assert_eq!(pp.rxn_mt, vec![0]);
        assert_eq!(pp.rxn_xs.len(), pp.n_grid);
        assert_eq!(pp.n_product, 1);
        assert_eq!(pp.prod_eout_kind, vec![PHOTON_EOUT_KIND_NONE]);
        assert_eq!(pp.prod_dist_slot, vec![PHOTON_DIST_SLOT_NONE]);
        assert!(pp.rxn_xs.iter().all(|&w| w == 0.0));
        assert!(pp.prod_yield_grid.iter().all(|&w| w == 0.0));
        assert!(pp.photon_prod.iter().all(|&w| w == 0.0));
    }

    /// Empty material errors cleanly (mirrors `extract_material_xs`).
    #[test]
    fn empty_material_errors() {
        let err = extract_photon_production_xs(&[], "294", &[0.0, 1.0]).unwrap_err();
        assert!(matches!(err, NuclideXsError::EmptyMaterial));
    }
}
