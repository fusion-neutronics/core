//! Per-law outgoing-energy FLATTENING: turn ONE `Reaction` (or one of its
//! products) into the tight, variable-length per-slot arrays the shared flat
//! samplers in this module tree consume.
//!
//! Every `*Slot` here is the extraction half of a sibling sampler:
//! [`EoutSlot`] -> `tabulated_continuous_eout` / `tabulated_equiprobable`,
//! [`CorrSlot`] -> `correlated_angle_energy`, [`KalbachSlot`] ->
//! `kalbach_mann`, [`EvapSlot`] -> `evaporation`, [`MaxwellSlot`] ->
//! `maxwell`, [`WattSlot`] -> `watt`, [`NbpsSlot`] -> `nbody_phase_space`.
//! [`EoutSlot::from_reaction`] also assigns the slot's `EOUT_KIND_*` code, the
//! discriminant [`inelastic_dispatch`](super::inelastic_dispatch) branches on.
//!
//! Moved here verbatim from yamc-gpu's `neutron::xs::distributions` (issue
//! #111, stream unification) so the CPU production transport, which builds
//! without the `gpu` feature, and the GPU host-side extraction share ONE
//! flattening implementation and therefore identical buffer contents.
//! yamc-gpu keeps what is genuinely GPU-buffer-layout work: the
//! `build_per_mt_*_buffers` loops that concatenate these per-slot arrays
//! across `MT_SLOTS`, nuclides, and materials into the global CSR buffers,
//! plus the CSR base offsets built at concatenation time in `translate.rs`.
//!
//! Bit-sensitive: the arrays produced here feed samplers whose RNG schedule is
//! matched draw-for-draw against the cubecl kernel. Any change to the
//! normalisation, the CDF fallback, or the row/point ordering changes sampled
//! outcomes on BOTH backends.

use super::elastic_mu_cm::{ANGLE_INTERP_HISTOGRAM, ANGLE_INTERP_LINLIN};
use super::inelastic_dispatch::{
    EOUT_KIND_CONTINUOUS_TABULAR, EOUT_KIND_CORRELATED, EOUT_KIND_EVAPORATION,
    EOUT_KIND_KALBACH_MANN, EOUT_KIND_LEVEL_INELASTIC, EOUT_KIND_MAXWELL,
    EOUT_KIND_NBODY_PHASE_SPACE, EOUT_KIND_TABULATED, EOUT_KIND_WATT,
};
use yamc_nuclide::particle_type::ParticleType;
use yamc_nuclide::reaction::Reaction;
use yamc_nuclide::reaction_product::{
    AngleEnergyDistribution, EnergyDistribution, ReactionProduct, Tabulated1D, TabulatedProbability,
};
use yamc_nuclide::secondary_correlated::{
    CorrTable, CorrelatedAngleEnergy, Tabular as CorrTabular,
};

/// Maximum number of Evaporation sub-distributions retained per MT
/// slot. A neutron product may carry several Evaporation laws gated
/// by `applicability(E_in)`; the CPU
/// (`ReactionProduct::sample_distribution_index`) stochastically
/// picks one component per collision weighted by applicability. The
/// GPU mirrors this: each component's `θ(E_in)` is resampled onto the
/// slot's shared incident-energy grid and stored component-major in
/// the tight `evap_theta` buffer (component `c`'s row starts at
/// `evap_theta_offset[slot] + c * n_energies[slot]`); the kernel draws
/// one uniform per collision to select the component.
/// 4 covers the widest case in endf-b8.1 (the (n,4n)/MT37 channels of
/// the heavy actinides, which carry four equally-weighted components).
/// A product with more components than this keeps only the first
/// `MAX_EVAP_COMPONENTS` (none observed in endf-b8.1).
///
/// Owned here (issue #111) rather than in yamc-gpu's layout constants:
/// [`EvapSlot::from_evaps`] is the code that enforces the cap, and
/// yamc-physics cannot depend on yamc-gpu. yamc-gpu re-exports it.
pub const MAX_EVAP_COMPONENTS: usize = 4;
/// Maximum number of `CorrelatedAngleEnergy` sub-distribution components
/// kept per neutron product. A product carrying several equally-weighted
/// correlated laws gated by applicability (e.g. F19 MT16 n,2n, two laws at
/// 0.5/0.5) keeps them ALL: the CPU
/// (`ReactionProduct::sample_distribution_index`) draws one component per
/// collision, so collapsing to `distribution[0]` biases the (n,xn)
/// outgoing-energy spectrum (the +9% MT16 residual, issue #111). The rows
/// are stored component-major in the tight `corr_*` buffers (component `c`
/// occupies the `n_per_comp` rows at `corr_ae_offset[slot] + c *
/// n_per_comp`); the kernel/twin draw one uniform per collision to select
/// the component. A product with more components than this, or with UNEQUAL
/// applicability (none observed in endf-b8.1), falls back to a single
/// distribution (`distribution[0]`), so behaviour is unchanged for any such
/// exotic case. 2 covers every multi-law correlated product in endf-b8.1.
///
/// Owned here (issue #111) rather than in yamc-gpu's layout constants:
/// [`CorrSlot::from_components`] is the code that enforces the cap, and
/// yamc-physics cannot depend on yamc-gpu. yamc-gpu re-exports it.
pub const MAX_CORR_COMPONENTS: usize = 4;

/// Per-MT outgoing-energy distribution slice extracted from a
/// `Reaction`. Tight variable-length layout (issue #104): the per-row
/// arrays carry exactly `n_energies` incident-energy rows and the
/// `(x, p, cdf)` arrays carry exactly `sum(n_x)` outgoing points, back
/// to back with no per-axis padding or stride
/// subsampling. The per-slot / per-row CSR base offsets are built at
/// concatenation time from the `n_energies` / `n_x` counts (mirrors the
/// elastic / per-MT-angle families). The kernel reads the slot's rows
/// via `eout_ae_offset[mat_slot]` and each row's points via
/// `eout_x_offset[ae_row]`.
pub struct EoutSlot {
    pub kind: u32,
    pub n_energies: u32,
    /// `1` when the outer `ContinuousTabular.histogram_interp` flag is
    /// set -- the kernel then suppresses the stochastic E_in bracket
    /// pick and the bracket-bound stretch, mirroring CPU's
    /// `ContinuousTabular::sample`. `0` for the default
    /// (lin-lin in incident-energy) path.
    pub histogram_interp: u32,
    pub energy_grid: Vec<f64>, // length n_energies
    pub n_x: Vec<u32>,         // length n_energies
    pub x: Vec<f64>,           // length sum(n_x)
    pub p: Vec<f64>,           // length sum(n_x)
    pub cdf: Vec<f64>,         // length sum(n_x)
    /// Per-(E_in slice) interpolation discriminant for the inner
    /// `Tabular` (Histogram = 0, LinLin = 1). Mirrors
    /// `secondary_correlated::Interpolation` / `ANGLE_INTERP_*`.
    pub interp: Vec<u32>, // length n_energies
    /// Per-(E_in slice) discrete-line prefix count. `n_discrete[i]` is
    /// `Tabular.n_discrete` for slice `i` -- bins `0..n_discrete` are
    /// treated as discrete photon lines (kernel returns the bin
    /// endpoint and skips bracket-bound stretch).
    pub n_discrete: Vec<u32>, // length n_energies
}

impl EoutSlot {
    /// Default slot: `LevelInelastic` kind, no continuum data. The
    /// kernel falls back to the closed-form Q-value energy formula
    /// for these slots. Public so yamc-gpu's slice-S3 photon-production
    /// extractor can append an empty continuous-tabular slot. Tight
    /// layout (issue #104): no rows, so every per-row / per-point `Vec`
    /// is empty.
    pub fn empty() -> Self {
        Self {
            kind: EOUT_KIND_LEVEL_INELASTIC,
            n_energies: 0,
            histogram_interp: 0,
            energy_grid: Vec::new(),
            n_x: Vec::new(),
            x: Vec::new(),
            p: Vec::new(),
            cdf: Vec::new(),
            interp: Vec::new(),
            n_discrete: Vec::new(),
        }
    }

    /// Pull the outgoing-energy distribution from a reaction's first
    /// neutron product. Recognises `LevelInelastic` (kept as the
    /// closed-form path), `ContinuousTabular` (encoded into the
    /// `eout_*` flat CDF buffers), and `CorrelatedAngleEnergy`
    /// (sets `kind = EOUT_KIND_CORRELATED` as a marker -- actual
    /// data lives in the parallel `corr_*` buffers built by
    /// `CorrSlot::from_reaction`). Anything else falls back to the
    /// empty slot, so the kernel uses the closed-form energy.
    pub fn from_reaction(reaction: &Reaction) -> Self {
        let Some(neutron_product) = first_neutron_product(reaction) else {
            return Self::empty();
        };

        for d in &neutron_product.distribution {
            match d {
                AngleEnergyDistribution::UncorrelatedAngleEnergy { energy, .. } => {
                    let Some(energy_dist) = energy.as_ref() else {
                        continue;
                    };
                    return match energy_dist {
                        EnergyDistribution::LevelInelastic { .. } => Self::empty(),
                        EnergyDistribution::ContinuousTabular {
                            energy,
                            energy_out,
                            histogram_interp,
                        } => Self::from_continuous_tabular(energy, energy_out, *histogram_interp),
                        EnergyDistribution::Evaporation { .. } => {
                            // Mark the slot so the kernel routes to
                            // the `evap_*` buffers; the actual θ(E_in)
                            // and `u` are encoded by
                            // `EvapSlot::from_reaction`.
                            let mut s = Self::empty();
                            s.kind = EOUT_KIND_EVAPORATION;
                            s
                        }
                        EnergyDistribution::Maxwell { .. } => {
                            // Mark the slot so the kernel routes to
                            // the `maxwell_*` buffers; the actual
                            // θ(E_in) and `u` are encoded by
                            // `MaxwellSlot::from_reaction`.
                            let mut s = Self::empty();
                            s.kind = EOUT_KIND_MAXWELL;
                            s
                        }
                        EnergyDistribution::Watt { .. } => {
                            // Mark the slot so the kernel routes to
                            // the `watt_*` buffers; the actual
                            // a(E_in), b(E_in), and `u` are encoded
                            // by `WattSlot::from_reaction`.
                            let mut s = Self::empty();
                            s.kind = EOUT_KIND_WATT;
                            s
                        }
                        EnergyDistribution::Tabulated { energy, energy_out } => {
                            Self::from_tabulated_equiprobable(energy, energy_out)
                        }
                        // DiscretePhoton needs a dedicated GPU
                        // sampler (and lives on the photon side
                        // anyway). Fall back to the closed form.
                        _ => Self::empty(),
                    };
                }
                AngleEnergyDistribution::Evaporation { .. } => {
                    let mut s = Self::empty();
                    s.kind = EOUT_KIND_EVAPORATION;
                    return s;
                }
                AngleEnergyDistribution::NBodyPhaseSpace { n_bodies, .. } => {
                    // CPU panics for n_bodies > 5 -- match the
                    // 3 / 4 / 5 cap by skipping the marker for
                    // out-of-range values. The kernel falls back to
                    // closed-form Q-value energy + isotropic mu.
                    if *n_bodies >= 3 && *n_bodies <= 5 {
                        let mut s = Self::empty();
                        s.kind = EOUT_KIND_NBODY_PHASE_SPACE;
                        return s;
                    }
                    return Self::empty();
                }
                AngleEnergyDistribution::CorrelatedAngleEnergy { .. } => {
                    // Mark the slot as correlated so the kernel
                    // routes to the `corr_*` buffers; the actual
                    // data is encoded by `CorrSlot::from_reaction`.
                    let mut s = Self::empty();
                    s.kind = EOUT_KIND_CORRELATED;
                    return s;
                }
                AngleEnergyDistribution::KalbachMann { .. } => {
                    // Mark the slot as Kalbach-Mann so the kernel
                    // routes to the `km_*` buffers; the actual data
                    // is encoded by `KalbachSlot::from_reaction`.
                    let mut s = Self::empty();
                    s.kind = EOUT_KIND_KALBACH_MANN;
                    return s;
                }
            }
        }
        Self::empty()
    }

    /// Populate the `eout_*` buffers (energy_grid, n_x, x) from a
    /// sparse equiprobable-bin distribution. `cdf` / `p` stay
    /// zero-length per row's contribution -- unused by
    /// `EOUT_KIND_TABULATED` since the sampler is a uniform bin pick,
    /// not CDF inversion (the parallel `cdf` / `p` arrays are still
    /// grown row-for-row with `x` so all three share one `x_offset`).
    /// Tight, full-resolution layout (issue #104): every incident
    /// energy and every outgoing bin is kept; rows are concatenated
    /// with no per-axis padding or subsampling.
    fn from_tabulated_equiprobable(energy: &[f64], energy_out: &[Vec<f64>]) -> Self {
        if energy.is_empty() || energy_out.is_empty() {
            return Self::empty();
        }
        let n_e = energy.len();

        let mut slot = Self::empty();
        slot.kind = EOUT_KIND_TABULATED;
        slot.n_energies = n_e as u32;
        slot.energy_grid.reserve(n_e);
        slot.n_x.reserve(n_e);
        slot.interp.reserve(n_e);
        slot.n_discrete.reserve(n_e);
        for (i, &e) in energy.iter().enumerate() {
            slot.energy_grid.push(e);
            slot.interp.push(ANGLE_INTERP_HISTOGRAM);
            slot.n_discrete.push(0);
            let xs: &[f64] = energy_out.get(i).map(|v| v.as_slice()).unwrap_or(&[]);
            slot.n_x.push(xs.len() as u32);
            for &x in xs {
                slot.x.push(x);
                // `cdf` / `p` intentionally left zero -- the kernel reads
                // them only on the `EOUT_KIND_CONTINUOUS_TABULAR` path; the
                // rows are still grown so the shared `x_offset` aligns.
                slot.cdf.push(0.0);
                slot.p.push(0.0);
            }
        }
        slot
    }

    /// Build an `EoutSlot` directly from a PHOTON product's
    /// `ContinuousTabular` outgoing-energy distribution (slice S3).
    /// `EoutSlot::from_reaction` only reads a reaction's *neutron*
    /// product; photon products live in `reaction.products` filtered by
    /// `ParticleType::Photon`, so S3 calls this with the photon product's
    /// already-destructured `(energy, energy_out, histogram_interp)`.
    /// Delegates to [`Self::from_continuous_tabular`] so the byte layout,
    /// normalisation, subsampling, and `n_discrete` conversion are
    /// identical to the neutron eout buffers -- the resulting buffers feed
    /// `sample_continuous_tabular_eout` byte-compatibly.
    pub fn from_photon_continuous_tabular(
        energy: &[f64],
        energy_out: &[TabulatedProbability],
        histogram_interp: bool,
    ) -> Self {
        Self::from_continuous_tabular(energy, energy_out, histogram_interp)
    }

    /// Slice S3 accessors. The flat photon CT buffers are assembled in
    /// yamc-gpu's `neutron::xs::photon_production` by reading each slot's
    /// arrays. They expose exactly the fields the `eout_*` flat buffers
    /// carry; no copies, just borrows.
    pub fn n_energies(&self) -> u32 {
        self.n_energies
    }
    pub fn histogram_interp(&self) -> u32 {
        self.histogram_interp
    }
    pub fn energy_grid(&self) -> &[f64] {
        &self.energy_grid
    }
    pub fn n_x(&self) -> &[u32] {
        &self.n_x
    }
    pub fn x(&self) -> &[f64] {
        &self.x
    }
    pub fn p(&self) -> &[f64] {
        &self.p
    }
    pub fn cdf(&self) -> &[f64] {
        &self.cdf
    }
    pub fn interp(&self) -> &[u32] {
        &self.interp
    }
    pub fn n_discrete(&self) -> &[u32] {
        &self.n_discrete
    }

    fn from_continuous_tabular(
        energy: &[f64],
        energy_out: &[TabulatedProbability],
        histogram_interp: bool,
    ) -> Self {
        if energy.is_empty() || energy_out.is_empty() {
            return Self::empty();
        }
        let n_e = energy.len();

        let mut slot = Self::empty();
        slot.kind = EOUT_KIND_CONTINUOUS_TABULAR;
        slot.n_energies = n_e as u32;
        slot.histogram_interp = if histogram_interp { 1 } else { 0 };
        slot.energy_grid.reserve(n_e);
        slot.n_x.reserve(n_e);
        slot.interp.reserve(n_e);
        slot.n_discrete.reserve(n_e);
        // Tight, full-resolution layout (issue #104): keep every incident
        // energy and every outgoing point; rows are concatenated with no
        // per-axis padding or stride-subsampling, so the
        // GPU samples the same data the CPU does. `n_discrete` carries over
        // verbatim (no subsampled-index remap needed).
        for (i, &e) in energy.iter().enumerate() {
            slot.energy_grid.push(e);
            // ContinuousTabular's `energy_out` length matches `energy`
            // when normalised; an empty / missing slice yields a 0-point
            // row (the kernel's `n_x < 2` guard falls back to closed form).
            let dist = energy_out.get(i);
            let Some(TabulatedProbability::Tabulated {
                x,
                p,
                c,
                interp,
                n_discrete,
            }) = dist
            else {
                slot.n_x.push(0);
                slot.interp.push(ANGLE_INTERP_HISTOGRAM);
                slot.n_discrete.push(0);
                continue;
            };
            let m_in = x.len();
            if m_in == 0 {
                slot.n_x.push(0);
                slot.interp.push(ANGLE_INTERP_HISTOGRAM);
                slot.n_discrete.push(0);
                continue;
            }

            let cdf_owned: Vec<f64>;
            let cdf_slice: &[f64] = if c.len() == m_in {
                c
            } else {
                cdf_owned = trapezoidal_cdf(x, p);
                &cdf_owned
            };
            // Renormalise so the last CDF entry is exactly 1.0; scale
            // the per-point PDF by the same factor so `∫ p dx = c`
            // stays consistent on the renormalised pair (required by
            // the kernel's quadratic LinLin CDF inversion).
            let cdf_max = cdf_slice.last().copied().unwrap_or(0.0);
            let pdf_scale = if cdf_max > 0.0 { 1.0 / cdf_max } else { 0.0 };

            slot.n_x.push(m_in as u32);
            slot.interp.push(match interp {
                yamc_nuclide::reaction_product::TabulatedInterp::Histogram => {
                    ANGLE_INTERP_HISTOGRAM
                }
                yamc_nuclide::reaction_product::TabulatedInterp::LinLin => ANGLE_INTERP_LINLIN,
            });
            slot.n_discrete.push(*n_discrete as u32);
            for j in 0..m_in {
                slot.x.push(x[j]);
                slot.cdf.push(if cdf_max > 0.0 {
                    cdf_slice[j] / cdf_max
                } else {
                    j as f64 / (m_in - 1).max(1) as f64
                });
                slot.p.push(p.get(j).copied().unwrap_or(0.0) * pdf_scale);
            }
        }
        slot
    }
}

/// Per-MT correlated angle-energy slice extracted from a `Reaction`.
/// Used when `EoutSlot::from_reaction` returns
/// `EOUT_KIND_CORRELATED` for that slot -- the eout buffers stay
/// empty and the `corr_*` buffers carry the joint `(E_out, mu)`
/// CDF data. Tight, variable-length layout (issue #104): the three
/// nesting levels (E_in rows -> E_out x-points -> mu points) are
/// concatenated full-resolution with no `MAX_CORR_*` stride and no
/// subsampling, so GPU and CPU sample byte-identical data. The
/// per-slot / per-row / per-x CSR offsets are built at concatenation
/// time (`push_corr_csr_offsets` in translate.rs) from the `n_x` /
/// `n_mu` counts, mirroring the elastic / eout families.
pub struct CorrSlot {
    /// Number of incident-energy rows TOTAL across all components, i.e.
    /// `n_components * n_per_comp`. The per-row arrays (`energy_grid`,
    /// `n_x`, `interp`, `n_discrete`) carry this many entries, laid out
    /// component-major. For a single-component slot this is just the one
    /// distribution's incident-energy count.
    pub n_energies: u32,
    /// Number of `CorrelatedAngleEnergy` components retained
    /// (1..=MAX_CORR_COMPONENTS). `>= 2` when the neutron product carries
    /// several equally-weighted correlated laws gated by applicability
    /// (F19 MT16 n,2n, two laws at 0.5/0.5): the CPU draws one component
    /// per collision (`ReactionProduct::sample_distribution_index`), so
    /// collapsing to `distribution[0]` biases the (n,xn) spectrum (the +9%
    /// MT16 residual, issue #111). The kernel/twin draw one uniform per
    /// collision to pick the component when `>= 2`. Per-component rows are
    /// `n_per_comp = n_energies / n_components`; component `c` starts at
    /// row `c * n_per_comp`.
    pub n_components: u32,
    pub energy_grid: Vec<f64>, // length n_energies (one per E_in row, component-major)
    pub n_x: Vec<u32>,         // length n_energies
    pub x: Vec<f64>,           // length sum(n_x) (one per E_out x-point)
    pub cdf: Vec<f64>,         // length sum(n_x)
    pub p: Vec<f64>,           // length sum(n_x) (per-point E_out PDF)
    pub interp: Vec<u32>,      // length n_energies (per-E_in Histogram/LinLin)
    pub n_discrete: Vec<u32>,  // length n_energies (per-E_in discrete prefix count)
    pub n_mu: Vec<u32>,        // length sum(n_x) (one per E_out x-point)
    pub mu: Vec<f64>,          // length sum(n_mu) (one per mu point)
    pub mu_cdf: Vec<f64>,      // length sum(n_mu)
    pub mu_pdf: Vec<f64>,      // length sum(n_mu)
    pub mu_interp: Vec<u32>,   // length sum(n_x) (one per E_out x-point)
}

impl CorrSlot {
    pub fn empty() -> Self {
        Self {
            n_energies: 0,
            n_components: 0,
            energy_grid: Vec::new(),
            n_x: Vec::new(),
            x: Vec::new(),
            cdf: Vec::new(),
            p: Vec::new(),
            interp: Vec::new(),
            n_discrete: Vec::new(),
            n_mu: Vec::new(),
            mu: Vec::new(),
            mu_cdf: Vec::new(),
            mu_pdf: Vec::new(),
            mu_interp: Vec::new(),
        }
    }

    /// Pull `CorrelatedAngleEnergy` data from a reaction's first neutron
    /// product, encode it into the tight `corr_*` buffers at full
    /// resolution (no subsampling, issue #104). When the product carries
    /// MORE THAN ONE correlated law gated by EQUAL applicability (the
    /// only multi-law correlated case in endf-b8.1: F19 MT16 n,2n with two
    /// 0.5/0.5 laws), all components are kept component-major so the
    /// kernel/twin can draw one per collision -- matching the CPU's
    /// per-collision applicability selection. Returns `Self::empty()` for
    /// any other distribution kind.
    pub fn from_reaction(reaction: &Reaction) -> Self {
        let Some(neutron_product) = first_neutron_product(reaction) else {
            return Self::empty();
        };
        // Collect every correlated sub-distribution with its applicability
        // (the per-incident-energy weight the CPU uses to pick which to
        // sample when there is more than one).
        let mut comps: Vec<(&CorrelatedAngleEnergy, Option<&Tabulated1D>)> = Vec::new();
        for (i, d) in neutron_product.distribution.iter().enumerate() {
            if let AngleEnergyDistribution::CorrelatedAngleEnergy { correlated } = d {
                comps.push((correlated, neutron_product.applicability.get(i)));
            }
        }
        if comps.is_empty() {
            return Self::empty();
        }
        Self::from_components(&comps)
    }

    /// Build the slot from one or more `(correlated, applicability)`
    /// components. A SINGLE component reduces to the original behaviour
    /// (no selector draw). With SEVERAL the CPU draws a component per
    /// collision weighted by `applicability(E_in)`; when those weights are
    /// EQUAL across the active components (every multi-law correlated
    /// product in endf-b8.1: 1/N each) the mixture is exactly "pick one of
    /// N uniformly", which the kernel/twin reproduce with a single uniform
    /// draw. We keep all components (each with its own incident-energy
    /// rows, component-major) whenever the weights are equal AND the
    /// per-component row counts match (so the component stride is fixed).
    /// Otherwise we fall back to `components[0]` (no such exotic correlated
    /// product exists in endf-b8.1, so nothing regresses).
    fn from_components(comps: &[(&CorrelatedAngleEnergy, Option<&Tabulated1D>)]) -> Self {
        let n_comp_raw = comps.len();
        let first_n_e = comps[0].0.energy.len();
        let equal_rows = comps.iter().all(|(c, _)| c.energy.len() == first_n_e);
        // Equal-applicability check on the union of the components' incident
        // energy grids (where the distributions live).
        let mut grid: Vec<f64> = Vec::new();
        for (c, _) in comps {
            grid.extend_from_slice(&c.energy);
        }
        grid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        grid.dedup();
        let apps: Vec<Option<&Tabulated1D>> = comps.iter().map(|(_, a)| *a).collect();
        let keep_all = (2..=MAX_CORR_COMPONENTS).contains(&n_comp_raw)
            && equal_rows
            && corr_applicabilities_equal(&apps, &grid);

        let mut slot = Self::empty();
        if keep_all {
            slot.n_components = n_comp_raw as u32;
            slot.n_energies = (n_comp_raw * first_n_e) as u32;
            for (c, _) in comps {
                Self::pack_correlated(&mut slot, c);
            }
        } else {
            slot.n_components = 1;
            slot.n_energies = comps[0].0.energy.len() as u32;
            Self::pack_correlated(&mut slot, comps[0].0);
        }
        slot
    }

    /// Append one `CorrelatedAngleEnergy`'s incident-energy rows (and their
    /// E_out + mu sub-tables) onto the tail of the slot's tight buffers.
    /// Does NOT touch `n_energies` / `n_components` (the caller sets those).
    fn pack_correlated(slot: &mut Self, corr: &CorrelatedAngleEnergy) {
        for (i, &e_in) in corr.energy.iter().enumerate() {
            slot.energy_grid.push(e_in);
            match corr.distributions.get(i) {
                Some(table) => Self::pack_corr_table(slot, table),
                None => {
                    // Keep the per-E_in row arrays aligned with energy_grid
                    // even when a distribution is missing for this index.
                    slot.n_x.push(0);
                    slot.interp.push(ANGLE_INTERP_HISTOGRAM);
                    slot.n_discrete.push(0);
                }
            }
        }
    }

    /// Pack one `CorrTable` (E_out CDF + per-bin angular sub-table)
    /// onto the tail of the slot's tight buffers (no `MAX_CORR_*`
    /// stride, issue #104). Appends one per-E_in row plus its
    /// `m_out` E_out x-points and their mu sub-tables in order.
    fn pack_corr_table(slot: &mut Self, table: &CorrTable) {
        let m_out = table.e_out.len();
        // Empty distribution: still emit a zero-length per-E_in row so the
        // per-E_in arrays (energy_grid / n_x / interp / n_discrete) stay
        // aligned (matches the old m_in == 0 early-return that left a
        // zero-filled row).
        slot.n_x.push(m_out as u32);
        slot.interp.push(match table.interpolation {
            yamc_nuclide::secondary_correlated::Interpolation::Histogram => ANGLE_INTERP_HISTOGRAM,
            yamc_nuclide::secondary_correlated::Interpolation::LinLin => ANGLE_INTERP_LINLIN,
        });
        // Tight layout: n_discrete carries directly (no subsampled remap).
        slot.n_discrete.push(table.n_discrete as u32);
        if m_out == 0 {
            return;
        }
        let cdf_owned: Vec<f64>;
        let cdf_slice: &[f64] = if table.c.len() == m_out {
            &table.c
        } else {
            cdf_owned = trapezoidal_cdf(&table.e_out, &table.p);
            &cdf_owned
        };
        let cdf_max = cdf_slice.last().copied().unwrap_or(0.0);
        // PDF normalisation: divide each point by the slice integral
        // so `∫ p dx = c_last = 1.0` after the CDF renormalisation.
        // Mirrors CPU's `Tabular::normalize` post-load.
        let pdf_scale = if cdf_max > 0.0 { 1.0 / cdf_max } else { 0.0 };

        for (j, &c) in cdf_slice.iter().enumerate().take(m_out) {
            slot.x.push(table.e_out[j]);
            slot.cdf.push(if cdf_max > 0.0 {
                c / cdf_max
            } else {
                j as f64 / (m_out - 1).max(1) as f64
            });
            slot.p
                .push(table.p.get(j).copied().unwrap_or(0.0) * pdf_scale);
            // Each (E_in_idx, E_out_idx) bin carries its own
            // angular sub-table. Append it to the mu / mu_cdf
            // buffers; bins without one push n_mu = 0 (the
            // kernel falls back to isotropic for those).
            match table.angle.get(j) {
                Some(ang) if !ang.x.is_empty() => Self::pack_corr_angle(slot, ang),
                _ => {
                    slot.n_mu.push(0);
                    slot.mu_interp.push(ANGLE_INTERP_HISTOGRAM);
                }
            }
        }
    }

    fn pack_corr_angle(slot: &mut Self, ang: &CorrTabular) {
        let n_mu_in = ang.x.len();
        let cdf_owned: Vec<f64>;
        let cdf_slice: &[f64] = if ang.c.len() == n_mu_in {
            &ang.c
        } else {
            cdf_owned = trapezoidal_cdf(&ang.x, &ang.p);
            &cdf_owned
        };
        let cdf_max = cdf_slice.last().copied().unwrap_or(0.0);
        let pdf_scale = if cdf_max > 0.0 { 1.0 / cdf_max } else { 0.0 };

        slot.n_mu.push(n_mu_in as u32);
        slot.mu_interp.push(match ang.interpolation {
            yamc_nuclide::secondary_correlated::Interpolation::Histogram => ANGLE_INTERP_HISTOGRAM,
            yamc_nuclide::secondary_correlated::Interpolation::LinLin => ANGLE_INTERP_LINLIN,
        });
        for (k, &c) in cdf_slice.iter().enumerate().take(n_mu_in) {
            slot.mu.push(ang.x[k]);
            slot.mu_cdf.push(if cdf_max > 0.0 {
                c / cdf_max
            } else {
                k as f64 / (n_mu_in - 1).max(1) as f64
            });
            slot.mu_pdf
                .push(ang.p.get(k).copied().unwrap_or(0.0) * pdf_scale);
        }
    }
}

/// Per-MT Kalbach-Mann slice extracted from a reaction's first
/// neutron product. Tight variable-length layout (issue #104): the
/// per-row arrays carry exactly `n_energies` incident-energy rows and
/// the `(x, p, cdf, r, a)` arrays carry exactly `sum(n_x)` outgoing
/// points, back to back with no per-axis padding or
/// stride subsampling. The per-slot / per-row CSR base offsets are
/// built at concatenation time from the `n_energies` / `n_x` counts
/// (mirrors the eout / correlated families). The kernel reads the
/// slot's rows via `km_ae_offset[mat_slot]` and each row's points via
/// `km_x_offset[ae_row]`.
pub struct KalbachSlot {
    pub n_energies: u32,
    pub energy_grid: Vec<f64>, // length n_energies
    pub interp: Vec<u32>,      // length n_energies
    pub n_discrete: Vec<u32>,  // length n_energies
    pub n_x: Vec<u32>,         // length n_energies
    pub x: Vec<f64>,           // length sum(n_x)
    pub p: Vec<f64>,           // length sum(n_x)
    pub cdf: Vec<f64>,         // length sum(n_x)
    pub r: Vec<f64>,           // length sum(n_x)
    pub a: Vec<f64>,           // length sum(n_x)
}

impl KalbachSlot {
    pub fn empty() -> Self {
        Self {
            n_energies: 0,
            energy_grid: Vec::new(),
            interp: Vec::new(),
            n_discrete: Vec::new(),
            n_x: Vec::new(),
            x: Vec::new(),
            p: Vec::new(),
            cdf: Vec::new(),
            r: Vec::new(),
            a: Vec::new(),
        }
    }

    pub fn from_reaction(reaction: &Reaction) -> Self {
        let Some(neutron_product) = first_neutron_product(reaction) else {
            return Self::empty();
        };
        let kalbach = neutron_product.distribution.iter().find_map(|d| match d {
            AngleEnergyDistribution::KalbachMann { kalbach } => Some(kalbach),
            _ => None,
        });
        let Some(kalbach) = kalbach else {
            return Self::empty();
        };
        Self::from_kalbach(kalbach)
    }

    fn from_kalbach(km: &yamc_nuclide::secondary_kalbach::KalbachMann) -> Self {
        if km.energy.is_empty() || km.distributions.is_empty() {
            return Self::empty();
        }
        let n_e = km.energy.len();
        let mut slot = Self::empty();
        slot.n_energies = n_e as u32;
        slot.energy_grid.reserve(n_e);
        slot.interp.reserve(n_e);
        slot.n_discrete.reserve(n_e);
        slot.n_x.reserve(n_e);
        // Tight, full-resolution layout (issue #104): keep every incident
        // energy and every outgoing point; rows are concatenated with no
        // per-axis padding or stride-subsampling, so the GPU
        // samples the same data the CPU does. A missing / empty `KMTable`
        // still pushes a 0-point row (the kernel's `n_kx < 2` guard falls
        // back to the closed-form energy).
        for (i, &e) in km.energy.iter().enumerate() {
            slot.energy_grid.push(e);
            match km.distributions.get(i) {
                Some(table) => Self::pack_table(&mut slot, table),
                None => {
                    slot.interp.push(ANGLE_INTERP_HISTOGRAM);
                    slot.n_discrete.push(0);
                    slot.n_x.push(0);
                }
            }
        }
        slot
    }

    /// Append one `KMTable` (E_out PDF/CDF + per-bin `(r, a)`) onto the
    /// slot's tight concatenated buffers as a single incident-energy row.
    /// Pushes exactly one entry onto each per-row array and `n_x` points
    /// onto each per-point array (issue #104).
    fn pack_table(slot: &mut Self, table: &yamc_nuclide::secondary_kalbach::KMTable) {
        let m_in = table.e_out.len();
        if m_in == 0 {
            slot.interp.push(ANGLE_INTERP_HISTOGRAM);
            slot.n_discrete.push(0);
            slot.n_x.push(0);
            return;
        }
        slot.interp.push(match table.interpolation {
            yamc_nuclide::secondary_kalbach::Interpolation::Histogram => ANGLE_INTERP_HISTOGRAM,
            yamc_nuclide::secondary_kalbach::Interpolation::LinLin => ANGLE_INTERP_LINLIN,
        });
        slot.n_discrete.push(table.n_discrete as u32);
        slot.n_x.push(m_in as u32);
        let cdf_max = table.c.last().copied().unwrap_or(0.0);
        for j in 0..m_in {
            slot.x.push(table.e_out[j]);
            // PDF and CDF: renormalise the CDF to 1.0 at the last
            // entry -- ENDF data is usually pre-normalised but small
            // drift breaks the kernel's `xi <= cdf[last]` guard.
            slot.p.push(table.p.get(j).copied().unwrap_or(0.0));
            slot.cdf.push(if cdf_max > 0.0 {
                table.c.get(j).copied().unwrap_or(0.0) / cdf_max
            } else {
                j as f64 / (m_in - 1).max(1) as f64
            });
            slot.r.push(table.r.get(j).copied().unwrap_or(0.0));
            slot.a.push(table.a.get(j).copied().unwrap_or(0.0));
        }
    }
}

/// Per-MT Evaporation slice. Extracted from a reaction's first
/// neutron product when the energy distribution is `Evaporation`
/// (either via `UncorrelatedAngleEnergy { energy:
/// Some(EnergyDistribution::Evaporation { theta, u }) }` or the
/// top-level `AngleEnergyDistribution::Evaporation { theta, u }`
/// variant). Empty otherwise.
pub struct EvapSlot {
    /// Number of incident-energy points on the SHARED grid (`energy_grid`,
    /// `u_grid`, and each component's `theta` row use these `n_energies`
    /// points). 0 marks an empty (non-Evaporation) slot.
    pub n_energies: u32,
    /// Number of Evaporation components retained (1..=MAX_EVAP_COMPONENTS).
    /// A product carrying several Evaporation laws gated by equal
    /// `applicability(E_in)` keeps them ALL: the CPU
    /// (`ReactionProduct::sample_distribution_index`) draws one component
    /// per collision, so collapsing to a single dominant curve biases the
    /// outgoing (n,xn) spectrum (it is the hotter/colder component, not the
    /// 1/N mixture). The kernel draws one uniform per collision to pick a
    /// component when `n_components >= 2`. When the components do NOT carry
    /// equal applicability (step-gated windows: endf-b8.1 Ar36 / Ar38 /
    /// Na22 / Na23 / Co58_m1 MT91), the slot falls back to
    /// `n_components == 1` with the applicability-dominant curve per
    /// incident-energy grid point -- exact for 0/1 step windows as long as
    /// the shared grid contains the window edges (see `from_evaps`).
    pub n_components: u32,
    pub energy_grid: Vec<f64>, // length n_energies (shared incident-energy grid)
    /// Component-major `θ(E_in)`: `theta[c * n_energies + i]` is component
    /// `c`'s theta at `energy_grid[i]`. Length `n_components * n_energies`
    /// (tight, issue #104). Each component is resampled onto the shared grid
    /// (the components' native breakpoint counts differ, e.g. 10 vs 2 in Ba140
    /// MT16, so a shared grid avoids per-component grid bookkeeping).
    pub theta: Vec<f64>,
    /// Restriction energy `u`, tabulated per incident-energy grid point
    /// (length n_energies, parallel to `energy_grid`). When a product
    /// carries MORE THAN ONE Evaporation distribution gated by
    /// `applicability` (e.g. ENDF File-6 Na23 MT 91, where a low-`u`
    /// distribution is active at 14 MeV and a high-`u` one at lower
    /// incident energies), `u` is incident-energy-dependent: a single
    /// scalar would clamp the outgoing spectrum to the wrong band. The
    /// kernel reads `u` at the same incident-energy bracket it uses for
    /// `theta`. The multi-Evaporation products in endf-b8.1 share a single
    /// `u` across their components, so one shared `u_grid` suffices.
    pub u_grid: Vec<f64>, // length n_energies
}

impl EvapSlot {
    pub fn empty() -> Self {
        Self {
            n_energies: 0,
            n_components: 0,
            energy_grid: Vec::new(),
            theta: Vec::new(),
            u_grid: Vec::new(),
        }
    }

    pub fn from_reaction(reaction: &Reaction) -> Self {
        let Some(neutron_product) = first_neutron_product(reaction) else {
            return Self::empty();
        };
        // Collect every Evaporation sub-distribution together with its
        // applicability (the per-incident-energy weight the CPU uses to
        // pick which distribution to sample per collision).
        let mut evaps: Vec<(&Tabulated1D, f64, Option<&Tabulated1D>)> = Vec::new();
        for (i, d) in neutron_product.distribution.iter().enumerate() {
            let applic = neutron_product.applicability.get(i);
            match d {
                AngleEnergyDistribution::UncorrelatedAngleEnergy { energy, .. } => {
                    if let Some(EnergyDistribution::Evaporation { theta, u }) = energy.as_ref() {
                        evaps.push((theta, *u, applic));
                    }
                }
                AngleEnergyDistribution::Evaporation { theta: Some(t), u } => {
                    evaps.push((t, *u, applic));
                }
                _ => {}
            }
        }
        if evaps.is_empty() {
            return Self::empty();
        }
        Self::from_evaps(&evaps)
    }

    /// Build the slot from one or more `(theta, u, applicability)`
    /// Evaporation distributions.
    ///
    /// A SINGLE distribution reduces to the original per-grid theta with
    /// constant `u` (single component, no per-collision selector draw).
    ///
    /// With SEVERAL distributions the CPU draws a component per collision
    /// weighted by `applicability(E_in)`. When those weights are EQUAL
    /// across the active components (endf-b8.1 (n,xn) mixtures, 1/N each,
    /// e.g. Ta182 MT17), the mixture is exactly "pick one of N components
    /// uniformly", which the kernel reproduces with a single uniform draw.
    /// We therefore keep all components (their `θ(E_in)` resampled onto a
    /// shared incident-energy grid) whenever the weights are equal. If the
    /// weights are NOT equal (0/1 step-gated windows: Ar36 / Ar38 / Na22 /
    /// Na23 / Co58_m1 MT91, where a high-`u` law applies at low E_in and a
    /// low-`u` law above the window edge) we fall back to the single-curve
    /// applicability-argmax per grid point, exact for step windows because
    /// the shared grid includes the applicability breakpoints.
    pub fn from_evaps(evaps: &[(&Tabulated1D, f64, Option<&Tabulated1D>)]) -> Self {
        // Shared incident-energy grid: union of every component's theta
        // breakpoints, sorted/deduped. The components share an E_in domain
        // (their applicability ranges coincide), so a shared grid loses no
        // resolution. Tight, full-resolution layout (issue #104): every grid
        // point is kept (no per-axis stride-subsampling), so the GPU
        // samples the same θ(E_in) the CPU does. The per-(slab,MT) CSR bases
        // (`evap_ae_offset` / `evap_theta_offset`) are built at concatenation
        // time in `translate.rs`.
        let mut grid: Vec<f64> = Vec::new();
        for (Tabulated1D::Tabulated1D { x, .. }, _, _) in evaps {
            grid.extend_from_slice(x);
        }
        // The applicability breakpoints are grid points too: the argmax
        // collapse below switches components exactly at those incident
        // energies, and the kernel reads `u` at the nearest-lower grid
        // point. Without them a mid-segment switch lands on the wrong
        // component's restriction energy `u` -- endf-b8.1 Ar38 MT91 has
        // theta breakpoints at [6.06, 7, 8, 20] MeV but its second law
        // (u = 2.2 MeV) applies from 11 MeV, so every E_in in [8, 20)
        // kept the first law's u = 5.9 MeV and hard-truncated the
        // outgoing spectrum at E_in - 5.9 (V&V: zero flux above 8.19 MeV
        // for a 14.06 MeV source; same pattern in Ar36 / Na22 / Co58_m1
        // MT91).
        for (_, _, applic) in evaps {
            if let Some(Tabulated1D::Tabulated1D { x, .. }) = applic {
                grid.extend_from_slice(x);
            }
        }
        grid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        grid.dedup();
        if grid.is_empty() {
            return Self::empty();
        }
        let n_e = grid.len();

        // Decide whether to keep all components (equal applicability) or
        // collapse to the dominant one. `keep_all` is true only when there are
        // >= 2 components (capped at MAX_EVAP_COMPONENTS, which bounds the
        // kernel's component loop) and their applicabilities are equal at every
        // grid point (within tolerance). Otherwise collapse.
        let n_comp_raw = evaps.len();
        let keep_all =
            (2..=MAX_EVAP_COMPONENTS).contains(&n_comp_raw) && applicabilities_equal(evaps, &grid);

        let mut slot = Self::empty();
        slot.n_energies = n_e as u32;
        slot.energy_grid.reserve(n_e);
        slot.u_grid.reserve(n_e);

        if keep_all {
            slot.n_components = n_comp_raw as u32;
            // Tight component-major theta: component `c`'s row of `n_e` points
            // starts at `c * n_e`.
            slot.theta = vec![0.0; n_comp_raw * n_e];
            for (i, &e_in) in grid.iter().enumerate() {
                slot.energy_grid.push(e_in);
                // `u` is shared across components in the targeted data; take
                // it from the first (any) component.
                slot.u_grid.push(evaps[0].1);
                for (c, (Tabulated1D::Tabulated1D { x: tx, y: ty, .. }, _, _)) in
                    evaps.iter().enumerate()
                {
                    slot.theta[c * n_e + i] = eval_tab1d(tx, ty, e_in);
                }
            }
        } else {
            // Single effective curve: applicability-argmax per grid point
            // (bit-equivalent to the CPU when applicability is a 0/1 step,
            // the overwhelmingly common ENDF case).
            slot.n_components = 1;
            slot.theta = vec![0.0; n_e];
            for (i, &e_in) in grid.iter().enumerate() {
                let mut best = 0usize;
                let mut best_w = f64::NEG_INFINITY;
                for (j, (_, _, applic)) in evaps.iter().enumerate() {
                    let w = applic.map(|a| a.evaluate(e_in)).unwrap_or(1.0);
                    if w > best_w {
                        best_w = w;
                        best = j;
                    }
                }
                let (Tabulated1D::Tabulated1D { x: tx, y: ty, .. }, u, _) = evaps[best];
                slot.energy_grid.push(e_in);
                slot.theta[i] = eval_tab1d(tx, ty, e_in);
                slot.u_grid.push(u);
            }
        }
        slot
    }
}

/// True when every Evaporation component carries EQUAL applicability at each
/// shared-grid incident energy where the reaction is active (the
/// fractional-mixture case the multi-component path handles exactly). Grid
/// points where all applicabilities are ~0 (below the channel threshold, where
/// the partial cross section vanishes and no neutron is emitted) are ignored.
fn applicabilities_equal(
    evaps: &[(&Tabulated1D, f64, Option<&Tabulated1D>)],
    grid: &[f64],
) -> bool {
    // A missing applicability means an implicit weight of 1; a mix of
    // present/absent is not a clean equal-weight mixture.
    if !evaps.iter().all(|(_, _, a)| a.is_some()) {
        return false;
    }
    for &e_in in grid {
        let ws: Vec<f64> = evaps
            .iter()
            .map(|(_, _, a)| a.map(|a| a.evaluate(e_in)).unwrap_or(1.0))
            .collect();
        let sum: f64 = ws.iter().sum();
        if sum <= 1e-9 {
            continue; // inactive (below threshold): partial xs ~ 0 here.
        }
        let w0 = ws[0];
        // Relative tolerance against the first weight. ENDF files store
        // equal weights as DECIMAL-ROUNDED 1/N (Ta182 MT17 carries
        // 0.33333 / 0.33333 / 0.33334), so the spread of an intended-equal
        // mixture can reach ~1e-4 relative; a step-gated (0/1) mixture
        // differs by the full weight. 1e-3 separates the two cleanly.
        if ws.iter().any(|w| (w - w0).abs() > 1e-3 * w0.max(1e-30)) {
            return false;
        }
    }
    true
}

/// True when every correlated component carries EQUAL applicability at each
/// incident energy where the reaction is active (the fractional-mixture case
/// the multi-component path handles exactly). Mirrors `applicabilities_equal`
/// for the `(correlated, applicability)` shape. Grid points where all
/// applicabilities are ~0 (below threshold, no neutron emitted) are ignored.
fn corr_applicabilities_equal(apps: &[Option<&Tabulated1D>], grid: &[f64]) -> bool {
    // A missing applicability means an implicit weight of 1; a mix of
    // present/absent is not a clean equal-weight mixture.
    if !apps.iter().all(|a| a.is_some()) {
        return false;
    }
    for &e_in in grid {
        let ws: Vec<f64> = apps
            .iter()
            .map(|a| a.map(|a| a.evaluate(e_in)).unwrap_or(1.0))
            .collect();
        let sum: f64 = ws.iter().sum();
        if sum <= 1e-9 {
            continue; // inactive (below threshold): partial xs ~ 0 here.
        }
        let w0 = ws[0];
        if ws.iter().any(|w| (w - w0).abs() > 1e-6 * w0.max(1e-30)) {
            return false;
        }
    }
    true
}

/// Linear interpolation of a tabulated `(x, y)` curve at `xq`, clamped to
/// the endpoints (matches `Tabulated1D::evaluate`'s flat extrapolation).
fn eval_tab1d(x: &[f64], y: &[f64], xq: f64) -> f64 {
    let n = x.len().min(y.len());
    if n == 0 {
        return 0.0;
    }
    if xq <= x[0] {
        return y[0];
    }
    if xq >= x[n - 1] {
        return y[n - 1];
    }
    let mut k = 0usize;
    while k + 1 < n {
        if xq >= x[k] && xq < x[k + 1] {
            let de = x[k + 1] - x[k];
            let f = if de > 0.0 { (xq - x[k]) / de } else { 0.0 };
            return y[k] + f * (y[k + 1] - y[k]);
        }
        k += 1;
    }
    y[n - 1]
}

/// Per-MT NBodyPhaseSpace slice. Extracted from a reaction's first
/// neutron product when the angle-energy distribution is
/// `NBodyPhaseSpace`. Empty otherwise.
pub struct NbpsSlot {
    pub n_bodies: u32,
    pub total_mass: f64,
}

impl NbpsSlot {
    pub fn empty() -> Self {
        Self {
            n_bodies: 0,
            total_mass: 0.0,
        }
    }

    pub fn from_reaction(reaction: &Reaction) -> Self {
        let Some(neutron_product) = first_neutron_product(reaction) else {
            return Self::empty();
        };
        for d in &neutron_product.distribution {
            if let AngleEnergyDistribution::NBodyPhaseSpace {
                n_bodies,
                total_mass,
                ..
            } = d
            {
                if *n_bodies >= 3 && *n_bodies <= 5 {
                    return Self {
                        n_bodies: *n_bodies as u32,
                        total_mass: *total_mass,
                    };
                }
                // Out-of-range body count: fall back to the
                // closed-form path. CPU panics on >5; the GPU
                // can't panic, so we silently no-op.
                return Self::empty();
            }
        }
        Self::empty()
    }
}

/// Per-MT Maxwell slice. Extracted from a reaction's first neutron
/// product when the energy distribution is `Maxwell` inside an
/// `UncorrelatedAngleEnergy`. Same shape as `EvapSlot` (tabulated
/// `θ(E_in)` plus restriction energy `u`); intentionally not
/// merged so a future nuclide carrying both Maxwell and Evaporation
/// across different MTs needs no special-casing on the extractor.
pub struct MaxwellSlot {
    pub n_energies: u32,
    pub energy_grid: Vec<f64>, // length n_energies (tight, issue #104)
    pub theta: Vec<f64>,       // length n_energies (tight, issue #104)
    pub u: f64,
}

impl MaxwellSlot {
    pub fn empty() -> Self {
        Self {
            n_energies: 0,
            energy_grid: Vec::new(),
            theta: Vec::new(),
            u: 0.0,
        }
    }

    pub fn from_reaction(reaction: &Reaction) -> Self {
        let Some(neutron_product) = first_neutron_product(reaction) else {
            return Self::empty();
        };
        for d in &neutron_product.distribution {
            if let AngleEnergyDistribution::UncorrelatedAngleEnergy { energy, .. } = d {
                let Some(energy_dist) = energy.as_ref() else {
                    continue;
                };
                if let EnergyDistribution::Maxwell { theta, u } = energy_dist {
                    let Tabulated1D::Tabulated1D { x, y, .. } = theta;
                    return Self::from_theta_u(x, y, *u);
                }
            }
        }
        Self::empty()
    }

    fn from_theta_u(theta_x: &[f64], theta_y: &[f64], u: f64) -> Self {
        if theta_x.is_empty() || theta_y.is_empty() {
            return Self::empty();
        }
        let n_in = theta_x.len().min(theta_y.len());
        // Tight, full-resolution layout (issue #104): keep every incident-energy
        // point (no per-axis stride-subsampling), so the GPU samples the
        // same θ(E_in) grid the CPU does. The per-slot CSR base is built at
        // concatenation time in `translate.rs` from the `n_energies` counts.
        let mut slot = Self::empty();
        slot.n_energies = n_in as u32;
        slot.u = u;
        slot.energy_grid.reserve(n_in);
        slot.theta.reserve(n_in);
        for i in 0..n_in {
            slot.energy_grid.push(theta_x[i]);
            slot.theta.push(theta_y[i]);
        }
        slot
    }
}

/// Per-MT inelastic Watt slice. Extracted from a reaction's first
/// neutron product when the energy distribution is `Watt` inside
/// an `UncorrelatedAngleEnergy`. Carries two tabulated parameters
/// (`a(E_in)` in eV, `b(E_in)` in 1/eV) on a shared incident-energy
/// grid, plus a restriction energy `u`.
pub struct WattSlot {
    pub n_energies: u32,
    pub energy_grid: Vec<f64>, // length n_energies (tight, issue #104)
    pub a: Vec<f64>,           // length n_energies (tight, issue #104)
    pub b: Vec<f64>,           // length n_energies (tight, issue #104)
    pub u: f64,
}

impl WattSlot {
    pub fn empty() -> Self {
        Self {
            n_energies: 0,
            energy_grid: Vec::new(),
            a: Vec::new(),
            b: Vec::new(),
            u: 0.0,
        }
    }

    pub fn from_reaction(reaction: &Reaction) -> Self {
        let Some(neutron_product) = first_neutron_product(reaction) else {
            return Self::empty();
        };
        for d in &neutron_product.distribution {
            if let AngleEnergyDistribution::UncorrelatedAngleEnergy { energy, .. } = d {
                let Some(energy_dist) = energy.as_ref() else {
                    continue;
                };
                if let EnergyDistribution::Watt { a, b, u } = energy_dist {
                    let Tabulated1D::Tabulated1D { x: a_x, y: a_y, .. } = a;
                    let Tabulated1D::Tabulated1D { x: b_x, y: b_y, .. } = b;
                    return Self::from_a_b_u(a_x, a_y, b_x, b_y, *u);
                }
            }
        }
        Self::empty()
    }

    /// Construct a slot from CPU-side `a(E_in)` and `b(E_in)` tabulations.
    /// The two parameters are assumed to share the same incident-energy
    /// breakpoints in source data -- we take `a`'s grid as the master and
    /// re-evaluate `b` onto it via linear interpolation if `b_x != a_x`.
    /// In practice ENDF File 5 Law 11 records always tabulate `a` and
    /// `b` on the same E_in axis, so the re-interpolation is a no-op
    /// path that exists purely for safety.
    fn from_a_b_u(a_x: &[f64], a_y: &[f64], b_x: &[f64], b_y: &[f64], u: f64) -> Self {
        if a_x.is_empty() || a_y.is_empty() || b_x.is_empty() || b_y.is_empty() {
            return Self::empty();
        }
        let n_a = a_x.len().min(a_y.len());
        // Tight, full-resolution layout (issue #104): keep every incident-energy
        // point on `a`'s master grid (no per-axis stride-subsampling), so
        // the GPU samples the same a(E_in) / b(E_in) the CPU does. The per-slot
        // CSR base (`watt_ae_offset`) is built at concatenation time in
        // `translate.rs`.
        let mut slot = Self::empty();
        slot.n_energies = n_a as u32;
        slot.u = u;
        slot.energy_grid.reserve(n_a);
        slot.a.reserve(n_a);
        slot.b.reserve(n_a);
        let same_grid = a_x == b_x && a_y.len() == b_y.len();
        for i in 0..n_a {
            let e_in = a_x[i];
            slot.energy_grid.push(e_in);
            slot.a.push(a_y[i]);
            slot.b.push(if same_grid {
                b_y[i]
            } else {
                // Linear interp `b` onto `a`'s grid.
                interp_lin(b_x, b_y, e_in)
            });
        }
        slot
    }
}

/// Linear interpolation helper. Clamps to endpoints; expects
/// `xs` strictly ascending (Tabulated1D guarantees this).
pub fn interp_lin(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let n = xs.len().min(ys.len());
    if n == 0 {
        return 0.0;
    }
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[n - 1] {
        return ys[n - 1];
    }
    for i in 0..n - 1 {
        if x >= xs[i] && x < xs[i + 1] {
            let dx = xs[i + 1] - xs[i];
            let t = if dx > 0.0 { (x - xs[i]) / dx } else { 0.0 };
            return ys[i] + t * (ys[i + 1] - ys[i]);
        }
    }
    ys[n - 1]
}

/// Trapezoidal CDF from a `(x, p)` pair, used when the data file's
/// `c` field wasn't populated. Mirrors yamc-nuclide's
/// `cumulative_from_pdf` shape: linear-segment integration with the
/// CDF starting at 0 and ending at the integral of `p` over `x`.
/// Caller renormalises if a `[0, 1]` range is required.
pub fn trapezoidal_cdf(x: &[f64], p: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    for i in 1..x.len() {
        let dx = x[i] - x[i - 1];
        out[i] = out[i - 1] + 0.5 * (p[i] + p[i - 1]) * dx;
    }
    out
}

pub fn first_neutron_product(reaction: &Reaction) -> Option<&ReactionProduct> {
    reaction
        .products
        .iter()
        .find(|p| p.particle == ParticleType::Neutron)
}
