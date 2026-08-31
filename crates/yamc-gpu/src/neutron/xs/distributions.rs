//! Flat per-MT / per-nuclide BUFFER BUILDERS (angle, eout, correlated,
//! Kalbach-Mann, evaporation, n-body, Maxwell, Watt, URR): the loops that
//! concatenate one nuclide's per-slot arrays across `MT_SLOTS` into the
//! tight buffers `translate.rs` then joins across materials. Building
//! blocks for the public `extract_*` functions in `super::extract`.
//!
//! The single-reaction layer those loops call (`EoutSlot` / `CorrSlot` /
//! `KalbachSlot` / `EvapSlot` / `NbpsSlot` / `MaxwellSlot` / `WattSlot` and
//! their numeric helpers) now lives in
//! [`yamc_physics::gpu::flat::eout_extract`] and is re-exported below, so the
//! CPU production transport and this GPU-side packing flatten each ENDF law
//! through ONE implementation (issue #111). What stays here is genuinely
//! GPU-buffer-layout work: slot ordering, concatenation, and the counts the
//! CSR base offsets are built from.

use super::*;
use yamc_nuclide::nuclide::Nuclide;
use yamc_nuclide::reaction::Reaction;
use yamc_nuclide::reaction_product::{AngleDistribution, ElasticAngleFlat, TabulatedInterp};

/// Single-source-of-truth per-law flattening (issue #111). Re-exported at the
/// old paths so `super::extract`, `super::photon_production`, and the tests
/// below are unchanged; the buffers they build are byte-for-byte what the
/// in-crate copies produced.
pub use yamc_physics::gpu::flat::eout_extract::{
    first_neutron_product, interp_lin, trapezoidal_cdf, CorrSlot, EoutSlot, EvapSlot, KalbachSlot,
    MaxwellSlot, NbpsSlot, WattSlot,
};
/// The per-reaction elastic/inelastic angular flatten, likewise owned by
/// yamc-physics (issue #111): the CPU transport's per-collision
/// `InelasticFlat` bundle and the `build_per_mt_angle_buffers` loop below
/// must produce the same table. Re-exported at its old path.
pub use yamc_physics::gpu::flat::inelastic_flat::elastic_flat_from_reaction;

/// Per-PHOTON-product angular distribution slice (slice S3). Tight,
/// variable-length layout (issue #104): the per-(E_in row) arrays carry
/// exactly `n_energies` incident-energy rows and the `(mu, cdf, pdf)` arrays
/// carry exactly `sum(n_mu)` mu points, back to back with no per-axis
/// padding or stride
/// subsampling, so the GPU samples the same data the CPU does. The per-product
/// (`pa_ae_offset`) / per-row (`pa_mu_offset`) CSR base offsets are built at
/// concatenation time from the `n_energies` / `n_mu` counts (mirrors the
/// elastic / eout / correlated families). Extracted from a PHOTON product's
/// `UncorrelatedAngleEnergy` angle (not a reaction's first neutron product).
/// The normalisation / CDF-fallback logic mirrors the neutron elastic /
/// inelastic angular flatten (`to_elastic_flat`) so the buffers feed
/// `invert_angle_cdf` byte-compatibly.
pub(super) struct PhotonAngleSlot {
    pub(super) n_energies: u32,
    pub(super) energy_grid: Vec<f64>, // length n_energies
    pub(super) n_mu: Vec<u32>,        // length n_energies
    pub(super) mu: Vec<f64>,          // length sum(n_mu)
    pub(super) cdf: Vec<f64>,         // length sum(n_mu)
    pub(super) pdf: Vec<f64>,         // length sum(n_mu)
    pub(super) interp: Vec<u32>,      // length n_energies
}

impl PhotonAngleSlot {
    /// Empty slot -- no rows, no points. With `n_energies == 0` the kernel
    /// falls back to isotropic `2 * draw_uniform - 1` (the CPU
    /// `AngleDistribution::sample` empty-distribution path). Tight layout
    /// (issue #104): every per-row / per-point `Vec` is empty.
    pub(super) fn empty() -> Self {
        Self {
            n_energies: 0,
            energy_grid: Vec::new(),
            n_mu: Vec::new(),
            mu: Vec::new(),
            cdf: Vec::new(),
            pdf: Vec::new(),
            interp: Vec::new(),
        }
    }

    /// Pack a photon product's `AngleDistribution`. Returns `empty()` for
    /// an empty angle (the kernel then samples isotropic mu). Mirrors the
    /// neutron angular flatten's per-(E_in slice) CDF-fallback and
    /// renormalisation exactly. Tight, full-resolution layout (issue #104):
    /// keeps every incident energy and every mu point (no per-axis
    /// stride-subsampling).
    pub(super) fn from_angle(angle: &AngleDistribution) -> Self {
        if angle.energy.is_empty() || angle.mu.is_empty() {
            return Self::empty();
        }

        let n_e = angle.energy.len();

        let mut slot = Self::empty();
        slot.n_energies = n_e as u32;
        slot.energy_grid.reserve(n_e);
        slot.n_mu.reserve(n_e);
        slot.interp.reserve(n_e);
        for (i, &e_in) in angle.energy.iter().enumerate() {
            slot.energy_grid.push(e_in);
            let tab = &angle.mu[i];
            let m_in = tab.x.len();
            if m_in == 0 {
                // Keep the per-row arrays aligned with energy_grid even when a
                // row carries no mu points (the kernel's single-point default
                // path handles n_mu < 2).
                slot.n_mu.push(0);
                slot.interp.push(ANGLE_INTERP_HISTOGRAM);
                continue;
            }
            slot.interp.push(match tab.interp {
                TabulatedInterp::Histogram => ANGLE_INTERP_HISTOGRAM,
                TabulatedInterp::LinLin => ANGLE_INTERP_LINLIN,
            });
            // Fall back to a trapezoidal CDF when the source only carried
            // a PDF (matches the neutron angular flatten).
            let cdf_owned: Vec<f64>;
            let cdf_slice: &[f64] = if tab.c.len() == m_in {
                &tab.c
            } else {
                cdf_owned = trapezoidal_cdf(&tab.x, &tab.p);
                &cdf_owned
            };
            slot.n_mu.push(m_in as u32);
            // Renormalise the CDF to end at 1.0 and scale the per-point
            // PDF by the same factor so `int p dx = c` stays consistent
            // for the kernel's quadratic LinLin inversion (matches the
            // neutron angular flatten).
            let cdf_max = cdf_slice.last().copied().unwrap_or(0.0);
            let pdf_scale = if cdf_max > 0.0 { 1.0 / cdf_max } else { 1.0 };
            for (j, &c) in cdf_slice.iter().enumerate().take(m_in) {
                slot.mu.push(tab.x[j]);
                slot.cdf.push(if cdf_max > 0.0 {
                    c / cdf_max
                } else {
                    j as f64 / (m_in - 1).max(1) as f64
                });
                slot.pdf
                    .push(tab.p.get(j).copied().unwrap_or(0.0) * pdf_scale);
            }
        }
        slot
    }
}

/// Build per-nuclide flat buffers `(eout_kind, eout_n_energies,
/// eout_energy_grid, eout_n_x, eout_x, eout_cdf)` across every MT
/// slot in `MT_INELASTIC_FIRST..=MT_INELASTIC_LAST`. Mirror of
/// `build_per_mt_angle_buffers` but for the outgoing-energy axis.
#[allow(clippy::type_complexity)]
pub(super) fn build_per_mt_eout_buffers<'a, F>(
    get_reaction: F,
) -> (
    Vec<u32>,
    Vec<u32>,
    Vec<u32>,
    Vec<f64>,
    Vec<u32>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<u32>,
    Vec<u32>,
)
where
    F: Fn(i32) -> Option<&'a Reaction>,
{
    // Tight variable-length layout (issue #104): per-slot scalars are
    // `[MT_INELASTIC_COUNT]`; the per-row / per-point arrays are concatenated
    // tight (no per-axis stride). The CSR base offsets are
    // built at concatenation time in `translate.rs` from the `n_energies` /
    // `n_x` counts (mirrors the elastic / per-MT-angle families).
    let mut eout_kind = Vec::with_capacity(MT_INELASTIC_COUNT);
    let mut eout_n_energies = Vec::with_capacity(MT_INELASTIC_COUNT);
    let mut eout_histogram_interp = Vec::with_capacity(MT_INELASTIC_COUNT);
    let mut eout_energy_grid = Vec::new();
    let mut eout_n_x = Vec::new();
    let mut eout_x = Vec::new();
    let mut eout_p = Vec::new();
    let mut eout_cdf = Vec::new();
    let mut eout_interp = Vec::new();
    let mut eout_n_discrete = Vec::new();

    for &mt in MT_SLOTS.iter() {
        let slot = match get_reaction(mt) {
            Some(rxn) => EoutSlot::from_reaction(rxn),
            None => EoutSlot::empty(),
        };
        eout_kind.push(slot.kind);
        eout_n_energies.push(slot.n_energies);
        eout_histogram_interp.push(slot.histogram_interp);
        eout_energy_grid.extend_from_slice(&slot.energy_grid);
        eout_n_x.extend_from_slice(&slot.n_x);
        eout_x.extend_from_slice(&slot.x);
        eout_p.extend_from_slice(&slot.p);
        eout_cdf.extend_from_slice(&slot.cdf);
        eout_interp.extend_from_slice(&slot.interp);
        eout_n_discrete.extend_from_slice(&slot.n_discrete);
    }

    (
        eout_kind,
        eout_n_energies,
        eout_histogram_interp,
        eout_energy_grid,
        eout_n_x,
        eout_x,
        eout_p,
        eout_cdf,
        eout_interp,
        eout_n_discrete,
    )
}

/// Build per-nuclide flat correlated buffers across every MT slot.
/// Empty for slots whose reaction isn't `CorrelatedAngleEnergy`.
#[allow(clippy::type_complexity)]
pub(super) fn build_per_mt_corr_buffers<'a, F>(
    get_reaction: F,
) -> (
    Vec<u32>,
    Vec<u32>,
    Vec<f64>,
    Vec<u32>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<u32>,
    Vec<u32>,
    Vec<u32>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<u32>,
)
where
    F: Fn(i32) -> Option<&'a Reaction>,
{
    // Tight, full-resolution per-MT correlated tables (issue #104): the data
    // arrays grow only with the populated slots' rows / x-points / mu-points,
    // so we cannot pre-size to a fixed cap. The per-slot / per-row / per-x CSR
    // offsets are built later (translate.rs) from the n_x / n_mu counts.
    let mut n_energies = Vec::with_capacity(MT_INELASTIC_COUNT);
    let mut n_components = Vec::with_capacity(MT_INELASTIC_COUNT);
    let mut energy_grid = Vec::new();
    let mut n_x = Vec::new();
    let mut x = Vec::new();
    let mut cdf = Vec::new();
    let mut p = Vec::new();
    let mut interp = Vec::new();
    let mut n_discrete = Vec::new();
    let mut n_mu = Vec::new();
    let mut mu = Vec::new();
    let mut mu_cdf = Vec::new();
    let mut mu_pdf = Vec::new();
    let mut mu_interp = Vec::new();

    for &mt in MT_SLOTS.iter() {
        let slot = match get_reaction(mt) {
            Some(rxn) => CorrSlot::from_reaction(rxn),
            None => CorrSlot::empty(),
        };
        n_energies.push(slot.n_energies);
        n_components.push(slot.n_components);
        energy_grid.extend_from_slice(&slot.energy_grid);
        n_x.extend_from_slice(&slot.n_x);
        x.extend_from_slice(&slot.x);
        cdf.extend_from_slice(&slot.cdf);
        p.extend_from_slice(&slot.p);
        interp.extend_from_slice(&slot.interp);
        n_discrete.extend_from_slice(&slot.n_discrete);
        n_mu.extend_from_slice(&slot.n_mu);
        mu.extend_from_slice(&slot.mu);
        mu_cdf.extend_from_slice(&slot.mu_cdf);
        mu_pdf.extend_from_slice(&slot.mu_pdf);
        mu_interp.extend_from_slice(&slot.mu_interp);
    }

    (
        n_energies,
        n_components,
        energy_grid,
        n_x,
        x,
        cdf,
        p,
        interp,
        n_discrete,
        n_mu,
        mu,
        mu_cdf,
        mu_pdf,
        mu_interp,
    )
}

/// Build per-nuclide flat NBodyPhaseSpace buffers across every MT slot.
/// Empty for slots whose reaction isn't `NBodyPhaseSpace`.
pub(super) fn build_per_mt_nbps_buffers<'a, F>(get_reaction: F) -> (Vec<u32>, Vec<f64>)
where
    F: Fn(i32) -> Option<&'a Reaction>,
{
    let mut n_bodies = Vec::with_capacity(MT_INELASTIC_COUNT);
    let mut total_mass = Vec::with_capacity(MT_INELASTIC_COUNT);
    for &mt in MT_SLOTS.iter() {
        let slot = match get_reaction(mt) {
            Some(rxn) => NbpsSlot::from_reaction(rxn),
            None => NbpsSlot::empty(),
        };
        n_bodies.push(slot.n_bodies);
        total_mass.push(slot.total_mass);
    }
    (n_bodies, total_mass)
}

/// Build per-nuclide flat Evaporation buffers across every MT slot.
/// Empty for slots whose reaction isn't `Evaporation`.
///
/// Tight variable-length layout (issue #104): `n_energies` / `n_components` are
/// per-MT-slot scalars `[MT_INELASTIC_COUNT]`; `energy_grid` / `u` carry exactly
/// `sum(n_energies)` rows and `theta` carries exactly `sum(n_components *
/// n_energies)` rows, back to back with no per-axis padding. The
/// per-(slab,MT) CSR bases (`evap_ae_offset` for energy_grid/u and
/// `evap_theta_offset` for the component-major theta) are built at concatenation
/// time in `translate.rs`. `MAX_EVAP_COMPONENTS` is retained as the component
/// loop bound, not a subsampled axis.
///
/// Returns `(n_energies, n_components, energy_grid, theta, u)`:
///   * `n_energies[slot]`  -- shared incident-energy point count
///   * `n_components[slot]` -- 0/1 single-curve, >= 2 multi-component mixture
///   * `energy_grid[evap_ae_offset[slot] + i]`           -- shared E_in grid
///   * `theta[evap_theta_offset[slot] + c * n_energies[slot] + i]`
///   * `u[evap_ae_offset[slot] + i]`                     -- shared restriction E
#[allow(clippy::type_complexity)]
pub(super) fn build_per_mt_evap_buffers<'a, F>(
    get_reaction: F,
) -> (Vec<u32>, Vec<u32>, Vec<f64>, Vec<f64>, Vec<f64>)
where
    F: Fn(i32) -> Option<&'a Reaction>,
{
    let mut n_energies = Vec::with_capacity(MT_INELASTIC_COUNT);
    let mut n_components = Vec::with_capacity(MT_INELASTIC_COUNT);
    let mut energy_grid = Vec::new();
    // Component-major theta, concatenated tight (issue #104).
    let mut theta = Vec::new();
    // Per incident-energy restriction energy, parallel to `energy_grid`.
    let mut u = Vec::new();
    for &mt in MT_SLOTS.iter() {
        let slot = match get_reaction(mt) {
            Some(rxn) => EvapSlot::from_reaction(rxn),
            None => EvapSlot::empty(),
        };
        n_energies.push(slot.n_energies);
        n_components.push(slot.n_components);
        energy_grid.extend_from_slice(&slot.energy_grid);
        theta.extend_from_slice(&slot.theta);
        u.extend_from_slice(&slot.u_grid);
    }
    (n_energies, n_components, energy_grid, theta, u)
}

/// Build per-nuclide flat Maxwell buffers across every MT slot.
/// Empty for slots whose reaction isn't `Maxwell`.
///
/// Tight variable-length layout (issue #104): `n_energies` / `u` are per-MT-slot
/// scalars `[MT_INELASTIC_COUNT]`; `energy_grid` / `theta` carry exactly
/// `sum(n_energies)` rows back to back with no per-axis padding. The
/// per-(slab,MT) CSR base offset (`maxwell_ae_offset`) is built at concatenation
/// time in `translate.rs` from the `n_energies` counts, mirroring the
/// elastic / per-MT-angle / eout families.
#[allow(clippy::type_complexity)]
pub(super) fn build_per_mt_maxwell_buffers<'a, F>(
    get_reaction: F,
) -> (Vec<u32>, Vec<f64>, Vec<f64>, Vec<f64>)
where
    F: Fn(i32) -> Option<&'a Reaction>,
{
    let mut n_energies = Vec::with_capacity(MT_INELASTIC_COUNT);
    let mut energy_grid = Vec::new();
    let mut theta = Vec::new();
    let mut u = Vec::with_capacity(MT_INELASTIC_COUNT);
    for &mt in MT_SLOTS.iter() {
        let slot = match get_reaction(mt) {
            Some(rxn) => MaxwellSlot::from_reaction(rxn),
            None => MaxwellSlot::empty(),
        };
        n_energies.push(slot.n_energies);
        energy_grid.extend_from_slice(&slot.energy_grid);
        theta.extend_from_slice(&slot.theta);
        u.push(slot.u);
    }
    (n_energies, energy_grid, theta, u)
}

/// Build per-nuclide flat Watt-inelastic buffers across every MT slot.
/// Empty for slots whose reaction isn't `Watt`.
///
/// Tight variable-length layout (issue #104): `n_energies` / `u` are per-MT-slot
/// scalars `[MT_INELASTIC_COUNT]`; `energy_grid` / `a` / `b` carry exactly
/// `sum(n_energies)` rows back to back with no per-axis padding. The
/// per-(slab,MT) CSR base (`watt_ae_offset`) is built at concatenation time in
/// `translate.rs`.
#[allow(clippy::type_complexity)]
pub(super) fn build_per_mt_watt_buffers<'a, F>(
    get_reaction: F,
) -> (Vec<u32>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)
where
    F: Fn(i32) -> Option<&'a Reaction>,
{
    let mut n_energies = Vec::with_capacity(MT_INELASTIC_COUNT);
    let mut energy_grid = Vec::new();
    let mut a_buf = Vec::new();
    let mut b_buf = Vec::new();
    let mut u = Vec::with_capacity(MT_INELASTIC_COUNT);
    for &mt in MT_SLOTS.iter() {
        let slot = match get_reaction(mt) {
            Some(rxn) => WattSlot::from_reaction(rxn),
            None => WattSlot::empty(),
        };
        n_energies.push(slot.n_energies);
        energy_grid.extend_from_slice(&slot.energy_grid);
        a_buf.extend_from_slice(&slot.a);
        b_buf.extend_from_slice(&slot.b);
        u.push(slot.u);
    }
    (n_energies, energy_grid, a_buf, b_buf, u)
}

/// Extract per-(material, nuclide) URR (unresolved resonance region) data
/// for a material (issue #210).
///
/// URR is intrinsically per-nuclide: the CPU applies an independent
/// probability-table band to EVERY in-range URR nuclide (issue #204). The
/// GPU mirrors that by emitting one slab row per nuclide, in the same
/// `weighted` order the per-nuclide inelastic / elastic / total pools use,
/// so the URR rows align 1:1 with the global slab index the kernel walks via
/// `mat_nuclide_meta`. Non-URR nuclides (or URR nuclides out of temperature
/// range) carry a `PRESENT = 0` meta row and a zero-length table; the kernel
/// skips them and keeps their smooth XS.
///
/// The smooth baselines the kernel takes the URR delta against come from
/// `nuc_partial_xs` (already density-weighted macroscopic, per slab), so no
/// per-nuclide smooth-micro buffer is packed here.
///
/// Returns `(urr_meta, urr_energy_grid, urr_cdf, urr_xs, urr_atom_density)`:
/// - `urr_meta` is `[n_nuclides × URR_META_COLS]` (`URR_META_ZA` carries the
///   per-nuclide stream key),
/// - `urr_energy_grid` / `urr_cdf` / `urr_xs` are the tight per-slab tables
///   concatenated back to back (the per-slab CSR bases are built downstream
///   in `translate.rs`),
/// - `urr_atom_density` is `[n_nuclides]` (0 for non-URR slabs).
#[allow(clippy::type_complexity)]
pub(super) fn build_urr_buffers(
    nuclides: &[(&Nuclide, f64)],
    temperature: &str,
) -> (Vec<u32>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = nuclides.len();
    let mut meta = Vec::with_capacity(n * URR_META_COLS);
    let mut energy: Vec<f64> = Vec::new();
    let mut cdf: Vec<f64> = Vec::new();
    let mut xs: Vec<f64> = Vec::new();
    let mut atom_density = Vec::with_capacity(n);

    for (nuc, density) in nuclides {
        // A nuclide contributes a URR slab only if it has URR data loaded for
        // this temperature (mirrors the CPU `urr_sample_for_nuclide` guards);
        // otherwise it gets a PRESENT = 0 row and a zero-length table.
        let urr = if nuc.urr_present && nuc.get_temp_idx(temperature).is_some() {
            nuc.urr_for_temp(temperature)
        } else {
            None
        };
        let Some(urr) = urr else {
            meta.extend(std::iter::repeat_n(0u32, URR_META_COLS));
            atom_density.push(0.0);
            continue;
        };

        // Tight CSR (issue #104): keep ALL energy points and ALL CDF bands.
        let n_urr_e = urr.energy.len();
        // CDF-band count from the first energy point (URR records carry a
        // constant n_cdf across energies).
        let n_urr_cdf = if urr.cdf_values.is_empty() {
            0
        } else {
            urr.cdf_values[0].len()
        };

        // A usable band needs at least two energy points (the kernel brackets
        // `energy` between grid points and reads the last point as
        // `grid[off + n_urr_e - 1]`) and at least one CDF band. A degenerate
        // record gets a PRESENT = 0 row so `PRESENT == 1` implies `n_urr_e >= 2`
        // in the kernel, keeping the endpoint read in bounds.
        if n_urr_e < 2 || n_urr_cdf == 0 {
            meta.extend(std::iter::repeat_n(0u32, URR_META_COLS));
            atom_density.push(0.0);
            continue;
        }

        let mut row = vec![0u32; URR_META_COLS];
        row[URR_META_PRESENT] = 1;
        row[URR_META_N_ENERGIES] = n_urr_e as u32;
        row[URR_META_N_CDF] = n_urr_cdf as u32;
        row[URR_META_INTERP] = match urr.interp {
            yamc_nuclide::urr::UrrInterpolation::LinLin => 0,
            yamc_nuclide::urr::UrrInterpolation::LogLog => 1,
        };
        // CPU `inelastic_flag > 0` means "include smooth inelastic in the URR
        // sum"; on GPU we collapse to a 0/1 boolean since the exact MT number
        // isn't needed (smooth inelastic comes from `nuc_partial_xs`).
        row[URR_META_INELASTIC_FLAG] = if urr.inelastic_flag > 0 { 1 } else { 0 };
        row[URR_META_ABSORPTION_FLAG] = urr.absorption_flag.max(0) as u32;
        row[URR_META_MULTIPLY_SMOOTH] = if urr.multiply_smooth { 1 } else { 0 };
        // Per-nuclide stream key (issue #204): each isotope draws an
        // independent probability-table band from the shared per-collision
        // base seed via `urr_nuclide_random(base, ZA)`.
        row[URR_META_ZA] = nuc.urr_stream_key();
        meta.extend_from_slice(&row);

        // Tight per-slab table appended to the concatenated buffers: energy is
        // `[n_urr_e]`, cdf `[n_urr_e × n_urr_cdf]`, xs `[n_urr_e × n_urr_cdf ×
        // URR_XS_COLS]`.
        for i in 0..n_urr_e {
            energy.push(urr.energy[i]);
            let cdf_row = &urr.cdf_values[i];
            let xs_row = &urr.xs_values[i];
            for j in 0..n_urr_cdf {
                cdf.push(cdf_row[j]);
                let xs_set = xs_row[j];
                // Column order matches URR_XS_TOTAL / ELASTIC / FISSION / NGAMMA.
                xs.push(xs_set.total);
                xs.push(xs_set.elastic);
                xs.push(xs_set.fission);
                xs.push(xs_set.n_gamma);
            }
        }

        atom_density.push(*density);
    }

    (meta, energy, cdf, xs, atom_density)
}

/// Build per-nuclide flat Kalbach-Mann buffers across every MT slot.
/// Empty for slots whose reaction isn't `KalbachMann`.
#[allow(clippy::type_complexity)]
pub(super) fn build_per_mt_km_buffers<'a, F>(
    get_reaction: F,
) -> (
    Vec<u32>,
    Vec<f64>,
    Vec<u32>,
    Vec<u32>,
    Vec<u32>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
)
where
    F: Fn(i32) -> Option<&'a Reaction>,
{
    // Tight variable-length layout (issue #104): per-slot scalars are
    // `[MT_INELASTIC_COUNT]`; the per-row / per-point arrays are concatenated
    // tight (no per-axis stride). The CSR base offsets are
    // built at concatenation time in `translate.rs` from the `n_energies` /
    // `n_x` counts (mirrors the eout / correlated families).
    let mut n_energies = Vec::with_capacity(MT_INELASTIC_COUNT);
    let mut energy_grid = Vec::new();
    let mut interp = Vec::new();
    let mut n_discrete = Vec::new();
    let mut n_x = Vec::new();
    let mut x = Vec::new();
    let mut p = Vec::new();
    let mut cdf = Vec::new();
    let mut r = Vec::new();
    let mut a = Vec::new();

    for &mt in MT_SLOTS.iter() {
        let slot = match get_reaction(mt) {
            Some(rxn) => KalbachSlot::from_reaction(rxn),
            None => KalbachSlot::empty(),
        };
        n_energies.push(slot.n_energies);
        energy_grid.extend_from_slice(&slot.energy_grid);
        interp.extend_from_slice(&slot.interp);
        n_discrete.extend_from_slice(&slot.n_discrete);
        n_x.extend_from_slice(&slot.n_x);
        x.extend_from_slice(&slot.x);
        p.extend_from_slice(&slot.p);
        cdf.extend_from_slice(&slot.cdf);
        r.extend_from_slice(&slot.r);
        a.extend_from_slice(&slot.a);
    }
    (
        n_energies,
        energy_grid,
        interp,
        n_discrete,
        n_x,
        x,
        p,
        cdf,
        r,
        a,
    )
}

/// Build per-nuclide flat buffers `(angle_n_energies, angle_energy_grid,
/// angle_n_mu, angle_mu, angle_cdf, angle_interp, scatter_in_cm)`
/// across every MT slot in `MT_INELASTIC_FIRST..=MT_INELASTIC_LAST`,
/// pulling each slot's data from its reaction (when present) or
/// padding with zeros (when not). Returns the seven vectors with the
/// shapes documented on `GpuNuclideXs`.
#[allow(clippy::type_complexity)]
pub(super) fn build_per_mt_angle_buffers<'a, F>(
    get_reaction: F,
) -> Result<
    (
        Vec<u32>,
        Vec<f64>,
        Vec<u32>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<u32>,
        Vec<u32>,
    ),
    NuclideXsError,
>
where
    F: Fn(i32) -> Option<&'a Reaction>,
{
    let mut angle_n_energies = Vec::with_capacity(MT_INELASTIC_COUNT);
    let mut angle_energy_grid = Vec::new();
    let mut angle_n_mu = Vec::new();
    let mut angle_mu = Vec::new();
    let mut angle_cdf = Vec::new();
    let mut angle_pdf = Vec::new();
    let mut angle_interp = Vec::new();
    let mut scatter_in_cm = Vec::with_capacity(MT_INELASTIC_COUNT);

    // Tight, full-resolution per-MT angular tables (issue #104): no per-axis
    // stride-subsampling. Built via the same `to_elastic_flat` the CPU
    // uses (`elastic_flat_from_reaction`); the CM/lab frame flag is a property
    // of the reaction, carried even when the slot has no angular data (so the
    // kernel still applies the CM->lab boost on empty-angle slots). The
    // per-slot / per-row CSR offsets are built at concatenation time from the
    // returned `n_energies` / `n_mu` counts, mirroring the elastic family.
    for &mt in MT_SLOTS.iter() {
        let (flat, cm) = match get_reaction(mt) {
            Some(rxn) => (
                elastic_flat_from_reaction(rxn),
                if rxn.scatter_in_cm { 1 } else { 0 },
            ),
            None => (ElasticAngleFlat::empty(), 0),
        };
        angle_n_energies.push(flat.energy_grid.len() as u32);
        angle_energy_grid.extend_from_slice(&flat.energy_grid);
        angle_n_mu.extend_from_slice(&flat.n_mu);
        angle_mu.extend_from_slice(&flat.mu);
        angle_cdf.extend_from_slice(&flat.cdf);
        angle_pdf.extend_from_slice(&flat.pdf);
        angle_interp.extend_from_slice(&flat.interp);
        scatter_in_cm.push(cm);
    }

    Ok((
        angle_n_energies,
        angle_energy_grid,
        angle_n_mu,
        angle_mu,
        angle_cdf,
        angle_pdf,
        angle_interp,
        scatter_in_cm,
    ))
}

// ---------------------------------------------------------------------------
// Correlated angle-energy multi-applicability parity (issue #111 Phase C)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod corr_parity {
    //! Statistical parity of the SHARED flat correlated angle-energy sampler
    //! (`CorrSlot::from_reaction` -> `sample_correlated_angle_energy`, used by
    //! the GPU kernel + its CPU twin) against the OpenMC-validated legacy
    //! `ReactionProduct::sample` (which draws an applicability uniform and
    //! picks among the product's distributions when `n_dist > 1`).
    //!
    //! Issue #111 localized a +9% (n,2n) outgoing-energy bias to the flat
    //! extractor collapsing a multi-applicability product (F19 MT16 has TWO
    //! equal-weight correlated distributions) down to `distribution[0]`. This
    //! test reproduces that bias and guards the fix: at 14.06 MeV the flat
    //! sampler's mean/std outgoing energy must match the legacy product
    //! sampler for every correlated MT.
    //!
    //!   cargo test -p yamc-gpu --release \
    //!       --lib neutron::xs::distributions::corr_parity -- --nocapture
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};
    use yamc_physics::gpu::flat::correlated_angle_energy::sample_correlated_angle_energy;
    use yamc_rng::next_xi;

    /// Compute the slot-local CSR offsets (prefix sums of `n_x` / `n_mu`)
    /// exactly as `translate.rs` does at concatenation time.
    fn slot_slices(slot: &CorrSlot) -> (Vec<u32>, Vec<u32>) {
        let mut x_offset = Vec::with_capacity(slot.n_x.len());
        let mut acc = 0u32;
        for &nx in &slot.n_x {
            x_offset.push(acc);
            acc += nx;
        }
        let mut mu_offset = Vec::with_capacity(slot.n_mu.len());
        let mut acc_mu = 0u32;
        for &nm in &slot.n_mu {
            mu_offset.push(acc_mu);
            acc_mu += nm;
        }
        (x_offset, mu_offset)
    }

    /// Sample the flat correlated slot exactly as the dispatch/kernel do:
    /// pick a component (one uniform when `n_components >= 2`), then sample
    /// the leaf correlated sampler on that component's row window.
    fn sample_corr_slot(
        slot: &CorrSlot,
        x_offset: &[u32],
        mu_offset: &[u32],
        e_in: f64,
        state: &mut u64,
    ) -> Option<(f64, Option<f64>)> {
        let nc = slot.n_components.max(1) as usize;
        let total = slot.energy_grid.len();
        let n_per = total / nc;
        let comp = if nc >= 2 {
            let xi = next_xi(state);
            ((xi * nc as f64) as usize).min(nc - 1)
        } else {
            0
        };
        let base = comp * n_per;
        sample_correlated_angle_energy(
            e_in,
            &slot.energy_grid[base..base + n_per],
            &slot.n_x[base..base + n_per],
            &slot.interp[base..base + n_per],
            &slot.n_discrete[base..base + n_per],
            &slot.x,
            &slot.p,
            &slot.cdf,
            &slot.n_mu,
            &slot.mu_interp,
            &slot.mu,
            &slot.mu_pdf,
            &slot.mu_cdf,
            &x_offset[base..base + n_per],
            mu_offset,
            state,
        )
    }

    fn mean_std(sum: f64, sum_sq: f64, n: f64) -> (f64, f64) {
        let mean = sum / n;
        let var = (sum_sq / n - mean * mean).max(0.0);
        (mean, var.sqrt())
    }

    /// Returns `None` when the F19 cache is absent (CI without data).
    fn f19() -> Option<Nuclide> {
        let path = yamc_test_cache::nuclide("F19")?;
        yamc_nuclide::nuclide::load_nuclide(&path, &yamc_nuclide::LoadScope::full()).ok()
    }

    #[test]
    fn f19_correlated_sampler_matches_legacy_product() {
        let Some(nuc) = f19() else {
            eprintln!("skip f19_correlated_sampler_matches_legacy_product -- no F19 cache");
            return;
        };
        let tidx = nuc.get_temp_idx("294").unwrap_or(0);
        let e_in = 14.06e6_f64;
        let n_samples = 2_000_000usize;

        // Every F19 MT whose neutron product carries a correlated law.
        let mts = [16, 22, 28, 91];
        eprintln!(
            "\n==== F19 correlated flat-vs-legacy @ {e_in:.4e} eV, N={n_samples} ====\n  {:<5} {:>6} {:>13} {:>13} {:>9} {:>13} {:>13}",
            "MT", "n_dist", "legacy_mean", "flat_mean", "mean_gap", "legacy_std", "flat_std"
        );

        let mut failures: Vec<String> = Vec::new();
        for mt in mts {
            let Some(rxn) = nuc.reactions[tidx].get(&mt) else {
                continue;
            };
            let Some(product) = first_neutron_product(rxn) else {
                continue;
            };
            let n_dist = product.distribution.len();

            // Legacy reference: the OpenMC-validated product sampler.
            let mut rng = StdRng::seed_from_u64(0xF19_0000 + mt as u64);
            let (mut lsum, mut lsq) = (0.0f64, 0.0f64);
            for _ in 0..n_samples {
                let (e_out, _mu) = product.sample(e_in, &mut rng);
                lsum += e_out;
                lsq += e_out * e_out;
            }
            let (lmean, lstd) = mean_std(lsum, lsq, n_samples as f64);

            // Flat path: the REAL extractor + shared sampler.
            let slot = CorrSlot::from_reaction(rxn);
            let (x_offset, mu_offset) = slot_slices(&slot);
            let mut state: u64 =
                crate::common::rng::expand_seed(0x9E37_79B9u32.wrapping_add(mt as u32));
            let (mut fsum, mut fsq, mut fcnt) = (0.0f64, 0.0f64, 0usize);
            for _ in 0..n_samples {
                if let Some((e_out, _mu)) =
                    sample_corr_slot(&slot, &x_offset, &mu_offset, e_in, &mut state)
                {
                    fsum += e_out;
                    fsq += e_out * e_out;
                    fcnt += 1;
                }
            }
            let (fmean, fstd) = mean_std(fsum, fsq, fcnt.max(1) as f64);

            let gap = if lmean != 0.0 {
                (fmean - lmean) / lmean * 100.0
            } else {
                f64::NAN
            };
            eprintln!(
                "  {mt:<5} {n_dist:>6} {lmean:>13.5e} {fmean:>13.5e} {gap:>+8.3}% {lstd:>13.5e} {fstd:>13.5e}"
            );
            // Tolerance: at N=2e6 the per-mean Monte-Carlo noise is ~0.1%; a
            // real spectrum shift (the +9% MT16 bias) is far above it.
            if gap.abs() > 0.5 {
                failures.push(format!("MT{mt} mean gap {gap:+.3}% (n_dist={n_dist})"));
            }
        }

        assert!(
            failures.is_empty(),
            "flat correlated sampler diverges from legacy product sampler:\n  {}",
            failures.join("\n  ")
        );
    }
}

#[cfg(test)]
mod evap_parity {
    //! `EvapSlot` multi-law collapse correctness (the Ar38 MT91 V&V bug).
    //!
    //! endf-b8.1 carries two shapes of multi-Evaporation neutron product:
    //!   * 0/1 step-gated windows (Ar36 / Ar38 / Na22 / Na23 / Co58_m1 MT91):
    //!     a high-`u` law below the window edge, a low-`u` law above. The
    //!     collapse must switch `u` exactly at the edge, which requires the
    //!     edge to be a shared-grid point even when the theta tables skip it
    //!     (Ar38's theta grid is [6.06, 7, 8, 20] MeV; the edge is 11 MeV).
    //!   * decimal-rounded 1/N mixtures (Ta182 MT17: 0.33333/0.33333/0.33334),
    //!     which must be kept as components, not collapsed to the argmax.
    use super::*;
    use yamc_nuclide::reaction_product::Tabulated1D;

    fn step(x: Vec<f64>, y: Vec<f64>) -> Tabulated1D {
        Tabulated1D::Tabulated1D {
            x,
            y,
            breakpoints: vec![3],
            interpolation: vec![1],
        }
    }

    fn linlin(x: Vec<f64>, y: Vec<f64>) -> Tabulated1D {
        let n = x.len() as i32;
        Tabulated1D::Tabulated1D {
            x,
            y,
            breakpoints: vec![n],
            interpolation: vec![2],
        }
    }

    /// Ar38 MT91 shape: theta breakpoints skip the 11 MeV applicability
    /// edge; the collapsed `u` grid must still switch components there.
    #[test]
    fn step_gated_u_switches_at_applicability_edge() {
        let theta = linlin(
            vec![6.056_76e6, 7.0e6, 8.0e6, 20.0e6],
            vec![1.041_35e6, 1.041_35e6, 1.066_2e6, 1.755_32e6],
        );
        let low = step(vec![6.056_76e6, 11.0e6, 20.0e6], vec![1.0, 0.0, 0.0]);
        let high = step(vec![6.056_76e6, 11.0e6, 20.0e6], vec![0.0, 1.0, 1.0]);
        let evaps = vec![(&theta, 5.9e6, Some(&low)), (&theta, 2.2e6, Some(&high))];
        let slot = EvapSlot::from_evaps(&evaps);

        assert_eq!(slot.n_components, 1, "step-gated laws must collapse");
        assert!(
            slot.energy_grid.contains(&11.0e6),
            "shared grid must include the applicability edge: {:?}",
            slot.energy_grid
        );
        for (&e, &u) in slot.energy_grid.iter().zip(&slot.u_grid) {
            let want = if e < 11.0e6 { 5.9e6 } else { 2.2e6 };
            assert_eq!(
                u, want,
                "u at E_in={e:.3e} must come from the applicable law \
                 (grid {:?}, u {:?})",
                slot.energy_grid, slot.u_grid
            );
        }
    }

    /// Ta182 MT17 shape: ENDF stores the equal weights decimal-rounded, so
    /// the equality check must tolerate ~1e-4 relative spread and keep all
    /// components for the per-collision uniform pick.
    #[test]
    fn rounded_equal_weights_keep_all_components() {
        let t0 = linlin(vec![13.8e6, 17.0e6, 20.0e6], vec![0.3e6, 0.5e6, 0.6e6]);
        let t1 = linlin(vec![13.8e6, 15.0e6, 20.0e6], vec![0.3e6, 0.4e6, 0.6e6]);
        let t2 = linlin(vec![13.8e6, 16.0e6, 20.0e6], vec![0.3e6, 0.45e6, 0.6e6]);
        let w0 = step(vec![13.8e6, 20.0e6], vec![0.33333, 0.33333]);
        let w1 = step(vec![13.8e6, 20.0e6], vec![0.33333, 0.33333]);
        let w2 = step(vec![13.8e6, 20.0e6], vec![0.33334, 0.33334]);
        let evaps = vec![
            (&t0, 13.8e6, Some(&w0)),
            (&t1, 13.8e6, Some(&w1)),
            (&t2, 13.8e6, Some(&w2)),
        ];
        let slot = EvapSlot::from_evaps(&evaps);
        assert_eq!(
            slot.n_components, 3,
            "decimal-rounded 1/N weights are an equal mixture"
        );
    }

    /// The real Ar38 data: `from_reaction(MT91)` must produce u = 2.2 MeV
    /// for every grid point at/above the 11 MeV window edge (the V&V run
    /// samples E_in = 14.06 MeV, which previously read u = 5.9 MeV and
    /// truncated the outgoing spectrum at 8.16 MeV). Skips without the
    /// endf-b8.1 Ar38 cache.
    ///
    /// Ar38 is not in `scripts/fetch_test_fixtures.py`, so this runs only on a
    /// machine that fetched it for something else, and never in CI.
    #[test]
    fn ar38_mt91_u_grid_matches_windows() {
        let path = yamc_test_cache::nuclide_path("Ar38");
        if !std::path::Path::new(&path).exists() {
            eprintln!("skip ar38_mt91_u_grid_matches_windows -- no Ar38 cache");
            return;
        }
        let Ok(nuc) = yamc_nuclide::nuclide::load_nuclide(&path, &yamc_nuclide::LoadScope::full())
        else {
            eprintln!("skip ar38_mt91_u_grid_matches_windows -- Ar38 cache unreadable");
            return;
        };
        // Loading is not the same as carrying what this reads. Since #389 a
        // cache dir is routinely populated at activation scope, holding cross
        // sections and none of the secondary distributions, and a `full`
        // request over one of those is NARROWED rather than refused: the load
        // succeeds, MT 91 is present, and it has no distribution to extract.
        // Asserting on that reports absent data as a physics failure, which is
        // the shape of issues #531 and #542.
        if nuc.load_scope.sections != yamc_nuclide::load_scope::SectionScope::Full {
            eprintln!(
                "skip ar38_mt91_u_grid_matches_windows -- Ar38 is cached at \
                 activation scope, with no secondary distributions"
            );
            return;
        }
        let tidx = nuc.get_temp_idx("294").unwrap_or(0);
        let Some(rxn) = nuc.reactions[tidx].get(&91) else {
            eprintln!("skip ar38_mt91_u_grid_matches_windows -- no MT91");
            return;
        };
        let slot = EvapSlot::from_reaction(rxn);
        assert!(slot.n_energies > 0, "Ar38 MT91 must extract an EvapSlot");
        assert!(
            slot.energy_grid.contains(&11.0e6),
            "grid must include the 11 MeV window edge: {:?}",
            slot.energy_grid
        );
        for (&e, &u) in slot.energy_grid.iter().zip(&slot.u_grid) {
            let want = if e < 11.0e6 { 5.9e6 } else { 2.2e6 };
            assert_eq!(u, want, "u at E_in={e:.3e}");
        }
    }
}
