//! Host-side dispatcher for inelastic-collision angle / E_out
//! sampling. Bit-identical to the cubecl kernel's per-MT branch
//! structure: slice-B angular fallback, then dispatch on the
//! per-slot `eout_kind` to slice-C (continuous tabular), slice-D
//! (correlated angle-energy), slice-E (Kalbach-Mann), slice-F
//! (Evaporation, n-body phase space, Maxwell, Watt), or slice-G
//! (tabulated equiprobable). Each branch delegates the physics to
//! the sibling `flat` samplers; this function only owns the layout-
//! aware data extraction and the dispatch.
//!
//! Moved here from yamc-gpu's `neutron::transport::dispatch` (issue
//! #111, stream unification) so the CPU production transport, which
//! builds without the `gpu` feature, and yamc-gpu's host-side twin
//! call one single-source-of-truth dispatcher. yamc-gpu re-exports
//! this module's constants from its `neutron::xs` layout module and
//! compile-time asserts `MT_INELASTIC_COUNT == MT_SLOTS.len()`.

use yamc_nuclide::nuclide::INELASTIC_MT_SLOTS;

/// Number of MT slots reserved per material in the GPU's flat per-MT
/// buffers. Materials whose nuclides don't have a particular MT have
/// zero xs in that slot -- the kernel just never selects it.
///
/// Derived from yamc-nuclide's [`INELASTIC_MT_SLOTS`], the single
/// definition of the slot order (issue #111): the CPU's per-collision
/// reaction walk and the GPU's per-MT slot sweep are the same sequence,
/// so the two must not be able to drift apart. The dispatcher below
/// bakes this count into its `slab * MT_INELASTIC_COUNT + slot`
/// addressing; yamc-gpu re-exports both and asserts they agree.
pub const MT_INELASTIC_COUNT: usize = INELASTIC_MT_SLOTS.len();

/// Outgoing-energy distribution kinds. Matches the variants of yamc-
/// nuclide's `EnergyDistribution` that the kernel currently supports;
/// any other variant (Maxwell, Watt, Evaporation, DiscretePhoton)
/// falls back to `EOUT_KIND_LEVEL_INELASTIC` so the kernel keeps
/// using its closed-form Q-value energy formula.
///
/// `LevelInelastic` is the original GPU path: the kernel computes
/// the outgoing energy from `E_cm = (A/(A+1))² · (E − (A+1)/A · |Q|)`
/// and treats the result as either the CM-frame or lab-frame
/// outgoing energy depending on `scatter_in_cm`.
///
/// `ContinuousTabular` is the slice-C upgrade: the kernel samples
/// `E_out` from a per-MT tabulated CDF (`eout_x`, `eout_cdf`) plus
/// stochastic bracket interpolation between adjacent incident-energy
/// slices. The kernel applies the same `scatter_in_cm` conversion to
/// the sampled `E_out` so frame semantics stay consistent.
pub const EOUT_KIND_LEVEL_INELASTIC: u32 = 0;
pub const EOUT_KIND_CONTINUOUS_TABULAR: u32 = 1;
/// Kalbach-Mann correlated angle-energy distribution. The slot's
/// (E_in, E_out) tabulated CDF lives in the parallel `km_*` buffers
/// alongside the per-(E_in, E_out) `r` and `a` Kalbach-Mann
/// parameters used to sample mu.
pub const EOUT_KIND_KALBACH_MANN: u32 = 3;
/// Evaporation E_out distribution: `p(E_out) ~ E_out · exp(-E_out/θ)`
/// for `0 < E_out < E_in - u`. The slot's tabulated `θ(E_in)` lives
/// in the parallel `evap_*` buffers alongside the restriction
/// energy `u`. Used by `UncorrelatedAngleEnergy/Evaporation`
/// (angular comes from the slice-B per-MT table) and the top-level
/// `Evaporation` AngleEnergy variant (mu is isotropic -- covered by
/// `n_ae == 0` in the slice-B table). C12 MT 91 / MT 28 are the
/// canonical examples.
pub const EOUT_KIND_EVAPORATION: u32 = 4;
/// N-body phase-space distribution (ENDF File 6, Law 6). The slot's
/// `n_bodies` (3 / 4 / 5) and `total_mass` (sum of product AWRs)
/// live in the parallel `nbps_*` buffers; `q_value` comes from the
/// existing `q_inelastic_per_mt` buffer (signed) and `awr` from
/// `target_mass_per_material`. Mu is isotropic in CM. Used by H2
/// MT 16 and some other light-element multi-neutron-out
/// evaluations.
pub const EOUT_KIND_NBODY_PHASE_SPACE: u32 = 5;
/// Maxwell fission spectrum (ENDF File 5, Law 7).
/// `p(E_out) ~ sqrt(E_out) · exp(-E_out / θ(E_in))` for
/// `0 < E_out < E_in - u`. Same parameter shape as Evaporation
/// (tabulated `θ(E_in)` plus a restriction energy `u`) but a
/// different rejection sampler -- kernel branches on `eout_kind`.
/// The slot's tabulated parameter and restriction energy live in
/// the parallel `maxwell_*` buffers (separate from `evap_*` even
/// though the shapes match: keeping the buffers distinct means
/// extraction stays one-distribution-per-slot, and a future
/// nuclide that carries both Maxwell and Evaporation on different
/// MTs needs no special-casing).
///
/// Doesn't appear on inelastic MTs in our ENDF/B-VIII.0 19-nuclide
/// survey (only fission-MT use, which takes a different dispatch
/// path on GPU); implemented here for TENDL / older-library
/// completeness and so we have one less "fall back to closed form"
/// row in the CPU-vs-GPU gap table.
pub const EOUT_KIND_MAXWELL: u32 = 6;
/// Watt fission spectrum on inelastic MTs (ENDF File 5, Law 11).
/// `p(E_out) ~ exp(-E_out / a(E_in)) · sinh(sqrt(b(E_in) · E_out))`
/// for `0 < E_out < E_in - u`. Distinct from the fission-MT Watt
/// path: that one lives on its own dispatch and uses
/// `fission_a_per_material` / `fission_b_per_material`. This kind
/// covers Watt-shaped inelastic continua, which carry their own
/// per-MT tabulated `a(E_in)` and `b(E_in)` plus restriction `u`,
/// so the slot data lives in the parallel `watt_*` buffers.
///
/// Sampler: draw `w` from a Maxwell with parameter `a` (3 RNG
/// draws), draw `r4` uniform on [0,1] for the correction, build
/// `E = w + a²b/4 + (2 r4 - 1) · sqrt(a²b · w)`, retry until
/// `E ≤ E_in - u`. Bit-matches `sample_watt_spectrum_params` in
/// `yamc-nuclide::sampling`.
///
/// Like Maxwell on inelastic MTs, doesn't appear in our ENDF/B-VIII.0
/// 19-nuclide survey but exists in the CPU code for TENDL / older
/// libraries.
pub const EOUT_KIND_WATT: u32 = 7;
/// Tabulated equiprobable E_out (older ENDF format predating
/// File 5 Law 1's continuous-tabular). The slot carries an
/// incident-energy grid plus, for each incident energy, a list
/// of equiprobable outgoing-energy bins. The sampler picks the
/// incident-energy bracket (nearest-lower index, no
/// interpolation -- matches CPU's `find_energy_index`), then
/// picks one of the `n_x[i]` outgoing-energy bins uniformly:
/// `idx = floor(xi · n_x[i])`, `E_out = x[idx]`. One RNG draw
/// per sample, no rejection.
///
/// Buffer layout reuses the `eout_*` arrays already populated
/// for `ContinuousTabular` -- the data shape is identical
/// (incident energy axis, per-energy out-bin count, flat
/// out-energy table), and the `eout_cdf` slot is unused for
/// this kind (zero-padded by the extractor). Saves three
/// storage-buffer descriptor bindings versus a dedicated
/// `tab_*` family of buffers.
///
/// Older ENDF format, rare in B-VIII; doesn't appear in the
/// 19-nuclide ENDF/B-VIII.0 test set but exists in legacy
/// libraries and some TENDL evaluations.
pub const EOUT_KIND_TABULATED: u32 = 8;
/// `CorrelatedAngleEnergy` (slice D): the kernel samples `E_out`
/// from a per-MT CDF the same way `ContinuousTabular` does, then
/// samples `mu` from a per-`(E_in, E_out)` angular CDF (the
/// `corr_mu_*` buffers) instead of the slice-B per-MT angular
/// table. Covers Fe56 MT 91 (continuum inelastic) and other
/// nuclides that store correlated continuum data.
pub const EOUT_KIND_CORRELATED: u32 = 2;

/// Sample the outgoing angle and energy for an inelastic collision on
/// the flat per-MT buffer layout: slice-B tabulated CM angular sample
/// (with `xi3` isotropic fallback), then the per-slot `eout_kind`
/// dispatch, then the optional CM-to-lab conversion. Shared by the CPU
/// production transport and yamc-gpu's host-side twin (issue #111).
///
/// Returns `(mu, e_out, ok)`. `ok == false` means the sampled CM
/// energy was non-positive or the CM-to-lab conversion failed; `e_out`
/// then falls back to `e_in`.
#[allow(clippy::too_many_arguments)]
pub fn sample_inelastic_kinematics(
    e_in: f64,
    target_mass: f64,
    e_cm_closed_form: f64,
    xi3_isotropic_fallback: f64,
    slab: usize,
    selected_slot: usize,
    state: &mut u64,
    angle_n_energies: &[u32],
    angle_ae_offset: &[u32],
    angle_energy_grid: &[f64],
    angle_n_mu: &[u32],
    angle_mu_offset: &[u32],
    angle_mu: &[f64],
    angle_cdf: &[f64],
    angle_pdf: &[f64],
    angle_interp: &[u32],
    eout_kind: &[u32],
    eout_n_energies: &[u32],
    eout_ae_offset: &[u32],
    eout_energy_grid: &[f64],
    eout_n_x: &[u32],
    eout_x_offset: &[u32],
    eout_x: &[f64],
    eout_cdf: &[f64],
    eout_histogram_interp: &[u32],
    eout_p: &[f64],
    eout_interp: &[u32],
    eout_n_discrete: &[u32],
    corr_n_energies: &[u32],
    corr_n_components: &[u32],
    corr_ae_offset: &[u32],
    corr_energy_grid: &[f64],
    corr_n_x: &[u32],
    corr_x_offset: &[u32],
    corr_x: &[f64],
    corr_cdf: &[f64],
    corr_p: &[f64],
    corr_interp: &[u32],
    corr_n_discrete: &[u32],
    corr_n_mu: &[u32],
    corr_mu_offset: &[u32],
    corr_mu: &[f64],
    corr_mu_cdf: &[f64],
    corr_mu_pdf: &[f64],
    corr_mu_interp: &[u32],
    scatter_in_cm_per_mt: &[u32],
    km_n_energies: &[u32],
    km_ae_offset: &[u32],
    km_energy_grid: &[f64],
    km_interp: &[u32],
    km_n_discrete: &[u32],
    km_n_x: &[u32],
    km_x_offset: &[u32],
    km_x: &[f64],
    km_p: &[f64],
    km_c: &[f64],
    km_r: &[f64],
    km_a: &[f64],
    evap_n_energies: &[u32],
    evap_n_components: &[u32],
    evap_ae_offset: &[u32],
    evap_theta_offset: &[u32],
    evap_energy_grid: &[f64],
    evap_theta: &[f64],
    evap_u: &[f64],
    nbps_n_bodies: &[u32],
    nbps_total_mass: &[f64],
    maxwell_n_energies: &[u32],
    maxwell_ae_offset: &[u32],
    maxwell_energy_grid: &[f64],
    maxwell_theta: &[f64],
    maxwell_u: &[f64],
    watt_n_energies: &[u32],
    watt_ae_offset: &[u32],
    watt_energy_grid: &[f64],
    watt_a: &[f64],
    watt_b: &[f64],
    watt_u: &[f64],
    q_value: f64,
) -> (f64, f64, bool) {
    let mat_slot = slab * MT_INELASTIC_COUNT + selected_slot;
    let n_ae = angle_n_energies[mat_slot] as usize;

    // Slice B: tabulated CM-frame angular sample. Identical shape to
    // the elastic-angle table (per-slot energy grid → per-bracket
    // (μ, CDF, PDF) table with linlin/histogram interp) and uses the
    // same `xi3` isotropic fallback when the slot is empty; delegate
    // to `yamc-physics::flat::elastic_mu_cm`.
    // Tight CSR layout (issue #104): the slot's incident-energy rows start at
    // this global base; (mu, cdf, pdf) are read from the full arrays via the
    // per-row global `mu_offset`. No per-axis stride.
    let eg_off = angle_ae_offset[mat_slot] as usize;
    let mut mu_sampled = crate::gpu::flat::elastic_mu_cm::sample_elastic_mu_cm(
        e_in,
        xi3_isotropic_fallback,
        &angle_energy_grid[eg_off..eg_off + n_ae],
        &angle_n_mu[eg_off..eg_off + n_ae],
        &angle_interp[eg_off..eg_off + n_ae],
        angle_mu,
        angle_cdf,
        angle_pdf,
        &angle_mu_offset[eg_off..eg_off + n_ae],
        state,
    );

    // Slice C: override the closed-form `e_cm` with a sampled value
    // when continuum tabular data is present. Mirrors the GPU
    // kernel's `eout_kind == 1 && n_eout > 0` branch in the same RNG
    // order (xi_eeb followed by xi_x, both only drawn when their
    // guards are taken).
    let mut e_cm = e_cm_closed_form;
    let kind = eout_kind[mat_slot];
    let n_eout = eout_n_energies[mat_slot] as usize;
    if kind == EOUT_KIND_CONTINUOUS_TABULAR && n_eout > 0 {
        // Slice C: per-slot tabulated continuous E_out with optional
        // discrete head, linlin/histogram per-bracket interp, and a
        // global `histogram_outer` flag that suppresses both the
        // stochastic bracket pick and the bracket-bound stretch.
        // Tight CSR layout (issue #104): the slot's ae-rows start at
        // `eg_off_e`; (x, p, cdf) are read from the full arrays via the
        // per-row global `x_offset`. No per-axis stride.
        let eg_off_e = eout_ae_offset[mat_slot] as usize;
        let hist_outer = eout_histogram_interp[mat_slot] != 0;
        if let Some(e_sampled) =
            crate::gpu::flat::tabulated_continuous_eout::sample_tabulated_continuous_eout(
                e_in,
                &eout_energy_grid[eg_off_e..eg_off_e + n_eout],
                &eout_n_x[eg_off_e..eg_off_e + n_eout],
                &eout_interp[eg_off_e..eg_off_e + n_eout],
                &eout_n_discrete[eg_off_e..eg_off_e + n_eout],
                hist_outer,
                eout_x,
                eout_p,
                eout_cdf,
                &eout_x_offset[eg_off_e..eg_off_e + n_eout],
                state,
            )
        {
            e_cm = e_sampled;
        }
    } else if kind == EOUT_KIND_CORRELATED {
        // Slice D: correlated angle-energy. E_out and μ come from
        // the same per-(mat × MT × E_in × E_out) joint distribution
        // -- μ at the chosen `(bin_e, j)` bin has its own angular
        // sub-table. The yamc-physics function returns `Some(_, None)`
        // when the chosen bin has no μ data, so the slice-B μ (or
        // isotropic fallback) survives.
        let n_corr_total = corr_n_energies[mat_slot] as usize;
        if n_corr_total > 0 {
            // Multi-component mixture (issue #111): the neutron product carried
            // several equally-weighted correlated laws (F19 MT16 n,2n, two at
            // 0.5/0.5). Pick one uniformly per collision -- mirrors the CPU
            // `ReactionProduct::sample_distribution_index` for equal
            // applicability. Only one extra draw when `>= 2` components, so
            // single-component slots stay bit-identical. Component `c` occupies
            // the `n_corr` rows at `corr_ae_offset[mat_slot] + c * n_corr`.
            let n_comp = corr_n_components[mat_slot].max(1) as usize;
            let n_corr = n_corr_total / n_comp;
            let comp = if n_comp >= 2 {
                let xi = yamc_rng::next_xi(state);
                ((xi * n_comp as f64) as usize).min(n_comp - 1)
            } else {
                0
            };
            // Tight CSR layout (issue #104), three nesting levels: the chosen
            // component's ae-rows start at `eg_off_c`; the per-row (x, p, cdf) /
            // n_mu / mu_interp and the per-x-point mu sub-tables are read from
            // the full arrays via the per-row `corr_x_offset` and per-x-point
            // `corr_mu_offset`. No per-axis stride.
            let eg_off_c = corr_ae_offset[mat_slot] as usize + comp * n_corr;
            if let Some((e_sampled, mu_opt)) =
                crate::gpu::flat::correlated_angle_energy::sample_correlated_angle_energy(
                    e_in,
                    &corr_energy_grid[eg_off_c..eg_off_c + n_corr],
                    &corr_n_x[eg_off_c..eg_off_c + n_corr],
                    &corr_interp[eg_off_c..eg_off_c + n_corr],
                    &corr_n_discrete[eg_off_c..eg_off_c + n_corr],
                    corr_x,
                    corr_p,
                    corr_cdf,
                    corr_n_mu,
                    corr_mu_interp,
                    corr_mu,
                    corr_mu_pdf,
                    corr_mu_cdf,
                    &corr_x_offset[eg_off_c..eg_off_c + n_corr],
                    corr_mu_offset,
                    state,
                )
            {
                e_cm = e_sampled;
                if let Some(mu) = mu_opt {
                    mu_sampled = mu;
                }
            }
        }
    } else if kind == EOUT_KIND_KALBACH_MANN {
        // Slice E: Kalbach-Mann correlated angle-energy. Override
        // both `e_cm` (was the closed-form Q-value) and `mu_sampled`
        // (was the slice-B isotropic / tabulated angular result).
        let mat_slot = slab * MT_INELASTIC_COUNT + selected_slot;
        let n_e = km_n_energies[mat_slot] as usize;
        if n_e > 0 {
            // Tight CSR (issue #104): the slot's ae-rows start at `eg_off`;
            // (x, p, c, r, a) are read from the full arrays via the per-row
            // global `x_offset`. No per-axis stride.
            let eg_off = km_ae_offset[mat_slot] as usize;
            if let Some((e_km, mu_km)) = crate::gpu::flat::kalbach_mann::sample_kalbach_mann(
                e_in,
                &km_energy_grid[eg_off..eg_off + n_e],
                &km_n_x[eg_off..eg_off + n_e],
                &km_interp[eg_off..eg_off + n_e],
                &km_n_discrete[eg_off..eg_off + n_e],
                km_x,
                km_p,
                km_c,
                km_r,
                km_a,
                &km_x_offset[eg_off..eg_off + n_e],
                state,
            ) {
                e_cm = e_km;
                mu_sampled = mu_km;
            }
        }
    } else if kind == EOUT_KIND_EVAPORATION {
        // Slice F: Evaporation E_out. Overrides `e_cm` only;
        // `mu_sampled` keeps the slice-B angular value (or
        // isotropic fallback) sampled above.
        let mat_slot = slab * MT_INELASTIC_COUNT + selected_slot;
        let n_pts = evap_n_energies[mat_slot] as usize;
        if n_pts > 0 {
            // Tight CSR (issue #104): the slot's E_in rows start at
            // `evap_ae_offset[mat_slot]`; component-major theta at
            // `evap_theta_offset[mat_slot]`. No per-axis stride.
            let off = evap_ae_offset[mat_slot] as usize;
            let eg = &evap_energy_grid[off..off + n_pts];
            // Multi-component mixture: the neutron product carried several
            // equally-weighted Evaporation laws; pick one uniformly per
            // collision (mirrors the CPU `sample_distribution_index` for
            // equal applicability). One uniform draw only when `>= 2`
            // components, so single-component slots stay bit-identical.
            let n_comp = evap_n_components[mat_slot].max(1) as usize;
            let comp = if n_comp >= 2 {
                let xi = yamc_rng::next_xi(state);
                ((xi * n_comp as f64) as usize).min(n_comp - 1)
            } else {
                0
            };
            // `u` is incident-energy dependent (multi-law applicability):
            // pick it at the nearest-lower grid point (NOT interpolated --
            // the applicability switch is a step), mirroring the kernel's
            // `u_idx` selection.
            let mut u_idx = n_pts - 1;
            if e_in <= eg[0] {
                u_idx = 0;
            } else if e_in < eg[n_pts - 1] {
                let mut k = 0usize;
                while k + 1 < n_pts {
                    if e_in >= eg[k] && e_in < eg[k + 1] {
                        u_idx = k;
                    }
                    k += 1;
                }
            }
            // Component-major theta: component `comp` occupies its own
            // `n_pts`-point row within this slot's region.
            let theta_off = evap_theta_offset[mat_slot] as usize + comp * n_pts;
            if let Some(e_evap) = crate::gpu::flat::evaporation::sample_evaporation(
                e_in,
                eg,
                &evap_theta[theta_off..theta_off + n_pts],
                evap_u[off + u_idx],
                state,
            ) {
                e_cm = e_evap;
            }
        }
    } else if kind == EOUT_KIND_NBODY_PHASE_SPACE {
        // Slice G: N-body phase space. Overrides both `e_cm` and
        // `mu_sampled` (NBPS is isotropic in CM regardless of any
        // slice-B table).
        let mat_slot = slab * MT_INELASTIC_COUNT + selected_slot;
        if let Some((e_nbps, mu_nbps)) =
            crate::gpu::flat::nbody_phase_space::sample_nbody_phase_space(
                e_in,
                target_mass,
                q_value,
                nbps_n_bodies[mat_slot],
                nbps_total_mass[mat_slot],
                state,
            )
        {
            e_cm = e_nbps;
            mu_sampled = mu_nbps;
        }
    } else if kind == EOUT_KIND_MAXWELL {
        // Maxwell E_out. Overrides `e_cm` only; `mu_sampled` keeps
        // the slice-B angular value (or isotropic fallback)
        // sampled above.
        let mat_slot = slab * MT_INELASTIC_COUNT + selected_slot;
        let n_pts = maxwell_n_energies[mat_slot] as usize;
        if n_pts > 0 {
            // Tight CSR (issue #104): the slot's E_in rows start at
            // `maxwell_ae_offset[mat_slot]`. No per-axis stride.
            let off = maxwell_ae_offset[mat_slot] as usize;
            if let Some(e_max) = crate::gpu::flat::maxwell::sample_maxwell(
                e_in,
                &maxwell_energy_grid[off..off + n_pts],
                &maxwell_theta[off..off + n_pts],
                maxwell_u[mat_slot],
                state,
            ) {
                e_cm = e_max;
            }
        }
    } else if kind == EOUT_KIND_WATT {
        // Watt-inelastic E_out. Overrides `e_cm` only; `mu_sampled`
        // keeps the slice-B angular value (or isotropic fallback).
        let mat_slot = slab * MT_INELASTIC_COUNT + selected_slot;
        let n_pts = watt_n_energies[mat_slot] as usize;
        if n_pts > 0 {
            // Tight CSR (issue #104): the slot's E_in rows start at
            // `watt_ae_offset[mat_slot]`. No per-axis stride.
            let off = watt_ae_offset[mat_slot] as usize;
            if let Some(e_w) = crate::gpu::flat::watt::sample_watt_inelastic(
                e_in,
                &watt_energy_grid[off..off + n_pts],
                &watt_a[off..off + n_pts],
                &watt_b[off..off + n_pts],
                watt_u[mat_slot],
                state,
            ) {
                e_cm = e_w;
            }
        }
    } else if kind == EOUT_KIND_TABULATED {
        // Tabulated equiprobable E_out. Reuses the eout_*
        // buffers -- `cdf` is zero-padded on this branch.
        // Overrides `e_cm` only; `mu_sampled` keeps the slice-B
        // value (or isotropic fallback).
        let mat_slot = slab * MT_INELASTIC_COUNT + selected_slot;
        let n_eout = eout_n_energies[mat_slot] as usize;
        if n_eout > 0 {
            // Tight CSR (issue #104): the slot's ae-rows start at `eg_off`;
            // the equiprobable bins are read from the full `eout_x` via the
            // per-row global `x_offset`. No per-axis stride.
            let eg_off = eout_ae_offset[mat_slot] as usize;
            if let Some(e_t) =
                crate::gpu::flat::tabulated_equiprobable::sample_tabulated_equiprobable(
                    e_in,
                    &eout_energy_grid[eg_off..eg_off + n_eout],
                    &eout_n_x[eg_off..eg_off + n_eout],
                    eout_x,
                    &eout_x_offset[eg_off..eg_off + n_eout],
                    state,
                )
            {
                e_cm = e_t;
            }
        }
    }

    if e_cm <= 0.0 {
        return (mu_sampled, e_in, false);
    }

    if scatter_in_cm_per_mt[mat_slot] == 1 {
        match crate::gpu::flat::cm_to_lab::cm_to_lab(e_in, e_cm, mu_sampled, target_mass) {
            Some((mu_lab, e_lab)) => (mu_lab, e_lab, true),
            None => (mu_sampled, e_in, false),
        }
    } else {
        (mu_sampled, e_cm, true)
    }
}
