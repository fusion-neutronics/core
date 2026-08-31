//! Public entry points that assemble `GpuNuclideXs` / per-MT score
//! buffers from yamc `Nuclide`s, plus extraction-local helpers.

use super::distributions::*;
use super::*;
use std::collections::HashMap;
use std::sync::Arc;
use yamc_nuclide::nuclide::{Nuclide, SCATTERING_MTS_NON_INELASTIC};
use yamc_nuclide::reaction::Reaction;
use yamc_nuclide::reaction_product::{
    AngleEnergyDistribution, ElasticAngleFlat, EnergyDistribution, FissionChiFlat, Tabulated1D,
    TabulatedInterp, TabulatedProbability,
};

/// Total cross section at a given energy, mirroring the CPU's
/// `FastXSGrid` accounting. Prefers the precomputed `fast_xs.lookup`
/// when populated (the canonical path for ENDF nuclides loaded via
/// Arrow); falls back to summing every non-redundant reaction in
/// `nuclide.reactions[temp_idx]` when `fast_xs` is empty (synthetic
/// test fixtures). The non-redundant filter is essential: ENDF
/// libraries store both aggregated reactions (MT 1, MT 4, etc.)
/// flagged `redundant = true` and the per-channel constituents
/// flagged `redundant = false`; summing only the constituents
/// avoids double-counting.
/// Build the per-MT macroscopic XS buffer used by the kernel's
/// `SCORE_PER_MT` accumulator path.
///
/// Returns a flat `Vec<f64>` of length `score_mts.len() * energy_grid.len()`,
/// laid out slot-major (`out[slot * n_grid + i_grid]`). Each slot's xs
/// is the density-weighted sum over `nuclides` of that nuclide's
/// reaction cross-section at MT `score_mts[slot]`. Missing reactions
/// contribute zero -- exactly the CPU's `Material::lookup_xs_by_mt`
/// fallback behaviour.
///
/// Special MT handling:
/// - `MT 1` (total) is summed across non-redundant reactions per
///   nuclide via `total_xs_at` so the returned curve matches the CPU's
///   total cross section.
/// - `MT 4` (inelastic aggregate) -- if a nuclide has no MT 4 stored,
///   the value is summed over MT 51..=91 from the same nuclide. ENDF
///   libraries vary on whether MT 4 is materialised; CPU's lookup
///   sees both forms.
/// - `MT 27` (absorption) -- if a nuclide has no MT 27 stored, the
///   value falls back to `σ_t − σ_e − Σ_inelastic` so cell-only
///   absorption tallies don't silently zero out.
///
/// All other MTs: direct `reactions.get(&mt).cross_section_at(e)`
/// lookup. The `energy_grid` passed in must be the same one used for
/// the rest of the GPU's per-material XS arrays so the kernel's
/// existing `idx_lo` / `idx_hi` / `frac` indices are valid.
pub fn extract_score_xs_per_mt(
    nuclides: &[(&Nuclide, f64)],
    temperature: &str,
    score_mts: &[i32],
    energy_grid: &[f64],
) -> Result<Vec<f64>, NuclideXsError> {
    let n_grid = energy_grid.len();
    let n_mts = score_mts.len();
    let mut out = vec![0.0_f64; n_mts * n_grid];

    if n_mts == 0 || n_grid == 0 {
        return Ok(out);
    }

    for (nuclide, density) in nuclides {
        let temp_idx = nuclide
            .get_temp_idx(temperature)
            .ok_or_else(|| NuclideXsError::TemperatureNotLoaded(temperature.to_string()))?;
        let reactions = &nuclide.reactions[temp_idx];

        for (slot, &mt) in score_mts.iter().enumerate() {
            let off = slot * n_grid;
            match mt {
                1 => {
                    // Total -- sum across non-redundant reactions.
                    for (i, &e) in energy_grid.iter().enumerate() {
                        out[off + i] += density * total_xs_at(nuclide, temp_idx, e);
                    }
                }
                4 => {
                    // Inelastic aggregate -- prefer MT 4 if present,
                    // otherwise sum MT 51..=91 from the same nuclide.
                    if let Some(rxn) = reactions.get(&4) {
                        for (i, &e) in energy_grid.iter().enumerate() {
                            out[off + i] += density * rxn.cross_section_at(e).unwrap_or(0.0);
                        }
                    } else {
                        for (i, &e) in energy_grid.iter().enumerate() {
                            let mut s = 0.0_f64;
                            for mt_il in MT_INELASTIC_FIRST..=MT_INELASTIC_LAST {
                                if let Some(rxn) = reactions.get(&mt_il) {
                                    s += rxn.cross_section_at(e).unwrap_or(0.0);
                                }
                            }
                            out[off + i] += density * s;
                        }
                    }
                }
                27 => {
                    // Absorption -- prefer MT 27 if present, otherwise
                    // derive as σ_t − σ_e − Σ_inelastic to match the
                    // CPU's accounting.
                    if let Some(rxn) = reactions.get(&27) {
                        for (i, &e) in energy_grid.iter().enumerate() {
                            out[off + i] += density * rxn.cross_section_at(e).unwrap_or(0.0);
                        }
                    } else {
                        let elastic = reactions.get(&MT_ELASTIC);
                        for (i, &e) in energy_grid.iter().enumerate() {
                            let total = total_xs_at(nuclide, temp_idx, e);
                            let xs_e = elastic
                                .map(|r| r.cross_section_at(e).unwrap_or(0.0))
                                .unwrap_or(0.0);
                            let mut xs_il = 0.0_f64;
                            for mt_il in MT_INELASTIC_FIRST..=MT_INELASTIC_LAST {
                                if let Some(rxn) = reactions.get(&mt_il) {
                                    xs_il += rxn.cross_section_at(e).unwrap_or(0.0);
                                }
                            }
                            // Multi-neutron channels (MT 16, 17) aren't
                            // absorbing; subtract elastic + every
                            // inelastic-with-yield-1 channel only.
                            let derived = (total - xs_e - xs_il).max(0.0);
                            out[off + i] += density * derived;
                        }
                    }
                }
                _ => {
                    // Direct per-MT lookup; missing → 0.
                    if let Some(rxn) = reactions.get(&mt) {
                        for (i, &e) in energy_grid.iter().enumerate() {
                            out[off + i] += density * rxn.cross_section_at(e).unwrap_or(0.0);
                        }
                    }
                }
            }
        }
    }

    Ok(out)
}

/// Parse the temperature label resolved by yamc into Kelvin.
///
/// Shared with `Material::temperature_k` rather than reimplemented here. This
/// used to be a private copy, and the copy was right while `Material`'s was
/// stale, which is issue #478: the GPU ran a 900 K material at 900 K and the
/// CPU ran it at 294 K.
use yamc_nuclide::temperature::label_to_kelvin_or_default as parse_temperature_k;

fn total_xs_at(nuclide: &Nuclide, temp_idx: usize, energy: f64) -> f64 {
    if let Some(fast_grid) = nuclide.fast_xs.get(temp_idx) {
        if !fast_grid.energy.is_empty() {
            let (total, _abs, _scat, _fis) = fast_grid.lookup(energy);
            return total;
        }
    }
    nuclide.reactions[temp_idx]
        .values()
        .filter(|r| !r.redundant)
        .map(|r| r.cross_section_at(energy).unwrap_or(0.0))
        .sum()
}

/// Sum the per-MT inelastic cross sections across all `MT_INELASTIC_COUNT`
/// slots at each of the `n_grid` energy points, giving the aggregate
/// inelastic xs used for branch-probability lookups. `xs_inelastic_per_mt`
/// is the flat `[slot * n_grid + i]` buffer.
fn aggregate_inelastic_across_slots(xs_inelastic_per_mt: &[f64], n_grid: usize) -> Vec<f64> {
    let mut total = vec![0.0_f64; n_grid];
    for slot in 0..MT_INELASTIC_COUNT {
        let off = slot * n_grid;
        for (i, t) in total.iter_mut().enumerate() {
            *t += xs_inelastic_per_mt[off + i];
        }
    }
    total
}

/// Largest tolerated share of a nuclide's total cross section that may sit in
/// neutron-emitting MTs the kernel cannot sample (issue #106).
///
/// Zero would be the principled number, but it would refuse real data for no
/// practical gain: on ENDF/B-VIII.1 the only affected nuclide is La139, whose
/// unslotted MT 152-200 series reaches 2.5e-6 of its total, and only above
/// 20 MeV. 0.1% is far below the statistical error of any transport run while
/// still catching a channel that would actually move a result.
const UNSLOTTED_SCATTER_TOLERANCE: f64 = 1e-3;

/// Refuse when the nuclide's unslotted neutron-emitting MTs exceed
/// [`UNSLOTTED_SCATTER_TOLERANCE`] of its total cross section anywhere on the
/// grid.
///
/// The comparison is against the NUCLIDE's own total, not the material's
/// macroscopic total, which is the conservative direction: a trace nuclide with
/// a large unslotted fraction is refused even though its contribution to the
/// material is small. That is deliberate. The alternative, silently
/// over-absorbing, is the bug this guards against, and a wrong answer is worse
/// than a fallback to the CPU.
fn check_unslotted_scatter_mts(
    nuclide: &Nuclide,
    reactions: &HashMap<i32, Arc<Reaction>>,
    xs_total: &[f64],
    energy_grid: &[f64],
) -> Result<(), NuclideXsError> {
    let mut offenders: Vec<i32> = Vec::new();
    let mut worst = 0.0_f64;
    let mut summed: Vec<f64> = vec![0.0; energy_grid.len()];
    for (&mt, rxn) in reactions.iter() {
        if mt == MT_ELASTIC
            || rxn.redundant
            || MT_SLOTS.contains(&mt)
            || !SCATTERING_MTS_NON_INELASTIC.contains(&mt)
        {
            continue;
        }
        offenders.push(mt);
        for (i, &e) in energy_grid.iter().enumerate() {
            summed[i] += rxn.cross_section_at(e).unwrap_or(0.0);
        }
    }
    if offenders.is_empty() {
        return Ok(());
    }
    for (i, &total) in xs_total.iter().enumerate() {
        if total > 0.0 {
            worst = worst.max(summed[i] / total);
        }
    }
    if worst > UNSLOTTED_SCATTER_TOLERANCE {
        offenders.sort_unstable();
        return Err(NuclideXsError::UnslottedScatterMts {
            nuclide: nuclide.name.clone().unwrap_or_else(|| "(unnamed)".into()),
            mts: offenders,
            max_fraction: worst,
        });
    }
    Ok(())
}

/// Derived absorption xs: `σ_t − σ_e − σ_inelastic − σ_f` per energy point,
/// clamped non-negative. This mirrors the CPU's accounting and auto-includes
/// every non-modeled capture-like channel ((n,γ), (n,p), (n,α), …) without
/// enumerating them. All slices must have the same length.
fn derive_absorption(
    xs_total: &[f64],
    xs_elastic: &[f64],
    xs_inelastic: &[f64],
    xs_fission: &[f64],
) -> Vec<f64> {
    (0..xs_total.len())
        .map(|i| (xs_total[i] - xs_elastic[i] - xs_inelastic[i] - xs_fission[i]).max(0.0))
        .collect()
}

/// Per-energy average fission multiplicity `ν̄ = (ν·σ_f) / σ_f` where σ_f is
/// non-zero, else `0.0` (no fission → ν̄ is unused). Both slices must have
/// the same length.
fn nu_bar_from_nu_sigma_f(nu_sigma_f: &[f64], xs_fission: &[f64]) -> Vec<f64> {
    (0..xs_fission.len())
        .map(|i| {
            if xs_fission[i] > 0.0 {
                nu_sigma_f[i] / xs_fission[i]
            } else {
                0.0
            }
        })
        .collect()
}

/// Per-energy delayed-neutron fraction `beta(E)`: the fission-rate-weighted
/// `nu_d-sigma_f / nu-sigma_f`, or `0.0` where the material emits no fission
/// neutrons at all (issue #364). Clamped to `[0, 1]` because `nu_d` and `nu_t`
/// come from different blocks of the evaluation. Both slices must have the same
/// length.
fn beta_from_nu_sigma_f(nu_delayed_sigma_f: &[f64], nu_sigma_f: &[f64]) -> Vec<f64> {
    (0..nu_sigma_f.len())
        .map(|i| {
            if nu_sigma_f[i] > 0.0 {
                (nu_delayed_sigma_f[i] / nu_sigma_f[i]).clamp(0.0, 1.0)
            } else {
                0.0
            }
        })
        .collect()
}

/// Extract `(log E, σ_elastic, σ_absorption)` arrays from a yamc
/// `Nuclide` at the given temperature. Resamples both reactions onto
/// the parent (top-level) energy grid using yamc's existing
/// linear-interpolation lookup so values below thresholds are 0 and
/// values above the grid clamp at the last entry. Returns the result
/// plus the atomic-weight ratio for the kinematics step.
///
/// The output `xs_*` arrays are microscopic (the values stored in the
/// nuclide). For a single-nuclide pseudo-material the caller can
/// treat them as macroscopic; for real materials, multiply by atomic
/// density and sum across nuclides on the host before upload.
pub fn extract_xs_from_nuclide(
    nuclide: &Nuclide,
    temperature: &str,
) -> Result<GpuNuclideXs, NuclideXsError> {
    let temp_idx = nuclide
        .get_temp_idx(temperature)
        .ok_or_else(|| NuclideXsError::TemperatureNotLoaded(temperature.to_string()))?;

    let energy_grid = nuclide
        .energy
        .as_ref()
        .and_then(|e| e.get(temperature))
        .ok_or_else(|| NuclideXsError::MissingEnergyGrid(temperature.to_string()))?;

    let reactions = &nuclide.reactions[temp_idx];
    let elastic = reactions
        .get(&MT_ELASTIC)
        .ok_or_else(|| NuclideXsError::MissingReaction {
            mt: MT_ELASTIC,
            temperature: temperature.to_string(),
        })?;
    // MT 102 (radiative capture) is no longer required. The absorption
    // xs is computed below as `σ_t − σ_e − σ_inelastic_modeled`,
    // mirroring the CPU's `FastXSGrid` accounting where σ_a is total
    // minus the channels handled by neutron-emitting branches. A
    // nuclide with no radiative-capture channel (e.g. He4) just
    // contributes zero there, so its presence is not needed.

    // Resample onto the parent grid via the existing linear
    // interpolator. yamc populates `Reaction::energy` at load time,
    // so the lookup is a binary search per point -- not the fastest
    // way to resample but this runs once per launch, not per particle.
    let log_energy_grid: Vec<f64> = energy_grid.iter().map(|e| e.ln()).collect();
    let xs_elastic: Vec<f64> = energy_grid
        .iter()
        .map(|&e| elastic.cross_section_at(e).unwrap_or(0.0))
        .collect();
    // σ_t per the CPU's authoritative source: prefer
    // `nuclide.fast_xs[temp_idx].lookup(e).0` (the precomputed
    // total) when available; otherwise sum all non-redundant
    // reactions (ENDF marks reactions like MT 1 / MT 4 as
    // `redundant` when individual constituents are present, so
    // summing only non-redundant entries gives the same total).
    let xs_total: Vec<f64> = energy_grid
        .iter()
        .map(|&e| total_xs_at(nuclide, temp_idx, e))
        .collect();

    // Per-MT inelastic xs + Q, in slot order `MT_SLOTS`. Slots the
    // nuclide doesn't have are all-zero -- the kernel just never
    // samples them. Aggregated `xs_inelastic` is the sum across
    // slots at each energy. Slots 0..=40 are MT 51..=91 (single-
    // neutron-out); slots 41–42 are MT 16/17 (multi-neutron-out,
    // weight-multiplier handled in the kernel).
    let n_grid = energy_grid.len();
    let mut xs_inelastic_per_mt: Vec<f64> = vec![0.0; MT_INELASTIC_COUNT * n_grid];
    let mut q_inelastic_per_mt: Vec<f64> = vec![0.0; MT_INELASTIC_COUNT];
    // Per-MT yield ν(E). Default 1.0 everywhere (CPU's fallback when
    // a product has no yield curve); slots that do carry a yield
    // overwrite via `evaluate`.
    let mut yield_per_mt: Vec<f64> = vec![1.0; MT_INELASTIC_COUNT * n_grid];
    for (slot, &mt) in MT_SLOTS.iter().enumerate() {
        if let Some(rxn) = reactions.get(&mt) {
            // Skip redundant aggregates: some libraries store both an
            // aggregate (e.g. MT 16) flagged `redundant` and its
            // level-specific constituents. Summing both would
            // double-count and over-subtract from derived absorption.
            // The CPU scatter loader applies the same guard.
            if rxn.redundant {
                continue;
            }
            let off = slot * n_grid;
            for (i, &e) in energy_grid.iter().enumerate() {
                xs_inelastic_per_mt[off + i] = rxn.cross_section_at(e).unwrap_or(0.0);
            }
            q_inelastic_per_mt[slot] = rxn.q_value;
            // First neutron product carries the yield. Mirrors
            // `sample_from_products_with_awr` in yamc::inelastic.
            for product in &rxn.products {
                if !product.is_particle_type(&yamc_nuclide::particle_type::ParticleType::Neutron) {
                    continue;
                }
                if let Some(y) = &product.product_yield {
                    for (i, &e) in energy_grid.iter().enumerate() {
                        yield_per_mt[off + i] = y.evaluate(e);
                    }
                }
                break;
            }
        }
    }
    // Aggregate xs across MT slots for branch-probability use.
    let xs_inelastic = aggregate_inelastic_across_slots(&xs_inelastic_per_mt, n_grid);

    // Issue #106: refuse a nuclide whose unslotted neutron-emitting channels
    // are big enough to bias the answer. Anything the kernel has no slot for
    // is invisible to `xs_inelastic` and therefore lands in the derived
    // absorption below, so the GPU would kill neutrons the CPU scatters.
    check_unslotted_scatter_mts(nuclide, reactions, &xs_total, energy_grid)?;

    // Per-energy fission xs (sum across MT 18 / 19 / 20 / 21 / 38) and
    // ν̄(E) for the fission-branch sampling. Mirrors the multi-nuclide
    // path in `extract_material_xs`. Empty for non-fissionable
    // nuclides -- kernel just sees σ_f = 0 everywhere and never samples
    // the branch.
    let mut xs_fission: Vec<f64> = vec![0.0; n_grid];
    let mut nu_bar: Vec<f64> = vec![0.0; n_grid];
    let mut beta_delayed: Vec<f64> = vec![0.0; n_grid];
    if nuclide.fissionable {
        let mut nu_sigma_f: Vec<f64> = vec![0.0; n_grid];
        let mut nu_delayed_sigma_f: Vec<f64> = vec![0.0; n_grid];
        // Delayed groups' total yield nu_d(E) (issue #364); `None` when the
        // evaluation has no delayed data, which leaves beta zero everywhere.
        let delayed = nuclide.delayed_neutrons(temperature);
        for &fmt in &[18, 19, 20, 21, 38] {
            if let Some(rxn) = reactions.get(&fmt) {
                for (i, &e) in energy_grid.iter().enumerate() {
                    let xs_f = rxn.cross_section_at(e).unwrap_or(0.0);
                    xs_fission[i] += xs_f;
                    let nu = nuclide
                        .fission_nu
                        .as_ref()
                        .map(|n| n.evaluate(e))
                        .unwrap_or(2.5);
                    nu_sigma_f[i] += xs_f * nu;
                    if let Some(d) = delayed {
                        nu_delayed_sigma_f[i] += xs_f * d.nu(e);
                    }
                }
            }
        }
        nu_bar = nu_bar_from_nu_sigma_f(&nu_sigma_f, &xs_fission);
        beta_delayed = beta_from_nu_sigma_f(&nu_delayed_sigma_f, &nu_sigma_f);
    }

    // Derived absorption xs: everything not modeled as elastic,
    // inelastic, or fission. Clamped non-negative to absorb tiny
    // rounding drifts.
    let xs_absorption = derive_absorption(&xs_total, &xs_elastic, &xs_inelastic, &xs_fission);

    let target_mass = nuclide
        .atomic_weight_ratio
        .ok_or(NuclideXsError::MissingAtomicWeightRatio)?;
    // Parse the resolved temperature string ("294" / "294K") into
    // Kelvin for the kernel's free-gas thermal scattering branch.
    // Falls back to 294 K (room temperature) if the label doesn't
    // parse as a number -- matches the CPU's default in
    // `Material::new`.
    let temperature_k = parse_temperature_k(temperature);

    // Per-MT angular distributions: pull each MT slot's first
    // `UncorrelatedAngleEnergy` distribution out of its neutron
    // product and pack it into the GPU-ready flat buffers. Slots
    // without data are zero-padded; the kernel falls back to
    // isotropic-in-CM sampling when it sees `n_energies == 0`.
    let (
        angle_n_energies,
        angle_energy_grid,
        angle_n_mu,
        angle_mu,
        angle_cdf,
        angle_pdf,
        angle_interp,
        scatter_in_cm,
    ) = build_per_mt_angle_buffers(|mt| reactions.get(&mt).map(|arc| arc.as_ref()))?;

    let (
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
    ) = build_per_mt_eout_buffers(|mt| reactions.get(&mt).map(|arc| arc.as_ref()));

    let (
        corr_n_energies,
        corr_n_components,
        corr_energy_grid,
        corr_n_x,
        corr_x,
        corr_cdf,
        corr_p,
        corr_interp,
        corr_n_discrete,
        corr_n_mu,
        corr_mu,
        corr_mu_cdf,
        corr_mu_pdf,
        corr_mu_interp,
    ) = build_per_mt_corr_buffers(|mt| reactions.get(&mt).map(|arc| arc.as_ref()));

    let (
        km_n_energies,
        km_energy_grid,
        km_interp,
        km_n_discrete,
        km_n_x,
        km_x,
        km_p,
        km_c,
        km_r,
        km_a,
    ) = build_per_mt_km_buffers(|mt| reactions.get(&mt).map(|arc| arc.as_ref()));

    let (evap_n_energies, evap_n_components, evap_energy_grid, evap_theta, evap_u) =
        build_per_mt_evap_buffers(|mt| reactions.get(&mt).map(|arc| arc.as_ref()));

    let (maxwell_n_energies, maxwell_energy_grid, maxwell_theta, maxwell_u) =
        build_per_mt_maxwell_buffers(|mt| reactions.get(&mt).map(|arc| arc.as_ref()));

    let (watt_n_energies, watt_energy_grid, watt_a, watt_b, watt_u) =
        build_per_mt_watt_buffers(|mt| reactions.get(&mt).map(|arc| arc.as_ref()));

    let (nbps_n_bodies, nbps_total_mass) =
        build_per_mt_nbps_buffers(|mt| reactions.get(&mt).map(|arc| arc.as_ref()));

    let (urr_meta, urr_energy_grid, urr_cdf, urr_xs, urr_atom_density) =
        build_urr_buffers(&[(nuclide, 1.0)], temperature);

    // Watt parameters for prompt fission χ-spectrum sampling. Same
    // shape as `extract_material_xs` -- pull from the dominant fission
    // MT's first Watt-shaped neutron product, fall back to typical
    // thermal-fission `(0.988 MeV, 2.249 / MeV)`.
    let (fission_watt_a, fission_watt_b) =
        extract_watt_params_single(nuclide, temperature).unwrap_or((0.988e6, 2.249e-6));
    let fission_eout = extract_fission_eout_single(nuclide, temperature);
    let fission_eout_delayed =
        FissionEoutSlot::delayed_from_nuclides(&[(nuclide, 1.0)], temperature);

    Ok(GpuNuclideXs {
        // Single nuclide: fine and coarse grids are the same nuclide grid, so
        // this path stays bit-identical to before issue #88.
        coarse_log_energy_grid: log_energy_grid.clone(),
        log_energy_grid,
        xs_elastic,
        xs_absorption,
        xs_inelastic,
        xs_inelastic_per_mt,
        q_inelastic_per_mt,
        yield_per_mt,
        target_mass,
        temperature_k,
        angle_n_energies,
        angle_energy_grid,
        angle_n_mu,
        angle_mu,
        angle_cdf,
        angle_pdf,
        angle_interp,
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
        corr_n_energies,
        corr_n_components,
        corr_energy_grid,
        corr_n_x,
        corr_x,
        corr_cdf,
        corr_p,
        corr_interp,
        corr_n_discrete,
        corr_n_mu,
        corr_mu,
        corr_mu_cdf,
        corr_mu_pdf,
        corr_mu_interp,
        scatter_in_cm,
        xs_fission,
        nu_bar,
        beta_delayed,
        fission_watt_a,
        fission_watt_b,
        fission_eout_kind: fission_eout.kind,
        fission_eout_n_energies: fission_eout.n_energies,
        fission_eout_energy_grid: fission_eout.energy_grid,
        fission_eout_n_x: fission_eout.n_x,
        fission_eout_x: fission_eout.x,
        fission_eout_cdf: fission_eout.cdf,
        fission_eout_p: fission_eout.p,
        fission_eout_interp: fission_eout.interp,
        fission_eout_delayed_kind: fission_eout_delayed.kind,
        fission_eout_delayed_n_energies: fission_eout_delayed.n_energies,
        fission_eout_delayed_energy_grid: fission_eout_delayed.energy_grid,
        fission_eout_delayed_n_x: fission_eout_delayed.n_x,
        fission_eout_delayed_x: fission_eout_delayed.x,
        fission_eout_delayed_cdf: fission_eout_delayed.cdf,
        fission_eout_delayed_p: fission_eout_delayed.p,
        fission_eout_delayed_interp: fission_eout_delayed.interp,
        km_n_energies,
        km_energy_grid,
        km_interp,
        km_n_discrete,
        km_n_x,
        km_x,
        km_p,
        km_c,
        km_r,
        km_a,
        evap_n_energies,
        evap_n_components,
        evap_energy_grid,
        evap_theta,
        evap_u,
        nbps_n_bodies,
        nbps_total_mass,
        maxwell_n_energies,
        maxwell_energy_grid,
        maxwell_theta,
        maxwell_u,
        watt_n_energies,
        watt_energy_grid,
        watt_a,
        watt_b,
        watt_u,
        urr_meta,
        urr_energy_grid,
        urr_cdf,
        urr_xs,
        urr_atom_density,
    })
}

/// The exact UNION of the material's per-nuclide energy grids at `temperature`
/// (issue #88). This mirrors the CPU's `Material::unified_energy_grid_neutron`
/// exactly -- concatenate every nuclide's grid, sort ascending, dedup with a
/// `1e-12` tolerance -- so the GPU's collision / nuclide-selection cross
/// sections preserve every isotope's resonances, closing the multi-isotope
/// epithermal residual on alloys (SS316 1-30 eV was 0.33-0.49 of CPU on the
/// finest single grid). For a single-nuclide material the union of one grid is
/// that grid (sort + dedup are no-ops on an already-sorted unique grid), so it
/// equals `finest_energy_grid` and the single-nuclide path stays bit-identical.
pub fn union_energy_grid(
    nuclides: &[(&Nuclide, f64)],
    temperature: &str,
) -> Result<Vec<f64>, NuclideXsError> {
    let mut all_energies: Vec<f64> = Vec::new();
    for (nuclide, _) in nuclides {
        let grid = nuclide
            .energy
            .as_ref()
            .and_then(|e| e.get(temperature))
            .ok_or_else(|| NuclideXsError::MissingEnergyGrid(temperature.to_string()))?;
        all_energies.extend_from_slice(grid);
    }
    all_energies.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());
    all_energies.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    if all_energies.is_empty() {
        return Err(NuclideXsError::MissingEnergyGrid(temperature.to_string()));
    }
    Ok(all_energies)
}

/// The GPU's COARSE energy grid for a multi-nuclide material: the FINEST
/// (most points) of the material's per-nuclide grids at `temperature`.
///
/// Issue #74: the previous master grid was the *first* nuclide's grid, and the
/// host iterates a material's nuclides from a `HashMap`, so *which* nuclide was
/// first -- and hence the master grid's resolution -- was non-deterministic.
/// When a coarse-grid isotope landed first, the dominant isotope's resonance
/// structure was resampled away, the GPU slowing-down spectrum diverged from the
/// CPU (which keeps every resonance via the union grid) by up to ~40%, and the
/// result varied run-to-run. Picking the finest grid is deterministic and
/// always preserves the most-resolved isotope's resonances.
///
/// The mathematically exact choice is the *union* of all nuclide grids (what the
/// CPU's `Material::unified_energy_grid_neutron` builds), but for a many-nuclide
/// material that union is ~10x larger (natFe ~108k points, SS316 ~363k) and the
/// GPU's slab-major per-MT buffers (`[n_slab x MT_INELASTIC_COUNT x n_grid]`)
/// scale with it -- multi-GB, beyond a single device. The finest single grid
/// keeps memory bounded to one nuclide's grid while removing the
/// order-dependence and the dominant-isotope smearing that caused the bug.
pub fn finest_energy_grid<'a>(
    nuclides: &[(&'a Nuclide, f64)],
    temperature: &str,
) -> Result<&'a [f64], NuclideXsError> {
    let mut best: Option<&'a [f64]> = None;
    for (nuclide, _) in nuclides {
        let grid = nuclide
            .energy
            .as_ref()
            .and_then(|e| e.get(temperature))
            .ok_or_else(|| NuclideXsError::MissingEnergyGrid(temperature.to_string()))?;
        if best.is_none_or(|b| grid.len() > b.len()) {
            best = Some(grid.as_slice());
        }
    }
    best.ok_or_else(|| NuclideXsError::MissingEnergyGrid(temperature.to_string()))
}

/// Aggregate macroscopic cross sections across nuclides weighted by
/// atomic densities. Returns a single `GpuNuclideXs` representing the
/// material as one effective lookup.
///
/// ```text
/// Σ_mat(E) = Σ_i N_i · σ_i(E)
/// ```
///
/// # Limits of this first-cut aggregation
///
/// - **Effective target_mass.** Real transport picks a nuclide at
///   each collision (weighted by Σ_i / Σ_total) and uses *that*
///   nuclide's mass for elastic kinematics. This aggregator instead
///   reports a number-density-weighted average mass:
///   `<A> = Σ_i N_i A_i / Σ_i N_i`. That's an approximation --
///   acceptable for tally integrals where the mean rate is what
///   matters, less acceptable for energy-spectrum tallies where the
///   per-collision energy loss spread depends on which nuclide was
///   hit. A future kernel that uploads per-nuclide arrays and samples
///   the nuclide on the GPU lifts this restriction.
///
/// The master grid is the FINEST of the material's per-nuclide grids (see
/// [`finest_energy_grid`]): deterministic and order-independent, preserving the
/// most-resolved isotope's resonances. Issue #74 fixed the prior "first
/// nuclide's grid" choice, which was `HashMap`-order-dependent and smeared the
/// dominant isotope's resonances when a coarse isotope happened to be first.
///
/// Caller passes `&[(&Nuclide, atomic_density)]`. Atomic density is
/// in whatever units the caller chooses; what matters is consistency
/// across nuclides so the weighted sum produces the right macroscopic
/// cross section in those same units.
pub fn extract_material_xs(
    nuclides: &[(&Nuclide, f64)],
    temperature: &str,
) -> Result<GpuNuclideXs, NuclideXsError> {
    if nuclides.is_empty() {
        return Err(NuclideXsError::EmptyMaterial);
    }

    // Dual energy grid (issue #88). The FINE grid is the exact UNION of the
    // material's per-nuclide grids -- the same grid the CPU's
    // `Material::unified_energy_grid_neutron` builds -- so every isotope's
    // resonances survive in the collision / nuclide-selection cross sections
    // (closing the epithermal residual on alloys). The COARSE grid is the
    // finest single per-nuclide grid; it backs only the 3D per-MT inelastic
    // buffers, which on the union grid would be multi-GB (SS316 ~2.6 GB).
    // Inelastic is a fast-energy channel with no epithermal resonance
    // structure, so the coarse grid costs no accuracy there. For a
    // single-nuclide material the union equals that nuclide's grid equals the
    // finest grid, so fine == coarse and the whole path stays bit-identical.
    let energy_grid = union_energy_grid(nuclides, temperature)?;
    let log_energy_grid: Vec<f64> = energy_grid.iter().map(|e| e.ln()).collect();
    let n_grid = energy_grid.len();
    let coarse_grid = finest_energy_grid(nuclides, temperature)?.to_vec();
    let coarse_log_energy_grid: Vec<f64> = coarse_grid.iter().map(|e| e.ln()).collect();
    let n_coarse = coarse_grid.len();

    // FINE (union) collision buffers: elastic, total, aggregate inelastic,
    // fission, and ν·σ_f. These drive the collision-branch probabilities and
    // the derived absorption, all resonance-critical.
    let mut xs_elastic_total = vec![0.0_f64; n_grid];
    let mut xs_total_aggregate = vec![0.0_f64; n_grid];
    let mut xs_inelastic_total = vec![0.0_f64; n_grid];
    // Per-energy fission xs (sum across MT 18 / 19 / 20 / 21 / 38 from
    // every nuclide, density-weighted) and ν̄ × σ_f (so the per-energy
    // average is taken weighted by reaction rate, not by density alone).
    let mut xs_fission_total = vec![0.0_f64; n_grid];
    let mut nu_sigma_f_total = vec![0.0_f64; n_grid];
    let mut nu_delayed_sigma_f_total = vec![0.0_f64; n_grid];

    // COARSE per-MT inelastic buffers. For each MT slot, the material's xs is
    // the density-weighted sum of contributing nuclides' xs at that MT. The
    // Q-value is the peak-xs-weighted average across nuclides -- for
    // single-nuclide materials this is exactly the nuclide's Q; for
    // multi-nuclide materials it's an approximation consistent with the rest of
    // the per-material aggregation. These curves carry no epithermal resonance
    // structure (inelastic is a fast-energy channel), so the coarse grid is
    // exact-enough and keeps the 3D buffers off the (much larger) union grid.
    let mut xs_inelastic_per_mt = vec![0.0_f64; MT_INELASTIC_COUNT * n_coarse];
    let mut q_inelastic_per_mt = vec![0.0_f64; MT_INELASTIC_COUNT];
    let mut q_weight_per_mt = vec![0.0_f64; MT_INELASTIC_COUNT];
    // Per-MT yield ν(E) accumulated as `Σ density × σ_MT × yield`;
    // divided by `Σ density × σ_MT` after the per-nuclide pass to
    // give a reaction-rate-weighted average yield. Slots that no
    // nuclide contributes to keep their default `1.0`.
    let mut yield_sigma_per_mt = vec![0.0_f64; MT_INELASTIC_COUNT * n_coarse];
    let mut yield_weight_per_mt = vec![0.0_f64; MT_INELASTIC_COUNT * n_coarse];

    let mut total_density = 0.0_f64;
    let mut weighted_mass = 0.0_f64;
    for (nuclide, density) in nuclides {
        let temp_idx = nuclide
            .get_temp_idx(temperature)
            .ok_or_else(|| NuclideXsError::TemperatureNotLoaded(temperature.to_string()))?;
        let reactions = &nuclide.reactions[temp_idx];
        let elastic =
            reactions
                .get(&MT_ELASTIC)
                .ok_or_else(|| NuclideXsError::MissingReaction {
                    mt: MT_ELASTIC,
                    temperature: temperature.to_string(),
                })?;
        // MT 102 is no longer required; absorption is derived as
        // `σ_t − σ_e − σ_inelastic` after the inelastic aggregation,
        // matching the CPU's accounting. A nuclide with no
        // radiative-capture channel just contributes zero there.
        let mass = nuclide
            .atomic_weight_ratio
            .ok_or(NuclideXsError::MissingAtomicWeightRatio)?;

        for (i, &e) in energy_grid.iter().enumerate() {
            xs_elastic_total[i] += density * elastic.cross_section_at(e).unwrap_or(0.0);
            xs_total_aggregate[i] += density * total_xs_at(nuclide, temp_idx, e);
        }

        // Aggregate inelastic on the FINE (union) grid -- the collision-branch
        // probability weight. Summed over the same non-redundant MT slots the
        // per-MT (coarse) buffers below use, so the two agree wherever the
        // grids coincide (single nuclide) and in the resonance-free fast region
        // where the inelastic channel actually lives.
        for &mt in MT_SLOTS.iter() {
            if let Some(rxn) = reactions.get(&mt) {
                if rxn.redundant {
                    continue;
                }
                for (i, &e) in energy_grid.iter().enumerate() {
                    xs_inelastic_total[i] += density * rxn.cross_section_at(e).unwrap_or(0.0);
                }
            }
        }

        // Per-MT inelastic aggregation on the COARSE grid. Slot order matches
        // `MT_SLOTS` (slots 0..=40 = MT 51..=91, slots 41–42 = MT 16/17).
        for (slot, &mt) in MT_SLOTS.iter().enumerate() {
            if let Some(rxn) = reactions.get(&mt) {
                // Skip redundant aggregates (see the matching guard in
                // `extract_xs_from_nuclide`): prevents double-counting a
                // library's aggregate MT against its constituents and
                // keeps derived absorption non-negative.
                if rxn.redundant {
                    continue;
                }
                let off = slot * n_coarse;
                let mut peak_xs = 0.0;
                for (i, &e) in coarse_grid.iter().enumerate() {
                    let xs = rxn.cross_section_at(e).unwrap_or(0.0);
                    xs_inelastic_per_mt[off + i] += density * xs;
                    if xs > peak_xs {
                        peak_xs = xs;
                    }
                }
                // Q-value for this MT, density × peak-xs weighted across
                // nuclides. The Q-value is energy-independent, so the
                // weight only needs to be nonzero whenever the nuclide
                // contributes this MT; the *peak* cross section over the
                // grid satisfies that for every channel. (A previous
                // version weighted by the cross section at the top of the
                // grid, which is zero for threshold reactions whose xs
                // falls off well below the grid maximum -- e.g. discrete-
                // level inelastic MT 51..=90 on light nuclides -- so their
                // Q collapsed to 0, the closed-form `e_cm = (A/(A+1))^2 *
                // (E - (A+1)/A*|Q|)` lost its threshold term, and those
                // collisions kept almost all their energy. That hardened
                // the flux spectrum on C12 / N14 / O16 / Li6.)
                let w = density * peak_xs;
                q_inelastic_per_mt[slot] += w * rxn.q_value;
                q_weight_per_mt[slot] += w;
                // First neutron product's yield curve, sampled at every
                // grid point. Reaction-rate weighted across nuclides.
                let mut nuclide_yield: Option<&Reaction> = Some(rxn.as_ref());
                if let Some(rxn_ref) = nuclide_yield {
                    if let Some(neutron_product) = rxn_ref.products.iter().find(|p| {
                        p.is_particle_type(&yamc_nuclide::particle_type::ParticleType::Neutron)
                    }) {
                        let yield_curve = neutron_product.product_yield.as_ref();
                        for (i, &e) in coarse_grid.iter().enumerate() {
                            let y = yield_curve.map(|c| c.evaluate(e)).unwrap_or(1.0);
                            let xs = rxn_ref.cross_section_at(e).unwrap_or(0.0);
                            yield_sigma_per_mt[off + i] += density * xs * y;
                            yield_weight_per_mt[off + i] += density * xs;
                        }
                    }
                }
                let _ = nuclide_yield.take();
            }
        }

        // Per-nuclide fission XS aggregation. Sum every fission MT
        // present (18 / 19 / 20 / 21 / 38) at each grid point; multiply
        // by ν̄(E) for the nu-σ_f integrand used to derive the
        // material-level ν̄. Non-fissionable nuclides contribute zero.
        if nuclide.fissionable {
            // Delayed groups' total yield nu_d(E), for the material's delayed
            // fraction (issue #364). `None` for an evaluation with no delayed data,
            // which then contributes nothing to nu_d-sigma_f and so leaves the
            // material's beta at zero.
            let delayed = nuclide.delayed_neutrons(temperature);
            for &fmt in &[18, 19, 20, 21, 38] {
                if let Some(rxn) = reactions.get(&fmt) {
                    for (i, &e) in energy_grid.iter().enumerate() {
                        let xs_f = rxn.cross_section_at(e).unwrap_or(0.0);
                        xs_fission_total[i] += density * xs_f;
                        if let Some(ref nu) = nuclide.fission_nu {
                            nu_sigma_f_total[i] += density * xs_f * nu.evaluate(e);
                        } else {
                            // No nu-bar table -- fall back to 2.5 (CPU
                            // does the same).
                            nu_sigma_f_total[i] += density * xs_f * 2.5;
                        }
                        if let Some(d) = delayed {
                            nu_delayed_sigma_f_total[i] += density * xs_f * d.nu(e);
                        }
                    }
                }
            }
        }

        total_density += density;
        weighted_mass += density * mass;
    }

    if total_density == 0.0 {
        return Err(NuclideXsError::ZeroTotalDensity);
    }
    // Density-weighted average mass, except for a single-nuclide material, where
    // the average IS that nuclide's mass and the division only adds rounding:
    // `(N * A) / N` comes out a full ulp above `A` for Am240 at 5 g/cm3 (exact
    // for W184, Fe56 and F19, which is why only some nuclides showed it), and the
    // CPU's elastic kinematics use `A` itself, so every non-free-gas elastic
    // scatter landed a few ulp off the CPU's (issue #111). The loop above already
    // errored on a nuclide with no atomic weight ratio.
    let target_mass = if let [(single, _)] = nuclides {
        single
            .atomic_weight_ratio
            .ok_or(NuclideXsError::MissingAtomicWeightRatio)?
    } else {
        weighted_mass / total_density
    };
    let temperature_k = parse_temperature_k(temperature);

    // Finalise per-MT Q-values (divide by accumulated weights). Slots
    // with zero weight (no nuclide contributes that MT) keep Q=0;
    // they're also xs=0 so the kernel never selects them.
    for slot in 0..MT_INELASTIC_COUNT {
        if q_weight_per_mt[slot] > 0.0 {
            q_inelastic_per_mt[slot] /= q_weight_per_mt[slot];
        }
    }

    // Finalise per-MT yields ν(E). Reaction-rate weighted average
    // across nuclides where weight > 0 (i.e. at least one nuclide
    // contributes that MT at that energy); fall back to `1.0` when
    // no nuclide carries the MT at that energy point -- matches the
    // CPU's `unwrap_or(1.0)` default in
    // `sample_from_products_with_awr`.
    let mut yield_per_mt = vec![1.0_f64; MT_INELASTIC_COUNT * n_coarse];
    for (i, slot_data) in yield_per_mt.iter_mut().enumerate() {
        if yield_weight_per_mt[i] > 0.0 {
            *slot_data = yield_sigma_per_mt[i] / yield_weight_per_mt[i];
        }
    }

    // `xs_inelastic_total` is the FINE (union) aggregate accumulated in the
    // per-nuclide loop above (not derived from the coarse per-MT buffers), so
    // the collision-branch probability and the derived absorption stay on the
    // resonance-faithful grid.

    // Derived absorption: `σ_t − σ_e − σ_inelastic − σ_fission` per
    // energy point. Pulling fission out of the absorption derivation
    // means the kernel can sample it as its own collision branch
    // (with ν̄ as the surviving-neutron weight multiplier and a Watt
    // χ-spectrum E_out resample), which is the slice-G upgrade. The
    // remaining `xs_absorption_total` still auto-includes (n,γ),
    // (n,p), (n,d), (n,α), (n,2p), (n,n'p) -- every non-modeled
    // capture-like channel.
    let xs_absorption_total = derive_absorption(
        &xs_total_aggregate,
        &xs_elastic_total,
        &xs_inelastic_total,
        &xs_fission_total,
    );

    // Per-energy ν̄ for the material -- `nu_sigma_f / σ_f` where σ_f is
    // non-zero, otherwise zero (no fission means ν̄ doesn't matter).
    let nu_bar_total = nu_bar_from_nu_sigma_f(&nu_sigma_f_total, &xs_fission_total);

    // Per-energy delayed fraction for the material: the fission-rate-weighted
    // `nu_d / nu_t` (issue #364). For a single fissile nuclide it reduces exactly
    // to that nuclide's `nu_d(E) / nu_t(E)`, which is what the CPU evaluates.
    let beta_delayed_total = beta_from_nu_sigma_f(&nu_delayed_sigma_f_total, &nu_sigma_f_total);

    // Per-MT angular distributions for the material. For each MT
    // slot, pick the first nuclide that has tabulated angular data
    // for that MT -- multi-nuclide angular aggregation would need a
    // per-collision stochastic nuclide pick, which the kernel doesn't
    // do today (same approximation as the Q-value aggregation
    // above). Single-nuclide materials get exact data.
    // Per-MT angular + outgoing-energy distributions, both pulled
    // from the first nuclide that has data for the slot. Multi-
    // nuclide aggregation would need a per-collision stochastic
    // nuclide pick (not done on GPU yet); single-nuclide materials
    // get exact data.
    let pick_first = |mt: i32| {
        for (nuclide, _density) in nuclides {
            let Some(temp_idx) = nuclide.get_temp_idx(temperature) else {
                continue;
            };
            let reactions = &nuclide.reactions[temp_idx];
            if let Some(rxn) = reactions.get(&mt) {
                return Some(rxn.as_ref());
            }
        }
        None
    };
    let (
        angle_n_energies,
        angle_energy_grid,
        angle_n_mu,
        angle_mu,
        angle_cdf,
        angle_pdf,
        angle_interp,
        scatter_in_cm,
    ) = build_per_mt_angle_buffers(pick_first)?;
    let (
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
    ) = build_per_mt_eout_buffers(pick_first);
    let (
        corr_n_energies,
        corr_n_components,
        corr_energy_grid,
        corr_n_x,
        corr_x,
        corr_cdf,
        corr_p,
        corr_interp,
        corr_n_discrete,
        corr_n_mu,
        corr_mu,
        corr_mu_cdf,
        corr_mu_pdf,
        corr_mu_interp,
    ) = build_per_mt_corr_buffers(pick_first);
    let (
        km_n_energies,
        km_energy_grid,
        km_interp,
        km_n_discrete,
        km_n_x,
        km_x,
        km_p,
        km_c,
        km_r,
        km_a,
    ) = build_per_mt_km_buffers(pick_first);

    let (evap_n_energies, evap_n_components, evap_energy_grid, evap_theta, evap_u) =
        build_per_mt_evap_buffers(pick_first);

    let (maxwell_n_energies, maxwell_energy_grid, maxwell_theta, maxwell_u) =
        build_per_mt_maxwell_buffers(pick_first);

    let (watt_n_energies, watt_energy_grid, watt_a, watt_b, watt_u) =
        build_per_mt_watt_buffers(pick_first);

    let (nbps_n_bodies, nbps_total_mass) = build_per_mt_nbps_buffers(pick_first);

    let (urr_meta, urr_energy_grid, urr_cdf, urr_xs, urr_atom_density) =
        build_urr_buffers(nuclides, temperature);

    // Watt parameters for the prompt fission χ-spectrum. Pull from the
    // dominant fission MT's first neutron product if it carries a
    // `Watt { a, b, .. }` distribution; otherwise fall back to a
    // typical thermal-fission Watt (a = 0.988 MeV, b = 2.249 / MeV
    // expressed in eV / 1-per-eV units). The MVP evaluates `a` and `b`
    // at a representative incident energy of 1 MeV -- the χ-spectrum
    // shape doesn't vary strongly with E_in across actinides at fast
    // energies, and a single (a, b) avoids burning per-energy memory
    // until the comparison shows it matters.
    let (fission_watt_a, fission_watt_b) =
        extract_watt_params(nuclides, temperature).unwrap_or((0.988e6, 2.249e-6));
    let fission_eout = FissionEoutSlot::from_nuclides(nuclides, temperature);
    let fission_eout_delayed = FissionEoutSlot::delayed_from_nuclides(nuclides, temperature);

    Ok(GpuNuclideXs {
        // Dual grid (issue #88): the FINE `log_energy_grid` is the union of the
        // per-nuclide grids (resonance-faithful collision / selection XS); the
        // COARSE grid backs the per-MT inelastic buffers. Equal for a
        // single-nuclide material, so that path stays bit-identical.
        coarse_log_energy_grid,
        log_energy_grid,
        xs_elastic: xs_elastic_total,
        xs_absorption: xs_absorption_total,
        xs_inelastic: xs_inelastic_total,
        xs_inelastic_per_mt,
        q_inelastic_per_mt,
        yield_per_mt,
        target_mass,
        temperature_k,
        angle_n_energies,
        angle_energy_grid,
        angle_n_mu,
        angle_mu,
        angle_cdf,
        angle_pdf,
        angle_interp,
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
        corr_n_energies,
        corr_n_components,
        corr_energy_grid,
        corr_n_x,
        corr_x,
        corr_cdf,
        corr_p,
        corr_interp,
        corr_n_discrete,
        corr_n_mu,
        corr_mu,
        corr_mu_cdf,
        corr_mu_pdf,
        corr_mu_interp,
        scatter_in_cm,
        xs_fission: xs_fission_total,
        nu_bar: nu_bar_total,
        beta_delayed: beta_delayed_total,
        fission_watt_a,
        fission_watt_b,
        fission_eout_kind: fission_eout.kind,
        fission_eout_n_energies: fission_eout.n_energies,
        fission_eout_energy_grid: fission_eout.energy_grid,
        fission_eout_n_x: fission_eout.n_x,
        fission_eout_x: fission_eout.x,
        fission_eout_cdf: fission_eout.cdf,
        fission_eout_p: fission_eout.p,
        fission_eout_interp: fission_eout.interp,
        fission_eout_delayed_kind: fission_eout_delayed.kind,
        fission_eout_delayed_n_energies: fission_eout_delayed.n_energies,
        fission_eout_delayed_energy_grid: fission_eout_delayed.energy_grid,
        fission_eout_delayed_n_x: fission_eout_delayed.n_x,
        fission_eout_delayed_x: fission_eout_delayed.x,
        fission_eout_delayed_cdf: fission_eout_delayed.cdf,
        fission_eout_delayed_p: fission_eout_delayed.p,
        fission_eout_delayed_interp: fission_eout_delayed.interp,
        km_n_energies,
        km_energy_grid,
        km_interp,
        km_n_discrete,
        km_n_x,
        km_x,
        km_p,
        km_c,
        km_r,
        km_a,
        evap_n_energies,
        evap_n_components,
        evap_energy_grid,
        evap_theta,
        evap_u,
        nbps_n_bodies,
        nbps_total_mass,
        maxwell_n_energies,
        maxwell_energy_grid,
        maxwell_theta,
        maxwell_u,
        watt_n_energies,
        watt_energy_grid,
        watt_a,
        watt_b,
        watt_u,
        urr_meta,
        urr_energy_grid,
        urr_cdf,
        urr_xs,
        urr_atom_density,
    })
}

/// Per-nuclide macroscopic total cross section for a single material,
/// resampled onto a shared energy grid.
///
/// Row `n` holds `atom_density[n] * sigma_total(E)` for the material's
/// `n`-th nuclide at every grid energy, laid out nuclide-major
/// (`n * n_grid + energy_idx`) to match the existing `xs_inelastic_per_mt`
/// convention. This is the per-collision weight table the on-device
/// nuclide selector walks (see [`crate::neutron::nuclide_select`]); summed
/// over nuclides it equals the aggregate macroscopic total assembled by
/// [`extract_material_xs`].
///
/// The per-nuclide values are identical to the CPU material's
/// `macroscopic_xs_neutron_total_by_nuclide` (both are `density *
/// total_micro_xs` on the same grid), so the device selector reproduces
/// `Material::sample_interacting_nuclide`.
#[derive(Debug, Clone)]
pub struct PerNuclideMacroXs {
    /// Number of nuclides in the material (rows in `macro_total_xs`).
    pub n_nuclides: usize,
    /// Length of the energy grid (columns per row).
    pub n_grid: usize,
    /// Flat `[n_nuclides * n_grid]`, nuclide-major: entry `n * n_grid + i`
    /// is nuclide `n`'s macroscopic total xs at `energy_grid[i]`.
    pub macro_total_xs: Vec<f64>,
}

/// Extract the per-nuclide macroscopic total xs table (see
/// [`PerNuclideMacroXs`]) for `nuclides` (`(nuclide, atom_density)` pairs)
/// at `temperature`, sampled on `energy_grid`. Pass the same grid used to
/// build the material's [`GpuNuclideXs`] so the rows align with the
/// aggregate buffers.
pub fn extract_per_nuclide_macro_total_xs(
    nuclides: &[(&Nuclide, f64)],
    temperature: &str,
    energy_grid: &[f64],
) -> Result<PerNuclideMacroXs, NuclideXsError> {
    if nuclides.is_empty() {
        return Err(NuclideXsError::EmptyMaterial);
    }
    let n_grid = energy_grid.len();
    let mut macro_total_xs = vec![0.0_f64; nuclides.len() * n_grid];
    for (n, (nuclide, density)) in nuclides.iter().enumerate() {
        let temp_idx = nuclide
            .get_temp_idx(temperature)
            .ok_or_else(|| NuclideXsError::TemperatureNotLoaded(temperature.to_string()))?;
        let row = n * n_grid;
        for (i, &e) in energy_grid.iter().enumerate() {
            macro_total_xs[row + i] = density * total_xs_at(nuclide, temp_idx, e);
        }
    }
    Ok(PerNuclideMacroXs {
        n_nuclides: nuclides.len(),
        n_grid,
        macro_total_xs,
    })
}

/// Per-nuclide elastic (MT 2) angular distribution pool for one material
/// (issue #74, Stage 2a). Where [`GpuNuclideXs::elastic_angle_*`] carries a
/// single material-blended slot (first nuclide with MT 2 data), this carries
/// one slot per nuclide so the transport can sample the SELECTED nuclide's
/// elastic CM cosine. All buffers are `[n_nuclides × …]`, nuclide-major in the
/// same order as the input `nuclides` slice (the order the per-collision
/// selector picks `chosen` against), so concatenating them material-major
/// aligns row `slab` with `mat_nuclide_meta`'s `[offset, count]`.
#[derive(Debug, Clone)]
pub struct PerNuclideElasticAngle {
    /// Number of nuclides (rows).
    pub n_nuclides: usize,
    /// Per-nuclide tabulated incident-energy count, `[n_nuclides]`. Zero means
    /// "no tabulated elastic angle, fall back to isotropic-in-CM".
    pub n_energies: Vec<u32>,
    /// Tight CSR (issue #104): incident-energy grids for all nuclides
    /// concatenated back to back, length `sum(n_energies)`.
    pub energy_grid: Vec<f64>,
    /// Per-(nuclide, incident-energy) outgoing-cosine count, one entry per
    /// ae-row (length `sum(n_energies)`).
    pub n_mu: Vec<u32>,
    /// Outgoing CM cosines, concatenated tight across all rows, length
    /// `sum(n_mu)`.
    pub mu: Vec<f64>,
    /// CDF for `mu`, same shape.
    pub cdf: Vec<f64>,
    /// Per-point PDF for `mu` / `cdf`, same shape.
    pub pdf: Vec<f64>,
    /// Per-ae-row interpolation flag, one entry per ae-row (length
    /// `sum(n_energies)`).
    pub interp: Vec<u32>,
}

/// Extract the per-nuclide elastic-angle pool (see [`PerNuclideElasticAngle`])
/// for `nuclides` at `temperature`. Each nuclide's MT 2 reaction is flattened
/// to the tight variable-length [`ElasticAngleFlat`] layout via the same
/// `to_elastic_flat` the CPU transport uses (no stride-subsampling); a nuclide
/// without MT 2 angular data contributes an empty slot (kernel falls back to
/// isotropic). A single-nuclide material yields one row, byte-identical to the
/// prior single material-blended slot.
pub fn extract_per_nuclide_elastic_angle(
    nuclides: &[(&Nuclide, f64)],
    temperature: &str,
) -> Result<PerNuclideElasticAngle, NuclideXsError> {
    if nuclides.is_empty() {
        return Err(NuclideXsError::EmptyMaterial);
    }
    let n = nuclides.len();
    let mut n_energies = Vec::with_capacity(n);
    let mut energy_grid = Vec::new();
    let mut n_mu = Vec::new();
    let mut mu = Vec::new();
    let mut cdf = Vec::new();
    let mut pdf = Vec::new();
    let mut interp = Vec::new();
    for (nuclide, _density) in nuclides {
        let temp_idx = nuclide
            .get_temp_idx(temperature)
            .ok_or_else(|| NuclideXsError::TemperatureNotLoaded(temperature.to_string()))?;
        let reactions = &nuclide.reactions[temp_idx];
        // Tight, full-resolution flatten (no MAX_* subsampling, issue #104),
        // via the same `to_elastic_flat` the CPU transport uses.
        let flat = match reactions.get(&MT_ELASTIC) {
            Some(rxn) => elastic_flat_from_reaction(rxn.as_ref()),
            None => ElasticAngleFlat::empty(),
        };
        n_energies.push(flat.energy_grid.len() as u32);
        energy_grid.extend_from_slice(&flat.energy_grid);
        n_mu.extend_from_slice(&flat.n_mu);
        mu.extend_from_slice(&flat.mu);
        cdf.extend_from_slice(&flat.cdf);
        pdf.extend_from_slice(&flat.pdf);
        interp.extend_from_slice(&flat.interp);
    }
    Ok(PerNuclideElasticAngle {
        n_nuclides: n,
        n_energies,
        energy_grid,
        n_mu,
        mu,
        cdf,
        pdf,
        interp,
    })
}

/// Per-(material, nuclide) inelastic distribution + reaction-type-partial pool
/// for one material (issue #74, Stage 2b). Where [`extract_material_xs`]
/// material-blends the per-MT inelastic distributions (the `pick_first`
/// closure) and drives the reaction-type four-way split from material-aggregate
/// partials, this carries one full set of per-MT distribution buffers AND the
/// four reaction partials (elastic / absorption / inelastic / fission) PER
/// NUCLIDE, so the kernel can:
///   1. pick the struck nuclide (Stage 1, by macroscopic total), then
///   2. split the reaction type from THAT nuclide's own partials (mirroring
///      CPU `Nuclide::sample_reaction_type`), then
///   3. sample THAT nuclide's own inelastic secondary distribution.
///
/// All per-MT fields are flat `[n_nuclides × MT_INELASTIC_COUNT × …]`,
/// nuclide-major in the same order as the input `nuclides` slice (= the
/// per-collision selector's `chosen` order), so concatenating them
/// material-major aligns the leading `slab` dimension with `mat_nuclide_meta`.
/// The four partials are flat `[n_nuclides × n_grid]` on the shared master
/// grid (the same grid the aggregate per-material XS uses), DENSITY-WEIGHTED
/// (macroscopic) so the kernel's reaction split is `σ^n_e / Σ^n_t` etc.
///
/// A single-nuclide material yields exactly one slab whose buffers are
/// byte-identical to the material-blended ones [`extract_material_xs`]
/// produces, and whose partials equal the material aggregate (one nuclide
/// contributes everything), so the kernel stays bit-for-bit on single-nuclide
/// materials.
#[derive(Debug, Clone)]
pub struct PerNuclideInelastic {
    /// Number of nuclides (slabs) in this material.
    pub n_nuclides: usize,
    /// Per-nuclide reaction partials on the master grid, DENSITY-WEIGHTED,
    /// flat `[n_nuclides × n_grid]` each. The kernel splits the reaction type
    /// after selecting nuclide `n` using `σ^n_e`, `σ^n_a`, `σ^n_i`, `σ^n_f`.
    pub sigma_elastic: Vec<f64>,
    pub sigma_absorption: Vec<f64>,
    pub sigma_inelastic: Vec<f64>,
    pub sigma_fission: Vec<f64>,
    /// SPARSE per-MT inelastic XS / yield (issue #212). Per (slab, MT slot) only
    /// the tight nonzero (above-threshold) range of the material's coarse grid is
    /// stored, concatenated in (slab, slot) order. `permt_i_start[slab *
    /// MT_INELASTIC_COUNT + slot]` is the first coarse-grid index where the slot's
    /// XS is nonzero (relative to the material's coarse grid) and
    /// `permt_n_stored[..]` the number of contiguous stored points (`0` = absent /
    /// all-zero slot). The values live tight in `xs_inelastic_per_mt_sparse` /
    /// `yield_per_mt_sparse`; both share the same offset and range, so one meta
    /// row serves both. Interior zeros within `[i_start, i_start + n_stored)` are
    /// preserved (stored as XS `0.0`, yield `1.0` -- the dense default), so the
    /// kernel's sparse lookup reproduces the dense buffer bit-for-bit.
    pub xs_inelastic_per_mt_sparse: Vec<f64>,
    pub yield_per_mt_sparse: Vec<f64>,
    /// Per (slab, MT slot) first nonzero coarse-grid index, `[n_nuclides ×
    /// MT_INELASTIC_COUNT]`.
    pub permt_i_start: Vec<u32>,
    /// Per (slab, MT slot) stored-point count, `[n_nuclides × MT_INELASTIC_COUNT]`
    /// (`0` = absent).
    pub permt_n_stored: Vec<u32>,
    pub q_inelastic_per_mt: Vec<f64>,
    pub angle_n_energies: Vec<u32>,
    pub angle_energy_grid: Vec<f64>,
    pub angle_n_mu: Vec<u32>,
    pub angle_mu: Vec<f64>,
    pub angle_cdf: Vec<f64>,
    pub angle_pdf: Vec<f64>,
    pub angle_interp: Vec<u32>,
    pub scatter_in_cm: Vec<u32>,
    pub eout_kind: Vec<u32>,
    pub eout_n_energies: Vec<u32>,
    pub eout_histogram_interp: Vec<u32>,
    pub eout_energy_grid: Vec<f64>,
    pub eout_n_x: Vec<u32>,
    pub eout_x: Vec<f64>,
    pub eout_p: Vec<f64>,
    pub eout_cdf: Vec<f64>,
    pub eout_interp: Vec<u32>,
    pub eout_n_discrete: Vec<u32>,
    pub corr_n_energies: Vec<u32>,
    pub corr_n_components: Vec<u32>,
    pub corr_energy_grid: Vec<f64>,
    pub corr_n_x: Vec<u32>,
    pub corr_x: Vec<f64>,
    pub corr_cdf: Vec<f64>,
    pub corr_p: Vec<f64>,
    pub corr_interp: Vec<u32>,
    pub corr_n_discrete: Vec<u32>,
    pub corr_n_mu: Vec<u32>,
    pub corr_mu: Vec<f64>,
    pub corr_mu_cdf: Vec<f64>,
    pub corr_mu_pdf: Vec<f64>,
    pub corr_mu_interp: Vec<u32>,
    pub km_n_energies: Vec<u32>,
    pub km_energy_grid: Vec<f64>,
    pub km_interp: Vec<u32>,
    pub km_n_discrete: Vec<u32>,
    pub km_n_x: Vec<u32>,
    pub km_x: Vec<f64>,
    pub km_p: Vec<f64>,
    pub km_c: Vec<f64>,
    pub km_r: Vec<f64>,
    pub km_a: Vec<f64>,
    pub evap_n_energies: Vec<u32>,
    pub evap_n_components: Vec<u32>,
    pub evap_energy_grid: Vec<f64>,
    pub evap_theta: Vec<f64>,
    pub evap_u: Vec<f64>,
    pub nbps_n_bodies: Vec<u32>,
    pub nbps_total_mass: Vec<f64>,
    pub maxwell_n_energies: Vec<u32>,
    pub maxwell_energy_grid: Vec<f64>,
    pub maxwell_theta: Vec<f64>,
    pub maxwell_u: Vec<f64>,
    pub watt_n_energies: Vec<u32>,
    pub watt_energy_grid: Vec<f64>,
    pub watt_a: Vec<f64>,
    pub watt_b: Vec<f64>,
    pub watt_u: Vec<f64>,
}

/// Extract the per-(material, nuclide) inelastic distribution + reaction-partial
/// pool (see [`PerNuclideInelastic`]) for `nuclides` (`(nuclide, atom_density)`
/// pairs) at `temperature`. Issue #88 splits the grids: the four reaction
/// partials live on the FINE (union) `fine_grid` -- resonance-critical, because
/// the kernel's per-collision reaction-type split reads them at the struck
/// nuclide's resonances -- while the per-MT inelastic XS / Q / yield live on
/// the COARSE `coarse_grid` (the finest single per-nuclide grid), matching the
/// material aggregate's coarse per-MT buffers and keeping the 3D
/// `[n_slab × MT × n_grid]` buffers off the multi-GB union grid. Each nuclide
/// supplies one slab:
///   - its per-MT inelastic distribution buffers, built by the same `build_*`
///     helpers [`extract_material_xs`] uses but keyed to THAT nuclide's
///     reactions (not material-blended);
///   - its per-MT inelastic XS / Q / yield on the coarse grid, DENSITY-WEIGHTED
///     (so the per-slab inelastic σ sums to the material aggregate);
///   - its four reaction partials on the fine grid, density-weighted.
///
/// For a single-nuclide material the union equals the finest grid, so
/// `fine_grid == coarse_grid` and the slab is byte-identical to the
/// material-blended data [`extract_material_xs`] produces.
pub fn extract_per_nuclide_inelastic(
    nuclides: &[(&Nuclide, f64)],
    temperature: &str,
    fine_grid: &[f64],
    coarse_grid: &[f64],
) -> Result<PerNuclideInelastic, NuclideXsError> {
    if nuclides.is_empty() {
        return Err(NuclideXsError::EmptyMaterial);
    }
    let n = nuclides.len();
    let n_grid = fine_grid.len();
    let n_coarse = coarse_grid.len();
    let mut pool = PerNuclideInelastic {
        n_nuclides: n,
        sigma_elastic: Vec::with_capacity(n * n_grid),
        sigma_absorption: Vec::with_capacity(n * n_grid),
        sigma_inelastic: Vec::with_capacity(n * n_grid),
        sigma_fission: Vec::with_capacity(n * n_grid),
        // Sparse per-MT storage (issue #212): capacity is a loose upper bound
        // (real data is 90-99.9% zeros, so the sparse buffers stay far smaller).
        xs_inelastic_per_mt_sparse: Vec::new(),
        yield_per_mt_sparse: Vec::new(),
        permt_i_start: Vec::with_capacity(n * MT_INELASTIC_COUNT),
        permt_n_stored: Vec::with_capacity(n * MT_INELASTIC_COUNT),
        q_inelastic_per_mt: Vec::with_capacity(n * MT_INELASTIC_COUNT),
        angle_n_energies: Vec::new(),
        angle_energy_grid: Vec::new(),
        angle_n_mu: Vec::new(),
        angle_mu: Vec::new(),
        angle_cdf: Vec::new(),
        angle_pdf: Vec::new(),
        angle_interp: Vec::new(),
        scatter_in_cm: Vec::new(),
        eout_kind: Vec::new(),
        eout_n_energies: Vec::new(),
        eout_histogram_interp: Vec::new(),
        eout_energy_grid: Vec::new(),
        eout_n_x: Vec::new(),
        eout_x: Vec::new(),
        eout_p: Vec::new(),
        eout_cdf: Vec::new(),
        eout_interp: Vec::new(),
        eout_n_discrete: Vec::new(),
        corr_n_energies: Vec::new(),
        corr_n_components: Vec::new(),
        corr_energy_grid: Vec::new(),
        corr_n_x: Vec::new(),
        corr_x: Vec::new(),
        corr_cdf: Vec::new(),
        corr_p: Vec::new(),
        corr_interp: Vec::new(),
        corr_n_discrete: Vec::new(),
        corr_n_mu: Vec::new(),
        corr_mu: Vec::new(),
        corr_mu_cdf: Vec::new(),
        corr_mu_pdf: Vec::new(),
        corr_mu_interp: Vec::new(),
        km_n_energies: Vec::new(),
        km_energy_grid: Vec::new(),
        km_interp: Vec::new(),
        km_n_discrete: Vec::new(),
        km_n_x: Vec::new(),
        km_x: Vec::new(),
        km_p: Vec::new(),
        km_c: Vec::new(),
        km_r: Vec::new(),
        km_a: Vec::new(),
        evap_n_energies: Vec::new(),
        evap_n_components: Vec::new(),
        evap_energy_grid: Vec::new(),
        evap_theta: Vec::new(),
        evap_u: Vec::new(),
        nbps_n_bodies: Vec::new(),
        nbps_total_mass: Vec::new(),
        maxwell_n_energies: Vec::new(),
        maxwell_energy_grid: Vec::new(),
        maxwell_theta: Vec::new(),
        maxwell_u: Vec::new(),
        watt_n_energies: Vec::new(),
        watt_energy_grid: Vec::new(),
        watt_a: Vec::new(),
        watt_b: Vec::new(),
        watt_u: Vec::new(),
    };

    for (nuclide, density) in nuclides {
        let temp_idx = nuclide
            .get_temp_idx(temperature)
            .ok_or_else(|| NuclideXsError::TemperatureNotLoaded(temperature.to_string()))?;
        let reactions = &nuclide.reactions[temp_idx];

        // Reaction partials on the master grid, density-weighted. Elastic from
        // MT 2; inelastic summed over the modeled MT slots; fission summed over
        // 18/19/20/21/38; absorption derived as σ_t − σ_e − σ_i − σ_f (the same
        // accounting `extract_material_xs` and CPU `FastXSGrid` use). This
        // matches the CPU `Nuclide::sample_reaction_type` scattering /
        // absorption / fission split exactly (scattering = σ_e + σ_i).
        let elastic =
            reactions
                .get(&MT_ELASTIC)
                .ok_or_else(|| NuclideXsError::MissingReaction {
                    mt: MT_ELASTIC,
                    temperature: temperature.to_string(),
                })?;
        for &e in fine_grid {
            let s_e = density * elastic.cross_section_at(e).unwrap_or(0.0);
            let s_t = density * total_xs_at(nuclide, temp_idx, e);
            let mut s_i = 0.0_f64;
            for (slot, &mt) in MT_SLOTS.iter().enumerate() {
                if let Some(rxn) = reactions.get(&mt) {
                    if rxn.redundant {
                        continue;
                    }
                    let _ = slot;
                    s_i += density * rxn.cross_section_at(e).unwrap_or(0.0);
                }
            }
            let mut s_f = 0.0_f64;
            if nuclide.fissionable {
                for &fmt in &[18, 19, 20, 21, 38] {
                    if let Some(rxn) = reactions.get(&fmt) {
                        s_f += density * rxn.cross_section_at(e).unwrap_or(0.0);
                    }
                }
            }
            let s_a = (s_t - s_e - s_i - s_f).max(0.0);
            pool.sigma_elastic.push(s_e);
            pool.sigma_inelastic.push(s_i);
            pool.sigma_fission.push(s_f);
            pool.sigma_absorption.push(s_a);
        }

        // Per-MT inelastic XS / Q / yield on the master grid, density-weighted.
        // Bit-for-bit the SAME accounting `extract_material_xs` uses for one
        // contributing nuclide -- including the peak-xs-weighted Q average
        // (`(w·q)/w`) and the reaction-rate-weighted yield (`(d·xs·y)/(d·xs)`,
        // default 1.0 where xs == 0). Matching this exactly is what keeps a
        // single-nuclide material's slab byte-identical to the prior material-
        // blended `q_inelastic_per_mt` / `yield_per_mt` (the kernel now reads
        // the per-slab pool for single-nuclide materials too).
        let mut nuc_xs_per_mt = vec![0.0_f64; MT_INELASTIC_COUNT * n_coarse];
        let mut nuc_q_per_mt = vec![0.0_f64; MT_INELASTIC_COUNT];
        let mut q_weight_per_mt = vec![0.0_f64; MT_INELASTIC_COUNT];
        let mut yield_sigma_per_mt = vec![0.0_f64; MT_INELASTIC_COUNT * n_coarse];
        let mut yield_weight_per_mt = vec![0.0_f64; MT_INELASTIC_COUNT * n_coarse];
        for (slot, &mt) in MT_SLOTS.iter().enumerate() {
            if let Some(rxn) = reactions.get(&mt) {
                if rxn.redundant {
                    continue;
                }
                let off = slot * n_coarse;
                let mut peak_xs = 0.0_f64;
                for (i, &e) in coarse_grid.iter().enumerate() {
                    let xs = rxn.cross_section_at(e).unwrap_or(0.0);
                    nuc_xs_per_mt[off + i] = density * xs;
                    if xs > peak_xs {
                        peak_xs = xs;
                    }
                }
                let w = density * peak_xs;
                nuc_q_per_mt[slot] += w * rxn.q_value;
                q_weight_per_mt[slot] += w;
                if let Some(neutron_product) = rxn.products.iter().find(|p| {
                    p.is_particle_type(&yamc_nuclide::particle_type::ParticleType::Neutron)
                }) {
                    let yield_curve = neutron_product.product_yield.as_ref();
                    for (i, &e) in coarse_grid.iter().enumerate() {
                        let y = yield_curve.map(|c| c.evaluate(e)).unwrap_or(1.0);
                        let xs = rxn.cross_section_at(e).unwrap_or(0.0);
                        yield_sigma_per_mt[off + i] += density * xs * y;
                        yield_weight_per_mt[off + i] += density * xs;
                    }
                }
            }
        }
        for slot in 0..MT_INELASTIC_COUNT {
            if q_weight_per_mt[slot] > 0.0 {
                nuc_q_per_mt[slot] /= q_weight_per_mt[slot];
            }
        }
        let mut nuc_yield_per_mt = vec![1.0_f64; MT_INELASTIC_COUNT * n_coarse];
        for (i, slot_data) in nuc_yield_per_mt.iter_mut().enumerate() {
            if yield_weight_per_mt[i] > 0.0 {
                *slot_data = yield_sigma_per_mt[i] / yield_weight_per_mt[i];
            }
        }
        // Compress each slot to its tight nonzero (above-threshold) range (issue
        // #212). `i_start` / `i_end` are the first / one-past-last coarse-grid
        // indices where this slot's XS is nonzero; the values are pushed tight
        // over `[i_start, i_end)`. Interior zeros inside the range are kept as-is
        // (XS `0.0`, yield `1.0`), so the sparse lookup reproduces the dense
        // buffer bit-for-bit; slots with no nonzero point store nothing
        // (`n_stored == 0`). Note `!= 0.0` (not a tolerance): below-threshold
        // `cross_section_at` returns exactly `0.0`, matching the dense zeros the
        // kernel used to read.
        for slot in 0..MT_INELASTIC_COUNT {
            let off = slot * n_coarse;
            let mut i_start = 0usize;
            let mut i_end = 0usize;
            let mut found = false;
            for i in 0..n_coarse {
                if nuc_xs_per_mt[off + i] != 0.0 {
                    if !found {
                        i_start = i;
                        found = true;
                    }
                    i_end = i + 1;
                }
            }
            if found {
                pool.permt_i_start.push(i_start as u32);
                pool.permt_n_stored.push((i_end - i_start) as u32);
                pool.xs_inelastic_per_mt_sparse
                    .extend_from_slice(&nuc_xs_per_mt[off + i_start..off + i_end]);
                pool.yield_per_mt_sparse
                    .extend_from_slice(&nuc_yield_per_mt[off + i_start..off + i_end]);
            } else {
                pool.permt_i_start.push(0);
                pool.permt_n_stored.push(0);
            }
        }
        pool.q_inelastic_per_mt.extend_from_slice(&nuc_q_per_mt);

        // Per-MT distribution buffers for THIS nuclide (its own reactions, not
        // a material blend). Same builders `extract_material_xs` calls.
        let get = |mt: i32| reactions.get(&mt).map(|arc| arc.as_ref());
        let (a_ne, a_eg, a_nmu, a_mu, a_cdf, a_pdf, a_interp, a_cm) =
            build_per_mt_angle_buffers(get)?;
        pool.angle_n_energies.extend_from_slice(&a_ne);
        pool.angle_energy_grid.extend_from_slice(&a_eg);
        pool.angle_n_mu.extend_from_slice(&a_nmu);
        pool.angle_mu.extend_from_slice(&a_mu);
        pool.angle_cdf.extend_from_slice(&a_cdf);
        pool.angle_pdf.extend_from_slice(&a_pdf);
        pool.angle_interp.extend_from_slice(&a_interp);
        pool.scatter_in_cm.extend_from_slice(&a_cm);

        let (e_kind, e_ne, e_hist, e_eg, e_nx, e_x, e_p, e_cdf, e_interp, e_ndisc) =
            build_per_mt_eout_buffers(get);
        pool.eout_kind.extend_from_slice(&e_kind);
        pool.eout_n_energies.extend_from_slice(&e_ne);
        pool.eout_histogram_interp.extend_from_slice(&e_hist);
        pool.eout_energy_grid.extend_from_slice(&e_eg);
        pool.eout_n_x.extend_from_slice(&e_nx);
        pool.eout_x.extend_from_slice(&e_x);
        pool.eout_p.extend_from_slice(&e_p);
        pool.eout_cdf.extend_from_slice(&e_cdf);
        pool.eout_interp.extend_from_slice(&e_interp);
        pool.eout_n_discrete.extend_from_slice(&e_ndisc);

        let (
            c_ne,
            c_nc,
            c_eg,
            c_nx,
            c_x,
            c_cdf,
            c_p,
            c_interp,
            c_ndisc,
            c_nmu,
            c_mu,
            c_mu_cdf,
            c_mu_pdf,
            c_mu_interp,
        ) = build_per_mt_corr_buffers(get);
        pool.corr_n_energies.extend_from_slice(&c_ne);
        pool.corr_n_components.extend_from_slice(&c_nc);
        pool.corr_energy_grid.extend_from_slice(&c_eg);
        pool.corr_n_x.extend_from_slice(&c_nx);
        pool.corr_x.extend_from_slice(&c_x);
        pool.corr_cdf.extend_from_slice(&c_cdf);
        pool.corr_p.extend_from_slice(&c_p);
        pool.corr_interp.extend_from_slice(&c_interp);
        pool.corr_n_discrete.extend_from_slice(&c_ndisc);
        pool.corr_n_mu.extend_from_slice(&c_nmu);
        pool.corr_mu.extend_from_slice(&c_mu);
        pool.corr_mu_cdf.extend_from_slice(&c_mu_cdf);
        pool.corr_mu_pdf.extend_from_slice(&c_mu_pdf);
        pool.corr_mu_interp.extend_from_slice(&c_mu_interp);

        let (k_ne, k_eg, k_interp, k_ndisc, k_nx, k_x, k_p, k_c, k_r, k_a) =
            build_per_mt_km_buffers(get);
        pool.km_n_energies.extend_from_slice(&k_ne);
        pool.km_energy_grid.extend_from_slice(&k_eg);
        pool.km_interp.extend_from_slice(&k_interp);
        pool.km_n_discrete.extend_from_slice(&k_ndisc);
        pool.km_n_x.extend_from_slice(&k_nx);
        pool.km_x.extend_from_slice(&k_x);
        pool.km_p.extend_from_slice(&k_p);
        pool.km_c.extend_from_slice(&k_c);
        pool.km_r.extend_from_slice(&k_r);
        pool.km_a.extend_from_slice(&k_a);

        let (ev_ne, ev_nc, ev_eg, ev_theta, ev_u) = build_per_mt_evap_buffers(get);
        pool.evap_n_energies.extend_from_slice(&ev_ne);
        pool.evap_n_components.extend_from_slice(&ev_nc);
        pool.evap_energy_grid.extend_from_slice(&ev_eg);
        pool.evap_theta.extend_from_slice(&ev_theta);
        pool.evap_u.extend_from_slice(&ev_u);

        let (mx_ne, mx_eg, mx_theta, mx_u) = build_per_mt_maxwell_buffers(get);
        pool.maxwell_n_energies.extend_from_slice(&mx_ne);
        pool.maxwell_energy_grid.extend_from_slice(&mx_eg);
        pool.maxwell_theta.extend_from_slice(&mx_theta);
        pool.maxwell_u.extend_from_slice(&mx_u);

        let (w_ne, w_eg, w_a, w_b, w_u) = build_per_mt_watt_buffers(get);
        pool.watt_n_energies.extend_from_slice(&w_ne);
        pool.watt_energy_grid.extend_from_slice(&w_eg);
        pool.watt_a.extend_from_slice(&w_a);
        pool.watt_b.extend_from_slice(&w_b);
        pool.watt_u.extend_from_slice(&w_u);

        let (nb_n, nb_mass) = build_per_mt_nbps_buffers(get);
        pool.nbps_n_bodies.extend_from_slice(&nb_n);
        pool.nbps_total_mass.extend_from_slice(&nb_mass);
    }

    Ok(pool)
}

/// Single-nuclide Watt-parameter extraction for `extract_xs_from_nuclide`.
/// Same semantics as `extract_watt_params` but takes one nuclide.
fn extract_watt_params_single(nuclide: &Nuclide, temperature: &str) -> Option<(f64, f64)> {
    extract_watt_params(&[(nuclide, 1.0)], temperature)
}

/// Pull Watt-spectrum `(a, b)` parameters from the first fissionable
/// nuclide that has a Watt-shaped prompt-neutron product on its MT 18
/// (or 19/20/21/38). `a` is returned in eV; `b` in 1/eV. Both are
/// evaluated at a representative `E_in = 1 MeV`. Returns `None` if no
/// nuclide in the material is fissionable, or none has a Watt product.
fn extract_watt_params(nuclides: &[(&Nuclide, f64)], temperature: &str) -> Option<(f64, f64)> {
    const REPRESENTATIVE_E_IN: f64 = 1.0e6; // 1 MeV
    for (nuclide, _density) in nuclides {
        if !nuclide.fissionable {
            continue;
        }
        let temp_idx = nuclide.get_temp_idx(temperature)?;
        let reactions = &nuclide.reactions[temp_idx];
        for fmt in &[18, 19, 20, 21, 38] {
            let Some(rxn) = reactions.get(fmt) else {
                continue;
            };
            for product in &rxn.products {
                if !product.is_particle_type(&yamc_nuclide::particle_type::ParticleType::Neutron) {
                    continue;
                }
                for ae in &product.distribution {
                    if let AngleEnergyDistribution::UncorrelatedAngleEnergy {
                        energy: Some(EnergyDistribution::Watt { a, b, .. }),
                        ..
                    } = ae
                    {
                        let a_val = a.evaluate(REPRESENTATIVE_E_IN);
                        let b_val = b.evaluate(REPRESENTATIVE_E_IN);
                        if a_val > 0.0 && b_val > 0.0 {
                            return Some((a_val, b_val));
                        }
                    }
                }
            }
        }
    }
    None
}

/// Per-material fission outgoing-energy table extracted from the first
/// fissionable nuclide's first-found fission MT (18/19/20/21/38).
/// Mirrors `EoutSlot`'s flat layout but per-material rather than per-MT
/// slot -- fission collapses to a single branch on GPU. `kind` records
/// which sampler the kernel should use:
/// - `EOUT_KIND_CONTINUOUS_TABULAR` (1): sample from `(x, cdf)`
/// - `EOUT_KIND_WATT` (7): fall back to Watt rejection using
///   `fission_watt_a` / `fission_watt_b`
///
/// Taking the FIRST fission MT that carries a neutron product is exact for the
/// 87 ENDF/B-VIII.1 fissionables that have MT 18 alone, and wrong for U240, the
/// one nuclide with partial channels. U240's MT 18 is redundant and carries no
/// neutron product, so the walk lands on MT 19 and every U240 fission on the GPU
/// uses first-chance fission's spectrum regardless of energy. The CPU samples the
/// channel that actually fissioned and keys its chi cache per MT (issues #418,
/// #425), so the two backends disagree on U240 until this table is per MT as
/// well. See issue #424.
///
/// `n_energies == 0` means no usable distribution was found (or the
/// nuclide is non-fissionable) -- kernel never enters the fission
/// branch in that case so the buffer is unused.
///
/// `p` carries the normalized PDF alongside `cdf` (same shape, same
/// `cdf_max` normalization) and `interp` the per-incident-energy-row
/// interpolation code (`0` histogram, `1` lin-lin), so the sampler can
/// do the interp-aware within-bin inversion instead of the biased
/// linear-in-c one. Rows whose source table lacks a usable PDF get
/// `p == 0.0` per point and `interp == 0`, which routes the sampler to
/// the legacy linear-in-c fallback.
struct FissionEoutSlot {
    kind: u32,
    n_energies: u32,
    energy_grid: Vec<f64>, // length n_energies (tight, issue #104)
    n_x: Vec<u32>,         // length n_energies
    x: Vec<f64>,           // length sum(n_x) (tight CSR, issue #104)
    cdf: Vec<f64>,         // length sum(n_x)
    p: Vec<f64>,           // length sum(n_x), normalized like `cdf`
    interp: Vec<u32>,      // length n_energies (0 histogram, 1 lin-lin)
}

impl FissionEoutSlot {
    /// Default slot: Watt fallback, no continuum data. Tight CSR
    /// (issue #104): an empty slot has zero rows and zero points, so
    /// every Vec is empty (`n_energies == 0`).
    fn empty() -> Self {
        Self {
            kind: EOUT_KIND_WATT,
            n_energies: 0,
            energy_grid: Vec::new(),
            n_x: Vec::new(),
            x: Vec::new(),
            cdf: Vec::new(),
            p: Vec::new(),
            interp: Vec::new(),
        }
    }

    /// Walk the per-material nuclide list, find the first fissionable
    /// nuclide, and pull the prompt-neutron product's outgoing-energy
    /// spectrum from its dominant fission MT.
    ///
    /// The prompt fission spectrum lives on the FIRST neutron product of
    /// the fission reaction (matching the CPU's `sample_fission_neutrons`,
    /// which samples `products[0]`). Four encodings appear in ENDF/B-VIII.1:
    /// `UncorrelatedAngleEnergy/ContinuousTabular` (U235, U238, Pu239),
    /// `CorrelatedAngleEnergy` (Th232, the energy-angle-correlated prompt
    /// spectrum), and `UncorrelatedAngleEnergy/{Maxwell,Evaporation}` (the
    /// closed-form fission spectra; Maxwell χ appears on Pu241, U237,
    /// Am244, Pu243, Pu245, Ra223, Ra226 in ENDF/B-VIII.1). Fission mu is
    /// sampled isotropically in lab on the GPU regardless, so for the
    /// correlated case only the E_out marginal is needed; it is packed
    /// into the same ContinuousTabular `fission_eout_*` buffers. The
    /// ContinuousTabular / Correlated cases route through the kernel's
    /// interp-aware fission E_out sampler; Maxwell / Evaporation route to
    /// the kernel's shared `maxwell_rejection_draw` /
    /// `evaporation_rejection_draw` helpers (the same code the inelastic
    /// path uses) with the tabulated θ(E_in) and restriction energy `u`
    /// packed into the otherwise-unused `fission_eout_x` (θ, column 0) and
    /// `fission_eout_cdf` (u, column 0) buffers -- no new GPU bindings.
    /// Returns the Watt-fallback empty slot if no recognised encoding is
    /// found.
    ///
    /// Earlier this scanned EVERY neutron product for the first
    /// ContinuousTabular match, so for Th232 (whose `products[0]` is
    /// `CorrelatedAngleEnergy`) it silently fell through to a LATER
    /// product carrying a much softer partial spectrum -- the GPU then
    /// emitted no prompt fission neutrons above the incident energy,
    /// producing a ~10% flux-weighted spectral error vs the CPU.
    fn from_nuclides(nuclides: &[(&Nuclide, f64)], temperature: &str) -> Self {
        for (nuclide, _density) in nuclides {
            if !nuclide.fissionable {
                continue;
            }
            let Some(temp_idx) = nuclide.get_temp_idx(temperature) else {
                continue;
            };
            let reactions = &nuclide.reactions[temp_idx];
            for fmt in &[18, 19, 20, 21, 38] {
                let Some(rxn) = reactions.get(fmt) else {
                    continue;
                };
                // The prompt spectrum is the FIRST neutron product, to
                // mirror the CPU's `products[0]` selection.
                let Some(product) = rxn.products.iter().find(|p| {
                    p.is_particle_type(&yamc_nuclide::particle_type::ParticleType::Neutron)
                }) else {
                    continue;
                };
                for ae in &product.distribution {
                    match ae {
                        AngleEnergyDistribution::UncorrelatedAngleEnergy {
                            energy:
                                Some(EnergyDistribution::ContinuousTabular {
                                    energy, energy_out, ..
                                }),
                            ..
                        } => {
                            return Self::from_continuous_tabular(energy, energy_out);
                        }
                        AngleEnergyDistribution::UncorrelatedAngleEnergy {
                            energy: Some(EnergyDistribution::Maxwell { theta, u }),
                            ..
                        } => {
                            let Tabulated1D::Tabulated1D { x, y, .. } = theta;
                            return Self::from_maxwell_or_evap(EOUT_KIND_MAXWELL, x, y, *u);
                        }
                        AngleEnergyDistribution::UncorrelatedAngleEnergy {
                            energy: Some(EnergyDistribution::Evaporation { theta, u }),
                            ..
                        } => {
                            let Tabulated1D::Tabulated1D { x, y, .. } = theta;
                            return Self::from_maxwell_or_evap(EOUT_KIND_EVAPORATION, x, y, *u);
                        }
                        AngleEnergyDistribution::CorrelatedAngleEnergy { correlated } => {
                            return Self::from_correlated(correlated);
                        }
                        _ => {}
                    }
                }
            }
        }
        Self::empty()
    }

    /// Walk the per-material nuclide list for the first fissionable nuclide with
    /// DELAYED neutron data, and pack its yield-weighted folded delayed spectrum
    /// (issue #364).
    ///
    /// Takes the first nuclide that has any, mirroring how `from_nuclides` takes
    /// the first fissionable nuclide's prompt spectrum, so a material's prompt and
    /// delayed rows come from the same nuclide whenever one nuclide dominates. The
    /// fold is `ContinuousTabular` by construction, so it reuses the same packing.
    /// Returns the empty slot when no nuclide has delayed data; `beta == 0` then
    /// keeps the kernel from ever reading it.
    fn delayed_from_nuclides(nuclides: &[(&Nuclide, f64)], temperature: &str) -> Self {
        for (nuclide, _density) in nuclides {
            if !nuclide.fissionable {
                continue;
            }
            let Some(delayed) = nuclide.delayed_neutrons(temperature) else {
                continue;
            };
            if let FissionChiFlat::Continuous {
                energy_grid,
                n_x,
                interp,
                x,
                p,
                c,
                max_x,
                ..
            } = delayed.chi_flat()
            {
                return Self::from_flat_continuous(energy_grid, n_x, interp, x, p, c, *max_x);
            }
        }
        Self::empty()
    }

    /// Pack an already-flattened `FissionChiFlat::Continuous` (stride `max_x` per
    /// incident row) into the tight CSR `fission_eout_*` layout.
    fn from_flat_continuous(
        energy_grid: &[f64],
        n_x: &[u32],
        interp: &[u32],
        x: &[f64],
        p: &[f64],
        c: &[f64],
        max_x: usize,
    ) -> Self {
        if energy_grid.is_empty() || max_x == 0 {
            return Self::empty();
        }
        let mut slot = Self::empty();
        slot.kind = EOUT_KIND_CONTINUOUS_TABULAR;
        slot.n_energies = energy_grid.len() as u32;
        slot.energy_grid = energy_grid.to_vec();
        slot.n_x = n_x.to_vec();
        slot.interp = interp.to_vec();
        for (row, &n) in n_x.iter().enumerate() {
            let base = row * max_x;
            let n = n as usize;
            slot.x.extend_from_slice(&x[base..base + n]);
            slot.cdf.extend_from_slice(&c[base..base + n]);
            slot.p.extend_from_slice(&p[base..base + n]);
        }
        slot
    }

    /// Pack a `CorrelatedAngleEnergy` prompt-fission spectrum's E_out
    /// marginal into the ContinuousTabular `fission_eout_*` buffers,
    /// dropping the per-(E_in, E_out) angular sub-tables (fission mu is
    /// isotropic in lab on the GPU). Tight CSR (issue #104): keeps every
    /// incident-energy point and every E_out point (no subsampling); rows
    /// are concatenated back-to-back, `n_x[i]` recording each row's length.
    /// Same CDF-normalisation as `from_continuous_tabular`, reading the
    /// correlated table's per-incident-energy `(e_out, p, c)` arrays.
    fn from_correlated(corr: &yamc_nuclide::secondary_correlated::CorrelatedAngleEnergy) -> Self {
        if corr.energy.is_empty() || corr.distributions.is_empty() {
            return Self::empty();
        }
        let n_e = corr.energy.len();

        let mut slot = Self::empty();
        slot.kind = EOUT_KIND_CONTINUOUS_TABULAR;
        slot.n_energies = n_e as u32;
        slot.energy_grid = Vec::with_capacity(n_e);
        slot.n_x = Vec::with_capacity(n_e);
        for src_i in 0..n_e {
            slot.energy_grid.push(corr.energy[src_i]);
            if src_i >= corr.distributions.len() {
                slot.n_x.push(0);
                slot.interp.push(0);
                continue;
            }
            let table = &corr.distributions[src_i];
            let m_in = table.e_out.len();
            if m_in == 0 {
                slot.n_x.push(0);
                slot.interp.push(0);
                continue;
            }
            let cdf_owned: Vec<f64>;
            let cdf_slice: &[f64] = if table.c.len() == m_in {
                &table.c
            } else {
                cdf_owned = trapezoidal_cdf(&table.e_out, &table.p);
                &cdf_owned
            };
            let cdf_max = cdf_slice.last().copied().unwrap_or(0.0);
            // PDF + row interp code for the interp-aware inversion. A row
            // without a usable PDF (wrong length or degenerate CDF) stores
            // p == 0.0 per point and forces histogram (0) so the sampler
            // keeps the legacy linear-in-c behaviour.
            let has_pdf = table.p.len() == m_in && cdf_max > 0.0;
            slot.interp.push(if has_pdf {
                match table.interpolation {
                    yamc_nuclide::secondary_correlated::Interpolation::Histogram => 0,
                    yamc_nuclide::secondary_correlated::Interpolation::LinLin => 1,
                }
            } else {
                0
            });

            slot.n_x.push(m_in as u32);
            for (j, (&e_out, &c)) in table.e_out.iter().zip(cdf_slice.iter()).enumerate() {
                slot.x.push(e_out);
                slot.cdf.push(if cdf_max > 0.0 {
                    c / cdf_max
                } else {
                    j as f64 / (m_in - 1).max(1) as f64
                });
                slot.p
                    .push(if has_pdf { table.p[j] / cdf_max } else { 0.0 });
            }
        }
        slot
    }

    /// Same normalisation as `EoutSlot::from_continuous_tabular` -- the
    /// kernel reuses the existing linear-in-c inelastic eout sampler for
    /// the fission branch so the buffer encoding matches. Tight CSR
    /// (issue #104): keeps the full incident-energy and E_out resolution
    /// (U235 MT18 has max_n_x = 643, Pu239 = 642, both above the old 512
    /// cap, so the full spectrum is now retained); rows are concatenated
    /// back-to-back with `n_x[i]` recording each row's length.
    fn from_continuous_tabular(energy: &[f64], energy_out: &[TabulatedProbability]) -> Self {
        if energy.is_empty() || energy_out.is_empty() {
            return Self::empty();
        }
        let n_e = energy.len();

        let mut slot = Self::empty();
        slot.kind = EOUT_KIND_CONTINUOUS_TABULAR;
        slot.n_energies = n_e as u32;
        slot.energy_grid = Vec::with_capacity(n_e);
        slot.n_x = Vec::with_capacity(n_e);
        for src_i in 0..n_e {
            slot.energy_grid.push(energy[src_i]);
            if src_i >= energy_out.len() {
                slot.n_x.push(0);
                slot.interp.push(0);
                continue;
            }
            let TabulatedProbability::Tabulated {
                x, p, c, interp, ..
            } = &energy_out[src_i];
            let m_in = x.len();
            if m_in == 0 {
                slot.n_x.push(0);
                slot.interp.push(0);
                continue;
            }
            let cdf_owned: Vec<f64>;
            let cdf_slice: &[f64] = if c.len() == m_in {
                c
            } else {
                cdf_owned = trapezoidal_cdf(x, p);
                &cdf_owned
            };
            let cdf_max = cdf_slice.last().copied().unwrap_or(0.0);
            // PDF + row interp code for the interp-aware inversion. A row
            // without a usable PDF (wrong length or degenerate CDF) stores
            // p == 0.0 per point and forces histogram (0) so the sampler
            // keeps the legacy linear-in-c behaviour.
            let has_pdf = p.len() == m_in && cdf_max > 0.0;
            slot.interp.push(if has_pdf {
                match interp {
                    TabulatedInterp::Histogram => 0,
                    TabulatedInterp::LinLin => 1,
                }
            } else {
                0
            });

            slot.n_x.push(m_in as u32);
            for (j, (&x_j, &c)) in x.iter().zip(cdf_slice.iter()).enumerate() {
                slot.x.push(x_j);
                slot.cdf.push(if cdf_max > 0.0 {
                    c / cdf_max
                } else {
                    j as f64 / (m_in - 1).max(1) as f64
                });
                slot.p.push(if has_pdf { p[j] / cdf_max } else { 0.0 });
            }
        }
        slot
    }

    /// Pack a Maxwell (ENDF File 5, Law 7) or Evaporation (Law 9) prompt-
    /// fission χ into the `fission_eout_*` buffers without adding any new
    /// GPU storage bindings. The fission χ's E_out is closed-form
    /// (`p(E) ∝ √E·exp(-E/θ)` for Maxwell, `p(E) ∝ E·exp(-E/θ)` for
    /// Evaporation, both capped at `E_in - u`), so there is no tabulated
    /// `(E_out, cdf)` table to store -- only the tabulated temperature
    /// `θ(E_in)` and the scalar restriction energy `u`. Tight CSR
    /// (issue #104): each E_in row carries a SINGLE point (`n_x[i] == 1`),
    /// so the tight `x` / `cdf` arrays have one entry per row. The slot
    /// reuses:
    ///   `kind`              -- `EOUT_KIND_MAXWELL` / `_EVAPORATION`
    ///   `n_energies`        -- number of θ(E_in) grid points
    ///   `energy_grid[i]`    -- θ's incident-energy axis
    ///   `n_x[i] == 1`       -- one stored point per row
    ///   `x[x_off(i)]`       -- θ value at grid point i (the row's point)
    ///   `cdf[x_off(0)]`     -- restriction energy `u` (scalar, row-0 slot)
    /// where `x_off(i)` is the per-row CSR base. The kernel's fission
    /// branch interpolates θ off `energy_grid` / `x` (linear, mirroring
    /// the inelastic Maxwell/Evaporation path and `Tabulated1D::evaluate`'s
    /// lin-lin default -- fission θ is tabulated lin-lin) and feeds the
    /// shared rejection helpers. `u` is read once from the material's
    /// row-0 single slot; the CPU `EnergyDistribution::{Maxwell,Evaporation}`
    /// carries a single scalar `u`, so a per-slice grid is unnecessary.
    fn from_maxwell_or_evap(kind: u32, theta_x: &[f64], theta_y: &[f64], u: f64) -> Self {
        if theta_x.is_empty() || theta_y.is_empty() {
            return Self::empty();
        }
        let n_e = theta_x.len().min(theta_y.len());
        if n_e == 0 {
            return Self::empty();
        }
        let mut slot = Self::empty();
        slot.kind = kind;
        slot.n_energies = n_e as u32;
        slot.energy_grid = Vec::with_capacity(n_e);
        // One stored point per E_in row (tight CSR): θ in `x`, and `u` in
        // the material's row-0 `cdf` slot (the rest of `cdf` stays zero).
        // `p` / `interp` are unused by the rejection samplers; keep them
        // zero-filled and shape-parallel to `x` / `n_x`.
        slot.n_x = vec![1u32; n_e];
        slot.x = Vec::with_capacity(n_e);
        slot.cdf = vec![0.0; n_e];
        slot.p = vec![0.0; n_e];
        slot.interp = vec![0u32; n_e];
        for i in 0..n_e {
            slot.energy_grid.push(theta_x[i]);
            slot.x.push(theta_y[i]);
        }
        slot.cdf[0] = u;
        slot
    }
}

/// Build a single-nuclide fission eout slot for `extract_xs_from_nuclide`.
fn extract_fission_eout_single(nuclide: &Nuclide, temperature: &str) -> FissionEoutSlot {
    FissionEoutSlot::from_nuclides(&[(nuclide, 1.0)], temperature)
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_inelastic_across_slots_sums_per_energy() {
        // n_grid = 2; put data in slots 0 and 1, leave the rest zero.
        let n_grid = 2;
        let mut per_mt = vec![0.0_f64; MT_INELASTIC_COUNT * n_grid];
        per_mt[0] = 1.0; // slot 0, e0
        per_mt[1] = 2.0; // slot 0, e1
        per_mt[n_grid] = 10.0; // slot 1, e0
        per_mt[n_grid + 1] = 20.0; // slot 1, e1
        assert_eq!(
            aggregate_inelastic_across_slots(&per_mt, n_grid),
            vec![11.0, 22.0]
        );
    }

    #[test]
    fn derive_absorption_subtracts_channels_and_clamps() {
        let total = [10.0, 5.0, 1.0];
        let elastic = [2.0, 1.0, 1.0];
        let inelastic = [1.0, 1.0, 1.0];
        let fission = [1.0, 0.0, 1.0];
        // e2: 1 - 1 - 1 - 1 = -2 -> clamped to 0.
        assert_eq!(
            derive_absorption(&total, &elastic, &inelastic, &fission),
            vec![6.0, 3.0, 0.0]
        );
    }

    #[test]
    fn nu_bar_from_nu_sigma_f_divides_or_zero() {
        let nu_sigma_f = [5.0, 0.0, 7.0];
        let xs_fission = [2.0, 0.0, 0.0];
        // e1 + e2 have zero fission xs -> nu_bar 0 (no division).
        assert_eq!(
            nu_bar_from_nu_sigma_f(&nu_sigma_f, &xs_fission),
            vec![2.5, 0.0, 0.0]
        );
    }

    /// A `CorrelatedAngleEnergy` prompt-fission spectrum (Th232's MT 18
    /// `products[0]` encoding) must extract into a populated
    /// ContinuousTabular fission slot carrying the full outgoing-energy
    /// range -- NOT fall through to the Watt-empty default. Regression
    /// guard for the bug where the GPU fission extraction only matched
    /// `UncorrelatedAngleEnergy/ContinuousTabular` and so emitted no
    /// prompt fission neutrons above the incident energy for Th232.
    #[test]
    fn fission_eout_from_correlated_keeps_full_spectrum() {
        use yamc_nuclide::secondary_correlated::{CorrTable, CorrelatedAngleEnergy, Interpolation};

        // Two incident energies, each a 4-point fission-like spectrum
        // reaching 8 MeV -- well above any single incident energy.
        let table = |scale: f64| CorrTable {
            interpolation: Interpolation::LinLin,
            n_discrete: 0,
            e_out: vec![1.0e4, 1.0e6, 4.0e6, 8.0e6],
            p: vec![0.0, 1.0e-6 * scale, 4.0e-7 * scale, 0.0],
            c: vec![0.0, 0.45, 0.9, 1.0],
            angle: vec![],
        };
        let corr = CorrelatedAngleEnergy {
            energy: vec![1.0e6, 2.0e6],
            distributions: vec![table(1.0), table(1.1)],
        };

        let slot = FissionEoutSlot::from_correlated(&corr);
        assert_eq!(slot.kind, EOUT_KIND_CONTINUOUS_TABULAR);
        assert_eq!(slot.n_energies, 2);
        // First incident slice keeps all 4 outgoing points.
        assert_eq!(slot.n_x[0], 4);
        // Highest outgoing energy survives (8 MeV) -- the prompt spectrum
        // extends above the incident energy, which the pre-fix path lost.
        assert_eq!(slot.x[3], 8.0e6);
        // CDF is normalised to end at 1.0.
        assert!((slot.cdf[3] - 1.0).abs() < 1e-12);
    }

    /// A Maxwell prompt-fission χ (the encoding Pu241 / U237 / Am244 /
    /// Pu243 / Pu245 / Ra223 / Ra226 carry in ENDF/B-VIII.1) must pack
    /// `kind = EOUT_KIND_MAXWELL`, the tabulated θ(E_in) into
    /// `fission_eout_x` column 0, and the scalar restriction energy `u`
    /// into `fission_eout_cdf[0]` -- NOT fall through to the Watt-empty
    /// default. Regression guard for the pre-fix path that coerced every
    /// Maxwell χ to the Watt fallback. Evaporation packs identically.
    #[test]
    fn fission_eout_from_maxwell_packs_theta_and_u() {
        // Pu241-like θ(E_in): 4 points, slightly rising; u negative
        // (the actual ENDF value, which makes the cap E_in - u huge).
        let theta_x = [1.0e-5, 1.0e6, 1.0e7, 2.0e7];
        let theta_y = [1.3597e6, 1.45e6, 1.55e6, 1.6049e6];
        let u = -3.0e7;

        let slot = FissionEoutSlot::from_maxwell_or_evap(EOUT_KIND_MAXWELL, &theta_x, &theta_y, u);
        assert_eq!(slot.kind, EOUT_KIND_MAXWELL);
        assert_eq!(slot.n_energies, 4);
        // θ is on the incident-energy grid.
        assert_eq!(slot.energy_grid[0], 1.0e-5);
        assert_eq!(slot.energy_grid[3], 2.0e7);
        // Tight CSR (issue #104): one stored point per E_in row, so θ
        // values are packed contiguously (row i at x[i]); `u` sits in the
        // material's row-0 cdf slot.
        assert_eq!(slot.n_x, vec![1u32; 4]);
        assert_eq!(slot.x[0], 1.3597e6);
        assert_eq!(slot.x[3], 1.6049e6);
        // Scalar u in cdf[0] (row-0 slot).
        assert_eq!(slot.cdf[0], -3.0e7);

        // Evaporation routes through the same packer with its own kind.
        let evap =
            FissionEoutSlot::from_maxwell_or_evap(EOUT_KIND_EVAPORATION, &theta_x, &theta_y, u);
        assert_eq!(evap.kind, EOUT_KIND_EVAPORATION);
        assert_eq!(evap.x[0], 1.3597e6);
        assert_eq!(evap.x[3], 1.6049e6);
        assert_eq!(evap.cdf[0], -3.0e7);
    }
}
