//! Tally scoring helpers: track-length segment, collision-event, and the
//! Woodcock flight (collision-density) estimator.

use super::*;

/// Score a single straight-line transport segment of length `dist` into
/// every per-history tally and the optional transmutation tally. Returns
/// the segment's end position so the caller can pass it to a tracker
/// (surface-crossing event) without recomputing.
///
/// Extracted from the inner transport loop in [`Model::simulate`] as
/// step 5 of Phase 1f. The same scoring code was duplicated in both
/// the surface-crossing and the collision branches; both branches now
/// call this helper. `#[inline(always)]` keeps the call-site free.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn score_track_length_segment(
    particle: &yamc_particle::particle::Particle,
    cell: &crate::geometry::cell::Cell,
    cell_material: Option<&Arc<Material>>,
    material_id: Option<u32>,
    dist: f64,
    urr_macro_xs: Option<&UrrMacroXs>,
    tallies: &[&Tally],
    transmutation_tallies: Option<&TransmutationTallies>,
    welford_worker: &mut yamc_tallies::welford::WelfordWorkerState,
) -> [f64; 3] {
    let material_ref = cell_material.map(|m| m.as_ref());
    let end_position = [
        particle.position[0] + particle.direction[0] * dist,
        particle.position[1] + particle.direction[1] * dist,
        particle.position[2] + particle.direction[2] * dist,
    ];
    for (tally_idx, tally) in tallies.iter().enumerate() {
        // Skip tallies that have opted into the collision estimator --
        // they get scored at collision sites, not at every cell crossing.
        if tally.estimator == yamc_tallies::Estimator::Collision {
            continue;
        }
        tally.score_track_length(
            welford_worker,
            tally_idx,
            dist,
            particle.weight,
            cell.cell_id,
            material_ref,
            material_id,
            particle.energy,
            end_position,
            particle.position,
            particle.direction,
            urr_macro_xs,
            particle.particle_type,
            particle.parent_nuclide,
        );
    }
    if let (Some(dep_tallies), Some(mat_id), Some(mat_ref)) =
        (transmutation_tallies, material_id, material_ref)
    {
        // Weighted track length: under survival biasing (and any future
        // weighted technique) the particle carries weight < 1, and the
        // transmutation rates must see it. Analog weight is 1, so this is
        // bit-identical for analog runs.
        dep_tallies.score(mat_id, particle.energy, dist * particle.weight, mat_ref);
    }
    end_position
}

/// Score all tallies that opted into `Estimator::Collision` at the
/// current particle position. Called once per neutron collision, right
/// after `particle.move_by(dist_collision)` lands the particle at the
/// collision site. The total macroscopic cross section `total_xs`
/// (Σ_t) is the URR-modified value if URR is active, else the smooth
/// material lookup; it's the same Σ_t the distance was sampled against,
/// so `weight / total_xs` is the unbiased collision-estimator flux
/// contribution.
#[allow(clippy::too_many_arguments)]
pub(crate) fn score_collision_event(
    particle: &yamc_particle::particle::Particle,
    cell: &crate::geometry::cell::Cell,
    material: &Material,
    material_id: Option<u32>,
    urr_macro_xs: Option<&UrrMacroXs>,
    photon_macro_xs: Option<&yamc_materials::MacroPhotonXS>,
    tallies: &[&Tally],
    welford_worker: &mut yamc_tallies::welford::WelfordWorkerState,
) {
    // Quick pre-check: avoid the lookup if no tally needs it.
    if !tallies
        .iter()
        .any(|t| t.estimator == yamc_tallies::Estimator::Collision)
    {
        return;
    }
    // Per-particle-type Σ_t: neutrons use the (URR-modified) MT-1 total
    // from the neutron table; photons use the macroscopic photon total.
    // Photons don't have neutron MT 1 in their XS table, so the old
    // `lookup_xs_by_mt(1, ...)` returned 0 and photons silently dropped
    // every collision-estimator contribution.
    let total_xs = match particle.particle_type {
        yamc_particle::ParticleType::Photon => photon_macro_xs.map_or(0.0, |x| x.total),
        _ => match urr_macro_xs {
            Some(u) => u.total,
            None => material.lookup_xs_by_mt(1, particle.energy),
        },
    };
    if total_xs <= 0.0 {
        return;
    }
    for (tally_idx, tally) in tallies.iter().enumerate() {
        if tally.estimator != yamc_tallies::Estimator::Collision {
            continue;
        }
        tally.score_collision(
            welford_worker,
            tally_idx,
            particle.weight,
            total_xs,
            particle.position,
            cell.cell_id,
            Some(material),
            material_id,
            urr_macro_xs,
            photon_macro_xs,
            particle.energy,
            particle.particle_type,
            particle.parent_nuclide,
            None, // analog photon heat -- supplied post-photon-collision instead
        );
    }
}

/// Woodcock (delta-tracking) flux estimator for `Estimator::TrackLength`
/// tallies, evaluated at one delta-collision.
///
/// Delta tracking never computes boundary crossings, so the classical
/// track-length estimator (weight x path-length-in-cell) is unavailable
/// -- there is no boundary distance to integrate against. The unbiased
/// equivalent is the collision-density estimator evaluated at every
/// delta-collision, real OR virtual: each one contributes
/// `weight x response / Sigma_maj`, where the response is the score's
/// cross section (1 for flux, Sigma_x for a reaction rate). The number
/// of delta-collisions along a path is Poisson with mean
/// `Sigma_maj x path`, so the expectation reproduces the track-length
/// integral -- with no boundary distance, and attributing each
/// contribution to the cell / mesh voxel that actually contains the
/// delta-collision point. Cross-cell flights, escapes (the post-flight
/// point is outside, so nothing is scored), and void-region flux are
/// therefore all handled exactly, removing the Phase 4.0 segment-
/// attribution bias.
///
/// Implemented by reusing [`Tally::score_collision`] with
/// `total_xs = Sigma_maj`. Only fires for TrackLength tallies;
/// Collision-estimator tallies score at real collisions via
/// [`score_collision_event`]. Tallies eligible per
/// [`woodcock_mesh_track_length_eligible`] are skipped here too: they
/// are scored with the true track-length estimator along each flight
/// segment via `score_woodcock_mesh_track_length` instead (issue
/// #350), so this collision-density path covers only the remaining
/// ineligible class.
#[allow(clippy::too_many_arguments)]
pub(crate) fn score_woodcock_flight_estimator(
    particle: &yamc_particle::particle::Particle,
    cell: &crate::geometry::cell::Cell,
    cell_material: Option<&Arc<Material>>,
    material_id: Option<u32>,
    sigma_maj: f64,
    urr_macro_xs: Option<&UrrMacroXs>,
    photon_macro_xs: Option<&yamc_materials::MacroPhotonXS>,
    tallies: &[&Tally],
    tl_mesh_eligible: &[bool],
    transmutation_tallies: Option<&TransmutationTallies>,
    welford_worker: &mut yamc_tallies::welford::WelfordWorkerState,
) {
    let material_ref = cell_material.map(|m| m.as_ref());
    // Zipped (not indexed) so this per-delta-collision loop carries no
    // bounds check on the eligibility slice.
    for (tally_idx, (tally, &mesh_eligible)) in tallies.iter().zip(tl_mesh_eligible).enumerate() {
        if tally.estimator != yamc_tallies::Estimator::TrackLength {
            continue;
        }
        if mesh_eligible {
            // Scored with the TRUE track-length estimator along the
            // full flight segment instead (issue #350).
            continue;
        }
        tally.score_collision(
            welford_worker,
            tally_idx,
            particle.weight,
            sigma_maj,
            particle.position,
            cell.cell_id,
            material_ref,
            material_id,
            urr_macro_xs,
            photon_macro_xs,
            particle.energy,
            particle.particle_type,
            particle.parent_nuclide,
            None,
        );
    }
    // Transmutation reaction rates: the delta-tracking equivalent of a
    // track-length contribution is `weight / Sigma_maj` per
    // delta-collision (weighted for survival biasing; analog weight is
    // 1, so analog runs are bit-identical).
    if let (Some(dep), Some(mid), Some(mref)) = (transmutation_tallies, material_id, material_ref) {
        dep.score(mid, particle.energy, particle.weight / sigma_maj, mref);
    }
}
