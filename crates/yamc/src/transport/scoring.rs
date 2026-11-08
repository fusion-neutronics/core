//! Tally scoring helpers: track-length segment, collision-event, and the
//! Woodcock flight (collision-density) estimator.
//!
//! Real collision scores use the arrival likelihood before the sampled
//! reaction updates it. The collision density's total cross-section ratio
//! cancels the estimator's inverse-total ratio; only the score response's
//! own ratio remains. Photon analog heating carries its inherited ratio.

use super::*;

/// Score a single straight-line transport segment of length `dist` into
/// every per-history tally and the optional transmutation tally. Returns
/// the segment's end position so the caller can pass it to a tracker
/// (surface-crossing event) without recomputing.
///
/// Shared by the surface-crossing and the collision branches of the
/// inner transport loop. `#[inline(always)]` keeps the call-site free.
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
    replica: Option<&super::replica::ReplicaSegment>,
) -> [f64; 3] {
    let material_ref = cell_material.map(|m| m.as_ref());
    let end_position = [
        particle.position[0] + particle.direction[0] * dist,
        particle.position[1] + particle.direction[1] * dist,
        particle.position[2] + particle.direction[2] * dist,
    ];
    for (tally_idx, tally) in tallies.iter().enumerate() {
        // Only track-length tallies score here: collision tallies at
        // collision sites, surface tallies at surface crossings.
        if tally.estimator != yamc_tallies::Estimator::TrackLength {
            continue;
        }
        if let Some(seg) = replica.filter(|_| welford_worker.has_replicas(tally_idx)) {
            // Each score's replica factors for this segment: the ratio and
            // the mean survival along it, times the score's own Σ'_x / Σ_x.
            let factors: Vec<Vec<f64>> = tally
                .scores
                .iter()
                .map(|score| seg.factors_for(score, particle.energy))
                .collect();
            tally.score_track_length_with(
                &mut |bin_idx, v| {
                    welford_worker.add_contribution(tally_idx, bin_idx, v);
                    let score = tally.score_index_of_bin(bin_idx);
                    welford_worker.add_replica_contribution(tally_idx, bin_idx, v, &factors[score]);
                },
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
    // Transmutation reaction rates are neutron rates: a photon's track in
    // the same material is not neutron flux.
    if let (Some(dep_tallies), Some(mat_id), Some(mat_ref), yamc_particle::ParticleType::Neutron) = (
        transmutation_tallies,
        material_id,
        material_ref,
        particle.particle_type,
    ) {
        // Weighted track length: under survival biasing (and any future
        // weighted technique) the particle carries weight < 1, and the
        // transmutation rates must see it. Analog weight is 1, so this is
        // bit-identical for analog runs.
        match replica.filter(|_| dep_tallies.replicas().is_some()) {
            // Every replica's score too, under the segment's replica factors.
            Some(seg) => dep_tallies.score_with_replicas(
                mat_id,
                particle.energy,
                dist * particle.weight,
                mat_ref,
                seg.track_factors(),
                seg.context(),
            ),
            None => dep_tallies.score(mat_id, particle.energy, dist * particle.weight, mat_ref),
        }
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
    replica: Option<&super::replica::ReplicaContext>,
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
    let point = replica.and_then(|context| {
        super::replica::ReplicaSegment::at_point(context, particle, cell.material_idx)
    });
    for (tally_idx, tally) in tallies.iter().enumerate() {
        if tally.estimator != yamc_tallies::Estimator::Collision {
            continue;
        }
        if let Some(point) = point
            .as_ref()
            .filter(|_| welford_worker.has_replicas(tally_idx))
        {
            let factors: Vec<Vec<f64>> = tally
                .scores
                .iter()
                .map(|score| point.factors_for(score, particle.energy))
                .collect();
            tally.score_collision_with(
                &mut |bin_idx, value| {
                    welford_worker.add_contribution(tally_idx, bin_idx, value);
                    let score = tally.score_index_of_bin(bin_idx);
                    welford_worker.add_replica_contribution(
                        tally_idx,
                        bin_idx,
                        value,
                        &factors[score],
                    );
                },
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
                None,
            );
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn score_photon_heat_event(
    particle: &yamc_particle::particle::Particle,
    cell: &crate::geometry::cell::Cell,
    material: &Material,
    material_id: Option<u32>,
    incoming_weight: f64,
    incoming_energy: f64,
    heat_score: f64,
    tallies: &[&Tally],
    welford_worker: &mut yamc_tallies::welford::WelfordWorkerState,
    replica: Option<&super::replica::ReplicaContext>,
) {
    let point = replica.and_then(|context| {
        super::replica::ReplicaSegment::at_point(context, particle, cell.material_idx)
    });
    for (tally_idx, tally) in tallies.iter().enumerate() {
        if tally.estimator != yamc_tallies::Estimator::Collision {
            continue;
        }
        if let Some(point) = point
            .as_ref()
            .filter(|_| welford_worker.has_replicas(tally_idx))
        {
            let factors: Vec<Vec<f64>> = tally
                .scores
                .iter()
                .map(|score| point.factors_for(score, incoming_energy))
                .collect();
            tally.score_collision_with(
                &mut |bin_idx, value| {
                    welford_worker.add_contribution(tally_idx, bin_idx, value);
                    let score = tally.score_index_of_bin(bin_idx);
                    welford_worker.add_replica_contribution(
                        tally_idx,
                        bin_idx,
                        value,
                        &factors[score],
                    );
                },
                incoming_weight,
                0.0,
                particle.position,
                cell.cell_id,
                Some(material),
                material_id,
                None,
                None,
                incoming_energy,
                yamc_particle::ParticleType::Photon,
                particle.parent_nuclide,
                Some(heat_score),
            );
        } else {
            tally.score_collision(
                welford_worker,
                tally_idx,
                incoming_weight,
                0.0,
                particle.position,
                cell.cell_id,
                Some(material),
                material_id,
                None,
                None,
                incoming_energy,
                yamc_particle::ParticleType::Photon,
                particle.parent_nuclide,
                Some(heat_score),
            );
        }
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
/// therefore all handled exactly, with none of the bias of attributing a
/// whole flight segment to a single cell.
///
/// Implemented by reusing [`Tally::score_collision`] with
/// `total_xs = Sigma_maj`. Only fires for TrackLength tallies;
/// Collision-estimator tallies score at real collisions via
/// [`score_collision_event`]. Tallies eligible per
/// [`woodcock_mesh_track_length_eligible`] are skipped here too: they
/// are scored with the true track-length estimator along each flight
/// segment via `score_woodcock_mesh_track_length` instead, so this
/// collision-density path covers only the remaining ineligible class.
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
    replica: Option<&super::replica::ReplicaSegment>,
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
            // full flight segment instead.
            continue;
        }
        if let Some(point) = replica.filter(|_| welford_worker.has_replicas(tally_idx)) {
            // Each score's replica factors at the delta collision: the
            // particle's ratio times the score's own Σ'_x / Σ_x.
            let factors: Vec<Vec<f64>> = tally
                .scores
                .iter()
                .map(|score| point.factors_for(score, particle.energy))
                .collect();
            tally.score_collision_with(
                &mut |bin_idx, v| {
                    welford_worker.add_contribution(tally_idx, bin_idx, v);
                    let score = tally.score_index_of_bin(bin_idx);
                    welford_worker.add_replica_contribution(tally_idx, bin_idx, v, &factors[score]);
                },
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
    if let (Some(dep), Some(mid), Some(mref), yamc_particle::ParticleType::Neutron) = (
        transmutation_tallies,
        material_id,
        material_ref,
        particle.particle_type,
    ) {
        let weighted = particle.weight / sigma_maj;
        match replica.filter(|_| dep.replicas().is_some()) {
            Some(point) => dep.score_with_replicas(
                mid,
                particle.energy,
                weighted,
                mref,
                point.track_factors(),
                point.context(),
            ),
            None => dep.score(mid, particle.energy, weighted, mref),
        }
    }
}

/// The surfaces that surface-current tallies score on, looked up by the id a
/// boundary crossing reports. Built once per run, and only when some tally
/// scores surfaces, so the crossing hot path pays one `None` check otherwise.
pub(crate) struct SurfaceScoring {
    surfaces: std::collections::HashMap<usize, Arc<crate::geo::Surface>>,
}

impl SurfaceScoring {
    /// Collect the surfaces the surface tallies name, or `None` when no tally
    /// scores surfaces. A surface id that is not in the geometry, or a mesh
    /// geometry (whose crossings are triangles, not the user's surfaces), is
    /// refused here, before any transport.
    pub(crate) fn build(
        geometry: &GeometryKind,
        tallies: &[&Tally],
    ) -> Result<Option<Self>, String> {
        let wanted: std::collections::BTreeSet<usize> = tallies
            .iter()
            .filter_map(|t| t.get_surface_filter())
            .flat_map(|f| f.surface_ids.iter().copied())
            .collect();
        if wanted.is_empty() {
            return Ok(None);
        }
        let csg = match geometry {
            GeometryKind::Csg(csg) => csg,
            #[cfg(feature = "mesh")]
            GeometryKind::Mesh(_) => {
                return Err(
                    "surface tallies score on CSG surfaces; this model's geometry is a \
                            mesh, so tally a cell instead, or build the boundary as a CSG surface"
                        .to_string(),
                );
            }
        };
        let mut surfaces = std::collections::HashMap::new();
        for cell in &csg.cells {
            for (surface, _sense) in cell.region.surfaces_with_sense() {
                if let Some(id) = surface.surface_id
                    && wanted.contains(&id)
                {
                    surfaces.entry(id).or_insert(surface);
                }
            }
        }
        if let Some(missing) = wanted.iter().find(|id| !surfaces.contains_key(id)) {
            return Err(format!(
                "a surface tally names surface id {missing}, but no cell in the geometry is \
                 bounded by it"
            ));
        }
        Ok(Some(Self { surfaces }))
    }

    /// Score a crossing into every surface tally, if the crossed surface is
    /// one they name. `point` is where the particle meets the surface.
    #[inline]
    pub(crate) fn score_crossing(
        &self,
        particle: &yamc_particle::particle::Particle,
        surface_id: Option<usize>,
        point: [f64; 3],
        tallies: &[&Tally],
        welford_worker: &mut yamc_tallies::welford::WelfordWorkerState,
    ) {
        let Some(id) = surface_id else { return };
        let Some(surface) = self.surfaces.get(&id) else {
            return;
        };
        let normal = surface_normal(surface, point);
        let d = particle.direction;
        let mu = (d[0] * normal[0] + d[1] * normal[1] + d[2] * normal[2]).clamp(-1.0, 1.0);
        for (tally_idx, tally) in tallies.iter().enumerate() {
            if tally.is_surface_tally() {
                tally.score_surface_crossing(
                    welford_worker,
                    tally_idx,
                    id,
                    mu,
                    particle.weight,
                    particle.energy,
                    particle.particle_type,
                );
            }
        }
    }
}

/// The unit normal of `surface` at `point`, pointing to its positive side
/// (`Surface.above`, where `evaluate > 0`): the gradient of `evaluate` by
/// central differences, which serves every surface kind alike, formula and
/// flux surfaces included.
fn surface_normal(surface: &crate::geo::Surface, point: [f64; 3]) -> [f64; 3] {
    let scale = 1.0 + point.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    let h = 1e-6 * scale;
    let at =
        |dx: f64, dy: f64, dz: f64| surface.evaluate((point[0] + dx, point[1] + dy, point[2] + dz));
    let g = [
        at(h, 0.0, 0.0) - at(-h, 0.0, 0.0),
        at(0.0, h, 0.0) - at(0.0, -h, 0.0),
        at(0.0, 0.0, h) - at(0.0, 0.0, -h),
    ];
    let norm = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
    if norm > 0.0 {
        [g[0] / norm, g[1] / norm, g[2] / norm]
    } else {
        [0.0, 0.0, 0.0]
    }
}

#[cfg(test)]
mod surface_normal_tests {
    use super::*;

    #[test]
    fn sphere_normal_points_outwards() {
        let sphere = crate::geo::Surface::sphere(0.0, 0.0, 0.0, 5.0, Some(1), None);
        let n = surface_normal(&sphere, [0.0, 5.0, 0.0]);
        assert!((n[1] - 1.0).abs() < 1e-9 && n[0].abs() < 1e-9 && n[2].abs() < 1e-9);
    }

    #[test]
    fn plane_normal_points_to_positive_side() {
        let plane = crate::geo::Surface::new_plane(1.0, 0.0, 0.0, 2.0, Some(1), None);
        let n = surface_normal(&plane, [2.0, 3.0, -1.0]);
        let above = plane.evaluate((3.0, 3.0, -1.0)) > 0.0;
        assert_eq!(n[0] > 0.0, above);
        assert!((n[0].abs() - 1.0).abs() < 1e-9);
    }
}
