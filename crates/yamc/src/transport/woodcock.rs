//! Woodcock (delta-tracking) flight helpers: vacuum-exit distance, geometry
//! exit clipping, and mesh track-length scoring/eligibility along flights.

use super::*;

/// Whether a tally can be scored with the TRUE track-length estimator
/// along Woodcock flight segments (issue #350), instead of the
/// collision-density equivalent.
///
/// Requirements, all checked once at run start:
/// - `Estimator::TrackLength` with a regular-mesh filter: the mesh DDA
///   walks a regular grid without any geometry knowledge, which is what
///   makes segment scoring possible under delta tracking at all.
/// - Only material-independent scores (flux): a delta flight crosses
///   cells the tracker never identifies, so cross-section responses
///   (heating, reaction rates, ...) cannot be resolved along the
///   segment. Those tallies keep the collision-density equivalent,
///   which evaluates the response at each delta-collision point where
///   the material IS known.
/// - No cell or material filters, for the same reason: the segment
///   spans unidentified cells.
pub(crate) fn woodcock_mesh_track_length_eligible(tally: &Tally) -> bool {
    let has_spatial_mesh = {
        #[cfg(feature = "mesh")]
        {
            // Regular/cylindrical meshes use the lazy DDA iterator;
            // unstructured meshes use the ray-based `get_bins_crossed`
            // (issue #355). Both apportion a straight segment without
            // any geometry knowledge, which is what segment scoring
            // under delta tracking requires.
            tally.get_mesh_filter().is_some() || tally.get_unstructured_mesh_filter().is_some()
        }
        #[cfg(not(feature = "mesh"))]
        {
            tally.get_mesh_filter().is_some()
        }
    };
    tally.estimator == yamc_tallies::Estimator::TrackLength
        && has_spatial_mesh
        && tally
            .scores
            .iter()
            .all(|s| matches!(s, yamc_tallies::Score::Flux(_)))
        && !tally.filters.iter().any(|f| {
            matches!(
                f,
                yamc_tallies::filter::Filter::Cell(_) | yamc_tallies::filter::Filter::Material(_)
            )
        })
}

/// True track-length scoring for eligible mesh tallies over one Woodcock
/// flight segment `[particle.last_position, end_position]` of length
/// `flight_distance`. The mesh DDA (the same iterator surface tracking
/// uses) apportions the track length to every crossed voxel. Identical
/// for neutrons and photons, vacuum or material landing cells: by
/// eligibility the scores are material-independent, so the cell /
/// material / URR arguments are never consulted.
///
/// Segments are always fully inside the geometry: flights are truncated
/// at the first true vacuum exit by [`woodcock_flight_exit`] (issue
/// #360), and the leaking caller passes the clipped segment.
pub(super) fn score_woodcock_mesh_track_length(
    particle: &yamc_particle::particle::Particle,
    end_position: [f64; 3],
    flight_distance: f64,
    tallies: &[&Tally],
    eligible: &[bool],
    welford_worker: &mut yamc_tallies::welford::WelfordWorkerState,
) {
    for (tally_idx, (tally, &is_eligible)) in tallies.iter().zip(eligible).enumerate() {
        if !is_eligible {
            continue;
        }
        tally.score_track_length(
            welford_worker,
            tally_idx,
            flight_distance,
            particle.weight,
            None,
            None,
            None,
            particle.energy,
            end_position,
            particle.last_position,
            particle.direction,
            None,
            particle.particle_type,
            particle.parent_nuclide,
        );
    }
}

/// Distance from `start` along `direction` to the geometry exit, capped
/// at `max_distance`. Walks cell-to-cell with `closest_boundary` (a
/// delta flight can cross several cells before exiting, so a single
/// boundary query from the pre-flight cell is not enough).
///
/// Fallback path only since issue #360: CSG vacuum exits are caught
/// pre-flight by [`woodcock_flight_exit`], so this fires just for mesh
/// backends (no precomputed vacuum surfaces) and numerical corner
/// cases, on leaking flights with eligible mesh tallies present.
pub(super) fn distance_to_geometry_exit(
    geometry: &GeometryKind,
    neighbor_lists: &mut NeighborLists,
    mut cell_index: usize,
    start: [f64; 3],
    direction: [f64; 3],
    max_distance: f64,
) -> f64 {
    const SURFACE_TOLERANCE: f64 = 1e-8;
    // Guard against pathological geometries; a real leak path crosses
    // far fewer cells than this.
    const MAX_STEPS: usize = 10_000;
    let mut traveled = 0.0;
    for _ in 0..MAX_STEPS {
        let pos = [
            start[0] + direction[0] * traveled,
            start[1] + direction[1] * traveled,
            start[2] + direction[2] * traveled,
        ];
        let Some(hit) = geometry.closest_boundary(cell_index, pos, direction) else {
            // No boundary ahead along the ray, yet the landing point was
            // outside: a geometry gap. Keep only the distance confidently
            // walked inside (also avoids propagating an infinite
            // `max_distance` from a xi=0 flight into the mesh DDA).
            return traveled;
        };
        traveled += hit.distance + SURFACE_TOLERANCE;
        if traveled >= max_distance {
            return max_distance;
        }
        let next = [
            start[0] + direction[0] * traveled,
            start[1] + direction[1] * traveled,
            start[2] + direction[2] * traveled,
        ];
        match neighbor_lists.find_cell(geometry.cells(), (next[0], next[1], next[2]), cell_index) {
            Some(idx) => cell_index = idx,
            None => return traveled.min(max_distance),
        }
    }
    traveled.min(max_distance)
}

/// Distance along a Woodcock flight to the first TRUE geometry exit
/// through a vacuum surface, or `None` when the full flight stays
/// inside (issue #360).
///
/// Delta tracking samples flights with no boundary checks, so without
/// this test a flight can tunnel through a vacuum boundary, cross
/// out-of-geometry space, and land in a disjoint body that surface
/// tracking could never reach. The check is cheap: only the geometry's
/// few precomputed vacuum surfaces are tested (closed-form
/// `distance_to_surface` roots), not the cells. A crossing whose far
/// side is still inside defined geometry is NOT an exit and is skipped
/// -- e.g. another body's infinite vacuum-plane extension slicing
/// through the current body must not kill the flight -- so each
/// candidate is confirmed with one `find_cell` probe just past the
/// crossing. Draws no RNG and allocates nothing: non-leaking flight
/// histories are bit-identical with the check on, and this CPU
/// function is the source of truth for a future GPU port.
pub(super) fn woodcock_flight_exit(
    geometry: &GeometryKind,
    neighbor_lists: &mut NeighborLists,
    vacuum_surfaces: &[Arc<crate::geo::Surface>],
    pre_flight_cell_idx: usize,
    start: [f64; 3],
    direction: [f64; 3],
    d_flight: f64,
) -> Option<f64> {
    if vacuum_surfaces.is_empty() {
        return None;
    }
    const SURFACE_TOLERANCE: f64 = 1e-8;
    // Parity with `distance_to_geometry_exit`; a real flight crosses
    // far fewer vacuum-surface roots than this.
    const MAX_STEPS: usize = 10_000;
    let mut traveled = 0.0;
    let mut cell_hint = pre_flight_cell_idx;
    for _ in 0..MAX_STEPS {
        let pos = [
            start[0] + direction[0] * traveled,
            start[1] + direction[1] * traveled,
            start[2] + direction[2] * traveled,
        ];
        // Nearest vacuum-surface root strictly ahead of `pos`.
        // `distance_to_surface` already discards roots <= 1e-12 and
        // parallel planes; the tolerance filter below also handles
        // particles born within epsilon of a boundary.
        let mut nearest: Option<f64> = None;
        for surface in vacuum_surfaces {
            if let Some(t) = surface.distance_to_surface(pos, direction) {
                if t > SURFACE_TOLERANCE && nearest.is_none_or(|n| t < n) {
                    nearest = Some(t);
                }
            }
        }
        let t = nearest?;
        let hit = traveled + t;
        if hit >= d_flight {
            // First candidate crossing is beyond the sampled flight.
            return None;
        }
        // Probe just past the crossing: outside the defined geometry
        // means a true exit; inside means the crossing was a closed
        // surface's near side or another body's surface extension --
        // advance past it and keep walking.
        let probe_dist = hit + SURFACE_TOLERANCE;
        let probe = [
            start[0] + direction[0] * probe_dist,
            start[1] + direction[1] * probe_dist,
            start[2] + direction[2] * probe_dist,
        ];
        match neighbor_lists.find_cell(geometry.cells(), (probe[0], probe[1], probe[2]), cell_hint)
        {
            None => return Some(hit),
            Some(idx) => {
                cell_hint = idx;
                traveled = probe_dist;
            }
        }
    }
    None
}
