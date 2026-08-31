//! Self-contained transport / collision free functions.
//!
//! These are the CPU transport-loop helpers extracted verbatim from
//! `model.rs`. They take no `&self` and form the bit-exact source of
//! truth that the GPU/WGSL kernels are hand-ported from, so they are
//! moved here unchanged (including their `#[inline(always)]` and
//! `#[cfg(...)]` attributes). `Model::run_internal` (which stays in
//! `model.rs`) calls into these via `use crate::transport::{...}`.

pub mod debug;

use crate::geo::BoundaryType;
use crate::geometry::backend::{BoundaryHit, GeometryKind};
use crate::geometry::neighbor_lists::NeighborLists;
use crate::track::{TrackEventType, Tracker};
use crate::util::fast_rng::FastRng;
use rand::RngExt;
use std::sync::Arc;
use yamc_element::photon::{
    isotropic_direction, ElementMicroXS, PhotonInteraction, MASS_ELECTRON_EV,
};
use yamc_materials::material::{CollisionData, MacroPhotonXS, Material, UrrMacroXs};
use yamc_nuclide::nuclide::{Nuclide, ReactionType};
use yamc_nuclide::reaction::Reaction;
use yamc_nuclide::reaction_product::AngleEnergyDistribution;
use yamc_particle::particle::ParticleType;
use yamc_physics::gpu::flat::inelastic_flat::InelasticFlatCache;
use yamc_physics::neutron::interaction::rotate_direction_fast;
use yamc_physics::util::bank::ParticleBank;
use yamc_tallies::tally::Tally;
use yani_transmute::TransmutationTallies;

// Debug loggers live in `transport_debug.rs` (alongside the counters /
// env-var gating they share). `is_debug_elastic` / `log_elastic_scatter`
// are called unconditionally (each has a real and a no-op `#[cfg]` variant).
use crate::transport::debug::{is_debug_elastic, log_elastic_scatter};

// The `debug_collision`-tracing loggers are only *called* inside
// `#[cfg(feature = "debug_collision")]` blocks, so gate the import to
// match (otherwise it is unused in the default build).
#[cfg(feature = "debug_collision")]
use crate::transport::debug::{log_collision_distance, log_reaction_selection, log_scatter_result};

// Per-event diagnostic counters used by the photon collision handlers
// when `debug_diagnostics` is on; re-exported from `yamc_physics`.
#[cfg(feature = "debug_diagnostics")]
use yamc_physics::photon_diag;

// Atomic ordering + the global debug counters (which live in
// `transport_debug.rs`) are only referenced inside
// `#[cfg(feature = "debug_runtime")]` blocks.
#[cfg(feature = "debug_runtime")]
use crate::transport::debug::{
    TOTAL_ABSORPTION_REACTIONS, TOTAL_COLLISIONS, TOTAL_FISSION_REACTIONS,
    TOTAL_SCATTERING_REACTIONS, TOTAL_SCATTER_MT2, TOTAL_SCATTER_MT_OTHER,
};
#[cfg(feature = "debug_runtime")]
use std::sync::atomic::Ordering;

mod fission;
mod photon;
mod scatter;
mod scoring;
mod woodcock;

use fission::*;
pub(crate) use photon::*;
use scatter::*;
pub(crate) use scoring::*;
pub(crate) use woodcock::*;

/// Output of [`sample_distance_to_collision`]. `dist_collision` is
/// always set; `collision_data` is populated only for neutrons (and
/// only when the cell has material), `photon_macro_xs` only for
/// photons. The lifetime `'a` ties the borrowed nuclide-name and
/// nuclide references inside `CollisionData` to the material.
pub(crate) struct CollisionSampling<'a> {
    /// Neutron path: `(distance, nuclide_name, nuclide, urr_random,
    /// nuclide_id)`. `None` for photons or void cells.
    collision_data: Option<CollisionData<'a>>,
    /// Photon path: macroscopic photon XS used to score the photon
    /// track. `None` for neutrons or void cells.
    photon_macro_xs: Option<MacroPhotonXS>,
    /// Sampled distance to the next collision in cm. `f64::INFINITY`
    /// for void cells or vanishing total XS -- the surface-crossing
    /// path will then dominate.
    dist_collision: f64,
}

/// Read-only inputs shared across the transport entry points
/// ([`transport_particle`], [`transport_particle_woodcock`]) and the
/// neutron collision handler ([`handle_neutron_collision`]). These are
/// the simulation-level borrows and model flags that are identical for
/// every particle history in a `Model::simulate` call, so they are
/// gathered once at the call site and threaded as a single `&TransportCtx`
/// rather than as a long positional argument list. Mirrors the borrowed
/// [`CollisionSampling`] pattern.
///
/// Per-history / per-step varying state (the particle, RNG, tracker,
/// neighbour lists, particle bank, Welford worker, and the genealogy IDs)
/// stays as explicit parameters -- only genuinely shared read-only inputs
/// live here. The lifetime `'a` ties the borrows to the `simulate` scope
/// that builds the context.
pub(crate) struct TransportCtx<'a> {
    /// Geometry being transported through.
    pub(crate) geometry: &'a GeometryKind,
    /// Energy (eV) below which photons are killed.
    pub(crate) photon_cutoff_energy: f64,
    /// Whether secondary-photon production / photon transport is enabled.
    pub(crate) transport_secondary_photons: bool,
    /// Whether D1S decay-photon sampling is used in place of prompt
    /// photon-production data.
    pub(crate) use_decay_photons: bool,
    /// Energy (eV) below which free-gas thermal scattering is applied.
    pub(crate) free_gas_threshold: f64,
    /// Survival biasing (implicit capture): absorption is folded into a
    /// per-collision weight reduction instead of being a terminal event.
    pub(crate) survival_biasing: bool,
    /// Russian-roulette trigger weight (only consulted with
    /// `survival_biasing` on).
    pub(crate) weight_cutoff: f64,
    /// Weight assigned to roulette survivors.
    pub(crate) weight_survive: f64,
    /// Tallies scored over the transport.
    pub(crate) tallies: &'a [&'a Tally],
    /// Per-tally flag, parallel to `tallies`: eligible for TRUE
    /// track-length scoring along Woodcock flight segments (issue #350)
    /// instead of the delta-collision collision-density equivalent.
    /// All `false` (or empty) outside woodcock/hybrid modes.
    pub(crate) woodcock_tl_mesh_eligible: &'a [bool],
    /// Optional transmutation reaction-rate tallies.
    pub(crate) transmutation_tallies: Option<&'a TransmutationTallies>,
    /// Per-nuclide D1S decay-photon data, indexed by nuclide slot.
    pub(crate) decay_photon_nuclide_data:
        &'a [Vec<yamc_physics::photon::decay_photon_production::DecayPhotonNuclideData>],
    /// Running count of lost (geometry-gap) particles.
    pub(crate) lost_particle_count: &'a std::sync::atomic::AtomicUsize,
    /// Collected lost-particle records (capped at `max_lost`).
    pub(crate) lost_particles_collected:
        &'a std::sync::Mutex<Vec<crate::util::lost_particle::LostParticle>>,
    /// Maximum number of lost particles to collect before dropping.
    pub(crate) max_lost: usize,
    /// Weight-window maps (one per particle type at most) applied at
    /// collisions. Empty when no weight windows are configured. CPU-only
    /// (the GPU dispatch refuses weight windows).
    pub(crate) weight_windows: &'a [&'a crate::variance_reduction::WeightWindowBounds],
    /// Whether `YAMC_DEBUG` first-particle tracing is enabled.
    pub(crate) debug: bool,
    /// Run-scoped cache of flattened per-(nuclide, MT) inelastic kinematics
    /// tables (issue #111). Built lazily on the first collision that needs a
    /// given reaction and shared read-only across transport threads; the
    /// continuum inelastic arm samples from it on the shared PCG stream,
    /// byte-identically to the GPU. Created per `Model::simulate` call, so it
    /// never outlives the nuclide data its keys are addresses of.
    pub(crate) inelastic_flat_cache: &'a InelasticFlatCache,
}

/// Sample the distance the particle will travel before its next
/// collision, plus the auxiliary data needed by downstream tally and
/// reaction-sampling code (the per-collision URR random, the chosen
/// interacting nuclide for neutrons, the macroscopic photon XS for
/// photons).
///
/// Photon path: pulls the photon macroscopic XS from the cell material
/// and samples `-ln(ξ) / Σ_t`. Neutron path: defers to
/// [`Material::sample_collision_data`], which fuses the energy-grid
/// lookup, distance sample, interacting-nuclide pick, and URR random
/// into a single function for cache-locality reasons.
///
/// Extracted from the inner transport loop in [`Model::simulate`] as
/// part of Phase 1f. Hot path: fires once per particle step.
/// `#[inline(always)]` here (rather than just `#[inline]`) closes a
/// ~1% regression observed when LLVM declined to inline the body --
/// the function returns a non-trivial struct, which raises its
/// inline-cost estimate above the default threshold.
#[inline(always)]
pub(crate) fn sample_distance_to_collision<'a>(
    particle: &yamc_particle::particle::Particle,
    cell_material: Option<&'a Arc<Material>>,
    rng: &mut FastRng,
) -> CollisionSampling<'a> {
    if particle.particle_type == ParticleType::Photon {
        match cell_material {
            Some(material_arc) => {
                let material = material_arc.as_ref();
                let xs = material.calculate_photon_xs(particle.energy);
                let dist_collision = if xs.total > 0.0 {
                    -rng.random().ln() / xs.total
                } else {
                    f64::INFINITY
                };
                CollisionSampling {
                    collision_data: None,
                    photon_macro_xs: Some(xs),
                    dist_collision,
                }
            }
            None => CollisionSampling {
                collision_data: None,
                photon_macro_xs: None,
                dist_collision: f64::INFINITY,
            },
        }
    } else {
        // Neutron path. Reuse the cached URR random only if the energy
        // hasn't changed since it was sampled -- different energies
        // require independent URR samples to avoid spurious correlation.
        let cached_urr_random: Option<f64> = if particle.urr_energy == particle.energy {
            yamc_particle::particle::urr_to_option(particle.urr_random)
        } else {
            None
        };
        let collision_data = match cell_material {
            Some(material_arc) => {
                let material = material_arc.as_ref();
                material.sample_collision_data(particle.energy, cached_urr_random, rng)
            }
            None => None,
        };
        let dist_collision = collision_data
            .as_ref()
            .map(|(d, _, _, _, _)| *d)
            .unwrap_or(f64::INFINITY);
        CollisionSampling {
            collision_data,
            photon_macro_xs: None,
            dist_collision,
        }
    }
}

/// Resolve which cell currently contains `particle`. Uses the cached
/// `current_cell_index` first (set after the previous transport step),
/// then the learned neighbour list keyed by `previous_cell_index`, then
/// finally the geometry's BVH lookup. On a hit, caches the result on
/// the particle and returns `Some(idx)`. On a miss (geometry gap),
/// records a lost-particle event via [`handle_lost_particle`] and
/// returns `None` -- the caller is expected to treat that as
/// "particle is dead, continue to the next one".
///
/// Extracted from the inner transport loop in [`Model::simulate`] as
/// part of Phase 1f. Hot path: this fires every particle step that
/// crosses a surface (i.e. typically once per collision / surface
/// crossing). The `#[inline]` annotation lets LLVM merge the body back
/// into `simulate` so the function-call boundary is free at runtime.
#[inline(always)]
pub(crate) fn find_or_lose_cell(
    particle: &mut yamc_particle::particle::Particle,
    geometry: &GeometryKind,
    neighbor_lists: &mut NeighborLists,
    lost_particle_count: &std::sync::atomic::AtomicUsize,
    lost_particles_collected: &std::sync::Mutex<Vec<crate::util::lost_particle::LostParticle>>,
    max_lost: usize,
) -> Option<usize> {
    if particle.current_cell_index != yamc_particle::particle::NO_CELL {
        return Some(particle.current_cell_index as usize);
    }
    let pos = (
        particle.position[0],
        particle.position[1],
        particle.position[2],
    );
    let found = if particle.previous_cell_index != yamc_particle::particle::NO_CELL {
        // The neighbor lists locate the CSG cell only; a fill host is
        // refined to the embedded mesh volume containing the point
        // (geometry.find_cell_index resolves fills itself).
        neighbor_lists
            .find_cell(geometry.cells(), pos, particle.previous_cell_index as usize)
            .map(|idx| geometry.resolve_fill_at(idx, pos))
    } else {
        geometry.find_cell_index(pos)
    };
    match found {
        Some(idx) => {
            particle.current_cell_index = idx as u32;
            Some(idx)
        }
        None => {
            handle_lost_particle(
                particle,
                geometry.cells(),
                lost_particle_count,
                lost_particles_collected,
                max_lost,
            );
            None
        }
    }
}

/// Record a lost-particle event when cell-finding fails for `particle`'s
/// position (geometry gap). Pushes a `LostParticle` to the per-batch list,
/// increments the atomic counter, kills the particle, and panics if the
/// `max_lost` budget is exceeded.
///
/// Extracted from the inner transport loop in [`Model::simulate`] as part
/// of the GPU-prep work -- the eventual WGSL kernel will need a
/// self-contained "transport one particle" entry-point with no closures
/// over outer scope, and this is the cold-path bookkeeping piece of that.
/// Cold path: only fires for geometries with cell-cover gaps, so the
/// extraction is zero CPU cost on healthy geometries.
#[inline(always)]
pub(crate) fn handle_lost_particle(
    particle: &mut yamc_particle::particle::Particle,
    cells: &[crate::geometry::cell::Cell],
    lost_particle_count: &std::sync::atomic::AtomicUsize,
    lost_particles_collected: &std::sync::Mutex<Vec<crate::util::lost_particle::LostParticle>>,
    max_lost: usize,
) {
    let prev_cell = yamc_particle::particle::cell_index_to_option(particle.previous_cell_index)
        .map(|idx| &cells[idx]);
    let lost = crate::util::lost_particle::LostParticle {
        particle_type: particle.particle_type,
        position: particle.position,
        direction: particle.direction,
        energy: particle.energy,
        last_cell_index: yamc_particle::particle::cell_index_to_option(
            particle.previous_cell_index,
        ),
        last_cell_id: prev_cell.and_then(|c| c.cell_id),
        last_cell_name: prev_cell.and_then(|c| c.name.clone()),
        surface_id: yamc_particle::particle::surface_id_to_option(particle.last_surface_id),
    };
    lost.print_diagnostic();

    let count = lost_particle_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    lost_particles_collected
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .push(lost);

    particle.alive = false;

    if count > max_lost {
        panic!(
            "Maximum lost particles exceeded ({count} > {max_lost}). \
             Most often this means the outermost surface of the geometry \
             does not have `boundary='vacuum'` -- particles that escape \
             the geometry have nowhere to terminate, so they are flagged \
             as lost. Set `boundary='vacuum'` on the outer surface, fix \
             any gaps between cells (the lost particle diagnostics above \
             show where each particle was last located), or pass \
             `max_lost_particles=` to `Model(...)` if leakage is \
             expected and acceptable."
        );
    }
}

/// Spatial tracking verification (issue #254): before accepting a
/// surface crossing, check that no surface FOREIGN to the current cell
/// intersects the open segment between the particle and the crossing.
/// The adjacency tracker resolves crossings against the current
/// volume's own surfaces only and never re-locates a particle in
/// space, so a mesh whose volumes overlap or self-intersect, or a
/// tracking state corrupted by an earlier mis-resolved crossing, is
/// otherwise transported through silently, producing confident wrong
/// tallies with zero lost particles. A blocked segment goes through
/// the normal lost-particle path (diagnostics, `model.lost_particles`,
/// `max_lost_particles` abort) and returns true so the caller skips
/// the bogus crossing.
///
/// Runs on every transmission crossing: the foreign-surface query is a
/// bounded any-hit BVH traversal (seeded at the crossing distance, so
/// it only ever visits the segment's neighbourhood) and is free at the
/// measured level on both flat and 1.5M-triangle curved meshes. CSG
/// geometries skip it entirely inside `crossing_blocked`.
#[inline(always)]
fn verify_crossing(
    ctx: &TransportCtx,
    particle: &mut yamc_particle::particle::Particle,
    cell_index: usize,
    dist_surface: f64,
) -> bool {
    if ctx.geometry.crossing_blocked(
        cell_index,
        particle.position,
        particle.direction,
        dist_surface,
    ) {
        handle_lost_particle(
            particle,
            ctx.geometry.cells(),
            ctx.lost_particle_count,
            ctx.lost_particles_collected,
            ctx.max_lost,
        );
        return true;
    }
    false
}

/// Handle a surface-crossing event: the particle has reached a cell
/// boundary before its sampled collision distance. If the crossed
/// surface has a Vacuum boundary the particle leaves the geometry --
/// record a leak termination and kill the particle. Otherwise it's a
/// transmission boundary -- record the crossing event, advance the
/// particle past the surface (with a small `SURFACE_TOLERANCE` nudge
/// to avoid landing exactly on the surface), and update the
/// `previous_cell_index` and `current_cell_index` hints so the next
/// iteration's [`find_or_lose_cell`] starts from the right place.
///
/// Extracted from the inner transport loop in [`Model::simulate`] as
/// step 4 of Phase 1f. Hot path: fires every time a particle crosses
/// a surface (typically more often than collision in low-density
/// geometries). `#[inline(always)]` per the lesson from #34 -- plain
/// `#[inline]` was unreliable because the call site is hot even when
/// the body isn't.
///
/// Many parameters: this is the cost of being self-contained for the
/// eventual GPU port. The WGSL kernel will see the same conceptual
/// surface (particle ref, cell hint, boundary hit, end position, etc.).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_surface_crossing<T: Tracker>(
    particle: &mut yamc_particle::particle::Particle,
    cell_index: usize,
    cell_id: Option<u32>,
    boundary_hit: &BoundaryHit,
    end_position: [f64; 3],
    tracker: &mut T,
    current_particle_id: u64,
    current_parent_id: Option<u64>,
    current_generation: u32,
) {
    if boundary_hit.boundary == BoundaryType::Vacuum {
        // Track leak event
        tracker.record_termination(
            current_particle_id,
            current_parent_id,
            current_generation,
            end_position,
            particle.direction,
            particle.energy,
            particle.weight,
            cell_id,
            TrackEventType::Leak,
        );
        particle.alive = false;
    } else {
        // Track surface crossing
        tracker.record_surface_crossing(
            current_particle_id,
            current_parent_id,
            current_generation,
            end_position,
            particle.direction,
            particle.energy,
            particle.weight,
            cell_id,
        );
        const SURFACE_TOLERANCE: f64 = 1e-8;
        particle.previous_cell_index = cell_index as u32;
        particle.last_surface_id =
            yamc_particle::particle::surface_id_to_u32(boundary_hit.surface_id);
        particle.move_by(boundary_hit.distance + SURFACE_TOLERANCE);
        // Use topology-based next cell if available (mesh geometry),
        // otherwise invalidate cache to trigger find_cell_index (CSG).
        particle.current_cell_index =
            yamc_particle::particle::cell_index_to_u32(boundary_hit.next_cell_index);
    }
}

/// Run a single particle through the transport loop until it dies (or
/// is killed by the cell-finding fallback, vacuum boundary, or photon
/// cutoff). Self-contained: no `&self` parameter, no closures over the
/// outer `Model::simulate` scope. The 20-parameter signature reflects
/// the cost of being self-contained -- this is intentional and is the
/// shape the eventual WGSL kernel will need.
///
/// Step 6 (final step) of Phase 1f. The earlier steps factored out
/// `find_or_lose_cell`, `sample_distance_to_collision`,
/// `score_track_length_segment`, `handle_surface_crossing`, and
/// `handle_lost_particle` into private helpers; this lift moves what
/// was left of the inner `while particle.alive` body into a single
/// callable function. `#[inline(always)]` per the lesson from #34.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn transport_particle<T: Tracker>(
    ctx: &TransportCtx,
    particle: &mut yamc_particle::particle::Particle,
    rng: &mut FastRng,
    // Per-particle 64-bit PCG state for sub-steps on the shared GPU/CPU
    // collision path (issues #111, #274); threaded into the collision handler.
    pcg: &mut u64,
    tracker: &mut T,
    neighbor_lists: &mut NeighborLists,
    particle_bank: &mut ParticleBank,
    current_particle_id: u64,
    current_parent_id: Option<u64>,
    current_generation: u32,
    particle_idx: usize,
    welford_worker: &mut yamc_tallies::welford::WelfordWorkerState,
) {
    let geometry = ctx.geometry;
    while particle.alive {
        let cell_index = match find_or_lose_cell(
            particle,
            geometry,
            neighbor_lists,
            ctx.lost_particle_count,
            ctx.lost_particles_collected,
            ctx.max_lost,
        ) {
            Some(idx) => idx,
            None => continue,
        };
        let cell = &geometry.cells()[cell_index];

        // Extract material_id for flux tallying
        let material_id = geometry
            .material_for(cell)
            .and_then(|m| m.get_material_id());

        // Photon energy cutoff check
        if particle.particle_type == ParticleType::Photon
            && particle.energy < ctx.photon_cutoff_energy
        {
            particle.alive = false;
            particle.weight = 0.0;
            continue;
        }

        let cell_material = geometry.material_for(cell);
        // Free-flight on the shared PCG stream (issue #111). For a survival-off
        // neutron in a non-URR fast-XS material -- the config whose reaction-type
        // split runs on PCG (see `handle_neutron_collision`) -- draw the flight
        // `xi1` from PCG so the per-step schedule matches the GPU kernel, whose
        // first per-step sample is the flight. The struck nuclide is selected
        // only if the collision wins the surface min (see the collision arm
        // below), so the per-step PCG draws line up with the GPU: `xi1` every
        // step, `xi_n` only on a real collision. `smooth_flight` takes a PCG draw
        // only when its preconditions hold; any other config falls through to
        // legacy fused FastRng sampling with the PCG stream untouched.
        let smooth_flight = if particle.particle_type == ParticleType::Neutron {
            match cell_material {
                Some(m) => {
                    // A band held at THIS energy is reused, so a crossing
                    // does not redraw it (#342); at a new energy
                    // `smooth_flight` draws one before the flight, matching
                    // the kernel's per-step order.
                    let held = if particle.urr_energy == particle.energy {
                        yamc_particle::particle::urr_to_option(particle.urr_random)
                    } else {
                        None
                    };
                    m.smooth_flight(particle.energy, held, pcg)
                }
                None => None,
            }
        } else {
            None
        };
        // Record the band the flight sampled against, so the collision's
        // nuclide selection, reaction split and tally scoring reuse it.
        if let Some(ref sf) = smooth_flight {
            if let Some(band) = sf.urr_random {
                particle.urr_random = yamc_particle::particle::urr_from_option(Some(band));
                particle.urr_energy = particle.energy;
            }
        }
        let CollisionSampling {
            collision_data,
            photon_macro_xs,
            dist_collision,
        } = match smooth_flight.map(|sf| sf.distance) {
            Some(distance) => CollisionSampling {
                collision_data: None,
                photon_macro_xs: None,
                dist_collision: distance,
            },
            None => sample_distance_to_collision(particle, cell_material, rng),
        };

        // Neutron-specific: debug collision logging and URR handling
        if particle.particle_type == ParticleType::Neutron {
            #[cfg(feature = "debug_collision")]
            if let Some((dist, _, _, urr_rand, _)) = &collision_data {
                let sigma_t = if *dist > 0.0 {
                    -1.0_f64.ln() / *dist
                } else {
                    0.0
                };
                log_collision_distance(particle_idx, particle.energy, sigma_t, *dist, *urr_rand);
            }

            // Store URR random on particle for correlated reaction sampling.
            // Only update when a URR band was actually sampled; otherwise LEAVE
            // any held band intact so it survives a void / non-URR excursion at
            // the same energy (issue #206). This matches OpenMC, whose URR seed
            // advances only when the energy changes: a neutron that backscatters
            // through a void and re-enters the same material at the same energy
            // must reuse its band, not redraw. The `urr_energy == energy` guard
            // on the next lookup invalidates the held band as soon as a collision
            // changes the energy.
            if let Some((_, _, _, Some(band), _)) = &collision_data {
                particle.urr_random = yamc_particle::particle::urr_from_option(Some(*band));
                particle.urr_energy = particle.energy;
            }
        }

        // Compute URR-modified macroscopic XS for tally scoring (neutrons only)
        let urr_macro_xs = if particle.particle_type == ParticleType::Neutron
            && !particle.urr_random.is_nan()
        {
            cell_material.and_then(|m| m.compute_urr_macro_xs(particle.energy, particle.urr_random))
        } else {
            None
        };

        // Debug first particle (only if YAMC_DEBUG is set)
        if ctx.debug && particle_idx == 0 && particle.alive {
            let pos = particle.position;
            let r = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
            eprintln!(
                "[DEBUG] P0: r={:.4}, cell_idx={}, has_mat={}, dist_coll={:.4}",
                r,
                cell_index,
                cell_material.is_some(),
                dist_collision
            );
        }

        if let Some(boundary_hit) =
            geometry.closest_boundary(cell_index, particle.position, particle.direction)
        {
            let dist_surface = boundary_hit.distance;
            if dist_surface < dist_collision {
                if verify_crossing(ctx, particle, cell_index, dist_surface) {
                    continue;
                }
                let end_position = score_track_length_segment(
                    particle,
                    cell,
                    cell_material,
                    material_id,
                    dist_surface,
                    urr_macro_xs.as_ref(),
                    ctx.tallies,
                    ctx.transmutation_tallies,
                    welford_worker,
                );
                handle_surface_crossing(
                    particle,
                    cell_index,
                    cell.cell_id,
                    &boundary_hit,
                    end_position,
                    tracker,
                    current_particle_id,
                    current_parent_id,
                    current_generation,
                );
            } else {
                let _end_position = score_track_length_segment(
                    particle,
                    cell,
                    cell_material,
                    material_id,
                    dist_collision,
                    urr_macro_xs.as_ref(),
                    ctx.tallies,
                    ctx.transmutation_tallies,
                    welford_worker,
                );
                particle.move_by(dist_collision);
                let material = cell_material.unwrap().as_ref();

                // Score collision-estimator tallies at the collision site,
                // before any reaction sampling changes the particle state.
                // No-op when no tally opted into Estimator::Collision.
                score_collision_event(
                    particle,
                    cell,
                    material,
                    material_id,
                    urr_macro_xs.as_ref(),
                    photon_macro_xs.as_ref(),
                    ctx.tallies,
                    welford_worker,
                );

                // ========== PHOTON COLLISION HANDLING ==========
                #[cfg(feature = "debug_diagnostics")]
                crate::transport::debug::BATCH_COLLISIONS
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if particle.particle_type == ParticleType::Photon {
                    if let Some(ref macro_xs) = photon_macro_xs {
                        let incoming_energy = particle.energy;
                        let incoming_weight = particle.weight;
                        let (tally_mt, bank_second_photon_energy) = handle_photon_collision(
                            particle,
                            material,
                            macro_xs,
                            particle_bank,
                            ctx.photon_cutoff_energy,
                            rng,
                        );

                        // Score photon heating using the analog collision
                        // estimator. heat = (E_in - E_out - banked photon
                        // energy) * weight: electron/positron kinetic energy
                        // deposits locally (coupled neutron-photon
                        // local-deposition convention; see handle_photon_collision), only
                        // banked photons are subtracted. Score can be
                        // negative (no clamping). Routes through the unified
                        // `score_collision` dispatch with `total_xs = 0` so
                        // the Σ_t-dependent arms (Flux, neutron heating) are
                        // skipped; this call fires regardless of
                        // `tally.estimator` so a `TrackLength` heating tally
                        // still captures photon heating.
                        let e_out = particle.energy; // 0 if photon absorbed
                        let heat_score =
                            (incoming_energy - e_out - bank_second_photon_energy) * incoming_weight;
                        if heat_score != 0.0 {
                            for (tally_idx, tally) in ctx.tallies.iter().enumerate() {
                                // TrackLength tallies score photon KERMA
                                // along tracks instead (issue #356).
                                if tally.estimator != yamc_tallies::Estimator::Collision {
                                    continue;
                                }
                                tally.score_collision(
                                    welford_worker,
                                    tally_idx,
                                    incoming_weight,
                                    0.0,
                                    particle.position,
                                    cell.cell_id,
                                    Some(material),
                                    material_id,
                                    None, // photons have no URR
                                    None, // PhotonXS scored pre-collision; this call only writes analog heating
                                    incoming_energy,
                                    ParticleType::Photon,
                                    particle.parent_nuclide,
                                    Some(heat_score),
                                );
                            }
                        }

                        // Record collision for tracking
                        tracker.record_collision(
                            current_particle_id,
                            current_parent_id,
                            current_generation,
                            particle.position,
                            particle.direction,
                            incoming_energy,
                            particle.energy,
                            particle.weight,
                            cell.cell_id,
                            tally_mt,
                            "photon_Z",
                            None,
                            None,
                        );

                        // Weight-window check: split high-weight photons and
                        // roulette low-weight ones toward the per-voxel target
                        // band. Only surviving (scattered) photons are windowed
                        // (photoelectric / pair-production already set
                        // `alive = false`). Mirrors the neutron site below;
                        // CPU-only. Applies to every photon at its collision
                        // (source, coupled, decay, fluorescence), matched by
                        // particle type inside `apply_weight_window`.
                        if particle.alive && !ctx.weight_windows.is_empty() {
                            apply_weight_window(
                                ctx.weight_windows,
                                particle,
                                cell,
                                rng,
                                particle_bank,
                                tracker,
                                current_particle_id,
                                current_parent_id,
                                current_generation,
                            );
                        }
                    }

                // ========== NEUTRON COLLISION HANDLING ==========
                } else {
                    // On the smooth PCG path the flight (`xi1`) was drawn above
                    // but the struck nuclide was deferred to here -- a real
                    // collision -- so `xi_n` is consumed in the same per-step
                    // position as the GPU (nuclide-select only on collision).
                    // `select_nuclide_smooth` draws nothing for a single-nuclide
                    // material. Other configs already carry `collision_data` from
                    // the legacy fused sampler.
                    let collision_data = match smooth_flight {
                        Some(sf) => {
                            let selected = match cell_material {
                                Some(m) => m.select_nuclide_smooth(
                                    particle.energy,
                                    sf.i_grid,
                                    sf.f,
                                    sf.urr_random,
                                    pcg,
                                ),
                                None => None,
                            };
                            selected.map(|(name, nuc, id)| (sf.distance, name, nuc, None, id))
                        }
                        None => collision_data,
                    };
                    handle_neutron_collision(
                        ctx,
                        particle,
                        cell,
                        material,
                        collision_data,
                        rng,
                        pcg,
                        tracker,
                        particle_bank,
                        current_particle_id,
                        current_parent_id,
                        current_generation,
                        particle_idx,
                    );
                }
            }
        } else {
            // No surface found -- particle is likely at the edge of a mesh
            // approximation (inscribed polyhedron gap). Kill the particle
            // rather than panicking. This is rare and has negligible
            // impact on statistics.
            particle.alive = false;
            particle.weight = 0.0;
        }
    } // end of particle transport while loop
}

/// Hybrid-dispatch budget for Woodcock tracking, in units of expected
/// fictitious collisions per cell crossing. A cell is surface-tracked
/// (rather than delta-tracked) when the expected number of fictitious
/// collisions to cross it,
///
///   `(Σ_global - Σ_local) · mean_chord(cell)`,
///
/// exceeds this budget -- i.e. when Woodcock would churn more than ~1
/// rejection's worth of work that one boundary computation would avoid.
/// This is *cells-per-mean-free-path aware*: a large low-density cell
/// (big chord, like a plasma chamber or air region) is surface-tracked,
/// while a *small* low-density cell (thin shell in a finely diced model)
/// stays on Woodcock -- preserving the boundary-skipping win that pure
/// efficiency thresholding would throw away. Dense cells
/// (Σ_local ≈ Σ_global) have ~0 expected fictitious collisions and stay
/// on Woodcock regardless of size. 1.0 ≈ "surface-track once Woodcock
/// averages more than one fictitious collision crossing the cell".
const WOODCOCK_FICTITIOUS_BUDGET: f64 = 1.0;

/// Mean chord length of a cell, estimated from its region bounding box
/// (4V/S, the isotropic mean chord of the box). Used only by the hybrid
/// dispatch heuristic, so the bbox over-approximation of the cell is
/// fine. An unbounded cell (infinite bbox) returns `f64::INFINITY`,
/// which makes the hybrid always surface-track it (correct: an unbounded
/// cell is effectively infinite chord, so Woodcock would churn).
pub(crate) fn cell_mean_chord(bbox: &crate::geo::BoundingBox) -> f64 {
    let [a, b, c] = bbox.width();
    if !(a.is_finite() && b.is_finite() && c.is_finite()) {
        return f64::INFINITY;
    }
    let surface = 2.0 * (a * b + b * c + c * a);
    if surface <= 0.0 {
        return 0.0;
    }
    4.0 * (a * b * c) / surface
}

/// Process a real photon collision: collision-estimator scoring, the
/// photon physics (`handle_photon_collision`), analog heating scoring,
/// and tracking. Shared by the Woodcock delta-tracking step and the
/// hybrid surface step so the photon collision is written once.
#[allow(clippy::too_many_arguments)]
pub(crate) fn do_photon_collision<T: Tracker>(
    particle: &mut yamc_particle::particle::Particle,
    cell: &crate::geometry::cell::Cell,
    material: &Material,
    material_id: Option<u32>,
    photon_macro_xs: &yamc_materials::MacroPhotonXS,
    particle_bank: &mut ParticleBank,
    photon_cutoff_energy: f64,
    rng: &mut FastRng,
    tracker: &mut T,
    tallies: &[&Tally],
    welford_worker: &mut yamc_tallies::welford::WelfordWorkerState,
    current_particle_id: u64,
    current_parent_id: Option<u64>,
    current_generation: u32,
    weight_windows: &[&crate::variance_reduction::WeightWindowBounds],
) {
    // Collision-estimator tallies at the collision site (flux = weight /
    // Σ_t), before the physics mutates state.
    score_collision_event(
        particle,
        cell,
        material,
        material_id,
        None, // photons have no URR
        Some(photon_macro_xs),
        tallies,
        welford_worker,
    );

    let incoming_energy = particle.energy;
    let incoming_weight = particle.weight;
    let (tally_mt, bank_second_photon_energy) = handle_photon_collision(
        particle,
        material,
        photon_macro_xs,
        particle_bank,
        photon_cutoff_energy,
        rng,
    );
    // Analog heating: (E_in - E_out - banked photon energy) * weight;
    // electron/positron kinetic energy deposits locally (see
    // handle_photon_collision).
    let e_out = particle.energy; // 0 if photon absorbed
    let heat_score = (incoming_energy - e_out - bank_second_photon_energy) * incoming_weight;
    if heat_score != 0.0 {
        for (tally_idx, tally) in tallies.iter().enumerate() {
            // TrackLength tallies score photon KERMA along tracks
            // instead (issue #356).
            if tally.estimator != yamc_tallies::Estimator::Collision {
                continue;
            }
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
                ParticleType::Photon,
                particle.parent_nuclide,
                Some(heat_score),
            );
        }
    }
    tracker.record_collision(
        current_particle_id,
        current_parent_id,
        current_generation,
        particle.position,
        particle.direction,
        incoming_energy,
        particle.energy,
        particle.weight,
        cell.cell_id,
        tally_mt,
        "photon_Z",
        None,
        None,
    );

    // Weight-window check for the Woodcock/hybrid photon path, mirroring the
    // surface path and the neutron site. Only surviving (scattered) photons are
    // windowed; CPU-only.
    if particle.alive && !weight_windows.is_empty() {
        apply_weight_window(
            weight_windows,
            particle,
            cell,
            rng,
            particle_bank,
            tracker,
            current_particle_id,
            current_parent_id,
            current_generation,
        );
    }
}

/// Woodcock (delta) tracking transport for neutrons + photons. Serves
/// both `TrackingMode::Woodcock` (pure, `hybrid = false`) and
/// `TrackingMode::Hybrid` (`hybrid = true`, the per-cell surface
/// fallback below).
///
/// In the pure path this implementation does **not** call
/// `closest_boundary` in the hot loop. The particle moves the full
/// sampled flight distance and the new cell is found via
/// `find_or_lose_cell` at the post-flight position. That is where the
/// algorithmic win over surface tracking lives.
///
/// To make rejection valid across cell crossings inside a flight, the
/// majorant is `GlobalMajorant` (max Σ_t over all materials in the
/// model). `LocalMajorant`'s per-material bound is unsafe here because
/// a flight crossing from a light material into a dense one could
/// produce Σ_t(new) / Σ_maj(old) > 1.
///
/// Both particle types share the flight/relocate machinery; only the
/// majorant, the cross-section the rejection samples against, and the
/// collision physics differ:
/// - **Neutrons** use `majorant` (the neutron `GlobalMajorant`), sample
///   URR if applicable, and run [`handle_neutron_collision`] (which
///   banks secondary photons when `transport_secondary_photons` / `use_decay_photons`
///   are set -- coupled and D1S photons then transport through this same
///   loop).
/// - **Photons** use `photon_majorant` (a `GlobalPhotonMajorant`), are
///   killed below `photon_cutoff_energy`, reject against
///   `calculate_photon_xs().total`, and run [`handle_photon_collision`]
///   plus analog heating scoring -- mirroring the surface path.
///
/// Per-step algorithm:
/// 1. Sample `d_flight = -ln(ξ) / Σ_maj(E)` at the current position.
/// 2. Score track-length over `[pos, pos + d_flight·direction]`.
/// 3. `particle.move_by(d_flight)`. No boundary check.
/// 4. Relocate at the new position. `None` → escape, kill particle.
/// 5. If the new cell is vacuum, loop (purely fictitious step).
/// 6. Otherwise, accept the collision with probability `Σ_t / Σ_maj`;
///    on accept score collision-estimator tallies and run the
///    type-specific collision handler.
/// 7. Loop continues with `cell_index` set to the post-flight cell.
///
/// When `hybrid` is set, a cell whose expected fictitious-collision count
/// to cross exceeds the budget is instead surface-stepped against its
/// local cross section (one `closest_boundary`, no rejection loop). This
/// keeps voids/large low-density regions from churning the rejection
/// loop while leaving dense cells on the pure path. `cell_mean_chords` is
/// only consulted in this branch, so it may be empty when `hybrid` is
/// false.
#[allow(clippy::too_many_arguments)]
pub(crate) fn transport_particle_woodcock<T: Tracker>(
    ctx: &TransportCtx,
    particle: &mut yamc_particle::particle::Particle,
    majorant: &dyn yamc_materials::Majorant,
    photon_majorant: Option<&dyn yamc_materials::Majorant>,
    cell_mean_chords: &[f64],
    hybrid: bool,
    rng: &mut FastRng,
    // Per-particle 64-bit PCG state for the shared GPU/CPU collision path
    // (issues #111, #274); threaded into the collision handler.
    pcg: &mut u64,
    tracker: &mut T,
    neighbor_lists: &mut NeighborLists,
    particle_bank: &mut ParticleBank,
    current_particle_id: u64,
    current_parent_id: Option<u64>,
    current_generation: u32,
    particle_idx: usize,
    welford_worker: &mut yamc_tallies::welford::WelfordWorkerState,
) {
    let geometry = ctx.geometry;
    // Any tally taking the true track-length path along flight segments
    // (issue #350)? Hoisted: the per-flight scoring below is skipped
    // entirely when no eligible mesh tally exists, keeping analog
    // Woodcock bit-identical.
    let any_tl_mesh = ctx.woodcock_tl_mesh_eligible.iter().any(|&b| b);
    // Resolve the initial cell once before the loop. Subsequent
    // iterations carry `cell_index` forward from the post-flight lookup,
    // avoiding a second `find_or_lose_cell` per loop body.
    let mut cell_index = match find_or_lose_cell(
        particle,
        geometry,
        neighbor_lists,
        ctx.lost_particle_count,
        ctx.lost_particles_collected,
        ctx.max_lost,
    ) {
        Some(idx) => idx,
        None => return,
    };

    while particle.alive {
        let is_photon = particle.particle_type == ParticleType::Photon;

        // Photon energy cutoff: kill low-energy photons before sampling
        // a flight, matching the surface-tracking path.
        if is_photon && particle.energy < ctx.photon_cutoff_energy {
            particle.alive = false;
            particle.weight = 0.0;
            continue;
        }

        let cell = &geometry.cells()[cell_index];
        let cell_material = geometry.material_for(cell);
        let material_id = cell_material.and_then(|m| m.get_material_id());

        // Global (space-uniform) majorant for this particle type. No
        // pre-flight URR sampling is needed: the neutron majorant is
        // already URR-worst-case, and the Woodcock flux estimator scores
        // at the *post*-flight delta-collision (where URR is re-sampled),
        // not over the pre-flight segment.
        let sigma_maj = if is_photon {
            // `photon_majorant` is `Some` whenever photons can appear
            // (validated at simulate_transport start).
            photon_majorant
                .expect("photon majorant must be built when photons transport under Woodcock")
                .sigma_max(material_id, particle.energy)
        } else {
            majorant.sigma_max(material_id, particle.energy)
        };

        if sigma_maj <= 0.0 {
            // No material with this particle's physics anywhere in the
            // model -- no collisions are possible. Kill the particle to
            // avoid an infinite-flight non-terminating loop.
            particle.alive = false;
            particle.weight = 0.0;
            continue;
        }

        // Hybrid dispatch (only when `hybrid` is set; pure Woodcock never
        // falls back and skips this decision entirely). Pure Woodcock
        // samples flights against the GLOBAL majorant (set by the densest
        // material in the model). In a void or low-density cell that
        // majorant is far above the local Σ_t, so most delta-collisions
        // are fictitious. Surface-track a cell when the expected number of
        // fictitious collisions to cross it -- `(Σ_global - Σ_local) ·
        // mean_chord` -- exceeds the budget; sample the flight against the
        // LOCAL cross section and stop at the boundary (one
        // `closest_boundary`, no fictitious collisions). This is
        // cells-per-mfp aware: a *large* low-density cell (plasma chamber,
        // air) is surface-tracked, a *small* one (thin shell) stays on
        // Woodcock to keep the boundary-skipping win. Dense cells
        // (Σ_local ≈ Σ_global) have ~0 expected fictitious collisions and
        // stay on Woodcock. Σ_local uses the cheap smooth total XS (not
        // the per-step-expensive URR worst-case `total_xs_majorant`); the
        // decision is a heuristic and only needs a rough cross section.
        let surface_step = if hybrid {
            let sigma_local = if is_photon {
                cell_material.map_or(0.0, |m| {
                    m.as_ref().calculate_photon_xs(particle.energy).total
                })
            } else {
                cell_material.map_or(0.0, |m| m.as_ref().lookup_xs_by_mt(1, particle.energy))
            };
            let mean_chord = cell_mean_chords
                .get(cell_index)
                .copied()
                .unwrap_or(f64::INFINITY);
            (sigma_maj - sigma_local).max(0.0) * mean_chord > WOODCOCK_FICTITIOUS_BUDGET
        } else {
            false
        };

        if surface_step {
            // Distance to the next real collision against the LOCAL cross
            // section (infinite for a void cell, so it always crosses).
            let CollisionSampling {
                collision_data,
                photon_macro_xs,
                dist_collision,
            } = sample_distance_to_collision(particle, cell_material, rng);

            // Mirror surface tracking's URR cache bookkeeping (neutrons):
            // the sampled band is stored on the particle for a consistent
            // Σ_score in the tally lookups.
            let urr_macro_xs = if is_photon {
                None
            } else {
                // Only update when a URR band was sampled; otherwise hold any
                // band from a URR excursion at the same energy (issue #206,
                // mirrors the surface-tracking path and OpenMC's seed-advances-
                // on-energy-change rule).
                if let Some((_, _, _, Some(band), _)) = &collision_data {
                    particle.urr_random = yamc_particle::particle::urr_from_option(Some(*band));
                    particle.urr_energy = particle.energy;
                }
                if !particle.urr_random.is_nan() {
                    cell_material
                        .and_then(|m| m.compute_urr_macro_xs(particle.energy, particle.urr_random))
                } else {
                    None
                }
            };

            match geometry.closest_boundary(cell_index, particle.position, particle.direction) {
                Some(boundary_hit) if boundary_hit.distance < dist_collision => {
                    // Cross the boundary -- no collision this step.
                    if verify_crossing(ctx, particle, cell_index, boundary_hit.distance) {
                        continue;
                    }
                    let end_position = score_track_length_segment(
                        particle,
                        cell,
                        cell_material,
                        material_id,
                        boundary_hit.distance,
                        urr_macro_xs.as_ref(),
                        ctx.tallies,
                        ctx.transmutation_tallies,
                        welford_worker,
                    );
                    handle_surface_crossing(
                        particle,
                        cell_index,
                        cell.cell_id,
                        &boundary_hit,
                        end_position,
                        tracker,
                        current_particle_id,
                        current_parent_id,
                        current_generation,
                    );
                }
                Some(_) => {
                    // Real collision in this cell (a finite collision
                    // distance implies a material cell -- a void has
                    // `dist_collision = inf` and takes the cross branch).
                    let _ = score_track_length_segment(
                        particle,
                        cell,
                        cell_material,
                        material_id,
                        dist_collision,
                        urr_macro_xs.as_ref(),
                        ctx.tallies,
                        ctx.transmutation_tallies,
                        welford_worker,
                    );
                    particle.move_by(dist_collision);
                    let material = cell_material
                        .expect("a finite collision distance implies a material cell")
                        .as_ref();
                    if is_photon {
                        let macro_xs =
                            photon_macro_xs.expect("photon collision distance implies photon XS");
                        do_photon_collision(
                            particle,
                            cell,
                            material,
                            material_id,
                            &macro_xs,
                            particle_bank,
                            ctx.photon_cutoff_energy,
                            rng,
                            tracker,
                            ctx.tallies,
                            welford_worker,
                            current_particle_id,
                            current_parent_id,
                            current_generation,
                            ctx.weight_windows,
                        );
                    } else {
                        score_collision_event(
                            particle,
                            cell,
                            material,
                            material_id,
                            urr_macro_xs.as_ref(),
                            photon_macro_xs.as_ref(),
                            ctx.tallies,
                            welford_worker,
                        );
                        handle_neutron_collision(
                            ctx,
                            particle,
                            cell,
                            material,
                            collision_data,
                            rng,
                            pcg,
                            tracker,
                            particle_bank,
                            current_particle_id,
                            current_parent_id,
                            current_generation,
                            particle_idx,
                        );
                    }
                }
                None => {
                    // No boundary -- geometry-gap edge case; kill,
                    // mirroring surface tracking.
                    particle.alive = false;
                    particle.weight = 0.0;
                }
            }
            if !particle.alive {
                continue; // leaked / collided into death / killed
            }
            // Relocate for the next iteration (a crossing invalidated the
            // cell cache; a collision left it valid -- either way this is
            // an O(1) cached lookup or a neighbour search).
            cell_index = match find_or_lose_cell(
                particle,
                geometry,
                neighbor_lists,
                ctx.lost_particle_count,
                ctx.lost_particles_collected,
                ctx.max_lost,
            ) {
                Some(idx) => idx,
                None => continue,
            };
            continue;
        }

        // ---- Woodcock step (efficient material cell) ----
        let xi: f64 = rng.random();
        let d_flight = -xi.ln() / sigma_maj;

        let pre_flight_cell_idx = cell_index;

        // Truncate the flight at the first TRUE vacuum exit (issue
        // #360): without this, a flight crosses vacuum boundaries
        // unchecked and can land in a disjoint body across
        // out-of-geometry space, where surface tracking would have
        // killed it at the boundary -- the modes would disagree on
        // means, gaps would attenuate as exp(-sigma_maj*d), and the
        // mesh scorer would deposit into gap voxels. Crossings whose
        // far side is still inside the geometry are skipped by the
        // helper, so flights never false-kill on another body's
        // surface extensions. No RNG is drawn: non-leaking histories
        // are bit-identical.
        if let Some(exit_dist) = woodcock_flight_exit(
            geometry,
            neighbor_lists,
            geometry.vacuum_surfaces(),
            pre_flight_cell_idx,
            particle.position,
            particle.direction,
            d_flight,
        ) {
            particle.move_by(exit_dist);
            if any_tl_mesh && exit_dist > 0.0 {
                score_woodcock_mesh_track_length(
                    particle,
                    particle.position,
                    exit_dist,
                    ctx.tallies,
                    ctx.woodcock_tl_mesh_eligible,
                    welford_worker,
                );
            }
            // Mirror handle_surface_crossing's vacuum branch: a clean
            // leak with the live weight (not the old silent
            // weight-zeroing kill).
            tracker.record_termination(
                current_particle_id,
                current_parent_id,
                current_generation,
                particle.position,
                particle.direction,
                particle.energy,
                particle.weight,
                cell.cell_id,
                TrackEventType::Leak,
            );
            particle.alive = false;
            continue;
        }

        // Move the particle the full sampled distance -- no boundary
        // check beyond the vacuum-exit truncation above. This is the
        // structural difference from the hybrid Phase 1-3
        // implementation.
        particle.move_by(d_flight);

        // Locate the post-flight cell directly via the neighbor list
        // (which falls back to a full geometry scan when the new
        // position isn't in any pre-flight neighbour). `None` here is
        // now a defensive fallback: true vacuum exits are caught by
        // woodcock_flight_exit above, so this only fires for mesh
        // backends (no precomputed vacuum surfaces) and numerical
        // corner cases. The delta-collision is outside the geometry,
        // so the flight estimator scores nothing -- exactly the
        // behaviour that keeps it unbiased on leakage. (True
        // track-length mesh tallies are the exception: they score the
        // in-geometry part of the leaking segment below.)
        let pos = (
            particle.position[0],
            particle.position[1],
            particle.position[2],
        );
        let post_idx = match neighbor_lists.find_cell(geometry.cells(), pos, pre_flight_cell_idx) {
            Some(idx) => idx,
            None => {
                // True track-length mesh tallies still see the
                // in-geometry part of a leaking flight: clip the segment
                // at the geometry exit and score it before the kill
                // (surface tracking scores the same partial segment on
                // its way out). Skipping it would systematically
                // under-score escape-adjacent mesh voxels.
                if any_tl_mesh {
                    let clipped = distance_to_geometry_exit(
                        geometry,
                        neighbor_lists,
                        pre_flight_cell_idx,
                        particle.last_position,
                        particle.direction,
                        d_flight,
                    );
                    if clipped > 0.0 {
                        let end = [
                            particle.last_position[0] + particle.direction[0] * clipped,
                            particle.last_position[1] + particle.direction[1] * clipped,
                            particle.last_position[2] + particle.direction[2] * clipped,
                        ];
                        score_woodcock_mesh_track_length(
                            particle,
                            end,
                            clipped,
                            ctx.tallies,
                            ctx.woodcock_tl_mesh_eligible,
                            welford_worker,
                        );
                    }
                }
                particle.alive = false;
                particle.weight = 0.0;
                continue;
            }
        };
        // True track-length mesh tallies (issue #350): the whole flight
        // segment lies inside the geometry, so apportion it across the
        // crossed mesh voxels with the DDA. Material-independent by
        // eligibility, hence shared by the photon and neutron paths.
        if any_tl_mesh {
            score_woodcock_mesh_track_length(
                particle,
                particle.position,
                d_flight,
                ctx.tallies,
                ctx.woodcock_tl_mesh_eligible,
                welford_worker,
            );
        }
        particle.previous_cell_index = pre_flight_cell_idx as u32;
        particle.current_cell_index = post_idx as u32;
        cell_index = post_idx;
        let new_cell = &geometry.cells()[cell_index];
        let new_material_id = geometry
            .material_for(new_cell)
            .and_then(|m| m.get_material_id());
        let new_cell_material = geometry.material_for(new_cell);

        if is_photon {
            // Photon total XS at the landing point (None in vacuum).
            let photon_macro_xs =
                new_cell_material.map(|m| m.as_ref().calculate_photon_xs(particle.energy));

            // Delta-tracking flux estimator (TrackLength tallies) at the
            // delta-collision point -- scored for vacuum and material
            // cells alike, since flux is nonzero in voids too.
            score_woodcock_flight_estimator(
                particle,
                new_cell,
                new_cell_material,
                new_material_id,
                sigma_maj,
                None, // photons have no URR
                photon_macro_xs.as_ref(),
                ctx.tallies,
                ctx.woodcock_tl_mesh_eligible,
                ctx.transmutation_tallies,
                welford_worker,
            );

            // Vacuum at landing point -- purely fictitious, no collision
            // physics. Loop continues with `cell_index` already advanced.
            let Some(new_material_arc) = new_cell_material else {
                continue;
            };
            let new_material = new_material_arc.as_ref();
            let photon_macro_xs = photon_macro_xs.expect("material present implies photon XS");

            // Rejection samples against the macroscopic photon total XS.
            let p_real = (photon_macro_xs.total / sigma_maj).clamp(0.0, 1.0);
            let reject_xi: f64 = rng.random();
            if reject_xi >= p_real {
                // Fictitious event -- loop continues.
                continue;
            }

            // Real photon collision (shared with the surface-step path).
            do_photon_collision(
                particle,
                new_cell,
                new_material,
                new_material_id,
                &photon_macro_xs,
                particle_bank,
                ctx.photon_cutoff_energy,
                rng,
                tracker,
                ctx.tallies,
                welford_worker,
                current_particle_id,
                current_parent_id,
                current_generation,
                ctx.weight_windows,
            );
            continue;
        }

        // ---- Neutron rejection + collision ----
        // URR probability-table band selection. The band MUST be held
        // across all delta-collisions at a given energy, not re-sampled
        // at each one. A "real flight" spans many virtual collisions at
        // constant energy (only a real collision changes the energy), and
        // the free-flight distance to the next real collision is
        // governed by the band's Σ_t. Re-sampling the band every
        // delta-collision makes that distance track the *mean* URR cross
        // section, which destroys resonance self-shielding and biases the
        // flux low (survival exp(-⟨Σ⟩·x) instead of the self-shielded
        // Σ_b·p_b·exp(-Σ_b·x)). Reusing the cached band -- exactly as
        // surface tracking does per flight -- makes the penetration
        // correlate with the sampled cross section, so Woodcock
        // reproduces the self-shielded flux. The cache is keyed on the
        // energy it was sampled at; a real collision changes the energy
        // and forces a fresh band on the next flight.
        let new_urr_in_range = new_cell_material
            .map(|m| m.as_ref().has_urr_in_range(particle.energy))
            .unwrap_or(false);
        let new_cached_urr: Option<f64> = if new_urr_in_range {
            let held = if particle.urr_energy == particle.energy {
                yamc_particle::particle::urr_to_option(particle.urr_random)
            } else {
                None
            };
            let band = held.unwrap_or_else(|| rng.random());
            particle.urr_random = yamc_particle::particle::urr_from_option(Some(band));
            particle.urr_energy = particle.energy;
            Some(band)
        } else {
            // Non-URR landing point: leave any held band intact so it
            // survives a void / non-URR excursion at the same energy.
            None
        };
        let new_urr_macro_xs = if let (Some(u), Some(m)) = (new_cached_urr, new_cell_material) {
            m.compute_urr_macro_xs(particle.energy, u)
        } else {
            None
        };

        // Delta-tracking flux estimator (TrackLength tallies) at the
        // delta-collision point -- scored for vacuum and material cells
        // alike, since flux is nonzero in voids too.
        score_woodcock_flight_estimator(
            particle,
            new_cell,
            new_cell_material,
            new_material_id,
            sigma_maj,
            new_urr_macro_xs.as_ref(),
            None, // neutron path: no photon macro XS
            ctx.tallies,
            ctx.woodcock_tl_mesh_eligible,
            ctx.transmutation_tallies,
            welford_worker,
        );

        // Vacuum at landing point -- purely fictitious, no collision
        // physics. Loop continues with `cell_index` already advanced.
        let Some(new_material_arc) = new_cell_material else {
            continue;
        };
        let new_material = new_material_arc.as_ref();

        // Σ_t at the new position (URR-consistent via new cached random).
        let (sigma_t, _) = new_material.total_xs_with_urr(particle.energy, new_cached_urr, rng);
        let p_real = (sigma_t / sigma_maj).clamp(0.0, 1.0);
        let reject_xi: f64 = rng.random();
        if reject_xi >= p_real {
            // Fictitious event -- loop continues from the post-flight
            // position.
            continue;
        }

        // Real collision. Sample reaction data with the URR random
        // active for the new cell so the sampled nuclide is consistent
        // with the rejection denominator.
        let collision_data =
            new_material.sample_collision_data(particle.energy, new_cached_urr, rng);

        score_collision_event(
            particle,
            new_cell,
            new_material,
            new_material_id,
            new_urr_macro_xs.as_ref(),
            None, // photon macro XS not needed on the neutron path
            ctx.tallies,
            welford_worker,
        );

        // Pass the real photon flags so secondary (coupled) and D1S
        // decay photons are banked; they transport through this same
        // Woodcock loop when popped from the particle bank.
        handle_neutron_collision(
            ctx,
            particle,
            new_cell,
            new_material,
            collision_data,
            rng,
            pcg,
            tracker,
            particle_bank,
            current_particle_id,
            current_parent_id,
            current_generation,
            particle_idx,
        );
    }
}

/// Outcome of the analog (survival-off) reaction-type split (issue #111): which
/// scatter sub-channel `xi2` selected and the shared angle seed `xi3`. `None`
/// (the `analog_split` local being unset) means the legacy FastRng path handled
/// selection, which happens under survival biasing or when the material has no
/// `fast_xs` grid. URR bands take this path as of #111 gap 1. Multi-nuclide
/// materials DO take this
/// path: the split runs against the struck nuclide's partials, and nuclide
/// selection moved onto the shared stream in #131.
struct AnalogSplit {
    /// `xi2` landed in the elastic sub-range (route to `scatter_elastic`);
    /// otherwise the inelastic/other-scatter sub-range (draw `xi_mt`).
    is_elastic: bool,
    /// Angle seed drawn once at the split, mirroring the GPU twin's `xi3`.
    xi3: f64,
}

/// True when this scatter constituent is a DISCRETE inelastic level: its first
/// neutron product carries an uncorrelated angle-energy distribution whose
/// energy is `LevelInelastic` (closed-form-Q two-body). Such levels route through
/// `scatter_inelastic_level` on the shared PCG stream (issue #111 sub-step 3),
/// bit-identically to the GPU; continuum / correlated / Kalbach distributions
/// stay on the legacy tabulated FastRng path.
fn constituent_is_discrete_level(reaction: &Reaction) -> bool {
    use yamc_nuclide::reaction_product::{EnergyDistribution, ParticleType};
    reaction
        .products
        .iter()
        .find(|p| p.is_particle_type(&ParticleType::Neutron))
        .and_then(|p| p.distribution.first())
        .map(|d| {
            matches!(
                d,
                AngleEnergyDistribution::UncorrelatedAngleEnergy {
                    energy: Some(EnergyDistribution::LevelInelastic { .. }),
                    ..
                }
            )
        })
        .unwrap_or(false)
}

/// Per-history budget on weight-window split copies. Splitting stops once this
/// many copies have been created within one source history, leaving further
/// particles intact (their weight conserved) rather than growing the population
/// without bound. This caps the cost of a deep, heavily-splitting history while
/// keeping tallies unbiased: unlike a hard per-history particle cap (which
/// dropped still-queued split particles and destroyed their weight, biasing
/// DeGVR flux low), the budget never discards a particle -- the bank always
/// drains -- it only limits how many extra copies deep penetration spawns.
const WW_SPLIT_BUDGET_PER_HISTORY: usize = 10_000;

/// Apply a mesh weight window to a post-collision particle: split it when
/// its weight exceeds the local `upper` bound, roulette it when at or below
/// `lower`, and kill it below `weight_floor`. Weight is conserved (a split
/// makes N copies at `w/N`; roulette conserves in expectation), so tally
/// means are unchanged. A particle outside the mesh/energy binning or in a
/// sentinel voxel is left untouched.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_weight_window<T: Tracker>(
    weight_windows: &[&crate::variance_reduction::WeightWindowBounds],
    particle: &mut yamc_particle::particle::Particle,
    cell: &crate::geometry::cell::Cell,
    rng: &mut FastRng,
    particle_bank: &mut ParticleBank,
    tracker: &mut T,
    current_particle_id: u64,
    current_parent_id: Option<u64>,
    current_generation: u32,
) {
    let Some(&ww) = weight_windows
        .iter()
        .find(|w| w.particle == particle.particle_type)
    else {
        return;
    };
    let Some((lower, upper)) = ww.window(particle.position, particle.energy) else {
        return;
    };
    let w = particle.weight;
    // Absolute weight floor: kill below it.
    if w < ww.weight_floor {
        tracker.record_termination(
            current_particle_id,
            current_parent_id,
            current_generation,
            particle.position,
            particle.direction,
            particle.energy,
            w,
            cell.cell_id,
            TrackEventType::RussianRoulette,
        );
        particle.alive = false;
        particle.weight = 0.0;
        return;
    }
    if w > upper {
        // Split into n = min(ceil(w/upper), max_split) equal-weight copies, but
        // bound the copy count by the per-history split budget still available.
        // Weight is conserved across exactly the copies created (each w/n), and
        // when the budget is spent the particle is left intact (weight w, no
        // split). Nothing is ever discarded, so tallies stay unbiased; the
        // budget only caps how far a deep, heavily-splitting history grows the
        // population. (A hard per-history particle cap that dropped queued split
        // particles used to destroy their weight and bias DeGVR flux low.)
        let n_target = ((w / upper).ceil() as u64).clamp(1, ww.max_split as u64);
        let budget_left =
            WW_SPLIT_BUDGET_PER_HISTORY.saturating_sub(particle_bank.ww_splits()) as u64;
        let n = n_target.min(1 + budget_left); // total copies including the original
        if n >= 2 {
            let split_weight = w / n as f64;
            particle.weight = split_weight;
            for _ in 1..n {
                let mut child = particle.clone();
                child.weight = split_weight;
                particle_bank.bank_secondary(child);
            }
            particle_bank.record_ww_splits((n - 1) as usize);
        }
    } else if w <= lower {
        // Roulette toward survival_factor * lower, clamped so a very
        // low-weight particle is boosted by at most max_split in one event.
        let survive_weight = (w * ww.max_split as f64).min(ww.survival_factor * lower);
        let xi: f64 = rng.random();
        match weight_cutoff_roulette(w, survive_weight, xi) {
            Some(surviving_weight) => particle.weight = surviving_weight,
            None => {
                tracker.record_termination(
                    current_particle_id,
                    current_parent_id,
                    current_generation,
                    particle.position,
                    particle.direction,
                    particle.energy,
                    w,
                    cell.cell_id,
                    TrackEventType::RussianRoulette,
                );
                particle.alive = false;
                particle.weight = 0.0;
            }
        }
    }
    // else: lower < w <= upper -- the particle is in-window; nothing to do.
}

/// Process a real neutron collision in `cell` (material `material`) at the current
/// particle position. Extracted from `transport_particle` so the inner loop is
/// easier to read; behaviour is preserved verbatim.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "debug_collision"), allow(unused_variables))]
pub(crate) fn handle_neutron_collision<T: Tracker>(
    ctx: &TransportCtx,
    particle: &mut yamc_particle::particle::Particle,
    cell: &crate::geometry::cell::Cell,
    material: &Material,
    collision_data: Option<CollisionData<'_>>,
    rng: &mut FastRng,
    // Per-particle 64-bit PCG state for sub-steps migrated onto the shared
    // GPU/CPU collision path (issues #111, #274). Currently consumed by
    // elastic scatter; other sub-steps still draw from `rng` until migrated.
    pcg: &mut u64,
    tracker: &mut T,
    particle_bank: &mut ParticleBank,
    current_particle_id: u64,
    current_parent_id: Option<u64>,
    current_generation: u32,
    particle_idx: usize,
) {
    use yamc_rng::next_xi;
    let transport_secondary_photons = ctx.transport_secondary_photons;
    let use_decay_photons = ctx.use_decay_photons;
    let free_gas_threshold = ctx.free_gas_threshold;
    let decay_photon_nuclide_data = ctx.decay_photon_nuclide_data;
    if let Some((_, nuclide_name, nuclide, _, nuclide_id)) = collision_data {
        // Reaction-type selection.
        //
        // Both the analog and the survival-biased split now run on the shared
        // PCG stream (issue #111 gaps 1 and 2). `xi2` partitions the *struck*
        // nuclide's (sigma_e, sigma_a, sigma_i, sigma_f) -- the GPU twin's
        // branch on the selected nuclide's partials (yamc-gpu shared.rs) --
        // then `xi3` seeds the scatter angle and a per-MT `xi_mt` (drawn in the
        // inelastic arm) selects the constituent. URR rides the band the flight
        // sampled against, which `reaction_partials` applies.
        //
        // Under survival biasing the denominator is the SCATTER cross section
        // rather than the total, so the survivor always scatters and the weight
        // carries the capture-and-fission fraction away. Fission progeny are
        // banked from the pre-discount weight below. That is OpenMC's scheme
        // (`physics.cpp`: fission sites first, then
        // `wgt -= wgt * absorption/total` with fission inside `absorption`,
        // then the survivor scatters), and for a non-fissile nuclide it is
        // identical to what the kernel already does, since its `sigma_sf`
        // collapses to `sigma_e + sigma_i` there.
        let mut analog_split: Option<AnalogSplit> = None;
        // Angle seed for an analog FISSION collision: the same `xi3` the split
        // drew, which the GPU kernel's fission branch reuses as the continuing
        // progeny's isotropic cosine instead of drawing again (issue #111).
        // `None` on the legacy no-fast-grid selection, which makes no split
        // draw -- the fission arm then draws its own, as the elastic arm does.
        let mut fission_angle_xi: Option<f64> = None;
        // `(scatter/total, fission/total)` for a survival-biased collision:
        // the surviving weight fraction, and the fission fraction the banked
        // progeny are scaled by. Applied after photon sampling.
        let mut survival_capture: Option<(f64, f64)> = None;
        let reaction_type = {
            match nuclide.reaction_partials(
                particle.energy,
                material.temperature(),
                yamc_particle::particle::urr_to_option(particle.urr_random),
                rng,
            ) {
                // No fast_xs grid: fall back to the legacy selection (which
                // carries the slow path). `elastic_reaction` /
                // `sample_inelastic_scatter_reaction` are then never reached
                // without `fast_xs`, so their fast-path lookups are safe.
                None => nuclide.sample_reaction_type(
                    particle.energy,
                    material.temperature(),
                    yamc_particle::particle::urr_to_option(particle.urr_random),
                    rng,
                ),
                Some(p) => {
                    // Analog: the denominator is the full total, mirroring
                    // shared.rs (sel_denom == sigma_t). Survival biasing
                    // narrows it to the scatter cross section, which makes the
                    // absorption and fission arms unreachable -- the survivor
                    // always scatters.
                    let sigma_t = p.sigma_e + p.sigma_a + p.sigma_i + p.sigma_f;
                    let sigma_scatter = p.sigma_e + p.sigma_i;
                    let survival_on = ctx.survival_biasing && sigma_scatter > 0.0;
                    let sel_denom = if survival_on { sigma_scatter } else { sigma_t };
                    if survival_on {
                        // Record it; the weight reduction and the fission
                        // banking happen after secondary-photon sampling below,
                        // so photons inherit the PRE-discount weight. That is
                        // OpenMC's order (`sample_neutron_reaction`: fission
                        // sites, then photons, then the absorption discount).
                        // Deferring costs no stream alignment, because neither
                        // consumes a draw the GPU also makes here.
                        survival_capture = Some((sigma_scatter / sigma_t, p.sigma_f / sigma_t));
                    }
                    let p_elastic = p.sigma_e / sel_denom;
                    let p_scatter = (p.sigma_e + p.sigma_i) / sel_denom;
                    let p_fission_or_scatter = (p.sigma_e + p.sigma_i + p.sigma_f) / sel_denom;
                    let xi2 = next_xi(pcg);
                    if xi2 >= p_fission_or_scatter {
                        Some(ReactionType::Absorption)
                    } else {
                        // Angle seed: one draw for every non-absorption channel,
                        // mirroring the GPU twin's single `xi3`. Consumed by the
                        // scatter branches as the CM/level angle seed and by the
                        // fission branch as the continuing progeny's isotropic
                        // cosine, exactly as the GPU twin consumes it.
                        let xi3 = next_xi(pcg);
                        if xi2 < p_scatter {
                            analog_split = Some(AnalogSplit {
                                is_elastic: xi2 < p_elastic,
                                xi3,
                            });
                            Some(ReactionType::Scattering)
                        } else {
                            fission_angle_xi = Some(xi3);
                            Some(ReactionType::Fission)
                        }
                    }
                }
            }
        };
        if let Some(reaction_type) = reaction_type {
            // Debug collision logging: log reaction selection
            #[cfg(feature = "debug_collision")]
            log_reaction_selection(particle_idx, particle.energy, &reaction_type, nuclide_name);

            // Debug: count collision
            #[cfg(feature = "debug_runtime")]
            TOTAL_COLLISIONS.fetch_add(1, Ordering::Relaxed);

            // Track the reaction for tally scoring
            let tally_mt: i32;
            // Save incoming energy and direction before reaction changes them
            // (needed for collision tally and photon production)
            let incoming_energy = particle.energy;
            let incoming_direction = particle.direction;

            // Sample secondary photons from neutron collision
            // IMPORTANT: This must happen BEFORE the reaction match to use
            // the neutron's pre-scatter weight. Correct ordering:
            // sample_secondary_photons() → absorption() → scatter()
            if transport_secondary_photons {
                if let Some(temp_idx) = nuclide.get_temp_idx(material.temperature()) {
                    if let Some(fast_grid) = nuclide.fast_xs.get(temp_idx) {
                        let (i_grid, f) = fast_grid.lookup_grid_index(incoming_energy);
                        if use_decay_photons {
                            // D1S mode: use precomputed decay photon data
                            // (photon yields were replaced with decay data at setup time).
                            // `nuclide_id` comes from `Material.sample_collision_data`;
                            // indexing `decay_photon_nuclide_data` by it avoids a String-keyed HashMap lookup.
                            if let Some(id) = nuclide_id {
                                let slot = id.get() as usize - 1;
                                if let Some(decay_photon_temps) =
                                    decay_photon_nuclide_data.get(slot)
                                {
                                    if let Some(temp_idx) =
                                        nuclide.get_temp_idx(material.temperature())
                                    {
                                        if let Some(decay_photon_nuc) =
                                            decay_photon_temps.get(temp_idx)
                                        {
                                            let (total_xs, _, _, _) =
                                                fast_grid.lookup(incoming_energy);
                                            yamc_physics::photon::decay_photon_production::sample_decay_photons(
                                                particle,
                                                decay_photon_nuc,
                                                i_grid,
                                                f,
                                                total_xs,
                                                particle_bank,
                                                rng,
                                            );
                                        }
                                    }
                                }
                            }
                        } else {
                            // Prompt mode: sample from nuclear data
                            let photon_prod_xs = fast_grid.lookup_photon_prod(i_grid, f);
                            if photon_prod_xs > 0.0 {
                                let (total_xs, _, _, _) = fast_grid.lookup(incoming_energy);
                                yamc_physics::photon::photon_production::sample_secondary_photons(
                                    particle,
                                    incoming_energy,
                                    incoming_direction,
                                    nuclide,
                                    fast_grid,
                                    i_grid,
                                    f,
                                    total_xs,
                                    photon_prod_xs,
                                    particle_bank,
                                    rng,
                                );
                            }
                        }
                    }
                }
            }

            // Implicit capture, applied here so the collision estimator and the
            // secondary photons above both saw the pre-discount weight, which is
            // OpenMC's order. Fission progeny are banked from that same
            // pre-discount weight, scaled by the nuclide's fission fraction,
            // because with survival biasing on fission never "happens" to the
            // survivor -- it is part of the absorbed weight.
            if let Some((scatter_frac, fission_frac)) = survival_capture {
                if fission_frac > 0.0 {
                    // Every progeny is banked here (the survivor scatters
                    // instead of continuing as a fission neutron), so there is
                    // no continuing walk to hand the split's angle seed to --
                    // that seed belongs to the scatter branch below. Draw the
                    // batch's first isotropic cosine here instead.
                    let mu_xi = next_xi(pcg);
                    let (_, fission_neutrons) = sample_fission_event(
                        nuclide,
                        material.temperature(),
                        particle,
                        fission_frac,
                        mu_xi,
                        pcg,
                        rng,
                    );
                    let produced = fission_neutrons.len();
                    for fission_n in fission_neutrons {
                        particle_bank.bank_secondary(fission_n);
                    }
                    charge_fission_progeny(particle_bank, produced, material, nuclide_name);
                }
                particle.weight *= scatter_frac;
            }

            match reaction_type {
                ReactionType::Scattering => {
                    // Debug: count scattering reaction type
                    #[cfg(feature = "debug_runtime")]
                    TOTAL_SCATTERING_REACTIONS.fetch_add(1, Ordering::Relaxed);

                    if let Some(split) = analog_split {
                        // Analog single-nuclide path (#111): `xi2` already chose
                        // elastic vs inelastic; route on the shared PCG stream.
                        if split.is_elastic {
                            let elastic = nuclide
                                .elastic_reaction(material.temperature())
                                .expect("elastic reaction present when sigma_e > 0");
                            // `scatter_elastic` always returns true (the free-gas
                            // sampler handles the degenerate CM-speed case
                            // internally), so there is no early-return to check.
                            scatter_elastic(
                                particle,
                                nuclide,
                                material,
                                elastic,
                                free_gas_threshold,
                                particle_idx,
                                split.xi3,
                                pcg,
                            );
                            tally_mt = 2;
                        } else {
                            // Per-MT inelastic constituent draw (mirrors the GPU
                            // twin's `xi_mt`), then route to the existing
                            // kinematics (those move onto PCG in a later #111
                            // sub-step).
                            let xi_mt = next_xi(pcg);
                            let constituent = nuclide
                                .sample_inelastic_scatter_reaction(
                                    particle.energy,
                                    material.temperature(),
                                    xi_mt,
                                )
                                .expect("non-elastic constituent present when sigma_i > 0");
                            let proceed = match constituent.mt_number {
                                // Discrete levels: closed-form-Q on the shared PCG
                                // stream, bit-identical to the GPU (issue #111
                                // sub-step 3). `split.xi3` is the angle seed drawn
                                // at the reaction-type split, as the GPU does.
                                50..=91 | 875..=890
                                    if constituent_is_discrete_level(constituent) =>
                                {
                                    scatter_inelastic_level(
                                        particle,
                                        nuclide,
                                        constituent,
                                        particle_idx,
                                        particle_bank,
                                        split.xi3,
                                        pcg,
                                    )
                                }
                                // Continuum (e.g. MT 91) / other tabulated
                                // levels: the shared per-law samplers on the
                                // PCG stream, reading the same flattened
                                // tables the GPU packs into its buffers, so
                                // these collisions are bit-identical to the
                                // GPU twin too (issue #111 sub-step 3).
                                // `None` means the flat layer does not
                                // recognise this reaction's outgoing-energy
                                // law, so the legacy FastRng sampler (which
                                // covers more laws) still handles it.
                                50..=91 | 875..=890 => scatter_inelastic_shared(
                                    particle,
                                    nuclide,
                                    constituent,
                                    particle_idx,
                                    particle_bank,
                                    split.xi3,
                                    pcg,
                                    ctx.inelastic_flat_cache,
                                )
                                .unwrap_or_else(|| {
                                    scatter_inelastic(
                                        particle,
                                        nuclide,
                                        constituent,
                                        particle_idx,
                                        particle_bank,
                                        rng,
                                    )
                                }),
                                // (n,xn) / (n,n'x): MT 5 / 16 / 17 / 22 / 28 /
                                // ... on the shared flat samplers and the PCG
                                // stream, reading the same tables the GPU packs
                                // into its per-MT buffers, so these collisions
                                // are bit-identical to the GPU twin (issue
                                // #111). `None` means the flat layer does not
                                // recognise this reaction's outgoing-energy law
                                // (or it emits no neutron), so the legacy
                                // FastRng `scatter_other` still handles it.
                                // Either way this arm always proceeds.
                                _ => {
                                    scatter_other_shared(
                                        particle,
                                        nuclide,
                                        constituent,
                                        particle_idx,
                                        particle_bank,
                                        split.xi3,
                                        pcg,
                                        ctx.inelastic_flat_cache,
                                    )
                                    .unwrap_or_else(|| {
                                        scatter_other(
                                            particle,
                                            nuclide,
                                            nuclide_name,
                                            constituent,
                                            particle_bank,
                                            rng,
                                        )
                                    });
                                    true
                                }
                            };
                            if !proceed {
                                return;
                            }
                            tally_mt = constituent.mt_number;
                        }
                    } else {
                        // Legacy path (survival biasing or multi-nuclide): pick
                        // the constituent from FastRng and route by MT. The
                        // elastic arm draws `xi3` here so its PCG schedule is
                        // unchanged from before the seed was threaded out of
                        // `scatter_elastic`.
                        let constituent_reaction = nuclide.sample_scattering_constituent(
                            particle.energy,
                            material.temperature(),
                            rng,
                        );

                        // Handle the sampled constituent reaction. Elastic and
                        // inelastic can terminate the history early (degenerate
                        // CM speed / no outgoing particles); they return `false`
                        // to signal the caller to return immediately, exactly as
                        // the original inline `return`s did.
                        let proceed = match constituent_reaction.mt_number {
                            2 => {
                                let xi3 = next_xi(pcg);
                                scatter_elastic(
                                    particle,
                                    nuclide,
                                    material,
                                    constituent_reaction,
                                    free_gas_threshold,
                                    particle_idx,
                                    xi3,
                                    pcg,
                                )
                            }
                            50..=91 | 875..=890 => scatter_inelastic(
                                particle,
                                nuclide,
                                constituent_reaction,
                                particle_idx,
                                particle_bank,
                                rng,
                            ),
                            _ => {
                                scatter_other(
                                    particle,
                                    nuclide,
                                    nuclide_name,
                                    constituent_reaction,
                                    particle_bank,
                                    rng,
                                );
                                true
                            }
                        };
                        if !proceed {
                            return;
                        }

                        // Use constituent reaction for tally scoring
                        tally_mt = constituent_reaction.mt_number;
                    }
                }
                ReactionType::Fission => {
                    // Debug: count fission reactions
                    #[cfg(feature = "debug_runtime")]
                    TOTAL_FISSION_REACTIONS.fetch_add(1, Ordering::Relaxed);

                    // Continuing progeny's isotropic cosine: the split's angle
                    // seed, as the GPU kernel reuses it. The legacy no-fast-grid
                    // selection drew no seed, so draw one here (mirroring its
                    // elastic arm).
                    let mu_xi = fission_angle_xi.unwrap_or_else(|| next_xi(pcg));
                    let (fission_mt, fission_neutrons) = sample_fission_event(
                        nuclide,
                        material.temperature(),
                        particle,
                        1.0,
                        mu_xi,
                        pcg,
                        rng,
                    );
                    tally_mt = fission_mt;

                    let produced = fission_neutrons.len();
                    if fission_neutrons.is_empty() {
                        particle.alive = false;
                    } else {
                        // First neutron continues as current particle
                        let first = &fission_neutrons[0];
                        particle.energy = first.energy;
                        particle.direction = first.direction;

                        // Bank remaining neutrons
                        for fission_n in fission_neutrons.into_iter().skip(1) {
                            particle_bank.bank_secondary(fission_n);
                        }
                    }
                    charge_fission_progeny(particle_bank, produced, material, nuclide_name);
                }
                ReactionType::Absorption => {
                    // Debug: count absorption reactions
                    #[cfg(feature = "debug_runtime")]
                    TOTAL_ABSORPTION_REACTIONS.fetch_add(1, Ordering::Relaxed);

                    let constituent_reaction = nuclide.sample_absorption_constituent(
                        particle.energy,
                        material.temperature(),
                        rng,
                    );
                    tally_mt = constituent_reaction.mt_number;

                    // Record absorption termination for tracking
                    tracker.record_termination(
                        current_particle_id,
                        current_parent_id,
                        current_generation,
                        particle.position,
                        particle.direction,
                        particle.energy,
                        particle.weight,
                        cell.cell_id,
                        TrackEventType::Absorption,
                    );

                    particle.alive = false;
                }
            }

            // Record collision for particle tracking
            // Note: distribution tracking requires changes to reaction sampling to capture
            // which distribution was used - for now pass None
            tracker.record_collision(
                current_particle_id,
                current_parent_id,
                current_generation,
                particle.position,
                particle.direction,
                incoming_energy,
                particle.energy,
                particle.weight,
                cell.cell_id,
                tally_mt,
                nuclide_name,
                None, // distribution name - would need reaction sampling changes
                None, // energy dist name - would need reaction sampling changes
            );

            // Debug history: record collision and print if particle enters problematic range
            #[cfg(feature = "debug_history")]
            {
                particle.record_collision(incoming_energy, particle.energy, tally_mt, nuclide_name);
                // Print history if particle is in 2000-4300 eV range after many collisions
                // This helps identify how particles accumulate in this range
                if particle.energy >= 2000.0
                    && particle.energy <= 4300.0
                    && particle.history.len() >= 5
                {
                    use std::sync::atomic::{AtomicUsize, Ordering};
                    static DEBUG_HISTORY_COUNT: AtomicUsize = AtomicUsize::new(0);
                    let count = DEBUG_HISTORY_COUNT.fetch_add(1, Ordering::Relaxed);
                    if count < 20 {
                        // Print first 20 particles
                        eprintln!("\n*** PARTICLE IN PROBLEMATIC RANGE (2000-4300 eV) WITH {} COLLISIONS, weight={} ***",
                            particle.history.len(), particle.weight);
                        particle.print_history();
                    }
                }
            }
            // Weight-cutoff Russian roulette: cull low-weight survivors so
            // implicit capture does not accumulate near-zero-weight
            // histories. Survival probability weight / weight_survive
            // keeps the expected weight unchanged; survivors continue at
            // weight_survive. Exactly one draw per roulette, on the shared PCG
            // stream and in the kernel's position (last thing in the collision,
            // only below the cutoff), so a survival-biased history stays in
            // lockstep with the GPU (issue #111 gap 2).
            if ctx.survival_biasing && particle.alive && particle.weight < ctx.weight_cutoff {
                let xi: f64 = next_xi(pcg);
                match weight_cutoff_roulette(particle.weight, ctx.weight_survive, xi) {
                    Some(surviving_weight) => particle.weight = surviving_weight,
                    None => {
                        tracker.record_termination(
                            current_particle_id,
                            current_parent_id,
                            current_generation,
                            particle.position,
                            particle.direction,
                            particle.energy,
                            particle.weight,
                            cell.cell_id,
                            TrackEventType::RussianRoulette,
                        );
                        particle.alive = false;
                        particle.weight = 0.0;
                    }
                }
            }
            // Weight-window check: split high-weight particles and roulette
            // low-weight ones toward the per-voxel target band. Independent
            // of survival biasing; composes after it. CPU-only.
            if particle.alive && !ctx.weight_windows.is_empty() {
                apply_weight_window(
                    ctx.weight_windows,
                    particle,
                    cell,
                    rng,
                    particle_bank,
                    tracker,
                    current_particle_id,
                    current_parent_id,
                    current_generation,
                );
            }
        } else {
            panic!(
                "No valid reaction found for nuclide {} at energy {}",
                nuclide_name, particle.energy
            );
        }
    } else {
        // Use slower method to get nuclide name for error message
        let nuclide_name = material.sample_interacting_nuclide(particle.energy, rng);
        panic!("Nuclide {nuclide_name} not found in material data");
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use yamc_particle::particle::Particle;
    use yamc_tallies::welford::WelfordWorkerState;
    use yamc_tallies::{Estimator, FluxScore, Score};

    // --- photon collision helpers ----------------------------------------

    #[test]
    fn bank_fluorescence_collect_auger_splits_photons_and_electrons() {
        // Relaxation cascade: one fluorescent photon above the cutoff, one
        // below, plus an Auger electron and a zero-energy electron.
        let particle = Particle::new([1.0, 2.0, 3.0], [1.0, 0.0, 0.0], 1.0e6);
        let mut bank = ParticleBank::new();
        let cutoff = 1000.0;
        let secondaries = vec![
            (5000.0, [1.0, 0.0, 0.0], true),  // photon > cutoff  -> banked
            (500.0, [0.0, 1.0, 0.0], true),   // photon < cutoff  -> dropped
            (3000.0, [0.0, 0.0, 1.0], false), // Auger electron   -> collected
            (0.0, [0.0, -1.0, 0.0], false),   // zero-energy e-   -> dropped
        ];

        let (banked, auger) =
            bank_fluorescence_collect_auger(secondaries, &particle, cutoff, &mut bank, false);

        // Only the above-cutoff photon contributes banked energy + a particle.
        assert_eq!(banked, 5000.0);
        assert_eq!(bank.len(), 1);
        let secondary = bank.pop_particle().unwrap();
        assert_eq!(secondary.particle_type, ParticleType::Photon);
        assert_eq!(secondary.energy, 5000.0);
        assert_eq!(secondary.position, particle.position);
        assert_eq!(secondary.weight, particle.weight);

        // Only the positive-energy electron is collected for downstream TTB.
        assert_eq!(auger.len(), 1);
        assert_eq!(auger[0].0, 3000.0);
        assert_eq!(auger[0].1, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn weight_cutoff_roulette_survives_below_threshold_and_kills_above() {
        // Survival probability = weight / weight_survive = 0.1 / 1.0 = 0.1.
        assert_eq!(weight_cutoff_roulette(0.1, 1.0, 0.05), Some(1.0)); // xi < p -> survive at weight_survive
        assert_eq!(weight_cutoff_roulette(0.1, 1.0, 0.5), None); // xi >= p -> killed
                                                                 // xi exactly at the threshold is killed (strict `<`).
        assert_eq!(weight_cutoff_roulette(0.1, 1.0, 0.1), None);
        // A weight already at weight_survive always survives (p = 1.0 > any xi < 1).
        assert_eq!(weight_cutoff_roulette(1.0, 1.0, 0.999), Some(1.0));
    }

    #[test]
    fn bank_fluorescence_collect_auger_at_cutoff_is_banked() {
        // Photons exactly at the cutoff are banked (>= comparison).
        let particle = Particle::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0e6);
        let mut bank = ParticleBank::new();
        let (banked, auger) = bank_fluorescence_collect_auger(
            vec![(1000.0, [1.0, 0.0, 0.0], true)],
            &particle,
            1000.0,
            &mut bank,
            false,
        );
        assert_eq!(banked, 1000.0);
        assert_eq!(bank.len(), 1);
        assert!(auger.is_empty());
    }

    // --- shared fixtures -------------------------------------------------

    /// Build a minimal void (no-data) material. `calculate_photon_xs`
    /// returns `total = 0.0` for it (empty `cached_elements`), and
    /// `lookup_xs_by_mt` returns 0.0 -- exactly what the void-branch tests
    /// of the sampling/scoring helpers need.
    fn empty_material() -> Material {
        Material::new(std::collections::HashMap::new(), "atom", "sum", None).unwrap()
    }

    /// Build a trivial spherical cell with the given id. The scoring
    /// helpers only read `cell.cell_id`, so the region content is
    /// irrelevant -- any valid region works.
    fn make_cell(cell_id: u32) -> crate::geometry::cell::Cell {
        use crate::geo::Surface;
        use crate::geo::{HalfspaceType, Region};
        let sphere = Surface::new_sphere(0.0, 0.0, 0.0, 1.0, Some(1), None);
        let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
        crate::geometry::cell::Cell::new(Some(cell_id), region, None, None)
    }

    /// Build a single-score flux tally with the requested estimator and no
    /// filters. After `update_cache`, such a tally has exactly one bin
    /// (bin 0) and scores `weight * track_length` (track-length) or
    /// `weight / Sigma_t` (collision) into it.
    fn flux_tally(estimator: Estimator) -> Tally {
        let mut t = Tally::new();
        t.scores = vec![Score::Flux(FluxScore)];
        t.estimator = estimator;
        t.update_cache();
        t
    }

    /// Read the contribution scored into bin 0 of tally 0 for the current
    /// (not-yet-finished) history. Returns 0.0 if nothing was scored.
    fn bin0(welford: &WelfordWorkerState) -> f64 {
        welford.tallies[0]
            .scratch_map
            .get(&0u32)
            .copied()
            .unwrap_or(0.0)
    }

    // --- cell_mean_chord -------------------------------------------------
    //
    // Implements the isotropic mean chord of the bbox, 4V / S, with
    //   V = a*b*c,  S = 2(ab + bc + ca).
    // Unbounded -> INFINITY; zero-surface (degenerate) -> 0.

    #[test]
    fn cell_mean_chord_unit_cube() {
        // a=b=c=1: V=1, S=6 -> 4*1/6 = 2/3.
        let bbox = crate::geo::BoundingBox::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let chord = cell_mean_chord(&bbox);
        assert!((chord - 2.0 / 3.0).abs() < 1e-15, "got {chord}");
    }

    #[test]
    fn cell_mean_chord_cuboid() {
        // a=2, b=3, c=4: V=24, S=2(6+12+8)=52 -> 4*24/52 = 96/52 = 24/13.
        let bbox = crate::geo::BoundingBox::new([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]);
        let chord = cell_mean_chord(&bbox);
        assert!((chord - 24.0 / 13.0).abs() < 1e-15, "got {chord}");
    }

    #[test]
    fn cell_mean_chord_offset_box_uses_widths_only() {
        // Translating the box must not change the chord (depends on widths,
        // not absolute position): widths 2x3x4 again -> 24/13.
        let bbox = crate::geo::BoundingBox::new([-5.0, 10.0, 7.0], [-3.0, 13.0, 11.0]);
        let chord = cell_mean_chord(&bbox);
        assert!((chord - 24.0 / 13.0).abs() < 1e-15, "got {chord}");
    }

    #[test]
    fn cell_mean_chord_unbounded_is_infinite() {
        let bbox =
            crate::geo::BoundingBox::new([f64::NEG_INFINITY, 0.0, 0.0], [f64::INFINITY, 1.0, 1.0]);
        assert_eq!(cell_mean_chord(&bbox), f64::INFINITY);
    }

    #[test]
    fn cell_mean_chord_degenerate_zero_surface() {
        // A box with two zero widths has zero surface area -> 0 (the
        // surface <= 0 guard), not NaN from a 0/0 division.
        let bbox = crate::geo::BoundingBox::new([0.0, 0.0, 0.0], [5.0, 0.0, 0.0]);
        assert_eq!(cell_mean_chord(&bbox), 0.0);
    }

    // --- sample_distance_to_collision -----------------------------------
    //
    // Photon path: -ln(xi)/Sigma_t when Sigma_t > 0, else INFINITY.
    // Neutron path: defers to Material::sample_collision_data.
    // Void cell (no material): always INFINITY. The Sigma_t > 0 branch
    // needs a material carrying loaded photon-element data (heavy on-disk
    // fixtures), so it is exercised by the data-backed integration tests in
    // crates/yamc/tests/, not here; the deterministic control-flow branches
    // (void, zero-XS) are unit-tested below.

    #[test]
    fn sample_distance_void_photon_is_infinite() {
        let mut p = Particle::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0e6);
        p.particle_type = ParticleType::Photon;
        let mut rng = FastRng::new(1);
        let out = sample_distance_to_collision(&p, None, &mut rng);
        assert_eq!(out.dist_collision, f64::INFINITY);
        assert!(out.photon_macro_xs.is_none());
        assert!(out.collision_data.is_none());
    }

    #[test]
    fn sample_distance_void_neutron_is_infinite() {
        let p = Particle::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 2.0e6);
        let mut rng = FastRng::new(1);
        let out = sample_distance_to_collision(&p, None, &mut rng);
        assert_eq!(out.dist_collision, f64::INFINITY);
        assert!(out.collision_data.is_none());
        assert!(out.photon_macro_xs.is_none());
    }

    #[test]
    fn sample_distance_photon_zero_xs_is_infinite() {
        // A material with no element data has total photon XS == 0, so the
        // `xs.total > 0.0` guard fails and the distance is INFINITY (rather
        // than -ln(xi)/0 = +/-inf/NaN). The returned `photon_macro_xs` is
        // still Some (the zero-valued lookup), which downstream code relies
        // on for the photon branch.
        let mat = Arc::new(empty_material());
        let mut p = Particle::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0e6);
        p.particle_type = ParticleType::Photon;
        let mut rng = FastRng::new(42);
        let out = sample_distance_to_collision(&p, Some(&mat), &mut rng);
        assert_eq!(out.dist_collision, f64::INFINITY);
        let xs = out.photon_macro_xs.expect("photon path returns Some(xs)");
        assert_eq!(xs.total, 0.0);
    }

    // Note: the neutron path with a material present defers to
    // Material::sample_collision_data, which requires a built neutron XS
    // cache (loaded nuclear data). That branch is covered by the
    // data-backed integration tests; the void (None material) neutron path
    // above already pins the in-isolation INFINITY behavior.

    // --- score_track_length_segment -------------------------------------
    //
    // For a flux-only TrackLength tally with no filters, each segment of
    // length `dist` contributes `weight * dist` to bin 0. The returned
    // end position is `position + direction * dist`.

    #[test]
    fn score_track_length_flux_contribution() {
        let cell = make_cell(1);
        let tally = flux_tally(Estimator::TrackLength);
        let tallies = [&tally];
        let mut welford = WelfordWorkerState::new(&[tally.num_bins()]);

        let mut p = Particle::new([1.0, 2.0, 3.0], [1.0, 0.0, 0.0], 1.0e6);
        p.weight = 0.5;
        let dist = 4.0;

        let end = score_track_length_segment(
            &p,
            &cell,
            None, // void cell: flux scoring needs no material data
            None,
            dist,
            None,
            &tallies,
            None,
            &mut welford,
        );

        // flux = weight * dist = 0.5 * 4.0 = 2.0
        assert!(
            (bin0(&welford) - 2.0).abs() < 1e-12,
            "got {}",
            bin0(&welford)
        );
        // end position = start + dir * dist along +x.
        assert!((end[0] - 5.0).abs() < 1e-12);
        assert_eq!(end[1], 2.0);
        assert_eq!(end[2], 3.0);
    }

    #[test]
    fn score_track_length_skips_collision_estimator_tally() {
        // A Collision-estimator tally must NOT be scored by the
        // track-length helper (it scores at collision sites instead).
        let cell = make_cell(1);
        let tally = flux_tally(Estimator::Collision);
        let tallies = [&tally];
        let mut welford = WelfordWorkerState::new(&[tally.num_bins()]);

        let p = Particle::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0e6);
        let _ = score_track_length_segment(
            &p,
            &cell,
            None,
            None,
            4.0,
            None,
            &tallies,
            None,
            &mut welford,
        );
        assert_eq!(bin0(&welford), 0.0);
    }

    // --- score_collision_event ------------------------------------------
    //
    // For a flux-only Collision-estimator tally, a neutron collision with
    // total XS Sigma_t contributes the unbiased estimator weight / Sigma_t
    // to bin 0. Sigma_t comes from the URR struct when supplied.

    #[test]
    fn score_collision_flux_contribution() {
        let cell = make_cell(1);
        let mat = empty_material();
        let tally = flux_tally(Estimator::Collision);
        let tallies = [&tally];
        let mut welford = WelfordWorkerState::new(&[tally.num_bins()]);

        let mut p = Particle::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0e6);
        p.weight = 2.0;
        // Supply Sigma_t via the URR struct so we don't depend on loaded
        // nuclear data: total = 4.0 -> flux = weight / Sigma_t = 2 / 4 = 0.5.
        let urr = UrrMacroXs {
            total: 4.0,
            elastic: 0.0,
            fission: 0.0,
            capture: 0.0,
            absorption: 0.0,
        };

        score_collision_event(
            &p,
            &cell,
            &mat,
            None,
            Some(&urr),
            None,
            &tallies,
            &mut welford,
        );

        assert!(
            (bin0(&welford) - 0.5).abs() < 1e-12,
            "got {}",
            bin0(&welford)
        );
    }

    #[test]
    fn score_collision_skips_track_length_tally() {
        // A TrackLength-estimator tally must be ignored by the
        // collision-event helper.
        let cell = make_cell(1);
        let mat = empty_material();
        let tally = flux_tally(Estimator::TrackLength);
        let tallies = [&tally];
        let mut welford = WelfordWorkerState::new(&[tally.num_bins()]);

        let p = Particle::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0e6);
        let urr = UrrMacroXs {
            total: 4.0,
            elastic: 0.0,
            fission: 0.0,
            capture: 0.0,
            absorption: 0.0,
        };
        score_collision_event(
            &p,
            &cell,
            &mat,
            None,
            Some(&urr),
            None,
            &tallies,
            &mut welford,
        );
        assert_eq!(bin0(&welford), 0.0);
    }

    #[test]
    fn score_collision_zero_xs_no_contribution() {
        // Sigma_t <= 0 (no URR, empty material -> MT-1 lookup is 0) must
        // score nothing, never dividing by zero.
        let cell = make_cell(1);
        let mat = empty_material();
        let tally = flux_tally(Estimator::Collision);
        let tallies = [&tally];
        let mut welford = WelfordWorkerState::new(&[tally.num_bins()]);

        let p = Particle::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0e6);
        score_collision_event(&p, &cell, &mat, None, None, None, &tallies, &mut welford);
        assert_eq!(bin0(&welford), 0.0);
    }

    // --- score_woodcock_flight_estimator --------------------------------
    //
    // The delta-tracking flux estimator scores weight / Sigma_maj into each
    // TrackLength tally at every delta-collision (the collision-density
    // estimator with total_xs = Sigma_maj). Collision-estimator tallies are
    // ignored (they score at real collisions).

    #[test]
    fn woodcock_flux_contribution() {
        let cell = make_cell(1);
        let tally = flux_tally(Estimator::TrackLength);
        let tallies = [&tally];
        let mut welford = WelfordWorkerState::new(&[tally.num_bins()]);

        let mut p = Particle::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0e6);
        p.weight = 3.0;
        let sigma_maj = 6.0; // flux = weight / Sigma_maj = 3 / 6 = 0.5

        score_woodcock_flight_estimator(
            &p,
            &cell,
            None, // flux scoring needs no material data
            None,
            sigma_maj,
            None,
            None,
            &tallies,
            &[false],
            None,
            &mut welford,
        );

        assert!(
            (bin0(&welford) - 0.5).abs() < 1e-12,
            "got {}",
            bin0(&welford)
        );
    }

    #[test]
    fn woodcock_skips_collision_estimator_tally() {
        // A Collision-estimator tally must not be scored by the Woodcock
        // flight estimator.
        let cell = make_cell(1);
        let tally = flux_tally(Estimator::Collision);
        let tallies = [&tally];
        let mut welford = WelfordWorkerState::new(&[tally.num_bins()]);

        let p = Particle::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0e6);
        score_woodcock_flight_estimator(
            &p,
            &cell,
            None,
            None,
            6.0,
            None,
            None,
            &tallies,
            &[false],
            None,
            &mut welford,
        );
        assert_eq!(bin0(&welford), 0.0);
    }

    // --- true track-length mesh scoring under Woodcock (issue #350) -----

    /// Single-score flux tally on a 2x1x1 mesh over [0,10]^3 (bin
    /// boundary at x=5) with the requested estimator and extra filters.
    fn mesh_flux_tally(
        estimator: Estimator,
        extra_filters: Vec<yamc_tallies::filter::Filter>,
    ) -> Tally {
        use yamc_tallies::filter::mesh::MeshFilter;
        use yamc_tallies::mesh::RegularRectangularMesh;
        let mesh = RegularRectangularMesh::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [2, 1, 1]);
        let mut t = Tally::new();
        t.scores = vec![Score::Flux(FluxScore)];
        t.estimator = estimator;
        t.filters = vec![yamc_tallies::filter::Filter::Mesh(MeshFilter::new(mesh))];
        t.filters.extend(extra_filters);
        t.update_cache();
        t
    }

    #[test]
    fn mesh_track_length_eligibility() {
        use yamc_tallies::filter::Filter;
        use yamc_tallies::tally::{Mt, ReactionRateScore};
        use yamc_tallies::CellFilter;

        // The one shape that qualifies: TrackLength + mesh + flux-only.
        assert!(woodcock_mesh_track_length_eligible(&mesh_flux_tally(
            Estimator::TrackLength,
            vec![]
        )));
        // Collision estimator: scores at real collisions, untouched.
        assert!(!woodcock_mesh_track_length_eligible(&mesh_flux_tally(
            Estimator::Collision,
            vec![]
        )));
        // No mesh filter: cell tallies keep the flight estimator.
        assert!(!woodcock_mesh_track_length_eligible(&flux_tally(
            Estimator::TrackLength
        )));
        // Cell filter: the segment spans unidentified cells.
        assert!(!woodcock_mesh_track_length_eligible(&mesh_flux_tally(
            Estimator::TrackLength,
            vec![Filter::Cell(CellFilter::from_id(1))]
        )));
        // XS-weighted score: needs the local material, ineligible.
        let mut rxrate = mesh_flux_tally(Estimator::TrackLength, vec![]);
        rxrate.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(
            105,
        )))];
        rxrate.update_cache();
        assert!(!woodcock_mesh_track_length_eligible(&rxrate));
    }

    #[test]
    fn mesh_segment_scoring_apportions_track_length() {
        // Flight from x=1 to x=7 (length 6) across the bin boundary at
        // x=5: bin 0 gets 4/6 of the track length, bin 1 gets 2/6,
        // each times weight.
        let tally = mesh_flux_tally(Estimator::TrackLength, vec![]);
        let tallies = [&tally];
        let eligible = [true];
        let mut welford = WelfordWorkerState::new(&[tally.num_bins()]);

        let mut p = Particle::new([1.0, 5.0, 5.0], [1.0, 0.0, 0.0], 1.0e6);
        p.weight = 2.0;
        p.move_by(6.0); // last_position = (1,5,5), position = (7,5,5)

        score_woodcock_mesh_track_length(&p, p.position, 6.0, &tallies, &eligible, &mut welford);

        let bin = |idx: u32| -> f64 {
            welford.tallies[0]
                .scratch_map
                .get(&idx)
                .copied()
                .unwrap_or(0.0)
        };
        assert!((bin(0) - 8.0).abs() < 1e-12, "bin 0 got {}", bin(0));
        assert!((bin(1) - 4.0).abs() < 1e-12, "bin 1 got {}", bin(1));
    }

    #[test]
    fn mesh_segment_scoring_skips_ineligible() {
        // With the eligibility flag false, the segment scorer must not
        // touch the tally (it is scored by the flight estimator instead).
        let tally = mesh_flux_tally(Estimator::TrackLength, vec![]);
        let tallies = [&tally];
        let mut welford = WelfordWorkerState::new(&[tally.num_bins()]);

        let mut p = Particle::new([1.0, 5.0, 5.0], [1.0, 0.0, 0.0], 1.0e6);
        p.move_by(6.0);
        score_woodcock_mesh_track_length(&p, p.position, 6.0, &tallies, &[false], &mut welford);
        assert_eq!(bin0(&welford), 0.0);
    }

    // --- woodcock_flight_exit (issue #360) -------------------------------

    use crate::geo::{HalfspaceType, Region, RegionExpr};
    use crate::geometry::Geometry;

    /// Vacuum sphere of the given radius at the origin (void cell).
    fn vacuum_sphere_geometry(radius: f64) -> GeometryKind {
        let sphere = Arc::new(crate::geo::Surface::new_sphere(
            0.0,
            0.0,
            0.0,
            radius,
            Some(1),
            None,
        ));
        let mut sphere = (*sphere).clone();
        sphere.boundary = crate::geo::BoundaryType::Vacuum;
        let sphere = Arc::new(sphere);
        let region = Region::new_from_halfspace(HalfspaceType::Below(sphere));
        let cell = crate::geometry::cell::Cell::new(Some(1), region, None, None);
        GeometryKind::Csg(Geometry::new(vec![cell], Vec::new()).unwrap())
    }

    /// Two vacuum-bounded boxes: A spans x[-2,2] y[-6,6] z[-1,1]; B spans
    /// x[4,8] y[-1,1] z[-1,1]. B's infinite y=+-1 planes slice through A.
    fn disjoint_boxes_geometry() -> GeometryKind {
        fn plane(a: f64, b: f64, c: f64, d: f64, id: usize) -> Arc<crate::geo::Surface> {
            Arc::new(crate::geo::Surface {
                surface_id: Some(id),
                kind: crate::geo::SurfaceKind::Plane { a, b, c, d },
                boundary: crate::geo::BoundaryType::Vacuum,
                name: None,
            })
        }
        fn boxed(
            id_base: usize,
            x: (f64, f64),
            y: (f64, f64),
            z: (f64, f64),
        ) -> (crate::geometry::cell::Cell, Vec<Arc<crate::geo::Surface>>) {
            let xs = (
                plane(1.0, 0.0, 0.0, x.0, id_base),
                plane(1.0, 0.0, 0.0, x.1, id_base + 1),
            );
            let ys = (
                plane(0.0, 1.0, 0.0, y.0, id_base + 2),
                plane(0.0, 1.0, 0.0, y.1, id_base + 3),
            );
            let zs = (
                plane(0.0, 0.0, 1.0, z.0, id_base + 4),
                plane(0.0, 0.0, 1.0, z.1, id_base + 5),
            );
            let expr = RegionExpr::Intersection(
                Box::new(RegionExpr::Intersection(
                    Box::new(RegionExpr::Intersection(
                        Box::new(RegionExpr::Halfspace(HalfspaceType::Above(xs.0.clone()))),
                        Box::new(RegionExpr::Halfspace(HalfspaceType::Below(xs.1.clone()))),
                    )),
                    Box::new(RegionExpr::Intersection(
                        Box::new(RegionExpr::Halfspace(HalfspaceType::Above(ys.0.clone()))),
                        Box::new(RegionExpr::Halfspace(HalfspaceType::Below(ys.1.clone()))),
                    )),
                )),
                Box::new(RegionExpr::Intersection(
                    Box::new(RegionExpr::Halfspace(HalfspaceType::Above(zs.0.clone()))),
                    Box::new(RegionExpr::Halfspace(HalfspaceType::Below(zs.1.clone()))),
                )),
            );
            let cell =
                crate::geometry::cell::Cell::new(Some(id_base as u32), Region { expr }, None, None);
            (cell, vec![xs.0, xs.1, ys.0, ys.1, zs.0, zs.1])
        }
        let (cell_a, _) = boxed(10, (-2.0, 2.0), (-6.0, 6.0), (-1.0, 1.0));
        let (cell_b, _) = boxed(20, (4.0, 8.0), (-1.0, 1.0), (-1.0, 1.0));
        GeometryKind::Csg(Geometry::new(vec![cell_a, cell_b], Vec::new()).unwrap())
    }

    fn run_exit(
        geometry: &GeometryKind,
        start: [f64; 3],
        direction: [f64; 3],
        d_flight: f64,
    ) -> Option<f64> {
        let mut nl = NeighborLists::new(geometry.num_cells());
        let cell_idx = geometry
            .cells()
            .iter()
            .position(|c| c.region.contains((start[0], start[1], start[2])))
            .expect("start must be inside a cell");
        woodcock_flight_exit(
            geometry,
            &mut nl,
            geometry.vacuum_surfaces(),
            cell_idx,
            start,
            direction,
            d_flight,
        )
    }

    #[test]
    fn flight_exit_found_at_boundary() {
        let g = vacuum_sphere_geometry(3.0);
        let exit = run_exit(&g, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 10.0)
            .expect("flight past the boundary must report the exit");
        assert!((exit - 3.0).abs() < 1e-6, "exit at {exit}, expected 3.0");
    }

    #[test]
    fn flight_shorter_than_boundary_is_none() {
        let g = vacuum_sphere_geometry(3.0);
        assert_eq!(run_exit(&g, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 2.0), None);
    }

    #[test]
    fn in_geometry_crossing_is_skipped() {
        // +y flight from (0,-3,0) inside box A crosses box B's infinite
        // y=-1 and y=+1 plane extensions while still inside A; the true
        // exit is A's y=6 face.
        let g = disjoint_boxes_geometry();
        let exit =
            run_exit(&g, [0.0, -3.0, 0.0], [0.0, 1.0, 0.0], 100.0).expect("must exit at y=6");
        assert!((exit - 9.0).abs() < 1e-6, "exit at {exit}, expected 9.0");
    }

    #[test]
    fn first_true_exit_wins_over_later_body() {
        // +x flight from inside A: exits A at x=2 even though body B
        // (and its far side) lie further along the ray.
        let g = disjoint_boxes_geometry();
        let exit = run_exit(&g, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 100.0).expect("must exit at x=2");
        assert!((exit - 2.0).abs() < 1e-6, "exit at {exit}, expected 2.0");
    }

    #[test]
    fn start_near_boundary_terminates() {
        // Born within 1e-8 of the boundary, flying outward: the helper
        // must return promptly (no loop) with a tiny or skipped exit.
        let g = vacuum_sphere_geometry(3.0);
        let exit = run_exit(&g, [3.0 - 5e-9, 0.0, 0.0], [1.0, 0.0, 0.0], 10.0);
        if let Some(d) = exit {
            assert!(d < 1e-6, "exit distance {d} should be tiny");
        }
    }
}
