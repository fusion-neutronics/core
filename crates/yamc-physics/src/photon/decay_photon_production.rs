use rand::RngExt;
/// D1S decay photon production from neutron collisions.
///
/// 1. At setup time, precompute D1S photon production XS for each nuclide
///    by replacing prompt photon yields with decay photon yields from the chain.
/// 2. At collision time, use precomputed data to sample one decay photon
///    with weight = neutron_weight * y_t where y_t = photon_prod / total.
use std::collections::HashMap;
use std::sync::Arc;

use crate::util::bank::ParticleBank;
use yamc_nuclide::buffer::F64Buffer;
use yamc_nuclide::nuclide::{FastXSGrid, Nuclide};
use yamc_particle::particle::{Particle, ParticleType};
use yani::chain::{ChainNuclide, DecaySourceDistribution};
use yani::reactions::reaction_type_to_mt;

// =============================================================================
// Data structures for precomputed D1S photon data
// =============================================================================

/// Pre-computed D1S photon production data for a single nuclide at one temperature.
pub struct DecayPhotonNuclideData {
    /// D1S photon production XS at each energy grid point.
    /// decay_photon_prod[i] = Σ { reaction_xs[i] × branching × photon_per_decay }
    pub photon_prod: Vec<f64>,
    /// D1S photon channels for sampling which reaction/product produced the photon.
    pub channels: Vec<DecayPhotonChannel>,
}

/// A single D1S photon production channel (one chain reaction → one decay source).
pub struct DecayPhotonChannel {
    /// Reaction XS at each energy grid point (same grid as nuclide energy grid).
    /// A view of the grid's own buffer when the MT column can be handed over
    /// whole (MT 102), a fresh one when it has to be de-interleaved.
    pub xs: F64Buffer,
    /// Target nuclide name for parent_nuclide tagging (e.g., "Mn56").
    pub target_name: String,
    /// Interned id for `target_name`; stamped on each D1S photon's `parent_nuclide`.
    pub target_id: yamc_nuclide::nuclide_registry::NuclideId,
    /// Discrete photon energies from decay spectrum [eV].
    pub energies: Vec<f64>,
    /// Discrete photon intensities (emission rates from chain).
    pub intensities: Vec<f64>,
    /// Constant yield = branching × photon_per_decay.
    pub yield_constant: f64,
}

impl DecayPhotonNuclideData {
    /// Interpolate the D1S photon production XS at a given grid index.
    #[inline]
    pub fn lookup_photon_prod(&self, i_grid: usize, interp_factor: f64) -> f64 {
        interp_value(&self.photon_prod, i_grid, interp_factor)
    }
}

// =============================================================================
// Precomputation of D1S photon production data
// =============================================================================

/// Precompute D1S photon production data for all nuclides.
///
/// Called once at setup time:
/// 1. Replace prompt photon products with decay products on reactions
/// 2. Recompute photon_prod XS using decay yields
///
/// Returns a `Vec<Vec<DecayPhotonNuclideData>>` indexed by `(NuclideId.get() - 1) as usize`,
/// where each inner Vec is indexed by `temp_idx` (matching `nuclide.fast_xs`).
/// Source nuclide names are interned into the registry so the hot path can
/// look up D1S data via a single Vec index instead of a string-keyed HashMap.
/// Nuclides without any D1S channels get an empty inner Vec.
pub fn precompute_decay_photon_data(
    chain: &HashMap<String, ChainNuclide>,
    nuclides: &HashMap<String, Arc<Nuclide>>,
    registry: &mut yamc_nuclide::nuclide_registry::NuclideRegistry,
) -> Vec<Vec<DecayPhotonNuclideData>> {
    let mut per_nuclide: HashMap<
        yamc_nuclide::nuclide_registry::NuclideId,
        Vec<DecayPhotonNuclideData>,
    > = HashMap::new();

    for (nuc_name, nuclide) in nuclides {
        let chain_nuclide = match chain.get(nuc_name) {
            Some(cn) => cn,
            None => continue,
        };

        let mut temp_data = Vec::with_capacity(nuclide.fast_xs.len());

        for fast_grid in &nuclide.fast_xs {
            let n_energy = fast_grid.energy.len();
            let mut channels = Vec::new();
            let mut photon_prod = vec![0.0f64; n_energy];

            for reaction in &chain_nuclide.reactions {
                let target_name = match &reaction.target {
                    Some(t) => t.as_str(),
                    None => continue,
                };

                let mt = match reaction_type_to_mt(&reaction.kind) {
                    Some(mt) => mt,
                    None => continue,
                };

                // Get reaction XS vector on the full energy grid
                let xs_vec = match get_reaction_xs_vector(fast_grid, mt) {
                    Some(xs) => xs,
                    None => continue,
                };

                // Walk the decay chain from the produced target and emit a
                // channel for every photon-emitting nuclide along it -- the
                // target itself, or a daughter when the target is a pure beta
                // emitter (e.g. Cs137 -> Ba137m). The path branching is the
                // fraction of decays that reach each emitter; the time-dependent
                // buildup of the daughter is handled later by the Bateman TCF.
                for path in yani_decay::descendant_paths(chain, target_name) {
                    let emitter = match chain.get(&path.emitter) {
                        Some(e) => e,
                        None => continue,
                    };
                    let emitter_lambda = match emitter.half_life {
                        Some(t_half) if t_half > 0.0 => std::f64::consts::LN_2 / t_half,
                        _ => continue,
                    };
                    let path_branching = path.path_branching();

                    for source in &emitter.sources {
                        if source.particle != "photon" {
                            continue;
                        }
                        let DecaySourceDistribution::Discrete {
                            energies,
                            intensities,
                        } = &source.distribution;

                        // photon_per_decay = sum(emission_rates) / λ
                        // (chain stores emission rates = λ × yield_per_decay)
                        let photon_per_decay: f64 =
                            intensities.iter().sum::<f64>() / emitter_lambda;
                        let yield_constant = reaction.branching * path_branching * photon_per_decay;

                        if yield_constant <= 0.0 {
                            continue;
                        }

                        // Accumulate photon_prod: photon_prod[i] += xs[i] * yield
                        for (i, &xs) in xs_vec.iter().enumerate() {
                            photon_prod[i] += xs * yield_constant;
                        }

                        let target_id = registry.intern(&path.emitter);
                        channels.push(DecayPhotonChannel {
                            xs: xs_vec.clone(),
                            target_name: path.emitter.clone(),
                            target_id,
                            energies: energies.clone(),
                            intensities: intensities.clone(),
                            yield_constant,
                        });
                    }
                }
            }

            temp_data.push(DecayPhotonNuclideData {
                photon_prod,
                channels,
            });
        }

        if temp_data.iter().any(|d| !d.channels.is_empty()) {
            let id = registry.intern(nuc_name);
            per_nuclide.insert(id, temp_data);
        }
    }

    // Flatten HashMap<NuclideId, _> into Vec indexed by (id.get() - 1).
    // Size to cover every interned id; nuclides without D1S channels get an
    // empty inner Vec so `vec[id].is_empty()` means "no D1S data".
    let len = registry.len();
    let mut flat: Vec<Vec<DecayPhotonNuclideData>> = (0..len).map(|_| Vec::new()).collect();
    for (id, data) in per_nuclide {
        let slot = id.get() as usize - 1;
        if slot < flat.len() {
            flat[slot] = data;
        }
    }
    flat
}

// =============================================================================
// Runtime D1S photon sampling
// =============================================================================

/// Sample a D1S decay photon from a neutron collision using precomputed data.
///
/// - Emit exactly one photon per collision with weight = neutron_weight * y_t
/// - y_t = photon_prod / total (using precomputed D1S photon_prod)
/// - Sample which channel produced the photon (proportional to xs * yield)
/// - Sample energy from decay spectrum, isotropic direction
/// - Tag with parent nuclide name
#[allow(clippy::too_many_arguments)]
pub fn sample_decay_photons<R: rand::Rng>(
    particle: &Particle,
    decay_photon_data: &DecayPhotonNuclideData,
    i_grid: usize,
    interp_factor: f64,
    micro_xs_total: f64,
    bank: &mut ParticleBank,
    rng: &mut R,
) {
    if micro_xs_total <= 0.0 || decay_photon_data.channels.is_empty() {
        return;
    }

    // Interpolate precomputed D1S photon_prod at collision energy
    let photon_prod = decay_photon_data.lookup_photon_prod(i_grid, interp_factor);
    if photon_prod <= 0.0 {
        return;
    }

    // y_t = photon_prod / total
    let y_t = photon_prod / micro_xs_total;

    // Emit one photon with weight *= y_t
    let photon_wgt = particle.weight * y_t;

    // Sample which channel (probability proportional to reaction_xs * yield_constant)
    let cutoff = rng.random::<f64>() * photon_prod;
    let mut prob = 0.0;
    let mut selected_idx = decay_photon_data.channels.len() - 1;

    for (idx, ch) in decay_photon_data.channels.iter().enumerate() {
        let xs = interp_value(&ch.xs, i_grid, interp_factor);
        if xs > 0.0 {
            prob += xs * ch.yield_constant;
            selected_idx = idx;
            if prob > cutoff {
                break;
            }
        }
    }

    let selected = &decay_photon_data.channels[selected_idx];

    // Sample energy from discrete decay spectrum
    let energy = sample_discrete_energy(&selected.energies, &selected.intensities, rng);
    if energy <= 0.0 {
        return;
    }

    // Sample isotropic direction (decay photons are isotropic)
    let direction = sample_isotropic_direction(rng);

    let mut photon = Particle::new(particle.position, direction, energy);
    photon.particle_type = ParticleType::Photon;
    photon.weight = photon_wgt;
    photon.current_cell_index = particle.current_cell_index;
    photon.parent_nuclide = Some(selected.target_id);

    bank.bank_secondary(photon);

    #[cfg(feature = "debug_diagnostics")]
    crate::photon_diag::DECAY_PHOTONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

// =============================================================================
// Utility functions
// =============================================================================

/// Interpolate a value from a pre-computed XS vector at a grid index.
#[inline]
fn interp_value(xs_vec: &[f64], i_grid: usize, interp_factor: f64) -> f64 {
    if i_grid + 1 < xs_vec.len() {
        xs_vec[i_grid] + interp_factor * (xs_vec[i_grid + 1] - xs_vec[i_grid])
    } else if !xs_vec.is_empty() {
        xs_vec[i_grid.min(xs_vec.len() - 1)]
    } else {
        0.0
    }
}

/// Get the full XS vector for a given MT from the fast grid.
/// Searches all available per-reaction XS stores and materializes the
/// per-MT column from the row-major flat buffers. Called once per chain
/// reaction at D1S precompute time -- setup cost, not hot path.
fn get_reaction_xs_vector(fast_grid: &FastXSGrid, mt: i32) -> Option<F64Buffer> {
    // For (n,gamma) MT 102, prefer xs_ngamma (canonical source) over photon_rxn_xs.
    // The clone is a refcount bump, not a copy of the grid.
    if mt == 102 && !fast_grid.xs_ngamma.is_empty() {
        return Some(fast_grid.xs_ngamma.clone());
    }
    // Helper: extract column `col` from row-major [n_energies, n_mts] flat buffer
    let extract_col = |flat: &[f64], n_mts: usize, col: usize| -> F64Buffer {
        if n_mts == 0 || flat.is_empty() {
            return F64Buffer::default();
        }
        let n_e = flat.len() / n_mts;
        (0..n_e).map(|i| flat[i * n_mts + col]).collect()
    };
    // Search photon_rxn_xs
    if let Some(i) = fast_grid
        .photon_rxn_mt_numbers
        .iter()
        .position(|&m| m == mt)
    {
        return Some(extract_col(
            &fast_grid.photon_rxn_xs,
            fast_grid.photon_rxn_mt_numbers.len(),
            i,
        ));
    }
    // Search scatter_mt_xs
    if let Some(i) = fast_grid.scatter_mt_numbers.iter().position(|&m| m == mt) {
        return Some(extract_col(
            &fast_grid.scatter_mt_xs,
            fast_grid.scatter_mt_numbers.len(),
            i,
        ));
    }
    // Search fission_mt_xs
    if let Some(i) = fast_grid.fission_mt_numbers.iter().position(|&m| m == mt) {
        return Some(extract_col(
            &fast_grid.fission_mt_xs,
            fast_grid.fission_mt_numbers.len(),
            i,
        ));
    }
    // Search absorption_mt_xs
    if let Some(i) = fast_grid
        .absorption_mt_numbers
        .iter()
        .position(|&m| m == mt)
    {
        return Some(extract_col(
            &fast_grid.absorption_mt_xs,
            fast_grid.absorption_mt_numbers.len(),
            i,
        ));
    }
    None
}

/// Sample an energy from a discrete probability mass function.
/// energies[i] with probability proportional to intensities[i].
fn sample_discrete_energy<R: rand::Rng>(energies: &[f64], intensities: &[f64], rng: &mut R) -> f64 {
    let total: f64 = intensities.iter().sum();
    if total <= 0.0 || energies.is_empty() {
        return 0.0;
    }

    let xi = rng.random::<f64>() * total;
    let mut cumulative = 0.0;
    for (i, &intensity) in intensities.iter().enumerate() {
        cumulative += intensity;
        if xi <= cumulative {
            return energies[i];
        }
    }

    // Fallback: return last energy (floating-point edge case)
    *energies.last().unwrap_or(&0.0)
}

/// Sample an isotropic direction (uniform on unit sphere).
fn sample_isotropic_direction<R: rand::Rng>(rng: &mut R) -> [f64; 3] {
    let mu = 2.0 * rng.random::<f64>() - 1.0; // cosine of polar angle [-1, 1]
    let phi = std::f64::consts::TAU * rng.random::<f64>(); // azimuthal angle [0, 2pi)
    let sin_theta = (1.0 - mu * mu).sqrt();
    [sin_theta * phi.cos(), sin_theta * phi.sin(), mu]
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Test arrow datasets live in `crates/yamc/tests/`. Resolve relative
    /// to this crate's manifest dir so `cargo test` works regardless of
    /// which crate's runner is invoked.
    fn td(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("yamc")
            .join("tests")
            .join(name)
    }

    #[test]
    fn test_chain_reaction_to_mt_common() {
        assert_eq!(reaction_type_to_mt("(n,gamma)"), Some(102));
        assert_eq!(reaction_type_to_mt("fission"), Some(18));
        assert_eq!(reaction_type_to_mt("(n,2n)"), Some(16));
        assert_eq!(reaction_type_to_mt("(n,3n)"), Some(17));
        assert_eq!(reaction_type_to_mt("(n,p)"), Some(103));
        assert_eq!(reaction_type_to_mt("(n,d)"), Some(104));
        assert_eq!(reaction_type_to_mt("(n,t)"), Some(105));
        assert_eq!(reaction_type_to_mt("(n,3He)"), Some(106));
        assert_eq!(reaction_type_to_mt("(n,a)"), Some(107));
        assert_eq!(reaction_type_to_mt("(n,4n)"), Some(37));
    }

    /// The D1S path used to carry its own copy of the reaction name table, in
    /// which `(n,2nd)` was mapped to MT 35. MT 35 is `(n,nd2a)`, so every
    /// `(n,2nd)` chain reaction scored decay photon production against another
    /// channel's cross section. There is one table now; this pins the value the
    /// D1S path actually resolves.
    #[test]
    fn d1s_resolves_two_neutron_deuteron_to_mt_11() {
        assert_eq!(reaction_type_to_mt("(n,2nd)"), Some(11));
        assert_eq!(reaction_type_to_mt("(n,nd2a)"), Some(35));
    }

    #[test]
    fn test_chain_reaction_to_mt_unknown() {
        assert_eq!(reaction_type_to_mt("beta-"), None);
        assert_eq!(reaction_type_to_mt("alpha"), None);
        assert_eq!(reaction_type_to_mt("ec/beta+"), None);
        assert_eq!(reaction_type_to_mt("unknown"), None);
    }

    #[test]
    fn test_sample_discrete_energy_single_line() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let energies = vec![846764.0];
        let intensities = vec![1.0];
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let e = sample_discrete_energy(&energies, &intensities, &mut rng);
            assert!((e - 846764.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sample_discrete_energy_multiple_lines() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let energies = vec![100.0, 200.0, 300.0];
        let intensities = vec![0.5, 0.3, 0.2];
        let mut rng = StdRng::seed_from_u64(42);

        let mut counts = [0usize; 3];
        let n = 10000;
        for _ in 0..n {
            let e = sample_discrete_energy(&energies, &intensities, &mut rng);
            if (e - 100.0).abs() < 1e-6 {
                counts[0] += 1;
            } else if (e - 200.0).abs() < 1e-6 {
                counts[1] += 1;
            } else if (e - 300.0).abs() < 1e-6 {
                counts[2] += 1;
            }
        }

        // Check approximate fractions (within 5% tolerance)
        let frac0 = counts[0] as f64 / n as f64;
        let frac1 = counts[1] as f64 / n as f64;
        let frac2 = counts[2] as f64 / n as f64;
        assert!((frac0 - 0.5).abs() < 0.05, "Expected ~0.5, got {frac0}");
        assert!((frac1 - 0.3).abs() < 0.05, "Expected ~0.3, got {frac1}");
        assert!((frac2 - 0.2).abs() < 0.05, "Expected ~0.2, got {frac2}");
    }

    #[test]
    fn test_sample_isotropic_direction_is_unit_vector() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..1000 {
            let d = sample_isotropic_direction(&mut rng);
            let mag = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            assert!(
                (mag - 1.0).abs() < 1e-10,
                "Direction should be unit vector, got magnitude {mag}"
            );
        }
    }

    #[test]
    fn test_sample_isotropic_direction_covers_sphere() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let mut rng = StdRng::seed_from_u64(42);
        let n = 10000;
        let mut sum_z = 0.0;

        for _ in 0..n {
            let d = sample_isotropic_direction(&mut rng);
            sum_z += d[2];
        }

        // Mean of z-component should be ~0 for isotropic
        let mean_z = sum_z / n as f64;
        assert!(
            mean_z.abs() < 0.05,
            "Mean z should be ~0 for isotropic, got {mean_z}"
        );
    }

    #[test]
    fn test_precompute_and_sample_decay_photons() {
        // Test D1S decay photon production using real Fe56 nuclear data
        if !td("Fe56.arrow").exists() {
            eprintln!("Skipping: Fe56.arrow not found");
            return;
        }

        use rand::rngs::StdRng;
        use rand::SeedableRng;
        use yani::chain::{ChainReaction, DecaySource, DecaySourceDistribution};

        let nuclide = yamc_nuclide::nuclide_loader::load_nuclide(
            td("Fe56.arrow"),
            &yamc_nuclide::LoadScope::full(),
        )
        .expect("Failed to load Fe56.arrow");

        // Create a mock chain where Fe56 (n,gamma) -> Mn56 with photon source
        let mut chain: HashMap<String, ChainNuclide> = HashMap::new();

        chain.insert(
            "Fe56".to_string(),
            ChainNuclide {
                name: "Fe56".to_string(),
                half_life: None,
                decay_energy: 0.0,
                reactions: vec![ChainReaction {
                    kind: "(n,gamma)".to_string(),
                    target: Some("Mn56".to_string()),
                    branching: 1.0,
                    q_value: None,
                }],
                decays: vec![],
                fission_yields: None,
                sources: vec![],
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
            },
        );

        // Mn56: t_half = 9284.4 s, λ = ln(2)/9284.4 = 7.4657e-5
        chain.insert(
            "Mn56".to_string(),
            ChainNuclide {
                name: "Mn56".to_string(),
                half_life: Some(9284.4),
                decay_energy: 0.0,
                reactions: vec![],
                decays: vec![],
                fission_yields: None,
                sources: vec![DecaySource {
                    particle: "photon".to_string(),
                    distribution: DecaySourceDistribution::Discrete {
                        energies: vec![846764.0, 1810726.0],
                        intensities: vec![7.381e-5, 2.030e-5],
                    },
                }],
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
            },
        );

        // Precompute D1S data
        let mut nuclides: HashMap<String, Arc<Nuclide>> = HashMap::new();
        nuclides.insert("Fe56".to_string(), Arc::new(nuclide));
        let mut registry = yamc_nuclide::nuclide_registry::NuclideRegistry::new();
        let decay_photon_data = precompute_decay_photon_data(&chain, &nuclides, &mut registry);

        let nuclide = nuclides.get("Fe56").unwrap();
        let fe56_id = registry
            .lookup("Fe56")
            .expect("Fe56 should have been interned during D1S build");
        let decay_photon_nuc = &decay_photon_data[fe56_id.get() as usize - 1];
        assert!(!decay_photon_nuc.is_empty(), "Fe56 should have D1S data");
        let decay_photon_temp = &decay_photon_nuc[0]; // First temperature

        // Verify precomputed photon_prod is non-zero
        assert!(
            decay_photon_temp.photon_prod.iter().any(|&x| x > 0.0),
            "D1S photon_prod should have non-zero values"
        );

        // Verify channels
        assert_eq!(
            decay_photon_temp.channels.len(),
            1,
            "Should have 1 channel: (n,gamma)->Mn56"
        );
        assert_eq!(decay_photon_temp.channels[0].target_name, "Mn56");

        let fast_grid = &nuclide.fast_xs[0];
        let (i_grid, f) = fast_grid.lookup_grid_index(1.0e6); // 1 MeV
        let (total_xs, _, _, _) = fast_grid.lookup(1.0e6);

        let neutron = Particle::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0e6);
        let mut bank = ParticleBank::new();
        let mut rng = StdRng::seed_from_u64(42);

        // Run many trials to collect photons
        let mut total_photons = 0;
        let mut all_energies = Vec::new();
        let mut all_parents = Vec::new();
        let mut all_weights = Vec::new();

        for _ in 0..5000 {
            bank.clear();
            sample_decay_photons(
                &neutron,
                decay_photon_temp,
                i_grid,
                f,
                total_xs,
                &mut bank,
                &mut rng,
            );
            while let Some(p) = bank.pop_particle() {
                total_photons += 1;
                all_energies.push(p.energy);
                all_parents.push(p.parent_nuclide);
                all_weights.push(p.weight);
                assert_eq!(p.particle_type, ParticleType::Photon);
                assert_eq!(p.position, neutron.position);
            }
        }

        // Every call produces exactly 1 decay photon
        assert_eq!(
            total_photons, 5000,
            "Should produce exactly 1 decay photon per call"
        );

        // All photon energies should be from the discrete spectrum
        for e in &all_energies {
            assert!(
                (e - 846764.0).abs() < 1.0 || (e - 1810726.0).abs() < 1.0,
                "Photon energy {e} not from discrete spectrum"
            );
        }

        // All photons should have correct parent nuclide
        let mn56_id = registry
            .lookup("Mn56")
            .expect("Mn56 should have been interned during D1S build");
        for parent in &all_parents {
            assert_eq!(*parent, Some(mn56_id), "Parent nuclide should be Mn56");
        }

        // Photon weight should be fractional (y_t * neutron_weight)
        for &w in &all_weights {
            assert!(
                w < neutron.weight,
                "Decay photon weight {w} should be less than neutron weight {}",
                neutron.weight
            );
            assert!(w > 0.0, "Decay photon weight should be positive");
        }

        // Check that the 846 keV line is more frequent than the 1.8 MeV line
        let count_846 = all_energies
            .iter()
            .filter(|e| (**e - 846764.0).abs() < 1.0)
            .count();
        let count_1810 = all_energies
            .iter()
            .filter(|e| (**e - 1810726.0).abs() < 1.0)
            .count();
        assert!(
            count_846 > count_1810,
            "846 keV line should be more frequent: {count_846} vs {count_1810}"
        );
    }

    #[test]
    fn test_decay_photon_chain_through_beta_only_parent() {
        // Fe56 (n,gamma) -> A (pure beta, NO photons) -> B (gamma emitter).
        // The directly-produced nuclide A emits nothing, so the single-
        // generation code skipped it entirely and produced zero photons. With
        // decay-chain expansion, B's photons must be emitted and tagged B.
        if !td("Fe56.arrow").exists() {
            eprintln!("Skipping: Fe56.arrow not found");
            return;
        }

        use rand::rngs::StdRng;
        use rand::SeedableRng;
        use yani::chain::{ChainReaction, DecaySource, DecaySourceDistribution};

        let nuclide = yamc_nuclide::nuclide_loader::load_nuclide(
            td("Fe56.arrow"),
            &yamc_nuclide::LoadScope::full(),
        )
        .expect("Failed to load Fe56.arrow");

        let mut chain: HashMap<String, ChainNuclide> = HashMap::new();
        chain.insert(
            "Fe56".to_string(),
            ChainNuclide {
                name: "Fe56".to_string(),
                half_life: None,
                decay_energy: 0.0,
                reactions: vec![ChainReaction {
                    kind: "(n,gamma)".to_string(),
                    target: Some("ParentA".to_string()),
                    branching: 1.0,
                    q_value: None,
                }],
                decays: vec![],
                fission_yields: None,
                sources: vec![],
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
            },
        );
        // ParentA: unstable, beta-decays to DaughterB, emits NO photons.
        chain.insert(
            "ParentA".to_string(),
            ChainNuclide {
                name: "ParentA".to_string(),
                half_life: Some(3600.0),
                decay_energy: 0.0,
                reactions: vec![],
                decays: vec![ChainReaction {
                    kind: "beta-".to_string(),
                    target: Some("DaughterB".to_string()),
                    branching: 1.0,
                    q_value: None,
                }],
                fission_yields: None,
                sources: vec![],
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
            },
        );
        // DaughterB: unstable, emits a gamma line.
        chain.insert(
            "DaughterB".to_string(),
            ChainNuclide {
                name: "DaughterB".to_string(),
                half_life: Some(600.0),
                decay_energy: 0.0,
                reactions: vec![],
                decays: vec![],
                fission_yields: None,
                sources: vec![DecaySource {
                    particle: "photon".to_string(),
                    distribution: DecaySourceDistribution::Discrete {
                        energies: vec![1000000.0],
                        intensities: vec![1.0e-3],
                    },
                }],
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
            },
        );

        let mut nuclides: HashMap<String, Arc<Nuclide>> = HashMap::new();
        nuclides.insert("Fe56".to_string(), Arc::new(nuclide));
        let mut registry = yamc_nuclide::nuclide_registry::NuclideRegistry::new();
        let decay_photon_data = precompute_decay_photon_data(&chain, &nuclides, &mut registry);

        let fe56_id = registry
            .lookup("Fe56")
            .expect("Fe56 should be interned: it produces a chain emitter");
        let temp = &decay_photon_data[fe56_id.get() as usize - 1][0];

        // Exactly one channel, tagged with the DAUGHTER (the emitter), not the
        // directly-produced beta-only parent.
        assert_eq!(temp.channels.len(), 1);
        assert_eq!(temp.channels[0].target_name, "DaughterB");
        assert!(temp.photon_prod.iter().any(|&x| x > 0.0));

        // Sample and confirm photons carry DaughterB's energy and parent tag.
        let nuclide = nuclides.get("Fe56").unwrap();
        let fast_grid = &nuclide.fast_xs[0];
        let (i_grid, f) = fast_grid.lookup_grid_index(1.0e6);
        let (total_xs, _, _, _) = fast_grid.lookup(1.0e6);
        let neutron = Particle::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0e6);
        let daughter_id = registry.lookup("DaughterB").unwrap();
        let mut bank = ParticleBank::new();
        let mut rng = StdRng::seed_from_u64(7);
        let mut n = 0;
        for _ in 0..1000 {
            bank.clear();
            sample_decay_photons(&neutron, temp, i_grid, f, total_xs, &mut bank, &mut rng);
            while let Some(p) = bank.pop_particle() {
                n += 1;
                assert_eq!(p.particle_type, ParticleType::Photon);
                assert!((p.energy - 1000000.0).abs() < 1.0);
                assert_eq!(p.parent_nuclide, Some(daughter_id));
            }
        }
        assert_eq!(n, 1000, "one daughter photon per collision");
    }

    #[test]
    fn test_decay_photon_nuclide_not_in_chain() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        if !td("Fe56.arrow").exists() {
            eprintln!("Skipping: Fe56.arrow not found");
            return;
        }

        let nuclide = yamc_nuclide::nuclide_loader::load_nuclide(
            td("Fe56.arrow"),
            &yamc_nuclide::LoadScope::full(),
        )
        .expect("Failed to load Fe56.arrow");

        let chain: HashMap<String, ChainNuclide> = HashMap::new(); // empty chain

        let mut nuclides: HashMap<String, Arc<Nuclide>> = HashMap::new();
        nuclides.insert("Fe56".to_string(), Arc::new(nuclide));
        let mut registry = yamc_nuclide::nuclide_registry::NuclideRegistry::new();
        let decay_photon_data = precompute_decay_photon_data(&chain, &nuclides, &mut registry);

        // Fe56 not in chain → no D1S data (and the registry stays empty of
        // source-nuclide entries, so the flat Vec is empty).
        assert!(
            decay_photon_data.iter().all(|v| v.is_empty()),
            "No D1S data when nuclide not in chain"
        );
        assert!(
            registry.lookup("Fe56").is_none(),
            "Fe56 should not be interned when not in chain"
        );

        // If we tried to sample with empty data, nothing should happen
        let neutron = Particle::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0e6);
        let mut bank = ParticleBank::new();
        let mut rng = StdRng::seed_from_u64(42);

        let empty_data = DecayPhotonNuclideData {
            photon_prod: vec![],
            channels: vec![],
        };
        sample_decay_photons(&neutron, &empty_data, 0, 0.0, 1.0, &mut bank, &mut rng);
        assert_eq!(bank.len(), 0, "No photons when no D1S data");
    }
}
