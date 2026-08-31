//! Photon cross-section / interaction-data preparation: macroscopic XS
//! plus the per-channel tables (Compton Doppler broadening, incoherent
//! form factor, pair production, atomic relaxation, bremsstrahlung/TTB).
//! Pure CPU data prep -- always built, no cubecl dependency.

pub mod atomic_relaxation_xs;
pub mod bremsstrahlung_xs;
pub mod compton_doppler_xs;
pub mod incoherent_form_factor_xs;
pub mod pair_production_xs;
pub mod photon_xs;

use std::sync::Arc;

use yamc_element::photon::{ElementMicroXS, PhotonInteraction};

/// The per-reaction channel a single-element-per-material table models.
///
/// The GPU photon kernel currently carries one form-factor / relaxation
/// table per material (one dominant element). The physically correct
/// dominant element for a given reaction is the one with the largest
/// *macroscopic* contribution to THAT reaction's cross section
/// (`atom_density * micro_xs`), not the one with the most atoms. These
/// reactions scale steeply with Z, so a trace high-Z element (e.g. W in
/// polyethylene) dominates the coherent / photoelectric channel even at
/// a few atom-percent, while H dominates the atom count.
///
/// This mirrors the CPU reference: `Material::sample_element` samples the
/// interacting element per collision proportional to its macroscopic XS
/// contribution, then runs that element's form-factor / relaxation
/// physics. The single-dominant-element packing is the bounded
/// approximation of that sampling.
#[derive(Clone, Copy, Debug)]
pub enum ReactionChannel {
    /// Coherent (Rayleigh) -- coherent form factor.
    Coherent,
    /// Incoherent (Compton) -- incoherent scattering function / Doppler.
    Incoherent,
    /// Photoelectric -- atomic-relaxation cascade + subshell PE XS.
    Photoelectric,
    /// Pair production -- Z-indexed screening constants.
    Pair,
}

impl ReactionChannel {
    /// Pull this channel's microscopic cross section out of a computed
    /// `ElementMicroXS`.
    #[inline]
    fn micro(self, m: &ElementMicroXS) -> f64 {
        match self {
            ReactionChannel::Coherent => m.coherent,
            ReactionChannel::Incoherent => m.incoherent,
            ReactionChannel::Photoelectric => m.photoelectric,
            ReactionChannel::Pair => m.pair_production,
        }
    }
}

/// Lower energy bound (eV) for the dominant-element comparison. Photon
/// transport never tracks photons below the default 1 keV cutoff, and the
/// photoelectric cross section spikes by orders of magnitude right at the
/// few-eV absorption edges of light elements (H ~13.6 eV, C ~11 eV). Those
/// near-threshold spikes are not transport-relevant and would otherwise
/// fool a peak/sum comparison into picking the light bulk element. Anchor
/// the comparison at and above this floor, where the steep Z-scaling makes
/// the high-Z element the unambiguous dominator.
const DOMINANCE_ENERGY_FLOOR_EV: f64 = 1.0e3;

/// Pick the element in a material that dominates a given reaction's
/// macroscopic cross section, returning its index into `mat_elements`.
///
/// The dominance score is the *peak* macroscopic contribution
/// `atom_density * micro_xs(reaction)` over the first element's log-energy
/// grid, restricted to energies at or above `DOMINANCE_ENERGY_FLOOR_EV`
/// (the transport cutoff). The peak over a transport-relevant energy band
/// (rather than a full-grid sum) is the physically meaningful metric: it
/// asks "which element has the strongest macroscopic presence in this
/// channel at energies photons actually reach", and is insensitive both to
/// grid-point density and to the unphysical few-eV photoelectric edge
/// spikes of light elements. Because the coherent / photoelectric / pair
/// channels scale steeply with Z, the peak reliably selects a trace high-Z
/// element over a bulk low-Z one (W beats C+H at every energy above 1 keV
/// in poly+W). For a single-element material the only element trivially
/// wins, so packing is unchanged from the previous "most atoms" pick.
///
/// Tie-break: when two elements have an (essentially) equal contribution,
/// the higher atomic number wins. This fixes the WC (50/50 C/W) case,
/// where the previous strict-`>` over an alphabetically ordered element
/// list let Carbon shadow Tungsten. Returns `None` when no element has a
/// positive contribution (e.g. a void material, or a channel with no
/// cross section over the grid).
///
/// Residual approximation: a material with two comparable high-Z elements
/// is still modelled with only one of them. Full per-collision element
/// selection (mirroring `Material::sample_element` on the GPU) is the
/// follow-up.
pub fn dominant_element_for_reaction(
    mat_elements: &[(String, Arc<PhotonInteraction>, f64)],
    reaction: ReactionChannel,
) -> Option<usize> {
    // Common reference energies: the first element's ln(E) grid. Every
    // element's `calculate_xs` accepts any energy, so this lets us compare
    // macroscopic contributions on the same energy set.
    let grid: &[f64] = mat_elements
        .iter()
        .find(|(_, el, _)| !el.energy.is_empty())
        .map(|(_, el, _)| el.energy.as_slice())
        .unwrap_or(&[]);

    let mut best_idx: Option<usize> = None;
    let mut best_score = 0.0_f64;
    let mut best_z: u32 = 0;

    for (idx, (_, el, density)) in mat_elements.iter().enumerate() {
        if *density <= 0.0 {
            continue;
        }
        // Peak transport-relevant macroscopic contribution of this
        // reaction channel. `max` over grid energies >= the floor (not a
        // sum, and not the unphysical few-eV edge region) so grid-point
        // density and the low-E photoelectric divergence cannot bias the
        // pick toward a light bulk element.
        let mut score = 0.0_f64;
        if grid.is_empty() {
            // No usable grid: fall back to a single mid-spectrum probe so
            // the steep Z-scaling still selects correctly.
            score = *density * reaction.micro(&el.calculate_xs(1.0e6));
        } else {
            let mut any_above_floor = false;
            for &log_e in grid {
                let energy = log_e.exp();
                if energy < DOMINANCE_ENERGY_FLOOR_EV {
                    continue;
                }
                any_above_floor = true;
                let contrib = *density * reaction.micro(&el.calculate_xs(energy));
                if contrib > score {
                    score = contrib;
                }
            }
            // Degenerate grid entirely below the floor: fall back to the
            // grid max so we still produce a deterministic pick.
            if !any_above_floor {
                if let Some(&log_e) = grid.last() {
                    score = *density * reaction.micro(&el.calculate_xs(log_e.exp()));
                }
            }
        }
        let z = el.atomic_number;
        // Strictly larger contribution wins; on a (near-)tie the higher Z
        // wins so alphabetical element ordering can never pick the lower-Z
        // element on an even split (WC fix).
        let is_better = match best_idx {
            None => score > 0.0,
            Some(_) => {
                let tie = (score - best_score).abs() <= best_score * 1e-12;
                if tie {
                    z > best_z
                } else {
                    score > best_score
                }
            }
        };
        if is_better {
            best_idx = Some(idx);
            best_score = score;
            best_z = z;
        }
    }

    best_idx
}
