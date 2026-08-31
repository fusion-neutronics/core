/// Independent transmutation operator for coupled transport-transmutation.
///
/// Manages the transmutation of all transmutable materials in a model.
/// In coupled mode, transport runs each timestep with updated compositions,
/// providing accurate tracking of composition changes.
use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;

use super::transmutation_stepper::TransmutationStepper;
use yamc_materials::material::Material;
use yani::{ChainNuclide, ChainParts, FissionYieldWeights, ReactionRates};

use super::reaction_type_to_mt;

/// Drives transmutation of all transmutable materials from pre-computed
/// reaction rates.
///
/// Used by both the coupled and independent transmutation paths; the run mode
/// is chosen by the caller (the `method` argument in `orchestrate`), not by
/// this type.
pub struct TransmutationDriver {
    /// Transmutation chain (Arc for cheap cloning)
    chain: Arc<HashMap<String, ChainNuclide>>,

    /// Which optional subsections `chain` was built from.
    parts: ChainParts,

    /// TransmutationStepper for solving Bateman equations
    stepper: Box<dyn TransmutationStepper>,
}

impl TransmutationDriver {
    /// Create a new TransmutationDriver.
    ///
    /// # Arguments
    /// * `chain` - Pre-loaded transmutation chain (from load_chain())
    /// * `stepper` - TransmutationStepper implementation (e.g., ForwardEulerStepper)
    ///
    /// The chain is assumed complete. A driver over a chain built without a
    /// subsection should be made with [`TransmutationDriver::with_parts`], so
    /// a rate needing what was left out is refused rather than solved without.
    pub fn new(
        chain: Arc<HashMap<String, ChainNuclide>>,
        stepper: Box<dyn TransmutationStepper>,
    ) -> Self {
        Self::with_parts(chain, ChainParts::default(), stepper)
    }

    /// A driver over a chain that may be missing optional subsections.
    pub fn with_parts(
        chain: Arc<HashMap<String, ChainNuclide>>,
        parts: ChainParts,
        stepper: Box<dyn TransmutationStepper>,
    ) -> Self {
        Self {
            chain,
            parts,
            stepper,
        }
    }

    /// Transmute a single material by one timestep.
    ///
    /// # Arguments
    /// * `material` - Current material composition
    /// * `rates` - Reaction rates for this material (sigma*phi for each nuclide/reaction)
    /// * `fy_weights` - Spectrum weights over each fissionable nuclide's
    ///   tabulated fission-yield energies (issue #379)
    /// * `dt` - Timestep duration [s]
    ///
    /// # Returns
    /// New material with updated composition (in sum mode)
    pub fn transmute_material(
        &self,
        material: &Material,
        rates: &ReactionRates,
        fy_weights: &FissionYieldWeights,
        dt: f64,
    ) -> Result<Material, Box<dyn Error>> {
        self.stepper
            .step(material, &self.chain, rates, fy_weights, self.parts, dt)
    }

    /// Compute approximate reaction rates for a material given the scalar flux.
    ///
    /// **Note:** This method uses a simple energy-grid average for cross sections,
    /// not a proper flux-weighted one-group collapse. It is intended for standalone
    /// testing and debugging only. For production coupled transport-transmutation, use
    /// `Model::transmute()` which scores flux-weighted reaction rates via
    /// `TransmutationTallies` during transport.
    ///
    /// For each nuclide in the material that exists in the chain,
    /// computes sigma*phi for each relevant reaction type.
    ///
    /// # Arguments
    /// * `material` - Material with loaded nuclear data
    /// * `flux` - Scalar neutron flux [n/cm^2/s] in this material
    ///
    /// # Returns
    /// Reaction rates: nuclide_name -> reaction_type -> rate [1/s]
    pub fn compute_reaction_rates(&self, material: &Material, flux: f64) -> ReactionRates {
        let mut rates: ReactionRates = HashMap::new();

        if flux <= 0.0 {
            return rates;
        }

        for nuclide_name in material.nuclides.keys() {
            // Check if this nuclide is in the chain
            let chain_nuclide = match self.chain.get(nuclide_name) {
                Some(cn) => cn,
                None => continue,
            };

            // Skip nuclides with no reactions
            if chain_nuclide.reactions.is_empty() {
                continue;
            }

            // Get unique reaction types for this nuclide
            let mut reaction_types: Vec<&str> = chain_nuclide
                .reactions
                .iter()
                .map(|r| r.kind.as_str())
                .collect();
            reaction_types.sort();
            reaction_types.dedup();

            let mut nuclide_rates: HashMap<String, f64> = HashMap::new();

            for &rx_type in &reaction_types {
                // Get the MT number for this reaction type
                let mt = match reaction_type_to_mt(rx_type) {
                    Some(mt) => mt,
                    None => continue, // Unknown reaction type, skip
                };

                // Compute one-group cross section from the material's microscopic data
                // Returns value in barns; convert to cm² for rate calculation
                let sigma_1g = compute_one_group_xs(material, nuclide_name, mt);

                if sigma_1g > 0.0 {
                    // rate = sigma [cm²] * phi [n/cm²/s] = [1/s]
                    nuclide_rates.insert(rx_type.to_string(), sigma_1g * 1.0e-24 * flux);
                }
            }

            if !nuclide_rates.is_empty() {
                rates.insert(nuclide_name.clone(), nuclide_rates);
            }
        }

        rates
    }

    /// Get a reference to the transmutation chain.
    pub fn chain(&self) -> &HashMap<String, ChainNuclide> {
        &self.chain
    }
}

/// Compute approximate one-group microscopic cross section for a nuclide/MT pair.
///
/// **Warning:** Uses a simple arithmetic average over all energy grid points, which
/// is NOT flux-weighted. This can be significantly inaccurate, especially for
/// resonance-dominated or thermal reactions. Only suitable for standalone testing.
/// The coupled `Model::transmute()` path uses `TransmutationTallies` for proper
/// flux-weighted reaction rate scoring during transport.
fn compute_one_group_xs(material: &Material, nuclide_name: &str, mt: i32) -> f64 {
    // Get nuclide data
    let nuclide_data = match material.nuclide_data.get(nuclide_name) {
        Some(nd) => nd,
        None => return 0.0,
    };

    // Get temperature for data lookup
    let temperature = if material.temperature().is_empty() {
        match crate::default_temperature(nuclide_data) {
            Some(t) => t,
            None => return 0.0,
        }
    } else {
        material.temperature().to_string()
    };

    // Get reactions for this temperature
    let reactions = match nuclide_data.reactions_for_temp(&temperature) {
        Some(r) => r,
        None => return 0.0,
    };

    // Get the cross section for this MT
    let reaction = match reactions.get(&mt) {
        Some(r) => r,
        None => return 0.0,
    };

    // Compute simple average of the cross section values
    if reaction.cross_section.is_empty() {
        return 0.0;
    }

    let sum: f64 = reaction.cross_section.iter().sum();
    sum / reaction.cross_section.len() as f64
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use crate::{mt_to_reaction_type, reaction_type_to_mt};

    #[test]
    fn test_reaction_type_to_mt() {
        assert_eq!(reaction_type_to_mt("(n,gamma)"), Some(102));
        assert_eq!(reaction_type_to_mt("(n,fission)"), Some(18));
        assert_eq!(reaction_type_to_mt("(n,2n)"), Some(16));
        assert_eq!(reaction_type_to_mt("unknown"), None);
    }

    #[test]
    fn the_chain_spelling_of_fission_resolves_to_mt_18() {
        // The chain files write MT 18 as "fission". While this did not resolve,
        // no fission rate was computed and MT 18 was never tallied, so fission
        // was absent from the transmutation network altogether and the
        // spectrum fold of issue #379 had nothing to act on.
        assert_eq!(reaction_type_to_mt("fission"), Some(18));
    }

    #[test]
    fn test_mt_to_reaction_type() {
        assert_eq!(mt_to_reaction_type(102), Some("(n,gamma)"));
        // Canonical, because the rate key this produces has to match the kind
        // the chain uses when the matrix builder looks the rate back up.
        assert_eq!(mt_to_reaction_type(18), Some("fission"));
        assert_eq!(mt_to_reaction_type(999), None);
    }
}
