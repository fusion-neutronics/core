//! Delayed fission neutrons: the groups' yields and their folded spectrum.
//!
//! ENDF splits the delayed neutrons into six precursor groups, each with its own
//! decay constant, its own yield `nu_d,g(E)`, and its own outgoing spectrum. The
//! spectra are much softer than the prompt spectrum (group means around 0.4 to 0.6
//! MeV against roughly 2 MeV prompt), so which spectrum a fission neutron is born
//! from matters even though delayed neutrons are only a fraction of a percent to a
//! couple of percent of the source.
//!
//! yamc emits delayed neutrons immediately rather than after a precursor decay, so
//! the group a neutron came from has no observable consequence: only the yield-
//! weighted mixture of the six spectra does. That mixture is a single fixed
//! distribution, because the group FRACTIONS are energy-independent even though
//! the absolute yields are not (exactly constant to six decimals from 1e-5 eV to
//! 20 MeV for U235, U238, Pu239 and Th232; see
//! `crates/yamc/tests/delayed_neutron_data.rs`). Folding the groups once therefore
//! costs no accuracy, and leaves one spectrum for the CPU and GPU to sample
//! instead of six.
//!
//! Issue #364.

use crate::reaction_product::{
    weighted_energy_mixture, AngleEnergyDistribution, EnergyDistribution, FissionChiFlat,
    ParticleType, ReactionProduct, Yield,
};

/// A nuclide's delayed-neutron groups, resolved once and cached.
#[derive(Debug, Clone)]
pub struct DelayedNeutronData {
    /// Per-group `nu_d,g(E)`. Summed to get `nu_d(E)`; the grids are tiny (4 to 15
    /// points for the evaluations checked) so summing per fission event is cheap.
    group_yields: Vec<Yield>,
    /// Yield-weighted fold of the groups' spectra, flattened for the shared
    /// CPU/GPU samplers.
    chi_flat: FissionChiFlat,
}

impl DelayedNeutronData {
    /// `nu_d(E)`: the delayed groups' total yield, in neutrons per fission.
    pub fn nu(&self, energy: f64) -> f64 {
        self.group_yields.iter().map(|y| y.evaluate(energy)).sum()
    }

    /// Per-group `nu_d,g(E)`, in the evaluation's group order.
    pub fn group_nu(&self, energy: f64) -> Vec<f64> {
        self.group_yields
            .iter()
            .map(|y| y.evaluate(energy))
            .collect()
    }

    /// The folded delayed spectrum, ready for the flat samplers.
    pub fn chi_flat(&self) -> &FissionChiFlat {
        &self.chi_flat
    }

    /// Build from a fission reaction's product list. Returns `None` unless there is
    /// at least one delayed group with both a yield and a foldable spectrum, so a
    /// nuclide with no delayed data (Am240) and one whose delayed spectra cannot be
    /// folded both read as "no delayed neutrons" and keep the prompt-only
    /// behaviour.
    pub fn from_products(products: &[ReactionProduct]) -> Option<Self> {
        let groups: Vec<(f64, &Yield, &EnergyDistribution)> = products
            .iter()
            .filter(|p| p.is_particle_type(&ParticleType::Neutron) && p.is_delayed())
            .filter_map(|p| {
                let y = p.product_yield.as_ref()?;
                // Weight the fold by the group's yield at the low end of its own
                // grid. Any incident energy gives the same mixture because the
                // group fractions are energy-independent; a fixed reference keeps
                // the cached fold independent of which fission event built it.
                let w = y.evaluate(WEIGHT_REFERENCE_EV);
                (w > 0.0).then_some((w, y, spectrum_of(p)?))
            })
            .collect();
        if groups.is_empty() {
            return None;
        }
        let parts: Vec<(f64, &EnergyDistribution)> =
            groups.iter().map(|(w, _, d)| (*w, *d)).collect();
        let chi_flat = weighted_energy_mixture(&parts)?.to_fission_chi_flat();
        if matches!(chi_flat, FissionChiFlat::None) {
            return None;
        }
        Some(Self {
            group_yields: groups.iter().map(|(_, y, _)| (*y).clone()).collect(),
            chi_flat,
        })
    }
}

/// Incident energy at which the group yields are read to weight the fold. The
/// group fractions do not depend on it (see the module docs); it only has to be
/// fixed. Thermal, the energy `beta` is conventionally quoted at.
const WEIGHT_REFERENCE_EV: f64 = 0.0253;

/// A product's outgoing-energy distribution, if it has an uncorrelated one.
fn spectrum_of(product: &ReactionProduct) -> Option<&EnergyDistribution> {
    product.distribution.first().and_then(|d| match d {
        AngleEnergyDistribution::UncorrelatedAngleEnergy { energy, .. } => energy.as_ref(),
        _ => None,
    })
}

/// Per-nuclide cache of [`DelayedNeutronData`], built lazily on first fission and
/// shared read-only across transport threads. Resets on `clone()` and skipped by
/// serde, mirroring [`crate::reaction_product::FissionChiFlatCache`].
#[derive(Debug, Default)]
pub struct DelayedNeutronCache(std::sync::OnceLock<Option<DelayedNeutronData>>);

impl Clone for DelayedNeutronCache {
    fn clone(&self) -> Self {
        Self(std::sync::OnceLock::new())
    }
}

impl DelayedNeutronCache {
    /// Return the cached delayed data, building it with `build` on first access.
    pub fn get_or_build(
        &self,
        build: impl FnOnce() -> Option<DelayedNeutronData>,
    ) -> Option<&DelayedNeutronData> {
        self.0.get_or_init(build).as_ref()
    }
}
