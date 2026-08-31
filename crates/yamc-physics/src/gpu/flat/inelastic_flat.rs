//! ONE reaction's complete inelastic kinematics data in the flat layout
//! [`sample_inelastic_kinematics`](super::inelastic_dispatch::sample_inelastic_kinematics)
//! reads, plus the lazily-populated cache that lets the CPU production
//! transport obtain it per collision without re-flattening (issue #111).
//!
//! yamc-gpu builds the same arrays for a whole material: every MT slot of
//! every nuclide concatenated into global CSR buffers, addressed as
//! `slab * MT_INELASTIC_COUNT + slot`. The CPU transport needs exactly one
//! (nuclide, MT) at a time, so [`InelasticFlat`] holds that single slot and
//! the addressing collapses:
//!
//!   * `slab = 0`, `selected_slot = 0` (see [`InelasticFlat::SLAB`] /
//!     [`InelasticFlat::SLOT`]), so `mat_slot == 0` and every per-slot scalar
//!     array here has length 1;
//!   * every CSR base offset is 0 (`*_ae_offset == [0]`), and the per-row /
//!     per-x-point offsets (`eout_x_offset`, `corr_x_offset`,
//!     `corr_mu_offset`, `km_x_offset`, `angle_mu_offset`) are the plain
//!     zero-based cumulative sums of this slot's own counts.
//!
//! Every public field of [`InelasticFlat`] is named after, and is passed
//! straight through to, the parameter of the same name on
//! `sample_inelastic_kinematics`; [`InelasticFlat::sample_kinematics`] does
//! exactly that forwarding.
//!
//! Where the cache lives: yamc-nuclide cannot depend on yamc-physics, and the
//! per-law flattening this builds on ([`super::eout_extract`]) lives in
//! yamc-physics, so unlike the angular-only
//! `yamc_nuclide::reaction_product::InelasticAngleFlatCache` this cache
//! CANNOT be a field on `Nuclide`. It is instead a standalone
//! [`InelasticFlatCache`] the caller owns (one per simulation run), keyed by
//! (nuclide identity, MT). Same shape as the angle-side precedent otherwise:
//! interior mutability, lazily built per key on first use, shared read-only
//! (`Arc`) across transport threads.
//!
//! Bit-sensitive: the arrays are produced by the same extractors yamc-gpu
//! packs into its buffers, so a CPU collision routed through this cache
//! samples byte-identical data to the GPU kernel's twin.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::eout_extract::{
    CorrSlot, EoutSlot, EvapSlot, KalbachSlot, MaxwellSlot, NbpsSlot, WattSlot,
};
use yamc_nuclide::nuclide::Nuclide;
use yamc_nuclide::reaction::Reaction;
use yamc_nuclide::reaction_product::{AngleEnergyDistribution, ElasticAngleFlat};

/// Flatten a reaction's elastic angular distribution into the tight,
/// variable-length [`ElasticAngleFlat`] layout (issue #104), reusing the
/// exact same `AngleDistribution::to_elastic_flat` the CPU transport uses
/// so both backends sample byte-identical data with no stride-subsampling.
/// Returns an empty table (kernel falls back to isotropic) when the
/// reaction has no usable neutron angular data.
///
/// Moved here from yamc-gpu's `neutron::xs::distributions` (issue #111) so
/// [`InelasticFlat::from_reaction`] and the GPU's `build_per_mt_angle_buffers`
/// share one definition; yamc-gpu re-exports it at its old path.
pub fn elastic_flat_from_reaction(reaction: &Reaction) -> ElasticAngleFlat {
    let angle = super::eout_extract::first_neutron_product(reaction).and_then(|p| {
        p.distribution.iter().find_map(|d| match d {
            AngleEnergyDistribution::UncorrelatedAngleEnergy { angle, .. } => Some(angle),
            _ => None,
        })
    });
    match angle {
        Some(a) => a.to_elastic_flat(),
        None => ElasticAngleFlat::empty(),
    }
}

/// Zero-based CSR row starts for a tight per-row count array:
/// `out[i] = counts[..i].sum()`. The GPU's `push_*_csr_offsets` helpers build
/// the same running sum but biased by the global buffer base; a single-slot
/// bundle has no prior rows, so the base is 0.
fn csr_offsets(counts: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(counts.len());
    let mut acc = 0u32;
    for &n in counts {
        out.push(acc);
        acc += n;
    }
    out
}

/// One reaction's flattened inelastic kinematics: the angular table, the
/// outgoing-energy law (whichever of continuous-tabular / correlated /
/// Kalbach-Mann / evaporation / n-body / Maxwell / Watt / tabulated it
/// carries), and the per-slot scalars, all in the single-slot layout
/// described at the module level.
///
/// Field names match the parameters of
/// [`sample_inelastic_kinematics`](super::inelastic_dispatch::sample_inelastic_kinematics)
/// one-for-one, so a caller can pass `&flat.eout_x` and friends straight
/// through (or just call [`Self::sample_kinematics`]).
#[derive(Debug, Clone)]
pub struct InelasticFlat {
    // Slice B: tabulated CM angular table.
    pub angle_n_energies: Vec<u32>,
    pub angle_ae_offset: Vec<u32>,
    pub angle_energy_grid: Vec<f64>,
    pub angle_n_mu: Vec<u32>,
    pub angle_mu_offset: Vec<u32>,
    pub angle_mu: Vec<f64>,
    pub angle_cdf: Vec<f64>,
    pub angle_pdf: Vec<f64>,
    pub angle_interp: Vec<u32>,
    // Slice C / tabulated-equiprobable: outgoing-energy table + the
    // `EOUT_KIND_*` discriminant the dispatcher branches on.
    pub eout_kind: Vec<u32>,
    pub eout_n_energies: Vec<u32>,
    pub eout_ae_offset: Vec<u32>,
    pub eout_energy_grid: Vec<f64>,
    pub eout_n_x: Vec<u32>,
    pub eout_x_offset: Vec<u32>,
    pub eout_x: Vec<f64>,
    pub eout_cdf: Vec<f64>,
    pub eout_histogram_interp: Vec<u32>,
    pub eout_p: Vec<f64>,
    pub eout_interp: Vec<u32>,
    pub eout_n_discrete: Vec<u32>,
    // Slice D: correlated angle-energy.
    pub corr_n_energies: Vec<u32>,
    pub corr_n_components: Vec<u32>,
    pub corr_ae_offset: Vec<u32>,
    pub corr_energy_grid: Vec<f64>,
    pub corr_n_x: Vec<u32>,
    pub corr_x_offset: Vec<u32>,
    pub corr_x: Vec<f64>,
    pub corr_cdf: Vec<f64>,
    pub corr_p: Vec<f64>,
    pub corr_interp: Vec<u32>,
    pub corr_n_discrete: Vec<u32>,
    pub corr_n_mu: Vec<u32>,
    pub corr_mu_offset: Vec<u32>,
    pub corr_mu: Vec<f64>,
    pub corr_mu_cdf: Vec<f64>,
    pub corr_mu_pdf: Vec<f64>,
    pub corr_mu_interp: Vec<u32>,
    /// `1` when the reaction's secondary kinematics are CM-frame (the
    /// dispatcher then applies the CM-to-lab boost), `0` for lab-frame.
    pub scatter_in_cm_per_mt: Vec<u32>,
    // Slice E: Kalbach-Mann.
    pub km_n_energies: Vec<u32>,
    pub km_ae_offset: Vec<u32>,
    pub km_energy_grid: Vec<f64>,
    pub km_interp: Vec<u32>,
    pub km_n_discrete: Vec<u32>,
    pub km_n_x: Vec<u32>,
    pub km_x_offset: Vec<u32>,
    pub km_x: Vec<f64>,
    pub km_p: Vec<f64>,
    pub km_c: Vec<f64>,
    pub km_r: Vec<f64>,
    pub km_a: Vec<f64>,
    // Slice F: Evaporation / n-body phase space / Maxwell / Watt.
    pub evap_n_energies: Vec<u32>,
    pub evap_n_components: Vec<u32>,
    pub evap_ae_offset: Vec<u32>,
    pub evap_theta_offset: Vec<u32>,
    pub evap_energy_grid: Vec<f64>,
    pub evap_theta: Vec<f64>,
    pub evap_u: Vec<f64>,
    pub nbps_n_bodies: Vec<u32>,
    pub nbps_total_mass: Vec<f64>,
    pub maxwell_n_energies: Vec<u32>,
    pub maxwell_ae_offset: Vec<u32>,
    pub maxwell_energy_grid: Vec<f64>,
    pub maxwell_theta: Vec<f64>,
    pub maxwell_u: Vec<f64>,
    pub watt_n_energies: Vec<u32>,
    pub watt_ae_offset: Vec<u32>,
    pub watt_energy_grid: Vec<f64>,
    pub watt_a: Vec<f64>,
    pub watt_b: Vec<f64>,
    pub watt_u: Vec<f64>,
    /// The reaction's Q value (signed), the dispatcher's `q_value` argument.
    /// On GPU this is `q_inelastic_per_mt[mat_slot]`, which for a single
    /// nuclide is a peak-cross-section-weighted average over one term, i.e.
    /// this same value.
    pub q_value: f64,
}

impl InelasticFlat {
    /// Slab index for a single-reaction bundle: there is one slab.
    pub const SLAB: usize = 0;
    /// MT-slot index for a single-reaction bundle: there is one slot, so
    /// `mat_slot = SLAB * MT_INELASTIC_COUNT + SLOT == 0`.
    pub const SLOT: usize = 0;

    /// Flatten one reaction through the SAME per-law extractors yamc-gpu
    /// packs into its per-material buffers ([`super::eout_extract`]), then
    /// wrap them in the single-slot layout: per-slot scalars become
    /// length-1 arrays and the CSR offsets become zero-based.
    pub fn from_reaction(reaction: &Reaction) -> Self {
        let angle = elastic_flat_from_reaction(reaction);
        let eout = EoutSlot::from_reaction(reaction);
        let corr = CorrSlot::from_reaction(reaction);
        let km = KalbachSlot::from_reaction(reaction);
        let evap = EvapSlot::from_reaction(reaction);
        let nbps = NbpsSlot::from_reaction(reaction);
        let maxwell = MaxwellSlot::from_reaction(reaction);
        let watt = WattSlot::from_reaction(reaction);

        // Per-row / per-x-point CSR starts, zero-based (this slot's rows are
        // the only rows). `angle` already carries its own `mu_offset` from
        // `to_elastic_flat`, built the same way.
        let eout_x_offset = csr_offsets(&eout.n_x);
        let corr_x_offset = csr_offsets(&corr.n_x);
        let corr_mu_offset = csr_offsets(&corr.n_mu);
        let km_x_offset = csr_offsets(&km.n_x);

        Self {
            // `build_per_mt_angle_buffers` derives the slot's row count from
            // the flattened grid length, not from a slot field.
            angle_n_energies: vec![angle.energy_grid.len() as u32],
            angle_ae_offset: vec![0],
            angle_energy_grid: angle.energy_grid,
            angle_n_mu: angle.n_mu,
            angle_mu_offset: angle.mu_offset,
            angle_mu: angle.mu,
            angle_cdf: angle.cdf,
            angle_pdf: angle.pdf,
            angle_interp: angle.interp,
            eout_kind: vec![eout.kind],
            eout_n_energies: vec![eout.n_energies],
            eout_ae_offset: vec![0],
            eout_energy_grid: eout.energy_grid,
            eout_n_x: eout.n_x,
            eout_x_offset,
            eout_x: eout.x,
            eout_cdf: eout.cdf,
            eout_histogram_interp: vec![eout.histogram_interp],
            eout_p: eout.p,
            eout_interp: eout.interp,
            eout_n_discrete: eout.n_discrete,
            corr_n_energies: vec![corr.n_energies],
            corr_n_components: vec![corr.n_components],
            corr_ae_offset: vec![0],
            corr_energy_grid: corr.energy_grid,
            corr_n_x: corr.n_x,
            corr_x_offset,
            corr_x: corr.x,
            corr_cdf: corr.cdf,
            corr_p: corr.p,
            corr_interp: corr.interp,
            corr_n_discrete: corr.n_discrete,
            corr_n_mu: corr.n_mu,
            corr_mu_offset,
            corr_mu: corr.mu,
            corr_mu_cdf: corr.mu_cdf,
            corr_mu_pdf: corr.mu_pdf,
            corr_mu_interp: corr.mu_interp,
            scatter_in_cm_per_mt: vec![u32::from(reaction.scatter_in_cm)],
            km_n_energies: vec![km.n_energies],
            km_ae_offset: vec![0],
            km_energy_grid: km.energy_grid,
            km_interp: km.interp,
            km_n_discrete: km.n_discrete,
            km_n_x: km.n_x,
            km_x_offset,
            km_x: km.x,
            km_p: km.p,
            km_c: km.cdf,
            km_r: km.r,
            km_a: km.a,
            evap_n_energies: vec![evap.n_energies],
            evap_n_components: vec![evap.n_components],
            evap_ae_offset: vec![0],
            evap_theta_offset: vec![0],
            evap_energy_grid: evap.energy_grid,
            evap_theta: evap.theta,
            evap_u: evap.u_grid,
            nbps_n_bodies: vec![nbps.n_bodies],
            nbps_total_mass: vec![nbps.total_mass],
            maxwell_n_energies: vec![maxwell.n_energies],
            maxwell_ae_offset: vec![0],
            maxwell_energy_grid: maxwell.energy_grid,
            maxwell_theta: maxwell.theta,
            maxwell_u: vec![maxwell.u],
            watt_n_energies: vec![watt.n_energies],
            watt_ae_offset: vec![0],
            watt_energy_grid: watt.energy_grid,
            watt_a: watt.a,
            watt_b: watt.b,
            watt_u: vec![watt.u],
            q_value: reaction.q_value,
        }
    }

    /// True when this slot actually carries an outgoing-energy LAW the
    /// dispatcher can sample, i.e. the flattening recognised the reaction's
    /// secondary-energy distribution AND that law's arrays are non-empty.
    ///
    /// False means the slot fell back to `EOUT_KIND_LEVEL_INELASTIC`, which
    /// the dispatcher services from the caller's closed-form `e_cm` alone.
    /// That is correct for a genuine `LevelInelastic` distribution, but it is
    /// also where [`super::eout_extract::EoutSlot::from_reaction`] parks a law
    /// it does not recognise (`DiscretePhoton`, an out-of-range n-body count,
    /// a product with no distribution at all). A CPU caller that has a richer
    /// legacy sampler for those cases should consult this before routing here,
    /// so an unrecognised law falls back instead of silently degrading to
    /// closed-form-Q kinematics.
    pub fn has_outgoing_energy_law(&self) -> bool {
        use super::inelastic_dispatch::{
            EOUT_KIND_CONTINUOUS_TABULAR, EOUT_KIND_CORRELATED, EOUT_KIND_EVAPORATION,
            EOUT_KIND_KALBACH_MANN, EOUT_KIND_MAXWELL, EOUT_KIND_NBODY_PHASE_SPACE,
            EOUT_KIND_TABULATED, EOUT_KIND_WATT,
        };
        // Each arm repeats the `n > 0` (or in-range body count) guard the
        // matching dispatcher branch takes before it overrides `e_cm`; an
        // empty law would leave the closed form in place.
        match self.eout_kind[Self::SLOT] {
            EOUT_KIND_CONTINUOUS_TABULAR | EOUT_KIND_TABULATED => {
                self.eout_n_energies[Self::SLOT] > 0
            }
            EOUT_KIND_CORRELATED => self.corr_n_energies[Self::SLOT] > 0,
            EOUT_KIND_KALBACH_MANN => self.km_n_energies[Self::SLOT] > 0,
            EOUT_KIND_EVAPORATION => self.evap_n_energies[Self::SLOT] > 0,
            EOUT_KIND_NBODY_PHASE_SPACE => (3..=5).contains(&self.nbps_n_bodies[Self::SLOT]),
            EOUT_KIND_MAXWELL => self.maxwell_n_energies[Self::SLOT] > 0,
            EOUT_KIND_WATT => self.watt_n_energies[Self::SLOT] > 0,
            _ => false,
        }
    }

    /// Sample the outgoing `(mu, e_out, ok)` for a collision on this
    /// reaction, forwarding every array to
    /// [`sample_inelastic_kinematics`](super::inelastic_dispatch::sample_inelastic_kinematics)
    /// at the single-slot address (`slab = 0`, `selected_slot = 0`). The
    /// caller supplies what is not a property of the reaction: the incident
    /// energy, the target mass (the nuclide's AWR), the closed-form CM energy
    /// (`mass_ratio * (e_in - threshold)`, from [`Self::q_value`]), the
    /// isotropic-fallback uniform `xi3` drawn at the reaction-type split, and
    /// the shared 64-bit PCG `state`.
    pub fn sample_kinematics(
        &self,
        e_in: f64,
        target_mass: f64,
        e_cm_closed_form: f64,
        xi3_isotropic_fallback: f64,
        state: &mut u64,
    ) -> (f64, f64, bool) {
        super::inelastic_dispatch::sample_inelastic_kinematics(
            e_in,
            target_mass,
            e_cm_closed_form,
            xi3_isotropic_fallback,
            Self::SLAB,
            Self::SLOT,
            state,
            &self.angle_n_energies,
            &self.angle_ae_offset,
            &self.angle_energy_grid,
            &self.angle_n_mu,
            &self.angle_mu_offset,
            &self.angle_mu,
            &self.angle_cdf,
            &self.angle_pdf,
            &self.angle_interp,
            &self.eout_kind,
            &self.eout_n_energies,
            &self.eout_ae_offset,
            &self.eout_energy_grid,
            &self.eout_n_x,
            &self.eout_x_offset,
            &self.eout_x,
            &self.eout_cdf,
            &self.eout_histogram_interp,
            &self.eout_p,
            &self.eout_interp,
            &self.eout_n_discrete,
            &self.corr_n_energies,
            &self.corr_n_components,
            &self.corr_ae_offset,
            &self.corr_energy_grid,
            &self.corr_n_x,
            &self.corr_x_offset,
            &self.corr_x,
            &self.corr_cdf,
            &self.corr_p,
            &self.corr_interp,
            &self.corr_n_discrete,
            &self.corr_n_mu,
            &self.corr_mu_offset,
            &self.corr_mu,
            &self.corr_mu_cdf,
            &self.corr_mu_pdf,
            &self.corr_mu_interp,
            &self.scatter_in_cm_per_mt,
            &self.km_n_energies,
            &self.km_ae_offset,
            &self.km_energy_grid,
            &self.km_interp,
            &self.km_n_discrete,
            &self.km_n_x,
            &self.km_x_offset,
            &self.km_x,
            &self.km_p,
            &self.km_c,
            &self.km_r,
            &self.km_a,
            &self.evap_n_energies,
            &self.evap_n_components,
            &self.evap_ae_offset,
            &self.evap_theta_offset,
            &self.evap_energy_grid,
            &self.evap_theta,
            &self.evap_u,
            &self.nbps_n_bodies,
            &self.nbps_total_mass,
            &self.maxwell_n_energies,
            &self.maxwell_ae_offset,
            &self.maxwell_energy_grid,
            &self.maxwell_theta,
            &self.maxwell_u,
            &self.watt_n_energies,
            &self.watt_ae_offset,
            &self.watt_energy_grid,
            &self.watt_a,
            &self.watt_b,
            &self.watt_u,
            self.q_value,
        )
    }
}

/// Lazily-populated cache of [`InelasticFlat`] bundles, keyed by
/// (nuclide identity, MT) (issue #111).
///
/// Mirrors `yamc_nuclide::reaction_product::InelasticAngleFlatCache`: entries
/// are built on the first collision that needs them and handed out as `Arc`
/// clones, so reads after the first take only a shared lock and transport
/// threads share one copy of the data.
///
/// OWNERSHIP: the caller owns the cache (one per simulation run) rather than
/// `Nuclide` owning it, because the flattening lives in yamc-physics and
/// yamc-nuclide must not depend on yamc-physics. Nuclide identity is the
/// address of the `Nuclide` the reaction belongs to, which is stable and
/// unique for as long as the loaded data is alive: the model holds every
/// nuclide behind an `Arc` for the whole run and the data is immutable during
/// transport. The cache must therefore not outlive the nuclide data it was
/// filled from (a run-scoped cache does not), otherwise a later allocation
/// could reuse a freed nuclide's address.
#[derive(Debug, Default)]
pub struct InelasticFlatCache(RwLock<HashMap<(usize, i32), Arc<InelasticFlat>>>);

impl InelasticFlatCache {
    /// Return the flat bundle for `reaction` (which must be `nuclide`'s
    /// reaction for its MT), building it on first use.
    pub fn get_or_build(&self, nuclide: &Nuclide, reaction: &Reaction) -> Arc<InelasticFlat> {
        let key = (std::ptr::from_ref(nuclide) as usize, reaction.mt_number);
        if let Some(flat) = self.0.read().unwrap_or_else(|p| p.into_inner()).get(&key) {
            return flat.clone();
        }
        let mut w = self.0.write().unwrap_or_else(|p| p.into_inner());
        w.entry(key)
            .or_insert_with(|| Arc::new(InelasticFlat::from_reaction(reaction)))
            .clone()
    }

    /// Number of cached (nuclide, MT) bundles. Diagnostics / tests.
    pub fn len(&self) -> usize {
        self.0.read().unwrap_or_else(|p| p.into_inner()).len()
    }

    /// True while nothing has been built yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yamc_nuclide::particle_type::ParticleType;
    use yamc_nuclide::reaction_product::ReactionProduct;

    /// The single-slot addressing the whole module rests on: `mat_slot`
    /// (`slab * MT_INELASTIC_COUNT + slot`) must be 0 so the length-1
    /// per-slot arrays are in range.
    #[test]
    fn single_slot_address_is_zero() {
        let mat_slot = InelasticFlat::SLAB * super::super::inelastic_dispatch::MT_INELASTIC_COUNT
            + InelasticFlat::SLOT;
        assert_eq!(mat_slot, 0);
    }

    #[test]
    fn csr_offsets_are_zero_based_running_sums() {
        assert_eq!(csr_offsets(&[]), Vec::<u32>::new());
        assert_eq!(csr_offsets(&[3, 0, 5, 2]), vec![0, 3, 3, 8]);
    }

    /// One MT-91-shaped reaction whose neutron product carries `dist`.
    fn reaction_with(dist: Vec<AngleEnergyDistribution>) -> Reaction {
        Reaction {
            cross_section: vec![1.0, 1.0].into(),
            threshold_idx: 0,
            energy: vec![1.0e5, 2.0e7].into(),
            mt_number: 91,
            q_value: -1.0e6,
            products: vec![ReactionProduct {
                particle: ParticleType::Neutron,
                emission_mode: "prompt".to_string(),
                decay_rate: 0.0,
                applicability: Vec::new(),
                distribution: dist,
                product_yield: None,
            }],
            scatter_in_cm: true,
            redundant: false,
        }
    }

    /// The fallback guard a CPU caller relies on to keep its richer legacy
    /// sampler: a reaction the flattening does not recognise must report
    /// "no outgoing-energy law", NOT silently accept the closed-form-Q slot.
    #[test]
    fn unrecognised_law_reports_no_outgoing_energy_law() {
        // No products at all.
        let mut bare = reaction_with(Vec::new());
        bare.products.clear();
        assert!(!InelasticFlat::from_reaction(&bare).has_outgoing_energy_law());
        // A neutron product with no distribution.
        assert!(!InelasticFlat::from_reaction(&reaction_with(Vec::new())).has_outgoing_energy_law());
        // An n-body body count outside the supported 3..=5, which
        // `EoutSlot::from_reaction` deliberately parks on the closed form.
        let out_of_range = reaction_with(vec![AngleEnergyDistribution::NBodyPhaseSpace {
            n_bodies: 6,
            total_mass: 56.0,
            awr: 55.4,
            q_value: -1.0e6,
        }]);
        assert!(!InelasticFlat::from_reaction(&out_of_range).has_outgoing_energy_law());
    }

    /// ... and a law the flattening DOES encode must report that it has one,
    /// so the caller routes to the shared sampler rather than the legacy path.
    #[test]
    fn supported_law_reports_an_outgoing_energy_law() {
        let nbps = reaction_with(vec![AngleEnergyDistribution::NBodyPhaseSpace {
            n_bodies: 4,
            total_mass: 56.0,
            awr: 55.4,
            q_value: -1.0e6,
        }]);
        let flat = InelasticFlat::from_reaction(&nbps);
        assert!(flat.has_outgoing_energy_law());
        assert_eq!(flat.nbps_n_bodies[InelasticFlat::SLOT], 4);
    }
}
