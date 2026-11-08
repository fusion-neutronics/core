//! Continuous-energy neutron interaction data for one nuclide.

use std::collections::{BTreeMap, BTreeSet};

use crate::ace::{self, MetastableScheme, Table};
use crate::data::{gnds_name, sum_rule, temperature_str, ATOMIC_SYMBOL, EV_PER_MEV, K_BOLTZMANN};
use crate::error::{Error, Result};
use crate::function::Tabulated1D;
use crate::material::Material;
use crate::mf::mf4::{AngleAtEnergy, AngleDistribution};
use crate::reaction::{photon_products_ace, Reaction};
use crate::urr::ProbabilityTables;

/// Reactions whose cross sections are sums of others even where the sum rules
/// do not say so.
///
/// The first five are the particle production cross sections, which count
/// every reaction that emits that particle; MT=444 is damage energy, which is
/// a weighted sum rather than a channel.
const ALWAYS_REDUNDANT: [i32; 6] = [203, 204, 205, 206, 207, 444];

/// The transmutation reactions an ACE file may give only as separate levels.
const SUMMED_IF_ABSENT: [i32; 6] = [16, 103, 104, 105, 106, 107];

/// Continuous-energy neutron interaction data for one nuclide.
///
/// Built from an ENDF evaluation, which is at a single temperature, or from
/// one or more ACE tables, which carry one temperature each and are merged
/// with [`IncidentNeutron::add_temperature_from_ace`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct IncidentNeutron {
    pub atomic_number: u32,
    pub mass_number: u32,
    /// Zero for the ground state.
    pub metastable: u32,
    pub atomic_weight_ratio: Option<f64>,
    /// Temperatures as kT in eV.
    pub k_ts: Vec<f64>,
    /// The reactions, by MT.
    pub reactions: BTreeMap<i32, Reaction>,
    /// The nuclide's energy grid at each temperature, in eV. Empty for an
    /// ENDF evaluation, which gives each reaction its own grid.
    pub energy: BTreeMap<String, Vec<f64>>,
    /// Unresolved resonance probability tables, by temperature.
    pub urr: BTreeMap<String, ProbabilityTables>,
    /// A name given explicitly, overriding the one the Z, A and metastable
    /// state imply.
    pub name_override: Option<String>,
}

impl IncidentNeutron {
    /// An empty nuclide.
    pub fn new(atomic_number: u32, mass_number: u32, metastable: u32) -> IncidentNeutron {
        IncidentNeutron {
            atomic_number,
            mass_number,
            metastable,
            ..Default::default()
        }
    }

    /// The nuclide's name in GNDS convention, e.g. `"Am242_m1"`.
    pub fn name(&self) -> String {
        self.name_override
            .clone()
            .unwrap_or_else(|| gnds_name(self.atomic_number, self.mass_number, self.metastable))
    }

    /// The atomic symbol, e.g. `"Zr"`.
    pub fn atomic_symbol(&self) -> &'static str {
        ATOMIC_SYMBOL
            .get(self.atomic_number as usize)
            .copied()
            .unwrap_or("?")
    }

    /// The temperatures data exists at, as the strings that key the cross
    /// sections: `"294K"` and the like.
    pub fn temperatures(&self) -> Vec<String> {
        self.k_ts
            .iter()
            .map(|&k_t| temperature_str(k_t / K_BOLTZMANN))
            .collect()
    }

    /// Whether the nuclide has a reaction with this MT.
    pub fn contains(&self, mt: i32) -> bool {
        self.reactions.contains_key(&mt)
    }

    /// One reaction, by MT.
    pub fn get(&self, mt: i32) -> Option<&Reaction> {
        self.reactions.get(&mt)
    }

    /// One reaction, by name or alias: `"elastic"`, `"(n,2n)"`, `"n,2n"`.
    pub fn get_by_name(&self, name: &str) -> Option<&Reaction> {
        let mt = crate::reaction::reaction_mt(name)
            .or_else(|| crate::reaction::reaction_mt(&format!("({name})")))?;
        self.get(mt)
    }

    /// Read a nuclide from an ENDF material.
    ///
    /// Every MT with an MF=3 cross section becomes a reaction.
    pub fn from_endf(material: &Material) -> Result<IncidentNeutron> {
        let metadata = material.mf1_mt451().ok_or(Error::Unsupported {
            what: "a neutron evaluation with no MF=1 MT=451",
        })?;
        let (z, a) = (metadata.za / 1000, metadata.za % 1000);
        let mut data = IncidentNeutron::new(z as u32, a as u32, metadata.liso as u32);

        for &(mf, mt) in material.section_data.keys() {
            if mf == 3 {
                data.reactions
                    .insert(mt, Reaction::from_endf(mt, material)?);
            }
        }
        Ok(data)
    }

    /// Read a nuclide from a continuous-energy neutron ACE table.
    pub fn from_ace(table: &Table, scheme: MetastableScheme) -> Result<IncidentNeutron> {
        if table.data_type()? != ace::TableType::NeutronContinuous {
            return Err(Error::BadAceTable {
                what: format!("{} is not a continuous-energy neutron table", table.name),
            });
        }
        let meta = ace::get_metadata(table.zaid()?, scheme)?;

        // `Table::kt` is the raw value from the file, in MeV; kTs is in eV.
        let mut data = IncidentNeutron {
            atomic_number: meta.z,
            mass_number: meta.mass_number,
            metastable: meta.metastable,
            atomic_weight_ratio: Some(table.atomic_weight_ratio),
            k_ts: vec![table.kt * EV_PER_MEV],
            ..Default::default()
        };
        let t = data.temperatures().remove(0);

        // The energy grid, and the summed cross sections stored beside it.
        let n = table.nxs[3].max(0) as usize;
        let i = table.jxs[1].max(0) as usize;
        let at = |k: usize| table.xss.get(k).copied().unwrap_or(0.0);
        let column = |c: usize| -> Vec<f64> { (0..n).map(|k| at(i + c * n + k)).collect() };

        let energy: Vec<f64> = column(0).into_iter().map(|e| e * EV_PER_MEV).collect();
        let total_xs = column(1);
        let absorption_xs = column(2);
        let heating: Vec<f64> = column(4).into_iter().map(|h| h * EV_PER_MEV).collect();
        data.energy.insert(t.clone(), energy.clone());

        // Three redundant reactions the table gives outright rather than as
        // channels: the total, the absorption, and the heating number, which
        // becomes a heating "cross section" once multiplied by the total.
        let redundant = |mt: i32, y: Vec<f64>| {
            let mut rx = Reaction::new(mt);
            let mut xs = Tabulated1D::new(energy.clone(), y);
            xs.threshold_idx = Some(0);
            rx.xs.insert(t.clone(), xs);
            rx.redundant = true;
            rx
        };
        data.reactions.insert(1, redundant(1, total_xs.clone()));
        if absorption_xs.iter().any(|&v| v != 0.0) {
            data.reactions.insert(101, redundant(101, absorption_xs));
        }
        let heating_xs: Vec<f64> = heating
            .iter()
            .zip(&total_xs)
            .map(|(&h, &s)| h * s)
            .collect();
        data.reactions.insert(301, redundant(301, heating_xs));

        for i_reaction in 0..=table.nxs[4] {
            let rx = Reaction::from_ace(table, i_reaction)?;
            data.reactions.insert(rx.mt, rx);
        }

        // Some photon production is assigned to an MT with no cross section of
        // its own, usually MT=4. A redundant reaction is built from the
        // components so the photons have somewhere to live.
        let n_photon = table.nxs[6].max(0);
        let photon_mts: BTreeSet<i32> = (0..n_photon)
            .map(|k| (table.xss[(table.jxs[13] + k) as usize] as i64 / 1000) as i32)
            .collect();
        for mt in photon_mts {
            if data.contains(mt) || sum_rule(mt).is_none() {
                // Photon production for a reaction with neither a cross
                // section nor a sum rule has nowhere to go. The Python reader
                // warns; there is nothing else to do with it.
                continue;
            }
            let mts = data.reaction_components(mt);
            if mts.is_empty() {
                continue;
            }
            let mut rx = data.redundant_reaction(mt, &mts);
            rx.products.extend(photon_products_ace(table, &rx)?);
            data.reactions.insert(mt, rx);
        }

        // An ACE file sometimes gives only the individual levels of a
        // transmutation reaction, e.g. MT=600-649 rather than MT=103. The
        // summation is built explicitly so it can be tallied.
        for mt in SUMMED_IF_ABSENT {
            if data.contains(mt) {
                continue;
            }
            let mts = data.reaction_components(mt);
            if mts.is_empty() {
                continue;
            }
            let rx = data.redundant_reaction(mt, &mts);
            data.reactions.insert(mt, rx);
        }

        // Mark what is a sum of other channels rather than a channel.
        let redundant: Vec<i32> = data
            .reactions
            .keys()
            .copied()
            .filter(|&mt| ALWAYS_REDUNDANT.contains(&mt) || data.reaction_components(mt) != [mt])
            .collect();
        for mt in redundant {
            if let Some(rx) = data.reactions.get_mut(&mt) {
                rx.redundant = true;
            }
        }

        if let Some(urr) = ProbabilityTables::from_ace(table) {
            data.urr.insert(t, urr);
        }

        Ok(data)
    }

    /// Add the cross sections of the same nuclide at another temperature.
    ///
    /// The reactions must already exist: a table that brings a reaction this
    /// one does not have is a different evaluation, and the extra reaction is
    /// dropped rather than half-added.
    pub fn add_temperature_from_ace(
        &mut self,
        table: &Table,
        scheme: MetastableScheme,
    ) -> Result<()> {
        let data = IncidentNeutron::from_ace(table, scheme)?;
        let t = data.temperatures().remove(0);
        if self.temperatures().contains(&t) {
            // Already present. The Python reader warns and returns.
            return Ok(());
        }
        if data.name() != self.name() {
            return Err(Error::Mismatched {
                what: "the nuclide of an added temperature and of the data it is added to",
            });
        }

        self.k_ts.extend(data.k_ts);
        self.energy.insert(t.clone(), data.energy[&t].clone());
        for (mt, rx) in data.reactions {
            if let Some(existing) = self.reactions.get_mut(&mt) {
                existing.xs.insert(t.clone(), rx.xs[&t].clone());
            }
        }
        if let Some(urr) = data.urr.get(&t) {
            self.urr.insert(t, urr.clone());
        }
        Ok(())
    }

    /// Which reactions make up a redundant one.
    ///
    /// The sum rules are applied recursively, and only reactions the nuclide
    /// actually has are returned. An MT that is not a sum, or whose components
    /// are all absent, gives itself back, or nothing if it is absent too.
    pub fn reaction_components(&self, mt: i32) -> Vec<i32> {
        let mut mts = Vec::new();
        if let Some(rule) = sum_rule(mt) {
            for &component in rule {
                mts.extend(self.reaction_components(component));
            }
        }
        if !mts.is_empty() {
            return mts;
        }
        if self.contains(mt) {
            vec![mt]
        } else {
            Vec::new()
        }
    }

    /// Build a redundant reaction by summing its components.
    ///
    /// The sum starts at the lowest threshold of any component, since below
    /// that every one of them is zero.
    pub fn redundant_reaction(&self, mt: i32, mts: &[i32]) -> Reaction {
        let mut rx = Reaction::new(mt);
        for t in self.temperatures() {
            let Some(grid) = self.energy.get(&t) else {
                continue;
            };
            let parts: Vec<&Tabulated1D> = mts
                .iter()
                .filter_map(|m| self.reactions.get(m).and_then(|r| r.xs.get(&t)))
                .collect();
            let idx = parts
                .iter()
                .map(|xs| xs.threshold_idx.unwrap_or(0))
                .min()
                .unwrap_or(0)
                .min(grid.len());
            let energy = grid[idx..].to_vec();
            let y: Vec<f64> = energy
                .iter()
                .map(|&e| parts.iter().map(|xs| xs.eval(e)).sum())
                .collect();
            let mut xs = Tabulated1D::new(energy, y);
            xs.threshold_idx = Some(idx);
            rx.xs.insert(t, xs);
        }
        rx.redundant = true;
        rx
    }

    /// The removal cross section: the total less the part of elastic
    /// scattering that goes forward.
    ///
    /// Point-kernel shielding treats a forward-scattered neutron as still in
    /// the uncollided beam, so it is not removed. `mu_cutoff` is the cosine of
    /// the cone's half angle; zero is the forward hemisphere.
    pub fn removal_xs(&self, temperature: &str, mu_cutoff: f64) -> Result<Tabulated1D> {
        let total =
            self.get(1)
                .and_then(|rx| rx.xs.get(temperature))
                .ok_or(Error::Unsupported {
                    what: "a removal cross section without the total cross section",
                })?;
        let elastic_rx = self.get(2).ok_or(Error::Unsupported {
            what: "a removal cross section without elastic scattering",
        })?;
        let elastic = elastic_rx.xs.get(temperature).ok_or(Error::Unsupported {
            what: "a removal cross section without elastic scattering",
        })?;

        let angle = elastic_rx
            .products
            .first()
            .and_then(|p| p.distribution.first())
            .and_then(|d| match d {
                crate::angle_energy::AngleEnergy::Uncorrelated(u) => u.angle.as_ref(),
                _ => None,
            });

        let (energies, forward) = match angle {
            Some(angle) if !angle.energy.is_empty() => {
                (angle.energy.clone(), angle.forward_fraction(mu_cutoff))
            }
            // Isotropic scattering sends the same fraction forward at every
            // energy, so the elastic grid will do.
            _ => (
                elastic.x.clone(),
                vec![(1.0 - mu_cutoff) / 2.0; elastic.x.len()],
            ),
        };

        let y: Vec<f64> = energies
            .iter()
            .zip(&forward)
            .map(|(&e, &f)| total.eval(e) - f * elastic.eval(e))
            .collect();
        Ok(Tabulated1D::new(energies, y))
    }
}

/// The fraction of an isotropic distribution that goes into the forward cone.
///
/// Only here so the meaning of the constant in [`IncidentNeutron::removal_xs`]
/// is written down somewhere: the cosine is uniform on `[-1, 1]`, so the cone
/// `[mu, 1]` holds `(1 - mu)/2` of it.
pub fn isotropic_forward_fraction(mu_cutoff: f64) -> f64 {
    (1.0 - mu_cutoff) / 2.0
}

/// Whether an angular distribution is isotropic at every energy.
pub fn is_isotropic(angle: &AngleDistribution) -> bool {
    angle
        .mu
        .iter()
        .all(|mu| matches!(mu, AngleAtEnergy::Isotropic(_)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const AM244: &[u8] = include_bytes!("../fixtures/n-095_Am_244.endf.xz");

    fn li6() -> IncidentNeutron {
        let table = crate::testdata::ace_tables(crate::testdata::LI6_ACE).remove(0);
        IncidentNeutron::from_ace(&table, MetastableScheme::Mcnp).unwrap()
    }

    #[test]
    fn an_endf_evaluation_gives_one_reaction_per_cross_section() {
        let m = Material::from_str(&crate::testdata::text(AM244)).unwrap();
        let n = IncidentNeutron::from_endf(&m).unwrap();
        assert_eq!(n.name(), "Am244");
        assert_eq!((n.atomic_number, n.mass_number, n.metastable), (95, 244, 0));
        assert_eq!(n.atomic_symbol(), "Am");

        // Exactly the MTs that have an MF=3 section, and nothing else.
        let mf3: Vec<i32> = m
            .section_data
            .keys()
            .filter(|&&(mf, _)| mf == 3)
            .map(|&(_, mt)| mt)
            .collect();
        assert_eq!(n.reactions.keys().copied().collect::<Vec<_>>(), mf3);

        // An ENDF evaluation is at one temperature and gives no shared grid.
        assert!(n.k_ts.is_empty());
        assert!(n.energy.is_empty());
        assert!(n.urr.is_empty());
    }

    #[test]
    fn reactions_are_reachable_by_name() {
        let m = Material::from_str(&crate::testdata::text(AM244)).unwrap();
        let n = IncidentNeutron::from_endf(&m).unwrap();
        assert_eq!(n.get_by_name("elastic").unwrap().mt, 2);
        assert_eq!(n.get_by_name("(n,elastic)").unwrap().mt, 2);
        // The bare form, which the lookup wraps in parentheses.
        assert_eq!(n.get_by_name("n,gamma").unwrap().mt, 102);
        assert!(n.get_by_name("(n,nonsense)").is_none());
    }

    #[test]
    fn an_ace_table_brings_its_temperature_and_its_grid() {
        let n = li6();
        assert_eq!(n.name(), "Li6");
        assert_eq!(n.temperatures(), ["294K"]);
        assert_eq!(n.k_ts.len(), 1);
        assert!(n.atomic_weight_ratio.is_some());

        // One grid, shared by every reaction of that temperature.
        let grid = &n.energy["294K"];
        assert_eq!(grid.len(), 721);
        assert!(grid.windows(2).all(|w| w[1] > w[0]), "the grid ascends");
        // Li6 has no unresolved region.
        assert!(n.urr.is_empty());
    }

    #[test]
    fn the_summed_cross_sections_come_from_the_main_energy_block() {
        let n = li6();
        // The total, and the heating, are always there; absorption only when
        // it is nonzero.
        for mt in [1, 301] {
            let rx = &n.reactions[&mt];
            assert!(rx.redundant, "MT={mt} is a sum, not a channel");
            assert_eq!(rx.xs["294K"].x.len(), n.energy["294K"].len());
        }
        // The total is at least the elastic scattering it contains.
        let total = &n.reactions[&1].xs["294K"];
        let elastic = &n.reactions[&2].xs["294K"];
        for &e in &[0.0253, 1.0e3, 1.0e6, 1.0e7] {
            assert!(total.eval(e) >= elastic.eval(e), "at {e} eV");
        }
    }

    #[test]
    fn a_reaction_the_table_gives_only_as_levels_is_summed() {
        let n = li6();
        // Li6 gives MT=51..55 and MT=91 but no MT=4, so the components of the
        // inelastic summation are the levels themselves.
        let components = n.reaction_components(4);
        assert!(!components.is_empty());
        assert!(components.iter().all(|mt| (51..=91).contains(mt)));

        // A reaction that is its own only component is not a sum.
        assert_eq!(n.reaction_components(2), [2]);
        assert!(!n.reactions[&2].redundant);

        // A reaction the nuclide does not have has no components.
        assert!(n.reaction_components(18).is_empty());
    }

    #[test]
    fn a_summed_reaction_is_the_sum_of_its_parts() {
        let n = li6();
        // MT=4 is the inelastic summation, which Li6 gives only as levels.
        let mts = n.reaction_components(4);
        assert_eq!(mts, [51, 52, 53, 54, 55, 91]);
        let rx = n.redundant_reaction(4, &mts);
        assert!(rx.redundant);
        let xs = &rx.xs["294K"];
        for &e in &[1.0e3, 1.0e6, 1.0e7] {
            let want: f64 = mts.iter().map(|m| n.reactions[m].xs["294K"].eval(e)).sum();
            assert!((xs.eval(e) - want).abs() <= 1e-12 * want.abs().max(1e-30));
        }
    }

    #[test]
    fn the_particle_production_cross_sections_are_marked_redundant() {
        let n = li6();
        for mt in ALWAYS_REDUNDANT {
            if let Some(rx) = n.get(mt) {
                assert!(rx.redundant, "MT={mt} counts other reactions");
            }
        }
    }

    #[test]
    fn adding_a_temperature_that_is_already_there_changes_nothing() {
        let table = crate::testdata::ace_tables(crate::testdata::LI6_ACE).remove(0);
        let mut n = IncidentNeutron::from_ace(&table, MetastableScheme::Mcnp).unwrap();
        let before = n.clone();
        n.add_temperature_from_ace(&table, MetastableScheme::Mcnp)
            .unwrap();
        assert_eq!(n, before);
    }

    #[test]
    fn the_removal_cross_section_takes_the_forward_cone_off_the_total() {
        let m = Material::from_str(&crate::testdata::text(AM244)).unwrap();
        let n = IncidentNeutron::from_endf(&m).unwrap();

        let total = &n.get(1).unwrap().xs["0K"];
        let elastic = &n.get(2).unwrap().xs["0K"];

        // At a cutoff of -1 the whole of elastic scattering is forward, so
        // removal is the total less all of it.
        let all = n.removal_xs("0K", -1.0).unwrap();
        for (i, &e) in all.x.iter().enumerate() {
            let want = total.eval(e) - elastic.eval(e);
            assert!((all.y[i] - want).abs() <= 1e-9 * want.abs().max(1e-30));
        }

        // Raising the cutoff can only take less off, so removal rises.
        let half = n.removal_xs("0K", 0.0).unwrap();
        let most = n.removal_xs("0K", 0.5).unwrap();
        for (i, &e) in all.x.iter().enumerate() {
            assert!(all.y[i] <= half.y[i] + 1e-9, "at {e} eV");
            assert!(half.y[i] <= most.y[i] + 1e-9, "at {e} eV");
            assert!(most.y[i] <= total.eval(e) + 1e-9, "at {e} eV");
        }
    }

    #[test]
    fn a_removal_cross_section_needs_the_total_and_the_elastic() {
        let n = IncidentNeutron::new(3, 6, 0);
        assert!(n.removal_xs("0K", 0.0).is_err());
    }
}
