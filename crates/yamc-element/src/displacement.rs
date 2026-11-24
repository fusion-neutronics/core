//! Displacement threshold energies and the NRT displacement model.
//!
//! Both the transport-free path (a damage-energy cross section, MT=444, folded
//! against a spectrum) and the transport path (the damage-energy tally) end in
//! the same conversion from deposited damage energy to displacements per atom.
//! It lives here, beside the other per-element properties, so the two cannot
//! drift apart on the model or on the threshold energies it uses.
//!
//! # The model
//!
//! The Norgett-Robinson-Torrens (NRT) model counts `0.8 * T_d / (2 * E_d)`
//! stable displacements for a primary knock-on atom whose damage energy
//! `T_d` exceeds `2 * E_d / 0.8`, one between `E_d` and that, and none below
//! `E_d` (ASTM E521; M. J. Norgett, M. T. Robinson, I. M. Torrens, Nucl. Eng.
//! Des. 33 (1975) 50). [`nrt_dpa`] applies the linear branch only. That is the
//! standard practice when the input is damage energy already integrated over
//! the recoil spectrum, as MT=444 and a damage-energy tally both are: the
//! per-recoil threshold steps cannot be applied to a sum over recoils, and
//! the recoils they would change carry a negligible share of the damage energy
//! in a fast spectrum.
//!
//! # The threshold energies
//!
//! `E_d` is the spatially averaged displacement threshold energy. The defaults
//! in [`default_displacement_energy`] are the ASTM E521 recommended values
//! where that standard gives one, and otherwise the average displacement
//! energies compiled in Table 2.4 of the OECD-NEA report "Primary Radiation
//! Damage in Materials" (NEA/NSC/DOC(2015)9, 2015). Both are as tabulated in
//! that report's Table 2.4, which reproduces the ASTM E521 values beside the
//! measured ones. An element in neither has no default: published values for
//! it spread too widely for one to be picked silently, so a caller has to
//! supply it.

use std::fmt;

/// The fraction of the damage energy the NRT model counts as producing stable
/// displacements, the `0.8` in `0.8 * T_d / (2 * E_d)`.
pub const NRT_EFFICIENCY: f64 = 0.8;

/// Where a displacement threshold energy came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DisplacementEnergySource {
    /// The ASTM E521 recommended value.
    AstmE521,
    /// The OECD-NEA 2015 report's Table 2.4, for an element ASTM E521 does not
    /// cover.
    OecdNea2015,
    /// Supplied by the user.
    User,
}

impl DisplacementEnergySource {
    /// The label a report carries: `"ASTM E521"`, `"OECD-NEA 2015"` or
    /// `"user"`.
    pub fn label(self) -> &'static str {
        match self {
            Self::AstmE521 => "ASTM E521",
            Self::OecdNea2015 => "OECD-NEA 2015",
            Self::User => "user",
        }
    }
}

impl fmt::Display for DisplacementEnergySource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// One element's displacement threshold energy and where it came from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DisplacementEnergy {
    /// The spatially averaged displacement threshold energy `E_d` [eV].
    pub energy_ev: f64,
    /// Where the value came from.
    pub source: DisplacementEnergySource,
}

use DisplacementEnergySource::{AstmE521, OecdNea2015};

/// The default table, `(symbol, E_d [eV], source)`.
///
/// All values are from NEA/NSC/DOC(2015)9 Table 2.4, "Average Ed" column.
/// Those it attributes to ASTM E521 are marked so; the rest are the values
/// that table gives for elements ASTM E521 does not cover, with the original
/// measurement or compilation it cites noted per entry.
const DEFAULTS: &[(&str, f64, DisplacementEnergySource)] = &[
    // Graphite, 30 eV [Zinkle, 1997]. The table also lists 40 eV for diamond;
    // graphite is the form carbon takes in the materials this is run on.
    ("C", 30.0, OecdNea2015),
    // [Lucasson, 1975].
    ("Mg", 20.0, OecdNea2015),
    ("Al", 25.0, AstmE521),
    ("Ti", 30.0, AstmE521),
    ("V", 40.0, AstmE521),
    ("Cr", 40.0, AstmE521),
    ("Mn", 40.0, AstmE521),
    ("Fe", 40.0, AstmE521),
    ("Co", 40.0, AstmE521),
    ("Ni", 40.0, AstmE521),
    ("Cu", 30.0, AstmE521),
    // [Lucasson, 1975].
    ("Zn", 29.0, OecdNea2015),
    ("Zr", 40.0, AstmE521),
    ("Nb", 60.0, AstmE521),
    ("Mo", 60.0, AstmE521),
    // [Lucasson, 1975].
    ("Pd", 41.0, OecdNea2015),
    // [Lucasson, 1975].
    ("Ag", 39.0, OecdNea2015),
    // [Lucasson, 1975].
    ("Cd", 30.0, OecdNea2015),
    ("Ta", 90.0, AstmE521),
    ("W", 90.0, AstmE521),
    // [Lucasson, 1975].
    ("Re", 60.0, OecdNea2015),
    // [Lucasson, 1975].
    ("Pt", 44.0, OecdNea2015),
    // [Lucasson, 1975]. The table also lists 40 eV [Vajda, 1977].
    ("Au", 43.0, OecdNea2015),
    ("Pb", 25.0, AstmE521),
    // [Lucasson, 1975].
    ("Th", 44.0, OecdNea2015),
];

/// The default displacement threshold energy for an element, by symbol
/// (`"Fe"`, case sensitive), or `None` where neither ASTM E521 nor the OECD-NEA
/// 2015 report gives one.
///
/// See the module documentation for the sources. `None` is deliberate: it
/// means the caller must supply the value, not that a generic one applies.
pub fn default_displacement_energy(symbol: &str) -> Option<DisplacementEnergy> {
    DEFAULTS
        .iter()
        .find(|(s, _, _)| *s == symbol)
        .map(|&(_, energy_ev, source)| DisplacementEnergy { energy_ev, source })
}

/// Every element with a default displacement threshold energy, in the order
/// of the table (ascending atomic number).
pub fn default_displacement_energies() -> impl Iterator<Item = (&'static str, DisplacementEnergy)> {
    DEFAULTS
        .iter()
        .map(|&(symbol, energy_ev, source)| (symbol, DisplacementEnergy { energy_ev, source }))
}

/// Check a user-supplied displacement threshold energy: a known element symbol
/// and a positive, finite energy in eV.
pub fn check_displacement_energy(symbol: &str, energy_ev: f64) -> Result<(), String> {
    if !yamc_nuclide::data::ELEMENT_NAMES.contains_key(symbol) {
        return Err(format!(
            "'{symbol}' is not an element symbol; displacement energies are keyed by \
             symbol, case sensitive (e.g. 'Fe', 'W')"
        ));
    }
    if !(energy_ev.is_finite() && energy_ev > 0.0) {
        return Err(format!(
            "the displacement energy for {symbol} is {energy_ev} eV; it must be positive \
             and finite"
        ));
    }
    Ok(())
}

/// NRT displacements per atom from a damage energy per atom.
///
/// `dpa = 0.8 * damage_energy_ev / (2 * displacement_energy_ev)`, the linear
/// branch of the NRT model (see the module documentation for why the
/// per-recoil threshold steps are not applied to an integrated damage energy).
/// Both arguments in eV; the damage energy is per atom of the element whose
/// `E_d` is given.
pub fn nrt_dpa(damage_energy_ev: f64, displacement_energy_ev: f64) -> f64 {
    NRT_EFFICIENCY * damage_energy_ev / (2.0 * displacement_energy_ev)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tungsten_and_iron_are_the_astm_values() {
        let w = default_displacement_energy("W").unwrap();
        assert_eq!(w.energy_ev, 90.0);
        assert_eq!(w.source, DisplacementEnergySource::AstmE521);
        let fe = default_displacement_energy("Fe").unwrap();
        assert_eq!(fe.energy_ev, 40.0);
        assert_eq!(fe.source.label(), "ASTM E521");
        assert_eq!(
            default_displacement_energy("Re").unwrap().source.label(),
            "OECD-NEA 2015"
        );
    }

    #[test]
    fn an_element_in_neither_source_has_no_default() {
        assert!(default_displacement_energy("Li").is_none());
        assert!(default_displacement_energy("fe").is_none());
    }

    #[test]
    fn every_default_is_a_known_element_once() {
        let mut seen = std::collections::HashSet::new();
        for (symbol, e) in default_displacement_energies() {
            check_displacement_energy(symbol, e.energy_ev).unwrap();
            assert!(seen.insert(symbol), "{symbol} listed twice");
        }
    }

    #[test]
    fn nrt_is_the_linear_branch() {
        // The issue's hand check: 2.05e-9 eV/s per atom of W at 90 eV.
        let rate = nrt_dpa(2.05e-9, 90.0);
        assert!((rate - 9.111e-12).abs() < 1e-14, "{rate}");
        assert_eq!(nrt_dpa(225.0, 90.0), 1.0);
    }

    #[test]
    fn overrides_are_checked() {
        assert!(check_displacement_energy("Fe", 40.0).is_ok());
        assert!(check_displacement_energy("Xx", 40.0).is_err());
        assert!(check_displacement_energy("Fe", 0.0).is_err());
        assert!(check_displacement_energy("Fe", -1.0).is_err());
        assert!(check_displacement_energy("Fe", f64::NAN).is_err());
        assert!(check_displacement_energy("Fe", f64::INFINITY).is_err());
    }
}
