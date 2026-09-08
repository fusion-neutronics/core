//! What final states an evaluation says a reaction can leave its product in.
//!
//! A reaction that can leave its product in a metastable state has to say so
//! in MF=8, one subsection per final state, with each state's cross section in
//! MF=10 or its share of the reaction in MF=9. An evaluation that lists only
//! the ground state is not merely less accurate: the isomer is absent from any
//! network built from it, so no code can produce it, and a measurement that
//! sees its decay heat cannot be reproduced by any means. TENDL-2017 omitting
//! the 1706 keV state from `Os190(n,n')` is why both yani and FISPACT-II sit
//! at a third of the measured heat on the FNS osmium foil, and it is invisible
//! from the outside: the reaction is present, its cross section is reasonable,
//! and only the state list is short.
//!
//! [`extract_production`] reports that state list so it can be compared across
//! libraries. It is a read, not a conversion: nothing is written, no decay data
//! is needed, and the answer is what the file says rather than what a network
//! made of it would contain.
//!
//! Why this is not [`crate::branching`]. That module answers a different
//! question, how a reaction's rate splits between the isomers of the product,
//! which needs decay data to turn a level index into an isomeric state and
//! silently drops any state it cannot resolve. Asking which states exist has
//! to happen before that mapping, or a state the decay library does not know
//! about disappears from the answer, and those are exactly the states worth
//! finding.

use std::error::Error;
use std::path::Path;

use endf::Material;

use crate::branching::mt_to_type;

/// One final state of one reaction, as the evaluation lists it.
#[derive(Debug, Clone, PartialEq)]
pub struct ProductionState {
    /// Excitation energy above the product's ground state, in eV. From MF=8
    /// where the evaluation gives it, otherwise `QM - QI`.
    pub excitation_energy: f64,
    /// The product's nuclear level index, LFS. Zero is the ground state.
    ///
    /// Not an isomeric-state ordinal: two evaluators number the levels of one
    /// nuclide differently, so `Ir190`'s 377 keV isomer is level 3 in
    /// ENDF/B-VIII.1 and level 37 in JEFF-4.0 and TENDL-2025. The energy is
    /// what compares across libraries; this is for finding the record again.
    pub level_index: i64,
    /// The product's ground-state name, e.g. `"Ir190"`.
    ///
    /// The ground state even for an excited final state, because naming the
    /// isomer would need decay data to say which isomeric ordinal this level
    /// is, and this function deliberately reads no decay data. Pair the name
    /// with `excitation_energy` to identify the state.
    pub product: String,
    /// `"cross_section"` for a state given in MF=10, `"yield"` for MF=9.
    pub source: &'static str,
}

/// The final states one reaction of one parent can leave its product in.
#[derive(Debug, Clone, PartialEq)]
pub struct ProductionChannel {
    /// The evaluated nuclide, named as it names itself in MF=1 MT=451.
    pub parent: String,
    /// The reaction's MT number, which is what the format states.
    pub mt: i32,
    /// The transmutation reaction name, where yani has one for this MT.
    /// `None` for an MT no chain reaction covers, which is reported rather
    /// than dropped: an isomer reached only by an unnamed channel is still a
    /// state the evaluation carries.
    pub reaction: Option<String>,
    /// Every final state, ground state included, in level order.
    pub states: Vec<ProductionState>,
}

impl ProductionChannel {
    /// The excited states, which is what a completeness comparison is about.
    pub fn excited(&self) -> impl Iterator<Item = &ProductionState> {
        self.states.iter().filter(|s| s.excitation_energy > 0.0)
    }
}

/// Read one evaluation's radionuclide production.
///
/// Empty for an evaluation with no MF=9 or MF=10 at all, which is the common
/// case: most reactions cannot leave their product excited, and an evaluation
/// that says nothing is different from one that lists the ground state alone.
/// The second is a statement and appears here with a single state; the first
/// is a silence and does not appear.
pub fn extract_production(material: &Material) -> Vec<ProductionChannel> {
    let Some(meta) = material.mf1_mt451() else {
        return Vec::new();
    };
    let parent = endf::gnds_name(
        (meta.za / 1000) as u32,
        (meta.za % 1000) as u32,
        meta.liso as u32,
    );
    let names = mt_to_type();
    let mut out = Vec::new();
    for (mt, states) in endf::radionuclide_production::radionuclide_production(material) {
        let mut listed: Vec<ProductionState> = states
            .iter()
            .map(|state| ProductionState {
                excitation_energy: state.excitation_energy(),
                level_index: state.lfs,
                product: endf::gnds_name((state.zap / 1000) as u32, (state.zap % 1000) as u32, 0),
                source: if state.cross_section.is_some() {
                    "cross_section"
                } else {
                    "yield"
                },
            })
            .collect();
        listed.sort_by_key(|s| s.level_index);
        out.push(ProductionChannel {
            parent: parent.clone(),
            mt,
            reaction: names.get(&(mt as i64)).cloned(),
            states: listed,
        });
    }
    out
}

/// Read the radionuclide production of many evaluations, one at a time.
///
/// Streamed rather than parsed into a held set, for the reason issue #53 gives:
/// a whole neutron sublibrary does not fit in memory, and a survey is the case
/// that wants every file rather than a scoped few. Channels come back in file
/// order, so the answer does not depend on how the work was scheduled.
pub fn production_from_files<P: AsRef<Path>>(
    paths: &[P],
) -> Result<Vec<ProductionChannel>, Box<dyn Error>> {
    let mut out = Vec::new();
    for path in paths {
        let display = path.as_ref().display().to_string();
        let material = Material::from_file(path.as_ref())
            .map_err(|e| -> Box<dyn Error> { format!("{display}: {e}").into() })?;
        out.extend(extract_production(&material));
    }
    Ok(out)
}
