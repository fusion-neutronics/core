//! Curated material collections.
//!
//! A collection is a read-only, name-keyed table of published material
//! compositions that resolves to an ordinary [`Material`]. Everything
//! downstream (transport, transmutation, mixing) is therefore unchanged: a
//! collection is a lookup, not a new material type.
//!
//! [`pnnl`] is the PNNL Compendium of Material Composition Data for Radiation
//! Transport Modeling (PNNL-15870, Rev. 2), bundled in the binary.

pub mod pnnl;

use crate::material::Material;

/// One entry in a curated collection.
///
/// `composition` keys are element symbols (`"Fe"`) or nuclide names
/// (`"U235"`), and the values are atom fractions. Element symbols are
/// expanded with yamc's own natural abundances when the entry is turned into
/// a [`Material`]; nuclide names are used as given.
#[derive(Debug, Clone, PartialEq)]
pub struct CollectionEntry {
    /// Name exactly as printed in the source document.
    pub name: &'static str,
    /// The source document's own entry number, for citation.
    pub number: u32,
    /// Mass density in g/cm3.
    pub density: f64,
    /// Chemical formula where the source states one.
    pub formula: Option<&'static str>,
    /// `(element symbol or nuclide name, atom fraction)`, in source order.
    pub composition: Vec<(&'static str, f64)>,
}

impl CollectionEntry {
    /// Build a [`Material`] from this entry.
    ///
    /// The material carries the entry's name and density; everything else
    /// (id, volume, temperature, transmutable) is left at the `Material`
    /// default for the caller to set.
    pub fn to_material(&self) -> Result<Material, String> {
        use std::collections::HashMap;
        use yamc_nuclide::composition;

        let mut nuclides: HashMap<String, f64> = HashMap::new();
        let mut order: Vec<String> = Vec::new();
        for (key, fraction) in &self.composition {
            // A key with no digits is an element symbol; expand it with
            // yamc's abundances so the collection never pins the source
            // document's abundance vintage. Anything else is a nuclide.
            let expanded = if key.chars().any(|c| c.is_ascii_digit()) {
                composition::validate_nuclide_name(key)?;
                HashMap::from([(key.to_string(), *fraction)])
            } else {
                composition::expand_element(key, *fraction, "atom")?
            };
            let mut names: Vec<&String> = expanded.keys().collect();
            names.sort();
            for n in names {
                if !order.contains(n) {
                    order.push(n.clone());
                }
            }
            composition::merge_nuclides(&mut nuclides, &expanded);
        }

        let mut material = Material::new(nuclides, "atom", "g/cm3", Some(self.density))?;
        material.nuclide_input_order = Some(order);
        material.name = Some(self.name.to_string());
        Ok(material)
    }
}

/// Parse the bundled pipe-delimited table format.
///
/// `number|name|density|formula|<key> <fraction> <key> <fraction> ...`
/// with `#` comment lines. Panics on a malformed line: the input is a
/// generated file compiled into the binary, so a failure here is a build
/// defect, not user input.
fn parse_table(data: &'static str) -> Vec<CollectionEntry> {
    data.lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|line| {
            let mut fields = line.splitn(5, '|');
            let mut next = |what: &str| {
                fields
                    .next()
                    .unwrap_or_else(|| panic!("collection row missing {what}: {line}"))
            };
            let number = next("number")
                .parse()
                .unwrap_or_else(|e| panic!("bad number in {line}: {e}"));
            let name = next("name");
            let density = next("density")
                .parse()
                .unwrap_or_else(|e| panic!("bad density in {line}: {e}"));
            let formula = next("formula");
            let mut parts = next("composition").split_whitespace();
            let mut composition = Vec::new();
            while let Some(key) = parts.next() {
                let value = parts
                    .next()
                    .unwrap_or_else(|| panic!("odd composition field count in {line}"));
                composition.push((
                    key,
                    value
                        .parse()
                        .unwrap_or_else(|e| panic!("bad fraction in {line}: {e}")),
                ));
            }
            CollectionEntry {
                name,
                number,
                density,
                formula: (!formula.is_empty()).then_some(formula),
                composition,
            }
        })
        .collect()
}
