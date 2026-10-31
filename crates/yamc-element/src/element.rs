// Provides functionality for working with natural elements and their isotopic abundances
use yamc_nuclide::data::ELEMENT_NUCLIDES;

/// Extract the element symbol from a nuclide name.
///
/// Strips trailing digits and metastable suffixes to get the element symbol.
/// Examples: "Fe56" -> "Fe", "Li6" -> "Li", "Am241m" -> "Am", "H1" -> "H"
pub fn element_symbol_from_nuclide(name: &str) -> String {
    name.chars().take_while(|c| c.is_alphabetic()).collect()
}

/// Represents a chemical element identified by its symbol (e.g. `"Fe"`).
///
/// Provides helper methods to enumerate naturally occurring isotopes (nuclides)
/// using the internally defined natural abundance database.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Element {
    /// Chemical symbol of the element (case sensitive, e.g. "Fe").
    pub name: String,
}

impl Element {
    pub fn new<S: Into<String>>(name: S) -> Self {
        Self { name: name.into() }
    }

    /// Return the list of nuclide (isotope) names (e.g. ["Fe54", "Fe56", ...]) for this element.
    pub fn get_nuclides(&self) -> Vec<String> {
        ELEMENT_NUCLIDES
            .get(self.name.as_str())
            .map(|v| v.iter().map(|s| s.to_string()).collect())
            .unwrap_or_default()
    }
}
