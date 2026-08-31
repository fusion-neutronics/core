// Global configuration for the materials library
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Mutex;

// Global configuration for nuclear data file paths
pub static CONFIG: Lazy<Mutex<Config>> = Lazy::new(|| Mutex::new(Config::new()));

/// Check if a value is a keyword or a directory (both bypass file-existence checks).
fn is_keyword_or_directory(value: &str) -> bool {
    crate::url_cache::is_keyword(value) || std::path::Path::new(value).is_dir()
}

/// Global configuration container for the nuclear data library.
///
/// The configuration is primarily a mapping from nuclide names (e.g. "Li6")
/// to the file system path of the Arrow IPC directory that stores the reaction /
/// energy data for that nuclide. Helper methods are provided to set a single path,
/// bulk insert many paths, or query the mapping.
///
/// A single global instance is exposed via the `CONFIG` static (a
/// `Lazy<Mutex<Config>>`). Most code should obtain a guard with
/// [`Config::global`] rather than accessing the mutex directly to keep usage
/// consistent and centralized.
/// Default library for the decay / reactions / fission-yield transmutation
/// subsections when the user has not set them explicitly.
pub const DEFAULT_TRANSMUTATION_LIBRARY: &str = "endf-b8.1";

/// Where one optional transmutation subsection comes from.
///
/// Three states, because "the user has not chosen" and "the user chose not to
/// have it" are different answers and only one of them is an error when the
/// subsection turns out to be needed. `Option<String>` cannot hold both: it
/// spelled them the same way, so `transmutation_fission_yields = None` read as
/// off and behaved as endf-b8.1.
///
/// Only the parts a calculation can do without are modelled this way. The
/// decay subsection is not one of them: half-lives and decay energies come
/// from it and a chain without it is empty, so it keeps a plain
/// `Option<String>` that falls back to the default library.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum SubsectionSource {
    /// Not set. The default library supplies it, as it always has.
    #[default]
    Default,
    /// Deliberately not loaded. Nothing is downloaded, and a calculation that
    /// turns out to need it fails with an error naming what it needed.
    Off,
    /// A library keyword or a path to the subsection directory.
    Set(String),
}

impl SubsectionSource {
    /// The source to resolve, or `None` when the subsection is not to be loaded.
    pub fn resolve(&self) -> Option<String> {
        match self {
            Self::Default => Some(DEFAULT_TRANSMUTATION_LIBRARY.to_string()),
            Self::Off => None,
            Self::Set(source) => Some(source.clone()),
        }
    }

    /// Whether the user turned this subsection off, as opposed to leaving it alone.
    pub fn is_off(&self) -> bool {
        matches!(self, Self::Off)
    }
}

impl From<Option<&str>> for SubsectionSource {
    /// `None` is "back to the default", matching every other optional source
    /// on [`Config`]. [`Self::Off`] has no spelling here on purpose: turning a
    /// subsection off is a deliberate act and the caller has to name it.
    fn from(value: Option<&str>) -> Self {
        match value {
            None => Self::Default,
            Some(source) => Self::Set(source.to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    /// Map of nuclide name -> absolute or relative path to its Arrow data directory.
    pub cross_sections: HashMap<String, String>,
    /// Optional global default cross section source keyword
    pub default_cross_section: Option<String>,
    /// Source (library keyword or path to a `decay/` subsection) for the decay
    /// part of the transmutation network. `None` means the default library.
    pub transmutation_decay_data: Option<String>,
    /// Source for the reaction-topology part (`reactions/` subsection), or
    /// [`SubsectionSource::Off`] for a decay-only calculation.
    pub transmutation_reactions: SubsectionSource,
    /// Source for the fission-yields part (`fission_yields/` subsection), or
    /// [`SubsectionSource::Off`] for a material nothing in which fissions.
    pub transmutation_fission_yields: SubsectionSource,
    /// Source for the isomeric branching overlay (`branching/` subsection).
    /// `None` means no branching overlay is applied.
    pub transmutation_branch_ratios: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

impl Config {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Config {
            cross_sections: HashMap::new(),
            default_cross_section: None,
            transmutation_decay_data: None,
            transmutation_reactions: SubsectionSource::Default,
            transmutation_fission_yields: SubsectionSource::Default,
            transmutation_branch_ratios: None,
        }
    }

    /// Set a cross section file path for a nuclide, or set a global default if only a keyword/directory is provided
    pub fn set_cross_section(&mut self, nuclide_or_keyword: &str, path: Option<&str>) {
        match path {
            Some(p) => {
                // Validate file exists when a path is provided (keywords and directories bypass)
                if !is_keyword_or_directory(p) && !std::path::Path::new(p).exists() {
                    panic!(
                        "Cross section file for '{nuclide_or_keyword}' does not exist at path: {p}"
                    );
                }
                self.cross_sections
                    .insert(nuclide_or_keyword.to_string(), p.to_string());
            }
            None => {
                // No path: value must be a keyword or directory to set as global default
                if !is_keyword_or_directory(nuclide_or_keyword) {
                    panic!(
                        "Invalid cross section source: '{}'. Must be a keyword ({}) or a directory path.",
                        nuclide_or_keyword,
                        "tendl-2025, fendl-3.2d, endf-b8.1"
                    );
                }
                self.default_cross_section = Some(nuclide_or_keyword.to_string());
            }
        }
    }

    /// Get a cross section file path for a nuclide, falling back to the global default if not set
    pub fn get_cross_section(&self, nuclide: &str) -> Option<String> {
        self.cross_sections
            .get(nuclide)
            .cloned()
            .or_else(|| self.default_cross_section.clone())
    }

    /// Set multiple cross section file paths at once, or set a global keyword/directory
    pub fn set_cross_sections<T>(&mut self, input: T)
    where
        T: IntoCrossSectionsInput,
    {
        input.apply(self);
    }

    /// Effective source for the decay subsection (setting or default library).
    pub fn get_transmutation_decay_data(&self) -> String {
        self.transmutation_decay_data
            .clone()
            .unwrap_or_else(|| DEFAULT_TRANSMUTATION_LIBRARY.to_string())
    }

    /// Effective source for the reactions subsection, or `None` when it is off.
    pub fn get_transmutation_reactions(&self) -> Option<String> {
        self.transmutation_reactions.resolve()
    }

    /// Effective source for the fission-yields subsection, or `None` when off.
    pub fn get_transmutation_fission_yields(&self) -> Option<String> {
        self.transmutation_fission_yields.resolve()
    }

    /// Effective source for the branching overlay, or `None` if unset (no
    /// branching overlay is applied).
    pub fn get_transmutation_branch_ratios(&self) -> Option<String> {
        self.transmutation_branch_ratios.clone()
    }

    /// Clear all cross section mappings, default, and transmutation settings.
    pub fn clear(&mut self) {
        self.cross_sections.clear();
        self.default_cross_section = None;
        self.transmutation_decay_data = None;
        self.transmutation_reactions = SubsectionSource::Default;
        self.transmutation_fission_yields = SubsectionSource::Default;
        self.transmutation_branch_ratios = None;
    }
}

/// Trait to allow flexible input types for set_cross_sections
pub trait IntoCrossSectionsInput {
    fn apply(self, config: &mut Config);
}

impl IntoCrossSectionsInput for HashMap<String, String> {
    fn apply(self, config: &mut Config) {
        for (nuclide, path) in self {
            if crate::url_cache::is_keyword(&path) {
                // If value is a keyword, set as global default
                config.default_cross_section = Some(path.clone());
            } else if !std::path::Path::new(&path).is_dir() && !std::path::Path::new(&path).exists()
            {
                // Validate file exists (keywords and directories bypass)
                panic!("Cross section file for '{nuclide}' does not exist at path: {path}");
            }
            config.cross_sections.insert(nuclide, path);
        }
    }
}

impl IntoCrossSectionsInput for &str {
    fn apply(self, config: &mut Config) {
        if !is_keyword_or_directory(self) {
            panic!(
                "Invalid cross section source: '{}'. Must be a keyword (tendl-2025, fendl-3.2d, endf-b8.1) or a directory path.",
                self,
            );
        }
        config.default_cross_section = Some(self.to_string());
    }
}

impl IntoCrossSectionsInput for String {
    fn apply(self, config: &mut Config) {
        IntoCrossSectionsInput::apply(self.as_str(), config);
    }
}

impl Config {
    /// Get the global configuration instance
    pub fn global() -> std::sync::MutexGuard<'static, Self> {
        CONFIG
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

#[cfg(test)]
mod subsection_source_tests {
    //! The three states have to stay distinguishable, because the whole point
    //! of the enum is that `Option<String>` could not tell two of them apart.

    use super::*;

    #[test]
    fn untouched_resolves_to_the_default_library() {
        assert_eq!(
            SubsectionSource::Default.resolve().as_deref(),
            Some(DEFAULT_TRANSMUTATION_LIBRARY)
        );
    }

    #[test]
    fn off_resolves_to_nothing_at_all() {
        assert_eq!(SubsectionSource::Off.resolve(), None);
        assert!(SubsectionSource::Off.is_off());
    }

    #[test]
    fn a_set_source_is_returned_as_given() {
        let source = SubsectionSource::Set("tendl-2025".to_string());
        assert_eq!(source.resolve().as_deref(), Some("tendl-2025"));
        assert!(!source.is_off());
    }

    /// `None` resets to the default; it is not a way to spell "off". The two
    /// were the same value once, and a chain silently lost its reactions
    /// whenever a caller cleared the setting.
    #[test]
    fn none_resets_to_the_default() {
        assert_eq!(SubsectionSource::from(None), SubsectionSource::Default);
        assert_eq!(
            SubsectionSource::from(None).resolve().as_deref(),
            Some(DEFAULT_TRANSMUTATION_LIBRARY)
        );
        assert_eq!(
            SubsectionSource::from(Some("jeff-4.0")),
            SubsectionSource::Set("jeff-4.0".to_string())
        );
    }

    /// A fresh config still supplies both, so nothing that never touched these
    /// settings sees a change.
    #[test]
    fn a_new_config_supplies_both_subsections() {
        let config = Config::new();
        assert_eq!(
            config.get_transmutation_reactions().as_deref(),
            Some(DEFAULT_TRANSMUTATION_LIBRARY)
        );
        assert_eq!(
            config.get_transmutation_fission_yields().as_deref(),
            Some(DEFAULT_TRANSMUTATION_LIBRARY)
        );
    }

    #[test]
    fn clear_puts_both_back_to_the_default() {
        let mut config = Config::new();
        config.transmutation_reactions = SubsectionSource::Off;
        config.transmutation_fission_yields = SubsectionSource::Off;
        config.clear();
        assert!(!config.transmutation_reactions.is_off());
        assert!(!config.transmutation_fission_yields.is_off());
    }
}
