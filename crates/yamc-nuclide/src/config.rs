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

/// Refuse a branching-only library keyword (`irdff-ii`) as a cross-section
/// source, as the other invalid sources here are refused.
fn refuse_branching_only(value: &str) {
    if crate::url_cache::is_branching_only(value) {
        panic!(
            "{}",
            crate::url_cache::branching_only_error(value, "the neutron cross-section library")
        );
    }
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

/// The isomeric-branching sources, in the shape they were given.
///
/// One source on its own and a list of one name the same source, but each
/// reads back as it was written: a setting shows what was assigned, so reading
/// it and assigning the result again changes nothing.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum BranchSources {
    /// Nothing set: no branching overlay.
    #[default]
    Unset,
    /// One source, given on its own.
    One(String),
    /// Sources given as a list, in the order a channel takes its split from
    /// them. An empty list is no overlay.
    List(Vec<String>),
}

impl BranchSources {
    /// The sources in order, whatever shape they were given in.
    pub fn sources(&self) -> Vec<String> {
        match self {
            Self::Unset => Vec::new(),
            Self::One(source) => vec![source.clone()],
            Self::List(sources) => sources.clone(),
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
    /// Sources for the isomeric branching overlay (`branching/` subsections),
    /// in the order a `(parent, reaction)` channel takes its split from them:
    /// the first source that evaluates the channel supplies all of it. Each is
    /// a library keyword or a path. None at all means no branching overlay is
    /// applied.
    pub transmutation_branch_ratios: BranchSources,
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
            transmutation_branch_ratios: BranchSources::Unset,
        }
    }

    /// Set a cross section file path for a nuclide, or set a global default if only a keyword/directory is provided
    pub fn set_cross_section(&mut self, nuclide_or_keyword: &str, path: Option<&str>) {
        refuse_branching_only(path.unwrap_or(nuclide_or_keyword));
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
                        nuclide_or_keyword, "tendl-2025, fendl-3.2d, endf-b8.1"
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

    /// Why `nuclides` have no neutron cross-section source, saying how to give
    /// them one.
    ///
    /// There is no default neutron library: unlike the photon data
    /// (`url_cache::DEFAULT_PHOTON_LIBRARY`) and the transmutation chain
    /// ([`DEFAULT_TRANSMUTATION_LIBRARY`]), which library the neutrons see is
    /// the calculation's central choice, so it is made by the caller and a
    /// calculation that needs it and finds none stops here.
    pub fn missing_cross_sections(&self, nuclides: &[&str]) -> String {
        let named = nuclides
            .iter()
            .map(|n| format!("'{n}'"))
            .collect::<Vec<_>>()
            .join(", ");
        if self.default_cross_section.is_none() && self.cross_sections.is_empty() {
            format!(
                "no neutron cross-section library is configured, so there is no data for \
                 {named}. Set one with `cross_section_data`, e.g. \
                 `yamc.cross_section_data = \"endf-b8.1\"` (or `yani.`): a library keyword \
                 ({}), a local release directory, or a {{nuclide: path}} dict. Or load \
                 the material's data yourself with `Material.read_nuclear_data`.",
                crate::url_cache::cross_section_keywords().join(", "),
            )
        } else {
            format!(
                "no neutron cross sections for {named}: `cross_section_data` maps {} \
                 nuclide(s) and names no library for the rest. Add them to the mapping, or \
                 set a library keyword or local release directory as the default.",
                self.cross_sections.len(),
            )
        }
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

    /// The branching overlay's sources, in order; empty when none is set (no
    /// branching overlay is applied). [`Config::transmutation_branch_ratios`]
    /// keeps the shape they were set in.
    pub fn get_transmutation_branch_ratios(&self) -> Vec<String> {
        self.transmutation_branch_ratios.sources()
    }

    /// Set the branching overlay's sources, in the order a channel takes its
    /// split from them, in the shape they were given. No source turns the
    /// overlay off.
    ///
    /// Refuses an empty name and a source given twice: the second entry of a
    /// source could never supply a channel, so listing it again is a mistake
    /// in the order rather than a request.
    pub fn set_transmutation_branch_ratios(
        &mut self,
        setting: BranchSources,
    ) -> Result<(), String> {
        let sources = setting.sources();
        for (i, source) in sources.iter().enumerate() {
            if source.trim().is_empty() {
                return Err(format!(
                    "transmutation_branch_ratios entry {i} is empty: each source is a library \
                     keyword or a path"
                ));
            }
            if sources[..i].contains(source) {
                return Err(format!(
                    "'{source}' is listed twice in transmutation_branch_ratios: a channel takes \
                     its split from the first source that evaluates it, so a second entry of \
                     the same source could never supply one"
                ));
            }
        }
        self.transmutation_branch_ratios = setting;
        Ok(())
    }

    /// Clear all cross section mappings, default, and transmutation settings.
    pub fn clear(&mut self) {
        self.cross_sections.clear();
        self.default_cross_section = None;
        self.transmutation_decay_data = None;
        self.transmutation_reactions = SubsectionSource::Default;
        self.transmutation_fission_yields = SubsectionSource::Default;
        self.transmutation_branch_ratios = BranchSources::Unset;
    }
}

/// [`Config::missing_cross_sections`] for one nuclide, against the global
/// configuration.
///
/// Called where a load finds no source, which can be inside code already
/// holding [`CONFIG`]; it then answers without the configuration rather than
/// waiting on the lock.
pub fn missing_cross_section_message(nuclide: &str) -> String {
    match CONFIG.try_lock() {
        Ok(config) => config.missing_cross_sections(&[nuclide]),
        // A panic elsewhere while holding the lock leaves the configuration
        // as readable as it was; the poison says nothing about it.
        Err(std::sync::TryLockError::Poisoned(poisoned)) => {
            poisoned.into_inner().missing_cross_sections(&[nuclide])
        }
        Err(std::sync::TryLockError::WouldBlock) => format!(
            "no neutron cross-section source for '{nuclide}'. Set one with \
             `cross_section_data`, or load the material's data with \
             `Material.read_nuclear_data`."
        ),
    }
}

/// Trait to allow flexible input types for set_cross_sections
pub trait IntoCrossSectionsInput {
    fn apply(self, config: &mut Config);
}

impl IntoCrossSectionsInput for HashMap<String, String> {
    fn apply(self, config: &mut Config) {
        for (nuclide, path) in self {
            refuse_branching_only(&path);
            if crate::url_cache::is_keyword(&path) {
                // If value is a keyword, set as global default
                config.default_cross_section = Some(path.clone());
            } else if !crate::storage::exists(std::path::Path::new(&path)) {
                // Validate the path through the active storage backend, not
                // `std::fs`: a browser host's in-memory files (`InMemoryStorage`)
                // have no filesystem behind them, and `std::path::Path::exists`
                // is always false there, so every non-keyword path would
                // panic. Keywords bypass; the backend's
                // `exists` already covers directories.
                panic!("Cross section file for '{nuclide}' does not exist at path: {path}");
            }
            config.cross_sections.insert(nuclide, path);
        }
    }
}

impl IntoCrossSectionsInput for &str {
    fn apply(self, config: &mut Config) {
        refuse_branching_only(self);
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

    /// The branching sources are an ordered list: kept in order and in the
    /// shape they were given, emptied by `clear`, and refused where an entry is
    /// empty or repeated.
    #[test]
    fn branching_sources_are_an_ordered_list() {
        let mut config = Config::new();
        assert!(config.get_transmutation_branch_ratios().is_empty());
        let sources = vec!["irdff-ii".to_string(), "endf-b8.1".to_string()];
        config
            .set_transmutation_branch_ratios(BranchSources::List(sources.clone()))
            .expect("two sources");
        assert_eq!(config.get_transmutation_branch_ratios(), sources);
        let twice = config
            .set_transmutation_branch_ratios(BranchSources::List(vec![
                "irdff-ii".into(),
                "irdff-ii".into(),
            ]))
            .expect_err("a source listed twice");
        assert!(twice.contains("listed twice"), "{twice}");
        assert!(config
            .set_transmutation_branch_ratios(BranchSources::One(String::new()))
            .is_err());
        // A refused list leaves the setting as it was.
        assert_eq!(config.get_transmutation_branch_ratios(), sources);
        // A list of one keeps its shape; the sources are the same as one given
        // on its own.
        let one = BranchSources::List(vec!["endf-b8.1".to_string()]);
        config
            .set_transmutation_branch_ratios(one.clone())
            .expect("a list of one");
        assert_eq!(config.transmutation_branch_ratios, one);
        assert_eq!(
            config.get_transmutation_branch_ratios(),
            BranchSources::One("endf-b8.1".to_string()).sources()
        );
        config.clear();
        assert_eq!(config.transmutation_branch_ratios, BranchSources::Unset);
        assert!(config.get_transmutation_branch_ratios().is_empty());
    }

    /// A branching-only library is not a cross-section source.
    #[test]
    #[should_panic(expected = "supplies isomeric branching only")]
    fn irdff_ii_is_not_a_cross_section_source() {
        Config::new().set_cross_sections("irdff-ii");
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

#[cfg(test)]
mod missing_cross_section_tests {
    use super::*;

    #[test]
    fn a_missing_library_says_how_to_set_one() {
        let empty = Config::new().missing_cross_sections(&["Fe56", "Fe57"]);
        assert!(
            empty.contains("no neutron cross-section library is configured"),
            "{empty}"
        );
        assert!(empty.contains("cross_section_data"), "{empty}");
        assert!(empty.contains("'Fe56', 'Fe57'"), "{empty}");
        assert!(empty.contains("endf-b8.1"), "{empty}");

        let mut mapped = Config::new();
        mapped
            .cross_sections
            .insert("Li6".to_string(), "Li6.arrow".to_string());
        let partial = mapped.missing_cross_sections(&["Fe56"]);
        assert!(partial.contains("maps 1 nuclide(s)"), "{partial}");
        assert!(partial.contains("'Fe56'"), "{partial}");
    }
}
