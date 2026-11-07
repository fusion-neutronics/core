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

/// Where one optional transmutation subsection comes from.
///
/// Three states, because "the user has not chosen" and "the user chose not to
/// have it" are different answers, and only the first is an error when a
/// calculation reads the subsection. `Option<String>` cannot hold both.
///
/// Only the parts a calculation can do without are modelled this way. The
/// decay subsection is not one of them: half-lives and decay energies come
/// from it and a chain without it is empty, so it keeps a plain
/// `Option<String>`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum SubsectionSource {
    /// Not set. There is no default library, so a calculation that reads the
    /// subsection is refused before any work, naming the setting (see
    /// [`Config::transmutation_sources`]).
    #[default]
    Unset,
    /// Deliberately not loaded. Nothing is downloaded, and a calculation that
    /// turns out to need it fails with an error naming what it needed.
    Off,
    /// A library keyword or a path to the subsection directory.
    Set(String),
}

impl SubsectionSource {
    /// The source to resolve, or `None` when there is none: unset or off.
    pub fn source(&self) -> Option<&str> {
        match self {
            Self::Set(source) => Some(source),
            Self::Unset | Self::Off => None,
        }
    }

    /// Whether the user turned this subsection off, as opposed to leaving it alone.
    pub fn is_off(&self) -> bool {
        matches!(self, Self::Off)
    }
}

impl From<Option<&str>> for SubsectionSource {
    /// `None` is unset, matching every other optional source on [`Config`].
    /// [`Self::Off`] has no spelling here on purpose: turning a subsection off
    /// is a deliberate act and the caller has to name it.
    fn from(value: Option<&str>) -> Self {
        match value {
            None => Self::Unset,
            Some(source) => Self::Set(source.to_string()),
        }
    }
}

/// What a calculation reads from the transmutation chain. Every one reads the
/// decay data; the variants differ in what they read beside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainUse {
    /// The decay data alone: activity, decay heat, decay photons, dose,
    /// clearance, and a schedule that only cools.
    Decay,
    /// The decay data and which nuclide each neutron reaction makes: the
    /// activation products a D1S decay-photon run, `Model.radionuclides` and
    /// `PulseSchedule.time_correct_tally` follow.
    Activation,
    /// A schedule that irradiates: the decay data, the reactions and the
    /// fission yields.
    Irradiation,
}

impl ChainUse {
    /// What a transmutation reads: [`Self::Irradiation`] when its schedule
    /// irradiates, the decay data alone when it only cools.
    pub fn transmutation(irradiates: bool) -> Self {
        if irradiates {
            Self::Irradiation
        } else {
            Self::Decay
        }
    }

    fn reads_reactions(self) -> bool {
        matches!(self, Self::Activation | Self::Irradiation)
    }

    fn reads_fission_yields(self) -> bool {
        matches!(self, Self::Irradiation)
    }
}

/// The transmutation sources one calculation loads, from
/// [`Config::transmutation_sources`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransmutationSources {
    /// The decay subsection's source: a library keyword or a path.
    pub decay: String,
    /// The reactions subsection's source, or `None` when it is not loaded:
    /// turned off, or unset where the calculation does not read it.
    pub reactions: Option<String>,
    /// The fission-yields subsection's source, or `None` as for `reactions`.
    pub fission_yields: Option<String>,
    /// The isomeric-branching sources, in order; empty for no overlay.
    pub branching: Vec<String>,
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
#[derive(Debug, Clone)]
pub struct Config {
    /// Map of nuclide name -> absolute or relative path to its Arrow data directory.
    pub cross_sections: HashMap<String, String>,
    /// Optional global default cross section source keyword
    pub default_cross_section: Option<String>,
    /// Source (library keyword or path to a `decay/` subsection) for the decay
    /// part of the transmutation network. `None` is unset: there is no
    /// default library.
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
            transmutation_reactions: SubsectionSource::Unset,
            transmutation_fission_yields: SubsectionSource::Unset,
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
                        nuclide_or_keyword,
                        crate::url_cache::cross_section_keywords().join(", ")
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
    /// There is no default cross-section library: neutron and photon data
    /// follow the caller's selection, and a transport calculation that needs
    /// cross sections and finds no source stops here. Nor is there a default
    /// transmutation library (see [`Config::transmutation_sources`]).
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

    /// The transmutation sources a calculation of `chain_use` loads, or why it
    /// cannot run.
    ///
    /// There is no default transmutation library, as there is no default
    /// cross-section library: which evaluation the decay data, reactions and
    /// fission yields come from is the caller's choice. So a setting the
    /// calculation reads that is unset refuses it, before anything is
    /// resolved or downloaded, with a message naming each such setting and
    /// how to set it. A reactions or fission-yields setting turned off is a
    /// choice, and is not loaded; one left unset where the calculation does
    /// not read it is not loaded either.
    pub fn transmutation_sources(
        &self,
        chain_use: ChainUse,
    ) -> Result<TransmutationSources, String> {
        let mut unset = Vec::new();
        if self.transmutation_decay_data.is_none() {
            unset.push("transmutation_decay_data");
        }
        let reads = [
            (
                chain_use.reads_reactions(),
                &self.transmutation_reactions,
                "transmutation_reactions",
            ),
            (
                chain_use.reads_fission_yields(),
                &self.transmutation_fission_yields,
                "transmutation_fission_yields",
            ),
        ];
        for (read, setting, name) in reads {
            if read && *setting == SubsectionSource::Unset {
                unset.push(name);
            }
        }
        match &self.transmutation_decay_data {
            Some(decay) if unset.is_empty() => Ok(TransmutationSources {
                decay: decay.clone(),
                reactions: self.transmutation_reactions.source().map(str::to_string),
                fission_yields: self
                    .transmutation_fission_yields
                    .source()
                    .map(str::to_string),
                branching: self.get_transmutation_branch_ratios(),
            }),
            _ => Err(unset_transmutation_message(&unset, chain_use)),
        }
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
        self.transmutation_reactions = SubsectionSource::Unset;
        self.transmutation_fission_yields = SubsectionSource::Unset;
        self.transmutation_branch_ratios = BranchSources::Unset;
    }
}

/// Why a calculation of `chain_use` cannot run with the transmutation settings
/// `unset` (each one it reads and finds unset), saying how to set them.
fn unset_transmutation_message(unset: &[&str], chain_use: ChainUse) -> String {
    fn what(setting: &str) -> (&str, &str) {
        match setting {
            "transmutation_decay_data" => ("decay data", "decay"),
            "transmutation_reactions" => ("reactions", "reactions"),
            "transmutation_fission_yields" => ("fission yields", "fission_yields"),
            other => (other, other),
        }
    }
    let mut lines = Vec::new();
    for setting in unset {
        let (needs, subsection) = what(setting);
        let off = match (*setting, chain_use) {
            ("transmutation_reactions", ChainUse::Irradiation) => {
                ", or to False for a material with no reaction rates"
            }
            ("transmutation_fission_yields", ChainUse::Irradiation) => {
                ", or to False if nothing in the material fissions"
            }
            _ => "",
        };
        lines.push(format!(
            "this calculation needs {needs}, and {setting} is not set. {} Set it to one of \
             them or to a local folder holding it{off}.",
            crate::url_cache::publishers_of(subsection)
        ));
    }
    format!(
        "no transmutation data is configured: {} There is no default library, as there is \
         none for cross sections.",
        lines.join(" ")
    )
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
                "Invalid cross section source: '{}'. Must be a keyword ({}) or a directory path.",
                self,
                crate::url_cache::cross_section_keywords().join(", ")
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
    fn untouched_has_no_source() {
        assert_eq!(SubsectionSource::default(), SubsectionSource::Unset);
        assert_eq!(SubsectionSource::Unset.source(), None);
        assert!(!SubsectionSource::Unset.is_off());
    }

    #[test]
    fn off_has_no_source_either() {
        assert_eq!(SubsectionSource::Off.source(), None);
        assert!(SubsectionSource::Off.is_off());
    }

    #[test]
    fn a_set_source_is_returned_as_given() {
        let source = SubsectionSource::Set("tendl-2025".to_string());
        assert_eq!(source.source(), Some("tendl-2025"));
        assert!(!source.is_off());
    }

    /// `None` unsets; it is not a way to spell "off". The two were the same
    /// value once, and a chain silently lost its reactions whenever a caller
    /// cleared the setting.
    #[test]
    fn none_unsets() {
        assert_eq!(SubsectionSource::from(None), SubsectionSource::Unset);
        assert_eq!(
            SubsectionSource::from(Some("jeff-4.0")),
            SubsectionSource::Set("jeff-4.0".to_string())
        );
    }

    /// There is no default library: a fresh config supplies no subsection.
    #[test]
    fn a_new_config_supplies_no_transmutation_data() {
        let config = Config::new();
        assert_eq!(config.transmutation_decay_data, None);
        assert_eq!(config.transmutation_reactions, SubsectionSource::Unset);
        assert_eq!(config.transmutation_fission_yields, SubsectionSource::Unset);
        for chain_use in [ChainUse::Decay, ChainUse::Activation, ChainUse::Irradiation] {
            assert!(config.transmutation_sources(chain_use).is_err());
        }
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
    fn clear_unsets_every_transmutation_source() {
        let mut config = Config::new();
        config.transmutation_decay_data = Some("endf-b8.1".to_string());
        config.transmutation_reactions = SubsectionSource::Off;
        config.transmutation_fission_yields = SubsectionSource::Set("jeff-4.0".to_string());
        config.clear();
        assert_eq!(config.transmutation_decay_data, None);
        assert_eq!(config.transmutation_reactions, SubsectionSource::Unset);
        assert_eq!(config.transmutation_fission_yields, SubsectionSource::Unset);
    }
}

#[cfg(test)]
mod transmutation_sources_tests {
    //! A calculation reads the subsections its use names, and an unset one
    //! refuses it, naming the setting, where it used to read endf-b8.1.

    use super::*;

    fn unset_named(config: &Config, chain_use: ChainUse) -> Vec<&'static str> {
        let message = config
            .transmutation_sources(chain_use)
            .expect_err("a source it reads is unset");
        [
            "transmutation_decay_data",
            "transmutation_reactions",
            "transmutation_fission_yields",
            "transmutation_branch_ratios",
        ]
        .into_iter()
        .filter(|setting| message.contains(&format!("yamc.{setting} = ")))
        .collect()
    }

    #[test]
    fn each_use_names_the_unset_settings_it_reads() {
        let config = Config::new();
        assert_eq!(
            unset_named(&config, ChainUse::Decay),
            ["transmutation_decay_data"]
        );
        assert_eq!(
            unset_named(&config, ChainUse::Activation),
            ["transmutation_decay_data", "transmutation_reactions"]
        );
        assert_eq!(
            unset_named(&config, ChainUse::Irradiation),
            [
                "transmutation_decay_data",
                "transmutation_reactions",
                "transmutation_fission_yields"
            ]
        );
    }

    /// Only what is unset is named, so a caller who set two of three is told
    /// about the third alone.
    #[test]
    fn a_set_source_is_not_named() {
        let mut config = Config::new();
        config.transmutation_decay_data = Some("endf-b8.1".to_string());
        config.transmutation_reactions = SubsectionSource::Set("tendl-2025".to_string());
        assert_eq!(
            unset_named(&config, ChainUse::Irradiation),
            ["transmutation_fission_yields"]
        );
        let message = config
            .transmutation_sources(ChainUse::Irradiation)
            .expect_err("fission yields unset");
        assert!(
            message.contains("transmutation_fission_yields is not set"),
            "{message}"
        );
    }

    #[test]
    fn the_message_says_how_to_set_each_one() {
        let message = Config::new()
            .transmutation_sources(ChainUse::Irradiation)
            .expect_err("nothing is set");
        assert!(
            message.starts_with("no transmutation data is configured"),
            "{message}"
        );
        for (needs, setting, publishers) in [
            (
                "decay data",
                "transmutation_decay_data",
                "Libraries that publish decay data: endf-b8.1, jeff-4.0, jendl-5.0.",
            ),
            (
                "reactions",
                "transmutation_reactions",
                "Libraries that publish reactions: tendl-2025, tendl-2017, fendl-3.2d, \
                 endf-b8.1, jeff-4.0, jendl-5.0.",
            ),
            (
                "fission yields",
                "transmutation_fission_yields",
                "Libraries that publish fission yields: endf-b8.1, jeff-4.0, jendl-5.0.",
            ),
        ] {
            let sentence = format!("this calculation needs {needs}, and {setting} is not set.");
            assert!(message.contains(&sentence), "{sentence}: {message}");
            assert!(message.contains(publishers), "{publishers}: {message}");
        }
        assert!(message.contains("There is no default library"), "{message}");
        assert!(message.contains("a local folder"), "{message}");
        // No package prefix, so it reads the same from either package.
        assert!(
            !message.contains("yamc.") && !message.contains("yani."),
            "{message}"
        );
        assert!(!message.contains("irdff-ii"), "{message}");
        assert!(
            message.contains("or to False if nothing in the material fissions"),
            "{message}"
        );
        // A decay calculation has no use for the False of the other two.
        let decay = Config::new()
            .transmutation_sources(ChainUse::Decay)
            .expect_err("decay unset");
        assert!(!decay.contains("False"), "{decay}");
    }

    /// Turning a subsection off is a choice: it is not loaded, and the
    /// calculation runs without it.
    #[test]
    fn off_is_a_choice() {
        let mut config = Config::new();
        config.transmutation_decay_data = Some("endf-b8.1".to_string());
        config.transmutation_reactions = SubsectionSource::Off;
        config.transmutation_fission_yields = SubsectionSource::Off;
        let sources = config
            .transmutation_sources(ChainUse::Irradiation)
            .expect("each source is set or off");
        assert_eq!(
            sources,
            TransmutationSources {
                decay: "endf-b8.1".to_string(),
                reactions: None,
                fission_yields: None,
                branching: Vec::new(),
            }
        );
    }

    /// A use that does not read a subsection runs with it unset and does not
    /// load it, and loads what is set as before.
    #[test]
    fn a_use_runs_without_what_it_does_not_read() {
        let mut config = Config::new();
        config.transmutation_decay_data = Some("decay-dir".to_string());
        let decay = config
            .transmutation_sources(ChainUse::Decay)
            .expect("only decay is read");
        assert_eq!(decay.reactions, None);
        assert_eq!(decay.fission_yields, None);

        config.transmutation_reactions = SubsectionSource::Set("jeff-4.0".to_string());
        config
            .set_transmutation_branch_ratios(BranchSources::One("irdff-ii".to_string()))
            .expect("one branching source");
        let activation = config
            .transmutation_sources(ChainUse::Activation)
            .expect("decay and reactions are set");
        assert_eq!(activation.decay, "decay-dir");
        assert_eq!(activation.reactions.as_deref(), Some("jeff-4.0"));
        assert_eq!(activation.fission_yields, None);
        assert_eq!(activation.branching, ["irdff-ii"]);
        assert_eq!(
            config.transmutation_sources(ChainUse::Decay),
            Ok(activation)
        );
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

#[cfg(test)]
mod invalid_source_tests {
    use super::*;

    /// A source that is neither a keyword nor a directory is refused with every
    /// keyword a cross-section source can be, from the one list of them, and
    /// none of the branching-only ones.
    #[test]
    fn an_invalid_source_lists_every_cross_section_keyword() {
        let refusal = |set: fn()| -> String {
            let payload = std::panic::catch_unwind(set).expect_err("an invalid source panics");
            payload
                .downcast_ref::<String>()
                .cloned()
                .unwrap_or_default()
        };
        for message in [
            refusal(|| Config::new().set_cross_section("not-a-library", None)),
            refusal(|| Config::new().set_cross_sections("not-a-library")),
        ] {
            assert!(
                message.starts_with("Invalid cross section source: 'not-a-library'"),
                "{message}"
            );
            for keyword in crate::url_cache::cross_section_keywords() {
                assert!(message.contains(keyword), "{keyword} missing: {message}");
            }
            assert!(!message.contains("irdff-ii"), "{message}");
        }
    }
}
