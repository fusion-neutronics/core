//! Loading the configured chain for an irradiation with a source it reads
//! unset is refused, naming each such setting, before anything is resolved or
//! read, where an unset decay, reactions or fission-yields setting used to
//! read endf-b8.1.
//!
//! In a file of its own because it needs the global configuration in a known
//! state, which a test sharing a binary with one that configures it could not
//! rely on.

use yamc_nuclide::config::{ChainUse, Config, SubsectionSource};

fn load(chain_use: ChainUse) -> String {
    match yani_transmute::load_configured_chain(chain_use) {
        Ok(_) => panic!("a chain loaded from a decay folder that does not exist"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn unset_sources_are_refused_before_anything_is_read() {
    let missing = std::env::temp_dir().join("yani-no-such-decay-folder");
    {
        let mut config = Config::global();
        config.clear();
        config.transmutation_decay_data = Some(missing.to_string_lossy().into_owned());
    }

    // The decay folder does not exist, so a refusal that came after reading
    // anything would be about that folder rather than the unset settings.
    let err = load(ChainUse::Irradiation);
    assert!(
        err.starts_with("no transmutation data is configured"),
        "{err}"
    );
    assert!(err.contains("transmutation_reactions is not set"), "{err}");
    assert!(
        err.contains("transmutation_fission_yields is not set"),
        "{err}"
    );
    assert!(
        !err.contains("transmutation_decay_data is not set"),
        "decay is set: {err}"
    );

    // A decay calculation reads neither, so it gets as far as the folder.
    let decay = load(ChainUse::Decay);
    assert!(
        !decay.contains("no transmutation data is configured"),
        "{decay}"
    );

    // Turned off is a choice: the irradiation now gets as far as the folder too.
    {
        let mut config = Config::global();
        config.transmutation_reactions = SubsectionSource::Off;
        config.transmutation_fission_yields = SubsectionSource::Off;
    }
    let off = load(ChainUse::Irradiation);
    assert!(
        !off.contains("no transmutation data is configured"),
        "{off}"
    );
    Config::global().clear();
}
