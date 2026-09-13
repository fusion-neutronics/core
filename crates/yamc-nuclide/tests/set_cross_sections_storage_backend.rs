//! `set_cross_sections` validates paths through the active storage backend
//! (fusion-neutronics/core#99).
//!
//! A browser host populates `InMemoryStorage` with `/Li6.arrow/version.json`
//! and friends and then names `/Li6.arrow` in the config. The check used to go
//! through `std::path::Path::exists`, which knows nothing about that backend,
//! so on wasm32 every non-keyword path panicked and the documented in-memory
//! flow was closed. Swapping the process-global backend is why this lives in
//! its own test binary rather than beside the unit tests.

use std::collections::HashMap;
use std::path::Path;

use yamc_nuclide::config::Config;
use yamc_nuclide::storage::in_memory_storage::InMemoryStorage;
use yamc_nuclide::storage::{set_storage, NativeStorage};

/// Run `f` with an in-memory backend holding `files`, restoring the native
/// backend afterwards (also on panic). The backend is process-global, so the
/// tests in this binary take turns through one lock.
fn with_in_memory_files<R>(files: &[&str], f: impl FnOnce() -> R) -> R {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _serial = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            set_storage(Box::new(NativeStorage));
        }
    }
    let mem = InMemoryStorage::new();
    for path in files {
        mem.add_file(*path, b"{}".to_vec());
    }
    set_storage(Box::new(mem));
    let _restore = Restore;
    f()
}

#[test]
fn a_path_present_only_in_the_storage_backend_is_accepted() {
    with_in_memory_files(
        &[
            "/Li6.arrow/version.json",
            "/Li6.arrow/nuclide.arrow",
            "/Li6.arrow/reactions.arrow",
        ],
        || {
            // The directory exists for the backend (files sit under it) and for
            // nothing on the real filesystem.
            assert!(!Path::new("/Li6.arrow").exists());
            let mut config = Config::new();
            config.set_cross_sections(HashMap::from([(
                "Li6".to_string(),
                "/Li6.arrow".to_string(),
            )]));
            assert_eq!(
                config.cross_sections.get("Li6").map(String::as_str),
                Some("/Li6.arrow")
            );
        },
    );
}

#[test]
fn a_path_the_backend_does_not_have_is_still_refused() {
    let outcome = std::panic::catch_unwind(|| {
        with_in_memory_files(&["/Li6.arrow/version.json"], || {
            let mut config = Config::new();
            config.set_cross_sections(HashMap::from([(
                "Li7".to_string(),
                "/Li7.arrow".to_string(),
            )]));
        })
    });
    let err = outcome.expect_err("a missing path must still panic");
    let msg = err
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(
        msg.contains("does not exist at path: /Li7.arrow"),
        "unexpected panic message: {msg}"
    );
}

#[test]
fn keywords_still_bypass_the_backend() {
    with_in_memory_files(&[], || {
        let mut config = Config::new();
        config.set_cross_sections(HashMap::from([(
            "Li6".to_string(),
            "endf-b8.1".to_string(),
        )]));
        assert_eq!(config.default_cross_section.as_deref(), Some("endf-b8.1"));
    });
}
