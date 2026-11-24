//! What built a nuclear data folder, as the folder records it.
//!
//! Every `version.json` and `provenance.json` a converter writes carries a
//! `converter` stamp, and a release's `manifest.json` lists each distinct one
//! as `converters` and records the parts of the identity apart in its
//! `provenance` (`builder`, `builder_commit`, `builder_dirty`). The stamp is
//! [`BuildIdentity::stamp`], formatted here once so every writer agrees.

/// The build of core that wrote some data: the package and version that ran
/// it, and the git commit it was compiled from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuildIdentity<'a> {
    /// The package and its version, such as `"yani-python 0.22.1"`. A wheel
    /// built from source reports `0.0.0.dev0`, which is why the commit is
    /// recorded beside it.
    pub package: &'a str,
    /// The full hash of the core commit the package was compiled from, or
    /// `None` when it was built outside a git checkout of core.
    pub commit: Option<&'a str>,
    /// Whether a tracked file differed from `commit` when it was compiled.
    /// `None` when the commit is unknown.
    pub dirty: Option<bool>,
}

impl BuildIdentity<'_> {
    /// The one-line stamp written as `converter`:
    /// `"yani-python 0.22.1 (core <commit>)"`, with `", dirty"` after the
    /// commit when the tree had changes, or the package alone when the
    /// commit is unknown.
    pub fn stamp(&self) -> String {
        match (self.commit, self.dirty) {
            (Some(commit), Some(true)) => format!("{} (core {commit}, dirty)", self.package),
            (Some(commit), _) => format!("{} (core {commit})", self.package),
            (None, _) => self.package.to_string(),
        }
    }

    /// Whether the build is known to differ from its commit. An unknown
    /// commit is not dirty: there is nothing to differ from, and the record
    /// says so with a null commit.
    pub fn is_dirty(&self) -> bool {
        self.commit.is_some() && self.dirty == Some(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMMIT: &str = "250c439e0000000000000000000000000000abcd";

    #[test]
    fn the_stamp_names_the_package_and_the_commit() {
        let clean = BuildIdentity {
            package: "yani-python 0.22.1",
            commit: Some(COMMIT),
            dirty: Some(false),
        };
        assert_eq!(clean.stamp(), format!("yani-python 0.22.1 (core {COMMIT})"));
        assert!(!clean.is_dirty());

        let dirty = BuildIdentity {
            dirty: Some(true),
            ..clean
        };
        assert_eq!(
            dirty.stamp(),
            format!("yani-python 0.22.1 (core {COMMIT}, dirty)")
        );
        assert!(dirty.is_dirty());
    }

    #[test]
    fn an_unknown_commit_is_the_package_alone_and_not_dirty() {
        let unknown = BuildIdentity {
            package: "yani-python 0.0.0.dev0",
            commit: None,
            dirty: None,
        };
        assert_eq!(unknown.stamp(), "yani-python 0.0.0.dev0");
        assert!(!unknown.is_dirty());
    }
}
