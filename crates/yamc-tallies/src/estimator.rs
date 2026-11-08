//! Tally estimator: how track-length contributions vs collision events
//! are turned into score contributions.
//!
//! `TrackLength` is the default -- every cell crossing contributes
//! `weight × Δs` and the variance is integrated over the entire path
//! the particle travels. `Collision` instead fires only at collision
//! events, with the per-event flux contribution `weight / Σ_t` (and
//! analogous expressions for reaction rates / heating).
//!
//! The estimator lives on the [`Tally`](crate::Tally) so dispatch is
//! a single per-hook check in the transport loop: track-length tallies
//! are visited at every cell crossing, collision tallies are visited
//! at every collision site. Each score type declares which estimator(s)
//! are valid for it via [`Score::required_estimator`](crate::Score),
//! and [`Tally::validate`](crate::Tally::validate) refuses mismatched
//! combinations up front so simulations don't silently misreport.

use std::fmt;
use std::str::FromStr;

/// Estimator for a tally: how a particle's history is turned into
/// score contributions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub enum Estimator {
    /// Score at every cell crossing, weighted by the segment length
    /// in each crossed bin. Lower variance in optically thin regions;
    /// touches more bins per history. Default.
    #[default]
    TrackLength,
    /// Score at every collision event, weighted by `1 / Σ_t`. Lower
    /// variance in optically thick regions; touches one bin per
    /// collision instead of one per cell crossing. Heating-family
    /// scores require this estimator.
    Collision,
    // NOTE: if a surface-current / surface-crossing estimator is ever
    // added here, it must be validate-rejected under both delta-tracking
    // modes -- `TrackingMode::Woodcock` and `TrackingMode::Hybrid` (see
    // `Model::simulate_transport`): pure Woodcock and the hybrid surface
    // step do not produce the exact surface-crossing points such an
    // estimator needs, so it would silently misreport. Surface tracking
    // is the only mode that can serve a surface estimator.
}

impl Estimator {
    /// Lowercase display name (`"track-length"` / `"collision"`).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::TrackLength => "track-length",
            Self::Collision => "collision",
        }
    }
}

impl fmt::Display for Estimator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Estimator {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Exactly one accepted spelling per estimator -- no case-folding, no
        // hyphen/underscore variants. Anything else fails hard.
        match s {
            "track-length" => Ok(Self::TrackLength),
            "collision" => Ok(Self::Collision),
            other => Err(format!(
                "unknown estimator {other:?}; expected \"track-length\" or \"collision\""
            )),
        }
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_track_length() {
        assert_eq!(Estimator::default(), Estimator::TrackLength);
    }

    #[test]
    fn parse_accepts_only_canonical_spelling() {
        assert_eq!(
            Estimator::from_str("track-length").unwrap(),
            Estimator::TrackLength
        );
        assert_eq!(
            Estimator::from_str("collision").unwrap(),
            Estimator::Collision
        );
    }

    #[test]
    fn parse_rejects_variations_and_garbage() {
        // Only the exact canonical spelling is accepted -- variants fail hard.
        for bad in [
            "tracklength",
            "track_length",
            "Track-Length",
            "Track_Length",
            "TRACK-LENGTH",
            "Collision",
            "COLLISION",
            "surface",
            "",
        ] {
            assert!(
                Estimator::from_str(bad).is_err(),
                "{bad:?} should be rejected"
            );
        }
    }

    #[test]
    fn display_round_trip() {
        for e in [Estimator::TrackLength, Estimator::Collision] {
            assert_eq!(Estimator::from_str(&e.to_string()).unwrap(), e);
        }
    }
}
