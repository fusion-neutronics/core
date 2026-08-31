/// Which categories of progress output `simulate_transport` prints. Each flag
/// is independent and additive; an empty set is fully silent -- not even the
/// end-of-run summary. Progress is reported in source particles (not batches).
///
/// Cost per checkpoint (measured separately from the transport itself) is at
/// most a handful of `println!`s plus, for `tally_stream`, a per-scalar-tally
/// line from the running aggregate statistics (mesh tallies are skipped).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Verbose {
    /// Periodic `Progress: N/total particles (P%)` lines.
    pub progress: bool,
    /// Add elapsed wall-clock and ETA to the progress lines (implies progress).
    pub eta: bool,
    /// Per-tally running mean ± std (rel. err) at each checkpoint.
    pub tally_stream: bool,
    /// End-of-run completion line and per-tally final statistics block.
    pub summary: bool,
    /// One line per nuclear-data arrow file as it loads (nuclide + path), plus
    /// the transmutation chain file load. Non-overwriting (unlike progress).
    pub nuclear_data: bool,
}

impl Default for Verbose {
    /// `progress` + `eta` + `summary`. `tally` is OFF by default: per-tally
    /// running stats only refresh at chunk boundaries, so they go stale
    /// between updates on large runs; progress and ETA stay live (kept
    /// current by the heartbeat) and the final stats appear in the summary.
    /// Enable `tally` explicitly to stream convergence (shown with its
    /// checkpoint age so its staleness is visible).
    fn default() -> Self {
        Self {
            progress: true,
            eta: true,
            tally_stream: false,
            summary: true,
            nuclear_data: false,
        }
    }
}

impl Verbose {
    /// The fully-silent set: no terminal output at all.
    pub fn silent() -> Self {
        Self {
            progress: false,
            eta: false,
            tally_stream: false,
            summary: false,
            nuclear_data: false,
        }
    }

    /// Build from a list of case-insensitive flag names. Accepted:
    /// `"progress"`, `"eta"`, `"tally"` (aliases `"tally-stream"`, `"stream"`),
    /// `"summary"`, and `"nuclear_data"` (`-` and `_` are interchangeable). An
    /// empty list yields [`Verbose::silent`].
    pub fn from_flags<I, S>(flags: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut v = Verbose::silent();
        for flag in flags {
            match flag
                .as_ref()
                .to_ascii_lowercase()
                .replace('_', "-")
                .as_str()
            {
                "progress" => v.progress = true,
                "eta" => v.eta = true,
                "tally" | "tally-stream" | "stream" => v.tally_stream = true,
                "summary" => v.summary = true,
                "nuclear-data" | "nuclear data" => v.nuclear_data = true,
                other => {
                    return Err(format!(
                        "unknown verbose flag {other:?}; expected \"progress\", \"eta\", \
                         \"tally\", \"summary\", or \"nuclear_data\""
                    ))
                }
            }
        }
        Ok(v)
    }

    /// The active flag names, in canonical order.
    pub fn flags(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.progress {
            out.push("progress".to_string());
        }
        if self.eta {
            out.push("eta".to_string());
        }
        if self.tally_stream {
            out.push("tally".to_string());
        }
        if self.summary {
            out.push("summary".to_string());
        }
        if self.nuclear_data {
            out.push("nuclear_data".to_string());
        }
        out
    }

    /// Whether a progress line is printed at all.
    pub fn shows_progress(&self) -> bool {
        self.progress || self.eta
    }

    /// Whether nothing is printed at all.
    pub fn is_silent(&self) -> bool {
        !self.progress && !self.eta && !self.tally_stream && !self.summary && !self.nuclear_data
    }
}

/// Particle-transport algorithm used by `simulate_transport`. Selected
/// at the model level because tracking is a property of the transport
/// loop -- once chosen, every particle in the simulation uses the same
/// method.
///
/// - [`TrackingMode::Surface`] (default) -- standard ray-tracing,
///   distance-to-nearest-boundary each step. Always correct and a good
///   default; the right choice for typical fusion geometries that mix
///   dense and sparse regions and contain voids (vacuum vessel, ports).
/// - [`TrackingMode::Woodcock`] -- *pure* delta tracking. Free flights
///   are sampled against a global majorant `Σ_maj` and cells are crossed
///   with no boundary computation. Fastest on geometrically dense,
///   void-free models (many cells per mean free path with similar `Σ_t`,
///   e.g. finely diced or CAD-tessellated geometry). Avoid it when the
///   model has large voids or low-density regions: there `Σ_maj` far
///   exceeds the local `Σ_t`, so the rejection loop churns and it can be
///   much slower than surface tracking.
/// - [`TrackingMode::Hybrid`] -- delta tracking with an automatic
///   per-cell fallback to surface tracking in cells where delta tracking
///   would be inefficient (voids, large low-density regions). The safe
///   general-purpose Woodcock mode: it keeps the boundary-skipping win in
///   dense cells and never goes pathological in voids. Use it for a
///   Woodcock run on any model that also contains voids/low-density
///   regions (e.g. a tokamak with a vacuum vessel and air).
///
/// All three give the same answer within statistics; they differ only in
/// speed. Naming honours E. R. Woodcock's 1965 paper which introduced the
/// algorithm; "delta tracking" is the common American synonym.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum TrackingMode {
    /// Surface tracking: compute distance to the nearest cell boundary
    /// each step. Default; always correct.
    #[default]
    Surface,
    /// Pure Woodcock (delta) tracking against a global majorant, with no
    /// surface fallback. Crosses cells without computing boundary
    /// distances. Fastest on dense, void-free geometry; pathological in
    /// large voids / low-density regions (use [`TrackingMode::Hybrid`]
    /// there). Both estimators are accepted: a flux-score regular- or
    /// cylindrical-mesh track-length tally scores TRUE track length along
    /// each delta flight segment (every crossed voxel gets its share), so
    /// fine meshes converge much faster than with collision scoring; all
    /// other track-length tallies (cell tallies, and cross-section-weighted
    /// scores such as heating or reaction rates) use the equivalent
    /// delta-tracking collision-density estimator. Coupled neutron-photon
    /// and decay-photon transport are likewise supported. Flights
    /// terminate at vacuum boundaries exactly like surface tracking
    /// (each flight is checked against the geometry's vacuum surfaces),
    /// so disjoint vacuum-bounded bodies see no tunneling.
    Woodcock,
    /// Woodcock (delta) tracking with an automatic per-cell fallback to
    /// surface tracking in cells where delta tracking would be
    /// inefficient (voids, large low-density regions). Same results as
    /// [`TrackingMode::Woodcock`] within statistics; never pathological
    /// in voids. The recommended Woodcock mode for models with voids.
    Hybrid,
}

impl std::str::FromStr for TrackingMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "surface" => Ok(TrackingMode::Surface),
            "woodcock" => Ok(TrackingMode::Woodcock),
            "hybrid" => Ok(TrackingMode::Hybrid),
            other => Err(format!(
                "tracking_mode must be one of: surface, woodcock, hybrid (got {other:?})"
            )),
        }
    }
}

impl std::fmt::Display for TrackingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            TrackingMode::Surface => "surface",
            TrackingMode::Woodcock => "woodcock",
            TrackingMode::Hybrid => "hybrid",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nuclear_data_flag_parses_and_round_trips() {
        let v = Verbose::from_flags(["nuclear_data"]).unwrap();
        assert!(v.nuclear_data);
        assert_eq!(v.flags(), vec!["nuclear_data".to_string()]);
        // hyphen and space spellings are accepted too
        assert!(Verbose::from_flags(["nuclear-data"]).unwrap().nuclear_data);
        assert!(Verbose::from_flags(["nuclear data"]).unwrap().nuclear_data);
    }

    #[test]
    fn nuclear_data_off_by_default_and_not_silent_when_set() {
        assert!(!Verbose::default().nuclear_data);
        assert!(!Verbose::from_flags(["nuclear_data"]).unwrap().is_silent());
    }

    #[test]
    fn unknown_verbose_flag_errors() {
        assert!(Verbose::from_flags(["nope"]).is_err());
    }
}
