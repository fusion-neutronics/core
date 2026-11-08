//! The score-kind discriminant and the per-quantity score structs
//! (flux, heating, damage-energy, production, reaction-rate, photon XS) with
//! their named constructors.

use crate::mt::Mt;

/// Discriminant for score cache grouping (maps 1:1 to cache index vectors).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ScoreKind {
    Flux,
    Heating,
    HeatingLocal,
    Production,
    DamageEnergy,
    ReactionRate,
    PhotonXS,
    Current,
}

// ---------------------------------------------------------------------------
// Score structs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FluxScore;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HeatingScore;

/// Which crossings a surface current counts, by the side of the surface they
/// cross to: `Above` is towards `Surface.above` (outwards through a sphere or
/// cylinder, towards +x through an x plane), `Below` towards `Surface.below`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CurrentDirection {
    /// Above minus below: each crossing adds its weight with the sign of its side.
    Net,
    /// Crossings towards `Surface.above`, counted positive.
    Above,
    /// Crossings towards `Surface.below`, counted positive.
    Below,
}

/// Current through a surface, scored only by surface tallies.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CurrentScore {
    pub direction: CurrentDirection,
}

impl CurrentScore {
    pub const NET: Self = Self {
        direction: CurrentDirection::Net,
    };
    pub const ABOVE: Self = Self {
        direction: CurrentDirection::Above,
    };
    pub const BELOW: Self = Self {
        direction: CurrentDirection::Below,
    };

    /// The score's name: `"current-net"`, `"current-above"` or `"current-below"`.
    pub fn name(&self) -> &'static str {
        match self.direction {
            CurrentDirection::Net => "current-net",
            CurrentDirection::Above => "current-above",
            CurrentDirection::Below => "current-below",
        }
    }

    /// What one crossing of weight `weight` adds, given `mu`, the cosine
    /// between direction and the normal to the positive side; `None` when
    /// this direction does not count it. A crossing exactly along the
    /// surface (`mu == 0`) goes with the above side, as in `Net`.
    #[inline]
    pub fn contribution(&self, weight: f64, mu: f64) -> Option<f64> {
        let above = mu.is_sign_positive();
        match self.direction {
            CurrentDirection::Net => Some(weight.copysign(mu)),
            CurrentDirection::Above => above.then_some(weight),
            CurrentDirection::Below => (!above).then_some(weight),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HeatingLocalScore;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DamageEnergyScore;

/// Particle production score (H1–He4).
///
/// The `name` field is a redundant display string derived from `mt`
/// (e.g. `Mt::H1_PRODUCTION` → `"H1-production"`); serde skips it on
/// disk and any reader that needs the human-readable form can recompute
/// it from `mt` via the named constructors below.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ProductionScore {
    pub mt: Mt,
    #[serde(skip, default = "empty_static_str")]
    pub name: &'static str,
}

fn empty_static_str() -> &'static str {
    ""
}

impl ProductionScore {
    pub const H1: Self = Self {
        mt: Mt::H1_PRODUCTION,
        name: "H1-production",
    };
    pub const H2: Self = Self {
        mt: Mt::H2_PRODUCTION,
        name: "H2-production",
    };
    pub const H3: Self = Self {
        mt: Mt::H3_PRODUCTION,
        name: "H3-production",
    };
    pub const HE3: Self = Self {
        mt: Mt::HE3_PRODUCTION,
        name: "He3-production",
    };
    pub const HE4: Self = Self {
        mt: Mt::HE4_PRODUCTION,
        name: "He4-production",
    };
}

/// Reaction-rate score (total, elastic, fission, arbitrary MT, named reactions).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ReactionRateScore {
    pub mt: Mt,
    pub display_name: Option<String>,
}

impl ReactionRateScore {
    /// Unnamed MT -- round-trips as integer in Python.
    pub fn from_mt(mt: Mt) -> Self {
        Self {
            mt,
            display_name: None,
        }
    }

    /// Named reaction -- round-trips as string in Python.
    pub fn named(mt: Mt, name: impl Into<String>) -> Self {
        Self {
            mt,
            display_name: Some(name.into()),
        }
    }

    // Well-known named constructors
    pub fn total() -> Self {
        Self::named(Mt::TOTAL, "total")
    }
    pub fn elastic() -> Self {
        Self::named(Mt::ELASTIC, "elastic")
    }
    pub fn inelastic() -> Self {
        Self::named(Mt::INELASTIC, "inelastic")
    }
    pub fn fission() -> Self {
        Self::named(Mt::FISSION, "fission")
    }
    pub fn absorption() -> Self {
        Self::named(Mt::ABSORPTION, "absorption")
    }
}

/// Photon cross-section component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PhotonComponent {
    Coherent,
    Incoherent,
    Photoelectric,
    PairProduction,
}

impl PhotonComponent {
    /// Cache-index tag (matches the fast-path `component` byte).
    pub fn as_u8(self) -> u8 {
        match self {
            Self::Coherent => 0,
            Self::Incoherent => 1,
            Self::Photoelectric => 2,
            Self::PairProduction => 3,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Coherent => "coherent-scatter",
            Self::Incoherent => "incoherent-scatter",
            Self::Photoelectric => "photoelectric",
            Self::PairProduction => "pair-production",
        }
    }
}

/// Photon cross-section score.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PhotonXSScore {
    pub component: PhotonComponent,
}

impl PhotonXSScore {
    pub fn mt(self) -> Mt {
        match self.component {
            PhotonComponent::Coherent => Mt::COHERENT,
            PhotonComponent::Incoherent => Mt::INCOHERENT,
            PhotonComponent::Photoelectric => Mt::PHOTOELECTRIC,
            PhotonComponent::PairProduction => Mt::PAIR_PRODUCTION,
        }
    }
}
