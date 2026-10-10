//! String round-trip for [`Score`]: `Display` (score -> name) and `FromStr`
//! (name -> score), kept together as the canonical name<->score mapping.

use super::*;
use crate::mt::Mt;
use std::fmt;
use std::str::FromStr;

impl fmt::Display for Score {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl FromStr for Score {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some((_, make)) = NAMED_SCORES.iter().find(|(name, _)| *name == s) {
            return Ok(make());
        }

        // Try parsing as integer MT number
        if let Ok(mt_int) = s.parse::<i32>() {
            // Map well-known integer strings to named variants
            // The named integer MTs use their dedicated constructors; every
            // other integer (301/901/444/203-207/502/504/516/522 plus the
            // generic MT fallthrough) is handled once by `from_mt_number`.
            return match mt_int {
                1 => Ok(Self::ReactionRate(ReactionRateScore::total())),
                2 => Ok(Self::ReactionRate(ReactionRateScore::elastic())),
                4 => Ok(Self::ReactionRate(ReactionRateScore::inelastic())),
                18 => Ok(Self::ReactionRate(ReactionRateScore::fission())),
                27 => Ok(Self::ReactionRate(ReactionRateScore::absorption())),
                _ => {
                    Self::from_mt_number(mt_int).map_err(|_| format!("Invalid MT number: {mt_int}"))
                }
            };
        }

        // Try looking up in the comprehensive REACTION_MT map from data.rs
        if let Some(&mt_int) = yamc_nuclide::data::REACTION_MT.get(s) {
            let mt = Mt::try_from(mt_int)
                .map_err(|_| format!("Invalid MT number {mt_int} for reaction '{s}'"))?;
            return Ok(Self::ReactionRate(ReactionRateScore::named(
                mt,
                s.to_string(),
            )));
        }

        Err(unknown_score_message(s))
    }
}

/// Builds the score a name in [`NAMED_SCORES`] stands for.
type MakeScore = fn() -> Score;

/// The score names `from_str` accepts directly, each with its constructor.
///
/// The parser and the "Unknown score" message both read this table, so the
/// list of valid names in the error cannot drift from what actually parses.
const NAMED_SCORES: &[(&str, MakeScore)] = &[
    ("flux", || Score::Flux(FluxScore)),
    ("heating", || Score::Heating(HeatingScore)),
    ("heating-local", || Score::HeatingLocal(HeatingLocalScore)),
    ("damage-energy", || Score::DamageEnergy(DamageEnergyScore)),
    // Production scores
    ("H1-production", || Score::Production(ProductionScore::H1)),
    ("H2-production", || Score::Production(ProductionScore::H2)),
    ("H3-production", || Score::Production(ProductionScore::H3)),
    ("He3-production", || Score::Production(ProductionScore::HE3)),
    ("He4-production", || Score::Production(ProductionScore::HE4)),
    // Common reaction names
    ("total", || Score::ReactionRate(ReactionRateScore::total())),
    ("elastic", || {
        Score::ReactionRate(ReactionRateScore::elastic())
    }),
    ("inelastic", || {
        Score::ReactionRate(ReactionRateScore::inelastic())
    }),
    ("fission", || {
        Score::ReactionRate(ReactionRateScore::fission())
    }),
    ("absorption", || {
        Score::ReactionRate(ReactionRateScore::absorption())
    }),
    // Photon scores
    ("coherent-scatter", || {
        Score::PhotonXS(PhotonXSScore {
            component: PhotonComponent::Coherent,
        })
    }),
    ("incoherent-scatter", || {
        Score::PhotonXS(PhotonXSScore {
            component: PhotonComponent::Incoherent,
        })
    }),
    ("photoelectric", || {
        Score::PhotonXS(PhotonXSScore {
            component: PhotonComponent::Photoelectric,
        })
    }),
    ("pair-production", || {
        Score::PhotonXS(PhotonXSScore {
            component: PhotonComponent::PairProduction,
        })
    }),
];

/// Build the error for a score string nothing recognises: the bad input, a
/// case-insensitive "did you mean" when one exists, the named scores, and the
/// reaction-name and MT-number forms that are also accepted.
fn unknown_score_message(s: &str) -> String {
    let suggestion = NAMED_SCORES
        .iter()
        .map(|(name, _)| *name)
        .chain(yamc_nuclide::data::REACTION_MT.keys().copied())
        .find(|name| name.eq_ignore_ascii_case(s));
    let did_you_mean = match suggestion {
        Some(name) => format!(" Did you mean '{name}'?"),
        None => String::new(),
    };
    let names: Vec<String> = NAMED_SCORES
        .iter()
        .map(|(name, _)| format!("'{name}'"))
        .collect();
    format!(
        "Unknown score: '{s}'.{did_you_mean} Valid named scores are: {}. A score can also be \
         an ENDF reaction name such as '(n,gamma)' or '(n,t)', or an MT number such as 102 \
         (as an int or a string).",
        names.join(", ")
    )
}

// ----------------------------- Tests -----------------------------
