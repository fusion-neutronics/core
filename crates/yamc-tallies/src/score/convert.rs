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
        // First check for special named scores
        match s {
            "flux" => return Ok(Self::Flux(FluxScore)),
            "heating" => return Ok(Self::Heating(HeatingScore)),
            "heating-local" => return Ok(Self::HeatingLocal(HeatingLocalScore)),
            "damage-energy" => return Ok(Self::DamageEnergy(DamageEnergyScore)),

            // Production scores
            "H1-production" => return Ok(Self::Production(ProductionScore::H1)),
            "H2-production" => return Ok(Self::Production(ProductionScore::H2)),
            "H3-production" => return Ok(Self::Production(ProductionScore::H3)),
            "He3-production" => return Ok(Self::Production(ProductionScore::HE3)),
            "He4-production" => return Ok(Self::Production(ProductionScore::HE4)),

            // Common reaction names
            "total" => return Ok(Self::ReactionRate(ReactionRateScore::total())),
            "elastic" => return Ok(Self::ReactionRate(ReactionRateScore::elastic())),
            "inelastic" => return Ok(Self::ReactionRate(ReactionRateScore::inelastic())),
            "fission" => return Ok(Self::ReactionRate(ReactionRateScore::fission())),
            "absorption" => return Ok(Self::ReactionRate(ReactionRateScore::absorption())),

            // Photon scores
            "coherent-scatter" => {
                return Ok(Self::PhotonXS(PhotonXSScore {
                    component: PhotonComponent::Coherent,
                }))
            }
            "incoherent-scatter" => {
                return Ok(Self::PhotonXS(PhotonXSScore {
                    component: PhotonComponent::Incoherent,
                }))
            }
            "photoelectric" => {
                return Ok(Self::PhotonXS(PhotonXSScore {
                    component: PhotonComponent::Photoelectric,
                }))
            }
            "pair-production" => {
                return Ok(Self::PhotonXS(PhotonXSScore {
                    component: PhotonComponent::PairProduction,
                }))
            }

            _ => {}
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

        Err(format!("Unknown score: '{s}'"))
    }
}

// ----------------------------- Tests -----------------------------
