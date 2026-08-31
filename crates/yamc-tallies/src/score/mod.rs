//! Tally scores: what physical quantity a tally accumulates.
//!
//! The score-kind structs live in [`kinds`]; the name<->score string mapping
//! lives in [`convert`]. This module owns the [`Score`] enum-of-structs and its
//! query methods (`kind`/`name`/`mt`/`from_mt*`/`required_estimator`), and
//! re-exports the kinds so `crate::score::<T>` paths stay stable.

use super::estimator::Estimator;
use super::mt::Mt;

mod convert;
pub mod kinds;

pub use kinds::*;

/// Represents a tally score -- what physical quantity to accumulate.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Score {
    Flux(FluxScore),
    Heating(HeatingScore),
    HeatingLocal(HeatingLocalScore),
    Production(ProductionScore),
    DamageEnergy(DamageEnergyScore),
    ReactionRate(ReactionRateScore),
    PhotonXS(PhotonXSScore),
}

impl Score {
    /// Get the `ScoreKind` discriminant (used by cache grouping).
    pub fn kind(&self) -> ScoreKind {
        match self {
            Self::Flux(_) => ScoreKind::Flux,
            Self::Heating(_) => ScoreKind::Heating,
            Self::HeatingLocal(_) => ScoreKind::HeatingLocal,
            Self::Production(_) => ScoreKind::Production,
            Self::DamageEnergy(_) => ScoreKind::DamageEnergy,
            Self::ReactionRate(_) => ScoreKind::ReactionRate,
            Self::PhotonXS(_) => ScoreKind::PhotonXS,
        }
    }

    /// Display name for the score.
    pub fn name(&self) -> String {
        match self {
            Self::Flux(_) => "flux".to_string(),
            Self::Heating(_) => "heating".to_string(),
            Self::HeatingLocal(_) => "heating-local".to_string(),
            Self::DamageEnergy(_) => "damage-energy".to_string(),
            Self::Production(p) => p.name.to_string(),
            Self::ReactionRate(r) => match &r.display_name {
                Some(name) => name.clone(),
                None => r.mt.to_string(),
            },
            Self::PhotonXS(p) => p.component.name().to_string(),
        }
    }

    /// MT number, if applicable (returns `None` for flux).
    pub fn mt(&self) -> Option<Mt> {
        match self {
            Self::Flux(_) => None,
            Self::Heating(_) => Some(Mt::HEATING),
            Self::HeatingLocal(_) => Some(Mt::HEATING_LOCAL),
            Self::DamageEnergy(_) => Some(Mt::DAMAGE_ENERGY),
            Self::Production(p) => Some(p.mt),
            Self::ReactionRate(r) => Some(r.mt),
            Self::PhotonXS(p) => Some(p.mt()),
        }
    }

    /// Convert to `i32` representation (MT number).
    /// Panics for `Flux` (which has no MT).
    pub fn to_i32(&self) -> i32 {
        match self {
            Self::Flux(_) => panic!(
                "Direct integer value for flux score is not supported. Use 'flux' string only."
            ),
            _ => self.mt().unwrap().as_i32(),
        }
    }

    /// Convert to MT number if applicable (returns `None` for flux).
    pub fn to_mt(&self) -> Option<i32> {
        self.mt().map(|m| m.as_i32())
    }

    /// Which estimator(s) this score supports under the current
    /// codebase. `None` means either estimator is valid.
    ///
    /// - Flux: track-length and collision (`Tally::score_track_length`
    ///   and `Tally::score_collision`).
    /// - Heating / HeatingLocal: track-length (neutron KERMA) and
    ///   collision. Both flavors flow through one unified dispatch:
    ///   neutron collision uses `(heating_xs / Σ_t) · weight`; photon
    ///   collision uses the precomputed analog value
    ///   `(E_in − E_out − Σ E_secondary) · weight` passed in by the
    ///   transport loop. The photon-analog path fires at every photon
    ///   collision regardless of `tally.estimator`, so a photon-transport
    ///   heating tally still scores correctly even when its estimator is
    ///   `TrackLength`.
    /// - ReactionRate: track-length and collision
    ///   (`(σ_r / Σ_t) · weight` per collision, URR-aware for the
    ///   total/elastic/fission/absorption/capture MTs).
    /// - Production: track-length and collision
    ///   (`(σ_production_mt / Σ_t) · weight` per neutron collision, for
    ///   the H1–He4 production MTs 203–207).
    /// - DamageEnergy: track-length and collision
    ///   (`(damage_energy_xs / Σ_t) · weight` per neutron collision, MT 444).
    /// - PhotonXS: track-length and collision
    ///   (`(σ_component / Σ_t) · weight` per photon collision for each
    ///   of Coherent / Incoherent / Photoelectric / PairProduction).
    ///
    /// Every score type now supports both estimators; this function
    /// returns `None` for every variant. It is kept (rather than
    /// deleted) so future score types start out as `Some(TrackLength)`
    /// by default until collision support is wired.
    pub fn required_estimator(&self) -> Option<Estimator> {
        match self {
            Self::Flux(_) => None,
            Self::Heating(_) | Self::HeatingLocal(_) => None,
            Self::ReactionRate(_) => None,
            Self::Production(_) => None,
            Self::DamageEnergy(_) => None,
            Self::PhotonXS(_) => None,
        }
    }

    /// Base units string for this score type.
    pub fn base_units(&self) -> &'static str {
        match self {
            Self::Flux(_) => "cm",
            Self::Heating(_) | Self::HeatingLocal(_) | Self::DamageEnergy(_) => "eV",
            Self::Production(_) => "particles",
            Self::ReactionRate(_) | Self::PhotonXS(_) => "reactions",
        }
    }

    /// Construct a `Score` from a raw integer MT number.
    ///
    /// Special MT numbers (301, 901, 444, 203–207, 502/504/516/522) map
    /// to their dedicated variants. Everything else becomes an unnamed
    /// `ReactionRate` (round-trips as integer in Python).
    pub fn from_mt_number(i: i32) -> Result<Self, String> {
        match i {
            301 => Ok(Self::Heating(HeatingScore)),
            901 => Ok(Self::HeatingLocal(HeatingLocalScore)),
            444 => Ok(Self::DamageEnergy(DamageEnergyScore)),
            203 => Ok(Self::Production(ProductionScore::H1)),
            204 => Ok(Self::Production(ProductionScore::H2)),
            205 => Ok(Self::Production(ProductionScore::H3)),
            206 => Ok(Self::Production(ProductionScore::HE3)),
            207 => Ok(Self::Production(ProductionScore::HE4)),
            502 => Ok(Self::PhotonXS(PhotonXSScore {
                component: PhotonComponent::Coherent,
            })),
            504 => Ok(Self::PhotonXS(PhotonXSScore {
                component: PhotonComponent::Incoherent,
            })),
            516 => Ok(Self::PhotonXS(PhotonXSScore {
                component: PhotonComponent::PairProduction,
            })),
            522 => Ok(Self::PhotonXS(PhotonXSScore {
                component: PhotonComponent::Photoelectric,
            })),
            _ => {
                let mt = Mt::try_from(i)?;
                Ok(Self::ReactionRate(ReactionRateScore::from_mt(mt)))
            }
        }
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- ScoreKind / kind() ----

    #[test]
    fn test_kind_flux() {
        let s = Score::Flux(FluxScore);
        assert_eq!(s.kind(), ScoreKind::Flux);
    }

    #[test]
    fn test_kind_heating() {
        let s = Score::Heating(HeatingScore);
        assert_eq!(s.kind(), ScoreKind::Heating);
    }

    #[test]
    fn test_kind_heating_local() {
        let s = Score::HeatingLocal(HeatingLocalScore);
        assert_eq!(s.kind(), ScoreKind::HeatingLocal);
    }

    #[test]
    fn test_kind_damage_energy() {
        let s = Score::DamageEnergy(DamageEnergyScore);
        assert_eq!(s.kind(), ScoreKind::DamageEnergy);
    }

    #[test]
    fn test_kind_production() {
        let s = Score::Production(ProductionScore::H1);
        assert_eq!(s.kind(), ScoreKind::Production);
    }

    #[test]
    fn test_kind_reaction_rate() {
        let s = Score::ReactionRate(ReactionRateScore::total());
        assert_eq!(s.kind(), ScoreKind::ReactionRate);
    }

    #[test]
    fn test_kind_photon_xs() {
        let s = Score::PhotonXS(PhotonXSScore {
            component: PhotonComponent::Coherent,
        });
        assert_eq!(s.kind(), ScoreKind::PhotonXS);
    }

    // ---- name() ----

    #[test]
    fn test_name_flux() {
        assert_eq!(Score::Flux(FluxScore).name(), "flux");
    }

    #[test]
    fn test_name_heating() {
        assert_eq!(Score::Heating(HeatingScore).name(), "heating");
    }

    #[test]
    fn test_name_heating_local() {
        assert_eq!(
            Score::HeatingLocal(HeatingLocalScore).name(),
            "heating-local"
        );
    }

    #[test]
    fn test_name_damage_energy() {
        assert_eq!(
            Score::DamageEnergy(DamageEnergyScore).name(),
            "damage-energy"
        );
    }

    #[test]
    fn test_name_production_all() {
        assert_eq!(
            Score::Production(ProductionScore::H1).name(),
            "H1-production"
        );
        assert_eq!(
            Score::Production(ProductionScore::H2).name(),
            "H2-production"
        );
        assert_eq!(
            Score::Production(ProductionScore::H3).name(),
            "H3-production"
        );
        assert_eq!(
            Score::Production(ProductionScore::HE3).name(),
            "He3-production"
        );
        assert_eq!(
            Score::Production(ProductionScore::HE4).name(),
            "He4-production"
        );
    }

    #[test]
    fn test_name_reaction_rate_named() {
        assert_eq!(
            Score::ReactionRate(ReactionRateScore::total()).name(),
            "total"
        );
        assert_eq!(
            Score::ReactionRate(ReactionRateScore::elastic()).name(),
            "elastic"
        );
        assert_eq!(
            Score::ReactionRate(ReactionRateScore::fission()).name(),
            "fission"
        );
        assert_eq!(
            Score::ReactionRate(ReactionRateScore::absorption()).name(),
            "absorption"
        );
        assert_eq!(
            Score::ReactionRate(ReactionRateScore::inelastic()).name(),
            "inelastic"
        );
    }

    #[test]
    fn test_name_reaction_rate_unnamed() {
        // Unnamed uses MT number as string
        let s = Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(102)));
        assert_eq!(s.name(), "102");
    }

    #[test]
    fn test_name_photon_xs_all() {
        assert_eq!(
            Score::PhotonXS(PhotonXSScore {
                component: PhotonComponent::Coherent
            })
            .name(),
            "coherent-scatter"
        );
        assert_eq!(
            Score::PhotonXS(PhotonXSScore {
                component: PhotonComponent::Incoherent
            })
            .name(),
            "incoherent-scatter"
        );
        assert_eq!(
            Score::PhotonXS(PhotonXSScore {
                component: PhotonComponent::Photoelectric
            })
            .name(),
            "photoelectric"
        );
        assert_eq!(
            Score::PhotonXS(PhotonXSScore {
                component: PhotonComponent::PairProduction
            })
            .name(),
            "pair-production"
        );
    }

    // ---- mt() ----

    #[test]
    fn test_mt_flux_is_none() {
        assert!(Score::Flux(FluxScore).mt().is_none());
    }

    #[test]
    fn test_mt_heating() {
        assert_eq!(Score::Heating(HeatingScore).mt(), Some(Mt::HEATING));
    }

    #[test]
    fn test_mt_heating_local() {
        assert_eq!(
            Score::HeatingLocal(HeatingLocalScore).mt(),
            Some(Mt::HEATING_LOCAL)
        );
    }

    #[test]
    fn test_mt_damage_energy() {
        assert_eq!(
            Score::DamageEnergy(DamageEnergyScore).mt(),
            Some(Mt::DAMAGE_ENERGY)
        );
    }

    #[test]
    fn test_mt_production() {
        assert_eq!(
            Score::Production(ProductionScore::H1).mt(),
            Some(Mt::H1_PRODUCTION)
        );
        assert_eq!(
            Score::Production(ProductionScore::HE4).mt(),
            Some(Mt::HE4_PRODUCTION)
        );
    }

    #[test]
    fn test_mt_reaction_rate() {
        assert_eq!(
            Score::ReactionRate(ReactionRateScore::total()).mt(),
            Some(Mt::TOTAL)
        );
    }

    #[test]
    fn test_mt_photon_xs() {
        let s = Score::PhotonXS(PhotonXSScore {
            component: PhotonComponent::Coherent,
        });
        assert_eq!(s.mt(), Some(Mt::COHERENT));
    }

    // ---- to_i32() ----

    #[test]
    fn test_to_i32_heating() {
        assert_eq!(Score::Heating(HeatingScore).to_i32(), 301);
    }

    #[test]
    fn test_to_i32_reaction_rate() {
        assert_eq!(Score::ReactionRate(ReactionRateScore::total()).to_i32(), 1);
    }

    #[test]
    #[should_panic(expected = "Direct integer value for flux score is not supported")]
    fn test_to_i32_flux_panics() {
        Score::Flux(FluxScore).to_i32();
    }

    // ---- to_mt() ----

    #[test]
    fn test_to_mt_flux_is_none() {
        assert!(Score::Flux(FluxScore).to_mt().is_none());
    }

    #[test]
    fn test_to_mt_heating() {
        assert_eq!(Score::Heating(HeatingScore).to_mt(), Some(301));
    }

    // ---- base_units() ----

    #[test]
    fn test_base_units() {
        assert_eq!(Score::Flux(FluxScore).base_units(), "cm");
        assert_eq!(Score::Heating(HeatingScore).base_units(), "eV");
        assert_eq!(Score::HeatingLocal(HeatingLocalScore).base_units(), "eV");
        assert_eq!(Score::DamageEnergy(DamageEnergyScore).base_units(), "eV");
        assert_eq!(
            Score::Production(ProductionScore::H1).base_units(),
            "particles"
        );
        assert_eq!(
            Score::ReactionRate(ReactionRateScore::total()).base_units(),
            "reactions"
        );
        assert_eq!(
            Score::PhotonXS(PhotonXSScore {
                component: PhotonComponent::Coherent
            })
            .base_units(),
            "reactions"
        );
    }

    // ---- Display ----

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Score::Flux(FluxScore)), "flux");
        assert_eq!(format!("{}", Score::Heating(HeatingScore)), "heating");
        assert_eq!(
            format!("{}", Score::ReactionRate(ReactionRateScore::fission())),
            "fission"
        );
    }

    // ---- FromStr (string parsing) ----

    #[test]
    fn test_from_str_flux() {
        let s: Score = "flux".parse().unwrap();
        assert_eq!(s.kind(), ScoreKind::Flux);
    }

    #[test]
    fn test_from_str_heating() {
        let s: Score = "heating".parse().unwrap();
        assert_eq!(s.kind(), ScoreKind::Heating);
    }

    #[test]
    fn test_from_str_heating_local() {
        let s: Score = "heating-local".parse().unwrap();
        assert_eq!(s.kind(), ScoreKind::HeatingLocal);
    }

    #[test]
    fn test_from_str_damage_energy() {
        let s: Score = "damage-energy".parse().unwrap();
        assert_eq!(s.kind(), ScoreKind::DamageEnergy);
    }

    #[test]
    fn test_from_str_productions() {
        let names = [
            "H1-production",
            "H2-production",
            "H3-production",
            "He3-production",
            "He4-production",
        ];
        for name in &names {
            let s: Score = name.parse().unwrap();
            assert_eq!(s.kind(), ScoreKind::Production, "Failed for {name}");
        }
    }

    #[test]
    fn test_from_str_reaction_names() {
        for name in &["total", "elastic", "inelastic", "fission", "absorption"] {
            let s: Score = name.parse().unwrap();
            assert_eq!(s.kind(), ScoreKind::ReactionRate, "Failed for {name}");
            assert_eq!(s.name(), *name);
        }
    }

    #[test]
    fn test_from_str_photon_scores() {
        let cases = [
            ("coherent-scatter", PhotonComponent::Coherent),
            ("incoherent-scatter", PhotonComponent::Incoherent),
            ("photoelectric", PhotonComponent::Photoelectric),
            ("pair-production", PhotonComponent::PairProduction),
        ];
        for (name, expected_comp) in &cases {
            let s: Score = name.parse().unwrap();
            match &s {
                Score::PhotonXS(pxs) => {
                    assert_eq!(pxs.component, *expected_comp, "Failed for {name}")
                }
                _ => panic!("Expected PhotonXS for {name}"),
            }
        }
    }

    #[test]
    fn test_from_str_integer_well_known() {
        // Well-known integer strings that map to named variants
        let s: Score = "1".parse().unwrap();
        assert_eq!(s.name(), "total");

        let s: Score = "2".parse().unwrap();
        assert_eq!(s.name(), "elastic");

        let s: Score = "18".parse().unwrap();
        assert_eq!(s.name(), "fission");

        let s: Score = "27".parse().unwrap();
        assert_eq!(s.name(), "absorption");

        let s: Score = "4".parse().unwrap();
        assert_eq!(s.name(), "inelastic");
    }

    #[test]
    fn test_from_str_integer_special_scores() {
        let s: Score = "301".parse().unwrap();
        assert_eq!(s.kind(), ScoreKind::Heating);

        let s: Score = "901".parse().unwrap();
        assert_eq!(s.kind(), ScoreKind::HeatingLocal);

        let s: Score = "444".parse().unwrap();
        assert_eq!(s.kind(), ScoreKind::DamageEnergy);
    }

    #[test]
    fn test_from_str_integer_production() {
        let s: Score = "203".parse().unwrap();
        assert_eq!(s.name(), "H1-production");

        let s: Score = "207".parse().unwrap();
        assert_eq!(s.name(), "He4-production");
    }

    #[test]
    fn test_from_str_integer_photon_xs() {
        let s: Score = "502".parse().unwrap();
        assert_eq!(s.name(), "coherent-scatter");

        let s: Score = "504".parse().unwrap();
        assert_eq!(s.name(), "incoherent-scatter");

        let s: Score = "516".parse().unwrap();
        assert_eq!(s.name(), "pair-production");

        let s: Score = "522".parse().unwrap();
        assert_eq!(s.name(), "photoelectric");
    }

    #[test]
    fn test_from_str_integer_arbitrary_mt() {
        // Arbitrary MT like 102 (n,gamma) should become unnamed ReactionRate
        let s: Score = "102".parse().unwrap();
        assert_eq!(s.kind(), ScoreKind::ReactionRate);
        assert_eq!(s.name(), "102"); // unnamed, so MT number as string
    }

    #[test]
    fn test_from_str_reaction_name_from_data() {
        // "(n,gamma)" should be looked up from REACTION_MT map
        let s: Score = "(n,gamma)".parse().unwrap();
        assert_eq!(s.kind(), ScoreKind::ReactionRate);
        assert_eq!(s.name(), "(n,gamma)");
    }

    #[test]
    fn test_from_str_unknown() {
        let result = "not_a_real_score_xyz".parse::<Score>();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown score"));
    }

    #[test]
    fn test_from_str_invalid_integer() {
        // 0 is out of MT range
        let result = "0".parse::<Score>();
        assert!(result.is_err());

        // 1000 is out of MT range
        let result = "1000".parse::<Score>();
        assert!(result.is_err());
    }

    // ---- from_mt_number ----

    #[test]
    fn test_from_mt_number_heating() {
        let s = Score::from_mt_number(301).unwrap();
        assert_eq!(s.kind(), ScoreKind::Heating);
    }

    #[test]
    fn test_from_mt_number_heating_local() {
        let s = Score::from_mt_number(901).unwrap();
        assert_eq!(s.kind(), ScoreKind::HeatingLocal);
    }

    #[test]
    fn test_from_mt_number_damage_energy() {
        let s = Score::from_mt_number(444).unwrap();
        assert_eq!(s.kind(), ScoreKind::DamageEnergy);
    }

    #[test]
    fn test_from_mt_number_productions() {
        let cases = [
            (203, "H1-production"),
            (204, "H2-production"),
            (205, "H3-production"),
            (206, "He3-production"),
            (207, "He4-production"),
        ];
        for (mt, expected_name) in &cases {
            let s = Score::from_mt_number(*mt).unwrap();
            assert_eq!(s.name(), *expected_name);
        }
    }

    #[test]
    fn test_from_mt_number_photon_xs() {
        let cases = [
            (502, PhotonComponent::Coherent),
            (504, PhotonComponent::Incoherent),
            (516, PhotonComponent::PairProduction),
            (522, PhotonComponent::Photoelectric),
        ];
        for (mt, expected_comp) in &cases {
            let s = Score::from_mt_number(*mt).unwrap();
            match &s {
                Score::PhotonXS(pxs) => assert_eq!(pxs.component, *expected_comp),
                _ => panic!("Expected PhotonXS for MT {mt}"),
            }
        }
    }

    #[test]
    fn test_from_mt_number_arbitrary() {
        let s = Score::from_mt_number(102).unwrap();
        assert_eq!(s.kind(), ScoreKind::ReactionRate);
        assert_eq!(s.to_i32(), 102);
    }

    #[test]
    fn test_from_mt_number_invalid() {
        assert!(Score::from_mt_number(0).is_err());
        assert!(Score::from_mt_number(-1).is_err());
        assert!(Score::from_mt_number(1000).is_err());
    }

    // ---- PhotonComponent ----

    #[test]
    fn test_photon_component_as_u8() {
        assert_eq!(PhotonComponent::Coherent.as_u8(), 0);
        assert_eq!(PhotonComponent::Incoherent.as_u8(), 1);
        assert_eq!(PhotonComponent::Photoelectric.as_u8(), 2);
        assert_eq!(PhotonComponent::PairProduction.as_u8(), 3);
    }

    #[test]
    fn test_photon_component_name() {
        assert_eq!(PhotonComponent::Coherent.name(), "coherent-scatter");
        assert_eq!(PhotonComponent::Incoherent.name(), "incoherent-scatter");
        assert_eq!(PhotonComponent::Photoelectric.name(), "photoelectric");
        assert_eq!(PhotonComponent::PairProduction.name(), "pair-production");
    }

    // ---- PhotonXSScore::mt() ----

    #[test]
    fn test_photon_xs_score_mt() {
        assert_eq!(
            PhotonXSScore {
                component: PhotonComponent::Coherent
            }
            .mt(),
            Mt::COHERENT
        );
        assert_eq!(
            PhotonXSScore {
                component: PhotonComponent::Incoherent
            }
            .mt(),
            Mt::INCOHERENT
        );
        assert_eq!(
            PhotonXSScore {
                component: PhotonComponent::Photoelectric
            }
            .mt(),
            Mt::PHOTOELECTRIC
        );
        assert_eq!(
            PhotonXSScore {
                component: PhotonComponent::PairProduction
            }
            .mt(),
            Mt::PAIR_PRODUCTION
        );
    }

    // ---- ReactionRateScore constructors ----

    #[test]
    fn test_reaction_rate_from_mt() {
        let rr = ReactionRateScore::from_mt(Mt::new(102));
        assert_eq!(rr.mt, Mt::new(102));
        assert!(rr.display_name.is_none());
    }

    #[test]
    fn test_reaction_rate_named() {
        let rr = ReactionRateScore::named(Mt::TOTAL, "total");
        assert_eq!(rr.mt, Mt::TOTAL);
        assert_eq!(rr.display_name, Some("total".to_string()));
    }

    // ---- ProductionScore constants ----

    #[test]
    fn test_production_score_constants() {
        assert_eq!(ProductionScore::H1.mt, Mt::H1_PRODUCTION);
        assert_eq!(ProductionScore::H1.name, "H1-production");
        assert_eq!(ProductionScore::H2.mt, Mt::H2_PRODUCTION);
        assert_eq!(ProductionScore::H3.mt, Mt::H3_PRODUCTION);
        assert_eq!(ProductionScore::HE3.mt, Mt::HE3_PRODUCTION);
        assert_eq!(ProductionScore::HE4.mt, Mt::HE4_PRODUCTION);
    }

    // ---- Round-trip: name -> parse -> name ----

    #[test]
    fn test_roundtrip_named_scores() {
        let names = [
            "flux",
            "heating",
            "heating-local",
            "damage-energy",
            "H1-production",
            "H2-production",
            "H3-production",
            "He3-production",
            "He4-production",
            "total",
            "elastic",
            "inelastic",
            "fission",
            "absorption",
            "coherent-scatter",
            "incoherent-scatter",
            "photoelectric",
            "pair-production",
        ];
        for name in &names {
            let parsed: Score = name.parse().unwrap();
            assert_eq!(parsed.name(), *name, "Round-trip failed for {name}");
        }
    }
}
