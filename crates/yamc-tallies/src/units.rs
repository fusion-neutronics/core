//! Units of tally results.
//!
//! Transport scores per source particle. A score's unit per source particle
//! comes from the score (flux is a track length in cm, heating an energy in
//! eV, a reaction rate a count) times the units of any `energy_function`
//! filter. [`SourceStrength`] turns a result per source particle into one per
//! second or per pulse, and [`conversion`] works out the factor from a
//! result's own unit to a unit string such as `"W/cm3"`, dividing by each
//! bin's volume when the target is per volume.
//!
//! Unit strings are parsed by `yamc-units`, which yani's material outputs
//! share.
use crate::filter::Filter;
use crate::score::Score;
use crate::tally::Tally;
use yamc_units::{describe, parse, Dims, Quantity, LENGTH, MASS, PULSE, TIME};
/// How a source strength is counted: source particles per second (steady
/// state) or per pulse (ICF, one shot).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StrengthPer {
    Second,
    Pulse,
}

impl StrengthPer {
    /// `"second"` or `"pulse"`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Second => "second",
            Self::Pulse => "pulse",
        }
    }

    /// Parse `"second"` / `"s"` or `"pulse"`.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "second" | "s" => Ok(Self::Second),
            "pulse" => Ok(Self::Pulse),
            other => Err(format!(
                "unknown per={other:?}: expected \"second\" (a steady source, particles per second) or \"pulse\" (particles per pulse)"
            )),
        }
    }

    fn symbol(self) -> &'static str {
        match self {
            Self::Second => "s",
            Self::Pulse => "pulse",
        }
    }
}

/// Source particles per second or per pulse. Results are linear in it, so it
/// is applied after transport.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SourceStrength {
    pub value: f64,
    pub per: StrengthPer,
}

impl SourceStrength {
    /// A strength, refused unless finite and positive.
    pub fn new(value: f64, per: StrengthPer) -> Result<Self, String> {
        if value.is_finite() && value > 0.0 {
            Ok(Self { value, per })
        } else {
            Err(format!(
                "source strength must be finite and positive, got {value}"
            ))
        }
    }

    fn quantity(self) -> Quantity {
        Quantity::base(
            1.0,
            if self.per == StrengthPer::Second {
                TIME
            } else {
                PULSE
            },
        )
        .powi(-1)
    }
}

impl std::fmt::Display for SourceStrength {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:e} per {}", self.value, self.per.name())
    }
}

/// The unit of a tally score per source particle, as text and as a
/// quantity. `None` for the quantity when an `energy_function` filter has no
/// units, or units this module does not know: such a result still reports a
/// unit string but cannot be converted.
struct ScoreUnit {
    text: String,
    quantity: Result<Quantity, String>,
}

fn score_unit(tally: &Tally, score: &Score) -> ScoreUnit {
    let base = score.base_units();
    let base_quantity = match base {
        "cm" => Quantity::base(1.0, LENGTH),
        "eV" => Quantity::base(1.0, 1),
        _ => Quantity::ONE,
    };
    let ef_units = tally.filters.iter().find_map(|f| match f {
        Filter::EnergyFunction(ef) => Some(ef.units.clone()),
        _ => None,
    });
    match ef_units {
        None => ScoreUnit {
            text: base.to_string(),
            quantity: Ok(base_quantity),
        },
        Some(None) => ScoreUnit {
            text: format!("(energy_function)·{base}"),
            quantity: Err(
                "the tally's energy_function has no units, so its results cannot be converted: give the energy_function its units".into(),
            ),
        },
        Some(Some(u)) => ScoreUnit {
            text: format!("{u}·{base}"),
            quantity: parse(&u)
                .map(|q| q * base_quantity)
                .map_err(|e| format!("energy_function units {u:?}: {e}")),
        },
    }
}

/// Per score, the unit of a tally's results with `strength` applied (or per
/// source particle without one): `"eV/s"`, `"cm/pulse"`,
/// `"reactions/source-particle"`.
pub fn result_units(tally: &Tally, strength: Option<SourceStrength>) -> Vec<String> {
    let per = strength.map_or("source-particle", |s| s.per.symbol());
    tally
        .scores
        .iter()
        .map(|s| format!("{}/{per}", score_unit(tally, s).text))
        .collect()
}

/// What converting a score's results to a unit takes: a factor, and what each
/// bin is also divided by.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Conversion {
    pub factor: f64,
    pub per: Per,
}

/// What a bin is divided by to reach a unit: nothing, its volume in cm³, or
/// its mass in g.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Per {
    Bin,
    Volume,
    Mass,
}

/// The conversion of `score`'s results, with `strength` applied, to `target`.
pub fn conversion(
    tally: &Tally,
    score: &Score,
    strength: Option<SourceStrength>,
    target: &str,
) -> Result<Conversion, String> {
    let unit = score_unit(tally, score);
    let mut from = unit.quantity?;
    if let Some(s) = strength {
        from = from * s.quantity();
    }
    let to = parse(target)?;
    let mut per_volume_dims = from.dims;
    per_volume_dims[LENGTH] -= 3;
    let mut per_mass_dims = from.dims;
    per_mass_dims[MASS] -= 1;
    let per = if to.dims == from.dims {
        Per::Bin
    } else if to.dims == per_volume_dims {
        Per::Volume
    } else if to.dims == per_mass_dims {
        Per::Mass
    } else {
        let have = format!(
            "{}/{}",
            unit.text,
            strength.map_or("source-particle", |s| s.per.symbol())
        );
        return Err(mismatch(&have, target, from.dims, to.dims, strength));
    };
    Ok(Conversion {
        factor: from.factor / to.factor,
        per,
    })
}

fn mismatch(
    have: &str,
    target: &str,
    from: Dims,
    to: Dims,
    strength: Option<SourceStrength>,
) -> String {
    let head = format!("cannot convert {have} to {target:?}");
    let wants_time = to[TIME] != 0 && from[TIME] == 0;
    let wants_pulse = to[PULSE] != 0 && from[PULSE] == 0;
    match strength {
        None if wants_time || wants_pulse => format!(
            "{head}: the result is per source particle; call with_source_strength(...) first"
        ),
        Some(s) if s.per == StrengthPer::Pulse && wants_time => format!(
            "{head}: the source strength is per pulse, so the result is per pulse (for example J/pulse, not W)"
        ),
        Some(s) if s.per == StrengthPer::Second && wants_pulse => format!(
            "{head}: the source strength is per second; pass per=\"pulse\" to with_source_strength for a per-pulse source"
        ),
        Some(s) if from[TIME] + from[PULSE] != to[TIME] + to[PULSE] => format!(
            "{head}: the source strength is per {}, so the unit must be per {} too (for example W or eV/s for a steady source, J/pulse for a pulsed one)",
            s.per.name(),
            s.per.name()
        ),
        _ => format!(
                "{head}: the result is {}, the unit is {} (a per-volume unit, such as /cm3, divides by each bin's volume, and a per-mass one, such as /g or Gy, by its mass)",
                describe(from),
                describe(to)
            ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::{FluxScore, HeatingScore, ReactionRateScore};
    use yamc_units::J_PER_EV;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * a.abs().max(b.abs())
    }

    fn tally(score: Score) -> Tally {
        let mut t = Tally::new();
        t.scores = vec![score];
        t
    }

    #[test]
    fn heating_converts_per_mass_and_to_gray() {
        let t = tally(Score::Heating(HeatingScore));
        let s = &t.scores[0];
        let per_s = Some(SourceStrength::new(1.0, StrengthPer::Second).unwrap());
        let c = conversion(&t, s, per_s, "W/g").unwrap();
        assert_eq!(c.per, Per::Mass);
        assert!(close(c.factor, J_PER_EV));
        // A gray is a J/kg, so eV/s per g is 1.602e-16 Gy/s.
        let c = conversion(&t, s, per_s, "Gy/s").unwrap();
        assert_eq!(c.per, Per::Mass);
        assert!(close(c.factor, J_PER_EV * 1000.0));
        assert!(conversion(&t, s, None, "Gy").unwrap().per == Per::Mass);
    }

    #[test]
    fn heating_converts_with_and_without_strength() {
        let t = tally(Score::Heating(HeatingScore));
        let s = &t.scores[0];
        assert!(close(conversion(&t, s, None, "MeV").unwrap().factor, 1e-6));
        let per_s = SourceStrength::new(5e15, StrengthPer::Second).unwrap();
        let c = conversion(&t, s, Some(per_s), "W").unwrap();
        assert!(close(c.factor, J_PER_EV));
        assert_eq!(c.per, Per::Bin);
        assert_eq!(
            conversion(&t, s, Some(per_s), "W/cm3").unwrap().per,
            Per::Volume
        );
        let err = conversion(&t, s, None, "W").unwrap_err();
        assert!(err.contains("with_source_strength"), "{err}");
        let per_pulse = SourceStrength::new(1e19, StrengthPer::Pulse).unwrap();
        assert!(close(
            conversion(&t, s, Some(per_pulse), "J/pulse")
                .unwrap()
                .factor,
            J_PER_EV
        ));
        let err = conversion(&t, s, Some(per_pulse), "W").unwrap_err();
        assert!(err.contains("per pulse"), "{err}");
        let err = conversion(&t, s, Some(per_s), "J").unwrap_err();
        assert!(err.contains("per second"), "{err}");
    }

    #[test]
    fn flux_is_a_track_length_until_divided_by_volume() {
        let t = tally(Score::Flux(FluxScore));
        let per_s = SourceStrength::new(1.0, StrengthPer::Second).unwrap();
        let c = conversion(&t, &t.scores[0], Some(per_s), "1/(cm2 s)").unwrap();
        assert!(c.per == Per::Volume && close(c.factor, 1.0));
        let c = conversion(&t, &t.scores[0], Some(per_s), "m/s").unwrap();
        assert!(c.per == Per::Bin && close(c.factor, 0.01));
        let err = conversion(&t, &t.scores[0], Some(per_s), "eV/s").unwrap_err();
        assert!(err.contains("length"), "{err}");
    }

    #[test]
    fn dose_coefficients_give_dose_rate_per_volume() {
        use crate::filter::energy_function::EnergyFunctionFilter;
        let mut t = tally(Score::Flux(FluxScore));
        t.filters
            .push(Filter::EnergyFunction(EnergyFunctionFilter::with_units(
                vec![1.0, 2.0, 3.0, 4.0],
                vec![1.0; 4],
                crate::Interpolation::Cubic,
                "pSv·cm²",
            )));
        let per_s = SourceStrength::new(1.0, StrengthPer::Second).unwrap();
        let c = conversion(&t, &t.scores[0], Some(per_s), "Sv/h").unwrap();
        assert_eq!(c.per, Per::Volume);
        assert!(close(c.factor, 1e-12 * 3600.0));
        assert_eq!(result_units(&t, Some(per_s)), vec!["pSv·cm²·cm/s"]);
    }

    #[test]
    fn reaction_rates_are_counts() {
        let t = tally(Score::ReactionRate(ReactionRateScore::total()));
        assert_eq!(result_units(&t, None), vec!["reactions/source-particle"]);
        let per_s = SourceStrength::new(2.0, StrengthPer::Second).unwrap();
        assert_eq!(result_units(&t, Some(per_s)), vec!["reactions/s"]);
        let c = conversion(&t, &t.scores[0], Some(per_s), "1/(cm3 s)").unwrap();
        assert_eq!(c.per, Per::Volume);
    }

    #[test]
    fn strength_must_be_positive() {
        assert!(SourceStrength::new(0.0, StrengthPer::Second).is_err());
        assert!(SourceStrength::new(f64::NAN, StrengthPer::Pulse).is_err());
        assert!(StrengthPer::parse("hour").is_err());
    }
}
