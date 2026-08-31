//! Core irradiation/cooling timeline for transmutation.
//!
//! A [`Schedule`] is the numeric `(rate, dt)` timeline that drives a
//! transmutation calculation: each step is either an irradiation pulse (a
//! source rate held for a duration) or a decay-only cooldown. The Python
//! `PulseSchedule` wrapper builds one of these and adds the per-pulse neutron
//! source objects it needs for decay-photon shutdown-dose identity grouping;
//! all of the timeline logic (duration units, derivation, validation) lives
//! here in the core.

/// Convert a duration of `value` in `unit` to seconds.
///
/// Units: `s`, `min`, `h`, `d`, `a` (Julian year, 365.25 days), plus their
/// spelled-out aliases. `value` must be non-negative.
pub fn duration_to_seconds(value: f64, unit: &str) -> Result<f64, String> {
    if !value.is_finite() || value < 0.0 {
        return Err("duration value must be a finite, non-negative number".to_string());
    }
    let per_unit = match unit {
        "s" | "sec" | "second" | "seconds" => 1.0,
        "min" | "minute" | "minutes" => 60.0,
        "h" | "hr" | "hour" | "hours" => 3600.0,
        "d" | "day" | "days" => 86400.0,
        "a" | "y" | "yr" | "year" | "years" => 365.25 * 86400.0,
        other => {
            return Err(format!(
                "unknown duration unit {other:?}; use one of: s, min, h, d, a (year)"
            ))
        }
    };
    Ok(value * per_unit)
}

/// One step of an irradiation/cooling timeline.
#[derive(Clone, Debug)]
pub struct ScheduleStep {
    /// Source strength in n/s for a transport schedule, or a dimensionless flux
    /// multiplier for standalone `Material::transmute`. `0.0` for a decay-only
    /// step (a cooldown, or a pulse with rate 0).
    pub rate: f64,
    /// Step duration in seconds.
    pub dt: f64,
    /// `true` for an irradiation pulse, `false` for a cooldown. A sourceless
    /// pulse is still a pulse; used for display/diagnostics only.
    pub is_pulse: bool,
}

/// An irradiation/cooling timeline: the validated `(rate, dt)` steps that drive
/// a transmutation calculation.
#[derive(Clone, Debug)]
pub struct Schedule {
    steps: Vec<ScheduleStep>,
}

impl Schedule {
    /// Build a schedule, requiring at least one step.
    pub fn new(steps: Vec<ScheduleStep>) -> Result<Self, String> {
        if steps.is_empty() {
            return Err("schedule must have at least one step".to_string());
        }
        Ok(Self { steps })
    }

    /// Number of steps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether the schedule has no steps (always `false` for a constructed
    /// `Schedule`; present for lint completeness).
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// The steps in order.
    pub fn steps(&self) -> &[ScheduleStep] {
        &self.steps
    }

    /// Per-step durations (seconds), in order.
    pub fn timesteps(&self) -> Vec<f64> {
        self.steps.iter().map(|s| s.dt).collect()
    }

    /// Per-step rates, in order.
    pub fn source_rates(&self) -> Vec<f64> {
        self.steps.iter().map(|s| s.rate).collect()
    }

    /// Number of irradiation pulses (steps with `is_pulse == true`).
    pub fn pulse_count(&self) -> usize {
        self.steps.iter().filter(|s| s.is_pulse).count()
    }

    /// Elapsed time (seconds) at the end of each step, i.e. the running sum of
    /// step durations.
    pub fn cumulative_times(&self) -> Vec<f64> {
        let mut acc = 0.0;
        self.steps
            .iter()
            .map(|s| {
                acc += s.dt;
                acc
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_units() {
        assert_eq!(duration_to_seconds(1.0, "h").unwrap(), 3600.0);
        assert_eq!(duration_to_seconds(2.0, "d").unwrap(), 2.0 * 86400.0);
        assert_eq!(duration_to_seconds(1.0, "a").unwrap(), 365.25 * 86400.0);
        assert_eq!(duration_to_seconds(30.0, "s").unwrap(), 30.0);
    }

    #[test]
    fn duration_rejects_unknown_unit_and_negative() {
        assert!(duration_to_seconds(1.0, "fortnight").is_err());
        assert!(duration_to_seconds(-1.0, "s").is_err());
    }

    #[test]
    fn duration_rejects_non_finite() {
        assert!(duration_to_seconds(f64::NAN, "s").is_err());
        assert!(duration_to_seconds(f64::INFINITY, "s").is_err());
    }

    #[test]
    fn schedule_derivations() {
        let sched = Schedule::new(vec![
            ScheduleStep {
                rate: 1e20,
                dt: 3600.0,
                is_pulse: true,
            },
            ScheduleStep {
                rate: 0.0,
                dt: 7200.0,
                is_pulse: false,
            },
        ])
        .unwrap();
        assert_eq!(sched.len(), 2);
        assert_eq!(sched.timesteps(), vec![3600.0, 7200.0]);
        assert_eq!(sched.source_rates(), vec![1e20, 0.0]);
        assert_eq!(sched.cumulative_times(), vec![3600.0, 10800.0]);
        assert_eq!(sched.pulse_count(), 1);
    }

    #[test]
    fn empty_schedule_rejected() {
        assert!(Schedule::new(vec![]).is_err());
    }
}
