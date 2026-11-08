//! Precision-based stopping criteria for a simulation.
//!
//! A [`ConvergenceTarget`] ends a run early once a tally's chosen statistic
//! drops to or below a threshold. Targets are evaluated at each run checkpoint
//! against a tally's aggregate (per-history total) statistics.

use crate::tally::Tally;

/// Which aggregate statistic a [`ConvergenceTarget`] watches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceMetric {
    /// Aggregate relative error (standard error of the mean / mean).
    RelativeError,
    /// Aggregate standard error of the mean.
    StandardDeviation,
    /// Aggregate variance of the variance.
    VarianceOfVariance,
}

/// Which tally a [`ConvergenceTarget`] applies to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TallySelector {
    /// Apply to the tally with this name.
    Name(String),
    /// Apply to the tally with this id.
    Id(u32),
}

/// A criterion for finishing a simulation early once a tally is precise
/// enough. The run stops at the first checkpoint where every active
/// convergence target is satisfied.
#[derive(Debug, Clone)]
pub struct ConvergenceTarget {
    /// Statistic to watch.
    pub metric: ConvergenceMetric,
    /// Stop once the metric is at or below this value.
    pub threshold: f64,
    /// Tally to watch; `None` means the threshold must hold for every tally.
    pub tally: Option<TallySelector>,
}

impl ConvergenceTarget {
    /// A convergence target on `metric <= threshold` applying to every tally.
    pub fn new(metric: ConvergenceMetric, threshold: f64) -> Self {
        Self {
            metric,
            threshold,
            tally: None,
        }
    }

    /// Restrict this target to the tally with the given name.
    pub fn for_name(mut self, name: impl Into<String>) -> Self {
        self.tally = Some(TallySelector::Name(name.into()));
        self
    }

    /// Restrict this target to the tally with the given id.
    pub fn for_id(mut self, id: u32) -> Self {
        self.tally = Some(TallySelector::Id(id));
        self
    }

    /// Whether this convergence target applies to `tally`.
    pub fn targets(&self, tally: &Tally) -> bool {
        match &self.tally {
            None => true,
            Some(TallySelector::Name(n)) => tally.name.as_deref() == Some(n.as_str()),
            Some(TallySelector::Id(id)) => tally.tally_id == Some(*id),
        }
    }

    /// Whether the current metric `value` meets this target. A non-positive
    /// or non-finite value (not yet estimable) is treated as not satisfied,
    /// so a run never stops before the statistic is meaningful.
    pub fn is_satisfied_by(&self, value: f64) -> bool {
        value > 0.0 && value.is_finite() && value <= self.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tally::Tally;

    fn named_tally(name: &str, id: u32) -> Tally {
        let mut t = Tally::new();
        t.name = Some(name.to_string());
        t.tally_id = Some(id);
        t
    }

    #[test]
    fn new_targets_every_tally_with_no_selector() {
        let target = ConvergenceTarget::new(ConvergenceMetric::RelativeError, 0.05);
        assert_eq!(target.threshold, 0.05);
        assert_eq!(target.metric, ConvergenceMetric::RelativeError);
        assert!(target.tally.is_none());
        // With no selector it applies to any tally.
        assert!(target.targets(&named_tally("flux", 1)));
        assert!(target.targets(&Tally::new()));
    }

    #[test]
    fn for_name_targets_only_the_matching_name() {
        let target =
            ConvergenceTarget::new(ConvergenceMetric::StandardDeviation, 1.0).for_name("flux");
        assert_eq!(target.tally, Some(TallySelector::Name("flux".to_string())));
        assert!(target.targets(&named_tally("flux", 1)));
        assert!(!target.targets(&named_tally("heating", 1)));
        assert!(!target.targets(&Tally::new())); // unnamed never matches a name
    }

    #[test]
    fn for_id_targets_only_the_matching_id() {
        let target = ConvergenceTarget::new(ConvergenceMetric::VarianceOfVariance, 0.1).for_id(7);
        assert_eq!(target.tally, Some(TallySelector::Id(7)));
        assert!(target.targets(&named_tally("flux", 7)));
        assert!(!target.targets(&named_tally("flux", 8)));
        assert!(!target.targets(&Tally::new())); // no id never matches an id
    }

    #[test]
    fn is_satisfied_by_threshold_edges() {
        let target = ConvergenceTarget::new(ConvergenceMetric::RelativeError, 0.05);
        assert!(target.is_satisfied_by(0.04)); // below threshold
        assert!(target.is_satisfied_by(0.05)); // exactly at threshold (<=)
        assert!(!target.is_satisfied_by(0.06)); // above threshold
    }

    #[test]
    fn is_satisfied_by_rejects_not_yet_estimable_values() {
        let target = ConvergenceTarget::new(ConvergenceMetric::RelativeError, 0.05);
        assert!(!target.is_satisfied_by(0.0)); // non-positive
        assert!(!target.is_satisfied_by(-1.0)); // negative
        assert!(!target.is_satisfied_by(f64::NAN)); // not finite
        assert!(!target.is_satisfied_by(f64::INFINITY));
    }
}
