//! Decay branching ratio uncertainty, for the parents where the evaluation
//! fixes the joint distribution exactly (issue #140).
//!
//! MT=457 gives each decay mode a ratio and a sigma, and no covariance between
//! the modes. The ratios of a parent sum to a fixed total, so the modes cannot
//! be independent, and how their errors are shared is only determined by the
//! data where there is one degree of freedom: a parent with exactly two modes.
//! There `r1 = b1 + sigma z` and `r2 = T - r1` with `T = b1 + b2` is the whole
//! distribution, and it holds when
//!
//! - both modes state the same sigma, which is how an evaluator writes one
//!   number for a complementary pair (Bi212 0.3594 and 0.6406, both 0.0006),
//!   or only one mode states a sigma, the other being its complement by the sum
//!   rule (K35 p 0.0037 +- 0.0015 beside EC 0.9963 with none); and
//! - the smaller ratio is at least [`MIN_SIGMAS`] sigmas from zero, so the
//!   Gaussian the sigma describes stays inside `[0, T]` (the chance of a draw
//!   outside is below 3e-7).
//!
//! Every other parent is held at nominal and reported by why: three or more
//! modes with a sigma (the split of the error between them is not stated),
//! two modes with different sigmas (a sum rule allows only one), two modes too
//! wide to sample without truncating, and several modes with no sigma at all.
//! A spontaneous-fission mode is a mode like any other here: it is a share of
//! the parent's decays, whether or not the chain models a product for it.
//!
//! Only the inventory moves. A parent's per-decay emission (its lines and
//! decay energy) is the decay scheme's and stays nominal, since no correlation
//! between a ratio and an absolute line intensity is published, and the report
//! lists it under `not_perturbed`.

use std::collections::{BTreeSet, HashMap};

use yani::ChainNuclide;

use crate::covariance_sample::{name_ordinal, standard_normals};

/// Keeps the decay-branching streams clear of every other per-nuclide stream.
pub(crate) const DECAY_BRANCHING_STREAM: u32 = 0xDB2A_0C5E;

/// How many sigmas the smaller ratio must stand from zero to be sampled.
pub(crate) const MIN_SIGMAS: f64 = 5.0;

/// Two stated sigmas closer than this, relatively, are the same number.
const SAME_SIGMA: f64 = 1.0e-9;

/// One two-mode parent the sum rule lets us sample.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TwoModes {
    pub(crate) parent: String,
    /// The row of `decays` the deviate is applied to: the one stating the
    /// sigma, or the first where both state it.
    pub(crate) drawn: usize,
    /// The complementary row, which takes `total` minus the drawn ratio.
    pub(crate) other: usize,
    /// The drawn row's nominal ratio.
    pub(crate) ratio: f64,
    /// The two ratios' sum, which every replica keeps.
    pub(crate) total: f64,
    pub(crate) sigma: f64,
}

impl TwoModes {
    /// The smaller of the two nominal ratios.
    pub(crate) fn smaller(&self) -> f64 {
        self.ratio.min(self.total - self.ratio)
    }

    /// The parent's ratios, row by row, with the drawn row at `drawn_ratio`.
    fn ratios(&self, drawn_ratio: f64) -> Vec<f64> {
        let mut out = vec![0.0; 2];
        out[self.drawn] = drawn_ratio;
        out[self.other] = self.total - drawn_ratio;
        out
    }

    /// Ratios with the drawn row moved by `delta` and the other by `-delta`,
    /// for a sensitivity.
    pub(crate) fn shifted(&self, delta: f64) -> Vec<f64> {
        self.ratios(self.ratio + delta)
    }
}

/// The multi-mode parents of a chain, sorted into the one sampled case and the
/// reasons the rest are not.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Candidates {
    /// Sampled, sorted by parent.
    pub(crate) two_modes: Vec<TwoModes>,
    /// Several modes and no stated sigma on any.
    pub(crate) without: BTreeSet<String>,
    /// Three or more modes, at least one with a sigma.
    pub(crate) three_or_more_modes: BTreeSet<String>,
    /// Two modes stating different sigmas.
    pub(crate) unequal_sigmas: BTreeSet<String>,
    /// Two modes whose smaller ratio is under [`MIN_SIGMAS`] sigmas.
    pub(crate) too_wide: BTreeSet<String>,
}

/// Sort the unstable parents of `chain` with more than one decay mode.
///
/// A single mode is not listed anywhere: its ratio is the whole of the
/// parent's decays whatever sigma the tape writes beside it.
pub(crate) fn candidates(chain: &HashMap<String, ChainNuclide>) -> Candidates {
    let mut out = Candidates::default();
    for (name, cn) in chain {
        if cn.half_life.is_none_or(|t| t <= 0.0) || cn.decays.len() < 2 {
            continue;
        }
        // A stored 0.0 is MT=457's "not stated", as a null is.
        let sigmas: Vec<Option<f64>> = cn
            .decays
            .iter()
            .map(|d| d.branching_uncertainty.filter(|s| *s > 0.0))
            .collect();
        let stated: Vec<(usize, f64)> = sigmas
            .iter()
            .enumerate()
            .filter_map(|(j, s)| s.map(|s| (j, s)))
            .collect();
        if stated.is_empty() {
            out.without.insert(name.clone());
            continue;
        }
        if cn.decays.len() > 2 {
            out.three_or_more_modes.insert(name.clone());
            continue;
        }
        let (drawn, sigma) = stated[0];
        if let Some(&(_, second)) = stated.get(1) {
            if (sigma - second).abs() > SAME_SIGMA * sigma.max(second) {
                out.unequal_sigmas.insert(name.clone());
                continue;
            }
        }
        let other = 1 - drawn;
        let two = TwoModes {
            parent: name.clone(),
            drawn,
            other,
            ratio: cn.decays[drawn].branching,
            total: cn.decays[drawn].branching + cn.decays[other].branching,
            sigma,
        };
        if two.smaller() < MIN_SIGMAS * sigma {
            out.too_wide.insert(name.clone());
            continue;
        }
        out.two_modes.push(two);
    }
    out.two_modes.sort_by(|a, b| a.parent.cmp(&b.parent));
    out
}

/// One replica's ratios for every sampled parent, keyed by parent, one entry
/// per row of its `decays`.
///
/// One deviate per `(seed, replica, parent)`, on a stream keyed on the parent's
/// name, so a parent splits the same way in every spectrum and every material
/// of a run. A draw landing outside `[0, T]` is clamped to it and counted in
/// `floored`; the validity rule makes that a few-in-ten-million event.
pub(crate) fn sample(
    two_modes: &[TwoModes],
    base_seed: u64,
    replica: u64,
    floored: &mut usize,
) -> HashMap<String, Vec<f64>> {
    let replica_seed = yamc_rng::history_seed(base_seed, replica);
    two_modes
        .iter()
        .map(|t| {
            let seed = yamc_rng::secondary_seed(
                replica_seed,
                name_ordinal(&t.parent) ^ DECAY_BRANCHING_STREAM,
            );
            let mut state = yamc_rng::expand_seed(seed);
            let z = standard_normals(&mut state, 1)[0];
            let mut drawn = t.ratio + t.sigma * z;
            if !(0.0..=t.total).contains(&drawn) {
                *floored += 1;
                drawn = drawn.clamp(0.0, t.total);
            }
            (t.parent.clone(), t.ratios(drawn))
        })
        .collect()
}

/// Give a chain nuclide's decay modes the ratios in `ratios`, row by row.
///
/// Nothing else stored on the nuclide depends on a ratio: the matrix reads
/// `branching` for the target and the light particles alike, and the loss
/// stays the decay constant.
pub(crate) fn set_decay_branchings(cn: &mut ChainNuclide, ratios: &[f64]) {
    debug_assert_eq!(cn.decays.len(), ratios.len(), "{}", cn.name);
    for (d, r) in cn.decays.iter_mut().zip(ratios) {
        d.branching = *r;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mode(kind: &str, target: Option<&str>, b: f64, s: Option<f64>) -> yani::ChainReaction {
        yani::ChainReaction {
            kind: kind.to_string(),
            target: target.map(str::to_string),
            branching: b,
            q_value: None,
            branching_uncertainty: s,
        }
    }

    fn parent(name: &str, decays: Vec<yani::ChainReaction>) -> ChainNuclide {
        ChainNuclide {
            name: name.to_string(),
            half_life: Some(3600.0),
            half_life_uncertainty: None,
            decay_energy: 0.0,
            decay_energy_uncertainty: None,
            decay_energy_components: [None; 3],
            reactions: Vec::new(),
            decays,
            fission_yields: None,
            sources: Vec::new(),
        }
    }

    fn chain_of(nuclides: Vec<ChainNuclide>) -> HashMap<String, ChainNuclide> {
        nuclides.into_iter().map(|c| (c.name.clone(), c)).collect()
    }

    #[test]
    fn each_parent_lands_in_exactly_one_category() {
        let chain = chain_of(vec![
            // Equal sigmas: sampled on the first row.
            parent(
                "Equal",
                vec![
                    mode("beta-", Some("A"), 0.6406, Some(6e-4)),
                    mode("alpha", Some("B"), 0.3594, Some(6e-4)),
                ],
            ),
            // Only the minor mode states one, with the dominant one written as
            // 0.0: sampled on the stated row.
            parent(
                "OneSided",
                vec![
                    mode("ec/beta+", Some("A"), 0.9963, Some(0.0)),
                    mode("p", Some("B"), 0.0037, Some(5e-4)),
                ],
            ),
            parent(
                "Unequal",
                vec![
                    mode("beta-", Some("A"), 0.5, Some(0.01)),
                    mode("alpha", Some("B"), 0.5, Some(0.02)),
                ],
            ),
            // 0.0037 / 0.0015 is 2.5 sigmas.
            parent(
                "Wide",
                vec![
                    mode("ec/beta+", Some("A"), 0.9963, None),
                    mode("p", Some("B"), 0.0037, Some(0.0015)),
                ],
            ),
            parent(
                "Three",
                vec![
                    mode("beta-", Some("A"), 0.5, Some(0.01)),
                    mode("beta-,n", Some("B"), 0.3, None),
                    mode("beta-,2n", Some("C"), 0.2, None),
                ],
            ),
            parent(
                "Unstated",
                vec![
                    mode("beta-", Some("A"), 0.5, Some(0.0)),
                    mode("sf", None, 0.5, None),
                ],
            ),
            // A single mode's ratio is the whole: nothing to list.
            parent("Single", vec![mode("beta-", Some("A"), 1.0, Some(0.1))]),
        ]);
        let c = candidates(&chain);
        let sampled: Vec<(&str, usize)> = c
            .two_modes
            .iter()
            .map(|t| (t.parent.as_str(), t.drawn))
            .collect();
        assert_eq!(sampled, vec![("Equal", 0), ("OneSided", 1)]);
        assert_eq!(c.unequal_sigmas, BTreeSet::from(["Unequal".to_string()]));
        assert_eq!(c.too_wide, BTreeSet::from(["Wide".to_string()]));
        assert_eq!(c.three_or_more_modes, BTreeSet::from(["Three".to_string()]));
        assert_eq!(c.without, BTreeSet::from(["Unstated".to_string()]));
    }

    #[test]
    fn a_draw_keeps_the_pair_s_total_and_is_reproducible() {
        let t = TwoModes {
            parent: "Bi212".to_string(),
            drawn: 1,
            other: 0,
            ratio: 0.3594,
            total: 1.0,
            sigma: 6e-4,
        };
        let mut floored = 0;
        let a = sample(std::slice::from_ref(&t), 3, 17, &mut floored);
        let b = sample(std::slice::from_ref(&t), 3, 17, &mut floored);
        assert_eq!(a, b);
        let r = &a["Bi212"];
        assert!((r[0] + r[1] - 1.0).abs() <= f64::EPSILON, "{r:?}");
        assert_ne!(r[1], 0.3594, "the draw moved the ratio");
        assert_eq!(floored, 0);
    }

    #[test]
    fn the_stream_does_not_collide_with_any_other() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
        let chain = yani::parse_chain_arrow(&path).expect("parse chain");
        // Stream 0 is the cross sections, keyed on the bare name ordinal.
        let tags = [
            0,
            crate::uncertainty::HALF_LIFE_STREAM,
            crate::uncertainty::DECAY_ENERGY_STREAM,
            DECAY_BRANCHING_STREAM,
        ];
        let mut keys = std::collections::HashSet::new();
        for name in chain.keys() {
            for t in tags {
                let k = name_ordinal(name) ^ t;
                assert!(keys.insert(k), "{name} collides under tag {t:#x}");
                assert_ne!(
                    k,
                    crate::statistical::STATISTICAL_STREAM,
                    "{name} hits the statistical stream"
                );
                for i in 0..64u32 {
                    assert_ne!(
                        k,
                        crate::flux_uncertainty::FLUX_STREAM ^ i,
                        "{name} hits flux stream {i}"
                    );
                }
            }
        }
    }
}
