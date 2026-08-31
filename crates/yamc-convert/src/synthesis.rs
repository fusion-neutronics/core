//! The redundant MTs, built from their partials.
//!
//! An evaluation gives the partial channels; a transport code wants the sums.
//! MT 1 (total), 3 (non-elastic), 4 (inelastic), 27 (absorption) and 101
//! (disappearance) are therefore synthesized at conversion time rather than at
//! every simulation start.
//!
//! The sets below are taken from the ENDF summation rules in [`endf::data`]
//! rather than written out here. That is not tidiness: a hand-written copy of
//! the MT 101 set once included thirteen channels that re-emit a neutron, such
//! as (n,npd), (n,2n2p) and (n,n3p), which do not belong in a disappearance
//! sum. They were counted into MT 101, and through it into MT 27, MT 3 and
//! MT 1, so a nuclide's total came out above its own evaluated total. Deriving
//! the sets from one source means a change to the rules cannot leave two
//! definitions disagreeing.

use std::collections::{BTreeMap, BTreeSet};

use endf::data::sum_rule;

/// The MTs this module builds. Excluded from every sum, since summing a sum
/// double counts.
pub const SYNTHETIC_MTS: [i32; 5] = [1, 3, 4, 27, 101];

/// Channels that emit a neutron and are not level inelastic.
///
/// The candidates, less anything the disappearance rule claims: (n,2n) and
/// friends belong here, and the (n,npd)-shaped channels belong in exactly one
/// of the two groups, never both.
fn scattering_non_inelastic() -> BTreeSet<i32> {
    let mut set: BTreeSet<i32> = [
        2, 5, 11, 16, 17, 22, 23, 24, 25, 28, 29, 30, 32, 33, 34, 35, 36, 37, 41, 42, 44, 45,
    ]
    .into_iter()
    .chain(152..201)
    .chain(875..891)
    .collect();
    for mt in absorption() {
        set.remove(&mt);
    }
    set
}

/// Level inelastic scattering, MT 50 to 91.
fn inelastic() -> BTreeSet<i32> {
    sum_rule(4).unwrap_or(&[]).iter().copied().collect()
}

/// Neutron disappearance: capture and charged-particle-out channels.
fn absorption() -> BTreeSet<i32> {
    sum_rule(101).unwrap_or(&[]).iter().copied().collect()
}

/// The fission total and its partials.
fn fission() -> BTreeSet<i32> {
    let mut set: BTreeSet<i32> = sum_rule(18).unwrap_or(&[]).iter().copied().collect();
    set.insert(18);
    set
}

/// Whether an MT emits a neutron, either as level inelastic or otherwise.
pub fn is_scattering(mt: i32) -> bool {
    inelastic().contains(&mt) || scattering_non_inelastic().contains(&mt)
}

/// The three groups MT 3 is built from must be disjoint, or a channel is
/// counted twice and every sum above it is wrong.
///
/// Checked at run time rather than left as a comment, because the failure is
/// silent: the numbers stay plausible and only disagree with the evaluation's
/// own total by the size of the double-counted channels.
pub fn groups_are_disjoint() -> Result<(), String> {
    let (inel, scat, abs, fiss) = (
        inelastic(),
        scattering_non_inelastic(),
        absorption(),
        fission(),
    );
    let overlap =
        |a: &BTreeSet<i32>, b: &BTreeSet<i32>| -> Vec<i32> { a.intersection(b).copied().collect() };
    for (what, mts) in [
        ("disappearance and scattering", overlap(&abs, &scat)),
        ("disappearance and inelastic", overlap(&abs, &inel)),
        ("disappearance and fission", overlap(&abs, &fiss)),
        ("scattering and fission", overlap(&scat, &fiss)),
        ("inelastic and fission", overlap(&inel, &fiss)),
    ] {
        if !mts.is_empty() {
            return Err(format!(
                "the {what} sets share {mts:?}; MT 3 would double count them"
            ));
        }
    }
    Ok(())
}

/// Place a reaction's cross section on the nuclide's full energy grid.
///
/// A threshold reaction's array starts partway up the grid, at the index the
/// parser recorded, and is zero below it. Getting this offset wrong shifts a
/// whole cross section in energy, which no schema check would notice.
pub fn on_grid(values: &[f64], threshold_idx: usize, n_energy: usize) -> Vec<f64> {
    let mut out = vec![0.0; n_energy];
    if values.is_empty() || threshold_idx >= n_energy {
        return out;
    }
    let n = values.len().min(n_energy - threshold_idx);
    out[threshold_idx..threshold_idx + n].copy_from_slice(&values[..n]);
    out
}

/// Build the redundant MTs from the partials already on the grid.
///
/// `partials` maps MT to a cross section on the full energy grid, and must
/// exclude [`SYNTHETIC_MTS`]: an evaluation carries its own MT 1 and often its
/// own MT 101, and summing those back in would double count everything under
/// them.
/// Whether this MT is a total whose own residual levels are also present.
///
/// MT 16 and MT 875 to 890 both describe (n,2n): the first is the total, the
/// rest are the levels of the residual nucleus. An evaluation may carry both,
/// and adding both counts the reaction twice.
///
/// JEFF-4.0's Be9 is the case that found this. Its evaluation has no MF=3
/// MT 16 at all, so the parser builds MT 16 as the sum of MT 875 to 890 and
/// flags it redundant; summing the flat scattering set then added the sum AND
/// its parts, putting MT 3 (and so the total) 35% above the evaluation's own
/// MT 1 at 7.5 MeV. A total cross section larger than the evaluated total is
/// impossible, which is what makes this a defect rather than a convention.
///
/// One nuclide in 240 published files trips it, which is why a diff against
/// published data could not find it: the Python converter does the same thing,
/// so both were wrong together. OpenMC is immune because it skips redundant
/// reactions before summing (`src/nuclide.cpp`, `if (rx->redundant_) continue`).
fn covered_by_its_own_levels(mt: i32, partials: &BTreeMap<i32, Vec<f64>>) -> bool {
    match sum_rule(mt) {
        Some(rule) => rule.iter().any(|c| partials.contains_key(c)),
        None => false,
    }
}

/// The neutron-emitting channels other than elastic, summed.
///
/// MT 3 is this plus fission plus disappearance, and the `scattering` column of
/// fast_xs.arrow is this plus elastic. Both callers take the sum from here so a
/// change to the rules cannot leave the two disagreeing, the same reason the
/// sets above come from the ENDF rules rather than from a hand-written copy.
///
/// Exposed rather than inlined into [`synthesize`] because the difference is
/// not only tidiness: fast_xs.arrow used to reach the same quantity as
/// `MT 3 - MT 27`, which cancels the absorption back out and loses the low bits
/// of everything smaller than it. See `fast_xs::write_fast_xs`.
pub fn non_elastic_scattering(partials: &BTreeMap<i32, Vec<f64>>, n_energy: usize) -> Vec<f64> {
    let mut out = vec![0.0; n_energy];
    for (&mt, xs) in partials {
        if SYNTHETIC_MTS.contains(&mt) {
            continue;
        }
        // Skip a total whose own levels are in the same sum: taking both
        // counts the channel twice. Same rule the fission branch below uses.
        if mt != 2 && is_scattering(mt) && !covered_by_its_own_levels(mt, partials) {
            for (o, v) in out.iter_mut().zip(xs) {
                *o += v;
            }
        }
    }
    out
}

pub fn synthesize(partials: &BTreeMap<i32, Vec<f64>>, n_energy: usize) -> BTreeMap<i32, Vec<f64>> {
    let zeros = || vec![0.0; n_energy];
    let add = |acc: &mut Vec<f64>, xs: &[f64]| {
        for (a, b) in acc.iter_mut().zip(xs) {
            *a += b;
        }
    };

    let mut out = BTreeMap::new();

    let mut mt4 = zeros();
    for mt in inelastic() {
        if let Some(xs) = partials.get(&mt) {
            add(&mut mt4, xs);
        }
    }

    let mut mt101 = zeros();
    for mt in absorption() {
        if let Some(xs) = partials.get(&mt) {
            add(&mut mt101, xs);
        }
    }

    // MT 18 where the evaluation gives it, otherwise its partials. Never both,
    // which is why this is not a plain sum over the fission set.
    let mut fission_xs = zeros();
    match partials.get(&18) {
        Some(xs) => fission_xs.copy_from_slice(xs),
        None => {
            for mt in [19, 20, 21, 38] {
                if let Some(xs) = partials.get(&mt) {
                    add(&mut fission_xs, xs);
                }
            }
        }
    }

    let mut mt27 = fission_xs.clone();
    add(&mut mt27, &mt101);

    // Non-elastic: every neutron-emitting channel except elastic, plus fission,
    // plus disappearance. Built from the partials present rather than from the
    // set, so a channel the evaluation does not carry contributes nothing.
    let mut mt3 = non_elastic_scattering(partials, n_energy);
    add(&mut mt3, &fission_xs);
    add(&mut mt3, &mt101);

    let mut mt1 = partials.get(&2).cloned().unwrap_or_else(zeros);
    add(&mut mt1, &mt3);

    out.insert(1, mt1);
    out.insert(3, mt3);
    out.insert(4, mt4);
    out.insert(27, mt27);
    out.insert(101, mt101);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sets MT 3 is built from must not overlap.
    ///
    /// This is the check that would have caught issue #23 the day it was
    /// written, instead of after it reached published data.
    #[test]
    fn the_summation_groups_are_disjoint() {
        groups_are_disjoint().expect("the ENDF summation rules partition these channels");
    }

    /// The disappearance set must not contain a channel that re-emits a
    /// neutron. Pinned by name, since these are the exact thirteen that were
    /// wrong before.
    #[test]
    fn disappearance_excludes_neutron_emitting_channels() {
        let abs = absorption();
        for mt in [11, 16, 17, 22, 23, 24, 25, 28, 29, 30, 32, 33, 34] {
            assert!(
                !abs.contains(&mt),
                "MT {mt} re-emits a neutron and cannot be a disappearance channel"
            );
        }
    }

    /// A huge absorption must not round away the scattering beside it.
    ///
    /// `scattering` in fast_xs.arrow was once reached as MT 2 + MT 3 - MT 27,
    /// which cancels the absorption back out through an intermediate the size
    /// of MT 3. These are TENDL-2025 Mo86's partials at 1.125e-5 eV: 2e12 barns
    /// of (n,p) and (n,alpha) against 37.8 barns of elastic, where that
    /// intermediate has an ulp of 4.9e-4 barns. The subtraction lands BELOW
    /// elastic, which a cross section that elastic is a part of cannot be.
    #[test]
    fn a_huge_absorption_does_not_round_away_the_scattering() {
        let elastic = 37.78729_f64;
        let partials: BTreeMap<i32, Vec<f64>> = [
            (2, vec![elastic]),
            (102, vec![1070.986]),
            (103, vec![2_053_910_000_000.0]),
            (107, vec![2_059_280_000_000.0]),
            (111, vec![416_142.1]),
        ]
        .into_iter()
        .collect();

        let derived = synthesize(&partials, 1);
        // No neutron-emitting channel but elastic is open this low, so the
        // scattering cross section IS the elastic one, exactly.
        let direct = elastic + non_elastic_scattering(&partials, 1)[0];
        assert_eq!(direct, elastic);

        // Pin the arithmetic that used to be here, so this cannot come back
        // looking like a harmless simplification.
        let by_subtraction = elastic + derived[&3][0] - derived[&27][0];
        assert!(
            by_subtraction < elastic,
            "MT 3 - MT 27 is supposed to lose precision here; if it no longer \
             does, this test is pinning nothing"
        );
    }

    #[test]
    fn a_threshold_reaction_is_zero_below_its_threshold() {
        let xs = on_grid(&[1.0, 2.0, 3.0], 2, 6);
        assert_eq!(xs, vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0]);
    }

    /// A cross section longer than the grid above its threshold is clipped
    /// rather than panicking, matching the Python writer.
    #[test]
    fn an_overlong_cross_section_is_clipped() {
        let xs = on_grid(&[1.0, 2.0, 3.0, 4.0], 2, 4);
        assert_eq!(xs, vec![0.0, 0.0, 1.0, 2.0]);
    }

    /// Total equals elastic plus non-elastic, and non-elastic equals the sum of
    /// everything else, on a hand-built nuclide where the answer is known.
    #[test]
    fn the_sums_add_up() {
        let n = 3;
        let mut partials = BTreeMap::new();
        partials.insert(2, vec![10.0; n]); // elastic
        partials.insert(51, vec![1.0; n]); // level inelastic
        partials.insert(16, vec![2.0; n]); // (n,2n), scattering
        partials.insert(102, vec![4.0; n]); // capture, disappearance
        let out = synthesize(&partials, n);

        assert_eq!(out[&4], vec![1.0; n], "MT 4 is the inelastic levels");
        assert_eq!(out[&101], vec![4.0; n], "MT 101 is the disappearance sum");
        assert_eq!(
            out[&27],
            vec![4.0; n],
            "MT 27 is fission plus disappearance"
        );
        assert_eq!(out[&3], vec![7.0; n], "MT 3 is 1 + 2 + 4");
        assert_eq!(out[&1], vec![17.0; n], "MT 1 is elastic plus non-elastic");
    }

    /// A redundant MT already present in the evaluation must not be summed back
    /// in. This is the double count that inflates every sum above it.
    #[test]
    fn an_evaluations_own_redundant_mts_are_not_summed_back_in() {
        let n = 2;
        let mut partials = BTreeMap::new();
        partials.insert(2, vec![10.0; n]);
        partials.insert(102, vec![4.0; n]);
        // The caller is required to exclude these; assert the sum is unchanged
        // if one slips through the door the other side.
        partials.insert(101, vec![999.0; n]);
        partials.insert(1, vec![999.0; n]);
        let out = synthesize(&partials, n);
        assert_eq!(out[&1], vec![14.0; n], "a stray MT 1 or 101 was summed in");
    }

    /// Fission comes from MT 18 where it exists, not from 18 and its partials.
    #[test]
    fn fission_is_not_counted_twice() {
        let n = 2;
        let mut partials = BTreeMap::new();
        partials.insert(18, vec![5.0; n]);
        partials.insert(19, vec![3.0; n]);
        partials.insert(20, vec![2.0; n]);
        let out = synthesize(&partials, n);
        assert_eq!(
            out[&27],
            vec![5.0; n],
            "MT 18 and its partials were both counted"
        );
    }
}

#[cfg(test)]
mod double_count_tests {
    use super::*;

    /// A total and its own residual levels must not both enter MT 3.
    ///
    /// The shape that found this: JEFF-4.0's Be9 has no evaluated MT 16, so the
    /// parser builds it as the sum of MT 875 to 890. Adding both the sum and
    /// its parts put the total cross section 35% above the evaluation's own
    /// MT 1, which cannot happen. Both converters did it, so a diff against the
    /// published data could not see it; only the physical bound could.
    #[test]
    fn a_total_and_its_own_levels_are_not_both_summed() {
        let n = 3;
        let mut partials: BTreeMap<i32, Vec<f64>> = BTreeMap::new();
        partials.insert(2, vec![10.0; n]); // elastic
                                           // (n,2n) given BOTH ways: the total and the two levels it sums.
        partials.insert(875, vec![1.0; n]);
        partials.insert(876, vec![2.0; n]);
        partials.insert(16, vec![3.0; n]);

        let out = synthesize(&partials, n);
        let mt3 = out.get(&3).expect("MT 3 is synthesized");
        assert_eq!(
            mt3[0], 3.0,
            "MT 3 must count (n,2n) once. Got {}, which is the 3 barns of \
             MT 875+876 plus another 3 from MT 16.",
            mt3[0]
        );

        // And the total must not exceed elastic plus one copy of the channel.
        let mt1 = out.get(&1).expect("MT 1 is synthesized");
        assert_eq!(mt1[0], 13.0, "MT 1 must be elastic plus MT 3, counted once");
    }

    /// With only the total present it is used, since there are no levels.
    #[test]
    fn a_total_with_no_levels_present_is_still_counted() {
        let n = 2;
        let mut partials: BTreeMap<i32, Vec<f64>> = BTreeMap::new();
        partials.insert(2, vec![10.0; n]);
        partials.insert(16, vec![3.0; n]);
        let out = synthesize(&partials, n);
        assert_eq!(
            out.get(&3).unwrap()[0],
            3.0,
            "MT 16 is the only form of (n,2n) here and must be counted"
        );
    }
}
