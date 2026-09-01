//! Every column of `products.arrow` and `distributions.arrow`, against the
//! parsed evaluation the writer was handed.
//!
//! # Why the writers are called directly
//!
//! `entry::convert_neutron_transport` is the only production caller of these
//! two writers, and it refuses an ACE source outright, so there is no hermetic
//! route to them through the entry point. Every test here hands
//! `write_products` and `write_distributions` the same `IncidentNeutron` the
//! entry point would have built.
//!
//! # Which fixture is here for what
//!
//! `Li6.ace.xz` is the only vendored input that drives both writers with a full
//! product set: 18 products, uncorrelated and Kalbach-Mann distributions, ACE
//! energy laws 2, 3/33 and 4, and both yield forms.
//! `n-092_U_235_trimmed.endf.xz` is the only source of delayed products with
//! real decay constants. `n-026_Fe_056_trimmed.endf.xz` is the zero-row
//! distributions case. `n-003_Li_006_trimmed.endf.xz` drives two of the three
//! refusals. `synthetic-laws.ace.xz` cannot be read as a nuclide at all, so its
//! seven laws are decoded per DLW locator and wrapped in a scaffold built here;
//! that is the only way any vendored bytes reach laws 7, 9, 11, 61 and 66.
//!
//! # Four inputs are built here rather than parsed
//!
//! `constructed_two_distribution_product`, `constructed_correlated_regions`,
//! `constructed_kalbach_mann_with_lines` and
//! `constructed_watt_with_distinct_grids` assemble an `IncidentNeutron` by
//! hand. Each exists because no vendored evaluation discriminates the column
//! it covers, each is named in the doc comment of the test that uses it, and
//! each gives its fields deliberately different values: two columns that hold
//! the same thing on a real fixture cannot discriminate a swap between them,
//! which is the only reason to build an input at all. None of the four is
//! evaluation parity and no number in them is claimed to be physical.
//!
//! # No ravel is recomputed by calling the writer's own flattener
//!
//! `yamc_convert::univariate_flat::flatten` and `distributions::ravel3` are
//! what these tests exist to check, so calling either to build an expectation
//! would restate the writer rather than test it. The `(x, p, c)` triples come
//! from `expect_flat` and `expect_mu` below, which are written from the
//! `endf::univariate` and `endf::mf::mf4` types directly.

mod section_values;
use section_values::*;

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use arrow_array::RecordBatch;
use endf::angle_energy::{
    AngleEnergy, CorrelatedAngleEnergy, KalbachMann, UncorrelatedAngleEnergy,
};
use endf::function::Tabulated1D;
use endf::mf::mf4::AngleAtEnergy;
use endf::mf::mf5::EnergyDistribution;
use endf::product::{Product, Yield};
use endf::univariate::{Discrete, Interpolation, Mixture, Tabular, Uniform, Univariate};
use endf::IncidentNeutron;

// ---------------------------------------------------------------------------
// Writing, and reading a whole column back.
// ---------------------------------------------------------------------------

fn products_of(data: &IncidentNeutron, dir: &Path) -> RecordBatch {
    yamc_convert::products::write_products(data, dir).expect("products.arrow is written");
    section(dir, "products.arrow")
}

fn distributions_of(data: &IncidentNeutron, dir: &Path) -> RecordBatch {
    yamc_convert::distributions::write_distributions(data, dir)
        .expect("distributions.arrow is written");
    section(dir, "distributions.arrow")
}

fn i32_column(batch: &RecordBatch, col: &str) -> Vec<i32> {
    (0..batch.num_rows())
        .map(|r| i32_at(batch, col, r))
        .collect()
}

fn str_column(batch: &RecordBatch, col: &str) -> Vec<String> {
    (0..batch.num_rows())
        .map(|r| str_at(batch, col, r))
        .collect()
}

/// The `(reaction_mt, product_idx, dist_idx)` key the reader joins on.
fn triples(batch: &RecordBatch) -> Vec<(i32, i32, i32)> {
    (0..batch.num_rows())
        .map(|r| {
            (
                i32_at(batch, "reaction_mt", r),
                i32_at(batch, "product_idx", r),
                i32_at(batch, "dist_idx", r),
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The traversal the two files share.
// ---------------------------------------------------------------------------

/// One entry per `products.arrow` row: MT, product index, product.
///
/// A restatement of `products::product_rows`, kept local so the row order the
/// later tests index by is the one
/// `product_rows_are_the_reaction_walk_in_mt_order` pins against the file.
fn product_walk(data: &IncidentNeutron) -> Vec<(i32, usize, &Product)> {
    let mut out = Vec::new();
    for (&mt, rx) in &data.reactions {
        for (idx, product) in rx.products.iter().enumerate() {
            out.push((mt, idx, product));
        }
    }
    out
}

/// One entry per `distributions.arrow` row: the key, the distribution and the
/// applicability that belongs to it.
type DistRow<'a> = (i32, usize, usize, &'a AngleEnergy, Option<&'a Tabulated1D>);

fn distribution_walk(data: &IncidentNeutron) -> Vec<DistRow<'_>> {
    let mut out = Vec::new();
    for (mt, product_idx, product) in product_walk(data) {
        for (dist_idx, dist) in product.distribution.iter().enumerate() {
            out.push((
                mt,
                product_idx,
                dist_idx,
                dist,
                product.applicability.get(dist_idx),
            ));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// What a sub-table must be written as, restated from the endf types.
// ---------------------------------------------------------------------------

/// The interpolation code the format draws its one distinction with.
fn interp_code(interp: Interpolation) -> i32 {
    match interp {
        Interpolation::Histogram => 1,
        _ => 2,
    }
}

/// The CDF the file supplied, which is the one the format must carry.
///
/// Panics rather than falling back, because the fallback is the converter's
/// own trapezoid integration: if it ever fires on a vendored fixture then the
/// comparisons below stop being a comparison against the file and this helper
/// has to grow a second branch, which is worth failing over rather than
/// absorbing.
fn stored_cdf(what: &str, c: &Option<Vec<f64>>, n: usize) -> Vec<f64> {
    match c {
        Some(c) if c.len() == n => c.clone(),
        _ => panic!(
            "{what}: the parsed table carries no CDF of its own length, so the writer \
             integrated one instead and this test is no longer comparing against the file"
        ),
    }
}

/// The `(x, p, c, interp, n_discrete)` one sub-table must be written as.
///
/// Written from `endf::univariate::Univariate` and not by calling
/// `univariate_flat::flatten`, which is the function under test. Two of the
/// five values are conventions with no counterpart on the parsed side and are
/// restated here as such: a `Discrete` is always written with interpolation 1
/// whatever the ACE INTT said, and a `Uniform` likewise, since
/// `endf::univariate::Uniform` carries no interpolation at all.
fn expect_flat(u: &Univariate) -> (Vec<f64>, Vec<f64>, Vec<f64>, i32, i32) {
    match u {
        Univariate::Tabular(t) => (
            t.x.clone(),
            t.p.clone(),
            stored_cdf("a tabular outgoing table", &t.c, t.x.len()),
            interp_code(t.interpolation),
            0,
        ),
        Univariate::Discrete(d) => (
            d.x.clone(),
            d.p.clone(),
            stored_cdf("a discrete line set", &d.c, d.x.len()),
            1,
            d.x.len() as i32,
        ),
        Univariate::Uniform(u) => {
            assert!(
                u.b > u.a,
                "no vendored fixture holds a degenerate Uniform, so the writer's \
                 zero-density branch is not the one under test here"
            );
            let density = 1.0 / (u.b - u.a);
            (vec![u.a, u.b], vec![density, density], vec![0.0, 1.0], 1, 0)
        }
        // The format has no mixture: the parts are concatenated with the
        // discrete lines in front, because `n_discrete` counts from the front.
        Univariate::Mixture(m) => {
            let (mut x, mut p, mut c) = (Vec::new(), Vec::new(), Vec::new());
            let mut n_discrete = 0;
            let mut continuous_interp = None;
            for part in &m.distribution {
                let (px, pp, pc, part_interp, part_discrete) = expect_flat(part);
                if part_discrete > 0 {
                    n_discrete += part_discrete;
                } else if continuous_interp.is_none() {
                    continuous_interp = Some(part_interp);
                }
                x.extend(px);
                p.extend(pp);
                c.extend(pc);
            }
            (x, p, c, continuous_interp.unwrap_or(2), n_discrete)
        }
    }
}

/// The same for a cosine table, whose parsed type is a different enum.
fn expect_mu(mu: &AngleAtEnergy) -> (Vec<f64>, Vec<f64>, Vec<f64>, i32) {
    match mu {
        AngleAtEnergy::Tabular(t) => (
            t.x.clone(),
            t.p.clone(),
            stored_cdf("a cosine table", &t.c, t.x.len()),
            interp_code(t.interpolation),
        ),
        AngleAtEnergy::Isotropic(u) => {
            let density = 1.0 / (u.b - u.a);
            (vec![u.a, u.b], vec![density, density], vec![0.0, 1.0], 1)
        }
        AngleAtEnergy::Tabulated(_) => {
            panic!("no vendored fixture reaches the MF=4 LTT=2 branch of angle_at_energy")
        }
        AngleAtEnergy::Legendre(_) => panic!("the writer refuses Legendre, so this cannot be here"),
    }
}

// ---------------------------------------------------------------------------
// Comparing a ravel.
// ---------------------------------------------------------------------------

/// A `(3, total)` C-order ravel, one third at a time.
///
/// Compared in thirds rather than as one vector because the failure the module
/// docstring names is an interleaved layout, and an interleaved write differs
/// from this one in the first third: naming the third in the message says
/// which row went wrong instead of giving an index into a flat array of
/// thousands.
#[track_caller]
fn assert_ravel3(what: &str, written: &[f64], x: &[f64], p: &[f64], c: &[f64]) {
    let total = x.len();
    assert_eq!(p.len(), total, "{what}: the p row is a different length");
    assert_eq!(c.len(), total, "{what}: the c row is a different length");
    assert_eq!(
        written.len(),
        3 * total,
        "{what}: the ravel is (3, {total}), so it holds {} values",
        3 * total
    );
    assert_f64_slice_eq(&format!("{what} x"), &written[..total], x);
    assert_f64_slice_eq(&format!("{what} p"), &written[total..2 * total], p);
    assert_f64_slice_eq(&format!("{what} c"), &written[2 * total..], c);
}

/// The running start of each sub-table.
fn expect_offsets(lengths: &[usize]) -> Vec<i32> {
    let mut out = Vec::with_capacity(lengths.len());
    let mut at = 0i32;
    for &n in lengths {
        out.push(at);
        at += n as i32;
    }
    out
}

/// The offsets, and the four properties the reader depends on.
///
/// `stride` is the first dimension of the ravel the offsets index into, and it
/// is not always 3: `km_data` and `corr_eout_data` are `(5, total)`, so a
/// closing check that divided by 3 would pass on a file with the wrong number
/// of rows entirely.
#[track_caller]
fn assert_offsets(what: &str, written: &[i32], lengths: &[usize], data_len: usize, stride: usize) {
    assert_i32_slice_eq(what, written, &expect_offsets(lengths));
    assert_eq!(written.len(), lengths.len(), "{what}: one offset per table");
    assert_eq!(written[0], 0, "{what}: the first table starts at zero");
    assert!(
        written.windows(2).all(|w| w[0] <= w[1]),
        "{what}: the offsets are not non-decreasing: {written:?}"
    );
    // The reader takes the end of the LAST table as the total point count, so
    // this is the only one of the four that catches an off-by-one at the end.
    let last = *written.last().expect("at least one sub-table") as usize;
    assert_eq!(
        last + lengths.last().expect("at least one sub-table"),
        data_len / stride,
        "{what}: the last table does not end where the ({stride}, total) ravel does"
    );
}

// ---------------------------------------------------------------------------
// Nullability.
// ---------------------------------------------------------------------------

/// The nullable scalar columns of `distributions.arrow`.
///
/// A row fills the ones its own variant uses and leaves the rest null, and the
/// reader defaults each null to zero, so a null in the wrong place is a
/// different physics rather than a missing value: a null `energy_threshold`
/// opens a threshold reaction at every energy, and a null `nbody_n` is zero
/// bodies.
const SCALAR_COLUMNS: [&str; 10] = [
    "energy_restriction_u",
    "energy_threshold",
    "energy_mass_ratio",
    "energy_primary_flag",
    "energy_atomic_weight_ratio",
    "energy_discrete_energy",
    "nbody_n",
    "nbody_total_mass",
    "nbody_atomic_weight_ratio",
    "nbody_q_value",
];

/// The columns a Kalbach-Mann row fills, and every other row leaves null.
const KM_COLUMNS: [&str; 7] = [
    "km_energies",
    "km_breakpoints",
    "km_interpolation",
    "km_data",
    "km_offsets",
    "km_interp",
    "km_n_discrete",
];

/// The columns a correlated row fills, and every other row leaves null.
const CORR_COLUMNS: [&str; 10] = [
    "corr_energies",
    "corr_breakpoints",
    "corr_interpolation",
    "corr_eout_data",
    "corr_eout_offsets",
    "corr_eout_interp",
    "corr_eout_n_discrete",
    "corr_mu_data",
    "corr_mu_offsets",
    "corr_mu_interp",
];

/// Every `list<int32>` column of `distributions.arrow`.
const INT_LIST_COLUMNS: [&str; 21] = [
    "applicability_shape",
    "applicability_breakpoints",
    "applicability_interpolation",
    "angle_mu_offsets",
    "angle_mu_interpolation",
    "energy_dist_interpolation",
    "energy_dist_offsets",
    "energy_dist_out_interp",
    "energy_dist_n_discrete",
    "corr_breakpoints",
    "corr_interpolation",
    "corr_eout_offsets",
    "corr_eout_interp",
    "corr_eout_n_discrete",
    "corr_mu_offsets",
    "corr_mu_interp",
    "km_breakpoints",
    "km_interpolation",
    "km_offsets",
    "km_interp",
    "km_n_discrete",
];

#[track_caller]
fn assert_null(batch: &RecordBatch, columns: &[&str], row: usize, what: &str) {
    for col in columns {
        assert!(
            is_null(batch, col, row),
            "{what}: {col} should be null on row {row}"
        );
    }
}

#[track_caller]
fn assert_not_null(batch: &RecordBatch, columns: &[&str], row: usize, what: &str) {
    for col in columns {
        assert!(
            !is_null(batch, col, row),
            "{what}: {col} should not be null on row {row}"
        );
    }
}

/// The scalar columns this row is allowed to fill, and no others.
///
/// Both directions matter and no single fixture exercises both on every
/// column: on Li6 the four `nbody_*` columns are null on every row, so this
/// sweeps them there only in the null direction. The law-66 row of the laws
/// fixture is where they are filled.
#[track_caller]
fn assert_only_scalars(batch: &RecordBatch, row: usize, filled: &[&str], what: &str) {
    for col in SCALAR_COLUMNS {
        let expected_null = !filled.contains(&col);
        assert_eq!(
            is_null(batch, col, row),
            expected_null,
            "{what}: row {row} has {col} {}, which its variant does not agree with",
            if expected_null { "filled" } else { "null" }
        );
    }
}

// ---------------------------------------------------------------------------
// The laws fixture, which no parser will take whole.
// ---------------------------------------------------------------------------

/// Walk the DLW linked list for one reaction.
///
/// LDLW is JXS(10) and gives one locator per reaction; DLW is JXS(11) and each
/// distribution's own first word points at the next one for the same reaction.
/// This reimplements the crate's own helper at
/// `crates/endf/src/angle_energy.rs:276-285`, which lives inside a
/// `#[cfg(test)]` module and so cannot be imported. If the DLW layout ever
/// changes, this walk breaks in a way that looks like a converter defect and
/// is not one, which is what that reference is here to say.
fn dlw_chain(table: &endf::Table, i_reaction: usize) -> Vec<i64> {
    let (ldlw, dlw) = (table.jxs[10], table.jxs[11]);
    let mut chain = Vec::new();
    let mut lnw = table.xss[ldlw as usize + i_reaction - 1] as i64;
    while lnw > 0 {
        chain.push(lnw);
        lnw = table.xss[(dlw + lnw - 1) as usize] as i64;
    }
    chain
}

/// A Q value that cannot have come from anywhere but this call.
///
/// Law 66 is the only law that takes its Q from the caller rather than from the
/// block, and the golden dumper passes 0.0, which would make a dropped field
/// indistinguishable from a copied one.
fn law_q(law: i64) -> f64 {
    -1.0e6 * law as f64 - 0.5
}

/// The seven laws of `synthetic-laws.ace.xz`, each on a reaction of its own.
///
/// The table is a DLW block and nothing else: JXS(1) ESZ, JXS(3) MTR, JXS(6)
/// LSIG and JXS(7) SIG are all zero, so `IncidentNeutron::from_ace` fails and
/// no writer can be handed this fixture the ordinary way. Each law is decoded
/// straight out of the vendored bytes by `endf::AngleEnergy::from_ace`; only
/// the `Reaction` and `Product` around it are built here. MTs are `1000 + law`
/// so a failure names the law, and they ascend in law order, which is the order
/// the `BTreeMap` walk then writes them in.
fn laws_nuclide() -> (endf::Table, IncidentNeutron, Vec<i64>) {
    let table = ace_table(LAWS_ACE);
    let dlw = table.jxs[11];
    let mut data = IncidentNeutron::new(1, 1, 0);
    let mut laws = Vec::new();
    for i_reaction in 1..=(table.nxs[5] as usize) {
        for lnw in dlw_chain(&table, i_reaction) {
            // The law number sits one word past the locator, which is where
            // `AngleEnergy::from_ace` reads it from too.
            let law = table.xss[(dlw + lnw) as usize] as i64;
            let dist = AngleEnergy::from_ace(&table, dlw, lnw, Some(law_q(law)))
                .expect("every law in this table is one the reader knows");
            let mt = 1000 + law as i32;
            let mut reaction = endf::Reaction::new(mt);
            reaction.products.push(Product {
                name: "neutron".to_string(),
                distribution: vec![dist],
                ..Default::default()
            });
            data.reactions.insert(mt, reaction);
            laws.push(law);
        }
    }
    (table, data, laws)
}

/// The parsed distribution of the one product on MT `1000 + law`.
fn law_dist(data: &IncidentNeutron, law: i64) -> &AngleEnergy {
    &data.reactions[&(1000 + law as i32)].products[0].distribution[0]
}

// ---------------------------------------------------------------------------
// Inputs built by hand, for the branches no vendored evaluation reaches.
// ---------------------------------------------------------------------------

/// A tabulated sub-table carrying the CDF an ACE table stores beside it.
///
/// The CDF is supplied rather than left `None` so the writer's own trapezoid
/// integration stays out of the comparison, which is the same rule
/// `stored_cdf` enforces on the parsed fixtures.
fn tabular(x: Vec<f64>, p: Vec<f64>, interpolation: Interpolation, c: Vec<f64>) -> Univariate {
    Univariate::Tabular(Tabular::with_cdf(x, p, interpolation, c))
}

/// A set of discrete lines carrying its own CDF, for the same reason.
fn discrete_lines(x: Vec<f64>, p: Vec<f64>, c: Vec<f64>) -> Univariate {
    Univariate::Discrete(Discrete { x, p, c: Some(c) })
}

/// One reaction holding one neutron product, and nothing else at all.
fn one_product_nuclide(
    mt: i32,
    distribution: Vec<AngleEnergy>,
    applicability: Vec<Tabulated1D>,
) -> IncidentNeutron {
    let mut data = IncidentNeutron::new(3, 6, 0);
    let mut reaction = endf::Reaction::new(mt);
    reaction.products.push(Product {
        name: "neutron".to_string(),
        distribution,
        applicability,
        ..Default::default()
    });
    data.reactions.insert(mt, reaction);
    data
}

/// A product with TWO distributions, and a different applicability for each.
///
/// CONSTRUCTED INPUT. Nothing here was parsed from an evaluation and nothing
/// here is evaluation parity: the only claim it can support is about how the
/// writer indexes. It exists because every product of Li6, U235, Fe56 and the
/// laws scaffold carries exactly one distribution, so `dist_idx` is 0 on every
/// row those fixtures produce and so is the index into `product.applicability`.
/// Replacing `dist_idx: dist_idx as i32` with `dist_idx: 0`, or
/// `product.applicability.get(dist_idx)` with `.get(0)`, leaves every
/// vendored-fixture assertion in this file green.
///
/// The two distributions differ in both of their scalars and the two
/// applicability tabulations differ in y, deliberately: equal rows cannot
/// discriminate an index that never advances.
fn constructed_two_distribution_product() -> IncidentNeutron {
    let level = |threshold: f64, mass_ratio: f64| {
        AngleEnergy::Uncorrelated(UncorrelatedAngleEnergy {
            angle: None,
            energy: Some(EnergyDistribution::LevelInelastic {
                threshold,
                mass_ratio,
            }),
        })
    };
    one_product_nuclide(
        16,
        vec![level(1.5e6, 0.75), level(7.25e6, 0.25)],
        vec![
            Tabulated1D::new(vec![1.0e3, 2.0e7], vec![1.0, 0.0]),
            Tabulated1D::new(vec![1.0e3, 2.0e7], vec![0.0, 1.0]),
        ],
    )
}

/// A correlated distribution whose two region columns disagree and whose
/// outgoing tables hold different numbers of discrete lines.
///
/// CONSTRUCTED INPUT, and not evaluation parity: the values below are not from
/// any evaluation. Law 61 of `synthetic-laws.ace.xz` is the only vendored
/// correlated distribution in the tree, and on it `corr_breakpoints` and
/// `corr_interpolation` are both `[1, 2]` while `corr_eout_n_discrete` is
/// `[0, 0]`. Swapping the two writer assignments, or pushing a literal 0 for
/// the line count, therefore produces a byte-identical file. Here the
/// breakpoints are `[1, 3]` against interpolation `[2, 1]`, which differ at
/// every position, and the three outgoing tables hold 0, 1 and 2 lines.
fn constructed_correlated_regions() -> IncidentNeutron {
    let isotropic = || Univariate::Uniform(Uniform::new(-1.0, 1.0));
    let mu_table = || {
        tabular(
            vec![-1.0, 1.0],
            vec![0.25, 0.75],
            Interpolation::LinearLinear,
            vec![0.0, 1.0],
        )
    };
    let c = CorrelatedAngleEnergy {
        breakpoints: vec![1, 3],
        interpolation: vec![2, 1],
        energy: vec![1.0e3, 1.0e6, 2.0e7],
        energy_out: vec![
            tabular(
                vec![1.0e5, 5.0e5],
                vec![1.5e-6, 5.0e-7],
                Interpolation::LinearLinear,
                vec![0.0, 1.0],
            ),
            Univariate::Mixture(Mixture::new(
                vec![0.4, 0.6],
                vec![
                    discrete_lines(vec![2.0e5], vec![0.4], vec![0.4]),
                    tabular(
                        vec![3.0e5, 9.0e5],
                        vec![1.0e-6, 5.0e-7],
                        Interpolation::Histogram,
                        vec![0.7, 1.0],
                    ),
                ],
            )),
            discrete_lines(vec![1.1e6, 1.3e6], vec![0.25, 0.75], vec![0.25, 1.0]),
        ],
        // One cosine table per outgoing POINT: two, then three, then two, so
        // the writer's missing-mu fallback does not fire here either.
        mu: vec![
            vec![isotropic(), mu_table()],
            vec![isotropic(), mu_table(), isotropic()],
            vec![mu_table(), isotropic()],
        ],
    };
    one_product_nuclide(22, vec![AngleEnergy::Correlated(c)], Vec::new())
}

/// A Kalbach-Mann distribution with discrete lines in one outgoing table.
///
/// CONSTRUCTED INPUT, and not evaluation parity. Li6 is the only vendored
/// source of the variant and every outgoing table of both its Kalbach-Mann
/// rows is a pure `Tabular`, so `km_n_discrete` is all zeros there and
/// `n_discrete.push(0)` in `write_kalbach_mann` writes the same file. One
/// mixture is enough to tell those apart. The precompound and slope
/// tabulations are as long as their flattened tables, matching what
/// `KalbachMann::from_ace` produces, so the writer's zero-padding branch stays
/// out of this.
fn constructed_kalbach_mann_with_lines() -> IncidentNeutron {
    let k = KalbachMann {
        breakpoints: vec![2],
        interpolation: vec![1],
        energy: vec![1.0e6, 1.4e7],
        energy_out: vec![
            tabular(
                vec![1.0e5, 9.0e5],
                vec![1.5e-6, 5.0e-7],
                Interpolation::LinearLinear,
                vec![0.0, 1.0],
            ),
            Univariate::Mixture(Mixture::new(
                vec![0.3, 0.7],
                vec![
                    discrete_lines(vec![2.0e5, 4.0e5], vec![0.1, 0.2], vec![0.1, 0.3]),
                    tabular(
                        vec![5.0e5, 1.5e6],
                        vec![1.0e-6, 4.0e-7],
                        Interpolation::Histogram,
                        vec![0.6, 1.0],
                    ),
                ],
            )),
        ],
        precompound: vec![
            Tabulated1D::new(vec![1.0e5, 9.0e5], vec![0.11, 0.12]),
            Tabulated1D::new(
                vec![2.0e5, 4.0e5, 5.0e5, 1.5e6],
                vec![0.21, 0.22, 0.23, 0.24],
            ),
        ],
        slope: vec![
            Tabulated1D::new(vec![1.0e5, 9.0e5], vec![0.51, 0.52]),
            Tabulated1D::new(
                vec![2.0e5, 4.0e5, 5.0e5, 1.5e6],
                vec![0.61, 0.62, 0.63, 0.64],
            ),
        ],
    };
    one_product_nuclide(28, vec![AngleEnergy::KalbachMann(k)], Vec::new())
}

/// A Watt spectrum whose two tabulated parameters sit on DIFFERENT grids.
///
/// CONSTRUCTED INPUT, and not evaluation parity. Law 11 of
/// `synthetic-laws.ace.xz` is the only Watt spectrum any vendored bytes
/// produce, and it gives `a` and `b` the same x, `[1.0, 2.0e7]`, so writing
/// `a.x` into `energy_param2_x` (the copy-paste a four-assignment block
/// invites) produces a byte-identical file. The two y vectors there differ by
/// eleven orders of magnitude, so the pair as a whole is discriminated and
/// only the two grids are not. These two differ in length as well as in value.
fn constructed_watt_with_distinct_grids() -> IncidentNeutron {
    let watt = AngleEnergy::Uncorrelated(UncorrelatedAngleEnergy {
        angle: None,
        energy: Some(EnergyDistribution::WattEnergy {
            u: 1.25e6,
            a: Tabulated1D::new(vec![1.0e3, 2.0e7], vec![9.0e5, 1.1e6]),
            b: Tabulated1D::new(vec![2.0e3, 5.0e6, 1.9e7], vec![3.0e-6, 3.2e-6, 3.4e-6]),
        }),
    });
    one_product_nuclide(18, vec![watt], Vec::new())
}

// ---------------------------------------------------------------------------
// products.arrow
// ---------------------------------------------------------------------------

/// A failure means `products.arrow` and `distributions.arrow` stopped being
/// written from one traversal, which is the contract `products.rs:4-8` states
/// and nothing checks. The reader joins the two files on
/// `(reaction_mt, product_idx, dist_idx)` and builds a `HashMap` keyed on it,
/// so a duplicate triple silently drops a distribution and a gap in `dist_idx`
/// leaves a product whose secondary energy is sampled from nothing.
///
/// Li6 pins two thirds of that key and cannot pin the third. All 18 of its
/// products carry exactly one distribution, so `dist_idx` is 0 on every row it
/// produces, and so it is on every row of every other vendored fixture in this
/// file: writing a literal 0 satisfies all of them. The constructed
/// two-distribution product at the end is the only thing here that says
/// `dist_idx` counts, and its helper says why it is built rather than parsed.
#[test]
fn product_rows_are_the_reaction_walk_in_mt_order() {
    let data = li6_ace();
    let walk = product_walk(&data);
    let tmp = scratch();
    let products = products_of(&data, tmp.path());
    let distributions = distributions_of(&data, tmp.path());
    assert_schema_is_declared(&products, "products.arrow");

    assert_eq!(
        products.num_rows(),
        walk.len(),
        "one row per parsed product"
    );

    let written_mt = i32_column(&products, "reaction_mt");
    let expected_mt: Vec<i32> = walk.iter().map(|&(mt, _, _)| mt).collect();
    assert_i32_slice_eq("products.arrow reaction_mt", &written_mt, &expected_mt);
    // The literal too, so the traversal itself is pinned rather than merely
    // restated: a walk that dropped MT 55's third and fourth photons would
    // agree with the expression above and not with this.
    assert_i32_slice_eq(
        "products.arrow reaction_mt against the Li6 product set",
        &written_mt,
        &[
            2, 5, 5, 51, 51, 52, 52, 53, 53, 54, 54, 55, 55, 55, 55, 91, 91, 102,
        ],
    );

    let written_idx = i32_column(&products, "product_idx");
    let expected_idx: Vec<i32> = walk.iter().map(|&(_, idx, _)| idx as i32).collect();
    assert_i32_slice_eq("products.arrow product_idx", &written_idx, &expected_idx);
    assert_i32_slice_eq(
        "products.arrow product_idx against the Li6 product set",
        &written_idx,
        &[0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 0],
    );

    let written_n = i32_column(&products, "n_distribution");
    let expected_n: Vec<i32> = walk
        .iter()
        .map(|&(_, _, p)| p.distribution.len() as i32)
        .collect();
    assert_i32_slice_eq("products.arrow n_distribution", &written_n, &expected_n);

    // The join invariant: exactly the triples n_distribution promises, once
    // each, and nothing else.
    let mut expected_triples = Vec::new();
    for (row, &(mt, idx, _)) in walk.iter().enumerate() {
        for d in 0..written_n[row] {
            expected_triples.push((mt, idx as i32, d));
        }
    }
    // `expected_triples` holds no duplicate, because `(mt, idx)` is unique per
    // product and `d` ascends within it, so the equality is also the check
    // that no triple appears twice.
    let written_triples = triples(&distributions);
    assert_eq!(
        written_triples, expected_triples,
        "the distribution rows are not products.arrow expanded by n_distribution"
    );

    // The dist_idx third of the key, on a CONSTRUCTED input: see
    // `constructed_two_distribution_product`. Everything above passes with the
    // column hard-coded to zero, because no vendored product has a second
    // distribution.
    let two = constructed_two_distribution_product();
    let two_tmp = scratch();
    let two_products = products_of(&two, two_tmp.path());
    let two_distributions = distributions_of(&two, two_tmp.path());
    assert_i32_slice_eq(
        "the constructed product's n_distribution",
        &i32_column(&two_products, "n_distribution"),
        &[2],
    );
    assert_eq!(
        triples(&two_distributions),
        vec![(16, 0, 0), (16, 0, 1)],
        "n_distribution promises two rows, numbered by dist_idx and not both zero"
    );
}

/// A failure means a product's particle type, emission mode or decay constant
/// is no longer the one the evaluation gave. `decay_rate` is the one that
/// cannot be told from a default on an ACE fixture, which is why the fissile
/// ENDF evaluation is here: its six delayed groups carry real decay constants
/// from MF=1 MT=455, and a writer that dropped the field would still pass on
/// Li6, where every value is 0.0.
#[test]
fn product_identity_columns_are_the_parsed_products_own_fields() {
    let data = li6_ace();
    let walk = product_walk(&data);
    let tmp = scratch();
    let products = products_of(&data, tmp.path());

    let mut names: BTreeMap<&str, usize> = BTreeMap::new();
    for (row, &(mt, idx, product)) in walk.iter().enumerate() {
        let what = format!("Li6 MT {mt} product {idx}");
        assert_eq!(
            str_at(&products, "particle", row),
            product.name,
            "{what}: particle"
        );
        assert_eq!(
            str_at(&products, "emission_mode", row),
            product.emission_mode.name(),
            "{what}: emission_mode"
        );
        // The literal beside the call, so a rename on either side is visible
        // rather than agreeing with itself.
        assert_eq!(
            str_at(&products, "emission_mode", row),
            "prompt",
            "{what}: every Li6 product is prompt"
        );
        assert_eq!(
            f64_at(&products, "decay_rate", row),
            product.decay_rate,
            "{what}: decay_rate"
        );
        assert_eq!(f64_at(&products, "decay_rate", row), 0.0, "{what}: prompt");
        *names.entry(product.name.as_str()).or_insert(0) += 1;
    }
    assert_eq!(
        names,
        BTreeMap::from([("neutron", 8), ("photon", 10)]),
        "Li6 gives 8 neutron products from the DLW and elastic chains and 10 photons from MTRP"
    );

    // The delayed groups, which no ACE fixture in the tree carries.
    let u235 = endf_nuclide(U235_ENDF);
    let u_walk = product_walk(&u235);
    let u_tmp = scratch();
    let u_products = products_of(&u235, u_tmp.path());
    assert_eq!(
        u_products.num_rows(),
        7,
        "one prompt and six delayed groups"
    );

    let mut delayed = 0;
    for (row, &(mt, idx, product)) in u_walk.iter().enumerate() {
        let what = format!("U235 MT {mt} product {idx}");
        assert_eq!(str_at(&u_products, "particle", row), product.name, "{what}");
        assert_eq!(
            str_at(&u_products, "emission_mode", row),
            product.emission_mode.name(),
            "{what}: emission_mode"
        );
        assert_eq!(
            f64_at(&u_products, "decay_rate", row),
            product.decay_rate,
            "{what}: decay_rate"
        );
        if product.emission_mode == endf::EmissionMode::Delayed {
            delayed += 1;
            assert!(
                f64_at(&u_products, "decay_rate", row) > 0.0,
                "{what}: a delayed group's decay constant cannot be the prompt default"
            );
        }
    }
    assert_eq!(delayed, 6, "MF=1 MT=455 gives six precursor groups");
    assert_eq!(
        str_column(&u_products, "emission_mode"),
        vec!["prompt", "delayed", "delayed", "delayed", "delayed", "delayed", "delayed"],
        "the literal names endf::EmissionMode::name produces"
    );

    // The one-row photon case.
    let fe56 = endf_nuclide(FE56_ENDF);
    let fe_tmp = scratch();
    let fe_products = products_of(&fe56, fe_tmp.path());
    assert_eq!(fe_products.num_rows(), 1);
    assert_eq!(
        str_at(&fe_products, "particle", 0),
        product_walk(&fe56)[0].2.name
    );
    assert_eq!(str_at(&fe_products, "particle", 0), "photon");
}

/// A failure means a multiplicity was reshaped on the way out. `yield_shape` is
/// the column `products.rs:48-53` warns about: the reader splits on `shape[1]`,
/// so a transposed `[n, 2]` loads without error and turns Li6's 32-point MT 5
/// yield into a 2-point one. `yield_breakpoints` and `yield_interpolation` are
/// adjacent `list<int32>` columns filled from adjacent lines and the yamc
/// loader never reads either, so nothing but a direct comparison can catch a
/// swap.
#[test]
fn product_yield_columns_round_trip_both_yield_forms() {
    let mut polynomial = 0;
    let mut tabulated = 0;

    for (label, data) in [
        ("Li6", li6_ace()),
        ("U235", endf_nuclide(U235_ENDF)),
        ("Fe56", endf_nuclide(FE56_ENDF)),
    ] {
        let walk = product_walk(&data);
        let tmp = scratch();
        let products = products_of(&data, tmp.path());

        for (row, &(mt, idx, product)) in walk.iter().enumerate() {
            let what = format!("{label} MT {mt} product {idx}");
            let written = f64_list(&products, "yield_data", row);
            let shape = i32_list(&products, "yield_shape", row);
            let breakpoints = i32_list(&products, "yield_breakpoints", row);
            let interpolation = i32_list(&products, "yield_interpolation", row);

            match &product.yield_ {
                Yield::Polynomial(p) => {
                    polynomial += 1;
                    assert_eq!(str_at(&products, "yield_type", row), "Polynomial", "{what}");
                    assert_f64_slice_eq(&format!("{what} yield_data"), &written, &p.coefficients);
                    assert_eq!(shape.len(), 1, "{what}: a polynomial's shape is [n]");
                    assert_eq!(shape[0] as usize, written.len(), "{what}: shape[0]");
                    // Empty and not null: products.arrow uses sections::int_lists
                    // where every list column in distributions.arrow uses the
                    // or_null variant, and the reader cannot tell the two apart.
                    assert_not_null(
                        &products,
                        &["yield_breakpoints", "yield_interpolation"],
                        row,
                        &what,
                    );
                    assert!(breakpoints.is_empty(), "{what}: yield_breakpoints");
                    assert!(interpolation.is_empty(), "{what}: yield_interpolation");
                }
                Yield::Tabulated(t) => {
                    tabulated += 1;
                    let n = t.x.len();
                    assert_eq!(
                        str_at(&products, "yield_type", row),
                        "Tabulated1D",
                        "{what}"
                    );
                    assert_eq!(written.len(), 2 * n, "{what}: x and y end to end");
                    // Bit-exact: no arithmetic happens here, the eV conversion
                    // was done in the parser at function.rs:198-202.
                    assert_f64_slice_eq(&format!("{what} yield x"), &written[..n], &t.x);
                    assert_f64_slice_eq(&format!("{what} yield y"), &written[n..], &t.y);
                    // `[2, n]` and not `[n, 2]`. The reader splits the data
                    // column at `shape[1]`, so the transposed shape loads and
                    // turns Li6's 32-point MT 5 yield into a 2-point one; this
                    // one comparison is the whole defence, and n is 32 there,
                    // so the two shapes are distinguishable.
                    assert_i32_slice_eq(&format!("{what} yield_shape"), &shape, &[2, n as i32]);
                    assert_i32_slice_eq(
                        &format!("{what} yield_breakpoints"),
                        &breakpoints,
                        &t.breakpoints,
                    );
                    assert_i32_slice_eq(
                        &format!("{what} yield_interpolation"),
                        &interpolation,
                        &t.interpolation,
                    );
                    assert!(
                        !breakpoints.is_empty(),
                        "{what}: Tabulated1D::from_ace falls back to vec![n_pairs], so a \
                         tabulated yield always has a region"
                    );
                    assert_eq!(
                        breakpoints.len(),
                        interpolation.len(),
                        "{what}: one interpolation code per region"
                    );
                }
            }
        }
    }

    // Neither form is covered by accident: Li6 alone gives 7 polynomial and 11
    // tabulated yields, and the two ENDF fixtures add 8 more tabulated.
    assert_eq!(polynomial, 7, "the polynomial branch was exercised");
    assert_eq!(tabulated, 19, "the tabulated branch was exercised");
}

// ---------------------------------------------------------------------------
// distributions.arrow
// ---------------------------------------------------------------------------

/// A failure means the distribution table no longer lines up with the product
/// table. The zero-row case matters as much as the populated one: Fe56's single
/// photon product has an empty distribution vector, and "a distributions.arrow
/// with no rows" and "no distributions.arrow" are different answers to the
/// loader.
///
/// What the Li6 half of this cannot say, stated so nobody reads more into it.
/// The expectation is rebuilt by reading `products.arrow` rather than the
/// parsed evaluation, so a defect that dropped the same product from BOTH
/// writers agrees with itself here; only
/// `product_rows_are_the_reaction_walk_in_mt_order`, which walks the parsed
/// side, catches that. And every Li6 product holds one distribution, so the
/// no-gap sweep over `dist_idx` is satisfied by a column of zeros. The
/// CONSTRUCTED two-distribution product at the end is what makes `dist_idx`
/// mean something, and it is the only part of this test that checks a row
/// against the distribution it claims to be.
#[test]
fn distribution_rows_join_products_one_for_one() {
    let data = li6_ace();
    let tmp = scratch();
    let products = products_of(&data, tmp.path());
    let distributions = distributions_of(&data, tmp.path());
    assert_schema_is_declared(&distributions, "distributions.arrow");
    assert_eq!(distributions.num_rows(), 18, "Li6 has 18 distributions");

    let written = triples(&distributions);
    let mut expected = Vec::new();
    for row in 0..products.num_rows() {
        let mt = i32_at(&products, "reaction_mt", row);
        let idx = i32_at(&products, "product_idx", row);
        for d in 0..i32_at(&products, "n_distribution", row) {
            expected.push((mt, idx, d));
        }
    }
    assert_eq!(
        written, expected,
        "the two files must be in the same row order, not merely hold the same keys"
    );

    // Per product the dist_idx values ascend from zero with no gaps: the reader
    // loops `for di in 0..n_dist` and skips a missing one silently.
    let mut seen: BTreeMap<(i32, i32), Vec<i32>> = BTreeMap::new();
    for &(mt, idx, dist) in &written {
        seen.entry((mt, idx)).or_default().push(dist);
    }
    for ((mt, idx), dists) in &seen {
        let expected: Vec<i32> = (0..dists.len() as i32).collect();
        assert_eq!(dists, &expected, "MT {mt} product {idx}: dist_idx");
    }
    let unique: BTreeSet<_> = written.iter().collect();
    assert_eq!(unique.len(), written.len(), "no triple appears twice");

    // Row d against distribution d, on a CONSTRUCTED input: see
    // `constructed_two_distribution_product`. `write_distributions` picks the
    // applicability with `product.applicability.get(dist_idx)`, and with one
    // distribution per product `.get(0)` is the same call.
    let two = constructed_two_distribution_product();
    let two_tmp = scratch();
    let two_batch = distributions_of(&two, two_tmp.path());
    assert_eq!(two_batch.num_rows(), 2);
    let product = &two.reactions[&16].products[0];
    for (row, dist) in product.distribution.iter().enumerate() {
        let what = format!("the constructed product's distribution {row}");
        let AngleEnergy::Uncorrelated(u) = dist else {
            panic!("{what}: built as uncorrelated");
        };
        let Some(EnergyDistribution::LevelInelastic {
            threshold,
            mass_ratio,
        }) = &u.energy
        else {
            panic!("{what}: built as a level law");
        };
        assert_eq!(i32_at(&two_batch, "dist_idx", row), row as i32, "{what}");
        assert_eq!(
            f64_at(&two_batch, "energy_threshold", row),
            *threshold,
            "{what}: the row holds the OTHER distribution's threshold"
        );
        assert_eq!(
            f64_at(&two_batch, "energy_mass_ratio", row),
            *mass_ratio,
            "{what}: energy_mass_ratio"
        );

        let app = &product.applicability[row];
        let n = app.x.len();
        let written = f64_list(&two_batch, "applicability_data", row);
        assert_eq!(written.len(), 2 * n, "{what}: applicability x and y");
        assert_f64_slice_eq(&format!("{what} applicability x"), &written[..n], &app.x);
        assert_f64_slice_eq(&format!("{what} applicability y"), &written[n..], &app.y);
    }
    // And the two rows do differ, which is what makes the pairing above
    // checkable rather than a comparison of two copies of one thing.
    assert_ne!(
        f64_list(&two_batch, "applicability_data", 0),
        f64_list(&two_batch, "applicability_data", 1),
        "the constructed applicabilities are built distinct on purpose"
    );

    // Written, and empty. Fe56's one photon product carries no distribution.
    let fe56 = endf_nuclide(FE56_ENDF);
    let fe_tmp = scratch();
    let fe_distributions = distributions_of(&fe56, fe_tmp.path());
    assert!(
        !absent(fe_tmp.path(), "distributions.arrow"),
        "an absent file and an empty one are different answers to the loader"
    );
    assert_eq!(fe_distributions.num_rows(), 0);
    let fe_products = products_of(&fe56, fe_tmp.path());
    assert_eq!(i32_at(&fe_products, "n_distribution", 0), 0);
}

/// A failure means the angular tables or the scalar energy-law parameters were
/// reshaped, interleaved or dropped. The interleaved `(x, p, c)` layout is the
/// failure `distributions.rs:24-26` names: it loads, and then samples densities
/// out of the cosine column. The null-versus-empty distinction is part of the
/// format here and invisible at load, because the reader's `try_get_f64_list`
/// maps both to an empty `Vec`.
///
/// Two of these columns are pinned on Li6 and not discriminated by it.
/// `energy_primary_flag` is 0 on all seven of its discrete-photon rows, so the
/// comparison below cannot tell the column from a literal 0; the one vendored
/// row that carries 1 is the law-2 row of `synthetic-laws.ace.xz`, asserted in
/// `the_laws_only_an_assembled_table_can_reach_are_written_from_the_parsed_blocks`.
/// And the four `nbody_*` columns are null on every Li6 row, so
/// `assert_only_scalars` sweeps them here in the null direction only.
#[test]
fn an_uncorrelated_distribution_is_the_parsed_angle_and_energy_tables() {
    let table = ace_table(LI6_ACE);
    let data = li6_ace();
    let walk = distribution_walk(&data);
    let tmp = scratch();
    let batch = distributions_of(&data, tmp.path());

    let mut types: BTreeMap<String, usize> = BTreeMap::new();
    let mut energy_types: BTreeMap<String, usize> = BTreeMap::new();
    let mut applicable = 0;
    let mut angle_interp_seen: BTreeSet<i32> = BTreeSet::new();
    let mut angle_grids: Vec<usize> = Vec::new();

    for (row, &(mt, product_idx, dist_idx, dist, applicability)) in walk.iter().enumerate() {
        let what = format!("Li6 MT {mt} product {product_idx} distribution {dist_idx}");

        let ty = str_at(&batch, "type", row);
        let expected_ty = match dist {
            AngleEnergy::Uncorrelated(_) => "uncorrelated",
            AngleEnergy::Correlated(_) => "correlated",
            AngleEnergy::KalbachMann(_) => "kalbach-mann",
            AngleEnergy::NBodyPhaseSpace(_) => "nbody",
        };
        assert_eq!(ty, expected_ty, "{what}: type");
        *types.entry(ty).or_insert(0) += 1;

        // Nothing in Li6 is correlated or N-body, and no row may pretend to
        // be: the N-body columns are swept by assert_only_scalars below.
        assert_null(&batch, &CORR_COLUMNS, row, &what);

        let applicability_columns = [
            "applicability_data",
            "applicability_shape",
            "applicability_breakpoints",
            "applicability_interpolation",
        ];
        match applicability {
            Some(app) => {
                applicable += 1;
                assert_not_null(&batch, &applicability_columns, row, &what);
                let n = app.x.len();
                let written = f64_list(&batch, "applicability_data", row);
                assert_eq!(written.len(), 2 * n, "{what}: applicability x and y");
                assert_f64_slice_eq(&format!("{what} applicability x"), &written[..n], &app.x);
                assert_f64_slice_eq(&format!("{what} applicability y"), &written[n..], &app.y);
                assert_i32_slice_eq(
                    &format!("{what} applicability_shape"),
                    &i32_list(&batch, "applicability_shape", row),
                    &[2, n as i32],
                );
                let breakpoints = i32_list(&batch, "applicability_breakpoints", row);
                let interpolation = i32_list(&batch, "applicability_interpolation", row);
                assert_i32_slice_eq(
                    &format!("{what} applicability_breakpoints"),
                    &breakpoints,
                    &app.breakpoints,
                );
                assert_i32_slice_eq(
                    &format!("{what} applicability_interpolation"),
                    &interpolation,
                    &app.interpolation,
                );
                assert!(!breakpoints.is_empty(), "{what}: at least one region");
                assert_eq!(
                    breakpoints.len(),
                    interpolation.len(),
                    "{what}: one per region"
                );
            }
            // Null, not an empty list. `float_lists_or_null` writes null and the
            // published files hold null; an empty list loads identically and is
            // not the same file.
            None => assert_null(&batch, &applicability_columns, row, &what),
        }

        let angle_columns = [
            "angle_energies",
            "angle_mu_data",
            "angle_mu_offsets",
            "angle_mu_interpolation",
        ];
        let angle = match dist {
            AngleEnergy::Uncorrelated(u) => u.angle.as_ref().filter(|a| !a.energy.is_empty()),
            _ => None,
        };
        match angle {
            Some(angle) => {
                assert_not_null(&batch, &angle_columns, row, &what);
                assert_f64_slice_eq(
                    &format!("{what} angle_energies"),
                    &f64_list(&batch, "angle_energies", row),
                    &angle.energy,
                );
                // write_angle takes the energies from angle.energy and the
                // tables from angle.mu and never checks the two agree, while
                // the reader indexes the offsets by the energy index.
                assert_eq!(
                    list_len(&batch, "angle_energies", row),
                    angle.mu.len(),
                    "{what}: one cosine table per incident energy"
                );

                let (mut x, mut p, mut c, mut lengths, mut interp) =
                    (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
                for mu in &angle.mu {
                    let (mx, mp, mc, mi) = expect_mu(mu);
                    lengths.push(mx.len());
                    x.extend(mx);
                    p.extend(mp);
                    c.extend(mc);
                    interp.push(mi);
                }
                let data_col = f64_list(&batch, "angle_mu_data", row);
                assert_ravel3(&format!("{what} angle_mu_data"), &data_col, &x, &p, &c);
                assert_offsets(
                    &format!("{what} angle_mu_offsets"),
                    &i32_list(&batch, "angle_mu_offsets", row),
                    &lengths,
                    data_col.len(),
                    3,
                );
                // Li6's cosine tables parse as LinearLinear, so a neutron row is
                // all 2s; the photon rows are Isotropic, which the converter
                // writes as 1 by convention (univariate_flat.rs:102-113) because
                // endf's Uniform carries no interpolation to copy.
                assert_i32_slice_eq(
                    &format!("{what} angle_mu_interpolation"),
                    &i32_list(&batch, "angle_mu_interpolation", row),
                    &interp,
                );
                angle_interp_seen.extend(interp);
                angle_grids.push(angle.energy.len());
            }
            None => assert_null(&batch, &angle_columns, row, &what),
        }

        let energy = match dist {
            AngleEnergy::Uncorrelated(u) => u.energy.as_ref(),
            _ => None,
        };
        match energy {
            None => assert!(
                is_null(&batch, "energy_dist_type", row),
                "{what}: energy_dist_type must be null where there is no energy distribution"
            ),
            Some(energy) => {
                let written = str_at(&batch, "energy_dist_type", row);
                let expected = match energy {
                    EnergyDistribution::ContinuousTabular { .. } => "continuous",
                    EnergyDistribution::LevelInelastic { .. } => "level",
                    EnergyDistribution::DiscretePhoton { .. } => "discrete_photon",
                    other => panic!("{what}: Li6 does not hold {other:?}"),
                };
                assert_eq!(written, expected, "{what}: energy_dist_type");
                *energy_types.entry(written).or_insert(0) += 1;
            }
        }

        match energy {
            Some(EnergyDistribution::LevelInelastic {
                threshold,
                mass_ratio,
            }) => {
                assert_only_scalars(
                    &batch,
                    row,
                    &["energy_threshold", "energy_mass_ratio"],
                    &what,
                );
                assert_eq!(
                    f64_at(&batch, "energy_threshold", row),
                    *threshold,
                    "{what}: energy_threshold"
                );
                assert_eq!(
                    f64_at(&batch, "energy_mass_ratio", row),
                    *mass_ratio,
                    "{what}: energy_mass_ratio"
                );
                // Two adjacent Float64 columns filled from two adjacent lines.
                // A transposition is unmistakable only because the two differ
                // by six orders of magnitude on this fixture.
                assert!(
                    f64_at(&batch, "energy_threshold", row) > 1.0e5,
                    "{what}: the laboratory threshold is an energy in eV"
                );
                assert!(
                    f64_at(&batch, "energy_mass_ratio", row) < 1.0,
                    "{what}: (A / (A + 1))^2 is below one"
                );
            }
            Some(EnergyDistribution::DiscretePhoton {
                primary_flag,
                energy,
                atomic_weight_ratio,
            }) => {
                assert_only_scalars(
                    &batch,
                    row,
                    &[
                        "energy_primary_flag",
                        "energy_atomic_weight_ratio",
                        "energy_discrete_energy",
                    ],
                    &what,
                );
                assert_eq!(
                    i32_at(&batch, "energy_primary_flag", row),
                    *primary_flag as i32,
                    "{what}: energy_primary_flag decides whether the energy is the photon's \
                     or a binding energy"
                );
                assert_eq!(
                    f64_at(&batch, "energy_discrete_energy", row),
                    *energy,
                    "{what}: energy_discrete_energy in eV"
                );
                assert_eq!(
                    f64_at(&batch, "energy_atomic_weight_ratio", row),
                    *atomic_weight_ratio,
                    "{what}: energy_atomic_weight_ratio"
                );
                // The parser takes it from the ACE header, not from the block.
                assert_eq!(
                    f64_at(&batch, "energy_atomic_weight_ratio", row),
                    table.atomic_weight_ratio,
                    "{what}: and the header is where it came from"
                );
                assert_eq!(
                    f64_at(&batch, "energy_atomic_weight_ratio", row),
                    5.96345,
                    "{what}: Li6's atomic weight ratio"
                );
            }
            // The continuous rows fill no scalar column at all, and neither
            // does the elastic row with no energy distribution or either
            // Kalbach-Mann row.
            _ => assert_only_scalars(&batch, row, &[], &what),
        }
    }

    assert_eq!(
        types,
        BTreeMap::from([
            ("kalbach-mann".to_string(), 2),
            ("uncorrelated".to_string(), 16)
        ]),
        "Li6 gives 16 uncorrelated distributions and 2 Kalbach-Mann"
    );
    assert_eq!(
        energy_types,
        BTreeMap::from([
            ("continuous".to_string(), 3),
            ("discrete_photon".to_string(), 7),
            ("level".to_string(), 5),
        ]),
        "ACE laws 4, 2 and 3/33, with the elastic row leaving energy_dist_type null"
    );
    assert_eq!(
        applicable, 7,
        "one applicability per DLW chain entry, and NXS(5) is 7"
    );
    // The literal grid sizes, so a fixture that quietly lost its angular
    // detail could not leave the comparisons above passing on nothing: the
    // elastic row is 51 energies, the MT 51 to 55 neutrons run 28, 24, 21, 19
    // and 18, and every photon is the two-point isotropic pair.
    assert_eq!(
        angle_grids,
        vec![51, 2, 28, 2, 24, 2, 21, 2, 19, 2, 18, 2, 2, 2, 2, 2],
        "the angular grids Li6 holds, in row order, with the two Kalbach-Mann rows absent"
    );
    assert_eq!(
        angle_interp_seen,
        BTreeSet::from([1, 2]),
        "both interpolation codes appear, so the column is not a constant that happens to fit"
    );
}

/// Every column of one continuous-tabular row, against the law it came from.
///
/// Returns the interpolation codes and discrete-line counts it derived, so a
/// caller can pin the ones a literal makes sharper.
#[track_caller]
fn check_continuous_row(
    batch: &RecordBatch,
    row: usize,
    what: &str,
    breakpoints: &[i32],
    interpolation: &[i32],
    energy: &[f64],
    energy_out: &[Univariate],
) -> (Vec<i32>, Vec<i32>) {
    assert_not_null(batch, &CONTINUOUS_COLUMNS, row, what);

    // Already eV: ace_incident_grid converted them at mf5.rs:315-318.
    assert_f64_slice_eq(
        &format!("{what} energy_dist_energies"),
        &f64_list(batch, "energy_dist_energies", row),
        energy,
    );
    assert_eq!(
        energy.len(),
        energy_out.len(),
        "{what}: one outgoing table per incident energy"
    );

    // The two ENDF fields share one column, breakpoints first. The reader
    // splits at len / 2, so both halves have to be there and in that order.
    let concatenated: Vec<i32> = breakpoints.iter().chain(interpolation).copied().collect();
    let written_interp = i32_list(batch, "energy_dist_interpolation", row);
    assert_i32_slice_eq(
        &format!("{what} energy_dist_interpolation"),
        &written_interp,
        &concatenated,
    );
    assert_eq!(written_interp.len() % 2, 0, "{what}: an even length");
    assert_eq!(
        breakpoints.len(),
        interpolation.len(),
        "{what}: one interpolation code per region"
    );
    let half = written_interp.len() / 2;
    assert_i32_slice_eq(
        &format!("{what} the interpolation half the reader takes"),
        &written_interp[half..],
        interpolation,
    );

    let (mut x, mut p, mut c, mut lengths, mut interp, mut n_discrete) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let mut leading_discrete: Vec<Vec<f64>> = Vec::new();
    for out in energy_out {
        let (ox, op, oc, oi, on) = expect_flat(out);
        lengths.push(ox.len());
        leading_discrete.push(ox[..on as usize].to_vec());
        x.extend(ox);
        p.extend(op);
        c.extend(oc);
        interp.push(oi);
        n_discrete.push(on);
    }

    let data_col = f64_list(batch, "energy_dist_data", row);
    assert_ravel3(&format!("{what} energy_dist_data"), &data_col, &x, &p, &c);
    let offsets = i32_list(batch, "energy_dist_offsets", row);
    assert_offsets(
        &format!("{what} energy_dist_offsets"),
        &offsets,
        &lengths,
        data_col.len(),
        3,
    );
    // A Discrete gives 1 always, hard-coded at univariate_flat.rs:98 whatever
    // the ACE INTT said, which is a convention with no counterpart in
    // endf::univariate::Discrete; a Mixture takes the code of its first
    // continuous part.
    let written_out_interp = i32_list(batch, "energy_dist_out_interp", row);
    assert_i32_slice_eq(
        &format!("{what} energy_dist_out_interp"),
        &written_out_interp,
        &interp,
    );
    let written_discrete = i32_list(batch, "energy_dist_n_discrete", row);
    assert_i32_slice_eq(
        &format!("{what} energy_dist_n_discrete"),
        &written_discrete,
        &n_discrete,
    );

    for (i, &n) in written_discrete.iter().enumerate() {
        assert!(
            (n as usize) <= lengths[i],
            "{what}: sub-table {i} labels {n} of its {} points as discrete lines",
            lengths[i]
        );
        // The part a load cannot see: the leading n_discrete points of the
        // written table must be the parsed Discrete component's own energies,
        // which is what makes "discrete lines come first" true rather than
        // merely counted.
        let start = offsets[i] as usize;
        assert_f64_slice_eq(
            &format!("{what} sub-table {i} discrete lines"),
            &data_col[start..start + n as usize],
            &leading_discrete[i],
        );
    }

    (written_out_interp, written_discrete)
}

/// The columns a continuous-tabular row fills, and every other row leaves null.
const CONTINUOUS_COLUMNS: [&str; 6] = [
    "energy_dist_energies",
    "energy_dist_interpolation",
    "energy_dist_data",
    "energy_dist_offsets",
    "energy_dist_out_interp",
    "energy_dist_n_discrete",
];

/// The parsed pieces of a continuous-tabular law, where the row holds one.
type Continuous<'a> = (&'a [i32], &'a [i32], &'a [f64], &'a [Univariate]);

fn as_continuous(dist: &AngleEnergy) -> Option<Continuous<'_>> {
    match dist {
        AngleEnergy::Uncorrelated(u) => match &u.energy {
            Some(EnergyDistribution::ContinuousTabular {
                breakpoints,
                interpolation,
                energy,
                energy_out,
            }) => Some((breakpoints, interpolation, energy, energy_out)),
            _ => None,
        },
        _ => None,
    }
}

/// A failure means an outgoing-energy table was raveled wrongly, its region
/// boundaries were written in the wrong half of the concatenated interpolation
/// column, or its discrete-line count no longer matches the points it labels.
/// Writing the interpolation before the breakpoints loads fine and inverts the
/// histogram decision at `nuclide_arrow.rs:1288-1290`, which issue #499 already
/// cost once, and a wrong `n_discrete` loads cleanly and samples a plausible
/// spectrum with the wrong shape, which is what `univariate_flat.rs:8-20` says.
///
/// Li6 is the fixture for the values and the synthetic law 4 is the fixture for
/// the two interpolation and discrete-line codes: every one of Li6's 67
/// outgoing tables is histogram-interpolated, so its `energy_dist_out_interp`
/// is a column of 1s that a writer could hard-code and still pass. The
/// synthetic row's scaffold is built by hand, as
/// `the_laws_only_an_assembled_table_can_reach_are_written_from_the_parsed_blocks`
/// explains, but the three outgoing tables in it are parsed out of the
/// vendored bytes.
#[test]
fn a_continuous_tabular_energy_distribution_keeps_its_discrete_lines_first() {
    let data = li6_ace();
    let walk = distribution_walk(&data);
    let tmp = scratch();
    let batch = distributions_of(&data, tmp.path());

    let mut incident_grids: Vec<usize> = Vec::new();
    let mut discrete_lines = 0;
    for (row, &(mt, product_idx, dist_idx, dist, _)) in walk.iter().enumerate() {
        let what = format!("Li6 MT {mt} product {product_idx} distribution {dist_idx}");
        let Some((breakpoints, interpolation, energy, energy_out)) = as_continuous(dist) else {
            assert_null(&batch, &CONTINUOUS_COLUMNS, row, &what);
            continue;
        };
        incident_grids.push(energy.len());
        let (_, n_discrete) = check_continuous_row(
            &batch,
            row,
            &what,
            breakpoints,
            interpolation,
            energy,
            energy_out,
        );
        discrete_lines += n_discrete.iter().sum::<i32>() as usize;
    }

    assert_eq!(
        incident_grids,
        vec![16, 8, 43],
        "Li6's three law-4 products, in row order: the MT 5 photon, the MT 91 photon and MT 102"
    );
    // Not a vacuous column on this fixture: the MT 91 photon carries 56
    // discrete lines at each of 8 incident energies and MT 102 carries 35 at
    // each of 43, while the MT 5 photon's 16 tables are pure continuum.
    assert_eq!(
        discrete_lines,
        8 * 56 + 43 * 35,
        "the discrete-line counts Li6's mixtures hold"
    );

    // The same columns on the synthetic law 4, whose three outgoing tables are
    // a plain continuum, a pure set of discrete lines and a mixture of the two.
    // Li6 cannot discriminate `energy_dist_out_interp`, because every one of
    // its 67 outgoing tables is histogram-interpolated and so writes a 1, and
    // it holds no `Univariate::Discrete` at all.
    let (_, laws, _) = laws_nuclide();
    let laws_tmp = scratch();
    let laws_batch = distributions_of(&laws, laws_tmp.path());
    let row = (0..laws_batch.num_rows())
        .find(|&r| i32_at(&laws_batch, "reaction_mt", r) == 1004)
        .expect("the law 4 row");
    let (breakpoints, interpolation, energy, energy_out) =
        as_continuous(law_dist(&laws, 4)).expect("law 4 is a continuous tabulation");
    let (out_interp, n_discrete) = check_continuous_row(
        &laws_batch,
        row,
        "law 4",
        breakpoints,
        interpolation,
        energy,
        energy_out,
    );
    // The middle table is written with INTT=2 in the file and must still come
    // out as 1, because a discrete line is sampled by its own rule; the third
    // is a mixture whose continuous part is a histogram.
    assert_i32_slice_eq("law 4 energy_dist_out_interp", &out_interp, &[2, 1, 1]);
    assert_i32_slice_eq("law 4 energy_dist_n_discrete", &n_discrete, &[0, 2, 1]);
}

/// A failure means the two Kalbach parameters were dropped, padded or raveled
/// at the wrong stride. `km_data` is a `(5, total)` ravel while `angle_mu_data`
/// and `energy_dist_data` are `(3, total)`, so an offset check that divided by
/// 3 would be wrong here and this test states the divisor.
///
/// `km_n_discrete` is all zeros on Li6, because every outgoing table of both
/// its Kalbach-Mann rows is a pure `Tabular`, so nothing in the loop below can
/// tell that column from a literal zero. The CONSTRUCTED row at the end of
/// this test is here for that one column, and its helper says why it is built
/// rather than parsed.
#[test]
fn a_kalbach_mann_distribution_carries_its_precompound_and_slope_rows() {
    let data = li6_ace();
    let walk = distribution_walk(&data);
    let tmp = scratch();
    let batch = distributions_of(&data, tmp.path());

    let mut incident_grids: Vec<usize> = Vec::new();
    for (row, &(mt, product_idx, dist_idx, dist, _)) in walk.iter().enumerate() {
        let what = format!("Li6 MT {mt} product {product_idx} distribution {dist_idx}");
        let AngleEnergy::KalbachMann(k) = dist else {
            assert_null(&batch, &KM_COLUMNS, row, &what);
            continue;
        };
        incident_grids.push(k.energy.len());
        assert_not_null(&batch, &KM_COLUMNS, row, &what);

        assert_f64_slice_eq(
            &format!("{what} km_energies"),
            &f64_list(&batch, "km_energies", row),
            &k.energy,
        );
        assert_eq!(
            k.energy.len(),
            k.energy_out.len(),
            "{what}: one outgoing table per incident energy"
        );

        // Separate columns here, where the continuous-tabular case puts the
        // same two ENDF fields end to end in one. Both shapes are asserted so a
        // future unification cannot quietly change one of them.
        let breakpoints = i32_list(&batch, "km_breakpoints", row);
        let interpolation = i32_list(&batch, "km_interpolation", row);
        assert_i32_slice_eq(
            &format!("{what} km_breakpoints"),
            &breakpoints,
            &k.breakpoints,
        );
        assert_i32_slice_eq(
            &format!("{what} km_interpolation"),
            &interpolation,
            &k.interpolation,
        );
        assert!(
            !breakpoints.is_empty(),
            "{what}: ace_incident_grid falls back to vec![n_energy_in], so this is never empty"
        );
        assert_eq!(
            breakpoints.len(),
            interpolation.len(),
            "{what}: one per region"
        );

        let (mut x, mut p, mut c, mut lengths, mut interp, mut n_discrete) = (
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        for out in &k.energy_out {
            let (ox, op, oc, oi, on) = expect_flat(out);
            lengths.push(ox.len());
            x.extend(ox);
            p.extend(op);
            c.extend(oc);
            interp.push(oi);
            n_discrete.push(on);
        }
        let total: usize = lengths.iter().sum();

        let data_col = f64_list(&batch, "km_data", row);
        assert_eq!(
            data_col.len(),
            5 * total,
            "{what}: km_data is (5, {total}), not (3, {total})"
        );
        assert_ravel3(
            &format!("{what} km_data rows 0 to 2"),
            &data_col[..3 * total],
            &x,
            &p,
            &c,
        );

        // r and a are tabulated against the same outgoing energies. The
        // writer's zero-padding branch at distributions.rs:411-418 cannot fire
        // on ACE data, because KalbachMann::from_ace builds both from
        // out.data[3] and out.data[4], which are as long as the table itself.
        let mut precompound = Vec::new();
        let mut slope = Vec::new();
        for (i, &n) in lengths.iter().enumerate() {
            assert_eq!(
                k.precompound[i].y.len(),
                n,
                "{what}: the precompound fraction is one value per outgoing point, so the \
                 writer never pads"
            );
            assert_eq!(k.slope[i].y.len(), n, "{what}: and so is the slope");
            precompound.extend_from_slice(&k.precompound[i].y);
            slope.extend_from_slice(&k.slope[i].y);
        }
        assert_f64_slice_eq(
            &format!("{what} km_data row 3, the precompound fraction"),
            &data_col[3 * total..4 * total],
            &precompound,
        );
        assert_f64_slice_eq(
            &format!("{what} km_data row 4, the slope"),
            &data_col[4 * total..],
            &slope,
        );

        assert_offsets(
            &format!("{what} km_offsets"),
            &i32_list(&batch, "km_offsets", row),
            &lengths,
            data_col.len(),
            5,
        );
        assert_i32_slice_eq(
            &format!("{what} km_interp"),
            &i32_list(&batch, "km_interp", row),
            &interp,
        );
        assert_i32_slice_eq(
            &format!("{what} km_n_discrete"),
            &i32_list(&batch, "km_n_discrete", row),
            &n_discrete,
        );
    }
    assert_eq!(
        incident_grids,
        vec![16, 2],
        "Li6's two Kalbach-Mann products: the MT 5 neutron and the MT 91 neutron"
    );

    // The other half of the shape claim above, read off the MT 5 photon, whose
    // law-4 distribution concatenates the same two fields into one column.
    let continuous_row = walk
        .iter()
        .position(|&(mt, idx, _, _, _)| mt == 5 && idx == 1)
        .expect("the MT 5 photon");
    let AngleEnergy::Uncorrelated(u) = walk[continuous_row].3 else {
        panic!("the MT 5 photon is uncorrelated");
    };
    let Some(EnergyDistribution::ContinuousTabular {
        breakpoints,
        interpolation,
        ..
    }) = &u.energy
    else {
        panic!("the MT 5 photon is a law-4 continuous tabulation");
    };
    assert_eq!(
        list_len(&batch, "energy_dist_interpolation", continuous_row),
        breakpoints.len() + interpolation.len(),
        "the continuous case writes both fields into one column"
    );

    // `km_n_discrete`, on a CONSTRUCTED input: see
    // `constructed_kalbach_mann_with_lines`. The loop above passes with the
    // column hard-coded to zero, because Li6 holds no Kalbach-Mann mixture.
    let built = constructed_kalbach_mann_with_lines();
    let built_tmp = scratch();
    let built_batch = distributions_of(&built, built_tmp.path());
    assert_eq!(built_batch.num_rows(), 1);
    let AngleEnergy::KalbachMann(k) = &built.reactions[&28].products[0].distribution[0] else {
        panic!("the constructed row is Kalbach-Mann");
    };
    let (mut n_discrete, mut leading_lines) = (Vec::new(), Vec::new());
    for out in &k.energy_out {
        let (ox, _, _, _, on) = expect_flat(out);
        leading_lines.push(ox[..on as usize].to_vec());
        n_discrete.push(on);
    }
    let written_discrete = i32_list(&built_batch, "km_n_discrete", 0);
    assert_i32_slice_eq(
        "the constructed km_n_discrete",
        &written_discrete,
        &n_discrete,
    );
    assert_i32_slice_eq(
        "the constructed km_n_discrete, whose two tables differ so neither a \
         hard-coded count nor a copied one can pass",
        &written_discrete,
        &[0, 2],
    );
    // And the points the count labels are the mixture's own lines, which is
    // what makes it "lines first" rather than a number beside the table.
    let built_data = f64_list(&built_batch, "km_data", 0);
    let built_offsets = i32_list(&built_batch, "km_offsets", 0);
    for (i, &n) in written_discrete.iter().enumerate() {
        let start = built_offsets[i] as usize;
        assert_f64_slice_eq(
            &format!("the constructed km sub-table {i} discrete lines"),
            &built_data[start..start + n as usize],
            &leading_lines[i],
        );
    }
}

/// A failure means one of the five laws no vendored nuclide reaches was written
/// wrongly. `synthetic-laws.ace.xz` cannot be read as an `IncidentNeutron` (its
/// JXS(1) ESZ, JXS(3) MTR, JXS(6) LSIG and JXS(7) SIG are all zero, so
/// `IncidentNeutron::from_ace` fails), so the scaffold around the distributions
/// is SYNTHETIC while the distributions themselves come out of
/// `endf::AngleEnergy::from_ace` reading the vendored bytes. The comparison is
/// still file against parser; only the enclosing `Reaction` and `Product` are
/// built by hand.
///
/// Three things here are pinned rather than discriminated, and are worth
/// reading as such. `corr_eout_n_discrete` is `[0, 0]`, because both outgoing
/// tables of law 61 are pure `Tabular`. `corr_breakpoints` and
/// `corr_interpolation` both hold `[1, 2]` on that same row, so this test
/// cannot tell the two columns apart and stays green with the writer's two
/// assignments swapped. Those two gaps are closed by
/// `correlated_region_columns_and_line_counts_come_from_the_distribution`, on
/// an input built by hand rather than parsed. The third is not closable at
/// all: the choice between writing an interpolation VALUE and an index into
/// `corr_mu_interp` in row 3 of `corr_eout_data` cannot be resolved by any
/// test, so the convention `distributions.rs:327-334` describes is what is
/// asserted.
#[test]
fn the_laws_only_an_assembled_table_can_reach_are_written_from_the_parsed_blocks() {
    let (table, data, laws) = laws_nuclide();
    assert_eq!(
        laws,
        vec![2, 4, 7, 9, 11, 61, 66],
        "the seven laws tools/make_laws_ace.py writes"
    );
    let tmp = scratch();
    let batch = distributions_of(&data, tmp.path());
    assert_schema_is_declared(&batch, "distributions.arrow");
    assert_eq!(batch.num_rows(), 7, "one row per law");
    assert_i32_slice_eq(
        "the scaffold's MTs, which are 1000 + the law number",
        &i32_column(&batch, "reaction_mt"),
        &[1002, 1004, 1007, 1009, 1011, 1061, 1066],
    );

    let row_of = |law: i64| -> usize {
        (0..batch.num_rows())
            .find(|&r| i32_at(&batch, "reaction_mt", r) == 1000 + law as i32)
            .expect("every law has a row")
    };

    // -- law 2: the only discrete photon in the tree with a set flag ---------
    //
    // Li6 has seven discrete-photon rows and every one carries
    // `primary_flag = 0`, so the uncorrelated test cannot tell that column
    // from a literal zero. This row carries 1, which is the difference between
    // the stored energy being the photon's own and it being a binding energy
    // the sampler subtracts.
    {
        let row = row_of(2);
        let what = "law 2";
        let (primary_flag, photon_energy, awr) = match law_dist(&data, 2) {
            AngleEnergy::Uncorrelated(unc) => match &unc.energy {
                Some(EnergyDistribution::DiscretePhoton {
                    primary_flag,
                    energy,
                    atomic_weight_ratio,
                }) => (primary_flag, energy, atomic_weight_ratio),
                other => panic!("{what}: parsed as {other:?}"),
            },
            other => panic!("{what}: parsed as {other:?}"),
        };
        assert_eq!(str_at(&batch, "type", row), "uncorrelated", "{what}");
        assert_eq!(
            str_at(&batch, "energy_dist_type", row),
            "discrete_photon",
            "{what}"
        );
        assert_only_scalars(
            &batch,
            row,
            &[
                "energy_primary_flag",
                "energy_atomic_weight_ratio",
                "energy_discrete_energy",
            ],
            what,
        );
        assert_eq!(
            i32_at(&batch, "energy_primary_flag", row),
            *primary_flag as i32,
            "{what}: energy_primary_flag"
        );
        assert_eq!(
            i32_at(&batch, "energy_primary_flag", row),
            1,
            "{what}: the one vendored row where the flag is not zero, which is what \
             stops the column from being indistinguishable from a hard-coded 0"
        );
        assert_eq!(
            f64_at(&batch, "energy_discrete_energy", row),
            *photon_energy,
            "{what}: energy_discrete_energy in eV"
        );
        assert_eq!(
            f64_at(&batch, "energy_atomic_weight_ratio", row),
            *awr,
            "{what}: energy_atomic_weight_ratio"
        );
        assert_eq!(
            f64_at(&batch, "energy_atomic_weight_ratio", row),
            table.atomic_weight_ratio,
            "{what}: which the parser took from the ACE header"
        );
    }

    // -- laws 7 and 9: one tabulated parameter and a restriction energy ------
    for law in [7, 9] {
        let row = row_of(law);
        let what = format!("law {law}");
        let (name, u, theta) = match law_dist(&data, law) {
            AngleEnergy::Uncorrelated(unc) => match &unc.energy {
                Some(EnergyDistribution::MaxwellEnergy { u, theta }) => ("maxwell", u, theta),
                Some(EnergyDistribution::Evaporation { u, theta }) => ("evaporation", u, theta),
                other => panic!("{what}: parsed as {other:?}"),
            },
            other => panic!("{what}: parsed as {other:?}"),
        };
        assert_eq!(str_at(&batch, "type", row), "uncorrelated", "{what}");
        assert_eq!(str_at(&batch, "energy_dist_type", row), name, "{what}");
        assert_f64_slice_eq(
            &format!("{what} energy_param_x"),
            &f64_list(&batch, "energy_param_x", row),
            &theta.x,
        );
        assert_f64_slice_eq(
            &format!("{what} energy_param_y"),
            &f64_list(&batch, "energy_param_y", row),
            &theta.y,
        );
        assert_only_scalars(&batch, row, &["energy_restriction_u"], &what);
        assert_eq!(
            f64_at(&batch, "energy_restriction_u", row),
            *u,
            "{what}: the reader defaults a null restriction energy to zero, which is a \
             different spectrum"
        );
        assert_null(&batch, &["energy_param2_x", "energy_param2_y"], row, &what);

        // The deliberate loss, asserted rather than left to be discovered: the
        // parsed Tabulated1D has regions and the format has nowhere to put
        // them, so the reader reconstructs it with none.
        assert!(
            !theta.breakpoints.is_empty(),
            "{what}: the parsed side does carry region boundaries"
        );
        assert_null(&batch, &INT_LIST_COLUMNS, row, &what);
    }

    // -- law 11: the only row that can tell param from param2 ----------------
    {
        let row = row_of(11);
        let what = "law 11";
        let (u, a, b) = match law_dist(&data, 11) {
            AngleEnergy::Uncorrelated(unc) => match &unc.energy {
                Some(EnergyDistribution::WattEnergy { u, a, b }) => (u, a, b),
                other => panic!("{what}: parsed as {other:?}"),
            },
            other => panic!("{what}: parsed as {other:?}"),
        };
        assert_eq!(str_at(&batch, "type", row), "uncorrelated");
        assert_eq!(str_at(&batch, "energy_dist_type", row), "watt");
        assert_f64_slice_eq(
            "law 11 energy_param_x",
            &f64_list(&batch, "energy_param_x", row),
            &a.x,
        );
        assert_f64_slice_eq(
            "law 11 energy_param_y",
            &f64_list(&batch, "energy_param_y", row),
            &a.y,
        );
        assert_f64_slice_eq(
            "law 11 energy_param2_x",
            &f64_list(&batch, "energy_param2_x", row),
            &b.x,
        );
        assert_f64_slice_eq(
            "law 11 energy_param2_y",
            &f64_list(&batch, "energy_param2_y", row),
            &b.y,
        );
        assert_only_scalars(&batch, row, &["energy_restriction_u"], what);
        assert_eq!(f64_at(&batch, "energy_restriction_u", row), *u);
        // a and b differ by more than eleven orders of magnitude in this
        // fixture (9.0e5 against 3.0e-6), so a swap of the two pairs is
        // unmistakable rather than merely unequal.
        assert!(
            a.y.iter().all(|&v| v > 1.0e5) && b.y.iter().all(|&v| v < 1.0e-3),
            "law 11: the fixture's a and b are far apart, which is what makes the swap visible"
        );
        assert_null(&batch, &INT_LIST_COLUMNS, row, what);
    }

    // -- law 61: correlated angle and energy ---------------------------------
    {
        let row = row_of(61);
        let what = "law 61";
        let AngleEnergy::Correlated(c) = law_dist(&data, 61) else {
            panic!("{what}: parsed as something else");
        };
        assert_eq!(str_at(&batch, "type", row), "correlated");
        assert!(
            is_null(&batch, "energy_dist_type", row),
            "{what}: a correlated law fills no energy_dist_type"
        );
        assert_f64_slice_eq(
            "law 61 corr_energies",
            &f64_list(&batch, "corr_energies", row),
            &c.energy,
        );
        let breakpoints = i32_list(&batch, "corr_breakpoints", row);
        let interpolation = i32_list(&batch, "corr_interpolation", row);
        assert_i32_slice_eq("law 61 corr_breakpoints", &breakpoints, &c.breakpoints);
        assert_i32_slice_eq(
            "law 61 corr_interpolation",
            &interpolation,
            &c.interpolation,
        );
        // Two regions, which is what separates the correlated law's two
        // columns from the continuous law's single concatenated one: a
        // concatenated write would be four values long. It does NOT separate
        // the two columns from each other, because both hold [1, 2] here, and
        // that takes the constructed row in
        // correlated_region_columns_and_line_counts_come_from_the_distribution.
        assert_eq!(breakpoints, vec![1, 2], "law 61 is deliberately two-region");
        assert_eq!(interpolation, vec![1, 2]);

        let (mut x, mut p, mut cdf, mut lengths, mut interp, mut n_discrete) = (
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        for out in &c.energy_out {
            let (ox, op, oc, oi, on) = expect_flat(out);
            lengths.push(ox.len());
            x.extend(ox);
            p.extend(op);
            cdf.extend(oc);
            interp.push(oi);
            n_discrete.push(on);
        }
        let total: usize = lengths.iter().sum();

        // One cosine table per outgoing POINT, in row-major (incident,
        // outgoing) order. The writer's missing-mu fallback at
        // distributions.rs:317-324 cannot fire, because
        // CorrelatedAngleEnergy::from_ace fills mu[i] to the point count.
        let (mut mx, mut mp, mut mc, mut mu_lengths, mut mu_interp) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (i, &n) in lengths.iter().enumerate() {
            assert_eq!(
                c.mu[i].len(),
                n,
                "{what}: one cosine table per outgoing energy of incident energy {i}"
            );
            for mu in &c.mu[i] {
                let (ux, up, uc, ui, un) = expect_flat(mu);
                assert_eq!(un, 0, "{what}: a cosine table holds no discrete lines");
                mu_lengths.push(ux.len());
                mx.extend(ux);
                mp.extend(up);
                mc.extend(uc);
                mu_interp.push(ui);
            }
        }

        let eout = f64_list(&batch, "corr_eout_data", row);
        assert_eq!(
            eout.len(),
            5 * total,
            "{what}: corr_eout_data is (5, {total})"
        );
        assert_ravel3(
            "law 61 corr_eout_data rows 0 to 2",
            &eout[..3 * total],
            &x,
            &p,
            &cdf,
        );
        assert_offsets(
            "law 61 corr_eout_offsets",
            &i32_list(&batch, "corr_eout_offsets", row),
            &lengths,
            eout.len(),
            5,
        );
        assert_eq!(
            i32_list(&batch, "corr_eout_offsets", row),
            vec![0, 2],
            "a two-point table then a three-point one"
        );
        // Deliberately non-constant, so a writer that hard-coded 2 is caught.
        assert_i32_slice_eq(
            "law 61 corr_eout_interp",
            &i32_list(&batch, "corr_eout_interp", row),
            &interp,
        );
        assert_eq!(i32_list(&batch, "corr_eout_interp", row), vec![2, 1]);
        assert_i32_slice_eq(
            "law 61 corr_eout_n_discrete",
            &i32_list(&batch, "corr_eout_n_discrete", row),
            &n_discrete,
        );

        let mu_data = f64_list(&batch, "corr_mu_data", row);
        assert_ravel3("law 61 corr_mu_data", &mu_data, &mx, &mp, &mc);
        let mu_offsets = i32_list(&batch, "corr_mu_offsets", row);
        assert_offsets(
            "law 61 corr_mu_offsets",
            &mu_offsets,
            &mu_lengths,
            mu_data.len(),
            3,
        );
        // Global across incident energies, not restarting per energy: the
        // second incident energy's first table does not begin at zero.
        assert_eq!(
            mu_offsets[lengths[0]],
            mu_lengths[..lengths[0]].iter().sum::<usize>() as i32
        );
        assert!(mu_offsets[lengths[0]] > 0, "{what}: the offsets are global");

        let written_mu_interp = i32_list(&batch, "corr_mu_interp", row);
        assert_i32_slice_eq("law 61 corr_mu_interp", &written_mu_interp, &mu_interp);
        assert_eq!(
            written_mu_interp,
            vec![1, 2, 2, 1, 2],
            "isotropic cosines flatten to 1 and lin-lin tabulations to 2, so this column is \
             deliberately not uniform"
        );

        // Rows 3 and 4 of the outgoing ravel are not data: they repeat the
        // per-point cosine interpolation and offset as f64, and the reader's
        // legacy fallback depends on the two agreeing.
        for k in 0..total {
            assert_eq!(
                eout[3 * total + k],
                written_mu_interp[k] as f64,
                "{what}: corr_eout_data row 3 point {k} against corr_mu_interp"
            );
            assert_eq!(
                eout[4 * total + k],
                mu_offsets[k] as f64,
                "{what}: corr_eout_data row 4 point {k} against corr_mu_offsets"
            );
        }
    }

    // -- law 66: N-body phase space ------------------------------------------
    {
        let row = row_of(66);
        let what = "law 66";
        let AngleEnergy::NBodyPhaseSpace(n) = law_dist(&data, 66) else {
            panic!("{what}: parsed as something else");
        };
        assert_eq!(str_at(&batch, "type", row), "nbody");
        assert_only_scalars(
            &batch,
            row,
            &[
                "nbody_n",
                "nbody_total_mass",
                "nbody_atomic_weight_ratio",
                "nbody_q_value",
            ],
            what,
        );
        // Non-null is half the assertion: the reader does unwrap_or(0) and zero
        // bodies is a kinematically impossible distribution that loads without
        // complaint.
        assert_eq!(i32_at(&batch, "nbody_n", row), n.n_particles as i32);
        assert_eq!(i32_at(&batch, "nbody_n", row), 4);
        assert_eq!(f64_at(&batch, "nbody_total_mass", row), n.total_mass);
        assert_eq!(f64_at(&batch, "nbody_total_mass", row), 3.98);
        // Asserted against the ACE header rather than the parsed struct: that
        // is the provenance the parser chose at angle_energy.rs:247, and it is
        // the one field here not read out of the distribution block.
        assert_eq!(
            f64_at(&batch, "nbody_atomic_weight_ratio", row),
            table.atomic_weight_ratio
        );
        assert_eq!(f64_at(&batch, "nbody_atomic_weight_ratio", row), 0.999167);
        assert_eq!(f64_at(&batch, "nbody_q_value", row), n.q_value);
        assert_eq!(
            f64_at(&batch, "nbody_q_value", row),
            law_q(66),
            "the Q comes from the caller, so a dropped field is only visible against a \
             distinctive one"
        );
    }

    // A row fills the columns its own discriminant names and no others, which
    // is what lets the reader switch on `type` and read nothing else.
    for row in 0..batch.num_rows() {
        let ty = str_at(&batch, "type", row);
        let what = format!("the {ty} row");
        if ty != "nbody" {
            assert_null(
                &batch,
                &[
                    "nbody_n",
                    "nbody_total_mass",
                    "nbody_atomic_weight_ratio",
                    "nbody_q_value",
                ],
                row,
                &what,
            );
        }
        if ty != "correlated" {
            assert_null(&batch, &CORR_COLUMNS, row, &what);
        } else {
            assert_null(&batch, &CONTINUOUS_COLUMNS, row, &what);
        }
        // Nothing in this fixture is Kalbach-Mann, and no row may pretend to be.
        assert_null(&batch, &KM_COLUMNS, row, &what);
    }

    // -- the two discriminants, across both fixtures -------------------------
    let li6 = li6_ace();
    let li6_tmp = scratch();
    let li6_batch = distributions_of(&li6, li6_tmp.path());

    let mut types: BTreeSet<String> = BTreeSet::new();
    let mut energy_types: BTreeSet<String> = BTreeSet::new();
    for b in [&batch, &li6_batch] {
        for row in 0..b.num_rows() {
            types.insert(str_at(b, "type", row));
            if !is_null(b, "energy_dist_type", row) {
                energy_types.insert(str_at(b, "energy_dist_type", row));
            }
        }
    }
    // The reader refuses an unknown discriminant, so a fifth endf::AngleEnergy
    // variant must not be able to slip through as an unwritten row.
    assert_eq!(
        types,
        ["correlated", "kalbach-mann", "nbody", "uncorrelated"]
            .map(String::from)
            .into(),
        "the four the reader accepts at nuclide_arrow.rs:1134-1137"
    );
    assert_eq!(
        energy_types,
        [
            "continuous",
            "discrete_photon",
            "evaporation",
            "level",
            "maxwell",
            "watt"
        ]
        .map(String::from)
        .into(),
        "the six the reader accepts at nuclide_arrow.rs:1442-1452"
    );
}

/// A failure means a correlated row's region boundaries, its interpolation
/// codes or its discrete-line counts came from somewhere other than the
/// distribution they belong to.
///
/// The input is CONSTRUCTED and not parsed, and
/// `constructed_correlated_regions` says why. Law 61 of
/// `synthetic-laws.ace.xz` is the only correlated distribution any vendored
/// bytes produce, and on it `corr_breakpoints` and `corr_interpolation` both
/// hold `[1, 2]` while `corr_eout_n_discrete` is `[0, 0]`, so two real writer
/// defects (the two region assignments swapped, and a hard-coded line count)
/// leave every other test in this file green. Nothing here is evaluation
/// parity: the values in a correlated row are checked against parsed bytes in
/// `the_laws_only_an_assembled_table_can_reach_are_written_from_the_parsed_blocks`,
/// and this test checks only that the writer copies the right field.
#[test]
fn correlated_region_columns_and_line_counts_come_from_the_distribution() {
    let data = constructed_correlated_regions();
    let tmp = scratch();
    let batch = distributions_of(&data, tmp.path());
    assert_eq!(batch.num_rows(), 1);
    let AngleEnergy::Correlated(c) = &data.reactions[&22].products[0].distribution[0] else {
        panic!("the constructed row is correlated");
    };
    assert_eq!(str_at(&batch, "type", 0), "correlated");
    assert_f64_slice_eq(
        "the constructed corr_energies",
        &f64_list(&batch, "corr_energies", 0),
        &c.energy,
    );

    let breakpoints = i32_list(&batch, "corr_breakpoints", 0);
    let interpolation = i32_list(&batch, "corr_interpolation", 0);
    assert_i32_slice_eq(
        "the constructed corr_breakpoints",
        &breakpoints,
        &c.breakpoints,
    );
    assert_i32_slice_eq(
        "the constructed corr_interpolation",
        &interpolation,
        &c.interpolation,
    );
    // The literals as well, because the two comparisons above are what a swap
    // has to get past and they only bite while the two disagree. These are
    // built to disagree at every position.
    assert_i32_slice_eq("the constructed corr_breakpoints", &breakpoints, &[1, 3]);
    assert_i32_slice_eq(
        "the constructed corr_interpolation",
        &interpolation,
        &[2, 1],
    );

    let (mut interp, mut n_discrete, mut leading_lines) = (Vec::new(), Vec::new(), Vec::new());
    for out in &c.energy_out {
        let (ox, _, _, oi, on) = expect_flat(out);
        leading_lines.push(ox[..on as usize].to_vec());
        interp.push(oi);
        n_discrete.push(on);
    }
    let written_discrete = i32_list(&batch, "corr_eout_n_discrete", 0);
    assert_i32_slice_eq(
        "the constructed corr_eout_n_discrete",
        &written_discrete,
        &n_discrete,
    );
    assert_i32_slice_eq(
        "the constructed corr_eout_n_discrete, three tables with three different \
         counts so neither a hard-coded nor a copied count can pass",
        &written_discrete,
        &[0, 1, 2],
    );
    assert_i32_slice_eq(
        "the constructed corr_eout_interp",
        &i32_list(&batch, "corr_eout_interp", 0),
        &interp,
    );

    // The points those counts label are the mixture's and the discrete table's
    // own energies, so the count means "these leading points are lines" rather
    // than being a number beside the table. `corr_eout_data` is a (5, total)
    // ravel and the offsets index its first row, which is where the energies
    // are.
    let eout = f64_list(&batch, "corr_eout_data", 0);
    let offsets = i32_list(&batch, "corr_eout_offsets", 0);
    for (i, &n) in written_discrete.iter().enumerate() {
        let start = offsets[i] as usize;
        assert_f64_slice_eq(
            &format!("the constructed corr sub-table {i} discrete lines"),
            &eout[start..start + n as usize],
            &leading_lines[i],
        );
    }
}

/// A failure means a Watt spectrum's second parameter was paired with the
/// first one's energy grid.
///
/// The input is CONSTRUCTED and not parsed, and
/// `constructed_watt_with_distinct_grids` says why: on law 11, the only Watt
/// spectrum in the tree, `a.x` and `b.x` are the same two numbers, so
/// `energy_param2_x = a.x.clone()` writes a file identical to the correct one
/// and every other test here stays green. This is not evaluation parity; law
/// 11 in
/// `the_laws_only_an_assembled_table_can_reach_are_written_from_the_parsed_blocks`
/// is what checks Watt against parsed bytes.
#[test]
fn a_watt_spectrum_pairs_each_parameter_with_its_own_energy_grid() {
    let data = constructed_watt_with_distinct_grids();
    let tmp = scratch();
    let batch = distributions_of(&data, tmp.path());
    assert_eq!(batch.num_rows(), 1);
    let AngleEnergy::Uncorrelated(unc) = &data.reactions[&18].products[0].distribution[0] else {
        panic!("the constructed row is uncorrelated");
    };
    let Some(EnergyDistribution::WattEnergy { u, a, b }) = &unc.energy else {
        panic!("the constructed row is a Watt spectrum");
    };
    assert_eq!(str_at(&batch, "energy_dist_type", 0), "watt");
    // The grids are built to differ in length and in value, which is the whole
    // reason this test exists.
    assert_ne!(a.x, b.x, "the two parameters are on different grids here");

    assert_f64_slice_eq(
        "the constructed energy_param_x",
        &f64_list(&batch, "energy_param_x", 0),
        &a.x,
    );
    assert_f64_slice_eq(
        "the constructed energy_param_y",
        &f64_list(&batch, "energy_param_y", 0),
        &a.y,
    );
    assert_f64_slice_eq(
        "the constructed energy_param2_x",
        &f64_list(&batch, "energy_param2_x", 0),
        &b.x,
    );
    assert_f64_slice_eq(
        "the constructed energy_param2_y",
        &f64_list(&batch, "energy_param2_y", 0),
        &b.y,
    );
    assert_only_scalars(
        &batch,
        0,
        &["energy_restriction_u"],
        "the constructed watt row",
    );
    assert_eq!(f64_at(&batch, "energy_restriction_u", 0), *u);
}

/// A failure means a refusal turned into invented data. The three errors are
/// deliberate: the format holds neutrons and photons only, it holds tabulated
/// angular distributions only, and it holds the processed ACE energy laws only.
/// Silently converting any of them would be inventing a tabulation the
/// evaluation did not give, which is what the messages say.
#[test]
fn the_writers_refuse_what_the_format_cannot_hold() {
    let tmp = scratch();

    // A recoil product the format has no column for.
    let li6 = endf_nuclide(LI6_ENDF);
    let err = yamc_convert::products::write_products(&li6, tmp.path())
        .expect_err("MT 105 emits H3 and He4")
        .to_string();
    assert!(
        err.contains("MT 105"),
        "the message names the reaction: {err}"
    );
    assert!(err.contains("H3"), "and the particle: {err}");

    // Legendre coefficients, which cannot become a tabulation without
    // inventing one.
    let err = yamc_convert::distributions::write_distributions(&li6, tmp.path())
        .expect_err("the raw evaluation gives MT 2 as Legendre coefficients")
        .to_string();
    assert!(
        err.contains("MT 2"),
        "the message names the reaction: {err}"
    );
    assert!(err.contains("Legendre"), "and the representation: {err}");

    // An unprocessed energy law. The LF=5 general evaporation belongs to
    // MF=5 MT=455, and the delayed products it describes hang off reaction
    // MT 18, which is the MT the message names: U235 carries no MF=5 MT=18.
    let u235 = endf_nuclide(U235_ENDF);
    let err = yamc_convert::distributions::write_distributions(&u235, tmp.path())
        .expect_err("the delayed products on MT 18 are LF=5")
        .to_string();
    assert!(
        err.contains("MT 18"),
        "the message names the reaction: {err}"
    );
    assert!(
        err.contains("LF=5") || err.contains("general evaporation"),
        "and the law: {err}"
    );
}
