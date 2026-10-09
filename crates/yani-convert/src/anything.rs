//! MT=5, `(n,anything)`: the products of the part of a reaction an
//! evaluation does not split into reactions of its own.
//!
//! Where every other chain reaction names one residual, MT=5 is a total with
//! several products, each with a multiplicity against incident energy, given
//! in the evaluation's MF=6 MT=5. ENDF/B-VIII.1, JEFF-4.0 and FENDL-3.2d put
//! reactions there well inside a fusion spectrum (Fe54's (n,np) and most of
//! its (n,alpha), 0.54 b of MT=5 at 14 MeV); TENDL starts it at about 30 MeV.
//! This module reads those products into `(n,X)` rows of
//! `reactions/reactions.arrow`: one per residual nucleus and one per light
//! particle (H1, H2, H3, He3, He4), each carrying its multiplicity linearized
//! exactly as the branching subsection linearizes its curves. Neutrons and
//! photons are products too, but not inventory, and are read only to check
//! the evaluation's own conservation.
//!
//! What the rows mean is the solver's business (yani-transmute's branching
//! rule): the residuals are a complete list, so their multiplicities are
//! shares of the MT=5 total of the transport library, applied pointwise; each
//! light particle is its multiplicity times that total. A complete list is one
//! that leaves a residual per reaction, and that holds for every target heavy
//! enough not to break up into light particles alone. A light one does
//! (TENDL-2025's Li6 leaves a nucleus heavier than He4 in 7% of its MT=5
//! reactions at 14 MeV, Be9 in 66%), and normalising its few residuals to the
//! whole of MT=5 would multiply them; so where the residual multiplicities
//! average below [`ONE_RESIDUAL_PER_REACTION`] over MT=5, every product is
//! written as a multiplicity (`share` false), the residuals included, and the
//! list is named under `residuals_not_one_per_reaction`. Nothing here
//! normalises, caps or fills a number. What cannot be right is recorded in the
//! reactions subsection's `provenance.json` under `mt5`, by nuclide:
//!
//! * `not_conserving`: where the products, neutrons included and weighted by
//!   multiplicity, do not give back the charge and mass number of the target
//!   plus the neutron, by more than [`CONSERVATION_TOLERANCE`] at some energy
//!   where MT=5 is open. ENDF/B-VIII.1's Cr50 to Cr54 are the large ones: their
//!   V51 (Cr52) and similar "multiplicities" run to 1e9, a cross section in
//!   barns divided by a vanishing MT=5. Every parent's worst imbalance is
//!   under `conservation`, defects or not, so the check is whole.
//! * `impossible_multiplicity`: a product more numerous than the target plus
//!   the neutron has nucleons for, `floor((A + 1) / A_product)`, which the
//!   solver clips and refuses by the size of the rate it would carry.
//! * `residual_not_given`: an evaluation whose MF=6 MT=5 does not leave a
//!   residual per reaction while its products do not carry the target's
//!   charge (95 ENDF/B-VIII.1 evaluations, Fe58 and the Zr isotopes among
//!   them, which list light particles and no residual at all). The reaction
//!   still removes its parent at the MT=5 rate and still makes its light
//!   particles, but where its residuals go is not in the evaluation; a row
//!   with a null target says so, and the solver reports that share as
//!   unmodelled. A target light enough to break up entirely (Be9 into two
//!   alphas) is told apart by charge: its products carry at least
//!   [`BREAKUP_CHARGE_SHARE`] of the target's charge over the MT=5 cross
//!   section, where a missing residual leaves them carrying a few percent.
//! * `no_product_data`: MT=5 in MF=3 with no MF=6 MT=5 at all, written as the
//!   same null-target row.
//! * `unnamed_products`: a residual with neither decay data nor a stand-in
//!   ([`endf::chain::replace_missing_in`], the rule every other reaction's
//!   product goes by), left out.
//! * `mf10_mt5`: MF=9 or MF=10 MT=5 radionuclide production, which is not
//!   read. Where both are given (ENDF/B-VIII.1's Cr and W isotopes, Pt190,
//!   Pt192, Al27, Si29 and U235) MF=10 repeats MF=6's yield times MT=5 for the
//!   products both name, and the isomer partials it adds for W are zero
//!   wherever MT=5 is open below 20 MeV; Al27's and Si29's partials sit where
//!   their own MF=3 has no MT=5 at all. MF=6 is the one file that gives every
//!   product, and the isomer split comes with it, from its LIP. Each MF=9 and
//!   MF=10 state is listed with its largest value and whether MF=6 gives its
//!   product and isomer, so a case where MF=10 would add something is visible.

use std::collections::{BTreeMap, BTreeSet};

use endf::decay::Decay;
use endf::function::Tabulated1D;
use endf::Material;

use crate::branching::{linearize, merge_duplicates, BranchingRow, DEFAULT_LINEARIZE_TOL};

/// The chain kind MT=5's rows are written under. The reader's
/// `yani::reactions::ANYTHING`, which this crate does not link.
pub const ANYTHING: &str = "(n,X)";

/// How far the products of MT=5 may sit from conserving the target's charge
/// and mass number, at some energy where MT=5 is open, before the evaluation
/// is listed as not conserving: as a fraction of the target's Z, and of its
/// A plus one. The published evaluations that conserve at all do so to a few
/// parts in a thousand (ENDF/B-VIII.1 Fe54 3.8e-3, Fe56 4.5e-3, each at an
/// energy where MT=5 is a few percent of its peak), so this lists the ones
/// that do not without burying them under rounding.
pub const CONSERVATION_TOLERANCE: f64 = 1.0e-2;

/// Below this share of the target's charge, carried by the products of a list
/// that does not leave a residual per reaction, weighted by MT=5, the residual
/// is not given rather than a target that breaks up entirely. The published
/// cases are far either side: the 95 ENDF/B-VIII.1 lists with no residual
/// carry 0.1% to 6% of the charge in their light particles, a light target's
/// breakup most of it. See the module documentation.
pub const BREAKUP_CHARGE_SHARE: f64 = 0.5;

/// The MT=5-weighted average of an evaluation's summed residual multiplicity
/// at and above which its residuals are one per reaction, a list to be read as
/// shares. The heavy targets give 0.98 to 1.03 (TENDL-2025's Ni58 1.02 at 42
/// MeV, its rounding), the light ones that break up 0.07 (Li6) to 0.68 (B10),
/// so the line sits between them with room on either side.
pub const ONE_RESIDUAL_PER_REACTION: f64 = 0.9;

/// The light particles by ZA, with the names the chain gives them.
const LIGHT: [(i64, &str); 5] = [
    (1001, "H1"),
    (1002, "H2"),
    (1003, "H3"),
    (2003, "He3"),
    (2004, "He4"),
];

/// One product of MF=6 MT=5, its multiplicity linearized.
#[derive(Debug, Clone, PartialEq)]
pub struct Product {
    /// ZA of the product, 1000*Z + A; 1 is the neutron.
    pub zap: i64,
    /// MF=6's product modifier: the residual's isomeric state.
    pub lip: i64,
    pub energy: Vec<f64>,
    pub multiplicity: Vec<f64>,
}

/// One evaluation's MT=5, read but not yet named against a decay library.
///
/// What [`read`] takes from a neutron evaluation, so a converter streaming
/// the sublibrary holds this and not the evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct Evaluation {
    /// The parent, named from MF=1 as `Chain::from_endf` names it.
    pub parent: String,
    pub z: i64,
    pub a: i64,
    /// MT=5's QI, the Q every other reaction row carries.
    pub q_value: f64,
    /// MT=5 in MF=3, as tabulated (with its own interpolation laws).
    pub sigma: Tabulated1D,
    /// MF=6 MT=5's products other than photons, the neutron among them;
    /// `None` when the evaluation has no MF=6 MT=5.
    pub products: Option<Vec<Product>>,
    /// The MF=9 and MF=10 MT=5 states, described for the record.
    pub radionuclide_states: Vec<serde_json::Value>,
}

/// One `(n,X)` row of `reactions/reactions.arrow`.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    /// The chain nuclide made: a residual or a light particle. `None` for the
    /// row that says the evaluation gives no residual.
    pub target: Option<String>,
    /// The target's isomeric state, 0 for a ground state and a light
    /// particle; `None` for the row with no target.
    pub lip: Option<i32>,
    pub energy: Vec<f64>,
    pub multiplicity: Vec<f64>,
    /// Whether the multiplicity is a share of MT=5 with the parent's other
    /// share rows (a residual of a list leaving one per reaction) rather
    /// than a multiplicity times MT=5; `None` for the row with no target.
    pub share: Option<bool>,
}

/// One parent's `(n,X)` rows.
#[derive(Debug, Clone, PartialEq)]
pub struct Rows {
    pub q_value: f64,
    pub rows: Vec<Row>,
}

/// Everything [`name_all`] produces: the rows by parent, and the record.
#[derive(Debug, Clone, Default)]
pub struct Named {
    pub rows: BTreeMap<String, Rows>,
    /// The `mt5` entry of the reactions subsection's provenance.
    pub record: serde_json::Map<String, serde_json::Value>,
}

/// Read an evaluation's MT=5, or `None` when its MF=3 has no MT=5 or one that
/// is zero at every energy, which makes nothing to model.
pub fn read(material: &Material) -> Option<Evaluation> {
    let meta = material.mf1_mt451()?;
    let (z, a) = (meta.za / 1000, meta.za % 1000);
    let parent = endf::gnds_name(z as u32, a as u32, meta.liso as u32);
    let mf3 = material.mf3(5)?;
    if !mf3.sigma.y.iter().any(|&s| s > 0.0) {
        return None;
    }
    let products: Option<Vec<Product>> = material.mf6(5).map(|mf6| {
        mf6.products
            .iter()
            // ZAP 0 is the photon, which is not inventory and conserves
            // nothing a nucleus count can see.
            .filter(|p| p.zap > 0)
            .map(|p| {
                let (energy, multiplicity) = linearize(&p.yield_, DEFAULT_LINEARIZE_TOL);
                Product {
                    zap: p.zap,
                    lip: p.lip,
                    energy,
                    multiplicity,
                }
            })
            .collect()
    });
    let mut radionuclide_states = Vec::new();
    for (mf, section) in [(9, material.mf9(5)), (10, material.mf10(5))] {
        let Some(section) = section else { continue };
        for level in &section.levels {
            let (peak_e, peak) = level.func.x.iter().zip(&level.func.y).fold(
                (f64::NAN, 0.0f64),
                |best, (&e, &v)| {
                    if v > best.1 {
                        (e, v)
                    } else {
                        best
                    }
                },
            );
            let in_mf6 = products.as_ref().is_some_and(|ps| {
                ps.iter()
                    .any(|p| p.zap == level.izap && (level.lfs == 0 || p.lip > 0))
            });
            radionuclide_states.push(serde_json::json!({
                "mf": mf,
                "izap": level.izap,
                "lfs": level.lfs,
                "largest_value": peak,
                "at_eV": if peak > 0.0 { Some(peak_e) } else { None },
                "mf6_gives_state": in_mf6,
            }));
        }
    }
    Some(Evaluation {
        parent,
        z,
        a,
        q_value: mf3.qi,
        sigma: mf3.sigma.clone(),
        products,
        radionuclide_states,
    })
}

/// A linearized curve at `e`, as the solver reads one: zero below its first
/// point, linear between points, flat above its last. At a repeated point,
/// the value after it.
fn curve_at(energy: &[f64], values: &[f64], e: f64) -> f64 {
    let (Some(&first), Some(&last)) = (energy.first(), energy.last()) else {
        return 0.0;
    };
    if e < first {
        return 0.0;
    }
    if e >= last {
        return values[values.len() - 1];
    }
    let i = energy.partition_point(|&x| x <= e);
    let (x0, x1, y0, y1) = (energy[i - 1], energy[i], values[i - 1], values[i]);
    if x1 == x0 {
        y1
    } else {
        y0 + (e - x0) / (x1 - x0) * (y1 - y0)
    }
}

/// MT=5's cross section at `e`: the tape's own law inside its range, zero
/// outside it.
fn sigma_at(sigma: &Tabulated1D, e: f64) -> f64 {
    match (sigma.x.first(), sigma.x.last()) {
        (Some(&lo), Some(&hi)) if (lo..=hi).contains(&e) => sigma.eval(e),
        _ => 0.0,
    }
}

/// The energies a check walks: every point of MT=5 and of every product, where
/// MT=5 is open.
fn open_grid(evaluation: &Evaluation, products: &[Product]) -> Vec<f64> {
    let mut grid: Vec<f64> = evaluation
        .sigma
        .x
        .iter()
        .copied()
        .chain(products.iter().flat_map(|p| p.energy.iter().copied()))
        .filter(|&e| sigma_at(&evaluation.sigma, e) > 0.0)
        .collect();
    grid.sort_by(f64::total_cmp);
    grid.dedup();
    grid
}

/// The MT=5-weighted average over its whole range of `f(E)`, trapezoid on
/// [`open_grid`].
fn mt5_average(evaluation: &Evaluation, products: &[Product], f: impl Fn(f64) -> f64) -> f64 {
    let grid = open_grid(evaluation, products);
    let (mut num, mut den) = (0.0, 0.0);
    for w in grid.windows(2) {
        let (e0, e1) = (w[0], w[1]);
        let (s0, s1) = (
            sigma_at(&evaluation.sigma, e0),
            sigma_at(&evaluation.sigma, e1),
        );
        num += 0.5 * (f(e0) * s0 + f(e1) * s1) * (e1 - e0);
        den += 0.5 * (s0 + s1) * (e1 - e0);
    }
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

/// The share of the target's charge `emitted` carry, weighted by MT=5 over
/// its whole range; `products` are all of the evaluation's, for the grid.
fn charge_share(evaluation: &Evaluation, products: &[Product], emitted: &[&Product]) -> f64 {
    mt5_average(evaluation, products, |e| {
        emitted
            .iter()
            .map(|p| curve_at(&p.energy, &p.multiplicity, e) * (p.zap / 1000) as f64)
            .sum()
    }) / evaluation.z as f64
}

/// The worst imbalance of charge and mass number over the energies where
/// MT=5 is open, both relative, with where each is worst, and the residuals'
/// summed multiplicity there.
fn conservation(evaluation: &Evaluation, products: &[Product]) -> serde_json::Value {
    let (z_t, a_t) = (evaluation.z as f64, (evaluation.a + 1) as f64);
    let mut worst_z = (0.0f64, f64::NAN, f64::NAN);
    let mut worst_a = (0.0f64, f64::NAN);
    let mut worst_below_20 = 0.0f64;
    for e in open_grid(evaluation, products) {
        let (mut z, mut a, mut residuals) = (0.0, 0.0, 0.0);
        for p in products {
            let m = curve_at(&p.energy, &p.multiplicity, e);
            z += m * (p.zap / 1000) as f64;
            a += m * (p.zap % 1000) as f64;
            if !is_light(p) && p.zap != 1 {
                residuals += m;
            }
        }
        let dz = (z_t - z) / z_t;
        let da = (a_t - a) / a_t;
        if dz.abs() > worst_z.0.abs() || worst_z.1.is_nan() {
            worst_z = (dz, e, residuals);
        }
        if da.abs() > worst_a.0.abs() || worst_a.1.is_nan() {
            worst_a = (da, e);
        }
        if e <= 2.0e7 {
            worst_below_20 = worst_below_20.max(dz.abs()).max(da.abs());
        }
    }
    serde_json::json!({
        "charge": worst_z.0,
        "charge_at_eV": worst_z.1,
        "residual_multiplicity_there": worst_z.2,
        "mass_number": worst_a.0,
        "mass_number_at_eV": worst_a.1,
        "worst_at_or_below_20_MeV": worst_below_20,
    })
}

fn is_light(p: &Product) -> bool {
    p.lip == 0 && LIGHT.iter().any(|(za, _)| *za == p.zap)
}

/// What a product is called in the chain: the light particle's name, the
/// residual's own name where the decay library has it, or the stand-in every
/// other reaction's product gets. `None` where there is neither.
fn chain_name(p: &Product, decay: &DecayIndex) -> Option<String> {
    if let Some((_, name)) = LIGHT.iter().find(|(za, _)| *za == p.zap && p.lip == 0) {
        return Some(name.to_string());
    }
    let (z, a) = (p.zap / 1000, p.zap % 1000);
    if z <= 0 || a <= 0 || p.lip < 0 {
        return None;
    }
    let name = endf::gnds_name(z as u32, a as u32, p.lip as u32);
    if decay.names.contains(&name) {
        return Some(name);
    }
    endf::chain::replace_missing_in(
        &name,
        |n| decay.names.contains(n),
        decay
            .library
            .iter()
            .map(|(n, stable, t)| (n.as_str(), *stable, *t)),
    )
}

/// The decay library's nuclides, as `replace_missing` reads them.
pub struct DecayIndex {
    names: BTreeSet<String>,
    /// `(name, stable, half-life [s])`, ascending by name.
    library: Vec<(String, bool, f64)>,
}

impl DecayIndex {
    /// Index the decay evaluations a chain is built from, skipping the
    /// neutron's own as `Chain::from_endf` does.
    pub fn new(decay: &[Material]) -> DecayIndex {
        let mut by_name: BTreeMap<String, (bool, f64)> = BTreeMap::new();
        for material in decay {
            let Ok(data) = Decay::from_material(material) else {
                continue;
            };
            if data.nuclide.atomic_number == 0 {
                continue;
            }
            by_name.insert(
                data.nuclide.name.clone(),
                (data.nuclide.stable, data.half_life.map_or(0.0, |(t, _)| t)),
            );
        }
        DecayIndex {
            names: by_name.keys().cloned().collect(),
            library: by_name
                .into_iter()
                .map(|(name, (stable, t))| (name, stable, t))
                .collect(),
        }
    }
}

/// The isomeric state a chain name carries, 0 for a ground state.
fn state_of(name: &str) -> i32 {
    endf::zam(name).map_or(0, |(_, _, m)| m as i32)
}

/// Name every evaluation's products against the decay library and build the
/// rows, with the record of everything found on the way.
///
/// `parents` are the nuclides the chain carries (it has decay data for), so
/// an evaluation of something the chain cannot hold writes nothing, as its
/// other reactions do not either.
pub fn name_all(
    evaluations: &BTreeMap<String, Evaluation>,
    decay: &DecayIndex,
    parents: &BTreeSet<String>,
) -> Named {
    let mut out = Named::default();
    let mut conservation_all = serde_json::Map::new();
    let mut not_conserving = Vec::new();
    let mut impossible = Vec::new();
    let mut residual_not_given = Vec::new();
    let mut residuals_not_one = Vec::new();
    let mut no_product_data = Vec::new();
    let mut unnamed = Vec::new();
    let mut radionuclide = Vec::new();
    let mut zero_curves = 0usize;

    for (parent, evaluation) in evaluations {
        if !parents.contains(parent) {
            continue;
        }
        if !evaluation.radionuclide_states.is_empty() {
            radionuclide.push(serde_json::json!({
                "nuclide": parent,
                "states": evaluation.radionuclide_states,
            }));
        }
        let Some(products) = &evaluation.products else {
            no_product_data.push(serde_json::json!(parent));
            out.rows.insert(
                parent.clone(),
                Rows {
                    q_value: evaluation.q_value,
                    rows: vec![Row {
                        target: None,
                        lip: None,
                        energy: Vec::new(),
                        multiplicity: Vec::new(),
                        share: None,
                    }],
                },
            );
            continue;
        };

        let balance = conservation(evaluation, products);
        let worst = |key: &str| balance[key].as_f64().unwrap_or(0.0).abs();
        if worst("charge") > CONSERVATION_TOLERANCE || worst("mass_number") > CONSERVATION_TOLERANCE
        {
            let mut entry = balance.clone();
            entry["nuclide"] = serde_json::json!(parent);
            not_conserving.push(entry);
        }
        conservation_all.insert(parent.clone(), balance);

        // The most of one product the target and the neutron have nucleons
        // for, at any energy where MT=5 is open.
        let grid = open_grid(evaluation, products);
        for p in products.iter().filter(|p| p.zap != 1) {
            let bound = ((evaluation.a + 1) / (p.zap % 1000).max(1)) as f64;
            let worst = grid
                .iter()
                .map(|&e| (e, curve_at(&p.energy, &p.multiplicity, e)))
                .fold(
                    (f64::NAN, 0.0f64),
                    |best, (e, m)| {
                        if m > best.1 {
                            (e, m)
                        } else {
                            best
                        }
                    },
                );
            if worst.1 > bound {
                impossible.push(serde_json::json!({
                    "nuclide": parent,
                    "zap": p.zap,
                    "lip": p.lip,
                    "multiplicity": worst.1,
                    "at_eV": worst.0,
                    "bound": bound,
                }));
            }
        }

        // One residual per reaction, or a target that breaks up: the
        // MT=5-weighted residual multiplicity says which (see
        // `ONE_RESIDUAL_PER_REACTION`).
        let heavy: Vec<&Product> = products
            .iter()
            .filter(|p| p.zap != 1 && !is_light(p))
            .collect();
        let residual_sum = mt5_average(evaluation, products, |e| {
            heavy
                .iter()
                .map(|p| curve_at(&p.energy, &p.multiplicity, e))
                .sum()
        });
        let one_per_reaction = residual_sum >= ONE_RESIDUAL_PER_REACTION;
        if !one_per_reaction && !heavy.is_empty() {
            residuals_not_one.push(serde_json::json!({
                "nuclide": parent,
                "residual_multiplicity": residual_sum,
            }));
        }

        let mut rows: Vec<BranchingRow> = Vec::new();
        for p in products.iter().filter(|p| p.zap != 1) {
            if !p.multiplicity.iter().any(|&m| m != 0.0) {
                zero_curves += 1;
                continue;
            }
            let Some(target) = chain_name(p, decay) else {
                unnamed.push(serde_json::json!({
                    "nuclide": parent,
                    "zap": p.zap,
                    "lip": p.lip,
                }));
                continue;
            };
            rows.push(BranchingRow {
                nuclide: parent.clone(),
                reaction: ANYTHING.to_string(),
                target,
                quantity: "yield".to_string(),
                energy: p.energy.clone(),
                values: p.multiplicity.clone(),
                states: Vec::new(),
                normalisation: None,
            });
        }
        // Two products the chain books to one nuclide (a stand-in, or an
        // isomer the decay data does not have) make that nuclide as their sum.
        let (rows, _) = merge_duplicates(rows);
        let mut written: Vec<Row> = rows
            .into_iter()
            .map(|r| {
                let light = LIGHT.iter().any(|(_, n)| *n == r.target);
                Row {
                    lip: Some(state_of(&r.target)),
                    share: Some(one_per_reaction && !light),
                    target: Some(r.target),
                    energy: r.energy,
                    multiplicity: r.values,
                }
            })
            .collect();
        if !one_per_reaction {
            let emitted: Vec<&Product> = products.iter().filter(|p| p.zap != 1).collect();
            let carried = charge_share(evaluation, products, &emitted);
            if carried < BREAKUP_CHARGE_SHARE {
                residual_not_given.push(serde_json::json!({
                    "nuclide": parent,
                    "charge_share_carried": carried,
                    "residual_multiplicity": residual_sum,
                }));
                written.push(Row {
                    target: None,
                    lip: None,
                    energy: Vec::new(),
                    multiplicity: Vec::new(),
                    share: None,
                });
            }
        }
        if !written.is_empty() {
            out.rows.insert(
                parent.clone(),
                Rows {
                    q_value: evaluation.q_value,
                    rows: written,
                },
            );
        }
    }

    let record = &mut out.record;
    record.insert("parents".to_string(), serde_json::json!(out.rows.len()));
    record.insert(
        "conservation_tolerance".to_string(),
        serde_json::json!(CONSERVATION_TOLERANCE),
    );
    record.insert(
        "conservation".to_string(),
        serde_json::Value::Object(conservation_all),
    );
    record.insert(
        "not_conserving".to_string(),
        serde_json::json!(not_conserving),
    );
    record.insert(
        "impossible_multiplicity".to_string(),
        serde_json::json!(impossible),
    );
    record.insert(
        "residual_not_given".to_string(),
        serde_json::json!(residual_not_given),
    );
    record.insert(
        "residuals_not_one_per_reaction".to_string(),
        serde_json::json!(residuals_not_one),
    );
    record.insert(
        "no_product_data".to_string(),
        serde_json::json!(no_product_data),
    );
    record.insert("unnamed_products".to_string(), serde_json::json!(unnamed));
    record.insert(
        "zero_multiplicities_left_out".to_string(),
        serde_json::json!(zero_curves),
    );
    record.insert("mf10_mt5".to_string(), serde_json::json!(radionuclide));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curve_at_is_zero_below_and_flat_above() {
        let (e, v) = ([1.0, 2.0, 3.0], [0.0, 2.0, 4.0]);
        assert_eq!(curve_at(&e, &v, 0.5), 0.0);
        assert_eq!(curve_at(&e, &v, 1.5), 1.0);
        assert_eq!(curve_at(&e, &v, 9.0), 4.0);
        // A jump reads its value after the point.
        assert_eq!(
            curve_at(&[1.0, 2.0, 2.0, 3.0], &[0.0, 1.0, 5.0, 5.0], 2.0),
            5.0
        );
    }

    #[test]
    fn a_light_particle_is_named_and_an_isomeric_one_is_not_light() {
        let p = |zap, lip| Product {
            zap,
            lip,
            energy: vec![1.0],
            multiplicity: vec![1.0],
        };
        assert!(is_light(&p(2004, 0)));
        assert!(!is_light(&p(2004, 1)));
        assert!(!is_light(&p(25053, 0)));
    }
}
