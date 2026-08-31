//! The four per-element photon sections.
//!
//! `element.arrow` is the cross sections and form factors on one union energy
//! grid. `subshells.arrow` is the per-subshell photoionization and the
//! relaxation cascade. `compton.arrow` and `bremsstrahlung.arrow` carry data no
//! evaluation holds, which the parser attaches from the auxiliary tabulations.
//!
//! Unlike the neutron route there is no NJOY step: a photoatomic evaluation is
//! already pointwise, so this reads MF=23, MF=27 and MF=28 and writes them.
//!
//! # The logarithms are stored, not recomputed
//!
//! Several columns are the natural log of the column beside them. They are
//! written rather than derived on load because the sampler interpolates in log
//! space on every collision, and a log per lookup is the kind of cost that only
//! shows up in a profile. Zero and negative values map to `ln(1e-300)` rather
//! than to negative infinity, which keeps an interpolation across a threshold
//! finite.

use std::collections::BTreeMap;
use std::error::Error;
use std::path::Path;

use endf::function::Tabulated1D;
use endf::incident_photon::{compton_profile_cdfs, compton_subshell_map, SUBSHELLS};
use endf::IncidentPhoton;

use crate::sections::*;

/// The MTs that are whole-atom cross sections rather than per-subshell ones.
const ELEMENT_MTS: [i32; 6] = [501, 502, 504, 515, 517, 522];

/// The per-subshell photoionization range, MT 534 (K) to MT 572.
const SUBSHELL_MTS: std::ops::RangeInclusive<i32> = 534..=572;

/// `ln`, with non-positive values floored rather than sent to negative
/// infinity. An infinity here propagates through the sampler's interpolation
/// and comes out as a NaN energy several steps later, where nothing points
/// back to the cross section that was zero.
fn safe_log(values: &[f64]) -> Vec<f64> {
    values
        .iter()
        .map(|&v| if v > 0.0 { v.ln() } else { 1e-300f64.ln() })
        .collect()
}

/// The union of every reaction's energy grid, sorted and deduplicated.
fn union_grid(data: &IncidentPhoton) -> Vec<f64> {
    let mut grid: Vec<f64> = data
        .reactions
        .values()
        .filter_map(|rx| rx.xs.as_ref())
        .flat_map(|xs| xs.x.iter().copied())
        .collect();
    grid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    grid.dedup();
    grid
}

/// The relaxation cascade for one subshell as a flat `(n, 4)` row-major table.
///
/// Columns are secondary subshell, tertiary subshell, energy and probability.
/// The two subshell columns are indices into [`SUBSHELLS`] rather than names so
/// the whole table is one `f64` array; a radiative transition has no tertiary
/// subshell and is written as index 0, which is the empty name.
fn transitions_table(t: &endf::incident_photon::Transitions) -> (Vec<f64>, Vec<i32>) {
    let index = |s: &str| SUBSHELLS.iter().position(|&x| x == s).unwrap_or(0) as f64;
    let n = t.energy.len();
    let mut data = Vec::with_capacity(4 * n);
    for i in 0..n {
        data.push(index(t.secondary_subshell.get(i).copied().unwrap_or("")));
        data.push(index(t.tertiary_subshell.get(i).copied().unwrap_or("")));
        data.push(t.energy[i]);
        data.push(t.probability.get(i).copied().unwrap_or(0.0));
    }
    (data, vec![n as i32, 4])
}

/// x and y of a tabulated function, or two empty columns when it is absent.
fn xy(f: Option<&Tabulated1D>) -> (Vec<f64>, Vec<f64>) {
    match f {
        Some(t) => (t.x.clone(), t.y.clone()),
        None => (Vec::new(), Vec::new()),
    }
}

/// Write `element.arrow`.
fn write_element(data: &IncidentPhoton, grid: &[f64], dir: &Path) -> Result<(), Box<dyn Error>> {
    let z = data.atomic_number;

    // Each whole-atom cross section on the union grid, by name.
    let mut xs: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for (&mt, rx) in &data.reactions {
        if !ELEMENT_MTS.contains(&mt) {
            continue;
        }
        let (Some(name), Some(f)) = (rx.name(), rx.xs.as_ref()) else {
            continue;
        };
        xs.insert(name, grid.iter().map(|&e| f.eval(e)).collect());
    }
    let take = |k: &str| xs.get(k).cloned().unwrap_or_default();

    let coherent = data.reactions.get(&502);
    let incoherent = data.reactions.get(&504);

    // The coherent form factor, and its cumulative integral against the SQUARE
    // of the momentum transfer with the factor of Z^2 divided out. That is the
    // form the sampler rejects against, and building it here keeps a
    // per-collision integral out of the hot path.
    let (int_ff_x, int_ff_y) = match coherent.and_then(|rx| rx.scattering_factor.as_ref()) {
        Some(ff) => {
            let squared = Tabulated1D {
                x: ff.x.iter().map(|v| v * v).collect(),
                y: ff.y.iter().map(|v| v * v / (z as f64).powi(2)).collect(),
                ..ff.clone()
            };
            (squared.x.clone(), squared.integral())
        }
        None => (Vec::new(), Vec::new()),
    };
    let (ff_x, ff_y) = xy(coherent.and_then(|rx| rx.scattering_factor.as_ref()));
    let (ar_x, ar_y) = xy(coherent.and_then(|rx| rx.anomalous_real.as_ref()));
    let (ai_x, ai_y) = xy(coherent.and_then(|rx| rx.anomalous_imag.as_ref()));
    let (iff_x, iff_y) = xy(incoherent.and_then(|rx| rx.scattering_factor.as_ref()));

    write_section(
        &dir.join("element.arrow"),
        "element.arrow",
        vec![
            strings(&[data.name().to_string()]),
            ints(&[z as i32]),
            float_list(&safe_log(grid)),
            float_list(&take("coherent")),
            float_list(&take("incoherent")),
            float_list(&take("photoelectric")),
            float_list(&take("pair_production_nuclear")),
            float_list(&take("pair_production_electron")),
            float_list(&take("heating")),
            float_list(&int_ff_x),
            float_list(&int_ff_y),
            float_list(&ff_x),
            float_list(&ff_y),
            float_list(&ar_x),
            float_list(&ar_y),
            float_list(&ai_x),
            float_list(&ai_y),
            float_list(&iff_x),
            float_list(&iff_y),
        ],
    )
}

/// One subshell's row, built while writing `subshells.arrow` and needed again
/// by `compton.arrow` for the occupancy grouping.
struct Subshell {
    designator: String,
    num_electrons: f64,
}

/// Write `subshells.arrow`, returning the rows in the order written.
///
/// Written in MT order, which is most-bound-first, because the Compton
/// subshell map groups by occupancy down the same sequence.
fn write_subshells(
    data: &IncidentPhoton,
    grid: &[f64],
    dir: &Path,
) -> Result<Vec<Subshell>, Box<dyn Error>> {
    let mut designator = Vec::new();
    let mut binding_energy = Vec::new();
    let mut num_electrons = Vec::new();
    let mut xs_col = Vec::new();
    let mut ln_xs_col = Vec::new();
    let mut threshold_idx = Vec::new();
    let mut transitions_data = Vec::new();
    let mut transitions_shape = Vec::new();
    let mut rows = Vec::new();

    for (&mt, rx) in &data.reactions {
        if !SUBSHELL_MTS.contains(&mt) {
            continue;
        }
        let (Some(name), Some(f)) = (rx.name(), rx.xs.as_ref()) else {
            continue;
        };
        let Some(&threshold) = f.x.first() else {
            continue;
        };

        // The last grid point at or below the threshold, so the stored cross
        // section starts one point before the edge rather than on it.
        let idx = grid.partition_point(|&e| e <= threshold).saturating_sub(1);
        let values: Vec<f64> = grid[idx..].iter().map(|&e| f.eval(e)).collect();

        let (mut be, mut ne) = (0.0, 0.0);
        let (mut tdata, mut tshape) = (Vec::new(), Vec::new());
        if let Some(ar) = &data.atomic_relaxation {
            if let Some(&b) = ar.binding_energy.get(name) {
                be = b;
            }
            if let Some(&n) = ar.num_electrons.get(name) {
                ne = n;
            }
            if let Some(t) = ar.transitions.get(name) {
                let (d, s) = transitions_table(t);
                tdata = d;
                tshape = s;
            }
        }

        rows.push(Subshell {
            designator: name.to_string(),
            num_electrons: ne,
        });
        designator.push(name.to_string());
        binding_energy.push(be);
        num_electrons.push(ne);
        ln_xs_col.push(safe_log(&values));
        xs_col.push(values);
        threshold_idx.push(idx as i32);
        transitions_data.push(tdata);
        transitions_shape.push(tshape);
    }

    if rows.is_empty() {
        return Ok(rows);
    }

    write_section(
        &dir.join("subshells.arrow"),
        "subshells.arrow",
        vec![
            strings(&designator),
            floats(&binding_energy),
            floats(&num_electrons),
            float_lists(&xs_col),
            float_lists(&ln_xs_col),
            ints(&threshold_idx),
            // Null, not an empty list, for a subshell with no cascade. The
            // outermost shells have none, and the published files write those
            // as null; an empty list loads the same and is not the same file.
            float_lists_or_null(&transitions_data),
            int_lists_or_null(&transitions_shape),
        ],
    )?;
    Ok(rows)
}

/// Write `compton.arrow`.
fn write_compton(
    data: &IncidentPhoton,
    subshells: &[Subshell],
    dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let Some(profiles) = &data.compton_profiles else {
        return Ok(());
    };
    let Some(first) = profiles.j.first() else {
        return Ok(());
    };

    // The map groups Compton shells onto relaxation subshells by walking both
    // occupancy lists in order, so the subshell rows have to BE in order.
    // Checked rather than assumed: a reordering upstream would otherwise write
    // a map that pairs the wrong shells, which loads and samples without
    // complaint.
    let positions: Vec<usize> = subshells
        .iter()
        .filter_map(|s| SUBSHELLS.iter().position(|&x| x == s.designator))
        .collect();
    if positions.windows(2).any(|w| w[0] > w[1]) {
        return Err(
            "the subshell rows are not in most-bound-first order, so the \
                    Compton subshell map would pair the wrong shells"
                .into(),
        );
    }

    let j: Vec<Vec<f64>> = profiles.j.iter().map(|t| t.y.clone()).collect();
    let pz = first.x.clone();
    let cdf = compton_profile_cdfs(&j, &pz);
    let occupancies: Vec<f64> = subshells.iter().map(|s| s.num_electrons).collect();
    let (offsets, indices, weights) = compton_subshell_map(&profiles.num_electrons, &occupancies);

    let shape = |rows: &[Vec<f64>]| {
        vec![
            rows.len() as i32,
            rows.first().map(Vec::len).unwrap_or(0) as i32,
        ]
    };
    let flatten = |rows: &[Vec<f64>]| rows.concat();

    write_section(
        &dir.join("compton.arrow"),
        "compton.arrow",
        vec![
            float_list(&profiles.num_electrons),
            float_list(&profiles.binding_energy),
            float_list(&pz),
            float_list(&flatten(&j)),
            int_list(&shape(&j)),
            float_list(&flatten(&cdf)),
            int_list(&shape(&cdf)),
            int_list(&offsets.iter().map(|&v| v as i32).collect::<Vec<_>>()),
            int_list(&indices.iter().map(|&v| v as i32).collect::<Vec<_>>()),
            float_list(&weights),
        ],
    )
}

/// Write `bremsstrahlung.arrow`.
fn write_bremsstrahlung(data: &IncidentPhoton, dir: &Path) -> Result<(), Box<dyn Error>> {
    let Some(brem) = &data.bremsstrahlung else {
        return Ok(());
    };
    let shape = vec![
        brem.dcs.len() as i32,
        brem.dcs.first().map(Vec::len).unwrap_or(0) as i32,
    ];
    write_section(
        &dir.join("bremsstrahlung.arrow"),
        "bremsstrahlung.arrow",
        vec![
            floats(&[brem.i]),
            float_list(&brem.electron_energy),
            float_list(&brem.photon_energy),
            float_list(&brem.num_electrons),
            float_list(&brem.ionization_energy),
            float_list(&brem.dcs.concat()),
            int_list(&shape),
        ],
    )
}

/// Write every photon section for one element.
///
/// `element.arrow` is always written; the other three are absent when the
/// evaluation and the auxiliary data have nothing for them, which is what the
/// reader takes a missing file to mean.
pub fn write_photon(data: &IncidentPhoton, dir: &Path) -> Result<(), Box<dyn Error>> {
    let grid = union_grid(data);
    if grid.is_empty() {
        return Err(format!(
            "{}: no photon reaction carries an energy grid, so there is nothing \
             to write element.arrow on",
            data.name()
        )
        .into());
    }
    write_element(data, &grid, dir)?;
    let subshells = write_subshells(data, &grid, dir)?;
    write_compton(data, &subshells, dir)?;
    write_bremsstrahlung(data, dir)?;
    Ok(())
}
