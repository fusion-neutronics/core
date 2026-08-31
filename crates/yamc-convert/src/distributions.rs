//! `distributions.arrow`: the secondary angle and energy of every product.
//!
//! One row per `(reaction_mt, product_idx, dist_idx)`, joined against
//! `products.arrow`, whose `n_distribution` says how many rows to expect. The
//! traversal is [`crate::products::product_rows`] so the two files cannot
//! disagree about which product is which.
//!
//! # Every column is nullable and most are null
//!
//! The schema is the union of four distribution types and six energy laws, so
//! a row fills the columns its own type uses and leaves the rest null. That is
//! what the reader expects: it switches on `type` and `energy_dist_type` and
//! reads only the columns that discriminant names.
//!
//! # The ragged layout
//!
//! A distribution holds one sub-table per incident energy, and they have
//! different lengths. Each is written as a flat array of every column end to
//! end, plus offsets giving each sub-table's start. The flat array is a
//! C-order ravel of shape `(k, total_points)`: all the x values, then all the
//! p values, and so on, NOT point-by-point interleaving. Offsets index a
//! single row of that, not the whole array.
//!
//! Getting the ravel order wrong is the failure mode worth naming, because it
//! does not fail to load. It produces a distribution whose densities are read
//! out of the energy column, which samples without complaint.

use std::error::Error;
use std::path::Path;

use endf::angle_energy::{AngleEnergy, CorrelatedAngleEnergy, KalbachMann, NBodyPhaseSpace};
use endf::function::Tabulated1D;
use endf::mf::mf4::{AngleAtEnergy, AngleDistribution};
use endf::mf::mf5::EnergyDistribution;
use endf::univariate::{Interpolation, Tabular, Univariate};
use endf::IncidentNeutron;

use crate::products::product_rows;
use crate::sections::*;
use crate::univariate_flat::{flatten, Flat};

/// One row under construction. Every field starts empty, and writing a row
/// means filling in the ones its type uses.
#[derive(Default)]
struct Row {
    reaction_mt: i32,
    product_idx: i32,
    dist_idx: i32,
    ty: String,

    applicability_data: Vec<f64>,
    applicability_shape: Vec<i32>,
    applicability_breakpoints: Vec<i32>,
    applicability_interpolation: Vec<i32>,

    angle_energies: Vec<f64>,
    angle_mu_data: Vec<f64>,
    angle_mu_offsets: Vec<i32>,
    angle_mu_interpolation: Vec<i32>,

    energy_dist_type: Option<String>,
    energy_dist_energies: Vec<f64>,
    energy_dist_interpolation: Vec<i32>,
    energy_dist_data: Vec<f64>,
    energy_dist_offsets: Vec<i32>,
    energy_dist_out_interp: Vec<i32>,
    energy_dist_n_discrete: Vec<i32>,
    energy_param_x: Vec<f64>,
    energy_param_y: Vec<f64>,
    energy_param2_x: Vec<f64>,
    energy_param2_y: Vec<f64>,
    energy_restriction_u: Option<f64>,
    energy_threshold: Option<f64>,
    energy_mass_ratio: Option<f64>,
    energy_primary_flag: Option<i32>,
    energy_atomic_weight_ratio: Option<f64>,
    energy_discrete_energy: Option<f64>,

    corr_energies: Vec<f64>,
    corr_breakpoints: Vec<i32>,
    corr_interpolation: Vec<i32>,
    corr_eout_data: Vec<f64>,
    corr_eout_offsets: Vec<i32>,
    corr_eout_interp: Vec<i32>,
    corr_eout_n_discrete: Vec<i32>,
    corr_mu_data: Vec<f64>,
    corr_mu_offsets: Vec<i32>,
    corr_mu_interp: Vec<i32>,

    km_energies: Vec<f64>,
    km_breakpoints: Vec<i32>,
    km_interpolation: Vec<i32>,
    km_data: Vec<f64>,
    km_offsets: Vec<i32>,
    km_interp: Vec<i32>,
    km_n_discrete: Vec<i32>,

    nbody_n: Option<i32>,
    nbody_total_mass: Option<f64>,
    nbody_atomic_weight_ratio: Option<f64>,
    nbody_q_value: Option<f64>,
}

/// A set of sub-tables as `(flat_data, offsets, interp, n_discrete)`.
///
/// `flat_data` is the C-order ravel of `(3, total)`: every x, then every p,
/// then every c.
fn ravel3(tables: &[Flat]) -> (Vec<f64>, Vec<i32>, Vec<i32>, Vec<i32>) {
    let total: usize = tables.iter().map(|t| t.len()).sum();
    let mut data = Vec::with_capacity(3 * total);
    let mut offsets = Vec::with_capacity(tables.len());
    let mut interp = Vec::with_capacity(tables.len());
    let mut n_discrete = Vec::with_capacity(tables.len());

    let mut at = 0;
    for t in tables {
        offsets.push(at as i32);
        at += t.len();
        interp.push(t.interp);
        n_discrete.push(t.n_discrete as i32);
    }
    for t in tables {
        data.extend_from_slice(&t.x);
    }
    for t in tables {
        data.extend_from_slice(&t.p);
    }
    for t in tables {
        data.extend_from_slice(&t.c);
    }
    (data, offsets, interp, n_discrete)
}

/// A tabulated function as `(data, shape, breakpoints, interpolation)`, with x
/// and y end to end and shape `[2, n]`.
fn tabulated_columns(t: &Tabulated1D) -> (Vec<f64>, Vec<i32>, Vec<i32>, Vec<i32>) {
    let mut data = t.x.clone();
    data.extend_from_slice(&t.y);
    (
        data,
        vec![2, t.x.len() as i32],
        t.breakpoints.clone(),
        t.interpolation.clone(),
    )
}

/// An angular distribution at one incident energy, as a flat sub-table.
///
/// The format stores a tabulated density in the cosine, so the two forms that
/// are not already that have to become it. Legendre coefficients cannot, which
/// is why they are refused: converting them here would be inventing a
/// tabulation the evaluation did not give, and the ACE tables this reads are
/// already tabulated because NJOY did that conversion properly.
fn angle_at_energy(angle: &AngleAtEnergy, mt: i32) -> Result<Flat, Box<dyn Error>> {
    Ok(match angle {
        AngleAtEnergy::Tabular(t) => flatten(&Univariate::Tabular(t.clone())),
        AngleAtEnergy::Isotropic(u) => flatten(&Univariate::Uniform(u.clone())),
        // ENDF's MF=4 LTT=2 form: a density with no CDF beside it, so one is
        // built from the density rather than taken from the file.
        AngleAtEnergy::Tabulated(t) => {
            let interp = match t.interpolation.first() {
                Some(1) => Interpolation::Histogram,
                _ => Interpolation::LinearLinear,
            };
            flatten(&Univariate::Tabular(Tabular::new(
                t.x.clone(),
                t.y.clone(),
                interp,
            )))
        }
        AngleAtEnergy::Legendre(_) => {
            return Err(format!(
                "MT {mt} gives its angular distribution as Legendre coefficients, \
                 which this format has no column for. An ACE table from NJOY is \
                 tabulated, so this means the distribution came from somewhere \
                 else; converting it here would be inventing a tabulation the \
                 evaluation did not give"
            )
            .into())
        }
    })
}

/// Fill the angle columns from an angular distribution.
fn write_angle(row: &mut Row, angle: &AngleDistribution, mt: i32) -> Result<(), Box<dyn Error>> {
    if angle.energy.is_empty() {
        // Isotropic at every energy, which the reader takes from an empty
        // `angle_energies` rather than from a uniform table per energy.
        return Ok(());
    }
    let tables = angle
        .mu
        .iter()
        .map(|m| angle_at_energy(m, mt))
        .collect::<Result<Vec<_>, _>>()?;
    let (data, offsets, interp, _) = ravel3(&tables);
    row.angle_energies = angle.energy.clone();
    row.angle_mu_data = data;
    row.angle_mu_offsets = offsets;
    row.angle_mu_interpolation = interp;
    Ok(())
}

/// Fill the energy columns from a secondary energy distribution.
fn write_energy(row: &mut Row, energy: &EnergyDistribution, mt: i32) -> Result<(), Box<dyn Error>> {
    match energy {
        EnergyDistribution::ContinuousTabular {
            breakpoints,
            interpolation,
            energy,
            energy_out,
        } => {
            let tables: Vec<Flat> = energy_out.iter().map(flatten).collect();
            let (data, offsets, out_interp, n_discrete) = ravel3(&tables);
            row.energy_dist_type = Some("continuous".into());
            row.energy_dist_energies = energy.clone();
            row.energy_dist_data = data;
            row.energy_dist_offsets = offsets;
            row.energy_dist_out_interp = out_interp;
            row.energy_dist_n_discrete = n_discrete;
            // The breakpoints and then the interpolation, sharing one
            // column: there is no separate breakpoint column for a continuous
            // energy distribution, and the published files concatenate them
            // (Fe56 MT 5 holds `[59, 22]` for a 59-point incident grid).
            //
            // The reader only asks whether every value is 1, to decide
            // `histogram_interp`, so writing the interpolation alone loads the
            // same. It also loses the region boundaries, which nothing reads
            // back today and which cannot be recovered from the file once
            // dropped.
            row.energy_dist_interpolation = breakpoints.clone();
            row.energy_dist_interpolation
                .extend_from_slice(interpolation);
        }
        EnergyDistribution::MaxwellEnergy { u, theta } => {
            row.energy_dist_type = Some("maxwell".into());
            row.energy_param_x = theta.x.clone();
            row.energy_param_y = theta.y.clone();
            row.energy_restriction_u = Some(*u);
        }
        EnergyDistribution::Evaporation { u, theta } => {
            row.energy_dist_type = Some("evaporation".into());
            row.energy_param_x = theta.x.clone();
            row.energy_param_y = theta.y.clone();
            row.energy_restriction_u = Some(*u);
        }
        EnergyDistribution::WattEnergy { u, a, b } => {
            row.energy_dist_type = Some("watt".into());
            row.energy_param_x = a.x.clone();
            row.energy_param_y = a.y.clone();
            row.energy_param2_x = b.x.clone();
            row.energy_param2_y = b.y.clone();
            row.energy_restriction_u = Some(*u);
        }
        EnergyDistribution::LevelInelastic {
            threshold,
            mass_ratio,
        } => {
            row.energy_dist_type = Some("level".into());
            row.energy_threshold = Some(*threshold);
            row.energy_mass_ratio = Some(*mass_ratio);
        }
        EnergyDistribution::DiscretePhoton {
            primary_flag,
            energy,
            atomic_weight_ratio,
        } => {
            row.energy_dist_type = Some("discrete_photon".into());
            row.energy_primary_flag = Some(*primary_flag as i32);
            row.energy_discrete_energy = Some(*energy);
            row.energy_atomic_weight_ratio = Some(*atomic_weight_ratio);
        }
        // The remaining ENDF laws have no place in this format. They are also
        // not what an ACE table holds: NJOY turns them into one of the above.
        // Refused rather than skipped, because a product written with no
        // energy distribution samples its secondary energy from nothing.
        other => {
            return Err(format!(
                "MT {mt} has a secondary energy distribution this format cannot \
                 hold ({}). An ACE table from NJOY carries only the processed \
                 laws, so this came from an unprocessed evaluation",
                match other {
                    EnergyDistribution::ArbitraryTabulated { .. } => "LF=1, arbitrary tabulated",
                    EnergyDistribution::GeneralEvaporation { .. } => "LF=5, general evaporation",
                    EnergyDistribution::MadlandNix { .. } => "LF=12, Madland-Nix",
                    _ => "an unrecognised law",
                }
            )
            .into())
        }
    }
    Ok(())
}

/// Fill the correlated columns.
///
/// Two ragged levels: one outgoing energy table per incident energy, and one
/// cosine table per outgoing energy. The outgoing table's ravel is `(5, n)`,
/// where the last two rows are not data but indices into the cosine arrays,
/// stored as floats because the column is `f64`.
fn write_correlated(row: &mut Row, c: &CorrelatedAngleEnergy) {
    let eout: Vec<Flat> = c.energy_out.iter().map(flatten).collect();
    let total: usize = eout.iter().map(|t| t.len()).sum();

    // Every cosine table, flattened, in the order the outgoing energies index
    // them. The index of each is what goes into the outgoing table's rows.
    let mut mu_tables: Vec<Flat> = Vec::new();
    let mut mu_offsets_per_point: Vec<f64> = Vec::with_capacity(total);
    let mut mu_interp_per_point: Vec<f64> = Vec::with_capacity(total);
    let mut mu_interp_values: Vec<i32> = Vec::new();
    let mut at = 0usize;
    for (i, table) in eout.iter().enumerate() {
        for j in 0..table.len() {
            let flat =
                c.mu.get(i)
                    .and_then(|m| m.get(j))
                    .map(flatten)
                    .unwrap_or(Flat {
                        x: vec![-1.0, 1.0],
                        p: vec![0.5, 0.5],
                        c: vec![0.0, 1.0],
                        interp: 2,
                        n_discrete: 0,
                    });
            mu_offsets_per_point.push(at as f64);
            at += flat.len();
            // The interpolation VALUE, not an index into `corr_mu_interp`.
            // The reader spells this `mu_interp[mu_interp_indices[j]]`, which
            // reads as an index, and the two only agree because
            // `corr_mu_interp` is uniform in every correlated distribution in
            // the published library (62 of 62 checked). Writing the index
            // instead would load identically today and differ from every
            // published file, so the corpus convention wins and the ambiguity
            // is worth an issue rather than a unilateral change here.
            mu_interp_per_point.push(flat.interp as f64);
            mu_interp_values.push(flat.interp);
            mu_tables.push(flat);
        }
    }

    let (mu_data, mu_offsets, _, _) = ravel3(&mu_tables);

    // The outgoing table's own ravel, with the two index rows appended.
    let mut eout_data = Vec::with_capacity(5 * total);
    let mut eout_offsets = Vec::with_capacity(eout.len());
    let mut eout_interp = Vec::with_capacity(eout.len());
    let mut eout_n_discrete = Vec::with_capacity(eout.len());
    let mut at = 0usize;
    for t in &eout {
        eout_offsets.push(at as i32);
        at += t.len();
        eout_interp.push(t.interp);
        eout_n_discrete.push(t.n_discrete as i32);
    }
    for t in &eout {
        eout_data.extend_from_slice(&t.x);
    }
    for t in &eout {
        eout_data.extend_from_slice(&t.p);
    }
    for t in &eout {
        eout_data.extend_from_slice(&t.c);
    }
    eout_data.extend_from_slice(&mu_interp_per_point);
    eout_data.extend_from_slice(&mu_offsets_per_point);

    row.corr_energies = c.energy.clone();
    row.corr_breakpoints = c.breakpoints.clone();
    row.corr_interpolation = c.interpolation.clone();
    row.corr_eout_data = eout_data;
    row.corr_eout_offsets = eout_offsets;
    row.corr_eout_interp = eout_interp;
    row.corr_eout_n_discrete = eout_n_discrete;
    row.corr_mu_data = mu_data;
    row.corr_mu_offsets = mu_offsets;
    row.corr_mu_interp = mu_interp_values;
}

/// Fill the Kalbach-Mann columns.
///
/// The ravel is `(5, n)`: outgoing energy, density, cumulative, then the
/// precompound fraction `r` and the slope `a`, which are tabulated against the
/// same outgoing energies rather than separately.
fn write_kalbach_mann(row: &mut Row, k: &KalbachMann) {
    let tables: Vec<Flat> = k.energy_out.iter().map(flatten).collect();
    let total: usize = tables.iter().map(|t| t.len()).sum();

    let mut data = Vec::with_capacity(5 * total);
    let mut offsets = Vec::with_capacity(tables.len());
    let mut interp = Vec::with_capacity(tables.len());
    let mut n_discrete = Vec::with_capacity(tables.len());
    let mut at = 0usize;
    for t in &tables {
        offsets.push(at as i32);
        at += t.len();
        interp.push(t.interp);
        n_discrete.push(t.n_discrete as i32);
    }
    for t in &tables {
        data.extend_from_slice(&t.x);
    }
    for t in &tables {
        data.extend_from_slice(&t.p);
    }
    for t in &tables {
        data.extend_from_slice(&t.c);
    }
    // `r` and `a` are one value per outgoing energy point. Padded rather than
    // truncated when the evaluation gives fewer, since a short row would shift
    // every later sub-table's values by the shortfall.
    for source in [&k.precompound, &k.slope] {
        for (i, t) in tables.iter().enumerate() {
            let values = source.get(i).map(|f| f.y.as_slice()).unwrap_or(&[]);
            for j in 0..t.len() {
                data.push(values.get(j).copied().unwrap_or(0.0));
            }
        }
    }

    row.km_energies = k.energy.clone();
    row.km_breakpoints = k.breakpoints.clone();
    row.km_interpolation = k.interpolation.clone();
    row.km_data = data;
    row.km_offsets = offsets;
    row.km_interp = interp;
    row.km_n_discrete = n_discrete;
}

fn write_nbody(row: &mut Row, n: &NBodyPhaseSpace) {
    row.nbody_n = Some(n.n_particles as i32);
    row.nbody_total_mass = Some(n.total_mass);
    row.nbody_atomic_weight_ratio = Some(n.atomic_weight_ratio);
    row.nbody_q_value = Some(n.q_value);
}

/// Build one row.
fn build_row(
    mt: i32,
    product_idx: usize,
    dist_idx: usize,
    dist: &AngleEnergy,
    applicability: Option<&Tabulated1D>,
) -> Result<Row, Box<dyn Error>> {
    let mut row = Row {
        reaction_mt: mt,
        product_idx: product_idx as i32,
        dist_idx: dist_idx as i32,
        ..Default::default()
    };

    if let Some(app) = applicability {
        let (data, shape, breakpoints, interpolation) = tabulated_columns(app);
        row.applicability_data = data;
        row.applicability_shape = shape;
        row.applicability_breakpoints = breakpoints;
        row.applicability_interpolation = interpolation;
    }

    match dist {
        AngleEnergy::Uncorrelated(u) => {
            row.ty = "uncorrelated".into();
            if let Some(angle) = &u.angle {
                write_angle(&mut row, angle, mt)?;
            }
            if let Some(energy) = &u.energy {
                write_energy(&mut row, energy, mt)?;
            }
        }
        AngleEnergy::Correlated(c) => {
            row.ty = "correlated".into();
            write_correlated(&mut row, c);
        }
        AngleEnergy::KalbachMann(k) => {
            row.ty = "kalbach-mann".into();
            write_kalbach_mann(&mut row, k);
        }
        AngleEnergy::NBodyPhaseSpace(n) => {
            row.ty = "nbody".into();
            write_nbody(&mut row, n);
        }
    }

    Ok(row)
}

/// Write `distributions.arrow`.
pub fn write_distributions(data: &IncidentNeutron, dir: &Path) -> Result<(), Box<dyn Error>> {
    let mut rows = Vec::new();
    for (mt, product_idx, product) in product_rows(data) {
        for (dist_idx, dist) in product.distribution.iter().enumerate() {
            // The applicability is per distribution and absent when there is
            // only one, which is what the reader takes an empty column to mean.
            let applicability = product.applicability.get(dist_idx);
            rows.push(build_row(mt, product_idx, dist_idx, dist, applicability)?);
        }
    }

    let columns = vec![
        ints(&rows.iter().map(|r| r.reaction_mt).collect::<Vec<_>>()),
        ints(&rows.iter().map(|r| r.product_idx).collect::<Vec<_>>()),
        ints(&rows.iter().map(|r| r.dist_idx).collect::<Vec<_>>()),
        strings(&rows.iter().map(|r| r.ty.clone()).collect::<Vec<_>>()),
        float_lists_or_null(&col(&rows, |r| r.applicability_data.clone())),
        int_lists_or_null(&col(&rows, |r| r.applicability_shape.clone())),
        int_lists_or_null(&col(&rows, |r| r.applicability_breakpoints.clone())),
        int_lists_or_null(&col(&rows, |r| r.applicability_interpolation.clone())),
        float_lists_or_null(&col(&rows, |r| r.angle_energies.clone())),
        float_lists_or_null(&col(&rows, |r| r.angle_mu_data.clone())),
        int_lists_or_null(&col(&rows, |r| r.angle_mu_offsets.clone())),
        int_lists_or_null(&col(&rows, |r| r.angle_mu_interpolation.clone())),
        opt_strings(&col(&rows, |r| r.energy_dist_type.clone())),
        float_lists_or_null(&col(&rows, |r| r.energy_dist_energies.clone())),
        int_lists_or_null(&col(&rows, |r| r.energy_dist_interpolation.clone())),
        float_lists_or_null(&col(&rows, |r| r.energy_dist_data.clone())),
        int_lists_or_null(&col(&rows, |r| r.energy_dist_offsets.clone())),
        int_lists_or_null(&col(&rows, |r| r.energy_dist_out_interp.clone())),
        int_lists_or_null(&col(&rows, |r| r.energy_dist_n_discrete.clone())),
        float_lists_or_null(&col(&rows, |r| r.energy_param_x.clone())),
        float_lists_or_null(&col(&rows, |r| r.energy_param_y.clone())),
        float_lists_or_null(&col(&rows, |r| r.energy_param2_x.clone())),
        float_lists_or_null(&col(&rows, |r| r.energy_param2_y.clone())),
        opt_floats(&col(&rows, |r| r.energy_restriction_u)),
        opt_floats(&col(&rows, |r| r.energy_threshold)),
        opt_floats(&col(&rows, |r| r.energy_mass_ratio)),
        opt_ints(&col(&rows, |r| r.energy_primary_flag)),
        opt_floats(&col(&rows, |r| r.energy_atomic_weight_ratio)),
        opt_floats(&col(&rows, |r| r.energy_discrete_energy)),
        float_lists_or_null(&col(&rows, |r| r.corr_energies.clone())),
        int_lists_or_null(&col(&rows, |r| r.corr_breakpoints.clone())),
        int_lists_or_null(&col(&rows, |r| r.corr_interpolation.clone())),
        float_lists_or_null(&col(&rows, |r| r.corr_eout_data.clone())),
        int_lists_or_null(&col(&rows, |r| r.corr_eout_offsets.clone())),
        int_lists_or_null(&col(&rows, |r| r.corr_eout_interp.clone())),
        int_lists_or_null(&col(&rows, |r| r.corr_eout_n_discrete.clone())),
        float_lists_or_null(&col(&rows, |r| r.corr_mu_data.clone())),
        int_lists_or_null(&col(&rows, |r| r.corr_mu_offsets.clone())),
        int_lists_or_null(&col(&rows, |r| r.corr_mu_interp.clone())),
        float_lists_or_null(&col(&rows, |r| r.km_energies.clone())),
        int_lists_or_null(&col(&rows, |r| r.km_breakpoints.clone())),
        int_lists_or_null(&col(&rows, |r| r.km_interpolation.clone())),
        float_lists_or_null(&col(&rows, |r| r.km_data.clone())),
        int_lists_or_null(&col(&rows, |r| r.km_offsets.clone())),
        int_lists_or_null(&col(&rows, |r| r.km_interp.clone())),
        int_lists_or_null(&col(&rows, |r| r.km_n_discrete.clone())),
        opt_ints(&col(&rows, |r| r.nbody_n)),
        opt_floats(&col(&rows, |r| r.nbody_total_mass)),
        opt_floats(&col(&rows, |r| r.nbody_atomic_weight_ratio)),
        opt_floats(&col(&rows, |r| r.nbody_q_value)),
    ];

    write_section(
        &dir.join("distributions.arrow"),
        "distributions.arrow",
        columns,
    )
}

/// One column's worth of values, in row order.
fn col<T>(rows: &[Row], f: impl Fn(&Row) -> T) -> Vec<T> {
    rows.iter().map(f).collect()
}
