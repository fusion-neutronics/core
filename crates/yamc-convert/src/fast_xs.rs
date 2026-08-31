//! `fast_xs.arrow`: the lookup accelerator the transport hot path reads.
//!
//! Nothing here is new physics. Every number is derivable from
//! `reactions.arrow`, and this file exists so a collision does not have to
//! re-derive it: a logarithmic index into the energy grid, the four summed
//! cross sections a collision needs before it knows which reaction happened,
//! and the per-MT matrices it walks once it does.
//!
//! It is also the bulk of the data. One row per temperature, and for U238 the
//! file is a few hundred megabytes, which is why the reader loads it in a
//! second pass after dropping the other sections.
//!
//! # What the four summed columns are
//!
//! `[total, absorption, scattering, fission]`, and the definitions matter
//! because two of them have a plausible wrong answer:
//!
//! - absorption is MT 101, the disappearance sum, NOT MT 101 + fission. MT 27
//!   is the one that includes fission, and using it would double count every
//!   fission against the total.
//! - total is elastic plus MT 3, NOT MT 1, which is the same channels the
//!   other three columns are built from and so agrees with their sum to the
//!   rounding of the largest term. It is not exactly their sum, and does not
//!   have to be: neither sampler divides by this column. Both rebuild the
//!   denominator from the partials they are about to pick between
//!   (`sigma_t = sigma_e + sigma_a + sigma_i + sigma_f`, in
//!   `yamc::transport` and in `yamc_gpu`'s `shared.rs`), so a total a couple
//!   of ulp off theirs cannot leave a sliver of probability pointing at no
//!   reaction. What this column is for is the free-flight distance,
//!   `-ln(xi) / sigma_t`, where that difference is far below the sampling
//!   noise.

use std::collections::BTreeMap;
use std::error::Error;
use std::path::Path;

use endf::IncidentNeutron;

use crate::sections::*;

/// The MT numbers `fission_mt_numbers` may hold.
///
/// MT 18 is the total, 19 to 21 and 38 the first, second, third and fourth
/// chance partials. An evaluation gives either the total or the partials.
pub const FISSION_MTS: [i32; 5] = [18, 19, 20, 21, 38];

/// The number of bins in the logarithmic index. Fixed, not scaled to the grid:
/// the published data uses 8000 for a 631-point H1 grid and for an
/// 80,222-point U235 one alike.
const LOG_BINS: usize = 8000;

/// Whether a reaction puts a neutron into the transport.
///
/// Read off the products rather than from an MT table, which is what makes it
/// agree with the published data exactly on every nuclide checked: it is the
/// same question the sampler asks later.
fn emits_neutron(rx: &endf::Reaction) -> bool {
    rx.products.iter().any(|p| p.name == "neutron")
}

/// A reaction's cross section on the full energy grid, zero below threshold.
fn on_grid(rx: &endf::Reaction, temperature: &str, n_energy: usize) -> Option<Vec<f64>> {
    let xs = rx.xs.get(temperature)?;
    Some(crate::synthesis::on_grid(
        &xs.y,
        xs.threshold_idx.unwrap_or(0),
        n_energy,
    ))
}

/// The logarithmic index: for each of `LOG_BINS + 1` equally log-spaced
/// points, the index of the last grid energy at or below it.
///
/// The sampler uses this to start its binary search, so an index that is too
/// high skips energies that should have been searched.
fn log_grid_index(energy: &[f64]) -> (f64, f64, Vec<i32>) {
    let log_e_min = energy[0].ln();
    let log_e_max = energy[energy.len() - 1].ln();
    let delta = (log_e_max - log_e_min) / LOG_BINS as f64;
    let inv_log_delta = 1.0 / delta;

    let mut index = Vec::with_capacity(LOG_BINS + 1);
    let mut at = 0usize;
    for bin in 0..=LOG_BINS {
        let e = (log_e_min + bin as f64 * delta).exp();
        while at + 1 < energy.len() && energy[at + 1] <= e {
            at += 1;
        }
        index.push(at as i32);
    }
    // The top bin is the top of the grid by construction, set rather than
    // derived, because `exp(ln(e_max))` need not return `e_max`.
    //
    // This is the one place the output does NOT match the published files for
    // every nuclide, and it is deliberate. Whether the round trip lands on or
    // below `e_max` depends on the value: the published Li6, H1 and U235 hold
    // the last grid index here, and the published Fe56 and Al27 hold one less,
    // from the same generator. The reader uses this entry only as the upper
    // bracket of a search (`log_grid_index[bin + 1] + 1`), so one short can
    // exclude the topmost energy from the search while the last index never
    // can. The safe value is therefore the same one for every nuclide.
    if let Some(last) = index.last_mut() {
        *last = (energy.len() - 1) as i32;
    }
    (log_e_min, inv_log_delta, index)
}

/// Row-major `[n_energy, n_mts]`, one column per MT in `mts` order.
fn matrix(columns: &[Vec<f64>], n_energy: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n_energy * columns.len());
    for i in 0..n_energy {
        for c in columns {
            out.push(c[i]);
        }
    }
    out
}

/// Write `fast_xs.arrow`.
pub fn write_fast_xs(data: &IncidentNeutron, dir: &Path) -> Result<(), Box<dyn Error>> {
    let temperatures = data.temperatures();

    let mut temperature_col = Vec::new();
    let mut log_e_min_col = Vec::new();
    let mut inv_log_delta_col = Vec::new();
    let mut log_grid_index_col = Vec::new();
    let mut xs_col = Vec::new();
    let mut xs_shape_col = Vec::new();
    let mut energy_col = Vec::new();
    let mut scatter_mt_numbers_col = Vec::new();
    let mut scatter_mt_xs_col = Vec::new();
    let mut scatter_mt_shape_col = Vec::new();
    let mut fission_mt_numbers_col = Vec::new();
    let mut fission_mt_xs_col = Vec::new();
    let mut fission_mt_shape_col = Vec::new();
    let mut has_partial_fission_col = Vec::new();
    let mut xs_ngamma_col = Vec::new();
    let mut photon_prod_col = Vec::new();

    for temperature in &temperatures {
        let Some(energy) = data.energy.get(temperature) else {
            continue;
        };
        if energy.is_empty() {
            continue;
        }
        let n_energy = energy.len();

        // The per-MT columns, split by whether the reaction is a fission or
        // some other neutron-emitting channel. Redundant reactions are left
        // out: MT 4 and MT 16 are sums over the level partials beside them,
        // and including both would count the same scattering twice.
        let zeros = || vec![0.0; n_energy];
        let sum_of = |columns: &[Vec<f64>]| {
            let mut out = zeros();
            for c in columns {
                for (o, v) in out.iter_mut().zip(c) {
                    *o += v;
                }
            }
            out
        };

        let mut scatter_mts = Vec::new();
        let mut scatter_cols: Vec<Vec<f64>> = Vec::new();
        let mut fission_mts = Vec::new();
        let mut fission_cols: Vec<Vec<f64>> = Vec::new();
        for (&mt, rx) in &data.reactions {
            if rx.redundant {
                continue;
            }
            let Some(column) = on_grid(rx, temperature, n_energy) else {
                continue;
            };
            if !emits_neutron(rx) {
                // Absorption is taken from MT 27 below rather than summed
                // here, so a channel that emits no neutron has no column.
                continue;
            } else if FISSION_MTS.contains(&mt) {
                fission_mts.push(mt);
                fission_cols.push(column);
            } else {
                scatter_mts.push(mt);
                scatter_cols.push(column);
            }
        }

        let fission = sum_of(&fission_cols);

        // The redundant sums, ALWAYS re-derived from the partials and never
        // taken from the evaluation's own stored copy.
        //
        // That distinction is the whole point. A stored MT 3 reaches the ACE
        // file through NJOY's PENDF in ENDF's 11-character field, so it carries
        // seven significant digits; the partials it is a sum of carry full
        // precision. Building `total` and `scattering` on the rounded copy put
        // FENDL's Mo95 `scattering` BELOW its own elastic channel at 4,922 of
        // 11,830 energies, which cannot happen: elastic is part of scattering.
        // NJOY's own ACE total agrees with the re-derived sum to nine digits
        // and with the stored one to seven.
        //
        // The comment that used to sit here claimed the two differed by ~1e-13
        // and that matching the published files was worth more. Both halves
        // were wrong: the residual is 1e-5, six decades larger, and the file
        // did not match the published one either.
        let partials: BTreeMap<i32, Vec<f64>> = data
            .reactions
            .iter()
            .filter(|(mt, _)| !crate::synthesis::SYNTHETIC_MTS.contains(mt))
            .filter_map(|(&mt, rx)| on_grid(rx, temperature, n_energy).map(|c| (mt, c)))
            .collect();
        let mut synthesized = crate::synthesis::synthesize(&partials, n_energy);
        let mut derive = |mt: i32| -> Vec<f64> {
            synthesized
                .remove(&mt)
                .unwrap_or_else(|| vec![0.0; n_energy])
        };
        let elastic = data
            .reactions
            .get(&2)
            .and_then(|rx| on_grid(rx, temperature, n_energy))
            .unwrap_or_else(zeros);
        let non_elastic = derive(3);
        let absorption_and_fission = derive(27);

        // Absorption is MT 27 less what fissions, which is the disappearance
        // sum. NOT MT 101, which is a redundant reaction carried from the ACE
        // file and disagrees with this by 5e-5 barns on Li6.
        let absorption: Vec<f64> = (0..n_energy)
            .map(|i| absorption_and_fission[i] - fission[i])
            .collect();

        // Scattering is MT 2 plus the neutron-emitting part of MT 3, summed
        // straight from the partials rather than reached by taking MT 3 and
        // subtracting the absorption and fission back off it.
        //
        // The two are the same quantity in exact arithmetic and are not in
        // floating point, because the subtraction runs through an intermediate
        // the size of MT 3. TENDL-2025's Mo86 has (n,p) and (n,alpha) near
        // 2e12 barns at 1e-5 eV against 37.8 barns of elastic, so that
        // intermediate has an ulp of 4.9e-4 barns: adding elastic to it threw
        // away everything below that, and the cancellation left `scattering`
        // 1.8e-4 barns BELOW its own elastic channel at 96 energies. Nb85_m1
        // does the same. Summing non-negative terms cannot go below any one of
        // them, whatever the absorption alongside is doing.
        //
        // Every nuclide with a large thermal absorption was losing precision
        // here, not only the two that crossed the check's tolerance.
        let non_elastic_scatter = crate::synthesis::non_elastic_scattering(&partials, n_energy);
        let scattering: Vec<f64> = (0..n_energy)
            .map(|i| elastic[i] + non_elastic_scatter[i])
            .collect();

        let ngamma = data
            .reactions
            .get(&102)
            .and_then(|rx| on_grid(rx, temperature, n_energy))
            .unwrap_or_else(zeros);

        // MT 2 + MT 3, not MT 1: the ACE total is acer's own sum and differs
        // from the channels it is a sum of. `total == absorption + scattering
        // + fission` then holds to the rounding of the largest term, since the
        // two sides are the same channels added in a different order. It is no
        // longer an identity by construction, which it was while scattering
        // was total minus the other two, and that is the point: an identity
        // that holds because one side was defined as the other cannot detect
        // that the side it was defined from is wrong.
        let mut xs = Vec::with_capacity(4 * n_energy);
        for i in 0..n_energy {
            let total = elastic[i] + non_elastic[i];
            xs.extend_from_slice(&[total, absorption[i], scattering[i], fission[i]]);
        }

        // The scattering column can never be less than any single channel it
        // is a sum of, and elastic is the largest of them at low energy. This
        // is checked rather than assumed because it has been wrong twice, both
        // times silently: building scattering from the ACE's seven-digit MT 3
        // put it 9.4e-5 barns below elastic on FENDL's Mo95 at 4,922 energies,
        // and reaching it by subtracting the absorption back off MT 3 put it
        // 1.8e-4 barns below on TENDL-2025's Mo86. Every schema check and every
        // load passed both times.
        //
        // Rounding can no longer trip this, now that the sum above is over
        // non-negative terms and includes elastic itself. What is left for it
        // to catch is a negative partial, a double count, and the next person
        // who reintroduces a subtraction here.
        if let Some(elastic_col) = scatter_mts.iter().position(|&mt| mt == 2) {
            for i in 0..n_energy {
                let e = scatter_cols[elastic_col][i];
                // The tolerance is set by the DATA, not by float precision.
                // An ACE cross section carries about seven significant digits,
                // so the elastic column and a sum built from other columns can
                // legitimately disagree at 1e-9 relative: TENDL's Ag96 has
                // elastic 14.13199 against a summed 14.131989985704422, and a
                // 1e-9 bound rejected the whole conversion over 1.4e-8 barns.
                // 1e-6 is comfortably above ACE's own resolution and still an
                // order of magnitude below the 1e-5 that caught Mo95, let alone
                // the 35% that caught Be9.
                if scattering[i] < e - e.abs() * 1e-6 - f64::EPSILON {
                    return Err(format!(
                        "at {:.6e} eV the scattering cross section is {} barns, \
                         below the {} barns of elastic scattering alone, which \
                         it is a sum over. Something it was built from is \
                         rounded or double counted.",
                        energy[i], scattering[i], e
                    )
                    .into());
                }
            }
        }

        let (log_e_min, inv_log_delta, index) = log_grid_index(energy);

        let n_scatter = scatter_mts.len();
        let n_fission = fission_mts.len();

        temperature_col.push(temperature.clone());
        log_e_min_col.push(log_e_min);
        inv_log_delta_col.push(inv_log_delta);
        log_grid_index_col.push(index);
        xs_col.push(xs);
        xs_shape_col.push(vec![n_energy as i32, 4]);
        energy_col.push(energy.clone());
        scatter_mt_xs_col.push(matrix(&scatter_cols, n_energy));
        scatter_mt_shape_col.push(vec![n_energy as i32, n_scatter as i32]);
        fission_mt_xs_col.push(matrix(&fission_cols, n_energy));
        fission_mt_shape_col.push(vec![n_energy as i32, n_fission as i32]);
        // True when the evaluation gives the chance-by-chance partials rather
        // than only the MT 18 total.
        has_partial_fission_col.push(fission_mts.iter().any(|&mt| mt != 18));
        xs_ngamma_col.push(ngamma);
        // Never populated in the published data; every consumer takes it as
        // absent rather than as zero photon production.
        photon_prod_col.push(zeros());
        scatter_mt_numbers_col.push(scatter_mts);
        fission_mt_numbers_col.push(fission_mts);
    }

    if temperature_col.is_empty() {
        return Err("no energy grid to build fast_xs.arrow from".into());
    }

    write_section(
        &dir.join("fast_xs.arrow"),
        "fast_xs.arrow",
        vec![
            strings(&temperature_col),
            floats(&log_e_min_col),
            floats(&inv_log_delta_col),
            int_lists(&log_grid_index_col),
            float_lists(&xs_col),
            int_lists(&xs_shape_col),
            float_lists(&energy_col),
            int_lists(&scatter_mt_numbers_col),
            float_lists(&scatter_mt_xs_col),
            int_lists(&scatter_mt_shape_col),
            int_lists(&fission_mt_numbers_col),
            float_lists(&fission_mt_xs_col),
            int_lists(&fission_mt_shape_col),
            bools(&has_partial_fission_col),
            float_lists(&xs_ngamma_col),
            float_lists(&photon_prod_col),
        ],
    )
}
