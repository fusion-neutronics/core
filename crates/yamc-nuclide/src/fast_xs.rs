//! Building the transport lookup ([`FastXSGrid`]) from a nuclide's reactions.
//!
//! The lookup is derived data: per temperature, the union energy grid, an
//! 8000-bin logarithmic index into it, the four summed cross sections a
//! collision reads before it knows which reaction happened, and the per-MT
//! matrices it walks once it does. Every number in it comes from the
//! reactions the loader has already parsed, and building it is a single pass
//! over those (Fe56: 6 ms natively), which is less than decoding the file that
//! used to carry it (28 ms), so it is built on every load rather than read.
//!
//! # What the four summed columns are
//!
//! `[total, absorption, scattering, fission]`, and the definitions matter
//! because two of them have a plausible wrong answer:
//!
//! - absorption is MT 27 less fission, the disappearance sum, NOT MT 101 as
//!   the evaluation stores it: the stored copy is the ACE file's seven-digit
//!   one and disagrees by 5e-5 barns on Li6.
//! - total is elastic plus MT 3, NOT MT 1, which is the same channels the
//!   other three columns are built from and so agrees with their sum to the
//!   rounding of the largest term. Neither sampler divides by this column
//!   (both rebuild the denominator from the partials they pick between), so a
//!   total a couple of ulp off theirs cannot leave a sliver of probability
//!   pointing at no reaction. It sets the free-flight distance, where that
//!   difference is far below the sampling noise.
//!
//! MT 3 and MT 27 are re-derived here from the partials through
//! [`crate::synthesis`], never taken from the `redundant` rows `reactions.arrow`
//! carries for them. Those rows are the evaluation's own stored copies when it
//! had them, rounded to seven digits by the ACE format, and building the sums
//! on a rounded copy once put FENDL Mo95's scattering below its own elastic
//! channel at 4922 energies. The partials carry full precision.
//!
//! # Order
//!
//! Every loop here runs in ascending MT. The converter that used to write this
//! table iterated a `BTreeMap`, and the sums are floating-point, so the order
//! of addition is part of the result: iterating the loader's `HashMap` in its
//! own order would change the low bits from one process to the next.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use crate::blend::build_log_grid_index;
use crate::buffer::F64Buffer;
use crate::fission_photon::FissionPhotonRelease;
use crate::nuclide::{flatten_row_major, is_fission_mt, FastXSGrid};
use crate::particle_type::ParticleType;
use crate::reaction::Reaction;
use crate::synthesis::{non_elastic_scattering, on_grid, synthesize, FISSION_MTS, SYNTHETIC_MTS};

/// Whether a reaction puts a neutron into the transport.
///
/// Read off the products rather than from an MT table: it is the same question
/// the sampler asks later, so the two cannot disagree about a channel.
fn emits_neutron(reaction: &Reaction) -> bool {
    reaction
        .products
        .iter()
        .any(|p| p.is_particle_type(&ParticleType::Neutron))
}

/// Element-wise sum of `columns`, in the order given.
fn sum_of(columns: &[Vec<f64>], n_energy: usize) -> Vec<f64> {
    let mut out = vec![0.0; n_energy];
    for column in columns {
        for (o, v) in out.iter_mut().zip(column) {
            *o += v;
        }
    }
    out
}

impl FastXSGrid {
    /// Build the lookup for one temperature from every reaction at it.
    ///
    /// `energy` is the nuclide's union grid at this temperature; the grid is
    /// shared into the result rather than copied, so the accelerator and the
    /// nuclide hold one allocation between them. `reactions` must be every MT
    /// the file carries at this temperature, redundant rows included: the
    /// synthesised sums exclude them by rule, not by their absence, and a total
    /// built over a subset would be silently short. `photon_release` scales the
    /// photon production of fission channels for the delayed photons; `None`
    /// leaves that scaling off, as an evaluation without fission
    /// energy release data does.
    ///
    /// `label` names the nuclide and temperature in an error.
    ///
    /// Errors on a grid that cannot be indexed logarithmically (empty, or not
    /// rising from a positive first energy) and on a scattering sum that falls
    /// below the elastic channel it contains, which means a partial was rounded,
    /// negative or double counted.
    pub fn build(
        energy: &F64Buffer,
        reactions: &HashMap<i32, Arc<Reaction>>,
        photon_release: Option<&FissionPhotonRelease>,
        label: &str,
    ) -> Result<FastXSGrid, String> {
        let grid = energy.as_slice();
        let n_energy = grid.len();
        if n_energy == 0 {
            return Err(format!(
                "{label}: no energy grid to build the transport lookup from"
            ));
        }
        // A grid that is present but has no positive extent cannot be indexed
        // logarithmically, and nothing downstream would notice: `ln(0.0)` is
        // negative infinity, `inv_log_delta` comes out NaN, and every lookup
        // would take bin 0's bracket for the whole grid.
        let (first, last) = (grid[0], grid[n_energy - 1]);
        if !(first > 0.0 && last > first && last.is_finite()) {
            return Err(format!(
                "{label}: the energy grid runs from {first:e} to {last:e} eV, which \
                 cannot be indexed logarithmically. The grid has to start above zero \
                 and end above where it starts."
            ));
        }

        let mut mts: Vec<i32> = reactions.keys().copied().collect();
        mts.sort_unstable();

        // The partials, on the full grid: every MT except the synthesised
        // sums. Redundant rows STAY (MT 16 beside its own MT 875 to 890): the
        // double-count defence inside the sums is the summation rules, not the
        // flag, and the two are not the same set.
        let mut partials: BTreeMap<i32, Vec<f64>> = BTreeMap::new();
        for &mt in &mts {
            if SYNTHETIC_MTS.contains(&mt) {
                continue;
            }
            let reaction = &reactions[&mt];
            partials.insert(
                mt,
                on_grid(
                    reaction.cross_section.as_slice(),
                    reaction.threshold_idx,
                    n_energy,
                ),
            );
        }

        // The per-MT columns, split by whether the reaction is a fission or
        // some other neutron-emitting channel. Redundant reactions are left
        // out here: MT 4 and MT 16 are sums over the level partials beside
        // them, and a column for both would offer the same scattering twice.
        // A channel that emits no neutron has no column; absorption is taken
        // from MT 27 below.
        let mut scatter_mts = Vec::new();
        let mut scatter_cols: Vec<Vec<f64>> = Vec::new();
        let mut scatter_mt_reactions = Vec::new();
        let mut fission_mts = Vec::new();
        let mut fission_cols: Vec<Vec<f64>> = Vec::new();
        let mut fission_mt_reactions = Vec::new();
        for &mt in &mts {
            let reaction = &reactions[&mt];
            if reaction.redundant || !emits_neutron(reaction) {
                continue;
            }
            let column = match partials.get(&mt) {
                Some(column) => column.clone(),
                None => on_grid(
                    reaction.cross_section.as_slice(),
                    reaction.threshold_idx,
                    n_energy,
                ),
            };
            if FISSION_MTS.contains(&mt) {
                fission_mts.push(mt);
                fission_cols.push(column);
                fission_mt_reactions.push(Arc::clone(reaction));
            } else {
                scatter_mts.push(mt);
                scatter_cols.push(column);
                scatter_mt_reactions.push(Arc::clone(reaction));
            }
        }

        let zeros = || vec![0.0; n_energy];
        let fission = sum_of(&fission_cols, n_energy);

        // The redundant sums, re-derived from the partials and never taken
        // from the evaluation's own stored copy (see the module doc).
        let mut synthesized = synthesize(&partials, n_energy);
        let mut derive = |mt: i32| synthesized.remove(&mt).unwrap_or_else(zeros);
        let elastic = partials.get(&2).cloned().unwrap_or_else(zeros);
        let non_elastic = derive(3);
        let absorption_and_fission = derive(27);

        // Absorption is MT 27 less what fissions: the disappearance sum.
        let absorption: Vec<f64> = (0..n_energy)
            .map(|i| absorption_and_fission[i] - fission[i])
            .collect();

        // Scattering is MT 2 plus the neutron-emitting part of MT 3, summed
        // straight from the partials rather than reached by taking MT 3 and
        // subtracting the absorption and fission back off it: that subtraction
        // runs through an intermediate the size of MT 3, and on a nuclide with
        // 2e12 barns of (n,p) at thermal it threw away everything below
        // 4.9e-4 barns of the elastic beside it. Summing non-negative terms
        // cannot go below any one of them.
        let non_elastic_scatter = non_elastic_scattering(&partials, n_energy);
        let scattering: Vec<f64> = (0..n_energy)
            .map(|i| elastic[i] + non_elastic_scatter[i])
            .collect();

        let ngamma = partials.get(&102).cloned().unwrap_or_else(zeros);

        // MT 2 + MT 3, not MT 1: the ACE total is acer's own sum and differs
        // from the channels it is a sum of.
        let mut xs = Vec::with_capacity(n_energy);
        for i in 0..n_energy {
            xs.push([
                elastic[i] + non_elastic[i],
                absorption[i],
                scattering[i],
                fission[i],
            ]);
        }

        // The scattering column can never be less than any single channel it
        // is a sum of, and elastic is the largest of them at low energy. This
        // has been wrong twice, both times silently, and both times the sum had
        // been built through a rounded copy or a subtraction. The tolerance is
        // set by the data, not by float precision: an ACE cross section carries
        // about seven significant digits, so the elastic column and a sum built
        // from other columns can legitimately disagree at 1e-9 relative.
        if let Some(elastic_col) = scatter_mts.iter().position(|&mt| mt == 2) {
            for i in 0..n_energy {
                let e = scatter_cols[elastic_col][i];
                if scattering[i] < e - e.abs() * 1e-6 - f64::EPSILON {
                    return Err(format!(
                        "{label}: at {:.6e} eV the scattering cross section is {} barns, \
                         below the {} barns of elastic scattering alone, which it is a \
                         sum over. Something it was built from is rounded, negative or \
                         double counted.",
                        grid[i], scattering[i], e
                    ));
                }
            }
        }

        let (log_e_min, inv_log_delta, log_grid_index) = build_log_grid_index(grid);
        let scatter_mt_xs: F64Buffer = flatten_row_major(&scatter_cols, n_energy).into();
        let fission_mt_xs: F64Buffer = flatten_row_major(&fission_cols, n_energy).into();
        // True when the evaluation gives the chance-by-chance partials rather
        // than only the MT 18 total; the sampler then draws which one.
        let has_partial_fission = fission_mts.iter().any(|&mt| mt != 18);
        let elastic_idx = scatter_mts.iter().position(|&mt| mt == 2);
        let inelastic_walk_order =
            FastXSGrid::build_inelastic_walk_order(&scatter_mts, elastic_idx);
        let reaction_absorption = reactions.get(&101).map(Arc::clone);
        let xs_ngamma: F64Buffer = ngamma.into();

        // The photon-producing channels: every reaction with a photon product
        // whose cross section is positive somewhere on the grid, in MT order.
        let mut photon_rxn_mt_numbers: Vec<i32> = Vec::new();
        let mut photon_rxn_xs_per_mt: Vec<Vec<f64>> = Vec::new();
        let mut photon_rxn_reactions: Vec<Arc<Reaction>> = Vec::new();
        for &mt in &mts {
            let reaction = &reactions[&mt];
            let has_photon_products = reaction
                .products
                .iter()
                .any(|p| p.is_particle_type(&ParticleType::Photon));
            if !has_photon_products {
                continue;
            }
            let xs_vec: Vec<f64> = grid
                .iter()
                .map(|&e| reaction.cross_section_at(e).unwrap_or(0.0))
                .collect();
            if xs_vec.iter().any(|&x| x > 0.0) {
                photon_rxn_mt_numbers.push(mt);
                photon_rxn_xs_per_mt.push(xs_vec);
                photon_rxn_reactions.push(Arc::clone(reaction));
            }
        }

        // The absorption-only channels the decay-photon lookup reads: whatever
        // is left once scattering, fission, photon production, capture and the
        // summed MTs are set aside.
        let mut absorption_mt_numbers: Vec<i32> = Vec::new();
        let mut absorption_mt_reactions: Vec<Arc<Reaction>> = Vec::new();
        {
            let mut covered: HashSet<i32> = HashSet::new();
            covered.extend(&scatter_mts);
            covered.extend(&fission_mts);
            covered.extend(&photon_rxn_mt_numbers);
            // `xs_ngamma` is never empty here, so MT 102 is always covered.
            if !xs_ngamma.is_empty() {
                covered.insert(102);
            }
            covered.extend(&[1, 4, 101, 1001]);
            for &mt in &mts {
                if covered.contains(&mt) {
                    continue;
                }
                let reaction = &reactions[&mt];
                let xs_vec: Vec<f64> = grid
                    .iter()
                    .map(|&e| reaction.cross_section_at(e).unwrap_or(0.0))
                    .collect();
                if xs_vec.iter().any(|&x| x > 0.0) {
                    absorption_mt_numbers.push(mt);
                    absorption_mt_reactions.push(Arc::clone(reaction));
                }
            }
        }

        // Delayed-photon scaling f(E) = (prompt + delayed) / prompt, one value
        // per grid point, applied to fission photon production
        // only. Empty when the evaluation has no fission energy release data,
        // in which case every consumer takes its `f = 1.0` branch.
        let delayed_photon_scaling: F64Buffer = match photon_release {
            Some(release) => grid.iter().map(|&e| release.scaling(e)).collect(),
            None => F64Buffer::default(),
        };

        // Photon production: sum over photon-producing reactions of cross
        // section times photon yield, with the delayed scaling on fission.
        let mut photon_prod = vec![0.0f64; n_energy];
        for (j, &mt) in photon_rxn_mt_numbers.iter().enumerate() {
            let xs_vec = &photon_rxn_xs_per_mt[j];
            let reaction = &photon_rxn_reactions[j];
            for (i, &e) in grid.iter().enumerate() {
                let rxn_xs = xs_vec[i];
                if rxn_xs <= 0.0 {
                    continue;
                }
                let f = if is_fission_mt(mt) && !delayed_photon_scaling.is_empty() {
                    delayed_photon_scaling[i]
                } else {
                    1.0
                };
                for product in &reaction.products {
                    if product.is_particle_type(&ParticleType::Photon) {
                        let y = product
                            .product_yield
                            .as_ref()
                            .map(|yld| yld.evaluate(e))
                            .unwrap_or(1.0);
                        photon_prod[i] += f * rxn_xs * y;
                    }
                }
            }
        }

        Ok(FastXSGrid {
            log_grid_index,
            log_e_min,
            inv_log_delta,
            xs,
            energy: energy.clone(),
            scatter_mt_numbers: scatter_mts,
            scatter_mt_xs,
            scatter_mt_reactions,
            elastic_idx,
            inelastic_walk_order,
            reaction_absorption,
            fission_mt_numbers: fission_mts,
            fission_mt_xs,
            fission_mt_reactions,
            has_partial_fission,
            xs_ngamma,
            photon_prod: photon_prod.into(),
            photon_rxn_mt_numbers,
            // Both tables are built on first use from the reactions kept here,
            // by `columns_on_grid`, which evaluates exactly as the columns
            // above were: a neutron-only run never asks.
            photon_rxn_xs: std::sync::OnceLock::new(),
            photon_rxn_reactions,
            absorption_mt_numbers,
            absorption_mt_reactions,
            absorption_mt_xs: std::sync::OnceLock::new(),
            delayed_photon_scaling,
        })
    }
}

/// Each reaction's cross section at every point of `grid`, flattened row-major
/// `[grid.len(), reactions.len()]`: the layout of the lazily built
/// photon-producing and absorption-only tables.
///
/// Evaluated with [`Reaction::cross_section_at`], as [`FastXSGrid::build`]
/// evaluates the same columns to choose the channels, so a table built late
/// holds exactly what one built at load time would have.
pub(crate) fn columns_on_grid(grid: &[f64], reactions: &[Arc<Reaction>]) -> F64Buffer {
    let columns: Vec<Vec<f64>> = reactions
        .iter()
        .map(|reaction| {
            grid.iter()
                .map(|&e| reaction.cross_section_at(e).unwrap_or(0.0))
                .collect()
        })
        .collect();
    flatten_row_major(&columns, grid.len()).into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reaction_product::ReactionProduct;

    /// A reaction on the grid's tail from `threshold_idx`, with the given
    /// products.
    fn reaction(
        mt: i32,
        xs: Vec<f64>,
        threshold_idx: usize,
        grid: &F64Buffer,
        redundant: bool,
        products: Vec<ReactionProduct>,
    ) -> Arc<Reaction> {
        let energy = if threshold_idx > 0 {
            grid.tail(threshold_idx)
        } else {
            grid.clone()
        };
        Arc::new(Reaction {
            cross_section: xs.into(),
            threshold_idx,
            energy,
            mt_number: mt,
            q_value: 0.0,
            products,
            scatter_in_cm: false,
            redundant,
        })
    }

    fn neutron_out() -> Vec<ReactionProduct> {
        vec![ReactionProduct {
            particle: ParticleType::Neutron,
            emission_mode: "prompt".to_string(),
            decay_rate: 0.0,
            applicability: Vec::new(),
            distribution: Vec::new(),
            product_yield: None,
        }]
    }

    fn grid() -> F64Buffer {
        vec![1.0, 10.0, 100.0, 1000.0].into()
    }

    fn photon_out() -> Vec<ReactionProduct> {
        vec![ReactionProduct {
            particle: ParticleType::Photon,
            emission_mode: "prompt".to_string(),
            decay_rate: 0.0,
            applicability: Vec::new(),
            distribution: Vec::new(),
            product_yield: None,
        }]
    }

    /// The photon-producing and absorption-only tables are not built with the
    /// lookup, and hold what the load-time build would have when asked for.
    #[test]
    fn the_photon_and_absorption_tables_are_built_on_first_use() {
        let g = grid();
        let n = g.len();
        let mut reactions: HashMap<i32, Arc<Reaction>> = HashMap::new();
        reactions.insert(2, reaction(2, vec![10.0; n], 0, &g, false, neutron_out()));
        reactions.insert(102, reaction(102, vec![4.0; n], 0, &g, false, photon_out()));
        // (n,alpha) above the second point: no neutron out, no photon.
        reactions.insert(107, reaction(107, vec![3.0, 5.0], 2, &g, false, Vec::new()));

        let built = FastXSGrid::build(&g, &reactions, None, "test").unwrap();
        assert_eq!(built.photon_rxn_mt_numbers, vec![102]);
        assert_eq!(built.absorption_mt_numbers, vec![107]);
        assert!(built.photon_rxn_xs.get().is_none(), "built eagerly");
        assert!(built.absorption_mt_xs.get().is_none(), "built eagerly");
        // Photon production is still summed at load: transport reads it to
        // decide whether a collision emits photons at all.
        assert_eq!(built.photon_prod.as_slice(), &[4.0; 4]);

        assert_eq!(built.photon_rxn_xs(), &[4.0; 4]);
        assert_eq!(built.absorption_mt_xs(), &[0.0, 0.0, 3.0, 5.0]);
        assert_eq!(built.photon_rxn_xs_interp(1, 0.5, 0), 4.0);
    }

    /// Elastic, one level-inelastic channel, one (n,2n) and capture: the four
    /// summed columns are the sums the module doc names, the matrices hold the
    /// neutron-emitting channels in ascending MT, and capture has no column.
    #[test]
    fn the_columns_are_the_sums_of_the_partials() {
        let g = grid();
        let n = g.len();
        let mut reactions: HashMap<i32, Arc<Reaction>> = HashMap::new();
        reactions.insert(2, reaction(2, vec![10.0; n], 0, &g, false, neutron_out()));
        reactions.insert(51, reaction(51, vec![1.0; 3], 1, &g, false, neutron_out()));
        reactions.insert(16, reaction(16, vec![2.0; 2], 2, &g, false, neutron_out()));
        reactions.insert(102, reaction(102, vec![4.0; n], 0, &g, false, Vec::new()));
        // The evaluation's own stored total, rounded: must never be read.
        reactions.insert(1, reaction(1, vec![999.0; n], 0, &g, true, Vec::new()));

        let built = FastXSGrid::build(&g, &reactions, None, "test").unwrap();
        assert_eq!(built.scatter_mt_numbers, vec![2, 16, 51]);
        assert!(built.fission_mt_numbers.is_empty());
        assert_eq!(built.elastic_idx, Some(0));
        assert!(!built.has_partial_fission);
        // Energy point 3 (1000 eV): every channel open.
        assert_eq!(built.xs[3], [10.0 + 1.0 + 2.0 + 4.0, 4.0, 13.0, 0.0]);
        // Energy point 0 (1 eV): only elastic and capture.
        assert_eq!(built.xs[0], [14.0, 4.0, 10.0, 0.0]);
        // Row-major [n_energy, n_mts]: point 3 holds MT 2, MT 16, MT 51.
        assert_eq!(&built.scatter_mt_xs.as_slice()[9..12], &[10.0, 2.0, 1.0]);
        assert_eq!(built.xs_ngamma.as_slice(), &[4.0; 4]);
        assert_eq!(built.log_grid_index.len(), 8001);
        assert!(
            built.energy.shares_with(&g),
            "the grid is shared, not copied"
        );
    }

    #[test]
    fn a_grid_that_cannot_be_indexed_is_refused() {
        let g: F64Buffer = vec![0.0, 1.0].into();
        let err = FastXSGrid::build(&g, &HashMap::new(), None, "Xx1 at 294 K").unwrap_err();
        assert!(err.contains("Xx1 at 294 K"), "{err}");
        assert!(err.contains("logarithmically"), "{err}");
        let err =
            FastXSGrid::build(&F64Buffer::default(), &HashMap::new(), None, "Xx1").unwrap_err();
        assert!(err.contains("no energy grid"), "{err}");
    }

    /// A negative partial pulls the scattering sum below elastic; the build
    /// refuses rather than handing transport a lookup that lies.
    #[test]
    fn scattering_below_elastic_is_refused() {
        let g = grid();
        let n = g.len();
        let mut reactions: HashMap<i32, Arc<Reaction>> = HashMap::new();
        reactions.insert(2, reaction(2, vec![10.0; n], 0, &g, false, neutron_out()));
        reactions.insert(16, reaction(16, vec![-1.0; n], 0, &g, false, neutron_out()));
        let err = FastXSGrid::build(&g, &reactions, None, "bad").unwrap_err();
        assert!(err.contains("below"), "{err}");
    }

    /// The fission column and flag: MT 18 alone is not partial fission; the
    /// chance partials are.
    #[test]
    fn partial_fission_is_read_off_the_fission_mts() {
        let g = grid();
        let n = g.len();
        let mut reactions: HashMap<i32, Arc<Reaction>> = HashMap::new();
        reactions.insert(2, reaction(2, vec![1.0; n], 0, &g, false, neutron_out()));
        reactions.insert(18, reaction(18, vec![5.0; n], 0, &g, false, neutron_out()));
        let total = FastXSGrid::build(&g, &reactions, None, "t").unwrap();
        assert_eq!(total.fission_mt_numbers, vec![18]);
        assert!(!total.has_partial_fission);
        assert_eq!(total.xs[0], [6.0, 0.0, 1.0, 5.0]);

        reactions.remove(&18);
        reactions.insert(19, reaction(19, vec![3.0; n], 0, &g, false, neutron_out()));
        reactions.insert(20, reaction(20, vec![2.0; n], 0, &g, false, neutron_out()));
        let partial = FastXSGrid::build(&g, &reactions, None, "p").unwrap();
        assert_eq!(partial.fission_mt_numbers, vec![19, 20]);
        assert!(partial.has_partial_fission);
        assert_eq!(partial.xs[0], [6.0, 0.0, 1.0, 5.0]);
    }
}
