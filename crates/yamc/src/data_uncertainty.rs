//! What nuclear-data uncertainty a model's transport could carry, before any
//! uncertainty run.
//!
//! Transport perturbs the partial reactions it samples and rebuilds every
//! total from them, so the question per nuclide is which of those partials the
//! evaluation states a covariance for, which take one from the summed reaction
//! they belong to, and which are held at nominal. [`Model::data_uncertainty_coverage`]
//! answers it from the same cell fields the uncertainty runs sample
//! ([`yani_transmute::covariance_fold::transport_fields`]), so the report and
//! the run cannot disagree about what is covered.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use yamc_nuclide::nuclide::Nuclide;
use yani_transmute::covariance_fold::{transport_fields, DerivedTerm, Read, TransportField};
use yani_transmute::covariance_sample::{FieldRepair, Sampler};

use crate::model::Model;

/// One transport reaction the evaluation covers.
#[derive(Debug, Clone, PartialEq)]
pub struct ReactionCoverage {
    /// The reaction whose covariance this one takes: itself, or the summed
    /// reaction it is a component of (MT 4 for a level of MT 51 to 91, and so
    /// on), in which case the components move together.
    pub via: i32,
    /// The largest relative standard deviation the evaluation states for it
    /// on any covariance cell, after any repair. Zero when it is covered only
    /// by an absolute block. For a reaction whose covariance an NC block derives from others
    /// (ENDF/B-VIII.1 Pb208 elastic above 1.5 MeV), the largest relative
    /// standard deviation of its cross section at any point of its energy
    /// grid, its own cells and the named reactions' together, absolute
    /// blocks included: the sigma a run applies to it.
    pub max_relative_sigma: f64,
}

/// One nuclide's coverage.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NuclideCoverage {
    /// The partial reactions a run would perturb, by MT.
    pub perturbed: BTreeMap<i32, ReactionCoverage>,
    /// The partial reactions no covariance a run applies reaches, held at
    /// nominal.
    pub held_at_nominal: BTreeSet<i32>,
    /// Covariance cells in the nuclide's field, relative and absolute.
    pub cells: usize,
    /// Short-range (`lb = 8`) blocks in the field.
    pub short_range_blocks: usize,
    /// What repairing the evaluated covariance did, when it was not positive
    /// semidefinite.
    pub repair: Option<FieldRepair>,
    /// The library the covariance came from, as its data folder records it,
    /// or `None` when it records none.
    pub library: Option<String>,
    /// What the library's own documentation says is wrong with this
    /// covariance (see `yani_transmute::covariance_provenance`).
    pub warnings: Vec<String>,
    /// The fission multiplicities the evaluation carries MF=31 covariance
    /// for (452 total, 455 delayed, 456 prompt), ascending. Reported so the
    /// data is visible; no run samples it yet, which `not_perturbed` says.
    pub nubar_covariance: Vec<i32>,
    /// Whether the evaluation carries MF=35, the covariance of the fission
    /// spectrum. Reported, not sampled, as for `nubar_covariance`.
    pub spectrum_covariance: bool,
}

/// What nuclear-data uncertainty a model's transport could carry.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DataUncertaintyCoverage {
    /// Every nuclide with covariance data, by name.
    pub nuclides: BTreeMap<String, NuclideCoverage>,
    /// Nuclides whose data carries no covariance at all: every reaction of
    /// theirs would be held at nominal.
    pub without_data: BTreeSet<String>,
    /// Inputs a run holds at nominal whatever the data, so a small sigma is
    /// not read as one these were included in.
    pub not_perturbed: Vec<String>,
    /// Every nuclide whose evaluation carries MF=34, the covariance of
    /// angular distributions: per reaction, the sorted (L, L1) pairs of
    /// Legendre orders a covariance block correlates. Reported so the data is
    /// visible; no run samples it yet, which `not_perturbed` says.
    pub angular_covariance: BTreeMap<String, BTreeMap<i32, Vec<(i32, i32)>>>,
}

/// The inputs no transport uncertainty run perturbs, whatever the evaluation
/// carries.
const NOT_PERTURBED: [&str; 8] = [
    "angular and energy distributions of secondaries (MF=34, MF=35, MF=6)",
    "resonance-parameter covariance (MF=32)",
    "fission multiplicity and spectrum (MF=31, MF=35)",
    "unresolved-resonance probability tables beyond the cross-section covariance",
    "photon production and photon interaction data",
    "heating, KERMA and damage-energy responses, which move only through the flux",
    "short-range (lb = 8) covariance, which averages away along a track",
    "the material composition and density",
];

/// The largest relative standard deviation the field gives partial `mt` of
/// `nuclide` at any point of its energy grid, its own cells and the `terms`
/// an NC derivation adds together, or `None` where the nuclide has no such
/// partial or grid at `temperature`.
///
/// At each point the partial's change is linear in the draw: `1` on its own
/// relative cell and `w_t / σ` on each named reaction's, `1 / σ` on its own
/// absolute cell and `w_t / (σ_t σ)` on each named one's, as the runs apply
/// it ([`crate::xs_perturbation::grid_terms`]). Its variance is that
/// gradient sandwiched with the field's covariance.
fn derived_relative_sigma(
    nuclide: &Nuclide,
    temperature: &str,
    transport: &TransportField,
    mt: i32,
    terms: &[DerivedTerm],
) -> Option<f64> {
    use crate::xs_perturbation::{cell_at, cells_by_mt, grid_terms, on_full_grid};
    let field = transport.field.as_ref()?;
    let t = nuclide
        .loaded_temperatures
        .iter()
        .position(|l| l == temperature)
        .or_else(|| (nuclide.loaded_temperatures.len() == 1).then_some(0))?;
    let energies = nuclide
        .energy
        .as_ref()?
        .get(&nuclide.loaded_temperatures[t])?
        .as_slice();
    let reactions = nuclide.reactions.get(t)?;
    let partial = reactions.get(&mt)?;
    let xs = on_full_grid(partial, energies.len());
    let derived = grid_terms(terms, partial, reactions, energies);
    let relative_cells = cells_by_mt(&field.relative_cells);
    let absolute_cells = cells_by_mt(&field.absolute_cells);
    let sandwich = |gradient: &[(usize, f64)], covariance: &[f64], n: usize| -> f64 {
        gradient
            .iter()
            .flat_map(|(a, ga)| {
                gradient
                    .iter()
                    .map(move |(b, gb)| ga * gb * covariance[a * n + b])
            })
            .sum()
    };
    let mut worst: f64 = 0.0;
    let (mut g, mut h) = (Vec::new(), Vec::new());
    for (i, &e) in energies.iter().enumerate().skip(partial.threshold_idx) {
        if xs[i] <= 0.0 {
            continue;
        }
        g.clear();
        h.clear();
        let cell = |cells: &BTreeMap<i32, crate::xs_perturbation::Cells>, of: i32| {
            cells.get(&of).and_then(|c| cell_at(c, e))
        };
        if let Some(k) = cell(&relative_cells, mt) {
            g.push((k, 1.0));
        }
        if let Some(k) = cell(&absolute_cells, mt) {
            h.push((k, 1.0 / xs[i]));
        }
        for term in derived.iter().filter(|d| d.weight[i] != 0.0) {
            if let Some(k) = cell(&relative_cells, term.mt) {
                g.push((k, term.weight[i] / xs[i]));
            }
            if let (Some(k), true) = (cell(&absolute_cells, term.mt), term.xs[i] > 0.0) {
                h.push((k, term.weight[i] / (term.xs[i] * xs[i])));
            }
        }
        let variance = sandwich(&g, &field.relative, field.relative_cells.len())
            + sandwich(&h, &field.absolute, field.absolute_cells.len());
        worst = worst.max(variance.max(0.0).sqrt());
    }
    Some(worst)
}

impl Model {
    /// Which partial reactions of every nuclide in the model's materials the
    /// evaluation states a covariance for, which take one from the summed
    /// reaction they belong to, and which are held at nominal.
    ///
    /// Loads each material's nuclear data, with covariance, into the model, as
    /// a transport run would load it, and factorizes each nuclide's field,
    /// which reports any repair and leaves the factorization cached for a
    /// later run. A nuclide in several materials is reported from the first
    /// one it appears in.
    pub fn data_uncertainty_coverage(&mut self) -> Result<DataUncertaintyCoverage, String> {
        let mut report = DataUncertaintyCoverage {
            not_perturbed: NOT_PERTURBED.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        };
        for material_arc in self.geometry.materials_mut().iter_mut() {
            let mut material = (**material_arc).clone();
            material
                .ensure_nuclides_loaded()
                .map_err(|e| format!("loading nuclear data: {e}"))?;
            material
                .ensure_covariance_loaded()
                .map_err(|e| format!("loading covariance: {e}"))?;
            material
                .ensure_angular_covariance_loaded()
                .map_err(|e| format!("loading angular covariance: {e}"))?;
            material
                .ensure_fission_covariance_loaded()
                .map_err(|e| format!("loading fission covariance: {e}"))?;
            for (name, nuclide) in &material.nuclide_data {
                let Some(blocks) = &nuclide.angular_covariance else {
                    continue;
                };
                let by_mt = report.angular_covariance.entry(name.clone()).or_default();
                for b in blocks.iter() {
                    by_mt.entry(b.mt).or_default().push((b.l, b.l1));
                }
                for pairs in by_mt.values_mut() {
                    pairs.sort_unstable();
                    pairs.dedup();
                }
            }
            let material = Arc::new(material);
            *material_arc = Arc::clone(&material);

            let (fields, without_data) = transport_fields(&material);
            report.without_data.extend(without_data);
            let cell_fields: BTreeMap<String, _> = fields
                .iter()
                .filter(|(name, _)| !report.nuclides.contains_key(*name))
                .filter_map(|(name, t)| t.field.clone().map(|f| (name.clone(), f)))
                .collect();
            let repairs = Sampler::new(&cell_fields, &[]).field_repairs();

            for (name, transport) in fields {
                if report.nuclides.contains_key(&name) {
                    continue;
                }
                let mut coverage = NuclideCoverage::default();
                if let Some(field) = &transport.field {
                    coverage.cells = field.relative_cells.len() + field.absolute_cells.len();
                    coverage.short_range_blocks = field.short.len();
                }
                coverage.repair = repairs.get(&name).copied();
                if let Some(nd) = material.nuclide_data.get(&name) {
                    if let Some(blocks) = &nd.nubar_covariance {
                        let mts: BTreeSet<i32> = blocks.iter().map(|b| b.mt).collect();
                        coverage.nubar_covariance = mts.into_iter().collect();
                    }
                    coverage.spectrum_covariance = nd
                        .spectrum_covariance
                        .as_ref()
                        .is_some_and(|b| !b.is_empty());
                    coverage.library = nd
                        .data_source
                        .as_deref()
                        .filter(|s| yamc_nuclide::storage::url_cache::is_keyword(s))
                        .map(str::to_string)
                        .or_else(|| nd.library.clone())
                        .map(|l| {
                            yani_transmute::covariance_provenance::library_keyword(&l).to_string()
                        });
                }
                if let Some(library) = &coverage.library {
                    coverage.warnings =
                        yani_transmute::covariance_provenance::known_problems(library, &name);
                }
                for (mt, read) in &transport.reads {
                    let via = match read {
                        Read::Own => *mt,
                        Read::Parent(sum) => *sum,
                        Read::Nominal => {
                            coverage.held_at_nominal.insert(*mt);
                            continue;
                        }
                    };
                    let derived = transport.derived.get(mt).and_then(|terms| {
                        derived_relative_sigma(
                            material.nuclide_data.get(&name)?,
                            material.temperature(),
                            &transport,
                            *mt,
                            terms,
                        )
                    });
                    let max_relative_sigma = derived.unwrap_or_else(|| {
                        transport.field.as_ref().map_or(0.0, |f| {
                            let n = f.relative_cells.len();
                            f.relative_cells
                                .iter()
                                .enumerate()
                                .filter(|(_, c)| c.mt == via)
                                .map(|(k, _)| f.relative[k * n + k].max(0.0).sqrt())
                                .fold(0.0, f64::max)
                        })
                    });
                    coverage.perturbed.insert(
                        *mt,
                        ReactionCoverage {
                            via,
                            max_relative_sigma,
                        },
                    );
                }
                report.nuclides.insert(name, coverage);
            }
        }
        // A nuclide with data in one material is not without data overall.
        let covered: BTreeSet<String> = report.nuclides.keys().cloned().collect();
        report.without_data.retain(|n| !covered.contains(n));
        Ok(report)
    }
}

impl Model {
    /// Refuse a nuclear-data uncertainty run in any configuration the
    /// replica weights do not yet carry exactly, naming what to change.
    ///
    /// Each refusal is a mode whose weight bookkeeping is not implemented:
    /// rather than a sigma that silently leaves part of the physics nominal,
    /// the run stops before it starts.
    pub(crate) fn validate_data_uncertainty(
        &self,
        data: &crate::model::TransportDataUncertainty,
        mpi_size: usize,
        transmutation: bool,
    ) -> Result<(), String> {
        let refuse = |what: &str| {
            Err(format!(
                "data_uncertainty does not yet support {what}; the replica weights would \
                 leave part of the physics nominal and report too small a sigma"
            ))
        };
        if data.replicas < 2 {
            return Err(format!(
                "data_uncertainty needs at least 2 replicas to estimate a spread, got {}",
                data.replicas
            ));
        }
        if mpi_size > 1 {
            return refuse("MPI runs");
        }
        if transmutation {
            return refuse("simulate_transmutation");
        }
        if self.tracking_mode != crate::model::TrackingMode::Surface {
            return refuse("tracking_mode other than 'surface' (delta tracking)");
        }
        if self.survival_biasing().is_some() {
            return refuse("survival biasing (implicit capture)");
        }
        if !self.weight_windows().is_empty() {
            return refuse("weight windows");
        }
        if self.has_photons() {
            return refuse("photon transport (secondary, decay or source photons)");
        }
        for tally in &self.tallies {
            let name = tally.name.as_deref().unwrap_or("<unnamed>");
            if tally.estimator == yamc_tallies::Estimator::Collision {
                return refuse(&format!("collision-estimator tallies (tally '{name}')"));
            }
            if !tally.multiply_density {
                return refuse(&format!(
                    "overlay tallies, which read their own nominal data (tally '{name}')"
                ));
            }
            if !tally.nuclides.is_empty() {
                return refuse(&format!("per-nuclide tally bins (tally '{name}')"));
            }
            // A mesh splits one flight across voxels, and each voxel's replica
            // factor is the likelihood ratio integrated over its own part of
            // the flight, which the per-segment factor does not resolve.
            if tally.filters.iter().any(|f| {
                matches!(f, yamc_tallies::filter::Filter::Mesh(_))
                    || f.type_name().to_lowercase().contains("mesh")
            }) {
                return refuse(&format!("mesh tallies (tally '{name}')"));
            }
        }
        Ok(())
    }

    /// Load every material's nuclear data with covariance, before the run
    /// prepares its tables, so the replica weights read the same data the run
    /// transports.
    pub(crate) fn load_covariance_for_replicas(&mut self) -> Result<(), String> {
        for material_arc in self.geometry.materials_mut().iter_mut() {
            let mut material = (**material_arc).clone();
            material
                .ensure_nuclides_loaded()
                .map_err(|e| format!("loading nuclear data: {e}"))?;
            material
                .ensure_covariance_loaded()
                .map_err(|e| format!("loading covariance: {e}"))?;
            *material_arc = Arc::new(material);
        }
        Ok(())
    }
}
