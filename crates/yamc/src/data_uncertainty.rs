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

use yani_transmute::covariance_fold::{transport_fields, Read};
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
    /// by an absolute or short-range block.
    pub max_relative_sigma: f64,
}

/// One nuclide's coverage.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NuclideCoverage {
    /// The partial reactions a run would perturb, by MT.
    pub perturbed: BTreeMap<i32, ReactionCoverage>,
    /// The partial reactions no covariance reaches, held at nominal.
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
                    let max_relative_sigma = transport
                        .field
                        .as_ref()
                        .map(|f| {
                            let n = f.relative_cells.len();
                            f.relative_cells
                                .iter()
                                .enumerate()
                                .filter(|(_, c)| c.mt == via)
                                .map(|(k, _)| f.relative[k * n + k].max(0.0).sqrt())
                                .fold(0.0, f64::max)
                        })
                        .unwrap_or(0.0);
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
