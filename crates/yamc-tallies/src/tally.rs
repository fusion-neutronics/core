use crate::accumulator::TallyAccumulator;
use crate::filter::Filter;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

// Re-export score types so `use yamc::tallies::tally::*` still works
pub use crate::mt::Mt;
pub use crate::score::{
    DamageEnergyScore, FluxScore, HeatingLocalScore, HeatingScore, PhotonComponent, PhotonXSScore,
    ProductionScore, ReactionRateScore, Score, ScoreKind,
};

mod bin_index;
mod mesh_slice;
mod overlay_xs;

/// Pre-loaded microscopic cross section data for overlay tallying.
///
/// When `multiply_density = false`, the tally is decoupled from the cell material
/// and scores a virtual response across the entire geometry (including void).
/// Whether that response is microscopic or macroscopic depends on `densities`:
/// an empty map scores bare microscopic cross sections (barns) per nuclide, while
/// a populated map (a material response) weights each nuclide's microscopic XS by
/// its atom density, giving a macroscopic response (e.g. a silicon dose map).
#[derive(Debug)]
pub struct OverlayXsData {
    // --- Neutron data ---
    /// Unified energy grid for neutron cross sections
    energy_grid: Vec<f64>,
    /// Microscopic cross sections: nuclide -> MT -> xs(E) in barns
    micro_xs: HashMap<String, HashMap<i32, Vec<f64>>>,
    /// O(1) log-grid lookup table (maps a log(E) bin to a starting index in
    /// `energy_grid`). 32-bit for the same reason as
    /// `FastXSGrid::log_grid_index` (issue #482): the values are grid indices,
    /// and a grid long enough to overflow `u32` would be 34 GB of `f64`. Built
    /// monotone below, so nothing to validate.
    log_grid_index: Vec<u32>,
    /// Log of minimum energy
    log_e_min: f64,
    /// Inverse of log bin width
    inv_log_delta: f64,

    // --- Photon data ---
    /// Photon interaction data per overlay nuclide's element: nuclide -> element
    photon_elements: HashMap<String, Arc<yamc_element::photon::PhotonInteraction>>,

    // --- Response weighting (issue #341) ---
    /// The overlay's own nuclide list, in a stable order. Iterated when
    /// collapsing a material response into one combined bin (`combine`).
    /// Distinct from `Tally::nuclides`, which for a material response is
    /// just `[Total]` and carries no per-nuclide entries.
    nuclide_names: Vec<String>,
    /// Per-nuclide number density (atoms / barn-cm). Empty for a unit-density
    /// nuclide overlay; populated from the response material's composition.
    /// Missing entries default to `1.0` via [`OverlayXsData::density`].
    densities: HashMap<String, f64>,
    /// When true, the overlay scores a single combined macroscopic bin
    /// `Σ_i N_i·σ_i` (material response). When false, one bin per nuclide at
    /// unit density (the historical microscopic overlay).
    combine: bool,
}

/// Number of logarithmic bins for overlay XS lookup (same as MaterialFastXS)
const OVERLAY_N_LOG_BINS: usize = 8192;

impl OverlayXsData {
    /// Build log-grid lookup table from an energy grid.
    fn build_log_grid(energy_grid: &[f64]) -> (Vec<u32>, f64, f64) {
        let n = energy_grid.len();
        if n < 2 {
            return (vec![0u32; OVERLAY_N_LOG_BINS + 1], 0.0, 1.0);
        }
        let e_min = energy_grid[0].max(1e-11);
        let e_max = energy_grid[n - 1];
        let log_e_min = e_min.ln();
        let log_e_max = e_max.ln();
        let log_delta = (log_e_max - log_e_min) / OVERLAY_N_LOG_BINS as f64;
        let inv_log_delta = 1.0 / log_delta;

        let mut log_grid_index = vec![0u32; OVERLAY_N_LOG_BINS + 1];
        let mut i_grid = 0;
        for (i_bin, slot) in log_grid_index.iter_mut().enumerate() {
            let log_e = log_e_min + (i_bin as f64) * log_delta;
            let e = log_e.exp();
            while i_grid < n - 1 && energy_grid[i_grid] < e {
                i_grid += 1;
            }
            *slot = i_grid.saturating_sub(1) as u32;
        }

        (log_grid_index, log_e_min, inv_log_delta)
    }
}

/// A nuclide bin for per-nuclide tally breakdown.
///
/// When a tally has `nuclides` set, each score is broken down by nuclide
/// contribution. `Total` uses the material macroscopic cross section (sum
/// over all nuclides), while `Specific` uses a single nuclide's macroscopic
/// cross section.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum NuclideBin {
    /// Total material macroscopic cross section (default behaviour)
    Total,
    /// Per-nuclide macroscopic cross section for the named nuclide
    Specific(String),
}

impl fmt::Display for NuclideBin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NuclideBin::Total => write!(f, "total"),
            NuclideBin::Specific(name) => write!(f, "{name}"),
        }
    }
}

/// A tally -- a scoring channel that accumulates per-bin estimates of
/// some quantity (flux, reaction rate, heating, etc.) over a Monte
/// Carlo simulation.
///
/// # Quick configuration guide
///
/// - **Score** (`scores`) -- what's being measured (`Flux`, `Heating`, …).
/// - **Filters** (`filters`) -- which events count (cell, material, energy,
///   mesh, …). The `Vec<Filter>` is the cross-product binning.
/// - **Estimator** (`estimator`) -- track-length vs collision. See
///   [`crate::Estimator`].
///
/// # Memory layout
///
/// Per bin: a 24-byte `WelfordBin` (count + mean + m2) per bin in each
/// rayon worker's welford state, plus a sparse per-history scratch
/// hash map shared across the worker's tallies.
///
/// # Serialization
///
/// Goes via [`TallySerde`] -- only the user-supplied configuration
/// (id, name, scores, nuclides, filters, units, multiply_density,
/// estimator). Atomic batch counters and the accumulator are rebuilt
/// with their `::new()` defaults on deserialize; tally *results*
/// (accumulated values) are never serialized -- a tally on disk is a
/// *specification*, not a *result*.
#[derive(Debug, serde::Deserialize)]
#[serde(from = "TallySerde")]
pub struct Tally {
    // Input specification fields
    pub tally_id: Option<u32>,
    pub name: Option<String>,
    pub scores: Vec<Score>,
    pub nuclides: Vec<NuclideBin>,
    pub filters: Vec<Filter>,
    pub units: String,

    /// How particle histories turn into score contributions. Default
    /// `Estimator::TrackLength` matches the historical behavior. Set
    /// to `Estimator::Collision` to score at collision sites instead
    /// (lower variance in optically-thick regions, fewer bins touched
    /// per history). See [`crate::estimator`].
    pub estimator: crate::Estimator,

    // Batch tracking
    pub n_batches: AtomicU32, // Total batches configured (for compatibility)
    pub particles_per_chunk: AtomicU32,

    // --- Overlay (multiply_density=false) ---
    /// When false, the tally is decoupled from the cell material and scores a
    /// virtual response across the entire geometry (including void). This does
    /// *not* by itself mean a microscopic result: with no `overlay_material` it
    /// scores bare microscopic XS per nuclide, but a material response supplies
    /// per-nuclide densities via `overlay_material` and is therefore macroscopic.
    /// Default: true (standard macroscopic tally that scales by the cell's density).
    pub multiply_density: bool,

    /// Overlay *material* response (issue #341): per-nuclide atom densities
    /// (atoms / barn-cm) of a virtual material the tally responds to. When
    /// `Some`, the overlay weights each nuclide's microscopic XS by its
    /// density and collapses the result into a single combined macroscopic
    /// bin, giving the material's response across the whole geometry
    /// (including void). `None` for a normal tally or a unit-density nuclide
    /// overlay. Only meaningful when `multiply_density == false`.
    pub overlay_material: Option<std::collections::BTreeMap<String, f64>>,

    /// Internal scoring state: atomic per-bin values, finalized per-history
    /// Welford statistics, score-index caches, and the overlay XS cache. All
    /// `Tally` scoring methods delegate to this accumulator internally.
    pub(crate) accumulator: TallyAccumulator,
}

/// On-disk shape of [`Tally`] -- only the user-supplied configuration.
#[derive(PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TallySerde {
    pub tally_id: Option<u32>,
    pub name: Option<String>,
    pub scores: Vec<Score>,
    pub nuclides: Vec<NuclideBin>,
    pub filters: Vec<Filter>,
    pub units: String,
    pub multiply_density: bool,
    /// Estimator. `#[serde(default)]` keeps backwards compatibility
    /// with serialized tallies that pre-date the field -- they
    /// reconstruct as track-length, matching their original behavior.
    #[serde(default)]
    pub estimator: crate::Estimator,
    /// Overlay material response (issue #341). `#[serde(default,
    /// skip_serializing_if)]` keeps backwards compatibility: tallies that
    /// pre-date the field reconstruct as `None`, and ordinary tallies don't
    /// emit it. `BTreeMap` keeps the JSON key order (and thus `PartialEq`)
    /// deterministic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overlay_material: Option<std::collections::BTreeMap<String, f64>>,
}

impl Tally {
    /// Project the user-supplied configuration into its [`TallySerde`]
    /// form. The single source of truth for what counts as a tally's
    /// *configuration*: both serialization and `PartialEq` go through
    /// this projection, so they can never drift apart -- and adding a
    /// field to `TallySerde` fails to compile here until the projection
    /// (and therefore both consumers) is updated.
    fn to_serde(&self) -> TallySerde {
        TallySerde {
            tally_id: self.tally_id,
            name: self.name.clone(),
            scores: self.scores.clone(),
            nuclides: self.nuclides.clone(),
            filters: self.filters.clone(),
            units: self.units.clone(),
            multiply_density: self.multiply_density,
            estimator: self.estimator,
            overlay_material: self.overlay_material.clone(),
        }
    }
}

/// Configuration equality: two tallies are equal iff their user-supplied
/// configurations (id, name, scores, nuclides, filters, units,
/// multiply_density, estimator) are equal -- exactly the surface that
/// serializes. Runtime accumulator state and batch counters are
/// execution detail, not identity, and are deliberately excluded (which
/// is why this cannot be `#[derive]`d). Used by `combine_results` to
/// verify that same-name tallies from different runs measure the same
/// quantity.
impl PartialEq for Tally {
    fn eq(&self, other: &Self) -> bool {
        self.to_serde() == other.to_serde()
    }
}

// Manual `Serialize` impl rather than `#[serde(into = "TallySerde")]`
// because `into` requires `Tally: Clone`, which is impossible while the
// struct contains `AtomicU32` counters.
impl serde::Serialize for Tally {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        self.to_serde().serialize(ser)
    }
}

impl From<TallySerde> for Tally {
    fn from(s: TallySerde) -> Self {
        let mut t = Tally::new();
        t.tally_id = s.tally_id;
        t.name = s.name;
        t.scores = s.scores;
        t.nuclides = s.nuclides;
        t.filters = s.filters;
        t.units = s.units;
        t.multiply_density = s.multiply_density;
        t.estimator = s.estimator;
        t.overlay_material = s.overlay_material;
        // Allocate accumulator storage now that the bin layout is known.
        // `initialize_batches_shared` (which `model.simulate_transport`
        // calls) only resets the values vec; it won't allocate it.
        // Pass 1 as the placeholder batch count -- the real count gets
        // written by the later `initialize_batches_shared` call.
        t.initialize_batches(1);
        t
    }
}

impl Default for Tally {
    fn default() -> Self {
        Self::new()
    }
}

impl Tally {
    /// Get the number of energy bins from the energy filter (1 if no energy filter)
    pub fn num_energy_bins(&self) -> usize {
        self.filters
            .iter()
            .find_map(|f| {
                if let Filter::Energy(ef) = f {
                    Some(ef.num_bins())
                } else {
                    None
                }
            })
            .unwrap_or(1)
    }

    /// Get the number of nuclide bins (1 if nuclides is empty, otherwise nuclides.len())
    #[inline]
    pub fn num_nuclide_bins(&self) -> usize {
        if self.nuclides.is_empty() {
            1
        } else {
            self.nuclides.len()
        }
    }

    /// Iterate over nuclide bins. Yields `(index, &NuclideBin)`.
    /// When nuclides is empty, yields a single `(0, NuclideBin::Total)`.
    pub fn nuclides_iter(&self) -> Vec<(usize, NuclideBin)> {
        if self.nuclides.is_empty() {
            vec![(0, NuclideBin::Total)]
        } else {
            self.nuclides
                .iter()
                .enumerate()
                .map(|(i, n)| (i, n.clone()))
                .collect()
        }
    }

    /// Default nuclide bin used when no nuclide filter is set.
    const DEFAULT_NUCLIDE_BIN: NuclideBin = NuclideBin::Total;

    /// Get the total number of bins
    /// (scores * cell_bins * material_bins * nuclide_bins * parent_nuclide_bins * energy_bins * mesh_bins).
    #[inline]
    pub fn num_bins(&self) -> usize {
        self.scores.len()
            * self.num_cell_bins()
            * self.num_material_bins()
            * self.num_nuclide_bins()
            * self.num_parent_nuclide_bins()
            * self.num_energy_bins()
            * self.num_mesh_bins()
    }

    /// Get the number of parent nuclide bins (1 if no ParentNuclideFilter).
    ///
    /// An empty ParentNuclideFilter (a D1S dose tally on a material with no
    /// gamma-emitting activation products) collapses to a single all-zero bin
    /// rather than zero bins, so the tally keeps its energy dimension and
    /// reports a zero spectrum. This matches the GPU dispatch, which already
    /// clamps the parent dimension to `max(1, ...)`.
    pub fn num_parent_nuclide_bins(&self) -> usize {
        self.get_parent_nuclide_filter()
            .map(|f| f.num_bins())
            .unwrap_or(1)
            .max(1)
    }

    /// Get the ParentNuclideFilter if present
    pub fn get_parent_nuclide_filter(&self) -> Option<&crate::ParentNuclideFilter> {
        self.filters.iter().find_map(|f| {
            if let Filter::ParentNuclide(pnf) = f {
                Some(pnf)
            } else {
                None
            }
        })
    }

    /// Get the CellFilter if present.
    pub fn get_cell_filter(&self) -> Option<&crate::CellFilter> {
        self.filters.iter().find_map(|f| {
            if let Filter::Cell(cf) = f {
                Some(cf)
            } else {
                None
            }
        })
    }

    /// Get the MaterialFilter if present.
    pub fn get_material_filter(&self) -> Option<&crate::MaterialFilter> {
        self.filters.iter().find_map(|f| {
            if let Filter::Material(mf) = f {
                Some(mf)
            } else {
                None
            }
        })
    }

    /// Number of cell bins (1 when no CellFilter is set or filter is single-cell).
    #[inline]
    pub fn num_cell_bins(&self) -> usize {
        self.get_cell_filter().map(|cf| cf.num_bins()).unwrap_or(1)
    }

    /// Number of material bins (1 when no MaterialFilter is set or filter is single-material).
    #[inline]
    pub fn num_material_bins(&self) -> usize {
        self.get_material_filter()
            .map(|mf| mf.num_bins())
            .unwrap_or(1)
    }

    /// Get the number of mesh bins (1 if no mesh filter)
    fn num_mesh_bins(&self) -> usize {
        if let Some(f) = self.get_mesh_filter() {
            return f.num_bins();
        }
        #[cfg(feature = "mesh")]
        if let Some(f) = self.get_unstructured_mesh_filter() {
            return f.num_bins();
        }
        1
    }

    /// Get the energy filter if present
    pub fn get_energy_filter(&self) -> Option<&crate::EnergyFilter> {
        self.filters.iter().find_map(|f| {
            if let Filter::Energy(ef) = f {
                Some(ef)
            } else {
                None
            }
        })
    }

    /// Get the energy-function filter if present (`energy_function=` /
    /// `dose_coefficients=`).
    ///
    /// Unlike the other filter accessors this one reaches a filter that binds
    /// no bins: it multiplies the score by a tabulated curve and drops events
    /// whose energy falls off the table. `validate()` allows at most one per
    /// tally, so the first match is the only match.
    pub fn get_energy_function_filter(&self) -> Option<&crate::EnergyFunctionFilter> {
        self.filters.iter().find_map(|f| {
            if let Filter::EnergyFunction(ef) = f {
                Some(ef)
            } else {
                None
            }
        })
    }

    /// Get the mesh filter if present
    pub fn get_mesh_filter(&self) -> Option<&crate::MeshFilter> {
        self.filters.iter().find_map(|f| {
            if let Filter::Mesh(mf) = f {
                Some(mf)
            } else {
                None
            }
        })
    }

    /// Get the unstructured mesh filter if present
    #[cfg(feature = "mesh")]
    pub fn get_unstructured_mesh_filter(&self) -> Option<&crate::UnstructuredMeshFilter> {
        self.filters.iter().find_map(|f| {
            if let Filter::UnstructuredMesh(umf) = f {
                Some(umf)
            } else {
                None
            }
        })
    }

    /// True if this tally has any spatial-mesh filter (regular or unstructured).
    pub fn has_mesh_filter(&self) -> bool {
        self.filters.iter().any(|f| {
            matches!(f, Filter::Mesh(_)) || {
                #[cfg(feature = "mesh")]
                {
                    matches!(f, Filter::UnstructuredMesh(_))
                }
                #[cfg(not(feature = "mesh"))]
                {
                    false
                }
            }
        })
    }

    /// Score a track-length contribution into a per-rayon-worker
    /// per-history scratch. Hot path called once per cell crossing
    /// per tally. Per-bin contributions are routed into the worker's
    /// sparse `scratch_map`; the history-end fold (called separately
    /// by the simulation loop) walks the touched bins and applies
    /// one Welford update per bin to the worker's dense state.
    #[allow(clippy::too_many_arguments)]
    pub fn score_track_length(
        &self,
        scratch: &mut crate::welford::WelfordWorkerState,
        tally_idx: usize,
        track_length: f64,
        weight: f64,
        cell_id: Option<u32>,
        material: Option<&yamc_materials::Material>,
        material_id: Option<u32>,
        energy: f64,
        position: [f64; 3],
        last_position: [f64; 3],
        direction: [f64; 3],
        urr_xs: Option<&yamc_materials::UrrMacroXs>,
        particle_type: yamc_particle::ParticleType,
        parent_nuclide: Option<yamc_nuclide::nuclide_registry::NuclideId>,
    ) {
        self.score_track_length_with(
            &mut |bin_idx, v| {
                scratch.add_contribution(tally_idx, bin_idx, v);
            },
            track_length,
            weight,
            cell_id,
            material,
            material_id,
            energy,
            position,
            last_position,
            direction,
            urr_xs,
            particle_type,
            parent_nuclide,
        );
    }

    /// Closure-parameterised body that `score_track_length` routes
    /// through. The 9 per-step scoring sites in this function and the
    /// slow path call `score_fn(bin_idx, v)` rather than touching
    /// storage directly.
    #[allow(clippy::too_many_arguments)]
    fn score_track_length_with(
        &self,
        score_fn: &mut dyn FnMut(usize, f64),
        track_length: f64,
        weight: f64,
        cell_id: Option<u32>,
        material: Option<&yamc_materials::Material>,
        material_id: Option<u32>,
        energy: f64,
        position: [f64; 3],
        last_position: [f64; 3],
        direction: [f64; 3],
        urr_xs: Option<&yamc_materials::UrrMacroXs>,
        particle_type: yamc_particle::ParticleType,
        parent_nuclide: Option<yamc_nuclide::nuclide_registry::NuclideId>,
    ) {
        // Check if cache is initialized
        if !self.accumulator.cache_initialized.load(Ordering::Acquire) {
            return self.score_track_length_slow_with(
                score_fn,
                track_length,
                weight,
                cell_id,
                material,
                material_id,
                energy,
                position,
                last_position,
                direction,
                urr_xs,
                particle_type,
                parent_nuclide,
            );
        }

        // If there's a mesh filter, use slow path (mesh crossing is complex)
        let has_mesh = self.get_mesh_filter().is_some();
        #[cfg(feature = "mesh")]
        let has_mesh = has_mesh || self.get_unstructured_mesh_filter().is_some();
        // If there's a ParentNuclideFilter, use slow path (multi-bin parent nuclide)
        let has_parent_filter = self.get_parent_nuclide_filter().is_some();
        // If there are per-nuclide bins, use slow path
        let has_nuclides = !self.nuclides.is_empty();
        // Multi-cell / multi-material filters bin rather than gate -- use slow path.
        let has_multi_cell = self.num_cell_bins() > 1;
        let has_multi_material = self.num_material_bins() > 1;
        if has_mesh || has_parent_filter || has_nuclides || has_multi_cell || has_multi_material {
            return self.score_track_length_slow_with(
                score_fn,
                track_length,
                weight,
                cell_id,
                material,
                material_id,
                energy,
                position,
                last_position,
                direction,
                urr_xs,
                particle_type,
                parent_nuclide,
            );
        }

        let has_flux = self
            .accumulator
            .cached_has_flux_score
            .get()
            .copied()
            .unwrap_or(false);
        let flux_indices = self
            .accumulator
            .cached_flux_score_indices
            .get()
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        let has_heating = self
            .accumulator
            .cached_has_heating_score
            .get()
            .copied()
            .unwrap_or(false);
        let heating_indices = self
            .accumulator
            .cached_heating_score_indices
            .get()
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        let has_heating_local = self
            .accumulator
            .cached_has_heating_local_score
            .get()
            .copied()
            .unwrap_or(false);
        let heating_local_indices = self
            .accumulator
            .cached_heating_local_score_indices
            .get()
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        let cell_filter_id = self
            .accumulator
            .cached_cell_filter_id
            .get()
            .copied()
            .unwrap_or(None);
        let energy_bins = self.accumulator.cached_energy_bins.get().unwrap_or(&None);
        let num_energy_bins = self
            .accumulator
            .cached_num_energy_bins
            .get()
            .copied()
            .unwrap_or(1);

        let production_indices = self
            .accumulator
            .cached_production_score_indices
            .get()
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        let has_production = !production_indices.is_empty();

        let has_damage_energy = self
            .accumulator
            .cached_has_damage_energy_score
            .get()
            .copied()
            .unwrap_or(false);
        let damage_energy_indices = self
            .accumulator
            .cached_damage_energy_score_indices
            .get()
            .map(|v| v.as_slice())
            .unwrap_or(&[]);

        let energy_function_filter = self
            .accumulator
            .cached_energy_function_filter
            .get()
            .unwrap_or(&None);
        let mt_indices = self
            .accumulator
            .cached_mt_score_indices
            .get()
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        let has_mt = !mt_indices.is_empty();

        let cached_pt_filter = self
            .accumulator
            .cached_particle_type_filter
            .get()
            .copied()
            .unwrap_or(None);
        let has_photon = self
            .accumulator
            .cached_has_photon_score
            .get()
            .copied()
            .unwrap_or(false);
        let photon_indices = self
            .accumulator
            .cached_photon_score_indices
            .get()
            .map(|v| v.as_slice())
            .unwrap_or(&[]);

        // Early filter: if particle type filter is set and doesn't match, skip
        if let Some(required_pt) = cached_pt_filter {
            if required_pt != particle_type {
                return;
            }
        }

        let is_neutron = particle_type == yamc_particle::ParticleType::Neutron;
        let is_photon = particle_type == yamc_particle::ParticleType::Photon;

        // Fast path: if no applicable scores, skip
        if !has_flux
            && !(is_neutron
                && (has_heating
                    || has_heating_local
                    || has_production
                    || has_damage_energy
                    || has_mt))
            && !(is_photon && (has_photon || has_heating || has_heating_local || has_mt))
        {
            return;
        }

        // Fast cell filter check
        if let Some(required_cell_id) = cell_filter_id {
            match cell_id {
                Some(id) if id == required_cell_id => {}
                _ => return,
            }
        }

        // Check material filter
        for filter in &self.filters {
            if let Filter::Material(mf) = filter {
                if !mf.matches(material_id) {
                    return;
                }
            }
        }

        // Get energy bin using cached bins
        // Energy bin convention: [E0, E1], (E1, E2], ... (En-1, En]
        let energy_bin = if let Some(ref bins) = energy_bins {
            if energy < bins[0] || energy > bins[bins.len() - 1] {
                return;
            }
            match bins.binary_search_by(|&bin| {
                if bin < energy {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                }
            }) {
                Ok(i) => i,
                Err(i) if i > 0 && i < bins.len() => i - 1,
                Err(0) => 0, // Handle E == bins[0] case
                _ => return,
            }
        } else {
            0
        };

        let num_e_bins = num_energy_bins.max(1);

        // Score all flux scores - directly into values array (no RwLock!)
        // Multiply track_length by weight for proper statistical weighting
        let mut weighted_track_length = track_length * weight;

        // Apply EnergyFunctionFilter weight if present (for dose calculations)
        if let Some(ef) = energy_function_filter {
            if let Some(energy_weight) = ef.get_weight(energy) {
                weighted_track_length *= energy_weight;
            } else {
                // Energy outside filter range, skip all scoring
                return;
            }
        }

        for &score_idx in flux_indices {
            // Note: mesh bins = 1 in fast path, so calculation simplifies to score_idx * num_e_bins + energy_bin
            let bin_idx = score_idx * num_e_bins + energy_bin;
            score_fn(bin_idx, weighted_track_length);
        }

        // Heating scores: track-length KERMA for both particle types.
        // Neutrons use MT 301; photons use the macroscopic photon
        // heating estimate (issue #356) -- the analog collision deposit
        // now belongs to collision-estimator tallies only.
        if has_heating {
            if let Some(material) = material {
                let heating_xs = if is_neutron {
                    material.lookup_heating_xs(energy)
                } else if is_photon {
                    material.calculate_photon_xs(energy).heating
                } else {
                    0.0
                };
                // Neutron KERMA (MT 301) is legitimately negative in
                // endothermic charged-particle windows (e.g. O16
                // 7.4-13 MeV); guard on `!= 0.0` so those negatives fold
                // in (issue #84). Photon heating is non-negative.
                if heating_xs != 0.0 {
                    for &score_idx in heating_indices {
                        let bin_idx = score_idx * num_e_bins + energy_bin;
                        score_fn(bin_idx, heating_xs * weighted_track_length);
                    }
                }
            }
        }

        // Heating-local scores: track-length KERMA for both particle
        // types (photons reuse the heating estimate -- with no electron
        // transport, local and total deposition coincide; issue #356).
        if has_heating_local {
            if let Some(material) = material {
                let heating_local_xs = if is_neutron {
                    material.lookup_heating_local_xs(energy)
                } else if is_photon {
                    material.calculate_photon_xs(energy).heating
                } else {
                    0.0
                };
                // See heating note above: negative neutron KERMA (MT 901)
                // must fold in (issue #84).
                if heating_local_xs != 0.0 {
                    for &score_idx in heating_local_indices {
                        let bin_idx = score_idx * num_e_bins + energy_bin;
                        score_fn(bin_idx, heating_local_xs * weighted_track_length);
                    }
                }
            }
        }

        // Neutron-specific scores: only fire for neutrons
        if is_neutron {
            // Score all production scores (H1, H2, H3, He3, He4)
            if has_production {
                if let Some(material) = material {
                    for &(score_idx, mt) in production_indices {
                        let production_xs = material.lookup_production_xs(mt, energy);
                        if production_xs > 0.0 {
                            let bin_idx = score_idx * num_e_bins + energy_bin;
                            score_fn(bin_idx, production_xs * weighted_track_length);
                        }
                    }
                }
            }

            // Score all damage-energy scores (MT 444)
            if has_damage_energy {
                if let Some(material) = material {
                    let damage_energy_xs = material.lookup_damage_energy_xs(energy);
                    if damage_energy_xs > 0.0 {
                        for &score_idx in damage_energy_indices {
                            let bin_idx = score_idx * num_e_bins + energy_bin;
                            score_fn(bin_idx, damage_energy_xs * weighted_track_length);
                        }
                    }
                }
            }

            // Score all MT reaction scores (Score::ReactionRate variants)
            if has_mt {
                if let Some(material) = material {
                    for &(score_idx, mt) in mt_indices {
                        let xs = material.macro_xs_by_mt(mt, energy, urr_xs);
                        if xs > 0.0 {
                            let bin_idx = score_idx * num_e_bins + energy_bin;
                            score_fn(bin_idx, xs * weighted_track_length);
                        }
                    }
                }
            }
        }

        // Photon-specific scores: only fire for photons
        if is_photon && (has_photon || has_mt) {
            if let Some(material) = material {
                let photon_xs = material.calculate_photon_xs(energy);
                for &(score_idx, component) in photon_indices {
                    let xs = match component {
                        0 => photon_xs.coherent,
                        1 => photon_xs.incoherent,
                        2 => photon_xs.photoelectric,
                        3 => photon_xs.pair_production,
                        _ => unreachable!(),
                    };
                    if xs > 0.0 {
                        let bin_idx = score_idx * num_e_bins + energy_bin;
                        score_fn(bin_idx, xs * weighted_track_length);
                    }
                }
                // ReactionRate scores on the photon path. MT=TOTAL is the
                // particle-agnostic total interaction rate -- the photon total
                // macroscopic XS (Σ_total,photon · track length), matching the
                // GPU SCORE_TOTAL fast path and OpenMC's photon `total`. A
                // specific neutron MT is a category error for a photon (it does
                // not undergo a neutron reaction), so it scores nothing -- the
                // GPU rejects it at validation; here it is silently skipped.
                for &(score_idx, mt) in mt_indices {
                    if mt == Mt::TOTAL.as_i32() && photon_xs.total > 0.0 {
                        let bin_idx = score_idx * num_e_bins + energy_bin;
                        score_fn(bin_idx, photon_xs.total * weighted_track_length);
                    }
                }
            }
        }
    }

    /// Resolve a scalar overlay (microscopic) cross section for one
    /// nuclide bin. For `Specific(name)` this is just `lookup(name)`;
    /// for `Total` it sums `lookup` over every `Specific` nuclide bin
    /// in `self.nuclides`. Collapses the otherwise-identical
    /// `filter_map(..).sum()` overlay blocks in
    /// `score_track_length_slow_with` to a single call differing only
    /// in the per-nuclide `lookup` (MT / heating function).
    #[inline]
    fn overlay_xs_for_bin(&self, nuc_bin: &NuclideBin, lookup: impl Fn(&str) -> f64) -> f64 {
        match nuc_bin {
            NuclideBin::Specific(name) => lookup(name),
            NuclideBin::Total => self
                .nuclides
                .iter()
                .filter_map(|n| match n {
                    NuclideBin::Specific(name) => Some(lookup(name)),
                    _ => None,
                })
                .sum::<f64>(),
        }
    }

    /// Resolve a density-weighted overlay cross section for one nuclide bin,
    /// given a closure returning the *microscopic* XS (barns) for a nuclide.
    ///
    /// - **Material response** (`ovl.combine()`): collapse the whole virtual
    ///   material into one macroscopic value `Σ_i N_i·σ_i`, iterating the
    ///   overlay's *own* nuclide list. (We must use `ovl.nuclide_names()`,
    ///   not `self.nuclides`: for a material response `self.nuclides` is just
    ///   `[Total]` and carries no `Specific` entries, so `overlay_xs_for_bin`
    ///   would sum to zero.)
    /// - **Nuclide overlay**: per-bin as before, but each microscopic XS is
    ///   scaled by `ovl.density(name)`, which is `1.0` for the historical
    ///   unit-density overlay - so that path is unchanged.
    #[inline]
    fn overlay_xs_weighted(
        &self,
        ovl: &OverlayXsData,
        nuc_bin: &NuclideBin,
        lookup_micro: impl Fn(&str) -> f64,
    ) -> f64 {
        if ovl.combine() {
            ovl.nuclide_names()
                .iter()
                .map(|n| ovl.density(n) * lookup_micro(n))
                .sum::<f64>()
        } else {
            self.overlay_xs_for_bin(nuc_bin, |name| ovl.density(name) * lookup_micro(name))
        }
    }

    /// Slow path for score_track_length when cache is not initialized.
    /// Closure-parameterised over the per-bin storage write so the
    /// same body serves both the default (atomic-CAS) and Stage 3
    /// (scratch write) entry points. See `score_track_length` for the
    /// public wrapper.
    #[allow(clippy::too_many_arguments)]
    fn score_track_length_slow_with(
        &self,
        score_fn: &mut dyn FnMut(usize, f64),
        track_length: f64,
        weight: f64,
        cell_id: Option<u32>,
        material: Option<&yamc_materials::Material>,
        material_id: Option<u32>,
        energy: f64,
        position: [f64; 3],
        last_position: [f64; 3],
        direction: [f64; 3],
        urr_xs: Option<&yamc_materials::UrrMacroXs>,
        particle_type: yamc_particle::ParticleType,
        parent_nuclide: Option<yamc_nuclide::nuclide_registry::NuclideId>,
    ) {
        // Get the energy bin if there's an energy filter
        let energy_bin = if let Some(energy_filter) = self.get_energy_filter() {
            match energy_filter.get_bin(energy) {
                Some(bin) => bin,
                None => return,
            }
        } else {
            0
        };

        // Mesh-crossings handling: see the streaming dispatch at the end of
        // this function. Previously we collected the crossings into a
        // `SmallVec<[MeshCrossing; 8]>` and iterated it inside the
        // per-(score, nuc) loop -- on the 200³ flux mesh that allocation
        // alone showed ~27% of transport time in samply (tracks routinely
        // exceed 8 inline slots and spill to heap). The current version
        // hoists the (score, nuc) loop above the crossings and streams the
        // iterator once.

        // Check particle type filter early
        for filter in &self.filters {
            if let Filter::ParticleType(ptf) = filter {
                if !ptf.matches(particle_type) {
                    return;
                }
            }
        }

        let is_neutron = particle_type == yamc_particle::ParticleType::Neutron;
        let is_photon = particle_type == yamc_particle::ParticleType::Photon;

        // For each score, check if it's a flux, heating, heating-local, or production score
        // Multiply track_length by weight for proper statistical weighting
        let mut weighted_track_length = track_length * weight;

        // Apply EnergyFunctionFilter weight if present (for dose calculations)
        for filter in &self.filters {
            if let Filter::EnergyFunction(ef) = filter {
                if let Some(energy_weight) = ef.get_weight(energy) {
                    weighted_track_length *= energy_weight;
                } else {
                    // Energy outside filter range, skip all scoring
                    return;
                }
                break;
            }
        }

        // Cell and Material filters bin (and gate) this event: a missing bin
        // means the event is outside the filter, so skip the score.
        let cell_bin: usize = if let Some(cf) = self.get_cell_filter() {
            match cell_id.and_then(|id| cf.get_bin(id)) {
                Some(bin) => bin,
                None => return,
            }
        } else {
            0
        };

        let material_bin: usize = if let Some(mf) = self.get_material_filter() {
            match mf.get_bin(material_id) {
                Some(bin) => bin,
                None => return,
            }
        } else {
            0
        };

        // Determine parent nuclide bin to score into (always 0 or 1)
        let parent_bin: usize = if let Some(pnf) = self.get_parent_nuclide_filter() {
            match parent_nuclide.and_then(|id| pnf.get_bin(id)) {
                Some(bin) => bin,
                None => return, // No match → skip scoring
            }
        } else {
            0 // No filter → single bin 0
        };

        // Iterate over nuclide bins without allocation.
        // When no nuclides filter is set, use a single Total bin from the static const.
        let nuclide_bins: &[NuclideBin] = if self.nuclides.is_empty() {
            std::slice::from_ref(&Self::DEFAULT_NUCLIDE_BIN)
        } else {
            &self.nuclides
        };

        // Helper: compute XS-based contribution for a score, given a nuclide bin.
        // When multiply_density=false (overlay mode), use cached microscopic XS.
        // For NuclideBin::Total, use material macroscopic XS (sum of all nuclides)
        // For NuclideBin::Specific(name), use per-nuclide macroscopic XS
        //
        // First pass: collect `(score_idx, nuc_idx, base_contribution)`
        // tuples for every `(score, nuc)` pair where `base_contribution` is
        // `Some`. The base contribution is per-(score, nuc) and independent
        // of which mesh crossing we're applying it to, so it can be hoisted
        // out of the crossings loop and computed once.
        let mut contribs: SmallVec<[(usize, usize, f64); 8]> = SmallVec::new();
        let overlay = self.overlay_xs();
        for (score_idx, score) in self.scores.iter().enumerate() {
            for (nuc_idx, nuc_bin) in nuclide_bins.iter().enumerate() {
                let base_contribution = match score {
                    Score::Flux(_) => Some(weighted_track_length),
                    Score::Heating(_) => {
                        // Overlay mode: use microscopic XS from cache (both neutron & photon)
                        if let Some(ovl) = overlay {
                            let xs = if is_neutron {
                                self.overlay_xs_weighted(ovl, nuc_bin, |name| {
                                    ovl.lookup_neutron(name, 301, energy)
                                })
                            } else if is_photon {
                                // Track-length KERMA estimator for photon overlay
                                self.overlay_xs_weighted(ovl, nuc_bin, |name| {
                                    ovl.lookup_photon_heating(name, energy)
                                })
                            } else {
                                0.0
                            };
                            // Neutron KERMA (MT 301) can be negative; fold it
                            // in (issue #84). Photon heating stays >= 0.
                            if (is_neutron && xs != 0.0) || (!is_neutron && xs > 0.0) {
                                Some(xs * weighted_track_length)
                            } else {
                                None
                            }
                        } else if let Some(material) = material {
                            let heating_xs = if is_neutron {
                                match nuc_bin {
                                    NuclideBin::Total => material.lookup_heating_xs(energy),
                                    NuclideBin::Specific(name) => {
                                        material.lookup_xs_by_mt_for_nuclide(name, 301, energy)
                                    }
                                }
                            } else if is_photon {
                                // Track-length photon KERMA (issue #356);
                                // per-nuclide photon splits need the
                                // element mapping and stay zero.
                                match nuc_bin {
                                    NuclideBin::Total => {
                                        material.calculate_photon_xs(energy).heating
                                    }
                                    NuclideBin::Specific(_) => 0.0,
                                }
                            } else {
                                0.0
                            };
                            // Negative neutron KERMA (MT 301) must fold in
                            // (issue #84); photon heating stays >= 0.
                            if (is_neutron && heating_xs != 0.0)
                                || (!is_neutron && heating_xs > 0.0)
                            {
                                Some(heating_xs * weighted_track_length)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    Score::HeatingLocal(_) => {
                        if let Some(ovl) = overlay {
                            let xs = if is_neutron {
                                self.overlay_xs_weighted(ovl, nuc_bin, |name| {
                                    ovl.lookup_neutron(name, 901, energy)
                                })
                            } else if is_photon {
                                self.overlay_xs_weighted(ovl, nuc_bin, |name| {
                                    ovl.lookup_photon_heating(name, energy)
                                })
                            } else {
                                0.0
                            };
                            // Negative neutron KERMA (MT 901) must fold in
                            // (issue #84); photon heating stays >= 0.
                            if (is_neutron && xs != 0.0) || (!is_neutron && xs > 0.0) {
                                Some(xs * weighted_track_length)
                            } else {
                                None
                            }
                        } else if let Some(material) = material {
                            let xs = if is_neutron {
                                match nuc_bin {
                                    NuclideBin::Total => material.lookup_heating_local_xs(energy),
                                    NuclideBin::Specific(name) => {
                                        material.lookup_xs_by_mt_for_nuclide(name, 901, energy)
                                    }
                                }
                            } else if is_photon {
                                // Photon KERMA; local == total without
                                // electron transport (issue #356).
                                match nuc_bin {
                                    NuclideBin::Total => {
                                        material.calculate_photon_xs(energy).heating
                                    }
                                    NuclideBin::Specific(_) => 0.0,
                                }
                            } else {
                                0.0
                            };
                            // Negative neutron KERMA (MT 901) must fold in
                            // (issue #84); photon heating stays >= 0.
                            if (is_neutron && xs != 0.0) || (!is_neutron && xs > 0.0) {
                                Some(xs * weighted_track_length)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    Score::Production(p) => {
                        if !is_neutron {
                            None
                        } else if let Some(ovl) = overlay {
                            let mt = p.mt.as_i32();
                            let xs = self.overlay_xs_weighted(ovl, nuc_bin, |name| {
                                ovl.lookup_neutron(name, mt, energy)
                            });
                            if xs > 0.0 {
                                Some(xs * weighted_track_length)
                            } else {
                                None
                            }
                        } else if let Some(material) = material {
                            let mt = p.mt.as_i32();
                            let xs = match nuc_bin {
                                NuclideBin::Total => material.lookup_production_xs(mt, energy),
                                NuclideBin::Specific(name) => {
                                    material.lookup_xs_by_mt_for_nuclide(name, mt, energy)
                                }
                            };
                            if xs > 0.0 {
                                Some(xs * weighted_track_length)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    Score::DamageEnergy(_) => {
                        if !is_neutron {
                            None
                        } else if let Some(ovl) = overlay {
                            let xs = self.overlay_xs_weighted(ovl, nuc_bin, |name| {
                                ovl.lookup_neutron(name, 444, energy)
                            });
                            if xs > 0.0 {
                                Some(xs * weighted_track_length)
                            } else {
                                None
                            }
                        } else if let Some(material) = material {
                            let xs = match nuc_bin {
                                NuclideBin::Total => material.lookup_damage_energy_xs(energy),
                                NuclideBin::Specific(name) => {
                                    material.lookup_xs_by_mt_for_nuclide(name, 444, energy)
                                }
                            };
                            if xs > 0.0 {
                                Some(xs * weighted_track_length)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    Score::ReactionRate(r) => {
                        if is_photon {
                            // ReactionRate(MT=TOTAL) is the particle-agnostic
                            // total interaction rate. On the photon path it is
                            // the photon total macroscopic XS (Σ_total,photon ·
                            // track length), matching the GPU SCORE_TOTAL fast
                            // path and OpenMC's photon `total`. A specific
                            // neutron MT is a category error for a photon, so it
                            // scores nothing (the GPU rejects it at validation).
                            // Per-nuclide photon totals are not split out, so we
                            // only score the material-level Total bin.
                            if r.mt == Mt::TOTAL && matches!(nuc_bin, NuclideBin::Total) {
                                material.and_then(|m| {
                                    let xs = m.calculate_photon_xs(energy).total;
                                    (xs > 0.0).then_some(xs * weighted_track_length)
                                })
                            } else {
                                None
                            }
                        } else if !is_neutron {
                            None
                        } else if let Some(ovl) = overlay {
                            let mt = r.mt.as_i32();
                            let xs = self.overlay_xs_weighted(ovl, nuc_bin, |name| {
                                ovl.lookup_neutron(name, mt, energy)
                            });
                            if xs > 0.0 {
                                Some(xs * weighted_track_length)
                            } else {
                                None
                            }
                        } else if let Some(material) = material {
                            let mt = r.mt.as_i32();
                            let xs = match nuc_bin {
                                NuclideBin::Total => material.macro_xs_by_mt(mt, energy, urr_xs),
                                NuclideBin::Specific(name) => {
                                    material.lookup_xs_by_mt_for_nuclide(name, mt, energy)
                                }
                            };
                            if xs > 0.0 {
                                Some(xs * weighted_track_length)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    Score::PhotonXS(p) => {
                        if !is_photon {
                            None
                        } else if let Some(ovl) = overlay {
                            // Overlay mode: per-nuclide photon component XS.
                            // `overlay_xs_weighted` serves both the unit-density
                            // nuclide overlay and the density-weighted material
                            // response (collapsed to one combined bin). Photon
                            // data is element-level, so summing density_i × the
                            // per-atom XS over isotopes of one element yields
                            // N_element × XS -- correct, no double counting.
                            let component = |name: &str| -> f64 {
                                ovl.lookup_photon(name, energy)
                                    .map(|micro| match p.component {
                                        PhotonComponent::Coherent => micro.coherent,
                                        PhotonComponent::Incoherent => micro.incoherent,
                                        PhotonComponent::Photoelectric => micro.photoelectric,
                                        PhotonComponent::PairProduction => micro.pair_production,
                                    })
                                    .unwrap_or(0.0)
                            };
                            let xs = self.overlay_xs_weighted(ovl, nuc_bin, component);
                            if xs > 0.0 {
                                Some(xs * weighted_track_length)
                            } else {
                                None
                            }
                        } else if !matches!(nuc_bin, NuclideBin::Total) {
                            // Normal mode: photon XS per-nuclide not supported
                            None
                        } else if let Some(material) = material {
                            let photon_xs = material.calculate_photon_xs(energy);
                            let xs = match p.component {
                                PhotonComponent::Coherent => photon_xs.coherent,
                                PhotonComponent::Incoherent => photon_xs.incoherent,
                                PhotonComponent::Photoelectric => photon_xs.photoelectric,
                                PhotonComponent::PairProduction => photon_xs.pair_production,
                            };
                            if xs > 0.0 {
                                Some(xs * weighted_track_length)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                };

                if let Some(base) = base_contribution {
                    contribs.push((score_idx, nuc_idx, base));
                }
            }
        }

        // No score on this event -- skip the (possibly expensive) mesh
        // traversal entirely.
        if contribs.is_empty() {
            return;
        }

        // Second pass: stream the mesh crossings once and, for each
        // crossing, apply all contribs. The crossings iterator is lazy --
        // we never collect into an intermediate buffer (the previous
        // `SmallVec<[MeshCrossing; 8]>` allocation spilled to heap for any
        // track that crossed more than 8 voxels, which is the typical case
        // on 200³+ meshes and accounted for ~27% inclusive time in
        // `score_track_length_slow_with` per the samply profile).
        // Macro to keep the per-crossing body inline at every call site --
        // the previous `let apply_crossing = |...| { ... }` closure form
        // produced break-even or marginally regressive timings on dense
        // meshes (rustc didn't always inline through the dyn FnMut), so
        // we duplicate the small body across the three mesh dispatch
        // branches instead.
        macro_rules! apply_crossing {
            ($bin:expr, $length_fraction:expr) => {{
                let length_fraction = $length_fraction;
                let bin = $bin;
                for &(score_idx, nuc_idx, base) in &contribs {
                    let contrib = base * length_fraction;
                    if let Some(bin_idx) = self.get_bin_index_7d(
                        score_idx,
                        cell_bin,
                        material_bin,
                        nuc_idx,
                        parent_bin,
                        energy_bin,
                        bin,
                    ) {
                        score_fn(bin_idx, contrib);
                    }
                }
            }};
        }

        if let Some(mesh_filter) = self.get_mesh_filter() {
            for crossing in mesh_filter.get_bins_crossed_iter(last_position, position, direction) {
                apply_crossing!(crossing.bin, crossing.length_fraction);
            }
            return;
        }
        #[cfg(feature = "mesh")]
        if let Some(umf) = self.get_unstructured_mesh_filter() {
            // Unstructured mesh still allocates its result Vec internally;
            // the streaming saving applies only to the regular-mesh path.
            for crossing in umf.get_bins_crossed(last_position, position, direction) {
                apply_crossing!(crossing.bin, crossing.length_fraction);
            }
            return;
        }
        // No mesh filter -- one synthetic crossing covering the full track.
        apply_crossing!(0, 1.0);
    }

    /// Score a single collision event.
    ///
    /// Per-event contributions:
    ///   - `Flux`: `weight / Σ_t` (units: cm)
    ///   - `Heating` (neutron): `(heating_xs / Σ_t) · weight` (units: eV)
    ///   - `HeatingLocal` (neutron): `(heating_local_xs / Σ_t) · weight`
    ///   - `Heating` / `HeatingLocal` (photon): `photon_heat_score_ev`
    ///     -- the precomputed analog value
    ///     `(E_in − E_out − Σ E_banked_photons) · weight`, supplied by
    ///     the transport loop because it depends on collision outcomes,
    ///     not cross-section ratios. Electron/positron kinetic energy
    ///     deposits locally and is included (coupled neutron-photon
    ///     local-deposition convention); only banked photons (fluorescence, annihilation,
    ///     TTB bremsstrahlung) are subtracted.
    ///   - `ReactionRate(mt)`: `(σ_r / Σ_t) · weight` (units: reactions);
    ///     URR-aware via `urr_xs` for total / elastic / fission /
    ///     absorption / capture so reaction-rate scores in URR-active
    ///     energy ranges stay consistent with the transport sampling.
    ///   - `Production(mt)` (neutron): `(σ_production_mt / Σ_t) · weight`
    ///     (units: particles), for the H1–He4 production MTs 203–207.
    ///   - `DamageEnergy` (neutron): `(damage_energy_xs / Σ_t) · weight`
    ///     (units: eV), MT 444.
    ///   - `PhotonXS(component)` (photon): `(σ_component / Σ_t) · weight`
    ///     for `Coherent` / `Incoherent` / `Photoelectric` / `PairProduction`.
    ///     The caller supplies the precomputed `MacroPhotonXS` so the
    ///     match arm picks the right component without re-running the
    ///     XS calculation.
    ///
    /// `total_xs` is the macroscopic total cross section the collision
    /// was sampled against (URR-modified neutron Σ_t when neutron+URR
    /// is active, smooth `MacroPhotonXS::total` for photons). For the
    /// post-photon-collision dispatch that only needs to write the
    /// analog photon heat, the caller passes `total_xs = 0` and
    /// `photon_heat_score_ev = Some(h)` -- `Σ_t`-dependent arms (Flux,
    /// neutron heating, ReactionRate, Production, DamageEnergy,
    /// PhotonXS) are then skipped and the analog arm fires regardless
    /// of `tally.estimator`. The collision estimator converges to the
    /// same physical quantity as the track-length estimator, so
    /// identical seeds must agree to within statistical noise.
    /// Score a collision-estimator contribution into the per-rayon-
    /// worker per-history scratch. Hot path called once per collision
    /// per tally (when the tally has opted into
    /// `Estimator::Collision`). Same scratch-write architecture as
    /// `score_track_length`.
    #[allow(clippy::too_many_arguments)]
    pub fn score_collision(
        &self,
        scratch: &mut crate::welford::WelfordWorkerState,
        tally_idx: usize,
        weight: f64,
        total_xs: f64,
        position: [f64; 3],
        cell_id: Option<u32>,
        material: Option<&yamc_materials::Material>,
        material_id: Option<u32>,
        urr_xs: Option<&yamc_materials::UrrMacroXs>,
        photon_xs: Option<&yamc_materials::MacroPhotonXS>,
        energy: f64,
        particle_type: yamc_particle::ParticleType,
        parent_nuclide: Option<yamc_nuclide::nuclide_registry::NuclideId>,
        photon_heat_score_ev: Option<f64>,
    ) {
        self.score_collision_with(
            &mut |bin_idx, v| {
                scratch.add_contribution(tally_idx, bin_idx, v);
            },
            weight,
            total_xs,
            position,
            cell_id,
            material,
            material_id,
            urr_xs,
            photon_xs,
            energy,
            particle_type,
            parent_nuclide,
            photon_heat_score_ev,
        );
    }

    /// Closure-parameterised body that `score_collision` routes
    /// through. The per-bin loop calls `score_fn(bin_idx, v)` rather
    /// than touching storage directly -- same pattern as
    /// `score_track_length_with`.
    #[allow(clippy::too_many_arguments)]
    fn score_collision_with(
        &self,
        score_fn: &mut dyn FnMut(usize, f64),
        weight: f64,
        total_xs: f64,
        position: [f64; 3],
        cell_id: Option<u32>,
        material: Option<&yamc_materials::Material>,
        material_id: Option<u32>,
        urr_xs: Option<&yamc_materials::UrrMacroXs>,
        photon_xs: Option<&yamc_materials::MacroPhotonXS>,
        energy: f64,
        particle_type: yamc_particle::ParticleType,
        parent_nuclide: Option<yamc_nuclide::nuclide_registry::NuclideId>,
        photon_heat_score_ev: Option<f64>,
    ) {
        // Nothing to score if there is no Σ_t-dependent contribution
        // *and* no analog photon-heat contribution to write.
        if total_xs <= 0.0 && photon_heat_score_ev.is_none() {
            return;
        }

        // Particle-type filter is a hard gate.
        for filter in &self.filters {
            if let Filter::ParticleType(ptf) = filter {
                if !ptf.matches(particle_type) {
                    return;
                }
            }
        }

        // Energy filter -- bin or skip.
        let energy_bin: usize = if let Some(ef) = self.get_energy_filter() {
            match ef.get_bin(energy) {
                Some(bin) => bin,
                None => return,
            }
        } else {
            0
        };

        // Mesh filter -- single bin at the collision position (vs the
        // track-length estimator, which integrates over all bins the
        // segment touches). Unstructured meshes resolve the containing
        // tetrahedron with yamt's element-BVH point query (issue #354);
        // a collision outside the mesh volume scores nothing.
        let mesh_bin: usize = if let Some(mesh_filter) = self.get_mesh_filter() {
            match mesh_filter.get_bin(position) {
                Some(bin) => bin,
                None => return,
            }
        } else {
            #[cfg(feature = "mesh")]
            if let Some(umf) = self.get_unstructured_mesh_filter() {
                match umf.get_bin(position) {
                    Some(bin) => bin,
                    None => return,
                }
            } else {
                0
            }
            #[cfg(not(feature = "mesh"))]
            0
        };

        // Energy-function weighting (e.g. dose conversion), kept SEPARATE from
        // the weight/Σ_t factor (issues #378, #382).
        //
        // The two factors are not interchangeable. `weight/Σ_t` converts a
        // collision into a track-length equivalent and applies only to
        // Σ_t-dependent scores. The energy function is a user-supplied response
        // f(E) and applies to every score, including the analog photon heat,
        // which is already an eV deposit and must NOT take weight/Σ_t. Folding
        // both into one number forced that arm to choose between taking neither
        // or taking both, and it took neither -- so an energy function on a
        // collision-estimator photon heating tally acted as a gate but never as
        // a weight, and the tally disagreed with the same tally under the
        // track-length estimator.
        //
        // Off the table drops the whole event, matching every other arm.
        let mut ef_weight = 1.0_f64;
        for filter in &self.filters {
            if let Filter::EnergyFunction(ef) = filter {
                match ef.get_weight(energy) {
                    Some(energy_weight) => ef_weight = energy_weight,
                    None => return,
                }
                break;
            }
        }
        let base_weight = if total_xs > 0.0 {
            weight / total_xs * ef_weight
        } else {
            0.0
        };

        // Cell and material filters bin and gate this event.
        let cell_bin: usize = if let Some(cf) = self.get_cell_filter() {
            match cell_id.and_then(|id| cf.get_bin(id)) {
                Some(bin) => bin,
                None => return,
            }
        } else {
            0
        };
        let material_bin: usize = if let Some(mf) = self.get_material_filter() {
            match mf.get_bin(material_id) {
                Some(bin) => bin,
                None => return,
            }
        } else {
            0
        };

        // Parent-nuclide filter (decay-photon tracking).
        let parent_bin: usize = if let Some(pnf) = self.get_parent_nuclide_filter() {
            match parent_nuclide.and_then(|id| pnf.get_bin(id)) {
                Some(bin) => bin,
                None => return,
            }
        } else {
            0
        };

        // Per-collision contributions for each supported score. All
        // bin into nuclide 0 (collision-estimator scores are material-
        // level here; per-nuclide collision flavors would need
        // σ_r/Σ_t-style weighting and aren't implemented yet).
        //
        //   Flux:                 weight / Σ_t                    (units: cm)
        //   Heating (neutron):    (heating_xs/Σ_t) · weight × ef
        //   HeatingLocal (n):     (heating_local_xs/Σ_t) · weight × ef
        //   Heating (photon):     photon_heat_score_ev × ef       (analog, eV)
        //   HeatingLocal (photon): photon_heat_score_ev × ef
        //   ReactionRate(mt):     (σ_r/Σ_t) · weight × ef         (URR-aware)
        //   Production(mt):       (σ_production_mt/Σ_t) · weight × ef (neutron-only)
        //   DamageEnergy:         (damage_energy_xs/Σ_t) · weight × ef (neutron-only)
        //   PhotonXS(component):  (σ_component/Σ_t) · weight × ef     (photon-only)
        //
        // where `ef` is the optional energy-function weighting. It is folded
        // into `base_weight = weight/Σ_t · ef` for the Σ_t-dependent arms and
        // applied on its own to the analog photon heat, which takes `ef` but
        // not `weight/Σ_t` (issue #382).
        // `allow_negative` lets neutron KERMA (MT 301/901) fold in its
        // physically-negative endothermic windows (issue #84); damage
        // energy (MT 444) keeps the `> 0.0` guard.
        let neutron_xs_contrib = |lookup: fn(&yamc_materials::Material, f64) -> f64,
                                  allow_negative: bool| {
            if total_xs <= 0.0 {
                return None;
            }
            material.and_then(|m| {
                let xs = lookup(m, energy);
                let keep = if allow_negative { xs != 0.0 } else { xs > 0.0 };
                keep.then_some(xs * base_weight)
            })
        };
        for (score_idx, score) in self.scores.iter().enumerate() {
            let contrib = match (score, particle_type) {
                (Score::Flux(_), _) if total_xs > 0.0 => Some(base_weight),
                // Photon heating routing (issues #358/#356):
                // - Collision-estimator tallies take the analog eV
                //   deposit dispatched after each real photon collision
                //   (photon_heat_score_ev, total_xs == 0 on that call).
                // - TrackLength tallies reaching this path (the Woodcock
                //   flight estimator, total_xs = Sigma_maj) use the
                //   collision-density KERMA equivalent instead.
                // - Overlay tallies (multiply_density = false) score
                //   per-atom KERMA along tracks in eV*barn; the analog
                //   eV deposit would mix units and double count.
                (Score::Heating(_), yamc_particle::ParticleType::Photon)
                | (Score::HeatingLocal(_), yamc_particle::ParticleType::Photon)
                    if self.multiply_density && self.estimator == crate::Estimator::Collision =>
                {
                    // The eV deposit takes the energy function but NOT
                    // weight/Σ_t (issue #382). `energy` here is the
                    // pre-collision photon energy, the same E the track-length
                    // arm below evaluates f at and the same E the energy filter
                    // binned on, which is what makes the two arms estimators of
                    // the same integral.
                    photon_heat_score_ev.map(|heat| heat * ef_weight)
                }
                (Score::Heating(_), yamc_particle::ParticleType::Photon)
                | (Score::HeatingLocal(_), yamc_particle::ParticleType::Photon)
                    if self.multiply_density
                        && self.estimator == crate::Estimator::TrackLength
                        && total_xs > 0.0 =>
                {
                    photon_xs
                        .and_then(|pxs| (pxs.heating > 0.0).then_some(pxs.heating * base_weight))
                }
                (Score::Heating(_), yamc_particle::ParticleType::Neutron) => {
                    neutron_xs_contrib(yamc_materials::Material::lookup_heating_xs, true)
                }
                (Score::HeatingLocal(_), yamc_particle::ParticleType::Neutron) => {
                    neutron_xs_contrib(yamc_materials::Material::lookup_heating_local_xs, true)
                }
                // ReactionRate(MT=TOTAL) is a particle-agnostic total
                // interaction rate. On the photon path it means the photon
                // total macroscopic XS (Σ_total,photon · flux), matching the
                // GPU SCORE_TOTAL fast path and OpenMC's photon `total`. A
                // *specific* neutron MT on a photon is a category error: a
                // photon does not undergo a neutron reaction, so it scores 0
                // (the GPU likewise rejects it at validation -- see
                // `validate_tallies`). The neutron path is unchanged:
                // `macro_xs_by_mt(mt)` over the neutron MT table.
                (Score::ReactionRate(rr), yamc_particle::ParticleType::Photon)
                    if total_xs > 0.0 =>
                {
                    if rr.mt == Mt::TOTAL {
                        photon_xs
                            .and_then(|pxs| (pxs.total > 0.0).then_some(pxs.total * base_weight))
                    } else {
                        None
                    }
                }
                (Score::ReactionRate(rr), _) if total_xs > 0.0 => material.and_then(|m| {
                    let xs = m.macro_xs_by_mt(rr.mt.as_i32(), energy, urr_xs);
                    (xs > 0.0).then_some(xs * base_weight)
                }),
                (Score::Production(p), yamc_particle::ParticleType::Neutron) if total_xs > 0.0 => {
                    material.and_then(|m| {
                        let xs = m.lookup_production_xs(p.mt.as_i32(), energy);
                        (xs > 0.0).then_some(xs * base_weight)
                    })
                }
                (Score::DamageEnergy(_), yamc_particle::ParticleType::Neutron) => {
                    neutron_xs_contrib(yamc_materials::Material::lookup_damage_energy_xs, false)
                }
                (Score::PhotonXS(p), yamc_particle::ParticleType::Photon) if total_xs > 0.0 => {
                    photon_xs.and_then(|pxs| {
                        let xs = match p.component {
                            PhotonComponent::Coherent => pxs.coherent,
                            PhotonComponent::Incoherent => pxs.incoherent,
                            PhotonComponent::Photoelectric => pxs.photoelectric,
                            PhotonComponent::PairProduction => pxs.pair_production,
                        };
                        (xs > 0.0).then_some(xs * base_weight)
                    })
                }
                _ => None,
            };
            let Some(v) = contrib else { continue };
            if v == 0.0 {
                continue;
            }
            if let Some(bin_idx) = self.get_bin_index_7d(
                score_idx,
                cell_bin,
                material_bin,
                0,
                parent_bin,
                energy_bin,
                mesh_bin,
            ) {
                score_fn(bin_idx, v);
            }
        }
    }

    /// Create a new tally specification
    pub fn new() -> Self {
        Self {
            tally_id: None,
            name: None,
            scores: Vec::new(),
            nuclides: Vec::new(),
            filters: Vec::new(),
            units: String::new(),
            n_batches: AtomicU32::new(0),
            particles_per_chunk: AtomicU32::new(0),
            multiply_density: true,
            overlay_material: None,
            estimator: crate::Estimator::default(),
            accumulator: TallyAccumulator::new(),
        }
    }

    /// Update cached fields for fast scoring. Call after setting scores and filters.
    /// Can be called from shared reference (thread-safe one-time initialization).
    pub fn update_cache(&self) {
        // Only initialize once
        if self
            .accumulator
            .cache_initialized
            .swap(true, Ordering::AcqRel)
        {
            return;
        }

        self.accumulator
            .cached_num_energy_bins
            .get_or_init(|| self.num_energy_bins());

        self.accumulator.cached_energy_bins.get_or_init(|| {
            self.filters.iter().find_map(|f| {
                if let Filter::Energy(ef) = f {
                    Some(ef.bins.clone())
                } else {
                    None
                }
            })
        });

        // Fast-path cell gate: only cached for the common single-cell case.
        // Multi-cell filters bin rather than gate, and route through the slow path.
        self.accumulator.cached_cell_filter_id.get_or_init(|| {
            self.filters.iter().find_map(|f| {
                if let Filter::Cell(cf) = f {
                    if cf.cell_ids.len() == 1 {
                        Some(cf.cell_ids[0])
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
        });

        let flux_indices: Vec<usize> = self
            .scores
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                if matches!(s, Score::Flux(_)) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        self.accumulator
            .cached_has_flux_score
            .get_or_init(|| !flux_indices.is_empty());
        self.accumulator
            .cached_flux_score_indices
            .get_or_init(|| flux_indices);

        let heating_indices: Vec<usize> = self
            .scores
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                if matches!(s, Score::Heating(_)) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        self.accumulator
            .cached_has_heating_score
            .get_or_init(|| !heating_indices.is_empty());
        self.accumulator
            .cached_heating_score_indices
            .get_or_init(|| heating_indices);

        let heating_local_indices: Vec<usize> = self
            .scores
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                if matches!(s, Score::HeatingLocal(_)) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        self.accumulator
            .cached_has_heating_local_score
            .get_or_init(|| !heating_local_indices.is_empty());
        self.accumulator
            .cached_heating_local_score_indices
            .get_or_init(|| heating_local_indices);

        // Cache production score indices (score_idx, mt_number)
        self.accumulator
            .cached_production_score_indices
            .get_or_init(|| {
                self.scores
                    .iter()
                    .enumerate()
                    .filter_map(|(i, s)| match s {
                        Score::Production(p) => Some((i, p.mt.as_i32())),
                        _ => None,
                    })
                    .collect()
            });

        // Cache damage-energy score indices (MT 444)
        let damage_energy_indices: Vec<usize> = self
            .scores
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                if matches!(s, Score::DamageEnergy(_)) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        self.accumulator
            .cached_has_damage_energy_score
            .get_or_init(|| !damage_energy_indices.is_empty());
        self.accumulator
            .cached_damage_energy_score_indices
            .get_or_init(|| damage_energy_indices);

        // Cache EnergyFunctionFilter for dose calculations
        self.accumulator
            .cached_energy_function_filter
            .get_or_init(|| {
                self.filters.iter().find_map(|f| {
                    if let Filter::EnergyFunction(ef) = f {
                        Some(ef.clone())
                    } else {
                        None
                    }
                })
            });

        // Cache MT reaction score indices
        self.accumulator.cached_mt_score_indices.get_or_init(|| {
            self.scores
                .iter()
                .enumerate()
                .filter_map(|(i, s)| match s {
                    Score::ReactionRate(r) => Some((i, r.mt.as_i32())),
                    _ => None,
                })
                .collect()
        });

        // Cache particle type filter
        self.accumulator
            .cached_particle_type_filter
            .get_or_init(|| {
                self.filters.iter().find_map(|f| {
                    if let Filter::ParticleType(ptf) = f {
                        Some(ptf.particle_type)
                    } else {
                        None
                    }
                })
            });

        // Cache photon score indices (score_idx, component)
        let photon_indices: Vec<(usize, u8)> = self
            .scores
            .iter()
            .enumerate()
            .filter_map(|(i, s)| match s {
                Score::PhotonXS(p) => Some((i, p.component.as_u8())),
                _ => None,
            })
            .collect();
        self.accumulator
            .cached_has_photon_score
            .get_or_init(|| !photon_indices.is_empty());
        self.accumulator
            .cached_photon_score_indices
            .get_or_init(|| photon_indices);
    }

    /// Initialize storage for simulation (mutable version).
    /// Sets up the per-history Welford state via the
    /// `simulate_transport`-side worker pool; here we only reset
    /// counters and the cached scoring indices, plus the small
    /// per-bin `values` atomic vector used by the GPU writeback path.
    pub fn initialize_batches(&mut self, n_batches: usize) {
        self.update_cache();
        let num_bins = self.num_bins();
        self.accumulator.values = (0..num_bins).map(|_| AtomicU64::new(0)).collect();
        self.n_batches.store(n_batches as u32, Ordering::Relaxed);
        self.accumulator.n_realizations.store(0, Ordering::Relaxed);
    }

    /// Initialize through shared reference (for use with Arc<Tally>)
    pub fn initialize_batches_shared(&self, n_batches: usize) {
        self.update_cache();
        let num_bins = self.num_bins();
        for i in 0..num_bins.min(self.accumulator.values.len()) {
            self.accumulator.values[i].store(0, Ordering::Relaxed);
        }
        self.n_batches.store(n_batches as u32, Ordering::Relaxed);
        self.accumulator.n_realizations.store(0, Ordering::Relaxed);
    }

    /// Per-particle chunk size bookkeeping for compatibility with the
    /// existing simulation loop. With per-history Welford there's no
    /// batch fold to perform here; the per-rayon-worker `WorkerState`
    /// already holds the per-history stats and the global combine
    /// happens once in `simulate_transport`'s post-batch reduce.
    pub fn accumulate_batch(&self, particles_in_batch: u32) {
        if particles_in_batch == 0 {
            return;
        }
        self.particles_per_chunk
            .store(particles_in_batch, Ordering::Relaxed);
    }

    /// Store a single per-history bin value atomically. Used by yamc's GPU
    /// dispatch path to fold a batch's worth of kernel-side accumulation into
    /// the host-side tally before `accumulate_batch` finalises the realization.
    ///
    /// Bin indices come from [`Tally::get_bin_index_7d`]; callers are expected
    /// to pass a valid index.
    pub fn store_bin_value(&self, bin_idx: usize, value: f64) {
        self.accumulator.values[bin_idx].store(value.to_bits(), Ordering::Relaxed);
    }

    /// Install the finalized per-history Welford global stats for this
    /// tally. Called once by the simulation loop after the rayon
    /// fold/reduce completes and the worker state has been finalized. The
    /// read accessors (`get_mean`, `get_std_dev`, `total_mean`,
    /// `total_std`) route through this state.
    pub fn install_finalized(&self, stats: crate::welford::WelfordTallyStats) {
        // n_realizations is the variance estimator's sample count;
        // for per-history Welford this is the total source-history
        // count.
        self.accumulator
            .n_realizations
            // Saturate rather than wrap: the MPI rank fold (and very
            // large runs) can push the u64 history count past u32::MAX.
            // The exact count stays available via `get_n_histories`.
            .store(
                stats.n_histories.min(u32::MAX as u64) as u32,
                Ordering::Relaxed,
            );
        *self
            .accumulator
            .welford_finalized
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(stats);
    }

    /// Install the per-tally convergence history (aggregate statistics versus
    /// number of histories) recorded by the run loop at each checkpoint.
    /// Read back by [`finalize`](Self::finalize) into the `TallyResult`.
    pub fn install_convergence_history(&self, history: Vec<crate::result::ConvergencePoint>) {
        *self
            .accumulator
            .convergence_history
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = history;
    }

    /// The recorded convergence history (empty if the run did not record one).
    pub fn get_convergence_history(&self) -> Vec<crate::result::ConvergencePoint> {
        self.accumulator
            .convergence_history
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Per-bin mean. Reads from the per-history Welford state
    /// installed at the end of `simulate_transport`.
    pub fn get_mean(&self) -> Vec<f64> {
        if let Some(stats) = self
            .accumulator
            .welford_finalized
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
        {
            return stats.mean.clone();
        }
        vec![0.0; self.num_bins()]
    }

    /// Per-bin standard error of the mean.
    /// `sqrt(m2 / ((n-1) × n))` from the per-history Welford state.
    pub fn get_std_dev(&self) -> Vec<f64> {
        if let Some(stats) = self
            .accumulator
            .welford_finalized
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
        {
            return stats.std_err();
        }
        vec![0.0; self.num_bins()]
    }

    /// Sum-of-bin-means as a single number, computed in place without
    /// allocating a `Vec<f64>`.
    pub fn total_mean(&self) -> f64 {
        if let Some(stats) = self
            .accumulator
            .welford_finalized
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
        {
            return stats.mean.iter().sum();
        }
        0.0
    }

    /// Standard deviation of the summed bin total, assuming bin scores
    /// are independent across batches (true for the score patterns used
    /// here -- each particle contributes to one bin at a time, so per-
    /// batch bin scores are uncorrelated). Computed in place without
    /// allocating the per-bin std vector -- for huge meshes this is the
    /// difference between ~ms and ~tens-of-ms per call.
    ///
    /// Returns 0.0 when fewer than 2 batches have been folded in
    /// (Bessel's correction divides by n-1).
    pub fn total_std(&self) -> f64 {
        if let Some(stats) = self
            .accumulator
            .welford_finalized
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
        {
            let n = stats.n_histories as f64;
            if n <= 1.0 {
                return 0.0;
            }
            let mut var_total = 0.0;
            for &m in stats.m2.iter() {
                var_total += (m / ((n - 1.0) * n)).max(0.0);
            }
            return var_total.sqrt();
        }
        0.0
    }

    /// Raw per-bin Welford sum of squared deviations from the finalized
    /// state. Empty if no Welford state was installed (e.g. GPU runs).
    /// This is the exact merge state `combine_results` consumes;
    /// `get_std_dev` is derived from it.
    pub fn get_m2(&self) -> Vec<f64> {
        if let Some(stats) = self
            .accumulator
            .welford_finalized
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
        {
            return stats.m2.clone();
        }
        Vec::new()
    }

    /// Moments of the per-history total score from the finalized Welford
    /// state. `AggMoments::ZERO` if none was installed (e.g. GPU runs).
    pub fn get_agg(&self) -> crate::welford::AggMoments {
        if let Some(stats) = self
            .accumulator
            .welford_finalized
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
        {
            return stats.agg;
        }
        crate::welford::AggMoments::ZERO
    }

    /// Raw empirical PDF of the per-history total score from the finalized
    /// Welford state. Empty if none was installed.
    pub fn get_score_pdf(&self) -> crate::welford::ScorePdf {
        if let Some(stats) = self
            .accumulator
            .welford_finalized
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
        {
            return stats.score_pdf.clone();
        }
        crate::welford::ScorePdf::default()
    }

    /// Exact total source-history count from the finalized Welford
    /// state. `get_n_realizations` is the legacy u32 mirror and
    /// saturates at ~4.29e9 histories; this does not. Returns 0 if no
    /// Welford state was installed.
    pub fn get_n_histories(&self) -> u64 {
        if let Some(stats) = self
            .accumulator
            .welford_finalized
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
        {
            return stats.n_histories;
        }
        0
    }

    /// Get relative error for each bin
    pub fn get_rel_error(&self) -> Vec<f64> {
        let means = self.get_mean();
        let std_devs = self.get_std_dev();

        means
            .iter()
            .zip(std_devs.iter())
            .map(|(&m, &s)| if m > 0.0 { s / m } else { 0.0 })
            .collect()
    }

    /// Get the total count across all batches (sum of normalized values * n)
    pub fn total_count(&self) -> Vec<u64> {
        let means = self.get_mean();
        // Prefer the exact u64 history count: the u32 `n_realizations`
        // mirror saturates above ~4.29e9 (reachable via the MPI rank
        // fold) and would corrupt the counts here. Paths without Welford
        // state (GPU) still rely on `n_realizations`.
        let n_histories = self.get_n_histories();
        let n = if n_histories > 0 {
            n_histories as f64
        } else {
            self.accumulator.n_realizations.load(Ordering::Relaxed) as f64
        };
        let ppb = self.particles_per_chunk.load(Ordering::Relaxed) as f64;

        means
            .iter()
            .map(|&m| {
                // mean is per-particle, so total = mean * particles_per_chunk * n_batches
                (m * ppb * n) as u64
            })
            .collect()
    }

    /// Snapshot the currently-accumulated statistics into a `TallyResult`.
    ///
    /// This is the finalize step for PR A of the `SimulationResults` refactor:
    /// it produces an immutable result value from the current batch state
    /// without disturbing any existing scoring machinery. PR B will wrap a
    /// set of these into `SimulationResults` and return it from
    /// `model.simulate_transport()`.
    ///
    /// Requires the `Tally` to be shared through an `Arc` so that the result
    /// can hold a back-reference to the config without copying.
    pub fn finalize(self: &Arc<Self>) -> crate::TallyResult {
        let (shape, dim_labels) = crate::result::compute_shape_and_dims(self);
        crate::TallyResult {
            tally: Arc::clone(self),
            mean: self.get_mean(),
            standard_deviation: self.get_std_dev(),
            relative_error: self.get_rel_error(),
            m2: self.get_m2(),
            n_histories: self.get_n_histories(),
            total_count: self.total_count(),
            // Figure of merit is populated by `TallyResult::with_fom` --
            // requires the simulation's elapsed wall-clock time, which
            // `finalize` doesn't have. `SimulationResults::from_tallies`
            // chains `.with_fom(elapsed_secs)` after this.
            figure_of_merit: Vec::new(),
            aggregate_figure_of_merit: 0.0,
            agg: self.get_agg(),
            score_pdf: self.get_score_pdf(),
            convergence_history: self.get_convergence_history(),
            shape,
            dim_labels,
            n_batches: self.get_n_realizations(),
            particles_per_chunk: self.particles_per_chunk.load(Ordering::Relaxed),
            // Provenance fields are filled by `with_fom` (elapsed) and
            // `SimulationResults::from_tallies_with_run` (run_indices).
            elapsed_secs: 0.0,
            run_indices: Vec::new(),
        }
    }

    /// Create a new tally with name and units (for simulation results)
    pub fn with_name_and_units(name: &str, units: &str) -> Self {
        Self {
            tally_id: None,
            name: Some(name.to_string()),
            scores: Vec::new(),
            nuclides: Vec::new(),
            filters: Vec::new(),
            units: units.to_string(),
            n_batches: AtomicU32::new(0),
            particles_per_chunk: AtomicU32::new(0),
            multiply_density: true,
            overlay_material: None,
            estimator: crate::Estimator::default(),
            accumulator: TallyAccumulator::new(),
        }
    }

    /// Set scores from a mix of integers and score names
    pub fn set_scores_mixed(&mut self, scores: Vec<Score>) {
        self.scores = scores;
    }

    /// Validate that the tally configuration is valid
    /// A per-nuclide axis folds a nuclide's cross section into the score, so it
    /// cannot apply to a score that has no cross section (issue #305).
    ///
    /// Split out of [`Self::validate`] so the Python constructor can enforce it
    /// at construction, where the user can still fix the call, rather than at
    /// the start of a run. One implementation, both entry points.
    pub fn validate_nuclide_axis(&self) -> Result<(), String> {
        // A per-nuclide axis folds a nuclide's cross section into the score,
        // so it cannot apply to a score that has no cross section. `flux` is
        // the only such score today (`Score::mt()` is `None` for it, and for
        // nothing else). Both axes are affected: `response=` (the virtual
        // overlay, `multiply_density == false`) and `nuclides=` (the real
        // in-material breakdown).
        //
        // This used to be accepted and silently do nothing, in three flavours
        // (issue #305): `flux` + `response=` returned plain flux with the
        // response dropped; `['flux', 'heating']` + `response=` returned
        // response-weighted heating next to plain flux in one array with no
        // marker; and `flux` + `nuclides=['Fe56', 'total']` duplicated the same
        // flux into every nuclide bin, which reads as a per-nuclide breakdown
        // that does not exist.
        //
        // The whole tally is refused rather than the axis being applied only
        // where it means something: a tally is the unit of "what am I
        // measuring", and one whose axis applies to some of its scores but not
        // others hands back an array with two meanings in it. Scoring flux in
        // its own tally is one extra line and says what the user meant.
        let axis = if !self.multiply_density {
            Some("a response (response=)")
        } else if self
            .nuclides
            .iter()
            .any(|n| matches!(n, NuclideBin::Specific(_)))
        {
            Some("a per-nuclide breakdown (nuclides=)")
        } else {
            None
        };
        if let Some(axis) = axis {
            if let Some(score) = self.scores.iter().find(|s| s.mt().is_none()) {
                return Err(format!(
                    "Tally '{}': {axis} cannot apply to the '{}' score, which has no cross section to fold it into -- the axis would be silently ignored (and with several scores, applied to some of them and not others). Score '{}' in its own tally.",
                    self.display_name(),
                    score.name(),
                    score.name(),
                ));
            }
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut filter_type_counts: HashMap<&str, u32> = HashMap::new();

        for filter in &self.filters {
            let type_name = filter.type_name();
            *filter_type_counts.entry(type_name).or_insert(0) += 1;
        }

        for (filter_type, count) in filter_type_counts {
            if count > 1 {
                return Err(format!(
                    "Tally '{}' has {} filters of type {}. Multiple filters of the same type are not allowed.",
                    self.display_name(),
                    count,
                    filter_type
                ));
            }
        }

        // Overlay tallies (multiply_density=false) need a response target:
        // either >=1 specific nuclide (unit-density nuclide overlay) or a
        // non-empty material response (issue #341). The two are mutually
        // exclusive -- a material response collapses to one combined bin
        // (`nuclides == [Total]`) and carries its own nuclide list + densities.
        if !self.multiply_density {
            let has_specific = self
                .nuclides
                .iter()
                .any(|n| matches!(n, NuclideBin::Specific(_)));
            let has_material = self
                .overlay_material
                .as_ref()
                .is_some_and(|m| !m.is_empty());
            if has_specific && has_material {
                return Err(format!(
                    "Tally '{}': a material response and specific nuclides are mutually exclusive",
                    self.display_name()
                ));
            }
            if !has_specific && !has_material {
                return Err(format!(
                    "Tally '{}': multiply_density=false requires a response target -- at least one specific nuclide or a non-empty material",
                    self.display_name()
                ));
            }
        }

        self.validate_nuclide_axis()?;

        // Validate score/estimator compatibility. Heating-family scores
        // require the collision estimator; everything else is permissive.
        for score in &self.scores {
            if let Some(required) = score.required_estimator() {
                if required != self.estimator {
                    return Err(format!(
                        "Tally '{}' uses estimator {} but score {:?} requires {}",
                        self.display_name(),
                        self.estimator,
                        score.name(),
                        required,
                    ));
                }
            }
        }

        Ok(())
    }

    /// Pre-load microscopic cross section data for overlay tallying.
    ///
    /// Called before transport when `multiply_density = false`. Loads:
    /// - Neutron: microscopic XS on a unified energy grid for each overlay nuclide
    /// - Photon: `PhotonInteraction` data for each overlay nuclide's element
    ///
    /// # Arguments
    /// * `mt_filter` - List of MT numbers needed by all tallies
    /// * `photon_data_paths` - Map of element symbol → Arrow path for photon data
    /// * `needs_neutron` / `needs_photon` - which data set is load-bearing for
    ///   this run. The caller knows which species will actually score the tally
    ///   (its `ParticleType` filter combined with what the model transports).
    ///   Missing data for a species that WILL score used to warn and then score
    ///   the plain, un-responded quantity (or a flat zero), i.e. a wrong answer
    ///   behind a stderr line, so that case is an error now (issue #288).
    ///   Missing data for a species that cannot score stays a warning.
    pub fn prepare_overlay_xs(
        &self,
        mt_filter: &[i32],
        photon_data_paths: &HashMap<String, String>,
        needs_neutron: bool,
        needs_photon: bool,
    ) -> Result<(), String> {
        if self.multiply_density {
            return Ok(());
        }

        let mut fatal: Option<String> = None;

        self.accumulator.cached_overlay_xs.get_or_init(|| {
            // Determine the overlay's nuclide list, per-nuclide densities, and
            // whether to collapse them into one combined macroscopic bin:
            //
            // - Material response (issue #341): the nuclide list and the real
            //   atom densities (atoms/barn-cm) come from `overlay_material`,
            //   and `combine = true` → one combined bin `Σ_i N_i·σ_i`.
            // - Nuclide overlay (historical): the nuclides come from the
            //   `Specific` bins, scored at unit density (no `densities`
            //   entries → `density()` returns 1.0), `combine = false` → one
            //   bin per nuclide.
            let (overlay_nuclides, densities, combine): (Vec<String>, HashMap<String, f64>, bool) =
                if let Some(mat) = &self.overlay_material {
                    let names: Vec<String> = mat.keys().cloned().collect();
                    let dens: HashMap<String, f64> =
                        mat.iter().map(|(k, v)| (k.clone(), *v)).collect();
                    (names, dens, true)
                } else {
                    let names: Vec<String> = self
                        .nuclides
                        .iter()
                        .filter_map(|n| match n {
                            NuclideBin::Specific(name) => Some(name.clone()),
                            NuclideBin::Total => None,
                        })
                        .collect();
                    (names, HashMap::new(), false)
                };

            if overlay_nuclides.is_empty() {
                return None;
            }

            // --- Neutron overlay ---
            // Create a temporary material with the overlay nuclides to get microscopic XS.
            // We use atom fraction = 1.0 for each nuclide: `calculate_microscopic_xs_neutron`
            // returns per-nuclide microscopic XS (barns), which is independent of the fraction.
            // Any real atom-density weighting (material response) is applied at scoring time.
            let nuclides: std::collections::HashMap<String, f64> = overlay_nuclides
                .iter()
                .map(|nuc| (nuc.clone(), 1.0))
                .collect();
            let mut temp_mat =
                yamc_materials::Material::new(nuclides, "atom", "sum", None).unwrap();
            // Default to 294 K for overlay nuclides when data has multiple temperatures
            temp_mat.set_temperature("294");

            // Load nuclear data for the overlay nuclides (from global config).
            // This may fail if the global config doesn't have paths for overlay nuclides.
            let mut neutron_loaded = false;
            let mut neutron_problem: Option<String> = None;
            match temp_mat.ensure_nuclides_loaded() {
                Ok(()) if !temp_mat.nuclide_data.is_empty() => {
                    neutron_loaded = true;
                }
                Ok(()) => {
                    neutron_problem =
                        Some("no neutron data available for the overlay nuclides".to_string());
                }
                Err(e) => {
                    neutron_problem = Some(format!("failed to load neutron data: {e}"));
                }
            }
            if let Some(problem) = neutron_problem {
                let message = format!(
                    "overlay tally '{}': {problem}. A response (response=) needs the \
                     overlay nuclides' own data, which is loaded from the global \
                     configuration (`yamc.cross_section_data`), not from the cell materials. \
                     Without it the tally would silently score the plain, un-responded \
                     quantity.",
                    self.display_name()
                );
                if needs_neutron {
                    fatal = Some(message);
                } else {
                    // No neutron will score this tally in this run.
                    eprintln!("[WARNING] {message}");
                }
            }

            // Get microscopic XS on unified grid (only if neutron data was loaded)
            let (micro_xs, energy_grid) = if neutron_loaded {
                let micro = temp_mat.calculate_microscopic_xs_neutron(Some(&mt_filter.to_vec()));
                let grid = temp_mat.unified_energy_grid_neutron();
                (micro, grid)
            } else {
                (HashMap::new(), Vec::new())
            };

            // Build log-grid lookup
            let (log_grid_index, log_e_min, inv_log_delta) =
                OverlayXsData::build_log_grid(&energy_grid);

            // --- Photon overlay ---
            let mut photon_elements: HashMap<String, Arc<yamc_element::photon::PhotonInteraction>> =
                HashMap::new();

            for nuc in &overlay_nuclides {
                let elem_sym = yamc_element::element::element_symbol_from_nuclide(nuc);

                // Skip if we already loaded this element for a different nuclide
                if photon_elements.contains_key(nuc) {
                    continue;
                }

                // Try global cache first
                if let Some(arc) = yamc_element::photon::get_element_by_name(&elem_sym) {
                    photon_elements.insert(nuc.clone(), arc);
                    continue;
                }

                // Try loading from photon_data_paths
                if let Some(path) = photon_data_paths.get(&elem_sym) {
                    match yamc_element::photon::get_or_load_element(&elem_sym, path) {
                        Ok(arc) => {
                            photon_elements.insert(nuc.clone(), arc);
                        }
                        Err(e) => {
                            let message = format!(
                                "overlay tally '{}': failed to load photon data for {}: {e}",
                                self.display_name(),
                                elem_sym
                            );
                            if needs_photon {
                                fatal = Some(message);
                            } else {
                                eprintln!("[WARNING] {message}");
                            }
                        }
                    }
                } else if needs_photon {
                    // A photon-scoring overlay with no element data would score a
                    // flat zero, which reads as "no response here" rather than
                    // "no data" (issue #288).
                    fatal = Some(format!(
                        "overlay tally '{}': no photon data for element {} -- a photon \
                         response needs per-element photon data (load it on a material, \
                         or set `yamc.photon_data`), else the tally scores zero \
                         everywhere",
                        self.display_name(),
                        elem_sym
                    ));
                }
            }

            Some(OverlayXsData {
                energy_grid,
                micro_xs,
                log_grid_index,
                log_e_min,
                inv_log_delta,
                photon_elements,
                nuclide_names: overlay_nuclides,
                densities,
                combine,
            })
        });

        match fatal {
            Some(message) => Err(message),
            None => Ok(()),
        }
    }

    /// Get the overlay XS data, if initialized.
    #[inline]
    fn overlay_xs(&self) -> Option<&OverlayXsData> {
        self.accumulator
            .cached_overlay_xs
            .get()
            .and_then(|opt| opt.as_ref())
    }

    /// Get the display name of the tally
    pub fn display_name(&self) -> String {
        self.name
            .clone()
            .unwrap_or_else(|| "Unnamed Tally".to_string())
    }

    /// Get number of realizations (batches accumulated)
    pub fn get_n_realizations(&self) -> u32 {
        self.accumulator.n_realizations.load(Ordering::Relaxed)
    }

    /// Derive physical units for each score based on score type and filters.
    ///
    /// Returns a `Vec<String>` with one entry per score. The units account for:
    /// - Score type (flux → cm, heating → eV, reactions → reactions, etc.)
    /// - Mesh filter (adds / cm³ denominator)
    /// - EnergyFunctionFilter (uses user-supplied units if provided)
    pub fn derive_units(&self) -> Vec<String> {
        let has_mesh = self.filters.iter().any(|f| {
            matches!(f, Filter::Mesh(_)) || {
                #[cfg(feature = "mesh")]
                {
                    matches!(f, Filter::UnstructuredMesh(_))
                }
                #[cfg(not(feature = "mesh"))]
                {
                    false
                }
            }
        });

        let energy_fn_units = self.filters.iter().find_map(|f| {
            if let Filter::EnergyFunction(ef) = f {
                ef.units.clone()
            } else {
                None
            }
        });

        self.scores
            .iter()
            .map(|score| {
                let base = score.base_units();
                if let Some(ref u) = energy_fn_units {
                    if has_mesh {
                        format!("{u} · {base} / cm³ / source-particle")
                    } else {
                        format!("{u} · {base} / source-particle")
                    }
                } else if has_mesh {
                    format!("{base} / cm³ / source-particle")
                } else {
                    format!("{base} / source-particle")
                }
            })
            .collect()
    }
}

impl fmt::Display for Tally {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Tally name: {}", self.display_name())?;
        let means = self.get_mean();
        let std_devs = self.get_std_dev();
        let rel_errors = self.get_rel_error();
        let total_counts = self.total_count();
        let n_realizations = self.accumulator.n_realizations.load(Ordering::Relaxed);
        let units = self.derive_units();

        for (i, score) in self.scores.iter().enumerate() {
            let score_name = match score {
                Score::ReactionRate(r) if r.display_name.is_none() => {
                    format!("MT {}", r.mt)
                }
                _ => {
                    // Capitalize the first character of the name
                    let name = score.name();
                    let mut chars = name.chars();
                    match chars.next() {
                        None => String::new(),
                        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
                    }
                }
            };
            writeln!(f, "  Score {score_name}:")?;
            let mean = means.get(i).copied().unwrap_or(0.0);
            let std_dev = std_devs.get(i).copied().unwrap_or(0.0);
            let rel_error = rel_errors.get(i).copied().unwrap_or(0.0);
            let total = total_counts.get(i).copied().unwrap_or(0);
            let unit = units.get(i).map(|u| u.as_str()).unwrap_or("per particle");

            writeln!(f, "    Mean: {mean:.6e} [{unit}]")?;
            writeln!(f, "    Std Dev: {std_dev:.6e} [{unit}]")?;
            writeln!(
                f,
                "    Rel Error: {:.4} ({:.2}%)",
                rel_error,
                rel_error * 100.0
            )?;
            writeln!(f, "    Total count: {}", total)?;
        }
        writeln!(f, "    Batches: {n_realizations}")?;
        writeln!(
            f,
            "    Particles per batch: {}",
            self.particles_per_chunk.load(Ordering::Relaxed)
        )
    }
}

/// Initialize tallies from user tally specifications
pub fn create_tallies_from_specs(tally_specs: &[Tally]) -> Vec<Tally> {
    let mut tallies = Vec::new();

    // Always add leakage tally as the first tally
    tallies.push(Tally::with_name_and_units("Leakage", "particles"));

    // Add user-specified tallies
    for (i, spec) in tally_specs.iter().enumerate() {
        let name = spec
            .name
            .clone()
            .unwrap_or_else(|| format!("Tally {}", i + 1));

        let mut tally = Tally::with_name_and_units(&name, "");
        tally.tally_id = spec.tally_id;
        tally.scores = spec.scores.clone();
        tally.nuclides = spec.nuclides.clone();
        tally.filters = spec.filters.clone();

        if let Err(err) = tally.validate() {
            panic!("Invalid tally configuration: {err}");
        }

        tallies.push(tally);
    }

    tallies
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::mesh::MeshFilter;
    use crate::mesh::RegularRectangularMesh;

    fn make_mesh_tally() -> Tally {
        // 3x4x5 mesh from (0,0,0) to (3,4,5). 60 bins total. Set each
        // bin to its index value so the slice-extraction tests can
        // verify which bin landed where.
        let mesh = RegularRectangularMesh::new([0.0, 0.0, 0.0], [3.0, 4.0, 5.0], [3, 4, 5]);
        let mesh_filter = MeshFilter::new(mesh);

        let mut tally = Tally::new();
        tally.scores = vec![Score::Flux(FluxScore)];
        tally.filters = vec![Filter::Mesh(mesh_filter)];
        let arc = std::sync::Arc::new(tally);
        let stats = crate::welford::WelfordTallyStats {
            mean: (0..60).map(|i| i as f64).collect(),
            m2: vec![0.0; 60],
            n_histories: 1,
            agg: crate::welford::AggMoments::ZERO,
            score_pdf: crate::welford::ScorePdf::default(),
        };
        arc.install_finalized(stats);
        // Unwrap the Arc back -- tests want owned Tally to mutate filters etc.
        std::sync::Arc::try_unwrap(arc).ok().unwrap()
    }

    /// Pre-rip-out: exercised the GPU-dispatch writeback into the
    /// per-batch atomic-CAS accumulator. With per-history Welford
    /// the GPU dispatch installs `WelfordTallyStats` directly
    /// via `install_finalized`, so this test no longer
    /// reflects how the path is wired. Kept disabled as a marker;
    /// a fresh GPU-writeback test will be added when the GPU
    /// dispatcher is updated for the per-history-only path.
    #[ignore]
    #[test]
    fn gpu_writeback_shape_yields_correct_stats() {
        // 4-bin Flux tally (no filters so num_bins == 1 score × 1 bin… use mesh)
        let mesh = RegularRectangularMesh::new([0.0, 0.0, 0.0], [2.0, 2.0, 1.0], [2, 2, 1]);
        let mut tally = Tally::new();
        tally.scores = vec![Score::Flux(FluxScore)];
        tally.filters = vec![Filter::Mesh(MeshFilter::new(mesh))];
        // 3 batches of fixed per-bin values; particles_per_chunk = 100.
        // Per-bin per-batch normalized values are values[batch][bin] / 100.
        let raw_values: [[f64; 4]; 3] = [
            [10.0, 20.0, 30.0, 40.0],
            [12.0, 18.0, 32.0, 38.0],
            [8.0, 22.0, 28.0, 42.0],
        ];
        let particles_per_chunk: u32 = 100;
        tally.initialize_batches(raw_values.len());

        for batch_values in raw_values.iter() {
            for (bin, &v) in batch_values.iter().enumerate() {
                tally.store_bin_value(bin, v);
            }
            tally.accumulate_batch(particles_per_chunk);
        }

        // Expected per-bin mean = average of (raw / 100) over 3 batches.
        let n = raw_values.len() as f64;
        let mut expected_mean = [0.0_f64; 4];
        let mut expected_m2 = [0.0_f64; 4];
        for batch_values in raw_values.iter() {
            for (bin, &v) in batch_values.iter().enumerate() {
                let normalized = v / particles_per_chunk as f64;
                expected_mean[bin] += normalized;
            }
        }
        for m in expected_mean.iter_mut() {
            *m /= n;
        }
        for batch_values in raw_values.iter() {
            for (bin, &v) in batch_values.iter().enumerate() {
                let normalized = v / particles_per_chunk as f64;
                let d = normalized - expected_mean[bin];
                expected_m2[bin] += d * d;
            }
        }

        let mean = tally.get_mean();
        let std_dev = tally.get_std_dev();
        assert_eq!(mean.len(), 4);
        assert_eq!(std_dev.len(), 4);

        for bin in 0..4 {
            assert!(
                (mean[bin] - expected_mean[bin]).abs() < 1e-12,
                "bin {bin}: get_mean = {} vs expected {}",
                mean[bin],
                expected_mean[bin]
            );
            // std_err = sqrt(var / n) = sqrt((M2 / (n-1)) / n)
            let expected_std = (expected_m2[bin] / ((n - 1.0) * n)).sqrt();
            assert!(
                (std_dev[bin] - expected_std).abs() < 1e-12,
                "bin {bin}: get_std_dev = {} vs expected {}",
                std_dev[bin],
                expected_std
            );
        }
    }

    #[test]
    fn test_extract_mesh_slice_xy() {
        let tally = make_mesh_tally();

        // XY slice at z=0.5 -> iz=0
        let values = tally
            .extract_mesh_slice("xy", Some(0.5), 0, None, "mean")
            .unwrap();

        // Should be 4 rows (ny) x 3 cols (nx)
        assert_eq!(values.len(), 4);
        assert_eq!(values[0].len(), 3);

        // iz=0: mesh_bin = (0 * 4 + iy) * 3 + ix
        // values[iy][ix] should equal (0*4 + iy)*3 + ix = iy*3 + ix
        for (iy, row) in values.iter().enumerate() {
            for (ix, &v) in row.iter().enumerate() {
                assert_eq!(v, (iy * 3 + ix) as f64);
            }
        }
    }

    #[test]
    fn test_extract_mesh_slice_xz() {
        let tally = make_mesh_tally();

        // XZ slice at y=1.5 -> iy=1
        let values = tally
            .extract_mesh_slice("xz", Some(1.5), 0, None, "mean")
            .unwrap();

        // Should be 5 rows (nz) x 3 cols (nx)
        assert_eq!(values.len(), 5);
        assert_eq!(values[0].len(), 3);

        // iy=1: mesh_bin = (iz * 4 + 1) * 3 + ix
        for (iz, row) in values.iter().enumerate() {
            for (ix, &v) in row.iter().enumerate() {
                let expected = (iz * 4 + 1) * 3 + ix;
                assert_eq!(v, expected as f64);
            }
        }
    }

    #[test]
    fn test_extract_mesh_slice_yz() {
        let tally = make_mesh_tally();

        // YZ slice at x=2.5 -> ix=2
        let values = tally
            .extract_mesh_slice("yz", Some(2.5), 0, None, "mean")
            .unwrap();

        // Should be 5 rows (nz) x 4 cols (ny)
        assert_eq!(values.len(), 5);
        assert_eq!(values[0].len(), 4);

        // ix=2: mesh_bin = (iz * 4 + iy) * 3 + 2
        for (iz, row) in values.iter().enumerate() {
            for (iy, &v) in row.iter().enumerate() {
                let expected = (iz * 4 + iy) * 3 + 2;
                assert_eq!(v, expected as f64);
            }
        }
    }

    #[test]
    fn test_extract_mesh_slice_default_center() {
        let tally = make_mesh_tally();

        // No slice_coord -> uses center (z=2.5 -> iz=2)
        let values = tally
            .extract_mesh_slice("xy", None, 0, None, "mean")
            .unwrap();

        // iz=2: mesh_bin = (2 * 4 + iy) * 3 + ix
        for (iy, row) in values.iter().enumerate() {
            for (ix, &v) in row.iter().enumerate() {
                let expected = (2 * 4 + iy) * 3 + ix;
                assert_eq!(v, expected as f64);
            }
        }
    }

    #[test]
    fn test_extract_mesh_slice_out_of_bounds() {
        let tally = make_mesh_tally();

        let result = tally.extract_mesh_slice("xy", Some(10.0), 0, None, "mean");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("outside mesh bounds"));
    }

    #[test]
    fn test_extract_mesh_slice_no_mesh_filter() {
        let mut tally = Tally::new();
        tally.scores = vec![Score::Flux(FluxScore)];

        let result = tally.extract_mesh_slice("xy", None, 0, None, "mean");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no MeshFilter"));
    }

    #[test]
    fn test_extract_mesh_slice_invalid_basis() {
        let tally = make_mesh_tally();

        let result = tally.extract_mesh_slice("ab", None, 0, None, "mean");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid basis"));
    }

    #[test]
    fn test_extract_mesh_slice_invalid_value_type() {
        let tally = make_mesh_tally();

        let result = tally.extract_mesh_slice("xy", None, 0, None, "invalid");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid value_type"));
    }

    /// Test that all scores with both string and MT integer representations
    /// parse to the same Score variant via FromStr.
    #[test]
    fn test_score_string_and_int_parse_equivalence() {
        // (string_name, mt_int_string, expected_variant_name)
        let cases: Vec<(&str, &str, &str)> = vec![
            ("total", "1", "total"),
            ("elastic", "2", "elastic"),
            ("inelastic", "4", "inelastic"),
            ("fission", "18", "fission"),
            ("absorption", "27", "absorption"),
            ("heating", "301", "heating"),
            ("heating-local", "901", "heating-local"),
            ("damage-energy", "444", "damage-energy"),
            ("H1-production", "203", "H1-production"),
            ("H2-production", "204", "H2-production"),
            ("H3-production", "205", "H3-production"),
            ("He3-production", "206", "He3-production"),
            ("He4-production", "207", "He4-production"),
        ];

        for (string_name, mt_str, expected_name) in &cases {
            let from_string: Score = string_name
                .parse()
                .unwrap_or_else(|e| panic!("Failed to parse string score '{string_name}': {e}"));
            let from_int: Score = mt_str
                .parse()
                .unwrap_or_else(|e| panic!("Failed to parse int score '{mt_str}': {e}"));

            assert_eq!(
                from_string.name(),
                *expected_name,
                "String '{string_name}' should produce name '{expected_name}', got '{}'",
                from_string.name()
            );
            assert_eq!(
                from_int.name(),
                *expected_name,
                "Int '{mt_str}' should produce name '{expected_name}', got '{}'",
                from_int.name()
            );
            assert_eq!(
                from_string.name(),
                from_int.name(),
                "String '{string_name}' and int '{mt_str}' should produce same Score variant"
            );
        }
    }

    /// Test that Score::to_i32() returns correct MT numbers and round-trips via FromStr.
    #[test]
    fn test_score_to_i32_roundtrip() {
        // (Score variant, expected MT, MT that FromStr maps back to this variant)
        let scores: Vec<(Score, i32, i32)> = vec![
            (Score::Heating(HeatingScore), 301, 301),
            (Score::HeatingLocal(HeatingLocalScore), 901, 901),
            (Score::DamageEnergy(DamageEnergyScore), 444, 444),
            (Score::Production(ProductionScore::H1), 203, 203),
            (Score::Production(ProductionScore::H2), 204, 204),
            (Score::Production(ProductionScore::H3), 205, 205),
            (Score::Production(ProductionScore::HE3), 206, 206),
            (Score::Production(ProductionScore::HE4), 207, 207),
            (Score::ReactionRate(ReactionRateScore::total()), 1, 1),
            (Score::ReactionRate(ReactionRateScore::elastic()), 2, 2),
            (Score::ReactionRate(ReactionRateScore::inelastic()), 4, 4),
            (Score::ReactionRate(ReactionRateScore::fission()), 18, 18),
            (Score::ReactionRate(ReactionRateScore::absorption()), 27, 27),
        ];

        for (score, expected_mt, fromstr_mt) in &scores {
            assert_eq!(
                score.to_i32(),
                *expected_mt,
                "{} should have MT {}",
                score.name(),
                expected_mt
            );

            // Parsing the fromstr_mt should give back the same variant
            let reparsed: Score = fromstr_mt.to_string().parse().unwrap();
            assert_eq!(
                reparsed.name(),
                score.name(),
                "Reparsing MT {fromstr_mt} should give '{}', got '{}'",
                score.name(),
                reparsed.name()
            );
        }
    }

    /// Test that unknown MT numbers stay as unnamed ReactionRate.
    #[test]
    fn test_score_unknown_mt_stays_as_reaction_rate() {
        let score: Score = "102".parse().unwrap();
        assert!(matches!(score, Score::ReactionRate(ref r) if r.display_name.is_none()));
        assert_eq!(score.to_i32(), 102);
        assert_eq!(score.name(), "102");

        let score2: Score = "16".parse().unwrap();
        assert!(matches!(score2, Score::ReactionRate(ref r) if r.display_name.is_none()));
    }

    #[test]
    fn test_base_units_flux() {
        assert_eq!(Score::Flux(FluxScore).base_units(), "cm");
    }

    #[test]
    fn test_base_units_heating() {
        assert_eq!(Score::Heating(HeatingScore).base_units(), "eV");
        assert_eq!(Score::HeatingLocal(HeatingLocalScore).base_units(), "eV");
        assert_eq!(Score::DamageEnergy(DamageEnergyScore).base_units(), "eV");
    }

    #[test]
    fn test_base_units_production() {
        assert_eq!(
            Score::Production(ProductionScore::H1).base_units(),
            "particles"
        );
        assert_eq!(
            Score::Production(ProductionScore::HE4).base_units(),
            "particles"
        );
    }

    #[test]
    fn test_base_units_reactions() {
        assert_eq!(
            Score::ReactionRate(ReactionRateScore::total()).base_units(),
            "reactions"
        );
        assert_eq!(
            Score::ReactionRate(ReactionRateScore::absorption()).base_units(),
            "reactions"
        );
        assert_eq!(
            Score::ReactionRate(ReactionRateScore::fission()).base_units(),
            "reactions"
        );
        assert_eq!(
            Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(102))).base_units(),
            "reactions"
        );
    }

    #[test]
    fn test_derive_units_simple_flux() {
        let mut tally = Tally::new();
        tally.scores = vec![Score::Flux(FluxScore)];
        let units = tally.derive_units();
        assert_eq!(units, vec!["cm / source-particle"]);
    }

    #[test]
    fn test_derive_units_with_mesh() {
        let mesh = RegularRectangularMesh::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2, 2, 2]);
        let mut tally = Tally::new();
        tally.scores = vec![Score::Flux(FluxScore)];
        tally.filters = vec![Filter::Mesh(MeshFilter::new(mesh))];
        let units = tally.derive_units();
        assert_eq!(units, vec!["cm / cm³ / source-particle"]);
    }

    #[test]
    fn test_derive_units_mixed_scores() {
        let mut tally = Tally::new();
        tally.scores = vec![
            Score::Flux(FluxScore),
            Score::Heating(HeatingScore),
            Score::ReactionRate(ReactionRateScore::absorption()),
        ];
        let units = tally.derive_units();
        assert_eq!(
            units,
            vec![
                "cm / source-particle",
                "eV / source-particle",
                "reactions / source-particle",
            ]
        );
    }

    #[test]
    fn test_derive_units_energy_function_filter_with_units() {
        use crate::EnergyFunctionFilter;
        let ef = EnergyFunctionFilter::with_units(
            vec![1.0, 10.0, 100.0, 1000.0],
            vec![1.0, 2.0, 3.0, 4.0],
            "pSv·cm²",
        );
        let mut tally = Tally::new();
        tally.scores = vec![Score::Flux(FluxScore)];
        tally.filters = vec![Filter::EnergyFunction(ef)];
        let units = tally.derive_units();
        assert_eq!(units, vec!["pSv·cm² · cm / source-particle"]);
    }

    #[test]
    fn test_derive_units_energy_function_filter_without_units() {
        use crate::EnergyFunctionFilter;
        let ef =
            EnergyFunctionFilter::new(vec![1.0, 10.0, 100.0, 1000.0], vec![1.0, 2.0, 3.0, 4.0]);
        let mut tally = Tally::new();
        tally.scores = vec![Score::Flux(FluxScore)];
        tally.filters = vec![Filter::EnergyFunction(ef)];
        let units = tally.derive_units();
        assert_eq!(units, vec!["cm / source-particle"]);
    }

    // ---- Per-nuclide tally tests ----

    #[test]
    fn test_nuclide_bin_default_empty() {
        let tally = Tally::new();
        assert!(tally.nuclides.is_empty());
        assert_eq!(tally.num_nuclide_bins(), 1);
    }

    #[test]
    fn test_nuclide_bin_count() {
        let mut tally = Tally::new();
        tally.nuclides = vec![
            NuclideBin::Specific("Li6".to_string()),
            NuclideBin::Specific("Li7".to_string()),
            NuclideBin::Total,
        ];
        assert_eq!(tally.num_nuclide_bins(), 3);
    }

    #[test]
    fn test_nuclides_iter_empty() {
        let tally = Tally::new();
        let items = tally.nuclides_iter();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].0, 0);
        assert_eq!(items[0].1, NuclideBin::Total);
    }

    #[test]
    fn test_nuclides_iter_with_bins() {
        let mut tally = Tally::new();
        tally.nuclides = vec![NuclideBin::Specific("U235".to_string()), NuclideBin::Total];
        let items = tally.nuclides_iter();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0], (0, NuclideBin::Specific("U235".to_string())));
        assert_eq!(items[1], (1, NuclideBin::Total));
    }

    #[test]
    fn test_5d_indexing() {
        // 2 scores, 3 nuclide bins, 1 parent, 2 energy bins, 1 mesh
        let mut tally = Tally::new();
        tally.scores = vec![Score::Flux(FluxScore), Score::Flux(FluxScore)];
        tally.nuclides = vec![
            NuclideBin::Specific("A".to_string()),
            NuclideBin::Specific("B".to_string()),
            NuclideBin::Total,
        ];
        // No energy filter = 1 energy bin, no mesh = 1 mesh bin
        // Total bins = 2 * 3 * 1 * 1 * 1 = 6
        assert_eq!(tally.num_bins(), 6);

        // score=0, nuc=0 → 0
        assert_eq!(tally.get_bin_index_7d(0, 0, 0, 0, 0, 0, 0), Some(0));
        // score=0, nuc=1 → 1
        assert_eq!(tally.get_bin_index_7d(0, 0, 0, 1, 0, 0, 0), Some(1));
        // score=0, nuc=2 → 2
        assert_eq!(tally.get_bin_index_7d(0, 0, 0, 2, 0, 0, 0), Some(2));
        // score=1, nuc=0 → 3
        assert_eq!(tally.get_bin_index_7d(1, 0, 0, 0, 0, 0, 0), Some(3));
        // score=1, nuc=2 → 5
        assert_eq!(tally.get_bin_index_7d(1, 0, 0, 2, 0, 0, 0), Some(5));
        // Out of range
        assert_eq!(tally.get_bin_index_7d(2, 0, 0, 0, 0, 0, 0), None);
        assert_eq!(tally.get_bin_index_7d(0, 0, 0, 3, 0, 0, 0), None);
    }

    #[test]
    fn test_5d_backward_compat() {
        // When nuclides is empty, 5D indexing with nuclide_bin=0 should
        // produce the same index as the old 4D scheme.
        let mut tally = Tally::new();
        tally.scores = vec![Score::Flux(FluxScore), Score::Flux(FluxScore)];
        // No nuclides → num_nuclide_bins = 1

        // 2 scores × 1 nuclide × 1 parent × 1 energy × 1 mesh = 2 bins
        assert_eq!(tally.num_bins(), 2);
        assert_eq!(tally.get_bin_index_7d(0, 0, 0, 0, 0, 0, 0), Some(0));
        assert_eq!(tally.get_bin_index_7d(1, 0, 0, 0, 0, 0, 0), Some(1));
        // 4D-style call (zero cell/material/nuclide) should give same result
        assert_eq!(tally.get_bin_index_7d(0, 0, 0, 0, 0, 0, 0), Some(0));
        assert_eq!(tally.get_bin_index_7d(1, 0, 0, 0, 0, 0, 0), Some(1));
        // 3D-style call (zero cell/material/nuclide/parent) should give same result
        assert_eq!(tally.get_bin_index_7d(0, 0, 0, 0, 0, 0, 0), Some(0));
        assert_eq!(tally.get_bin_index_7d(1, 0, 0, 0, 0, 0, 0), Some(1));
    }

    #[test]
    fn test_nuclide_bin_display() {
        assert_eq!(NuclideBin::Total.to_string(), "total");
        assert_eq!(NuclideBin::Specific("Li6".to_string()).to_string(), "Li6");
    }

    #[test]
    fn test_num_bins_with_nuclides_and_energy() {
        use crate::EnergyFilter;

        let mut tally = Tally::new();
        tally.scores = vec![Score::Flux(FluxScore)];
        tally.nuclides = vec![
            NuclideBin::Specific("Li6".to_string()),
            NuclideBin::Specific("Li7".to_string()),
        ];
        let ef = EnergyFilter::new(vec![0.0, 1e6, 20e6]);
        tally.filters = vec![Filter::Energy(ef)];
        // 1 score * 2 nuclides * 1 parent * 2 energy * 1 mesh = 4
        assert_eq!(tally.num_bins(), 4);
    }

    #[test]
    fn test_empty_parent_nuclide_filter_collapses_to_one_bin() {
        use crate::{EnergyFilter, ParentNuclideFilter};

        let mut tally = Tally::new();
        tally.scores = vec![Score::Flux(FluxScore)];
        let ef = EnergyFilter::new(vec![0.0, 1e6, 20e6]);
        // An empty parent filter (material with no gamma-emitting activation
        // products) must collapse to one all-zero parent bin, preserving the
        // energy dimension rather than zeroing out the whole tally.
        tally.filters = vec![
            Filter::Energy(ef),
            Filter::ParentNuclide(ParentNuclideFilter::new(vec![])),
        ];
        assert_eq!(tally.num_parent_nuclide_bins(), 1);
        // 1 score * 1 cell * 1 material * 1 nuclide * 1 parent * 2 energy * 1 mesh = 2
        assert_eq!(tally.num_bins(), 2);
    }

    #[test]
    fn test_num_bins_with_multi_cell_filter() {
        use crate::CellFilter;

        let mut tally = Tally::new();
        tally.scores = vec![Score::Flux(FluxScore), Score::Flux(FluxScore)];
        tally.filters = vec![Filter::Cell(CellFilter {
            cell_ids: vec![1, 2, 3],
        })];
        // 2 scores * 3 cells * 1 material * 1 nuclide * 1 parent * 1 energy * 1 mesh = 6
        assert_eq!(tally.num_bins(), 6);
        assert_eq!(tally.num_cell_bins(), 3);
    }

    #[test]
    fn test_num_bins_with_multi_material_filter() {
        use crate::MaterialFilter;

        let mut tally = Tally::new();
        tally.scores = vec![Score::Flux(FluxScore)];
        tally.filters = vec![Filter::Material(MaterialFilter {
            material_ids: vec![10, 20],
        })];
        // 1 score * 1 cell * 2 materials * 1 nuclide * 1 parent * 1 energy * 1 mesh = 2
        assert_eq!(tally.num_bins(), 2);
        assert_eq!(tally.num_material_bins(), 2);
    }

    #[test]
    fn test_7d_indexing_layout() {
        // score → cell → material → nuclide → parent → energy → mesh
        // Choose a shape that makes the stride arithmetic easy to eyeball:
        // 2 scores × 3 cells × 2 materials × 1 × 1 × 1 × 1 = 12 bins.
        use crate::{CellFilter, MaterialFilter};

        let mut tally = Tally::new();
        tally.scores = vec![Score::Flux(FluxScore), Score::Flux(FluxScore)];
        tally.filters = vec![
            Filter::Cell(CellFilter {
                cell_ids: vec![1, 2, 3],
            }),
            Filter::Material(MaterialFilter {
                material_ids: vec![10, 20],
            }),
        ];
        assert_eq!(tally.num_bins(), 12);

        // (score, cell, material, nuclide=0, parent=0, energy=0, mesh=0)
        // stride_mat = 1; stride_cell = 2 * 1 = 2; stride_score = 3 * 2 = 6
        assert_eq!(tally.get_bin_index_7d(0, 0, 0, 0, 0, 0, 0), Some(0));
        assert_eq!(tally.get_bin_index_7d(0, 0, 1, 0, 0, 0, 0), Some(1));
        assert_eq!(tally.get_bin_index_7d(0, 1, 0, 0, 0, 0, 0), Some(2));
        assert_eq!(tally.get_bin_index_7d(0, 2, 1, 0, 0, 0, 0), Some(5));
        assert_eq!(tally.get_bin_index_7d(1, 0, 0, 0, 0, 0, 0), Some(6));
        assert_eq!(tally.get_bin_index_7d(1, 2, 1, 0, 0, 0, 0), Some(11));
        // Out of range
        assert_eq!(tally.get_bin_index_7d(0, 3, 0, 0, 0, 0, 0), None);
        assert_eq!(tally.get_bin_index_7d(0, 0, 2, 0, 0, 0, 0), None);
    }

    #[test]
    fn test_7d_indexing_collapses_when_single_cell_and_material() {
        // When cell_bin and material_bin are 0 (and the filters are 1-bin,
        // or absent), the 7D index must collapse to the plain
        // (score, nuclide) layout: index = score * num_nuclides + nuclide.
        let mut tally = Tally::new();
        tally.scores = vec![Score::Flux(FluxScore), Score::Flux(FluxScore)];
        tally.nuclides = vec![NuclideBin::Specific("A".to_string()), NuclideBin::Total];
        let num_nuclides = 2;
        for score in 0..2 {
            for nuclide in 0..2 {
                assert_eq!(
                    tally.get_bin_index_7d(score, 0, 0, nuclide, 0, 0, 0),
                    Some(score * num_nuclides + nuclide),
                );
            }
        }
    }

    /// A per-nuclide axis cannot apply to `flux`, which has no cross section
    /// to fold it into (issue #305). All three shapes used to be accepted and
    /// silently do nothing.
    #[test]
    fn validate_rejects_a_nuclide_axis_on_flux() {
        // 1. response= (virtual overlay) on flux alone.
        let mut t = Tally::new();
        t.scores = vec![Score::Flux(FluxScore)];
        t.multiply_density = false;
        t.nuclides = vec![NuclideBin::Specific("Fe56".to_string())];
        let err = t.validate().expect_err("flux + response must be refused");
        assert!(err.contains("response="), "{err}");
        assert!(err.contains("flux"), "{err}");

        // 2. response= on a tally that also carries a score the overlay DOES
        // apply to: the array would mix response-weighted heating with plain
        // flux, so the whole tally is refused.
        let mut mixed = Tally::new();
        mixed.scores = vec![Score::Flux(FluxScore), Score::Heating(HeatingScore)];
        mixed.estimator = crate::Estimator::Collision;
        mixed.multiply_density = false;
        mixed.nuclides = vec![NuclideBin::Specific("Fe56".to_string())];
        assert!(
            mixed.validate().is_err(),
            "a mixed-score overlay tally must be refused, not partly applied"
        );

        // 3. nuclides= (the real in-material breakdown) on flux: this used to
        // duplicate the same flux into every nuclide bin.
        let mut per_nuclide = Tally::new();
        per_nuclide.scores = vec![Score::Flux(FluxScore)];
        per_nuclide.nuclides = vec![NuclideBin::Specific("Fe56".to_string()), NuclideBin::Total];
        let err = per_nuclide
            .validate()
            .expect_err("flux + nuclides must be refused");
        assert!(err.contains("nuclides="), "{err}");
    }

    /// The rule is about scores with no cross section, so everything that has
    /// one keeps working, and so does plain flux.
    #[test]
    fn validate_still_accepts_a_nuclide_axis_on_scored_reactions() {
        let mut overlay = Tally::new();
        overlay.scores = vec![Score::Heating(HeatingScore)];
        overlay.estimator = crate::Estimator::Collision;
        overlay.multiply_density = false;
        overlay.nuclides = vec![NuclideBin::Specific("Fe56".to_string())];
        overlay
            .validate()
            .expect("heating + response is the feature");

        let mut breakdown = Tally::new();
        breakdown.scores = vec![Score::ReactionRate(ReactionRateScore {
            mt: Mt::new(102),
            display_name: None,
        })];
        breakdown.nuclides = vec![NuclideBin::Specific("Fe56".to_string())];
        breakdown
            .validate()
            .expect("(n,gamma) + nuclides is the feature");

        let mut plain_flux = Tally::new();
        plain_flux.scores = vec![Score::Flux(FluxScore)];
        plain_flux
            .validate()
            .expect("flux with no nuclide axis is fine");

        // `nuclides=['total']` is a single aggregate bin, not a per-nuclide
        // breakdown, so it stays accepted on flux (one bin, plain flux).
        let mut total_only = Tally::new();
        total_only.scores = vec![Score::Flux(FluxScore)];
        total_only.nuclides = vec![NuclideBin::Total];
        total_only
            .validate()
            .expect("nuclides=['total'] is not a per-nuclide axis");
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn validate_accepts_both_estimators_on_unstructured_mesh() {
        // The collision path resolves the containing tet with yamt's
        // point query (issue #354), so both estimators are valid.
        let mesh = std::sync::Arc::new(
            yamt::MeshGeometry::from_arrow(std::path::Path::new("../yamt/tests/data/cube.arrow"))
                .unwrap(),
        );
        let umf = crate::UnstructuredMeshFilter::new(mesh, 0);

        let mut tally = Tally::new();
        tally.scores = vec![Score::Flux(FluxScore)];
        tally.filters = vec![Filter::UnstructuredMesh(umf)];

        tally.estimator = crate::Estimator::TrackLength;
        assert!(tally.validate().is_ok());
        tally.estimator = crate::Estimator::Collision;
        assert!(tally.validate().is_ok());
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn unstructured_get_bin_resolves_containing_tet() {
        // get_bin agrees with the crossing walk: the tet containing the
        // midpoint of a short segment is among the crossed bins.
        let mesh = std::sync::Arc::new(
            yamt::MeshGeometry::from_arrow(std::path::Path::new("../yamt/tests/data/cube.arrow"))
                .unwrap(),
        );
        let umf = crate::UnstructuredMeshFilter::new(mesh, 0);

        let mid = [0.5, 0.5, 0.5];
        let bin = umf.get_bin(mid).expect("cube centre must be inside");
        assert!(bin < umf.num_bins());
        let crossings = umf.get_bins_crossed([0.45, 0.5, 0.5], [0.55, 0.5, 0.5], [1.0, 0.0, 0.0]);
        assert!(
            crossings.iter().any(|c| c.bin == bin),
            "containing tet {bin} not among crossed bins"
        );
        assert_eq!(umf.get_bin([5.0, 5.0, 5.0]), None, "outside must be None");
    }

    #[test]
    fn overlay_material_serde_roundtrip() {
        use std::collections::BTreeMap;
        let mut tally = Tally::new();
        tally.scores = vec![Score::Heating(HeatingScore)];
        tally.multiply_density = false;
        // Material response: single combined bin backed by per-nuclide densities.
        tally.nuclides = vec![NuclideBin::Total];
        tally.overlay_material = Some(BTreeMap::from([
            ("Fe54".to_string(), 4.9e-3),
            ("Fe56".to_string(), 7.7e-2),
        ]));

        let json = serde_json::to_string(&tally).unwrap();
        assert!(json.contains("overlay_material"));
        assert!(json.contains("Fe56"));

        let restored: Tally = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.overlay_material, tally.overlay_material);
        // The field is part of tally identity (PartialEq routes via to_serde).
        assert_eq!(restored, tally);

        // A unit-density nuclide overlay (no material) is a different tally.
        let mut nuclide_overlay = Tally::new();
        nuclide_overlay.scores = vec![Score::Heating(HeatingScore)];
        nuclide_overlay.multiply_density = false;
        nuclide_overlay.nuclides = vec![NuclideBin::Specific("Fe56".to_string())];
        assert_ne!(nuclide_overlay, tally);
    }

    #[test]
    fn overlay_material_absent_field_is_none() {
        // Tallies that predate the field (no `overlay_material` key) must
        // deserialize as None -- `skip_serializing_if` also keeps it out of
        // the JSON for ordinary tallies.
        let mut tally = Tally::new();
        tally.scores = vec![Score::Flux(FluxScore)];
        let json = serde_json::to_string(&tally).unwrap();
        assert!(!json.contains("overlay_material"));
        let restored: Tally = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.overlay_material, None);
    }
}
