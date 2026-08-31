use crate::interpolate_linear;
use rand::RngExt;
use std::collections::HashMap;
use yamc_element::bremsstrahlung::Bremsstrahlung;
use yamc_element::photon::PhotonInteraction;
use yamc_nuclide::config::CONFIG;
use yamc_nuclide::load_scope::LoadScope;
use yamc_nuclide::nuclide::{get_or_load_nuclide, Nuclide};
use yamc_nuclide::nuclide_registry::NuclideId;

/// Tuple returned from [`Material::sample_collision_data`]:
/// `(distance, nuclide_name, nuclide, urr_random, nuclide_id)`.
/// `nuclide_id` is populated only when D1S setup has run; otherwise `None`.
pub type CollisionData<'a> = (
    f64,
    &'a str,
    &'a Arc<Nuclide>,
    Option<f64>,
    Option<NuclideId>,
);

/// The struck nuclide chosen at a collision, without the (already-sampled)
/// flight distance: `(name, nuclide, nuclide_id)`. Returned by
/// [`Material::select_nuclide_smooth`] and combined with the flight distance
/// into a [`CollisionData`] by the caller.
pub type SelectedNuclide<'a> = (&'a str, &'a Arc<Nuclide>, Option<NuclideId>);

/// Free-flight sample on the smooth (non-URR) PCG path (issue #111): the
/// sampled `distance` plus the energy-grid bracket (`i_grid`, interpolation
/// fraction `f`) and material total `sigma_t` at the collision energy, so the
/// caller can select the struck nuclide *after* the surface-vs-collision min
/// (mirroring the GPU, which draws nuclide-selection only on a real collision).
#[derive(Debug, Clone, Copy)]
pub struct SmoothFlight {
    pub distance: f64,
    pub i_grid: usize,
    pub f: f64,
    pub sigma_t: f64,
    /// URR probability-table band this flight sampled against, when the
    /// material had a URR nuclide in range at this energy. Threaded back onto
    /// the particle so the nuclide selection, the reaction split and the tally
    /// scoring at this collision all reuse the SAME band (issues #204, #342).
    pub urr_random: Option<f64>,
}
use std::sync::Arc;

mod composition;
mod loading;
mod macro_xs;
mod photon;

/// Fast O(1) energy grid lookup using logarithmic binning.
/// Instead of O(log n) binary search, this uses a pre-computed grid that maps
/// log(E) bins to starting indices in the energy grid.
#[derive(Debug, Clone, Default)]
pub struct MaterialFastXS {
    /// Maps a log(E) bin index to a starting index in the energy grid. 32-bit
    /// for the same reason as `FastXSGrid::log_grid_index` (issue #482): the
    /// values are grid indices, and a grid long enough to overflow `u32` would
    /// be 34 GB of `f64`. Built monotone here, so nothing to validate.
    log_grid_index: Vec<u32>,
    /// Log of minimum energy
    log_e_min: f64,
    /// Inverse of log bin width for fast bin calculation
    inv_log_delta: f64,
    /// Number of log bins
    n_log_bins: usize,
    /// Pre-computed total macroscopic XS at each energy point
    total_xs: Vec<f64>,
    /// Number of energy points
    n_energy: usize,
}

impl MaterialFastXS {
    /// Number of logarithmic bins
    const N_LOG_BINS: usize = 8192;

    /// Build the fast lookup structure from energy grid and total XS
    #[allow(clippy::needless_range_loop)]
    pub fn new(energy_grid: &[f64], total_xs: &[f64]) -> Option<Self> {
        let n = energy_grid.len();
        if n < 2 || total_xs.len() != n {
            return None;
        }

        let e_min = energy_grid[0].max(1e-11); // Avoid log(0)
        let e_max = energy_grid[n - 1];
        let log_e_min = e_min.ln();
        let log_e_max = e_max.ln();
        let log_delta = (log_e_max - log_e_min) / Self::N_LOG_BINS as f64;
        let inv_log_delta = 1.0 / log_delta;

        // Build logarithmic grid index
        let mut log_grid_index = vec![0u32; Self::N_LOG_BINS + 1];
        let mut i_grid = 0;
        for i_bin in 0..=Self::N_LOG_BINS {
            let log_e = log_e_min + (i_bin as f64) * log_delta;
            let e = log_e.exp();
            // Find first energy grid point >= e
            while i_grid < n - 1 && energy_grid[i_grid] < e {
                i_grid += 1;
            }
            log_grid_index[i_bin] = i_grid.saturating_sub(1) as u32;
        }

        Some(Self {
            log_grid_index,
            log_e_min,
            inv_log_delta,
            n_log_bins: Self::N_LOG_BINS,
            total_xs: total_xs.to_vec(),
            n_energy: n,
        })
    }

    /// Fast O(1) lookup of total macroscopic cross-section at given energy.
    /// Returns (sigma_t, grid_index, interpolation_factor) for reuse.
    #[inline]
    pub fn lookup_total(&self, energy: f64, energy_grid: &[f64]) -> (f64, usize, f64) {
        // Handle boundary cases
        if energy <= energy_grid[0] {
            return (self.total_xs[0], 0, 0.0);
        }
        if energy >= energy_grid[self.n_energy - 1] {
            return (self.total_xs[self.n_energy - 1], self.n_energy - 1, 0.0);
        }

        // Use logarithmic grid for O(1) bin lookup
        let log_e = energy.ln();
        let bin = ((log_e - self.log_e_min) * self.inv_log_delta) as usize;
        let bin = bin.min(self.n_log_bins - 1);

        // Get narrow search range from log grid
        let i_low = self.log_grid_index[bin] as usize;
        let i_high = (self.log_grid_index[bin + 1] as usize + 1).min(self.n_energy);

        // Short linear search in narrow range (typically 1-3 elements)
        let mut i_grid = i_low;
        while i_grid < i_high - 1 && energy_grid[i_grid + 1] <= energy {
            i_grid += 1;
        }
        let i_grid = i_grid.min(self.n_energy - 2);

        // Linear interpolation
        let e0 = energy_grid[i_grid];
        let e1 = energy_grid[i_grid + 1];
        let f = (energy - e0) / (e1 - e0);
        let sigma_t =
            self.total_xs[i_grid] + f * (self.total_xs[i_grid + 1] - self.total_xs[i_grid]);

        (sigma_t, i_grid, f)
    }
}

// Cached microscopic cross sections: (MT filter, nuclide -> MT -> xs_values on unified grid).
type MicroscopicXsCache = (Vec<i32>, Arc<HashMap<String, HashMap<i32, Vec<f64>>>>);

/// How a material's density value is interpreted.
///
/// Parsed once in [`Material::new`] from the user-supplied unit string so the
/// rest of the codebase can match on a typed value instead of re-parsing
/// strings. The serde wire format keeps the canonical strings via
/// [`DensityUnits::as_str`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DensityUnits {
    /// Grams per cubic centimetre. Covers both `"g/cm3"` and `"g/cc"`
    /// (aliases, identity factor). Canonical string is `"g/cm3"`. The
    /// composition values are relative fractions.
    GramsPerCc,
    /// Kilograms per cubic metre (`"kg/m3"`); divide by 1000 to reach g/cm³.
    /// The composition values are relative fractions.
    KgPerM3,
    /// Total atom density mode (`"atom/barn-cm"`): the density value is the
    /// total atom density in atoms/barn-cm and the composition values are
    /// relative atom fractions (requires `fraction_type = "atom"`). Per-nuclide
    /// atom densities are `N_i = (x_i / sum x) * total`.
    AtomPerBarnCm,
    /// Raw per-nuclide atom-density mode (`"sum"`): the composition values are
    /// themselves absolute atom densities (atoms/barn-cm). Not part of the
    /// public `Material(...)` surface -- it is the canonical representation the
    /// transmutation stepper stores via [`Material::from_nuclide_densities`] /
    /// the `from_atom_densities` constructor, kept here for serde round-trips.
    Sum,
}

impl DensityUnits {
    /// Parse a user-supplied density unit string into a [`DensityUnits`].
    ///
    /// `"g/cm3"` and `"g/cc"` are accepted aliases for [`DensityUnits::GramsPerCc`].
    /// `"sum"` is accepted for serde round-trips of transmuted materials but is
    /// not a user-facing `Material(...)` unit. Returns an error on an
    /// unsupported unit.
    pub fn parse(input: &str) -> Result<Self, String> {
        match input {
            "g/cm3" | "g/cc" => Ok(DensityUnits::GramsPerCc),
            "kg/m3" => Ok(DensityUnits::KgPerM3),
            "atom/barn-cm" => Ok(DensityUnits::AtomPerBarnCm),
            "sum" => Ok(DensityUnits::Sum),
            other => Err(format!("Unsupported density unit: '{other}'")),
        }
    }

    /// Canonical string for this unit, used for serialization and display.
    pub fn as_str(&self) -> &'static str {
        match self {
            DensityUnits::GramsPerCc => "g/cm3",
            DensityUnits::KgPerM3 => "kg/m3",
            DensityUnits::AtomPerBarnCm => "atom/barn-cm",
            DensityUnits::Sum => "sum",
        }
    }
}

/// Whether a material's nuclide fractions are atom or mass fractions.
///
/// Parsed once in [`Material::new`]. `"volume"` is intentionally **not** a
/// variant here: it is only a `mix_materials` input, never a stored fraction
/// type. The serde wire format keeps the canonical strings via
/// [`FractionType::as_str`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FractionType {
    /// Atom fractions (`"atom"`).
    Atom,
    /// Mass fractions (`"mass"`).
    Mass,
}

impl FractionType {
    /// Parse a user-supplied fraction-type string into a [`FractionType`].
    ///
    /// Exactly one accepted spelling per variant (`"atom"` / `"mass"`);
    /// anything else (including the old `"weight"`) fails hard.
    pub fn parse(input: &str) -> Result<Self, String> {
        match input {
            "atom" => Ok(FractionType::Atom),
            "mass" => Ok(FractionType::Mass),
            other => Err(format!("fraction must be 'atom' or 'mass', got '{other}'")),
        }
    }

    /// Canonical string for this fraction type, used for serialization.
    pub fn as_str(&self) -> &'static str {
        match self {
            FractionType::Atom => "atom",
            FractionType::Mass => "mass",
        }
    }
}

/// Represents a heterogeneous collection of nuclides (or elements expanded to
/// their naturally abundant isotopes) along with the material density and
/// nuclear data needed for transport / analysis.
///
/// A `Material` is built complete: [`Material::new`] takes the composition,
/// the fraction type and the density together, and there is no incremental
/// `add_nuclide`. Elements and chemical formulas are expanded to isotopes by
/// natural abundance in [`yamc_nuclide::composition`] before they get here.
/// Temperature (a label, via [`Material::set_temperature`]) and volume are set
/// afterwards. On demand the structure loads Arrow nuclide data (through the
/// global [`yamc_nuclide::config::Config`]) and builds a unified energy grid for
/// neutrons so reaction cross sections for different nuclides can be
/// interpolated on a common axis.
///
/// Key cached members:
/// * `unified_energy_grid_neutron` – lazily constructed common energy grid.
/// * `macroscopic_xs_neutron` – map MT -> Σ(E) on the unified grid.
/// * `macroscopic_xs_neutron_total_by_nuclide` – optional per‑nuclide Σ_t(E) when
///   requested (used for sampling interacting nuclides).
///
/// Typical workflow:
/// 1. Create with [`Material::new`].
/// 2. Populate composition (nuclides or elements) and set density.
/// 3. Optionally set temperature (clears caches if changed).
/// 4. Call cross section building methods (e.g.
///    [`Material::calculate_macroscopic_xs`]) or sampling utilities.
///
/// Serialization goes via [`MaterialSerde`] -- only the user-supplied
/// fields persist. Loaded nuclide data and macroscopic XS caches are
/// rebuilt on demand via `read_nuclear_data` / `calculate_macroscopic_xs`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(into = "MaterialSerde", try_from = "MaterialSerde")]
pub struct Material {
    /// Optional name of the material
    pub name: Option<String>,
    /// Unique identifier for the material
    pub material_id: Option<u32>,
    /// Composition of the material as a map of nuclide names to their fractions.
    /// Interpretation depends on `fraction_type`: atom fractions ("atom") or mass fractions ("mass").
    pub nuclides: HashMap<String, f64>,
    /// User-input order of nuclides, when known (set by the Python binding).
    /// Each name appears at most once; expanded entries (e.g. from elements or formulas)
    /// are ordered alphabetically within their parent input key. None when not tracked
    /// (e.g. constructed directly from the Rust API).
    pub nuclide_input_order: Option<Vec<String>>,
    /// Whether nuclide fractions are atom or weight fractions. Default [`FractionType::Atom`].
    /// Must not be changed once nuclides have been added.
    pub fraction_type: FractionType,
    /// Density of the material in g/cm³
    pub density: Option<f64>,
    /// Density unit (default: [`DensityUnits::GramsPerCc`])
    pub density_units: DensityUnits,
    /// Volume of the material in cm³
    pub volume: Option<f64>,
    /// Temperature label, the key that selects a cross-section set (e.g. `"294"`).
    ///
    /// Private, and `temperature_k` with it, because the two must agree and
    /// nothing kept them in step when they were public: `temperature_k` was set
    /// once in `Material::new` and never again, so the CPU free-gas kernel ran
    /// every material at 294 K (issue #478). Go through
    /// [`Material::set_temperature`], which maintains both, or
    /// [`Material::temperature`] to read.
    ///
    /// Empty means unresolved. [`Material::resolve_temperature`] fills it in
    /// from the loaded nuclide data.
    temperature: String,
    /// `temperature` in Kelvin, cached for the collision hot path.
    ///
    /// Always parsed from `temperature` by
    /// [`yamc_nuclide::temperature::label_to_kelvin_or_default`], the same
    /// function the GPU extractor uses, so the two backends cannot disagree.
    temperature_k: f64,
    /// Loaded nuclide data (name -> `Arc<Nuclide>`) shared for this material instance
    pub nuclide_data: HashMap<String, Arc<Nuclide>>,
    /// Macroscopic cross sections for different MT numbers (neutron only for now)
    /// Map of MT number (i32) -> cross sections
    pub macroscopic_xs_neutron: HashMap<i32, Vec<f64>>,
    /// Unified energy grid for neutrons
    pub unified_energy_grid_neutron: Vec<f64>,
    /// Per-nuclide macroscopic total cross section (MT=1) on the unified grid.
    /// Indexed parallel to `sorted_nuclide_keys` (dense Vec, no String lookup).
    pub macroscopic_xs_neutron_total_by_nuclide: Option<Vec<Vec<f64>>>,
    /// Per-nuclide macroscopic cross sections for ALL requested MTs.
    /// Indexed parallel to `sorted_nuclide_keys`; inner HashMap is MT -> `Vec<f64>`.
    /// Populated when any tally uses per-nuclide scoring (`tally.nuclides`).
    pub macroscopic_xs_neutron_by_nuclide: Option<Vec<HashMap<i32, Vec<f64>>>>,
    /// Fast O(1) energy grid lookup structure (built after calculate_macroscopic_xs)
    pub fast_xs: Option<MaterialFastXS>,
    /// Cached atom densities (atoms/barn-cm) per nuclide, populated during calculate_macroscopic_xs
    pub cached_atoms_per_barn_cm: Option<HashMap<String, f64>>,
    /// Marks this material for the coupled transport-plus-transmutation driver.
    ///
    /// Only `Model::transmute` reads it, via `find_transmutable_cells`: in a
    /// geometry full of materials it is how you say which ones to deplete.
    /// `transmute_material` (the transport-free path behind
    /// `Material.transmute()`) does not consult it, because there is exactly one
    /// material in play and the caller has already named it.
    pub transmutable: bool,
    /// Cached microscopic XS on the unified grid (nuclide -> MT -> xs_values).
    /// Wrapped in Arc for cheap cloning. Valid while nuclide_data + temperature unchanged.
    /// The `Vec<i32>` stores which MTs were cached (for filter validation).
    pub cached_microscopic_xs: Option<MicroscopicXsCache>,
    /// Photon element indices: for each nuclide (sorted by name), the index of its
    /// element in the global photon element store. Parallel to sorted nuclide keys.
    /// Populated by `init_photon_data()` when `transport_secondary_photons` is enabled.
    pub element_indices: Vec<(String, usize)>,
    /// Cached Arc references to photon elements, parallel to `element_indices`.
    /// Populated once during `init_photon_data()` to avoid RwLock + Arc clone
    /// on every `get_element_by_index()` call in the hot path.
    pub cached_elements: Vec<(String, Arc<yamc_element::photon::PhotonInteraction>)>,
    /// Atom densities (atoms/barn-cm) aligned 1:1 with `cached_elements`.
    /// Populated after `calculate_macroscopic_xs` + `init_photon_data` so the
    /// photon hot path avoids a `HashMap<String, _>` lookup per collision.
    pub cached_element_atom_densities: Vec<f64>,
    /// Paths to photon interaction Arrow data, keyed by element symbol.
    /// Example: {"Fe" => "path/to/Fe.arrow", "O" => "path/to/O.arrow"}
    /// Set via `read_nuclear_data()` when photon data is provided.
    pub photon_data_paths: HashMap<String, String>,
    /// Thick-target bremsstrahlung data (electron and positron).
    /// Populated by `init_bremsstrahlung()` when TTB electron treatment is enabled.
    pub ttb: Option<Bremsstrahlung>,
    /// Cached sorted nuclide names for deterministic iteration over
    /// `macroscopic_xs_neutron_total_by_nuclide`. Populated alongside that map
    /// in `calculate_macroscopic_xs` so the hot path avoids re-sorting every collision.
    pub sorted_nuclide_keys: Option<Vec<String>>,
    /// NuclideIds aligned 1:1 with `sorted_nuclide_keys`. Populated at D1S setup
    /// so the hot path can read the interned id without a HashMap<String, _> lookup.
    /// `None` when D1S is not used; shape matches `sorted_nuclide_keys` when set.
    pub sorted_nuclide_ids: Option<Vec<NuclideId>>,
    /// `Arc<Nuclide>` handles aligned 1:1 with `sorted_nuclide_keys`. Populated
    /// alongside the keys in `calculate_macroscopic_xs`. Lets `sample_collision_data`
    /// index directly into this Vec instead of doing a `HashMap<String, _>.get(name)`
    /// on every collision.
    pub sorted_nuclides: Option<Vec<Arc<Nuclide>>>,
    /// Sorted MT numbers for macroscopic neutron XS (companion to flat buffer).
    /// Parallel to the column dimension of `macroscopic_xs_flat`.
    pub macroscopic_xs_mt_numbers: Vec<i32>,
    /// Flat row-major buffer: `[n_energies × n_mts]` macroscopic neutron XS.
    /// Access XS at (energy index i, MT index j) via
    /// `macroscopic_xs_flat[i * n_mts + j]`.
    pub macroscopic_xs_flat: Vec<f64>,
}

/// Macroscopic photon cross sections computed per-collision.
#[derive(Debug, Clone, Default)]
pub struct MacroPhotonXS {
    pub total: f64,
    pub coherent: f64,
    pub incoherent: f64,
    pub photoelectric: f64,
    pub pair_production: f64,
    pub heating: f64,
}

/// URR-modified macroscopic cross-sections for tally scoring.
/// Contains the key MTs affected by URR probability tables.
pub struct UrrMacroXs {
    pub total: f64,      // MT 1
    pub elastic: f64,    // MT 2
    pub fission: f64,    // MT 18
    pub capture: f64,    // MT 102 (n,gamma)
    pub absorption: f64, // MT 27 (= capture + fission)
}

/// Result of sampling the URR probability table for a single nuclide at one energy.
///
/// Produced by [`Material::urr_sample_for_nuclide`]; bundles the smooth microscopic
/// values the callers still need alongside the URR-sampled microscopic cross-sections.
struct UrrNuclideSample {
    /// Smooth microscopic total from `fast_xs.lookup` (used to scale the macroscopic total).
    xs_total: f64,
    /// Smooth microscopic inelastic = (scattering - elastic).max(0); unmodified by URR.
    xs_inelastic: f64,
    /// `true` when this nuclide's URR table includes an inelastic contribution.
    inelastic_in_table: bool,
    /// URR-sampled microscopic total.
    urr_total: f64,
    /// URR-sampled microscopic elastic.
    urr_elastic: f64,
    /// URR-sampled microscopic capture (n,gamma).
    urr_capture: f64,
    /// URR-sampled microscopic fission.
    urr_fission: f64,
}

/// On-disk shape of [`Material`]. Only the user-supplied configuration --
/// loaded nuclide data, macroscopic XS caches, photon-interaction handles,
/// etc. are rebuilt by `read_nuclear_data` / `calculate_macroscopic_xs`
/// after deserialize.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct MaterialSerde {
    pub name: Option<String>,
    pub material_id: Option<u32>,
    pub nuclides: HashMap<String, f64>,
    pub nuclide_input_order: Option<Vec<String>>,
    pub fraction_type: String,
    pub density: Option<f64>,
    pub density_units: String,
    pub volume: Option<f64>,
    pub temperature: String,
    pub transmutable: bool,
}

impl From<Material> for MaterialSerde {
    fn from(m: Material) -> Self {
        Self {
            name: m.name,
            material_id: m.material_id,
            nuclides: m.nuclides,
            nuclide_input_order: m.nuclide_input_order,
            fraction_type: m.fraction_type.as_str().to_string(),
            density: m.density,
            density_units: m.density_units.as_str().to_string(),
            volume: m.volume,
            temperature: m.temperature,
            transmutable: m.transmutable,
        }
    }
}

impl TryFrom<MaterialSerde> for Material {
    type Error = String;
    fn try_from(spec: MaterialSerde) -> Result<Self, Self::Error> {
        let mut m = Material::new(
            spec.nuclides,
            &spec.fraction_type,
            &spec.density_units,
            spec.density,
        )?;
        m.name = spec.name;
        if let Some(id) = spec.material_id {
            m.set_material_id(id);
        }
        m.nuclide_input_order = spec.nuclide_input_order;
        m.volume = spec.volume;
        m.set_temperature(&spec.temperature);
        m.transmutable = spec.transmutable;
        Ok(m)
    }
}

impl Material {
    /// Create a new material with composition and density specified upfront.
    ///
    /// # Arguments
    /// * `nuclides` - Map of nuclide names to fractions (atom or mass, per `fraction_type`)
    /// * `fraction_type` - `"atom"` for atom fractions or `"mass"` for mass fractions
    /// * `density_unit` - `"g/cm3"`, `"g/cc"`, `"kg/m3"`, or `"sum"` (absolute atom densities)
    /// * `density_value` - Density value (required for g/cm3 etc., optional for "sum")
    pub fn new(
        nuclides: HashMap<String, f64>,
        fraction_type: &str,
        density_unit: &str,
        density_value: Option<f64>,
    ) -> Result<Material, String> {
        let fraction_type = FractionType::parse(fraction_type)?;

        // Validate each nuclide name and fraction
        for (name, &fraction) in &nuclides {
            yamc_nuclide::composition::validate_nuclide_name(name)?;
            if fraction < 0.0 {
                return Err(format!("Fraction for nuclide '{name}' cannot be negative"));
            }
        }

        // Validate density
        let density_units = DensityUnits::parse(density_unit)?;
        let density = match density_units {
            DensityUnits::Sum => {
                if let Some(v) = density_value {
                    if v <= 0.0 {
                        return Err("Density must be positive".into());
                    }
                }
                density_value
            }
            DensityUnits::AtomPerBarnCm => {
                // Total atom density + relative atom fractions; mass fractions
                // are incoherent with an absolute atom-density total.
                if fraction_type == FractionType::Mass {
                    return Err(
                        "units='atom/barn-cm' requires fraction='atom' (a total atom \
                         density is incoherent with mass fractions)"
                            .into(),
                    );
                }
                let v = density_value.ok_or_else(|| {
                    format!("Density value is required for unit '{density_unit}'")
                })?;
                if v <= 0.0 {
                    return Err("Density must be positive".into());
                }
                Some(v)
            }
            DensityUnits::GramsPerCc | DensityUnits::KgPerM3 => {
                let v = density_value.ok_or_else(|| {
                    format!("Density value is required for unit '{density_unit}'")
                })?;
                if v <= 0.0 {
                    return Err("Density must be positive".into());
                }
                Some(v)
            }
        };

        Ok(Material {
            name: None,
            material_id: None,
            nuclides,
            nuclide_input_order: None,
            fraction_type,
            density,
            density_units,
            volume: None,
            temperature: String::new(),
            temperature_k: yamc_nuclide::temperature::DEFAULT_TEMPERATURE_K,
            nuclide_data: HashMap::new(),
            macroscopic_xs_neutron: HashMap::new(),
            unified_energy_grid_neutron: Vec::new(),
            macroscopic_xs_neutron_total_by_nuclide: None,
            macroscopic_xs_neutron_by_nuclide: None,
            fast_xs: None,
            cached_atoms_per_barn_cm: None,
            transmutable: false,
            cached_microscopic_xs: None,
            element_indices: Vec::new(),
            cached_elements: Vec::new(),
            cached_element_atom_densities: Vec::new(),
            photon_data_paths: HashMap::new(),
            ttb: None,
            sorted_nuclide_keys: None,
            sorted_nuclide_ids: None,
            sorted_nuclides: None,
            macroscopic_xs_mt_numbers: Vec::new(),
            macroscopic_xs_flat: Vec::new(),
        })
    }

    /// Create a material from absolute nuclide densities, inheriting metadata
    /// from a template material. Used for transmutation where compositions change
    /// but material identity (name, ID, temperature, etc.) should be preserved.
    ///
    /// The resulting material is in "sum" mode (nuclide values are atoms/barn-cm).
    ///
    /// Carries the template's loaded `nuclide_data` forward, so the result can
    /// still compute cross sections. For a transmutation RESULT that is not
    /// wanted, since it pins the whole loaded library for the lifetime of the
    /// returned list; use [`Material::from_nuclide_densities_without_data`].
    pub fn from_nuclide_densities(
        nuclide_densities: HashMap<String, f64>,
        template: &Material,
    ) -> Material {
        let nuclide_data = template.nuclide_data.clone();
        Self::from_nuclide_densities_with(nuclide_densities, template, nuclide_data)
    }

    /// The shared body of the two constructors above, taking the
    /// `nuclide_data` map to install rather than always cloning the
    /// template's.
    ///
    /// `from_nuclide_densities_without_data` used to clone that map and then
    /// throw it away, which on a transmutation step is 556 `Arc` increments and
    /// 556 decrements, on refcounts every thread shares -- once per step per
    /// replica, to produce an empty map (issue #576, finding 6).
    fn from_nuclide_densities_with(
        nuclide_densities: HashMap<String, f64>,
        template: &Material,
        nuclide_data: HashMap<String, Arc<Nuclide>>,
    ) -> Material {
        Material {
            name: template.name.clone(),
            material_id: template.material_id,
            nuclides: nuclide_densities,
            nuclide_input_order: template.nuclide_input_order.clone(),
            fraction_type: FractionType::Atom,
            density: None,
            density_units: DensityUnits::Sum,
            volume: template.volume,
            temperature: template.temperature.clone(),
            temperature_k: template.temperature_k,
            nuclide_data,
            macroscopic_xs_neutron: HashMap::new(),
            unified_energy_grid_neutron: Vec::new(),
            macroscopic_xs_neutron_total_by_nuclide: None,
            macroscopic_xs_neutron_by_nuclide: None,
            fast_xs: None,
            cached_atoms_per_barn_cm: None,
            transmutable: template.transmutable,
            cached_microscopic_xs: None,
            element_indices: Vec::new(),
            cached_elements: Vec::new(),
            cached_element_atom_densities: Vec::new(),
            photon_data_paths: template.photon_data_paths.clone(),
            ttb: None,
            sorted_nuclide_keys: None,
            sorted_nuclide_ids: None,
            sorted_nuclides: None,
            macroscopic_xs_mt_numbers: Vec::new(),
            macroscopic_xs_flat: Vec::new(),
        }
    }

    /// As [`Material::from_nuclide_densities`], but WITHOUT carrying the
    /// template's loaded `nuclide_data` forward.
    ///
    /// The transmutation stepper returns one material per timestep. Cloning
    /// `nuclide_data` into each of them hands every step a strong reference to
    /// the whole loaded cross-section set, and since `GLOBAL_NUCLIDE_CACHE`
    /// holds only `Weak` refs the caller's result list then becomes the sole
    /// owner and pins it for as long as the results are held. The pointwise data
    /// is dead by then: the per-spectrum collapse has already reduced it to
    /// scalar reaction rates, and the stepper reads only densities, the chain,
    /// those rates and the fission-yield weights.
    pub fn from_nuclide_densities_without_data(
        nuclide_densities: HashMap<String, f64>,
        template: &Material,
    ) -> Material {
        Self::from_nuclide_densities_with(nuclide_densities, template, HashMap::new())
    }

    /// Set the name of the material
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = Some(name.into());
    }

    /// Populate `sorted_nuclide_ids` by interning each `sorted_nuclide_keys`
    /// entry into the supplied registry. Called once at D1S setup time so the
    /// hot path can read interned ids without a `HashMap<String, _>` lookup.
    /// No-op if `sorted_nuclide_keys` is not set.
    pub fn populate_sorted_nuclide_ids(
        &mut self,
        registry: &mut yamc_nuclide::nuclide_registry::NuclideRegistry,
    ) {
        let Some(ref keys) = self.sorted_nuclide_keys else {
            self.sorted_nuclide_ids = None;
            return;
        };
        let ids: Vec<NuclideId> = keys.iter().map(|k| registry.intern(k)).collect();
        self.sorted_nuclide_ids = Some(ids);
    }

    /// Get the name of the material
    pub fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Set the material ID
    pub fn set_material_id(&mut self, material_id: u32) {
        self.material_id = Some(material_id);
    }

    /// Get the material ID
    #[inline]
    pub fn get_material_id(&self) -> Option<u32> {
        self.material_id
    }

    /// Set whether this material is transmutable
    pub fn set_transmutable(&mut self, transmutable: bool) {
        self.transmutable = transmutable;
    }

    /// Get whether this material is transmutable
    pub fn get_transmutable(&self) -> bool {
        self.transmutable
    }

    /// Check if material is ready for transmutation.
    ///
    /// Validates that the material has required fields set for transmutation:
    /// - Must be marked transmutable
    /// - Must have at least one nuclide
    /// - Must have volume specified for tally normalization
    pub fn validate_for_transmutation(&self) -> Result<(), String> {
        if !self.transmutable {
            return Err("Material not marked as transmutable".into());
        }
        if self.nuclides.is_empty() {
            return Err("Material has no nuclides".into());
        }
        if self.volume.is_none() {
            return Err(
                "Transmutable material must have volume specified for tally normalization".into(),
            );
        }
        Ok(())
    }

    /// Drop the loaded nuclear data, and everything derived from it.
    ///
    /// `Material::transmute` loads the cross sections it needs into the
    /// material and leaves them there, so a second call on the same material
    /// does no Arrow decoding at all (issue #576, finding 3). On a chain like
    /// ENDF/B-8.1 that is a few hundred nuclides, which is the right trade for
    /// a material transmuted more than once and the wrong one for a sweep over
    /// thousands of DISTINCT compositions -- so the release is explicit rather
    /// than automatic.
    ///
    /// The material is fully usable afterwards: the next call that needs the
    /// data loads it again.
    pub fn release_nuclear_data(&mut self) {
        self.nuclide_data.clear();
        self.invalidate_xs_cache();
    }

    /// Clear all cached cross section data
    pub fn invalidate_xs_cache(&mut self) {
        self.macroscopic_xs_neutron.clear();
        self.macroscopic_xs_neutron_total_by_nuclide = None;
        self.macroscopic_xs_neutron_by_nuclide = None;
        self.sorted_nuclide_keys = None;
        self.sorted_nuclides = None;
        self.unified_energy_grid_neutron.clear();
        self.fast_xs = None;
        self.cached_atoms_per_barn_cm = None;
        self.cached_element_atom_densities.clear();
        self.cached_microscopic_xs = None;
        self.element_indices.clear();
        self.macroscopic_xs_mt_numbers.clear();
        self.macroscopic_xs_flat.clear();
    }

    /// Clear exactly the caches derived from the temperature.
    ///
    /// Narrower than [`Material::invalidate_xs_cache`], which also drops
    /// `cached_atoms_per_barn_cm`, `cached_elements`, `element_indices` and
    /// `cached_element_atom_densities`. None of those four depend on
    /// temperature: the first three come from composition, density and the
    /// photon element data, and dropping them here is not merely wasteful. Only
    /// `init_photon_data` rebuilds `element_indices`, and `cached_elements` is
    /// documented as parallel to it, so clearing one and not the other leaves
    /// `calculate_photon_xs` indexing `cached_element_atom_densities[idx]` for
    /// `idx` in `0..cached_elements.len()` past the end of an emptied vector --
    /// guarded only by a `debug_assert_eq!`, so a release build panics (#481).
    pub fn invalidate_temperature_caches(&mut self) {
        self.macroscopic_xs_neutron.clear();
        self.macroscopic_xs_neutron_total_by_nuclide = None;
        self.macroscopic_xs_neutron_by_nuclide = None;
        self.sorted_nuclide_keys = None;
        self.sorted_nuclides = None;
        self.unified_energy_grid_neutron.clear();
        self.fast_xs = None;
        self.cached_microscopic_xs = None;
        self.macroscopic_xs_mt_numbers.clear();
        self.macroscopic_xs_flat.clear();
    }

    /// Invalidate only density-dependent caches (macroscopic XS, fast lookup).
    /// Preserves unified energy grid and microscopic XS cache.
    pub fn invalidate_density_caches(&mut self) {
        self.macroscopic_xs_neutron.clear();
        self.macroscopic_xs_neutron_total_by_nuclide = None;
        self.macroscopic_xs_neutron_by_nuclide = None;
        self.sorted_nuclide_keys = None;
        self.sorted_nuclides = None;
        self.fast_xs = None;
        self.cached_atoms_per_barn_cm = None;
        self.cached_element_atom_densities.clear();
        self.macroscopic_xs_mt_numbers.clear();
        self.macroscopic_xs_flat.clear();
        // Keep: unified_energy_grid_neutron, cached_microscopic_xs
    }

    /// Sample the distance to the next collision for a neutron at the given energy.
    /// Uses the total macroscopic cross section (MT=1).
    /// Returns None if the cross section is zero or not available.
    #[inline]
    pub fn sample_distance_to_collision<R: rand::Rng + ?Sized>(
        &self,
        energy: f64,
        rng: &mut R,
    ) -> Option<f64> {
        // Use fast O(1) lookup if available
        let sigma_t = if let Some(ref fast_xs) = self.fast_xs {
            let (sigma, _, _) = fast_xs.lookup_total(energy, &self.unified_energy_grid_neutron);
            sigma
        } else {
            // Fallback to slow O(log n) lookup
            let xs_vec = self.macroscopic_xs_neutron.get(&1).unwrap_or_else(|| {
                panic!("sample_distance_to_collision: macroscopic_xs_neutron[1] missing. Did you call calculate_macroscopic_xs?");
            });
            if self.unified_energy_grid_neutron.is_empty() || xs_vec.is_empty() {
                panic!("sample_distance_to_collision: energy grid or cross section vector is empty. Did you call calculate_macroscopic_xs?");
            }
            crate::interpolate_linear(&self.unified_energy_grid_neutron, xs_vec, energy)
        };

        if sigma_t <= 0.0 {
            panic!("sample_distance_to_collision: total cross section is zero or negative at energy {energy}. Check your nuclear data and energy grid.");
        }
        let xi: f64 = rng.random_range(0.0..1.0);
        Some(-xi.ln() / sigma_t)
    }

    /// Combined sampling of distance to collision AND interacting nuclide in one energy lookup.
    /// More efficient than calling `sample_distance_to_collision` and
    /// `sample_interacting_nuclide_data` separately because it reuses the O(1) energy grid index.
    ///
    /// Returns (distance, nuclide_name, nuclide_data, urr_random, nuclide_id) or None.
    ///
    /// URR semantics: when any nuclide in the material has URR probability-table data covering
    /// `energy`, a single per-collision base seed (`urr_random`) is drawn and **each in-range
    /// URR nuclide derives an independent probability-table band from it** (issue #204), so a
    /// multi-isotope material samples statistically independent resonance structure per isotope.
    /// The URR-modified Σ_t is then used for both distance sampling and nuclide selection.
    /// If `cached_urr_random` is `Some`, the base seed is reused for consistency across XS
    /// lookups at the same energy (the struck nuclide's reaction re-derives the same band);
    /// otherwise a new draw is taken from `rng`. When no URR is in range, no random is consumed
    /// and the smooth fast path runs.
    #[inline]
    pub fn sample_collision_data<R: rand::Rng + ?Sized>(
        &self,
        energy: f64,
        cached_urr_random: Option<f64>,
        rng: &mut R,
    ) -> Option<CollisionData<'_>> {
        let Some(ref fast_xs) = self.fast_xs else {
            // Slow fallback path: no per-nuclide macroscopic tables yet → no URR, no D1S id.
            let distance = self.sample_distance_to_collision(energy, rng)?;
            let (name, nuclide) = self.sample_interacting_nuclide_data(energy, rng)?;
            return Some((distance, name, nuclide, None, None));
        };

        let (sigma_t_smooth, i_grid, f) =
            fast_xs.lookup_total(energy, &self.unified_energy_grid_neutron);
        if sigma_t_smooth <= 0.0 {
            return None;
        }

        let by_nuclide = self.macroscopic_xs_neutron_total_by_nuclide.as_ref()?;
        let sorted_keys = self
            .sorted_nuclide_keys
            .as_ref()
            .expect("sorted_nuclide_keys must be set when by_nuclide is enabled");
        let sorted_ids = self.sorted_nuclide_ids.as_deref();
        let sorted_nuclides = self.sorted_nuclides.as_deref();

        // Helper: lookup smooth per-nuclide macroscopic total at this grid index.
        let smooth_nuc_xs = |idx: usize| -> f64 {
            let xs_vec = &by_nuclide[idx];
            if xs_vec.len() > i_grid + 1 {
                let xs0 = xs_vec[i_grid];
                let xs1 = xs_vec[i_grid + 1];
                xs0 + f * (xs1 - xs0)
            } else {
                0.0
            }
        };

        // Resolve the Nuclide for a given index.
        let lookup_nuclide = |idx: usize, name: &str| -> Option<&Arc<Nuclide>> {
            sorted_nuclides
                .and_then(|v| v.get(idx))
                .or_else(|| self.nuclide_data.get(name))
        };

        if self.has_urr_in_range(energy) {
            // URR path: one per-collision base seed, from which each in-range URR
            // nuclide derives an *independent* probability-table band (issue #204,
            // see `nuclide_macro_total_with_urr` -> `urr_sample_for_nuclide`).
            // Pass 1 computes URR-modified per-nuclide contributions and Σ_t_urr;
            // Pass 2 samples the struck nuclide using those URR-modified contributions.
            let urr_random = cached_urr_random.unwrap_or_else(|| rng.random::<f64>());

            let n = sorted_keys.len();
            let mut per_nuclide_urr: Vec<f64> = Vec::with_capacity(n);
            let mut sigma_t_urr = 0.0;
            for (idx, name) in sorted_keys.iter().enumerate() {
                let xs_smooth = smooth_nuc_xs(idx);
                let xs_urr = match lookup_nuclide(idx, name) {
                    Some(nuc) => {
                        self.nuclide_macro_total_with_urr(nuc, xs_smooth, energy, urr_random)
                    }
                    None => xs_smooth,
                };
                per_nuclide_urr.push(xs_urr);
                sigma_t_urr += xs_urr;
            }

            if sigma_t_urr <= 0.0 {
                return None;
            }

            let xi: f64 = rng.random_range(0.0..1.0);
            let distance = -xi.ln() / sigma_t_urr;

            let xi_nuclide = rng.random_range(0.0..sigma_t_urr);
            let mut accum = 0.0;
            for (idx, &xs_urr) in per_nuclide_urr.iter().enumerate() {
                accum += xs_urr;
                if xi_nuclide < accum {
                    let name = &sorted_keys[idx];
                    let nuclide = lookup_nuclide(idx, name)?;
                    let nuc_id = sorted_ids.and_then(|ids| ids.get(idx).copied());
                    return Some((distance, name.as_str(), nuclide, Some(urr_random), nuc_id));
                }
            }
            return None;
        }

        // Smooth path: no URR-in-range nuclides → existing fast logic, no random consumed for URR.
        let xi: f64 = rng.random_range(0.0..1.0);
        let distance = -xi.ln() / sigma_t_smooth;

        // Single-nuclide fast path: nuclide is trivially the only one.
        if by_nuclide.len() == 1 {
            let name = sorted_keys.first()?;
            let nuclide = lookup_nuclide(0, name)?;
            let nuc_id = sorted_ids.and_then(|ids| ids.first().copied());
            return Some((distance, name.as_str(), nuclide, None, nuc_id));
        }

        // Multi-nuclide smooth: sample nuclide proportional to smooth per-nuclide contributions.
        let xi_nuclide = rng.random_range(0.0..sigma_t_smooth);
        let mut accum = 0.0;
        for (idx, name) in sorted_keys.iter().enumerate() {
            let xs = smooth_nuc_xs(idx);
            accum += xs;
            if xi_nuclide < accum {
                let nuclide = lookup_nuclide(idx, name)?;
                let nuc_id = sorted_ids.and_then(|ids| ids.get(idx).copied());
                return Some((distance, name.as_str(), nuclide, None, nuc_id));
            }
        }
        None
    }

    /// Sample the free-flight distance on the smooth (non-URR) PCG path, drawing
    /// `xi1` from the per-particle PCG `state` (issue #111).
    ///
    /// This is the front half of the shared GPU/CPU collision path: the GPU
    /// kernel draws the flight as its *first* per-step PCG sample
    /// (`-ln(xi1) / sigma_t`) and selects the struck nuclide only after the
    /// surface-vs-collision min wins (see [`Material::select_nuclide_smooth`]).
    /// Splitting flight from selection lets the CPU match that draw schedule:
    /// `xi1` every step, `xi_n` only on a real collision.
    ///
    /// Returns `None` -- **without consuming a PCG draw** -- when the fast path
    /// or per-nuclide tables are unavailable, or URR is in range at `energy`.
    /// The caller then falls back to the legacy fused `sample_collision_data` on
    /// the FastRng stream. The PCG sample is taken only once all preconditions
    /// hold, so a fallback never desynchronises the PCG stream.
    #[inline]
    pub fn smooth_flight(
        &self,
        energy: f64,
        held_urr: Option<f64>,
        state: &mut u64,
    ) -> Option<SmoothFlight> {
        let fast_xs = self.fast_xs.as_ref()?;
        // Per-nuclide tables are required by `select_nuclide_smooth`; bail to the
        // legacy path together if they are absent.
        self.macroscopic_xs_neutron_total_by_nuclide.as_ref()?;
        let (sigma_t_smooth, i_grid, f) =
            fast_xs.lookup_total(energy, &self.unified_energy_grid_neutron);
        if sigma_t_smooth <= 0.0 {
            return None;
        }

        // URR: the base uniform is drawn once per ENERGY and BEFORE the flight,
        // matching the kernel's per-step order (issues #342, #111). A band the
        // particle already holds at this energy is reused, so a boundary
        // crossing does not redraw. Every draw below is `next_xi`, the mapping
        // the GPU uses, so the two streams stay aligned.
        let mut urr_random = None;
        let mut sigma_t = sigma_t_smooth;
        if self.has_urr_in_range(energy) {
            let band = match held_urr {
                Some(b) => b,
                None => yamc_rng::next_xi(state),
            };
            urr_random = Some(band);
            let s = self.urr_macro_total(energy, i_grid, f, band);
            // A non-positive URR total would be unphysical; fall back to the
            // smooth total rather than bailing, because the band draw above has
            // already advanced the stream and a bail here would desync it.
            if s > 0.0 {
                sigma_t = s;
            }
        }

        let xi = yamc_rng::next_xi(state);
        Some(SmoothFlight {
            distance: -xi.ln() / sigma_t,
            i_grid,
            f,
            sigma_t,
            urr_random,
        })
    }

    /// Material macroscopic total with every in-range URR nuclide perturbed by
    /// the band `urr_random` selects for it. Each nuclide derives its own band
    /// from the shared base (issue #204), so isotopes stay uncorrelated.
    fn urr_macro_total(&self, energy: f64, i_grid: usize, f: f64, urr_random: f64) -> f64 {
        (0..self.nuclide_count_for_selection())
            .map(|idx| self.nuclide_total_with_urr(energy, idx, i_grid, f, urr_random))
            .sum()
    }

    /// Number of nuclides the selection walk sees, i.e. the length of the
    /// sorted per-nuclide macroscopic table.
    fn nuclide_count_for_selection(&self) -> usize {
        self.macroscopic_xs_neutron_total_by_nuclide
            .as_ref()
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// One nuclide's macroscopic total at the bracket, URR-perturbed when its
    /// table covers `energy`. Falls back to the smooth value otherwise.
    fn nuclide_total_with_urr(
        &self,
        energy: f64,
        idx: usize,
        i_grid: usize,
        f: f64,
        urr_random: f64,
    ) -> f64 {
        let Some(by_nuclide) = self.macroscopic_xs_neutron_total_by_nuclide.as_ref() else {
            return 0.0;
        };
        let xs_vec = &by_nuclide[idx];
        let smooth = if xs_vec.len() > i_grid + 1 {
            let xs0 = xs_vec[i_grid];
            let xs1 = xs_vec[i_grid + 1];
            xs0 + f * (xs1 - xs0)
        } else {
            0.0
        };
        let Some(keys) = self.sorted_nuclide_keys.as_ref() else {
            return smooth;
        };
        let Some(name) = keys.get(idx) else {
            return smooth;
        };
        let nuclide = match self
            .sorted_nuclides
            .as_deref()
            .and_then(|v| v.get(idx))
            .or_else(|| self.nuclide_data.get(name))
        {
            Some(n) => n,
            None => return smooth,
        };
        self.nuclide_macro_total_with_urr(nuclide, smooth, energy, urr_random)
    }

    /// Select the struck nuclide on the smooth (non-URR) PCG path, drawing
    /// `xi_n` from the per-particle PCG `state` (issue #111). Call only on a real
    /// collision (after the surface-vs-collision min), using the bracket from
    /// [`Material::smooth_flight`].
    ///
    /// A single-nuclide material draws nothing (the lone nuclide is trivially the
    /// target -- byte-identical PCG stream to the pre-multi-nuclide path). A
    /// multi-nuclide material draws one `xi_n` and walks the per-nuclide
    /// macroscopic totals at the energy bracket, exact twin of the GPU kernel
    /// block (yamc-gpu shared.rs): `xi_n = next_xi * sigma_t_nuc`, then the first
    /// nuclide whose cumulative total exceeds `xi_n` (clamped to the last on a
    /// floating-point overshoot).
    #[inline]
    pub fn select_nuclide_smooth(
        &self,
        energy: f64,
        i_grid: usize,
        f: f64,
        urr_random: Option<f64>,
        state: &mut u64,
    ) -> Option<SelectedNuclide<'_>> {
        let by_nuclide = self.macroscopic_xs_neutron_total_by_nuclide.as_ref()?;
        let sorted_keys = self.sorted_nuclide_keys.as_ref()?;
        let sorted_ids = self.sorted_nuclide_ids.as_deref();
        let sorted_nuclides = self.sorted_nuclides.as_deref();

        let lookup_nuclide = |idx: usize, name: &str| -> Option<&Arc<Nuclide>> {
            sorted_nuclides
                .and_then(|v| v.get(idx))
                .or_else(|| self.nuclide_data.get(name))
        };
        // A nuclide's share of the collision density follows the cross section
        // that actually governed the flight, so an in-range URR nuclide is
        // weighted by its PERTURBED total (issue #347). Without a band this is
        // the plain smooth total.
        let smooth_nuc_xs = |idx: usize| -> f64 {
            match urr_random {
                Some(band) => self.nuclide_total_with_urr(energy, idx, i_grid, f, band),
                None => {
                    let xs_vec = &by_nuclide[idx];
                    if xs_vec.len() > i_grid + 1 {
                        let xs0 = xs_vec[i_grid];
                        let xs1 = xs_vec[i_grid + 1];
                        xs0 + f * (xs1 - xs0)
                    } else {
                        0.0
                    }
                }
            }
        };

        // Single-nuclide fast path: no PCG draw (matches the GPU `count > 1` gate).
        if by_nuclide.len() == 1 {
            let name = sorted_keys.first()?;
            let nuclide = lookup_nuclide(0, name)?;
            let nuc_id = sorted_ids.and_then(|ids| ids.first().copied());
            return Some((name.as_str(), nuclide, nuc_id));
        }

        // Multi-nuclide: sample proportional to per-nuclide macroscopic totals at
        // the bracket. Scale `xi_n` by the sum of those totals (the GPU's
        // `sigma_t_nuc`), not the flight's `sigma_t`, so the two backends share
        // the exact selection arithmetic.
        let sigma_t_nuc: f64 = (0..by_nuclide.len()).map(smooth_nuc_xs).sum();
        let xi_n = yamc_rng::next_xi(state) * sigma_t_nuc;
        let mut accum = 0.0;
        for (idx, name) in sorted_keys.iter().enumerate() {
            accum += smooth_nuc_xs(idx);
            if xi_n < accum {
                let nuclide = lookup_nuclide(idx, name)?;
                let nuc_id = sorted_ids.and_then(|ids| ids.get(idx).copied());
                return Some((name.as_str(), nuclide, nuc_id));
            }
        }
        // Floating-point overshoot (xi_n >= sigma_t_nuc): clamp to the last.
        let last = by_nuclide.len() - 1;
        let name = sorted_keys.get(last)?;
        let nuclide = lookup_nuclide(last, name)?;
        let nuc_id = sorted_ids.and_then(|ids| ids.get(last).copied());
        Some((name.as_str(), nuclide, nuc_id))
    }

    /// URR-modified macroscopic total contribution for a single nuclide at this energy.
    ///
    /// Returns `smooth_macro_total` unchanged if the nuclide has no URR data, the energy is
    /// outside the URR range, or per-temperature fast XS is unavailable. Otherwise samples
    /// the URR probability table (using the supplied `urr_random`) and returns the URR-modified
    /// macroscopic total = `smooth_macro_total * (urr_micro_total / smooth_micro_total)`.
    ///
    /// This is the per-nuclide primitive used by [`Material::sample_collision_data`] to compute
    /// the URR-consistent Σ_t and per-nuclide contributions for nuclide selection.
    fn nuclide_macro_total_with_urr(
        &self,
        nuclide: &Nuclide,
        smooth_macro_total: f64,
        energy: f64,
        urr_random: f64,
    ) -> f64 {
        let Some(sample) = self.urr_sample_for_nuclide(nuclide, energy, urr_random) else {
            return smooth_macro_total;
        };

        if sample.xs_total > 1e-20 {
            smooth_macro_total * (sample.urr_total / sample.xs_total)
        } else {
            smooth_macro_total
        }
    }

    /// Shared per-nuclide URR probability-table sampling.
    ///
    /// Runs the guard chain (URR present, temperature index, in-range URR data,
    /// per-temperature fast XS), then extracts the smooth microscopic partials from
    /// `fast_xs` and samples the URR table with `urr_random`. Returns `None` when the
    /// nuclide has no applicable URR contribution at this energy (the early-return
    /// semantics callers rely on); otherwise returns the smooth values the callers
    /// still need alongside the URR-sampled cross-sections.
    fn urr_sample_for_nuclide(
        &self,
        nuclide: &Nuclide,
        energy: f64,
        urr_random: f64,
    ) -> Option<UrrNuclideSample> {
        if !nuclide.urr_present {
            return None;
        }
        let temp_idx = nuclide.get_temp_idx(&self.temperature)?;
        let urr = nuclide
            .urr_data
            .get(temp_idx)
            .and_then(|opt| opt.as_ref())?;
        if !urr.energy_in_bounds(energy) {
            return None;
        }
        let fast_grid = nuclide.fast_xs.get(temp_idx)?;

        let (xs_total, xs_absorption, xs_scattering, xs_fission) = fast_grid.lookup(energy);
        let (i_grid, f) = fast_grid.lookup_grid_index(energy);
        let xs_elastic = fast_grid
            .elastic_idx
            .map(|idx| fast_grid.scatter_xs_interp(i_grid, f, idx))
            .unwrap_or(0.0);
        let xs_inelastic = (xs_scattering - xs_elastic).max(0.0);
        // `xs_absorption` is `FastXSGrid::lookup`'s disappearance PARTIAL, which
        // already excludes fission (the four partials sum to the total). This is
        // the SCORING-side twin of the transport-side bug fixed in #154: OpenMC
        // writes `capture *= (micro.absorption - micro.fission)` because ITS
        // absorption includes fission, and importing that expression here
        // subtracted fission a second time, clamping in-band capture to zero for
        // every nuclide whose fission exceeds its capture. In-band that made a
        // capture tally read zero and an absorption tally (built as
        // `macro_capture + macro_fission`) report just the fission rate.
        let xs_capture = xs_absorption;
        let xs_ngamma = if !urr.multiply_smooth && !fast_grid.xs_ngamma.is_empty() {
            Some(fast_grid.lookup_ngamma(i_grid, f))
        } else {
            None
        };

        let smooth_absorption = xs_capture + xs_fission;
        // `urr_random` is the per-collision base seed shared across the
        // material's nuclides; derive this nuclide's independent probability
        // table band from it (issue #204). Isotopes' resonance structures are
        // statistically independent, so each must draw its own band rather than
        // all sharing one random (which over-transmits multi-isotope materials).
        let r = yamc_nuclide::urr::urr_nuclide_random(urr_random, nuclide.urr_stream_key());
        let (urr_total, urr_elastic, urr_capture, urr_fission, _smooth) = urr.sample(
            energy,
            r,
            xs_elastic,
            smooth_absorption,
            xs_fission,
            xs_inelastic,
            xs_ngamma,
        );

        Some(UrrNuclideSample {
            xs_total,
            xs_inelastic,
            inelastic_in_table: urr.inelastic_flag > 0,
            urr_total,
            urr_elastic,
            urr_capture,
            urr_fission,
        })
    }

    /// Compute URR-modified macroscopic cross-sections for tally scoring.
    /// Returns per-MT macroscopic values (total, elastic, fission, capture, absorption)
    /// with URR probability table modifications applied, or None if no nuclide has URR.
    pub fn compute_urr_macro_xs(&self, energy: f64, urr_random: f64) -> Option<UrrMacroXs> {
        let owned_atoms;
        let atoms_per_bcm = if let Some(ref cached) = self.cached_atoms_per_barn_cm {
            cached
        } else {
            owned_atoms = self
                .get_atoms_per_barn_cm()
                .unwrap_or_else(|e| panic!("{e}"));
            &owned_atoms
        };
        let temperature = &self.temperature;

        let mut macro_total = 0.0;
        let mut macro_elastic = 0.0;
        let mut macro_fission = 0.0;
        let mut macro_capture = 0.0;
        let mut any_urr = false;

        for (name, nuclide) in &self.nuclide_data {
            let n_density = match atoms_per_bcm.get(name) {
                Some(&n) => n,
                None => continue,
            };

            // Skip nuclides with no matching temperature (matches the per-nuclide
            // temperature guard inside `urr_sample_for_nuclide`, but here a missing
            // index drops the nuclide entirely rather than falling back to smooth XS).
            if nuclide.get_temp_idx(temperature).is_none() {
                continue;
            }

            // Check if this nuclide has URR data in range
            let urr_applied =
                if let Some(sample) = self.urr_sample_for_nuclide(nuclide, energy, urr_random) {
                    // Accumulate URR-modified microscopic XS * atom density
                    // Total is sum of partials + inelastic (inelastic is unmodified by URR)
                    let urr_inelastic = if sample.inelastic_in_table {
                        sample.xs_inelastic
                    } else {
                        0.0
                    };
                    macro_total += n_density
                        * (sample.urr_elastic
                            + urr_inelastic
                            + sample.urr_capture
                            + sample.urr_fission);
                    macro_elastic += n_density * sample.urr_elastic;
                    macro_fission += n_density * sample.urr_fission;
                    macro_capture += n_density * sample.urr_capture;
                    any_urr = true;
                    true
                } else {
                    false
                };

            // Non-URR nuclide (or URR not applicable): accumulate smooth contributions
            if !urr_applied {
                macro_total += self.lookup_nuclide_macro_xs_by_mt(name, n_density, 1, energy);
                macro_elastic += self.lookup_nuclide_macro_xs_by_mt(name, n_density, 2, energy);
                macro_fission += self.lookup_nuclide_macro_xs_by_mt(name, n_density, 18, energy);
                // MT 101 (disappearance), not MT 102: `absorption` below is built
                // as `macro_capture + macro_fission`, which is OpenMC's
                // `micro.absorption = capture + fission` where its `capture` is
                // `absorption - fission`, i.e. disappearance. Asking for MT 102
                // here would drop the charged-particle absorption channels and,
                // before #362, returned zero outright for a fissile nuclide.
                macro_capture += self.lookup_nuclide_macro_xs_by_mt(name, n_density, 101, energy);
            }
        }

        if any_urr {
            let absorption = macro_capture + macro_fission;
            Some(UrrMacroXs {
                total: macro_total,
                elastic: macro_elastic,
                fission: macro_fission,
                capture: macro_capture,
                absorption,
            })
        } else {
            None
        }
    }

    /// Macroscopic total cross section Σ_t(E), with URR probability-table modifications
    /// applied if any nuclide has URR data in range at this energy.
    ///
    /// Decouples the Σ_t-at-a-point query from nuclide sampling. Useful for callers
    /// that need Σ_t without committing to a reaction sample (e.g. Woodcock tracking,
    /// flux/source biasing previews, visualisation).
    ///
    /// If no nuclide has URR data in range, returns `(sigma_t_smooth, None)` and does
    /// not consume a random number. Otherwise samples `urr_random` (or reuses
    /// `cached_urr_random` if `Some`), applies URR consistently across all in-range
    /// URR nuclides via [`Material::compute_urr_macro_xs`], and returns
    /// `(sigma_t_with_urr, Some(urr_random))`. Cache the returned random so subsequent
    /// queries -- or a later [`Material::sample_collision_data`] call -- at the same
    /// energy stay consistent.
    ///
    /// URR semantics match those used for tally scoring: all in-range URR nuclides
    /// receive URR-modified contributions, each deriving an independent probability
    /// table band from the shared base seed (issue #204); non-URR nuclides keep their
    /// smooth contributions. For single-URR-nuclide materials this matches the Σ_t used
    /// internally by `sample_collision_data`; for multi-nuclide materials with multiple
    /// URR nuclides it may differ slightly (`sample_collision_data` applies URR only to
    /// the sampled nuclide).
    pub fn total_xs_with_urr<R: rand::Rng + ?Sized>(
        &self,
        energy: f64,
        cached_urr_random: Option<f64>,
        rng: &mut R,
    ) -> (f64, Option<f64>) {
        let sigma_t_smooth = self.lookup_xs_by_mt(1, energy);

        if !self.has_urr_in_range(energy) {
            return (sigma_t_smooth, None);
        }

        let urr_random = cached_urr_random.unwrap_or_else(|| rng.random::<f64>());
        match self.compute_urr_macro_xs(energy, urr_random) {
            Some(urr_macro) => (urr_macro.total, Some(urr_random)),
            None => (sigma_t_smooth, None),
        }
    }

    /// URR-aware upper bound on the macroscopic total cross section at
    /// `energy`. Used by Woodcock majorant construction.
    ///
    /// For energies outside any URR nuclide's URR range this returns
    /// the same value as `lookup_xs_by_mt(1, energy)`. Inside a URR
    /// range, replaces each URR-bearing nuclide's smooth contribution
    /// with its worst-case URR-sampled contribution (the max value
    /// across the probability-table CDF). The result is guaranteed
    /// `≥ Σ_t(urr_random=anything, energy)`.
    ///
    /// Construction-time cost is `O(n_urr_nuclides × n_cdf)` per
    /// energy, which is fine because the majorant is built once at
    /// simulation start.
    pub fn total_xs_majorant(&self, energy: f64) -> f64 {
        let smooth_total = self.lookup_xs_by_mt(1, energy);
        if !self.has_urr_in_range(energy) {
            return smooth_total;
        }

        let owned_atoms;
        let atoms_per_bcm = if let Some(ref cached) = self.cached_atoms_per_barn_cm {
            cached
        } else {
            owned_atoms = self
                .get_atoms_per_barn_cm()
                .unwrap_or_else(|e| panic!("{e}"));
            &owned_atoms
        };

        let mut adjusted = smooth_total;
        for (name, nuclide) in &self.nuclide_data {
            if !nuclide.urr_present {
                continue;
            }
            let Some(temp_idx) = nuclide.get_temp_idx(&self.temperature) else {
                continue;
            };
            let Some(urr) = nuclide.urr_data.get(temp_idx).and_then(|opt| opt.as_ref()) else {
                continue;
            };
            if !urr.energy_in_bounds(energy) {
                continue;
            }
            let Some(&n_density) = atoms_per_bcm.get(name) else {
                continue;
            };
            let Some(fast_grid) = nuclide.fast_xs.get(temp_idx) else {
                continue;
            };

            // Smooth micro Σ_t for this nuclide (subtract its smooth
            // macro contribution, then add the URR-max macro
            // contribution).
            let (smooth_micro_total, _, smooth_micro_scatter, smooth_micro_fission) =
                fast_grid.lookup(energy);
            let (i_grid, f) = fast_grid.lookup_grid_index(energy);
            let smooth_micro_elastic = fast_grid
                .elastic_idx
                .map(|idx| fast_grid.scatter_xs_interp(i_grid, f, idx))
                .unwrap_or(0.0);
            // Inelastic isn't modified by URR -- it stays at smooth.
            let smooth_micro_inelastic = (smooth_micro_scatter - smooth_micro_elastic).max(0.0);

            // Worst-case URR micro from this nuclide.
            let urr_max_in_table = urr.max_total_in_table(energy);
            let urr_max_micro = if urr.multiply_smooth {
                // Table values are multipliers on smooth components.
                // The contributions URR-modifies are elastic +
                // capture + fission (matches `sample()`). Inelastic
                // is fixed at smooth.
                let smooth_partials = smooth_micro_elastic
                    + smooth_micro_fission
                    + (smooth_micro_total - smooth_micro_scatter - smooth_micro_fission).max(0.0);
                // Worst case: all URR-modified partials scaled by the
                // single largest factor in the table. This is a (mild)
                // over-bound -- the table actually applies separate
                // factors per partial -- and is what we want for a
                // majorant.
                let scaled = smooth_partials * urr_max_in_table;
                let inelastic_contrib = if urr.inelastic_flag > 0 {
                    smooth_micro_inelastic
                } else {
                    0.0
                };
                scaled + inelastic_contrib
            } else {
                // Table values are absolute -- max sum of partials
                // from `urr.max_total_in_table` plus inelastic.
                let inelastic_contrib = if urr.inelastic_flag > 0 {
                    smooth_micro_inelastic
                } else {
                    0.0
                };
                urr_max_in_table + inelastic_contrib
            };

            // Adjust the macro total: remove smooth contribution, add
            // URR-max contribution.
            adjusted += n_density * (urr_max_micro - smooth_micro_total);
        }

        // The URR-max can in principle be below the smooth value for
        // some nuclides at some energies; the majorant must still
        // bound the smooth case as well. Take the larger of the two.
        adjusted.max(smooth_total)
    }

    /// True if **any** nuclide in this material carries URR probability-
    /// table data at the material's temperature, regardless of energy.
    /// Energy-independent companion to [`Self::has_urr_in_range`] --
    /// useful for validation passes that need to reject configurations
    /// at simulation start (e.g. Woodcock tracking with a smooth-Σ_t
    /// majorant cannot bound URR-sampled Σ_t and would bias the
    /// rejection loop).
    pub fn has_urr_data(&self) -> bool {
        for nuclide in self.nuclide_data.values() {
            if !nuclide.urr_present {
                continue;
            }
            let Some(temp_idx) = nuclide.get_temp_idx(&self.temperature) else {
                continue;
            };
            if nuclide
                .urr_data
                .get(temp_idx)
                .and_then(|opt| opt.as_ref())
                .is_some()
            {
                return true;
            }
        }
        false
    }

    /// True if any nuclide in this material has URR probability-table data
    /// covering `energy` at the material's temperature.
    pub fn has_urr_in_range(&self, energy: f64) -> bool {
        for nuclide in self.nuclide_data.values() {
            if !nuclide.urr_present {
                continue;
            }
            let Some(temp_idx) = nuclide.get_temp_idx(&self.temperature) else {
                continue;
            };
            if let Some(urr) = nuclide.urr_data.get(temp_idx).and_then(|opt| opt.as_ref()) {
                if urr.energy_in_bounds(energy) {
                    return true;
                }
            }
        }
        false
    }

    /// Helper: compute N * sigma_mt for a single nuclide using the material's macroscopic XS grid.
    /// Falls back to micro XS * atom density if the nuclide has fast_xs data.
    fn lookup_nuclide_macro_xs_by_mt(
        &self,
        name: &str,
        n_density: f64,
        mt: i32,
        energy: f64,
    ) -> f64 {
        // Use nuclide microscopic XS from fast_xs if available
        if let Some(nuclide) = self.nuclide_data.get(name) {
            let temp_idx = nuclide.get_temp_idx(&self.temperature);
            if let Some(idx) = temp_idx {
                if let Some(fast_grid) = nuclide.fast_xs.get(idx) {
                    let (total, absorption, scattering, fission) = fast_grid.lookup(energy);
                    let micro = match mt {
                        1 => total,
                        2 => {
                            // Elastic from scatter_mt_xs via elastic_idx
                            let (i_grid, f) = fast_grid.lookup_grid_index(energy);
                            fast_grid
                                .elastic_idx
                                .map(|idx| fast_grid.scatter_xs_interp(i_grid, f, idx))
                                .unwrap_or(0.0)
                        }
                        18 => fission,
                        // MT 27 is ENDF absorption = MT 18 + MT 101, so it
                        // INCLUDES fission, and that is what OpenMC's
                        // `"absorption"` score reports: `nuclide.cpp` sets
                        // `micro.absorption = capture + fission` and the tally
                        // scores `macro_xs().absorption * flux`. `absorption`
                        // from `lookup` is the disappearance PARTIAL (its four
                        // partials sum to the total), so fission is added back
                        // here (issue #362).
                        27 => absorption + fission,
                        // MT 101, neutron disappearance: the partial as stored.
                        101 => absorption,
                        // MT 102 is RADIATIVE CAPTURE specifically, which the
                        // library carries as its own column. It used to be
                        // derived as `(absorption - fission).max(0.0)`, which is
                        // OpenMC's expression but not OpenMC's input: OpenMC
                        // subtracts fission from an absorption that includes it,
                        // while this one excludes it. That read ZERO for every
                        // fissile nuclide (U235 at 10 keV: absorption 1.06 b
                        // against fission 2.91 b) and, where fission is absent,
                        // still conflated capture with the charged-particle
                        // channels the partial also holds -- 149x high for Fe56
                        // at 14 MeV, 5x for W184 (issue #362).
                        102 => {
                            let (i_grid, f) = fast_grid.lookup_grid_index(energy);
                            if fast_grid.xs_ngamma.is_empty() {
                                // No MT 102 column in this library: the
                                // disappearance partial is the closest stand-in,
                                // and equals capture wherever the
                                // charged-particle channels are closed.
                                absorption
                            } else {
                                fast_grid.lookup_ngamma(i_grid, f)
                            }
                        }
                        _ => {
                            // For other MTs, fall back to scattering or 0
                            if mt == 4 {
                                let elastic = {
                                    let (i_grid, f) = fast_grid.lookup_grid_index(energy);
                                    fast_grid
                                        .elastic_idx
                                        .map(|idx| fast_grid.scatter_xs_interp(i_grid, f, idx))
                                        .unwrap_or(0.0)
                                };
                                (scattering - elastic).max(0.0)
                            } else {
                                return 0.0;
                            }
                        }
                    };
                    return n_density * micro;
                }
            }
        }
        0.0
    }

    pub fn volume(&mut self, value: Option<f64>) -> Result<Option<f64>, String> {
        if let Some(v) = value {
            if v <= 0.0 {
                return Err(String::from("Volume must be positive"));
            }
            self.volume = Some(v);
        }
        Ok(self.volume)
    }

    /// The temperature label, or `""` if it has not been resolved yet.
    pub fn temperature(&self) -> &str {
        &self.temperature
    }

    /// The temperature in Kelvin, for physics kernels.
    ///
    /// Always consistent with [`Material::temperature`]; that is the point of
    /// both fields being private.
    pub fn temperature_k(&self) -> f64 {
        self.temperature_k
    }

    /// Write the label and its Kelvin value without touching the caches.
    ///
    /// The label is normalised with `strip_k`, so `"294K"` and `"294"` name the
    /// same temperature everywhere. Nuclide data is keyed by the stripped form
    /// (`Nuclide::get_temp_idx` compares strings exactly), so storing the
    /// suffixed form here would make an otherwise valid temperature unfindable.
    ///
    /// Private, and it must stay that way: a caller that changes the
    /// temperature without invalidating the derived caches leaves the material
    /// reporting one temperature while its cached grid and cross sections
    /// belong to another (#481). Use [`Material::set_temperature`].
    fn assign_temperature(&mut self, temperature: impl AsRef<str>) {
        let label = yamc_nuclide::temperature::strip_k(temperature.as_ref());
        self.temperature_k = yamc_nuclide::temperature::label_to_kelvin_or_default(label);
        self.temperature = String::from(label);
    }

    /// Set the temperature and drop everything derived from the old one.
    ///
    /// Invalidation goes through [`Material::invalidate_temperature_caches`]
    /// rather than clearing a hand-written subset. The subset this used to clear left
    /// `cached_microscopic_xs`, `fast_xs` and both by-nuclide tables holding
    /// the previous temperature's data (#481), and `fast_xs` surviving an
    /// emptied grid is itself a panic: `MaterialFastXS::lookup_total` indexes
    /// `energy_grid[0]` unguarded.
    pub fn set_temperature(&mut self, temperature: impl AsRef<str>) {
        self.assign_temperature(temperature);
        self.invalidate_temperature_caches();
    }

    /// Resolve temperature with fallback chain:
    /// 1. Use provided temperature if Some
    /// 2. Fall back to material.temperature if set AND available in nuclide data
    /// 3. Fall back to the only temperature if nuclide data has exactly one
    /// 4. Otherwise panic with available temperatures listed
    pub fn resolve_temperature(&mut self, provided: Option<&str>) -> String {
        // Collect all available temperatures from nuclide data
        let mut all_temps: std::collections::HashSet<String> = std::collections::HashSet::new();
        for nuclide in self.nuclide_data.values() {
            for temp in &nuclide.available_temperatures {
                all_temps.insert(temp.clone());
            }
        }

        // 1. Use provided temperature if given.
        // Normalised, because callers pass the on-disk spelling: nuclide data
        // is keyed by the stripped form, so `"900K"` would otherwise be
        // rejected as unavailable while `"900"` succeeded on the same data
        // (#481).
        if let Some(temp) = provided {
            return yamc_nuclide::temperature::strip_k(temp).to_string();
        }

        // Photon-only mode: no nuclide data loaded, return material temperature
        if self.nuclide_data.is_empty() && !self.temperature.is_empty() {
            return self.temperature.clone();
        }

        // 2. Fall back to material.temperature if set AND available in nuclide data
        if !self.temperature.is_empty() && all_temps.contains(&self.temperature) {
            return self.temperature.clone();
        }

        // 3. If material.temperature is empty and there's only one temp, auto-detect and set it
        if self.temperature.is_empty() && all_temps.len() == 1 {
            let detected_temp = all_temps.into_iter().next().unwrap();
            // Through the setter: this branch is why `temperature_k` could go
            // stale even when the user never touched the temperature at all.
            //
            // `assign_temperature`, not `set_temperature`: this fills in an
            // empty label with the temperature the caches are about to be built
            // at, so there is nothing derived from a *different* temperature to
            // throw away, and invalidating here would drop a grid the caller is
            // part-way through building.
            self.assign_temperature(&detected_temp);
            return detected_temp;
        }

        // 4. If material.temperature is explicitly set but NOT available, error
        if !self.temperature.is_empty() {
            let mut temps_list: Vec<String> = all_temps.into_iter().collect();
            temps_list.sort();
            panic!(
                "Temperature '{}' not available. Available temperatures: {:?}",
                self.temperature, temps_list
            );
        }

        // 5. Panic with helpful error (multiple temps available, none specified)
        let mut temps_list: Vec<String> = all_temps.into_iter().collect();
        temps_list.sort();
        panic!(
            "Temperature not specified or not available. Please provide a temperature argument, \
             set material.temperature to an available value, or ensure nuclide data has only one temperature. \
             Available temperatures: {temps_list:?}"
        );
    }

    /// Validate that all loaded nuclides have compatible temperatures
    /// Ensures materials don't mix nuclides with different available temperatures
    pub fn validate_temperature_consistency(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.nuclide_data.is_empty() {
            return Ok(());
        }

        // Collect all available temperatures for each nuclide
        let mut nuclide_temps: Vec<(String, Vec<String>)> = Vec::new();
        for (name, nuclide) in &self.nuclide_data {
            let mut temps = nuclide.available_temperatures.clone();
            temps.sort();
            nuclide_temps.push((name.clone(), temps));
        }

        // Find common temperatures across all nuclides
        if let Some((_first_name, first_temps)) = nuclide_temps.first() {
            let common_temps: Vec<String> = first_temps
                .iter()
                .filter(|t| nuclide_temps.iter().all(|(_, temps)| temps.contains(t)))
                .cloned()
                .collect();

            if common_temps.is_empty() {
                // No common temperature - build detailed error message
                let mut msg = format!(
                    "Material '{}' has nuclides with incompatible temperature data. \
                     No common temperature found across all nuclides.\n",
                    self.name.as_deref().unwrap_or("unnamed")
                );
                for (name, temps) in &nuclide_temps {
                    msg.push_str(&format!(
                        "  {}: [{}]\n",
                        name,
                        temps
                            .iter()
                            .map(|s| format!("'{s}'"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }
                msg.push_str(
                    "All nuclides in a material must have at least one common temperature.",
                );
                return Err(msg.into());
            }

            // If material temperature is set, verify it's in the common set
            if !self.temperature.is_empty() && !common_temps.contains(&self.temperature) {
                let temp_list = common_temps
                    .iter()
                    .map(|s| format!("'{s}'"))
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(format!(
                    "Material temperature '{}' is not available in all nuclides. \
                     Common temperatures across all nuclides: {}",
                    self.temperature, temp_list
                )
                .into());
            }
        }

        Ok(())
    }

    pub fn get_nuclides(&self) -> Vec<String> {
        let mut nuclides: Vec<String> = self.nuclides.keys().cloned().collect();
        nuclides.sort(); // Sort alphabetically for consistent output
        nuclides
    }

    /// Sample which nuclide a neutron interacts with at a given energy, using per-nuclide macroscopic total xs
    /// Returns the nuclide name as a String, or None if not possible
    #[inline]
    pub fn sample_interacting_nuclide<R: rand::Rng + ?Sized>(
        &self,
        energy: f64,
        rng: &mut R,
    ) -> String {
        let by_nuclide = self.macroscopic_xs_neutron_total_by_nuclide.as_ref().expect("macroscopic_xs_neutron_total_by_nuclide is None: call calculate_macroscopic_xs with by_nuclide=true first");

        // Use cached sorted keys for deterministic iteration order
        let sorted_keys = self
            .sorted_nuclide_keys
            .as_ref()
            .expect("sorted_nuclide_keys must be set when by_nuclide is enabled");

        // Fast path: single nuclide - no sampling needed
        if sorted_keys.len() == 1 {
            return sorted_keys[0].clone();
        }

        let mut xs_by_nuclide = Vec::with_capacity(by_nuclide.len());
        let mut total = 0.0;
        for (idx, nuclide) in sorted_keys.iter().enumerate() {
            let xs_vec = &by_nuclide[idx];
            if xs_vec.is_empty() || self.unified_energy_grid_neutron.is_empty() {
                continue;
            }
            let xs = crate::interpolate_linear(&self.unified_energy_grid_neutron, xs_vec, energy);
            if xs > 0.0 {
                xs_by_nuclide.push((nuclide, xs));
                total += xs;
            }
        }
        if xs_by_nuclide.is_empty() || total <= 0.0 {
            // Only build debug info if we're about to panic
            let mut debug_info = String::new();
            for (nuclide, xs_vec) in sorted_keys.iter().zip(by_nuclide.iter()) {
                if xs_vec.is_empty() || self.unified_energy_grid_neutron.is_empty() {
                    debug_info.push_str(&format!("{nuclide}: EMPTY\n"));
                } else {
                    let xs = crate::interpolate_linear(
                        &self.unified_energy_grid_neutron,
                        xs_vec,
                        energy,
                    );
                    debug_info.push_str(&format!("{nuclide}: xs = {xs}\n"));
                }
            }
            panic!(
                "No nuclide has nonzero macroscopic total cross section at energy {energy}. Details:\n{debug_info}"
            );
        }
        let xi = rng.random_range(0.0..total);
        let mut accum = 0.0;
        for (nuclide, xs) in xs_by_nuclide {
            accum += xs;
            if xi < accum {
                return nuclide.clone();
            }
        }
        panic!("Failed to sample nuclide: numerical error in sampling loop");
    }

    /// Sample which nuclide a neutron interacts with, returning references directly.
    /// This avoids String allocation and HashMap lookup in the hot path.
    /// Returns (nuclide_name, nuclide_data) or None if sampling fails.
    #[inline]
    pub fn sample_interacting_nuclide_data<R: rand::Rng + ?Sized>(
        &self,
        energy: f64,
        rng: &mut R,
    ) -> Option<(&str, &Arc<Nuclide>)> {
        let by_nuclide = self.macroscopic_xs_neutron_total_by_nuclide.as_ref()?;

        // Use cached sorted keys for deterministic iteration order
        let sorted_keys = self.sorted_nuclide_keys.as_ref()?;

        // Fast path: single nuclide - no sampling needed
        if sorted_keys.len() == 1 {
            let name = &sorted_keys[0];
            return self.nuclide_data.get(name).map(|n| (name.as_str(), n));
        }

        let mut xs_by_nuclide = Vec::with_capacity(by_nuclide.len());
        let mut total = 0.0;
        for (idx, nuclide) in sorted_keys.iter().enumerate() {
            let xs_vec = &by_nuclide[idx];
            if xs_vec.is_empty() || self.unified_energy_grid_neutron.is_empty() {
                continue;
            }
            let xs = crate::interpolate_linear(&self.unified_energy_grid_neutron, xs_vec, energy);
            if xs > 0.0 {
                xs_by_nuclide.push((nuclide.as_str(), xs));
                total += xs;
            }
        }

        if xs_by_nuclide.is_empty() || total <= 0.0 {
            return None;
        }

        let xi = rng.random_range(0.0..total);
        let mut accum = 0.0;
        for (name, xs) in xs_by_nuclide {
            accum += xs;
            if xi < accum {
                return self.nuclide_data.get(name).map(|n| (name, n));
            }
        }

        None
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_symbol_from_nuclide() {
        use yamc_element::element::element_symbol_from_nuclide;
        assert_eq!(element_symbol_from_nuclide("Fe56"), "Fe");
        assert_eq!(element_symbol_from_nuclide("Li6"), "Li");
        assert_eq!(element_symbol_from_nuclide("Am241m"), "Am");
        assert_eq!(element_symbol_from_nuclide("H1"), "H");
        assert_eq!(element_symbol_from_nuclide("U235"), "U");
        assert_eq!(element_symbol_from_nuclide("Pb208"), "Pb");
    }

    #[test]
    fn test_new_invalid_fraction_type_errors() {
        // Invalid percent/fraction type must fail-fast in Material::new.
        let result = Material::new(HashMap::new(), "atomm", "sum", None);
        assert!(result.is_err(), "invalid fraction type must return Err");
        let err = result.unwrap_err();
        assert!(
            err.contains("fraction must be 'atom' or 'mass'"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn test_new_valid_inputs_store_enums() {
        // g/cc aliases g/cm3; weight + kg/m3 parse to the expected enum variants.
        let m_gcc = Material::new(
            HashMap::from([("Fe56".into(), 1.0)]),
            "atom",
            "g/cc",
            Some(7.874),
        )
        .unwrap();
        assert_eq!(m_gcc.density_units, DensityUnits::GramsPerCc);
        assert_eq!(m_gcc.density_units.as_str(), "g/cm3");
        assert_eq!(m_gcc.fraction_type, FractionType::Atom);

        let m_kg = Material::new(
            HashMap::from([("Fe56".into(), 1.0)]),
            "mass",
            "kg/m3",
            Some(7874.0),
        )
        .unwrap();
        assert_eq!(m_kg.density_units, DensityUnits::KgPerM3);
        assert_eq!(m_kg.fraction_type, FractionType::Mass);
        // kg/m3 still divides by 1000 to reach g/cm3.
        assert!((m_kg.get_mass_density().unwrap() - 7.874).abs() < 1e-10);
    }

    #[test]
    fn test_density_units_and_fraction_type_parse_roundtrip() {
        // parse(as_str()) is the identity over canonical strings, and g/cc maps to g/cm3.
        assert_eq!(
            DensityUnits::parse("g/cm3").unwrap(),
            DensityUnits::GramsPerCc
        );
        assert_eq!(
            DensityUnits::parse("g/cc").unwrap(),
            DensityUnits::GramsPerCc
        );
        assert_eq!(DensityUnits::parse("kg/m3").unwrap(), DensityUnits::KgPerM3);
        assert_eq!(DensityUnits::parse("sum").unwrap(), DensityUnits::Sum);
        assert!(DensityUnits::parse("lb/ft3")
            .unwrap_err()
            .contains("Unsupported density unit: 'lb/ft3'"));

        assert_eq!(FractionType::parse("atom").unwrap(), FractionType::Atom);
        assert_eq!(FractionType::parse("mass").unwrap(), FractionType::Mass);
        assert!(FractionType::parse("volume").is_err());
        // The old "weight" spelling is no longer accepted.
        assert!(FractionType::parse("weight").is_err());
    }

    #[test]
    fn test_serde_wire_format_unchanged() {
        // The serde wire shape (MaterialSerde) keeps the canonical strings,
        // and a round-trip reconstructs the same enum-typed fields.
        let mat = Material::new(
            HashMap::from([("Fe56".into(), 1.0)]),
            "mass",
            "kg/m3",
            Some(7874.0),
        )
        .unwrap();
        let wire: MaterialSerde = mat.into();
        assert_eq!(wire.fraction_type, "mass");
        assert_eq!(wire.density_units, "kg/m3");

        let back = Material::try_from(wire).unwrap();
        assert_eq!(back.fraction_type, FractionType::Mass);
        assert_eq!(back.density_units, DensityUnits::KgPerM3);
        assert_eq!(back.density, Some(7874.0));

        // Deserializing an invalid stored unit must error (not panic).
        let bad = MaterialSerde {
            name: None,
            material_id: None,
            nuclides: HashMap::from([("Fe56".to_string(), 1.0)]),
            nuclide_input_order: None,
            fraction_type: "mass".to_string(),
            density: Some(7874.0),
            density_units: "furlong".to_string(),
            volume: None,
            temperature: String::new(),
            transmutable: false,
        };
        assert!(Material::try_from(bad).is_err());
    }

    #[test]
    fn test_average_molar_mass_ao() {
        // H2O: 2 parts H1, 1 part O16 → M_avg ≈ (2*1.007825 + 1*15.994915) / 3
        let mat = Material::new(
            HashMap::from([("H1".into(), 2.0), ("O16".into(), 1.0)]),
            "atom",
            "sum",
            None,
        )
        .unwrap();
        let amm = mat.average_molar_mass().unwrap();
        let expected = (2.0 * 1.00782503207 + 1.0 * 15.99491461956) / 3.0;
        assert!(
            (amm - expected).abs() / expected < 1e-6,
            "Expected ~{expected}, got {amm}"
        );
    }

    #[test]
    fn test_get_mass_density_gcm3() {
        let mat = Material::new(
            HashMap::from([("Fe56".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(7.874),
        )
        .unwrap();
        let rho = mat.get_mass_density().unwrap();
        assert!((rho - 7.874).abs() < 1e-10);
    }

    #[test]
    fn test_get_mass_density_kgm3() {
        let mat = Material::new(
            HashMap::from([("Fe56".into(), 1.0)]),
            "atom",
            "kg/m3",
            Some(7874.0),
        )
        .unwrap();
        let rho = mat.get_mass_density().unwrap();
        assert!((rho - 7.874).abs() < 1e-10);
    }

    #[test]
    fn test_mix_materials_vo_basic() {
        // 50% water, 50% iron by volume
        let water = Material::new(
            HashMap::from([("H1".into(), 2.0), ("O16".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(1.0),
        )
        .unwrap();

        let iron = Material::new(
            HashMap::from([("Fe56".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(7.874),
        )
        .unwrap();

        let mixed =
            Material::mix_materials(&[&water, &iron], &[0.5, 0.5], "volume", None, None).unwrap();

        let rho = mixed.get_mass_density().unwrap();
        // Expected: 0.5 * 1.0 + 0.5 * 7.874 = 4.437
        assert!(
            (rho - 4.437).abs() < 0.01,
            "Expected ~4.437 g/cm3, got {rho}"
        );
        // Check all nuclides present
        assert!(mixed.nuclides.contains_key("H1"));
        assert!(mixed.nuclides.contains_key("O16"));
        assert!(mixed.nuclides.contains_key("Fe56"));
    }

    #[test]
    fn test_mix_materials_transmutable_propagation() {
        let mut mat1 = Material::new(
            HashMap::from([("H1".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(1.0),
        )
        .unwrap();
        mat1.transmutable = true;

        let mat2 = Material::new(
            HashMap::from([("Fe56".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(7.874),
        )
        .unwrap();

        let mixed =
            Material::mix_materials(&[&mat1, &mat2], &[0.5, 0.5], "volume", None, None).unwrap();
        assert!(mixed.transmutable, "Transmutable flag should propagate");
    }

    #[test]
    fn test_mix_materials_fracs_validation() {
        let mat = Material::new(
            HashMap::from([("H1".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(1.0),
        )
        .unwrap();

        // Wrong number of fractions
        let result = Material::mix_materials(&[&mat], &[0.5, 0.5], "atom", None, None);
        assert!(result.is_err());

        // Negative fraction
        let result = Material::mix_materials(&[&mat, &mat], &[-0.5, 1.5], "atom", None, None);
        assert!(result.is_err());

        // ao fractions not summing to 1.0
        let result = Material::mix_materials(&[&mat, &mat], &[0.3, 0.3], "atom", None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_expand_formula_h2o() {
        let nuclides =
            yamc_nuclide::composition::expand_formula("H2O", "atom", None, None, None).unwrap();
        // H expands to H1, H2; O expands to O16, O17, O18
        let has_h = nuclides.keys().any(|n| n.starts_with('H'));
        let has_o = nuclides.keys().any(|n| n.starts_with('O'));
        assert!(has_h, "Should contain hydrogen isotopes");
        assert!(has_o, "Should contain oxygen isotopes");
    }

    #[test]
    fn test_expand_formula_brackets() {
        let nuclides =
            yamc_nuclide::composition::expand_formula("(NH4)2SO4", "atom", None, None, None)
                .unwrap();
        let has_n = nuclides.keys().any(|n| n.starts_with('N'));
        let has_h = nuclides.keys().any(|n| n.starts_with('H'));
        let has_s = nuclides.keys().any(|n| n.starts_with('S'));
        let has_o = nuclides.keys().any(|n| n.starts_with('O'));
        assert!(has_n, "Should contain nitrogen isotopes");
        assert!(has_h, "Should contain hydrogen isotopes");
        assert!(has_s, "Should contain sulfur isotopes");
        assert!(has_o, "Should contain oxygen isotopes");
    }

    #[test]
    fn test_expand_formula_invalid() {
        // Empty formula
        assert!(yamc_nuclide::composition::expand_formula("", "atom", None, None, None).is_err());
        // Decimal multiplier
        assert!(
            yamc_nuclide::composition::expand_formula("H2.5O", "atom", None, None, None).is_err()
        );
        // Unknown element
        assert!(
            yamc_nuclide::composition::expand_formula("Xx2O", "atom", None, None, None).is_err()
        );
        // Unbalanced parens
        assert!(
            yamc_nuclide::composition::expand_formula("(NH4", "atom", None, None, None).is_err()
        );
    }

    #[test]
    fn test_formula_parse_counts() {
        // Verify element proportions for (NH4)2SO4 via expand_formula
        // Formula: N=2, H=8, S=1, O=4 → total atoms = 15
        let nuclides =
            yamc_nuclide::composition::expand_formula("(NH4)2SO4", "atom", None, None, None)
                .unwrap();

        // Sum fractions by element
        let mut element_sums: HashMap<String, f64> = HashMap::new();
        for (name, frac) in &nuclides {
            let elem: String = name
                .chars()
                .take_while(|c| c.is_ascii_alphabetic())
                .collect();
            *element_sums.entry(elem).or_default() += frac;
        }

        let total: f64 = element_sums.values().sum();
        // Normalize and check ratios match 2:8:1:4
        assert!((element_sums["N"] / total - 2.0 / 15.0).abs() < 1e-10);
        assert!((element_sums["H"] / total - 8.0 / 15.0).abs() < 1e-10);
        assert!((element_sums["S"] / total - 1.0 / 15.0).abs() < 1e-10);
        assert!((element_sums["O"] / total - 4.0 / 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_read_nuclear_data_or_keyword_rejects_non_keyword_non_directory() {
        let mut mat =
            Material::new(HashMap::from([("Li6".into(), 1.0)]), "atom", "sum", None).unwrap();
        let err = mat
            .read_nuclear_data_or_keyword("not-a-keyword-and-not-a-dir")
            .expect_err("non-keyword, non-directory source must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("neither a registered data keyword nor a directory"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn test_keyword_load_registers_global_default_cross_section() {
        // A material loaded from a keyword/directory source must register
        // that source as the GLOBAL fallback (when none is configured) so
        // nuclides that enter the problem later by name only -- e.g. the
        // transmutation products `simulate_transmutation` preloads -- can
        // resolve cross sections. Without the fallback every product warns
        // "not found" and silently skips transport (V&V: second-generation
        // product densities collapsed 100-1000x, Ar36 -> Cl36 -> S36).
        //
        // The registration happens up front (before the per-nuclide load),
        // so the assertion is independent of the directory's contents.
        let saved = {
            let mut config = CONFIG.lock().unwrap_or_else(|p| p.into_inner());
            std::mem::take(&mut config.default_cross_section)
        };

        let dir = std::env::temp_dir();
        let dir_str = dir.to_str().unwrap().to_string();
        let mut mat =
            Material::new(HashMap::from([("Li6".into(), 1.0)]), "atom", "sum", None).unwrap();
        let _ = mat.read_nuclear_data_or_keyword(&dir_str);
        let registered = {
            let config = CONFIG.lock().unwrap_or_else(|p| p.into_inner());
            config.default_cross_section.clone()
        };

        // An explicit user configuration must WIN over the material source.
        {
            let mut config = CONFIG.lock().unwrap_or_else(|p| p.into_inner());
            config.default_cross_section = Some("user-configured".into());
        }
        let mut mat2 =
            Material::new(HashMap::from([("Li6".into(), 1.0)]), "atom", "sum", None).unwrap();
        let _ = mat2.read_nuclear_data_or_keyword(&dir_str);
        let kept = {
            let mut config = CONFIG.lock().unwrap_or_else(|p| p.into_inner());
            let kept = config.default_cross_section.clone();
            config.default_cross_section = saved;
            kept
        };

        assert_eq!(registered.as_deref(), Some(dir_str.as_str()));
        assert_eq!(kept.as_deref(), Some("user-configured"));
    }

    #[test]
    fn test_total_xs_with_urr_no_urr_nuclides_returns_smooth() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut material = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        material.unified_energy_grid_neutron = vec![1.0, 10.0, 100.0];
        material
            .macroscopic_xs_neutron
            .insert(1, vec![2.0, 2.0, 2.0]);
        let mut rng = StdRng::seed_from_u64(0);

        let (sigma_t, urr_random) = material.total_xs_with_urr(5.0, None, &mut rng);

        assert!((sigma_t - 2.0).abs() < 1e-12);
        assert!(
            urr_random.is_none(),
            "no urr_random should be returned when no nuclide has URR data"
        );
    }

    #[test]
    fn test_total_xs_with_urr_does_not_consume_rng_without_urr() {
        use rand::rngs::StdRng;
        use rand::RngExt;
        use rand::SeedableRng;
        let mut material = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        material.unified_energy_grid_neutron = vec![1.0, 10.0, 100.0];
        material
            .macroscopic_xs_neutron
            .insert(1, vec![3.0, 3.0, 3.0]);

        let mut rng_a = StdRng::seed_from_u64(123);
        let mut rng_b = StdRng::seed_from_u64(123);

        let _ = material.total_xs_with_urr(5.0, None, &mut rng_a);
        // rng_b draws nothing; the two RNGs should still be in lockstep.
        let next_a: f64 = rng_a.random();
        let next_b: f64 = rng_b.random();
        assert_eq!(
            next_a, next_b,
            "rng must not be advanced when no URR nuclide is in range"
        );
    }

    /// Issue #478: `temperature_k` was set once in `Material::new` and never
    /// again, so every material transported at 294 K whatever its label said.
    #[test]
    fn temperature_k_tracks_the_label() {
        let mut m = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        assert_eq!(
            m.temperature_k(),
            yamc_nuclide::temperature::DEFAULT_TEMPERATURE_K,
            "an unresolved material starts at room temperature"
        );

        m.set_temperature("900");
        assert_eq!(m.temperature(), "900");
        assert_eq!(
            m.temperature_k(),
            900.0,
            "the Kelvin value the free-gas kernel reads must follow the label"
        );

        m.set_temperature("600K");
        assert_eq!(
            m.temperature_k(),
            600.0,
            "the on-disk `600K` spelling must parse the same as `600`"
        );
        assert_eq!(
            m.temperature(),
            "600",
            "the label must be stored stripped, because nuclide data is keyed \
             by the stripped form and `get_temp_idx` compares exactly (#481)"
        );
    }

    /// Issue #481: `set_temperature` cleared four of the temperature-dependent
    /// caches and left the rest holding the previous temperature's data.
    ///
    /// `fast_xs` mattered most: it is the "already built, skip" guard at
    /// `model.rs`, so a material whose temperature changed after a build was
    /// skipped at setup and transported on the old tables. `fast_xs` surviving
    /// an emptied grid is also a panic in its own right, since
    /// `MaterialFastXS::lookup_total` indexes `energy_grid[0]` unguarded.
    #[test]
    fn set_temperature_clears_every_temperature_dependent_cache() {
        use std::sync::Arc;

        let mut m = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        m.set_temperature("294");

        // Fabricate every cache derived from the temperature.
        m.unified_energy_grid_neutron = vec![1.0, 10.0, 100.0];
        m.macroscopic_xs_neutron.insert(1, vec![2.0, 2.0, 2.0]);
        m.macroscopic_xs_mt_numbers = vec![1];
        m.macroscopic_xs_flat = vec![2.0, 2.0, 2.0];
        m.fast_xs = MaterialFastXS::new(&[1.0, 10.0, 100.0], &[2.0, 2.0, 2.0]);
        m.cached_microscopic_xs = Some((vec![1], Arc::new(HashMap::new())));
        m.macroscopic_xs_neutron_total_by_nuclide = Some(vec![vec![2.0, 2.0, 2.0]]);
        m.macroscopic_xs_neutron_by_nuclide = Some(vec![HashMap::new()]);

        m.set_temperature("900");

        assert!(
            m.unified_energy_grid_neutron.is_empty(),
            "the unified grid is built per temperature"
        );
        assert!(m.macroscopic_xs_neutron.is_empty(), "macroscopic XS");
        assert!(m.macroscopic_xs_mt_numbers.is_empty(), "flat buffer index");
        assert!(m.macroscopic_xs_flat.is_empty(), "flat buffer");
        assert!(
            m.fast_xs.is_none(),
            "fast_xs is the skip guard at simulation setup, and outliving the \
             grid it indexes into is a panic"
        );
        assert!(
            m.cached_microscopic_xs.is_none(),
            "cached_microscopic_xs is keyed by MT filter alone, so it carries \
             no temperature of its own and must be dropped"
        );
        assert!(
            m.macroscopic_xs_neutron_total_by_nuclide.is_none(),
            "per-nuclide totals"
        );
        assert!(
            m.macroscopic_xs_neutron_by_nuclide.is_none(),
            "per-nuclide per-MT"
        );
    }

    /// Issue #481: a query for a temperature other than the material's own used
    /// to overwrite the material's caches, then hand the label back, leaving
    /// every cache holding the queried temperature's data under the original
    /// label. The builders now take the temperature explicitly and only write
    /// through when it is the material's own.
    #[test]
    fn a_foreign_temperature_query_leaves_the_caches_alone() {
        let mut m = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        m.set_temperature("294");

        let fabricated = vec![1.0, 10.0, 100.0];
        m.unified_energy_grid_neutron = fabricated.clone();

        // 900 is not this material's temperature, so this must neither return
        // nor disturb the cached 294 grid.
        let grid_900 = m.unified_energy_grid_neutron_at("900");
        assert!(
            grid_900.is_empty(),
            "no nuclide data is loaded, so the 900 grid is empty; the point is \
             that it did not come from the 294 cache"
        );
        assert_eq!(
            m.unified_energy_grid_neutron, fabricated,
            "the material's own cached grid must survive a query at another \
             temperature"
        );
        assert_eq!(
            m.temperature(),
            "294",
            "and the label must still be the material's own"
        );

        // The material's own temperature still reads and writes through.
        let grid_294 = m.unified_energy_grid_neutron_at("294");
        assert_eq!(grid_294, fabricated, "own temperature reads the cache");
    }

    /// The two backends must not disagree about what a label means. The GPU
    /// extractor parses labels with the same function, so this pins the
    /// contract they share.
    #[test]
    fn temperature_k_matches_the_shared_parser() {
        for label in ["250", "294", "600K", "900", "1200", "2500K"] {
            let mut m = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
            m.set_temperature(label);
            assert_eq!(
                m.temperature_k(),
                yamc_nuclide::temperature::label_to_kelvin_or_default(label),
                "label {label:?} must mean one thing"
            );
        }
    }

    /// A serde round trip goes through `set_temperature`, which is exactly how
    /// the stale 294.0 used to survive being written out and read back.
    #[test]
    fn temperature_k_survives_a_serde_round_trip() {
        let mut m = Material::new(
            HashMap::from([("Fe56".to_string(), 1.0)]),
            "atom",
            "g/cm3",
            Some(7.874),
        )
        .unwrap();
        m.set_temperature("900");

        // The same conversions serde drives, without pulling in a JSON
        // dependency: `Material` is `#[serde(into/try_from = "MaterialSerde")]`.
        let wire: MaterialSerde = m.clone().into();
        let back: Material = wire.try_into().expect("round trip");

        assert_eq!(back.temperature(), "900");
        assert_eq!(
            back.temperature_k(),
            900.0,
            "temperature_k must be rebuilt from the label on deserialize"
        );
    }

    /// `from_nuclide_densities` copies the label from its template, so it has to
    /// copy a Kelvin value that agrees with it.
    #[test]
    fn temperature_k_is_consistent_after_templating() {
        let mut template = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        template.set_temperature("1200");

        let derived = Material::from_nuclide_densities(
            HashMap::from([("Fe56".to_string(), 1e-3)]),
            &template,
        );

        assert_eq!(derived.temperature(), "1200");
        assert_eq!(derived.temperature_k(), 1200.0);
        assert_eq!(
            derived.temperature_k(),
            yamc_nuclide::temperature::label_to_kelvin_or_default(derived.temperature()),
            "the copied pair must still agree"
        );
    }
}
