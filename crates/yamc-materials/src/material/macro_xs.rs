use super::*;

impl Material {
    /// Fast lookup of macroscopic heating cross section (MT 301) at given energy.
    /// Uses the same O(1) fast_xs grid index as total cross section.
    /// Returns 0.0 if heating xs not available.
    #[inline]
    pub fn lookup_heating_xs(&self, energy: f64) -> f64 {
        self.lookup_xs_by_mt(301, energy)
    }

    /// Fast lookup of macroscopic heating-local cross section (MT 901) at given energy.
    /// Uses the same O(1) fast_xs grid index as total cross section.
    /// Returns 0.0 if heating-local xs not available.
    #[inline]
    pub fn lookup_heating_local_xs(&self, energy: f64) -> f64 {
        self.lookup_xs_by_mt(901, energy)
    }

    /// URR-aware macroscopic cross section for a given MT.
    ///
    /// When `urr` is `Some`, MTs 1 / 2 / 18 / 27 / 102 (total / elastic /
    /// fission / absorption / capture) read from the URR sample so
    /// reaction-rate scoring stays consistent with the transport ladder.
    /// All other MTs -- and the no-URR case -- fall through to the smooth
    /// `lookup_xs_by_mt` table. This is the single source of truth used
    /// by both the track-length and collision tally paths.
    #[inline]
    pub fn macro_xs_by_mt(&self, mt: i32, energy: f64, urr: Option<&UrrMacroXs>) -> f64 {
        match (urr, mt) {
            (Some(u), 1) => u.total,
            (Some(u), 2) => u.elastic,
            (Some(u), 18) => u.fission,
            (Some(u), 27) => u.absorption,
            (Some(u), 102) => u.capture,
            _ => self.lookup_xs_by_mt(mt, energy),
        }
    }

    /// Fast lookup of macroscopic cross section for any MT at given energy.
    /// Uses the same O(1) fast_xs grid index as total cross section.
    /// Returns 0.0 if the MT is not available.
    #[inline]
    pub fn lookup_xs_by_mt(&self, mt: i32, energy: f64) -> f64 {
        // Use fast O(1) lookup if available
        if let Some(ref fast_xs) = self.fast_xs {
            // Reuse the grid index from fast_xs
            let (_, i_grid, f) = fast_xs.lookup_total(energy, &self.unified_energy_grid_neutron);

            // Use flat buffer when available (GPU-friendly contiguous layout)
            let n_mts = self.macroscopic_xs_mt_numbers.len();
            if let Some(n_energies) = self.macroscopic_xs_flat.len().checked_div(n_mts) {
                if let Ok(mt_idx) = self.macroscopic_xs_mt_numbers.binary_search(&mt) {
                    if i_grid + 1 < n_energies {
                        let xs0 = self.macroscopic_xs_flat[i_grid * n_mts + mt_idx];
                        let xs1 = self.macroscopic_xs_flat[(i_grid + 1) * n_mts + mt_idx];
                        return xs0 + f * (xs1 - xs0);
                    }
                }
            } else if let Some(xs_vec) = self.macroscopic_xs_neutron.get(&mt) {
                // Fallback to HashMap when flat buffer not yet built
                if i_grid + 1 < xs_vec.len() {
                    let xs0 = xs_vec[i_grid];
                    let xs1 = xs_vec[i_grid + 1];
                    return xs0 + f * (xs1 - xs0);
                }
            }
        } else {
            // Fallback to slow interpolation
            if let Some(xs_vec) = self.macroscopic_xs_neutron.get(&mt) {
                return crate::interpolate_linear(
                    &self.unified_energy_grid_neutron,
                    xs_vec,
                    energy,
                );
            }
        }
        0.0
    }

    /// Look up production cross section (MT 203-207) for the given production type.
    /// Returns 0.0 if the production cross section is not available.
    #[inline]
    pub fn lookup_production_xs(&self, production_mt: i32, energy: f64) -> f64 {
        self.lookup_xs_by_mt(production_mt, energy)
    }

    /// Fast lookup of macroscopic damage-energy cross section (MT 444) at given energy.
    /// Uses the same O(1) fast_xs grid index as total cross section.
    /// Returns 0.0 if damage-energy xs not available.
    #[inline]
    pub fn lookup_damage_energy_xs(&self, energy: f64) -> f64 {
        self.lookup_xs_by_mt(444, energy)
    }

    /// Build a unified energy grid for all nuclides for neutrons across all MT reactions
    /// This method also stores the result in the material's unified_energy_grid_neutron property
    pub fn unified_energy_grid_neutron(&mut self) -> Vec<f64> {
        // Ensure nuclides are loaded before proceeding. This has to happen
        // before `resolve_temperature`, which reports photon-only mode when
        // `nuclide_data` is empty.
        if let Err(e) = self.ensure_nuclides_loaded() {
            panic!("Error loading nuclides: {e}");
        }
        let temperature = self.resolve_temperature(None);
        self.unified_energy_grid_neutron_at(&temperature)
    }

    /// The unified grid at `temperature`, caching it only when that is the
    /// material's own temperature.
    ///
    /// The gate is the point. A query for some other temperature computes into
    /// locals and stores nothing, so it cannot leave the cache holding one
    /// temperature's grid while the label says another (#481).
    pub(crate) fn unified_energy_grid_neutron_at(&mut self, temperature: &str) -> Vec<f64> {
        // Ensure nuclides are loaded before proceeding
        if let Err(e) = self.ensure_nuclides_loaded() {
            panic!("Error loading nuclides: {e}");
        }
        // An unlabelled material has no temperature of its own to protect, and
        // the caller has just named one. Adopt it, so the derived caches are
        // stored rather than thrown away: without this a default-constructed
        // Material queried at an explicit temperature caches nothing, and
        // `sample_interacting_nuclide` then panics on its own `expect` (#481).
        if self.temperature.is_empty() {
            self.assign_temperature(temperature);
        }
        let store = temperature == self.temperature.as_str();

        // Check the cache before doing any work. The widening below cannot
        // change a hit: a cached grid at this material's own temperature means
        // that temperature was already loaded.
        if store && !self.unified_energy_grid_neutron.is_empty() {
            return self.unified_energy_grid_neutron.clone();
        }

        // A nuclide may advertise this temperature without having parsed it in,
        // because the original load was narrowed to the material's own. Widen
        // before validating, or a temperature the data plainly contains is
        // reported as unavailable (#481). This is the single choke point: the
        // microscopic and macroscopic builders both reach the grid first.
        if let Err(e) = self.ensure_temperature_loaded(temperature) {
            panic!("Error loading temperature '{temperature}': {e}");
        }

        let temperature = temperature.to_string();

        // Validate temperature availability before building grid.
        // `ensure_temperature_loaded` ran above, so anything the data offers is
        // now parsed in and this only rejects temperatures genuinely absent.
        for (nuclide_name, nuclide_data) in &self.nuclide_data {
            if !nuclide_data.available_temperatures.contains(&temperature) {
                let mut available: Vec<String> = nuclide_data.available_temperatures.clone();
                available.sort();
                let temp_list = if available.is_empty() {
                    "NONE".to_string()
                } else {
                    available
                        .iter()
                        .map(|s| format!("'{s}'"))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                panic!(
                    "Temperature '{temperature}' not available for nuclide '{nuclide_name}'. Available temperatures: {temp_list}"
                );
            }
        }

        // If not cached, build the grid
        let mut all_energies = Vec::new();
        let _particle = "neutron"; // This is now specifically for neutrons

        for nuclide in self.nuclides.keys() {
            if let Some(nuclide_data) = self.nuclide_data.get(nuclide) {
                // Check if there's a top-level energy grid
                if let Some(energy_map) = &nuclide_data.energy {
                    if let Some(energy_grid) = energy_map.get(&temperature) {
                        all_energies.extend(energy_grid);
                    }
                }
            }
        }

        // Sort and deduplicate
        all_energies.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());
        all_energies.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

        // Cache the result, but only when it belongs to this material's own
        // temperature (see the doc comment).
        if store {
            self.unified_energy_grid_neutron = all_energies.clone();
        }

        all_energies
    }

    /// Calculate microscopic cross sections for neutrons on the unified energy grid
    ///
    /// This method interpolates the microscopic cross sections for each nuclide
    /// onto the unified energy grid for all available MT reactions, or only for the specified MTs if provided.
    /// If mt_filter is Some, only those MTs will be included (by int match).
    /// Returns a nested HashMap: nuclide -> mt -> cross_section values
    pub fn calculate_microscopic_xs_neutron(
        &mut self,
        mt_filter: Option<&Vec<i32>>,
    ) -> HashMap<String, HashMap<i32, Vec<f64>>> {
        // Ensure nuclides are loaded before proceeding
        if let Err(e) = self.ensure_nuclides_loaded() {
            panic!("Error loading nuclides: {e}");
        }
        let temperature = self.resolve_temperature(None);
        self.calculate_microscopic_xs_neutron_at(&temperature, mt_filter)
    }

    /// Microscopic cross sections interpolated onto the grid for `temperature`.
    ///
    /// Takes the temperature explicitly so the grid it interpolates onto and
    /// the reactions it reads come from the same temperature. Reading the grid
    /// from a cache filled at one temperature while looking reactions up at
    /// another produced vectors of mismatched length (#481).
    pub(crate) fn calculate_microscopic_xs_neutron_at(
        &mut self,
        temperature: &str,
        mt_filter: Option<&Vec<i32>>,
    ) -> HashMap<String, HashMap<i32, Vec<f64>>> {
        // Ensure nuclides are loaded before proceeding
        if let Err(e) = self.ensure_nuclides_loaded() {
            panic!("Error loading nuclides: {e}");
        }

        let grid = self.unified_energy_grid_neutron_at(temperature);
        let mut micro_xs: HashMap<String, HashMap<i32, Vec<f64>>> = HashMap::new();
        let temperature = temperature.to_string();

        // Unified logic: iterate all reactions; if a filter is provided skip non-matching MTs.
        let mt_set_opt: Option<std::collections::HashSet<i32>> =
            mt_filter.map(|v| v.iter().copied().collect());
        for nuclide_name in self.nuclides.keys() {
            if let Some(nuclide_data) = self.nuclide_data.get(nuclide_name) {
                let mut nuclide_reactions_map: HashMap<i32, Vec<f64>> = HashMap::new();
                if let Some(temp_reactions) = nuclide_data.reactions_for_temp(&temperature) {
                    if let Some(energy_map) = &nuclide_data.energy {
                        if let Some(energy_grid) = energy_map.get(&temperature) {
                            for (&mt, reaction) in temp_reactions {
                                if let Some(ref set) = mt_set_opt {
                                    if !set.contains(&mt) {
                                        continue;
                                    }
                                }
                                let threshold_idx = reaction.threshold_idx;
                                if threshold_idx < energy_grid.len() {
                                    let reaction_energy = &energy_grid[threshold_idx..];
                                    if reaction.cross_section.len() == reaction_energy.len() {
                                        let mut xs_values = Vec::with_capacity(grid.len());
                                        for &grid_energy in &grid {
                                            if grid_energy < reaction_energy[0] {
                                                xs_values.push(0.0);
                                            } else {
                                                let xs = interpolate_linear(
                                                    reaction_energy,
                                                    &reaction.cross_section,
                                                    grid_energy,
                                                );
                                                xs_values.push(xs);
                                            }
                                        }
                                        nuclide_reactions_map.insert(mt, xs_values);
                                    }
                                }
                            }
                        }
                    }
                }
                if !nuclide_reactions_map.is_empty() {
                    micro_xs.insert(nuclide_name.clone(), nuclide_reactions_map);
                }
            }
        }
        micro_xs
    }

    /// Calculate macroscopic cross sections for neutrons on the unified energy grid
    ///
    /// This method calculates the total macroscopic cross section by:
    /// 1. Interpolating the microscopic cross sections onto the unified grid
    /// 2. Multiplying by atom density for each nuclide
    /// 3. Summing over all nuclides
    ///
    /// If mt_filter is Some, only those MTs will be included (by string match).
    /// If by_nuclide is true, populates the struct field with per-nuclide macroscopic total xs (MT=1) on the unified grid.
    pub fn calculate_macroscopic_xs(
        &mut self,
        mt_filter: &Vec<i32>,
        by_nuclide: bool,
    ) -> (Vec<f64>, HashMap<i32, Vec<f64>>) {
        // Ensure nuclides are loaded before proceeding. Must precede
        // `resolve_temperature`, which reports photon-only mode on empty data.
        if let Err(e) = self.ensure_nuclides_loaded() {
            panic!("Error loading nuclides: {e}");
        }
        let temperature = self.resolve_temperature(None);
        self.calculate_macroscopic_xs_at(&temperature, mt_filter, by_nuclide)
    }

    /// Macroscopic cross sections at `temperature`.
    ///
    /// Every cache write below is gated on `temperature` being the material's
    /// own, so a query for another temperature computes into locals and leaves
    /// the material untouched. This replaces a set-temperature / compute /
    /// restore-label sequence that left all thirteen derived caches holding the
    /// queried temperature's data under the original label, and that lost the
    /// original label entirely if the computation panicked (#481).
    pub(crate) fn calculate_macroscopic_xs_at(
        &mut self,
        temperature: &str,
        mt_filter: &Vec<i32>,
        by_nuclide: bool,
    ) -> (Vec<f64>, HashMap<i32, Vec<f64>>) {
        // Ensure nuclides are loaded before proceeding
        if let Err(e) = self.ensure_nuclides_loaded() {
            panic!("Error loading nuclides: {e}");
        }

        // Adopt an empty label for the same reason as the grid builder, and
        // before computing `store`, so the two agree.
        if self.temperature.is_empty() {
            self.assign_temperature(temperature);
        }
        let store = temperature == self.temperature.as_str();

        // Get the energy grid
        let energy_grid = self.unified_energy_grid_neutron_at(temperature);
        // ...existing code...

        // If by_nuclide is true, ensure MT=1 is in the filter
        if by_nuclide && !mt_filter.contains(&1) {
            panic!("If by_nuclide is true, mt_filter must contain 1 (total). Otherwise, per-nuclide total cross section makes no sense.");
        }

        // Use cached microscopic XS if available and covering all requested MTs.
        // Only this material's own temperature may read or write that cache:
        // the key holds the MT filter alone, so honouring it for a foreign
        // temperature returns the wrong temperature's data on the wrong grid.
        let cache_valid = store
            && match &self.cached_microscopic_xs {
                Some((cached_filter, _)) => mt_filter.iter().all(|mt| cached_filter.contains(mt)),
                None => false,
            };
        let micro_xs: Arc<HashMap<String, HashMap<i32, Vec<f64>>>> = if cache_valid {
            Arc::clone(&self.cached_microscopic_xs.as_ref().unwrap().1)
        } else {
            let computed = self.calculate_microscopic_xs_neutron_at(temperature, Some(mt_filter));
            let arc = Arc::new(computed);
            if store {
                self.cached_microscopic_xs = Some((mt_filter.clone(), Arc::clone(&arc)));
            }
            arc
        };

        // Create a map to hold macroscopic cross sections for each MT (i32)
        let mut macro_xs: HashMap<i32, Vec<f64>> = HashMap::new();
        // Find all unique MT numbers across all nuclides (as i32)
        let mut all_mts = std::collections::HashSet::new();
        for nuclide_data in micro_xs.values() {
            for &mt in nuclide_data.keys() {
                if mt_filter.contains(&mt) {
                    all_mts.insert(mt);
                }
            }
        }
        // Get the grid length (from any MT reaction of any nuclide, all should have same length)
        let grid_length = micro_xs
            .values()
            .next()
            .and_then(|xs| xs.values().next())
            .map_or(0, |v| v.len());
        // Initialize macro_xs with zeros for each MT
        for &mt in &all_mts {
            macro_xs.insert(mt, vec![0.0; grid_length]);
        }
        // Calculate macroscopic cross section for each MT
        // Get atoms per barn-cm for all nuclides
        let atoms_per_bcm_map = self
            .get_atoms_per_barn_cm()
            .unwrap_or_else(|e| panic!("{e}"));
        self.cached_atoms_per_barn_cm = Some(atoms_per_bcm_map.clone());
        // Optionally: collect per-nuclide macroscopic total xs (MT=1) if requested.
        // We accumulate into a temporary HashMap, then convert to a dense Vec
        // aligned with sorted_nuclide_keys for O(1) hot-path access.
        let mut by_nuclide_tmp: Option<HashMap<String, Vec<f64>>> = if by_nuclide {
            Some(HashMap::new())
        } else {
            None
        };

        // Name order, not `HashMap` order. Every nuclide accumulates into the
        // same `macro_values[i]` below, so the iteration order IS the summation
        // order, and float addition is not associative: walking the map
        // directly moved the last bit of the macroscopic cross section at most
        // grid points from one run to the next. Measured on the eight-nuclide
        // steel of `tests/macro_xs_reproducibility.rs`, 76352 of 151285 points
        // of MT 1 differed between two builds in a single process (#598).
        //
        // Same defect as #502 (`matrix.rs`), #576 (`composition.rs`) and the
        // four sites of #597. This one was held back from that sweep because it
        // feeds transport: it is the cross section a collision samples against,
        // so re-ordering the sum moves transport results in the last bit.
        //
        // The rest of this file already sorts wherever it builds an index
        // (the energy grid, `sorted_nuclide_keys`, `macroscopic_xs_mt_numbers`);
        // the one place taking a cross-nuclide float sum did not.
        let mut nuclide_names: Vec<&String> = self.nuclides.keys().collect();
        nuclide_names.sort_unstable();

        for nuclide in nuclide_names {
            let atoms_per_bcm = atoms_per_bcm_map.get(nuclide);
            let nuclide_data = micro_xs.get(nuclide);
            // Always try to store per-nuclide MT=1 if by_nuclide is true
            if let Some(by_nuclide_map) = by_nuclide_tmp.as_mut() {
                if let (Some(nuclide_data), Some(atoms_per_bcm)) = (nuclide_data, atoms_per_bcm) {
                    if let Some(xs_values) = nuclide_data.get(&1) {
                        let macro_vec: Vec<f64> =
                            xs_values.iter().map(|&xs| atoms_per_bcm * xs).collect();
                        by_nuclide_map.insert(nuclide.clone(), macro_vec);
                    } else {
                        by_nuclide_map.insert(nuclide.clone(), vec![0.0; energy_grid.len()]);
                    }
                } else {
                    by_nuclide_map.insert(nuclide.clone(), vec![0.0; energy_grid.len()]);
                }
            }
            if let (Some(nuclide_data), Some(atoms_per_bcm)) = (nuclide_data, atoms_per_bcm) {
                for (&mt, xs_values) in nuclide_data {
                    if let Some(macro_values) = macro_xs.get_mut(&mt) {
                        for (i, &xs) in xs_values.iter().enumerate() {
                            macro_values[i] += atoms_per_bcm * xs;
                        }
                    }
                }
            }
        }

        // Everything below writes derived state onto the material, so it
        // runs only when this really is the material's own temperature.
        // A foreign-temperature query returns its numbers and leaves no
        // trace (#481).
        if store {
            // If by_nuclide was requested, convert HashMap to dense Vec aligned with sorted keys
            if by_nuclide {
                if let Some(mut map) = by_nuclide_tmp {
                    let mut keys: Vec<String> = map.keys().cloned().collect();
                    keys.sort();
                    // Build sorted_nuclides aligned with the sorted keys so the hot
                    // path can index directly (skipping a HashMap<String, _> lookup
                    // per collision). Missing entries are silently dropped -- the
                    // HashMap access was already fallible in the same way.
                    let nuclides: Vec<Arc<Nuclide>> = keys
                        .iter()
                        .filter_map(|k| self.nuclide_data.get(k).cloned())
                        .collect();
                    self.sorted_nuclides = if nuclides.len() == keys.len() {
                        Some(nuclides)
                    } else {
                        None
                    };
                    // Convert HashMap to Vec in sorted key order
                    let by_nuclide_vec: Vec<Vec<f64>> = keys
                        .iter()
                        .map(|k| map.remove(k).unwrap_or_default())
                        .collect();
                    self.sorted_nuclide_keys = Some(keys);
                    self.macroscopic_xs_neutron_total_by_nuclide = Some(by_nuclide_vec);
                } else {
                    self.sorted_nuclide_keys = None;
                    self.sorted_nuclides = None;
                    self.macroscopic_xs_neutron_total_by_nuclide = None;
                }
            } else {
                self.macroscopic_xs_neutron_total_by_nuclide = None;
                self.sorted_nuclide_keys = None;
                self.sorted_nuclides = None;
            }

            // Build element atom density cache for the photon hot path
            if !self.cached_elements.is_empty() {
                self.cached_element_atom_densities = self
                    .cached_elements
                    .iter()
                    .map(|(name, _)| atoms_per_bcm_map.get(name).copied().unwrap_or(0.0))
                    .collect();
            }
            // Cache the results in the material
            self.macroscopic_xs_neutron = macro_xs.clone();

            // Build fast O(1) energy lookup structure if we have total XS (MT=1)
            if let Some(total_xs) = macro_xs.get(&1) {
                self.fast_xs = MaterialFastXS::new(&energy_grid, total_xs);
            }

            // Build flat row-major buffer for macroscopic XS (GPU-friendly layout)
            {
                let mut mt_numbers: Vec<i32> = macro_xs.keys().copied().collect();
                mt_numbers.sort();
                let n_mts = mt_numbers.len();
                let n_energies = energy_grid.len();
                let mut flat = vec![0.0; n_energies * n_mts];
                for (j, &mt) in mt_numbers.iter().enumerate() {
                    if let Some(xs_vec) = macro_xs.get(&mt) {
                        for (i, &xs) in xs_vec.iter().enumerate() {
                            flat[i * n_mts + j] = xs;
                        }
                    }
                }
                self.macroscopic_xs_mt_numbers = mt_numbers;
                self.macroscopic_xs_flat = flat;
            }
        }

        // All hierarchical MTs are now constructed in Python and present in the JSON files.
        // No need to generate or copy hierarchical MTs here.
        (energy_grid, macro_xs)
    }

    /// Populate per-nuclide macroscopic cross sections for all MTs in the current cache.
    ///
    /// After `calculate_macroscopic_xs` has been called, this method builds
    /// `macroscopic_xs_neutron_by_nuclide`: nuclide → (MT → Vec<f64>) using the
    /// same microscopic XS cache and atom densities.
    ///
    /// Called by the model when any tally has `nuclides` set.
    pub fn populate_per_nuclide_xs(&mut self) {
        let atoms_per_bcm_map = match &self.cached_atoms_per_barn_cm {
            Some(m) => m.clone(),
            None => {
                let m = self
                    .get_atoms_per_barn_cm()
                    .unwrap_or_else(|e| panic!("{e}"));
                self.cached_atoms_per_barn_cm = Some(m.clone());
                m
            }
        };

        let micro_xs = match &self.cached_microscopic_xs {
            Some((_, arc)) => Arc::clone(arc),
            None => return, // No microscopic XS computed yet
        };

        // Build per-nuclide data into a temporary HashMap, then convert to a dense
        // Vec aligned with sorted_nuclide_keys.
        let grid_length = self.unified_energy_grid_neutron.len();
        let mut by_nuc_tmp: HashMap<String, HashMap<i32, Vec<f64>>> = HashMap::new();

        for nuclide in self.nuclides.keys() {
            let atoms_per_bcm = match atoms_per_bcm_map.get(nuclide) {
                Some(&v) => v,
                None => continue,
            };
            let nuc_micro = match micro_xs.get(nuclide) {
                Some(d) => d,
                None => continue,
            };

            let mut nuc_macro: HashMap<i32, Vec<f64>> = HashMap::new();
            for (&mt, xs_values) in nuc_micro {
                let macro_vec: Vec<f64> = xs_values.iter().map(|&xs| atoms_per_bcm * xs).collect();
                nuc_macro.insert(mt, macro_vec);
            }
            // Ensure a zero vector for any MT in the material total that this nuclide doesn't have
            for &mt in self.macroscopic_xs_neutron.keys() {
                nuc_macro
                    .entry(mt)
                    .or_insert_with(|| vec![0.0; grid_length]);
            }
            by_nuc_tmp.insert(nuclide.clone(), nuc_macro);
        }

        // Convert to Vec aligned with sorted_nuclide_keys
        let by_nuc_vec: Vec<HashMap<i32, Vec<f64>>> = match &self.sorted_nuclide_keys {
            Some(keys) => keys
                .iter()
                .map(|k| by_nuc_tmp.remove(k).unwrap_or_default())
                .collect(),
            None => {
                // No sorted keys -- build our own sorted order
                let mut keys: Vec<String> = by_nuc_tmp.keys().cloned().collect();
                keys.sort();
                keys.iter()
                    .map(|k| by_nuc_tmp.remove(k).unwrap_or_default())
                    .collect()
            }
        };

        self.macroscopic_xs_neutron_by_nuclide = Some(by_nuc_vec);
    }

    /// Look up the macroscopic cross section for a specific nuclide and MT at given energy.
    ///
    /// Uses the same O(1) fast grid index as `lookup_xs_by_mt`.
    /// Returns 0.0 if the nuclide, MT, or energy is not available.
    #[inline]
    pub fn lookup_xs_by_mt_for_nuclide(&self, nuclide: &str, mt: i32, energy: f64) -> f64 {
        if let Some(ref by_nuc) = self.macroscopic_xs_neutron_by_nuclide {
            // Binary search the sorted nuclide keys to find the index
            let nuc_xs = self
                .sorted_nuclide_keys
                .as_ref()
                .and_then(|keys| keys.binary_search_by(|k| k.as_str().cmp(nuclide)).ok())
                .and_then(|idx| by_nuc.get(idx));
            if let Some(nuc_xs) = nuc_xs {
                if let Some(xs_vec) = nuc_xs.get(&mt) {
                    // Reuse fast O(1) lookup if available
                    if let Some(ref fast_xs) = self.fast_xs {
                        let (_, i_grid, f) =
                            fast_xs.lookup_total(energy, &self.unified_energy_grid_neutron);
                        if i_grid + 1 < xs_vec.len() {
                            return xs_vec[i_grid] + f * (xs_vec[i_grid + 1] - xs_vec[i_grid]);
                        }
                    }
                    // Fallback to linear interpolation
                    return crate::interpolate_linear(
                        &self.unified_energy_grid_neutron,
                        xs_vec,
                        energy,
                    );
                }
            }
        }
        0.0
    }

    /// Calculate macroscopic neutron cross sections with flexible reaction parameter.
    ///
    /// This method calculates the macroscopic cross section by accepting either
    /// integer MT numbers or string reaction names (like "(n,gamma)", "fission").
    ///
    /// # Arguments
    /// * `reaction` - Either an integer MT number or a string reaction name
    /// * `temperature` - Optional temperature string. If not provided, uses fallback chain:
    ///   material.temperature -> single available temperature -> error
    ///
    /// # Returns
    /// * A tuple of (cross_section_values, energy_grid)
    pub fn macroscopic_cross_section<R>(
        &mut self,
        reaction: R,
        temperature: Option<&str>,
    ) -> (Vec<f64>, Vec<f64>)
    where
        R: Into<yamc_nuclide::nuclide::ReactionIdentifier>,
    {
        // Ensure nuclides are loaded first (needed for temperature resolution)
        if let Err(e) = self.ensure_nuclides_loaded() {
            panic!("Error loading nuclides: {e}");
        }

        // Resolve temperature with fallback chain
        let resolved_temp = self.resolve_temperature(temperature);

        // Convert reaction identifier to MT number
        let reaction_id: yamc_nuclide::nuclide::ReactionIdentifier = reaction.into();
        let mt = match reaction_id {
            yamc_nuclide::nuclide::ReactionIdentifier::Mt(mt_num) => mt_num,
            yamc_nuclide::nuclide::ReactionIdentifier::Name(name) => {
                yamc_nuclide::data::REACTION_MT
                    .get(name.as_str())
                    .copied()
                    .unwrap_or_else(|| panic!("Unknown reaction name '{name}'"))
            }
        };

        // Calculate macroscopic cross sections for this MT.
        // Always pass by_nuclide=true so per-nuclide data is available for
        // downstream methods like sample_interacting_nuclide.
        // by_nuclide requires MT=1 (total) in the filter, so always include it.
        let mut mt_filter = vec![mt];
        if mt != 1 {
            mt_filter.push(1);
        }
        let (energy_grid, xs_map) =
            self.calculate_macroscopic_xs_at(&resolved_temp, &mt_filter, true);

        // Extract the cross section for the requested MT
        let xs_values = xs_map
            .get(&mt)
            .unwrap_or_else(|| panic!("No cross section data found for MT {mt}"))
            .clone();

        (xs_values, energy_grid)
    }

    /// Calculate the neutron mean free path at a given energy
    ///
    /// This method calculates the mean free path of a neutron at a specific energy
    /// by interpolating the total macroscopic cross section and then taking 1/Σ.
    ///
    /// If the total macroscopic cross section hasn't been calculated yet, it will
    /// automatically call calculate_total_xs_neutron() first.
    ///
    /// # Arguments
    /// * `energy` - The energy of the neutron in eV
    ///
    /// # Returns
    /// * The mean free path in cm, or None if there's no cross section data
    pub fn mean_free_path_neutron(&mut self, energy: f64) -> Option<f64> {
        // Ensure we have a total cross section
        if !self.macroscopic_xs_neutron.contains_key(&1) {
            let mt_filter = vec![1];
            self.calculate_macroscopic_xs(&mt_filter, false);
        }
        // If we still don't have a total cross section, return None
        if !self.macroscopic_xs_neutron.contains_key(&1) {
            return None;
        }
        // Get the total cross section and energy grid
        let total_xs = &self.macroscopic_xs_neutron[&1];

        // If we have an empty cross section array, return None
        if total_xs.is_empty() || self.unified_energy_grid_neutron.is_empty() {
            return None;
        }

        // Make sure the energy grid and cross section have the same length
        if total_xs.len() != self.unified_energy_grid_neutron.len() {
            eprintln!("Error: Energy grid and cross section lengths don't match");
            return None;
        }

        // Interpolate to get the cross section at the requested energy
        // Using linear-linear interpolation
        let cross_section = interpolate_linear(&self.unified_energy_grid_neutron, total_xs, energy);

        // Mean free path = 1/Σ
        // Check for zero to avoid division by zero
        if cross_section <= 0.0 {
            None
        } else {
            Some(1.0 / cross_section)
        }
    }

    /// Returns a sorted list of all unique MT numbers available in this material (across all nuclides).
    /// Ensures all nuclide JSON data is loaded.
    pub fn reaction_mts(&mut self) -> Result<Vec<i32>, Box<dyn std::error::Error>> {
        // Ensure all nuclides are loaded using the global config
        self.ensure_nuclides_loaded()?;
        let mut mt_set = std::collections::HashSet::new();
        for nuclide in self.nuclide_data.values() {
            if let Some(mts) = nuclide.reaction_mts() {
                for mt in mts {
                    mt_set.insert(mt);
                }
            }
        }
        let mut mt_vec: Vec<i32> = mt_set.into_iter().collect();
        mt_vec.sort();
        Ok(mt_vec)
    }
}
