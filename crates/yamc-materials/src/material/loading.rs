use super::*;

impl Material {
    /// Read nuclear data (nuclide + optional photon element data) for this material.
    ///
    /// # Arguments
    /// * `nuclide_path_map` - Map of nuclide names to file paths (e.g., {"Li6": "Li6.arrow"})
    /// * `photon_data_paths` - Optional map of element symbols to photon Arrow data paths
    pub fn read_nuclear_data(
        &mut self,
        nuclide_path_map: &HashMap<String, String>,
        photon_data_paths: Option<&HashMap<String, String>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Store photon data paths if provided
        if let Some(paths) = photon_data_paths {
            self.photon_data_paths = paths.clone();
        }
        // Collect needed nuclide names
        let mut nuclide_names: Vec<String> = self.nuclides.keys().cloned().collect();
        nuclide_names.sort(); // ensure deterministic alphabetical load order

        // Build merged source map: explicit entries override, missing filled from CONFIG
        let mut merged: HashMap<String, String> = HashMap::new();

        // Start with global config entries for required nuclides, with proper error handling
        let cfg = yamc_nuclide::config::CONFIG
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        for n in &nuclide_names {
            if let Some(p) = cfg.get_cross_section(n) {
                merged.insert(n.clone(), p.clone());
            }
        }
        drop(cfg);

        // Override with any provided mapping entries (even if extra keys not in composition)
        for (k, v) in nuclide_path_map {
            merged.insert(k.clone(), v.clone());
        }
        let source_map: &HashMap<String, String> = &merged;

        // Load nuclides using the centralized function in the nuclide module
        use std::collections::HashSet;

        // Only filter by temperature if explicitly set, otherwise load all temperatures
        let temp_filter = if !self.temperature.is_empty() {
            let mut temp_set: HashSet<String> = HashSet::new();
            temp_set.insert(self.temperature.clone());
            Some(temp_set)
        } else {
            None
        };

        let photon_only = photon_data_paths.is_some() && source_map.is_empty();
        for nuclide_name in nuclide_names {
            if photon_only && !source_map.contains_key(&nuclide_name) {
                // Photon-only mode: skip nuclides without neutron data
                continue;
            }
            let nuclide = get_or_load_nuclide(
                &nuclide_name,
                source_map,
                &LoadScope::full().with_temperatures(temp_filter.clone()),
            )?;
            self.nuclide_data.insert(nuclide_name, nuclide);
        }

        // Validate that all nuclides have compatible temperatures
        self.validate_temperature_consistency()?;

        // Clear cached data since new nuclear data affects cross sections
        self.invalidate_xs_cache();
        Ok(())
    }

    /// Make sure every nuclide has `temperature` parsed in, loading it if the
    /// data offers it but the current load scope left it out.
    ///
    /// `read_nuclear_data` narrows the load to `self.temperature`, so a material
    /// built at 294 K holds only 294 K reactions even though its Arrow files
    /// carry every temperature. `available_temperatures` then advertises a
    /// temperature that `loaded_temperatures` cannot serve, and the request dies
    /// downstream as a missing MT rather than as a temperature problem (#481).
    ///
    /// The re-request goes back through `get_or_load_nuclide` with a widened
    /// scope so the global cache's own `covers`/`union` logic does the widening
    /// once, shared. `Nuclide::auto_load_additional_temperature` is the wrong
    /// lever: it takes `&mut self` against an `Arc<Nuclide>`, so calling it means
    /// `Arc::make_mut` forking this material's copy away from the global cache,
    /// and it resolves files only through the global CONFIG, which never sees
    /// the explicit paths handed to `read_nuclear_data`.
    ///
    /// Nuclides whose data does not offer the temperature at all are left alone;
    /// reporting that is `resolve_temperature`'s job, which can name every
    /// nuclide at once instead of failing on whichever came first.
    pub fn ensure_temperature_loaded(
        &mut self,
        temperature: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let wanted = yamc_nuclide::temperature::strip_k(temperature).to_string();
        if wanted.is_empty() {
            return Ok(());
        }

        let stale: Vec<String> = self
            .nuclide_data
            .iter()
            .filter(|(_, n)| {
                !n.loaded_temperatures.contains(&wanted)
                    && yamc_nuclide::temperature::resolve(&wanted, &n.available_temperatures)
                        .is_ok()
            })
            .map(|(name, _)| name.clone())
            .collect();
        if stale.is_empty() {
            return Ok(());
        }

        for name in stale {
            let (path, mut temps, scope) = {
                let nuclide = &self.nuclide_data[&name];
                let temps: std::collections::HashSet<String> =
                    nuclide.loaded_temperatures.iter().cloned().collect();
                (nuclide.data_path.clone(), temps, nuclide.load_scope.clone())
            };
            // `data_path` is stamped by the Arrow loader whichever route loaded
            // the nuclide, so it survives the explicit-path case that CONFIG
            // does not know about. Fall back to CONFIG for the nuclides that
            // reached us some other way.
            let source = match path.or_else(|| {
                let cfg = yamc_nuclide::config::CONFIG
                    .lock()
                    .unwrap_or_else(|p| p.into_inner());
                cfg.get_cross_section(&name)
            }) {
                Some(p) => p,
                None => {
                    // Skipping quietly would put the request back on the path
                    // that ends in `No cross section data found for MT n`, the
                    // original #481 symptom with no mention of temperature.
                    return Err(format!(
                        "nuclide '{name}' offers temperature '{wanted}' but only \
                         has {:?} loaded, and there is no path to load it from: \
                         no `data_path` was recorded and no cross section is \
                         configured for it",
                        self.nuclide_data[&name].loaded_temperatures
                    )
                    .into());
                }
            };
            temps.insert(wanted.clone());

            // Widen the temperature axis and nothing else. `LoadScope::full()`
            // here would union to Full with every MT, re-reading the nine
            // full-grid transport sections for a nuclide that `transmute_material`
            // deliberately loaded at activation scope (issue #401).
            let path_map = HashMap::from([(name.clone(), source)]);
            let widened =
                get_or_load_nuclide(&name, &path_map, &scope.with_temperatures(Some(temps)))?;
            self.nuclide_data.insert(name, widened);
        }

        // No invalidation. Widening only adds temperatures: the reactions and
        // grid the caches were built from are still there, unchanged, under the
        // same key. Clearing here would mean a query at some other temperature
        // destroys the caches belonging to this material's own -- the exact
        // incoherence the rest of this work exists to remove.
        Ok(())
    }

    /// Re-read any nuclide that was loaded without its MF=33 covariance.
    ///
    /// The covariance axis of [`LoadScope`] has the same problem the temperature
    /// axis has: a material built by `read_nuclear_data` holds nuclides loaded
    /// at whatever scope that call used, and `transmute_material`'s own loading
    /// loop resolves paths through the global CONFIG, which never sees the
    /// explicit paths handed to `read_nuclear_data`. Without this, asking for
    /// uncertainty on such a material reports every nuclide as having no
    /// covariance data, which is indistinguishable from an evaluation that
    /// genuinely has none.
    ///
    /// Widening goes back through `get_or_load_nuclide` so the global cache's
    /// own `covers`/`union` logic does it once and shares the result, exactly as
    /// [`Self::ensure_temperature_loaded`] does.
    ///
    /// A nuclide re-read and still carrying no covariance is NOT an error: most
    /// evaluations have no MF=33, and `covariance.arrow` is optional by
    /// construction. Nor is a nuclide with no path to re-read from -- it simply
    /// keeps what it has, and the fold's coverage report is what names it. The
    /// alternative, failing the whole transmutation because one trace daughter
    /// cannot be located, would be worse than a reported gap.
    pub fn ensure_covariance_loaded(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let narrow: Vec<String> = self
            .nuclide_data
            .iter()
            .filter(|(_, n)| !n.load_scope.covariance)
            .map(|(name, _)| name.clone())
            .collect();

        for name in narrow {
            let (path, scope) = {
                let nuclide = &self.nuclide_data[&name];
                (nuclide.data_path.clone(), nuclide.load_scope.clone())
            };
            let source = path.or_else(|| {
                let cfg = yamc_nuclide::config::CONFIG
                    .lock()
                    .unwrap_or_else(|p| p.into_inner());
                cfg.get_cross_section(&name)
            });
            let Some(source) = source else {
                continue;
            };

            // Widen the covariance axis and nothing else. `LoadScope::full()`
            // would union to Full with every MT and re-read the transport
            // sections an activation load deliberately skipped (issue #401).
            let path_map = HashMap::from([(name.clone(), source)]);
            let widened = get_or_load_nuclide(&name, &path_map, &scope.with_covariance(true))?;
            self.nuclide_data.insert(name, widened);
        }
        Ok(())
    }

    /// Read nuclear data from either a JSON mapping, a keyword string, or a directory path.
    /// Keywords and directories are applied to all nuclides in this material.
    pub fn read_nuclear_data_or_keyword(
        &mut self,
        source: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let is_keyword = yamc_nuclide::url_cache::is_keyword(source);
        if !is_keyword && !std::path::Path::new(source).is_dir() {
            return Err(format!(
                "'{source}' is neither a registered data keyword nor a directory. \
                 Pass a dict mapping nuclide names to file paths, or use a keyword \
                 (e.g. 'endf-b8.1') or a directory containing per-nuclide files."
            )
            .into());
        }

        // Apply the keyword or directory to every nuclide in this material.
        let mut source_map = HashMap::new();
        for nuclide_name in self.nuclides.keys() {
            source_map.insert(nuclide_name.clone(), source.to_string());
        }

        // For keyword form, register a photon data source for each unique
        // element symbol in this material's composition so that photon /
        // coupled transport has cross sections (see `Model::initialize_xs`
        // for the fail-loud check that depends on photon_data_paths being
        // non-empty).
        //
        // The photon library is DECOUPLED from the neutron library: we
        // default it to `endf-b8.1` regardless of the neutron keyword
        // (`source`), because endf-b8.1 ships full atomic-relaxation tables
        // (fluorescence + Auger) whereas other libraries (e.g. fendl-3.2d)
        // publish photon cross sections with no relaxation, which would
        // silently suppress fluorescence. The neutron data still comes from
        // `source`. `or_insert_with` preserves any explicit per-element
        // photon path the user already set (those win over the default).
        if is_keyword {
            use yamc_element::element::element_symbol_from_nuclide;
            let photon_source = yamc_nuclide::url_cache::DEFAULT_PHOTON_LIBRARY.to_string();
            for nuclide_name in self.nuclides.keys() {
                let elem = element_symbol_from_nuclide(nuclide_name);
                self.photon_data_paths
                    .entry(elem)
                    .or_insert_with(|| photon_source.clone());
            }
        }

        // Register the keyword / directory as the GLOBAL fallback source when
        // the user has not configured one. Nuclides that enter the problem
        // later by name only -- the transmutation products preloaded in
        // `Model::simulate_transmutation` -- resolve through
        // `Config::get_cross_section`, which knows nothing about this
        // material's source. Without a fallback every such product warns
        // "not found" and silently skips neutron transport, which collapsed
        // second-generation product densities 100-1000x in the coupled
        // depletion V&V (Ar36 -> Cl36 -> S36; Cd110 -> Cd109 -> Cd108). The
        // `is_none` guard keeps any explicit user configuration winning.
        {
            let mut config = yamc_nuclide::config::CONFIG
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if config.default_cross_section.is_none() {
                config.default_cross_section = Some(source.to_string());
            }
        }

        self.read_nuclear_data(&source_map, None)?;
        Ok(())
    }

    /// Read nuclear data from a keyword string that will be applied to all nuclides in this material
    pub fn read_nuclear_data_keyword(
        &mut self,
        keyword: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.read_nuclear_data_or_keyword(keyword)
    }

    /// Load nuclear data from extracted input data (pure Rust, no PyO3 dependencies)
    pub fn load_nuclear_data_from_input(
        &mut self,
        dict_data: Option<HashMap<String, String>>,
        keyword_data: Option<String>,
        photon_data: Option<HashMap<String, String>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(map) = dict_data {
            self.read_nuclear_data(&map, photon_data.as_ref())
        } else if let Some(keyword) = keyword_data {
            // Store photon data paths if provided
            if let Some(paths) = photon_data {
                self.photon_data_paths = paths;
            }
            self.read_nuclear_data_keyword(&keyword)
        } else {
            let empty_map = HashMap::new();
            self.read_nuclear_data(&empty_map, photon_data.as_ref())
        }
    }

    /// Directly load a nuclide from a file path and insert into nuclide_data.
    /// If the nuclide already exists it will be overwritten.
    /// Only loads the temperature needed for this material (from self.temperature field).
    pub fn load_nuclide_from_file(
        &mut self,
        nuclide_name: &str,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Only load the temperature we need (from material's temperature setting)
        // Temperature keys are stored without 'K' suffix (e.g., "294" not "294K")
        let temp_filter: Option<std::collections::HashSet<String>> = {
            let temp = yamc_nuclide::temperature::strip_k(&self.temperature);
            Some(std::iter::once(temp.to_string()).collect())
        };
        let nuclide = yamc_nuclide::nuclide_loader::load_nuclide(
            path,
            &LoadScope::full().with_temperatures(temp_filter),
        )?;
        self.nuclide_data
            .insert(nuclide_name.to_string(), Arc::new(nuclide));
        Ok(())
    }

    /// Ensure all nuclides are loaded, using the global configuration if needed
    pub fn ensure_nuclides_loaded(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Presence is not enough. An entry loaded at ACTIVATION scope carries
        // only the MTs the transmutation network names, and transport would go
        // looking for one that is not there.
        //
        // That was unreachable while `Material::transmute` worked on a private
        // clone. It stopped being so when the preload started writing back into
        // the caller's material (issue #576, finding 3): transmute a material
        // and then put it in a Model, and its composition nuclides are present
        // but too narrow.
        //
        // The test is `mts.is_some()` rather than `has_transport_data`, which
        // also demands `SectionScope::Full`. A trimmed data directory carrying
        // none of the transport sections is recorded as `XsOnly` by
        // `narrow_to_present_sections` even when the caller asked for
        // everything, so `has_transport_data` is false for a nuclide that was
        // loaded as fully as its directory allows -- and reloading those on
        // every call would be a regression, not a fix. Only the activation
        // preload sets a concrete MT set.
        let nuclide_names: Vec<String> = self
            .nuclides
            .keys()
            .filter(|name| match self.nuclide_data.get(*name) {
                None => true,
                Some(n) => n.load_scope.mts.is_some(),
            })
            .cloned()
            .collect();

        if nuclide_names.is_empty() {
            return Ok(());
        }

        // Get the global configuration with proper error handling
        let config = CONFIG
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        // In photon-only mode (photon data loaded but no neutron data), skip
        // nuclides that have no neutron data path available.
        let photon_only = !self.photon_data_paths.is_empty() && self.nuclide_data.is_empty();

        // Load any missing nuclides
        for nuclide_name in nuclide_names {
            // Build a temporary source map. The nuclide's own `data_path`
            // first, the way `ensure_temperature_loaded` widens: it is stamped
            // by the Arrow loader whichever route loaded the entry, so a
            // material built by `read_nuclear_data` from explicit paths widens
            // from those rather than from whatever CONFIG happens to name.
            let mut source_map = HashMap::new();
            let recorded = self
                .nuclide_data
                .get(&nuclide_name)
                .and_then(|n| n.data_path.clone());
            if let Some(path) = recorded.or_else(|| config.get_cross_section(&nuclide_name)) {
                source_map.insert(nuclide_name.clone(), path);
            }

            if photon_only && source_map.is_empty() {
                // Photon-only mode: skip nuclides without neutron data
                continue;
            }

            // Load nuclide with all available temperatures (no filter)
            // Temperature validation is deferred to resolve_temperature() which uses
            // fallback logic to find a valid temperature
            match get_or_load_nuclide(&nuclide_name, &source_map, &LoadScope::full()) {
                Ok(nuclide) => {
                    self.nuclide_data.insert(nuclide_name.clone(), nuclide);
                }
                Err(e) => {
                    return Err(format!("Failed to load nuclide '{nuclide_name}': {e}").into());
                }
            }
        }

        // Clear cached data since new nuclear data affects cross sections
        self.invalidate_xs_cache();
        Ok(())
    }

    /// Whether this material holds data for `nuclide_name` that transport can
    /// use: every section and every MT.
    ///
    /// Presence in `nuclide_data` is not the same question. A transmutation
    /// preload puts activation-scope entries there, which carry the network's
    /// cross sections but none of the transport sections (issue #401).
    ///
    /// Only the section and MT axes are tested. Temperature coverage is left
    /// alone: `LoadScope::full()` names every temperature, so testing that axis
    /// would call every temperature-filtered load insufficient.
    pub fn has_transport_data(&self, nuclide_name: &str) -> bool {
        self.nuclide_data
            .get(nuclide_name)
            .is_some_and(|n| n.load_scope.wants_transport_sections() && n.load_scope.mts.is_none())
    }

    /// Load nuclear data for a single nuclide at transport scope, if what is
    /// already held does not reach that far.
    /// Does NOT invalidate caches (caller is responsible).
    ///
    /// Presence alone is not enough to skip the load: transmutation preloads
    /// its reachable products at activation scope, which carries the network's
    /// cross sections but none of the transport sections. When such a product
    /// later grows a density and enters transport, it has to be widened here
    /// (issue #401).
    /// Returns whether data was actually read, so a caller can skip the
    /// cross-section rebuild that only a real load makes necessary.
    pub fn ensure_nuclide_loaded(
        &mut self,
        nuclide_name: &str,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        if self.has_transport_data(nuclide_name) {
            return Ok(false);
        }

        let config = CONFIG
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let mut source_map = HashMap::new();
        if let Some(path) = config.get_cross_section(nuclide_name) {
            source_map.insert(nuclide_name.to_string(), path);
        }

        match get_or_load_nuclide(nuclide_name, &source_map, &LoadScope::full()) {
            Ok(nuclide) => {
                self.nuclide_data.insert(nuclide_name.to_string(), nuclide);
                Ok(true)
            }
            Err(e) => Err(format!("Failed to load nuclide '{nuclide_name}': {e}").into()),
        }
    }
}
