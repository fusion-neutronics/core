use super::*;

impl Material {
    // =========================================================================
    // Photon cross-section infrastructure
    // =========================================================================

    /// Initialize photon element data for this material.
    ///
    /// For each nuclide in the material, determines the corresponding element
    /// and loads photon interaction data from the global element cache (or from
    /// the provided path map). Populates `self.element_indices`.
    ///
    /// This should be called once during setup when `transport_secondary_photons=true`.
    ///
    /// # Arguments
    /// * `photon_data_paths` - Map of element name to Arrow data path
    ///   (e.g., {"Fe": "path/to/Fe.arrow", "Li": "path/to/Li.arrow"}).
    pub fn init_photon_data(
        &mut self,
        photon_data_paths: &HashMap<String, String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use yamc_element::element::element_symbol_from_nuclide;
        use yamc_element::photon::get_or_load_element;

        // Track which elements we've already loaded (avoid duplicates)
        let mut seen_elements: HashMap<
            String,
            (usize, Arc<yamc_element::photon::PhotonInteraction>),
        > = HashMap::new();
        self.element_indices.clear();
        self.cached_elements.clear();

        // Sort nuclide names for deterministic ordering
        let mut nuclide_names: Vec<String> = self.nuclides.keys().cloned().collect();
        nuclide_names.sort();

        for nuclide_name in &nuclide_names {
            let element_sym = element_symbol_from_nuclide(nuclide_name);

            // Check if we already loaded this element
            if let Some((idx, arc)) = seen_elements.get(&element_sym) {
                self.element_indices.push((nuclide_name.clone(), *idx));
                self.cached_elements
                    .push((nuclide_name.clone(), Arc::clone(arc)));
                continue;
            }

            // Look up the path for this element
            let path = photon_data_paths.get(&element_sym).ok_or_else(|| {
                format!(
                    "No photon data path provided for element '{element_sym}' (from nuclide '{nuclide_name}')"
                )
            })?;

            // Load from Arrow (or get from global cache)
            let element_data = get_or_load_element(&element_sym, path)?;
            let idx = element_data.index;

            seen_elements.insert(element_sym, (idx, Arc::clone(&element_data)));
            self.element_indices.push((nuclide_name.clone(), idx));
            self.cached_elements
                .push((nuclide_name.clone(), element_data));
        }

        // Build element atom density cache if atom densities are already available
        if let Some(ref atoms_per_bcm) = self.cached_atoms_per_barn_cm {
            self.cached_element_atom_densities = self
                .cached_elements
                .iter()
                .map(|(name, _)| atoms_per_bcm.get(name).copied().unwrap_or(0.0))
                .collect();
        }

        Ok(())
    }

    /// Initialize thick-target bremsstrahlung data for this material.
    ///
    /// Must be called after `init_photon_data()` and while the global TTB energy
    /// grid is still in LINEAR space. Computes the PDF, CDF, and yield tables
    /// for both electrons and positrons.
    #[allow(clippy::type_complexity)]
    pub fn init_bremsstrahlung(&mut self) {
        let ttb_e_grid = yamc_element::photon::ttb_e_grid();
        let ttb_k_grid = yamc_element::photon::ttb_k_grid();

        if ttb_e_grid.is_empty() || ttb_k_grid.is_empty() {
            eprintln!(
                "Warning: TTB energy grids not loaded -- skipping bremsstrahlung init for material {:?}",
                self.name
            );
            return;
        }

        let atoms_per_bcm = self.cached_atoms_per_barn_cm.as_ref().unwrap_or_else(|| {
            panic!("cached_atoms_per_barn_cm is None. Call calculate_macroscopic_xs first.")
        });

        // Collect per-element data needed by init_bremsstrahlung.
        // We need: (atom_density, Z, &dcs, &stopping_power_radiative, &n_electrons, &ionization_energy, mean_excitation_energy)

        // Deduplicate elements (multiple nuclides may share the same element)
        let mut seen: HashMap<usize, usize> = HashMap::new(); // element_index -> position in vecs
        let mut elem_atom_densities: Vec<f64> = Vec::new();
        let mut elem_indices: Vec<usize> = Vec::new();

        for (nuclide_name, i_element) in &self.element_indices {
            let i_element = *i_element;
            let atom_density = atoms_per_bcm.get(nuclide_name).copied().unwrap_or(0.0);

            if let Some(&pos) = seen.get(&i_element) {
                // Accumulate atom density for same element
                elem_atom_densities[pos] += atom_density;
            } else {
                let pos = elem_indices.len();
                seen.insert(i_element, pos);
                elem_atom_densities.push(atom_density);
                elem_indices.push(i_element);
            }
        }

        // Build element_data tuples referencing the photon interaction data
        let elements: Vec<std::sync::Arc<yamc_element::photon::PhotonInteraction>> = elem_indices
            .iter()
            .map(|&idx| {
                yamc_element::photon::get_element_by_index(idx)
                    .expect("Photon element not found in global cache")
            })
            .collect();

        let element_data: Vec<(f64, u32, &[Vec<f64>], &[f64], &[f64], &[f64], f64)> = elements
            .iter()
            .enumerate()
            .map(|(pos, elem)| {
                (
                    elem_atom_densities[pos],
                    elem.atomic_number,
                    elem.dcs.as_slice(),
                    elem.stopping_power_radiative.as_slice(),
                    elem.n_electrons.as_slice(),
                    elem.ionization_energy.as_slice(),
                    elem.mean_excitation_energy,
                )
            })
            .collect();

        // For the collision stopping power, we need per-element AWRs that match
        // the deduplicated element order. Use the AWR of the first nuclide for
        // each element (they differ only slightly for isotopes).
        let elem_awrs: Vec<f64> = {
            let mut awrs = Vec::with_capacity(elem_indices.len());
            for &i_element in &elem_indices {
                // Find first nuclide mapping to this element
                let awr = self
                    .element_indices
                    .iter()
                    .find(|(_, idx)| *idx == i_element)
                    .and_then(|(name, _)| self.nuclide_data.get(name))
                    .and_then(|n| n.atomic_weight_ratio)
                    .unwrap_or(1.0);
                awrs.push(awr);
            }
            awrs
        };

        self.ttb = Some(yamc_element::bremsstrahlung::init_bremsstrahlung(
            &ttb_e_grid,
            &ttb_k_grid,
            &element_data,
            &elem_awrs,
        ));
    }

    /// Compute macroscopic photon cross sections at the given energy.
    ///
    /// Loops over nuclides, gets the element microscopic XS via log-log
    /// interpolation, and sums `atom_density * micro_xs` for each component.
    /// Computes macroscopic photon cross sections from per-element microscopic data.
    ///
    /// # Arguments
    /// * `energy` - Photon energy in eV.
    ///
    /// # Returns
    /// `MacroPhotonXS` with total, coherent, incoherent, photoelectric, and
    /// pair production macroscopic cross sections in 1/cm (barn-cm units).
    #[inline]
    pub fn calculate_photon_xs(&self, energy: f64) -> MacroPhotonXS {
        debug_assert_eq!(
            self.cached_element_atom_densities.len(),
            self.cached_elements.len(),
            "cached_element_atom_densities not built; call calculate_macroscopic_xs first"
        );

        let mut macro_xs = MacroPhotonXS::default();

        for (idx, (_name, element)) in self.cached_elements.iter().enumerate() {
            let micro = element.calculate_xs(energy);
            let atom_density = self.cached_element_atom_densities[idx];

            macro_xs.total += atom_density * micro.total;
            macro_xs.coherent += atom_density * micro.coherent;
            macro_xs.incoherent += atom_density * micro.incoherent;
            macro_xs.photoelectric += atom_density * micro.photoelectric;
            macro_xs.pair_production += atom_density * micro.pair_production;
            macro_xs.heating += atom_density * micro.heating;
        }

        macro_xs
    }

    /// Sample which element a photon interacts with.
    ///
    /// Uses cumulative macroscopic XS as weights:
    /// `prob += atom_density * micro.total`, stops when `prob > cutoff`.
    /// Samples an element proportional to its macroscopic cross section contribution.
    ///
    /// # Arguments
    /// * `macro_xs_total` - Total macroscopic photon XS (from `calculate_photon_xs`).
    /// * `energy` - Photon energy in eV.
    /// * `rng` - Random number generator.
    ///
    /// # Returns
    /// `(element_index, Arc<PhotonInteraction>)` for the sampled element.
    pub fn sample_element<R: rand::Rng + ?Sized>(
        &self,
        macro_xs_total: f64,
        energy: f64,
        rng: &mut R,
    ) -> (usize, Arc<PhotonInteraction>) {
        let cutoff = rng.random_range(0.0..macro_xs_total);

        let mut prob = 0.0;
        for (idx, (_name, element)) in self.cached_elements.iter().enumerate() {
            let micro = element.calculate_xs(energy);
            let atom_density = self.cached_element_atom_densities[idx];

            prob += atom_density * micro.total;
            if prob > cutoff {
                return (element.index, Arc::clone(element));
            }
        }

        // Fallback: return last element (shouldn't happen unless rounding)
        let (_, element) = self
            .cached_elements
            .last()
            .expect("cached_elements is empty -- call init_photon_data first");
        (element.index, Arc::clone(element))
    }
}
