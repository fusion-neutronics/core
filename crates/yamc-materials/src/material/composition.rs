use super::*;

/// The entries of a nuclide map in name order.
///
/// `Material::nuclides` is a `HashMap`, and Rust seeds each instance
/// separately, so summing straight out of it adds the same numbers in a
/// different order on every process. Every sum below is over a few dozen terms
/// and lands in the atom densities the whole solve starts from, so an ulp there
/// moves the last bits of every inventory: `Material.transmute()` returned
/// results that disagreed at ~1e-15 between one run and the next, with no Monte
/// Carlo anywhere in the path. This is the same defect `matrix.rs` fixed for the
/// burnup matrix in issue #502, one layer further up.
///
/// Sorting a few dozen keys costs nothing at these sizes and is what makes a
/// golden inventory comparable across runs at all.
fn in_name_order(nuclides: &HashMap<String, f64>) -> Vec<(&str, f64)> {
    let mut entries: Vec<(&str, f64)> = nuclides.iter().map(|(n, &v)| (n.as_str(), v)).collect();
    entries.sort_unstable_by(|a, b| a.0.cmp(b.0));
    entries
}

impl Material {
    /// Convert material to 'sum' mode (absolute atom densities).
    ///
    /// Returns a new Material where nuclide values are absolute atom densities
    /// [atoms/barn-cm] and density_units is "sum". If already in sum mode,
    /// returns a clone.
    pub fn to_sum_mode(&self) -> Result<Material, String> {
        if self.density_units == DensityUnits::Sum {
            return Ok(self.clone());
        }

        let atoms_per_bcm = self.get_atoms_per_barn_cm()?;
        Ok(Material::from_nuclide_densities(atoms_per_bcm, self))
    }

    /// Calculate atoms per barn-centimeter for each nuclide in the material.
    ///
    /// Behavior depends on `density_units` and `fraction_type`:
    /// - `"sum"`: Nuclide fractions are absolute atom densities (atoms/barn-cm), returned directly.
    ///   `fraction_type` must be `"atom"` (mass fractions don't make sense for absolute densities).
    /// - `"g/cm3"`, `"g/cc"`, `"kg/m3"` with `fraction_type = "atom"` (atom fractions):
    ///   `N_i = rho * N_A / M_avg * x_i_normalized * 1e-24`
    /// - `"g/cm3"`, `"g/cc"`, `"kg/m3"` with `fraction_type = "mass"` (mass fractions):
    ///   `N_i = rho * N_A * (w_i_normalized / M_i) * 1e-24`
    ///
    /// Returns `Err` if density or nuclides are not set (except in "sum" mode where density is optional),
    /// or if a nuclide's atomic mass cannot be determined.
    pub fn get_atoms_per_barn_cm(&self) -> Result<HashMap<String, f64>, String> {
        if self.nuclides.is_empty() {
            return Err(
                "Cannot calculate atoms per barn-cm: Material has no nuclides defined".to_string(),
            );
        }

        let mut atoms_per_bcm = HashMap::new();

        match self.density_units {
            DensityUnits::Sum => {
                if self.fraction_type == FractionType::Mass {
                    return Err("Cannot use fraction='weight' with density unit 'sum': \
                         'sum' mode requires absolute atom densities (atoms/barn-cm)"
                        .to_string());
                }
                // Nuclide fractions ARE absolute atom densities in atoms/barn-cm
                for (nuclide, &fraction) in &self.nuclides {
                    if fraction < 0.0 {
                        return Err(format!(
                            "Nuclide '{nuclide}' has negative atom density {fraction} in 'sum' mode"
                        ));
                    }
                    atoms_per_bcm.insert(nuclide.clone(), fraction);
                }
            }
            DensityUnits::AtomPerBarnCm => {
                // density is the TOTAL atom density (atoms/barn-cm); composition
                // values are relative atom fractions. N_i = (x_i / sum x) * total.
                // fraction_type is validated as Atom at construction.
                let total = self.density.ok_or_else(|| {
                    "Cannot calculate atoms per barn-cm: Material has no density defined"
                        .to_string()
                })?;
                let total_fraction: f64 =
                    in_name_order(&self.nuclides).iter().map(|(_, v)| v).sum();
                if total_fraction <= 0.0 {
                    return Err("Total fraction is zero or negative".into());
                }
                for (nuclide, &fraction) in &self.nuclides {
                    atoms_per_bcm.insert(nuclide.clone(), total * fraction / total_fraction);
                }
            }
            DensityUnits::GramsPerCc | DensityUnits::KgPerM3 => {
                let density_gcm3 =
                    match self.density {
                        Some(d) => match self.density_units {
                            DensityUnits::KgPerM3 => d / 1000.0,
                            _ => d, // g/cm3 and g/cc are equivalent
                        },
                        None => return Err(
                            "Cannot calculate atoms per barn-cm: Material has no density defined"
                                .to_string(),
                        ),
                    };

                let mut nuclide_masses = HashMap::new();
                for nuclide in self.nuclides.keys() {
                    let mass = yamc_nuclide::composition::atomic_mass(nuclide).map_err(|e| {
                        format!("Cannot determine atomic mass for nuclide '{nuclide}': {e}")
                    })?;
                    nuclide_masses.insert(nuclide.clone(), mass);
                }

                const AVOGADRO: f64 = 6.02214076e23;
                let ordered = in_name_order(&self.nuclides);
                let total_fraction: f64 = ordered.iter().map(|(_, v)| v).sum();

                if self.fraction_type == FractionType::Mass {
                    // Weight fractions: N_i = rho * N_A * (w_i_normalized / M_i) * 1e-24
                    for (nuclide, &fraction) in &self.nuclides {
                        let w_i = fraction / total_fraction;
                        let mass = nuclide_masses[nuclide];
                        let atom_density = density_gcm3 * AVOGADRO * w_i / mass * 1.0e-24;
                        atoms_per_bcm.insert(nuclide.clone(), atom_density);
                    }
                } else {
                    // Atom fractions: N_i = rho * N_A / M_avg * x_i_normalized * 1e-24
                    let mut weighted_mass_sum = 0.0;
                    for (nuclide, fraction) in &ordered {
                        let mass = nuclide_masses[*nuclide];
                        weighted_mass_sum += fraction * mass;
                    }
                    let average_molar_mass = weighted_mass_sum / total_fraction;

                    for (nuclide, &fraction) in &self.nuclides {
                        let normalized_fraction = fraction / total_fraction;
                        let atom_density = density_gcm3 * AVOGADRO / average_molar_mass
                            * normalized_fraction
                            * 1.0e-24;
                        atoms_per_bcm.insert(nuclide.clone(), atom_density);
                    }
                }
            }
        }

        Ok(atoms_per_bcm)
    }

    /// Compute the average molar mass of this material (g/mol).
    ///
    /// For atom fractions ("atom"): `M_avg = sum(x_i * M_i) / sum(x_i)`
    /// For mass fractions ("mass"): `M_avg = sum(w_i) / sum(w_i / M_i)`
    pub fn average_molar_mass(&self) -> Result<f64, String> {
        if self.nuclides.is_empty() {
            return Err("Material has no nuclides".into());
        }
        let ordered = in_name_order(&self.nuclides);
        let total_frac: f64 = ordered.iter().map(|(_, v)| v).sum();
        if total_frac <= 0.0 {
            return Err("Total fraction is zero or negative".into());
        }

        if self.fraction_type == FractionType::Mass {
            let mut sum_w_over_m = 0.0;
            for (nuc, frac) in &ordered {
                let m = yamc_nuclide::composition::atomic_mass(nuc)?;
                sum_w_over_m += frac / m;
            }
            Ok(total_frac / sum_w_over_m)
        } else {
            let mut weighted_mass = 0.0;
            for (nuc, frac) in &ordered {
                let m = yamc_nuclide::composition::atomic_mass(nuc)?;
                weighted_mass += frac * m;
            }
            Ok(weighted_mass / total_frac)
        }
    }

    /// Return mass density in g/cm³ regardless of stored units.
    ///
    /// For "sum" mode, computes density from atom densities and molar masses.
    pub fn get_mass_density(&self) -> Result<f64, String> {
        if self.nuclides.is_empty() {
            return Err("Material has no nuclides".into());
        }
        const AVOGADRO: f64 = 6.02214076e23;

        match self.density_units {
            DensityUnits::Sum => {
                // Each nuclide value is atoms/barn-cm.  Convert to g/cm³.
                // rho = sum_i(N_i * M_i / N_A) where N_i in atoms/cm³ = value * 1e24
                let mut mass_density = 0.0;
                for (nuc, n_bcm) in in_name_order(&self.nuclides) {
                    let m = yamc_nuclide::composition::atomic_mass(nuc)?;
                    mass_density += n_bcm * 1.0e24 * m / AVOGADRO;
                }
                Ok(mass_density)
            }
            DensityUnits::AtomPerBarnCm => {
                // Derive per-nuclide atom densities (fractions * total), then
                // convert to g/cm³: rho = sum_i(N_i * 1e24 * M_i / N_A).
                let per = self.get_atoms_per_barn_cm()?;
                let mut mass_density = 0.0;
                for (nuc, n_bcm) in in_name_order(&per) {
                    let m = yamc_nuclide::composition::atomic_mass(nuc)?;
                    mass_density += n_bcm * 1.0e24 * m / AVOGADRO;
                }
                Ok(mass_density)
            }
            DensityUnits::GramsPerCc => self.density.ok_or_else(|| "Density value not set".into()),
            DensityUnits::KgPerM3 => {
                let d = self
                    .density
                    .ok_or_else(|| "Density value not set".to_string())?;
                Ok(d / 1000.0)
            }
        }
    }

    /// Total atom density (atoms/barn-cm), regardless of stored units.
    ///
    /// For `"atom/barn-cm"` this is the stored density value; for `"sum"` it is
    /// the sum of the per-nuclide values; for mass-density units it is derived
    /// from the per-nuclide atom densities. Used by the Python `density` getter
    /// so it always returns a float.
    pub fn total_atom_density(&self) -> Result<f64, String> {
        let per = self.get_atoms_per_barn_cm()?;
        Ok(in_name_order(&per).iter().map(|(_, v)| v).sum())
    }

    /// Mix multiple materials into a new material.
    ///
    /// Mixes materials by combining nuclide atom densities with the given fractions.
    ///
    /// # Arguments
    /// * `materials` -- slice of references to materials to mix
    /// * `fractions` -- mixing fractions (one per material)
    /// * `fraction_type` -- `"atom"` (atom), `"mass"` (mass), or `"volume"` (volume)
    /// * `name` -- optional name for the result (auto-generated if `None`)
    /// * `material_id` -- optional ID for the result
    pub fn mix_materials(
        materials: &[&Material],
        fractions: &[f64],
        fraction_type: &str,
        name: Option<String>,
        material_id: Option<u32>,
    ) -> Result<Material, String> {
        // Validate fraction_type
        let fraction_type = match fraction_type {
            "atom" => "atom",
            "mass" => "mass",
            "volume" => "volume",
            other => {
                return Err(format!(
                    "fraction must be 'atom', 'mass', or 'volume', got '{other}'"
                ))
            }
        };

        // --- validation ---
        if materials.len() != fractions.len() {
            return Err(format!(
                "Number of materials ({}) must match number of fractions ({})",
                materials.len(),
                fractions.len()
            ));
        }
        if materials.is_empty() {
            return Err("At least one material is required".into());
        }
        for (i, &f) in fractions.iter().enumerate() {
            if f < 0.0 {
                return Err(format!("Fraction at index {i} is negative ({f})"));
            }
        }
        let frac_sum: f64 = fractions.iter().sum();
        match fraction_type {
            "atom" | "mass" => {
                if (frac_sum - 1.0).abs() > 1e-8 {
                    return Err(format!(
                        "Fractions must sum to 1.0 for fraction='{fraction_type}', got {frac_sum}"
                    ));
                }
            }
            "volume" => {
                if frac_sum > 1.0 + 1e-8 {
                    return Err(format!(
                        "Volume fractions must sum to <= 1.0, got {frac_sum}"
                    ));
                }
            }
            _ => unreachable!(),
        }

        // --- gather per-material properties ---
        let n = materials.len();
        let mut atoms_per_bcm_vec: Vec<HashMap<String, f64>> = Vec::with_capacity(n);
        let mut amm_vec: Vec<f64> = Vec::with_capacity(n);
        let mut rho_vec: Vec<f64> = Vec::with_capacity(n);

        for (i, mat) in materials.iter().enumerate() {
            let abcm = mat.get_atoms_per_barn_cm()?;
            if abcm.is_empty() {
                return Err(format!("Material at index {i} has no nuclides or density"));
            }
            let amm = mat.average_molar_mass()?;
            let rho = mat.get_mass_density()?;
            atoms_per_bcm_vec.push(abcm);
            amm_vec.push(amm);
            rho_vec.push(rho);
        }

        // --- compute volume-based weights ---
        let mut wgts: Vec<f64> = match fraction_type {
            "atom" => {
                // w_i = frac_i * amm_i / rho_i, then normalize
                fractions
                    .iter()
                    .enumerate()
                    .map(|(i, &f)| f * amm_vec[i] / rho_vec[i])
                    .collect()
            }
            "mass" => {
                // w_i = frac_i / rho_i, then normalize
                fractions
                    .iter()
                    .enumerate()
                    .map(|(i, &f)| f / rho_vec[i])
                    .collect()
            }
            "volume" => {
                // Direct volume fractions
                fractions.to_vec()
            }
            _ => unreachable!(),
        };

        // Normalize for atom/mass
        if fraction_type != "volume" {
            let wgt_sum: f64 = wgts.iter().sum();
            if wgt_sum > 0.0 {
                for w in &mut wgts {
                    *w /= wgt_sum;
                }
            }
        }

        // --- accumulate nuclide densities ---
        const AVOGADRO: f64 = 6.02214076e23;
        let mut nuclides_per_cc: HashMap<String, f64> = HashMap::new();
        let mut mass_per_cc: HashMap<String, f64> = HashMap::new();

        for (i, abcm) in atoms_per_bcm_vec.iter().enumerate() {
            let wgt = wgts[i];
            for (nuc, &n_bcm) in abcm {
                let nuc_per_cc = wgt * 1.0e24 * n_bcm;
                *nuclides_per_cc.entry(nuc.clone()).or_insert(0.0) += nuc_per_cc;
                let m = yamc_nuclide::composition::atomic_mass(nuc)?;
                *mass_per_cc.entry(nuc.clone()).or_insert(0.0) += nuc_per_cc * m / AVOGADRO;
            }
        }

        // --- build result material ---
        // In name order, like every other sum in this file: these two totals
        // normalize every nuclide fraction below and become the mixed
        // material's density, so summing them in `HashMap` order left a mix of
        // the same inputs slightly different on each run. This is the one
        // function `in_name_order` did not reach when the rest of the file was
        // fixed (issue #576).
        let total_atoms_per_cc: f64 = in_name_order(&nuclides_per_cc).iter().map(|(_, v)| v).sum();
        let total_mass_per_cc: f64 = in_name_order(&mass_per_cc).iter().map(|(_, v)| v).sum();

        // Build nuclide atom fractions map
        let mut nuclide_fracs: HashMap<String, f64> = HashMap::new();
        if total_atoms_per_cc > 0.0 {
            for (nuc, &n_cc) in &nuclides_per_cc {
                let atom_frac = n_cc / total_atoms_per_cc;
                if atom_frac > 0.0 {
                    nuclide_fracs.insert(nuc.clone(), atom_frac);
                }
            }
        }

        let density = if total_mass_per_cc > 0.0 {
            Some(total_mass_per_cc)
        } else {
            None
        };

        let mut result = Material::new(nuclide_fracs, "atom", "g/cm3", density)?;

        // Propagate transmutable flag
        if materials.iter().any(|m| m.transmutable) {
            result.transmutable = true;
        }

        // Name
        let final_name = name.unwrap_or_else(|| {
            let parts: Vec<String> = materials
                .iter()
                .zip(fractions.iter())
                .map(|(m, f)| {
                    let default_name = format!("mat_{}", m.material_id.unwrap_or(0));
                    let mname = m.name.as_deref().unwrap_or(&default_name);
                    format!("{}({:.4})", mname, f)
                })
                .collect();
            parts.join("_")
        });
        result.name = Some(final_name);

        if let Some(id) = material_id {
            result.material_id = Some(id);
        }

        Ok(result)
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_atoms_per_barn_cm_sum_mode_success() {
        // "sum" mode returns the absolute atom densities unchanged.
        let nuclides = HashMap::from([("H1".to_string(), 0.0667), ("O16".to_string(), 0.0334)]);
        let mat = Material::new(nuclides.clone(), "atom", "sum", None).unwrap();

        let result = mat
            .get_atoms_per_barn_cm()
            .expect("valid sum-mode material should not error");

        assert_eq!(result, nuclides);
    }

    #[test]
    fn test_get_atoms_per_barn_cm_weight_sum_errors() {
        // fraction_type="mass" combined with density unit "sum" is invalid and
        // must return Err rather than panicking.
        let mat = Material {
            fraction_type: FractionType::Mass,
            ..Material::new(
                HashMap::from([("H1".to_string(), 1.0)]),
                "atom",
                "sum",
                None,
            )
            .unwrap()
        };

        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| mat.get_atoms_per_barn_cm()))
                .expect("get_atoms_per_barn_cm must not panic on invalid input");

        assert!(result.is_err(), "weight + sum should return Err");
        assert!(result.unwrap_err().contains("fraction='weight'"));
    }
}
