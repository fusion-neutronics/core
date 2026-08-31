//! Arrow IPC format reader for photon (photoatomic) nuclear data.
//!
//! Reads a `.arrow/` directory containing pre-computed photon interaction data.
//! Directory structure:
//! ```text
//! Fe.arrow/
//! ├── version.json
//! ├── element.arrow       # Z, energy grid, all XS (raw + ln), form factors
//! ├── subshells.arrow     # per-subshell photoionization + transitions
//! ├── compton.arrow       # Compton profiles + pre-computed CDFs
//! └── bremsstrahlung.arrow # DCS tables (already truncated at cutoff)
//! ```

use crate::photon::{
    AtomicTransition, ComptonRelaxTarget, ElectronSubshell, PhotonInteraction, SUBSHELLS,
};
use crate::photon_log::log_transform_xs;
use yamc_nuclide::arrow_helpers::{
    get_f64, get_f64_list, get_i32, get_i32_list, get_str, read_arrow_file, try_get_f64_list,
};
use yamc_nuclide::reaction_product::Tabulated1D;

use arrow_array::RecordBatch;

use std::collections::HashMap;
use std::error::Error;
use std::path::Path;

// =============================================================================
// Main reader
// =============================================================================

/// Read photon interaction data from an Arrow IPC directory, to LOOK at it.
///
/// Same read as [`read_photon_interaction_from_arrow`], with one difference
/// that matters to a caller holding a directory up for inspection rather than
/// loading a library to simulate with: it publishes none of the process-wide
/// grids. Those are first-write-wins, so an ordinary read of a candidate
/// directory installs its Compton momentum grid and TTB grids over the ones
/// every later element load and every photon collision in the process uses, and
/// a same-length grid with different values does it with no error at all.
///
/// The returned data is identical either way. Nothing in the read path reads
/// those grids back; they are consumed later, by transport.
pub fn inspect_photon_interaction_from_arrow(
    dir: &Path,
) -> Result<PhotonInteraction, Box<dyn Error>> {
    let _suppressed = crate::photon::SharedGridsSuppressed::new();
    read_photon_interaction_from_arrow(dir)
}

/// Read photon interaction data from an Arrow IPC directory.
pub fn read_photon_interaction_from_arrow(dir: &Path) -> Result<PhotonInteraction, Box<dyn Error>> {
    // 1. Read element.arrow
    let elem_batch = read_arrow_file(&dir.join("element.arrow"))?;

    let name = get_str(&elem_batch, "name", 0)?;
    let z = get_i32(&elem_batch, "Z", 0)? as u32;

    // Energy grid: use pre-computed ln(energy)
    let energy = get_f64_list(&elem_batch, "ln_energy", 0)?;
    let n_energy = energy.len();

    // Cross sections: read raw values and apply safe_log
    // to ensure consistent clamping of zero/tiny values to -900.
    let coherent_xs_raw = get_f64_list(&elem_batch, "coherent_xs", 0)?;
    let incoherent_xs_raw = get_f64_list(&elem_batch, "incoherent_xs", 0)?;
    let photoelectric_xs_raw = get_f64_list(&elem_batch, "photoelectric_xs", 0)?;
    let coherent_xs = log_transform_xs(&coherent_xs_raw);
    let incoherent_xs = log_transform_xs(&incoherent_xs_raw);
    let photoelectric_total_xs = log_transform_xs(&photoelectric_xs_raw);

    // Pair production and heating
    let pp_nuclear_raw = try_get_f64_list(&elem_batch, "pair_production_nuclear_xs", 0);
    let pp_electron_raw = try_get_f64_list(&elem_batch, "pair_production_electron_xs", 0);
    let heating_raw = try_get_f64_list(&elem_batch, "heating_xs", 0);

    // Default to zeros if empty
    let pp_nuclear_raw = if pp_nuclear_raw.is_empty() {
        vec![0.0; n_energy]
    } else {
        pp_nuclear_raw
    };
    let pp_electron_raw = if pp_electron_raw.is_empty() {
        vec![0.0; n_energy]
    } else {
        pp_electron_raw
    };
    let heating_raw = if heating_raw.is_empty() {
        vec![0.0; n_energy]
    } else {
        heating_raw
    };

    // Compute pair_production_total = nuclear + electron (in raw space), then log-transform
    let pp_total_raw: Vec<f64> = pp_nuclear_raw
        .iter()
        .zip(pp_electron_raw.iter())
        .map(|(&n, &e)| n + e)
        .collect();

    let pair_production_total_xs = log_transform_xs(&pp_total_raw);
    let pair_production_nuclear_xs = log_transform_xs(&pp_nuclear_raw);
    let pair_production_electron_xs = log_transform_xs(&pp_electron_raw);
    let heating_xs = log_transform_xs(&heating_raw);

    // Form factors
    let coh_ff_x = get_f64_list(&elem_batch, "coherent_int_ff_x", 0)?;
    let coh_ff_y = get_f64_list(&elem_batch, "coherent_int_ff_y", 0)?;
    let coherent_int_form_factor = Tabulated1D::Tabulated1D {
        x: coh_ff_x,
        y: coh_ff_y,
        breakpoints: Vec::new(),
        interpolation: Vec::new(),
    };

    let inc_ff_x = get_f64_list(&elem_batch, "incoherent_ff_x", 0)?;
    let inc_ff_y = get_f64_list(&elem_batch, "incoherent_ff_y", 0)?;
    let incoherent_form_factor = Tabulated1D::Tabulated1D {
        x: inc_ff_x,
        y: inc_ff_y,
        breakpoints: Vec::new(),
        interpolation: Vec::new(),
    };

    // 2. Read subshells.arrow
    let (shells, has_atomic_relaxation) = read_subshells(dir, n_energy)?;

    // Build 2D cross_sections array: cross_sections[i_grid][i_shell]
    let n_shells = shells.len();
    let mut cross_sections: Vec<Vec<f64>> = vec![vec![0.0; n_shells]; n_energy];
    for (i_shell, shell) in shells.iter().enumerate() {
        for (j, &log_xs) in shell.cross_section.iter().enumerate() {
            let i_grid = shell.threshold + j;
            if i_grid < n_energy {
                cross_sections[i_grid][i_shell] = log_xs;
            }
        }
    }

    // Expected per-subshell radiative (fluorescence) energy, used to correct the
    // photoelectric photon-KERMA coefficient. Only meaningful when the element
    // carries atomic-relaxation data; otherwise `atomic_relaxation` banks no
    // fluorescence, so the correction (and this vector) must be zero.
    let subshell_radiative_energy = if has_atomic_relaxation {
        crate::photon::compute_subshell_radiative_energy(&shells)
    } else {
        vec![0.0; n_shells]
    };

    // 3. Read compton.arrow, including the Compton-profile-shell -> relaxation-
    // subshell map. The association is computed at data-generation time by
    // occupancy grouping (a Compton (n, l) shell maps to its constituent (n, l, j)
    // subshells, weighted by occupancy) and shipped in compton.arrow, so the load
    // path no longer re-derives it from binding energies. Compton shells with no
    // clean counterpart (outer/valence) carry an empty target list and bank no
    // fluorescence.
    let (electron_pdf, binding_energy_cp, profile_pdf, profile_cdf, compton_relax_map) =
        read_compton(dir)?;

    // Expected fluorescence energy banked per Compton event: the event ionizes
    // the shell sampled from `electron_pdf` (Doppler broadening) and relaxes one
    // of its constituent subshells (chosen by occupancy weight), so this is the
    // electron_pdf- and occupancy-weighted per-shell radiative energy. Constant
    // (energy-independent). See PhotonInteraction::compton_radiative_energy.
    let compton_radiative_energy: f64 = electron_pdf
        .iter()
        .zip(&compton_relax_map)
        .map(|(&pdf, targets)| {
            pdf * targets
                .iter()
                .map(|t| t.weight * subshell_radiative_energy[t.shell_index])
                .sum::<f64>()
        })
        .sum();

    // 4. Read bremsstrahlung.arrow
    let (
        dcs,
        stopping_power_radiative,
        ionization_energy,
        n_electrons,
        mean_excitation_energy,
        ttb_electron_energy,
        ttb_photon_energy,
    ) = read_bremsstrahlung(dir, z)?;

    println!(
        "  Loaded photon data (Arrow): Z={}, {} energy points, {} subshells, {} Compton shells",
        z,
        n_energy,
        n_shells,
        profile_pdf.len()
    );

    Ok(PhotonInteraction {
        name,
        index: 0,
        atomic_number: z,
        energy,
        coherent_xs,
        incoherent_xs,
        photoelectric_total_xs,
        pair_production_total_xs,
        pair_production_nuclear_xs,
        pair_production_electron_xs,
        heating_xs,
        coherent_int_form_factor,
        incoherent_form_factor,
        electron_pdf,
        binding_energy: binding_energy_cp,
        profile_pdf,
        profile_cdf,
        shells,
        cross_sections,
        subshell_radiative_energy,
        compton_radiative_energy,
        compton_relax_map,
        has_atomic_relaxation,
        dcs,
        stopping_power_radiative,
        ionization_energy,
        n_electrons,
        mean_excitation_energy,
        ttb_electron_energy,
        ttb_photon_energy,
    })
}

// =============================================================================
// Subshell reader
// =============================================================================

fn read_subshells(
    dir: &Path,
    _n_energy: usize,
) -> Result<(Vec<ElectronSubshell>, bool), Box<dyn Error>> {
    let path = dir.join("subshells.arrow");
    if !path.exists() {
        return Ok((Vec::new(), false));
    }

    let batch = read_arrow_file(&path)?;
    let n_rows = batch.num_rows();

    // First pass: build shell_map (file subshell IDs → local indices)
    let mut shell_map: HashMap<i32, i32> = HashMap::new();
    shell_map.insert(0, -1); // ID 0 = no subshell

    let mut subshell_indices: Vec<i32> = Vec::with_capacity(n_rows);
    for row in 0..n_rows {
        let designator = get_str(&batch, "designator", row)?;
        let subshell_idx = SUBSHELLS
            .iter()
            .position(|&s| s == designator)
            .map(|i| (i + 1) as i32)
            .unwrap_or(0);
        shell_map.insert(subshell_idx, row as i32);
        subshell_indices.push(subshell_idx);
    }

    // Second pass: read shell data and transitions
    let mut shells: Vec<ElectronSubshell> = Vec::with_capacity(n_rows);
    let mut has_atomic_relaxation = false;

    for (row, &subshell_idx) in subshell_indices.iter().enumerate() {
        let binding_energy = get_f64(&batch, "binding_energy", row)?;
        let num_electrons = get_f64(&batch, "num_electrons", row)?;
        let threshold = get_i32(&batch, "threshold_idx", row)? as usize;

        // Use pre-computed ln(xs)
        let cross_section = get_f64_list(&batch, "ln_xs", row)?;

        // Read transitions
        let transitions = read_transitions_from_arrow(&batch, row, &shell_map)?;
        if !transitions.is_empty() {
            has_atomic_relaxation = true;
        }

        shells.push(ElectronSubshell {
            index_subshell: subshell_idx,
            binding_energy,
            num_electrons,
            threshold,
            cross_section,
            transitions,
        });
    }

    Ok((shells, has_atomic_relaxation))
}

/// Read transitions for a subshell row from the Arrow batch.
///
/// transitions_data is a flat array of [primary_id, secondary_id, energy, prob] × N
/// transitions_shape is [N, 4]
fn read_transitions_from_arrow(
    batch: &RecordBatch,
    row: usize,
    shell_map: &HashMap<i32, i32>,
) -> Result<Vec<AtomicTransition>, Box<dyn Error>> {
    let data = try_get_f64_list(batch, "transitions_data", row);
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let shape = get_i32_list(batch, "transitions_shape", row)?;
    if shape.len() < 2 || shape[1] != 4 {
        return Ok(Vec::new());
    }

    let n_transitions = shape[0] as usize;

    // Compute normalization factor
    let norm: f64 = (0..n_transitions).map(|i| data[i * 4 + 3]).sum();

    let mut transitions = Vec::with_capacity(n_transitions);
    let mut cumulative_prob = 0.0;

    for i in 0..n_transitions {
        let primary_id = data[i * 4] as i32;
        let secondary_id = data[i * 4 + 1] as i32;
        let energy = data[i * 4 + 2];
        let prob = if norm > 0.0 {
            data[i * 4 + 3] / norm
        } else {
            data[i * 4 + 3]
        };

        let primary_local = shell_map.get(&primary_id).copied().unwrap_or(-1);
        let secondary_local = shell_map.get(&secondary_id).copied().unwrap_or(-1);

        cumulative_prob += prob;

        transitions.push(AtomicTransition {
            primary_subshell: primary_local,
            secondary_subshell: secondary_local,
            energy,
            probability: cumulative_prob,
        });
    }

    Ok(transitions)
}

// =============================================================================
// Compton profile reader
// =============================================================================

#[allow(clippy::type_complexity)]
fn read_compton(
    dir: &Path,
) -> Result<
    (
        Vec<f64>,
        Vec<f64>,
        Vec<Vec<f64>>,
        Vec<Vec<f64>>,
        Vec<Vec<ComptonRelaxTarget>>,
    ),
    Box<dyn Error>,
> {
    let path = dir.join("compton.arrow");
    if !path.exists() {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()));
    }

    let batch = read_arrow_file(&path)?;

    // Read momentum grid and set global
    let pz = get_f64_list(&batch, "pz", 0)?;
    if !pz.is_empty() {
        crate::photon::set_compton_profile_pz(pz);
    }

    // Read per-shell data
    let mut num_electrons = get_f64_list(&batch, "num_electrons", 0)?;
    let binding_energy = get_f64_list(&batch, "binding_energy", 0)?;

    // Compton-profile shell -> constituent relaxation subshells (CSR layout).
    // Shipped by the data generator (occupancy grouping). Absent on pre-map data,
    // in which case the map is empty and no Compton fluorescence is banked (the
    // conservative behaviour that matches OpenMC on the same data).
    let map_offsets = get_i32_list(&batch, "subshell_map_offsets", 0).unwrap_or_default();
    let map_indices = get_i32_list(&batch, "subshell_map_indices", 0).unwrap_or_default();
    let map_weights = get_f64_list(&batch, "subshell_map_weights", 0).unwrap_or_default();
    let compton_relax_map: Vec<Vec<ComptonRelaxTarget>> = if map_offsets.len() >= 2 {
        (0..map_offsets.len() - 1)
            .map(|c| {
                let lo = map_offsets[c] as usize;
                let hi = map_offsets[c + 1] as usize;
                (lo..hi)
                    .map(|k| ComptonRelaxTarget {
                        shell_index: map_indices[k] as usize,
                        weight: map_weights[k],
                    })
                    .collect()
            })
            .collect()
    } else {
        Vec::new()
    };

    // Normalize electron_pdf so it sums to 1.0
    let total_electrons: f64 = num_electrons.iter().sum();
    if total_electrons > 0.0 {
        for v in &mut num_electrons {
            *v /= total_electrons;
        }
    }

    // Read J (PDF) -- stored as flat array with shape
    let j_data = get_f64_list(&batch, "J_data", 0)?;
    let j_shape = get_i32_list(&batch, "J_shape", 0)?;
    let (n_shells, n_pz) = if j_shape.len() >= 2 {
        (j_shape[0] as usize, j_shape[1] as usize)
    } else {
        return Ok((
            num_electrons,
            binding_energy,
            Vec::new(),
            Vec::new(),
            compton_relax_map,
        ));
    };

    let mut profile_pdf: Vec<Vec<f64>> = Vec::with_capacity(n_shells);
    for i in 0..n_shells {
        let start = i * n_pz;
        let end = start + n_pz;
        profile_pdf.push(j_data[start..end].to_vec());
    }

    // Read pre-computed CDF
    let cdf_data = get_f64_list(&batch, "J_cdf_data", 0)?;
    let cdf_shape = get_i32_list(&batch, "J_cdf_shape", 0)?;
    let profile_cdf = if !cdf_data.is_empty() && cdf_shape.len() >= 2 {
        let n_s = cdf_shape[0] as usize;
        let n_p = cdf_shape[1] as usize;
        let mut cdf = Vec::with_capacity(n_s);
        for i in 0..n_s {
            let start = i * n_p;
            let end = start + n_p;
            cdf.push(cdf_data[start..end].to_vec());
        }
        cdf
    } else {
        Vec::new()
    };

    Ok((
        num_electrons,
        binding_energy,
        profile_pdf,
        profile_cdf,
        compton_relax_map,
    ))
}

// =============================================================================
// Bremsstrahlung reader
// =============================================================================

#[allow(clippy::type_complexity)]
fn read_bremsstrahlung(
    dir: &Path,
    z: u32,
) -> Result<
    (
        Vec<Vec<f64>>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        f64,
        Vec<f64>,
        Vec<f64>,
    ),
    Box<dyn Error>,
> {
    let path = dir.join("bremsstrahlung.arrow");
    if !path.exists() {
        return Ok((
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            0.0,
            Vec::new(),
            Vec::new(),
        ));
    }

    let batch = read_arrow_file(&path)?;

    let mean_excitation_energy = get_f64(&batch, "I", 0)?;
    let electron_energy = get_f64_list(&batch, "electron_energy", 0)?;
    let photon_energy = get_f64_list(&batch, "photon_energy", 0)?;
    let ionization_energy = get_f64_list(&batch, "ionization_energy", 0)?;
    let n_electrons = get_f64_list(&batch, "num_electrons", 0)?;

    // Set global grids
    if !photon_energy.is_empty() {
        crate::photon::set_ttb_k_grid(photon_energy.clone());
    }
    if !electron_energy.is_empty() {
        crate::photon::set_ttb_e_grid(electron_energy.clone());
    }

    // Read DCS matrix (already truncated at cutoff by converter)
    let dcs_data = get_f64_list(&batch, "dcs_data", 0)?;
    let dcs_shape = get_i32_list(&batch, "dcs_shape", 0)?;

    let (n_e, n_k) = if dcs_shape.len() >= 2 {
        (dcs_shape[0] as usize, dcs_shape[1] as usize)
    } else {
        (0, 0)
    };

    let mut dcs: Vec<Vec<f64>> = Vec::with_capacity(n_e);
    for i in 0..n_e {
        let start = i * n_k;
        let end = start + n_k;
        dcs.push(dcs_data[start..end].to_vec());
    }

    // Compute radiative stopping power from DCS
    // S_rad(E) = Z^2 / beta^2 * E * integral(DCS(E,k) dk)
    let mass_electron_ev = 0.51099895000e6; // eV
    let mut stopping_power_radiative = Vec::with_capacity(n_e);
    for i in 0..n_e {
        let e = electron_energy[i];

        // Trapezoidal integration of DCS over reduced photon energy k
        let mut c = 0.0;
        for j in 1..n_k {
            c += 0.5 * (dcs[i][j] + dcs[i][j - 1]) * (photon_energy[j] - photon_energy[j - 1]);
        }

        let beta_sq =
            e * (e + 2.0 * mass_electron_ev) / ((e + mass_electron_ev) * (e + mass_electron_ev));
        let s_rad = (z as f64) * (z as f64) / beta_sq * e * c;
        stopping_power_radiative.push(s_rad);
    }

    Ok((
        dcs,
        stopping_power_radiative,
        ionization_energy,
        n_electrons,
        mean_excitation_energy,
        electron_energy,
        photon_energy,
    ))
}
