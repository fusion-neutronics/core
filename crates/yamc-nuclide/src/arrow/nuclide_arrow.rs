//! Arrow IPC format reader for neutron nuclear data.
//!
//! Reads a `.arrow/` directory containing pre-computed nuclear data in Apache Arrow IPC format.
//! This replaces the HDF5 reader and provides near-zero load times since the FastXSGrid
//! is pre-computed by the Python converter.

use crate::buffer::F64Buffer;
use crate::fission_photon::{FissionPhotonRelease, ReleaseFunction};
use crate::load_scope::{LoadScope, SectionScope};
use crate::nuclide::{FastXSGrid, FissionNuData, Nuclide};
use crate::particle_type::ParticleType;
use crate::reaction::Reaction;
use crate::reaction_product::{
    AngleDistribution, AngleEnergyDistribution, EnergyDistribution, ReactionProduct, Tabulated,
    Tabulated1D, TabulatedInterp, TabulatedProbability, Yield,
};
use crate::secondary_correlated;
use crate::secondary_kalbach;
use crate::urr::{UrrData, UrrInterpolation, UrrXsSet};

use crate::arrow_helpers::{
    borrow_f64_list, borrow_i32_list, borrow_nested_f64_list, get_bool, get_f64, get_f64_list,
    get_i32, get_i32_list, get_str, get_str_list, read_arrow_file, try_get_f64, try_get_f64_list,
    try_get_i32, try_get_i32_list, try_get_str,
};

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch};
use arrow_buffer::ScalarBuffer;

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::path::Path;
use std::sync::Arc;

// =============================================================================
// Main reader
// =============================================================================

/// Read the optional `fission_photon.arrow` section (issue #369).
///
/// Layout is one row per term, keyed by `role` ("prompt_photons" /
/// "delayed_photons"), so a term can be a polynomial or a table independently of
/// the other -- which is the real case: U235, U238 and Pu239 store a tabulated
/// prompt term alongside a polynomial delayed one.
///
/// Returns `Ok(None)` when the section is absent, which is every file published
/// before it existed and every nuclide without `fission_energy_release`. A
/// section that is present but unreadable, or that uses an interpolation scheme
/// this cannot evaluate exactly, is an ERROR -- falling back to no scaling would
/// silently restore the actinide photon deficit.
fn read_fission_photon_release(
    dir: &Path,
    nuclide: &str,
) -> Result<Option<FissionPhotonRelease>, Box<dyn Error>> {
    let path = dir.join("fission_photon.arrow");
    if !path.exists() {
        return Ok(None);
    }
    let batch = read_arrow_file(&path)?;

    let mut prompt = None;
    let mut delayed = None;
    for row in 0..batch.num_rows() {
        let role = get_str(&batch, "role", row)?;
        let kind = get_str(&batch, "kind", row)?;
        let context = format!("{nuclide} {role}");
        let func = match kind.as_str() {
            "polynomial" => {
                let coeffs = get_f64_list(&batch, "coefficients", row)?;
                if coeffs.is_empty() {
                    return Err(format!(
                        "{context}: polynomial fission energy release has no coefficients"
                    )
                    .into());
                }
                ReleaseFunction::Polynomial(coeffs)
            }
            "tabulated" => ReleaseFunction::from_tabulated(
                get_f64_list(&batch, "x", row)?,
                get_f64_list(&batch, "y", row)?,
                &get_i32_list(&batch, "interpolation", row)?,
                &get_i32_list(&batch, "breakpoints", row)?,
                &context,
            )?,
            other => {
                return Err(format!(
                    "{context}: unknown fission energy release kind {other:?}; expected \
                     \"polynomial\" or \"tabulated\""
                )
                .into())
            }
        };
        match role.as_str() {
            "prompt_photons" => prompt = Some(func),
            "delayed_photons" => delayed = Some(func),
            other => {
                return Err(format!(
                    "{nuclide}: unknown fission energy release role {other:?}; expected \
                     \"prompt_photons\" or \"delayed_photons\""
                )
                .into())
            }
        }
    }

    match (prompt, delayed) {
        (Some(prompt), Some(delayed)) => Ok(Some(FissionPhotonRelease { prompt, delayed })),
        // Both terms are needed to form the ratio, so half a section is a
        // converter bug worth surfacing rather than quietly ignoring.
        (p, d) => Err(format!(
            "{nuclide}: fission_photon.arrow is incomplete (prompt_photons {}, \
             delayed_photons {}); both terms are required to form the scaling",
            if p.is_some() { "present" } else { "MISSING" },
            if d.is_some() { "present" } else { "MISSING" },
        )
        .into()),
    }
}

/// Take an Arrow column view either by sharing its allocation or by copying out
/// of it (issue #476, task 1).
///
/// Sharing skips the copy, but a `ScalarBuffer` view holds an `Arc` on the whole
/// column, so it pins every row rather than the one it spans. That is a win when
/// the load keeps every row and a regression when it does not: an MT- or
/// temperature-filtered load exists precisely to drop most of the column, and
/// sharing one cell of it would keep all of it resident. So `share` comes from
/// [`LoadScope::is_unfiltered`], and a filtered load copies exactly as it did
/// before, leaving the phase-by-phase batch drops below still effective.
///
/// Used for `nuclide.arrow` and `reactions.arrow`, whose bytes are almost
/// entirely the grids and cross sections an unfiltered load keeps. `fast_xs.arrow`
/// deliberately never shares; see [`build_fast_xs_from_arrow`].
fn adopt(values: &ScalarBuffer<f64>, share: bool) -> F64Buffer {
    if share {
        F64Buffer::share(values.clone())
    } else {
        F64Buffer::from_slice(values)
    }
}

/// The sections a `Full` load has always required outright.
///
/// `urr.arrow`, `fission_photon.arrow` and the nu tables are optional already,
/// each handled where it is read.
const TRANSPORT_SECTIONS: [&str; 3] = ["products.arrow", "distributions.arrow", "fast_xs.arrow"];

/// Which file holds this load's cross sections.
///
/// Normally `reactions.arrow`, the whole published object. A download that read
/// only some MTs caches those as a spliced stream at `subset/reactions.arrow`
/// instead (see `SUBSET_REACTIONS` in `storage/url_cache.rs`), and the canonical
/// name is deliberately left absent so that nothing mistakes a partial for a
/// complete section.
///
/// The whole object wins whenever it is there, so a directory that has since
/// been topped up for transport reads the complete file and the leftover subset
/// is ignored.
///
/// Only a scope that reads some MTs may fall back to the subset. A transport
/// load reaching a subset-only directory finds no `reactions.arrow` and fails
/// naming it, which is the right answer: the file it needs has not been
/// downloaded. Silently reading the subset would hand transport a nuclide
/// missing whole channels.
fn reactions_path(dir: &Path, scope: &LoadScope) -> std::path::PathBuf {
    let whole = dir.join("reactions.arrow");
    if scope.wants_transport_sections() || crate::storage::exists(&whole) {
        return whole;
    }
    let subset = dir.join("subset").join("reactions.arrow");
    if crate::storage::exists(&subset) {
        return subset;
    }
    whole
}

/// Narrow a `Full` request to the sections the directory actually carries.
///
/// `convert_neutron_xs` writes `nuclide.arrow`, `reactions.arrow` and
/// `version.json` and nothing else. That is the whole point of it -- a
/// cross-sections-only conversion -- and its output drives `Material.transmute`
/// correctly, because transmutation asks for [`SectionScope::XsOnly`] anyway. It
/// was unreadable through `Nuclide.read_nuclear_data` only because that asks for
/// `Full`, and the reader then required the transport sections whatever was on
/// disk (issue #506).
///
/// A directory carrying NONE of them is that conversion, and loads at `XsOnly`.
/// The narrowed scope is recorded on the [`Nuclide`], so transport still cannot
/// pick one up by accident: that is the same guard which already keeps an
/// activation load away from transport.
///
/// A directory carrying SOME of them is a different fault -- an interrupted
/// conversion, or a cache directory half-updated between library versions --
/// where narrowing would hide the damage. That names what is missing and fails.
///
/// The `.absent` markers a hosted directory carries play no part here. They are
/// a download-cache record of a settled 404 (`url_cache.rs`) and this reader
/// never consults them, which is why writing `fast_xs.arrow.absent` by hand
/// changes nothing.
fn narrow_to_present_sections(dir: &Path, scope: &LoadScope) -> Result<LoadScope, Box<dyn Error>> {
    if !scope.wants_transport_sections() {
        return Ok(scope.clone());
    }

    // A directory with no `reactions.arrow` at all is not the cross-sections-only
    // conversion either, because that writes one. It is a download cache holding
    // only a ranged subset under `subset/`, and narrowing onto it would send this
    // transport request to a reactions table missing whole channels. Leave the
    // scope alone, so the read below fails naming the file it wanted.
    if !crate::storage::exists(&dir.join("reactions.arrow")) {
        return Ok(scope.clone());
    }

    let missing: Vec<&str> = TRANSPORT_SECTIONS
        .iter()
        .copied()
        .filter(|section| !crate::storage::exists(&dir.join(section)))
        .collect();

    if missing.is_empty() {
        return Ok(scope.clone());
    }
    if missing.len() < TRANSPORT_SECTIONS.len() {
        return Err(format!(
            "{}: missing {}. The directory carries some transport sections and not others, so \
             it is neither a complete conversion nor the cross-sections-only conversion \
             convert_neutron_xs writes. That is what an interrupted conversion or a directory \
             half-updated between library versions looks like. Re-convert or re-download this \
             nuclide.",
            dir.display(),
            missing.join(", "),
        )
        .into());
    }

    Ok(LoadScope {
        sections: SectionScope::XsOnly,
        mts: scope.mts.clone(),
        temperatures: scope.temperatures.clone(),
        // Narrowing is about the transport sections a directory does not have.
        // Covariance is a separate file and its own decision, so it survives.
        covariance: scope.covariance,
    })
}

/// Read a nuclide from a `{Name}.arrow/` section directory.
///
/// `scope` selects the subset to materialize. [`LoadScope::full`] reads
/// everything, which is what transport needs. A [`SectionScope::XsOnly`] scope
/// reads only `nuclide.arrow` and `reactions.arrow`, and only the MTs the scope
/// names: that is all a transport-free reaction-rate collapse touches, and it
/// skips the `fast_xs` accelerator that dominates the on-disk size. The scope is
/// recorded on the returned nuclide so the cache can refuse to hand a narrow
/// load to a caller that needs a wide one.
pub fn read_nuclide_from_arrow(dir: &Path, scope: &LoadScope) -> Result<Nuclide, Box<dyn Error>> {
    let temps_filter = scope.temperatures.as_ref();
    // Whether the large numeric columns can be shared rather than copied out of.
    // See `adopt`: sharing keeps the whole column alive, which only pays when
    // this load keeps every row of it.
    let share_buffers = scope.is_unfiltered();
    // Route the "does the directory exist?" probe through the configured
    // `Storage` backend so browser builds with `InMemoryStorage` work too
    // (`Path::is_dir` always returns false on `wasm32-unknown-unknown`).
    // `Storage::exists` matches a directory if any file under it has been
    // registered; on native it falls through to `Path::exists` which
    // accepts both files and directories. A path that points to a regular
    // file rather than a directory will fail at the first sub-file read
    // (`dir.join("nuclide.arrow")`) instead of here -- same end-user
    // outcome, broader compatibility.
    if !crate::storage::exists(dir) {
        return Err(format!("Arrow directory not found: {}", dir.display()).into());
    }

    // What this load can actually parse, which is not always what was asked
    // for: a cross-sections-only directory narrows a `Full` request rather than
    // failing it. Shadows the argument so every section decision below, and the
    // scope recorded on the nuclide, speaks for what is really on disk.
    let scope = &narrow_to_present_sections(dir, scope)?;

    // Check version.json
    let version_path = dir.join("version.json");
    if crate::storage::exists(&version_path) {
        let version_str = crate::storage::read_to_string(&version_path)?;
        let version: serde_json::Value = serde_json::from_str(&version_str)?;
        let fmt_version = version
            .get("format_version")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        if fmt_version != 1 {
            return Err(format!("Unsupported Arrow format version: {fmt_version}").into());
        }
    }

    // 1. Read nuclide.arrow - basic metadata
    let nuclide_batch = read_arrow_file(&dir.join("nuclide.arrow"))?;
    let name = get_str(&nuclide_batch, "name", 0)?;
    let z = get_i32(&nuclide_batch, "Z", 0)? as u32;
    let a = get_i32(&nuclide_batch, "A", 0)? as u32;
    let awr = get_f64(&nuclide_batch, "atomic_weight_ratio", 0)?;
    // Arrow files store temperatures with "K" suffix (e.g., "294K").
    // The rest of the codebase uses bare numbers (e.g., "294").
    // We strip "K" here for consistency, but keep the originals for Arrow file lookups.
    let all_temps_raw = get_str_list(&nuclide_batch, "temperatures", 0)?;
    let energy_temps_raw = get_str_list(&nuclide_batch, "energy_temperatures", 0)?;
    let energy_values = borrow_nested_f64_list(&nuclide_batch, "energy_values", 0)?;

    // Strip "K" suffix for normalized keys
    let strip_k = |s: &str| crate::temperature::strip_k(s).to_string();
    // Sorted numerically, not in file order: the published data lists
    // temperatures lexicographically ("1200" before "250"), which would leak a
    // storage detail into `available_temperatures` and make the order depend on
    // which vintage of a file a user happens to have. Every lookup keys by
    // label, so this is presentation only.
    let mut all_temps: Vec<String> = all_temps_raw.iter().map(|t| strip_k(t)).collect();
    all_temps.sort_by(|a, b| match (a.parse::<f64>(), b.parse::<f64>()) {
        (Ok(x), Ok(y)) => x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal),
        _ => a.cmp(b),
    });

    // Fission energy release for the delayed-photon scaling (issue #369), read
    // from its own optional section. OPTIONAL twice over: only evaluations with
    // `fission_energy_release` carry it (79 of 557 in ENDF/B-VIII.1), and files
    // published before the section existed have no such file at all. Either way
    // the scaling stays off and every consumer takes its `f = 1.0` branch, which
    // is what every file did before.
    //
    // A file that IS present but malformed is a hard error rather than a silent
    // fallback: quietly dropping to `f = 1.0` there would reintroduce exactly the
    // ~38% actinide photon deficit this fixes, with nothing to show for it.
    let photon_release = if scope.wants_transport_sections() {
        read_fission_photon_release(dir, &name)?
    } else {
        None
    };

    // Determine which temperatures to load (filter uses normalized keys)
    let loaded_temps: Vec<String> = if let Some(filter) = temps_filter {
        all_temps
            .iter()
            .filter(|t| filter.contains(t.as_str()))
            .cloned()
            .collect()
    } else {
        all_temps.clone()
    };

    if loaded_temps.is_empty() {
        return Err(
            format!("No matching temperatures found for {name}. Available: {all_temps:?}").into(),
        );
    }

    // Build energy grids per temperature (using normalized keys), keeping only
    // the temperatures this load actually asked for.
    //
    // The published files carry one full union grid per available temperature
    // (Fe56: seven grids, 2.4 MB, of which a 294 K load needs 0.34 MB), and the
    // reactions are already narrowed to `loaded_temps`, so a grid for a
    // temperature with no reactions behind it can never be interpolated on.
    // Every consumer reaches the map through `energy_grid(temperature)` with the
    // material's own temperature label.
    //
    // Grids whose label is not one of the file's `temperatures` (the 0 K grid,
    // which is listed under `energy_temperatures` only) are kept regardless: no
    // temperature filter can ever name them, so dropping them on a filtered load
    // would silently change what an unfiltered load provides.
    let energy_map: HashMap<String, F64Buffer> = energy_temps_raw
        .iter()
        .map(|t| strip_k(t))
        .zip(energy_values.iter())
        .filter(|(key, _)| {
            loaded_temps.iter().any(|t| t == key) || !all_temps.iter().any(|t| t == key)
        })
        // Adopt after filtering, so a temperature this load drops is never
        // copied in the first place.
        .map(|(key, values)| (key, adopt(values, share_buffers)))
        .collect();

    // Phase 1: Read reactions + products + distributions, build reactions_per_temp,
    // then drop the Arrow batches before loading the much larger fast_xs.arrow.
    // This avoids having ~117 MB (reactions) + 308 MB (fast_xs) live simultaneously.
    //
    // Under an XsOnly scope the products and distributions are skipped outright:
    // they describe what comes OUT of a reaction, which only transport samples.
    // A reaction-rate collapse needs sigma(E) and nothing else.
    let mut reactions_per_temp: Vec<HashMap<i32, Arc<Reaction>>> = Vec::new();
    let fissionable;

    {
        let reactions_batch = read_arrow_file(&reactions_path(dir, scope))?;

        // Build products lookup, keyed by MT, empty when the scope excludes them.
        let mut products_map: HashMap<i32, Vec<ReactionProduct>> = HashMap::new();
        if scope.wants_transport_sections() {
            let products_batch = read_arrow_file(&dir.join("products.arrow"))?;
            let distributions_batch = read_arrow_file(&dir.join("distributions.arrow"))?;

            // Build distributions lookup
            let mut distributions_map: HashMap<(i32, i32, i32), ParsedDistribution> =
                HashMap::new();
            for row in 0..distributions_batch.num_rows() {
                let mt = get_i32(&distributions_batch, "reaction_mt", row)?;
                let prod_idx = get_i32(&distributions_batch, "product_idx", row)?;
                let dist_idx = get_i32(&distributions_batch, "dist_idx", row)?;
                let dist = parse_distribution(&distributions_batch, row)?;
                distributions_map.insert((mt, prod_idx, dist_idx), dist);
            }
            // Drop distributions_batch early (16 MB)
            drop(distributions_batch);

            build_products_map(&products_batch, &distributions_map, &mut products_map)?;
            // Drop products_batch and distributions_map (no longer needed)
            drop(products_batch);
            drop(distributions_map);
        }

        let mut has_fission = false;
        let n_reactions = reactions_batch.num_rows();

        for temp_key in &loaded_temps {
            let mut temp_reactions: HashMap<i32, Arc<Reaction>> = HashMap::new();
            // Hoisted out of the MT loop: every reaction at this temperature
            // views the same grid.
            let energy_grid = energy_map.get(temp_key);

            for rxn_row in 0..n_reactions {
                let mt = get_i32(&reactions_batch, "mt", rxn_row)?;
                // Bail before the nested-list reads below, which are what
                // actually cost: on Fe56 the nine full-grid MTs the activation
                // path never asks for (total, elastic, nonelastic, inelastic,
                // absorption, disappearance, heating, damage) are 1.36 MB each
                // against 0.14 MB for a threshold reaction.
                if !scope.wants_mt(mt) {
                    continue;
                }
                let q_value = get_f64(&reactions_batch, "Q_value", rxn_row)?;
                let center_of_mass = get_bool(&reactions_batch, "center_of_mass", rxn_row)?;
                let redundant = get_bool(&reactions_batch, "redundant", rxn_row)?;
                let xs_temps = get_str_list(&reactions_batch, "xs_temperatures", rxn_row)?;
                let xs_values = borrow_nested_f64_list(&reactions_batch, "xs_values", rxn_row)?;
                let xs_thresholds = get_i32_list(&reactions_batch, "xs_threshold_idx", rxn_row)?;

                let temp_idx = match xs_temps
                    .iter()
                    .position(|t| crate::temperature::strip_k(t) == temp_key)
                {
                    Some(idx) => idx,
                    None => continue,
                };

                let cross_section = match xs_values.get(temp_idx) {
                    Some(values) => adopt(values, share_buffers),
                    None => continue,
                };

                let threshold_idx = if temp_idx < xs_thresholds.len() {
                    xs_thresholds[temp_idx] as usize
                } else {
                    0
                };

                // A view of the nuclide's grid, not a copy of it. This used
                // to be a `to_vec()` per (MT, temperature) pair over the same
                // few tens of thousands of points: on ENDF/B-VIII.1 Fe56, 38 MiB
                // of duplicate grid behind 2.3 MiB of distinct grid. Sharing is
                // unconditional here (no `share_buffers` check) because the
                // nuclide holds `energy_map` for the whole of its own lifetime,
                // so the slice pins nothing the load was going to release.
                let reaction_energy = match energy_grid {
                    Some(grid) if threshold_idx > 0 && threshold_idx < grid.len() => {
                        grid.tail(threshold_idx)
                    }
                    Some(grid) => grid.clone(),
                    None => F64Buffer::default(),
                };

                // `cross_section_at` binary-searches `energy` and indexes
                // `cross_section` with what it finds, so a file whose two
                // sections disagree panics on the first lookup rather than
                // failing to load (issue #507). Establishing the invariant here
                // costs one comparison per (MT, temperature) at load and keeps
                // that hot path a plain index.
                //
                // Skipped when the temperature carries no grid at all: that
                // leaves `energy` empty, which `cross_section_at` already
                // reports as no data, and is a different fault from two
                // sections that disagree.
                if energy_grid.is_some() && reaction_energy.len() != cross_section.len() {
                    return Err(format!(
                        "{}: {name} MT {mt} at {temp_key} K has {} energy points against {} \
                         cross section values (threshold index {threshold_idx}). nuclide.arrow \
                         and reactions.arrow disagree, which is what a truncated download, an \
                         interrupted conversion or a directory half-updated between library \
                         versions looks like. Re-convert or re-download this nuclide.",
                        dir.display(),
                        reaction_energy.len(),
                        cross_section.len(),
                    )
                    .into());
                }

                let products = products_map.get(&mt).cloned().unwrap_or_default();

                if crate::nuclide::is_fission_mt(mt) && !redundant {
                    has_fission = true;
                }

                let reaction = Reaction {
                    cross_section,
                    threshold_idx,
                    energy: reaction_energy,
                    mt_number: mt,
                    q_value,
                    products,
                    scatter_in_cm: center_of_mass,
                    redundant,
                };

                temp_reactions.insert(mt, Arc::new(reaction));
            }

            reactions_per_temp.push(temp_reactions);
        }

        fissionable = has_fission;
        // reactions_batch, products_map drop here -- freeing ~117 MB before fast_xs load
    }

    // Phase 2: Read fast_xs.arrow (308 MB for U238) -- reactions batch is already freed.
    //
    // This is the single largest section (61% of the TENDL-2025 library on disk)
    // and it is purely a transport lookup accelerator, so an XsOnly load leaves
    // the vector EMPTY rather than filling it with defaults. A default grid would
    // answer every lookup with zero; an empty one cannot be indexed at all, and
    // the scope recorded on the nuclide keeps transport from ever trying.
    let mut fast_xs_grids: Vec<FastXSGrid> = Vec::new();
    if scope.wants_transport_sections() {
        let fast_xs_batch = read_arrow_file(&dir.join("fast_xs.arrow"))?;

        for (temp_idx, temp_key) in loaded_temps.iter().enumerate() {
            let fast_row = find_temp_row(&fast_xs_batch, temp_key);
            let temp_reactions = &reactions_per_temp[temp_idx];

            match fast_row {
                Some(row) => {
                    let fast_grid = build_fast_xs_from_arrow(
                        &fast_xs_batch,
                        row,
                        temp_reactions,
                        energy_map.get(temp_key),
                        photon_release.as_ref(),
                        temp_key,
                    )
                    // Every rejection out of the builder should name the nuclide
                    // it came from, not only the temperature.
                    .map_err(|e| format!("{}: {e}", dir.display()))?;
                    fast_xs_grids.push(fast_grid);
                }
                None => {
                    fast_xs_grids.push(FastXSGrid::default());
                }
            }
        }
        // fast_xs_batch drops here -- freeing ~308 MB before URR/nu loads
    }

    // Phase 3: Read small optional files. Both are transport-only: URR tables
    // resolve the unresolved-resonance band during tracking, and nu-bar counts
    // fission progeny. A transmutation network takes its fission yields from the
    // chain, not from here.
    let mut urr_data: Vec<Option<UrrData>> = Vec::new();
    let mut urr_present = false;
    let urr_path = dir.join("urr.arrow");
    if !scope.wants_transport_sections() {
        urr_data.resize_with(loaded_temps.len(), || None);
    } else if urr_path.exists() {
        let urr_batch = read_arrow_file(&urr_path)?;
        for temp_key in &loaded_temps {
            match find_temp_row(&urr_batch, temp_key) {
                Some(row) => {
                    let urr = parse_urr(&urr_batch, row)?;
                    urr_present = true;
                    urr_data.push(Some(urr));
                }
                None => urr_data.push(None),
            }
        }
    } else {
        urr_data.resize_with(loaded_temps.len(), || None);
    }

    // MF=33 covariance, and only when asked for. Its own axis of the scope
    // rather than part of the transport/activation split: an uncertainty run
    // wants it with an XsOnly load and transport wants a Full load without it.
    // Absence is not an error at any scope -- most evaluations have no MF=33,
    // and every directory published before the section existed has no file.
    let covariance = if scope.covariance {
        crate::arrow::covariance_arrow::read_covariance(dir, &name)?.map(std::sync::Arc::new)
    } else {
        None
    };

    let mut fission_nu: Option<FissionNuData> = None;
    let total_nu_path = dir.join("total_nu.arrow");
    if !scope.wants_transport_sections() {
        // Left as None: the yield fallback below reads reaction products, which
        // an XsOnly load does not carry.
    } else if total_nu_path.exists() {
        let nu_batch = read_arrow_file(&total_nu_path)?;
        fission_nu = Some(parse_fission_nu(&nu_batch)?);
    } else if fissionable {
        // Fallback: extract nu-bar from fission reaction product yield.
        // When total_nu.arrow is missing, the fission product's neutron yield IS the nu-bar.
        // Compute nu from reaction products when a separate dataset is not available.
        if let Some(first_temp_reactions) = reactions_per_temp.first() {
            for &mt in &[18i32, 19, 20, 21, 38] {
                if let Some(rxn) = first_temp_reactions.get(&mt) {
                    if let Some(neutron_prod) = rxn.products.iter().find(|p| {
                        p.is_particle_type(&crate::reaction_product::ParticleType::Neutron)
                    }) {
                        if let Some(ref y) = neutron_prod.product_yield {
                            fission_nu = Some(fission_nu_from_yield(y));
                            break;
                        }
                    }
                }
            }
        }
    }

    let atomic_symbol = name
        .chars()
        .take_while(|c| c.is_alphabetic())
        .collect::<String>();

    let nuclide = Nuclide {
        name: Some(name),
        element: crate::nuclide::element_name_from_z(z),
        atomic_symbol: Some(atomic_symbol),
        atomic_number: Some(z),
        neutron_number: Some(a.saturating_sub(z)),
        mass_number: Some(a),
        atomic_weight_ratio: Some(awr),
        library: None,
        energy: Some(energy_map),
        reactions: reactions_per_temp,
        fissionable,
        available_temperatures: all_temps,
        loaded_temperatures: loaded_temps,
        data_path: Some(dir.to_string_lossy().to_string()),
        fission_nu,
        fast_xs: fast_xs_grids,
        urr_data,
        urr_present,
        fission_photon_release: photon_release,
        covariance,
        elastic_flat_cache: Default::default(),
        fission_chi_flat_cache: Default::default(),
        delayed_neutron_cache: Default::default(),
        inelastic_angle_flat_cache: Default::default(),
        load_scope: scope.clone(),
    };

    Ok(nuclide)
}

/// Parse `products.arrow` into a per-MT product lookup, resolving each product's
/// secondary distributions out of `distributions_map`.
fn build_products_map(
    products_batch: &RecordBatch,
    distributions_map: &HashMap<(i32, i32, i32), ParsedDistribution>,
    products_map: &mut HashMap<i32, Vec<ReactionProduct>>,
) -> Result<(), Box<dyn Error>> {
    for row in 0..products_batch.num_rows() {
        let mt = get_i32(products_batch, "reaction_mt", row)?;
        let prod_idx = get_i32(products_batch, "product_idx", row)?;
        let n_dist = get_i32(products_batch, "n_distribution", row)?;
        let product = parse_product(products_batch, row, mt, prod_idx, n_dist, distributions_map)?;
        products_map.entry(mt).or_default().push(product);
    }
    Ok(())
}

// =============================================================================
// Fast XS grid construction from pre-computed Arrow data
// =============================================================================

fn find_temp_row(batch: &RecordBatch, temp_key: &str) -> Option<usize> {
    let col = batch.column_by_name("temperature")?;
    let str_arr = col.as_string::<i32>();
    // Try exact match first, then try with "K" suffix (Arrow files store "294K")
    (0..str_arr.len()).find(|&i| {
        let v = str_arr.value(i);
        v == temp_key || crate::temperature::strip_k(v) == temp_key
    })
}

/// Narrow the on-disk `int32` log-grid index to the `u32` the runtime keeps,
/// checking the invariants [`FastXSGrid::lookup`] relies on and does not clamp.
///
/// The column is `int32` on disk and the values are only ever slice indices, so
/// the width says nothing true and the sign is a liability: `v as usize` turns a
/// negative into ~1.8e19 and the first symptom is a panic inside a cross-section
/// lookup. The quieter failure is worse. An entry that is in range but out of
/// order makes `partition_point` return 0, `saturating_sub(1)` pins `i_grid` to
/// `i_low`, and the lookup interpolates between the wrong pair of grid points:
/// a plausible-looking wrong cross section. Both are rejected here, once, at
/// load. Issue #482.
///
/// Deliberately NOT checked: that the last entry is `n_energy - 1`. Only the
/// Rust converter forces that; the Python writer's `searchsorted` rounding
/// leaves it short, as 38,232 of the 41,382 published rows are.
fn log_grid_index_u32(
    values: &[i32],
    n_energy: usize,
    temp_key: &str,
) -> Result<Vec<u32>, Box<dyn Error>> {
    if n_energy > 0 && values.len() < 2 {
        return Err(format!(
            "log_grid_index at {temp_key} K has {} entries; the lookup reads a \
             bin and its successor, so it needs at least 2",
            values.len()
        )
        .into());
    }
    let mut out = Vec::with_capacity(values.len());
    let mut previous = 0i32;
    for (i, &v) in values.iter().enumerate() {
        if v < 0 {
            return Err(format!(
                "log_grid_index[{i}] at {temp_key} K is {v}; a grid index cannot be negative"
            )
            .into());
        }
        if n_energy > 0 && v as usize >= n_energy {
            return Err(format!(
                "log_grid_index[{i}] at {temp_key} K is {v}, past the {n_energy}-point \
                 energy grid it indexes"
            )
            .into());
        }
        if i > 0 && v < previous {
            return Err(format!(
                "log_grid_index at {temp_key} K decreases at {i}: {previous} then {v}. \
                 A bracket that runs backwards silently interpolates between the \
                 wrong pair of grid points"
            )
            .into());
        }
        previous = v;
        out.push(v as u32);
    }
    Ok(out)
}

fn build_fast_xs_from_arrow(
    batch: &RecordBatch,
    row: usize,
    temp_reactions: &HashMap<i32, Arc<Reaction>>,
    // The nuclide's union grid for this temperature, so the accelerator can
    // point at it rather than keep a second copy of the same numbers.
    nuclide_grid: Option<&F64Buffer>,
    // Fission energy release for the delayed-photon scaling (issue #369). `None`
    // when the evaluation has none, or the published file predates the section;
    // the scaling then stays empty and every consumer takes its `f = 1.0` branch,
    // exactly as before.
    photon_release: Option<&FissionPhotonRelease>,
    // Names the temperature in a log-grid-index rejection, since one bad row
    // should say which of the nuclide's temperatures it is.
    temp_key: &str,
) -> Result<FastXSGrid, Box<dyn Error>> {
    let log_e_min = get_f64(batch, "log_e_min", row)?;
    let inv_log_delta = get_f64(batch, "inv_log_delta", row)?;

    // Nothing here shares out of `batch`, at any scope. `xs`, `scatter_mt_xs`
    // and `fission_mt_xs` are reshaped or column-selected on the way in and so
    // have to be rebuilt regardless; sharing the narrower columns alongside them
    // would keep the whole section resident (308 MB on U238, see phase 2 above)
    // *and* keep the rebuilds, for a net loss. The views below are read sources
    // only, which is still a win: they replace the intermediate `Vec` each
    // rebuild used to copy first.
    let xs_flat = borrow_f64_list(batch, "xs", row)?;
    let xs_shape = get_i32_list(batch, "xs_shape", row)?;
    let energy_col = borrow_f64_list(batch, "energy", row)?;
    // The one exception, and it does not touch `batch`: `fast_xs.arrow` repeats
    // the nuclide's union grid for this temperature, so where the two agree,
    // point at the grid the nuclide holds for its whole lifetime anyway rather
    // than keeping a second copy. The comparison is a memcmp over the grid.
    let energy = match nuclide_grid {
        Some(grid) if grid.as_slice() == energy_col.as_ref() => grid.clone(),
        _ => F64Buffer::from_slice(&energy_col),
    };

    // Below the grid read because the range check needs its length. `energy` is
    // `energy_col.len()` long either way, whether it aliased the nuclide grid or
    // copied the column.
    let log_grid_index = log_grid_index_u32(
        &borrow_i32_list(batch, "log_grid_index", row)?,
        energy_col.len(),
        temp_key,
    )?;

    let n_energy = if !xs_shape.is_empty() {
        xs_shape[0] as usize
    } else {
        energy.len()
    };
    let n_xs_cols = if xs_shape.len() >= 2 {
        xs_shape[1] as usize
    } else {
        4
    };

    // Reshape xs from flat [n_energy * 4] to Vec<[f64; 4]>
    let mut xs = Vec::with_capacity(n_energy);
    for i in 0..n_energy {
        let base = i * n_xs_cols;
        let mut arr = [0.0f64; 4];
        for j in 0..4.min(n_xs_cols) {
            if base + j < xs_flat.len() {
                arr[j] = xs_flat[base + j];
            }
        }
        xs.push(arr);
    }

    // Build scatter_mt_xs with Arc<Reaction> pointers
    let scatter_mt_numbers = get_i32_list(batch, "scatter_mt_numbers", row)?;
    // Read as a view: the buffer below selects columns out of it, so the
    // intermediate never needed to be a copy.
    let scatter_xs_flat = borrow_f64_list(batch, "scatter_mt_xs", row)?;
    let scatter_shape = get_i32_list(batch, "scatter_mt_shape", row)?;
    let n_scatter_energies = if !scatter_shape.is_empty() {
        scatter_shape[0] as usize
    } else {
        n_energy
    };

    // Pick the non-redundant MT columns and rebuild a flat row-major
    // [n_scatter_energies, n_kept_mts] buffer by selecting only those
    // columns from the disk layout. The disk buffer is already row-major;
    // we just need to drop redundant-reaction columns.
    let mut scatter_mt_numbers_out: Vec<i32> = Vec::new();
    let mut scatter_mt_reactions: Vec<Arc<Reaction>> = Vec::new();
    let mut scatter_kept_cols: Vec<usize> = Vec::new();
    let n_scatter_mts = scatter_mt_numbers.len();
    for (mt_idx, &mt) in scatter_mt_numbers.iter().enumerate() {
        if let Some(reaction) = temp_reactions.get(&mt) {
            // Skip redundant reactions (e.g. MT 16 when level-specific MT 875-890 exist)
            if reaction.redundant {
                continue;
            }
            scatter_mt_numbers_out.push(mt);
            scatter_mt_reactions.push(Arc::clone(reaction));
            scatter_kept_cols.push(mt_idx);
        }
    }
    let n_scatter_kept = scatter_kept_cols.len();
    let scatter_mt_xs: F64Buffer =
        if !scatter_xs_flat.is_empty() && n_scatter_mts > 0 && n_scatter_kept > 0 {
            let mut out = Vec::with_capacity(n_scatter_energies * n_scatter_kept);
            for i in 0..n_scatter_energies {
                for &col in &scatter_kept_cols {
                    let idx = i * n_scatter_mts + col;
                    out.push(scatter_xs_flat.get(idx).copied().unwrap_or(0.0));
                }
            }
            out.into()
        } else {
            F64Buffer::default()
        };
    let elastic_idx = scatter_mt_numbers_out.iter().position(|&mt| mt == 2);
    // Canonical non-elastic walk order (issue #111): the permutation of the
    // non-elastic scatter columns into `INELASTIC_MT_SLOTS` order, so the
    // per-collision cumulative walk visits reactions in the same sequence the
    // GPU kernel sweeps its per-MT slots in.
    let inelastic_walk_order =
        FastXSGrid::build_inelastic_walk_order(&scatter_mt_numbers_out, elastic_idx);

    // Build fission_mt_xs flat buffer (same pattern as scatter)
    let fission_mt_numbers = get_i32_list(batch, "fission_mt_numbers", row)?;
    let fission_xs_flat = borrow_f64_list(batch, "fission_mt_xs", row)?;
    let fission_shape = get_i32_list(batch, "fission_mt_shape", row)?;
    let n_fission_energies = if !fission_shape.is_empty() {
        fission_shape[0] as usize
    } else {
        n_energy
    };
    let has_partial_fission = get_bool(batch, "has_partial_fission", row)?;

    let mut fission_mt_numbers_out: Vec<i32> = Vec::new();
    let mut fission_mt_reactions: Vec<Arc<Reaction>> = Vec::new();
    let mut fission_kept_cols: Vec<usize> = Vec::new();
    let n_fission_mts = fission_mt_numbers.len();
    for (mt_idx, &mt) in fission_mt_numbers.iter().enumerate() {
        if let Some(reaction) = temp_reactions.get(&mt) {
            if reaction.redundant {
                continue;
            }
            fission_mt_numbers_out.push(mt);
            fission_mt_reactions.push(Arc::clone(reaction));
            fission_kept_cols.push(mt_idx);
        }
    }
    let n_fission_kept = fission_kept_cols.len();
    let fission_mt_xs: F64Buffer =
        if !fission_xs_flat.is_empty() && n_fission_mts > 0 && n_fission_kept > 0 {
            let mut out = Vec::with_capacity(n_fission_energies * n_fission_kept);
            for i in 0..n_fission_energies {
                for &col in &fission_kept_cols {
                    let idx = i * n_fission_mts + col;
                    out.push(fission_xs_flat.get(idx).copied().unwrap_or(0.0));
                }
            }
            out.into()
        } else {
            F64Buffer::default()
        };

    let reaction_absorption = temp_reactions.get(&101).map(Arc::clone);

    let xs_ngamma = F64Buffer::from_slice(&borrow_f64_list(batch, "xs_ngamma", row)?);
    let photon_prod = F64Buffer::from_slice(&borrow_f64_list(batch, "photon_prod", row)?);

    // photon_rxn_xs - compute from reactions (not in Arrow). Build per-MT
    // temporaries then flatten row-major at the end.
    let mut photon_rxn_mt_numbers: Vec<i32> = Vec::new();
    let mut photon_rxn_xs_perrx: Vec<Vec<f64>> = Vec::new();
    let mut photon_rxn_reactions: Vec<Arc<Reaction>> = Vec::new();
    {
        let mut sorted_mts: Vec<i32> = temp_reactions.keys().copied().collect();
        sorted_mts.sort();

        for mt in &sorted_mts {
            let reaction = &temp_reactions[mt];
            let has_photon_products = reaction
                .products
                .iter()
                .any(|p| p.is_particle_type(&ParticleType::Photon));
            if !has_photon_products {
                continue;
            }
            let xs_vec: Vec<f64> = energy
                .iter()
                .map(|&e| reaction.cross_section_at(e).unwrap_or(0.0))
                .collect();
            if xs_vec.iter().any(|&x| x > 0.0) {
                photon_rxn_mt_numbers.push(*mt);
                photon_rxn_xs_perrx.push(xs_vec);
                photon_rxn_reactions.push(Arc::clone(reaction));
            }
        }
    }

    // absorption_mt_xs - compute from reactions (not in Arrow). Per-MT
    // temporaries then flatten row-major at the end.
    let mut absorption_mt_numbers: Vec<i32> = Vec::new();
    let mut absorption_mt_xs_perrx: Vec<Vec<f64>> = Vec::new();
    {
        let mut covered: HashSet<i32> = HashSet::new();
        for mt in &scatter_mt_numbers_out {
            covered.insert(*mt);
        }
        for mt in &fission_mt_numbers_out {
            covered.insert(*mt);
        }
        for mt in &photon_rxn_mt_numbers {
            covered.insert(*mt);
        }
        if !xs_ngamma.is_empty() {
            covered.insert(102);
        }
        covered.extend(&[1, 4, 101, 1001]);

        let mut sorted_mts: Vec<i32> = temp_reactions.keys().copied().collect();
        sorted_mts.sort();

        for mt in &sorted_mts {
            if covered.contains(mt) {
                continue;
            }
            let reaction = &temp_reactions[mt];
            let xs_vec: Vec<f64> = energy
                .iter()
                .map(|&e| reaction.cross_section_at(e).unwrap_or(0.0))
                .collect();
            if xs_vec.iter().any(|&x| x > 0.0) {
                absorption_mt_numbers.push(*mt);
                absorption_mt_xs_perrx.push(xs_vec);
            }
        }
    }

    // Delayed-photon scaling f(E) = (prompt + delayed) / prompt, one value per
    // point of the nuclide's own energy grid (issue #369). Applied to FISSION
    // photon production only, which is what OpenMC's
    // `settings::delayed_photon_scaling` does (`physics.cpp`): a fission also
    // releases photons from the fission products' decay, and the prompt photon
    // production in the evaluation does not include them.
    //
    // Empty when the evaluation has no `fission_energy_release` data (Fe56) or the
    // published file predates the columns, in which case every consumer takes its
    // `f = 1.0` branch and nothing changes.
    let delayed_photon_scaling: F64Buffer = match photon_release {
        Some(release) => energy.iter().map(|&e| release.scaling(e)).collect(),
        _ => F64Buffer::default(),
    };

    // If photon_prod from Arrow is all zeros but we have photon-producing reactions,
    // recompute it (converter may not have computed it).
    let photon_prod = if !photon_rxn_xs_perrx.is_empty()
        && (photon_prod.is_empty() || photon_prod.iter().all(|&x| x == 0.0))
    {
        use crate::nuclide::is_fission_mt;
        use crate::particle_type::ParticleType;
        let n = energy.len();
        let mut pp = vec![0.0f64; n];
        for (j, mt) in photon_rxn_mt_numbers.iter().enumerate() {
            let xs_vec = &photon_rxn_xs_perrx[j];
            let reaction = &photon_rxn_reactions[j];
            for (i, &e) in energy.iter().enumerate() {
                let rxn_xs = xs_vec[i];
                if rxn_xs <= 0.0 {
                    continue;
                }
                let f = if is_fission_mt(*mt) && !delayed_photon_scaling.is_empty() {
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
                        pp[i] += f * rxn_xs * y;
                    }
                }
            }
        }
        pp.into()
    } else {
        photon_prod
    };

    // Flatten the Rust-computed per-MT XS vectors to row-major for storage.
    let n_e = energy.len();
    let photon_rxn_xs: F64Buffer =
        crate::nuclide::flatten_row_major(&photon_rxn_xs_perrx, n_e).into();
    let absorption_mt_xs: F64Buffer =
        crate::nuclide::flatten_row_major(&absorption_mt_xs_perrx, n_e).into();

    Ok(FastXSGrid {
        log_grid_index,
        log_e_min,
        inv_log_delta,
        xs,
        energy,
        scatter_mt_numbers: scatter_mt_numbers_out,
        scatter_mt_xs,
        scatter_mt_reactions,
        elastic_idx,
        inelastic_walk_order,
        reaction_absorption,
        fission_mt_numbers: fission_mt_numbers_out,
        fission_mt_xs,
        fission_mt_reactions,
        has_partial_fission,
        xs_ngamma,
        photon_prod,
        photon_rxn_mt_numbers,
        photon_rxn_xs,
        photon_rxn_reactions,
        absorption_mt_numbers,
        absorption_mt_xs,
        delayed_photon_scaling,
    })
}

// =============================================================================
// Discriminant columns
// =============================================================================

/// Error for a value in a discriminant column that this reader does not know.
///
/// These columns are declared as plain `utf8` in the schema, so the schema
/// itself cannot police their contents: the writer and the reader agree only by
/// both spelling the same literals. That makes them the one part of the format
/// where a mismatch is invisible to `test_schema_manifest.py` and to every
/// field-name check, which is how issue #379 (the chain spelling MT 18
/// "fission" against a reader that only knew "(n,fission)") went unnoticed.
///
/// Reaching an unknown value means the data was written by a converter this
/// build does not understand. Refusing it is the only safe answer: silently
/// returning `None` drops that row's physics and the simulation runs on, giving
/// a wrong answer rather than no answer.
fn unknown_discriminant(
    section: &str,
    column: &str,
    value: &str,
    row: usize,
    accepted: &[&str],
) -> Box<dyn Error> {
    format!(
        "{section}: unknown {column} {value:?} at row {row}. This build accepts: {}. \
         The data was written by a converter this reader does not understand; \
         re-download the library or update yamc.",
        accepted.join(", ")
    )
    .into()
}

// =============================================================================
// Distribution parsing
// =============================================================================

struct ParsedDistribution {
    applicability: Option<Tabulated1D>,
    angle_energy_dist: Option<AngleEnergyDistribution>,
}

fn parse_distribution(
    batch: &RecordBatch,
    row: usize,
) -> Result<ParsedDistribution, Box<dyn Error>> {
    let dist_type = get_str(batch, "type", row)?;

    let applicability = parse_applicability(batch, row);

    let angle_energy_dist = match dist_type.as_str() {
        "uncorrelated" => Some(parse_uncorrelated(batch, row)?),
        "correlated" => Some(parse_correlated(batch, row)?),
        "kalbach-mann" => Some(parse_kalbach_mann(batch, row)?),
        "nbody" => Some(parse_nbody(batch, row)?),
        other => {
            return Err(unknown_discriminant(
                "distributions.arrow",
                "type",
                other,
                row,
                &["uncorrelated", "correlated", "kalbach-mann", "nbody"],
            ))
        }
    };

    Ok(ParsedDistribution {
        applicability,
        angle_energy_dist,
    })
}

fn parse_applicability(batch: &RecordBatch, row: usize) -> Option<Tabulated1D> {
    let data = try_get_f64_list(batch, "applicability_data", row);
    let shape = try_get_i32_list(batch, "applicability_shape", row);
    if data.is_empty() || shape.is_empty() {
        return None;
    }

    let n = if shape.len() >= 2 {
        shape[1] as usize
    } else {
        data.len() / 2
    };
    if data.len() < 2 * n {
        return None;
    }

    let x = data[..n].to_vec();
    let y = data[n..2 * n].to_vec();
    let breakpoints = try_get_i32_list(batch, "applicability_breakpoints", row);
    let interpolation = try_get_i32_list(batch, "applicability_interpolation", row);

    Some(Tabulated1D::Tabulated1D {
        x,
        y,
        breakpoints,
        interpolation,
    })
}

fn parse_uncorrelated(
    batch: &RecordBatch,
    row: usize,
) -> Result<AngleEnergyDistribution, Box<dyn Error>> {
    // Parse angle distribution
    let angle_energies = try_get_f64_list(batch, "angle_energies", row);
    let angle_mu_data = try_get_f64_list(batch, "angle_mu_data", row);
    let angle_mu_offsets = try_get_i32_list(batch, "angle_mu_offsets", row);
    let angle_mu_interp = try_get_i32_list(batch, "angle_mu_interpolation", row);

    let angle = if !angle_energies.is_empty() {
        // mu_data layout: C-order ravel of shape (3, total_pts)
        // = [all_mu_values | all_pdf_values | all_cdf_values]
        // offsets index into each row (total_pts long), not the full array.
        let total_pts = angle_mu_data.len() / 3;
        let all_mu = &angle_mu_data[..total_pts];
        let all_pdf = &angle_mu_data[total_pts..2 * total_pts];
        let all_cdf = &angle_mu_data[2 * total_pts..3 * total_pts];

        let mut tables = Vec::with_capacity(angle_energies.len());
        for (idx, _) in angle_energies.iter().enumerate() {
            let start = if idx < angle_mu_offsets.len() {
                angle_mu_offsets[idx] as usize
            } else {
                0
            };
            let end = if idx + 1 < angle_mu_offsets.len() {
                angle_mu_offsets[idx + 1] as usize
            } else {
                total_pts
            };

            let x = all_mu[start..end].to_vec();
            let p = all_pdf[start..end].to_vec();
            let c = all_cdf[start..end].to_vec();

            let interp = if idx < angle_mu_interp.len() {
                match angle_mu_interp[idx] {
                    1 => TabulatedInterp::Histogram,
                    _ => TabulatedInterp::LinLin,
                }
            } else {
                TabulatedInterp::LinLin
            };

            let mut tab = Tabulated { x, p, c, interp };
            tab.normalize();
            tables.push(tab);
        }
        AngleDistribution {
            energy: angle_energies,
            mu: tables,
        }
    } else {
        // Isotropic (no angle data)
        AngleDistribution {
            energy: Vec::new(),
            mu: Vec::new(),
        }
    };

    let energy = parse_energy_distribution(batch, row)?;

    Ok(AngleEnergyDistribution::UncorrelatedAngleEnergy { angle, energy })
}

/// ENDF interpolation scheme 1: histogram (constant between points).
const ENDF_INTERP_HISTOGRAM: i32 = 1;

/// Whether a continuous-tabular distribution interpolates its incident-energy
/// axis as a histogram.
///
/// `energy_dist_interpolation` holds the region breakpoints and then the
/// interpolation codes, in one column: a continuous energy distribution has no
/// separate breakpoint column, so the writer concatenates the two ENDF fields
/// (`yamc-convert/src/distributions.rs`; the retired Python converter did the
/// same, and published data carries both layouts). Splitting at the halfway
/// point recovers both, since ENDF gives every region one breakpoint and one
/// code.
///
/// This used to ask whether EVERY value in the column was 1, which conflated the
/// two halves. A breakpoint is the index of its region's last point, so the
/// final one is the length of the incident grid, and the test therefore needed a
/// one-point grid to pass: it never fired on real data, and every distribution
/// declaring a histogram incident-energy region was silently sampled lin-lin on
/// both the CPU and GPU paths (issue #499).
///
/// The rule is OpenMC's, in `ContinuousTabular::sample`: histogram only for a
/// single region whose code says histogram. A multi-region distribution samples
/// lin-lin whatever its first code is.
fn continuous_histogram_interp(interp_flat: &[i32], row: usize) -> Result<bool, Box<dyn Error>> {
    if interp_flat.is_empty() {
        return Ok(false);
    }
    // Neither half can be told from the other at an odd length, and any guess
    // silently changes which secondary energies are sampled, so this refuses
    // rather than picking one.
    if !interp_flat.len().is_multiple_of(2) {
        return Err(format!(
            "distributions.arrow row {row}: energy_dist_interpolation has {} values. It holds one \
             breakpoint and one interpolation code per region, concatenated, so its length is \
             always even; an odd length means the file was written by something that did not \
             follow that layout and the two halves cannot be separated.",
            interp_flat.len()
        )
        .into());
    }

    let n_regions = interp_flat.len() / 2;
    let interpolation = &interp_flat[n_regions..];
    Ok(n_regions == 1 && interpolation[0] == ENDF_INTERP_HISTOGRAM)
}

fn parse_energy_distribution(
    batch: &RecordBatch,
    row: usize,
) -> Result<Option<EnergyDistribution>, Box<dyn Error>> {
    // A null here is legitimate and common: an uncorrelated distribution whose
    // secondary energy is not tabulated (elastic scattering, and 359 of the
    // 13061 distribution rows in the published endf-b8.1 data) carries angle
    // data only. An unrecognised non-null value is a different thing entirely
    // and is refused below.
    let dist_type = match try_get_str(batch, "energy_dist_type", row) {
        Some(t) => t,
        None => return Ok(None),
    };

    match dist_type.as_str() {
        "continuous" => {
            let energies = try_get_f64_list(batch, "energy_dist_energies", row);
            let data = try_get_f64_list(batch, "energy_dist_data", row);
            let offsets = try_get_i32_list(batch, "energy_dist_offsets", row);
            let out_interp = try_get_i32_list(batch, "energy_dist_out_interp", row);
            let n_discrete = try_get_i32_list(batch, "energy_dist_n_discrete", row);
            let interp_flat = try_get_i32_list(batch, "energy_dist_interpolation", row);

            let histogram_interp = continuous_histogram_interp(&interp_flat, row)?;

            // Parse tabulated distributions per energy
            // data layout: C-order ravel of shape (3, total_pts) = [all_x | all_p | all_c]
            let total_pts = data.len() / 3;
            let all_x = &data[..total_pts];
            let all_p = &data[total_pts..2 * total_pts];
            let all_c = &data[2 * total_pts..3 * total_pts];

            let mut tables = Vec::with_capacity(energies.len());
            for (idx, _) in energies.iter().enumerate() {
                let start = if idx < offsets.len() {
                    offsets[idx] as usize
                } else {
                    0
                };
                let end = if idx + 1 < offsets.len() {
                    offsets[idx + 1] as usize
                } else {
                    total_pts
                };

                let x = all_x[start..end].to_vec();
                let p = all_p[start..end].to_vec();
                let c = all_c[start..end].to_vec();

                let interp = if idx < out_interp.len() {
                    match out_interp[idx] {
                        1 => TabulatedInterp::Histogram,
                        _ => TabulatedInterp::LinLin,
                    }
                } else {
                    TabulatedInterp::LinLin
                };

                let nd = if idx < n_discrete.len() {
                    n_discrete[idx] as usize
                } else {
                    0
                };

                let mut tab_prob = TabulatedProbability::Tabulated {
                    x,
                    p,
                    c,
                    interp,
                    n_discrete: nd,
                };
                tab_prob.normalize();
                tables.push(tab_prob);
            }

            Ok(Some(EnergyDistribution::ContinuousTabular {
                energy: energies,
                energy_out: tables,
                histogram_interp,
            }))
        }
        "maxwell" => {
            let param_x = try_get_f64_list(batch, "energy_param_x", row);
            let param_y = try_get_f64_list(batch, "energy_param_y", row);
            let u = try_get_f64(batch, "energy_restriction_u", row).unwrap_or(0.0);
            Ok(Some(EnergyDistribution::Maxwell {
                theta: Tabulated1D::Tabulated1D {
                    x: param_x,
                    y: param_y,
                    breakpoints: Vec::new(),
                    interpolation: Vec::new(),
                },
                u,
            }))
        }
        "evaporation" => {
            let param_x = try_get_f64_list(batch, "energy_param_x", row);
            let param_y = try_get_f64_list(batch, "energy_param_y", row);
            let u = try_get_f64(batch, "energy_restriction_u", row).unwrap_or(0.0);
            Ok(Some(EnergyDistribution::Evaporation {
                theta: Tabulated1D::Tabulated1D {
                    x: param_x,
                    y: param_y,
                    breakpoints: Vec::new(),
                    interpolation: Vec::new(),
                },
                u,
            }))
        }
        "watt" => {
            let param_x = try_get_f64_list(batch, "energy_param_x", row);
            let param_y = try_get_f64_list(batch, "energy_param_y", row);
            let param2_x = try_get_f64_list(batch, "energy_param2_x", row);
            let param2_y = try_get_f64_list(batch, "energy_param2_y", row);
            let u = try_get_f64(batch, "energy_restriction_u", row).unwrap_or(0.0);
            Ok(Some(EnergyDistribution::Watt {
                a: Tabulated1D::Tabulated1D {
                    x: param_x,
                    y: param_y,
                    breakpoints: Vec::new(),
                    interpolation: Vec::new(),
                },
                b: Tabulated1D::Tabulated1D {
                    x: param2_x,
                    y: param2_y,
                    breakpoints: Vec::new(),
                    interpolation: Vec::new(),
                },
                u,
            }))
        }
        "level" => {
            let threshold = try_get_f64(batch, "energy_threshold", row).unwrap_or(0.0);
            let mass_ratio = try_get_f64(batch, "energy_mass_ratio", row).unwrap_or(0.0);
            Ok(Some(EnergyDistribution::LevelInelastic {
                threshold,
                mass_ratio,
            }))
        }
        "discrete_photon" => {
            let primary_flag = try_get_i32(batch, "energy_primary_flag", row).unwrap_or(0);
            let awr = try_get_f64(batch, "energy_atomic_weight_ratio", row).unwrap_or(0.0);
            let photon_energy = try_get_f64(batch, "energy_discrete_energy", row).unwrap_or(0.0);
            Ok(Some(EnergyDistribution::DiscretePhoton {
                primary_flag,
                energy: photon_energy,
                atomic_weight_ratio: awr,
            }))
        }
        other => Err(unknown_discriminant(
            "distributions.arrow",
            "energy_dist_type",
            other,
            row,
            &[
                "continuous",
                "maxwell",
                "evaporation",
                "watt",
                "level",
                "discrete_photon",
            ],
        )),
    }
}

fn parse_correlated(
    batch: &RecordBatch,
    row: usize,
) -> Result<AngleEnergyDistribution, Box<dyn Error>> {
    let energies = try_get_f64_list(batch, "corr_energies", row);
    let eout_data = try_get_f64_list(batch, "corr_eout_data", row);
    let eout_offsets = try_get_i32_list(batch, "corr_eout_offsets", row);
    let eout_interp = try_get_i32_list(batch, "corr_eout_interp", row);
    let eout_n_discrete = try_get_i32_list(batch, "corr_eout_n_discrete", row);
    let mu_data = try_get_f64_list(batch, "corr_mu_data", row);
    let mu_offsets = try_get_i32_list(batch, "corr_mu_offsets", row);
    let mu_interp = try_get_i32_list(batch, "corr_mu_interp", row);

    // eout_data layout: C-order ravel of shape (5, total_eout_pts)
    // = [all_x | all_p | all_c | all_mu_interp | all_mu_offsets]
    let total_eout_pts = eout_data.len() / 5;
    let eout_all_x = &eout_data[..total_eout_pts];
    let eout_all_p = &eout_data[total_eout_pts..2 * total_eout_pts];
    let eout_all_c = &eout_data[2 * total_eout_pts..3 * total_eout_pts];
    let eout_all_mu_interp = &eout_data[3 * total_eout_pts..4 * total_eout_pts];
    let eout_all_mu_offsets = &eout_data[4 * total_eout_pts..5 * total_eout_pts];

    // mu_data layout: C-order ravel of shape (3, total_mu_pts) = [all_mu | all_pdf | all_cdf]
    let total_mu_pts = mu_data.len() / 3;
    let mu_all_x = &mu_data[..total_mu_pts];
    let mu_all_p = &mu_data[total_mu_pts..2 * total_mu_pts];
    let mu_all_c = &mu_data[2 * total_mu_pts..3 * total_mu_pts];

    // `corr_mu_offsets` and `corr_mu_interp` carry one value per outgoing-energy
    // POINT, globally over the row: the writer fills them from rows 4 and 3 of
    // `corr_eout_data`, so they are the same integers that column holds as f64.
    // Read them from the int32 columns (no f64-to-int round trip, and no
    // per-incident-energy `Vec<usize>`), falling back to the float copy for a
    // file that predates them. Index by point, `eout_start + j`; indexing
    // `corr_mu_interp` by one of its own values, as this did, gave every mu
    // table in a row the interpolation of one of the row's first three points.
    // Issue #484.
    let mu_offsets_int = mu_offsets.len() == total_eout_pts;
    let mu_interp_int = mu_interp.len() == total_eout_pts;
    let mu_offset_at = |k: usize| -> usize {
        if mu_offsets_int {
            mu_offsets[k].max(0) as usize
        } else {
            eout_all_mu_offsets[k] as usize
        }
    };
    let mu_interp_at = |k: usize| -> i32 {
        if mu_interp_int {
            mu_interp[k]
        } else {
            eout_all_mu_interp[k] as i32
        }
    };

    let mut tables = Vec::with_capacity(energies.len());
    for (idx, _) in energies.iter().enumerate() {
        let eout_start = if idx < eout_offsets.len() {
            eout_offsets[idx] as usize
        } else {
            0
        };
        let eout_end = if idx + 1 < eout_offsets.len() {
            eout_offsets[idx + 1] as usize
        } else {
            total_eout_pts
        };

        let n_points = eout_end - eout_start;
        let eout_x = eout_all_x[eout_start..eout_end].to_vec();
        let eout_p = eout_all_p[eout_start..eout_end].to_vec();
        let eout_c = eout_all_c[eout_start..eout_end].to_vec();
        let interp_val = if idx < eout_interp.len() {
            eout_interp[idx]
        } else {
            2
        };
        let interpolation = match interp_val {
            1 => secondary_correlated::Interpolation::Histogram,
            _ => secondary_correlated::Interpolation::LinLin,
        };

        let n_disc = if idx < eout_n_discrete.len() {
            eout_n_discrete[idx] as usize
        } else {
            0
        };

        // Parse mu distributions for each outgoing energy
        let mut angle = Vec::with_capacity(n_points);
        for j in 0..n_points {
            let k = eout_start + j;
            let mu_start = mu_offset_at(k);
            // The offsets are GLOBAL into `mu_all_x`. The end of the LAST mu
            // sub-table in this corrtable is the FIRST mu offset of the NEXT
            // corrtable (point `eout_end`), not the end of the whole
            // concatenated mu array. An earlier `else { total_mu_pts }` here
            // made every corrtable's last angular distribution swallow all
            // remaining mu points (e.g. U238 MT5: a 181k-point non-monotonic
            // table), corrupting the angular sampling for that outgoing energy
            // on both the CPU and GPU paths.
            let mu_end = if j + 1 < n_points {
                mu_offset_at(k + 1)
            } else if eout_end < total_eout_pts {
                mu_offset_at(eout_end)
            } else {
                total_mu_pts
            };

            if mu_start < total_mu_pts && mu_end <= total_mu_pts && mu_end > mu_start {
                let mx = mu_all_x[mu_start..mu_end].to_vec();
                let mp = mu_all_p[mu_start..mu_end].to_vec();
                let mc = mu_all_c[mu_start..mu_end].to_vec();

                // This point's own code: 1 histogram, 2 lin-lin, 0 a Discrete
                // mu table, which is sampled from its own points and so takes
                // the lin-lin arm like any unrecognized code.
                let mi = match mu_interp_at(k) {
                    1 => secondary_correlated::Interpolation::Histogram,
                    _ => secondary_correlated::Interpolation::LinLin,
                };

                angle.push(secondary_correlated::Tabular {
                    x: mx,
                    p: mp,
                    c: mc,
                    interpolation: mi,
                    n_discrete: 0,
                });
            } else {
                angle.push(secondary_correlated::Tabular {
                    x: vec![-1.0, 1.0],
                    p: vec![0.5, 0.5],
                    c: vec![0.0, 1.0],
                    interpolation: secondary_correlated::Interpolation::LinLin,
                    n_discrete: 0,
                });
            }
        }

        let mut table = secondary_correlated::CorrTable {
            interpolation,
            n_discrete: n_disc,
            e_out: eout_x,
            p: eout_p,
            c: eout_c,
            angle,
        };
        table.normalize();
        tables.push(table);
    }

    Ok(AngleEnergyDistribution::CorrelatedAngleEnergy {
        correlated: secondary_correlated::CorrelatedAngleEnergy {
            energy: energies,
            distributions: tables,
        },
    })
}

fn parse_kalbach_mann(
    batch: &RecordBatch,
    row: usize,
) -> Result<AngleEnergyDistribution, Box<dyn Error>> {
    let energies = try_get_f64_list(batch, "km_energies", row);
    let data = try_get_f64_list(batch, "km_data", row);
    let offsets = try_get_i32_list(batch, "km_offsets", row);
    let interp = try_get_i32_list(batch, "km_interp", row);
    let n_discrete = try_get_i32_list(batch, "km_n_discrete", row);

    // km_data layout: C-order ravel of shape (5, total_pts)
    // = [all_eout | all_pdf | all_cdf | all_r | all_a]
    // offsets index into each row (total_pts long), not the full array.
    let total_pts = data.len() / 5;
    let all_eout = &data[..total_pts];
    let all_pdf = &data[total_pts..2 * total_pts];
    let all_cdf = &data[2 * total_pts..3 * total_pts];
    let all_r = &data[3 * total_pts..4 * total_pts];
    let all_a = &data[4 * total_pts..5 * total_pts];

    let mut tables = Vec::with_capacity(energies.len());
    for (idx, _) in energies.iter().enumerate() {
        let start = if idx < offsets.len() {
            offsets[idx] as usize
        } else {
            0
        };
        let end = if idx + 1 < offsets.len() {
            offsets[idx + 1] as usize
        } else {
            total_pts
        };

        let eout_x = all_eout[start..end].to_vec();
        let eout_p = all_pdf[start..end].to_vec();
        let eout_c = all_cdf[start..end].to_vec();
        let km_r = all_r[start..end].to_vec();
        let km_a = all_a[start..end].to_vec();

        let entry_interp = if idx < interp.len() {
            match interp[idx] {
                1 => secondary_kalbach::Interpolation::Histogram,
                _ => secondary_kalbach::Interpolation::LinLin,
            }
        } else {
            secondary_kalbach::Interpolation::LinLin
        };

        let n_disc = if idx < n_discrete.len() {
            n_discrete[idx] as usize
        } else {
            0
        };

        let mut table = secondary_kalbach::KMTable {
            interpolation: entry_interp,
            n_discrete: n_disc,
            e_out: eout_x,
            p: eout_p,
            c: eout_c,
            r: km_r,
            a: km_a,
        };
        table.normalize();
        tables.push(table);
    }

    Ok(AngleEnergyDistribution::KalbachMann {
        kalbach: secondary_kalbach::KalbachMann {
            energy: energies,
            distributions: tables,
        },
    })
}

fn parse_nbody(batch: &RecordBatch, row: usize) -> Result<AngleEnergyDistribution, Box<dyn Error>> {
    let n = try_get_i32(batch, "nbody_n", row).unwrap_or(0);
    let total_mass = try_get_f64(batch, "nbody_total_mass", row).unwrap_or(0.0);
    let awr = try_get_f64(batch, "nbody_atomic_weight_ratio", row).unwrap_or(0.0);
    let q_value = try_get_f64(batch, "nbody_q_value", row).unwrap_or(0.0);

    Ok(AngleEnergyDistribution::NBodyPhaseSpace {
        n_bodies: n,
        total_mass,
        awr,
        q_value,
    })
}

// =============================================================================
// Product parsing
// =============================================================================

fn parse_product(
    batch: &RecordBatch,
    row: usize,
    mt: i32,
    prod_idx: i32,
    n_dist: i32,
    distributions_map: &HashMap<(i32, i32, i32), ParsedDistribution>,
) -> Result<ReactionProduct, Box<dyn Error>> {
    let particle_str = get_str(batch, "particle", row)?;
    let emission_mode = get_str(batch, "emission_mode", row)?;
    let decay_rate = get_f64(batch, "decay_rate", row)?;
    let yield_type = get_str(batch, "yield_type", row)?;
    let yield_data = get_f64_list(batch, "yield_data", row)?;
    let yield_shape = get_i32_list(batch, "yield_shape", row)?;
    let yield_breakpoints = get_i32_list(batch, "yield_breakpoints", row)?;
    let yield_interpolation = get_i32_list(batch, "yield_interpolation", row)?;

    let particle = match particle_str.as_str() {
        "neutron" => ParticleType::Neutron,
        "photon" => ParticleType::Photon,
        other => {
            return Err(unknown_discriminant(
                "products.arrow",
                "particle",
                other,
                row,
                &["neutron", "photon"],
            ))
        }
    };

    let product_yield = parse_yield(
        &yield_type,
        &yield_data,
        &yield_shape,
        &yield_breakpoints,
        &yield_interpolation,
        row,
    )?;

    // Collect distributions and applicabilities
    let mut distributions: Vec<AngleEnergyDistribution> = Vec::new();
    let mut applicabilities: Vec<Tabulated1D> = Vec::new();
    for di in 0..n_dist {
        if let Some(dist_data) = distributions_map.get(&(mt, prod_idx, di)) {
            if let Some(ref ae_dist) = dist_data.angle_energy_dist {
                distributions.push(ae_dist.clone());
                if let Some(ref app) = dist_data.applicability {
                    applicabilities.push(app.clone());
                }
            }
        }
    }

    Ok(ReactionProduct {
        particle,
        emission_mode,
        decay_rate,
        applicability: applicabilities,
        distribution: distributions,
        product_yield,
    })
}

/// An empty `yield_data` gives `None`, which is how a product with no tabulated
/// yield is written. A non-empty yield with a `yield_type` this build does not
/// know is refused rather than dropped: dropping it would leave the product
/// with no multiplicity and quietly change the particle balance.
fn parse_yield(
    yield_type: &str,
    data: &[f64],
    shape: &[i32],
    breakpoints: &[i32],
    interpolation: &[i32],
    row: usize,
) -> Result<Option<Yield>, Box<dyn Error>> {
    if data.is_empty() {
        return Ok(None);
    }

    Ok(match yield_type {
        "Polynomial" => Some(Yield::Polynomial {
            coefficients: data.to_vec(),
        }),
        "Tabulated1D" => {
            let n = if shape.len() >= 2 {
                shape[1] as usize
            } else {
                data.len() / 2
            };
            if data.len() < 2 * n {
                return Ok(None);
            }
            let x = data[..n].to_vec();
            let y = data[n..2 * n].to_vec();
            Some(Yield::Tabulated1D {
                x,
                y,
                breakpoints: if breakpoints.is_empty() {
                    None
                } else {
                    Some(breakpoints.iter().map(|&v| v as usize).collect())
                },
                interpolation: if interpolation.is_empty() {
                    None
                } else {
                    Some(interpolation.iter().map(|&v| v as u32).collect())
                },
            })
        }
        other => {
            return Err(unknown_discriminant(
                "products.arrow",
                "yield_type",
                other,
                row,
                &["Polynomial", "Tabulated1D"],
            ))
        }
    })
}

// =============================================================================
// URR parsing
// =============================================================================

fn parse_urr(batch: &RecordBatch, row: usize) -> Result<UrrData, Box<dyn Error>> {
    let energy = get_f64_list(batch, "energy", row)?;
    let table_data = get_f64_list(batch, "table_data", row)?;
    let table_shape = get_i32_list(batch, "table_shape", row)?;
    let interp_val = get_i32(batch, "interpolation", row)?;
    let inelastic = get_i32(batch, "inelastic", row)?;
    let absorption = get_i32(batch, "absorption", row)?;
    let multiply_smooth = get_bool(batch, "multiply_smooth", row)?;

    let interp = match interp_val {
        5 => UrrInterpolation::LogLog,
        _ => UrrInterpolation::LinLin,
    };

    // table_shape = [n_energy, n_params, n_cdf]
    // n_params is 6: cumulative, total, elastic, fission, n_gamma, heating
    let n_energy = if !table_shape.is_empty() {
        table_shape[0] as usize
    } else {
        energy.len()
    };
    let n_params = if table_shape.len() >= 2 {
        table_shape[1] as usize
    } else {
        6
    };
    let n_cdf = if table_shape.len() >= 3 {
        table_shape[2] as usize
    } else if n_energy > 0 && n_params > 0 {
        table_data.len() / (n_energy * n_params)
    } else {
        0
    };

    let mut cdf_values = Vec::with_capacity(n_energy);
    let mut xs_values = Vec::with_capacity(n_energy);

    for ie in 0..n_energy {
        let mut cdfs = Vec::with_capacity(n_cdf);
        let mut xs_sets = Vec::with_capacity(n_cdf);

        for ic in 0..n_cdf {
            // C-order: [ie][ip][ic] → ie * n_params * n_cdf + ip * n_cdf + ic
            let base = ie * n_params * n_cdf;

            let cumul = table_data.get(base + ic).copied().unwrap_or(0.0);
            cdfs.push(cumul);

            let total = table_data.get(base + n_cdf + ic).copied().unwrap_or(0.0);
            let elastic = table_data
                .get(base + 2 * n_cdf + ic)
                .copied()
                .unwrap_or(0.0);
            let fission = table_data
                .get(base + 3 * n_cdf + ic)
                .copied()
                .unwrap_or(0.0);
            let n_gamma = table_data
                .get(base + 4 * n_cdf + ic)
                .copied()
                .unwrap_or(0.0);
            let heating = table_data
                .get(base + 5 * n_cdf + ic)
                .copied()
                .unwrap_or(0.0);

            xs_sets.push(UrrXsSet {
                total,
                elastic,
                fission,
                n_gamma,
                heating,
            });
        }

        cdf_values.push(cdfs);
        xs_values.push(xs_sets);
    }

    Ok(UrrData {
        interp,
        inelastic_flag: inelastic,
        absorption_flag: absorption,
        multiply_smooth,
        energy,
        cdf_values,
        xs_values,
    })
}

// =============================================================================
// Fission nu parsing
// =============================================================================

fn parse_fission_nu(batch: &RecordBatch) -> Result<FissionNuData, Box<dyn Error>> {
    let yield_type = get_str(batch, "yield_type", 0)?;
    let yield_data = get_f64_list(batch, "yield_data", 0)?;
    let yield_shape = get_i32_list(batch, "yield_shape", 0)?;

    match yield_type.as_str() {
        "Tabulated1D" => {
            let n = if yield_shape.len() >= 2 {
                yield_shape[1] as usize
            } else {
                yield_data.len() / 2
            };
            let energy = yield_data[..n].to_vec();
            let nu = yield_data[n..2 * n].to_vec();
            Ok(FissionNuData { energy, nu })
        }
        "Polynomial" => {
            let energy = vec![1e-5, 0.0253, 1.0, 1e6, 2e7];
            let nu: Vec<f64> = energy
                .iter()
                .map(|&e: &f64| {
                    yield_data
                        .iter()
                        .enumerate()
                        .map(|(i, &c)| c * e.powi(i as i32))
                        .sum()
                })
                .collect();
            Ok(FissionNuData { energy, nu })
        }
        _ => Err(format!("Unknown nu yield type: {yield_type}").into()),
    }
}

/// Extract FissionNuData from a reaction product Yield.
/// Used as fallback when total_nu.arrow is missing.
fn fission_nu_from_yield(y: &crate::reaction_product::Yield) -> FissionNuData {
    use crate::reaction_product::Yield;
    match y {
        Yield::Tabulated1D { x, y, .. } | Yield::Tabulated { x, y } => FissionNuData {
            energy: x.clone(),
            nu: y.clone(),
        },
        Yield::Polynomial { coefficients } => {
            let energy = vec![1e-5, 0.0253, 1.0, 1e3, 1e6, 5e6, 1e7, 1.5e7, 2e7];
            let nu: Vec<f64> = energy
                .iter()
                .map(|&e: &f64| {
                    coefficients
                        .iter()
                        .enumerate()
                        .map(|(i, &c)| c * e.powi(i as i32))
                        .sum()
                })
                .collect();
            FissionNuData { energy, nu }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::fs;
    use std::path::PathBuf;

    /// The published shape: monotone, in range, and stopping short of the last
    /// grid point. 38,232 of the 41,382 published rows end short, so a validator
    /// that demanded `n_energy - 1` would reject 92% of the data.
    #[test]
    fn accepts_a_monotone_table_that_stops_short_of_the_last_point() {
        let table = [0, 0, 1, 3, 3, 7];
        let out = log_grid_index_u32(&table, 64, "294").expect("valid table");
        assert_eq!(out, vec![0u32, 0, 1, 3, 3, 7]);
    }

    /// An index equal to `n_energy - 1` is the tight upper bound: `i_high` is
    /// `entry + 1`, an exclusive bound, so it may reach `n_energy` exactly.
    #[test]
    fn accepts_the_last_grid_point() {
        let out = log_grid_index_u32(&[0, 63], 64, "294").expect("valid table");
        assert_eq!(out, vec![0u32, 63]);
    }

    #[test]
    fn rejects_a_negative_entry() {
        let err = log_grid_index_u32(&[0, -1, 4], 64, "294").expect_err("negative index");
        let msg = err.to_string();
        assert!(msg.contains("log_grid_index[1]"), "{msg}");
        assert!(msg.contains("294"), "{msg}");
        assert!(msg.contains("cannot be negative"), "{msg}");
    }

    /// `n_energy` itself is out of range: it would make `i_high` one past the
    /// end of the grid and panic in the lookup's slice.
    #[test]
    fn rejects_an_entry_at_the_grid_length() {
        let err = log_grid_index_u32(&[0, 1, 64], 64, "900").expect_err("index past the grid");
        let msg = err.to_string();
        assert!(msg.contains("log_grid_index[2]"), "{msg}");
        assert!(msg.contains("64-point"), "{msg}");
        assert!(msg.contains("900"), "{msg}");
    }

    /// The quiet one: in range, out of order. No panic, just a bracket that
    /// excludes the energy and a lookup between the wrong pair of points.
    #[test]
    fn rejects_a_decrease() {
        let err = log_grid_index_u32(&[0, 5, 4, 9], 64, "294").expect_err("non-monotonic table");
        let msg = err.to_string();
        assert!(msg.contains("decreases at 2"), "{msg}");
        assert!(msg.contains("5 then 4"), "{msg}");
    }

    /// `lookup` reads `bin` and `bin + 1` after clamping to `len() - 2`, which
    /// underflows on a 1-entry table.
    #[test]
    fn rejects_a_table_too_short_to_index() {
        let err = log_grid_index_u32(&[0], 64, "294").expect_err("single-entry table");
        assert!(err.to_string().contains("needs at least 2"), "{err}");
    }

    /// An empty grid means the accelerator is never indexed, so an empty table
    /// is not a problem to report.
    #[test]
    fn passes_an_empty_table_through_for_an_empty_grid() {
        assert!(log_grid_index_u32(&[], 0, "294")
            .expect("empty is fine")
            .is_empty());
        assert_eq!(
            log_grid_index_u32(&[7], 0, "294").expect("no grid, no range to check"),
            vec![7u32]
        );
    }

    /// Same shape as the helper in `url_cache.rs`: a directory of touched
    /// section files, since `narrow_to_present_sections` decides on presence
    /// alone and never opens one.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let p = std::env::temp_dir().join(format!(
                "yamc-narrow-sections-{}-{}-{:?}",
                tag,
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = fs::remove_dir_all(&p);
            fs::create_dir_all(&p).expect("create temp dir");
            TempDir(p)
        }
        fn touch(&self, name: &str) {
            fs::File::create(self.0.join(name)).expect("touch");
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// What `convert_neutron_xs` writes: cross sections and nothing else.
    fn xs_only_dir(tag: &str) -> TempDir {
        let dir = TempDir::new(tag);
        dir.touch("nuclide.arrow");
        dir.touch("reactions.arrow");
        dir.touch("version.json");
        dir
    }

    #[test]
    fn a_full_directory_keeps_the_scope_it_was_asked_for() {
        let dir = xs_only_dir("full");
        for section in TRANSPORT_SECTIONS {
            dir.touch(section);
        }
        let scope = narrow_to_present_sections(&dir.0, &LoadScope::full()).expect("full load");
        assert_eq!(scope.sections, SectionScope::Full);
    }

    /// Issue #506: this is the directory that could not be opened at all.
    #[test]
    fn a_cross_sections_only_directory_narrows_instead_of_failing() {
        let dir = xs_only_dir("xs-only");
        let scope = narrow_to_present_sections(&dir.0, &LoadScope::full()).expect("xs-only load");
        assert_eq!(scope.sections, SectionScope::XsOnly);
    }

    /// The narrowed scope is what gets recorded on the nuclide, so it must not
    /// quietly widen the MT or temperature filters on its way through.
    #[test]
    fn narrowing_preserves_the_other_axes() {
        let dir = xs_only_dir("axes");
        let temperatures: HashSet<String> = ["294".to_string()].into_iter().collect();
        let requested = LoadScope::full().with_temperatures(Some(temperatures.clone()));
        let scope = narrow_to_present_sections(&dir.0, &requested).expect("xs-only load");
        assert_eq!(scope.sections, SectionScope::XsOnly);
        assert_eq!(scope.temperatures, Some(temperatures));
        assert!(scope.mts.is_none());
    }

    /// Half a transport set is an interrupted conversion, not a deliberate
    /// cross-sections-only one, and must not be narrowed past.
    #[test]
    fn a_partial_transport_set_names_what_is_missing() {
        let dir = xs_only_dir("partial");
        dir.touch("products.arrow");
        let err = narrow_to_present_sections(&dir.0, &LoadScope::full())
            .expect_err("a half-written directory should not load")
            .to_string();
        assert!(err.contains("distributions.arrow"), "{err}");
        assert!(err.contains("fast_xs.arrow"), "{err}");
        assert!(!err.contains("products.arrow"), "{err}");
    }

    /// An activation load never wanted the transport sections, so their absence
    /// is not a decision to make.
    #[test]
    fn an_activation_scope_passes_through_untouched() {
        let dir = xs_only_dir("activation");
        let requested = LoadScope::activation([102].into_iter().collect());
        let scope = narrow_to_present_sections(&dir.0, &requested).expect("activation load");
        assert_eq!(scope.sections, SectionScope::XsOnly);
        assert_eq!(scope.mts, requested.mts);
    }

    // --- issue #499: histogram_interp over the concatenated column ---------

    /// The shape the bug needed and never got: a one-point incident grid is the
    /// only way `all(v == 1)` could have been true.
    #[test]
    fn a_single_point_histogram_region_is_histogram() {
        assert!(continuous_histogram_interp(&[1, 1], 0).expect("even length"));
    }

    /// The real data the old test got wrong. Mo92/Mo94/Mo96/Mo97/Mo98/Na23 carry
    /// a 21-point grid, fendl-3.2d C12/Ni61/Ni62/Ni64 an 11-point one, and
    /// jeff-4.0 N15 a 9-point one, each one region declaring histogram.
    #[test]
    fn a_multi_point_histogram_region_is_histogram() {
        for breakpoint in [9, 11, 12, 17, 21] {
            assert!(
                continuous_histogram_interp(&[breakpoint, 1], 0).expect("even length"),
                "breakpoint {breakpoint} should still read as histogram"
            );
        }
    }

    /// Fe56 MT 5: one region over a 59-point grid, unit-base lin-lin. The
    /// overwhelming majority of published rows look like this, and they must
    /// keep sampling lin-lin.
    #[test]
    fn a_lin_lin_region_is_not_histogram() {
        for code in [2, 12, 22] {
            assert!(
                !continuous_histogram_interp(&[59, code], 0).expect("even length"),
                "code {code} is not histogram"
            );
        }
    }

    /// OpenMC samples multi-region distributions lin-lin whatever the first
    /// code says: fendl-3.2d Cu63/Cu65 are `[2, 7, 9]` with `[1, 2, 1]`, and
    /// jeff-4.0 N15 has `[9, 10]` with `[1, 2]`.
    #[test]
    fn multiple_regions_are_never_histogram() {
        assert!(!continuous_histogram_interp(&[2, 7, 9, 1, 2, 1], 0).expect("even length"));
        assert!(!continuous_histogram_interp(&[9, 10, 1, 2], 0).expect("even length"));
    }

    /// A distribution carrying no interpolation column at all.
    #[test]
    fn an_absent_column_is_not_histogram() {
        assert!(!continuous_histogram_interp(&[], 0).expect("empty is fine"));
    }

    /// The halves cannot be separated at an odd length, and guessing would
    /// quietly change sampled energies.
    #[test]
    fn an_odd_length_column_is_refused() {
        let err = continuous_histogram_interp(&[21, 17, 1], 7)
            .expect_err("an odd length is undecodable")
            .to_string();
        assert!(err.contains("row 7"), "{err}");
        assert!(err.contains("3 values"), "{err}");
    }
}
