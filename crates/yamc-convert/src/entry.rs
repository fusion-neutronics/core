//! Turning an evaluation into the cross-section data an activation
//! calculation reads.
//!
//! One reader, two ways in. An ENDF evaluation gives cross sections at 0 K with
//! the resonance region described by parameters rather than pointwise data, so
//! NJOY reconstructs and Doppler broadens it and writes ACE, which is then read
//! back. Someone who already has an ACE table skips that step and enters at the
//! same place.
//!
//! That is why this is not two implementations that could disagree: there is
//! exactly one path that produces cross sections, [`endf::IncidentNeutron::from_ace`],
//! and the ENDF entry point is the same path with a preprocessing step in front.
//!
//! What ACE costs, and why the entry points are named for the use rather than
//! the format: an ACE table is one temperature and carries neither MT 901
//! heating nor MF=1/MT=458, so `fission_photon.arrow` cannot be built from it.
//! For an activation calculation none of that matters. For transport it is a
//! silent downgrade, which is why the transport converter does not offer it.
//!
//! # Heating comes from the tapes, not the ACE table
//!
//! MT 901 is not in the ACE file at all, so the ENDF route keeps the two HEATR
//! tapes NJOY writes and builds it afterwards (see [`crate::heating`]). The ACE
//! route cannot: an ACE table has one KERMA and no way to recover the other.
//!
//! Measured against the published data: Li6 agrees to 9.9e-7 on MT 901 and
//! 5.9e-6 on MT 301, and U235, where the fission energy release replaces NJOY's
//! fragment-only fission heating, agrees to 2.6e-4 on both. That last figure is
//! also what MT 18 itself agrees to, so the residual is two independent NJOY
//! runs rather than anything this does with them.

use std::error::Error;
use std::path::{Path, PathBuf};

use endf::ace::MetastableScheme;
use endf::{IncidentNeutron, IncidentPhoton, Material};

use crate::{distributions, fast_xs, fission_nu, nuclide, products, reactions};

/// Where the evaluation comes from.
pub enum Source<'a> {
    /// An ENDF evaluation, reconstructed and broadened by NJOY first. This is
    /// the documented route: it is the only one that can reach an arbitrary
    /// temperature, and the only one a transport conversion could ever use.
    Endf {
        path: &'a Path,
        njoy_exec: &'a str,
        /// Temperatures in Kelvin. Empty means NJOY's own default.
        temperatures: Vec<f64>,
    },
    /// An ACE table that already exists. No NJOY needed, at the cost of being
    /// stuck with whatever temperature it was processed at.
    Ace { path: &'a Path },
}

/// What the conversion recorded about itself.
#[derive(Debug, Clone, Default)]
pub struct Provenance {
    pub library: String,
    /// The published release this output belongs to, which is what a consumer
    /// compares a cached copy against (issue #366). Identifies the DATA, not
    /// the code.
    pub data_version: String,
    pub created_utc: String,
}

/// What a source yields: the nuclide, and the fission energy release where
/// the source can supply one.
///
/// The release is `None` for an ACE table, which carries no MF=1/MT=458. That
/// is not a gap this can paper over, and it is why the transport converter
/// does not accept ACE: `fission_photon.arrow` would be missing for every
/// fissile nuclide.
struct Read {
    data: IncidentNeutron,
    release: Option<endf::FissionEnergyRelease>,
    /// The evaluation itself, for the sections that exist only on the tape.
    ///
    /// `None` for an ACE table. MF=33 covariance is not in ACE at all -- NJOY
    /// does not carry it through ACER -- so an ACE conversion cannot produce
    /// `covariance.arrow` however it is asked.
    material: Option<Material>,
}

/// Read a nuclide from either source.
fn read(source: &Source) -> Result<Read, Box<dyn Error>> {
    match source {
        Source::Ace { path } => {
            let text =
                std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
            let tables = endf::ace::tables_from_str(&text, None)?;
            let (first, rest) = tables
                .split_first()
                .ok_or_else(|| format!("{}: no ACE tables in the file", path.display()))?;
            // A file holding several temperatures is read as all of them.
            let mut data = IncidentNeutron::from_ace(first, MetastableScheme::Mcnp)?;
            for table in rest {
                data.add_temperature_from_ace(table, MetastableScheme::Mcnp)?;
            }
            Ok(Read {
                data,
                release: None,
                material: None,
            })
        }
        Source::Endf {
            path,
            njoy_exec,
            temperatures,
        } => {
            let material =
                Material::from_file(path).map_err(|e| format!("{}: {e}", path.display()))?;

            // NJOY writes into a directory of its own so a failed run leaves
            // nothing behind to be mistaken for output.
            //
            // The counter is what makes two conversions of the SAME evaluation
            // in one process safe. Keying on the process id and the file name
            // alone gave them one directory between them, and the first to
            // finish deleted it under the second: a library converted across
            // threads would fail, or worse, read half of someone else's ACE
            // file.
            static RUN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let work = std::env::temp_dir().join(format!(
                "yamc-convert-njoy-{}-{}-{}",
                std::process::id(),
                RUN.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("nuclide")
            ));
            let _ = std::fs::remove_dir_all(&work);
            std::fs::create_dir_all(&work)?;

            // Keep the HEATR tapes rather than letting them be cleaned up:
            // MT 901 is built from them afterwards and is not in the ACE file.
            // NJOY writes the second one at this path with "_local" appended.
            let heatr = work.join("heatr");
            // The 0 K grid comes off the RECONR output, which is the only
            // place it exists: every ACE table has been broadened to a
            // temperature. The published files carry it, and the reader keeps
            // it deliberately, so it is kept here too.
            let pendf = work.join("pendf");
            let options = endf::njoy::AceOptions {
                temperatures: temperatures.clone(),
                output_dir: work.clone(),
                njoy_exec: (*njoy_exec).to_string(),
                heatr: Some(Some(heatr.clone())),
                pendf: Some(Some(pendf.clone())),
                ..Default::default()
            };
            endf::njoy::make_ace(path, &material, &options).map_err(|e| {
                format!(
                    "NJOY failed on {}: {e}. The executable was {njoy_exec:?}; \
                     note that FENDL needs the IAEA-NDS build rather than the \
                     LANL one and the wrong choice is quiet",
                    path.display()
                )
            })?;

            let ace = work.join("ace");
            let text = std::fs::read_to_string(&ace)
                .map_err(|e| format!("{}: NJOY produced no ACE file ({e})", ace.display()))?;
            let tables = endf::ace::tables_from_str(&text, None)?;
            let (first, rest) = tables
                .split_first()
                .ok_or("NJOY produced an ACE file with no tables")?;
            // ACER writes one table per temperature, so all of them are read
            // rather than just the first. Taking only the first produced a
            // one-temperature nuclide from a six-temperature run, and the
            // remaining five were dropped without a word.
            let mut data = IncidentNeutron::from_ace(first, MetastableScheme::Mcnp)?;
            for table in rest {
                data.add_temperature_from_ace(table, MetastableScheme::Mcnp)?;
            }

            // The isomeric state, taken from the EVALUATION rather than left to
            // the ZAID the ACE carries.
            //
            // ACER records no isomeric state, so `metastable_zaid` encodes one
            // by adding 400 to the mass number, the way MCNP libraries do:
            // Am242 with LISO=1 is written 95642. Reading it back with
            // `MetastableScheme::Mcnp` then applies that scheme's Am242
            // exception, which says 95242 IS the metastable and 95642 the
            // ground state. Encoder and decoder disagree for exactly this
            // nuclide, so Am242 and Am242_m1 came out with each other's data,
            // under each other's names. A swap, not a rounding difference.
            //
            // MF=1/MT=451 knows the answer, so it is asked instead of inferred.
            if let Some(meta) = material.mf1_mt451() {
                let liso = meta.liso.max(0) as u32;
                if data.metastable != liso {
                    data.metastable = liso;
                    data.name_override = None;
                }
            }

            // MT 901 and the MT 301 correction. Not fatal if the tapes are
            // missing: a caller who turned HEATR off gets a conversion without
            // heating rather than no conversion, and the message says which.
            let local = work.join("heatr_local");
            if heatr.is_file() && local.is_file() {
                crate::heating::add_heating_local(&mut data, &material, &heatr, &local)?;
            } else {
                eprintln!(
                    "warning: no HEATR tapes at {} and {}, so MT 901 \
                     (heating-local) is absent from this conversion",
                    heatr.display(),
                    local.display()
                );
            }

            // The 0 K energy grid, under its own label. Nothing normally
            // interpolates on it, since a material's temperature is never 0 K,
            // but it is what the file format holds and dropping it would make
            // a regenerated file provide less than a downloaded one.
            if pendf.is_file() {
                match endf::get_materials(&pendf) {
                    Ok(materials) => {
                        if let Some(zero_k) = materials.first() {
                            // From MT 2, not MT 1. Both readers this format
                            // is shared with take the 0 K grid from elastic
                            // scattering, and RECONR does not always give the
                            // two the same range: FENDL's Cl35 has a negative
                            // File-3 elastic background at thermal, so MT 2's
                            // PENDF section is trimmed to start at 1.3954e-5 eV
                            // where MT 1 still starts at 1e-5, a ten-point
                            // difference in the grid this writes.
                            if let Some(section) = zero_k.mf3(2) {
                                data.energy
                                    .insert("0K".to_string(), section.sigma.x.clone());
                            }
                            // The 0 K elastic cross section too, and only that
                            // one: it is the unbroadened scattering an exact
                            // resonance treatment needs, and it is the single
                            // reaction the published files carry at 0 K.
                            if let (Some(section), Some(elastic)) =
                                (zero_k.mf3(2), data.reactions.get_mut(&2))
                            {
                                elastic.xs.insert("0K".to_string(), section.sigma.clone());
                            }
                        }
                    }
                    Err(e) => eprintln!(
                        "warning: could not read the PENDF tape at {} ({e}), so the \
                         0 K energy grid is absent from this conversion",
                        pendf.display()
                    ),
                }
            }

            let _ = std::fs::remove_dir_all(&work);

            // Read from the evaluation, for the same reason the heating
            // correction is: MT 458 is evaluated data and is not on any tape
            // NJOY writes.
            let release = crate::heating::fission_energy_release(&material)?;
            Ok(Read {
                data,
                release,
                material: Some(material),
            })
        }
    }
}

/// Write the cross-section data an activation calculation reads.
///
/// `nuclide.arrow`, `reactions.arrow` and `version.json`, which is exactly the
/// set `NEUTRON_XS_ONLY_SECTIONS` names. Not a partial transport conversion:
/// the products, secondary distributions and lookup accelerators a transport
/// run needs are a separate concern and their absence is what the scope means.
///
/// `covariance` adds `covariance.arrow`. Off by default because the matrices
/// are large and only an uncertainty calculation reads them, and ENDF-only
/// because ACER does not carry MF=33 through: asking for it from an ACE source
/// is an error rather than a silently empty section.
///
/// Returns the directory written.
pub fn convert_neutron_xs(
    source: &Source,
    output_dir: &Path,
    provenance: &Provenance,
    covariance: bool,
) -> Result<PathBuf, Box<dyn Error>> {
    let Read { data, material, .. } = read(source)?;
    let dir = output_dir.join(format!("{}.arrow", data.name()));
    std::fs::create_dir_all(&dir)?;

    nuclide::write_nuclide(&data, &dir)?;
    reactions::write_reactions(&data, &dir)?;
    // urr.arrow is transport-only, but it costs nothing to carry when the
    // evaluation has it and a later transport conversion would want it.
    nuclide::write_urr(&data, &dir)?;
    write_covariance_if_asked(covariance, material.as_ref(), &dir)?;

    write_version(&dir, provenance)?;
    Ok(dir)
}

/// Write `covariance.arrow` when it was asked for, and refuse when it cannot be.
///
/// An ACE source has no `Material`, and MF=33 is not in an ACE table, so a
/// caller who asked for covariance and would have got a directory without one
/// is told rather than left to discover the absence at load time -- where it is
/// indistinguishable from an evaluation that simply has no covariance.
fn write_covariance_if_asked(
    covariance: bool,
    material: Option<&Material>,
    dir: &Path,
) -> Result<(), Box<dyn Error>> {
    if !covariance {
        return Ok(());
    }
    let Some(material) = material else {
        return Err(
            "covariance must be converted from an ENDF evaluation: MF=33 is \
                    not carried through ACER, so an ACE table has none"
                .into(),
        );
    };
    crate::covariance::write_covariance(material, dir)?;
    Ok(())
}

/// Write `version.json`.
///
/// Last, and only once everything else is on disk: its presence is what a
/// resume takes as proof the conversion finished.
fn write_version(dir: &Path, provenance: &Provenance) -> Result<(), Box<dyn Error>> {
    let mut marker = serde_json::json!({
        "format_version": 1,
        "library": provenance.library,
        "data_version": provenance.data_version,
        "converter_version": concat!("yamc-convert ", env!("CARGO_PKG_VERSION")),
        "created_utc": provenance.created_utc,
    });

    // Where each MT's record batch sits in reactions.arrow, so an activation
    // reader can range-request just the channels its chain names rather than
    // pulling the full-grid transport MTs it never looks at (8.2x less on
    // Fe56). Absent for a photon element, which has no reactions table.
    // Best-effort: the index is an optimisation, and a reader that does not
    // find one falls back to fetching the whole file. Failing a conversion over
    // it would trade a working data set for a faster one.
    let reactions = dir.join("reactions.arrow");
    if reactions.exists() {
        match crate::reaction_ranges::index_reactions(&reactions) {
            Ok(ranges) => marker["reaction_ranges"] = ranges.to_json(),
            Err(e) => eprintln!("warning: no reaction_ranges for {}: {e}", dir.display()),
        }
    }

    let tmp = dir.join("version.json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(&marker)?)?;
    std::fs::rename(tmp, dir.join("version.json"))?;
    Ok(())
}

/// Write the full transport data set.
///
/// Every section `NEUTRON_SECTIONS` names: the three an activation run reads,
/// plus the products, their secondary distributions, the lookup accelerator,
/// and the two fissile-only sections.
///
/// ENDF only. An ACE table is one temperature, carries no MT 901 heating and
/// no MF=1/MT=458, so a transport conversion from one would be missing the
/// local KERMA and the fission photon scaling. Those absences are silent at
/// load time, which is why this refuses the source rather than warning.
///
/// Returns the directory written.
pub fn convert_neutron_transport(
    source: &Source,
    output_dir: &Path,
    provenance: &Provenance,
    covariance: bool,
) -> Result<PathBuf, Box<dyn Error>> {
    if matches!(source, Source::Ace { .. }) {
        return Err(
            "transport data must be converted from an ENDF evaluation: an ACE \
                    table holds one temperature, no MT 901 heating and no fission \
                    energy release, and every one of those is missing without a word \
                    once the file is written"
                .into(),
        );
    }

    let Read {
        data,
        release,
        material,
    } = read(source)?;
    let dir = output_dir.join(format!("{}.arrow", data.name()));
    std::fs::create_dir_all(&dir)?;

    nuclide::write_nuclide(&data, &dir)?;
    reactions::write_reactions(&data, &dir)?;
    nuclide::write_urr(&data, &dir)?;
    products::write_products(&data, &dir)?;
    distributions::write_distributions(&data, &dir)?;
    fast_xs::write_fast_xs(&data, &dir)?;
    fission_nu::write_total_nu(&data, &dir)?;
    if let Some(release) = &release {
        fission_nu::write_fission_photon(release, &dir)?;
    }
    write_covariance_if_asked(covariance, material.as_ref(), &dir)?;

    write_version(&dir, provenance)?;
    Ok(dir)
}

/// Where the auxiliary photon tabulations live.
///
/// None of this is in any evaluation: the photoatomic sublibrary carries no
/// Compton profiles, no bremsstrahlung cross sections and no density effect
/// correction, so a transport code takes them from separate published
/// tabulations. Optional because a conversion without them still produces a
/// usable `element.arrow`; it just has no `compton.arrow` or
/// `bremsstrahlung.arrow` beside it.
pub struct PhotonTabulations<'a> {
    pub compton_profiles: &'a Path,
    pub density_effect: &'a Path,
    pub bremsstrahlung: &'a Path,
}

/// Convert a photoatomic evaluation into the per-element photon sections.
///
/// `relaxation` is the atomic relaxation sublibrary for the same element,
/// which carries the binding energies, occupancies and the transition cascade.
/// Without it `subshells.arrow` is still written, with zero binding energies
/// and no transitions, which is a fluorescence-free atom rather than an error:
/// some libraries publish the two sublibraries separately and a caller may
/// legitimately have only one.
///
/// One ENDF file may hold several elements (FENDL bundles them), so every
/// material in the file is converted and all the directories are returned.
pub fn convert_photon(
    photoatomic: &Path,
    relaxation: Option<&Path>,
    tabulations: Option<&PhotonTabulations>,
    output_dir: &Path,
    provenance: &Provenance,
) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let materials =
        endf::get_materials(photoatomic).map_err(|e| format!("{}: {e}", photoatomic.display()))?;
    if materials.is_empty() {
        return Err(format!("{}: no evaluations in the file", photoatomic.display()).into());
    }

    // The relaxation file is matched to the photoatomic one by atomic number
    // rather than by position, since a bundled file need not list its elements
    // in the same order.
    let relaxation_materials = match relaxation {
        Some(path) => endf::get_materials(path).map_err(|e| format!("{}: {e}", path.display()))?,
        None => Vec::new(),
    };
    let relaxation_by_z: std::collections::HashMap<i64, &Material> = relaxation_materials
        .iter()
        .filter_map(|m| m.mf1_mt451().map(|meta| (meta.za / 1000, m)))
        .collect();

    let auxiliary = match tabulations {
        Some(t) => Some(
            endf::PhotonData::from_files(t.compton_profiles, t.density_effect, t.bremsstrahlung)
                .map_err(|e| format!("the auxiliary photon tabulations could not be read: {e}"))?,
        ),
        None => None,
    };

    let mut written = Vec::new();
    for material in &materials {
        let z = material.mf1_mt451().map(|meta| meta.za / 1000);
        let mut data =
            IncidentPhoton::from_endf(material, z.and_then(|z| relaxation_by_z.get(&z)).copied())?;
        if let Some(auxiliary) = &auxiliary {
            data.add_photon_data(auxiliary);
        }

        let dir = output_dir.join(format!("{}.arrow", data.name()));
        std::fs::create_dir_all(&dir)?;
        crate::photon::write_photon(&data, &dir)?;
        write_version(&dir, provenance)?;
        written.push(dir);
    }
    Ok(written)
}
