//! MT 901, heating-local, which is not in the ACE file.
//!
//! NJOY computes one heating number, but two KERMAs are wanted: one where an
//! outgoing photon deposits its energy where it was made, and one where it
//! carries the energy away. HEATR is run twice to get both, and the ACE table
//! carries only the second, as MT 301. The local one has to be built afterwards
//! from the HEATR tapes, which is what this does.
//!
//! Without it a heating tally counts only the energy deposited by the neutron
//! itself, and the photon contribution is silently missing. That is the whole
//! reason an ENDF conversion is worth the NJOY cost over reading an ACE table
//! someone else made.
//!
//! For a fissile nuclide both KERMAs also need correcting, because NJOY takes
//! the fission heating as the fragment energy alone. The evaluation's own
//! fission energy release says how much goes to fragments, betas and photons,
//! so the fission term is replaced rather than adjusted.

use std::collections::BTreeMap;
use std::error::Error;
use std::path::Path;

use endf::function::Tabulated1D;
use endf::{FissionEnergyRelease, IncidentNeutron, Material};

/// Evaluate an MF=3 section's cross section at each energy, zero outside the
/// range the section covers.
///
/// Absent section means zero, which is the right answer for a channel the
/// evaluation does not have rather than an error: MT 318 is only there for a
/// fissile nuclide.
///
/// The range check is the part that matters. `Tabulated1D::eval` CLAMPS below
/// the first abscissa, returning y[0], which is right for a cross section
/// tabulated from the bottom of the grid and wrong for a threshold reaction
/// that starts partway up it. HEATR writes MT 318, the fission KERMA, only
/// over the fission range: for Pa233 that is 248 points starting at 105.8 eV
/// with a nonzero first value, while MT 301 spans all 14,325 points from
/// 1e-5 eV. Clamping subtracted that first value at all 14,077 points below
/// the threshold, where the fission cross section is identically zero and
/// there is no fission KERMA to correct.
///
/// Two nuclides in ENDF/B-VIII.1 are affected, not the 42 an earlier count
/// here claimed. That count came from the EVALUATION's MF=3 MT 18, whose
/// leading zeros are an artefact of resonance-range fission living in MF=2
/// parameters; after RECONR reconstructs them sigma_f is nonzero from 1e-5 eV
/// and MT 318 spans the whole grid, leaving nothing for the clamp to act on.
/// Of the 88 cached evaluations carrying MT 18, exactly Pa233 (105.79 eV) and
/// Th232 (3994.41 eV) have a fission threshold above the bottom of the grid.
///
/// Only Pa233 was visible: its MT 301 dips to 20.9 eV-barn below the
/// threshold, so a 2.2e-6 offset is a 1.2e-7 relative error. Th232's MT 318
/// starts exactly on the MT 18 threshold point where sigma_f is still zero, so
/// its y[0] is 1e-14 and the offset is unmeasurable.
fn file3_xs(material: &Material, mt: i32, energies: &[f64]) -> Vec<f64> {
    let Some(section) = material.mf3(mt) else {
        return vec![0.0; energies.len()];
    };
    let sigma = &section.sigma;
    let (Some(&first), Some(&last)) = (sigma.x.first(), sigma.x.last()) else {
        return vec![0.0; energies.len()];
    };
    // The bound is a RELATIVE one, not an exact comparison.
    //
    // The energies come from the ACE table and the section from the HEATR
    // tape, parsed from different text, so the same grid point can differ in
    // the last bit: Cl35's first energy is 1e-5 on both sides and compares as
    // strictly less. An exact test zeroed the whole first point of the local
    // KERMA, 1.9e10 eV-barn, turning a fix into a worse defect than the one it
    // replaced.
    //
    // 1e-9 separates the two cases cleanly. A rounding difference is an ulp;
    // Pa233's MT 318 starts seven decades above the bottom of the grid. Note
    // the direction of the mistake to avoid: endf-python's own snapping uses
    // 1e-5 RELATIVE, which is wide enough to swallow real grid points (issue
    // #24). This has to be tight enough not to.
    const SAME_POINT: f64 = 1e-9;
    let below = |e: f64| e < first - first.abs() * SAME_POINT;
    let above = |e: f64| e > last + last.abs() * SAME_POINT;
    energies
        .iter()
        .map(|&e| {
            if below(e) || above(e) {
                0.0
            } else {
                sigma.eval(e)
            }
        })
        .collect()
}

/// The fission energy release of an evaluation, or `None` where it has none.
///
/// Read from the ORIGINAL evaluation and with the fission neutron yield. Two
/// separate things, each of which silently loses the whole heating correction:
///
/// The material. MT 458 is an evaluated quantity that a HEATR PENDF tape does
/// not carry, so reading it from a tape finds nothing for every fissile
/// nuclide there is. Measured on U235 that put MT 301 3.7% and MT 901 6.7%
/// below the published values, with MT 18 itself agreeing, so nothing pointed
/// at the heating.
///
/// The yield. An evaluation that gives one coefficient per component takes its
/// prompt neutron term from the Sher-Beck formula, which needs nu-bar. U235
/// and Pu239 use the tabulated form and do not care, which is what makes this
/// worth stating: 25 of the 79 fissile evaluations in ENDF/B-VIII.1 do care,
/// and they are the minor actinides (Np237, the Pu and Am isotopes above 240,
/// every Cm) rather than anything a first test would reach for.
///
/// Neither failure is swallowed with `.ok()`. Doing that made "this evaluation
/// is fissile and its release could not be built" indistinguishable from "this
/// evaluation is not fissile".
pub fn fission_energy_release(
    evaluation: &Material,
) -> Result<Option<FissionEnergyRelease>, Box<dyn Error>> {
    if evaluation.mf1_mt458().is_none() {
        return Ok(None);
    }
    let nu = evaluation.mf1_mt452(452).map(|section| &section.nu);
    Ok(Some(
        FissionEnergyRelease::from_material(evaluation, nu).map_err(|e| {
            format!(
                "the evaluation has a fission energy release (MF=1 MT=458) but it \
             could not be read ({e}); it drives the heating correction and the \
             fission photon scaling, and both would be silently absent"
            )
        })?,
    ))
}

/// Add MT 901 to a nuclide, and correct MT 301, from the two HEATR tapes.
///
/// `heatr` and `heatr_local` are the tapes NJOY wrote, one per temperature in
/// the same order as [`IncidentNeutron::temperatures`].
pub fn add_heating_local(
    data: &mut IncidentNeutron,
    evaluation: &Material,
    heatr: &Path,
    heatr_local: &Path,
) -> Result<(), Box<dyn Error>> {
    let temperatures = data.temperatures();

    // The tapes hold one material per temperature, in the order NJOY wrote
    // them, which is the order the temperatures were asked for.
    let tapes = endf::get_materials(heatr)?;
    let local_tapes = endf::get_materials(heatr_local)?;
    if tapes.len() != temperatures.len() || local_tapes.len() != temperatures.len() {
        return Err(format!(
            "the HEATR tapes hold {} and {} materials for {} temperatures; \
             they are matched positionally, so a mismatch would silently pair \
             a KERMA with the wrong temperature",
            tapes.len(),
            local_tapes.len(),
            temperatures.len()
        )
        .into());
    }

    let release = fission_energy_release(evaluation)?;

    let mut local_xs: BTreeMap<String, Tabulated1D> = BTreeMap::new();
    let mut corrected_301: BTreeMap<String, Vec<f64>> = BTreeMap::new();

    for ((temperature, m), m_local) in temperatures.iter().zip(&tapes).zip(&local_tapes) {
        let Some(kerma) = data
            .reactions
            .get(&301)
            .and_then(|rx| rx.xs.get(temperature))
        else {
            continue;
        };
        let energies = kerma.x.clone();

        let fission = release.as_ref().and_then(|_| {
            data.reactions
                .get(&18)
                .and_then(|rx| rx.xs.get(temperature))
                .map(|xs| energies.iter().map(|&e| xs.eval(e)).collect::<Vec<f64>>())
        });

        // MT 301, photons carrying their energy away. NJOY's fission heating is
        // the fragment energy alone, so replace it with fragments plus betas.
        if let (Some(f), Some(sigma_f)) = (release.as_ref(), fission.as_ref()) {
            let mt318 = file3_xs(m, 318, &energies);
            let corrected: Vec<f64> = kerma
                .y
                .iter()
                .zip(&mt318)
                .zip(sigma_f)
                .zip(&energies)
                .map(|(((y, sub), sf), &e)| y - sub + (f.fragments.eval(e) + f.betas.eval(e)) * sf)
                .collect();
            corrected_301.insert(temperature.clone(), corrected);
        }

        // MT 901, photons depositing locally. Straight off the second tape,
        // plus the same fission correction with the photon terms kept.
        let mut y = file3_xs(m_local, 301, &energies);
        if let (Some(f), Some(sigma_f)) = (release.as_ref(), fission.as_ref()) {
            let mt318 = file3_xs(m_local, 318, &energies);
            for (i, value) in y.iter_mut().enumerate() {
                let e = energies[i];
                *value = *value - mt318[i]
                    + (f.fragments.eval(e)
                        + f.prompt_photons.eval(e)
                        + f.delayed_photons.eval(e)
                        + f.betas.eval(e))
                        * sigma_f[i];
            }
        }
        local_xs.insert(
            temperature.clone(),
            Tabulated1D {
                x: energies,
                y,
                ..Default::default()
            },
        );
    }

    if local_xs.is_empty() {
        return Err("no MT 301 to build MT 901 from; was HEATR run?".into());
    }

    for (temperature, y) in corrected_301 {
        if let Some(xs) = data
            .reactions
            .get_mut(&301)
            .and_then(|rx| rx.xs.get_mut(&temperature))
        {
            xs.y = y;
        }
    }

    // The frame flag is MT 301's. MT 901 is the same KERMA with the photons
    // accounted differently, so it inherits how MT 301 was recorded rather
    // than defaulting: the published files carry `center_of_mass` on both.
    let center_of_mass = data
        .reactions
        .get(&301)
        .map(|rx| rx.center_of_mass)
        .unwrap_or_default();
    let mut reaction = endf::Reaction {
        mt: 901,
        redundant: true,
        center_of_mass,
        ..Default::default()
    };
    reaction.xs = local_xs;
    data.reactions.insert(901, reaction);
    Ok(())
}
