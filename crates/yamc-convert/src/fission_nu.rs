//! `total_nu.arrow` and `fission_photon.arrow`: the two fissile-only sections.
//!
//! Both are absent for a nuclide that does not fission, which the reader
//! treats as "this nuclide has neither" rather than as a missing file.

use std::error::Error;
use std::path::Path;

use endf::fission_energy::Component;
use endf::product::Yield;
use endf::{FissionEnergyRelease, IncidentNeutron};

use crate::fast_xs::FISSION_MTS;
use crate::sections::*;

/// Write `total_nu.arrow`, if the evaluation gives a total nu-bar.
///
/// An evaluation that gives prompt and delayed neutrons separately also gives
/// their sum in MF=1/MT=452, and the parser keeps it as a derived product
/// rather than dropping it. Without this file the reader falls back to taking
/// the fission product's own yield as the nu-bar, which is the PROMPT yield
/// and so about 1.6% low for U235.
///
/// Returns whether anything was written.
pub fn write_total_nu(data: &IncidentNeutron, dir: &Path) -> Result<bool, Box<dyn Error>> {
    // Searched across every fission MT, not just 18.
    //
    // The parser attaches the NU block to whichever reaction the ACE table
    // marks TY = 19, and for an evaluation that gives chance-by-chance
    // partials rather than the MT 18 total that is MT 19. Looking only at
    // MT 18 found nothing for U240 and every nuclide like it, so no
    // total_nu.arrow was written; the reader then falls back to treating the
    // fission product's own yield as the nu-bar, which is the PROMPT yield.
    // That is issue #364 again, one nuclide at a time and without a message.
    let Some(total) = FISSION_MTS.iter().find_map(|mt| {
        data.reactions
            .get(mt)
            .and_then(|rx| rx.derived_products.first())
    }) else {
        return Ok(false);
    };

    let (yield_type, yield_data, yield_shape, breakpoints, interpolation) = match &total.yield_ {
        Yield::Polynomial(p) => (
            "Polynomial",
            p.coefficients.clone(),
            vec![p.coefficients.len() as i32],
            Vec::new(),
            Vec::new(),
        ),
        Yield::Tabulated(t) => {
            let mut data = t.x.clone();
            data.extend_from_slice(&t.y);
            (
                "Tabulated1D",
                data,
                vec![2, t.x.len() as i32],
                t.breakpoints.clone(),
                t.interpolation.clone(),
            )
        }
    };

    write_section(
        &dir.join("total_nu.arrow"),
        "total_nu.arrow",
        vec![
            strings(std::slice::from_ref(&total.name)),
            strings(&[total.emission_mode.name().to_string()]),
            floats(&[total.decay_rate]),
            strings(&[yield_type.to_string()]),
            float_lists(&[yield_data]),
            int_lists(&[yield_shape]),
            int_lists(&[breakpoints]),
            int_lists(&[interpolation]),
        ],
    )?;
    Ok(true)
}

/// One term of the fission energy release, in the columns it is written to.
///
/// A term is a polynomial or a table, never both, so the unused half stays
/// empty and `kind` says which to read.
struct ReleaseColumns {
    kind: &'static str,
    coefficients: Vec<f64>,
    x: Vec<f64>,
    y: Vec<f64>,
    interpolation: Vec<i32>,
    breakpoints: Vec<i32>,
}

fn release_columns(c: &Component) -> ReleaseColumns {
    match c {
        Component::Polynomial(p) => ReleaseColumns {
            kind: "polynomial",
            coefficients: p.coefficients.clone(),
            x: Vec::new(),
            y: Vec::new(),
            interpolation: Vec::new(),
            breakpoints: Vec::new(),
        },
        Component::Tabulated(t) => ReleaseColumns {
            kind: "tabulated",
            coefficients: Vec::new(),
            x: t.x.clone(),
            y: t.y.clone(),
            interpolation: t.interpolation.clone(),
            breakpoints: t.breakpoints.clone(),
        },
    }
}

/// Write `fission_photon.arrow` (issue #369).
///
/// The prompt and delayed photon terms of the fission energy release, which
/// scale the fission photon yield. Both rows are written or neither: the
/// reader forms a ratio from the two and refuses half a section, since a
/// missing delayed term would silently restore the actinide photon deficit
/// the section exists to fix.
///
/// The two terms take different forms in the same evaluation. U235, U238 and
/// Pu239 all give a tabulated prompt term beside a polynomial delayed one, so
/// `kind` is per row rather than per file.
pub fn write_fission_photon(
    release: &FissionEnergyRelease,
    dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let terms = [
        ("prompt_photons", &release.prompt_photons),
        ("delayed_photons", &release.delayed_photons),
    ];

    let mut role = Vec::new();
    let mut kind = Vec::new();
    let mut coefficients = Vec::new();
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut interpolation = Vec::new();
    let mut breakpoints = Vec::new();

    for (name, component) in terms {
        let c = release_columns(component);
        if c.kind == "polynomial" && c.coefficients.is_empty() {
            return Err(format!(
                "the {name} term of the fission energy release is a polynomial \
                 with no coefficients, which the reader refuses; writing it \
                 would make the section unloadable"
            )
            .into());
        }
        role.push(name.to_string());
        kind.push(c.kind.to_string());
        coefficients.push(c.coefficients);
        x.push(c.x);
        y.push(c.y);
        interpolation.push(c.interpolation);
        breakpoints.push(c.breakpoints);
    }

    write_section(
        &dir.join("fission_photon.arrow"),
        "fission_photon.arrow",
        vec![
            strings(&role),
            strings(&kind),
            float_lists(&coefficients),
            float_lists(&x),
            float_lists(&y),
            int_lists(&interpolation),
            int_lists(&breakpoints),
        ],
    )
}
