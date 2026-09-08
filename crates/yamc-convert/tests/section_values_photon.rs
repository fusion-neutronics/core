//! Every column of `element.arrow`, `subshells.arrow`, `compton.arrow` and
//! `bremsstrahlung.arrow`, against the `endf::IncidentPhoton` the writer was
//! handed.
//!
//! `crates/yamc-convert/src/photon.rs` carries no unit tests of its own, and
//! the two integration tests that run the photon route assert `is_file()` and
//! nothing else, one of them skipping entirely on a clean checkout. So until
//! this file no test compared a single value in those four sections against
//! the evaluation it came from.
//!
//! # Why a slot swap is the failure to look for
//!
//! `write_section` builds each batch from a positional `Vec<ArrayRef>` and
//! Arrow validates the data type and the row count, never the meaning.
//! `element.arrow` declares sixteen consecutive `list<double>` fields, so
//! any permutation among them is type-valid and writes without complaint.
//! `read_photon_interaction_from_arrow`
//! (`crates/yamc-element/src/photon_arrow.rs:53`) does read every one of those
//! columns back on every load, but it reads them BY NAME and compares them
//! against nothing, so a permutation loads as wrong physics rather than as an
//! error.
//!
//! # Hermetic
//!
//! The hydrogen photoatomic evaluation and its relaxation file are
//! `include_bytes!` from `crates/endf/fixtures/`. The three auxiliary
//! tabulations, which are the only way `compton.arrow` and
//! `bremsstrahlung.arrow` get written at all, are the git-tracked text files
//! that ship in the wheel under `packages/yamc-core/python/yamc/data`. No
//! NJOY, no nuclear-data cache, no network.
//!
//! # Constructed inputs, and where they begin
//!
//! The four tests in the first half of this file convert a vendored
//! evaluation. The eight below the "A constructed element" divider hand
//! `yamc_convert::photon::write_photon` an `IncidentPhoton` this file builds,
//! because hydrogen is Z = 1 with one subshell and one Compton shell and so
//! cannot discriminate the Z squared divide, the subshell row order, the two
//! arguments of `compton_subshell_map`, `safe_log`'s floor or the relaxation
//! cascade layout. Those eight are not evaluation parity and each says so in
//! its own doc comment.

mod section_values;
use section_values::*;

use std::path::PathBuf;
use std::sync::OnceLock;

use endf::function::Tabulated1D;
use endf::incident_photon::{
    compton_subshell_map, AtomicRelaxation, Bremsstrahlung, ComptonProfiles, PhotonReaction,
    Transitions, SUBSHELLS,
};
use endf::{IncidentPhoton, PhotonData};

// ---------------------------------------------------------------------------
// Running the real conversion.
// ---------------------------------------------------------------------------

/// The three auxiliary tabulations, parsed once for the whole binary.
///
/// `PhotonData::from_files` reads 2.4 MB of text and resamples a 30-column
/// bremsstrahlung table onto 200 electron energies for all 100 elements.
/// Measured at about 130 ms in a debug build, so this is memoised for the
/// parsed side rather than for the writer: `convert_photon` reads the files
/// itself on every call and cannot be handed a `PhotonData`.
fn tabulations() -> &'static PhotonData {
    static ONCE: OnceLock<PhotonData> = OnceLock::new();
    ONCE.get_or_init(photon_tabulations)
}

/// One conversion of the hydrogen pair, and the value the writer was handed.
struct Converted {
    /// Held so the scratch tree outlives the assertions and still removes
    /// itself when one of them panics.
    _scratch: tempfile::TempDir,
    /// The `H.arrow` directory `convert_photon` returned.
    dir: PathBuf,
    /// The same `IncidentPhoton` the writer saw, rebuilt from the same bytes
    /// through the same public calls `entry::convert_photon` makes.
    data: IncidentPhoton,
}

/// Convert the hydrogen photoatomic fixture through `entry::convert_photon`.
///
/// `relaxation` decides whether `atom-001_H_000.endf` is supplied, which is
/// what fills the subshell binding energy and occupancy. `tabulations`
/// decides whether `compton.arrow` and `bremsstrahlung.arrow` are written at
/// all: without them `compton_profiles` and `bremsstrahlung` are `None` and
/// both writers return early having written nothing.
fn convert(relaxation: bool, tabulations_on: bool) -> Converted {
    let scratch = scratch();
    let photoatomic = write_fixture(H_PHOTOAT, scratch.path(), "photoat-001_H_000.endf");
    let relax = write_fixture(H_ATOM, scratch.path(), "atom-001_H_000.endf");

    let d = photon_tabulation_dir();
    let (compton, density, brems) = (
        d.join("compton_profiles_biggs1975.txt"),
        d.join("density_effect_sternheimer1982.txt"),
        d.join("bremsstrahlung_seltzer_berger1986.txt"),
    );
    assert!(
        compton.is_file() && density.is_file() && brems.is_file(),
        "the auxiliary photon tabulations are missing from {}",
        d.display()
    );
    let paths = yamc_convert::entry::PhotonTabulations {
        compton_profiles: &compton,
        density_effect: &density,
        bremsstrahlung: &brems,
    };

    let out = scratch.path().join("out");
    let written = yamc_convert::entry::convert_photon(
        &photoatomic,
        relaxation.then_some(relax.as_path()),
        tabulations_on.then_some(&paths),
        &out,
        &yamc_convert::entry::Provenance::default(),
    )
    .expect("the photon route converts");
    assert_eq!(
        written.len(),
        1,
        "one element in the file, one directory out"
    );

    let photoatomic_material = endf_material(H_PHOTOAT);
    let relax_material = endf_material(H_ATOM);
    let mut data =
        IncidentPhoton::from_endf(&photoatomic_material, relaxation.then_some(&relax_material))
            .expect("the photoatomic evaluation reads");
    if tabulations_on {
        data.add_photon_data(tabulations());
    }

    Converted {
        _scratch: scratch,
        dir: written.into_iter().next().unwrap(),
        data,
    }
}

/// The union of every reaction's abscissa, restated from `photon.rs:48-58`.
///
/// A restatement, so on its own it checks wiring rather than arithmetic. The
/// properties that do not depend on the writer's implementation are asserted
/// beside it: the result is strictly increasing, and every reaction's own
/// abscissa is a subset of it.
fn union_grid_expected(data: &IncidentPhoton) -> Vec<f64> {
    let mut grid: Vec<f64> = data
        .reactions
        .values()
        .filter_map(|rx| rx.xs.as_ref())
        .flat_map(|xs| xs.x.iter().copied())
        .collect();
    grid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    grid.dedup();
    grid
}

/// One reaction's cross section evaluated over the union grid.
fn eval_on(data: &IncidentPhoton, mt: i32, grid: &[f64]) -> Vec<f64> {
    let f = data
        .get(mt)
        .and_then(|rx| rx.xs.as_ref())
        .unwrap_or_else(|| panic!("MT {mt} has no cross section"));
    grid.iter().map(|&e| f.eval(e)).collect()
}

// ---------------------------------------------------------------------------
// element.arrow
// ---------------------------------------------------------------------------

/// A failure means a photon cross section landed in the wrong slot or on the
/// wrong grid.
///
/// `element.arrow` declares sixteen consecutive `list<double>` fields, so
/// any permutation among them is type-valid and `RecordBatch::try_new`
/// accepts it. The grid itself is load-bearing and subtle: MT 501 tabulates
/// 13.6 eV twice for the K edge and `Vec::dedup` collapses the pair, so the
/// union of a 2021-point channel is 2020 points.
///
/// The five channel comparisons run the same `Tabulated1D::eval` the writer
/// ran, so they pin WIRING (which MT reached which slot) and not arithmetic.
///
/// The literal spot values below are two different things and each one is
/// marked. FIXTURE means the literal was found in `photoat-001_H_000.endf` by
/// grepping the decompressed text, so it is an independent read of the
/// evaluation. GOLDEN means it is not in that file in any form: it is a
/// capture of the same `eval` the writer ran, at a grid point that falls
/// between two tabulated ones, so it detects a change to `eval` and does not
/// independently verify it. Four of the twelve are GOLDEN. Every
/// interpolation region in this evaluation is scheme 2, lin-lin, so `eval` is
/// plain arithmetic with no libm call and an exact comparison is portable.
#[test]
fn element_cross_section_columns_are_the_parsed_reactions_on_the_union_grid() {
    let c = convert(true, false);
    assert_eq!(
        c.dir.file_name().unwrap(),
        "H.arrow",
        "the output directory is named for the element"
    );

    let batch = section(&c.dir, "element.arrow");
    assert_schema_is_declared(&batch, "element.arrow");
    assert_eq!(batch.num_rows(), 1, "element.arrow is one row per element");

    assert_eq!(str_at(&batch, "name", 0), c.data.name());
    assert_eq!(str_at(&batch, "name", 0), "H");
    assert_eq!(i32_at(&batch, "Z", 0), c.data.atomic_number as i32);
    assert_eq!(i32_at(&batch, "Z", 0), 1);

    // The grid, and the two properties of it that hold whatever the writer
    // does internally.
    let grid = union_grid_expected(&c.data);
    assert_eq!(grid.len(), 2020, "the union of the H channels");
    assert!(
        grid.windows(2).all(|w| w[1] > w[0]),
        "the union grid must be strictly increasing"
    );
    for (&mt, rx) in &c.data.reactions {
        let Some(f) = rx.xs.as_ref() else { continue };
        for &e in &f.x {
            assert!(
                grid.binary_search_by(|g| g.partial_cmp(&e).unwrap())
                    .is_ok(),
                "MT {mt} tabulates {e:e}, which is not on the union grid"
            );
        }
    }
    // Where the missing point went: MT 501 carries 2021 abscissae with 13.6
    // eV twice, once each side of the K edge.
    let total = &c.data.get(501).unwrap().xs.as_ref().unwrap().x;
    assert_eq!(total.len(), 2021);
    assert_eq!(total.iter().filter(|&&e| e == 13.6).count(), 2);

    // ln_energy IS the grid: the raw energy column was retired.
    let ln_energy = f64_list(&batch, "ln_energy", 0);
    let expected: Vec<f64> = grid.iter().map(|e| e.ln()).collect();
    assert_f64_slice_eq("ln_energy", &ln_energy, &expected);
    assert_eq!(ln_energy.len(), 2020);
    assert_eq!(grid[0], 1.0);
    assert_eq!(ln_energy[0], 0.0, "ln(1 eV) is exactly zero");
    assert_eq!(grid[137], 13.6);
    assert_eq!(grid[2019], 1.0e11);

    // Each populated channel, against the reaction it names.
    for (col, mt) in [
        ("coherent_xs", 502),
        ("incoherent_xs", 504),
        ("photoelectric_xs", 522),
        ("pair_production_nuclear_xs", 517),
        ("pair_production_electron_xs", 515),
    ] {
        let written = f64_list(&batch, col, 0);
        assert_f64_slice_eq(col, &written, &eval_on(&c.data, mt, &grid));
        assert_eq!(written.len(), grid.len(), "{col} is on the union grid");
    }

    // Spot values. Coherent and incoherent are orders of magnitude apart at
    // the same index, so the pair cannot be swapped undetected. FIXTURE is a
    // literal grepped out of photoat-001_H_000.endf; GOLDEN is a capture of
    // the writer's own interpolation between two of them.
    let coherent = f64_list(&batch, "coherent_xs", 0);
    assert_eq!(coherent[0], 4.52522e-6); // FIXTURE, MT 502 at 1 eV
    assert_eq!(coherent[137], 8.943107736147576); // GOLDEN, interpolated
    assert_eq!(coherent[1000], 1.0422148052673345e-5); // GOLDEN, interpolated
    assert_eq!(coherent[2019], 4.6282e-16); // FIXTURE, MT 502 at 100 GeV

    let incoherent = f64_list(&batch, "incoherent_xs", 0);
    assert_eq!(incoherent[0], 9.5623e-8); // FIXTURE
    assert_eq!(incoherent[137], 1.769349772687295e-5); // GOLDEN, interpolated
    assert_eq!(incoherent[1000], 0.254789613); // FIXTURE, a tabulated point
    assert_eq!(incoherent[2019], 1.7042e-5); // FIXTURE

    // Below its threshold the photoelectric curve is CONSTANT at the edge
    // value, not zero: `Tabulated1D::eval` clamps below x[0]. Asserted
    // explicitly so nobody rewrites the comparison as zero-below-threshold.
    let photoelectric = f64_list(&batch, "photoelectric_xs", 0);
    assert_eq!(photoelectric[0], 6318358.25); // FIXTURE, MT 522 at its edge
    assert_eq!(photoelectric[137], 6318358.25); // FIXTURE, the same clamp
    assert!(
        photoelectric[..=137].iter().all(|&v| v == photoelectric[0]),
        "eval clamps below the 13.6 eV edge, so the first 138 points are flat"
    );
    assert_eq!(photoelectric[1000], 4.724903727980379e-9); // GOLDEN
    assert_eq!(photoelectric[2019], 7.736e-15); // FIXTURE

    // The two pair-production channels are identically zero over most of the
    // grid and separate only at the top, so the last point and the threshold
    // index are what tell them apart; comparing index 0 would leave the swap
    // invisible below about 1 MeV.
    let nuclear = f64_list(&batch, "pair_production_nuclear_xs", 0);
    let electron = f64_list(&batch, "pair_production_electron_xs", 0);
    assert_eq!(nuclear[2019], 0.009601); // FIXTURE
    assert_eq!(electron[2019], 0.0111); // FIXTURE
    let first_nuclear = nuclear.iter().position(|&v| v != 0.0).expect("rises");
    let first_electron = electron.iter().position(|&v| v != 0.0).expect("rises");
    assert_eq!(first_nuclear, 1026);
    assert_eq!(first_electron, 1296);
    assert!(
        first_nuclear < first_electron,
        "the nuclear-field threshold 1.022 MeV is below the electron-field 2.044 MeV"
    );
    assert_eq!(
        c.data.get(517).unwrap().xs.as_ref().unwrap().x[0],
        1_022_000.0
    );
    assert_eq!(
        c.data.get(515).unwrap().xs.as_ref().unwrap().x[0],
        2_044_000.0
    );

    // MT 525 is absent from this evaluation, as it is from every ENDF
    // photoatomic file: MF=23 does not carry it. There is no heating column to
    // check, and `constructed_mt_525_writes_no_heating_column` pins that even a
    // constructed MT 525 does not bring one back.
    assert!(!c.data.reactions.contains_key(&525), "H carries no MT 525");
}

/// A failure means an interpolated cross section is no longer the number the
/// evaluation implies.
///
/// Four of the twelve spot values in
/// `element_cross_section_columns_are_the_parsed_reactions_on_the_union_grid`
/// are GOLDEN: they fall between two tabulated points and were captured from a
/// run of the same `Tabulated1D::eval` the writer ran, so they detect a CHANGE
/// to `eval` without saying the value is right. This test derives all four
/// from the evaluation instead.
///
/// The two tabulated points bracketing each one are transcribed here and
/// checked to appear, side by side and in that order, in the decompressed
/// fixture text, so they are reads of the file rather than captures of the
/// code. The interpolant is then formed as the weighted average
/// `(y0 (x1 - e) + y1 (e - x0)) / (x1 - x0)`, deliberately NOT the slope form
/// `y0 + (e - x0) / (x1 - x0) * (y1 - y0)` that `eval` uses
/// (`crates/endf/src/function.rs:103`). The two forms agree to one ulp at worst
/// on these four points, which is why this is the one comparison in the
/// vendored half of this file made to a relative tolerance.
///
/// What this does NOT close. No mutation separates it from the golden
/// literals: any change to `eval` turns both red, so the gain is that the
/// expected value is now derived from the file and cannot be "refreshed" from
/// a run. And every interpolation region in this evaluation is scheme 2, so
/// nothing here checks `eval` in a log region; the expected value is a second
/// hand-written implementation of the same lin-lin rule, not an outside
/// source.
#[test]
fn element_interior_spot_values_are_the_lin_lin_interpolant_of_two_fixture_points() {
    /// One interpolated spot value and the two tabulated points it comes from.
    struct Probe {
        column: &'static str,
        mt: i32,
        /// Index into the union grid, which is where the written column is
        /// read.
        index: usize,
        below: (f64, f64),
        above: (f64, f64),
        /// The two points exactly as they appear in the fixture text, one
        /// eleven-column field each with the single space between them.
        below_text: &'static str,
        above_text: &'static str,
        /// The literal the vendored spot-value test asserts, so the two cannot
        /// drift apart.
        golden: f64,
    }

    let probes = [
        Probe {
            column: "coherent_xs",
            mt: 502,
            index: 137,
            below: (13.5990191, 8.93264869),
            above: (13.6538059, 9.51682411),
            below_text: "13.5990191 8.93264869",
            above_text: "13.6538059 9.51682411",
            golden: 8.943107736147576,
        },
        Probe {
            column: "coherent_xs",
            mt: 502,
            index: 1000,
            below: (629462.706, 1.16803e-5),
            above: (690192.132, 9.71529e-6),
            below_text: "629462.706 1.16803E-5",
            above_text: "690192.132 9.71529E-6",
            golden: 1.0422148052673345e-5,
        },
        Probe {
            column: "incoherent_xs",
            mt: 504,
            index: 137,
            below: (13.3350268, 1.70030e-5),
            above: (13.9234189, 1.85363e-5),
            below_text: "13.3350268 1.70030E-5",
            above_text: "13.9234189 1.85363E-5",
            golden: 1.769349772687295e-5,
        },
        Probe {
            column: "photoelectric_xs",
            mt: 522,
            index: 1000,
            below: (663552.073, 4.80501e-9),
            above: (683911.700, 4.46483e-9),
            below_text: "663552.073 4.80501E-9",
            above_text: "683911.700 4.46483E-9",
            golden: 4.724903727980379e-9,
        },
    ];

    let c = convert(true, false);
    let batch = section(&c.dir, "element.arrow");
    let grid = union_grid_expected(&c.data);
    let raw = text(H_PHOTOAT);

    for p in &probes {
        let (x0, y0) = p.below;
        let (x1, y1) = p.above;

        // In the file, as a pair, in that order.
        assert!(
            raw.contains(p.below_text),
            "`{}` is not in photoat-001_H_000.endf, so it is not a fixture read",
            p.below_text
        );
        assert!(
            raw.contains(p.above_text),
            "`{}` is not in photoat-001_H_000.endf, so it is not a fixture read",
            p.above_text
        );

        // And consecutive points of the reaction the column names, so the
        // transcription is the right bracket rather than two points from
        // elsewhere on the curve.
        let f = c
            .data
            .get(p.mt)
            .and_then(|rx| rx.xs.as_ref())
            .unwrap_or_else(|| panic!("MT {} has no cross section", p.mt));
        let k =
            f.x.iter()
                .position(|&v| v == x0)
                .unwrap_or_else(|| panic!("MT {} does not tabulate {x0:e}", p.mt));
        assert_eq!(f.y[k], y0, "MT {} at {x0:e}", p.mt);
        assert_eq!(f.x[k + 1], x1, "MT {} after {x0:e}", p.mt);
        assert_eq!(f.y[k + 1], y1, "MT {} after {x0:e}", p.mt);
        // One region, lin-lin, which is the rule the expected value uses.
        assert_eq!(f.breakpoints, vec![f.x.len() as i32]);
        assert_eq!(f.interpolation, vec![2]);

        // The union grid point really is inside that bin, so the value is
        // interpolated rather than tabulated or clamped.
        let e = grid[p.index];
        assert!(
            x0 < e && e < x1,
            "grid[{}] = {e:e} is not strictly between {x0:e} and {x1:e}",
            p.index
        );

        let expected = (y0 * (x1 - e) + y1 * (e - x0)) / (x1 - x0);
        let written = f64_list(&batch, p.column, 0)[p.index];
        assert_close(
            &format!("{}[{}]", p.column, p.index),
            written,
            expected,
            1e-15,
        );
        // The same number as the literal in the vendored spot-value test, bit
        // for bit. Refreshing that literal from a run after an `eval` change
        // fails the comparison above, and dropping this one would let the two
        // tests disagree in silence.
        assert_eq!(
            written, p.golden,
            "{}[{}] no longer matches the literal the spot-value test asserts",
            p.column, p.index
        );
    }
}

/// A failure means a form factor or an anomalous scattering term is in the
/// wrong slot.
///
/// These ten columns are the densest interchangeable block in the format, and
/// the real and imaginary anomalous terms share the same 297-point abscissa on
/// this element, so only their first ordinate can tell them apart. The MF=27
/// MT 505 to imaginary and MT 506 to real mapping is a deliberate crossing at
/// `crates/endf/src/incident_photon.rs:383-392` and is exactly the kind of
/// thing a slot swap reproduces.
///
/// The `coherent_int_ff_y` comparison rebuilds the writer's own expression, so
/// it pins the CONSTRUCTION (squared axes, the original region scheme carried
/// onto them) rather than the quadrature. The independent parts of that column
/// are the two asserted beside it: it starts at zero and never decreases. The
/// Z squared divide is NOT among them, because Z is 1 here; it is
/// discriminated by
/// `constructed_element_divides_the_integrated_form_factor_by_z_squared`.
#[test]
fn element_form_factor_columns_come_from_the_reaction_they_name() {
    let c = convert(true, false);
    let batch = section(&c.dir, "element.arrow");
    let z = c.data.atomic_number as f64;

    let coherent = c.data.get(502).expect("MT 502");
    let ff = coherent.scattering_factor.as_ref().expect("MF=27 MT=502");

    let ff_x = f64_list(&batch, "coherent_ff_x", 0);
    let ff_y = f64_list(&batch, "coherent_ff_y", 0);
    assert_f64_slice_eq("coherent_ff_x", &ff_x, &ff.x);
    assert_f64_slice_eq("coherent_ff_y", &ff_y, &ff.y);
    assert_eq!(ff_x.len(), 1253);
    assert_eq!(ff_x[0], 0.0);
    assert_eq!(ff_x[1252], 1.0e9);
    // Both verbatim fixture points. The first is the electron count at zero
    // momentum transfer, which on Z = 1 is indistinguishable from the literal
    // 1.0 and is written as the literal so it does not read as a Z check.
    assert_eq!(ff_y[0], 1.0);
    assert_eq!(ff_y[1252], 8.1829e-39);

    // The integral runs against the SQUARE of the momentum transfer.
    let int_x = f64_list(&batch, "coherent_int_ff_x", 0);
    let squared_x: Vec<f64> = ff.x.iter().map(|v| v * v).collect();
    assert_f64_slice_eq("coherent_int_ff_x", &int_x, &squared_x);
    for (i, (&a, &b)) in int_x.iter().zip(&ff_x).enumerate() {
        assert_eq!(
            a,
            b * b,
            "coherent_int_ff_x[{i}] is not coherent_ff_x[{i}] squared"
        );
    }
    assert_eq!(int_x[0], 0.0);
    assert_eq!(int_x[1252], 1.0e18);

    // `..ff.clone()` carries the original breakpoints and interpolation onto
    // the squared axes, which is what decides the quadrature rule per region.
    assert_eq!(ff.breakpoints, vec![1253]);
    assert_eq!(ff.interpolation, vec![2]);
    let squared = Tabulated1D {
        x: squared_x,
        y: ff.y.iter().map(|v| v * v / z.powi(2)).collect(),
        ..ff.clone()
    };
    let int_y = f64_list(&batch, "coherent_int_ff_y", 0);
    assert_f64_slice_eq("coherent_int_ff_y", &int_y, &squared.integral());
    assert_eq!(int_y[0], 0.0, "a cumulative integral starts at zero");
    assert!(
        int_y.windows(2).all(|w| w[1] >= w[0]),
        "the form factor is non-negative, so its integral cannot decrease"
    );
    // Hydrogen is Z = 1, so the divide above is a divide by one: changing
    // `.powi(2)` to `.powi(5)`, or deleting the divide, leaves this test
    // green. No vendored photoatomic fixture has Z > 1.
    assert_eq!(z, 1.0);

    // Real and imaginary anomalous terms. Same abscissa, so y[0] is the only
    // discriminator on hydrogen.
    let real = coherent.anomalous_real.as_ref().expect("MF=27 MT=506");
    let imag = coherent.anomalous_imag.as_ref().expect("MF=27 MT=505");
    let real_x = f64_list(&batch, "coherent_anomalous_real_x", 0);
    let real_y = f64_list(&batch, "coherent_anomalous_real_y", 0);
    let imag_x = f64_list(&batch, "coherent_anomalous_imag_x", 0);
    let imag_y = f64_list(&batch, "coherent_anomalous_imag_y", 0);
    assert_f64_slice_eq("coherent_anomalous_real_x", &real_x, &real.x);
    assert_f64_slice_eq("coherent_anomalous_real_y", &real_y, &real.y);
    assert_f64_slice_eq("coherent_anomalous_imag_x", &imag_x, &imag.x);
    assert_f64_slice_eq("coherent_anomalous_imag_y", &imag_y, &imag.y);
    assert_eq!(real_x.len(), 297);
    assert_eq!(real_x[0], 1.0);
    assert_eq!(real_x[296], 1.0e7);
    assert_eq!(real_y[0], -1.00260813);
    assert_eq!(imag_y[0], 0.0);
    // Why only the ordinates are checked: in the EVALUATION the two terms share
    // one abscissa, so exchanging the two written x slots produces a
    // byte-identical file. Asserted on the parsed side, not by comparing the
    // two written columns to each other, which would say nothing. The x slots
    // are discriminated by
    // `constructed_anomalous_scattering_columns_keep_their_own_abscissae`.
    assert_eq!(
        imag.x, real.x,
        "MT 505 and MT 506 are tabulated on the same 297 points here"
    );
    assert_ne!(
        real_y, imag_y,
        "a 505/506 swap is only visible in the ordinates on this element"
    );

    // The incoherent scattering function rises from zero to Z. It has 398
    // points against the coherent 1253, so the two length literals are
    // themselves a slot check.
    let iff = c
        .data
        .get(504)
        .and_then(|rx| rx.scattering_factor.as_ref())
        .expect("MF=27 MT=504");
    let iff_x = f64_list(&batch, "incoherent_ff_x", 0);
    let iff_y = f64_list(&batch, "incoherent_ff_y", 0);
    assert_f64_slice_eq("incoherent_ff_x", &iff_x, &iff.x);
    assert_f64_slice_eq("incoherent_ff_y", &iff_y, &iff.y);
    assert_eq!(iff_x.len(), 398);
    assert_eq!(iff_x[0], 0.0);
    assert_eq!(iff_x[397], 1.0e9);
    assert_eq!(iff_y[0], 0.0);
    // Z again, and again a literal: the incoherent scattering function rises
    // to the electron count, which is 1.0 here.
    assert_eq!(iff_y[397], 1.0);
}

// ---------------------------------------------------------------------------
// subshells.arrow
// ---------------------------------------------------------------------------

/// A failure means the per-shell photoelectric data is offset from the grid or
/// logged wrongly.
///
/// This test does NOT establish the row order, and an earlier name saying it
/// did was wrong. Hydrogen has exactly one photoionization channel, so
/// reversing the iteration in `write_subshells` leaves everything here green,
/// and so does deleting `write_compton`'s refusal to accept out-of-order rows.
/// Both are discriminated by the constructed tests in the second half of this
/// file. The `xs` and `ln_xs` comparisons likewise re-run the writer's own
/// `eval` and `ln`, so they pin which reaction reached the row and that the
/// log column is the log of the column beside it, not the quadrature or the
/// logarithm itself.
///
/// What is established here. `threshold_idx` is a `searchsorted` port and the
/// off-by-one it reimplements (side `left` instead of `right`) yields 136, one
/// point below the edge. `ln_xs`, not `xs`, is what the reader loads as the
/// sampled cross section. And `transitions_data` is NULL rather than an empty
/// list, which is byte fidelity with the published files and nothing more: an
/// earlier version of this comment justified it with a reader failure that
/// does not exist, since `read_transitions_from_arrow` returns at
/// `crates/yamc-element/src/photon_arrow.rs:303-305` BEFORE it reads
/// `transitions_shape` at :307, and `try_get_f64_list` collapses null and
/// empty to the same `Vec` anyway.
#[test]
fn subshell_rows_are_the_relaxation_shells_on_the_thresholded_grid() {
    let c = convert(true, false);
    let element = section(&c.dir, "element.arrow");
    let batch = section(&c.dir, "subshells.arrow");
    assert_schema_is_declared(&batch, "subshells.arrow");

    // BTreeMap order is ascending MT, which is most-bound-first, and is the
    // order write_compton refuses to proceed without.
    let shells: Vec<(i32, &str)> = c
        .data
        .reactions
        .range(534..=572)
        .filter(|(_, rx)| rx.xs.is_some())
        .map(|(&mt, rx)| (mt, rx.name().expect("a subshell MT has a name")))
        .collect();
    assert_eq!(batch.num_rows(), shells.len());
    assert_eq!(shells.len(), 1, "hydrogen has one photoionization channel");
    for (row, &(mt, name)) in shells.iter().enumerate() {
        assert_eq!(
            str_at(&batch, "designator", row),
            name,
            "row {row} is not MT {mt}"
        );
    }
    assert_eq!(shells[0], (534, "K"));

    let relaxation = c.data.atomic_relaxation.as_ref().expect("MF=28 was read");
    assert_eq!(
        f64_at(&batch, "binding_energy", 0),
        relaxation.binding_energy["K"]
    );
    assert_eq!(f64_at(&batch, "binding_energy", 0), 13.6);
    assert_eq!(
        f64_at(&batch, "num_electrons", 0),
        relaxation.num_electrons["K"]
    );
    assert_eq!(f64_at(&batch, "num_electrons", 0), 1.0);

    // The zero-default branch, which is what a caller with no relaxation
    // sublibrary gets, and the reason compton's subshell map comes out empty
    // there.
    let bare = convert(false, false);
    assert!(bare.data.atomic_relaxation.is_none());
    let bare_batch = section(&bare.dir, "subshells.arrow");
    assert_eq!(str_at(&bare_batch, "designator", 0), "K");
    assert_eq!(f64_at(&bare_batch, "binding_energy", 0), 0.0);
    assert_eq!(f64_at(&bare_batch, "num_electrons", 0), 0.0);

    // The threshold index, and the tight form of it. The doc comment at
    // photon.rs:191-192 says the stored curve "starts one point before the
    // edge rather than on it", which is not what happens: the subshell's own
    // abscissa feeds the union grid, so the threshold is always itself a grid
    // point and grid[idx] == threshold.
    let grid = union_grid_expected(&c.data);
    let k = c
        .data
        .get(534)
        .and_then(|rx| rx.xs.as_ref())
        .expect("MT 534");
    let idx = i32_at(&batch, "threshold_idx", 0);
    assert_eq!(idx, 137);
    let idx = idx as usize;
    assert_eq!(grid[idx], k.x[0]);
    assert_eq!(k.x[0], 13.6);
    assert!(
        grid[idx - 1] < k.x[0],
        "grid[136] is 13.5990191, which is what a side='left' searchsorted would have picked"
    );

    let xs = f64_list(&batch, "xs", 0);
    let expected: Vec<f64> = grid[idx..].iter().map(|&e| k.eval(e)).collect();
    assert_f64_slice_eq("subshells.xs", &xs, &expected);
    assert_eq!(xs.len(), 1883);
    assert_eq!(xs[0], 6318358.25);
    assert_eq!(xs[1882], 7.736e-15);
    // The invariant the reader relies on when it indexes cross_sections at
    // threshold + j.
    assert_eq!(xs.len() + idx, list_len(&element, "ln_energy", 0));

    let ln_xs = f64_list(&batch, "ln_xs", 0);
    assert!(
        xs.iter().all(|&v| v > 0.0),
        "every K-shell value is positive here, so safe_log's 1e-300 floor is \
         not exercised by any vendored fixture"
    );
    let expected_ln: Vec<f64> = xs.iter().map(|v| v.ln()).collect();
    assert_f64_slice_eq("subshells.ln_xs", &ln_xs, &expected_ln);

    // NTR is zero for hydrogen's only shell, so both cascade columns are null
    // rather than empty, and they have to be null together.
    assert!(relaxation.transitions.is_empty());
    for row in 0..batch.num_rows() {
        assert!(
            is_null(&batch, "transitions_data", row),
            "row {row} has no cascade, so transitions_data is null and not an empty list"
        );
        assert_eq!(
            is_null(&batch, "transitions_data", row),
            is_null(&batch, "transitions_shape", row),
            "row {row} disagrees between the two cascade columns, which the reader cannot load"
        );
    }
}

// ---------------------------------------------------------------------------
// compton.arrow and bremsstrahlung.arrow
// ---------------------------------------------------------------------------

/// A failure means the Compton profiles or the bremsstrahlung DCS were
/// reshaped, transposed or sourced from the wrong table.
///
/// The parsed side for these two sections is `endf::PhotonData` and the
/// `IncidentPhoton` AFTER `add_photon_data`, not the ENDF material, because no
/// evaluation carries them. Both shape columns are silently reshapeable: a
/// transposed `[31, 1]` or `[30, 200]` has the same element count and the
/// reader accepts it, so each shape is asserted element by element rather than
/// by its product.
///
/// `J_data` and `dcs_data` are compared against the parsed tabulation, which
/// pins the slot and the ravel offset but not the spline that produced the
/// DCS. `J_cdf_data` is the one column recomputed independently here.
///
/// The CSR subshell map is WIRING ONLY on this element and cannot be anything
/// else. Hydrogen has one Compton shell and one subshell, so both arguments of
/// `compton_subshell_map` are `[1.0]`, exchanging them changes nothing, and
/// `[0, 1] / [0] / [1.0]` is the only non-empty triple a one-shell element can
/// produce, whatever the grouping algorithm. Same for the shell-major ravel of
/// `J_data`: one row has one order. Both are discriminated by
/// `constructed_compton_shells_map_onto_the_subshell_rows_by_occupancy`.
#[test]
fn compton_and_bremsstrahlung_are_written_from_the_auxiliary_tabulations() {
    let c = convert(true, true);
    let subshells = section(&c.dir, "subshells.arrow");
    let batch = section(&c.dir, "compton.arrow");
    assert_schema_is_declared(&batch, "compton.arrow");
    assert_eq!(batch.num_rows(), 1, "compton.arrow is one row per element");

    let profiles = c
        .data
        .compton_profiles
        .as_ref()
        .expect("add_photon_data attached the Biggs profiles");

    let num_electrons = f64_list(&batch, "num_electrons", 0);
    assert_f64_slice_eq(
        "compton.num_electrons",
        &num_electrons,
        &profiles.num_electrons,
    );
    assert_eq!(num_electrons, vec![1.0]);
    assert_eq!(num_electrons.len(), profiles.j.len());

    // 13.61 from the Biggs tabulation, against 13.6 from MF=28 and MF=23. The
    // difference is what makes this a usable discriminator that the column
    // came from the tabulation and not from subshells.binding_energy.
    let binding_energy = f64_list(&batch, "binding_energy", 0);
    assert_f64_slice_eq(
        "compton.binding_energy",
        &binding_energy,
        &profiles.binding_energy,
    );
    assert_eq!(binding_energy, vec![13.61]);
    assert_ne!(
        binding_energy[0],
        f64_at(&subshells, "binding_energy", 0),
        "the Compton binding energy is not the relaxation one"
    );

    // Only the first shell's abscissa is written, so a ragged upstream would
    // be dropped without a word. The loop below that would notice runs exactly
    // once here, and `pz = first.x` against a last-shell source stays
    // undiscriminated: every row of the Biggs tabulation is built from the
    // same `data.pz`, so no input this writer can be handed is ragged.
    let pz = f64_list(&batch, "pz", 0);
    assert_f64_slice_eq("compton.pz", &pz, &profiles.j[0].x);
    assert_eq!(pz.len(), 31);
    assert_eq!(pz[0], 0.0);
    assert_eq!(pz[30], 100.0);
    for (s, row) in profiles.j.iter().enumerate() {
        assert_f64_slice_eq(&format!("the momentum grid of shell {s}"), &row.x, &pz);
        assert_eq!(row.y.len(), pz.len(), "shell {s} is ragged");
    }

    let j_data = f64_list(&batch, "J_data", 0);
    let expected_j: Vec<f64> = profiles
        .j
        .iter()
        .flat_map(|t| t.y.iter().copied())
        .collect();
    assert_f64_slice_eq("J_data", &j_data, &expected_j);
    // Hydrogen has one Compton shell, so the comparison above is one row and
    // the shell-major ravel ORDER cannot be tested here; the arithmetic is
    // pinned instead against the literal first three of the Z=1 row of
    // compton_profiles_biggs1975.txt.
    assert_f64_slice_eq(
        "the Z=1 row of the Biggs tabulation",
        &j_data[..3],
        &[0.849, 0.842, 0.824],
    );

    let j_shape = i32_list(&batch, "J_shape", 0);
    assert_eq!(j_shape.len(), 2);
    assert_eq!(j_shape[0], profiles.j.len() as i32);
    assert_eq!(j_shape[1], pz.len() as i32);
    assert_i32_slice_eq("J_shape", &j_shape, &[1, 31]);
    assert_eq!((j_shape[0] * j_shape[1]) as usize, j_data.len());

    // The trapezoid, accumulated here rather than by calling
    // compton_profile_cdfs, which is the function the writer called. The
    // factors are in the opposite order to the writer's, so the last bit can
    // differ and this is the one column compared to a relative tolerance.
    let cdf = f64_list(&batch, "J_cdf_data", 0);
    let mut expected_cdf: Vec<f64> = Vec::with_capacity(cdf.len());
    for row in &profiles.j {
        let mut running = 0.0;
        expected_cdf.push(0.0);
        for i in 1..row.y.len() {
            running += (pz[i] - pz[i - 1]) * (row.y[i - 1] + row.y[i]) / 2.0;
            expected_cdf.push(running);
        }
    }
    assert_f64_slice_close("J_cdf_data", &cdf, &expected_cdf, 1e-15);
    assert_eq!(cdf[0], 0.0);
    assert!(
        cdf.windows(2).all(|w| w[1] >= w[0]),
        "a cumulative distribution cannot decrease"
    );
    // Deliberately un-normalised: a profile is tabulated for positive momentum
    // only, so this row ends at 0.5021940797600001 and normalising it to 1.0
    // would be wrong. No separate assertion, because the element-wise
    // comparison above is against the un-normalised trapezoid and already
    // fails if the writer starts normalising.
    let cdf_shape = i32_list(&batch, "J_cdf_shape", 0);
    assert_i32_slice_eq(
        "J_cdf_shape",
        &cdf_shape,
        &[profiles.j.len() as i32, profiles.j[0].y.len() as i32],
    );
    assert_eq!((cdf_shape[0] * cdf_shape[1]) as usize, cdf.len());

    // The CSR map from Compton shells onto subshell rows.
    let offsets = i32_list(&batch, "subshell_map_offsets", 0);
    let indices = i32_list(&batch, "subshell_map_indices", 0);
    let weights = f64_list(&batch, "subshell_map_weights", 0);
    assert_eq!(offsets.len(), num_electrons.len() + 1);
    assert_eq!(offsets[0], 0);
    assert!(offsets.windows(2).all(|w| w[1] >= w[0]));
    assert_eq!(*offsets.last().unwrap() as usize, indices.len());
    assert_eq!(weights.len(), indices.len());
    assert_i32_slice_eq("subshell_map_offsets", &offsets, &[0, 1]);
    assert_i32_slice_eq("subshell_map_indices", &indices, &[0]);
    assert_f64_slice_eq("subshell_map_weights", &weights, &[1.0]);
    for &i in &indices {
        assert!(
            (i as usize) < subshells.num_rows(),
            "index {i} is not a row of subshells.arrow"
        );
    }
    // Checkable from the two written files alone: each group of subshell rows
    // has to carry the Compton shell's own occupancy, to the tolerance
    // compton_subshell_map itself enforces. One iteration here, comparing 1.0
    // against 1.0; the multi-shell case is in the constructed test.
    for shell in 0..num_electrons.len() {
        let (lo, hi) = (offsets[shell] as usize, offsets[shell + 1] as usize);
        if lo == hi {
            continue;
        }
        let grouped: f64 = indices[lo..hi]
            .iter()
            .map(|&i| f64_at(&subshells, "num_electrons", i as usize))
            .sum();
        assert_close(
            &format!("the occupancy of Compton shell {shell}"),
            grouped,
            num_electrons[shell],
            1e-3,
        );
        let total: f64 = weights[lo..hi].iter().sum();
        assert_close(
            &format!("the weights of Compton shell {shell}"),
            total,
            1.0,
            1e-12,
        );
    }

    // The empty branch. Without the relaxation file every subshell occupancy
    // is 0.0, the tolerance test fails on the first shell and the map is
    // dropped whole.
    let bare = convert(false, true);
    let bare_compton = section(&bare.dir, "compton.arrow");
    assert_eq!(
        f64_at(&section(&bare.dir, "subshells.arrow"), "num_electrons", 0),
        0.0
    );
    assert_i32_slice_eq(
        "subshell_map_offsets with no relaxation",
        &i32_list(&bare_compton, "subshell_map_offsets", 0),
        &[0, 0],
    );
    assert!(i32_list(&bare_compton, "subshell_map_indices", 0).is_empty());
    assert!(f64_list(&bare_compton, "subshell_map_weights", 0).is_empty());

    // bremsstrahlung.arrow.
    let brem_batch = section(&c.dir, "bremsstrahlung.arrow");
    assert_schema_is_declared(&brem_batch, "bremsstrahlung.arrow");
    assert_eq!(
        brem_batch.num_rows(),
        1,
        "I is the only non-list column, so the batch is one row"
    );
    let brem = c
        .data
        .bremsstrahlung
        .as_ref()
        .expect("add_photon_data attached the Seltzer-Berger data");

    assert_eq!(f64_at(&brem_batch, "I", 0), brem.i);
    assert_eq!(f64_at(&brem_batch, "I", 0), 19.2);

    let electron_energy = f64_list(&brem_batch, "electron_energy", 0);
    assert_f64_slice_eq("electron_energy", &electron_energy, &brem.electron_energy);
    let expected_ee: Vec<f64> = (0..200)
        .map(|i| 10f64.powf(3.0 + 6.0 * i as f64 / 199.0))
        .collect();
    assert_f64_slice_eq("electron_energy", &electron_energy, &expected_ee);
    assert_eq!(electron_energy.len(), 200);
    assert_close("electron_energy[0]", electron_energy[0], 1.0e3, 1e-12);
    assert_close("electron_energy[199]", electron_energy[199], 1.0e9, 1e-9);

    let photon_energy = f64_list(&brem_batch, "photon_energy", 0);
    assert_f64_slice_eq("photon_energy", &photon_energy, &brem.photon_energy);
    assert_eq!(photon_energy.len(), 30);

    // Two adjacent type-identical columns which, on hydrogen, only their
    // values tell apart.
    let brem_electrons = f64_list(&brem_batch, "num_electrons", 0);
    let ionization = f64_list(&brem_batch, "ionization_energy", 0);
    assert_f64_slice_eq(
        "bremsstrahlung.num_electrons",
        &brem_electrons,
        &brem.num_electrons,
    );
    assert_f64_slice_eq("ionization_energy", &ionization, &brem.ionization_energy);
    assert_eq!(brem_electrons, vec![1.0]);
    assert_eq!(ionization, vec![13.6]);
    assert_eq!(ionization.len(), brem_electrons.len());

    let dcs = f64_list(&brem_batch, "dcs_data", 0);
    assert_eq!(dcs.len(), 200 * 30);
    for (i, row) in brem.dcs.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(
                dcs[i * 30 + j],
                v,
                "dcs_data[{i} * 30 + {j}] is not dcs[{i}][{j}]"
            );
        }
    }
    // The ravel orientation, pinned without recomputing the spline: a
    // transposed ravel would put dcs[1][0] at flat index 1.
    assert_eq!(dcs[1], brem.dcs[0][1]);
    assert_ne!(
        brem.dcs[0][1], brem.dcs[1][0],
        "the two candidates for flat index 1 have to differ for this to mean anything"
    );

    let shape = i32_list(&brem_batch, "dcs_shape", 0);
    assert_i32_slice_eq("dcs_shape", &shape, &[200, 30]);
    assert_eq!(shape[0] as usize, electron_energy.len());
    assert_eq!(shape[1] as usize, photon_energy.len());
    assert_eq!((shape[0] * shape[1]) as usize, dcs.len());
}

// ---------------------------------------------------------------------------
// A constructed element, for the branches hydrogen cannot reach.
// ---------------------------------------------------------------------------

/// A hand-built `IncidentPhoton`. NOT A PARSED EVALUATION.
///
/// `photoat-001_H_000.endf` is the only photoatomic file in the tree and it is
/// Z = 1 with one subshell and one Compton shell. That leaves five writer
/// branches undiscriminated however the four tests above are written: the
/// `(z as f64).powi(2)` divide is a divide by one; the subshell row order is
/// the order of a single row; both arguments of `compton_subshell_map` are the
/// same `[1.0]`; `safe_log`'s floor is never reached because every K-shell
/// value is positive; and NTR is zero, so the relaxation cascade table is
/// never built. Each was measured to survive a source mutation with every
/// vendored test still green.
///
/// So this element is invented. Nothing asserted against it is agreement with
/// ENDF. It pins the writer's arithmetic, its column order and its ravel
/// order, and it is worth nothing at all as evaluation parity.
///
/// Every field is given a DISTINCT value, which is the entire point of
/// building one: two columns that are equal cannot discriminate a swap, and
/// that is exactly what went wrong on hydrogen.
///
/// The layout, chosen so every expected answer below is exact in binary:
///
/// * Z = 10, so the form factor divide is by 100 rather than by 1.
/// * Cross sections tabulated so the union grid is `[1, 5, 10, 20, 50, 100]`.
/// * Three subshells K, L1 and L2, binding energies 50, 20 and 5 eV,
///   occupancies 2, 2 and 6 summing to Z, thresholds equal to the binding
///   energies. The rows are then most-bound-first and every subshell column
///   changes under a reversal.
/// * The K-shell cross section is exactly zero at its own threshold, which is
///   the only way to reach `safe_log`'s floor.
/// * Two Compton shells, occupancies 2 and 8, grouping onto the three subshell
///   rows as {K} and {L1, L2}. The two arguments of `compton_subshell_map`
///   then differ in length as well as in value.
/// * Two Compton profile rows with different ordinates, so the shell-major
///   ravel of `J_data` is observable.
/// * MT 525 carries a cross section, which no vendored photoatomic file does.
fn constructed_element() -> IncidentPhoton {
    fn channel(mt: i32, x: &[f64], y: &[f64]) -> PhotonReaction {
        let mut rx = PhotonReaction::new(mt);
        rx.xs = Some(Tabulated1D::new(x.to_vec(), y.to_vec()));
        rx
    }

    let mut data = IncidentPhoton::new(10);

    let mut coherent = channel(502, &[1.0, 10.0, 100.0], &[2.0, 3.0, 5.0]);
    // A triangle: Z at zero momentum transfer falling to zero at 2. Squared
    // and divided by Z^2 that is a line from 1 to 0 over [0, 4], whose area is
    // exactly 2.
    coherent.scattering_factor = Some(Tabulated1D::new(vec![0.0, 2.0], vec![10.0, 0.0]));
    // Different abscissae as well as different ordinates, which hydrogen does
    // not have: there MT 505 and MT 506 share one 297-point grid.
    coherent.anomalous_real = Some(Tabulated1D::new(vec![1.0, 2.0], vec![-1.5, -0.5]));
    coherent.anomalous_imag = Some(Tabulated1D::new(vec![1.0, 3.0], vec![0.25, 0.75]));
    data.reactions.insert(502, coherent);

    let mut incoherent = channel(504, &[1.0, 10.0, 100.0], &[7.0, 11.0, 13.0]);
    incoherent.scattering_factor = Some(Tabulated1D::new(vec![0.0, 3.0], vec![0.0, 10.0]));
    data.reactions.insert(504, incoherent);

    data.reactions
        .insert(522, channel(522, &[10.0, 100.0], &[17.0, 19.0]));
    data.reactions
        .insert(525, channel(525, &[1.0, 10.0, 100.0], &[23.0, 29.0, 31.0]));

    // K is zero at its own threshold, which is what reaches the log floor.
    data.reactions
        .insert(534, channel(534, &[50.0, 100.0], &[0.0, 4.0]));
    data.reactions
        .insert(535, channel(535, &[20.0, 100.0], &[6.0, 8.0]));
    data.reactions
        .insert(536, channel(536, &[5.0, 100.0], &[9.0, 12.0]));

    let mut relaxation = AtomicRelaxation::default();
    for (shell, binding, electrons) in [("K", 50.0, 2.0), ("L1", 20.0, 2.0), ("L2", 5.0, 6.0)] {
        relaxation.binding_energy.insert(shell, binding);
        relaxation.num_electrons.insert(shell, electrons);
    }
    // One radiative transition, with no tertiary subshell, and one
    // non-radiative one that has both. The two rows differ in every column.
    relaxation.transitions.insert(
        "K",
        Transitions {
            secondary_subshell: vec!["L1", "L2"],
            tertiary_subshell: vec!["", "L2"],
            energy: vec![30.0, 40.0],
            probability: vec![0.4, 0.6],
        },
    );
    data.atomic_relaxation = Some(relaxation);

    data.compton_profiles = Some(ComptonProfiles {
        num_electrons: vec![2.0, 8.0],
        // Deliberately not the relaxation binding energies, so a column
        // sourced from the wrong place is visible.
        binding_energy: vec![51.0, 6.0],
        j: vec![
            Tabulated1D::new(vec![0.0, 1.0, 2.0], vec![0.9, 0.5, 0.1]),
            Tabulated1D::new(vec![0.0, 1.0, 2.0], vec![0.8, 0.4, 0.2]),
        ],
    });

    data
}

/// The real `write_photon` over a constructed element, into a scratch tree
/// that removes itself when an assertion panics.
fn write_constructed(data: &IncidentPhoton) -> (tempfile::TempDir, PathBuf) {
    let scratch = scratch();
    let dir = scratch.path().join("Ne.arrow");
    std::fs::create_dir_all(&dir).expect("the output directory is created");
    yamc_convert::photon::write_photon(data, &dir).expect("the constructed element writes");
    (scratch, dir)
}

/// The union grid of [`constructed_element`], as a fact about the fixture
/// rather than a restatement of `union_grid`.
const CONSTRUCTED_GRID: [f64; 6] = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0];

/// A failure means the cumulative coherent form factor lost its Z squared
/// normalisation.
///
/// CONSTRUCTED INPUT, not an evaluation: no vendored photoatomic file has
/// Z > 1, so on `photoat-001_H_000.endf` the divide at `photon.rs:115` is a
/// divide by one and `.powi(2)` can be changed to `.powi(5)`, or deleted
/// outright, with every other test in this file still green.
///
/// The expected integral is a closed form and not a rerun of the writer's
/// expression. The form factor is the straight line from `(0, Z)` to `(2, 0)`,
/// so the squared and normalised function is the line from `(0, 1)` to
/// `(4, 0)` and the area under it is exactly 2. With `.powi(5)` it would be
/// 0.002 and with no divide at all 200.
#[test]
fn constructed_element_divides_the_integrated_form_factor_by_z_squared() {
    let data = constructed_element();
    let (_scratch, dir) = write_constructed(&data);
    let batch = section(&dir, "element.arrow");
    assert_schema_is_declared(&batch, "element.arrow");

    assert_eq!(str_at(&batch, "name", 0), "Ne");
    assert_eq!(i32_at(&batch, "Z", 0), 10);

    // The form factor itself is copied through untouched, so the divide has to
    // show up in the integral and nowhere else.
    assert_f64_slice_eq(
        "coherent_ff_x",
        &f64_list(&batch, "coherent_ff_x", 0),
        &[0.0, 2.0],
    );
    assert_f64_slice_eq(
        "coherent_ff_y",
        &f64_list(&batch, "coherent_ff_y", 0),
        &[10.0, 0.0],
    );

    assert_f64_slice_eq(
        "coherent_int_ff_x",
        &f64_list(&batch, "coherent_int_ff_x", 0),
        &[0.0, 4.0],
    );
    assert_f64_slice_eq(
        "coherent_int_ff_y",
        &f64_list(&batch, "coherent_int_ff_y", 0),
        &[0.0, 2.0],
    );
}

/// A failure means the two anomalous scattering terms were crossed.
///
/// CONSTRUCTED INPUT, not an evaluation. On hydrogen MT 505 and MT 506 share
/// one 297-point abscissa, so exchanging the two x slots writes a
/// byte-identical file and only the ordinates separate them. Here the two
/// terms are tabulated on different abscissae of different lengths, so all
/// four slots are independent.
///
/// The MT 505 to imaginary and MT 506 to real mapping is a deliberate crossing
/// at `crates/endf/src/incident_photon.rs:383-392`; this test is downstream of
/// it and pins the writer's slots, not the parser's crossing.
#[test]
fn constructed_anomalous_scattering_columns_keep_their_own_abscissae() {
    let data = constructed_element();
    let (_scratch, dir) = write_constructed(&data);
    let batch = section(&dir, "element.arrow");

    assert_f64_slice_eq(
        "coherent_anomalous_real_x",
        &f64_list(&batch, "coherent_anomalous_real_x", 0),
        &[1.0, 2.0],
    );
    assert_f64_slice_eq(
        "coherent_anomalous_real_y",
        &f64_list(&batch, "coherent_anomalous_real_y", 0),
        &[-1.5, -0.5],
    );
    assert_f64_slice_eq(
        "coherent_anomalous_imag_x",
        &f64_list(&batch, "coherent_anomalous_imag_x", 0),
        &[1.0, 3.0],
    );
    assert_f64_slice_eq(
        "coherent_anomalous_imag_y",
        &f64_list(&batch, "coherent_anomalous_imag_y", 0),
        &[0.25, 0.75],
    );

    // The incoherent scattering function is the fifth same-typed neighbour of
    // those four and is given its own third length here.
    assert_f64_slice_eq(
        "incoherent_ff_x",
        &f64_list(&batch, "incoherent_ff_x", 0),
        &[0.0, 3.0],
    );
    assert_f64_slice_eq(
        "incoherent_ff_y",
        &f64_list(&batch, "incoherent_ff_y", 0),
        &[0.0, 10.0],
    );
}

/// A failure means `element.arrow` grew a heating column back.
///
/// `heating_xs` was declared but unfillable and is now gone. MT 525 is the MT
/// named `"heating"` (`crates/endf/src/incident_photon.rs:33`) and it reaches
/// an `IncidentPhoton` only through `from_ace`; the converter is ENDF-only,
/// and ENDF photoatomic MF=23 has no MT 525. The column was empty in all 261
/// published elements across endf-b8.1, fendl-3.2d and jendl-5.0, so nothing
/// ever read a value from it.
///
/// CONSTRUCTED INPUT, not an evaluation: `photoat-001_H_000.endf` has no
/// MF=23 MT=525 and is the only photoatomic file in the tree, so this is the
/// only way to hand the writer an MT 525 at all. If an ACE photoatomic route
/// is ever added, this test is what should go red, and the column, the schema
/// field and the reader all come back together rather than one at a time.
#[test]
fn constructed_mt_525_writes_no_heating_column() {
    let data = constructed_element();
    // The input really does carry a heating cross section, under the name the
    // writer used to ask for.
    let heating = data.get(525).expect("the constructed element has MT 525");
    assert_eq!(heating.name(), Some("heating"));
    assert_f64_slice_eq(
        "the MT 525 cross section handed to the writer",
        &heating.xs.as_ref().expect("MT 525 has a cross section").y,
        &[23.0, 29.0, 31.0],
    );

    let (_scratch, dir) = write_constructed(&data);
    let batch = section(&dir, "element.arrow");

    // The channels ELEMENT_MTS lists are on the union grid.
    assert_eq!(list_len(&batch, "ln_energy", 0), CONSTRUCTED_GRID.len());
    assert_eq!(list_len(&batch, "coherent_xs", 0), CONSTRUCTED_GRID.len());

    // The heating column is absent, not empty. An empty column is what the old
    // defect looked like, so asserting absence is what tells the two apart.
    assert!(
        batch.column_by_name("heating_xs").is_none(),
        "MT 525 was supplied and must not produce a heating_xs column"
    );
    assert_schema_is_declared(&batch, "element.arrow");
}

/// A failure means `subshells.arrow` is no longer written most-bound-first.
///
/// CONSTRUCTED INPUT, not an evaluation: hydrogen has one photoionization
/// channel, so on the vendored fixture the row order is the order of a single
/// row and replacing the loop at `photon.rs:180` with
/// `data.reactions.iter().rev()` leaves every other test in this file green.
/// The order is load bearing twice over: `compton_subshell_map` walks the two
/// occupancy lists in step, and the reader indexes `subshell_map_indices` into
/// these rows.
///
/// The element is written without Compton profiles on purpose, so that a
/// reversal is caught by the row assertions here rather than by
/// `write_compton`'s guard, which is exercised separately in
/// `constructed_out_of_order_subshell_rows_are_refused_by_write_compton`.
#[test]
fn constructed_subshell_rows_are_written_most_bound_first() {
    let mut data = constructed_element();
    data.compton_profiles = None;
    let (_scratch, dir) = write_constructed(&data);
    let batch = section(&dir, "subshells.arrow");
    assert_schema_is_declared(&batch, "subshells.arrow");
    assert_eq!(batch.num_rows(), 3);

    let designators: Vec<String> = (0..3).map(|r| str_at(&batch, "designator", r)).collect();
    assert_eq!(designators, ["K", "L1", "L2"]);

    let binding: Vec<f64> = (0..3)
        .map(|r| f64_at(&batch, "binding_energy", r))
        .collect();
    assert_eq!(binding, [50.0, 20.0, 5.0]);
    assert!(
        binding.windows(2).all(|w| w[0] > w[1]),
        "most-bound-first means the binding energy falls down the rows"
    );

    let electrons: Vec<f64> = (0..3).map(|r| f64_at(&batch, "num_electrons", r)).collect();
    assert_eq!(electrons, [2.0, 2.0, 6.0]);

    // The thresholds fall with the binding energies, so the index into the
    // union grid falls too. Grid: [1, 5, 10, 20, 50, 100].
    let grid = union_grid_expected(&data);
    assert_f64_slice_eq("the constructed union grid", &grid, &CONSTRUCTED_GRID);
    let thresholds: Vec<i32> = (0..3).map(|r| i32_at(&batch, "threshold_idx", r)).collect();
    assert_eq!(thresholds, [4, 3, 1]);

    // Each row's curve starts at its own threshold, so the three rows have
    // three different lengths as well as three different first values.
    assert_f64_slice_eq(
        "row 0, the K curve",
        &f64_list(&batch, "xs", 0),
        &[0.0, 4.0],
    );
    assert_f64_slice_eq(
        "row 1, the L1 curve",
        &f64_list(&batch, "xs", 1),
        &[6.0, 6.75, 8.0],
    );
    let l2 = f64_list(&batch, "xs", 2);
    assert_eq!(l2.len(), 5);
    assert_eq!(l2[0], 9.0);
    assert_eq!(l2[4], 12.0);
}

/// A failure means a zero cross section reaches the sampler as an infinity.
///
/// CONSTRUCTED INPUT, not an evaluation: every K-shell value in
/// `photoat-001_H_000.endf` is strictly positive, so `safe_log`'s floor
/// (`photon.rs:43`) is never taken there and changing `1e-300` to `1e-30`
/// leaves every other test in this file green. The constructed K-shell cross
/// section is exactly zero at its own threshold, which is the first point of
/// its stored curve.
///
/// The floor is compared against `1e-300f64.ln()` computed here rather than
/// against the literal -690.7755278982137, so a runner whose `ln` differs in
/// the last bit does not fail on the constant. The mutation this catches moves
/// the value by a factor of ten, not by an ulp.
#[test]
fn constructed_zero_cross_section_reaches_the_safe_log_floor() {
    let data = constructed_element();
    let (_scratch, dir) = write_constructed(&data);
    let batch = section(&dir, "subshells.arrow");

    let xs = f64_list(&batch, "xs", 0);
    let ln_xs = f64_list(&batch, "ln_xs", 0);
    assert_eq!(xs[0], 0.0, "the K curve starts at zero, which is the point");
    assert_eq!(ln_xs.len(), xs.len());
    assert_eq!(ln_xs[0], 1e-300f64.ln());
    assert!(
        ln_xs[0].is_finite(),
        "an infinity here comes back as a NaN energy several steps later"
    );
    // The positive values beside it are the ordinary log, so the floor is a
    // floor and not a blanket replacement.
    assert_eq!(ln_xs[1], 4.0f64.ln());
}

/// A failure means the relaxation cascade is transposed or its subshell
/// indices are wrong.
///
/// CONSTRUCTED INPUT, not an evaluation: `atom-001_H_000.endf` gives hydrogen
/// one subshell with NTR = 0, and it is the only relaxation file in the tree,
/// so `transitions_table` (`photon.rs:66-77`) is never called on any vendored
/// input and both cascade columns are only ever null. The NULL branch is
/// covered by the vendored test; this is the populated one.
///
/// The two subshell columns are indices into
/// `endf::incident_photon::SUBSHELLS` rather than names, so the whole table
/// can be one `f64` array. A radiative transition has no tertiary subshell and
/// is written as index 0, which is the empty name and not `"K"`.
#[test]
fn constructed_subshell_cascade_is_a_row_major_table_of_subshell_indices() {
    let data = constructed_element();
    let (_scratch, dir) = write_constructed(&data);
    let batch = section(&dir, "subshells.arrow");

    // The indices the flat table encodes, so the literals below are legible.
    assert_eq!(SUBSHELLS[0], "");
    assert_eq!(SUBSHELLS[2], "L1");
    assert_eq!(SUBSHELLS[3], "L2");

    // Row 0 is K, which is the only shell given a cascade. Two transitions of
    // four columns each: secondary, tertiary, energy, probability.
    assert!(!is_null(&batch, "transitions_data", 0));
    assert_i32_slice_eq(
        "K transitions_shape",
        &i32_list(&batch, "transitions_shape", 0),
        &[2, 4],
    );
    assert_f64_slice_eq(
        "K transitions_data",
        &f64_list(&batch, "transitions_data", 0),
        &[2.0, 0.0, 30.0, 0.4, 3.0, 3.0, 40.0, 0.6],
    );

    // L1 and L2 have none, so they stay null in a file whose first row does
    // not.
    for row in 1..3 {
        assert!(
            is_null(&batch, "transitions_data", row),
            "row {row} has no cascade, so transitions_data is null"
        );
        assert!(is_null(&batch, "transitions_shape", row));
    }
}

/// A failure means the Compton subshell map groups the wrong way round, or the
/// profile rows were ravelled in the wrong order.
///
/// CONSTRUCTED INPUT, not an evaluation. On hydrogen both arguments of
/// `compton_subshell_map` (`photon.rs:284`) are `[1.0]`, so exchanging them
/// changes nothing and the CSR triple can only be `[0, 1] / [0] / [1.0]`,
/// which is the single answer a one-shell element can give. One Compton shell
/// also makes every ravel order of `J_data` the same bytes. This element has
/// two Compton shells over three subshell rows, with different occupancies and
/// different profile ordinates.
#[test]
fn constructed_compton_shells_map_onto_the_subshell_rows_by_occupancy() {
    let data = constructed_element();
    let (_scratch, dir) = write_constructed(&data);
    let batch = section(&dir, "compton.arrow");
    let subshells = section(&dir, "subshells.arrow");
    assert_schema_is_declared(&batch, "compton.arrow");

    assert_f64_slice_eq(
        "compton.num_electrons",
        &f64_list(&batch, "num_electrons", 0),
        &[2.0, 8.0],
    );
    assert_f64_slice_eq(
        "compton.binding_energy",
        &f64_list(&batch, "binding_energy", 0),
        &[51.0, 6.0],
    );

    // Shell-major: row 0 then row 1, not the transpose.
    assert_f64_slice_eq("compton.pz", &f64_list(&batch, "pz", 0), &[0.0, 1.0, 2.0]);
    assert_f64_slice_eq(
        "J_data",
        &f64_list(&batch, "J_data", 0),
        &[0.9, 0.5, 0.1, 0.8, 0.4, 0.2],
    );
    assert_i32_slice_eq("J_shape", &i32_list(&batch, "J_shape", 0), &[2, 3]);

    // The trapezoid, in closed form: 0.5 * (0.9 + 0.5) * 1 then
    // 0.5 * (0.5 + 0.1) * 1 on top of it, and the same for the second row.
    assert_f64_slice_close(
        "J_cdf_data",
        &f64_list(&batch, "J_cdf_data", 0),
        &[0.0, 0.7, 1.0, 0.0, 0.6, 0.9],
        1e-15,
    );
    assert_i32_slice_eq("J_cdf_shape", &i32_list(&batch, "J_cdf_shape", 0), &[2, 3]);

    // Compton shell 0 (2 electrons) is K alone; shell 1 (8 electrons) is
    // L1 plus L2, weighted 2/8 and 6/8.
    let offsets = i32_list(&batch, "subshell_map_offsets", 0);
    assert_i32_slice_eq("subshell_map_offsets", &offsets, &[0, 1, 3]);
    assert_i32_slice_eq(
        "subshell_map_indices",
        &i32_list(&batch, "subshell_map_indices", 0),
        &[0, 1, 2],
    );
    assert_f64_slice_eq(
        "subshell_map_weights",
        &f64_list(&batch, "subshell_map_weights", 0),
        &[1.0, 0.25, 0.75],
    );
    assert_eq!(
        subshells.num_rows(),
        3,
        "the indices above are rows of subshells.arrow"
    );

    // The swap this element exists to catch, stated so the assertions above
    // are demonstrably not the same under both argument orders. Feeding the
    // subshell occupancies as the Compton ones yields one offset per subshell
    // row and stops after the first group.
    let (swapped, _, _) = compton_subshell_map(&[2.0, 2.0, 6.0], &[2.0, 8.0]);
    assert_eq!(
        swapped,
        vec![0usize, 1, 1, 1],
        "the two argument orders have to disagree for the CSR triple above to \
         discriminate between them"
    );
}

/// A failure means `write_compton` stopped refusing subshell rows it cannot
/// map.
///
/// CONSTRUCTED INPUT, and a deliberately malformed one. The guard at
/// `photon.rs:268-278` is unreachable through any parser: `write_subshells`
/// walks `data.reactions` in `BTreeMap` key order and names each row from
/// `PhotonReaction::name`, which reads the reaction's own `mt`, and every
/// constructor in `crates/endf` inserts a reaction under its own MT, so the
/// designators come out in ascending `SUBSHELLS` order by construction. The
/// only way to reach the guard is to insert a reaction under a key that is not
/// its `mt`, which is what this does. Nothing here says a real evaluation can
/// produce that state; it says the guard fires when it does.
///
/// The refusal happens after `element.arrow` and `subshells.arrow` are already
/// on disk, so a caller that ignores the error is left with a partial tree.
#[test]
fn constructed_out_of_order_subshell_rows_are_refused_by_write_compton() {
    let mut data = IncidentPhoton::new(10);

    let mut l2 = PhotonReaction::new(536);
    l2.xs = Some(Tabulated1D::new(vec![5.0, 100.0], vec![9.0, 12.0]));
    let mut k = PhotonReaction::new(534);
    k.xs = Some(Tabulated1D::new(vec![50.0, 100.0], vec![1.0, 4.0]));
    // Key 534 holds the reaction that names itself L2, key 535 the one that
    // names itself K, so the rows come out L2 then K: least bound first.
    data.reactions.insert(534, l2);
    data.reactions.insert(535, k);
    data.compton_profiles = Some(ComptonProfiles {
        num_electrons: vec![2.0, 8.0],
        binding_energy: vec![51.0, 6.0],
        j: vec![
            Tabulated1D::new(vec![0.0, 1.0, 2.0], vec![0.9, 0.5, 0.1]),
            Tabulated1D::new(vec![0.0, 1.0, 2.0], vec![0.8, 0.4, 0.2]),
        ],
    });

    let scratch = scratch();
    let dir = scratch.path().join("Ne.arrow");
    std::fs::create_dir_all(&dir).expect("the output directory is created");
    let err = yamc_convert::photon::write_photon(&data, &dir)
        .expect_err("out-of-order subshell rows are refused")
        .to_string();
    assert!(
        err.contains("not in most-bound-first order"),
        "the refusal came from somewhere else: {err}"
    );
    assert!(
        absent(&dir, "compton.arrow"),
        "the refusal has to leave compton.arrow unwritten"
    );
    assert_eq!(
        str_at(&section(&dir, "subshells.arrow"), "designator", 0),
        "L2",
        "the rows really were out of order, rather than the guard misfiring"
    );
}

// ---------------------------------------------------------------------------
// The gaps the pull request published, closed where a constructed input can
// reach the writer at all.
// ---------------------------------------------------------------------------

/// [`constructed_element`] with every `element.arrow` column the writer can
/// populate carrying its own distinct values. NOT A PARSED EVALUATION.
///
/// Three changes to the shared builder, each of them because two equal columns
/// cannot discriminate a swap between them:
///
/// * MT 517 and MT 515 are added, so the two pair-production columns are no
///   longer both empty. Their abscissae are already union grid points, so the
///   grid stays `[1, 5, 10, 20, 50, 100]`, and their ordinates do not start at
///   zero, unlike a real evaluation's, so `eval`'s clamp below the threshold is
///   visible on them too.
/// * The coherent form factor becomes the line from `(0, 10)` to `(3, 0)`.
///   Squared and divided by `Z^2 = 100` that is the line from `(0, 1)` to
///   `(9, 0)`, whose cumulative integral is `[0, 4.5]`. The shared builder's
///   `[0, 2]` is the same pair as its own `coherent_ff_x`.
/// * The incoherent form factor gets the abscissa `[0, 7]`, which the shared
///   builder shares with `coherent_anomalous_imag_x`.
fn constructed_element_fully_populated() -> IncidentPhoton {
    let mut data = constructed_element();

    for (mt, x, y) in [
        (517, [20.0, 100.0], [41.0, 43.0]),
        (515, [50.0, 100.0], [47.0, 53.0]),
    ] {
        let mut rx = PhotonReaction::new(mt);
        rx.xs = Some(Tabulated1D::new(x.to_vec(), y.to_vec()));
        data.reactions.insert(mt, rx);
    }

    data.reactions
        .get_mut(&502)
        .expect("MT 502")
        .scattering_factor = Some(Tabulated1D::new(vec![0.0, 3.0], vec![10.0, 0.0]));
    data.reactions
        .get_mut(&504)
        .expect("MT 504")
        .scattering_factor = Some(Tabulated1D::new(vec![0.0, 7.0], vec![0.0, 10.0]));

    data
}

/// A failure means two `element.arrow` columns were exchanged.
///
/// CONSTRUCTED INPUT, not an evaluation. `element.arrow` declares sixteen
/// consecutive `list<double>` fields and a permutation among any two that are
/// EMPTY on the input at hand writes a byte-identical file. On
/// [`constructed_element`] two of them are empty at once
/// (`pair_production_nuclear_xs` and `pair_production_electron_xs`), so
/// exchanging those two slots in `write_element`'s argument vector leaves all
/// eight constructed tests green.
///
/// Here all sixteen carry values that are pairwise different. The pairwise
/// comparison at the end of this test is the property that says so: no two of
/// the sixteen columns are equal, therefore every permutation of them changes
/// the file. Retiring the unfillable `heating_xs` strengthened this: it was
/// the one column that could never carry a value, so it was the one slot a
/// permutation could hide in.
#[test]
fn constructed_element_columns_are_pairwise_distinct_so_no_permutation_is_invisible() {
    /// The sixteen list columns, in the order `write_element` passes them.
    const LIST_COLUMNS: [&str; 16] = [
        "ln_energy",
        "coherent_xs",
        "incoherent_xs",
        "photoelectric_xs",
        "pair_production_nuclear_xs",
        "pair_production_electron_xs",
        "coherent_int_ff_x",
        "coherent_int_ff_y",
        "coherent_ff_x",
        "coherent_ff_y",
        "coherent_anomalous_real_x",
        "coherent_anomalous_real_y",
        "coherent_anomalous_imag_x",
        "coherent_anomalous_imag_y",
        "incoherent_ff_x",
        "incoherent_ff_y",
    ];

    let data = constructed_element_fully_populated();
    let (_scratch, dir) = write_constructed(&data);
    let batch = section(&dir, "element.arrow");
    assert_schema_is_declared(&batch, "element.arrow");

    // Those sixteen are every column but the two scalars, in that order, so
    // the pairwise property below cannot miss one that was added later.
    let names: Vec<String> = batch
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().to_string())
        .collect();
    assert_eq!(names.len(), 2 + LIST_COLUMNS.len());
    assert_eq!(names[0], "name");
    assert_eq!(names[1], "Z");
    for (i, expected) in LIST_COLUMNS.iter().enumerate() {
        assert_eq!(
            names[2 + i],
            *expected,
            "column {} is not the one write_element writes there",
            2 + i
        );
    }

    // The grid is the shared builder's, unchanged: both added channels are
    // tabulated on points it already carries.
    let grid = union_grid_expected(&data);
    assert_f64_slice_eq("the constructed union grid", &grid, &CONSTRUCTED_GRID);
    let ln_energy = f64_list(&batch, "ln_energy", 0);
    let expected_ln: Vec<f64> = grid.iter().map(|e| e.ln()).collect();
    assert_f64_slice_eq("ln_energy", &ln_energy, &expected_ln);
    assert_eq!(ln_energy[0], 0.0, "ln(1 eV) is exactly zero");

    // The five channels ELEMENT_MTS lets through, each against the reaction it
    // names. A restatement of the writer's own `eval`, so this pins WIRING,
    // which is what a permutation is.
    for (col, mt) in [
        ("coherent_xs", 502),
        ("incoherent_xs", 504),
        ("photoelectric_xs", 522),
        ("pair_production_nuclear_xs", 517),
        ("pair_production_electron_xs", 515),
    ] {
        let written = f64_list(&batch, col, 0);
        assert_f64_slice_eq(col, &written, &eval_on(&data, mt, &grid));
        assert_eq!(written.len(), grid.len(), "{col} is on the union grid");
    }

    // The two pair-production channels in closed form. Their first three
    // points are below the tabulated range, where `eval` clamps to y[0]: 41
    // and 47, not zero. The evaluation's own pair-production curves start at
    // zero, so on hydrogen that clamp cannot be told from a zero fill.
    assert_f64_slice_eq(
        "pair_production_nuclear_xs",
        &f64_list(&batch, "pair_production_nuclear_xs", 0),
        &[41.0, 41.0, 41.0, 41.0, 41.75, 43.0],
    );
    assert_f64_slice_eq(
        "pair_production_electron_xs",
        &f64_list(&batch, "pair_production_electron_xs", 0),
        &[47.0, 47.0, 47.0, 47.0, 47.0, 53.0],
    );
    // The other three at the bottom of the grid, where the three differ.
    assert_eq!(f64_list(&batch, "coherent_xs", 0)[0], 2.0);
    assert_eq!(f64_list(&batch, "incoherent_xs", 0)[0], 7.0);
    assert_eq!(f64_list(&batch, "photoelectric_xs", 0)[0], 17.0);

    // The ten copied columns, verbatim.
    for (col, expected) in [
        ("coherent_int_ff_x", vec![0.0, 9.0]),
        ("coherent_int_ff_y", vec![0.0, 4.5]),
        ("coherent_ff_x", vec![0.0, 3.0]),
        ("coherent_ff_y", vec![10.0, 0.0]),
        ("coherent_anomalous_real_x", vec![1.0, 2.0]),
        ("coherent_anomalous_real_y", vec![-1.5, -0.5]),
        ("coherent_anomalous_imag_x", vec![1.0, 3.0]),
        ("coherent_anomalous_imag_y", vec![0.25, 0.75]),
        ("incoherent_ff_x", vec![0.0, 7.0]),
        ("incoherent_ff_y", vec![0.0, 10.0]),
    ] {
        assert_f64_slice_eq(col, &f64_list(&batch, col, 0), &expected);
    }

    // All sixteen populated. There is no longer a column that cannot carry a
    // value, so there is no slot a permutation can hide in.
    let written: Vec<Vec<f64>> = LIST_COLUMNS
        .iter()
        .map(|&c| f64_list(&batch, c, 0))
        .collect();
    for (col, values) in LIST_COLUMNS.iter().zip(&written) {
        assert!(!values.is_empty(), "{col} is empty, so a swap with it is invisible");
    }

    // The property this element exists for.
    for i in 0..LIST_COLUMNS.len() {
        for j in (i + 1)..LIST_COLUMNS.len() {
            assert_ne!(
                written[i], written[j],
                "{} and {} carry the same values, so exchanging the two slots \
                 would write an identical file",
                LIST_COLUMNS[i], LIST_COLUMNS[j]
            );
        }
    }
}

/// A failure means `compton.arrow` stopped taking its momentum grid from the
/// first shell.
///
/// CONSTRUCTED INPUT, and a deliberately malformed one. No parser can produce
/// a ragged `ComptonProfiles`: `IncidentPhoton::add_photon_data`
/// (`crates/endf/src/incident_photon.rs:286-300`) builds every row as
/// `Tabulated1D::new(data.pz.clone(), row)`, one shared abscissa, so on every
/// input the writer can be handed through a parser `first.x` and `last.x` are
/// the same values and `photon.rs:281` could read either. Nothing here says an
/// evaluation can be ragged; it says what the writer does when it is handed
/// one, which is to use shell 0's abscissa for every shell and say nothing.
///
/// PINS TWO SUSPECTED DEFECTS, neither of them endorsed. `write_compton`
/// validates nothing about the shape of `profiles.j`, so when the rows differ
/// in length it writes a `J_shape` that describes the first row alone while
/// `J_data` concatenates all of them, and `compton_profile_cdfs`
/// (`crates/endf/src/incident_photon.rs:551-564`) stops each row at
/// `pz.len()`, leaving the tail of a longer row at zero and the cumulative
/// distribution decreasing. A writer that refused a ragged input, or that
/// wrote the true row length, would turn this test red; see the assertions
/// below before changing them.
#[test]
fn constructed_ragged_compton_profiles_are_written_on_the_first_shell_abscissa() {
    let mut data = constructed_element();
    data.compton_profiles = Some(ComptonProfiles {
        num_electrons: vec![2.0, 8.0],
        binding_energy: vec![51.0, 6.0],
        j: vec![
            Tabulated1D::new(vec![0.0, 1.0, 2.0], vec![0.9, 0.5, 0.1]),
            // A different abscissa, of a different length, which is what no
            // parser produces.
            Tabulated1D::new(vec![0.0, 4.0, 8.0, 10.0], vec![0.8, 0.4, 0.2, 0.05]),
        ],
    });

    let profiles = data.compton_profiles.as_ref().expect("just set");
    assert_ne!(
        profiles.j[0].x, profiles.j[1].x,
        "the two shells have to disagree about the momentum grid for this test \
         to discriminate between them"
    );

    let (_scratch, dir) = write_constructed(&data);
    let batch = section(&dir, "compton.arrow");
    assert_schema_is_declared(&batch, "compton.arrow");

    // Shell 0's abscissa, not shell 1's.
    let pz = f64_list(&batch, "pz", 0);
    assert_f64_slice_eq("compton.pz", &pz, &[0.0, 1.0, 2.0]);
    assert_f64_slice_eq("compton.pz", &pz, &profiles.j[0].x);

    // Every ordinate is still written, so the flat array is longer than the
    // shape says. DEFECT: `[2, 3]` describes six values and there are seven.
    let j_data = f64_list(&batch, "J_data", 0);
    assert_f64_slice_eq("J_data", &j_data, &[0.9, 0.5, 0.1, 0.8, 0.4, 0.2, 0.05]);
    let j_shape = i32_list(&batch, "J_shape", 0);
    assert_i32_slice_eq("J_shape", &j_shape, &[2, 3]);
    assert_ne!(
        (j_shape[0] * j_shape[1]) as usize,
        j_data.len(),
        "the shape and the data agreeing would mean the writer had learned to \
         refuse or to describe a ragged profile set; read this test's doc \
         comment before deleting the assertion"
    );

    // Shell 1's own abscissa is never used: its cumulative distribution is
    // integrated against shell 0's spacing. With its own [0, 4, 8, 10] the
    // first step would be 0.5 * (0.8 + 0.4) * 4 = 2.4, not 0.6. DEFECT: the
    // trailing 0.0 is the tail `compton_profile_cdfs` never reaches, so the
    // row falls from 0.9 back to zero.
    let cdf = f64_list(&batch, "J_cdf_data", 0);
    assert_f64_slice_close(
        "J_cdf_data",
        &cdf,
        &[0.0, 0.7, 1.0, 0.0, 0.6, 0.9, 0.0],
        1e-15,
    );
    assert!(
        cdf[6] < cdf[5],
        "the second row's tail is what makes this a defect rather than a \
         curiosity: a cumulative distribution that decreases"
    );
    assert_i32_slice_eq("J_cdf_shape", &i32_list(&batch, "J_cdf_shape", 0), &[2, 3]);
}

/// A failure means a `bremsstrahlung.arrow` column was sourced from the wrong
/// place.
///
/// CONSTRUCTED INPUT, not an evaluation. On hydrogen `ionization_energy` is
/// `[13.6]`, which is also the MF=28 K-shell binding energy and also the MF=23
/// MT=522 threshold, so a column wired to the relaxation data instead of to
/// the Seltzer-Berger tabulation writes identical bytes and no assertion
/// against the vendored fixture can tell the two apart. The element built here
/// has three subshells bound at 50, 20 and 5 eV and two ionization energies of
/// 21.5 and 7.25 eV, so every candidate source is a different number.
///
/// The DCS is 3 by 2 rather than 200 by 30, which is enough on its own to tell
/// a row-major ravel from a column-major one and a `[rows, cols]` shape from
/// its transpose. What it does not do is check the not-a-knot spline that
/// produces the real one: those numbers are the parser's and verifying them
/// needs a second spline implementation, which is a test of `crates/endf` and
/// not of this writer.
#[test]
fn constructed_bremsstrahlung_columns_come_from_the_attached_tabulation() {
    let mut data = constructed_element();
    data.bremsstrahlung = Some(Bremsstrahlung {
        i: 42.5,
        num_electrons: vec![4.0, 6.0],
        ionization_energy: vec![21.5, 7.25],
        electron_energy: vec![1000.0, 2000.0, 4000.0],
        photon_energy: vec![0.125, 0.5],
        dcs: vec![vec![1.5, 2.5], vec![3.5, 4.5], vec![5.5, 6.5]],
    });

    let (_scratch, dir) = write_constructed(&data);
    let batch = section(&dir, "bremsstrahlung.arrow");
    assert_schema_is_declared(&batch, "bremsstrahlung.arrow");
    assert_eq!(batch.num_rows(), 1);

    assert_eq!(f64_at(&batch, "I", 0), 42.5);
    assert_f64_slice_eq(
        "electron_energy",
        &f64_list(&batch, "electron_energy", 0),
        &[1000.0, 2000.0, 4000.0],
    );
    assert_f64_slice_eq(
        "photon_energy",
        &f64_list(&batch, "photon_energy", 0),
        &[0.125, 0.5],
    );
    assert_f64_slice_eq(
        "bremsstrahlung.num_electrons",
        &f64_list(&batch, "num_electrons", 0),
        &[4.0, 6.0],
    );

    let ionization = f64_list(&batch, "ionization_energy", 0);
    assert_f64_slice_eq("ionization_energy", &ionization, &[21.5, 7.25]);

    // None of the other places the column could have come from. The subshell
    // binding energies are read out of the file this same conversion wrote,
    // so this is a comparison between two written sections and not between two
    // expressions in this test.
    let subshells = section(&dir, "subshells.arrow");
    let bound: Vec<f64> = (0..subshells.num_rows())
        .map(|r| f64_at(&subshells, "binding_energy", r))
        .collect();
    assert_f64_slice_eq(
        "the relaxation binding energies",
        &bound,
        &[50.0, 20.0, 5.0],
    );
    for &v in &ionization {
        assert!(
            !bound.contains(&v),
            "{v} is a relaxation binding energy as well, so this element cannot \
             tell the two sources apart"
        );
    }
    let compton_bound = f64_list(&section(&dir, "compton.arrow"), "binding_energy", 0);
    assert_f64_slice_eq("the Compton binding energies", &compton_bound, &[51.0, 6.0]);
    for &v in &ionization {
        assert!(!compton_bound.contains(&v));
    }
    assert_ne!(ionization, f64_list(&batch, "num_electrons", 0));
    assert!(!ionization.contains(&f64_at(&batch, "I", 0)));

    // Row-major, one row per electron energy.
    let dcs = f64_list(&batch, "dcs_data", 0);
    assert_f64_slice_eq("dcs_data", &dcs, &[1.5, 2.5, 3.5, 4.5, 5.5, 6.5]);
    let shape = i32_list(&batch, "dcs_shape", 0);
    assert_i32_slice_eq("dcs_shape", &shape, &[3, 2]);
    assert_eq!(shape[0] as usize, list_len(&batch, "electron_energy", 0));
    assert_eq!(shape[1] as usize, list_len(&batch, "photon_energy", 0));
    assert_eq!((shape[0] * shape[1]) as usize, dcs.len());
}
