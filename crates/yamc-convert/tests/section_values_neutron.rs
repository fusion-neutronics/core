//! `nuclide.arrow`, `urr.arrow` and `reactions.arrow`, column by column,
//! against the `endf::IncidentNeutron` the writer was handed.
//!
//! The writers are called directly rather than through
//! `entry::convert_neutron_xs`, so these tests hold the identical
//! `IncidentNeutron` value the writer saw and compare against it rather than
//! against a second parse of the same bytes.
//!
//! Two routes are exercised, because they reach different branches. The ACE
//! route (`Li6.ace.xz`) carries one temperature, a real 721 point grid and
//! twelve threshold reactions. The ENDF route
//! (`n-003_Li_006_trimmed.endf.xz` through `IncidentNeutron::from_endf`) sets
//! no `k_ts`, no `atomic_weight_ratio` and no nuclide grid, and keys every
//! cross section at `"0K"`. The unresolved resonance tables come from
//! `synthetic-urr.ace.xz`, the only vendored input that reaches
//! `write_urr`'s write branch at all.
//!
//! Six of the fourteen tests are built from a HAND-CONSTRUCTED
//! `IncidentNeutron` rather than from a parsed evaluation, and their names all
//! begin with `constructed_`. They exist because of what the vendored bytes
//! cannot say:
//!
//! * every vendored neutron evaluation carries at most ONE temperature, so
//!   `temperatures`, `energy_temperatures` and `xs_temperatures` are one
//!   element lists with no permutation to catch;
//! * the single one that carries unresolved resonance tables carries a single
//!   row whose scalar columns are all equal to each other or to a constant;
//! * Li6 carries its OWN MT 1 and MT 101, and the ENDF route synthesizes
//!   nothing at all, so no vendored route writes a synthesized MT 1 or MT 101
//!   with a number in it;
//! * no vendored reaction has more than one cross section at a temperature
//!   outside `temperatures()`, so the extras phase has no order to get wrong.
//!
//! A writer that swapped two of those columns, that ignored a parsed field and
//! wrote a literal in its place, or that dropped a term from a sum whose other
//! terms Li6 happens not to carry, passes every test built from vendored bytes.
//! The constructed inputs give those fields deliberately DIFFERENT values,
//! which is the only thing that separates them. They are not evaluation parity
//! and nothing in them should be read as such.

mod section_values;

use std::collections::{BTreeMap, BTreeSet};

use arrow_array::RecordBatch;
use endf::{IncidentNeutron, ProbabilityTables, Reaction, Tabulated1D};

use section_values::*;

// ---------------------------------------------------------------------------
// Local helpers. Nothing here is shared: the sibling files have their own. The
// only one that asserts is `written_on_grid`, and it asserts only to turn a
// slice range panic into a sentence.
// ---------------------------------------------------------------------------

/// The `energy_temperatures` rule of `nuclide.rs:41-51`, restated.
///
/// `data.temperatures()` filtered to the keys `data.energy` holds, in that
/// order, then whatever else the map holds in its own (BTreeMap, so
/// lexicographic) order. This is a restatement of the writer rather than an
/// independent derivation, so on its own it is a wiring check: it pins which
/// list reached which column, not that the ordering rule is the right one. On
/// the single temperature Li6 fixture the two phases cannot be told apart at
/// all, which is why
/// `constructed_two_temperature_nuclide_separates_the_two_temperature_columns`
/// spells the expected list out as literals beside this.
fn expected_energy_temperatures(data: &IncidentNeutron) -> Vec<String> {
    let temperatures = data.temperatures();
    let mut out: Vec<String> = temperatures
        .iter()
        .filter(|t| data.energy.contains_key(*t))
        .cloned()
        .collect();
    out.extend(
        data.energy
            .keys()
            .filter(|t| !temperatures.contains(t))
            .cloned(),
    );
    out
}

/// The same two phase rule for one reaction's cross sections
/// (`reactions.rs:82-87`), which reads `rx.xs` rather than `data.energy`.
///
/// A restatement of the writer, with the same caveat: it pins which list
/// reached which column. The order it encodes is discriminated only in
/// `constructed_two_temperature_reactions_keep_each_curve_with_its_own_temperature`,
/// which is the one case in this file where the processed order and the map's
/// own order differ, and which spells the expected list out as a literal
/// rather than calling this.
fn expected_xs_temperatures(data: &IncidentNeutron, rx: &Reaction) -> Vec<String> {
    let temperatures = data.temperatures();
    let mut out: Vec<String> = temperatures
        .iter()
        .filter(|t| rx.xs.contains_key(*t))
        .cloned()
        .collect();
    out.extend(rx.xs.keys().filter(|t| !temperatures.contains(t)).cloned());
    out
}

/// Where each MT sits in a `reactions.arrow` read back as one row space.
fn rows_by_mt(batch: &RecordBatch) -> BTreeMap<i32, usize> {
    (0..batch.num_rows())
        .map(|row| (i32_at(batch, "mt", row), row))
        .collect()
}

/// The MT column, in file order.
fn written_mts(batch: &RecordBatch) -> Vec<i32> {
    (0..batch.num_rows())
        .map(|row| i32_at(batch, "mt", row))
        .collect()
}

/// One written row's first (and here only) cross section, laid onto an
/// `n_energy` long zero vector at the threshold index the same row carries.
///
/// Deliberately not `synthesis::on_grid`: that is the function the writer used
/// to build the synthesized rows, so reconstructing with it would restate the
/// writer instead of checking it. This works only from the file's own two
/// columns.
#[track_caller]
fn written_on_grid(
    batch: &RecordBatch,
    mt_rows: &BTreeMap<i32, usize>,
    mt: i32,
    n: usize,
) -> Vec<f64> {
    let row = mt_rows[&mt];
    let values = nested_f64_list(batch, "xs_values", row);
    let threshold = i32_list(batch, "xs_threshold_idx", row)[0] as usize;
    // Named, because without it the line below panics with "range end index
    // 722 out of range for slice of length 721", which names neither the MT
    // nor the column that is wrong. A threshold written one too high is a
    // defect this file is looking for, so it deserves a sentence.
    assert!(
        threshold + values[0].len() <= n,
        "MT {mt}: threshold {threshold} plus {} written values runs past the {n} point grid",
        values[0].len()
    );
    let mut out = vec![0.0; n];
    out[threshold..threshold + values[0].len()].copy_from_slice(&values[0]);
    out
}

/// `round(x, 6)` the way the URR fixture's generator writes it.
fn round6(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}

/// The index of the first value that is not zero.
fn first_nonzero(values: &[f64]) -> Option<usize> {
    values.iter().position(|&v| v != 0.0)
}

// ---------------------------------------------------------------------------
// nuclide.arrow
// ---------------------------------------------------------------------------

/// A failure means nuclide.arrow no longer describes the table it was written
/// from: the identity the loader keys everything else off, the kT list nothing
/// in this workspace reads back, or the energy grid every binary search
/// downstream runs on. `nuclide_section_matches_the_parsed_table` checks name,
/// Z and A against literals and then only that the grid is non-decreasing, so
/// a grid of 721 copies of one value, or a grid off by the MeV to eV factor,
/// passes it today.
///
/// What this fixture cannot establish, said here so nobody stops looking:
/// every list on it has exactly ONE element, so `temperatures` and
/// `energy_temperatures` both hold `["294K"]` and swapping the two
/// `string_list` arguments at nuclide.rs:65 and :67 passes everything below.
/// A one element list has no permutation either, so the kT to temperature
/// round trip cannot see one. Both are separated in
/// `constructed_two_temperature_nuclide_separates_the_two_temperature_columns`.
/// What is independent of the writer HERE is `energy_values` against the raw
/// ESZ block of the ACE table, and the AWR against the ACE header.
#[test]
fn nuclide_columns_are_the_parsed_tables_own_numbers() {
    let data = li6_ace();
    let table = ace_table(LI6_ACE);
    let dir = scratch();
    yamc_convert::nuclide::write_nuclide(&data, dir.path()).expect("nuclide.arrow is written");
    let batch = section(dir.path(), "nuclide.arrow");

    assert_eq!(batch.num_rows(), 1, "nuclide.arrow is one row per nuclide");

    // Against the parsed value AND against the literal. Comparing only against
    // data.name() passes when both sides are wrong the same way.
    assert_eq!(str_at(&batch, "name", 0), data.name());
    assert_eq!(str_at(&batch, "name", 0), "Li6");
    assert_eq!(i32_at(&batch, "Z", 0), data.atomic_number as i32);
    assert_eq!(i32_at(&batch, "Z", 0), 3);
    assert_eq!(i32_at(&batch, "A", 0), data.mass_number as i32);
    assert_eq!(i32_at(&batch, "A", 0), 6);

    // Exact: the AWR is copied from the ACE header with no arithmetic on the
    // way, so anything but bit equality is a defect rather than rounding.
    let awr = f64_at(&batch, "atomic_weight_ratio", 0);
    assert_eq!(
        awr,
        data.atomic_weight_ratio
            .expect("the ACE route sets an atomic weight ratio")
    );
    assert_eq!(awr, table.atomic_weight_ratio);
    assert_eq!(awr, 5.96345);

    let temperatures = str_list(&batch, "temperatures", 0);
    assert_eq!(temperatures, data.temperatures());
    assert_eq!(temperatures, vec!["294K".to_string()]);

    // One deterministic multiply on each side, so exact is right here too.
    let k_ts = f64_list(&batch, "kTs", 0);
    assert_f64_slice_eq("kTs", &k_ts, &data.k_ts);
    assert_f64_slice_eq("kTs", &k_ts, &[table.kt * endf::EV_PER_MEV]);
    // Through temperature_str rather than a reimplementation of its banker's
    // rounding (crates/endf/src/data.rs:321-328). Nothing in this workspace
    // reads kTs back, so nothing downstream would ever notice a kT that named
    // the wrong temperature. On one element this only says the single pair
    // agrees; the constructed test is where a permutation becomes possible.
    for (kt, t) in k_ts.iter().zip(&temperatures) {
        assert_eq!(
            &endf::data::temperature_str(kt / endf::K_BOLTZMANN),
            t,
            "kT {kt} does not name temperature {t}"
        );
    }

    let energy_temperatures = str_list(&batch, "energy_temperatures", 0);
    assert_eq!(energy_temperatures, expected_energy_temperatures(&data));
    assert_eq!(energy_temperatures, vec!["294K".to_string()]);

    let grids = nested_f64_list(&batch, "energy_values", 0);
    assert_eq!(
        grids.len(),
        energy_temperatures.len(),
        "one grid per energy temperature"
    );
    for (grid, t) in grids.iter().zip(&energy_temperatures) {
        assert_f64_slice_eq(&format!("energy_values[{t}]"), grid, &data.energy[t]);
        assert_eq!(grid.len(), 721, "the Li6 ACE grid is 721 points");
        // Strictly, not the >= the existing test allows: a duplicated point
        // passes that and breaks every binary search built on the grid.
        assert!(
            grid.windows(2).all(|w| w[1] > w[0]),
            "the grid at {t} is not strictly increasing"
        );
    }

    // The one derivation that does not go through the writer at all: the ESZ
    // block of the raw ACE table, in MeV, times EV_PER_MEV.
    let base = usize::try_from(table.jxs[1]).expect("JXS(1) locates the ESZ block");
    let n_energy = usize::try_from(table.nxs[3]).expect("NXS(3) is the grid length");
    let from_xss: Vec<f64> = (0..n_energy)
        .map(|k| table.xss[base + k] * endf::EV_PER_MEV)
        .collect();
    assert_f64_slice_eq(
        "energy_values against the raw ESZ block",
        &grids[0],
        &from_xss,
    );
    assert_eq!(grids[0][0], 9.999999999999999e-6);
    assert_eq!(grids[0][720], 2e8);
}

/// A failure means the ENDF branch of `write_nuclide` started inventing a
/// temperature or an energy grid the evaluation does not carry. It is the only
/// hermetic case that reaches the `atomic_weight_ratio` fallback, because
/// `IncidentNeutron::from_endf` never sets `k_ts`, never sets
/// `atomic_weight_ratio` and leaves `data.energy` empty
/// (crates/endf/src/incident_neutron.rs:41-43, :106-119).
///
/// This doc comment used to claim the test was also "the only hermetic case
/// that separates temperatures from energy_temperatures". It separates
/// nothing: on this route BOTH lists are empty, so swapping the two
/// `string_list` arguments at nuclide.rs:65 and :67 leaves every assertion
/// here true. That separation is done by
/// `constructed_two_temperature_nuclide_separates_the_two_temperature_columns`.
/// What the emptiness assertions below do carry is the other direction: a
/// writer that invented a temperature, a kT or a grid on this route fails
/// them, and empty against null is a real difference in the written file.
#[test]
fn the_endf_route_writes_no_temperature_and_no_grid_it_was_not_given() {
    let data = endf_nuclide(LI6_ENDF);
    let dir = scratch();
    yamc_convert::nuclide::write_nuclide(&data, dir.path()).expect("nuclide.arrow is written");
    let batch = section(dir.path(), "nuclide.arrow");

    assert_eq!(batch.num_rows(), 1);
    assert_eq!(str_at(&batch, "name", 0), "Li6");
    assert_eq!(i32_at(&batch, "Z", 0), 3);
    assert_eq!(i32_at(&batch, "A", 0), 6);

    // Pinning the fallback at nuclide.rs:64, so a later change that starts
    // writing a real AWR on this route is a deliberate edit rather than a
    // surprise. `from_endf` leaves the field None even though the evaluation's
    // MF=1 MT=451 does carry AWR = 5.961817.
    assert_eq!(data.atomic_weight_ratio, None);
    assert_eq!(f64_at(&batch, "atomic_weight_ratio", 0), 0.0);

    // Empty and PRESENT, not null: string_list and float_list_list at
    // sections.rs:101-135 always append a row, so an absent temperature is an
    // empty list here and never a null.
    for column in [
        "temperatures",
        "energy_temperatures",
        "energy_values",
        "kTs",
    ] {
        assert!(!is_null(&batch, column, 0), "{column} was written as null");
    }
    assert!(str_list(&batch, "temperatures", 0).is_empty());
    assert!(f64_list(&batch, "kTs", 0).is_empty());
    assert!(str_list(&batch, "energy_temperatures", 0).is_empty());
    assert_eq!(list_len(&batch, "energy_values", 0), 0);
}

// ---------------------------------------------------------------------------
// urr.arrow
// ---------------------------------------------------------------------------

/// A failure means urr.arrow disagrees with `endf::ProbabilityTables` on the
/// one vendored input that reaches `write_urr` at all. No test in this
/// repository has ever produced a urr.arrow: the only existing one asserts the
/// file is ABSENT for Li6 (sections.rs:136-146), so all eight columns and the
/// `(n_energy, 6, n_band)` ravel are otherwise unexercised.
///
/// What this fixture cannot establish, because it is ONE row on which every
/// scalar column is a constant: `interpolation` is 2, `table_shape` is
/// `[4, 6, 3]`, `multiply_smooth` is true and `inelastic` and `absorption` are
/// BOTH -1. A writer that pushed any of those literals instead of reading the
/// parsed field, or that swapped the `inelastic` and `absorption` arguments at
/// nuclide.rs:126-127, passes everything below. `multiply_smooth` is the worst
/// of the four to be blind to, since it flips the physical meaning of every
/// number in `table_data` from factors on the smooth background to absolute
/// cross sections, and it is the one this test least deserves credit for. All
/// four are separated in
/// `constructed_urr_rows_carry_their_own_scalars_in_processed_order`.
///
/// What this test does carry on its own: `table_data` against the generator's
/// closed form, and the position of `interpolation` in the argument list,
/// since its neighbour `inelastic` holds a different number.
///
/// Two things this deliberately does NOT assert, because the fixture's values
/// are invented rather than physical: that the q=0 cumulative row ends at 1.0
/// (it ends at 1.02) and that the total row exceeds the elastic row (it does
/// not).
#[test]
fn urr_columns_are_the_parsed_probability_tables_in_c_order() {
    let data = ace_nuclide(URR_ACE);
    let dir = scratch();
    let wrote = yamc_convert::nuclide::write_urr(&data, dir.path()).expect("no error");
    assert!(wrote, "synthetic-urr carries unresolved resonance tables");
    let batch = section(dir.path(), "urr.arrow");

    assert_eq!(batch.num_rows(), data.urr.len());
    assert_eq!(batch.num_rows(), 1);

    // The row order rule of nuclide.rs:96-105, which is the same shape as the
    // energy_temperatures rule and equally undiscriminated by a single
    // temperature fixture.
    let mut ordered: Vec<String> = data
        .temperatures()
        .into_iter()
        .filter(|t| data.urr.contains_key(t))
        .collect();
    ordered.extend(
        data.urr
            .keys()
            .filter(|t| !data.temperatures().contains(t))
            .cloned(),
    );
    let written: Vec<String> = (0..batch.num_rows())
        .map(|row| str_at(&batch, "temperature", row))
        .collect();
    assert_eq!(written, ordered);
    assert_eq!(written, vec!["294K".to_string()]);

    let tables = &data.urr[&written[0]];

    let energy = f64_list(&batch, "energy", 0);
    assert_f64_slice_eq("energy", &energy, &tables.energy);
    assert_f64_slice_eq("energy", &energy, &[1000.0, 2000.0, 3000.0, 4000.0]);
    assert!(
        energy.windows(2).all(|w| w[1] > w[0]),
        "the unresolved energies are not strictly increasing"
    );

    // The loader defaults n_params to 6 when this list is short
    // (nuclide_arrow.rs:1862-1866), so a dropped axis is absorbed silently.
    // The three shape identities that used to follow (shape[1] == 6,
    // shape[0] == energy.len(), and the flat length as the product) all follow
    // from the two literals pinned here and above, so they are gone.
    let shape = i32_list(&batch, "table_shape", 0);
    assert_i32_slice_eq("table_shape", &shape, &[4, 6, 3]);
    assert_eq!(
        shape,
        tables.shape.iter().map(|&n| n as i32).collect::<Vec<i32>>()
    );

    let flat = f64_list(&batch, "table_data", 0);
    assert_eq!(flat.len(), 72);

    let (n_e, n_b) = (shape[0] as usize, shape[2] as usize);

    // The C order identity, exact. Read this for what it is:
    // `ProbabilityTables::get` computes this same index expression over the
    // same Vec the writer clones (crates/endf/src/urr.rs:62-70), so it checks
    // that the copy was not permuted and is NOT independent evidence that the
    // ravel is C order. The generator oracle below carries that, and so does
    // the constructed test, which fills its table from an (e, q, b) formula.
    // The index is also the one the loader uses at nuclide_arrow.rs:1884-1905,
    // so a ravel that fails here loads as a different set of bands.
    for e in 0..n_e {
        for q in 0..6 {
            for b in 0..n_b {
                assert_eq!(
                    flat[(e * 6 + q) * n_b + b],
                    tables.get(e, q, b).expect("in range"),
                    "table_data[{e},{q},{b}]"
                );
            }
        }
    }

    // The independent oracle, and the reason this fixture is worth using:
    // tools/make_urr_ace.py writes round(1.0 + e + 0.1*q + 0.01*b, 6) with the
    // q == 5 heating row scaled by EV_PER_MEV. A relative tolerance is right
    // HERE and only here, because the oracle is a decimal literal
    // reconstructed in binary (1.0 + 0.01*2 need not be the f64 nearest 1.02)
    // and not because the converter did any arithmetic. Measured worst on this
    // fixture is 0.0; the tolerance is headroom for the reconstruction, not
    // slack for the writer, and the comparison against tables.get above stays
    // exact.
    for e in 0..n_e {
        for q in 0..6 {
            for b in 0..n_b {
                let mut want = round6(1.0 + e as f64 + 0.1 * q as f64 + 0.01 * b as f64);
                if q == 5 {
                    want *= endf::EV_PER_MEV;
                }
                assert_close(
                    &format!("table_data[{e},{q},{b}] against the generator"),
                    flat[(e * 6 + q) * n_b + b],
                    want,
                    1e-12,
                );
            }
        }
    }

    // An axis probe that does not reuse the ravel arithmetic above: only the
    // heating row is scaled by EV_PER_MEV, so it stands about 1.5e6 against
    // O(1) everywhere else. A transposed ravel moves the large values off that
    // stride and this fires.
    for e in 0..n_e {
        for b in 0..n_b {
            let heating = flat[(e * 6 + 5) * n_b + b];
            for q in 0..5 {
                let other = flat[(e * 6 + q) * n_b + b];
                assert!(
                    heating >= 1e3 * other,
                    "heating at ({e},{b}) is {heating:e}, not above quantity {q} at {other:e}"
                );
            }
        }
    }

    // The loader maps 5 to log-log and EVERYTHING else to lin-lin
    // (nuclide_arrow.rs:1849-1852), so a corrupted value degrades silently.
    assert_eq!(
        i32_at(&batch, "interpolation", 0),
        tables.interpolation as i32
    );
    assert_eq!(i32_at(&batch, "interpolation", 0), 2);

    assert_eq!(i32_at(&batch, "inelastic", 0), -1);
    assert_eq!(i32_at(&batch, "absorption", 0), -1);
    // Also as i64, so the i64 to i32 narrowing at nuclide.rs:112-113 is covered.
    assert_eq!(
        i64::from(i32_at(&batch, "inelastic", 0)),
        tables.inelastic_flag
    );
    assert_eq!(
        i64::from(i32_at(&batch, "absorption", 0)),
        tables.absorption_flag
    );

    assert!(bool_at(&batch, "multiply_smooth", 0));
    assert_eq!(
        bool_at(&batch, "multiply_smooth", 0),
        tables.multiply_smooth
    );
}

// ---------------------------------------------------------------------------
// reactions.arrow
// ---------------------------------------------------------------------------

/// A failure means reactions.arrow gained, lost or reordered a reaction, or
/// gave one a label that is not the format's own name for it. The existing
/// hermetic test only checks that each label resolves back through
/// `REACTION_MT`, which accepts any alias: writing "capture" for MT 102
/// resolves to 102 and passes, and its completeness guard is `checked >= 19`
/// on a 21 row file.
#[test]
fn reaction_rows_are_every_parsed_mt_then_the_synthesized_ones() {
    let data = li6_ace();
    let dir = scratch();
    yamc_convert::reactions::write_reactions(&data, dir.path())
        .expect("reactions.arrow is written");
    // Every batch, fused: reactions.arrow is one RecordBatch per row, so a
    // first-batch read would check MT 1 alone.
    let batch = section(dir.path(), "reactions.arrow");

    let mut synthetic: Vec<i32> = yamc_convert::synthesis::SYNTHETIC_MTS
        .iter()
        .copied()
        .filter(|mt| !data.reactions.contains_key(mt))
        .collect();
    synthetic.sort_unstable();
    let mut expected: Vec<i32> = data.reactions.keys().copied().collect();
    expected.extend(synthetic);

    let mts = written_mts(&batch);
    assert_eq!(mts, expected);
    // The literal order too, so the derivation above cannot go wrong the same
    // way the writer does. Note the tail: the synthesized rows are appended
    // AFTER the parsed ones, so the file is not globally ascending.
    assert_eq!(
        mts,
        vec![
            1, 2, 5, 51, 52, 53, 54, 55, 91, 101, 102, 203, 204, 205, 206, 207, 301, 444, 3, 4, 27
        ]
    );

    for (row, &mt) in mts.iter().enumerate() {
        let label = str_at(&batch, "label", row);
        assert_eq!(
            label,
            endf::reaction_name(mt).unwrap_or_else(|| mt.to_string()),
            "MT {mt} is not labelled with the format's own name for it"
        );
        // The round trip the loader does, kept as a second assertion because
        // it is what a consumer actually runs. It is the weaker of the two:
        // every alias resolves.
        assert_eq!(
            yamc_nuclide::data::REACTION_MT.get(label.as_str()).copied(),
            Some(mt),
            "label {label} does not resolve back to MT {mt}"
        );
    }
}

/// A failure means a reaction's Q value or one of its two boolean flags no
/// longer matches the parser. `center_of_mass` and `redundant` are adjacent
/// Boolean columns and `RecordBatch::try_new` validates position and type
/// only, so swapping the two `bools()` arguments at reactions.rs:101-102
/// writes a valid, loadable, wrong file; `redundant` also drives
/// `is_fission_mt(mt) && !redundant` in the loader at
/// nuclide_arrow.rs:503-505.
#[test]
fn reaction_flags_are_the_parsed_reactions_own_flags() {
    let data = li6_ace();
    let dir = scratch();
    yamc_convert::reactions::write_reactions(&data, dir.path())
        .expect("reactions.arrow is written");
    let batch = section(dir.path(), "reactions.arrow");
    let mt_rows = rows_by_mt(&batch);

    // Not a hand transcribed truth table: the two flags disagree on 14 of the
    // 18 parsed rows (MT 2 is center_of_mass true and redundant false, MT 102
    // is false and false, MT 203 to 207 are false and true), so comparing each
    // against the parsed side already fails on a swap, and a second
    // transcription would only test the transcription.
    let mut disagree = 0;
    for (&mt, rx) in &data.reactions {
        let row = mt_rows[&mt];
        assert_eq!(
            f64_at(&batch, "Q_value", row),
            rx.q_reaction,
            "MT {mt} Q value"
        );
        assert_eq!(
            bool_at(&batch, "center_of_mass", row),
            rx.center_of_mass,
            "MT {mt} center_of_mass"
        );
        assert_eq!(
            bool_at(&batch, "redundant", row),
            rx.redundant,
            "MT {mt} redundant"
        );
        if rx.center_of_mass != rx.redundant {
            disagree += 1;
        }
    }
    assert_eq!(
        disagree, 14,
        "the two boolean columns no longer disagree often enough for the \
         per-MT comparison to catch a swap between them"
    );

    // Two literal pins quoted from the fixture's MTR/LQR block, so a units
    // regression in the parser is visible without transcribing 21 numbers.
    assert_eq!(f64_at(&batch, "Q_value", mt_rows[&51]), -2.186e6);
    assert_eq!(f64_at(&batch, "Q_value", mt_rows[&102]), 7.251091e6);

    // A PIN, not a check. These three rows have no parsed side at all, so this
    // compares the writer's own literals at reactions.rs:130-132 against a
    // transcription of them and cannot tell you the literals are right. It
    // earns its place because `redundant = true` is what keeps a synthesized
    // row out of the sums (reactions.rs:43-45): changing the literal changes
    // the arithmetic of every row above it.
    //
    // One of the three is no longer only a pin:
    // `constructed_synthesized_rows_are_redundant_so_the_rest_sum_to_the_total`
    // checks `redundant` against the rule a consumer applies to it rather than
    // against a copy of the literal. `Q_value` and `center_of_mass` are still
    // transcriptions here and there, and a synthesized sum has no products, so
    // there is nothing downstream that reads either one to check them by.
    for mt in [3, 4, 27] {
        let row = mt_rows[&mt];
        assert_eq!(f64_at(&batch, "Q_value", row), 0.0, "MT {mt} Q value");
        assert!(!bool_at(&batch, "center_of_mass", row), "MT {mt}");
        assert!(bool_at(&batch, "redundant", row), "MT {mt}");
    }
}

/// A failure means the cross section a reaction row carries is not the curve
/// the parser produced, or the threshold index no longer lines the curve up
/// with the nuclide grid. Inside reactions.arrow alone the threshold is
/// silent; it becomes loud only when the whole directory loads, because the
/// loader compares `grid.tail(threshold_idx)` against the cross section length
/// at nuclide_arrow.rs:485-499. Reproducing that check here is what makes the
/// section self-sufficient.
///
/// `xs_temperatures` is wiring only here: Li6 has one temperature, so the list
/// is `["294K"]` on every row and `expected_xs_temperatures` restates
/// reactions.rs:82-87 rather than deriving it. The ordering rule and the
/// pairing of each curve with its own temperature are discriminated in
/// `constructed_two_temperature_reactions_keep_each_curve_with_its_own_temperature`.
#[test]
fn reaction_cross_sections_are_the_parsed_curves_unshifted() {
    let data = li6_ace();
    let dir = scratch();
    yamc_convert::reactions::write_reactions(&data, dir.path())
        .expect("reactions.arrow is written");
    let batch = section(dir.path(), "reactions.arrow");
    let mt_rows = rows_by_mt(&batch);

    for (&mt, rx) in &data.reactions {
        let row = mt_rows[&mt];
        let temperatures = str_list(&batch, "xs_temperatures", row);
        assert_eq!(
            temperatures,
            expected_xs_temperatures(&data, rx),
            "MT {mt} xs_temperatures"
        );
        assert_eq!(temperatures, vec!["294K".to_string()], "MT {mt}");

        let values = nested_f64_list(&batch, "xs_values", row);
        let thresholds = i32_list(&batch, "xs_threshold_idx", row);
        assert_eq!(values.len(), temperatures.len(), "MT {mt} xs_values");
        assert_eq!(thresholds.len(), temperatures.len(), "MT {mt} thresholds");

        for (i, t) in temperatures.iter().enumerate() {
            let xs = &rx.xs[t];
            // Exact: reactions.rs:88-91 clones `.y` with no grid alignment, no
            // scaling and no filtering, so a tolerance could only hide a
            // defect.
            assert_f64_slice_eq(&format!("MT {mt} xs_values[{t}]"), &values[i], &xs.y);
            assert_eq!(
                thresholds[i],
                xs.threshold_idx.unwrap_or(0) as i32,
                "MT {mt} xs_threshold_idx[{t}]"
            );
            // The loader's own invariant, reproduced here.
            assert_eq!(
                thresholds[i] as usize + values[i].len(),
                data.energy[t].len(),
                "MT {mt} at {t} does not span the grid from its threshold"
            );
            assert_eq!(data.energy[t].len(), 721);
        }
    }

    // Two pins so a wholesale parser shift is visible.
    let row = mt_rows[&51];
    assert_eq!(i32_list(&batch, "xs_threshold_idx", row)[0], 511);
    assert_eq!(nested_f64_list(&batch, "xs_values", row)[0].len(), 210);
}

/// A failure means MT 3, 4 or 27 is not the sum of the partials the same file
/// carries. Calling `synthesis::synthesize` here would prove nothing, so the
/// reconstruction is done from the file's own partial rows instead.
///
/// Every number compared below comes out of the written file, so what this
/// establishes is internal consistency: each synthesized row is the sum of the
/// rows beside it in the summation order stated below, and each one starts
/// where the fixture says it starts. A comparison of written MT 2 plus written
/// MT 3 against written MT 1 at 1e-6 relative used to sit here and has been
/// removed: MT 3 is pinned EXACTLY against those same partial rows a few lines
/// above, so that assertion could only ever fail on a property of the
/// evaluation's own seven digit ACE columns, never on a converter defect.
///
/// Two of the five synthesized MTs are not reached here at all. Li6 carries its
/// own MT 1 and its own MT 101, so those two rows are copies of the
/// evaluation's columns and the values the writer synthesized for them are
/// discarded. Both are checked on a constructed input in
/// `constructed_partials_pin_synthesized_mt_101_and_the_mt_3_term_of_mt_1`.
#[test]
fn the_synthesized_rows_are_the_sum_of_the_rows_beside_them() {
    let data = li6_ace();
    let dir = scratch();
    yamc_convert::reactions::write_reactions(&data, dir.path())
        .expect("reactions.arrow is written");
    let batch = section(dir.path(), "reactions.arrow");
    let mt_rows = rows_by_mt(&batch);
    let n = data.energy["294K"].len();
    assert_eq!(n, 721);

    let written = |mt: i32| nested_f64_list(&batch, "xs_values", mt_rows[&mt])[0].clone();
    let on_grid = |mt: i32| written_on_grid(&batch, &mt_rows, mt, n);

    // Ascending MT order, which is the order synthesis.rs:170-227 sums in:
    // inelastic() and absorption() are BTreeSets and non_elastic_scattering
    // walks the partials BTreeMap. Summing in that order gives bit identical
    // f64 results, so this is exact; any other order needs a 1e-12 relative
    // tolerance instead.
    let sum = |mts: &[i32]| -> Vec<f64> {
        let mut acc = vec![0.0; n];
        for &mt in mts {
            for (a, b) in acc.iter_mut().zip(on_grid(mt)) {
                *a += b;
            }
        }
        acc
    };

    for mt in [3, 4, 27] {
        assert_i32_slice_eq(
            &format!("MT {mt} xs_threshold_idx"),
            &i32_list(&batch, "xs_threshold_idx", mt_rows[&mt]),
            &[0],
        );
        assert_eq!(
            written(mt).len(),
            n,
            "MT {mt} is defined on the whole grid, whatever its partials' offsets were"
        );
    }

    assert_f64_slice_eq("MT 4", &written(4), &sum(&[51, 52, 53, 54, 55, 91]));
    assert_f64_slice_eq("MT 27", &written(27), &sum(&[102]));
    assert_f64_slice_eq("MT 3", &written(3), &sum(&[5, 51, 52, 53, 54, 55, 91, 102]));

    // The first nonzero index is what turns a one index shift loud, since the
    // written length cannot: it is 721 whatever the offset was. It is also the
    // only thing here that survives a shift applied to the PARTIAL rows too,
    // which moves both sides of the equalities above together.
    assert_eq!(first_nonzero(&written(4)), Some(512));
    assert_eq!(first_nonzero(&written(3)), Some(0));
    assert_eq!(first_nonzero(&written(27)), Some(0));

    // The one comparison here that reaches past the writer's own arithmetic.
    // Li6 carries its OWN MT 101 in the ACE ESZ block, so that row is PARSED,
    // while MT 27 is synthesized as the sum of MT 102, and on this fixture the
    // two are bit identical. So this pins the synthesized absorption against a
    // column the evaluation stored itself. It says nothing about a SYNTHESIZED
    // MT 101, which no VENDORED route in this file reaches: MT 101 is parsed on
    // the ACE route and empty on the ENDF route. The synthesized row is checked
    // in
    // `constructed_partials_pin_synthesized_mt_101_and_the_mt_3_term_of_mt_1`,
    // on a hand-built nuclide with two disappearance channels rather than
    // Li6's one. If a future fixture or parser change breaks the bit equality
    // below, 1e-6 relative is the honest fallback rather than deleting the
    // check.
    assert_f64_slice_eq("MT 101 against MT 27", &written(101), &written(27));
}

/// A failure means the writer changed how it handles an evaluation with no
/// nuclide grid. It is the only VENDORED case where `xs_temperatures` comes
/// from the extras branch at reactions.rs:87 rather than from the filtered list
/// at :82-86, and the only case anywhere in this file where the filtered list
/// is EMPTY: `temperatures()` has no entries and the "0K" key comes entirely
/// from `rx.xs`. With one extra key there is no order to check, which is what
/// `constructed_extra_temperatures_follow_the_processed_ones_in_map_order`
/// adds beside it.
///
/// It also pins what the writer actually does with the synthesized MTs on this
/// route, which is NOT "write no row": `n_energy` is 0 at every temperature so
/// the synthesis loop at reactions.rs:67-74 never runs, but the row loop at
/// :113-137 is unconditional, so MT 3, 27 and 101 each get a row whose
/// xs_temperatures, xs_values and xs_threshold_idx are all empty lists.
///
/// Those three rows are a SUSPECTED DEFECT, pinned and not endorsed. An
/// evaluation with no nuclide grid gains three reactions carrying no cross
/// section at all, and the loader skips its own grid length check in exactly
/// that situation (nuclide_arrow.rs:496), so they load as three silently empty
/// reactions rather than as an error. The fix belongs in reactions.rs, not
/// here. When it lands this test is EXPECTED to fail on the row count, and the
/// answer then is 38 rows, not a deleted test.
#[test]
fn the_endf_route_synthesizes_empty_rows_and_keys_its_cross_sections_at_0k() {
    let data = endf_nuclide(LI6_ENDF);
    assert!(
        data.energy.is_empty(),
        "the ENDF route carries no nuclide grid"
    );
    assert!(
        data.temperatures().is_empty(),
        "the ENDF route carries no k_ts"
    );

    let dir = scratch();
    yamc_convert::reactions::write_reactions(&data, dir.path())
        .expect("reactions.arrow is written");
    let batch = section(dir.path(), "reactions.arrow");
    let mt_rows = rows_by_mt(&batch);

    let mut expected: Vec<i32> = data.reactions.keys().copied().collect();
    assert_eq!(expected.len(), 38);
    // MT 1 and MT 4 are the evaluation's own, so only these three are appended.
    expected.extend([3, 27, 101]);
    assert_eq!(written_mts(&batch), expected);

    for (&mt, rx) in &data.reactions {
        let row = mt_rows[&mt];
        assert_eq!(
            str_list(&batch, "xs_temperatures", row),
            vec!["0K".to_string()],
            "MT {mt}"
        );
        assert_i32_slice_eq(
            &format!("MT {mt} xs_threshold_idx"),
            &i32_list(&batch, "xs_threshold_idx", row),
            &[0],
        );
        let values = nested_f64_list(&batch, "xs_values", row);
        assert_eq!(values.len(), 1, "MT {mt}");
        assert_f64_slice_eq(&format!("MT {mt} xs_values"), &values[0], &rx.xs["0K"].y);
        // Both hard-coded on this route at crates/endf/src/reaction.rs:361-362.
        assert!(bool_at(&batch, "center_of_mass", row), "MT {mt}");
        assert!(!bool_at(&batch, "redundant", row), "MT {mt}");
    }

    // No grid length invariant here: data.energy is empty, and the loader
    // skips its own check in the same situation (nuclide_arrow.rs:496).
    for mt in [3, 27, 101] {
        let row = mt_rows[&mt];
        assert!(
            str_list(&batch, "xs_temperatures", row).is_empty(),
            "MT {mt}"
        );
        assert_eq!(list_len(&batch, "xs_values", row), 0, "MT {mt}");
        assert!(
            i32_list(&batch, "xs_threshold_idx", row).is_empty(),
            "MT {mt}"
        );
        for column in ["xs_temperatures", "xs_values", "xs_threshold_idx"] {
            assert!(!is_null(&batch, column, row), "MT {mt} {column} is null");
        }
    }
}

// ---------------------------------------------------------------------------
// Constructed inputs.
//
// Everything below this line is built by hand rather than parsed from an
// evaluation. Each test names the writer branch that no vendored evaluation
// discriminates and why only a constructed input separates the two columns in
// question. None of it is parity with any evaluation: the numbers are invented
// and are chosen to differ from one another, which is the entire point, since
// two columns that hold the same value cannot show a swap between them.
// ---------------------------------------------------------------------------

/// kT in eV for a temperature in kelvin, so `temperatures()` names it back.
fn k_t(kelvin: f64) -> f64 {
    kelvin * endf::K_BOLTZMANN
}

/// A hand-built nuclide with TWO temperatures and THREE energy grids.
///
/// Parsed from nothing. The shape it exists to produce, which no vendored
/// fixture has:
///
/// * `temperatures()` is `["294K", "1200K"]`, in `k_ts` order, which is NOT
///   the order a BTreeMap keyed by those names walks: `"0K"` then `"1200K"`
///   then `"294K"`.
/// * `energy` holds a grid at a temperature that is not in `temperatures()`,
///   which is the 0 K PENDF grid the NJOY route adds (nuclide.rs:28-40).
/// * the three grids have three different lengths and share no value, so a
///   grid paired with the wrong temperature cannot pass unnoticed.
fn two_temperature_nuclide() -> IncidentNeutron {
    let mut data = IncidentNeutron::new(3, 6, 0);
    data.k_ts = vec![k_t(294.0), k_t(1200.0)];
    data.energy.insert("294K".to_string(), vec![1.0, 2.0, 3.0]);
    data.energy
        .insert("1200K".to_string(), vec![10.0, 20.0, 30.0, 40.0]);
    data.energy.insert("0K".to_string(), vec![100.0, 200.0]);
    data
}

/// CONSTRUCTED INPUT, not an evaluation. `temperatures` and
/// `energy_temperatures` are different lists here, and the writer must not
/// confuse them.
///
/// No vendored evaluation reaches this: on Li6 ACE both columns hold
/// `["294K"]` and on the ENDF route both are empty, so swapping the two
/// `string_list` arguments at nuclide.rs:65 and :67 writes a valid, loadable,
/// wrong file that every test built from vendored bytes accepts. That is why
/// the input is built by hand, and it is why nothing below should be read as
/// agreement with a real evaluation.
///
/// The same construction discriminates the two phase ordering rule the
/// writer's own comment argues for, which a one element list cannot.
#[test]
fn constructed_two_temperature_nuclide_separates_the_two_temperature_columns() {
    let data = two_temperature_nuclide();
    // The constructed input, restated: if `temperature_str` ever rendered
    // these kTs differently, everything below would be about another nuclide.
    assert_eq!(
        data.temperatures(),
        vec!["294K".to_string(), "1200K".to_string()]
    );

    let dir = scratch();
    yamc_convert::nuclide::write_nuclide(&data, dir.path()).expect("nuclide.arrow is written");
    let batch = section(dir.path(), "nuclide.arrow");

    let temperatures = str_list(&batch, "temperatures", 0);
    let energy_temperatures = str_list(&batch, "energy_temperatures", 0);
    assert_eq!(
        temperatures,
        vec!["294K".to_string(), "1200K".to_string()],
        "temperatures is data.temperatures(), which has no 0 K entry"
    );
    // Processed order first, then whatever the energy map holds that the
    // temperature list does not. Iterating data.energy directly would give
    // ["0K", "1200K", "294K"]; filtering to the temperature list would drop
    // the 0 K grid; writing this list into the other column would put a
    // temperature the nuclide was never processed at into `temperatures`.
    assert_eq!(
        energy_temperatures,
        vec!["294K".to_string(), "1200K".to_string(), "0K".to_string()],
        "energy_temperatures is the two phase list of nuclide.rs:41-51"
    );
    // The restated rule against the literal, on the only input in this file
    // where the rule's two phases hold different things. This says nothing
    // about the writer that the literal above has not already said; it keeps
    // the helper the Li6 test leans on honest.
    assert_eq!(energy_temperatures, expected_energy_temperatures(&data));
    let mut seen = BTreeSet::new();
    for t in &energy_temperatures {
        assert!(
            data.energy.contains_key(t),
            "energy_temperatures lists {t}, which this nuclide has no grid for"
        );
        assert!(
            seen.insert(t.clone()),
            "energy_temperatures lists {t} twice, which duplicates a whole grid"
        );
    }

    // Two distinct kTs, so the positional pairing with `temperatures` is
    // finally visible: a reversed pair names the wrong temperature.
    let k_ts = f64_list(&batch, "kTs", 0);
    assert_f64_slice_eq("kTs", &k_ts, &data.k_ts);
    for (kt, t) in k_ts.iter().zip(&temperatures) {
        assert_eq!(
            &endf::data::temperature_str(kt / endf::K_BOLTZMANN),
            t,
            "kT {kt} does not name temperature {t}"
        );
    }

    // Three grids of three different lengths: a grid paired with the wrong
    // temperature fails on the length before it fails on a value.
    let grids = nested_f64_list(&batch, "energy_values", 0);
    assert_eq!(
        grids.len(),
        energy_temperatures.len(),
        "one grid per energy temperature"
    );
    for (grid, t) in grids.iter().zip(&energy_temperatures) {
        assert_f64_slice_eq(&format!("energy_values[{t}]"), grid, &data.energy[t]);
    }
}

/// The value the constructed probability tables hold at one
/// (energy, quantity, band), distinct for every triple and every seed.
fn urr_cell(seed: f64, e: usize, q: usize, b: usize) -> f64 {
    seed + 100.0 * e as f64 + 10.0 * q as f64 + b as f64
}

/// A `ProbabilityTables` whose every field differs from every other row's.
///
/// `table` is filled from the (energy, quantity, band) triple rather than from
/// a running index, so the test can compare the written list against
/// `urr_cell(e, q, b)` instead of against `ProbabilityTables::get`, which
/// recomputes the writer's own index expression over the same Vec. What is
/// being checked is still a clone, since the writer reshapes nothing: a
/// mismatch means the copy was permuted or came from another row. The energies
/// are half integers so that no energy can collide with a table value.
fn distinct_tables(
    seed: f64,
    shape: [usize; 3],
    interpolation: i64,
    inelastic_flag: i64,
    absorption_flag: i64,
    multiply_smooth: bool,
) -> ProbabilityTables {
    let [n_e, n_q, n_b] = shape;
    let mut table = Vec::with_capacity(n_e * n_q * n_b);
    for e in 0..n_e {
        for q in 0..n_q {
            for b in 0..n_b {
                table.push(urr_cell(seed, e, q, b));
            }
        }
    }
    ProbabilityTables {
        energy: (0..n_e).map(|e| seed + 0.5 + e as f64).collect(),
        table,
        shape,
        interpolation,
        inelastic_flag,
        absorption_flag,
        multiply_smooth,
    }
}

/// CONSTRUCTED INPUT, not an evaluation. Every scalar column of urr.arrow,
/// told apart, and the row order with it.
///
/// No vendored evaluation reaches this: `synthetic-urr.ace.xz` is the only
/// input in the tree that produces a urr.arrow at all, and it gives ONE row on
/// which `interpolation` is 2, `table_shape` is `[4, 6, 3]`,
/// `multiply_smooth` is true and `inelastic` and `absorption` are both -1.
/// Against that row a writer that pushed a literal instead of reading the
/// parsed field, or that swapped `inelastic` with `absorption`, is green. So
/// the input here is built by hand with three rows whose scalars all differ,
/// including the two flags within a row and `multiply_smooth` across rows.
/// The numbers are invented and mean nothing physically.
#[test]
fn constructed_urr_rows_carry_their_own_scalars_in_processed_order() {
    // In the order the rows must come out: the processed temperatures first,
    // then what the map holds and `temperatures()` does not. The map's own
    // order is ["0K", "1200K", "294K"], so a writer that walked `data.urr`
    // directly writes these three rows permuted, and one that filtered to
    // `temperatures()` drops the 0 K row entirely.
    let expected: Vec<(&str, f64, ProbabilityTables)> = vec![
        (
            "294K",
            1000.0,
            distinct_tables(1000.0, [2, 6, 3], 2, 4, 102, false),
        ),
        (
            "1200K",
            2000.0,
            distinct_tables(2000.0, [3, 6, 4], 5, 18, 27, true),
        ),
        (
            "0K",
            3000.0,
            distinct_tables(3000.0, [1, 6, 2], 1, -1, 51, false),
        ),
    ];

    let mut data = IncidentNeutron::new(92, 235, 0);
    data.k_ts = vec![k_t(294.0), k_t(1200.0)];
    for (t, _, tables) in &expected {
        data.urr.insert((*t).to_string(), tables.clone());
    }

    let dir = scratch();
    let wrote = yamc_convert::nuclide::write_urr(&data, dir.path()).expect("no error");
    assert!(
        wrote,
        "the constructed nuclide has unresolved resonance data"
    );
    let batch = section(dir.path(), "urr.arrow");
    assert_eq!(batch.num_rows(), expected.len());

    let written: Vec<String> = (0..batch.num_rows())
        .map(|row| str_at(&batch, "temperature", row))
        .collect();
    assert_eq!(
        written,
        expected
            .iter()
            .map(|(t, _, _)| (*t).to_string())
            .collect::<Vec<String>>(),
        "the rows are the processed temperatures first, then the map's extras"
    );
    let mut seen = BTreeSet::new();
    for t in &written {
        assert!(seen.insert(t.clone()), "urr.arrow lists {t} twice");
    }

    for (row, (t, seed, tables)) in expected.iter().enumerate() {
        // Each column against its OWN row's field. No two rows share a value
        // in any of these, so a column filled from a literal, or copied from
        // the first row, or swapped with its neighbour, fails here.
        assert_eq!(
            i32_at(&batch, "interpolation", row),
            tables.interpolation as i32,
            "{t} interpolation"
        );
        assert_eq!(
            i64::from(i32_at(&batch, "inelastic", row)),
            tables.inelastic_flag,
            "{t} inelastic"
        );
        assert_eq!(
            i64::from(i32_at(&batch, "absorption", row)),
            tables.absorption_flag,
            "{t} absorption"
        );
        assert_ne!(
            tables.inelastic_flag, tables.absorption_flag,
            "{t}: the two flags must differ or the two assertions above cannot \
             see a swap between the columns"
        );
        assert_eq!(
            bool_at(&batch, "multiply_smooth", row),
            tables.multiply_smooth,
            "{t} multiply_smooth"
        );
        assert_i32_slice_eq(
            &format!("{t} table_shape"),
            &i32_list(&batch, "table_shape", row),
            &tables.shape.iter().map(|&n| n as i32).collect::<Vec<i32>>(),
        );
        assert_f64_slice_eq(
            &format!("{t} energy"),
            &f64_list(&batch, "energy", row),
            &tables.energy,
        );

        // The ravel, against the formula the table was filled from rather than
        // against `ProbabilityTables::get`, which would recompute the same
        // index over the same Vec. Every row has a different `n_energy` and
        // `n_band`, so a shape taken from another row does not even fit.
        let [n_e, n_q, n_b] = tables.shape;
        let flat = f64_list(&batch, "table_data", row);
        assert_eq!(flat.len(), n_e * n_q * n_b, "{t} table_data length");
        for e in 0..n_e {
            for q in 0..n_q {
                for b in 0..n_b {
                    assert_eq!(
                        flat[(e * n_q + q) * n_b + b],
                        urr_cell(*seed, e, q, b),
                        "{t} table_data[{e},{q},{b}]"
                    );
                }
            }
        }
    }
}

/// One cross section given from its threshold upwards, the way an ACE table
/// stores a threshold reaction.
fn partial_xs(x: Vec<f64>, y: Vec<f64>, threshold_idx: usize) -> Tabulated1D {
    Tabulated1D {
        threshold_idx: Some(threshold_idx),
        ..Tabulated1D::new(x, y)
    }
}

/// CONSTRUCTED INPUT, not an evaluation. Each temperature's curve stays with
/// its own temperature, its own threshold and its own grid.
///
/// No vendored evaluation reaches this: every vendored neutron evaluation has
/// exactly one temperature ("294K" on the ACE route, "0K" on the ENDF route),
/// so `xs_temperatures`, `xs_values` and `xs_threshold_idx` are one element
/// lists on every row of every other test here and no permutation of them
/// exists to be caught. The input is built by hand with two temperatures whose
/// curves have different lengths, different thresholds and no shared value, so
/// a writer that walked `rx.xs` in the map's own order ("1200K" before "294K")
/// or that paired a curve with the wrong name fails. Not parity with any
/// evaluation.
///
/// Both branches that build a row are covered, because they iterate different
/// lists: the parsed row at reactions.rs:82-95 walks `xs_temperatures`, and
/// the synthesized rows at :117-135 walk `temperatures` instead.
#[test]
fn constructed_two_temperature_reactions_keep_each_curve_with_its_own_temperature() {
    let mut data = two_temperature_nuclide();
    // The 0 K grid this builder carries is deliberately left in place:
    // `write_reactions` reads `data.energy` only for each PROCESSED
    // temperature's grid length (reactions.rs:68) and never iterates its keys,
    // so no row may gain a third entry from it.
    let mut elastic = Reaction::new(2);
    elastic.xs.insert(
        "294K".to_string(),
        partial_xs(vec![1.0, 2.0, 3.0], vec![7.0, 8.0, 9.0], 0),
    );
    elastic.xs.insert(
        "1200K".to_string(),
        partial_xs(vec![30.0, 40.0], vec![70.0, 80.0], 2),
    );
    data.reactions.insert(2, elastic);

    let dir = scratch();
    yamc_convert::reactions::write_reactions(&data, dir.path())
        .expect("reactions.arrow is written");
    let batch = section(dir.path(), "reactions.arrow");
    let mt_rows = rows_by_mt(&batch);

    let row = mt_rows[&2];
    let temperatures = str_list(&batch, "xs_temperatures", row);
    assert_eq!(
        temperatures,
        vec!["294K".to_string(), "1200K".to_string()],
        "MT 2 xs_temperatures is the processed order, not the map's"
    );
    let values = nested_f64_list(&batch, "xs_values", row);
    let thresholds = i32_list(&batch, "xs_threshold_idx", row);
    assert_f64_slice_eq("MT 2 at 294K", &values[0], &[7.0, 8.0, 9.0]);
    assert_f64_slice_eq("MT 2 at 1200K", &values[1], &[70.0, 80.0]);
    assert_i32_slice_eq("MT 2 xs_threshold_idx", &thresholds, &[0, 2]);
    // The loader's grid span invariant, at each temperature separately. This
    // is what a mispairing breaks even where the values happen to look
    // plausible: 0 + 3 is the 294K grid and 2 + 2 is the 1200K grid, and the
    // two curves exchanged satisfy neither.
    for (i, t) in temperatures.iter().enumerate() {
        assert_eq!(
            thresholds[i] as usize + values[i].len(),
            data.energy[t].len(),
            "MT 2 at {t} does not span that temperature's grid from its threshold"
        );
    }

    // The synthesized rows come from the other list. MT 2 is the only parsed
    // reaction, so MT 1 is the elastic curve and nothing else, laid on each
    // temperature's own grid: three points at 294K, four at 1200K with the
    // threshold's two leading zeros.
    let row = mt_rows[&1];
    assert_eq!(
        str_list(&batch, "xs_temperatures", row),
        vec!["294K".to_string(), "1200K".to_string()],
        "MT 1 xs_temperatures"
    );
    let values = nested_f64_list(&batch, "xs_values", row);
    assert_f64_slice_eq("MT 1 at 294K", &values[0], &[7.0, 8.0, 9.0]);
    assert_f64_slice_eq("MT 1 at 1200K", &values[1], &[0.0, 0.0, 70.0, 80.0]);
    assert_i32_slice_eq(
        "MT 1 xs_threshold_idx",
        &i32_list(&batch, "xs_threshold_idx", row),
        &[0, 0],
    );
}

/// A hand-built nuclide carrying a NON-ELASTIC partial in every summation
/// group the synthesized rows are built from.
///
/// Parsed from nothing. One temperature, a four point grid and six reactions
/// whose cross sections are distinct powers of two, so every subset sum is a
/// different number and a synthesized row built from the wrong set cannot land
/// on the right value by accident. Three of the six start above the bottom of
/// the grid, at two different thresholds, so a partial laid on the grid at the
/// wrong offset changes the sums as well.
///
/// What each channel is here for, and what it looks like on the grid:
///
/// * MT 2, elastic, `[1, 1, 1, 1]`. The one channel MT 3 excludes, and the
///   term MT 1 adds to MT 3.
/// * MT 16, (n,2n), `[0, 0, 2, 2]`. Neutron emitting and not level inelastic,
///   so it enters MT 3 and nothing else.
/// * MT 51, first inelastic level, `[0, 4, 4, 4]`. The only term of MT 4.
/// * MT 102, capture, `[8, 8, 8, 8]`, and MT 103, (n,p), `[0, 0, 16, 16]`.
///   Both disappearance, so MT 101 is their sum.
/// * MT 18, fission, `[0, 0, 32, 32]`. Reaches MT 27 and MT 3 through the
///   fission term and nothing else.
///
/// No vendored neutron evaluation has this shape. Li6 carries no MT 16, no
/// MT 18 and no MT 103, and it carries its OWN MT 1 and MT 101, so on the only
/// fixture that reaches these rows with real numbers MT 1 and MT 101 are
/// parsed columns rather than synthesized ones. The numbers here are invented.
fn multi_channel_nuclide() -> IncidentNeutron {
    let mut data = IncidentNeutron::new(26, 56, 0);
    data.k_ts = vec![k_t(294.0)];
    data.energy
        .insert("294K".to_string(), vec![1.0, 2.0, 3.0, 4.0]);
    for (mt, x, y, threshold) in [
        (2, vec![1.0, 2.0, 3.0, 4.0], vec![1.0; 4], 0),
        (16, vec![3.0, 4.0], vec![2.0; 2], 2),
        (18, vec![3.0, 4.0], vec![32.0; 2], 2),
        (51, vec![2.0, 3.0, 4.0], vec![4.0; 3], 1),
        (102, vec![1.0, 2.0, 3.0, 4.0], vec![8.0; 4], 0),
        (103, vec![3.0, 4.0], vec![16.0; 2], 2),
    ] {
        let mut rx = Reaction::new(mt);
        rx.xs
            .insert("294K".to_string(), partial_xs(x, y, threshold));
        data.reactions.insert(mt, rx);
    }
    data
}

/// CONSTRUCTED INPUT, not an evaluation. The synthesized MT 101 row, and the
/// MT 3 term of the synthesized MT 1 row.
///
/// No vendored evaluation reaches either. Li6 ACE carries its own MT 1 and its
/// own MT 101 in the ESZ block, so both rows are PARSED there and the
/// synthesized values are computed and thrown away; the ENDF route has no
/// nuclide grid, so it synthesizes nothing and writes those rows empty; and the
/// only other constructed nuclide in this file has elastic alone, which makes
/// its MT 3 identically zero and its MT 101 all zeros. Against all three, a
/// writer that dropped a channel from the disappearance sum, or dropped the
/// `add(&mut mt1, &mt3)` term of MT 1 entirely, stays green. So the input here
/// is built by hand with a partial in every group. The numbers are invented and
/// nothing below is parity with any evaluation.
///
/// Every expected row is written out as a literal, computed by hand from the
/// table in `multi_channel_nuclide`, and then again as a sum of the file's own
/// partial rows. The literals are the part that does not go through the writer.
#[test]
fn constructed_partials_pin_synthesized_mt_101_and_the_mt_3_term_of_mt_1() {
    let data = multi_channel_nuclide();
    assert_eq!(data.temperatures(), vec!["294K".to_string()]);
    for mt in [1, 3, 4, 27, 101] {
        assert!(
            !data.reactions.contains_key(&mt),
            "MT {mt} must be absent from the input or the writer copies it \
             instead of synthesizing it"
        );
    }

    let dir = scratch();
    yamc_convert::reactions::write_reactions(&data, dir.path())
        .expect("reactions.arrow is written");
    let batch = section(dir.path(), "reactions.arrow");
    let mt_rows = rows_by_mt(&batch);
    let n = data.energy["294K"].len();
    assert_eq!(n, 4);

    let on_grid = |mt: i32| written_on_grid(&batch, &mt_rows, mt, n);
    let sum = |mts: &[i32]| -> Vec<f64> {
        let mut acc = vec![0.0; n];
        for &mt in mts {
            for (a, b) in acc.iter_mut().zip(on_grid(mt)) {
                *a += b;
            }
        }
        acc
    };

    assert_eq!(
        written_mts(&batch),
        vec![2, 16, 18, 51, 102, 103, 1, 3, 4, 27, 101],
        "the parsed reactions then the five synthesized MTs, none of which the \
         input carries"
    );

    // The partials first, so a failure in the sums below cannot be blamed on
    // the rows they are summed from. Every value is a small integer, so every
    // sum in this test is exact in f64 whatever order it is taken in.
    assert_f64_slice_eq("MT 2", &on_grid(2), &[1.0, 1.0, 1.0, 1.0]);
    assert_f64_slice_eq("MT 16", &on_grid(16), &[0.0, 0.0, 2.0, 2.0]);
    assert_f64_slice_eq("MT 18", &on_grid(18), &[0.0, 0.0, 32.0, 32.0]);
    assert_f64_slice_eq("MT 51", &on_grid(51), &[0.0, 4.0, 4.0, 4.0]);
    assert_f64_slice_eq("MT 102", &on_grid(102), &[8.0, 8.0, 8.0, 8.0]);
    assert_f64_slice_eq("MT 103", &on_grid(103), &[0.0, 0.0, 16.0, 16.0]);

    // THE SYNTHESIZED DISAPPEARANCE ROW. Capture plus (n,p), and neither one
    // alone: 8 is MT 102 by itself and 16 is MT 103 by itself, so a sum missing
    // either term is a different number at every point above the (n,p)
    // threshold. Li6 has no MT 103 at all, which is why a disappearance set
    // that lost it passes every other test here.
    assert_f64_slice_eq("MT 101", &on_grid(101), &[8.0, 8.0, 24.0, 24.0]);
    assert_f64_slice_eq(
        "MT 101 against MT 102 + MT 103",
        &on_grid(101),
        &sum(&[102, 103]),
    );

    // MT 4 is the level inelastic sum and MT 27 is fission plus disappearance.
    assert_f64_slice_eq("MT 4", &on_grid(4), &[0.0, 4.0, 4.0, 4.0]);
    assert_f64_slice_eq("MT 4 against MT 51", &on_grid(4), &sum(&[51]));
    assert_f64_slice_eq("MT 27", &on_grid(27), &[8.0, 8.0, 56.0, 56.0]);
    assert_f64_slice_eq(
        "MT 27 against MT 18 + MT 101",
        &on_grid(27),
        &sum(&[18, 101]),
    );

    // MT 3 is everything but elastic: the neutron emitting channels, plus
    // fission, plus disappearance.
    assert_f64_slice_eq("MT 3", &on_grid(3), &[8.0, 12.0, 62.0, 62.0]);
    assert_f64_slice_eq(
        "MT 3 against its partials",
        &on_grid(3),
        &sum(&[16, 18, 51, 102, 103]),
    );

    // THE MT 3 TERM OF MT 1. Elastic is 1 barn flat, so a writer that wrote
    // MT 1 as elastic alone gets [1, 1, 1, 1] and this fires at index 0. On
    // Li6 the same defect is invisible because MT 1 is the evaluation's own
    // column, and on an elastic-only nuclide it is invisible because MT 3 is
    // zero.
    assert_f64_slice_eq("MT 1", &on_grid(1), &[9.0, 13.0, 63.0, 63.0]);
    assert_f64_slice_eq("MT 1 against MT 2 + MT 3", &on_grid(1), &sum(&[2, 3]));
    assert_ne!(
        on_grid(3),
        vec![0.0; n],
        "MT 3 must be non-zero or MT 1 cannot show that it added it"
    );

    // No two synthesized rows are equal, so a row copied from its neighbour is
    // visible. Without this the pairs (MT 27, MT 101) and (MT 3, MT 4) are the
    // easy mistakes to make, and on a nuclide with no fission MT 27 and MT 101
    // are identical.
    let synthesized: Vec<(i32, Vec<f64>)> = [1, 3, 4, 27, 101]
        .into_iter()
        .map(|mt| (mt, on_grid(mt)))
        .collect();
    for (i, (mt, xs)) in synthesized.iter().enumerate() {
        for (other, other_xs) in &synthesized[i + 1..] {
            assert_ne!(
                xs, other_xs,
                "MT {mt} and MT {other} hold the same curve, so a copy between \
                 them would pass"
            );
        }
        // Each one spans the whole grid from index 0, whatever its partials'
        // thresholds were.
        assert_i32_slice_eq(
            &format!("MT {mt} xs_threshold_idx"),
            &i32_list(&batch, "xs_threshold_idx", mt_rows[mt]),
            &[0],
        );
        assert_eq!(
            nested_f64_list(&batch, "xs_values", mt_rows[mt])[0].len(),
            n,
            "MT {mt} is not defined on the whole grid"
        );
    }
}

/// CONSTRUCTED INPUT, not an evaluation. `redundant = true` on a synthesized
/// row, checked by the rule a consumer applies to that column rather than by
/// transcribing the literal.
///
/// The three synthesized rows on Li6 (MT 3, 4 and 27) have no parsed side, so
/// `reaction_flags_are_the_parsed_reactions_own_flags` can only compare the
/// writer's literals at reactions.rs:130-132 against a copy of them, and says
/// so. This test replaces the copy for ONE of the three columns. A consumer
/// with no fast_xs grid reconstructs the total by summing every reaction whose
/// `redundant` flag is false (yamc-gpu `total_xs_at`,
/// crates/yamc-gpu/src/neutron/xs/extract.rs:151-163, and the same guard on the
/// scattering sum at crates/yamc-nuclide/src/nuclide/sampling.rs:499). On this
/// input that sum must come out as the MT 1 row exactly, and it does so only
/// because the five synthesized rows are excluded from it. A writer that
/// flagged them `false` makes the consumer count MT 3, 4, 27 and 101 a second
/// time, and the assertion below fires whether or not the transcription
/// elsewhere in this file was changed to match.
///
/// No vendored evaluation reaches this. It needs a nuclide whose parsed rows
/// are all non-redundant and partition the total, which Li6 is not: its MT 1,
/// MT 101 and MT 203 to 207 are parsed AND redundant, so the same sum there is
/// a statement about the evaluation rather than about the writer's literal.
///
/// Still a transcription after this, and still recorded as such: `Q_value` 0.0
/// and `center_of_mass` false on those rows. A synthesized sum has no products,
/// so nothing downstream reads either one, and there is no rule to check them
/// against.
#[test]
fn constructed_synthesized_rows_are_redundant_so_the_rest_sum_to_the_total() {
    let data = multi_channel_nuclide();
    let dir = scratch();
    yamc_convert::reactions::write_reactions(&data, dir.path())
        .expect("reactions.arrow is written");
    let batch = section(dir.path(), "reactions.arrow");
    let mt_rows = rows_by_mt(&batch);
    let n = data.energy["294K"].len();

    let mut plain = Vec::new();
    let mut redundant = Vec::new();
    let mut plain_sum = vec![0.0; n];
    let mut every_row_sum = vec![0.0; n];
    for row in 0..batch.num_rows() {
        let mt = i32_at(&batch, "mt", row);
        let xs = written_on_grid(&batch, &mt_rows, mt, n);
        for (a, b) in every_row_sum.iter_mut().zip(&xs) {
            *a += b;
        }
        if bool_at(&batch, "redundant", row) {
            redundant.push(mt);
        } else {
            plain.push(mt);
            for (a, b) in plain_sum.iter_mut().zip(&xs) {
                *a += b;
            }
        }
    }

    // The consumer's rule, first, so it is what fires. Exact: every value in
    // the file is a small integer.
    assert_f64_slice_eq(
        "the non-redundant rows summed, against the MT 1 row",
        &plain_sum,
        &written_on_grid(&batch, &mt_rows, 1, n),
    );
    // And the filter is load bearing: without it the same consumer counts the
    // sums as well as their parts.
    assert_ne!(
        every_row_sum, plain_sum,
        "every row is flagged the same way, so the redundant filter cannot be \
         shown to matter on this input"
    );

    // Which rows ended up on which side, as documentation. This part IS a
    // transcription of reactions.rs:132 and the parser's default; the sum above
    // is the assertion that does not depend on it.
    assert_eq!(plain, vec![2, 16, 18, 51, 102, 103]);
    assert_eq!(redundant, yamc_convert::synthesis::SYNTHETIC_MTS.to_vec());

    // The other two literals on those rows, pinned and not endorsed: there is
    // no parsed side and no consumer rule to check them against.
    for &mt in &redundant {
        let row = mt_rows[&mt];
        assert_eq!(f64_at(&batch, "Q_value", row), 0.0, "MT {mt} Q value");
        assert!(!bool_at(&batch, "center_of_mass", row), "MT {mt}");
    }
}

/// CONSTRUCTED INPUT, not an evaluation. TWO cross sections at temperatures the
/// nuclide was never processed at, so the order the extras phase writes them in
/// is finally visible.
///
/// No vendored evaluation reaches this. The extras branch at reactions.rs:87 is
/// reached only on the ENDF route, where `temperatures()` is empty and `rx.xs`
/// holds the single key "0K": one extra has no order. Everything else in this
/// file has one temperature and no extras at all. So a writer that reversed or
/// re-sorted the extras is green everywhere else, and the input here is built
/// by hand with three keys whose curves have three different lengths, three
/// different thresholds and no shared value.
///
/// What the assertion pins is the writer's rule and not a physical fact: the
/// processed temperatures in `k_ts` order, then whatever `rx.xs` holds that
/// `temperatures()` does not, in the map's own (BTreeMap, so lexicographic)
/// order. The load bearing half is that `xs_values` and `xs_threshold_idx` are
/// built by walking that same list, so each curve keeps its own name and its
/// own threshold whichever order the names come out in.
#[test]
fn constructed_extra_temperatures_follow_the_processed_ones_in_map_order() {
    let mut data = IncidentNeutron::new(3, 6, 0);
    data.k_ts = vec![k_t(294.0)];
    data.energy.insert("294K".to_string(), vec![1.0, 2.0, 3.0]);
    let mut elastic = Reaction::new(2);
    // Only "294K" is a processed temperature. "0K" is the unbroadened grid the
    // NJOY route leaves behind and "600K" stands for a broadened set that was
    // never named in k_ts; both are extras.
    elastic.xs.insert(
        "294K".to_string(),
        partial_xs(vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0], 0),
    );
    elastic.xs.insert(
        "0K".to_string(),
        partial_xs(vec![2.0, 3.0], vec![10.0, 20.0], 1),
    );
    // "600K" has no grid in `data.energy` at all, so its abscissae are its
    // own: four points starting at index 4 of a grid this nuclide does not
    // carry. Nothing checks a span for a temperature with no grid, which is
    // exactly the situation the extras branch exists for.
    elastic.xs.insert(
        "600K".to_string(),
        partial_xs(
            vec![5.0, 6.0, 7.0, 8.0],
            vec![100.0, 200.0, 300.0, 400.0],
            4,
        ),
    );
    data.reactions.insert(2, elastic);
    assert_eq!(data.temperatures(), vec!["294K".to_string()]);

    let dir = scratch();
    yamc_convert::reactions::write_reactions(&data, dir.path())
        .expect("reactions.arrow is written");
    let batch = section(dir.path(), "reactions.arrow");
    let mt_rows = rows_by_mt(&batch);

    let row = mt_rows[&2];
    let temperatures = str_list(&batch, "xs_temperatures", row);
    assert_eq!(
        temperatures,
        vec!["294K".to_string(), "0K".to_string(), "600K".to_string()],
        "the processed temperature first, then the two extras in the map's own \
         order, which is lexicographic and puts 0K before 600K"
    );
    // The restated rule beside the literal, on the only input in this file
    // where the extras phase holds more than one name.
    assert_eq!(
        temperatures,
        expected_xs_temperatures(&data, &data.reactions[&2])
    );

    // Each curve with its own name. Three lengths, three thresholds, no shared
    // value, so a permutation cannot pass on either column.
    let values = nested_f64_list(&batch, "xs_values", row);
    assert_f64_slice_eq("MT 2 at 294K", &values[0], &[1.0, 2.0, 3.0]);
    assert_f64_slice_eq("MT 2 at 0K", &values[1], &[10.0, 20.0]);
    assert_f64_slice_eq("MT 2 at 600K", &values[2], &[100.0, 200.0, 300.0, 400.0]);
    assert_i32_slice_eq(
        "MT 2 xs_threshold_idx",
        &i32_list(&batch, "xs_threshold_idx", row),
        &[0, 1, 4],
    );

    // The synthesized rows are built from `temperatures` rather than from
    // `rx.xs`, so no extra may leak into them: a nuclide processed at one
    // temperature has one synthesized curve, whatever else the reaction was
    // stored at.
    for mt in [1, 3, 4, 27, 101] {
        let row = mt_rows[&mt];
        assert_eq!(
            str_list(&batch, "xs_temperatures", row),
            vec!["294K".to_string()],
            "MT {mt} gained a temperature the nuclide was not processed at"
        );
        assert_eq!(
            nested_f64_list(&batch, "xs_values", row).len(),
            1,
            "MT {mt}"
        );
    }
    // MT 1 is elastic alone here, on the 294K grid only.
    assert_f64_slice_eq(
        "MT 1 at 294K",
        &nested_f64_list(&batch, "xs_values", mt_rows[&1])[0],
        &[1.0, 2.0, 3.0],
    );
}
