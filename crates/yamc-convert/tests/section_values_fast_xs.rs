//! `fast_xs.arrow`, column for column, against the evaluation it was built
//! from.
//!
//! The accelerator is the section a collision reads on every flight, and it is
//! also the one whose columns are interchangeable at the schema level: six
//! `list<int32>` fields and six `list<double>` fields that
//! `RecordBatch::try_new` matches by position and data type alone. Swapping any
//! two of the same type gives a file that satisfies the declared schema and
//! loads without complaint, so only a per-column value comparison can see it.
//!
//! `Li6.ace.xz` is the only vendored evaluation that reaches this writer with
//! real cross sections. `IncidentNeutron::from_endf` leaves `data.energy`
//! empty, so every ENDF fixture is refused for want of a grid (that refusal is
//! asserted below), and `entry::convert_neutron_transport` declines an ACE
//! source, so `write_fast_xs` is called here directly rather than through the
//! entry point. `synthetic-urr.ace.xz` reaches it too, with a grid of four
//! zeros, and its refusal is asserted in
//! `a_grid_with_no_positive_extent_is_refused`.
//!
//! # Where the input is CONSTRUCTED rather than parsed
//!
//! Li6 is not fissile, has no redundant reaction that emits a neutron, and has
//! exactly one temperature, so three of this writer's decisions cannot be
//! discriminated by any vendored evaluation in the tree. Each of the three was
//! checked by deleting the writer's code and watching every test here stay
//! green:
//!
//! - absorption is MT 27 LESS fission. On a nuclide whose fission column is
//!   zero everywhere the subtraction can simply be deleted. That is the exact
//!   wrong answer `fast_xs.rs:18-20` names, the one that double counts every
//!   fission against the total.
//! - the per-MT loop skips `rx.redundant`. Every redundant Li6 reaction also
//!   fails `emits_neutron`, so the guard never decides anything here, and it is
//!   the Be9 double-count defence `synthesis.rs:124-134` cites.
//! - the row set follows `temperatures()`, not `data.energy.keys()`. On Li6
//!   those are the same one-element set.
//!
//! Those three are covered by `IncidentNeutron` values built in this file, with
//! `built_nuclide` and `built_reaction`. They are NOT evaluation parity, and
//! nothing about them says the parser agrees with a real ACE table: what they
//! establish is that the writer's arithmetic and its filters do what the module
//! doc says. Their fields are given deliberately DISTINCT values, which is the
//! whole point, since two columns that happen to be equal cannot discriminate a
//! swap between them.

mod section_values;
use section_values::*;

use std::collections::BTreeMap;

use arrow_array::RecordBatch;
use endf::IncidentNeutron;
use yamc_convert::fast_xs::{write_fast_xs, FISSION_MTS};
use yamc_convert::synthesis;

/// The number of bins in the logarithmic index, as `fast_xs.rs:50` fixes it.
/// Duplicated rather than imported because the writer's constant is private,
/// which is the point: a change that misses one of the two sites shows up here.
const LOG_BINS: usize = 8000;

/// `fast_xs.arrow` written from a nuclide and read back through the loader's
/// own reader.
///
/// The scratch directory goes out of scope on the way out, which is safe:
/// `read_arrow_file` collects every batch into memory before returning
/// (`arrow_helpers.rs:71-87`), so the returned batch does not refer to the
/// file.
fn written(data: &IncidentNeutron) -> RecordBatch {
    let dir = scratch();
    write_fast_xs(data, dir.path()).expect("this nuclide has an energy grid");
    section(dir.path(), "fast_xs.arrow")
}

/// The Li6 table converted, with the section read back.
fn li6_fast_xs() -> (IncidentNeutron, RecordBatch) {
    let data = li6_ace();
    let batch = written(&data);
    (data, batch)
}

/// One of the four summed channels, out of the row-major `[n_energy, 4]`
/// buffer: 0 total, 1 absorption, 2 scattering, 3 fission.
fn channel(xs: &[f64], k: usize, n_energy: usize) -> Vec<f64> {
    (0..n_energy).map(|i| xs[i * 4 + k]).collect()
}

/// Whether a reaction puts a neutron into the transport.
///
/// One line, reimplemented rather than imported: `fast_xs::emits_neutron` is
/// private, and reimplementing it is what makes the comparison worth something,
/// since it asks the parsed products the same question the sampler asks later.
fn emits_neutron(rx: &endf::Reaction) -> bool {
    rx.products.iter().any(|p| p.name == "neutron")
}

/// One reaction's parsed cross section, on the nuclide grid.
///
/// This calls `synthesis::on_grid` rather than rebuilding the zero-fill from
/// `xs.x`, deliberately: `on_grid` ignores `xs.x` entirely and clips instead of
/// panicking when `y` runs past the end of the grid (`synthesis.rs:102-110`),
/// so a reimplementation would disagree for reasons that have nothing to do
/// with the writer. The threshold offset itself is pinned independently, from
/// `y` alone, in `the_per_mt_matrices_are_the_parsed_cross_sections_on_the_nuclide_grid`.
fn parsed_on_grid(data: &IncidentNeutron, mt: i32, t: &str, n_energy: usize) -> Vec<f64> {
    let xs = data.reactions[&mt].xs[t].clone();
    synthesis::on_grid(&xs.y, xs.threshold_idx.unwrap_or(0), n_energy)
}

/// Every reaction that is not one of the synthesized sums, on the grid: the map
/// the writer hands `synthesis::synthesize` (`fast_xs.rs:206-211`).
///
/// Redundant reactions stay in, which is the writer's distinction and not an
/// oversight: the per-MT matrices skip them, the partials keep them and filter
/// only `SYNTHETIC_MTS`.
fn partials_on_grid(data: &IncidentNeutron, t: &str, n_energy: usize) -> BTreeMap<i32, Vec<f64>> {
    data.reactions
        .iter()
        .filter(|(mt, _)| !synthesis::SYNTHETIC_MTS.contains(mt))
        .filter_map(|(&mt, rx)| {
            rx.xs.get(t).map(|xs| {
                (
                    mt,
                    synthesis::on_grid(&xs.y, xs.threshold_idx.unwrap_or(0), n_energy),
                )
            })
        })
        .collect()
}

/// The log lookup table rebuilt from the grid, WITHOUT the writer's top-entry
/// override.
///
/// Not the writer's running scan restated: this asks `partition_point` for each
/// bin independently, so a scan that failed to advance, or advanced twice,
/// would show up. `saturating_sub(1)` is what the running scan's `at = 0` start
/// amounts to at bin 0, whose probe energy lands below the first grid point.
fn scanned_index(grid: &[f64]) -> Vec<i32> {
    let lo = grid[0].ln();
    let delta = (grid[grid.len() - 1].ln() - lo) / LOG_BINS as f64;
    (0..=LOG_BINS)
        .map(|bin| {
            let e = (lo + bin as f64 * delta).exp();
            grid.partition_point(|&g| g <= e).saturating_sub(1) as i32
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Constructed inputs. See the module doc: these are NOT parsed evaluations, and
// they exist only for the branches no vendored evaluation reaches.
// ---------------------------------------------------------------------------

/// The temperature the constructed nuclides below are built at.
///
/// `temperatures()` derives this string from `k_ts` rather than storing it, so
/// the key used for `data.energy` and for every reaction's `xs` has to be the
/// one `temperature_str(294.0)` writes. When it is not, no reaction has a cross
/// section at the temperature the writer iterates and the row comes out empty,
/// which is loud rather than silent.
const BUILT_T: &str = "294K";

/// One constructed reaction: a cross section covering the whole grid from index
/// 0, and the products that decide which group the writer puts it in.
///
/// `emits` names the product particles, because `emits_neutron` reads the
/// products rather than an MT table (`fast_xs.rs:52-59`); a capture channel is
/// therefore built with a photon and no neutron.
fn built_reaction(
    mt: i32,
    grid: &[f64],
    y: &[f64],
    emits: &[&str],
    redundant: bool,
) -> endf::Reaction {
    let mut rx = endf::Reaction::new(mt);
    let mut xs = endf::Tabulated1D::new(grid.to_vec(), y.to_vec());
    xs.threshold_idx = Some(0);
    rx.xs.insert(BUILT_T.to_string(), xs);
    rx.products = emits.iter().map(|p| endf::Product::new(p)).collect();
    rx.redundant = redundant;
    rx
}

/// An `IncidentNeutron` with one temperature, one grid and the reactions given.
///
/// The Z and A are arbitrary and nothing in this writer reads them.
fn built_nuclide(grid: &[f64], reactions: Vec<endf::Reaction>) -> IncidentNeutron {
    let mut data = IncidentNeutron::new(94, 239, 0);
    data.k_ts = vec![294.0 * endf::K_BOLTZMANN];
    data.energy.insert(BUILT_T.to_string(), grid.to_vec());
    for rx in reactions {
        data.reactions.insert(rx.mt, rx);
    }
    data
}

/// A failure means a nuclide with no usable energy grid produced a file instead
/// of an error, or the row set stopped following `data.temperatures()`.
///
/// WIRING ONLY on the filter. On this fixture `temperatures()` and
/// `data.energy.keys()` are the same one-element set, `["294K"]`, so the
/// `filter` in the expected expression is inert and a writer rewritten to
/// iterate `data.energy.keys()` passes this test unchanged. That rewrite is
/// discriminated by
/// `a_grid_at_a_temperature_the_nuclide_has_no_kt_for_gets_no_row`, on a
/// constructed nuclide. The row column here is also a one-row loop, so it says
/// nothing about row ordering.
///
/// What IS discriminated is the pair of refusals. A nuclide with no grid at all
/// and a nuclide whose only grid is present and empty must both be errors: a
/// regression there publishes an accelerator with a missing temperature rather
/// than failing the conversion.
#[test]
fn the_fast_xs_row_set_is_the_temperatures_that_have_a_grid() {
    let (data, batch) = li6_fast_xs();
    assert_schema_is_declared(&batch, "fast_xs.arrow");

    let expected: Vec<String> = data
        .temperatures()
        .into_iter()
        .filter(|t| data.energy.get(t).is_some_and(|g| !g.is_empty()))
        .collect();
    assert_eq!(
        expected,
        vec!["294K".to_string()],
        "the Li6 table carries one kT, 2.53e-8 MeV, which rounds to 294K"
    );
    let written: Vec<String> = (0..batch.num_rows())
        .map(|row| str_at(&batch, "temperature", row))
        .collect();
    assert_eq!(
        written, expected,
        "the temperature column is the row's only identity; every other \
         assertion in this file maps a row through it rather than by index"
    );

    // The ENDF route reaches the writer with reactions but no energy grid at
    // all, because `IncidentNeutron::from_endf` never fills `data.energy`.
    let endf_dir = scratch();
    let refused = write_fast_xs(&endf_nuclide(LI6_ENDF), endf_dir.path())
        .expect_err("an ENDF evaluation has no nuclide-wide energy grid before NJOY");
    assert!(
        refused
            .to_string()
            .contains("no energy grid to build fast_xs.arrow from"),
        "the refusal must name the missing grid, not something else: {refused}"
    );
    assert!(
        absent(endf_dir.path(), "fast_xs.arrow"),
        "a refused conversion must leave no file behind for a later pass to read"
    );

    // The other refusal: a temperature that IS in `temperatures()` and IS in
    // `data.energy`, with an empty grid behind it.
    let denormal = ace_nuclide(DENORMAL_ACE);
    let temperatures = denormal.temperatures();
    assert_eq!(
        denormal.energy.get(&temperatures[0]).map(|g| g.len()),
        Some(0),
        "the denormal fixture's value is that its grid is present and empty; \
         without that this is the same case as the ENDF one"
    );
    let denormal_dir = scratch();
    let refused = write_fast_xs(&denormal, denormal_dir.path())
        .expect_err("an empty grid is skipped, and skipping every temperature is an error");
    assert!(
        refused
            .to_string()
            .contains("no energy grid to build fast_xs.arrow from"),
        "the refusal must name the missing grid, not something else: {refused}"
    );
    assert!(
        absent(denormal_dir.path(), "fast_xs.arrow"),
        "a refused conversion must leave no file behind for a later pass to read"
    );
}

/// A failure means the accelerator gained a row for a temperature the nuclide
/// was never processed at, or mapped a row onto the wrong one.
///
/// NO VENDORED EVALUATION REACHES THIS AND THE INPUT IS CONSTRUCTED, not
/// parsed: the ACE route has no 0 K grid and the ENDF route has no grid at all,
/// so `temperatures()` and `data.energy.keys()` are the same set on every
/// fixture in the tree, and every fixture has exactly one temperature. The NJOY
/// route is the one that differs: `entry.rs:225-232` puts a 0 K grid into
/// `data.energy` that is never in `temperatures()`, and a writer that iterated
/// the map would publish a row that every consumer then maps to a temperature
/// the nuclide has no cross sections at.
///
/// The two real temperatures are given DIFFERENT grid lengths and DIFFERENT
/// elastic cross sections on purpose, so the row-to-temperature mapping is
/// discriminated as well as the row set: `temperatures()` follows the order of
/// `k_ts` and gives `["294K", "1200K"]`, where the map's own lexicographic walk
/// gives `["0K", "1200K", "294K"]`.
#[test]
fn a_grid_at_a_temperature_the_nuclide_has_no_kt_for_gets_no_row() {
    let warm = [1.0e-5, 1.0e0, 1.0e3, 1.0e6];
    let hot = [2.0e-5, 3.0e0, 4.0e3, 5.0e6, 6.0e7];
    // The 0 K grid NJOY leaves behind. Nothing has a cross section at it.
    let unprocessed = [9.0e-4, 8.0e0, 7.0e4];
    let warm_elastic = [10.0, 11.0, 12.0, 13.0];
    let hot_elastic = [20.0, 21.0, 22.0, 23.0, 24.0];

    let mut data = IncidentNeutron::new(3, 6, 0);
    data.k_ts = vec![294.0 * endf::K_BOLTZMANN, 1200.0 * endf::K_BOLTZMANN];
    data.energy.insert("294K".to_string(), warm.to_vec());
    data.energy.insert("1200K".to_string(), hot.to_vec());
    data.energy.insert("0K".to_string(), unprocessed.to_vec());

    let mut elastic = endf::Reaction::new(2);
    elastic.products = vec![endf::Product::new("neutron")];
    for (t, grid, y) in [
        ("294K", warm.as_slice(), warm_elastic.as_slice()),
        ("1200K", hot.as_slice(), hot_elastic.as_slice()),
    ] {
        let mut xs = endf::Tabulated1D::new(grid.to_vec(), y.to_vec());
        xs.threshold_idx = Some(0);
        elastic.xs.insert(t.to_string(), xs);
    }
    data.reactions.insert(2, elastic);

    // The premise, pinned so this cannot quietly become the same set twice.
    assert_eq!(
        data.temperatures(),
        ["294K", "1200K"],
        "temperatures() follows the order of k_ts, which is not the order the \
         energy map is keyed in"
    );
    assert_eq!(
        data.energy.keys().collect::<Vec<_>>(),
        ["0K", "1200K", "294K"],
        "the energy map carries a grid at a temperature temperatures() does \
         not, and walks its keys lexicographically"
    );

    let batch = written(&data);
    let rows: Vec<String> = (0..batch.num_rows())
        .map(|row| str_at(&batch, "temperature", row))
        .collect();
    assert_eq!(
        rows,
        ["294K", "1200K"],
        "the 0K grid must produce no row, and the rows must come out in the \
         order temperatures() gives rather than the order the map is keyed in"
    );

    // Row for row, with the position pinned by the equality above.
    for (row, (grid, elastic_xs)) in [
        (warm.as_slice(), warm_elastic.as_slice()),
        (hot.as_slice(), hot_elastic.as_slice()),
    ]
    .into_iter()
    .enumerate()
    {
        let t = &rows[row];
        assert_f64_slice_eq(
            &format!("energy at {t}"),
            &f64_list(&batch, "energy", row),
            grid,
        );
        let xs = f64_list(&batch, "xs", row);
        assert_f64_slice_eq(
            &format!("xs total at {t}"),
            &channel(&xs, 0, grid.len()),
            elastic_xs,
        );
    }
}

/// A failure means the sampler's binary search starts in the wrong place.
///
/// If an entry is too high, `FastXSGrid::lookup` starts its narrowed search
/// past the answer, `partition_point` returns 0, `i_grid` pins at `i_low`, and
/// the interpolation fraction goes negative: it extrapolates backwards off a
/// pair that does not bracket E, which at a resonance edge returns a wildly
/// wrong or a negative cross section. Nothing raises, because
/// `log_grid_index_u32` (`nuclide_arrow.rs:714-753`) checks only that entries
/// are in range and non-decreasing and deliberately does not require the last
/// entry to be `n_energy - 1`. The energy column is checked here too because
/// the index means nothing without the grid it indexes.
///
/// Two things this does NOT do, both of which it used to claim.
///
/// It no longer loops over the grid asserting that each point lands inside its
/// bin's bracket. With the index pinned entry for entry against the scan below,
/// that loop is a fact about the fixture's grid rather than about the writer,
/// and could not fail while the equality held.
///
/// It does not discriminate the scan's boundary comparison. The writer advances
/// while `energy[at + 1] <= e`, and changing that to `<` leaves this test green,
/// because no bin's probe energy coincides exactly with a grid point on Li6.
/// That variant starts the search one point low, which the writer's own comment
/// at `fast_xs.rs:98-101` says is the safe direction, so it is recorded here
/// rather than built for.
#[test]
fn the_log_lookup_index_is_the_last_grid_point_at_or_below_every_bin() {
    let (data, batch) = li6_fast_xs();
    let temperature = str_at(&batch, "temperature", 0);
    let grid = &data.energy[&temperature];
    let n = grid.len();
    assert_eq!(n, 721, "the Li6 ACE grid is 721 points");

    // The grid itself: a verbatim clone of the parsed one, so exact.
    let energy = f64_list(&batch, "energy", 0);
    assert_f64_slice_eq("energy", &energy, grid);

    // One `ln` of one grid point, and one reciprocal of one difference. Both
    // exact: a tolerance here could only hide a wrong grid point or a wrong
    // bin count.
    let log_e_min = f64_at(&batch, "log_e_min", 0);
    assert_eq!(
        log_e_min,
        grid[0].ln(),
        "log_e_min is ln of the first energy"
    );
    assert!(
        log_e_min.exp() < grid[0],
        "bin 0's probe energy, {:e}, is expected to land BELOW the first grid \
         point, {:e}; the index recomputation's saturating_sub depends on it",
        log_e_min.exp(),
        grid[0]
    );
    let inv_log_delta = f64_at(&batch, "inv_log_delta", 0);
    let delta = (grid[n - 1].ln() - grid[0].ln()) / LOG_BINS as f64;
    assert_eq!(inv_log_delta, 1.0 / delta, "inv_log_delta is 1 / delta");

    let index = i32_list(&batch, "log_grid_index", 0);
    assert_eq!(
        index.len(),
        LOG_BINS + 1,
        "the lookup reads bin and bin + 1, so the table is LOG_BINS + 1 long"
    );

    // Value for value against an independent reconstruction.
    let mut expected = scanned_index(grid);
    assert_eq!(
        expected[LOG_BINS], 719,
        "this fixture is worth using because exp(ln(e_max)) lands at \
         199999999.99999928 eV, below grid[720] = 2e8, so the scan yields 719 \
         and only the deliberate override at fast_xs.rs:102-104 puts 720 in the \
         file; if the scan ever reaches 720 on its own, the override has stopped \
         being exercised here"
    );
    expected[LOG_BINS] = (n - 1) as i32;
    assert_i32_slice_eq("log_grid_index", &index, &expected);
}

/// The refusal that closed the defect this test used to pin.
///
/// `write_fast_xs` used to refuse only an EMPTY grid and accept a grid with no
/// positive extent, which is what `synthetic-urr.ace.xz` parses to: four zeros.
/// The accelerator it wrote was poisoned. `log_e_min` was `ln(0.0)`, negative
/// infinity, and `inv_log_delta` was `1.0 / ((-inf) - (-inf))`, NaN.
///
/// Nothing downstream caught it, which is what made it worth closing.
/// `log_grid_index_u32` (`nuclide_arrow.rs:714-753`) checks only that the
/// entries are in range and non-decreasing, and all 8001 of them were, so the
/// file loaded. `FastXSGrid::lookup` then computes
/// `((ln E - (-inf)) * NaN) as usize`, and a NaN cast to `usize` saturates to 0
/// in Rust, so every energy silently took bin 0's bracket for the whole grid.
/// No error, no warning, and the transport kept running.
#[test]
fn a_grid_with_no_positive_extent_is_refused() {
    let data = ace_nuclide(URR_ACE);
    let temperatures = data.temperatures();
    let grid = &data.energy[&temperatures[0]];
    assert_f64_slice_eq("the fixture's own grid", grid, &[0.0; 4]);

    let dir = scratch();
    let refused = write_fast_xs(&data, dir.path())
        .expect_err("a grid with no positive extent cannot be indexed logarithmically");
    let message = refused.to_string();
    assert!(
        message.contains("cannot be indexed logarithmically"),
        "the refusal must name why the grid was rejected: {message}"
    );
    assert!(
        message.contains(&data.name()),
        "the refusal must name the nuclide, so a directory-wide conversion \
         says which file to look at: {message}"
    );
    assert!(
        message.contains(&temperatures[0]),
        "the refusal must name the temperature, since a nuclide's other \
         temperatures may be fine: {message}"
    );
    assert!(
        absent(dir.path(), "fast_xs.arrow"),
        "a refused conversion must leave no file behind for a later pass to read"
    );
}

/// A failure means one of total, absorption, scattering or fission is not the
/// sum of the channels it is defined over, so a collision would sample a
/// reaction out of a set whose probabilities do not add up to the total it
/// flew a free path against.
///
/// What each assertion is worth, plainly.
///
/// Total against `synthesize(&partials)[&1]`, absorption against
/// `synthesize(&partials)[&27]` less the written fission, and scattering
/// against `elastic + non_elastic_scattering(&partials)` all run the same
/// functions the writer ran (`synthesis.rs:218-219` is `partials[2] + mt3`,
/// `fast_xs.rs:270` is `elastic + non_elastic`). They pin the value written
/// against the expression and prove nothing about the arithmetic.
///
/// Total against the ACE table's own MT 1 is the one independent oracle here,
/// and its bound is 1e-7 relative rather than the 1e-6 it was. The measured
/// worst residual is 4.252890222678634e-9 at 573554.7 eV, so 1e-7 still leaves
/// 23x of headroom, and 1e-6 was loose enough that scaling every term inside
/// `synthesis::non_elastic_scattering` by `(1.0 + 1e-6)` passed. That defect
/// moves `total` and `scattering` together, so the cross-grouping identity
/// below is blind to it and this bound is the only thing that can see it: at
/// 2e8 eV the non-elastic scattering is 73.6% of the total, so the scaling
/// shows up as a 7.36e-7 residual.
///
/// Absorption against the ACE table's own MT 101 is independent (MT 101 comes
/// straight off the ESZ block, `incident_neutron.rs:150-166`) but DEGENERATE:
/// MT 102 is Li6's only disappearance channel, so it compares a one-term sum
/// against its one term. The subtraction that makes absorption differ from
/// MT 27 is undiscriminated here because fission is zero everywhere; the
/// fissile case is
/// `the_absorption_column_is_the_disappearance_sum_less_fission`, on a
/// constructed nuclide.
///
/// The cross-grouping identity is the check that survives a channel landing on
/// one side only. It is deliberately not exact, per `fast_xs.rs:263-267`: an
/// identity that holds because one side was defined as the other cannot detect
/// that the side it was defined from is wrong.
#[test]
fn the_four_summed_channels_are_the_curves_they_are_summed_from() {
    let (data, batch) = li6_fast_xs();
    let temperature = str_at(&batch, "temperature", 0);
    let grid = &data.energy[&temperature];
    let n = grid.len();

    // The reader takes n_energy from xs_shape[0] and zero-fills past the end of
    // the flat buffer (nuclide_arrow.rs:806-816), so a short shape truncates
    // the nuclide's grid and a long one appends rows of zero cross section.
    let xs_shape = i32_list(&batch, "xs_shape", 0);
    assert_i32_slice_eq("xs_shape", &xs_shape, &[n as i32, 4]);
    let xs = f64_list(&batch, "xs", 0);
    assert_eq!(
        xs.len(),
        (xs_shape[0] * xs_shape[1]) as usize,
        "the flat xs buffer must hold exactly the shape it declares"
    );

    let total = channel(&xs, 0, n);
    let absorption = channel(&xs, 1, n);
    let scattering = channel(&xs, 2, n);
    let fission = channel(&xs, 3, n);

    let partials = partials_on_grid(&data, &temperature, n);
    let synthesized = synthesis::synthesize(&partials, n);
    let elastic = parsed_on_grid(&data, 2, &temperature, n);

    // Total: the writer's own arithmetic restated, then the real check against
    // the table's own stored MT 1. Not exact, and it cannot be: the stored copy
    // reaches the file through ENDF's 11-character field and so carries seven
    // significant digits where the re-derived sum carries full precision.
    assert_f64_slice_eq("xs total", &total, &synthesized[&1]);
    assert_f64_slice_close(
        "xs total against the ACE table's own MT 1",
        &total,
        &parsed_on_grid(&data, 1, &temperature, n),
        1e-7,
    );

    // Absorption is MT 27 less what fissions, which is the disappearance sum.
    let want_absorption: Vec<f64> = (0..n).map(|i| synthesized[&27][i] - fission[i]).collect();
    assert_f64_slice_eq("xs absorption", &absorption, &want_absorption);
    // The comment at fast_xs.rs:226-228 says the derived absorption and the
    // table's own MT 101 disagree by 5e-5 barns on Li6. On this vendored table
    // the measured difference is exactly zero at all 721 points, so this is
    // asserted exactly and the comment is what needs revisiting.
    assert_f64_slice_eq(
        "xs absorption against the ACE table's own MT 101",
        &absorption,
        &parsed_on_grid(&data, 101, &temperature, n),
    );

    // Scattering is elastic plus the neutron-emitting part of MT 3, summed from
    // non-negative terms rather than reached by subtracting absorption back off
    // MT 3. Restated arithmetic again.
    let non_elastic_scatter = synthesis::non_elastic_scattering(&partials, n);
    let want_scattering: Vec<f64> = (0..n)
        .map(|i| elastic[i] + non_elastic_scatter[i])
        .collect();
    assert_f64_slice_eq("xs scattering", &want_scattering, &scattering);

    // Li6 is not fissile, so the fourth slot is zeros. Against the literal
    // rather than against a row sum over a matrix with no columns, which was
    // the same literal reached the long way round.
    assert_f64_slice_eq("xs fission", &fission, &vec![0.0; n]);

    // The cross-grouping identity, at every energy. The bound is a few ulp of
    // TOTAL, the largest term, not of the partial being compared: absorption
    // comes out of a subtraction whose error scales with absorption plus
    // fission. `m` is the number of channels that entered the sums, and the
    // extra 1 pays for that subtraction. Measured worst on this fixture is 0.97
    // ulp of total, at 5400000.0 eV, so 10 * EPSILON leaves an order of
    // magnitude of headroom while staying about 4.5e5 times tighter than the
    // 1e-9 in use at crates/yamc/tests/urr_fissile_capture.rs:52-77, which
    // tolerates a whole channel vanishing from one side.
    let scatter_mts = i32_list(&batch, "scatter_mt_numbers", 0);
    let fission_mts = i32_list(&batch, "fission_mt_numbers", 0);
    let disappearance = endf::data::sum_rule(101)
        .unwrap_or(&[])
        .iter()
        .filter(|mt| data.reactions.contains_key(mt))
        .count();
    let m = scatter_mts.len() + fission_mts.len() + disappearance;
    for i in 0..n {
        let sum = absorption[i] + scattering[i] + fission[i];
        let residual = (total[i] - sum).abs();
        if total[i] == 0.0 {
            assert_eq!(
                sum, 0.0,
                "at {:e} eV the total is zero but its channels sum to {sum:e}",
                grid[i]
            );
            continue;
        }
        let bound = ((m + 1) as f64) * f64::EPSILON * total[i];
        assert!(
            residual <= bound,
            "at {:e} eV total is {:e} but absorption + scattering + fission is \
             {sum:e}, off by {residual:e}, which is {:.1} ulp of total over {} \
             channels: the two sides are the same channels in a different \
             order, so anything above the rounding of the largest term means a \
             channel is on one side only",
            grid[i],
            total[i],
            residual / (f64::EPSILON * total[i]),
            m
        );
    }

    // Radiative capture, against the parsed channel it is a copy of. The
    // comparison this used to make beside it, against the absorption slice, was
    // a self-comparison: MT 102 is Li6's only member of sum_rule(101), so
    // absorption IS on_grid(102) bit for bit and the two sides were one value.
    let ngamma = f64_list(&batch, "xs_ngamma", 0);
    assert_f64_slice_eq(
        "xs_ngamma",
        &ngamma,
        &parsed_on_grid(&data, 102, &temperature, n),
    );

    // Format parity only: the reader recomputes photon production whenever this
    // column is empty or all zeros (nuclide_arrow.rs:1012-1014), so no
    // transport answer depends on it. It is pinned because it is the one
    // list<double> whose contents are constant, which makes it the cheapest
    // tripwire for a positional swap among the six interchangeable ones.
    assert!(
        !is_null(&batch, "photon_prod", 0),
        "photon_prod is written as zeros, not as a null"
    );
    let photon_prod = f64_list(&batch, "photon_prod", 0);
    assert_f64_slice_eq("photon_prod", &photon_prod, &vec![0.0; n]);
}

/// A failure means every fission is counted twice against the total. That is
/// the wrong answer `fast_xs.rs:18-20` names outright: absorption is MT 27 LESS
/// fission, and MT 27 is the one that includes it.
///
/// NO VENDORED EVALUATION REACHES THIS AND THE INPUT IS CONSTRUCTED, not
/// parsed. Li6 is not fissile and no other ACE fixture carries a reaction set
/// at all, so on parsed data the fission column is zero everywhere and the
/// `- fission[i]` at `fast_xs.rs:229-231` can be deleted with every other test
/// in this file still green. Nothing here says anything about parity with a
/// real evaluation.
///
/// The fields are given distinct values for exactly that reason. Elastic,
/// capture and fission are 10, 4 and 1 barns at the first grid point and rise
/// differently, so absorption cannot be confused with capture plus fission, and
/// the two fission partials differ from each other at every energy, so a
/// swapped column or a column-major ravel in the fission matrix cannot hide
/// behind an equality. Every value is an exact binary fraction, so the writer's
/// association order and this test's agree bit for bit and the comparisons stay
/// exact.
///
/// Both shapes an evaluation may take are built: MT 18 alone, which is
/// `has_partial_fission == false`, and MT 19 beside MT 20, which is `true`. The
/// flag gates a random draw (`sample_fission_reaction` skips `draw_xi()` when it
/// is false, `yamc-nuclide/src/nuclide.rs:729-733`), so flipping it moves the
/// shared PCG stream out of step with the GPU twin.
#[test]
fn the_absorption_column_is_the_disappearance_sum_less_fission() {
    let grid = [1.0e-5, 1.0e0, 1.0e3, 1.0e6];
    let n = grid.len();
    let elastic = [10.0, 11.0, 12.0, 13.0];
    let capture = [4.0, 4.5, 5.0, 5.5];

    // The evaluation gives the fission total, MT 18.
    let mt18 = [1.0, 2.0, 3.0, 4.0];
    let batch = written(&built_nuclide(
        &grid,
        vec![
            built_reaction(2, &grid, &elastic, &["neutron"], false),
            built_reaction(18, &grid, &mt18, &["neutron"], false),
            built_reaction(102, &grid, &capture, &["photon"], false),
        ],
    ));
    assert_i32_slice_eq("xs_shape", &i32_list(&batch, "xs_shape", 0), &[n as i32, 4]);
    let xs = f64_list(&batch, "xs", 0);

    // The whole point of this test: absorption is the capture channel, and NOT
    // capture plus the 1 to 4 barns of fission beside it.
    assert_f64_slice_eq("xs absorption", &channel(&xs, 1, n), &capture);
    assert_f64_slice_eq("xs fission", &channel(&xs, 3, n), &mt18);
    assert_f64_slice_eq("xs scattering", &channel(&xs, 2, n), &elastic);
    let total: Vec<f64> = (0..n).map(|i| elastic[i] + capture[i] + mt18[i]).collect();
    assert_f64_slice_eq("xs total", &channel(&xs, 0, n), &total);

    assert_i32_slice_eq(
        "fission_mt_numbers",
        &i32_list(&batch, "fission_mt_numbers", 0),
        &[18],
    );
    assert_f64_slice_eq(
        "fission_mt_xs",
        &f64_list(&batch, "fission_mt_xs", 0),
        &mt18,
    );
    assert_i32_slice_eq(
        "fission_mt_shape",
        &i32_list(&batch, "fission_mt_shape", 0),
        &[n as i32, 1],
    );
    assert!(
        !bool_at(&batch, "has_partial_fission", 0),
        "MT 18 is the fission total, not a chance-by-chance partial"
    );

    // The same nuclide with the first and second chance partials instead.
    let mt19 = [1.0, 2.0, 3.0, 4.0];
    let mt20 = [0.25, 0.5, 0.75, 1.0];
    let batch = written(&built_nuclide(
        &grid,
        vec![
            built_reaction(2, &grid, &elastic, &["neutron"], false),
            built_reaction(19, &grid, &mt19, &["neutron"], false),
            built_reaction(20, &grid, &mt20, &["neutron"], false),
            built_reaction(102, &grid, &capture, &["photon"], false),
        ],
    ));
    let xs = f64_list(&batch, "xs", 0);
    let fission: Vec<f64> = (0..n).map(|i| mt19[i] + mt20[i]).collect();
    assert_f64_slice_eq("xs absorption, partials", &channel(&xs, 1, n), &capture);
    assert_f64_slice_eq("xs fission, partials", &channel(&xs, 3, n), &fission);
    let total: Vec<f64> = (0..n)
        .map(|i| elastic[i] + capture[i] + fission[i])
        .collect();
    assert_f64_slice_eq("xs total, partials", &channel(&xs, 0, n), &total);

    assert_i32_slice_eq(
        "fission_mt_numbers",
        &i32_list(&batch, "fission_mt_numbers", 0),
        &[19, 20],
    );
    // Row-major [n_energy, 2], with the two columns different at every energy.
    let expected: Vec<f64> = (0..n).flat_map(|i| [mt19[i], mt20[i]]).collect();
    assert_f64_slice_eq(
        "fission_mt_xs",
        &f64_list(&batch, "fission_mt_xs", 0),
        &expected,
    );
    assert_i32_slice_eq(
        "fission_mt_shape",
        &i32_list(&batch, "fission_mt_shape", 0),
        &[n as i32, 2],
    );
    assert!(
        bool_at(&batch, "has_partial_fission", 0),
        "MT 19 and MT 20 are the chance-by-chance partials"
    );
}

/// A failure means the matrix that answers "what is MT 51 at this energy" is
/// sheared or column-shifted, which loads without complaint and hands every
/// reaction someone else's cross section.
///
/// `shape[1]` is the row stride the reader de-ravels with
/// (`nuclide_arrow.rs:860-868`), so a stride one too large or too small returns
/// a real, plausible cross section from the wrong (energy, MT) cell at every
/// lookup.
///
/// Two parts of the expected set are WIRING ONLY on this fixture. The
/// `!rx.redundant` filter decides nothing: Li6's redundant reactions are 1, 101,
/// 203 to 207, 301 and 444, and every one of them also fails `emits_neutron`,
/// so deleting the writer's guard changes nothing here. The discriminating case
/// is `the_per_mt_columns_leave_out_a_redundant_total_whose_own_levels_are_present`,
/// on a constructed nuclide. And the fission group compares an empty set against
/// a predicate that yields the empty set, so it says nothing about fission; the
/// non-empty case, including the `has_partial_fission` truth table, is in
/// `the_absorption_column_is_the_disappearance_sum_less_fission`. It is still
/// asserted because empty is a different file from null.
///
/// The MT 51 pin below is the only assertion ON A PARSED EVALUATION that
/// survives a defect in `synthesis::on_grid`. Everything else that runs on Li6,
/// including the ACE MT 1 oracle in the test above, routes through `on_grid` on
/// both sides and cannot see a threshold shifted inside it. The two constructed
/// tests survive one too, but for a different reason: their expected values are
/// literals rather than anything `on_grid` produced.
#[test]
fn the_per_mt_matrices_are_the_parsed_cross_sections_on_the_nuclide_grid() {
    let (data, batch) = li6_fast_xs();
    let temperature = str_at(&batch, "temperature", 0);
    let n = data.energy[&temperature].len();

    let expected_scatter: Vec<i32> = data
        .reactions
        .iter()
        .filter(|(mt, rx)| {
            !rx.redundant
                && rx.xs.contains_key(&temperature)
                && emits_neutron(rx)
                && !FISSION_MTS.contains(mt)
        })
        .map(|(&mt, _)| mt)
        .collect();
    let scatter_mts = i32_list(&batch, "scatter_mt_numbers", 0);
    assert_i32_slice_eq("scatter_mt_numbers", &scatter_mts, &expected_scatter);
    assert_i32_slice_eq(
        "scatter_mt_numbers",
        &scatter_mts,
        &[2, 5, 51, 52, 53, 54, 55, 91],
    );

    let scatter_mt_shape = i32_list(&batch, "scatter_mt_shape", 0);
    assert_i32_slice_eq(
        "scatter_mt_shape",
        &scatter_mt_shape,
        &[n as i32, scatter_mts.len() as i32],
    );
    let scatter_mt_xs = f64_list(&batch, "scatter_mt_xs", 0);
    assert_eq!(
        scatter_mt_xs.len(),
        n * scatter_mts.len(),
        "the flat matrix must hold exactly the shape it declares"
    );

    // Every cell, against the parsed cross section it is a copy of. Exact:
    // there is no arithmetic between the evaluation and this buffer.
    let stride = scatter_mts.len();
    for (j, &mt) in scatter_mts.iter().enumerate() {
        let expected = parsed_on_grid(&data, mt, &temperature, n);
        let written: Vec<f64> = (0..n).map(|i| scatter_mt_xs[i * stride + j]).collect();
        assert_f64_slice_eq(
            &format!("scatter_mt_xs column {j} (MT {mt})"),
            &written,
            &expected,
        );
    }

    // The threshold zero-fill, pinned on a real threshold channel from the
    // parsed `y` alone rather than through `on_grid`. MT 51 starts at grid
    // index 511 with 210 points, and its first strictly positive value is
    // y[1], so the curve must sit at grid index 512 and nowhere else. A
    // one-point shift in either direction moves that edge.
    let mt51 = data.reactions[&51].xs[&temperature].clone();
    let threshold = mt51.threshold_idx.expect("MT 51 is a threshold reaction");
    assert_eq!(
        (threshold, mt51.y.len()),
        (511, 210),
        "MT 51 is this fixture's threshold channel: 210 points starting at grid \
         index 511. If that ever changes, the zero-fill below is checking \
         nothing"
    );
    let j51 = scatter_mts
        .iter()
        .position(|&mt| mt == 51)
        .expect("MT 51 has a column");
    let column: Vec<f64> = (0..n).map(|i| scatter_mt_xs[i * stride + j51]).collect();
    assert!(
        column[..threshold].iter().all(|&v| v == 0.0),
        "MT 51 must be exactly zero below its threshold at grid index {threshold}"
    );
    let parsed_first_positive = mt51
        .y
        .iter()
        .position(|&v| v > 0.0)
        .expect("MT 51 rises above zero somewhere");
    let first_positive = column
        .iter()
        .position(|&v| v > 0.0)
        .expect("so its column must too");
    assert_eq!(
        first_positive,
        threshold + parsed_first_positive,
        "MT 51's first positive point must land at the grid index its threshold \
         and its own array put it at"
    );
    assert_eq!(
        column[first_positive], mt51.y[parsed_first_positive],
        "and carry the parsed value"
    );
    assert_f64_slice_eq(
        "scatter_mt_xs MT 51 above its threshold",
        &column[threshold..],
        &mt51.y,
    );

    // Fission. Empty and PRESENT, not null: `int_lists` and `float_lists`
    // (sections.rs:138-155) append unconditionally, and the or_null variants
    // used by other sections would produce a different file that loads the same.
    let expected_fission: Vec<i32> = data
        .reactions
        .iter()
        .filter(|(mt, rx)| {
            !rx.redundant
                && rx.xs.contains_key(&temperature)
                && emits_neutron(rx)
                && FISSION_MTS.contains(mt)
        })
        .map(|(&mt, _)| mt)
        .collect();
    assert!(
        expected_fission.is_empty(),
        "Li6 is not fissile; if this fixture ever gains a fission channel the \
         assertions below stop being the empty case"
    );
    let fission_mts = i32_list(&batch, "fission_mt_numbers", 0);
    assert_i32_slice_eq("fission_mt_numbers", &fission_mts, &expected_fission);
    assert!(
        !is_null(&batch, "fission_mt_numbers", 0) && !is_null(&batch, "fission_mt_xs", 0),
        "the fission columns are empty lists, not nulls"
    );
    assert_eq!(list_len(&batch, "fission_mt_xs", 0), 0);
    assert_i32_slice_eq(
        "fission_mt_shape",
        &i32_list(&batch, "fission_mt_shape", 0),
        &[n as i32, 0],
    );
    assert_eq!(
        bool_at(&batch, "has_partial_fission", 0),
        fission_mts.iter().any(|&mt| mt != 18),
        "has_partial_fission is true only when the evaluation gives the \
         chance-by-chance partials rather than the MT 18 total"
    );
}

/// A failure means a reaction is counted twice in the per-MT matrix the sampler
/// picks from, so the same (n,2n) can be drawn under two MT numbers and the
/// column probabilities do not add up to the scattering channel beside them.
///
/// NO VENDORED EVALUATION REACHES THIS AND THE INPUT IS CONSTRUCTED, not
/// parsed. Deleting the `if rx.redundant { continue; }` guard at
/// `fast_xs.rs:169-171` leaves every test that runs on Li6 green, because every
/// redundant Li6 reaction also fails `emits_neutron` and so is dropped one line
/// later anyway.
///
/// The shape built here is the one `synthesis.rs:124-134` names. JEFF-4.0's Be9
/// has no evaluated MF=3 MT 16, so the parser builds MT 16 as the sum of the
/// residual levels MT 875 and MT 876 and flags it redundant. Both forms emit a
/// neutron, so the flag is the only thing that separates them, and counting
/// both put MT 3 (and with it the total) 35% above the evaluation's own MT 1.
/// The levels are given different values from each other and from the elastic
/// channel so a wrong column cannot land on a right number.
#[test]
fn the_per_mt_columns_leave_out_a_redundant_total_whose_own_levels_are_present() {
    let grid = [1.0e-5, 1.0e0, 1.0e3, 1.0e6];
    let n = grid.len();
    let elastic = [10.0, 11.0, 12.0, 13.0];
    let level0 = [1.0, 1.25, 1.5, 1.75];
    let level1 = [2.0, 2.5, 3.0, 3.5];
    let capture = [4.0, 4.5, 5.0, 5.5];
    let n2n: Vec<f64> = (0..n).map(|i| level0[i] + level1[i]).collect();

    let batch = written(&built_nuclide(
        &grid,
        vec![
            built_reaction(2, &grid, &elastic, &["neutron"], false),
            built_reaction(16, &grid, &n2n, &["neutron"], true),
            built_reaction(875, &grid, &level0, &["neutron"], false),
            built_reaction(876, &grid, &level1, &["neutron"], false),
            built_reaction(102, &grid, &capture, &["photon"], false),
        ],
    ));

    assert_i32_slice_eq(
        "scatter_mt_numbers",
        &i32_list(&batch, "scatter_mt_numbers", 0),
        &[2, 875, 876],
    );
    assert_i32_slice_eq(
        "scatter_mt_shape",
        &i32_list(&batch, "scatter_mt_shape", 0),
        &[n as i32, 3],
    );
    let expected: Vec<f64> = (0..n)
        .flat_map(|i| [elastic[i], level0[i], level1[i]])
        .collect();
    assert_f64_slice_eq(
        "scatter_mt_xs",
        &f64_list(&batch, "scatter_mt_xs", 0),
        &expected,
    );

    // And the summed channels above the matrix count (n,2n) once, which is the
    // consequence the missing column would otherwise show up as.
    let xs = f64_list(&batch, "xs", 0);
    let scattering: Vec<f64> = (0..n).map(|i| elastic[i] + n2n[i]).collect();
    assert_f64_slice_eq("xs scattering", &channel(&xs, 2, n), &scattering);
    let total: Vec<f64> = (0..n).map(|i| scattering[i] + capture[i]).collect();
    assert_f64_slice_eq("xs total", &channel(&xs, 0, n), &total);
}
