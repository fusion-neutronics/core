//! `total_nu.arrow` and `fission_photon.arrow`, column by column, against the
//! evaluation the writer was handed.
//!
//! Every column of both sections is a verbatim clone of something the parser
//! already built: `Vec::clone`, `extend_from_slice`, or a literal role string.
//! No arithmetic happens in `crates/yamc-convert/src/fission_nu.rs` at all, so
//! every numeric assertion here is exact equality and a tolerance anywhere
//! would only hide the wrong-slot and wrong-unit mistakes these tests exist to
//! find. The one arithmetic step anywhere in the chain is upstream, the NPLY=2
//! unit correction at `crates/endf/src/fission_energy.rs:132-138`, and nothing
//! here reaches it: a test that tried was dropped because the guard does not
//! gate the way its own comment says, which is its own issue rather than
//! something to pin from here.
//!
//! Neither section is reachable through `entry::convert_neutron_transport`
//! without NJOY, so both writers are called directly. That loses nothing:
//! `entry.rs:257` builds the release with `heating::fission_energy_release`,
//! and `entry.rs:402-405` then calls `write_total_nu` on the `IncidentNeutron`
//! unchanged and `write_fission_photon` on that release, behind
//! `if let Some(release)`. The only production step not exercised is the NJOY
//! run that fills the energy grids, which neither writer reads.
//!
//! # Two kinds of input, and they are not interchangeable
//!
//! The first two tests are handed VENDORED evaluations, and are parity checks
//! against them. Everything below the CONSTRUCTED INPUTS banner is handed input
//! this file BUILDS BY HAND, and is a parity check against nothing: no vendored
//! evaluation reaches those branches, so a constructed `IncidentNeutron`, a
//! constructed `FissionEnergyRelease`, or a parsed `Material` with one
//! coefficient replaced, is the only way to reach them at all. Each says so in
//! its own doc comment, and each gives its constructed fields values that
//! differ from every constant the writer might hard-code, because two fields
//! that happen to be equal on a fixture cannot tell a copy from a constant.
//!
//! Two of them go further and pin behaviour that is not evaluation parity in
//! any sense: which of two candidates the writer picks when an input carries
//! both, where no evaluation carries both and there is no evaluated answer.
//! Those two say in their own doc comments which part of what they assert is a
//! fact about fission and which part is a choice being held still.

mod section_values;
use section_values::*;

use endf::fission_energy::Component;
use endf::mf::mf1::FissionEnergyRelease as Mt458Component;
use endf::product::Yield;
use endf::{Polynomial, Tabulated1D};

/// The index of one MF=1/MT=458 component in the raw section, by its ENDF-102
/// name.
///
/// Read off `FISSION_ENERGY_COMPONENTS` rather than hard coded, so the
/// comparison follows the format's own ordering instead of a transcribed 3
/// and 4.
fn component_index(name: &str) -> usize {
    endf::mf::mf1::FISSION_ENERGY_COMPONENTS
        .iter()
        .position(|&n| n == name)
        .unwrap_or_else(|| panic!("no MF=1/MT=458 component named {name}"))
}

/// The written fission multiplicity must be the evaluation's TOTAL nu-bar.
///
/// A failure means one of two things. Either the multiplicity written is the
/// prompt yield rather than the total, which is issue #364 and a roughly 1.6%
/// undercount of fission neutrons for U235 with nothing on disk to say so, or
/// the tabulated pair was reshaped on the way out.
///
/// Nothing downstream would notice most of this. `parse_fission_nu`
/// (`nuclide_arrow.rs:1935-1967`) reads only `yield_type`, `yield_data` and
/// `yield_shape`, so a load round trip cannot see `particle`, `emission_mode`,
/// `decay_rate`, `yield_breakpoints` or `yield_interpolation` at all, and a
/// direct comparison is the only thing that can.
///
/// What this FIXTURE can and cannot separate, measured rather than assumed:
///
/// - `emission_mode` is discriminated: the derived total says "total" where the
///   prompt product beside it says "prompt", so reading the wrong product fails.
/// - `yield_breakpoints` and `yield_interpolation` are discriminated: they are
///   adjacent `list<int32>` columns filled from adjacent lines of the writer,
///   and their values differ here ([85] against [2]), so pushing them in the
///   wrong order fails rather than passing as a matched pair.
/// - `particle` and `decay_rate` are NOT. U235's derived total and its prompt
///   product both carry the name "neutron" and a zero decay rate, which are
///   also the two constants a writer would reach for, so a writer that
///   hard-coded either would pass every assertion below. All the comparisons
///   here defend is the `particle`/`emission_mode` transposition, since those
///   two adjacent utf8 columns do differ on this fixture. The hard-coding is
///   caught in `total_nu_columns_are_the_products_own_fields_on_every_fission_mt`,
///   which has to construct its input to do it.
/// - The writer's search across every fission MT rather than MT 18 alone
///   (`fission_nu.rs:26-41`) is not exercised: U235 puts fission on MT 18, as
///   does every other vendored fissile evaluation. Same constructed test.
///
/// The `Yield::Polynomial` arm of `write_total_nu` (`fission_nu.rs:44-50`) is
/// not reached here either: every derived total-nu product in every vendored
/// fixture is `Yield::Tabulated`. It is covered by
/// `total_nu_polynomial_yield_is_written_as_bare_coefficients`, again from a
/// constructed input.
#[test]
fn total_nu_is_the_derived_total_and_not_the_prompt_yield() {
    let data = endf_nuclide(U235_ENDF);
    let fission = data.reactions.get(&18).expect("U235 gives MT 18");
    let total = fission
        .derived_products
        .first()
        .expect("MF=1/MT=452 is attached to MT 18 as a derived product");

    let dir = scratch();
    assert!(
        yamc_convert::fission_nu::write_total_nu(&data, dir.path()).expect("the writer runs"),
        "U235 carries MF=1/MT=452 beside MF=1/MT=456, so a total nu-bar must be written"
    );
    let batch = section(dir.path(), "total_nu.arrow");
    assert_schema_is_declared(&batch, "total_nu.arrow");
    assert_eq!(batch.num_rows(), 1, "total_nu.arrow is one row per nuclide");

    // This product and the prompt one beside it are both called "neutron" on
    // this fixture, so the comparison catches the transposition with the
    // adjacent `emission_mode` column and nothing else. See the doc comment.
    assert_eq!(str_at(&batch, "particle", 0), total.name);

    assert_eq!(
        str_at(&batch, "emission_mode", 0),
        total.emission_mode.name()
    );
    assert_eq!(
        str_at(&batch, "emission_mode", 0),
        "total",
        "a \"prompt\" here means write_total_nu read products[0] rather than \
         derived_products[0], which is issue #364"
    );

    // Zero on this fixture: the total nu-bar covers prompt and delayed together
    // and has no single precursor, so it carries no decay rate. Zero is also
    // what a writer would hard-code, which is why this comparison alone does
    // not defend the column.
    assert_eq!(f64_at(&batch, "decay_rate", 0), total.decay_rate);

    let Yield::Tabulated(t) = &total.yield_ else {
        panic!("U235's MF=1/MT=452 is LNU=2, so the derived total is a Tabulated1D");
    };
    assert_eq!(str_at(&batch, "yield_type", 0), "Tabulated1D");

    let shape = i32_list(&batch, "yield_shape", 0);
    assert_i32_slice_eq("yield_shape", &shape, &[2, t.x.len() as i32]);
    assert_i32_slice_eq("yield_shape", &shape, &[2, 85]);

    // The shape is read from the column and then USED to split the data, which
    // is how the loader reads it (`nuclide_arrow.rs:1942-1948`). Worth being
    // straight about what that is worth: the two assertions above already pin
    // the shape to [2, 85], so the split is equivalent to splitting at half,
    // and it is those two that catch a transposed shape, not the split. The
    // length check below earns its place only by reporting a `yield_data` that
    // disagrees with the declared shape as that, instead of as a panic inside
    // `split_at` or a slice-length mismatch two lines later.
    let written = f64_list(&batch, "yield_data", 0);
    assert_eq!(
        written.len(),
        (shape[0] * shape[1]) as usize,
        "yield_data holds {} values but yield_shape declares {shape:?}, so the \
         loader would read past the energies or stop short of them",
        written.len()
    );
    let (energy, nu) = written.split_at(shape[1] as usize);
    assert_f64_slice_eq("total_nu energies", energy, &t.x);
    assert_f64_slice_eq("total_nu values", nu, &t.y);
    assert_eq!(energy[0], 1.0e-5);
    assert_eq!(nu[0], 2.42985);

    let breakpoints = i32_list(&batch, "yield_breakpoints", 0);
    let interpolation = i32_list(&batch, "yield_interpolation", 0);
    assert_i32_slice_eq("yield_breakpoints", &breakpoints, &t.breakpoints);
    assert_i32_slice_eq("yield_interpolation", &interpolation, &t.interpolation);
    // Asserted by name and separately, and their values differ on this
    // fixture, so pushing them in the wrong order at `fission_nu.rs:74-75`
    // fails here rather than passing as a matched pair.
    assert_i32_slice_eq("yield_breakpoints", &breakpoints, &[85]);
    assert_i32_slice_eq("yield_interpolation", &interpolation, &[2]);

    // The #364 regression pin. Not a re-derivation of the written values: the
    // MF=1/MT=456 prompt table is a different parsed object on the same
    // reaction, and the whole bug is writing that one instead.
    let prompt = fission
        .products
        .first()
        .expect("MF=1/MT=456 is attached to MT 18 as an ordinary product");
    assert_eq!(prompt.emission_mode, endf::EmissionMode::Prompt);
    let Yield::Tabulated(p) = &prompt.yield_ else {
        panic!("U235's MF=1/MT=456 is LNU=2 as well");
    };
    assert_ne!(
        nu,
        p.y.as_slice(),
        "the written nu-bar is the MF=1/MT=456 PROMPT table, not the MF=1/MT=452 total"
    );
    assert_eq!(p.y[0], 2.414, "the prompt yield at 1e-5 eV, for contrast");
    // Total is prompt plus delayed, so it can never be below prompt on a
    // shared abscissa. A swap of the two products fails this on every point,
    // not just the first.
    assert_f64_slice_eq("prompt energies", &p.x, &t.x);
    for (i, (&total_nu, &prompt_nu)) in nu.iter().zip(&p.y).enumerate() {
        assert!(
            total_nu > prompt_nu,
            "at {:e} eV the written total nu-bar is {total_nu}, which is not above \
             the prompt yield {prompt_nu}",
            energy[i]
        );
    }

    // Absence is the loader's signal for "this nuclide has no total nu-bar",
    // so an empty table would be a different answer rather than the same one.
    // This half does reach the writer and it has teeth: a search widened from
    // the fission MTs to `data.reactions.values()`, falling back to
    // `products.first()`, finds Li6's ordinary neutron products and writes a
    // nu-bar for a nuclide that does not fission at all.
    let non_fissile = scratch();
    assert!(
        !yamc_convert::fission_nu::write_total_nu(&li6_ace(), non_fissile.path())
            .expect("the writer runs"),
        "Li6 has no fission reaction and so no derived total nu-bar to write"
    );
    assert!(
        absent(non_fissile.path(), "total_nu.arrow"),
        "a nuclide with no total nu-bar must leave the section absent, not empty"
    );
}

/// Both fission energy release photon terms must survive in their own forms.
///
/// A failure means the photon release scaling was dropped, halved or made
/// unloadable. The reader forms `(prompt + delayed) / prompt` from the two
/// rows and refuses half a section (`nuclide_arrow.rs:107-118`), so the row
/// count and the role order are load-bearing. So is the interpolation column:
/// `ReleaseFunction::from_tabulated`
/// (`crates/yamc-nuclide/src/fission_photon.rs:74-83`) hard-errors on any
/// scheme other than 2, so writing the breakpoints ([18]) into that slot makes
/// the whole nuclide unloadable rather than merely wrong.
///
/// This test pins those two slots BY VALUE and does not run the loader. An
/// earlier draft fed the written columns back through `from_tabulated` and
/// called that a load test; it was deleted rather than kept, because every gate
/// inside that function (`x.len() == y.len()`, at least two points, one region,
/// scheme 2) is entailed by the value assertions that come first, so the call
/// could not fail and its result could not differ from the expected one. It
/// restated the assertions above it.
///
/// Note the asymmetry this pins: `write_fission_photon` pushes interpolation
/// BEFORE breakpoints (`fission_nu.rs:171-172`) while `write_total_nu` pushes
/// breakpoints BEFORE interpolation (`fission_nu.rs:74-75`), and the two
/// declared schemas genuinely differ in that order. Reading by name in both
/// tests is what keeps a "tidy-up" of one from silently corrupting the other.
///
/// The Li6 tail is a negative on the PRODUCTION GUARD and not on this writer:
/// `fission_energy_release` gives `Ok(None)` and the `if let Some(release)` at
/// `entry.rs:404` then never calls `write_fission_photon` at all, so no defect
/// inside the writer can make that half fail. The writer's own refusal path,
/// a polynomial term with no coefficients at `fission_nu.rs:145-152`, is
/// unreachable from any vendored evaluation and is covered separately by
/// `a_polynomial_release_term_with_no_coefficients_is_refused`.
///
/// Before this test no column of this section had ever been compared to
/// anything: the existing hermetic test (`sections.rs:597`) builds the
/// `FissionEnergyRelease` and stops one call short of the writer.
#[test]
fn fission_photon_carries_both_release_terms_in_their_own_forms() {
    let material = endf_material(U235_ENDF);
    let release = yamc_convert::heating::fission_energy_release(&material)
        .expect("the release reads")
        .expect("U235 carries MF=1/MT=458");

    let dir = scratch();
    yamc_convert::fission_nu::write_fission_photon(&release, dir.path()).expect("the writer runs");
    let batch = section(dir.path(), "fission_photon.arrow");
    assert_schema_is_declared(&batch, "fission_photon.arrow");
    assert_eq!(
        batch.num_rows(),
        2,
        "both terms are written or neither; the reader refuses half a section"
    );
    assert_eq!(str_at(&batch, "role", 0), "prompt_photons");
    assert_eq!(str_at(&batch, "role", 1), "delayed_photons");

    // `kind` is per row rather than per file, and U235 is the case that proves
    // it: LFC=1 puts an IFC=4 TAB1 on the prompt term while NPLY=0 sends the
    // delayed one down the Sher-Beck branch at `fission_energy.rs:146-149`.
    let raw = material.mf1_mt458().expect("U235 carries MF=1/MT=458");
    assert_eq!(raw.lfc, 1);
    assert_eq!(raw.nply, 0);
    assert!(matches!(release.prompt_photons, Component::Tabulated(_)));
    assert_eq!(str_at(&batch, "kind", 0), "tabulated");
    assert!(matches!(release.delayed_photons, Component::Polynomial(_)));
    assert_eq!(str_at(&batch, "kind", 1), "polynomial");

    // Row 0 is compared against the RAW MF=1/MT=458 EIFC table rather than
    // against the derived `Component`, so this checks the whole chain from the
    // evaluation's own record to the column and not just the last clone.
    let Mt458Component::Tabulated { eifc, .. } = &raw.components[component_index("EGP")] else {
        panic!("U235's EGP component is the LFC=1 tabulation");
    };
    let x = f64_list(&batch, "x", 0);
    let y = f64_list(&batch, "y", 0);
    assert_f64_slice_eq("prompt_photons x", &x, &eifc.x);
    assert_f64_slice_eq("prompt_photons y", &y, &eifc.y);
    assert_eq!(x.len(), 18);
    // Not implied by the comparisons above, which measure each column against
    // its own side of the same parse: a y column LONGER than x passes all of
    // them, and still produces a file the reader refuses at load
    // (`fission_photon.rs:52`). A y column SHORTER than x is caught either way,
    // by the `y[17]` literal below.
    assert_eq!(x.len(), y.len());
    assert_eq!(x[0], 1.0e-5);
    assert_eq!(x[17], 3.0e7);
    assert_eq!(y[0], 7.281253e6);
    assert_eq!(y[17], 1.262714e7);

    let interpolation = i32_list(&batch, "interpolation", 0);
    let breakpoints = i32_list(&batch, "breakpoints", 0);
    assert_i32_slice_eq("prompt_photons interpolation", &interpolation, &[2]);
    assert_i32_slice_eq("prompt_photons breakpoints", &breakpoints, &[18]);
    assert_i32_slice_eq(
        "prompt_photons interpolation",
        &interpolation,
        &eifc.interpolation,
    );
    assert_i32_slice_eq(
        "prompt_photons breakpoints",
        &breakpoints,
        &eifc.breakpoints,
    );

    // `float_lists`, not `float_lists_or_null`, so the unused half of a row is
    // an empty list and not a null. The two load identically and are still
    // different files, so this goes to the array rather than through a decoded
    // Vec.
    assert!(!is_null(&batch, "coefficients", 0));
    assert_eq!(list_len(&batch, "coefficients", 0), 0);

    // Row 1. The constant is the evaluation's own EGD coefficient and the
    // slope is the Sher-Beck one the parser supplies, so the two halves have
    // independent provenance and are asserted separately.
    let Mt458Component::Polynomial(egd) = &raw.components[component_index("EGD")] else {
        panic!("U235's EGD component is a single polynomial coefficient");
    };
    let coefficients = f64_list(&batch, "coefficients", 1);
    assert_eq!(coefficients.len(), 2);
    assert_eq!(coefficients[0], egd[0].0);
    assert_eq!(coefficients[0], 6.33e6);
    assert_eq!(
        coefficients[1], -0.075,
        "the Sher-Beck slope from fission_energy.rs:149"
    );
    let Component::Polynomial(delayed) = &release.delayed_photons else {
        unreachable!("checked above");
    };
    assert_f64_slice_eq(
        "delayed_photons coefficients",
        &coefficients,
        &delayed.coefficients,
    );
    for column in ["x", "y", "interpolation", "breakpoints"] {
        assert!(!is_null(&batch, column, 1), "{column} row 1 is null");
        assert_eq!(
            list_len(&batch, column, 1),
            0,
            "{column} row 1 is not empty"
        );
    }

    // The other shape this file takes. Am244 is LFC=0 with NPLY=2, so both
    // terms are polynomials of three coefficients and every list column is
    // empty on both rows.
    let am_material = endf_material(AM244_ENDF);
    let am_release = yamc_convert::heating::fission_energy_release(&am_material)
        .expect("the release reads")
        .expect("Am244 carries MF=1/MT=458");
    let am_dir = scratch();
    yamc_convert::fission_nu::write_fission_photon(&am_release, am_dir.path())
        .expect("the writer runs");
    let am_batch = section(am_dir.path(), "fission_photon.arrow");
    assert_eq!(am_batch.num_rows(), 2);
    assert_eq!(str_at(&am_batch, "role", 0), "prompt_photons");
    assert_eq!(str_at(&am_batch, "role", 1), "delayed_photons");

    let am_raw = am_material.mf1_mt458().expect("Am244 carries MF=1/MT=458");
    assert_eq!(am_raw.lfc, 0);
    assert_eq!(am_raw.nply, 2);
    for (row, name) in [(0, "EGP"), (1, "EGD")] {
        assert_eq!(str_at(&am_batch, "kind", row), "polynomial");
        let Mt458Component::Polynomial(pairs) = &am_raw.components[component_index(name)] else {
            panic!("Am244's {name} component is a polynomial");
        };
        // The (coefficient, uncertainty) pairs with the uncertainties dropped.
        // The NPLY=2 unit correction at `fission_energy.rs:132-138` only fires
        // when the second-order term is large enough to be the ENDF/B-VII.1
        // MeV mistake, and Am244's is exactly zero, so the raw coefficients
        // are an independent oracle here rather than a partial one.
        let expected: Vec<f64> = pairs.iter().map(|&(c, _)| c).collect();
        assert_eq!(expected.len(), 3, "NPLY=2 gives three coefficients");
        assert_eq!(expected[2], 0.0);
        assert_f64_slice_eq(
            &format!("Am244 {name} coefficients"),
            &f64_list(&am_batch, "coefficients", row),
            &expected,
        );
        for column in ["x", "y", "interpolation", "breakpoints"] {
            assert!(!is_null(&am_batch, column, row), "{column} is null");
            assert_eq!(list_len(&am_batch, column, row), 0, "{column} is not empty");
        }
    }

    // The negative, on the production guard rather than on the writer. Li6 has
    // no MF=1/MT=458, so there is no release and the `if let Some(release)` at
    // `entry.rs:404` never reaches `write_fission_photon`: `Ok(None)` is the
    // whole of "this nuclide has no fission_photon.arrow". It is distinct from
    // an `Err`, which is what a fissile evaluation whose release cannot be read
    // gives.
    let li6 = endf_material(LI6_ENDF);
    assert!(li6.mf1_mt458().is_none());
    assert!(
        yamc_convert::heating::fission_energy_release(&li6)
            .expect("the release reads")
            .is_none(),
        "a nuclide with no MF=1/MT=458 must give no release rather than an empty one"
    );
}

// ---------------------------------------------------------------------------
// Constructed inputs.
//
// Everything below is BUILT BY HAND rather than parsed, because no vendored
// evaluation reaches the branch under test. None of it is evaluation parity and
// none of it should be read as such: the values are chosen to be different from
// one another and from the constants the writer might hard-code, which is the
// one thing a fixture whose two columns happen to agree cannot do.
// ---------------------------------------------------------------------------

/// The `name` on the constructed total-nu product.
///
/// Deliberately not "neutron". Every real evaluation's total and prompt
/// products are both called "neutron", so that is exactly the constant a writer
/// could hard-code and stay green on every vendored fixture.
const SENTINEL_PARTICLE: &str = "sentinel-particle";

/// The `decay_rate` on the constructed total-nu product, in inverse seconds.
///
/// Deliberately not 0.0, for the same reason: a real total nu-bar has no
/// precursor and carries zero here, so zero is the constant to defend against.
const SENTINEL_DECAY_RATE: f64 = 3.75e-3;

/// The redundant MT 18 that stands beside the derived total in every
/// constructed nuclide here.
///
/// Its ordinary PROMPT product is what a writer that fell back to
/// `products.first()` would find, and its name, mode, decay rate and yield all
/// differ from every sentinel below, so that fallback fails on four columns
/// rather than passing unnoticed.
fn constructed_redundant_mt18() -> endf::Reaction {
    let mut mt18 = endf::Reaction::new(18);
    mt18.redundant = true;
    mt18.products.push(endf::Product {
        name: "neutron".to_string(),
        emission_mode: endf::EmissionMode::Prompt,
        decay_rate: 0.0,
        yield_: Yield::Tabulated(Tabulated1D::new(vec![1.0e-5, 2.0e7], vec![2.41, 4.20])),
        ..Default::default()
    });
    mt18
}

/// A derived total-nu product carrying the four fields `write_total_nu` copies.
///
/// Every other field is left at its default, because the writer reads no other
/// field: `fission_nu.rs:43-76` touches `name`, `emission_mode`, `decay_rate`
/// and `yield_` and nothing else.
fn constructed_total(
    name: &str,
    emission_mode: endf::EmissionMode,
    decay_rate: f64,
    yield_: Yield,
) -> endf::Product {
    endf::Product {
        name: name.to_string(),
        emission_mode,
        decay_rate,
        yield_,
        ..Default::default()
    }
}

/// A fissionable `IncidentNeutron` built by hand, with a derived total nu-bar
/// on `fission_mt`.
///
/// CONSTRUCTED, not parsed: there is no evaluation behind this and no parity
/// claim attached to it. The shape is the one the writer's own comment
/// (`fission_nu.rs:26-34`) describes for U240: MT 18 is present and redundant
/// and carries an ordinary PROMPT product, while the derived total sits on
/// whichever fission MT the caller asks for. Two consequences are deliberate.
/// A search narrowed to MT 18 finds a reaction with no derived product at all
/// and writes nothing, so the narrowing shows up as an absent file rather than
/// as wrong values. And a search that fell back to `products.first()` finds the
/// MT 18 prompt product, whose fields differ from the sentinel ones on every
/// column below.
fn constructed_fissile_nuclide(fission_mt: i32, total_yield: Yield) -> endf::IncidentNeutron {
    let mut mt18 = constructed_redundant_mt18();

    // Neither "total" (what the section is for) nor "prompt" (what #364 wrote
    // instead), so a writer that hard-codes either one fails.
    let total = constructed_total(
        SENTINEL_PARTICLE,
        endf::EmissionMode::Delayed,
        SENTINEL_DECAY_RATE,
        total_yield,
    );

    let mut data = endf::IncidentNeutron::new(92, 240, 0);
    if fission_mt == 18 {
        mt18.derived_products.push(total);
        data.reactions.insert(18, mt18);
    } else {
        data.reactions.insert(18, mt18);
        let mut chance = endf::Reaction::new(fission_mt);
        chance.derived_products.push(total);
        data.reactions.insert(fission_mt, chance);
    }
    data
}

/// A `FissionEnergyRelease` with the two photon terms given and a distinct
/// placeholder in each of the other five components.
///
/// CONSTRUCTED, not parsed. `write_fission_photon` reads `prompt_photons` and
/// `delayed_photons` and nothing else (`fission_nu.rs:130-133`), so the other
/// five only have to exist; they are given non-empty, mutually different
/// coefficients so that a writer reaching for the wrong component would take a
/// well formed term and change the outcome of the refusal test below.
fn constructed_release(
    prompt_photons: Component,
    delayed_photons: Component,
) -> endf::FissionEnergyRelease {
    let term = |zeroth: f64| Component::Polynomial(Polynomial::new(vec![zeroth, -0.075]));
    endf::FissionEnergyRelease {
        fragments: term(1.69e8),
        prompt_neutrons: term(4.94e6),
        delayed_neutrons: term(7.4e3),
        prompt_photons,
        delayed_photons,
        betas: term(6.5e6),
        neutrinos: term(8.75e6),
    }
}

/// `particle`, `emission_mode`, `decay_rate` and both region lists are copies of
/// the product's own fields, and the product is looked for on every fission MT
/// rather than on 18 alone.
///
/// CONSTRUCTED INPUT, and no vendored evaluation can stand in for it. Three
/// writer defects survive every assertion in
/// `total_nu_is_the_derived_total_and_not_the_prompt_yield`:
///
/// - `strings(std::slice::from_ref(&total.name))` replaced by a literal
///   `"neutron"`, which is right on every evaluation there is;
/// - `floats(&[total.decay_rate])` replaced by `floats(&[0.0])`, likewise;
/// - `FISSION_MTS.iter()` narrowed back to `[18].iter()`, the U240 regression
///   the writer's own comment (`fission_nu.rs:26-34`) is about, which is right
///   for every vendored fissile evaluation since all of them fission on 18. The
///   only test that covered the narrowing is `sections.rs:833`, which announces
///   a skip without NJOY and a full ENDF tree, so it covers nothing on CI.
///
/// A hard-coded `"total"` in the `emission_mode` slot survives the parity tests
/// as well, for the same reason, and is caught here too. So does truncating
/// either region list to its first element: U235's nu-bar is a single
/// interpolation region, so `t.breakpoints.clone()` and `vec![t.breakpoints[0]]`
/// write the same bytes there. The constructed yield below has TWO regions, of
/// different widths and different schemes, so a truncation fails and so does a
/// swap of the two lists.
///
/// The three sentinel fields are not physical: a real MF=1/MT=452 total is a
/// neutron yield, with no precursor, whose emission mode is `Total`. Being
/// different from every constant the writer might reach for is the whole point
/// of them, and it is the one thing the evaluations cannot be. The two-region
/// yield, by contrast, is an ordinary shape the format allows and the vendored
/// evaluations happen not to use for nu-bar.
#[test]
fn total_nu_columns_are_the_products_own_fields_on_every_fission_mt() {
    // The literal list, not the writer's own constant, so this cannot be
    // satisfied by narrowing the constant. Pinned so a sixth fission MT has to
    // arrive in the loop below as well as in the writer.
    let fission_mts = [18, 19, 20, 21, 38];
    assert_eq!(yamc_convert::fast_xs::FISSION_MTS, fission_mts);

    for mt in fission_mts {
        // Two regions: histogram up to the second point, then linear-linear to
        // the fourth. Nothing vendored has more than one region here.
        let yield_ = Yield::Tabulated(Tabulated1D::with_regions(
            vec![1.0e-5, 1.0, 1.0e6, 2.0e7],
            vec![2.44, 2.45, 3.10, 4.51],
            vec![2, 4],
            vec![1, 2],
        ));
        let data = constructed_fissile_nuclide(mt, yield_);
        let dir = scratch();
        assert!(
            yamc_convert::fission_nu::write_total_nu(&data, dir.path()).expect("the writer runs"),
            "the derived total nu-bar is on MT {mt}, which the writer must search"
        );
        let batch = section(dir.path(), "total_nu.arrow");

        assert_eq!(
            str_at(&batch, "particle", 0),
            SENTINEL_PARTICLE,
            "particle must be the product's own name, not the constant \"neutron\" (MT {mt})"
        );
        assert_eq!(
            str_at(&batch, "emission_mode", 0),
            "delayed",
            "emission_mode must be the product's own mode, not a constant and not \
             the MT 18 prompt product's (MT {mt})"
        );
        assert_eq!(
            f64_at(&batch, "decay_rate", 0),
            SENTINEL_DECAY_RATE,
            "decay_rate must be the product's own rate, not the constant 0.0 (MT {mt})"
        );
        assert_i32_slice_eq(
            "yield_breakpoints",
            &i32_list(&batch, "yield_breakpoints", 0),
            &[2, 4],
        );
        assert_i32_slice_eq(
            "yield_interpolation",
            &i32_list(&batch, "yield_interpolation", 0),
            &[1, 2],
        );
    }
}

/// The `Yield::Polynomial` arm of `write_total_nu`, which no evaluation reaches.
///
/// CONSTRUCTED INPUT. Every derived total-nu product in every vendored fixture
/// is `Yield::Tabulated` (LNU=2), so `fission_nu.rs:44-50` is dead code against
/// the fixtures and every defect in it survives both parity tests above: the
/// arm could label the coefficients "Tabulated1D", give them the Tabulated
/// arm's `[2, n]` shape, or push a region into columns a polynomial has none
/// of, and nothing in this repository would notice.
///
/// The three coefficients differ from one another in sign and magnitude, so a
/// reversed or duplicated write fails. There are three of them rather than two
/// so that a shape carrying the Tabulated arm's leading 2, rather than the
/// coefficient count, fails as well.
#[test]
fn total_nu_polynomial_yield_is_written_as_bare_coefficients() {
    let coefficients = vec![2.4367, 0.0611, -0.00123];
    let data =
        constructed_fissile_nuclide(18, Yield::Polynomial(Polynomial::new(coefficients.clone())));

    let dir = scratch();
    assert!(
        yamc_convert::fission_nu::write_total_nu(&data, dir.path()).expect("the writer runs"),
        "a derived total nu-bar is present, polynomial or not"
    );
    let batch = section(dir.path(), "total_nu.arrow");

    assert_eq!(str_at(&batch, "yield_type", 0), "Polynomial");
    assert_f64_slice_eq(
        "yield_data",
        &f64_list(&batch, "yield_data", 0),
        &coefficients,
    );
    assert_i32_slice_eq(
        "yield_shape",
        &i32_list(&batch, "yield_shape", 0),
        &[coefficients.len() as i32],
    );

    // A polynomial has no interpolation regions, so both list columns are
    // EMPTY and not null: `int_lists`, not an `or_null` variant. The loader
    // decodes the two identically (`arrow_helpers.rs:338-344`) and they are
    // still different files, so this goes to the array.
    for column in ["yield_breakpoints", "yield_interpolation"] {
        assert!(!is_null(&batch, column, 0), "{column} is null");
        assert_eq!(list_len(&batch, column, 0), 0, "{column} is not empty");
    }
}

/// A polynomial release term with no coefficients is refused, not written.
///
/// CONSTRUCTED INPUT. `FissionEnergyRelease::from_material` cannot build an
/// empty polynomial out of any vendored evaluation, so the writer's only
/// defensive path (`fission_nu.rs:145-152`) is unreachable from parsed bytes:
/// deleting the refusal outright leaves both parity tests above green, and
/// nothing else in the repository covers it.
///
/// What it defends: the reader rejects a polynomial term with no coefficients
/// at load (`arrow/nuclide_arrow.rs:71-76`), so writing one produces a nuclide
/// directory that cannot be read at all. Failing the conversion, with the term
/// named, is the difference between a build error and a library that only
/// breaks when someone tries to use it.
///
/// Both terms are exercised because the check sits inside the loop over them
/// (`fission_nu.rs:143-152`): a version that validated only the first term
/// would pass the prompt case and write the delayed one.
#[test]
fn a_polynomial_release_term_with_no_coefficients_is_refused() {
    let empty = || Component::Polynomial(Polynomial::new(Vec::new()));
    let well_formed = || Component::Polynomial(Polynomial::new(vec![6.33e6, -0.075]));

    for (role, release) in [
        (
            "prompt_photons",
            constructed_release(empty(), well_formed()),
        ),
        (
            "delayed_photons",
            constructed_release(well_formed(), empty()),
        ),
    ] {
        let dir = scratch();
        let error = match yamc_convert::fission_nu::write_fission_photon(&release, dir.path()) {
            Err(error) => error,
            Ok(()) => panic!("the empty {role} term must be refused, not written"),
        };
        let message = error.to_string();
        assert!(
            message.contains(role),
            "the refusal must name the term it refused, got: {message}"
        );
        assert!(
            absent(dir.path(), "fission_photon.arrow"),
            "{role}: the refusal must leave no half-written section behind"
        );
    }
}

// ---------------------------------------------------------------------------
// Which of two candidates the writer picks.
//
// `write_total_nu` makes two tie-breaks that no vendored evaluation ever puts
// to the test, because the parser attaches at most one derived product to at
// most one fission reaction: `fission_products_ace` pushes a single total
// (`reaction.rs:817-853`) and the ENDF route does the same
// (`reaction.rs:406-435`). Both are reachable only from a constructed nuclide,
// and neither is a parity claim.
// ---------------------------------------------------------------------------

/// With two derived products on one fission reaction, the FIRST one is written.
///
/// CONSTRUCTED INPUT, and this test pins a CHOICE rather than a fact. No parser
/// path produces two derived products on a reaction, so there is no evaluated
/// answer to "which of two totals is this nuclide's nu-bar" and none is being
/// claimed. What is not arbitrary is that the writer be deterministic about it
/// and not change its mind silently: `derived_products.first()`
/// (`fission_nu.rs:38`) is the behaviour, and replacing it with `.last()` is
/// invisible to every other test in the workspace.
///
/// Both orderings are built and the assertion in each is that the FRONT product
/// won, which is what makes this more than a restatement of `.first()`. A
/// writer that picked by `emission_mode == Total`, or by "the one with a
/// tabulated yield", satisfies one ordering and fails the other. The two
/// products differ on all four columns compared, so a failure names the column.
#[test]
fn total_nu_takes_the_first_derived_product_when_a_reaction_carries_two() {
    // A tabulated total against a polynomial one, so `yield_type` separates
    // them as well as the three scalar columns do.
    let tabulated = || {
        constructed_total(
            "first-of-two-totals",
            endf::EmissionMode::Total,
            1.25e-2,
            Yield::Tabulated(Tabulated1D::new(vec![1.0e-5, 2.0e7], vec![2.44, 4.55])),
        )
    };
    let polynomial = || {
        constructed_total(
            "second-of-two-totals",
            endf::EmissionMode::Delayed,
            7.5e-2,
            Yield::Polynomial(Polynomial::new(vec![9.99, -8.88])),
        )
    };

    for (label, front, back, front_yield_type) in [
        (
            "a Total ahead of a Delayed",
            tabulated(),
            polynomial(),
            "Tabulated1D",
        ),
        (
            "a Delayed ahead of a Total",
            polynomial(),
            tabulated(),
            "Polynomial",
        ),
    ] {
        // Read off before the product is moved into the reaction, so the
        // expectation is the front product's own fields and not a transcription.
        let (name, mode, decay_rate) = (
            front.name.clone(),
            front.emission_mode.name(),
            front.decay_rate,
        );

        let mut mt18 = constructed_redundant_mt18();
        mt18.derived_products.push(front);
        mt18.derived_products.push(back);
        let mut data = endf::IncidentNeutron::new(92, 240, 0);
        data.reactions.insert(18, mt18);

        let dir = scratch();
        assert!(
            yamc_convert::fission_nu::write_total_nu(&data, dir.path()).expect("the writer runs"),
            "{label}: a derived total is present, so a nu-bar must be written"
        );
        let batch = section(dir.path(), "total_nu.arrow");
        assert_eq!(
            batch.num_rows(),
            1,
            "{label}: total_nu.arrow is one row per nuclide, so the second derived \
             product is dropped rather than written as a second row the reader \
             would never look at"
        );

        assert_eq!(
            str_at(&batch, "particle", 0),
            name,
            "{label}: the written particle is not the front product's"
        );
        assert_eq!(
            str_at(&batch, "emission_mode", 0),
            mode,
            "{label}: the written emission_mode is not the front product's"
        );
        assert_eq!(
            f64_at(&batch, "decay_rate", 0),
            decay_rate,
            "{label}: the written decay_rate is not the front product's"
        );
        assert_eq!(
            str_at(&batch, "yield_type", 0),
            front_yield_type,
            "{label}: the written yield is not the front product's"
        );
    }
}

/// With a derived total on MT 18 AND on MT 19, MT 18 is the one written.
///
/// CONSTRUCTED INPUT: the parser attaches the NU block to exactly one reaction,
/// so no evaluation carries a derived total on two fission MTs at once, and
/// `find_map` over `FISSION_MTS` (`fission_nu.rs:35`) never has to break a tie.
/// Iterating that list backwards passes every other test in this file, because
/// every other nuclide here has its total on exactly one MT.
///
/// Unlike the two-derived-products tie above, this one pins a FACT and not just
/// a choice. MT 18 is total fission and MTs 19, 20, 21 and 38 are its first-,
/// second-, third- and fourth-chance partials, so where MT 18 has a total
/// nu-bar it is the nuclide's, and the one-row section holds the nuclide's.
/// Writing the first-chance nu-bar in its place would be low wherever
/// second-chance fission contributes, which for a fusion source is exactly the
/// energies that matter, and nothing on disk would say so.
#[test]
fn total_nu_prefers_mt_18_when_two_fission_mts_carry_a_total() {
    let mut mt18 = constructed_redundant_mt18();
    mt18.derived_products.push(constructed_total(
        "mt18-total-fission",
        endf::EmissionMode::Total,
        1.0e-3,
        Yield::Tabulated(Tabulated1D::new(vec![1.0e-5, 2.0e7], vec![2.44, 4.55])),
    ));

    let mut mt19 = endf::Reaction::new(19);
    mt19.derived_products.push(constructed_total(
        "mt19-first-chance",
        endf::EmissionMode::Delayed,
        2.0e-3,
        Yield::Polynomial(Polynomial::new(vec![9.99, -8.88])),
    ));

    let mut data = endf::IncidentNeutron::new(92, 240, 0);
    data.reactions.insert(18, mt18);
    data.reactions.insert(19, mt19);

    let dir = scratch();
    assert!(
        yamc_convert::fission_nu::write_total_nu(&data, dir.path()).expect("the writer runs"),
        "both fission MTs carry a derived total, so one of them must be written"
    );
    let batch = section(dir.path(), "total_nu.arrow");
    assert_eq!(batch.num_rows(), 1);

    assert_eq!(
        str_at(&batch, "particle", 0),
        "mt18-total-fission",
        "the written nu-bar is MT 19's first-chance total, not MT 18's"
    );
    assert_eq!(str_at(&batch, "emission_mode", 0), "total");
    assert_eq!(f64_at(&batch, "decay_rate", 0), 1.0e-3);
    assert_eq!(str_at(&batch, "yield_type", 0), "Tabulated1D");
}

/// A multi-region tabulated release term is never silently truncated to one
/// region.
///
/// CONSTRUCTED INPUT. Every vendored EGP tabulation is a single linear-linear
/// region (U235's is `[2]` and `[18]`), so `t.interpolation.clone()` and
/// `vec![t.interpolation[0]]` write the same bytes there, and so do the two
/// forms of the breakpoint copy.
///
/// This is the one place in the file where the writer's current behaviour is
/// arguably wrong, so the test says what it is doing. The writer copies both
/// region lists through, and the reader then refuses the file it produced:
/// `ReleaseFunction::from_tabulated` (`crates/yamc-nuclide/src/fission_photon.rs:66-73`)
/// errors on `interpolation.len() > 1 || breakpoints.len() > 1`. This input
/// therefore has no outcome that is both written and loadable, which is the
/// asymmetric validation this pull request reports: the writer refuses an empty
/// polynomial because the reader refuses one, and applies no equivalent check
/// to a tabulated term.
///
/// So the assertion is the invariant rather than today's return value. The
/// writer either refuses the term, naming it and leaving no file, or copies
/// both region lists whole; adding the missing refusal is then a passing change
/// rather than a spurious failure. What both arms rule out is the third
/// outcome, and it is the dangerous one: truncating to the first region turns a
/// file the reader REFUSES into a file the reader ACCEPTS, evaluating a
/// histogram tail as linear-linear with nothing on disk to say so. That is the
/// #369 class of error the reader's refusal exists to prevent. Neither arm is
/// vacuous, and a writer that simply always failed would be caught by
/// `fission_photon_carries_both_release_terms_in_their_own_forms` above.
#[test]
fn a_multi_region_release_table_is_never_silently_truncated_to_one_region() {
    // Two regions of different widths and different schemes: linear-linear to
    // the second point, histogram to the fourth. Truncating either list to its
    // first element therefore loses a value that differs from the one kept.
    let x = vec![1.0e-5, 1.0e3, 1.0e6, 3.0e7];
    let y = vec![6.60e6, 6.81e6, 7.42e6, 1.023e7];
    let breakpoints = vec![2, 4];
    let interpolation = vec![2, 1];
    let release = constructed_release(
        Component::Tabulated(Tabulated1D::with_regions(
            x.clone(),
            y.clone(),
            breakpoints.clone(),
            interpolation.clone(),
        )),
        Component::Polynomial(Polynomial::new(vec![6.33e6, -0.075])),
    );

    let dir = scratch();
    match yamc_convert::fission_nu::write_fission_photon(&release, dir.path()) {
        Ok(()) => {
            let batch = section(dir.path(), "fission_photon.arrow");
            assert_eq!(str_at(&batch, "kind", 0), "tabulated");
            assert_f64_slice_eq("prompt_photons x", &f64_list(&batch, "x", 0), &x);
            assert_f64_slice_eq("prompt_photons y", &f64_list(&batch, "y", 0), &y);
            assert_i32_slice_eq(
                "prompt_photons interpolation",
                &i32_list(&batch, "interpolation", 0),
                &interpolation,
            );
            assert_i32_slice_eq(
                "prompt_photons breakpoints",
                &i32_list(&batch, "breakpoints", 0),
                &breakpoints,
            );
        }
        Err(error) => {
            let message = error.to_string();
            assert!(
                message.contains("prompt_photons"),
                "a refusal must name the term it refused, got: {message}"
            );
            assert!(
                absent(dir.path(), "fission_photon.arrow"),
                "a refusal must leave no half-written section behind"
            );
        }
    }
}

/// Neither writer reports success for a section it could not write.
///
/// CONSTRUCTED INPUT, and a constructed destination: every other call in this
/// file writes into a scratch directory that exists, so nothing here or
/// anywhere else in the workspace exercises the `?` on `write_section`. The
/// directory named below is never created, so `File::create`
/// (`sections.rs:45`) fails with `NotFound` on every platform, with no `chmod`
/// and no assumption about who the test runs as.
///
/// What that defends. `write_total_nu` returns a bool the caller writes into
/// the manifest, so a swallowed error there means `entry.rs` records a
/// `total_nu.arrow` that is not on disk, and the reader then falls back to the
/// PROMPT yield, which is issue #364 arriving by a different route. A swallowed
/// error in `write_fission_photon` loses the delayed photon scaling the same
/// way. Both are silent: the conversion reports success and the wrong numbers
/// turn up in a transport result.
#[test]
fn neither_writer_reports_success_for_a_section_it_could_not_write() {
    let dir = scratch();
    let missing = dir.path().join("this-directory-is-never-created");
    assert!(!missing.exists(), "the destination must not exist");

    let data =
        constructed_fissile_nuclide(18, Yield::Polynomial(Polynomial::new(vec![2.44, 0.01])));
    let error = yamc_convert::fission_nu::write_total_nu(&data, &missing)
        .expect_err("write_total_nu must report a directory it cannot write into");
    assert_eq!(
        error
            .downcast_ref::<std::io::Error>()
            .map(std::io::Error::kind),
        Some(std::io::ErrorKind::NotFound),
        "the failure must be the IO one, not something further down: {error}"
    );
    assert!(absent(&missing, "total_nu.arrow"));

    let release = constructed_release(
        Component::Polynomial(Polynomial::new(vec![6.33e6, -0.075])),
        Component::Polynomial(Polynomial::new(vec![5.11e6, -0.081])),
    );
    let error = yamc_convert::fission_nu::write_fission_photon(&release, &missing)
        .expect_err("write_fission_photon must report a directory it cannot write into");
    assert_eq!(
        error
            .downcast_ref::<std::io::Error>()
            .map(std::io::Error::kind),
        Some(std::io::ErrorKind::NotFound),
        "the failure must be the IO one, not the empty-polynomial refusal: {error}"
    );
    assert!(absent(&missing, "fission_photon.arrow"));
}
