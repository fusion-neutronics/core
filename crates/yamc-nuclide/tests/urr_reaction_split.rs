//! Issue #372: inside a URR band, the elastic/inelastic split of the scatter
//! bucket must come from the sampled probability-table band, not from the smooth
//! elastic:scatter ratio.
//!
//! The probability table scales ELASTIC (and capture / fission) and leaves
//! inelastic alone, so `reaction_partials`' scatter bucket is exactly
//! `urr_elastic + inelastic` and `sigma_e` is `urr_elastic` itself. Re-deriving it
//! from the smooth ratio discards the band: for Fe58 at 1 MeV the smooth elastic
//! fraction is 0.843 while the true per-band fraction runs 0.709 to 0.902, so low
//! bands came out ~19% too elastic and ~46% short on inelastic. That put every
//! URR-bearing nuclide at the top of the V&V outlier list.
//!
//! These run only when the endf-b8.1 caches are present.

use rand::SeedableRng;

/// Any deterministic RNG: `reaction_partials` only draws from it when the caller
/// passes `None` for the band random, which these tests never do.
fn rng() -> rand::rngs::StdRng {
    rand::rngs::StdRng::seed_from_u64(1)
}

fn cache(n: &str) -> String {
    yamc_test_cache::nuclide_path(n)
}

/// The cache entry, or `None` when this machine has nothing full-scope for it.
///
/// Presence is not enough. Since #389 a cache dir is routinely populated at
/// activation scope, holding cross sections and none of the transport sections
/// this test needs, and the directory exists either way. Read errors are
/// therefore a skip like absence is: the alternative is a test that passes in
/// CI, where only the fixture list is fetched, and dies on any developer
/// machine that has run a transmutation.
fn load(name: &str) -> Option<yamc_nuclide::nuclide::Nuclide> {
    let p = cache(name);
    if !std::path::Path::new(&p).exists() {
        eprintln!("skipping {name} -- endf-b8.1-{name}.arrow cache absent");
        return None;
    }
    match yamc_nuclide::arrow::nuclide_arrow::read_nuclide_from_arrow(
        std::path::Path::new(&p),
        &yamc_nuclide::LoadScope::full(),
    ) {
        Ok(nuclide) => Some(nuclide),
        Err(e) => {
            eprintln!("skipping {name} -- cached at a narrower scope than full ({e})");
            None
        }
    }
}

/// `sigma_e` must track the sampled band rather than sitting at the smooth
/// elastic fraction. Sweeping the band random across [0, 1) has to move the
/// elastic FRACTION of the scatter bucket by a wide margin; before the fix it was
/// pinned to a single value for every band.
#[test]
fn elastic_fraction_varies_with_the_sampled_band() {
    let Some(nuc) = load("Fe58") else { return };
    let e = 1.0e6; // inside Fe58's 350 keV .. 3 MeV table
    let mut fracs: Vec<f64> = Vec::new();
    for i in 0..40 {
        let r = (i as f64 + 0.5) / 40.0;
        let mut rng = rng();
        let p = nuc
            .reaction_partials(e, "294", Some(r), &mut rng)
            .expect("partials in band");
        let scatter = p.sigma_e + p.sigma_i;
        assert!(scatter > 0.0, "expected a nonzero scatter bucket");
        fracs.push(p.sigma_e / scatter);
    }
    let (lo, hi) = (
        fracs.iter().cloned().fold(f64::MAX, f64::min),
        fracs.iter().cloned().fold(0.0_f64, f64::max),
    );
    assert!(
        hi - lo > 0.10,
        "elastic fraction of the scatter bucket spans only {lo:.4}..{hi:.4} across 40 \
         probability-table bands. It must follow the band (Fe58 at 1 MeV runs roughly \
         0.71..0.90); a near-constant span means the split is back on the smooth \
         elastic:scatter ratio and the band is being discarded (#372)"
    );
}

/// The partials must sum to the URR-sampled total. This is the invariant that
/// makes the split well-defined: `sigma_e + sigma_i` is the scatter bucket the URR
/// total was built from, so no probability mass is invented or lost.
#[test]
fn partials_sum_to_the_urr_total() {
    let mut loaded = 0usize;
    let mut checked = 0usize;
    for name in ["Fe58", "Mn55", "Ni62"] {
        let Some(nuc) = load(name) else { continue };
        loaded += 1;
        let temp_idx = nuc.get_temp_idx("294").expect("294 K");
        // A cache entry is a directory of sections and urr.arrow is an optional
        // one, so a nuclide fetched for something else loads perfectly well and
        // simply carries no URR block. That is absent data rather than a broken
        // split, and only Fe58 is in scripts/fetch_test_fixtures.py: Mn55 and
        // Ni62 are here only on a machine that happened to fetch them. Panicking
        // on their absence made a missing section look like a physics failure.
        let Some(urr) = nuc.urr_data.get(temp_idx).and_then(|o| o.as_ref()) else {
            eprintln!("skipping {name}: no URR data in the local cache");
            continue;
        };
        checked += 1;
        let (lo, hi) = (urr.energy[0], urr.energy[urr.energy.len() - 1]);
        for k in 1..5 {
            let e = lo + (hi - lo) * (k as f64) / 5.0;
            for &r in &[0.05_f64, 0.35, 0.65, 0.95] {
                let mut rng = rng();
                let p = nuc
                    .reaction_partials(e, "294", Some(r), &mut rng)
                    .expect("partials in band");
                let sum = p.sigma_e + p.sigma_i + p.sigma_a + p.sigma_f;
                // Expected total straight from the table, with the same smooth
                // inputs `reaction_partials` derives.
                let fg = &nuc.fast_xs[temp_idx];
                let (_t, xs_abs, xs_scat, xs_fis) = fg.lookup(e);
                let (i_grid, f) = fg.lookup_grid_index(e);
                let sel = fg
                    .elastic_idx
                    .map(|i| fg.scatter_xs_interp(i_grid, f, i))
                    .unwrap_or(0.0);
                let sinel = (xs_scat - sel).max(0.0);
                let band = yamc_nuclide::urr::urr_nuclide_random(r, nuc.urr_stream_key());
                let (total, _, _, _, _) =
                    urr.sample(e, band, sel, xs_abs + xs_fis, xs_fis, sinel, None);
                assert!(
                    (sum - total).abs() <= 1e-9 * total.max(1.0),
                    "{name} at {e:.4e} eV band r={r}: partials sum to {sum:.10e} but the \
                     URR total is {total:.10e} (#372)"
                );
                assert!(
                    p.sigma_e >= 0.0 && p.sigma_i >= 0.0,
                    "{name}: negative partial (sigma_e={:.4e}, sigma_i={:.4e})",
                    p.sigma_e,
                    p.sigma_i
                );
            }
        }
    }
    // Skipping absent nuclides one by one would otherwise let this pass while
    // checking nothing, which is the failure mode the panic above was really
    // guarding against.
    //
    // Conditioned on something having loaded, so it separates the two reasons
    // for checking nothing. No fixtures at all is an environment without the
    // data, which the rest of this file already treats as a skip. Fixtures that
    // load but carry no URR block is the suspicious one: urr.arrow is a
    // published section for Fe58, so that means it stopped being fetched.
    assert!(
        loaded == 0 || checked > 0,
        "{loaded} nuclide(s) loaded but none carried URR data, so this test \
         verified nothing. Fe58 is a test fixture and urr.arrow is a published \
         section for it, so this means the section stopped being fetched"
    );
}

/// Out of band nothing changes: the smooth ratio still governs, so a nuclide with
/// URR data behaves exactly like one without it outside the table's range.
#[test]
fn out_of_band_is_untouched() {
    let Some(nuc) = load("Fe58") else { return };
    let temp_idx = nuc.get_temp_idx("294").expect("294 K");
    let urr = nuc
        .urr_data
        .get(temp_idx)
        .and_then(|o| o.as_ref())
        .expect("urr");
    let below = urr.energy[0] * 0.5;
    let mut a = rng();
    let mut b = rng();
    let p1 = nuc.reaction_partials(below, "294", Some(0.1), &mut a);
    let p2 = nuc.reaction_partials(below, "294", Some(0.9), &mut b);
    let (p1, p2) = (p1.expect("partials"), p2.expect("partials"));
    assert_eq!(
        p1.sigma_e.to_bits(),
        p2.sigma_e.to_bits(),
        "below the table the band random must not change anything, but sigma_e moved \
         from {:.10e} to {:.10e}",
        p1.sigma_e,
        p2.sigma_e
    );
}
