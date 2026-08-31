//! Issue #364: the delayed-neutron data the transport path relies on.
//!
//! yamc samples nu_TOTAL neutrons per fission but drew every one of them from the
//! PROMPT spectrum, so its fission source was too hard and its slowing-down flux
//! read low against OpenMC (U238 -1.5%, U235 -0.65%, Pu239 -0.21%, in beta order).
//! The fix emits a `beta(E) = nu_d(E) / nu_t(E)` share of the progeny from the
//! delayed spectrum instead, which is far softer.
//!
//! To keep one spectrum for both backends to sample, the six ENDF delayed groups
//! are folded once into a single yield-weighted spectrum. That fold is EXACT rather
//! than an approximation, but only because of two properties of the data, which is
//! what these tests pin:
//!
//!  1. the group FRACTIONS `nu_d,g / nu_d` do not depend on incident energy (the
//!     absolute yields very much do), so one set of fold weights serves every
//!     energy;
//!  2. each group's spectrum does not depend on incident energy either (ENDF stores
//!     two identical rows spanning the range), so folding the groups' first rows
//!     loses nothing.
//!
//! Plus the two facts that make the fix worth making at all: the delayed spectra
//! really are much softer than the prompt one, and `beta` is small and falls with
//! energy.

use yamc_nuclide::nuclide::Nuclide;
use yamc_nuclide::reaction_product::{
    AngleEnergyDistribution, EnergyDistribution, ParticleType, TabulatedProbability,
};

/// Incident energies spanning the whole transport range.
const ENERGIES: [f64; 7] = [1.0e-5, 1.0e-2, 1.0, 1.0e3, 1.0e5, 1.0e6, 1.4e7];

/// Fissile nuclides with delayed data in ENDF/B-VIII.1.
const FISSILE: [&str; 4] = ["U235", "U238", "Pu239", "Th232"];

fn cache(n: &str) -> String {
    yamc_test_cache::nuclide_path(n)
}

fn load(name: &str) -> Option<Nuclide> {
    let p = cache(name);
    if !std::path::Path::new(&p).exists() {
        eprintln!("skipping {name} -- endf-b8.1-{name}.arrow cache absent");
        return None;
    }
    // A read error is a skip like absence is: since #389 a cache directory is
    // routinely populated at activation scope, holding cross sections and none
    // of the transport sections this needs, and the directory exists either
    // way. CI fetches only the fixture list; without this, any developer
    // machine that has run a transmutation fails instead of skipping.
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

/// Mean of a tabulated outgoing-energy row under its own interpolation law.
fn mean_of(row: &TabulatedProbability) -> f64 {
    let TabulatedProbability::Tabulated { x, p, interp, .. } = row;
    let (mut num, mut den) = (0.0, 0.0);
    for (i, w) in x.windows(2).enumerate() {
        let (a, b) = (w[0], w[1]);
        match interp {
            yamc_nuclide::reaction_product::TabulatedInterp::Histogram => {
                den += p[i] * (b - a);
                num += p[i] * 0.5 * (b * b - a * a);
            }
            yamc_nuclide::reaction_product::TabulatedInterp::LinLin => {
                let (pa, pb) = (p[i], p[i + 1]);
                den += 0.5 * (pa + pb) * (b - a);
                num += (b - a) * (a * (2.0 * pa + pb) + b * (pa + 2.0 * pb)) / 6.0;
            }
        }
    }
    num / den
}

/// `(prompt spectrum, delayed group spectra)` off the nuclide's first fission
/// reaction.
fn spectra(nuc: &Nuclide) -> (Option<&EnergyDistribution>, Vec<&EnergyDistribution>) {
    let grid = &nuc.fast_xs[nuc.get_temp_idx("294").expect("294 K")];
    let rxn = &grid.fission_mt_reactions[0];
    fn dist(p: &yamc_nuclide::reaction_product::ReactionProduct) -> Option<&EnergyDistribution> {
        p.distribution.first().and_then(|d| match d {
            AngleEnergyDistribution::UncorrelatedAngleEnergy { energy, .. } => energy.as_ref(),
            _ => None,
        })
    }
    let neutrons = || {
        rxn.products
            .iter()
            .filter(|p| p.is_particle_type(&ParticleType::Neutron))
    };
    let prompt = neutrons().find(|p| p.is_prompt()).and_then(dist);
    let delayed = neutrons()
        .filter(|p| p.is_delayed())
        .filter_map(dist)
        .collect();
    (prompt, delayed)
}

/// PROPERTY 1: the group fractions are energy-independent, which is what makes one
/// fixed set of fold weights exact. Checked across seven decades.
#[test]
fn delayed_group_fractions_do_not_depend_on_energy() {
    for name in FISSILE {
        let Some(nuc) = load(name) else { continue };
        let d = nuc
            .delayed_neutrons("294")
            .unwrap_or_else(|| panic!("{name} must carry delayed-neutron data"));

        let reference: Vec<f64> = {
            let g = d.group_nu(ENERGIES[0]);
            let s: f64 = g.iter().sum();
            g.iter().map(|v| v / s).collect()
        };
        assert_eq!(reference.len(), 6, "{name}: expected 6 delayed groups");
        for &e in &ENERGIES {
            let g = d.group_nu(e);
            let s: f64 = g.iter().sum();
            assert!(s > 0.0, "{name} at {e:.1e} eV: nu_d must be positive");
            for (i, (&raw, &want)) in g.iter().zip(&reference).enumerate() {
                let got = raw / s;
                assert!(
                    (got - want).abs() < 1.0e-6,
                    "{name} group {i} fraction at {e:.1e} eV is {got:.8} against {want:.8} at \
                     {:.1e} eV. The fold weights the groups at ONE reference energy, which is \
                     only exact while the fractions are energy-independent (#364)",
                    ENERGIES[0]
                );
            }
        }
    }
}

/// PROPERTY 2: each group's spectrum is energy-independent, so folding the groups'
/// FIRST incident-energy rows loses nothing.
#[test]
fn delayed_group_spectra_do_not_depend_on_energy() {
    for name in FISSILE {
        let Some(nuc) = load(name) else { continue };
        let (_, delayed) = spectra(&nuc);
        assert_eq!(delayed.len(), 6, "{name}: expected 6 delayed spectra");
        for (g, dist) in delayed.iter().enumerate() {
            let EnergyDistribution::ContinuousTabular { energy_out, .. } = dist else {
                panic!("{name} group {g}: delayed spectra are ContinuousTabular in ENDF/B-VIII.1");
            };
            let first = mean_of(&energy_out[0]);
            for (r, row) in energy_out.iter().enumerate() {
                let m = mean_of(row);
                assert!(
                    (m - first).abs() <= 1.0e-9 * first,
                    "{name} group {g}: incident row {r} has mean {m:.6e} eV against row 0's \
                     {first:.6e} eV. The fold uses row 0 only, which assumes the delayed \
                     spectra do not vary with incident energy (#364)"
                );
            }
        }
    }
}

/// The delayed spectra are much SOFTER than the prompt one -- the whole reason
/// sampling every progeny from the prompt spectrum biased the source hard.
#[test]
fn delayed_spectra_are_softer_than_prompt() {
    for name in FISSILE {
        let Some(nuc) = load(name) else { continue };
        let (prompt, delayed) = spectra(&nuc);
        let Some(EnergyDistribution::ContinuousTabular {
            energy_out: prompt_rows,
            ..
        }) = prompt
        else {
            // Th232's prompt chi is CorrelatedAngleEnergy, which the shared flat
            // path does not carry yet (issue #356), so it has no prompt row here.
            eprintln!("{name}: prompt chi is not ContinuousTabular, skipping the comparison");
            continue;
        };
        let prompt_mean = mean_of(&prompt_rows[0]);
        for (g, dist) in delayed.iter().enumerate() {
            let EnergyDistribution::ContinuousTabular { energy_out, .. } = dist else {
                panic!("{name} group {g}: expected ContinuousTabular");
            };
            let m = mean_of(&energy_out[0]);
            assert!(
                m < 0.5 * prompt_mean,
                "{name} group {g} mean outgoing energy {m:.4e} eV is not well below the prompt \
                 {prompt_mean:.4e} eV; if the delayed spectrum were as hard as the prompt one \
                 there would be nothing for #364 to fix"
            );
        }
    }
}

/// `beta(E) = nu_d / nu_t` is a small, decreasing fraction, and it ORDERS the flux
/// deficits the fix removes (U238 > U235 > Pu239).
#[test]
fn beta_is_small_and_falls_with_energy() {
    let mut fast: Vec<(&str, f64)> = Vec::new();
    for name in FISSILE {
        let Some(nuc) = load(name) else { continue };
        let d = nuc.delayed_neutrons("294").expect("delayed data");
        let nu_t = |e: f64| nuc.fission_nu.as_ref().expect("nu table").evaluate(e);
        let beta = |e: f64| d.nu(e) / nu_t(e);
        let (thermal, high) = (beta(0.0253), beta(1.4e7));
        assert!(
            thermal > 0.0 && thermal < 0.05,
            "{name}: thermal beta {thermal:.6} outside a physical range"
        );
        assert!(
            high < thermal,
            "{name}: beta rises with energy ({thermal:.6} thermal to {high:.6} at 14 MeV); \
             nu_t grows faster than nu_d, so it must fall"
        );
        fast.push((name, high));
    }
    if fast.len() == FISSILE.len() {
        let get = |n: &str| fast.iter().find(|(k, _)| *k == n).expect("present").1;
        assert!(
            get("U238") > get("U235") && get("U235") > get("Pu239"),
            "beta at 14 MeV must order U238 > U235 > Pu239 (it is the ordering of the flux \
             deficits #364 removes): U238 {:.6}, U235 {:.6}, Pu239 {:.6}",
            get("U238"),
            get("U235"),
            get("Pu239")
        );
    }
}

/// A nuclide with NO delayed data reads as `None`, which is what keeps its draw
/// schedule (and so its GPU parity) unchanged.
#[test]
fn a_nuclide_without_delayed_data_reads_none() {
    let Some(nuc) = load("Am240") else { return };
    assert!(nuc.fissionable, "Am240 is fissionable");
    assert!(
        nuc.delayed_neutrons("294").is_none(),
        "Am240's ENDF/B-VIII.1 evaluation carries no delayed groups, so it must read None and \
         skip the prompt-or-delayed draw entirely (#364)"
    );
}

/// The folded spectrum is normalised, softer than prompt, and bounded by the
/// groups' own means -- the end-to-end check on what transport actually samples.
#[test]
fn the_folded_spectrum_sits_inside_the_groups() {
    for name in FISSILE {
        let Some(nuc) = load(name) else { continue };
        let d = nuc.delayed_neutrons("294").expect("delayed data");
        let yamc_nuclide::reaction_product::FissionChiFlat::Continuous {
            n_x,
            x,
            p,
            c,
            max_x,
            ..
        } = d.chi_flat()
        else {
            panic!("{name}: the folded delayed chi must flatten to Continuous");
        };
        let n = n_x[0] as usize;
        let row = &c[..n];
        assert!(
            (row[n - 1] - 1.0).abs() < 1.0e-12,
            "{name}: folded cdf ends at {:.17e}, not 1",
            row[n - 1]
        );

        // The fold's mean must lie between the softest and hardest group.
        let (_, delayed) = spectra(&nuc);
        let group_means: Vec<f64> = delayed
            .iter()
            .map(|dist| {
                let EnergyDistribution::ContinuousTabular { energy_out, .. } = dist else {
                    panic!()
                };
                mean_of(&energy_out[0])
            })
            .collect();
        let (lo, hi) = (
            group_means.iter().cloned().fold(f64::MAX, f64::min),
            group_means.iter().cloned().fold(0.0, f64::max),
        );
        let xs = &x[..n];
        let ps = &p[..n];
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..n - 1 {
            // The fold of histogram groups is a histogram, so integrate that way.
            den += ps[i] * (xs[i + 1] - xs[i]);
            num += ps[i] * 0.5 * (xs[i + 1] * xs[i + 1] - xs[i] * xs[i]);
        }
        let folded = num / den;
        assert!(
            folded >= lo && folded <= hi,
            "{name}: folded mean {folded:.6e} eV is outside the groups' range \
             [{lo:.6e}, {hi:.6e}]"
        );
        assert!(
            *max_x >= n,
            "{name}: stride {max_x} shorter than row length {n}"
        );
    }
}
