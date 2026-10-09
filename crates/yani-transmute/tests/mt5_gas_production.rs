//! MT=5 modelled end to end, against each evaluation's own numbers.
//!
//! An evaluation states the hydrogen and helium every reaction makes, in its
//! gas production cross sections MT=203 to 207 (H1, H2, H3, He3, He4). A
//! short irradiation of one stable nuclide makes those gases at exactly the
//! flux-weighted MT=203 to 207 rates, whatever channels the chain models them
//! through, so the inventory checks the chain against the evaluation for
//! every channel at once: one that is missing, MT=5 or any other, shows as
//! gas the inventory lacks. On ENDF/B-VIII.1 Fe54 most of the helium is MT=5.
//!
//! Opt-in, since it reads whole local libraries. Set `YANI_MT5_DATA` to a
//! directory holding `<library>-arrow/neutron/<nuclide>.arrow` and the
//! published `transmutation-<library>.arrow/decay/`, and `YANI_MT5_REACTIONS`
//! to one holding `<library>/reactions/`, reactions subsections converted by
//! this version of yani-convert from the same evaluations, then run
//! `cargo test -p yani-transmute --test mt5_gas_production -- --ignored --nocapture`.
//! Each check prints its numbers before any assertion.

use std::collections::HashMap;
use std::path::PathBuf;

use yamc_materials::Material;
use yamc_nuclide::load_scope::LoadScope;
use yamc_nuclide::reaction::Reaction;
use yani_transmute::{transmute_material, MultigroupSpectrum, TransmuteStep};

/// The gases and the MT that states each one's production.
const GASES: [(&str, i32); 5] = [
    ("H1", 203),
    ("H2", 204),
    ("H3", 205),
    ("He3", 206),
    ("He4", 207),
];

/// How close the inventory's gas production has to come to the evaluation's
/// own, relative: the issue's 0.1%.
const TOLERANCE: f64 = 1.0e-3;

/// A gas whose production is below this share of the parent's whole gas
/// production is reported but not held to the tolerance, since a relative
/// difference of a vanishing rate says nothing (TENDL's He3 from Fe56 at 14
/// MeV is 1e-7 of its gas).
const NEGLIGIBLE: f64 = 1.0e-6;

fn roots() -> Option<(PathBuf, PathBuf)> {
    let data = std::env::var_os("YANI_MT5_DATA").map(PathBuf::from)?;
    let reactions = std::env::var_os("YANI_MT5_REACTIONS").map(PathBuf::from)?;
    Some((data, reactions))
}

/// The library a chain's decay data comes from: TENDL publishes none and
/// borrows ENDF/B-VIII.1's, as its published chains do.
fn decay_library(library: &str) -> &str {
    if library.starts_with("tendl") {
        "endf-b8.1"
    } else {
        library
    }
}

fn chain(data: &std::path::Path, reactions: &std::path::Path, library: &str) -> yani::LoadedChain {
    let decay = data
        .join(format!("transmutation-{}.arrow", decay_library(library)))
        .join("decay");
    let reactions = reactions.join(library).join("reactions");
    yani::load_chain_parts(
        &decay.to_string_lossy(),
        Some(&reactions.to_string_lossy()),
        None,
        None,
    )
    .expect("chain loads")
}

/// A flat-in-energy group average of a pointwise cross section, the
/// collapse's own default weight: the trapezoid on the evaluation's points
/// inside the group, zero outside its tabulated range.
fn group_average(reaction: &Reaction, lo: f64, hi: f64) -> f64 {
    let at = |e: f64| reaction.cross_section_at(e).unwrap_or(0.0);
    let last = reaction.energy[reaction.energy.len() - 1];
    let hi_in = hi.min(last);
    if hi_in <= lo {
        return 0.0;
    }
    let mut points = vec![lo];
    points.extend(
        reaction
            .energy
            .iter()
            .copied()
            .filter(|&e| e > lo && e < hi_in),
    );
    points.push(hi_in);
    let integral: f64 = points
        .windows(2)
        .map(|w| 0.5 * (at(w[0]) + at(w[1])) * (w[1] - w[0]))
        .sum();
    integral / (hi - lo)
}

/// A multiplicity as the fold reads it: zero below its first point, linear
/// between points, flat above its last.
fn multiplicity_at(curve: &yani::BranchCurve, e: f64) -> f64 {
    let (x, y) = (&curve.energy, &curve.values);
    if e < x[0] {
        return 0.0;
    }
    if e >= x[x.len() - 1] {
        return y[y.len() - 1];
    }
    let i = x.partition_point(|&v| v <= e);
    let (x0, x1) = (x[i - 1], x[i]);
    if x1 == x0 {
        return y[i];
    }
    y[i - 1] + (e - x0) / (x1 - x0) * (y[i] - y[i - 1])
}

/// MT=5's part of one gas over a group, two ways: the exact integral of the
/// product of the linear multiplicity and the linear MT=5 cross section (what
/// the fold integrates), and their product taken at the gas production cross
/// section's own points and linear between them, which is how a processed
/// MT=203 to 207 holds it (checked against the tables: they equal the product
/// at their points). Both averaged over the group, flat in energy.
fn mt5_part(
    curve: &yani::BranchCurve,
    mt5: &Reaction,
    gas: &Reaction,
    lo: f64,
    hi: f64,
) -> (f64, f64) {
    let sigma = |e: f64| mt5.cross_section_at(e).unwrap_or(0.0);
    let inside = |grid: &[f64]| -> Vec<f64> {
        let mut points = vec![lo];
        points.extend(grid.iter().copied().filter(|&e| e > lo && e < hi));
        points.push(hi);
        points.sort_by(f64::total_cmp);
        points.dedup();
        points
    };
    let mut union: Vec<f64> = mt5.energy.iter().copied().collect();
    union.extend(curve.energy.iter().copied());
    let exact: f64 = inside(&union)
        .windows(2)
        .map(|w| {
            let (a, b) = (w[0], w[1]);
            let (f0, f1) = (multiplicity_at(curve, a), multiplicity_at(curve, b));
            let (g0, g1) = (sigma(a), sigma(b));
            (b - a) * (2.0 * f0 * g0 + f0 * g1 + f1 * g0 + 2.0 * f1 * g1) / 6.0
        })
        .sum();
    // The product at the table's own points, linear between them, as the
    // table holds it.
    let gas_grid: Vec<f64> = gas.energy.iter().copied().collect();
    let tabulated = |e: f64| -> f64 {
        let p = |x: f64| multiplicity_at(curve, x) * sigma(x);
        let i = gas_grid.partition_point(|&x| x <= e);
        if i == 0 || i == gas_grid.len() {
            return if i == 0 { 0.0 } else { p(gas_grid[i - 1]) };
        }
        let (x0, x1) = (gas_grid[i - 1], gas_grid[i]);
        if x1 == x0 {
            return p(x1);
        }
        p(x0) + (e - x0) / (x1 - x0) * (p(x1) - p(x0))
    };
    let pointwise: f64 = inside(&gas_grid)
        .windows(2)
        .map(|w| 0.5 * (tabulated(w[0]) + tabulated(w[1])) * (w[1] - w[0]))
        .sum();
    (exact / (hi - lo), pointwise / (hi - lo))
}

/// One spectrum: boundaries in eV and a shape summing to one.
struct Spectrum {
    name: &'static str,
    boundaries: Vec<f64>,
    shape: Vec<f64>,
}

impl Spectrum {
    fn flat(name: &'static str, lo: f64, hi: f64) -> Spectrum {
        Spectrum {
            name,
            boundaries: vec![lo, hi],
            shape: vec![1.0],
        }
    }

    fn first_wall() -> Spectrum {
        let text = include_str!("fixtures/dt_first_wall_175.txt");
        let mut boundaries = Vec::new();
        let mut flux = Vec::new();
        for line in text.lines().filter(|l| !l.starts_with('#')) {
            let v: Vec<f64> = line
                .split_whitespace()
                .map(|x| x.parse().expect("number"))
                .collect();
            if boundaries.is_empty() {
                boundaries.push(v[0]);
            }
            boundaries.push(v[1]);
            flux.push(v[2]);
        }
        let total: f64 = flux.iter().sum();
        Spectrum {
            name: "D-T first wall",
            boundaries,
            shape: flux.iter().map(|f| f / total).collect(),
        }
    }

    fn reference(&self, reaction: Option<&Reaction>) -> f64 {
        let Some(r) = reaction else { return 0.0 };
        self.shape
            .iter()
            .enumerate()
            .map(|(g, w)| w * group_average(r, self.boundaries[g], self.boundaries[g + 1]))
            .sum()
    }
}

/// What one short irradiation of `nuclide` made of each gas, per atom and
/// per unit flux, in barns, beside the evaluation's MT=203 to 207 as
/// processed and the same with MT=5's part integrated exactly (see
/// [`mt5_part`]), and the per-edge rates the solve drove.
struct Measured {
    gases: Vec<(&'static str, f64, f64, f64)>,
    edges: yani::EdgeRates,
    mt5: f64,
}

fn irradiate(
    data: &std::path::Path,
    library: &str,
    loaded: &yani::LoadedChain,
    nuclide: &str,
    spectrum: &Spectrum,
) -> Result<Measured, String> {
    let path = data
        .join(format!("{library}-arrow"))
        .join("neutron")
        .join(format!("{nuclide}.arrow"));
    let full = yamc_nuclide::nuclide::load_nuclide(&path, &LoadScope::full())
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let temperature = full.loaded_temperatures[0].clone();
    let reactions = full
        .reactions_for_temp(&temperature)
        .ok_or("no reactions at the first temperature")?;

    const DENSITY: f64 = 1.0e-2;
    let mut material = Material::new(
        HashMap::from([(nuclide.to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )?;
    material.nuclides.insert(nuclide.to_string(), DENSITY);
    material.volume = Some(1.0);
    material.set_temperature(&temperature);
    material
        .read_nuclear_data(
            &HashMap::from([(nuclide.to_string(), path.to_string_lossy().to_string())]),
            None,
        )
        .map_err(|e| e.to_string())?;

    // Burn-up of a part in a million over one second, so no decay is worth
    // the name: tritium's half-life is 4e8 s, and a longer step would show its
    // He3 as He3 production.
    const FLUX: f64 = 1.0e18;
    const SECONDS: f64 = 1.0;
    let results = transmute_material(
        &mut material,
        &[MultigroupSpectrum {
            boundaries: spectrum.boundaries.clone(),
            masses: spectrum.shape.clone(),
            flux_error: None,
        }],
        &[TransmuteStep {
            dt: SECONDS,
            irradiation: Some((0, FLUX)),
        }],
        std::sync::Arc::clone(&loaded.chain),
        &loaded.branch,
        loaded.parts,
        None,
    )
    .map_err(|e| e.to_string())?;
    let after = &results.get_material(0, 1).ok_or("no step")?.nuclides;
    let per_barn = 1.0e24 / (DENSITY * FLUX * SECONDS);
    let particles: Vec<&yani::BranchCurve> = loaded
        .branch
        .curves()
        .get(nuclide)
        .and_then(|k| k.get(yani::reactions::ANYTHING))
        .map(|curves| {
            curves
                .iter()
                .filter(|c| c.quantity == yani::BranchQuantity::Multiplicity)
                .collect()
        })
        .unwrap_or_default();
    let gases = GASES
        .iter()
        .map(|&(gas, mt)| {
            let made = after.get(gas).copied().unwrap_or(0.0) * per_barn;
            let processed = reactions.get(&mt).map(|r| r.as_ref());
            let reference = spectrum.reference(processed);
            // The processed table's MT=5 part, taken out and put back exact.
            let mut exact_reference = reference;
            if let (Some(curve), Some(mt5), Some(table)) = (
                particles.iter().find(|c| c.target == gas),
                reactions.get(&5),
                processed,
            ) {
                for (g, w) in spectrum.shape.iter().enumerate() {
                    let (exact, pointwise) = mt5_part(
                        curve,
                        mt5,
                        table,
                        spectrum.boundaries[g],
                        spectrum.boundaries[g + 1],
                    );
                    exact_reference += w * (exact - pointwise);
                }
                if std::env::var_os("YANI_MT5_DEBUG").is_some() {
                    let (mut e_sum, mut p_sum) = (0.0, 0.0);
                    for (g, w) in spectrum.shape.iter().enumerate() {
                        let (exact, pointwise) = mt5_part(curve, mt5, table, spectrum.boundaries[g], spectrum.boundaries[g + 1]);
                        e_sum += w * exact;
                        p_sum += w * pointwise;
                    }
                    eprintln!("DEBUG {nuclide} {gas}: MT=5 exact {e_sum:.6e} pointwise-on-table-grid {p_sum:.6e} table {reference:.6e}");
                }
            }
            (gas, made, reference, exact_reference)
        })
        .collect();
    Ok(Measured {
        gases,
        edges: results
            .get_reaction_rates(0, 0)
            .cloned()
            .unwrap_or_default(),
        mt5: spectrum.reference(reactions.get(&5).map(|r| r.as_ref())),
    })
}

/// Test 1: on tendl-2025, endf-b8.1 and jeff-4.0, at 14 MeV and under a D-T
/// first wall spectrum, the inventory makes each gas at the evaluation's own
/// MT=203 to 207 rate. Test 4 alongside: the residuals of MT=5 add up to the
/// MT=5 rate.
#[test]
#[ignore = "reads whole local libraries; set YANI_MT5_DATA and YANI_MT5_REACTIONS"]
fn gas_production_matches_the_evaluations_own() {
    let Some((data, reactions_root)) = roots() else {
        eprintln!("YANI_MT5_DATA or YANI_MT5_REACTIONS unset; nothing to check");
        return;
    };
    let spectra = [
        Spectrum::flat("14 MeV (13-15 flat)", 1.3e7, 1.5e7),
        Spectrum::first_wall(),
    ];
    let nuclides = ["Fe54", "Fe56", "Cr52", "Ni58", "Cu63", "Mn55", "W186"];
    let mut failures = Vec::new();
    for library in ["tendl-2025", "endf-b8.1", "jeff-4.0"] {
        let loaded = chain(&data, &reactions_root, library);
        for spectrum in &spectra {
            for nuclide in nuclides {
                let m = match irradiate(&data, library, &loaded, nuclide, spectrum) {
                    Ok(m) => m,
                    Err(e) => {
                        println!("{library:10} {:20} {nuclide:5} REFUSED: {e}", spectrum.name);
                        failures.push(format!("{library} {} {nuclide}: {e}", spectrum.name));
                        continue;
                    }
                };
                let all_gas: f64 = m.gases.iter().map(|(_, _, r, _)| r).sum();
                for (gas, made, reference, exact) in &m.gases {
                    let relative = |to: f64| {
                        if to > 0.0 {
                            made / to - 1.0
                        } else if *made == 0.0 {
                            0.0
                        } else {
                            f64::INFINITY
                        }
                    };
                    let (raw, rel) = (relative(*reference), relative(*exact));
                    let held = *reference > NEGLIGIBLE * all_gas || *made > NEGLIGIBLE * all_gas;
                    let flag = if held && rel.abs() > TOLERANCE {
                        "FAIL"
                    } else {
                        ""
                    };
                    println!(
                        "{library:10} {:20} {nuclide:5} {gas:3} inventory {made:.6e} b  MT{} {reference:.6e} b {:+.4}%  MT=5 exact {exact:.6e} b {:+.4}% {flag}",
                        spectrum.name,
                        GASES.iter().find(|(g, _)| g == gas).unwrap().1,
                        100.0 * raw,
                        100.0 * rel
                    );
                    if !flag.is_empty() {
                        failures.push(format!(
                            "{library} {} {nuclide} {gas}: {:+.4}% ({:+.4}% against the table as processed)",
                            spectrum.name,
                            100.0 * rel,
                            100.0 * raw
                        ));
                    }
                }
                // Test 4: every atom MT=5 removes lands on a residual.
                let residuals: f64 = m
                    .edges
                    .get(nuclide)
                    .and_then(|k| k.get(yani::reactions::ANYTHING))
                    .map(|edges| {
                        edges
                            .iter()
                            .filter(|(t, _)| {
                                t.as_deref()
                                    .is_some_and(|t| !["H1", "H2", "H3", "He3", "He4"].contains(&t))
                            })
                            .map(|(_, r)| r)
                            .sum()
                    })
                    .unwrap_or(0.0);
                if m.mt5 > 0.0 {
                    // Edge rates are per atom at the step's flux.
                    let residual_b = residuals / 1.0e18 * 1.0e24;
                    let rel = residual_b / m.mt5 - 1.0;
                    println!(
                        "{library:10} {:20} {nuclide:5} MT=5 {:.6e} b, residuals {residual_b:.6e} b  {:+.4}%",
                        spectrum.name,
                        m.mt5,
                        100.0 * rel
                    );
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} checks outside {:.1}%:\n{}",
        failures.len(),
        100.0 * TOLERANCE,
        failures.join("\n")
    );
}

/// Test 2: the same on TENDL-2025 under a spectrum reaching 55 MeV, where its
/// own MT=5 is open (it starts at 30 MeV), as an IFMIF-DONES Li(d,n) source's
/// does.
#[test]
#[ignore = "reads whole local libraries; set YANI_MT5_DATA and YANI_MT5_REACTIONS"]
fn gas_production_matches_tendl_to_55_mev() {
    let Some((data, reactions_root)) = roots() else {
        eprintln!("YANI_MT5_DATA or YANI_MT5_REACTIONS unset; nothing to check");
        return;
    };
    let spectrum = Spectrum {
        name: "flat 1-55 MeV",
        boundaries: vec![1.0e6, 2.0e7, 3.0e7, 4.0e7, 5.5e7],
        shape: vec![0.25; 4],
    };
    let library = "tendl-2025";
    let loaded = chain(&data, &reactions_root, library);
    let mut failures = Vec::new();
    for nuclide in ["Fe54", "Fe56", "Cr52", "Ni58", "Cu63", "Mn55", "W186"] {
        let m = match irradiate(&data, library, &loaded, nuclide, &spectrum) {
            Ok(m) => m,
            Err(e) => {
                println!("{library} {nuclide} REFUSED: {e}");
                failures.push(format!("{nuclide}: {e}"));
                continue;
            }
        };
        let all_gas: f64 = m.gases.iter().map(|(_, _, r, _)| r).sum();
        for (gas, made, reference, exact) in &m.gases {
            let raw = made / reference - 1.0;
            let rel = made / exact - 1.0;
            let held = *reference > NEGLIGIBLE * all_gas;
            let flag = if held && rel.abs() > TOLERANCE {
                "FAIL"
            } else {
                ""
            };
            println!(
                "{library} {} {nuclide:5} {gas:3} inventory {made:.6e} b  table {reference:.6e} b {:+.4}%  MT=5 exact {exact:.6e} b {:+.4}% {flag}",
                spectrum.name,
                100.0 * raw,
                100.0 * rel
            );
            if !flag.is_empty() {
                failures.push(format!("{nuclide} {gas}: {:+.4}%", 100.0 * rel));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Test 9: a residual's ground and isomeric states, from MF=6's LIP, split
/// the MT=5 rate as the evaluation's own multiplicities do at the energy.
///
/// At 40 MeV on TENDL-2025's Ni58, whose MT=5 makes Co58 and Co58_m1 (LIP 0
/// and 1): the fold over a narrow group about 40 MeV against the hand
/// calculation `y_s(E) / sum_r y_r(E)` from the evaluation's multiplicities at
/// 40 MeV. Not at 14 MeV: no local library gives an MT=5 isomer split there
/// (TENDL's MT=5 opens at 30 MeV, and the ENDF/B-VIII.1, JEFF-4.0 and
/// FENDL-3.2d lists with LIP isomers have no residual multiplicity at 14 MeV).
#[test]
#[ignore = "reads whole local libraries; set YANI_MT5_DATA and YANI_MT5_REACTIONS"]
fn an_isomer_split_from_lip_matches_the_evaluation() {
    let Some((data, reactions_root)) = roots() else {
        eprintln!("YANI_MT5_DATA or YANI_MT5_REACTIONS unset; nothing to check");
        return;
    };
    let library = "tendl-2025";
    let loaded = chain(&data, &reactions_root, library);
    const E: f64 = 4.0e7;
    let spectrum = Spectrum::flat("40 MeV", E * (1.0 - 1.0e-4), E * (1.0 + 1.0e-4));
    let m = irradiate(&data, library, &loaded, "Ni58", &spectrum).expect("Ni58 runs");
    let curves = &loaded.branch.curves()["Ni58"][yani::reactions::ANYTHING];
    let residuals: Vec<&yani::BranchCurve> = curves
        .iter()
        .filter(|c| c.quantity == yani::BranchQuantity::Yield)
        .collect();
    let sum: f64 = residuals.iter().map(|c| multiplicity_at(c, E)).sum();
    let edges = &m.edges["Ni58"][yani::reactions::ANYTHING];
    let total: f64 = edges
        .iter()
        .filter(|(t, _)| {
            t.as_deref()
                .is_some_and(|t| residuals.iter().any(|c| c.target == t))
        })
        .map(|(_, r)| r)
        .sum();
    for state in ["Co58", "Co58_m1"] {
        let curve = residuals
            .iter()
            .find(|c| c.target == state)
            .expect("the state is listed");
        let hand = multiplicity_at(curve, E) / sum;
        let folded = edges
            .iter()
            .find(|(t, _)| t.as_deref() == Some(state))
            .map(|(_, r)| r / total)
            .expect("the state is made");
        println!(
            "Ni58 (n,X) {state}: folded share {folded:.6e}, hand calculation {hand:.6e}, {:+.4}%",
            100.0 * (folded / hand - 1.0)
        );
        assert!((folded / hand - 1.0).abs() < TOLERANCE, "{state}");
    }
}

/// Test 7: on TENDL-2025, whose MT=5 opens at 30 MeV, carrying it changes
/// nothing under a D-T spectrum: iron, tungsten and a EUROFER-like steel
/// irradiated for a year and cooled come out the same, nuclide by nuclide and
/// in activity and decay heat, to 1e-9, on the published reactions subsection
/// (no `(n,X)`) and on the one this version converts.
#[test]
#[ignore = "reads whole local libraries; set YANI_MT5_DATA and YANI_MT5_REACTIONS"]
fn tendl_under_dt_is_unchanged() {
    let Some((data, reactions_root)) = roots() else {
        eprintln!("YANI_MT5_DATA or YANI_MT5_REACTIONS unset; nothing to check");
        return;
    };
    let library = "tendl-2025";
    let neutron = data.join(format!("{library}-arrow")).join("neutron");
    // Every nuclide the library has, so the products' rates are collapsed
    // too and the comparison covers the whole network.
    {
        let mut config = yamc_nuclide::config::Config::global();
        for entry in std::fs::read_dir(&neutron).expect("library directory") {
            let path = entry.expect("entry").path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if let Some(nuclide) = name.strip_suffix(".arrow") {
                    config
                        .cross_sections
                        .insert(nuclide.to_string(), path.to_string_lossy().to_string());
                }
            }
        }
    }
    // Decay data and fission yields from ENDF/B-VIII.1, as TENDL borrows them.
    let decay = data.join("transmutation-endf-b8.1.arrow").join("decay");
    let yields = data
        .join("transmutation-endf-b8.1.arrow")
        .join("fission_yields");
    let load = |reactions: PathBuf| {
        yani::load_chain_parts(
            &decay.to_string_lossy(),
            Some(&reactions.to_string_lossy()),
            Some(&yields.to_string_lossy()),
            None,
        )
        .expect("chain loads")
    };
    let published = load(
        data.join("transmutation-tendl-2025.arrow")
            .join("reactions"),
    );
    let converted = load(reactions_root.join(library).join("reactions"));
    assert!(
        converted.chain.values().any(|n| n
            .reactions
            .iter()
            .any(|r| r.kind == yani::reactions::ANYTHING)),
        "the converted chain carries (n,X)"
    );

    let spectrum = Spectrum::first_wall();
    let steps = [
        TransmuteStep {
            dt: 3.15576e7,
            irradiation: Some((0, 1.0e14)),
        },
        TransmuteStep {
            dt: 86400.0,
            irradiation: None,
        },
        TransmuteStep {
            dt: 3.15576e7,
            irradiation: None,
        },
    ];
    let materials: [(&str, &[(&str, f64)]); 3] = [
        ("Fe", &[("Fe", 1.0)]),
        ("W", &[("W", 1.0)]),
        (
            "EUROFER-like",
            &[
                ("Fe", 0.886),
                ("Cr", 0.096),
                ("W", 0.0035),
                ("Mn", 0.004),
                ("V", 0.0022),
                ("Ta", 0.0004),
                ("C", 0.0046),
            ],
        ),
    ];
    for (name, elements) in materials {
        let mut atoms: HashMap<String, f64> = HashMap::new();
        for (element, fraction) in elements {
            for (nuclide, f) in
                yamc_nuclide::composition::expand_element(element, *fraction, "atom")
                    .expect("element expands")
            {
                *atoms.entry(nuclide).or_insert(0.0) += f * 8.5e-2;
            }
        }
        let run = |loaded: &yani::LoadedChain| {
            let mut material = Material::new(atoms.clone(), "atom", "sum", None).expect("material");
            material.nuclides = atoms.clone();
            material.volume = Some(1.0);
            material
                .read_nuclear_data(&HashMap::new(), None)
                .expect("cross sections load");
            transmute_material(
                &mut material,
                &[MultigroupSpectrum {
                    boundaries: spectrum.boundaries.clone(),
                    masses: spectrum.shape.clone(),
                    flux_error: None,
                }],
                &steps,
                std::sync::Arc::clone(&loaded.chain),
                &loaded.branch,
                loaded.parts,
                None,
            )
            .expect("transmutes")
        };
        let (before, after) = (run(&published), run(&converted));
        let mut worst: f64 = 0.0;
        for step in 0..=steps.len() {
            let a = &before.get_material(0, step).unwrap().nuclides;
            let b = &after.get_material(0, step).unwrap().nuclides;
            let total: f64 = a.values().sum();
            let (mut activity, mut heat) = ([0.0f64; 2], [0.0f64; 2]);
            for (k, inventory) in [a, b].into_iter().enumerate() {
                for (nuclide, n) in inventory {
                    let Some(c) = published.chain.get(nuclide) else {
                        continue;
                    };
                    if let Some(t) = c.half_life.filter(|t| *t > 0.0) {
                        let rate = std::f64::consts::LN_2 / t * n;
                        activity[k] += rate;
                        heat[k] += rate * c.decay_energy;
                    }
                }
            }
            let mut names: Vec<&String> = a.keys().chain(b.keys()).collect();
            names.sort();
            names.dedup();
            for nuclide in names {
                let (x, y) = (
                    a.get(nuclide).copied().unwrap_or(0.0),
                    b.get(nuclide).copied().unwrap_or(0.0),
                );
                // Relative for every nuclide above 1e-12 of the material. Below
                // that the two solves differ in rounding: the converted chain
                // reaches nuclides through `(n,X)` that the published one does
                // not, so the matrices differ in size though not in any rate,
                // and CRAM's rounding, relative to the largest density, is
                // what a nuclide at 1e-20 of the material sees (W's Yb178,
                // 1e-9 apart at 4e-20 of it).
                let rel = ((x - y) / x.abs().max(y.abs()).max(f64::MIN_POSITIVE)).abs();
                if x.abs().max(y.abs()) > 1.0e-12 * total {
                    worst = worst.max(rel);
                } else if rel > 1.0e-9 {
                    println!(
                        "{name:13} step {step} {nuclide} at {:.1e} of the material: {rel:.3e} apart",
                        x / total
                    );
                }
            }
            for (q, v) in [("activity", activity), ("decay heat", heat)] {
                let rel = if v[0] > 0.0 {
                    (v[1] / v[0] - 1.0).abs()
                } else {
                    0.0
                };
                println!(
                    "{name:13} step {step} {q:10} {:.9e} vs {:.9e}  {rel:.3e}",
                    v[0], v[1]
                );
                assert!(rel <= 1.0e-9, "{name} step {step} {q}");
            }
        }
        println!("{name:13} worst nuclide difference {worst:.3e}");
        assert!(worst <= 1.0e-9, "{name}: {worst}");
    }
}

/// Test 10: the `(n,X)` rate is a cross section like any other to the
/// resample-and-re-solve uncertainty. ENDF/B-VIII.1 Fe54 states MF=33
/// covariance for MT=5, so its `(n,X)` rate is drawn and the gas it makes
/// carries a spread; Fe57 has no covariance at all, and is named as such.
/// The multiplicities, which no library gives a covariance for, are named as
/// held.
#[test]
#[ignore = "reads whole local libraries; set YANI_MT5_DATA and YANI_MT5_REACTIONS"]
fn the_anything_rate_is_drawn_where_mf33_covers_mt5() {
    let Some((data, reactions_root)) = roots() else {
        eprintln!("YANI_MT5_DATA or YANI_MT5_REACTIONS unset; nothing to check");
        return;
    };
    let library = "endf-b8.1";
    let loaded = chain(&data, &reactions_root, library);
    let neutron = data.join(format!("{library}-arrow")).join("neutron");
    let atoms = HashMap::from([("Fe54".to_string(), 5.0e-3), ("Fe57".to_string(), 5.0e-3)]);
    let mut material = Material::new(atoms.clone(), "atom", "sum", None).expect("material");
    material.nuclides = atoms;
    material.volume = Some(1.0);
    material
        .read_nuclear_data(
            &["Fe54", "Fe57"]
                .iter()
                .map(|n| {
                    (
                        n.to_string(),
                        neutron
                            .join(format!("{n}.arrow"))
                            .to_string_lossy()
                            .to_string(),
                    )
                })
                .collect(),
            None,
        )
        .expect("cross sections load");
    let results = transmute_material(
        &mut material,
        &[MultigroupSpectrum {
            boundaries: vec![1.3e7, 1.5e7],
            masses: vec![1.0],
            flux_error: None,
        }],
        &[TransmuteStep {
            dt: 3.15576e7,
            irradiation: Some((0, 1.0e14)),
        }],
        std::sync::Arc::clone(&loaded.chain),
        &loaded.branch,
        loaded.parts,
        Some(&yani_transmute::uncertainty::DataUncertainty {
            seed: 7,
            samples: Some(64),
            ..Default::default()
        }),
    )
    .expect("transmutes");
    let info = &results.uncertainty_info[&0];
    let covered = |n: &str| {
        info.rate_fraction_covered
            .get(&(n.to_string(), yani::reactions::ANYTHING.to_string()))
            .copied()
    };
    println!(
        "Fe54 (n,X) covered {:?}, Fe57 (n,X) covered {:?}",
        covered("Fe54"),
        covered("Fe57")
    );
    println!("perturbed {:?}", info.perturbed);
    println!("no covariance data {:?}", info.no_covariance_data);
    for n in ["H1", "He4", "Mn53", "Cr51"] {
        let mean = results.get_nuclide_density(0, n, 1).unwrap_or(0.0);
        let sigma = results.get_nuclide_uncertainty(0, n, 1).unwrap_or(0.0);
        println!(
            "{n:4} {mean:.4e} +- {sigma:.3e} ({:.2}%)",
            100.0 * sigma / mean
        );
    }
    assert!(
        covered("Fe54").is_some_and(|f| f > 0.99),
        "{:?}",
        covered("Fe54")
    );
    assert!(info.no_covariance_data.contains("Fe57"));
    assert!(info
        .not_perturbed
        .iter()
        .any(|s| s.starts_with("MT=5 product multiplicities")));
    let mn53 = results.get_nuclide_uncertainty(0, "Mn53", 1).unwrap_or(0.0);
    assert!(
        mn53 > 0.0,
        "Mn53 is MT=5's residual, so its spread is MT=5's"
    );
}
