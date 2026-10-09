//! Hold the free-gas broadening to NJOY, and its adjoint weight to the
//! broadening.
//!
//! `broaden` is compared with BROADR's own output for ENDF/B-VIII.1 W186,
//! broadening NJOY's 0 K PENDF and reading the result at the energies BROADR
//! chose. `tests/reference/README.md` records how the reference was made.
//!
//! `broadened_weight` is held to the identity it exists for,
//! `∫ σ_T ψ dE = ∫ σ₀ w dE`, with the left side integrated independently from
//! `broaden` for random cross sections and several kinds of flux.

use std::path::{Path, PathBuf};

use endf::doppler::{broaden, broaden_within, broadened_weight, Flux, BROADR_REACH};

fn reference_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("reference")
        .join("doppler-w186.txt.xz")
}

/// One table of the reference: an MT at a temperature.
struct Table {
    temperature: f64,
    mt: i32,
    energy: Vec<f64>,
    sigma: Vec<f64>,
}

/// The mass ratio and the tables of the NJOY reference.
fn read_reference() -> (f64, Vec<Table>) {
    let raw = std::fs::read(reference_path()).expect("reading the NJOY reference");
    let mut text = String::new();
    std::io::Read::read_to_string(
        &mut lzma_rust2::XzReader::new(raw.as_slice(), true),
        &mut text,
    )
    .expect("decompressing the NJOY reference");
    let mut awr = 0.0;
    let mut tables = Vec::new();
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    while let Some(line) = lines.next() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        match fields[0] {
            "AWR" => awr = fields[1].parse().unwrap(),
            "TABLE" => {
                let n: usize = fields[3].parse().unwrap();
                let mut table = Table {
                    temperature: fields[1].parse().unwrap(),
                    mt: fields[2].parse().unwrap(),
                    energy: Vec::with_capacity(n),
                    sigma: Vec::with_capacity(n),
                };
                for _ in 0..n {
                    let mut point = lines.next().unwrap().split_whitespace();
                    table.energy.push(point.next().unwrap().parse().unwrap());
                    table.sigma.push(point.next().unwrap().parse().unwrap());
                }
                tables.push(table);
            }
            other => panic!("unexpected record {other:?} in the NJOY reference"),
        }
    }
    (awr, tables)
}

/// The reference stops at 1 keV, where W186's resolved resonances are still
/// dense, to stay small. BROADR itself broadened to the top of the resolved
/// range, 10 keV. The 0 K table runs 5% further, past the kernel's reach.
const REFERENCE_TOP: f64 = 1.0e3;

#[test]
fn broadening_matches_njoy_broadr() {
    let (awr, tables) = read_reference();
    for mt in [2, 102] {
        let cold = tables
            .iter()
            .find(|t| t.temperature == 0.0 && t.mt == mt)
            .unwrap();
        for hot in tables.iter().filter(|t| t.temperature > 0.0 && t.mt == mt) {
            let (at, want) = (&hot.energy, &hot.sigma);
            let t = hot.temperature;
            // Cut where BROADR cuts the kernel, the result is BROADR's to the
            // seven significant figures of the PENDF file (5e-7 is the
            // rounding of the reference alone).
            let njoy = broaden_within(&cold.energy, &cold.sigma, awr, t, at, BROADR_REACH).unwrap();
            // The exact kernel adds what BROADR's cut drops, which is positive
            // and largest in the minima beside strong resonances.
            let exact = broaden(&cold.energy, &cold.sigma, awr, t, at).unwrap();
            let (mut worst_njoy, mut worst_exact): (f64, f64) = (0.0, 0.0);
            for (((e, want), njoy), exact) in at.iter().zip(want).zip(&njoy).zip(&exact) {
                let relative = (njoy - want).abs() / want;
                worst_njoy = worst_njoy.max(relative);
                assert!(
                    relative < 1.5e-6,
                    "MT{mt} at {t} K, {e} eV: {njoy} against NJOY's {want}"
                );
                let relative = (exact - want) / want;
                worst_exact = worst_exact.max(relative.abs());
                assert!(
                    relative > -1e-6 && relative < 2e-4,
                    "MT{mt} at {t} K, {e} eV: exact {exact} against NJOY's {want}"
                );
            }
            println!(
                "MT{mt} at {t} K: {} points, worst relative difference from NJOY \
                 {worst_njoy:.2e} with BROADR's reach, {worst_exact:.2e} exact",
                at.len()
            );
        }
    }
}

/// Write the NJOY reference from the PENDF tapes NJOY wrote.
///
/// `DOPPLER_NJOY_DIR` names the directory holding `tape21` (RECONR's 0 K
/// PENDF) and `tape22` (BROADR's), made as `tests/reference/README.md` says.
#[test]
#[ignore = "writes the NJOY reference; run on purpose"]
fn regenerate_njoy_reference() {
    let dir = PathBuf::from(std::env::var("DOPPLER_NJOY_DIR").expect("set DOPPLER_NJOY_DIR"));
    let mut text = String::from(
        "# ENDF/B-VIII.1 W186 (MAT 7443) through NJOY2016 RECONR and BROADR, err=0.001.\n\
         # See tests/reference/README.md. Energies in eV, cross sections in barns.\n",
    );
    let cold = endf::get_materials(dir.join("tape21")).unwrap();
    let hot = endf::get_materials(dir.join("tape22")).unwrap();
    let awr = cold[0].mf3(102).unwrap().awr;
    text.push_str(&format!("AWR {awr}\n"));
    // Every point of the 0 K table, every third of the broadened ones, which
    // still samples every resonance BROADR resolved.
    for (material, limit, step) in cold
        .iter()
        .map(|m| (m, 1.05 * REFERENCE_TOP, 1))
        .chain(hot.iter().map(|m| (m, REFERENCE_TOP, 3)))
    {
        let temperature = material.mf1_mt451().unwrap().temp;
        for mt in [2, 102] {
            let sigma = &material.mf3(mt).unwrap().sigma;
            let points: Vec<(f64, f64)> = sigma
                .x
                .iter()
                .zip(&sigma.y)
                .filter(|(e, _)| **e <= limit)
                .step_by(step)
                .map(|(e, s)| (*e, *s))
                .collect();
            text.push_str(&format!("TABLE {temperature} {mt} {}\n", points.len()));
            for (e, s) in points {
                text.push_str(&format!("{e} {s}\n"));
            }
        }
    }
    let file = std::fs::File::create(reference_path()).unwrap();
    let mut xz = lzma_rust2::XzWriter::new(file, lzma_rust2::XzOptions::with_preset(9)).unwrap();
    std::io::Write::write_all(&mut xz, text.as_bytes()).unwrap();
    xz.finish().unwrap();
}

// ---------------------------------------------------------------------------
// The adjoint identity
// ---------------------------------------------------------------------------

/// A small deterministic generator, so the random cases are the same on
/// every run and platform.
struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        // SplitMix64.
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    fn log_uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo * (hi / lo).powf(self.uniform())
    }
}

/// A random 0 K cross section on `[lo, hi]`: a sparse background with a few
/// dense clusters standing in for resonances, and a step.
fn random_cross_section(rng: &mut Rng, lo: f64, hi: f64) -> (Vec<f64>, Vec<f64>) {
    let mut energy: Vec<f64> = (0..120).map(|_| rng.log_uniform(lo, hi)).collect();
    for _ in 0..4 {
        let centre = rng.log_uniform(lo, hi);
        let width = centre * 1e-4 * (1.0 + 20.0 * rng.uniform());
        energy.extend((0..40).map(|_| centre + width * (2.0 * rng.uniform() - 1.0)));
    }
    energy.push(lo);
    energy.push(hi);
    energy.retain(|e| *e >= lo && *e <= hi);
    energy.sort_by(f64::total_cmp);
    // A step: one energy twice.
    let step = energy[energy.len() / 2];
    energy.insert(energy.len() / 2, step);
    let sigma = energy
        .iter()
        .map(|_| 0.1 + 50.0 * rng.uniform().powi(3))
        .collect();
    (energy, sigma)
}

/// The flux density at `e` and its knots, evaluated independently of the
/// crate.
fn flux_at(flux: &Flux, e: f64) -> f64 {
    let group = |edges: &[f64]| {
        (edges[0] <= e && e < edges[edges.len() - 1])
            .then(|| edges.partition_point(|&v| v <= e) - 1)
    };
    match flux {
        Flux::Histogram { edges, density } => group(edges).map_or(0.0, |g| density[g]),
        Flux::Lethargy {
            edges,
            per_lethargy,
        } => group(edges).map_or(0.0, |g| per_lethargy[g] / e),
        Flux::Pointwise { energy, value } => {
            if e < energy[0] || e >= energy[energy.len() - 1] {
                return 0.0;
            }
            let i = energy.partition_point(|&v| v <= e) - 1;
            value[i] + (value[i + 1] - value[i]) * (e - energy[i]) / (energy[i + 1] - energy[i])
        }
    }
}

fn flux_knots(flux: &Flux) -> Vec<f64> {
    match flux {
        Flux::Histogram { edges, .. } | Flux::Lethargy { edges, .. } => edges.clone(),
        Flux::Pointwise { energy, .. } => energy.clone(),
    }
}

/// Eight-point Gauss-Legendre on [-1, 1].
#[allow(clippy::excessive_precision)]
const GL8: [(f64, f64); 8] = [
    (-0.960_289_856_497_536_2, 0.101_228_536_290_376_26),
    (-0.796_666_477_413_626_7, 0.222_381_034_453_374_47),
    (-0.525_532_409_916_329_0, 0.313_706_645_877_887_3),
    (-0.183_434_642_495_649_8, 0.362_683_783_378_362_0),
    (0.183_434_642_495_649_8, 0.362_683_783_378_362_0),
    (0.525_532_409_916_329_0, 0.313_706_645_877_887_3),
    (0.796_666_477_413_626_7, 0.222_381_034_453_374_47),
    (0.960_289_856_497_536_2, 0.101_228_536_290_376_26),
];

/// `∫ σ_T ψ dE`, with `σ_T` from `broaden` below `broaden_below` and the 0 K
/// cross section above it, integrated in `y = √(αE)` (where `σ_T` is smooth on
/// the scale of one) between the flux's knots, and the cross section's above
/// the limit.
fn broadened_rate(
    energy: &[f64],
    sigma: &[f64],
    awr: f64,
    temperature: f64,
    flux: &Flux,
    broaden_below: f64,
) -> f64 {
    let alpha = awr / (endf::K_BOLTZMANN * temperature);
    let mut knots = flux_knots(flux);
    knots.push(broaden_below);
    // Above the limit the integrand is the 0 K cross section itself, which
    // bends at its own points.
    knots.extend(energy.iter().filter(|e| **e > broaden_below));
    knots.sort_by(f64::total_cmp);
    knots.dedup();
    let (lo, hi) = (knots[0], *flux_knots(flux).last().unwrap());
    knots.retain(|k| *k >= lo && *k <= hi);

    let mut nodes = Vec::new();
    let mut weights = Vec::new();
    for w in knots.windows(2) {
        let (ya, yb) = ((alpha * w[0]).sqrt(), (alpha * w[1]).sqrt());
        let mut start = ya;
        while start < yb {
            let end = (start + 0.25f64.min(0.25 * start.max(0.01))).min(yb);
            let (mid, half) = (0.5 * (start + end), 0.5 * (end - start));
            for (u, wt) in GL8 {
                let y = mid + half * u;
                nodes.push(y * y / alpha);
                // dE = 2y dy / α.
                weights.push(wt * half * 2.0 * y / alpha);
            }
            start = end;
        }
    }
    let below: Vec<f64> = nodes
        .iter()
        .copied()
        .filter(|e| *e < broaden_below)
        .collect();
    let hot = broaden(energy, sigma, awr, temperature, &below).unwrap();
    let cold = broaden(energy, sigma, awr, 0.0, &nodes[below.len()..]).unwrap();
    nodes
        .iter()
        .zip(&weights)
        .zip(hot.iter().chain(&cold))
        .map(|((e, w), s)| w * s * flux_at(flux, *e))
        .sum()
}

fn check_identity(
    rng: &mut Rng,
    awr: f64,
    temperature: f64,
    flux: &Flux,
    sigma_range: (f64, f64),
    broaden_below: Option<f64>,
) {
    let weight = broadened_weight(flux, awr, temperature, broaden_below).unwrap();
    for _ in 0..3 {
        let (energy, sigma) = random_cross_section(rng, sigma_range.0, sigma_range.1);
        let direct = broadened_rate(
            &energy,
            &sigma,
            awr,
            temperature,
            flux,
            broaden_below.unwrap_or(f64::INFINITY),
        );
        let adjoint = weight.integrate(&energy, &sigma).unwrap();
        let relative = (adjoint - direct).abs() / direct.abs();
        println!("A={awr} T={temperature}: {relative:.1e} relative");
        assert!(
            relative < 1e-12,
            "A={awr} T={temperature}: ∫σ_T ψ = {direct}, ∫σ₀ w = {adjoint}, {relative:.2e} apart"
        );
    }
}

/// Group edges log-spaced across `[lo, hi]` with some jitter, and random
/// values with a few zero groups, so the flux has sharp edges in both
/// directions.
fn random_groups(rng: &mut Rng, lo: f64, hi: f64, groups: usize) -> (Vec<f64>, Vec<f64>) {
    let mut edges: Vec<f64> = (0..=groups)
        .map(|g| {
            let jitter = if g == 0 || g == groups {
                0.0
            } else {
                0.3 * (rng.uniform() - 0.5)
            };
            lo * (hi / lo).powf((g as f64 + jitter) / groups as f64)
        })
        .collect();
    edges.dedup();
    let values = (0..edges.len() - 1)
        .map(|_| {
            if rng.uniform() < 0.1 {
                0.0
            } else {
                rng.log_uniform(1e-3, 1e3)
            }
        })
        .collect();
    (edges, values)
}

#[test]
fn weight_reproduces_the_broadened_rate_for_a_group_flux() {
    // Tungsten at room temperature, a flat-in-energy group flux with sharp
    // edges from 1 eV to 1 keV.
    let mut rng = Rng(186);
    let (edges, density) = random_groups(&mut rng, 1.0, 1e3, 40);
    let flux = Flux::Histogram { edges, density };
    check_identity(&mut rng, 184.357, 293.6, &flux, (0.3, 3e3), None);
}

#[test]
fn weight_reproduces_the_broadened_rate_at_low_energy() {
    // A light target, hot, where the second exponential and the 1/v
    // continuation below the first point both matter, under a 1/E flux.
    let mut rng = Rng(1);
    let (edges, per_lethargy) = random_groups(&mut rng, 1e-4, 10.0, 30);
    let flux = Flux::Lethargy {
        edges,
        per_lethargy,
    };
    check_identity(&mut rng, 1.0, 600.0, &flux, (1e-3, 50.0), None);
}

#[test]
fn weight_reproduces_the_broadened_rate_for_a_pointwise_flux() {
    // A self-shielded shape: linear between points, with steps at group
    // edges, broadened only below 300 eV as BROADR's thnmax would.
    let mut rng = Rng(56);
    let (edges, _) = random_groups(&mut rng, 0.1, 1e3, 12);
    let mut energy = Vec::new();
    let mut value = Vec::new();
    for w in edges.windows(2) {
        let points = 2 + (rng.uniform() * 6.0) as usize;
        for k in 0..points {
            // The ends exactly on the edges, so each edge is a step.
            energy.push(if k + 1 == points {
                w[1]
            } else {
                w[0] * (w[1] / w[0]).powf(k as f64 / (points - 1) as f64)
            });
            value.push(rng.log_uniform(0.1, 10.0));
        }
    }
    let flux = Flux::Pointwise { energy, value };
    check_identity(&mut rng, 55.0, 1200.0, &flux, (0.01, 2e3), Some(300.0));
}

#[test]
fn weight_is_the_flux_well_above_the_doppler_width() {
    // Far from a group edge, and with E >> kT/A, w is the flux times the
    // free-gas rise of a constant, 2x D(x) = 1 + 1/(2x²) + 3/(4x⁴) + ...
    // (D is Dawson's integral, x² = AE/kT).
    let (awr, temperature) = (184.357, 293.6);
    let alpha = awr / (endf::K_BOLTZMANN * temperature);
    let edges = vec![1.0, 10.0, 100.0, 1e3, 1e4];
    let density = vec![3.0, 0.5, 2.0, 7.0];
    let flux = Flux::Histogram {
        edges: edges.clone(),
        density: density.clone(),
    };
    let weight = broadened_weight(&flux, awr, temperature, None).unwrap();
    for (w, d) in edges.windows(2).zip(&density) {
        let e = (w[0] * w[1]).sqrt();
        let got = weight.at(e) / d;
        let inv = 1.0 / (alpha * e);
        let rise = 0.5 * inv + 0.75 * inv * inv;
        assert!(
            (got - 1.0 - rise).abs() < 1e-11,
            "{e} eV: w/ψ = {got}, expected 1 + {rise:.6e}"
        );
        assert!((got - 1.0).abs() < 1e-3);
    }
    // Outside the flux's support by more than the kernel's reach, nothing.
    assert_eq!(weight.at(0.5), 0.0);
    assert_eq!(weight.at(2e4), 0.0);
}
