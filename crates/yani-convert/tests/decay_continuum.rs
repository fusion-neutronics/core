//! A real decay continuum, from the evaluation to yani's reader.
//!
//! JEFF-4.0 Cf252 gives its gamma spectrum as three weak lines beside a
//! linear-linear continuum, the spontaneous-fission photons, with FC = 0.249957
//! photons per decay. That continuum is nearly all of the nuclide's photon
//! emission, and reading its per-eV values as line intensities would put it
//! low by a factor of about its grid spacing in eV (roughly 2e5 here).

use endf::chain::Chain;
use endf::{Decay, Material};
use yani::{Continuum, DecaySourceDistribution, Interpolation};

const CF252: &[u8] = include_bytes!("../../endf/fixtures/dec-098_Cf_252.jeff40.endf.xz");

fn material() -> Material {
    let mut text = Vec::new();
    lzma_rs::xz_decompress(&mut &CF252[..], &mut text).expect("fixture decompresses");
    Material::from_str(&String::from_utf8(text).expect("fixture is UTF-8")).expect("parses")
}

/// Converted and read back through `yani::parse_chain_parts`, the reader a run
/// uses: the Cf252 photon sources as yani holds them. `tag` keeps the tests'
/// directories apart, since they run at once.
fn photon_sources(tag: &str) -> Vec<DecaySourceDistribution> {
    let decay = [material()];
    let chain = Chain::from_endf(
        &decay,
        &[],
        &endf::chain::q_values(&[]),
        &endf::chain::DEFAULT_REACTIONS,
    )
    .expect("chain builds");
    let sources = yani_convert::decay_sources(&decay).expect("sources read");
    let dir = std::env::temp_dir().join(format!("yani-convert-cf252-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    yani_convert::write_decay(&chain, &sources, &dir.join("decay")).expect("decay written");
    let (back, _branch) =
        yani::parse_chain_parts(&dir.join("decay"), None, None, None).expect("yani reads it");
    let _ = std::fs::remove_dir_all(&dir);
    back["Cf252"]
        .sources
        .iter()
        .filter(|s| s.particle == "photon")
        .map(|s| s.distribution.clone())
        .collect()
}

#[test]
fn the_continuum_arrives_as_a_continuum_with_its_law() {
    let sources = photon_sources("law");
    let continua: Vec<&DecaySourceDistribution> = sources
        .iter()
        .filter(|d| matches!(d, DecaySourceDistribution::Tabular { .. }))
        .collect();
    assert_eq!(continua.len(), 1, "{sources:?}");
    let DecaySourceDistribution::Tabular {
        energies,
        interpolation,
        ..
    } = continua[0]
    else {
        unreachable!()
    };
    assert_eq!(*interpolation, Some(Interpolation::LinearLinear));
    // The tape's own grid, 0 to 10 MeV in 15 points, stored as it was given.
    assert_eq!(energies.len(), 15);
    assert_eq!((energies[0], energies[14]), (0.0, 1.0e7));
    assert!(
        sources
            .iter()
            .any(|d| matches!(d, DecaySourceDistribution::Discrete { .. })),
        "the gamma and x-ray lines stay lines"
    );
}

/// Photons per decay from the continuum: FC times the integral of RP, as the
/// endf crate's own integral of the tape's TAB1 gives it. Summing the per-eV
/// values instead, as the readers did, gives less by a factor of about the grid
/// spacing in eV.
#[test]
fn the_continuum_emits_what_the_evaluation_states() {
    let decay = Decay::from_material(&material()).expect("decay data");
    let lambda = decay.decay_constant().expect("unstable").0;
    let gamma = &decay.spectra["gamma"];
    let rp = gamma.continuous.as_ref().expect("a continuum");
    let per_decay = gamma.continuous_normalization.0 * rp.integral().last().copied().unwrap();
    assert!((per_decay / 0.249957 - 1.0).abs() < 2e-5, "{per_decay}");

    let sources = photon_sources("rate");
    let continuum = sources
        .iter()
        .find(|d| matches!(d, DecaySourceDistribution::Tabular { .. }))
        .unwrap();
    let rate = continuum.emission_rate().expect("the law is stated");
    assert!(
        (rate / lambda / per_decay - 1.0).abs() < 1e-12,
        "{} photons per decay, the evaluation gives {per_decay}",
        rate / lambda
    );

    let DecaySourceDistribution::Tabular { intensities, .. } = continuum else {
        unreachable!()
    };
    let as_lines: f64 = intensities.iter().sum::<f64>() / lambda;
    assert!(
        as_lines < 1e-4 * per_decay,
        "{as_lines} against {per_decay}"
    );
}

/// The continuum's mean energy, read linear-linear, closes the evaluation's
/// gamma energy balance: FD times the line energies plus FC times the first
/// moment of RP is the stated mean gamma energy ER_AV to 0.2%. Read as a
/// histogram the same points miss it by far more, so this pins the law as
/// well as the units.
#[test]
fn the_linear_linear_reading_closes_the_energy_balance() {
    let decay = Decay::from_material(&material()).expect("decay data");
    let lambda = decay.decay_constant().expect("unstable").0;
    let gamma = &decay.spectra["gamma"];
    let lines: f64 = gamma
        .discrete
        .iter()
        .map(|l| gamma.discrete_normalization.0 * l.intensity.0 * l.energy.0)
        .sum();

    let sources = photon_sources("energy");
    let DecaySourceDistribution::Tabular {
        energies,
        intensities,
        ..
    } = sources
        .iter()
        .find(|d| matches!(d, DecaySourceDistribution::Tabular { .. }))
        .unwrap()
    else {
        unreachable!()
    };
    // The first moment of each piece, per decay, under either reading.
    let (mut linear, mut histogram) = (0.0, 0.0);
    for i in 0..energies.len() - 1 {
        let (a, b) = (energies[i], energies[i + 1]);
        let (ya, yb) = (intensities[i] / lambda, intensities[i + 1] / lambda);
        let slope = (yb - ya) / (b - a);
        linear += (ya - slope * a) * (b * b - a * a) / 2.0 + slope * (b.powi(3) - a.powi(3)) / 3.0;
        histogram += ya * (b * b - a * a) / 2.0;
    }
    let stated = gamma.energy_average.0;
    assert!(
        ((lines + linear) / stated - 1.0).abs() < 2e-3,
        "{} eV against ER_AV {stated}",
        lines + linear
    );
    assert!(
        ((lines + histogram) / stated - 1.0).abs() > 0.05,
        "the histogram reading should not close it: {}",
        lines + histogram
    );
}

/// The share of the continuum below the dose tables, which a contact dose
/// leaves out as it does a line there: 1 keV for the absorbed-air quantity and
/// 10 keV for the effective dose. The tape's first interval ramps linearly from
/// zero at 0 eV to 140 keV, so the share below a cut `c` is the triangle
/// `c * density(c) / 2` over the whole integral: 3.3e-4 below 10 keV and 3.3e-6
/// below 1 keV, measured here so the exclusion is shown to be small rather
/// than assumed.
#[test]
fn the_part_below_the_dose_tables_is_small() {
    let sources = photon_sources("below");
    let DecaySourceDistribution::Tabular {
        energies,
        intensities,
        interpolation,
    } = sources
        .iter()
        .find(|d| matches!(d, DecaySourceDistribution::Tabular { .. }))
        .unwrap()
    else {
        unreachable!()
    };
    assert_eq!(
        (energies[0], intensities[0], energies[1]),
        (0.0, 0.0, 1.4e5)
    );
    let continuum = Continuum::new(energies, intensities, *interpolation).expect("readable");
    let share = |cut: f64| 0.5 * cut * continuum.density(cut) / continuum.integral();
    let (below_1kev, below_10kev) = (share(1.0e3), share(1.0e4));
    assert!(
        (below_10kev / 3.2734e-4 - 1.0).abs() < 1e-3,
        "{below_10kev}"
    );
    assert!((below_1kev / 3.2734e-6 - 1.0).abs() < 1e-3, "{below_1kev}");
}
