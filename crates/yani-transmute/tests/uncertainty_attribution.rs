//! Where the inventory uncertainty comes from.
//!
//! Fe56 irradiated, then cooled, with two independent sources: a 5% sigma on
//! Mn56's half-life and a 10% error on every flux bin. The attribution has to
//! say, for Mn56, how much of its variance each source carries:
//!
//! - by source, exactly: each source alone, resampled, and the two sum to the
//!   total up to sampling noise, since they are independent;
//! - by contributor, to first order: Mn56's half-life is the only half-life
//!   contributor that can move Mn56 after cooling, so its first-order variance
//!   must equal the half-life source's exact one.
//!
//! Self-skips when the Fe56 fixture is missing.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use yamc_materials::Material;
use yani_transmute::flux_uncertainty::FluxError;
use yani_transmute::uncertainty::{DataUncertainty, Source};
use yani_transmute::{transmute_material, MultigroupSpectrum, TransmuteStep};

const HOUR: f64 = 3600.0;

fn chain(relative_sigma: f64) -> Arc<HashMap<String, yani::ChainNuclide>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    let mut chain = yani::parse_chain_arrow(&path).expect("parse chain");
    let mn56 = chain.get_mut("Mn56").expect("Mn56");
    mn56.half_life_uncertainty = Some(mn56.half_life.unwrap() * relative_sigma);
    Arc::new(chain)
}

fn iron() -> Option<Material> {
    let path = yamc_test_cache::nuclide("Fe56")?;
    let mut m = Material::new(
        HashMap::from([("Fe56".to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("Fe56 material");
    m.nuclides.insert("Fe56".to_string(), 8.5e-2);
    m.volume = Some(1.0);
    m.set_temperature("294");
    m.read_nuclear_data(&HashMap::from([("Fe56".to_string(), path)]), None)
        .expect("read Fe56");
    Some(m)
}

fn run(attribution: bool) -> Option<yani_transmute::TransmutationResults> {
    run_with(
        attribution,
        0.05,
        vec![Source::HalfLife, Source::FluxSpectrum],
    )
}

fn run_with(
    attribution: bool,
    relative_sigma: f64,
    sources: Vec<Source>,
) -> Option<yani_transmute::TransmutationResults> {
    let mut material = iron()?;
    let spectrum = MultigroupSpectrum {
        boundaries: vec![5.0e6, 1.0e7, 2.0e7],
        masses: vec![0.5, 0.5],
        flux_error: Some(FluxError::RelativeStdDev(vec![0.10, 0.10])),
    };
    let steps = vec![
        TransmuteStep {
            dt: 2.0 * HOUR,
            irradiation: Some((0, 1.0e14)),
        },
        TransmuteStep {
            dt: 5.0 * HOUR,
            irradiation: None,
        },
    ];
    let request = DataUncertainty {
        seed: 4,
        samples: Some(1024),
        sources,
        attribution,
        ..Default::default()
    };
    Some(
        transmute_material(
            &mut material,
            &[spectrum],
            &steps,
            chain(relative_sigma),
            &Default::default(),
            Default::default(),
            Some(&request),
        )
        .expect("transmute"),
    )
}

#[test]
fn the_breakdown_accounts_for_the_total() {
    let Some(results) = run(true) else {
        eprintln!("skipping -- Fe56 fixture absent");
        return;
    };
    let b = results
        .uncertainty_breakdown(0, "Mn56", 2)
        .expect("asked for");
    let half_life = b.by_source["half_life"];
    let flux = b.by_source["flux_spectrum"];
    assert!(half_life > 0.0 && flux > 0.0);
    // Independent sources: the parts sum to the total, to sampling noise.
    assert!(
        b.unattributed.abs() < 0.1 * b.variance,
        "unattributed {:.3e} of {:.3e}",
        b.unattributed,
        b.variance
    );
    // The only half-life that moves Mn56 is its own, so its first-order
    // contribution is the whole half-life source.
    let own = b
        .contributors
        .iter()
        .find(|(s, n, r, _)| s == "half_life" && n == "Mn56" && r.is_none())
        .expect("Mn56's half-life contributes")
        .3;
    assert!(
        (own / half_life - 1.0).abs() < 0.12,
        "first order {own:.3e} against exact {half_life:.3e}"
    );
    // A 5% half-life is close to linear in the inventory, so first order
    // explains its replicas. The flux has no first-order terms, so its
    // linearity, and that of both sources together, is not applicable rather
    // than read as nonlinear.
    let own = b.linearity["half_life"]
        .as_ref()
        .expect("half-life has terms");
    assert!(own.r2 > 0.99, "r2 {}", own.r2);
    assert!(own.residual_share < 0.02, "residual {}", own.residual_share);
    assert!(!own.flagged);
    assert!(own.by_contributor[&("half_life".to_string(), "Mn56".to_string())] > 0.99);
    assert!(b.linearity["flux_spectrum"].is_none());
    assert!(b.linearity["all"].is_none());
    // The initial composition carries nothing.
    assert_eq!(
        results
            .uncertainty_breakdown(0, "Mn56", 0)
            .unwrap()
            .variance,
        0.0
    );
}

/// Not asked for, not computed, and the totals do not move either way.
#[test]
fn attribution_is_off_unless_asked_and_changes_no_total() {
    let (Some(with), Some(without)) = (run(true), run(false)) else {
        eprintln!("skipping -- Fe56 fixture absent");
        return;
    };
    assert!(without.uncertainty_breakdown(0, "Mn56", 2).is_none());
    let a = with.uncertainty[&0].std_dev_at(1)["Mn56"];
    let b = without.uncertainty[&0].std_dev_at(1)["Mn56"];
    assert_eq!(a.to_bits(), b.to_bits());
}

/// A wide half-life moves the inventory nonlinearly, and the check sees it:
/// first order explains less of an 80% half-life's replicas than of a 5% one's.
#[test]
fn a_wide_input_reads_as_less_linear() {
    let (Some(narrow), Some(wide)) = (
        run_with(true, 0.05, vec![Source::HalfLife]),
        run_with(true, 0.80, vec![Source::HalfLife]),
    ) else {
        eprintln!("skipping -- Fe56 fixture absent");
        return;
    };
    let linearity = |r: &yani_transmute::TransmutationResults| {
        r.uncertainty_breakdown(0, "Mn56", 2)
            .expect("asked for")
            .linearity["all"]
            .clone()
            .expect("half-life alone is covered")
    };
    let (n, w) = (linearity(&narrow), linearity(&wide));
    assert!(w.residual_share > n.residual_share, "{w:?} against {n:?}");
    assert!(w.r2 < n.r2, "{w:?} against {n:?}");
}
