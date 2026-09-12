//! Issue #362: what a capture / absorption reaction-rate tally reports on a
//! fissile nuclide, and above the charged-particle thresholds.
//!
//! Two paths serve a reaction-rate tally, and only one was wrong.
//!
//! OUTSIDE a URR band, `Material::macro_xs_by_mt` falls through to
//! `lookup_xs_by_mt`, which reads the library's own per-MT macroscopic table. MT
//! 102 and MT 27 were always correct there.
//!
//! INSIDE a band it takes `u.capture` / `u.absorption` from the URR sample, and
//! `urr_sample_for_nuclide` derived capture as `(xs_absorption -
//! xs_fission).max(0.0)`. That is OpenMC's expression but not OpenMC's input:
//! `nuclide.cpp` computes `capture *= (micro.absorption - micro.fission)` on an
//! absorption that INCLUDES fission, whereas `FastXSGrid::lookup` returns a
//! disappearance partial that excludes it (its four partials sum to the total).
//! So in-band capture clamped to zero for every nuclide whose fission exceeds its
//! capture (U235 at 10 keV: absorption 1.06 b against fission 2.91 b), an MT 102
//! tally read exactly zero, and an absorption tally -- built as `macro_capture +
//! macro_fission` -- reported just the fission rate, the two agreeing to every
//! digit. Same mistake as #154, which was the transport-side twin.
//!
//! The convention is OpenMC's: `micro.absorption = capture + fission`
//! (`nuclide.cpp`), scored as `macro_xs().absorption * flux`, so MT 27 includes
//! fission and MT 102 is the library's own radiative-capture column.

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::energy::EnergyFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::Score;
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;

const N: usize = 40_000;

fn cache(n: &str) -> String {
    yamc_test_cache::nuclide_path(n)
}

fn tally(score: &str, bins: Option<Vec<f64>>) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(1)));
    if let Some(b) = bins {
        t.filters.push(Filter::Energy(EnergyFilter::new(b)));
    }
    t.scores = vec![score.parse::<Score>().expect("score name")];
    t.estimator = Estimator::TrackLength;
    t.initialize_batches(1);
    Arc::new(t)
}

/// `(summed tally means, in the order given)` for one sphere run.
fn run(nuclide: &str, density: f64, energy_ev: f64, tallies: Vec<Arc<Tally>>) -> Option<Vec<f64>> {
    if !std::path::Path::new(&cache(nuclide)).exists() {
        return None;
    }
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 5.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
    let mut m = Material::new(
        HashMap::from([(nuclide.to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(density),
    )
    .unwrap();
    m.set_material_id(1);
    m.set_temperature("294");
    // A skip, not a panic, like the absence check above. Since #389 a cache
    // directory is routinely populated at activation scope, holding cross
    // sections and none of the transport sections this needs, and the directory
    // exists either way. CI fetches only the fixture list and skips this
    // outright; without this, any developer machine that has run a
    // transmutation fails it instead.
    if m.read_nuclear_data(
        &HashMap::from([(nuclide.to_string(), cache(nuclide))]),
        None,
    )
    .is_err()
    {
        eprintln!("skipping {nuclide} -- cached at a narrower scope than full");
        return None;
    }
    let cell = Cell::new(Some(1), region, Some("c".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(m)]).unwrap();
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![energy_ev], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });
    let mut model = Model::new(
        geometry,
        vec![source],
        tallies.iter().map(Arc::clone).collect(),
    );
    model.verbose = Verbose::silent();
    model.gpu_max_steps_per_particle = 20_000;
    model.tracking_mode = TrackingMode::Surface;
    let settings = TransportSettings {
        total_particles: Some(N),
        seed: 4242,
        ..Default::default()
    };
    model.simulate_transport(&settings).expect("cpu run");
    Some(tallies.iter().map(|t| t.get_mean().iter().sum()).collect())
}

/// Absorption must EXCEED fission on a fissile material, because it is
/// `capture + fission`. Pre-fix the two were equal to seven digits.
#[test]
fn absorption_exceeds_fission_on_a_fissile_material() {
    // Scored INSIDE U235's URR band (2250 .. 24999 eV): that is where the
    // scoring path takes its cross sections from the URR sample rather than the
    // library's per-MT tables, and where the capture component was lost.
    let band = Some(vec![2250.0, 24999.0]);
    let Some(v) = run(
        "U235",
        18.95,
        1.0e6,
        vec![tally("absorption", band.clone()), tally("fission", band)],
    ) else {
        eprintln!("skipping -- endf-b8.1-U235.arrow cache absent");
        return;
    };
    let (absorption, fission) = (v[0], v[1]);
    assert!(
        fission > 0.0,
        "expected a nonzero fission rate, got {fission:.6e}"
    );
    assert!(
        absorption > fission * 1.01,
        "absorption {absorption:.8e} must exceed fission {fission:.8e} by the capture \
         rate; equal values mean the capture component was lost (#362)"
    );
}

/// An MT 102 tally must not read zero on a fissile nuclide.
#[test]
fn mt102_capture_is_nonzero_on_a_fissile_material() {
    let band = Some(vec![2250.0, 24999.0]);
    let Some(v) = run("U235", 18.95, 1.0e6, vec![tally("102", band)]) else {
        eprintln!("skipping -- endf-b8.1-U235.arrow cache absent");
        return;
    };
    assert!(
        v[0] > 0.0,
        "MT 102 radiative capture on U235 reads {:.6e}; the derived form clamped it to \
         zero for every fissile nuclide (#362)",
        v[0]
    );
}

/// INVARIANT, not a regression guard: this one passes on the pre-fix code too,
/// because above the band the tally path reads the library's per-MT table rather
/// than a derived value. Kept because it pins the distinction the derived form
/// blurred -- MT 102 is radiative capture, and above the (n,p) / (n,alpha)
/// thresholds it is orders of magnitude below neutron disappearance (for Fe56 at
/// 14 MeV the disappearance partial is 149x the capture column).
#[test]
fn mt102_is_capture_not_disappearance_at_fast_energies() {
    let fast = Some(vec![1.0e7, 2.0e7]);
    let Some(v) = run(
        "Fe56",
        7.87,
        1.406e7,
        vec![tally("102", fast.clone()), tally("101", fast)],
    ) else {
        eprintln!("skipping -- endf-b8.1-Fe56.arrow cache absent");
        return;
    };
    let (mt102, mt101) = (v[0], v[1]);
    assert!(
        mt101 > 0.0,
        "expected nonzero disappearance above 10 MeV, got {mt101:.6e}"
    );
    assert!(
        mt102 < 0.1 * mt101,
        "above 10 MeV Fe56's radiative capture {mt102:.6e} should be far below its \
         disappearance {mt101:.6e} (the charged-particle channels dominate there); \
         equal values mean MT 102 is being served the disappearance partial (#362)"
    );
}
