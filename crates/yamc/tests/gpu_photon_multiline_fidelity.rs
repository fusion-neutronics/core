//! GPU-vs-CPU photon spectral fidelity for a multi-line source in a
//! high-Z material whose photon data carries NO atomic-relaxation
//! tables (fendl-3.2d Pb).
//!
//! Such data (issue #41) has all subshell binding energies = 0, so the
//! CPU photoelectric path gives the photoelectron the FULL photon energy
//! and TTB-radiates the whole bremsstrahlung continuum. The GPU kernel
//! previously fell back to the Doppler K-shell binding energy (~88 keV
//! for Pb) when relaxation data was absent, robbing the photoelectron of
//! most of its kinetic energy and silencing the low-energy brem tail --
//! a roughly uniform spectral deficit on complex decay spectra. This
//! test pins the photoelectron-KE convention to the CPU's so the gap
//! stays closed.
//!
//! Needs the fendl-3.2d Pb photon Arrow (auto-downloaded via the keyword,
//! or a local cache); skips cleanly when unavailable (offline / no GPU).

#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::energy::EnergyFilter;
use yamc_tallies::filter::particle_type::ParticleTypeFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::{FluxScore, Score};
use yamc_tallies::tally::Tally;

/// Locally-cached fendl-3.2d Pb Arrow directories, if present. Lets the
/// test run on a box without network access. When absent the test falls
/// back to the `fendl-3.2d` keyword (auto-download).
const LOCAL_PB_PHOTON: &str =
    "/home/jon/yamc-org/cross_section_data_fendl_3.2d_arrow/fendl-3.2d-arrow/photon/Pb.arrow";
const LOCAL_PB208_NEUTRON: &str =
    "/home/jon/yamc-org/cross_section_data_fendl_3.2d_arrow/fendl-3.2d-arrow/neutron/Pb208.arrow";

fn bins() -> Vec<f64> {
    // 30 log-spaced bins, 1 keV (cutoff) to 2 MeV.
    let (lo, hi, n) = (1.0e3_f64, 2.0e6_f64, 30usize);
    let (ln_lo, ln_hi) = (lo.ln(), hi.ln());
    (0..=n)
        .map(|i| (ln_lo + (ln_hi - ln_lo) * (i as f64) / (n as f64)).exp())
        .collect()
}

/// Build a Pb sphere with a multi-line photon source. Returns `None`
/// (skip) when the AR-less fendl-3.2d Pb photon data can't be resolved.
fn build(lines: &[(f64, f64)]) -> Option<(Model, Arc<Tally>, TransportSettings)> {
    let sphere = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 1.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });
    let region = Region::new_from_halfspace(HalfspaceType::Below(sphere));
    let mut material = Material::new(
        HashMap::from([("Pb208".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(11.35),
    )
    .ok()?;
    material.set_material_id(1);
    material.set_temperature("294");

    // Photon + neutron data: prefer the local fendl-3.2d Pb cache, else
    // the keyword (auto-downloads). The point is the AR-less fendl-3.2d
    // PHOTON data; the neutron data only needs to load the nuclide.
    let pb_photon = if std::path::Path::new(LOCAL_PB_PHOTON).is_dir() {
        LOCAL_PB_PHOTON.to_string()
    } else {
        "fendl-3.2d".to_string()
    };
    let pb_neutron = if std::path::Path::new(LOCAL_PB208_NEUTRON).is_dir() {
        LOCAL_PB208_NEUTRON.to_string()
    } else {
        "fendl-3.2d".to_string()
    };
    let neutron = HashMap::from([("Pb208".to_string(), pb_neutron)]);
    let photon = HashMap::from([("Pb".to_string(), pb_photon)]);
    if material.read_nuclear_data(&neutron, Some(&photon)).is_err() {
        eprintln!("skipping -- could not resolve fendl-3.2d Pb data (offline?)");
        return None;
    }
    if material.init_photon_data(&photon).is_err() {
        eprintln!("skipping -- could not load fendl-3.2d Pb photon data (offline?)");
        return None;
    }

    let cell = Cell::new(Some(1), region, Some("pb".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).ok()?;

    let es: Vec<f64> = lines.iter().map(|l| l.0).collect();
    let ws: Vec<f64> = lines.iter().map(|l| l.1).collect();
    let source = ParticleSource::Photon(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(es, ws).ok()?),
        strength: 1.0,
    });

    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(1)));
    t.filters.push(Filter::ParticleType(ParticleTypeFilter::new(
        yamc_particle::particle::ParticleType::Photon,
    )));
    t.filters.push(Filter::Energy(EnergyFilter::new(bins())));
    t.scores = vec![Score::Flux(FluxScore)];
    t.initialize_batches(8);
    let t = Arc::new(t);

    let mut model = Model::new(geometry, vec![source], vec![t.clone()]);
    model.gpu_max_steps_per_particle = 10_000;
    model.transport_secondary_photons = true;
    let settings = TransportSettings {
        total_particles: Some(20_000 * 8),
        seed: 42,
        ..Default::default()
    };
    Some((model, t, settings))
}

#[test]
fn gpu_photon_multiline_pb_spectrum_matches_cpu() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }

    // Actinide-like multi-line spectrum spanning the photoelectric-
    // dominated low-energy region (where the brem tail lives) up to a
    // Compton-dominated 1.5 MeV line.
    let lines = [
        (3.0e4, 0.30),
        (6.0e4, 0.25),
        (1.0e5, 0.15),
        (2.0e5, 0.10),
        (4.0e5, 0.08),
        (8.0e5, 0.07),
        (1.5e6, 0.05),
    ];

    let Some((mut cpu_m, cpu_t, settings)) = build(&lines) else {
        return;
    };
    cpu_m
        .simulate_transport(&TransportSettings {
            threads: Some(1),
            ..settings
        })
        .unwrap();
    let cpu = cpu_t.get_mean();

    let Some((mut gpu_m, gpu_t, settings)) = build(&lines) else {
        return;
    };
    yamc::gpu::run_on_gpu(&mut gpu_m, &settings).expect("GPU dispatch");
    let gpu = gpu_t.get_mean();

    let edges = bins();
    let cpu_tot: f64 = cpu.iter().sum();
    let gpu_tot: f64 = gpu.iter().sum();
    eprintln!("\n| bin | E_lo (eV) | E_hi (eV) | CPU | GPU | GPU/CPU |");
    for i in 0..cpu.len() {
        let r = if cpu[i] > 0.0 {
            gpu[i] / cpu[i]
        } else {
            f64::NAN
        };
        eprintln!(
            "| {i} | {:.3e} | {:.3e} | {:.4e} | {:.4e} | {:.3} |",
            edges[i],
            edges[i + 1],
            cpu[i],
            gpu[i],
            r
        );
    }
    eprintln!(
        "TOTAL CPU={cpu_tot:.5e} GPU={gpu_tot:.5e} GPU/CPU={:.4}",
        gpu_tot / cpu_tot
    );

    // 1. Integrated flux agreement (the headline number).
    let total_ratio = gpu_tot / cpu_tot;
    assert!(
        (0.97..1.03).contains(&total_ratio),
        "multiline Pb GPU/CPU integrated photon flux {total_ratio:.4} outside [0.97, 1.03]"
    );

    // 2. Pure bremsstrahlung / scatter continuum BELOW the lowest source
    //    line (30 keV). Bins 0..=12 span 1 keV..~27 keV, so they contain
    //    no source-line peak -- every count there is a secondary photon,
    //    overwhelmingly photoelectron TTB. This is the band the
    //    Doppler-binding fallback suppressed (pre-fix it ran ~0.6x, with
    //    the deepest bins zeroed). Asserting on the continuum sum (which
    //    no peak bin can mask) makes this a real regression catcher.
    let cpu_cont: f64 = cpu[0..=12].iter().sum();
    let gpu_cont: f64 = gpu[0..=12].iter().sum();
    assert!(cpu_cont > 0.0, "CPU brem continuum empty -- bad test setup");
    let cont_ratio = gpu_cont / cpu_cont;
    assert!(
        (0.90..1.10).contains(&cont_ratio),
        "multiline Pb GPU/CPU bremsstrahlung-continuum (1-27 keV, below the lowest \
         source line) flux ratio {cont_ratio:.4} outside [0.90, 1.10] -- the \
         photoelectron-KE convention has regressed (GPU under-producing the brem \
         continuum vs CPU; see the Doppler-binding fallback in the photoelectric branch)"
    );
}
