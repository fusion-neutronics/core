//! Acceptance test for #58 (E13b): a model whose source list mixes a neutron
//! source and a photon source.
//!
//! Each history is one source particle, drawn by source strength. The GPU runs
//! the neutron share through the COUPLED kernel (which also emits secondary
//! photons -- the CPU auto-enables this for any photon-source model), and the
//! photon pass transports both those neutron-induced secondaries and the primary
//! photon source, folding everything per source particle. Pre-#58 such a model
//! was rejected at translation (`NonNeutronSource` / `PhotonSourceOnNeutronPath`).
//!
//! The GPU result must match the CPU mixed-source run statistically (PCG-32 vs
//! 64-bit RNG => parity within MC error, not bit-exact). We check three tally
//! routings: a neutron-filtered flux (neutron pass only), a photon-filtered flux
//! (primary source + neutron-induced secondaries), and an unfiltered flux (the
//! all-particle SUM = the "dual" path). The photon-filtered flux exceeds the
//! pure-photon-source flux precisely because of those secondaries. Self-skips
//! without the Fe data or an f64 GPU adapter.

#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings, Verbose};
use yamc_materials::Material;
use yamc_particle::particle::ParticleType;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::particle_type::ParticleTypeFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::{FluxScore, Score};
use yamc_tallies::tally::Tally;

const SEED: u64 = 990_058;
const N_PER_BATCH: usize = 20_000;
const N_BATCHES: usize = 8;

fn data_present() -> bool {
    std::path::Path::new("tests/Fe56.arrow").exists()
        && std::path::Path::new("tests/Fe.arrow").exists()
}

fn flux_tally(cell_id: u32, particle: Option<ParticleType>) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(cell_id)));
    if let Some(p) = particle {
        t.filters
            .push(Filter::ParticleType(ParticleTypeFilter::new(p)));
    }
    t.scores = vec![Score::Flux(FluxScore)];
    t.initialize_batches(N_BATCHES);
    Arc::new(t)
}

/// Fe sphere with BOTH a 14 MeV neutron source and a 2 MeV photon source, equal
/// strength (so ~half the histories are neutrons, half photons). Returns the
/// model plus (neutron-flux, photon-flux, total-flux) tallies.
fn mixed_fe_sphere() -> (Model, Arc<Tally>, Arc<Tally>, Arc<Tally>, TransportSettings) {
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

    let mut material = Material::new(
        HashMap::from([("Fe56".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.874),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let nm = HashMap::from([("Fe56".to_string(), "tests/Fe56.arrow".to_string())]);
    let photon_paths = HashMap::from([("Fe".to_string(), "tests/Fe.arrow".to_string())]);
    material
        .read_nuclear_data(&nm, Some(&photon_paths))
        .unwrap();
    material.init_photon_data(&photon_paths).unwrap();

    let cell = Cell::new(Some(1), region, Some("fe".into()), Some(0));
    let cell_id = cell.cell_id.unwrap();
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();

    let neutron_src = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![14.0e6], vec![1.0]).unwrap()),
        strength: 1.0,
    });
    let photon_src = ParticleSource::Photon(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![2.0e6], vec![1.0]).unwrap()),
        strength: 1.0,
    });

    let neutron_t = flux_tally(cell_id, Some(ParticleType::Neutron));
    let photon_t = flux_tally(cell_id, Some(ParticleType::Photon));
    let total_t = flux_tally(cell_id, None);

    let mut model = Model::new(
        geometry,
        vec![neutron_src, photon_src],
        vec![
            Arc::clone(&neutron_t),
            Arc::clone(&photon_t),
            Arc::clone(&total_t),
        ],
    );
    model.verbose = Verbose::silent();
    model.max_steps_per_particle = 5_000;
    model.photon_cutoff_energy = 1000.0;
    let settings = TransportSettings {
        total_particles: Some(N_PER_BATCH * N_BATCHES),
        seed: SEED,
        threads: Some(1),
        ..Default::default()
    };
    (model, neutron_t, photon_t, total_t, settings)
}

fn sum(t: &Arc<Tally>) -> f64 {
    t.get_mean().iter().sum()
}

#[test]
fn gpu_mixed_neutron_photon_source_matches_cpu() {
    if !data_present() {
        eprintln!("skipping gpu_mixed_neutron_photon_source_matches_cpu -- Fe data not found");
        return;
    }
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }

    // CPU reference (the mixed source "just works": sample_source picks a source
    // by strength, then transports that particle with its own physics).
    let (mut cpu_m, cpu_n, cpu_p, cpu_tot, settings) = mixed_fe_sphere();
    cpu_m.simulate_transport(&settings).unwrap();
    let (cn, cp, ct) = (sum(&cpu_n), sum(&cpu_p), sum(&cpu_tot));

    // GPU mixed run (must not reject).
    let (mut gpu_m, gpu_n, gpu_p, gpu_tot, settings) = mixed_fe_sphere();
    yamc::gpu::run_on_gpu(&mut gpu_m, &settings)
        .expect("mixed neutron+photon GPU dispatch must succeed");
    let (gn, gp, gt) = (sum(&gpu_n), sum(&gpu_p), sum(&gpu_tot));

    eprintln!(
        "mixed Fe sphere per-source-particle flux:\n  neutron  CPU={cn:.5e} GPU={gn:.5e} ratio={:.3}\n  photon   CPU={cp:.5e} GPU={gp:.5e} ratio={:.3}\n  total    CPU={ct:.5e} GPU={gt:.5e} ratio={:.3}",
        gn / cn,
        gp / cp,
        gt / ct,
    );

    assert!(
        cn > 0.0 && cp > 0.0 && gn > 0.0 && gp > 0.0,
        "both source types must produce flux on both backends (cn={cn} cp={cp} gn={gn} gp={gp})"
    );

    // Neutron-filtered tally: neutron pass only. Statistical parity.
    let rn = gn / cn;
    assert!(
        (0.90..=1.10).contains(&rn),
        "neutron-flux GPU/CPU ratio {rn:.3} outside [0.90, 1.10]"
    );
    // Photon-filtered tally: photon pass only.
    let rp = gp / cp;
    assert!(
        (0.88..=1.12).contains(&rp),
        "photon-flux GPU/CPU ratio {rp:.3} outside [0.88, 1.12]"
    );
    // Unfiltered (all-particle) tally: the neutron + photon SUM (dual path).
    let rt = gt / ct;
    assert!(
        (0.90..=1.10).contains(&rt),
        "total-flux GPU/CPU ratio {rt:.3} outside [0.90, 1.10]"
    );

    // The unfiltered total must equal the neutron + photon parts (per source
    // particle) on each backend, to MC error -- confirms the dual routing sums
    // the two passes rather than dropping or double-counting.
    let cpu_consistency = (cn + cp) / ct;
    let gpu_consistency = (gn + gp) / gt;
    assert!(
        (0.97..=1.03).contains(&cpu_consistency),
        "CPU total != neutron+photon (ratio {cpu_consistency:.3})"
    );
    assert!(
        (0.97..=1.03).contains(&gpu_consistency),
        "GPU total != neutron+photon (ratio {gpu_consistency:.3}) -- dual routing wrong"
    );
}
