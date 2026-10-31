//! Sample CPU's TTB bremsstrahlung function directly with controlled
//! electron energies. Print resulting photon energy distribution to
//! compare against GPU's TTB samples.

#![cfg(feature = "gpu")]

use std::collections::HashMap;
use std::sync::Arc;

use rand::SeedableRng;
use yamc::geo::{BoundaryType, Surface, SurfaceKind};
use yamc::geo::{HalfspaceType, Region};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::Model;
use yamc_materials::Material;
use yamc_physics::photon::bremsstrahlung::thick_target_bremsstrahlung;
use yamc_physics::util::bank::ParticleBank;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};

fn make_model(element: &str, nuclide: &str) -> Model {
    let cylinder = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Cylinder {
            axis: [0.0, 0.0, 1.0],
            origin: [0.0, 0.0, 0.0],
            radius: 1.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let z_bot = Surface {
        surface_id: Some(2),
        kind: SurfaceKind::Plane {
            a: 0.0,
            b: 0.0,
            c: 1.0,
            d: -50.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let z_top = Surface {
        surface_id: Some(3),
        kind: SurfaceKind::Plane {
            a: 0.0,
            b: 0.0,
            c: 1.0,
            d: 50.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(cylinder)))
        .intersection(&Region::new_from_halfspace(HalfspaceType::Above(Arc::new(
            z_bot,
        ))))
        .intersection(&Region::new_from_halfspace(HalfspaceType::Below(Arc::new(
            z_top,
        ))));

    let mut material = Material::new(
        HashMap::from([(nuclide.to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(1.0),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let mut nm = HashMap::new();
    nm.insert(
        nuclide.to_string(),
        format!("/home/jon/yamc-org/cross_section_data_endf_b8.1_arrow/endf-b8.1-arrow/neutron/{}.arrow", nuclide),
    );
    let mut photon_paths: HashMap<String, String> = HashMap::new();
    photon_paths.insert(
        element.to_string(),
        format!(
            "/home/jon/yamc-org/cross_section_data_endf_b8.1_arrow/endf-b8.1-arrow/photon/{}.arrow",
            element
        ),
    );
    material
        .read_nuclear_data(&nm, Some(&photon_paths))
        .unwrap();
    let cell = Cell::new(Some(1), region, Some("c".into()), Some(0));
    let geometry = Geometry::new(vec![cell.clone()], vec![Arc::new(material)]).unwrap();
    let source = ParticleSource::Photon(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![1.0e6_f64], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });
    let mut model = Model::new(geometry, vec![source], vec![]);
    model.transport_secondary_photons = true;
    model.electron_treatment = yamc::model::ElectronTreatment::Ttb;
    model
}

fn main() {
    for (element, nuclide) in &[("Be", "Be9")] {
        let mut model = make_model(element, nuclide);
        model.ensure_photon_data_for_gpu().unwrap();
        let mat = &model.geometry.materials()[0];
        let ttb_data = mat.ttb.as_ref().unwrap();

        for &electron_ke in &[50e3_f64, 100e3, 200e3, 400e3] {
            println!(
                "\n=== {} electron at {:.0} keV ===",
                element,
                electron_ke / 1e3
            );
            let mut bank = ParticleBank::new();
            let mut rng = rand::rngs::StdRng::seed_from_u64(42);
            let mut energies: Vec<f64> = Vec::new();
            // Call TTB sampler N times.
            for _ in 0..10_000 {
                let pos = [0.0, 0.0, 0.0];
                let dir = [0.0, 0.0, 1.0];
                bank.clear();
                let _ = thick_target_bremsstrahlung(
                    electron_ke,
                    false,
                    ttb_data,
                    pos,
                    dir,
                    1.0,
                    1000.0,
                    &mut bank,
                    &mut rng,
                );
                while let Some(p) = bank.pop_particle() {
                    energies.push(p.energy);
                }
            }
            println!(
                "Total TTB photons: {} (avg per call: {:.4})",
                energies.len(),
                energies.len() as f64 / 10_000.0
            );
            let mut bins = [0usize; 6];
            for &e in &energies {
                let b = if e < 1e4 {
                    0
                } else if e < 5e4 {
                    1
                } else if e < 1e5 {
                    2
                } else if e < 5e5 {
                    3
                } else if e < 1e6 {
                    4
                } else {
                    5
                };
                bins[b] += 1;
            }
            let n = energies.len() as f64;
            if n > 0.0 {
                println!("E bins (1-10k|10-50k|50-100k|100-500k|500k-1M|>1M):");
                for (i, label) in ["1-10k", "10-50k", "50-100k", "100-500k", "500k-1M", ">1M"]
                    .iter()
                    .enumerate()
                {
                    println!(
                        "  {:>8}: {:>6} ({:5.2}%)",
                        label,
                        bins[i],
                        100.0 * bins[i] as f64 / n
                    );
                }
                let total_e: f64 = energies.iter().sum();
                println!("Mean E: {:.3e} eV", total_e / n);
            }
        }
    }
}
