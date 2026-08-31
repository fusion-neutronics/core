//! Print TTB electron-yield values for Be / Fe so we can sanity-check
//! the kernel's N sampling. Compares against expectation: in pure Be
//! (Z=4), yield should be small (low Z = weak bremsstrahlung); in Fe
//! (Z=26) somewhat higher.
//!
//! Run with: `cargo run -p yamc --features gpu --example probe_ttb_yields --release`

#![cfg(feature = "gpu")]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, Surface, SurfaceKind};
use yamc::geo::{HalfspaceType, Region};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::Model;
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};

fn fe_or_be(element: &str, nuclide: &str) -> Model {
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
    for (element, nuclide) in &[("Be", "Be9"), ("Fe", "Fe56")] {
        let mut model = fe_or_be(element, nuclide);
        model.ensure_photon_data_for_gpu().unwrap();

        let mat = &model.geometry.materials()[0];
        let ttb = mat.ttb.as_ref().expect("ttb populated");
        let e_grid = yamc_element::photon::ttb_e_grid();

        println!("\n=== {} ===", element);
        println!("n_e = {}", e_grid.len());
        // Sample a few energies and print yield
        let probes = [10e3_f64, 50e3, 100e3, 300e3, 500e3, 800e3];
        println!("{:>12}  {:>12}  {:>10}", "ln(E)", "E (eV)", "yield");
        for &e in &probes {
            let log_e = e.ln();
            // Linear search for bracket
            let j = e_grid
                .iter()
                .rposition(|&x| x <= log_e)
                .unwrap_or(0)
                .min(e_grid.len() - 2);
            let e_l = e_grid[j];
            let e_r = e_grid[j + 1];
            let f = (log_e - e_l) / (e_r - e_l);
            let y_l = ttb.electron.yield_[j];
            let y_r = ttb.electron.yield_[j + 1];
            let y = (y_l + (y_r - y_l) * f).exp();
            println!("{:>12.4}  {:>12.4e}  {:>10.4}", log_e, e, y);
        }
    }
}
