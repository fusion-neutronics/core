//! End-to-end test for the GPU bremsstrahlung (TTB) data path.
//!
//! Slice 1 of TTB-on-GPU is data plumbing only: when
//! `Model::ensure_photon_data_for_gpu` runs on a model with
//! `electron_treatment = Ttb` and `transport_secondary_photons = true`,
//! every material should end up with a populated `material.ttb`
//! whose tables match the CPU's `init_bremsstrahlung` output.
//!
//! This test stands up an Fe broomstick (same shape as the verification
//! notebook), runs the GPU photon prep, asserts the extracted
//! `GpuBremsstrahlung` pack has `has_data[m] = 1` for the iron
//! material, and that the per-material PDF/CDF/yield arrays match the
//! values stored on `material.ttb` element-wise. Validates the flat-
//! buffer layout and the dispatch wiring.
//!
//! No GPU kernel involved at this slice; the test passes on hosts
//! without a Vulkan adapter (we don't call `run_on_gpu`).

#![cfg(feature = "gpu")]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, Surface, SurfaceKind};
use yamc::geo::{HalfspaceType, Region};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::gpu::translate_photon::translate_photon_for_gpu;
use yamc::model::Model;
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};

fn fe_broomstick(ttb_on: bool) -> Model {
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
        HashMap::from([("Fe56".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.874),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let mut nm = HashMap::new();
    nm.insert("Fe56".to_string(), "tests/Fe56.arrow".to_string());
    let mut photon_paths: HashMap<String, String> = HashMap::new();
    photon_paths.insert("Fe".to_string(), "tests/Fe.arrow".to_string());
    material
        .read_nuclear_data(&nm, Some(&photon_paths))
        .unwrap();
    let cell = Cell::new(Some(1), region, Some("fe".into()), Some(0));
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
    model.electron_treatment = if ttb_on {
        yamc::model::ElectronTreatment::Ttb
    } else {
        yamc::model::ElectronTreatment::Local
    };
    model
}

#[test]
fn gpu_prep_populates_material_ttb_when_enabled() {
    let mut model = fe_broomstick(true);
    model.ensure_photon_data_for_gpu().unwrap();

    // CPU's prep populates `material.ttb`; the GPU prep should now do
    // the same (and so the post-condition is identical).
    let mat = &model.geometry.materials()[0];
    assert!(
        mat.ttb.is_some(),
        "material.ttb should be Some after ensure_photon_data_for_gpu with ttb=true"
    );
    let ttb = mat.ttb.as_ref().unwrap();
    let n_e = ttb.electron.yield_.len();
    assert!(n_e > 1, "yield grid must have > 1 point, got {}", n_e);
    assert_eq!(ttb.positron.yield_.len(), n_e);
    assert_eq!(ttb.electron.pdf.len(), n_e);
    assert_eq!(ttb.electron.cdf.len(), n_e);
    assert_eq!(ttb.positron.pdf.len(), n_e);
    assert_eq!(ttb.positron.cdf.len(), n_e);
}

#[test]
fn gpu_prep_skips_ttb_when_disabled() {
    let mut model = fe_broomstick(false);
    model.ensure_photon_data_for_gpu().unwrap();

    // With `electron_treatment = Local`, the prep should still
    // populate `cached_elements` (photon transport needs that) but
    // leave `material.ttb` alone.
    let mat = &model.geometry.materials()[0];
    assert!(mat.ttb.is_none(), "ttb should be None when disabled");
    assert!(
        !mat.cached_elements.is_empty(),
        "photon data should still load"
    );
}

#[test]
fn translate_packs_ttb_into_inputs_when_enabled() {
    let mut model = fe_broomstick(true);
    model.ensure_photon_data_for_gpu().unwrap();

    let inputs = translate_photon_for_gpu(&model, 100, 7).expect("translate ok");
    let mat = &model.geometry.materials()[0];
    let cpu_ttb = mat.ttb.as_ref().unwrap();
    let n_e = cpu_ttb.electron.yield_.len();

    // GpuBremsstrahlung's per-material flag should mark Fe as ready.
    assert_eq!(
        inputs.ttb.has_data,
        vec![1u32],
        "Fe TTB should be populated"
    );
    assert_eq!(inputs.ttb.n_e as usize, n_e);
    assert_eq!(inputs.ttb.e_grid_log.len(), n_e);

    // Per-material yield arrays compared element-wise.
    for j in 0..n_e {
        assert_eq!(
            inputs.ttb.electron_yield[j], cpu_ttb.electron.yield_[j],
            "electron_yield[{}]",
            j
        );
        assert_eq!(
            inputs.ttb.positron_yield[j], cpu_ttb.positron.yield_[j],
            "positron_yield[{}]",
            j
        );
    }

    // Spot-check one PDF row (j = n_e/2) -- the row is lower-triangular,
    // so columns past j should be zero (matches CPU storage).
    let j = n_e / 2;
    let row_off = j * n_e;
    for i in 0..n_e {
        assert_eq!(
            inputs.ttb.electron_pdf[row_off + i],
            cpu_ttb.electron.pdf[j][i],
            "electron_pdf[{}][{}]",
            j,
            i
        );
        assert_eq!(
            inputs.ttb.electron_cdf[row_off + i],
            cpu_ttb.electron.cdf[j][i],
            "electron_cdf[{}][{}]",
            j,
            i
        );
    }
}

#[test]
fn translate_returns_empty_pack_when_ttb_disabled() {
    let mut model = fe_broomstick(false);
    model.ensure_photon_data_for_gpu().unwrap();

    let inputs = translate_photon_for_gpu(&model, 100, 7).expect("translate ok");
    // Degenerate single-grid-point pack with has_data = 0.
    assert_eq!(inputs.ttb.has_data, vec![0u32]);
    assert_eq!(inputs.ttb.n_e, 1);
}
