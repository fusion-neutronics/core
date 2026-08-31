//! End-to-end check of the wasm-bindgen geometry-plot API
//! (`sampleSlice`, `boundingBox`, `geometryJson`, `sampleSourcePoints`)
//! on a native target. These methods drive the in-browser interactive
//! viewer in commit C5; this test locks down the wire format so changes
//! that quietly break it get caught before the JS host does.

#![cfg(feature = "wasm")]

use std::sync::Arc;

use yamc::geo::{BoundaryType, Surface, SurfaceKind};
use yamc::geo::{HalfspaceType, Region};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::Model;
use yamc::wasm::simulation_wasm::WasmSimulation;
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};

fn two_sphere_model() -> Model {
    // Inner sphere (r=3) + outer sphere (r=10), Li-6 in both. Two cells
    // so sample_slice has interesting cell-id structure across the
    // boundary.
    let inner = Surface {
        surface_id: None,
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 3.0,
        },
        boundary: BoundaryType::Transmission,
        name: None,
    };
    let outer = Surface {
        surface_id: Some(2),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 10.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let inner_arc = Arc::new(inner);
    let outer_arc = Arc::new(outer);

    let core = Cell::new(
        Some(1),
        Region::new_from_halfspace(HalfspaceType::Below(inner_arc.clone())),
        Some("core".into()),
        Some(0),
    );
    let shell_region = Region::new_from_halfspace(HalfspaceType::Below(outer_arc.clone()))
        .intersection(&Region::new_from_halfspace(HalfspaceType::Above(
            inner_arc.clone(),
        )));
    let shell = Cell::new(Some(2), shell_region, Some("shell".into()), Some(0));

    let material = Material::new(
        std::collections::HashMap::from([("Li6".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(0.46),
    )
    .unwrap();

    let geometry = Geometry::new(vec![core, shell], vec![Arc::new(material)]).unwrap();

    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([1.5, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });
    Model::new(geometry, vec![source], vec![])
}

#[test]
fn tally_plot_html_works_on_a_freshly_simulated_mesh_tally() {
    // Build a model with a mesh tally, run a tiny in-wasm simulation,
    // then ask wasm to plot the tally -- same path the editor takes on
    // every Simulate click. The output should be a self-contained HTML
    // doc carrying the engineer's mesh + the per-bin data sourced from
    // the in-memory tally state (not a pre-baked snapshot).
    use std::collections::HashMap;
    use yamc::geometry::backend::GeometryKind;
    use yamc_materials::Material;
    use yamc_tallies::filter::mesh::MeshFilter;
    use yamc_tallies::filter::Filter;
    use yamc_tallies::mesh::RegularRectangularMesh;
    use yamc_tallies::mt::Mt;
    use yamc_tallies::score::{ReactionRateScore, Score};
    use yamc_tallies::tally::Tally;

    let mut model = two_sphere_model();
    // Add a mesh tally for tritium production over a small mesh.
    let mesh = RegularRectangularMesh::new([-10.0, -10.0, -10.0], [10.0, 10.0, 10.0], [4, 4, 4]);
    let mut tally = Tally::new();
    tally.filters.push(Filter::Mesh(MeshFilter::new(mesh)));
    tally.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(
        Mt::try_from(105).unwrap(),
    ))];
    tally.name = Some("tritium_mesh".into());
    // Replace the geometry's material so simulate_transport doesn't try
    // to load nuclear data we don't have on disk.
    let mat = Material::new(
        HashMap::from([("Li6".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(0.46),
    )
    .unwrap();
    if let GeometryKind::Csg(ref mut g) = model.geometry {
        g.materials = vec![std::sync::Arc::new(mat)];
    }
    model.tallies = vec![std::sync::Arc::new(tally)];

    let model_json = serde_json::to_string(&model).unwrap();
    let sim = WasmSimulation::new();
    sim.load_model_json(model_json).unwrap();

    // tallyPlotHtml works even WITHOUT first calling simulate_transport
    // (the tally has zero accumulator state, plot just shows zeros).
    let html = sim
        .tally_plot_html("{}".to_string())
        .expect("tallyPlotHtml");
    assert!(html.starts_with("<!DOCTYPE html>"));
    assert!(html.contains("MESH_LL"));
    assert!(html.contains("MESH_DIM"));
    assert!(html.contains("\"name\":\"core\"") || html.contains("\"name\":\"shell\""));
}

#[test]
fn tally_plot_html_errors_on_non_mesh_tally() {
    // Scalar (cell-filtered) tally -- should error with a useful message
    // rather than silently produce a bad plot.
    use yamc_tallies::filter::cell::CellFilter;
    use yamc_tallies::filter::Filter;
    use yamc_tallies::mt::Mt;
    use yamc_tallies::score::{ReactionRateScore, Score};
    use yamc_tallies::tally::Tally;

    let mut model = two_sphere_model();
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(1)));
    t.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(
        Mt::try_from(105).unwrap(),
    ))];
    model.tallies = vec![std::sync::Arc::new(t)];

    let sim = WasmSimulation::new();
    sim.load_model_json(serde_json::to_string(&model).unwrap())
        .unwrap();
    let err = sim.tally_plot_html("{}".to_string()).unwrap_err();
    assert!(err.contains("MeshFilter"), "got: {err}");
}

#[test]
fn sample_slice_returns_interleaved_grid_top_to_bottom() {
    let model_json = serde_json::to_string(&two_sphere_model()).unwrap();
    let sim = WasmSimulation::new();
    sim.load_model_json(model_json).unwrap();

    let params = r#"{"origin":[0.0,0.0,0.0],"width":[20.0,20.0],"pixels":[8,8],"basis":"xy"}"#;
    let flat = sim.sample_slice(params.to_string()).expect("sample_slice");

    assert_eq!(flat.len(), 8 * 8 * 2, "8x8 grid × 2 ints per pixel");

    // Centre pixel is inside the inner sphere → core (cell_id=1).
    // Pixel (3,4) sits closest to (0,0) for an 8x8 grid centered at origin.
    let centre_idx = (4 * 8 + 3) * 2;
    let cell_at_centre = flat[centre_idx];
    assert!(
        cell_at_centre == 1,
        "centre of xy slice should be in core (cell_id=1), got {cell_at_centre}"
    );

    // A corner pixel is outside the outer sphere → -1.
    let corner = flat[0];
    assert_eq!(
        corner, -1,
        "corner outside outer sphere should be void (-1)"
    );
}

#[test]
fn bounding_box_matches_geometry() {
    let model_json = serde_json::to_string(&two_sphere_model()).unwrap();
    let sim = WasmSimulation::new();
    sim.load_model_json(model_json).unwrap();
    let bb = sim.bounding_box().expect("bounding_box");
    assert_eq!(bb.len(), 6);
    // [cx, cy, cz, wx, wy, wz] -- concentric spheres at origin, outer r=10.
    assert!(bb[0].abs() < 1e-9, "cx: {}", bb[0]);
    assert!(bb[1].abs() < 1e-9, "cy: {}", bb[1]);
    assert!(bb[2].abs() < 1e-9, "cz: {}", bb[2]);
    // Widths ≈ 20 (diameter of outer sphere).
    for w in &bb[3..6] {
        assert!((*w - 20.0).abs() < 1e-6, "width {w} ≈ 20");
    }
}

#[test]
fn geometry_json_carries_cell_metadata() {
    let model_json = serde_json::to_string(&two_sphere_model()).unwrap();
    let sim = WasmSimulation::new();
    sim.load_model_json(model_json).unwrap();
    let json = sim.geometry_json().expect("geometry_json");
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    let cells = parsed["cells"].as_array().expect("cells array");
    assert_eq!(cells.len(), 2);
    let names: Vec<_> = cells
        .iter()
        .filter_map(|c| c["name"].as_str().map(str::to_string))
        .collect();
    assert!(names.contains(&"core".to_string()));
    assert!(names.contains(&"shell".to_string()));
}

#[test]
fn plot_html_returns_self_contained_viewer() {
    let model_json = serde_json::to_string(&two_sphere_model()).unwrap();
    let sim = WasmSimulation::new();
    sim.load_model_json(model_json).unwrap();

    // Empty params → auto-fit to bbox + default basis.
    let html = sim.plot_html("{}".to_string()).expect("plotHtml");
    assert!(html.starts_with("<!DOCTYPE html>"));
    assert!(html.contains("<script>"), "viewer should ship a script tag");
    assert!(html.contains("GEOMETRY_JSON"), "model data embedded");
    // CsgGeometry cell names from the model carry through.
    assert!(html.contains("\"core\""));
    assert!(html.contains("\"shell\""));

    // Explicit params override the auto-fit values.
    let params = r#"{"origin":[1.0,2.0,3.0],"width":[5.0,7.0],"pixels":[200,300],"basis":"xz"}"#;
    let html2 = sim
        .plot_html(params.to_string())
        .expect("plotHtml with params");
    // Verify the override took effect by checking the input control values.
    assert!(
        html2.contains("id=\"origin-x\" step=\"any\" value=\"1\"")
            || html2.contains("id=\"origin-x\" step=\"any\" value=\"1.0\"")
    );
    assert!(html2.contains("id=\"width-h\" step=\"any\" value=\"5\""));
}

#[test]
fn plot_html_defaults_match_model_plot_python_path() {
    // Regressions to lock down the V1 defaults so future drift gets caught.
    // model.plot() in Python ships color_by="cell", outline="cell",
    // total_pixels=40000, units="cm", uses bbox.center for origin and
    // bbox.width for size. The editor must mirror that or the recipient's
    // legend / colouring won't match the engineer's expectation.
    let model_json = serde_json::to_string(&two_sphere_model()).unwrap();
    let sim = WasmSimulation::new();
    sim.load_model_json(model_json).unwrap();
    let html = sim.plot_html("{}".to_string()).unwrap();

    // origin = bbox.center = (0, 0, 0) for our two concentric spheres.
    assert!(
        html.contains("id=\"origin-x\" step=\"any\" value=\"0\""),
        "origin-x default should be bbox center (0)"
    );
    // width-h = bbox.width[0] = 20 (outer sphere diameter).
    assert!(
        html.contains("id=\"width-h\" step=\"any\" value=\"20\""),
        "width-h default should be bbox width (20)"
    );
    // total-pixels = 40000 (matches Python).
    assert!(
        html.contains("id=\"total-pixels\" value=\"40000\""),
        "total-pixels default should be 40000"
    );
    // color_by=cell must be the radio that's checked.
    let color_by_cell_checked = regex_checked(&html, "color-by", "cell");
    assert!(
        color_by_cell_checked,
        "color-by=cell radio should be checked by default"
    );
    let color_by_mat_checked = regex_checked(&html, "color-by", "material");
    assert!(
        !color_by_mat_checked,
        "color-by=material radio should NOT be checked"
    );
    // outline=cell checked.
    assert!(regex_checked(&html, "outline", "cell"));
    // units=cm checked.
    assert!(regex_checked(&html, "units", "cm"));

    // Material legend pulls from cell.material_name (Python path), not
    // geom.materials -- so unreferenced materials never appear.
    assert!(html.contains("\"core\""), "cell name 'core' in legend");
    assert!(html.contains("\"shell\""), "cell name 'shell' in legend");
}

/// Find the radio input with `name=...` and `value=...` and check if the
/// surrounding tag has `checked` attribute. The HTML has the pattern:
///   <input type="radio" name="color-by" value="cell" checked> Cell
fn regex_checked(html: &str, name: &str, value: &str) -> bool {
    let needle = format!("name=\"{name}\" value=\"{value}\"");
    let start = match html.find(&needle) {
        Some(s) => s,
        None => return false,
    };
    // Look for `checked` (with or without a leading space) before the
    // closing `>` that ends this <input> tag.
    let after = &html[start..];
    let end = match after.find('>') {
        Some(e) => e,
        None => return false,
    };
    after[..end].contains("checked")
}

#[test]
fn plot_html_embeds_source_distribution_so_viewer_draws_points() {
    // Regression: previously plotHtml serialised `model.sources` directly
    // (an array tagged with Neutron/Photon variants), which the viewer's
    // inline sampler couldn't read -- the source dot never appeared. The
    // wire format must be {"sources": [{"strength","spatial","energy"}, ...]}.
    let model_json = serde_json::to_string(&two_sphere_model()).unwrap();
    let sim = WasmSimulation::new();
    sim.load_model_json(model_json).unwrap();
    let html = sim.plot_html("{}".to_string()).unwrap();

    // Find the `const SOURCE_DIST = ...;` line and parse the JSON literal.
    let needle = "const SOURCE_DIST = ";
    let start = html.find(needle).expect("SOURCE_DIST line present");
    let after = &html[start + needle.len()..];
    let end = after.find(";\n").expect("SOURCE_DIST line terminates");
    let dist_json = &after[..end];
    assert_ne!(
        dist_json.trim(),
        "null",
        "source distribution must not be null"
    );

    let v: serde_json::Value = serde_json::from_str(dist_json).expect("SOURCE_DIST is valid JSON");
    let sources = v["sources"].as_array().expect("sources is an array");
    assert_eq!(sources.len(), 1);
    let s = &sources[0];
    assert!(s["strength"].as_f64().is_some(), "strength field");
    assert!(s["spatial"].is_object(), "spatial field");
    assert!(s["energy"].is_object(), "energy field");
    // Our two_sphere_model puts a point source at (1.5, 0, 0). The
    // spatial entry should be a Point variant carrying that position.
    let xyz = &s["spatial"]["Point"]["xyz"];
    assert!(xyz.is_array(), "spatial.Point.xyz is [x,y,z]");
    assert_eq!(xyz[0].as_f64().unwrap(), 1.5);
}

#[test]
fn plot_html_hides_source_section_when_model_has_no_sources() {
    // Build a sources-less model: same geometry, empty sources vec.
    let mut m = two_sphere_model();
    m.sources.clear();
    let model_json = serde_json::to_string(&m).unwrap();
    let sim = WasmSimulation::new();
    sim.load_model_json(model_json).unwrap();
    let html = sim.plot_html("{}".to_string()).unwrap();

    // SOURCE_DIST must be `null` so the viewer's sampler short-circuits.
    let needle = "const SOURCE_DIST = ";
    let start = html.find(needle).unwrap();
    let after = &html[start + needle.len()..];
    let semi = after.find(';').unwrap();
    assert_eq!(
        after[..semi].trim(),
        "null",
        "no sources → SOURCE_DIST should be null, got {:?}",
        &after[..semi]
    );

    // The Source section in the controls panel hides itself when
    // n_samples is None (see build_interactive_html's
    // source_section_display = "none" path).
    assert!(
        html.contains("id=\"source-section\" style=\"display:none;\"")
            || html.contains("id=\"source-section\" style=\"display:none\""),
        "source-section should be hidden"
    );
}

#[test]
fn plot_html_errors_clearly_when_no_model_loaded() {
    let sim = WasmSimulation::new();
    let err = sim.plot_html("{}".to_string()).unwrap_err();
    assert!(err.contains("no model loaded"), "got: {err}");
}

#[test]
fn plot_html_rejects_bad_params_json() {
    let model_json = serde_json::to_string(&two_sphere_model()).unwrap();
    let sim = WasmSimulation::new();
    sim.load_model_json(model_json).unwrap();
    // Missing closing brace.
    let err = sim.plot_html("{\"basis\":\"xy\"".to_string()).unwrap_err();
    assert!(err.contains("plotHtml params"), "got: {err}");
}

#[test]
fn sample_source_points_returns_3n_floats_at_the_source_position() {
    let model_json = serde_json::to_string(&two_sphere_model()).unwrap();
    let sim = WasmSimulation::new();
    sim.load_model_json(model_json).unwrap();
    let pts = sim.sample_source_points(20, 42).expect("source points");
    assert_eq!(pts.len(), 20 * 3, "n=20 → 60 floats");
    // Point source at (1.5, 0, 0); every sample sits there.
    for i in 0..20 {
        let off = i * 3;
        assert!((pts[off] - 1.5).abs() < 1e-9, "x[{i}]={}", pts[off]);
        assert!(pts[off + 1].abs() < 1e-9, "y[{i}]={}", pts[off + 1]);
        assert!(pts[off + 2].abs() < 1e-9, "z[{i}]={}", pts[off + 2]);
    }
}
