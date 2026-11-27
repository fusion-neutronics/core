//! Tests for `yamc_plot::build_interactive_tally_html` -- focused on the
//! surface table plumbing, plus a couple of structural sanity checks so no
//! field is silently dropped.

use std::collections::HashMap;
use std::sync::Arc;

use yamc_geo::cell::GeoCell;
use yamc_geo::geometry::CsgGeometry;
use yamc_geo::region::{HalfspaceType, Region, RegionExpr};
use yamc_geo::surface::{BoundaryType, Surface, SurfaceKind};
use yamc_plot::{
    build_interactive_tally_html, build_surface_table, EmbeddedSlice, InteractiveTallyParams,
    MeshMeta,
};

fn sphere(name: &str, radius: f64) -> Arc<Surface> {
    Arc::new(Surface {
        surface_id: None,
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius,
        },
        boundary: BoundaryType::Transmission,
        name: Some(name.to_string()),
    })
}

fn make_csg() -> CsgGeometry {
    let inner = sphere("inner", 3.0);
    let outer = sphere("outer", 10.0);
    CsgGeometry {
        cells: vec![
            GeoCell {
                cell_id: 1,
                material_id: -1,
                name: Some("core".into()),
                material_name: None,
                region: Region {
                    expr: RegionExpr::Halfspace(HalfspaceType::Below(inner.clone())),
                },
            },
            GeoCell {
                cell_id: 2,
                material_id: -1,
                name: Some("shell".into()),
                material_name: None,
                region: Region {
                    expr: RegionExpr::Intersection(
                        Box::new(RegionExpr::Halfspace(HalfspaceType::Below(outer))),
                        Box::new(RegionExpr::Halfspace(HalfspaceType::Above(inner))),
                    ),
                },
            },
        ],
    }
}

fn make_params() -> InteractiveTallyParams {
    InteractiveTallyParams {
        initial_basis: "xy".into(),
        initial_slice_index: 0,
        colorscale: "Viridis".into(),
        log_scale: true,
        outline: Some("material".into()),
        outline_color: "#000000".into(),
        outline_thickness: 1,
        axis_units: "cm".into(),
        outline_pixels: 40_000,
        title: "test".into(),
        colorbar_title: "flux".into(),
        colorbar_auto: false,
        scaling_factor: 1.0,
        font_size: 18,
        show_colorbar: true,
        display_value: "mean".into(),
    }
}

fn make_mesh() -> MeshMeta {
    MeshMeta {
        lower_left: [-10.0, -10.0, -10.0],
        upper_right: [10.0, 10.0, 10.0],
        shape: [10, 10, 10],
        width: [20.0, 20.0, 20.0],
    }
}

fn make_slice() -> EmbeddedSlice {
    let data = vec![0.0_f64; 10 * 10];
    EmbeddedSlice {
        basis: "xy".into(),
        bin_index: 5,
        data,
    }
}

#[test]
fn surface_table_arg_emits_the_js_constant() {
    let csg = make_csg();
    let table = build_surface_table(&csg);
    assert_eq!(table.len(), 2);

    let mesh = make_mesh();
    let slices = vec![make_slice()];
    let html = build_interactive_tally_html(
        Some("{\"cells\":[]}"),
        Some("csg"),
        &mesh,
        &slices,
        &[],
        &make_params(),
        &HashMap::new(),
        &HashMap::new(),
        None,
        Some(&table),
    );

    // The JS const must be present and parse as a non-empty array.
    assert!(
        html.contains("const SURFACE_TABLE = ["),
        "SURFACE_TABLE missing"
    );
    // Both surface names round-trip.
    assert!(html.contains("\"name\":\"inner\""));
    assert!(html.contains("\"name\":\"outer\""));
    // The JS-side lookup function must be present.
    assert!(
        html.contains("function surfacesAtPoint"),
        "surfacesAtPoint missing"
    );
    // Tooltip uses position:fixed so the legend / colorbar don't clip it.
    assert!(
        html.contains("position: fixed"),
        "tooltip not position:fixed"
    );
}

#[test]
fn surface_table_none_emits_empty_array() {
    let mesh = make_mesh();
    let slices = vec![make_slice()];
    let html = build_interactive_tally_html(
        None,
        None,
        &mesh,
        &slices,
        &[],
        &make_params(),
        &HashMap::new(),
        &HashMap::new(),
        None,
        None,
    );
    assert!(html.contains("const SURFACE_TABLE = []"));
}

#[test]
fn axis_units_are_checked_against_one_table() {
    use yamc_plot::axis_unit_scale;
    assert_eq!(axis_unit_scale("cm"), Ok(1.0));
    assert_eq!(axis_unit_scale("m"), Ok(0.01));
    assert_eq!(axis_unit_scale("mm"), Ok(10.0));
    assert_eq!(axis_unit_scale("km"), Ok(1e-5));
    let err = axis_unit_scale("in").unwrap_err();
    assert!(err.contains("axis_units must be one of"), "{err}");
}

#[test]
fn an_auto_colorbar_label_follows_the_scaling_factor() {
    let mesh = make_mesh();
    let slices = vec![make_slice()];
    let mut params = make_params();
    params.colorbar_auto = true;
    let html = build_interactive_tally_html(
        None,
        None,
        &mesh,
        &slices,
        &[],
        &params,
        &HashMap::new(),
        &HashMap::new(),
        None,
        None,
    );
    assert!(html.contains("colorbarAuto: true"));
    assert!(html.contains("function colorbarText()"));
}
