//! Surface kinds the browser samplers cannot evaluate (e.g. a flux or an
//! implicit surface).
//!
//! `browser_can_sample` tells the plot paths to embed yamc's raster for such
//! a geometry, and every JS `surfaceEval` copy must answer NaN (unknown)
//! rather than 0 for them: 0 put a cell on both sides of the surface, so it
//! was drawn with the wrong fill and hovered as "on the surface" everywhere.

use std::collections::HashMap;
use std::sync::Arc;

use yamc_geo::cell::GeoCell;
use yamc_geo::geometry::CsgGeometry;
use yamc_geo::region::{HalfspaceType, Region, RegionExpr};
use yamc_geo::surface::{BoundaryType, FluxBoundary, ImplicitFormula, Surface, SurfaceKind};
use yamc_plot::{
    browser_can_sample, build_interactive_html, build_interactive_tally_html, EmbeddedSlice,
    InteractiveTallyParams, InteractiveViewParams, MeshMeta,
};

fn surface(kind: SurfaceKind) -> Arc<Surface> {
    Arc::new(Surface {
        surface_id: None,
        kind,
        boundary: BoundaryType::Transmission,
        name: None,
    })
}

fn below(s: Arc<Surface>) -> RegionExpr {
    RegionExpr::Halfspace(HalfspaceType::Below(s))
}

fn and(a: RegionExpr, b: RegionExpr) -> RegionExpr {
    RegionExpr::Intersection(Box::new(a), Box::new(b))
}

fn geometry(exprs: Vec<RegionExpr>) -> CsgGeometry {
    CsgGeometry {
        cells: exprs
            .into_iter()
            .enumerate()
            .map(|(i, expr)| GeoCell {
                cell_id: i as i32 + 1,
                material_id: -1,
                name: None,
                material_name: None,
                region: Region { expr },
            })
            .collect(),
    }
}

fn sphere() -> Arc<Surface> {
    surface(SurfaceKind::Sphere {
        x0: 0.0,
        y0: 0.0,
        z0: 0.0,
        radius: 900.0,
    })
}

/// A circular torus (R0 = 600, a = 150) given as a flux surface.
fn flux() -> Arc<Surface> {
    let boundary = FluxBoundary::new(
        1,
        vec![[0, 0], [1, 0]],
        vec![600.0, 150.0],
        vec![[-1, 0]],
        vec![150.0],
    )
    .unwrap();
    Arc::new(Surface::new_flux(Arc::new(boundary), None, None))
}

#[test]
fn browser_can_sample_only_the_kinds_surface_eval_knows() {
    let known = geometry(vec![
        and(
            below(sphere()),
            below(surface(SurfaceKind::Plane {
                a: 0.0,
                b: 0.0,
                c: 1.0,
                d: 0.0,
            })),
        ),
        below(surface(SurfaceKind::Cylinder {
            axis: [0.0, 0.0, 1.0],
            origin: [0.0, 0.0, 0.0],
            radius: 1.0,
        })),
        below(surface(SurfaceKind::ZTorus {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            a: 600.0,
            b: 150.0,
            c: 150.0,
        })),
    ]);
    assert!(browser_can_sample(&known));

    // One unknown leaf anywhere in any cell is enough, even under a complement.
    let blanket = and(
        below(sphere()),
        RegionExpr::Complement(Box::new(below(flux()))),
    );
    assert!(!browser_can_sample(&geometry(vec![below(flux()), blanket])));
    let x_torus = surface(SurfaceKind::XTorus {
        x0: 0.0,
        y0: 0.0,
        z0: 0.0,
        a: 600.0,
        b: 150.0,
        c: 150.0,
    });
    assert!(!browser_can_sample(&geometry(vec![
        below(sphere()),
        below(x_torus)
    ])));
    let formula = ImplicitFormula::new("sqrt(x^2 + y^2 + z^2) - 900").unwrap();
    let implicit = surface(SurfaceKind::Implicit {
        formula: Arc::new(formula),
    });
    assert!(!browser_can_sample(&geometry(vec![
        below(sphere()),
        RegionExpr::Complement(Box::new(below(implicit)))
    ])));
}

/// The text of the first `function surfaceEval` in `js`.
fn surface_eval(js: &str) -> &str {
    let start = js.find("function surfaceEval(").expect("no surfaceEval");
    let len = js[start..].find("\n}").expect("unterminated surfaceEval");
    &js[start..start + len]
}

fn assert_unknown_kind_is_nan(eval: &str) {
    assert!(
        eval.contains("return NaN;") && !eval.contains("return 0;"),
        "surfaceEval must return NaN for a kind it cannot evaluate:\n{eval}"
    );
}

#[test]
fn geometry_viewer_answers_nan_and_keeps_the_presampled_image() {
    let params = InteractiveViewParams {
        origin: (0.0, 0.0, 0.0),
        width: (10.0, 10.0),
        total_pixels: 100,
        basis: "xy".into(),
        color_by: "cell".into(),
        outline: None,
        axis_units: "cm".into(),
        bbox_widths: [10.0, 10.0, 10.0],
        outline_color: "#000000".into(),
        outline_thickness: 1,
        n_samples: None,
        plane_tolerance: 1.0,
        source_color: "#ff0000".into(),
        source_size: 3,
    };
    let html = build_interactive_html(
        "{\"cells\":[]}",
        "csg",
        "",
        &params,
        &HashMap::new(),
        &HashMap::new(),
        None,
        None,
        None,
        false,
        None,
    );
    // The worker pool's copy, then the page's own (hover + fallback sampler).
    let (worker, page) = html
        .split_once("const CSG_WORKER_SOURCE = `")
        .and_then(|(_, rest)| rest.split_once("`;"))
        .expect("no CSG_WORKER_SOURCE");
    assert_unknown_kind_is_nan(surface_eval(worker));
    assert_unknown_kind_is_nan(surface_eval(page));
    // Such a geometry is not re-sampled: the view stays on yamc's image.
    assert!(page.contains("const RESAMPLE_NOTICE"));
    assert!(page.contains("restorePresampledView();"));
}

#[test]
fn tally_viewer_answers_nan_and_does_not_resample_outlines() {
    let params = InteractiveTallyParams {
        initial_basis: "xy".into(),
        initial_slice_index: 0,
        colorscale: "Viridis".into(),
        log_scale: true,
        outline: Some("cell".into()),
        outline_color: "#000000".into(),
        outline_thickness: 1,
        axis_units: "cm".into(),
        outline_pixels: 40_000,
        title: "test".into(),
        colorbar_title: "flux".into(),
        scaling_factor: 1.0,
        font_size: 18,
        show_colorbar: true,
        display_value: "mean".into(),
    };
    let mesh = MeshMeta {
        lower_left: [-10.0, -10.0, -10.0],
        upper_right: [10.0, 10.0, 10.0],
        shape: [2, 2, 2],
        width: [10.0, 10.0, 10.0],
    };
    let slices = vec![EmbeddedSlice {
        basis: "xy".into(),
        bin_index: 0,
        data: vec![0.0; 4],
    }];
    let html = build_interactive_tally_html(
        Some("{\"cells\":[]}"),
        Some("csg"),
        &mesh,
        &slices,
        &[],
        &params,
        &HashMap::new(),
        &HashMap::new(),
        None,
        None,
    );
    assert_unknown_kind_is_nan(surface_eval(&html));
    // No outline plotter for such a geometry: only the pre-rendered outline.
    assert!(html.contains("const RESAMPLE_NOTICE"));
    assert!(html.contains("if (!RESAMPLE_NOTICE) {"));
}
