//! Unit tests for `yamc_plot::build_surface_table` -- the walker that
//! drives surface-hover tooltips in the interactive viewer.

use std::sync::Arc;

use yamc_geo::cell::GeoCell;
use yamc_geo::geometry::CsgGeometry;
use yamc_geo::region::{HalfspaceType, Region, RegionExpr};
use yamc_geo::surface::{BoundaryType, Surface, SurfaceKind};
use yamc_plot::build_surface_table;

fn sphere(name: Option<&str>, surface_id: Option<usize>, radius: f64) -> Arc<Surface> {
    Arc::new(Surface {
        surface_id,
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius,
        },
        boundary: BoundaryType::Transmission,
        name: name.map(str::to_string),
    })
}

fn cell_below(id: i32, surf: Arc<Surface>, name: &str) -> GeoCell {
    GeoCell {
        cell_id: id,
        material_id: -1,
        name: Some(name.to_string()),
        material_name: None,
        region: Region {
            expr: RegionExpr::Halfspace(HalfspaceType::Below(surf)),
        },
    }
}

fn cell_above(id: i32, surf: Arc<Surface>, name: &str) -> GeoCell {
    GeoCell {
        cell_id: id,
        material_id: -1,
        name: Some(name.to_string()),
        material_name: None,
        region: Region {
            expr: RegionExpr::Halfspace(HalfspaceType::Above(surf)),
        },
    }
}

#[test]
fn named_surfaces_carry_their_name_through() {
    let inner = sphere(Some("inner"), None, 5.0);
    let outer = sphere(Some("outer"), None, 10.0);
    let csg = CsgGeometry {
        cells: vec![
            cell_below(1, inner.clone(), "core"),
            cell_below(2, outer.clone(), "world"),
        ],
    };
    let table = build_surface_table(&csg);
    assert_eq!(table.len(), 2);
    assert_eq!(table[0].display_name, "inner");
    assert_eq!(table[1].display_name, "outer");
}

#[test]
fn surfaces_shared_between_cells_dedupe_by_arc_identity() {
    // Same Arc referenced by two cells (the usual case for an outer
    // boundary shared between core + shell) must appear only once.
    let inner = sphere(Some("inner"), None, 3.0);
    let outer = sphere(Some("outer"), None, 10.0);
    let csg = CsgGeometry {
        cells: vec![
            cell_below(1, inner.clone(), "core"),
            GeoCell {
                cell_id: 2,
                material_id: -1,
                name: Some("shell".into()),
                material_name: None,
                region: Region {
                    expr: RegionExpr::Intersection(
                        Box::new(RegionExpr::Halfspace(HalfspaceType::Below(outer.clone()))),
                        Box::new(RegionExpr::Halfspace(HalfspaceType::Above(inner.clone()))),
                    ),
                },
            },
        ],
    };
    let table = build_surface_table(&csg);
    assert_eq!(table.len(), 2, "inner sphere shared, must dedupe");
    let names: Vec<_> = table.iter().map(|e| e.display_name.as_str()).collect();
    assert!(names.contains(&"inner"));
    assert!(names.contains(&"outer"));
}

#[test]
fn unnamed_surface_with_id_falls_back_to_surface_id() {
    let surf = sphere(None, Some(7), 5.0);
    let csg = CsgGeometry {
        cells: vec![cell_below(1, surf, "c")],
    };
    let table = build_surface_table(&csg);
    assert_eq!(table[0].display_name, "surface 7");
}

#[test]
fn completely_anonymous_surfaces_get_kind_indexed_labels() {
    // Two distinct unnamed Spheres become "Sphere #1" + "Sphere #2".
    // Walker should keep indexing per kind, not globally.
    let a = sphere(None, None, 3.0);
    let b = sphere(None, None, 5.0);
    let plane = Arc::new(Surface {
        surface_id: None,
        kind: SurfaceKind::Plane {
            a: 0.0,
            b: 0.0,
            c: 1.0,
            d: 0.0,
        },
        boundary: BoundaryType::Transmission,
        name: None,
    });
    let csg = CsgGeometry {
        cells: vec![
            cell_below(1, a, "a"),
            cell_above(2, b, "b"),
            cell_below(3, plane, "c"),
        ],
    };
    let table = build_surface_table(&csg);
    let names: Vec<_> = table.iter().map(|e| e.display_name.as_str()).collect();
    assert_eq!(names, vec!["Sphere #1", "Sphere #2", "Plane #1"]);
}

#[test]
fn surfaces_dedupe_even_when_arcs_dont_share_identity() {
    // After a JSON round-trip, each cell holds its own freshly-allocated
    // Arc<Surface>, so Arc::ptr_eq returns false even for what was
    // logically the same surface. The walker has to fall back to
    // content equality or the table contains duplicates.
    //
    // Regression for the user-reported "surface name shows multiple times
    // in the hover tooltip" -- caused by the wasm plot_html path running
    // build_surface_table on a CsgGeometry rebuilt from JSON.
    let mk_inner = || sphere(Some("inner"), None, 3.0);
    let mk_outer = || sphere(Some("outer"), None, 10.0);
    // Different Arcs that wrap surfaces with identical content.
    let inner_a = mk_inner();
    let inner_b = mk_inner();
    let outer_a = mk_outer();
    let outer_b = mk_outer();
    assert!(
        !Arc::ptr_eq(&inner_a, &inner_b),
        "test setup: two Arcs with same content should NOT be ptr_eq"
    );
    let csg = CsgGeometry {
        cells: vec![
            cell_below(1, inner_a, "core"),
            GeoCell {
                cell_id: 2,
                material_id: -1,
                name: Some("shell".into()),
                material_name: None,
                region: Region {
                    expr: RegionExpr::Intersection(
                        Box::new(RegionExpr::Halfspace(HalfspaceType::Below(outer_a))),
                        Box::new(RegionExpr::Halfspace(HalfspaceType::Above(inner_b))),
                    ),
                },
            },
            cell_above(3, outer_b, "world"),
        ],
    };
    let table = build_surface_table(&csg);
    assert_eq!(table.len(), 2, "content-equal surfaces must dedupe");
    let names: Vec<_> = table.iter().map(|e| e.display_name.as_str()).collect();
    assert!(names.contains(&"inner"));
    assert!(names.contains(&"outer"));
}

#[test]
fn empty_geometry_yields_empty_table() {
    let csg = CsgGeometry { cells: vec![] };
    let table = build_surface_table(&csg);
    assert!(table.is_empty());
}
