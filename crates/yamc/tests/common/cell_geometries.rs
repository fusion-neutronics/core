//! Geometry constructors used by both correctness tests and benchmarks.
//!
//! Each builder produces a `Geometry` with a documented total-cell count
//! and a topology chosen to stress different aspects of cell-finding:
//!
//! * `build_cube_grid(n)` -- N×N×N axis-aligned unit cubes (N³ cells).
//! * `build_concentric_shells(n)` -- N nested spherical shells (N cells).
//! * `build_pin_lattice(n)` -- N×N pins, each a cylinder inside a square
//!   slab in z (2 N² cells, fuel + moderator alternating).

use std::sync::Arc;
use yamc::geo::Surface;
use yamc::geo::{HalfspaceType, Region};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;

// ---------------------------------------------------------------------
// Cube grid
// ---------------------------------------------------------------------

/// Build an N×N×N grid of unit cubes spanning [0, N]³.
///
/// Cell ordering: `index = i + N*(j + N*k)` so cell (i,j,k) sits at
/// `[i, i+1] × [j, j+1] × [k, k+1]`. This means [`expected_cube_grid_index`]
/// can answer "which cell is this point in?" analytically.
pub fn build_cube_grid(n: usize) -> Geometry {
    let x_planes: Vec<Arc<Surface>> = (0..=n)
        .map(|i| Arc::new(Surface::x_plane(i as f64, None, None)))
        .collect();
    let y_planes: Vec<Arc<Surface>> = (0..=n)
        .map(|j| Arc::new(Surface::y_plane(j as f64, None, None)))
        .collect();
    let z_planes: Vec<Arc<Surface>> = (0..=n)
        .map(|k| Arc::new(Surface::z_plane(k as f64, None, None)))
        .collect();

    let mut cells = Vec::with_capacity(n * n * n);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let region = Region::new_from_halfspace(HalfspaceType::Above(x_planes[i].clone()))
                    .intersection(&Region::new_from_halfspace(HalfspaceType::Below(
                        x_planes[i + 1].clone(),
                    )))
                    .intersection(&Region::new_from_halfspace(HalfspaceType::Above(
                        y_planes[j].clone(),
                    )))
                    .intersection(&Region::new_from_halfspace(HalfspaceType::Below(
                        y_planes[j + 1].clone(),
                    )))
                    .intersection(&Region::new_from_halfspace(HalfspaceType::Above(
                        z_planes[k].clone(),
                    )))
                    .intersection(&Region::new_from_halfspace(HalfspaceType::Below(
                        z_planes[k + 1].clone(),
                    )));
                cells.push(Cell::new(None, region, None, None));
            }
        }
    }
    Geometry::new(cells, Vec::new()).expect("cube grid build failed")
}

/// Analytical cell index for a point inside an N×N×N cube grid spanning
/// `[0, N]³`. Returns `None` if the point is outside the grid or on its
/// outer boundary (where `Cell::contains` excludes the boundary plane).
pub fn expected_cube_grid_index(n: usize, point: (f64, f64, f64)) -> Option<usize> {
    let (x, y, z) = point;
    let nf = n as f64;
    if !(0.0..nf).contains(&x) || !(0.0..nf).contains(&y) || !(0.0..nf).contains(&z) {
        return None;
    }
    let i = x.floor() as usize;
    let j = y.floor() as usize;
    let k = z.floor() as usize;
    Some(i + n * (j + n * k))
}

// ---------------------------------------------------------------------
// Concentric spheres
// ---------------------------------------------------------------------

/// Build N concentric spherical shells centred at the origin, with shell
/// `i` covering `i ≤ r < i+1` for `i = 0..N`. Total cells = N.
///
/// The bounding boxes of the outer shells fully enclose the inner shells'
/// boxes, which is adversarial for any acceleration that relies on AABB
/// pruning -- every BVH node will potentially overlap every shell.
pub fn build_concentric_shells(n: usize) -> Geometry {
    assert!(n >= 1, "concentric_shells: need at least one shell");
    let spheres: Vec<Arc<Surface>> = (1..=n)
        .map(|i| Arc::new(Surface::sphere(0.0, 0.0, 0.0, i as f64, None, None)))
        .collect();

    let mut cells = Vec::with_capacity(n);
    // Innermost cell: r < 1
    cells.push(Cell::new(
        None,
        Region::new_from_halfspace(HalfspaceType::Below(spheres[0].clone())),
        None,
        None,
    ));
    // Shells i = 1..N: i ≤ r < i+1
    for i in 1..n {
        let region = Region::new_from_halfspace(HalfspaceType::Above(spheres[i - 1].clone()))
            .intersection(&Region::new_from_halfspace(HalfspaceType::Below(
                spheres[i].clone(),
            )));
        cells.push(Cell::new(None, region, None, None));
    }
    Geometry::new(cells, Vec::new()).expect("concentric shells build failed")
}

/// Analytical shell index for a point in `build_concentric_shells(n)`.
/// Returns `None` if the point lies at or beyond the outermost shell.
pub fn expected_shell_index(n: usize, point: (f64, f64, f64)) -> Option<usize> {
    let r = (point.0 * point.0 + point.1 * point.1 + point.2 * point.2).sqrt();
    if r >= n as f64 {
        return None;
    }
    Some(r.floor() as usize)
}

// ---------------------------------------------------------------------
// Pin lattice
// ---------------------------------------------------------------------

/// Pin radius used by [`build_pin_lattice`]. Centred in the unit square,
/// well clear of the corners and edges.
pub const PIN_LATTICE_FUEL_RADIUS: f64 = 0.4;

/// Build an N×N pin lattice in the z-slab `[0, 1]`. Each pin sits in a
/// `[i, i+1] × [j, j+1]` square and contains a fuel cylinder of radius
/// 0.4 centred on the pin; the rest of the square is moderator.
///
/// Cell ordering: pin (i, j) contributes two consecutive cells --
/// `2*(i + N*j)` is the fuel, `2*(i + N*j) + 1` is the moderator.
/// Total cells = 2 N².
///
/// Mixed surface types (planes + cylinders) and non-uniform adjacency
/// (each moderator borders one fuel + four neighbouring moderators)
/// make this a useful contrast to the cube grid.
pub fn build_pin_lattice(n: usize) -> Geometry {
    let x_planes: Vec<Arc<Surface>> = (0..=n)
        .map(|i| Arc::new(Surface::x_plane(i as f64, None, None)))
        .collect();
    let y_planes: Vec<Arc<Surface>> = (0..=n)
        .map(|j| Arc::new(Surface::y_plane(j as f64, None, None)))
        .collect();
    let z_lo = Arc::new(Surface::z_plane(0.0, None, None));
    let z_hi = Arc::new(Surface::z_plane(1.0, None, None));

    let mut cells = Vec::with_capacity(2 * n * n);
    for j in 0..n {
        for i in 0..n {
            let cyl = Arc::new(Surface::z_cylinder(
                i as f64 + 0.5,
                j as f64 + 0.5,
                PIN_LATTICE_FUEL_RADIUS,
                None,
                None,
            ));
            // Square-and-slab bounds -- common factor for both cells.
            let bounds = Region::new_from_halfspace(HalfspaceType::Above(x_planes[i].clone()))
                .intersection(&Region::new_from_halfspace(HalfspaceType::Below(
                    x_planes[i + 1].clone(),
                )))
                .intersection(&Region::new_from_halfspace(HalfspaceType::Above(
                    y_planes[j].clone(),
                )))
                .intersection(&Region::new_from_halfspace(HalfspaceType::Below(
                    y_planes[j + 1].clone(),
                )))
                .intersection(&Region::new_from_halfspace(HalfspaceType::Above(
                    z_lo.clone(),
                )))
                .intersection(&Region::new_from_halfspace(HalfspaceType::Below(
                    z_hi.clone(),
                )));
            let inside_cyl = Region::new_from_halfspace(HalfspaceType::Below(cyl.clone()));
            let outside_cyl = Region::new_from_halfspace(HalfspaceType::Above(cyl));
            cells.push(Cell::new(
                None,
                bounds.intersection(&inside_cyl),
                None,
                None,
            ));
            cells.push(Cell::new(
                None,
                bounds.intersection(&outside_cyl),
                None,
                None,
            ));
        }
    }
    Geometry::new(cells, Vec::new()).expect("pin lattice build failed")
}
