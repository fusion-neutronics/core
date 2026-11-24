//! Correctness oracle for cell-finding under many-cell geometries.
//!
//! Validates that `Geometry::find_cell_index` and `NeighborLists::find_cell`
//! agree with an analytical answer (or with each other) on three families
//! of geometries that stress the cell-finder differently:
//!
//! * **Cube grid** -- uniform axis-aligned cells, regular adjacency.
//!   Easy case for AABB-based acceleration; hard for naive linear scans.
//! * **Concentric spheres** -- nested radial shells, AABBs all overlap.
//!   Adversarial for naive AABB-tree pruning; trivial for radial structures.
//! * **Pin lattice** -- N×N cylinders inside square pins. Mixed surface
//!   types (planes + cylinders), non-uniform adjacency.
//!
//! These tests are the regression net for any future cell-finder change
//! (BVH, pre-built neighbour tables, surface-shared adjacency, etc.):
//! anything new must agree with the linear-scan answer on every probe.

mod common;

use common::cell_geometries::{
    build_concentric_shells, build_cube_grid, build_pin_lattice, expected_cube_grid_index,
    expected_shell_index,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use yamc::geometry::neighbor_lists::NeighborLists;

// =====================================================================
// Cube grid (N×N×N unit cubes spanning [0, N]³)
// =====================================================================

#[test]
fn cube_grid_4x4x4_correctness_random_points() {
    let n = 4;
    let geometry = build_cube_grid(n);
    assert_eq!(geometry.cells.len(), n * n * n);

    let mut rng = StdRng::seed_from_u64(12345);
    for _ in 0..1000 {
        let p = (
            rng.random_range(0.001..(n as f64 - 0.001)),
            rng.random_range(0.001..(n as f64 - 0.001)),
            rng.random_range(0.001..(n as f64 - 0.001)),
        );
        let want = expected_cube_grid_index(n, p);
        let got = geometry.find_cell_index(p);
        assert_eq!(got, want, "cube grid disagreement at {p:?}");
        if let Some(idx) = got {
            assert!(geometry.cells[idx].contains(p));
        }
    }
}

#[test]
fn cube_grid_10x10x10_every_cell_center_is_findable() {
    let n = 10;
    let geometry = build_cube_grid(n);
    assert_eq!(geometry.cells.len(), 1000);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let p = (i as f64 + 0.5, j as f64 + 0.5, k as f64 + 0.5);
                let want = i + n * (j + n * k);
                assert_eq!(geometry.find_cell_index(p), Some(want));
            }
        }
    }
}

#[test]
fn cube_grid_10x10x10_outside_returns_none() {
    let n = 10;
    let geometry = build_cube_grid(n);
    for &p in &[
        (-0.5, 5.0, 5.0),
        (5.0, -0.5, 5.0),
        (5.0, 5.0, -0.5),
        (10.5, 5.0, 5.0),
        (5.0, 10.5, 5.0),
        (5.0, 5.0, 10.5),
    ] {
        assert_eq!(geometry.find_cell_index(p), None);
    }
}

#[test]
fn cube_grid_neighbor_list_agrees_with_linear_scan() {
    let n = 8;
    let geometry = build_cube_grid(n);
    let mut nl = NeighborLists::new(geometry.cells.len());
    let mut rng = StdRng::seed_from_u64(0xCAFE);
    for _ in 0..500 {
        let i = rng.random_range(0..n);
        let j = rng.random_range(0..n);
        let k = rng.random_range(0..n);
        let prev = i + n * (j + n * k);
        let di = rng.random_range(-1i64..=1);
        let dj = rng.random_range(-1i64..=1);
        let dk = rng.random_range(-1i64..=1);
        let ni = (i as i64 + di).clamp(0, n as i64 - 1) as usize;
        let nj = (j as i64 + dj).clamp(0, n as i64 - 1) as usize;
        let nk = (k as i64 + dk).clamp(0, n as i64 - 1) as usize;
        let p = (ni as f64 + 0.5, nj as f64 + 0.5, nk as f64 + 0.5);
        let want = geometry.find_cell_index(p);
        let got = nl.find_cell(&geometry.cells, p, prev);
        assert_eq!(
            got, want,
            "neighbor-list disagreement at prev={prev}, p={p:?}"
        );
    }
}

// =====================================================================
// Concentric spheres (N nested shells from r=0 to r=N)
// =====================================================================

#[test]
fn concentric_shells_radial_points_agree() {
    let n = 50;
    let geometry = build_concentric_shells(n);
    assert_eq!(geometry.cells.len(), n);

    // Walk along +x axis: at r = i + 0.5 we should be in shell i.
    for i in 0..n {
        let r = i as f64 + 0.5;
        let p = (r, 0.0, 0.0);
        let want = expected_shell_index(n, p);
        let got = geometry.find_cell_index(p);
        assert_eq!(got, want, "shell disagreement at radius {r}");
        assert_eq!(want, Some(i));
    }
}

#[test]
fn concentric_shells_random_points_agree() {
    let n = 100;
    let geometry = build_concentric_shells(n);
    let mut rng = StdRng::seed_from_u64(0xBEEF);
    for _ in 0..500 {
        // Sample random points inside the outermost shell (radius < n).
        let p = (
            rng.random_range(-(n as f64 - 0.5)..(n as f64 - 0.5)),
            rng.random_range(-(n as f64 - 0.5)..(n as f64 - 0.5)),
            rng.random_range(-(n as f64 - 0.5)..(n as f64 - 0.5)),
        );
        let want = expected_shell_index(n, p);
        let got = geometry.find_cell_index(p);
        assert_eq!(got, want, "shell disagreement at {p:?}");
    }
}

#[test]
fn concentric_shells_outside_returns_none() {
    let n = 20;
    let geometry = build_concentric_shells(n);
    for &p in &[(25.0, 0.0, 0.0), (0.0, -25.0, 0.0), (15.0, 15.0, 15.0)] {
        assert_eq!(geometry.find_cell_index(p), None);
    }
}

#[test]
fn concentric_shells_neighbor_list_agrees() {
    let n = 30;
    let geometry = build_concentric_shells(n);
    let mut nl = NeighborLists::new(geometry.cells.len());
    // Each shell only has up to 2 neighbours (inner, outer); take a step
    // each direction and check the result.
    for i in 0..n {
        for &di in &[-1i64, 0, 1] {
            let j = (i as i64 + di).clamp(0, n as i64 - 1) as usize;
            let p = (j as f64 + 0.5, 0.0, 0.0);
            let want = geometry.find_cell_index(p);
            let got = nl.find_cell(&geometry.cells, p, i);
            assert_eq!(got, want, "shell neighbor disagreement: prev={i}, p={p:?}");
        }
    }
}

// =====================================================================
// Pin lattice (N×N cylinders inside square pins, fuel + moderator)
// =====================================================================

#[test]
fn pin_lattice_centers_are_fuel() {
    let n = 5;
    let geometry = build_pin_lattice(n);
    // 2 cells per pin (fuel, moderator), interleaved as [fuel, mod, fuel, mod, ...].
    assert_eq!(geometry.cells.len(), 2 * n * n);
    for j in 0..n {
        for i in 0..n {
            // Pin center → fuel cell.
            let p = (i as f64 + 0.5, j as f64 + 0.5, 0.5);
            let pin_idx = i + n * j;
            let fuel_cell = 2 * pin_idx;
            let got = geometry.find_cell_index(p);
            assert_eq!(got, Some(fuel_cell), "pin center {p:?} should be fuel");
        }
    }
}

#[test]
fn pin_lattice_corners_are_moderator() {
    let n = 5;
    let geometry = build_pin_lattice(n);
    // A point near a square corner (well outside any cylinder) is moderator.
    for j in 0..n {
        for i in 0..n {
            let p = (i as f64 + 0.05, j as f64 + 0.05, 0.5);
            let pin_idx = i + n * j;
            let mod_cell = 2 * pin_idx + 1;
            let got = geometry.find_cell_index(p);
            assert_eq!(got, Some(mod_cell), "corner {p:?} should be moderator");
        }
    }
}

#[test]
fn pin_lattice_z_slab_bounded() {
    let n = 3;
    let geometry = build_pin_lattice(n);
    // Below z=0 and above z=1: outside the slab → None.
    assert_eq!(geometry.find_cell_index((1.5, 1.5, -0.1)), None);
    assert_eq!(geometry.find_cell_index((1.5, 1.5, 1.1)), None);
    // Inside the slab → some cell.
    assert!(geometry.find_cell_index((1.5, 1.5, 0.5)).is_some());
}

#[test]
fn pin_lattice_random_points_consistent_with_linear_scan() {
    let n = 6;
    let geometry = build_pin_lattice(n);
    let mut rng = StdRng::seed_from_u64(0xD0F0);
    for _ in 0..500 {
        let p = (
            rng.random_range(0.001..(n as f64 - 0.001)),
            rng.random_range(0.001..(n as f64 - 0.001)),
            rng.random_range(0.001..0.999),
        );
        // Reference: brute-force scan via the same `Cell::contains` the
        // index-based lookup uses internally.
        let scan = geometry.cells.iter().position(|c| c.contains(p));
        let got = geometry.find_cell_index(p);
        assert_eq!(got, scan, "pin lattice disagreement at {p:?}");
    }
}
