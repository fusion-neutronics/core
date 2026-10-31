//! Cell-finding performance baseline across multiple geometry topologies.
//!
//! Measures `Geometry::find_cell_index` (cold linear scan) and
//! `NeighborLists::find_cell` (warm steady-state) on three families:
//!
//! * **Cube grid** -- N×N×N axis-aligned unit cubes (N³ cells).
//! * **Concentric shells** -- N nested spherical shells (N cells).
//! * **Pin lattice** -- N×N cylinders inside square pins (2 N² cells).
//!
//! These provide a baseline before any acceleration-structure change
//! (BVH, pre-built neighbor tables, etc.) -- anything new must beat
//! these numbers across all three topologies, not just the cube grid.
//!
//! Run with:
//!     cargo bench --bench cell_finding
//!
//! Each sample evaluates `QUERIES_PER_BATCH` query points; criterion
//! reports per-batch time, so divide to get the per-query cost.
//!
//! Geometry builders are intentionally duplicated from
//! `tests/common/cell_geometries.rs` because criterion benches and
//! integration tests cannot share a module.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::sync::Arc;
use yamc::geo::Surface;
use yamc::geo::{HalfspaceType, Region};
use yamc::geometry::cell::Cell;
use yamc::geometry::neighbor_lists::NeighborLists;
use yamc::geometry::Geometry;

const QUERIES_PER_BATCH: usize = 1000;
const PIN_LATTICE_FUEL_RADIUS: f64 = 0.4;

// ---------------------------------------------------------------------
// Geometry builders (mirror of tests/common/cell_geometries.rs)
// ---------------------------------------------------------------------

fn build_cube_grid(n: usize) -> Geometry {
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

fn build_concentric_shells(n: usize) -> Geometry {
    let spheres: Vec<Arc<Surface>> = (1..=n)
        .map(|i| Arc::new(Surface::sphere(0.0, 0.0, 0.0, i as f64, None, None)))
        .collect();
    let mut cells = Vec::with_capacity(n);
    cells.push(Cell::new(
        None,
        Region::new_from_halfspace(HalfspaceType::Below(spheres[0].clone())),
        None,
        None,
    ));
    for i in 1..n {
        let region = Region::new_from_halfspace(HalfspaceType::Above(spheres[i - 1].clone()))
            .intersection(&Region::new_from_halfspace(HalfspaceType::Below(
                spheres[i].clone(),
            )));
        cells.push(Cell::new(None, region, None, None));
    }
    Geometry::new(cells, Vec::new()).expect("concentric shells build failed")
}

fn build_pin_lattice(n: usize) -> Geometry {
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

// ---------------------------------------------------------------------
// Query generators
// ---------------------------------------------------------------------

/// Random points uniformly inside the cube grid.
fn cube_grid_points(n: usize, count: usize, seed: u64) -> Vec<(f64, f64, f64)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let nf = n as f64;
    (0..count)
        .map(|_| {
            (
                rng.random_range(0.001..(nf - 0.001)),
                rng.random_range(0.001..(nf - 0.001)),
                rng.random_range(0.001..(nf - 0.001)),
            )
        })
        .collect()
}

/// (prev_cell, point) pairs simulating a particle stepping into one of
/// six axis neighbours on the cube grid.
fn cube_grid_step_queries(n: usize, count: usize, seed: u64) -> Vec<(usize, (f64, f64, f64))> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            let i = rng.random_range(0..n);
            let j = rng.random_range(0..n);
            let k = rng.random_range(0..n);
            let prev = i + n * (j + n * k);
            let dir = rng.random_range(0..6);
            let (di, dj, dk) = match dir {
                0 => (-1i64, 0i64, 0i64),
                1 => (1, 0, 0),
                2 => (0, -1, 0),
                3 => (0, 1, 0),
                4 => (0, 0, -1),
                _ => (0, 0, 1),
            };
            let ni = (i as i64 + di).clamp(0, n as i64 - 1) as usize;
            let nj = (j as i64 + dj).clamp(0, n as i64 - 1) as usize;
            let nk = (k as i64 + dk).clamp(0, n as i64 - 1) as usize;
            let p = (ni as f64 + 0.5, nj as f64 + 0.5, nk as f64 + 0.5);
            (prev, p)
        })
        .collect()
}

/// Random points inside the outermost shell.
fn shells_points(n: usize, count: usize, seed: u64) -> Vec<(f64, f64, f64)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let r_max = n as f64 - 0.001;
    (0..count)
        .map(|_| {
            (
                rng.random_range(-r_max..r_max),
                rng.random_range(-r_max..r_max),
                rng.random_range(-r_max..r_max),
            )
        })
        .collect()
}

/// (prev_shell, point) -- radial steps inward or outward by one shell.
fn shells_step_queries(n: usize, count: usize, seed: u64) -> Vec<(usize, (f64, f64, f64))> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            let prev = rng.random_range(0..n);
            let dr = rng.random_range(-1i64..=1);
            let next = (prev as i64 + dr).clamp(0, n as i64 - 1) as usize;
            // Sample on the +x axis at r = next + 0.5 (clearly inside that shell).
            let p = (next as f64 + 0.5, 0.0, 0.0);
            (prev, p)
        })
        .collect()
}

/// Random points inside the pin-lattice slab.
fn pin_lattice_points(n: usize, count: usize, seed: u64) -> Vec<(f64, f64, f64)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let nf = n as f64;
    (0..count)
        .map(|_| {
            (
                rng.random_range(0.001..(nf - 0.001)),
                rng.random_range(0.001..(nf - 0.001)),
                rng.random_range(0.001..0.999),
            )
        })
        .collect()
}

/// (prev_pin_cell, point) -- a step from a fuel/moderator cell into one
/// of its neighbours within or across pins.
fn pin_lattice_step_queries(n: usize, count: usize, seed: u64) -> Vec<(usize, (f64, f64, f64))> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            let i = rng.random_range(0..n);
            let j = rng.random_range(0..n);
            // 50/50: step within the pin (fuel <-> moderator) or to a neighbour.
            let kind = rng.random_range(0..2);
            let pin_idx = i + n * j;
            let prev = 2 * pin_idx + rng.random_range(0..2);
            let p = if kind == 0 {
                // Stay in the same pin: pick fuel-center or moderator-corner.
                if rng.random_range(0..2) == 0 {
                    (i as f64 + 0.5, j as f64 + 0.5, 0.5)
                } else {
                    (i as f64 + 0.05, j as f64 + 0.05, 0.5)
                }
            } else {
                // Step to an axis-adjacent pin's center (fuel).
                let dir = rng.random_range(0..4);
                let (di, dj) = match dir {
                    0 => (-1i64, 0i64),
                    1 => (1, 0),
                    2 => (0, -1),
                    _ => (0, 1),
                };
                let ni = (i as i64 + di).clamp(0, n as i64 - 1) as usize;
                let nj = (j as i64 + dj).clamp(0, n as i64 - 1) as usize;
                (ni as f64 + 0.5, nj as f64 + 0.5, 0.5)
            };
            (prev, p)
        })
        .collect()
}

// ---------------------------------------------------------------------
// Warm-up helpers
// ---------------------------------------------------------------------

/// Walk every (cell -> 6-axis-neighbour) transition once on the cube
/// grid so the `NeighborLists` is fully populated.
fn warm_cube_grid_neighbors(geometry: &Geometry, n: usize) -> NeighborLists {
    let mut nl = NeighborLists::new(geometry.cells.len());
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let prev = i + n * (j + n * k);
                for &(di, dj, dk) in &[
                    (-1i64, 0i64, 0i64),
                    (1, 0, 0),
                    (0, -1, 0),
                    (0, 1, 0),
                    (0, 0, -1),
                    (0, 0, 1),
                ] {
                    let ni = i as i64 + di;
                    let nj = j as i64 + dj;
                    let nk = k as i64 + dk;
                    if ni < 0 || nj < 0 || nk < 0 {
                        continue;
                    }
                    let (ni, nj, nk) = (ni as usize, nj as usize, nk as usize);
                    if ni >= n || nj >= n || nk >= n {
                        continue;
                    }
                    let p = (ni as f64 + 0.5, nj as f64 + 0.5, nk as f64 + 0.5);
                    nl.find_cell(&geometry.cells, p, prev);
                }
            }
        }
    }
    nl
}

/// Walk every shell -> {inner, outer} transition once.
fn warm_shells_neighbors(geometry: &Geometry, n: usize) -> NeighborLists {
    let mut nl = NeighborLists::new(geometry.cells.len());
    for i in 0..n {
        for &di in &[-1i64, 1] {
            let j = (i as i64 + di).clamp(0, n as i64 - 1) as usize;
            let p = (j as f64 + 0.5, 0.0, 0.0);
            nl.find_cell(&geometry.cells, p, i);
        }
    }
    nl
}

/// Walk every pin's local + axis-neighbour transitions once.
fn warm_pin_lattice_neighbors(geometry: &Geometry, n: usize) -> NeighborLists {
    let mut nl = NeighborLists::new(geometry.cells.len());
    for j in 0..n {
        for i in 0..n {
            let pin_idx = i + n * j;
            let fuel = 2 * pin_idx;
            let moderator = 2 * pin_idx + 1;
            // Fuel <-> moderator within the pin.
            let fuel_pt = (i as f64 + 0.5, j as f64 + 0.5, 0.5);
            let mod_pt = (i as f64 + 0.05, j as f64 + 0.05, 0.5);
            nl.find_cell(&geometry.cells, mod_pt, fuel);
            nl.find_cell(&geometry.cells, fuel_pt, moderator);
            // Moderator -> axis-neighbouring pins (centre).
            for &(di, dj) in &[(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                let ni = i as i64 + di;
                let nj = j as i64 + dj;
                if ni < 0 || nj < 0 {
                    continue;
                }
                let (ni, nj) = (ni as usize, nj as usize);
                if ni >= n || nj >= n {
                    continue;
                }
                let p = (ni as f64 + 0.5, nj as f64 + 0.5, 0.5);
                nl.find_cell(&geometry.cells, p, moderator);
            }
        }
    }
    nl
}

// ---------------------------------------------------------------------
// Bench groups
// ---------------------------------------------------------------------

const CUBE_GRID_SIZES: &[usize] = &[4, 8, 10, 15, 20];
const SHELL_SIZES: &[usize] = &[10, 50, 100, 500, 1000];
const PIN_LATTICE_SIZES: &[usize] = &[3, 6, 10, 15, 22];

fn bench_cube_grid(c: &mut Criterion) {
    let mut cold = c.benchmark_group("cube_grid/cold_linear_scan");
    for &n in CUBE_GRID_SIZES {
        let geometry = build_cube_grid(n);
        let n_cells = geometry.cells.len();
        let points = cube_grid_points(n, QUERIES_PER_BATCH, 0xA11C);
        cold.throughput(criterion::Throughput::Elements(QUERIES_PER_BATCH as u64));
        cold.bench_with_input(
            BenchmarkId::from_parameter(n_cells),
            &points,
            |b, points| {
                b.iter(|| {
                    for &p in points {
                        black_box(geometry.find_cell_index(p));
                    }
                });
            },
        );
    }
    cold.finish();

    let mut warm = c.benchmark_group("cube_grid/warm_neighbor_list");
    for &n in CUBE_GRID_SIZES {
        let geometry = build_cube_grid(n);
        let n_cells = geometry.cells.len();
        let queries = cube_grid_step_queries(n, QUERIES_PER_BATCH, 0xB2EE);
        warm.throughput(criterion::Throughput::Elements(QUERIES_PER_BATCH as u64));
        warm.bench_with_input(
            BenchmarkId::from_parameter(n_cells),
            &queries,
            |b, queries| {
                b.iter_with_setup(
                    || warm_cube_grid_neighbors(&geometry, n),
                    |mut nl| {
                        for &(prev, p) in queries {
                            black_box(nl.find_cell(&geometry.cells, p, prev));
                        }
                    },
                );
            },
        );
    }
    warm.finish();
}

fn bench_concentric_shells(c: &mut Criterion) {
    let mut cold = c.benchmark_group("concentric_shells/cold_linear_scan");
    for &n in SHELL_SIZES {
        let geometry = build_concentric_shells(n);
        let n_cells = geometry.cells.len();
        let points = shells_points(n, QUERIES_PER_BATCH, 0xC3DD);
        cold.throughput(criterion::Throughput::Elements(QUERIES_PER_BATCH as u64));
        cold.bench_with_input(
            BenchmarkId::from_parameter(n_cells),
            &points,
            |b, points| {
                b.iter(|| {
                    for &p in points {
                        black_box(geometry.find_cell_index(p));
                    }
                });
            },
        );
    }
    cold.finish();

    let mut warm = c.benchmark_group("concentric_shells/warm_neighbor_list");
    for &n in SHELL_SIZES {
        let geometry = build_concentric_shells(n);
        let n_cells = geometry.cells.len();
        let queries = shells_step_queries(n, QUERIES_PER_BATCH, 0xD4CC);
        warm.throughput(criterion::Throughput::Elements(QUERIES_PER_BATCH as u64));
        warm.bench_with_input(
            BenchmarkId::from_parameter(n_cells),
            &queries,
            |b, queries| {
                b.iter_with_setup(
                    || warm_shells_neighbors(&geometry, n),
                    |mut nl| {
                        for &(prev, p) in queries {
                            black_box(nl.find_cell(&geometry.cells, p, prev));
                        }
                    },
                );
            },
        );
    }
    warm.finish();
}

fn bench_pin_lattice(c: &mut Criterion) {
    let mut cold = c.benchmark_group("pin_lattice/cold_linear_scan");
    for &n in PIN_LATTICE_SIZES {
        let geometry = build_pin_lattice(n);
        let n_cells = geometry.cells.len();
        let points = pin_lattice_points(n, QUERIES_PER_BATCH, 0xE5BB);
        cold.throughput(criterion::Throughput::Elements(QUERIES_PER_BATCH as u64));
        cold.bench_with_input(
            BenchmarkId::from_parameter(n_cells),
            &points,
            |b, points| {
                b.iter(|| {
                    for &p in points {
                        black_box(geometry.find_cell_index(p));
                    }
                });
            },
        );
    }
    cold.finish();

    let mut warm = c.benchmark_group("pin_lattice/warm_neighbor_list");
    for &n in PIN_LATTICE_SIZES {
        let geometry = build_pin_lattice(n);
        let n_cells = geometry.cells.len();
        let queries = pin_lattice_step_queries(n, QUERIES_PER_BATCH, 0xF6AA);
        warm.throughput(criterion::Throughput::Elements(QUERIES_PER_BATCH as u64));
        warm.bench_with_input(
            BenchmarkId::from_parameter(n_cells),
            &queries,
            |b, queries| {
                b.iter_with_setup(
                    || warm_pin_lattice_neighbors(&geometry, n),
                    |mut nl| {
                        for &(prev, p) in queries {
                            black_box(nl.find_cell(&geometry.cells, p, prev));
                        }
                    },
                );
            },
        );
    }
    warm.finish();
}

criterion_group!(
    benches,
    bench_cube_grid,
    bench_concentric_shells,
    bench_pin_lattice
);
criterion_main!(benches);
