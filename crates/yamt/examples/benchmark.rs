//! Performance benchmark for YAMT mesh queries at different mesh sizes.
//!
//! Generates synthetic cube meshes with small / medium / large triangle counts,
//! then times the core query operations:
//!
//!   - BVH build
//!   - ray_fire
//!   - point_in_volume
//!   - find_volume
//!   - find_element
//!   - closest_to_surface
//!
//! Run with:
//!
//!   cargo run --example benchmark --release

use std::hint::black_box;
use std::time::Instant;

use yamt::mesh::topology::{build_tet_aabbs, build_tet_adjacency, build_triangle_aabbs};
use yamt::{build_topology, ArrowMeshData, MeshGeometry};

fn main() {
    println!("YAMT Performance Benchmark");
    println!("==========================");

    #[cfg(feature = "simd")]
    println!("SIMD level: {}", yamt::simd_level());

    #[cfg(not(feature = "simd"))]
    println!("SIMD: disabled");

    println!();

    let sizes = [("Small", 3), ("Medium", 15), ("Large", 40)];

    // Print header
    println!(
        "{:<8} {:>8} {:>8} {:>8}  {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Size",
        "Verts",
        "Tris",
        "Tets",
        "Build(ms)",
        "RayFire",
        "PtInVol",
        "FindVol",
        "FindElem",
        "Closest"
    );
    println!("{}", "-".repeat(116));

    for (label, n) in sizes {
        run_benchmark(label, n);
    }

    println!();
    println!("Query columns show nanoseconds per query.");
    println!("Closest is O(n) triangles -- see PLAN-SPEED-UP.md for BVH acceleration.");
}

fn run_benchmark(label: &str, n: usize) {
    let mesh = generate_cube_mesh(n);

    let num_verts = mesh.vertices.len();
    let num_tris = mesh.triangles.len();
    let num_tets = mesh.tetrahedra.len();

    // -- Build --
    let t0 = Instant::now();
    let geom = MeshGeometry::from_topology(
        build_topology(mesh).expect("generated benchmark mesh is positively oriented"),
    );
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Use points that are clearly inside the unit cube
    let origin = [0.3, 0.4, 0.5];
    let direction = [1.0, 0.0, 0.0];

    let iters = 10_000u64;
    let closest_iters = 1_000u64;

    // -- ray_fire --
    let t0 = Instant::now();
    for _ in 0..iters {
        black_box(geom.ray_fire(0, origin, direction, None));
    }
    let ray_fire_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

    // -- point_in_volume --
    let t0 = Instant::now();
    for _ in 0..iters {
        black_box(geom.point_in_volume(0, origin));
    }
    let piv_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

    // -- find_volume --
    let t0 = Instant::now();
    for _ in 0..iters {
        black_box(geom.find_volume(origin));
    }
    let fv_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

    // -- find_element --
    let t0 = Instant::now();
    for _ in 0..iters {
        black_box(geom.find_element(0, origin));
    }
    let fe_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

    // -- closest_to_surface (O(n) -- fewer iterations) --
    let t0 = Instant::now();
    for _ in 0..closest_iters {
        black_box(geom.closest_to_surface(0, origin));
    }
    let closest_ns = t0.elapsed().as_nanos() as f64 / closest_iters as f64;

    println!(
        "{:<8} {:>8} {:>8} {:>8}  {:>10.2} {:>10.0} {:>10.0} {:>10.0} {:>10.0} {:>10.0}",
        label,
        num_verts,
        num_tris,
        num_tets,
        build_ms,
        ray_fire_ns,
        piv_ns,
        fv_ns,
        fe_ns,
        closest_ns
    );
}

// ---------------------------------------------------------------------------
// Mesh generation
// ---------------------------------------------------------------------------

/// Generate a unit cube mesh subdivided into an NxNxN grid.
///
/// Each grid cell is decomposed into 6 tetrahedra (Kuhn triangulation).
/// The 6 outer faces become triangulated surfaces.
///
/// Totals:
///   - Vertices:  (N+1)^3
///   - Triangles: 12 * N^2  (2 per face cell, 6 faces)
///   - Tetrahedra: 6 * N^3
fn generate_cube_mesh(n: usize) -> ArrowMeshData {
    // -- Vertices --
    let mut vertices = Vec::with_capacity((n + 1) * (n + 1) * (n + 1));
    let vid =
        |i: usize, j: usize, k: usize| -> u32 { (i + j * (n + 1) + k * (n + 1) * (n + 1)) as u32 };
    for k in 0..=n {
        for j in 0..=n {
            for i in 0..=n {
                vertices.push([
                    i as f64 / n as f64,
                    j as f64 / n as f64,
                    k as f64 / n as f64,
                ]);
            }
        }
    }

    // -- Tetrahedra (6 per cell, Kuhn triangulation) --
    let mut tetrahedra = Vec::with_capacity(6 * n * n * n);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let v0 = vid(i, j, k);
                let v1 = vid(i + 1, j, k);
                let v2 = vid(i, j + 1, k);
                let v3 = vid(i + 1, j + 1, k);
                let v4 = vid(i, j, k + 1);
                let v5 = vid(i + 1, j, k + 1);
                let v6 = vid(i, j + 1, k + 1);
                let v7 = vid(i + 1, j + 1, k + 1);
                tetrahedra.extend_from_slice(&[
                    [v0, v1, v3, v7],
                    [v0, v1, v7, v5],
                    [v0, v2, v7, v3],
                    [v0, v2, v6, v7],
                    [v0, v4, v5, v7],
                    [v0, v4, v7, v6],
                ]);
            }
        }
    }

    // -- Surface triangles (2 per face cell, 6 faces) --
    // Surface 1: z=0  Surface 2: z=1
    // Surface 3: y=0  Surface 4: y=1
    // Surface 5: x=0  Surface 6: x=1
    let mut triangles: Vec<[u32; 3]> = Vec::with_capacity(12 * n * n);
    let mut triangle_surface_ids: Vec<u32> = Vec::with_capacity(12 * n * n);
    let push_quad = |tris: &mut Vec<[u32; 3]>,
                     sids: &mut Vec<u32>,
                     surface: u32,
                     a: u32,
                     b: u32,
                     c: u32,
                     d: u32,
                     flip: bool| {
        if flip {
            tris.push([a, c, b]);
            tris.push([a, d, c]);
        } else {
            tris.push([a, b, c]);
            tris.push([a, c, d]);
        }
        sids.push(surface);
        sids.push(surface);
    };

    for j in 0..n {
        for i in 0..n {
            // z=0 (normal -z, flip winding)
            push_quad(
                &mut triangles,
                &mut triangle_surface_ids,
                1,
                vid(i, j, 0),
                vid(i + 1, j, 0),
                vid(i + 1, j + 1, 0),
                vid(i, j + 1, 0),
                true,
            );
            // z=1 (normal +z)
            push_quad(
                &mut triangles,
                &mut triangle_surface_ids,
                2,
                vid(i, j, n),
                vid(i + 1, j, n),
                vid(i + 1, j + 1, n),
                vid(i, j + 1, n),
                false,
            );
        }
    }

    for k in 0..n {
        for i in 0..n {
            // y=0 (normal -y, flip)
            push_quad(
                &mut triangles,
                &mut triangle_surface_ids,
                3,
                vid(i, 0, k),
                vid(i + 1, 0, k),
                vid(i + 1, 0, k + 1),
                vid(i, 0, k + 1),
                true,
            );
            // y=1 (normal +y)
            push_quad(
                &mut triangles,
                &mut triangle_surface_ids,
                4,
                vid(i, n, k),
                vid(i + 1, n, k),
                vid(i + 1, n, k + 1),
                vid(i, n, k + 1),
                false,
            );
        }
    }

    for k in 0..n {
        for j in 0..n {
            // x=0 (normal -x, flip)
            push_quad(
                &mut triangles,
                &mut triangle_surface_ids,
                5,
                vid(0, j, k),
                vid(0, j + 1, k),
                vid(0, j + 1, k + 1),
                vid(0, j, k + 1),
                true,
            );
            // x=1 (normal +x)
            push_quad(
                &mut triangles,
                &mut triangle_surface_ids,
                6,
                vid(n, j, k),
                vid(n, j + 1, k),
                vid(n, j + 1, k + 1),
                vid(n, j, k + 1),
                false,
            );
        }
    }

    let n_tris = triangles.len();
    let n_tets = tetrahedra.len();
    ArrowMeshData {
        triangle_aabbs: build_triangle_aabbs(&triangles, &vertices),
        tet_aabbs: build_tet_aabbs(&tetrahedra, &vertices),
        tet_adjacency: build_tet_adjacency(&tetrahedra),
        triangle_physical_groups: vec![1; n_tris],
        triangle_surface_ids,
        triangles,
        tet_volume_ids: vec![1; n_tets],
        tet_physical_groups: vec![2; n_tets],
        tetrahedra,
        vertices,
        physical_groups_json: concat!(
            r#"{"1": {"name": "mat:water", "dim": 3}, "#,
            r#""2": {"name": "boundary:vacuum", "dim": 2, "surface_ids": [1, 2, 3, 4, 5, 6]}}"#
        )
        .to_string(),
        surface_volumes_json: "[[0, null], [0, null], [0, null], [0, null], [0, null], [0, null]]"
            .to_string(),
        volume_measures: Vec::new(),
    }
}
