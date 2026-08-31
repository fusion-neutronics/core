/// BCC (body-centered cubic) lattice point generation for volume interiors.
use super::aabb_bvh::TriangleBvh;

/// Generate BCC lattice points inside a closed surface mesh.
///
/// The BCC lattice places points at:
/// - Corner positions: (i*h, j*h, k*h)
/// - Body-center positions: ((i+0.5)*h, (j+0.5)*h, (k+0.5)*h)
///
/// where h = target_edge_length * sqrt(3)/2 (BCC cell size that produces
/// edges of approximately target_edge_length in the Delaunay tet mesh).
///
/// Points are filtered to keep only those strictly inside the boundary.
/// Uses a BVH for O(log n) inside/outside tests and nearest-point queries.
pub fn generate_bcc_interior_points(
    boundary_vertices: &[[f64; 3]],
    boundary_triangles: &[[usize; 3]],
    target_edge_length: f64,
) -> Vec<[f64; 3]> {
    let h = target_edge_length * 2.0 / 3.0_f64.sqrt();
    let (min_bb, max_bb) = bounding_box(boundary_vertices);
    let tdbg = std::env::var("YAMM_TIME_DBG").is_ok();
    let mut tp = std::time::Instant::now();

    // Build BVH for fast inside/outside tests and nearest-surface queries
    let tri_data: Vec<([f64; 3], [f64; 3], [f64; 3])> = boundary_triangles
        .iter()
        .map(|tri| {
            (
                boundary_vertices[tri[0]],
                boundary_vertices[tri[1]],
                boundary_vertices[tri[2]],
            )
        })
        .collect();
    let bvh = TriangleBvh::new(&tri_data);
    if tdbg {
        eprintln!("    [bcc] bvh build={:.2}s", tp.elapsed().as_secs_f64());
        tp = std::time::Instant::now();
    }

    let margin = target_edge_length * 0.5;
    let margin_sq = margin * margin * 0.25;

    // Build a coarse occupancy grid to skip BCC candidates in empty regions.
    // For thin geometries (e.g., toroidal coils), this can reject 90%+ of
    // candidates before any BVH query, turning 9M candidates into <1M.
    let voxel_size = h * 4.0; // coarse grid, 4x BCC cell size
    let nx = ((max_bb[0] - min_bb[0]) / voxel_size).ceil() as usize + 1;
    let ny = ((max_bb[1] - min_bb[1]) / voxel_size).ceil() as usize + 1;
    let nz = ((max_bb[2] - min_bb[2]) / voxel_size).ceil() as usize + 1;
    let mut occupied = vec![false; nx * ny * nz];

    // Mark voxels that contain or are near boundary triangles
    for tri in boundary_triangles {
        let verts = [
            boundary_vertices[tri[0]],
            boundary_vertices[tri[1]],
            boundary_vertices[tri[2]],
        ];
        // Find AABB of triangle and mark all overlapping voxels
        let mut tri_min = verts[0];
        let mut tri_max = verts[0];
        for v in &verts[1..] {
            for i in 0..3 {
                tri_min[i] = tri_min[i].min(v[i]);
                tri_max[i] = tri_max[i].max(v[i]);
            }
        }
        // Dilate by margin + voxel_size to ensure we don't miss interior points near boundaries
        let dilate = margin + voxel_size;
        let ix0 = ((tri_min[0] - dilate - min_bb[0]) / voxel_size)
            .floor()
            .max(0.0) as usize;
        let iy0 = ((tri_min[1] - dilate - min_bb[1]) / voxel_size)
            .floor()
            .max(0.0) as usize;
        let iz0 = ((tri_min[2] - dilate - min_bb[2]) / voxel_size)
            .floor()
            .max(0.0) as usize;
        let ix1 = ((tri_max[0] + dilate - min_bb[0]) / voxel_size).ceil() as usize;
        let iy1 = ((tri_max[1] + dilate - min_bb[1]) / voxel_size).ceil() as usize;
        let iz1 = ((tri_max[2] + dilate - min_bb[2]) / voxel_size).ceil() as usize;
        for ix in ix0..=ix1.min(nx - 1) {
            for iy in iy0..=iy1.min(ny - 1) {
                for iz in iz0..=iz1.min(nz - 1) {
                    occupied[ix * ny * nz + iy * nz + iz] = true;
                }
            }
        }
    }

    if tdbg {
        eprintln!(
            "    [bcc] occupancy mark={:.2}s (grid {}x{}x{})",
            tp.elapsed().as_secs_f64(),
            nx,
            ny,
            nz
        );
        tp = std::time::Instant::now();
    }
    let is_occupied = |p: &[f64; 3]| -> bool {
        let ix = ((p[0] - min_bb[0]) / voxel_size) as usize;
        let iy = ((p[1] - min_bb[1]) / voxel_size) as usize;
        let iz = ((p[2] - min_bb[2]) / voxel_size) as usize;
        if ix >= nx || iy >= ny || iz >= nz {
            return false;
        }
        occupied[ix * ny * nz + iy * nz + iz]
    };

    // Collect BCC candidate points, pre-filtered by occupancy grid
    let mut candidates = Vec::new();

    // Generate corner lattice points
    let mut x = min_bb[0] + margin;
    while x <= max_bb[0] - margin {
        let mut y = min_bb[1] + margin;
        while y <= max_bb[1] - margin {
            let mut z = min_bb[2] + margin;
            while z <= max_bb[2] - margin {
                let p = [x, y, z];
                if is_occupied(&p) {
                    candidates.push(p);
                }
                z += h;
            }
            y += h;
        }
        x += h;
    }

    // Generate body-center lattice points (offset by half cell in each direction)
    let half = h * 0.5;
    let mut x = min_bb[0] + margin + half;
    while x <= max_bb[0] - margin {
        let mut y = min_bb[1] + margin + half;
        while y <= max_bb[1] - margin {
            let mut z = min_bb[2] + margin + half;
            while z <= max_bb[2] - margin {
                let p = [x, y, z];
                if is_occupied(&p) {
                    candidates.push(p);
                }
                z += h;
            }
            y += h;
        }
        x += h;
    }

    if tdbg {
        eprintln!(
            "    [bcc] lattice collect={:.2}s",
            tp.elapsed().as_secs_f64()
        );
    }
    // Filter candidates: must be inside boundary and far enough from surface.
    // Use BVH nearest_point_on_surface (O(log n)) instead of O(n) vertex scan.
    // Parallelize with rayon for large candidate sets.
    use rayon::prelude::*;
    eprintln!(
        "    [bcc] {} candidates, {} boundary tris, filtering...",
        candidates.len(),
        boundary_triangles.len()
    );
    let t_filter = std::time::Instant::now();
    let mut points: Vec<[f64; 3]> = if candidates.len() > 1_000 {
        candidates
            .par_iter()
            .filter(|p| {
                if !bvh.is_point_inside(p) {
                    return false;
                }
                let nearest = bvh.nearest_point_on_surface(p);
                let dx = p[0] - nearest[0];
                let dy = p[1] - nearest[1];
                let dz = p[2] - nearest[2];
                dx * dx + dy * dy + dz * dz > margin_sq
            })
            .copied()
            .collect()
    } else {
        candidates
            .iter()
            .filter(|p| {
                if !bvh.is_point_inside(p) {
                    return false;
                }
                let nearest = bvh.nearest_point_on_surface(p);
                let dx = p[0] - nearest[0];
                let dy = p[1] - nearest[1];
                let dz = p[2] - nearest[2];
                dx * dx + dy * dy + dz * dz > margin_sq
            })
            .copied()
            .collect()
    };

    eprintln!(
        "    [bcc] filtered to {} interior points in {:.1}s",
        points.len(),
        t_filter.elapsed().as_secs_f64()
    );

    // Sort points along a Hilbert-like space-filling curve for better
    // spatial locality during Delaunay insertion (reduces cavity search time).
    hilbert_sort_3d(&mut points, &min_bb, &max_bb);

    points
}

/// Sort 3D points along a Hilbert-like space-filling curve.
///
/// Uses a recursive octant-based approach that approximates the 3D Hilbert curve.
/// Points are assigned a sort key based on their position in a recursive octree
/// subdivision, then sorted by that key.
fn hilbert_sort_3d(points: &mut [[f64; 3]], min: &[f64; 3], max: &[f64; 3]) {
    if points.len() <= 1 {
        return;
    }
    let inv = [
        if max[0] > min[0] {
            1.0 / (max[0] - min[0])
        } else {
            0.0
        },
        if max[1] > min[1] {
            1.0 / (max[1] - min[1])
        } else {
            0.0
        },
        if max[2] > min[2] {
            1.0 / (max[2] - min[2])
        } else {
            0.0
        },
    ];
    let depth = 16; // 16 levels of recursion → 48-bit key
    let mut keyed: Vec<(u64, [f64; 3])> = points
        .iter()
        .map(|p| {
            let nx = ((p[0] - min[0]) * inv[0]).clamp(0.0, 1.0);
            let ny = ((p[1] - min[1]) * inv[1]).clamp(0.0, 1.0);
            let nz = ((p[2] - min[2]) * inv[2]).clamp(0.0, 1.0);
            (hilbert_key_3d(nx, ny, nz, depth), *p)
        })
        .collect();
    keyed.sort_unstable_by_key(|k| k.0);
    for (i, (_, p)) in keyed.into_iter().enumerate() {
        points[i] = p;
    }
}

/// Compute a 3D Hilbert-like curve index for a point in [0,1]^3.
fn hilbert_key_3d(mut x: f64, mut y: f64, mut z: f64, depth: u32) -> u64 {
    let mut key = 0u64;
    for _ in 0..depth {
        let xi = if x >= 0.5 { 1u8 } else { 0 };
        let yi = if y >= 0.5 { 1u8 } else { 0 };
        let zi = if z >= 0.5 { 1u8 } else { 0 };
        let octant = (xi << 2) | (yi << 1) | zi;
        // Use a Gray-code-like mapping for better locality
        let mapped = HILBERT_OCTANT_MAP[octant as usize];
        key = (key << 3) | mapped as u64;
        // Zoom into the octant
        x = (x - xi as f64 * 0.5) * 2.0;
        y = (y - yi as f64 * 0.5) * 2.0;
        z = (z - zi as f64 * 0.5) * 2.0;
        // Apply rotation for this octant to maintain Hilbert locality
        let (nx, ny, nz) = hilbert_rotate_3d(x, y, z, octant);
        x = nx;
        y = ny;
        z = nz;
    }
    key
}

/// Rotation table for 3D Hilbert curve octants.
/// Maps octant index to a Gray-code ordering that improves spatial locality.
const HILBERT_OCTANT_MAP: [u8; 8] = [0, 1, 3, 2, 7, 6, 4, 5];

/// Apply coordinate rotation/reflection for a given octant to maintain
/// Hilbert curve locality across recursive levels.
fn hilbert_rotate_3d(x: f64, y: f64, z: f64, octant: u8) -> (f64, f64, f64) {
    match octant {
        0 => (z, x, y),
        1 => (y, z, x),
        2 => (y, z, x),
        3 => (1.0 - x, 1.0 - y, z),
        4 => (1.0 - x, 1.0 - y, z),
        5 => (1.0 - y, 1.0 - z, x),
        6 => (1.0 - y, 1.0 - z, x),
        7 => (z, x, y),
        _ => (x, y, z),
    }
}

// Ray casting and inside/outside tests now use TriangleBvh::is_point_inside()

fn bounding_box(vertices: &[[f64; 3]]) -> ([f64; 3], [f64; 3]) {
    let mut min = [f64::MAX; 3];
    let mut max = [f64::MIN; 3];
    for v in vertices {
        for i in 0..3 {
            min[i] = min[i].min(v[i]);
            max[i] = max[i].max(v[i]);
        }
    }
    (min, max)
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit cube boundary: 8 vertices, 12 triangles.
    fn unit_cube() -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
        let verts = vec![
            [0.0, 0.0, 0.0], // 0
            [1.0, 0.0, 0.0], // 1
            [1.0, 1.0, 0.0], // 2
            [0.0, 1.0, 0.0], // 3
            [0.0, 0.0, 1.0], // 4
            [1.0, 0.0, 1.0], // 5
            [1.0, 1.0, 1.0], // 6
            [0.0, 1.0, 1.0], // 7
        ];
        // Outward-facing triangles (CCW when viewed from outside)
        let tris = vec![
            // Bottom (z=0), normal -Z
            [0, 2, 1],
            [0, 3, 2],
            // Top (z=1), normal +Z
            [4, 5, 6],
            [4, 6, 7],
            // Front (y=0), normal -Y
            [0, 1, 5],
            [0, 5, 4],
            // Back (y=1), normal +Y
            [2, 3, 7],
            [2, 7, 6],
            // Left (x=0), normal -X
            [0, 4, 7],
            [0, 7, 3],
            // Right (x=1), normal +X
            [1, 2, 6],
            [1, 6, 5],
        ];
        (verts, tris)
    }

    #[test]
    fn bcc_generates_interior_points() {
        let (verts, tris) = unit_cube();
        let points = generate_bcc_interior_points(&verts, &tris, 0.3);
        assert!(
            !points.is_empty(),
            "Should generate interior points for a unit cube"
        );
        // All points should be strictly inside [0, 1]^3
        for p in &points {
            assert!(p[0] > 0.0 && p[0] < 1.0, "x={} out of bounds", p[0]);
            assert!(p[1] > 0.0 && p[1] < 1.0, "y={} out of bounds", p[1]);
            assert!(p[2] > 0.0 && p[2] < 1.0, "z={} out of bounds", p[2]);
        }
    }

    #[test]
    fn bcc_no_points_for_tiny_target() {
        let (verts, tris) = unit_cube();
        // With a very large target edge length relative to the box, should get few/no interior points
        let points = generate_bcc_interior_points(&verts, &tris, 5.0);
        // May or may not generate points depending on margin, but shouldn't panic
        let _ = points;
    }

    #[test]
    fn ray_test_inside_cube() {
        let (verts, tris) = unit_cube();
        let tri_data: Vec<([f64; 3], [f64; 3], [f64; 3])> = tris
            .iter()
            .map(|tri| (verts[tri[0]], verts[tri[1]], verts[tri[2]]))
            .collect();
        let bvh = TriangleBvh::new(&tri_data);
        assert!(bvh.is_point_inside(&[0.5, 0.5, 0.5]));
        assert!(!bvh.is_point_inside(&[2.0, 0.5, 0.5]));
    }
}
