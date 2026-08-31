//! Local (GPU-free) correctness gate for the rectangular mesh voxel-walk that
//! the GPU mesh-tally path uses (issue #234).
//!
//! `yamc_gpu::common::tallies::rect_mesh_crossings` is the plain-Rust twin of
//! the `#[cube]` kernel's inline Amanatides-Woo DDA. It operates on the packed
//! `mesh_params` descriptor rather than on a `RegularRectangularMesh`, so it
//! can be shared by the kernel. This test asserts it reproduces
//! `RegularRectangularMesh::bins_crossed` (the CPU tally's own binner) exactly,
//! over a spread of segment geometries: starting inside/outside, axis-parallel,
//! diagonal, grazing, and exiting through every face. No GPU is required (it
//! compares two CPU implementations), but it is gated behind `feature = "gpu"`
//! because `rect_mesh_crossings` lives in the optional `yamc-gpu` crate.
#![cfg(feature = "gpu")]

use yamc_gpu::common::tallies::rect_mesh_crossings;
use yamc_tallies::mesh::RegularRectangularMesh;

/// Pack a rectangular mesh into the `mesh_params` layout `build_tallies_pack`
/// emits: `[lower_left[3], upper_right[3], inv_width[3], width[3], shape[3]]`.
fn pack(m: &RegularRectangularMesh) -> Vec<f64> {
    let ll = m.lower_left();
    let ur = m.upper_right();
    let w = m.width();
    let s = m.shape();
    let mut p = Vec::with_capacity(15);
    p.extend_from_slice(&ll);
    p.extend_from_slice(&ur);
    p.extend_from_slice(&[1.0 / w[0], 1.0 / w[1], 1.0 / w[2]]);
    p.extend_from_slice(&w);
    p.extend_from_slice(&[s[0] as f64, s[1] as f64, s[2] as f64]);
    p
}

fn unit_dir(r0: [f64; 3], r1: [f64; 3]) -> [f64; 3] {
    let d = [r1[0] - r0[0], r1[1] - r0[1], r1[2] - r0[2]];
    let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    [d[0] / n, d[1] / n, d[2] / n]
}

/// Assert the twin and the reference mesh produce the identical crossing list.
fn assert_same(m: &RegularRectangularMesh, r0: [f64; 3], r1: [f64; 3]) {
    let params = pack(m);
    let dir = unit_dir(r0, r1);
    let reference = m.bins_crossed(r0, r1, dir);
    let mut twin: Vec<(u32, f64)> = Vec::new();
    rect_mesh_crossings(&params, r0, r1, dir, |bin, lf| twin.push((bin, lf)));

    assert_eq!(
        twin.len(),
        reference.len(),
        "crossing count differs for r0={r0:?} r1={r1:?}: twin {} vs ref {}",
        twin.len(),
        reference.len()
    );
    for (i, (t, rc)) in twin.iter().zip(reference.iter()).enumerate() {
        assert_eq!(
            t.0 as usize, rc.bin,
            "voxel bin differs at crossing {i} for r0={r0:?} r1={r1:?}: twin {} vs ref {}",
            t.0, rc.bin
        );
        let df = (t.1 - rc.length_fraction).abs();
        assert!(
            df <= 1e-12 * rc.length_fraction.abs().max(1.0),
            "length_fraction differs at crossing {i} for r0={r0:?} r1={r1:?}: \
             twin {} vs ref {} (|d|={df:.3e})",
            t.1,
            rc.length_fraction
        );
    }
}

#[test]
fn twin_matches_reference_cube_mesh() {
    // 4x4x4 mesh over [0,4]^3 (unit voxels).
    let m = RegularRectangularMesh::new([0.0, 0.0, 0.0], [4.0, 4.0, 4.0], [4, 4, 4]);

    // Straight diagonals, axis-parallel rays, off-axis, starting inside.
    assert_same(&m, [0.5, 0.5, 0.5], [3.5, 3.5, 3.5]); // main diagonal
    assert_same(&m, [0.5, 0.5, 0.5], [3.5, 0.5, 0.5]); // +x axis-parallel
    assert_same(&m, [0.5, 0.5, 0.5], [0.5, 3.5, 0.5]); // +y axis-parallel
    assert_same(&m, [0.5, 0.5, 0.5], [0.5, 0.5, 3.5]); // +z axis-parallel
    assert_same(&m, [3.5, 3.5, 3.5], [0.5, 0.5, 0.5]); // reverse diagonal
    assert_same(&m, [0.2, 1.7, 3.1], [3.9, 2.2, 0.4]); // generic interior
    assert_same(&m, [1.5, 0.5, 2.5], [1.5, 3.5, 2.5]); // -? along y only

    // Starting OUTSIDE the mesh, entering through a face.
    assert_same(&m, [-2.0, 2.0, 2.0], [6.0, 2.0, 2.0]); // through +x
    assert_same(&m, [2.0, -2.0, 2.0], [2.0, 6.0, 2.0]); // through +y
    assert_same(&m, [-1.0, -1.0, -1.0], [5.0, 5.0, 5.0]); // corner-to-corner
    assert_same(&m, [-1.0, 0.3, 0.7], [5.0, 3.7, 2.1]); // oblique entry

    // Missing the mesh entirely.
    assert_same(&m, [-2.0, -2.0, -2.0], [-2.0, 6.0, -2.0]);
    assert_same(&m, [10.0, 10.0, 10.0], [11.0, 12.0, 13.0]);

    // Short segment fully inside one voxel.
    assert_same(&m, [1.2, 1.2, 1.2], [1.3, 1.25, 1.22]);
}

#[test]
fn twin_matches_reference_nonuniform_offset_mesh() {
    // Non-cubic voxels, offset origin, asymmetric shape.
    let m = RegularRectangularMesh::new([-3.0, 1.0, -0.5], [5.0, 7.0, 2.5], [8, 3, 6]);

    assert_same(&m, [-2.5, 1.5, 0.0], [4.5, 6.5, 2.0]);
    assert_same(&m, [-10.0, 4.0, 1.0], [10.0, 4.0, 1.0]); // long +x sweep
    assert_same(&m, [0.0, 4.0, -5.0], [0.0, 4.0, 5.0]); // +z sweep through
    assert_same(&m, [-2.9, 6.9, 2.4], [4.9, 1.1, -0.4]); // near-corner diagonal
    assert_same(&m, [1.234, 2.345, 0.123], [4.9, 6.9, 2.4]);
    assert_same(&m, [-2.99, 1.01, -0.49], [4.99, 6.99, 2.49]); // near-corner span
}

#[test]
fn twin_matches_reference_fine_mesh_many_segments() {
    // A finer mesh + a deterministic pseudo-random spread of segments.
    let m = RegularRectangularMesh::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [10, 12, 7]);
    // Simple LCG so the test is deterministic without an rng dependency.
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // map to [-1.5, 1.5]
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 3.0 - 1.5
    };
    for _ in 0..2000 {
        let r0 = [next(), next(), next()];
        let r1 = [next(), next(), next()];
        // skip degenerate (near-zero-length) segments
        let dl =
            ((r1[0] - r0[0]).powi(2) + (r1[1] - r0[1]).powi(2) + (r1[2] - r0[2]).powi(2)).sqrt();
        if dl < 1e-3 {
            continue;
        }
        assert_same(&m, r0, r1);
    }
}
