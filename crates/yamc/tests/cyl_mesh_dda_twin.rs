//! Local (GPU-free) correctness gate for the cylindrical mesh voxel-walk that
//! the GPU mesh-tally path uses (issue #279).
//!
//! `yamc_gpu::common::tallies::cyl_mesh_crossings` / `cyl_mesh_bin_at` are the
//! plain-Rust twins of the `#[cube]` kernel's analytic (r, phi, z) DDA. They
//! operate on the packed `mesh_params` descriptor rather than on a
//! `CylindricalMesh`, so they can be shared by the kernel. This test asserts
//! they reproduce `CylindricalMesh::bins_crossed` / `get_bin` (the CPU tally's
//! own binner) exactly, over a spread of segment geometries: through-axis,
//! axis-parallel, central-hole re-entry, the `full_phi` seam, and partial-phi
//! wedge entry. No GPU is required (it compares two CPU implementations), but it
//! is gated behind `feature = "gpu"` because the twins live in the optional
//! `yamc-gpu` crate.
//!
//! This is the gate the plan requires to pass before the GPU kernel (which
//! mirrors these twins under the cubecl constraints) is trusted.
#![cfg(feature = "gpu")]

use yamc_gpu::common::tallies::{cyl_mesh_bin_at, cyl_mesh_crossings};
use yamc_tallies::mesh::CylindricalMesh;

/// Pack a cylindrical mesh into the `mesh_params` layout
/// `build_mesh_descriptor` emits: header `[origin[3], nr, nphi, nz, full_phi]`
/// then grids `r_grid`, `r_grid_sq (=r*r)`, `phi_grid`, `z_grid`.
fn pack(m: &CylindricalMesh) -> Vec<f64> {
    let o = m.origin();
    let r = m.r_grid();
    let phi = m.phi_grid();
    let z = m.z_grid();
    let shape = m.shape();
    let mut p = Vec::new();
    p.extend_from_slice(&o);
    p.push(shape[0] as f64);
    p.push(shape[1] as f64);
    p.push(shape[2] as f64);
    p.push(if m.full_phi() { 1.0 } else { 0.0 });
    p.extend_from_slice(r);
    p.extend(r.iter().map(|x| x * x));
    p.extend_from_slice(phi);
    p.extend_from_slice(z);
    p
}

fn unit_dir(r0: [f64; 3], r1: [f64; 3]) -> [f64; 3] {
    let d = [r1[0] - r0[0], r1[1] - r0[1], r1[2] - r0[2]];
    let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    [d[0] / n, d[1] / n, d[2] / n]
}

/// Assert the twin walk reproduces `CylindricalMesh::bins_crossed` exactly.
fn assert_same(m: &CylindricalMesh, r0: [f64; 3], r1: [f64; 3]) {
    let params = pack(m);
    let dir = unit_dir(r0, r1);
    let reference = m.bins_crossed(r0, r1, dir);
    let mut twin: Vec<(u32, f64)> = Vec::new();
    cyl_mesh_crossings(&params, r0, r1, dir, |bin, lf| twin.push((bin, lf)));

    assert_eq!(
        twin.len(),
        reference.len(),
        "crossing count differs for r0={r0:?} r1={r1:?}: twin {} vs ref {}\n twin={twin:?}\n ref={:?}",
        twin.len(),
        reference.len(),
        reference.iter().map(|c| (c.bin, c.length_fraction)).collect::<Vec<_>>(),
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

/// Assert the twin point-binning reproduces `CylindricalMesh::get_bin` exactly.
fn assert_bin_at(m: &CylindricalMesh, pos: [f64; 3]) {
    let params = pack(m);
    let reference = m.get_bin(pos).map(|b| b as u32);
    let twin = cyl_mesh_bin_at(&params, pos);
    assert_eq!(
        twin, reference,
        "get_bin differs at pos={pos:?}: twin {twin:?} vs ref {reference:?}"
    );
}

/// Full-2pi mesh whose innermost ring touches the axis (r_min = 0), so the seam
/// wraps and through-axis tracks flip phi by pi without crossing a phi wall.
fn full_phi_mesh() -> CylindricalMesh {
    CylindricalMesh::uniform(
        [0.0, 0.0, 0.0],
        (0.0, 10.0),
        (0.0, std::f64::consts::TAU),
        (-10.0, 10.0),
        [5, 4, 5],
    )
}

/// Central-hole mesh (r_min = 2): tracks through the middle leave and re-enter,
/// producing two crossing groups with an untallied gap.
fn central_hole_mesh() -> CylindricalMesh {
    CylindricalMesh::uniform(
        [0.0, 0.0, 0.0],
        (2.0, 10.0),
        (0.0, std::f64::consts::TAU),
        (-6.0, 6.0),
        [4, 6, 3],
    )
}

/// Partial-phi wedge (upper half plane): the phi ends are hard mesh boundaries,
/// not a seam, so entry through a phi wall is exercised.
fn half_phi_mesh() -> CylindricalMesh {
    CylindricalMesh::uniform(
        [0.0, 0.0, 0.0],
        (0.0, 8.0),
        (0.0, std::f64::consts::PI),
        (-5.0, 5.0),
        [4, 3, 4],
    )
}

/// Non-uniform grids + offset origin: catches any assumption of uniform spacing
/// or a zero origin in the descriptor parsing.
fn nonuniform_offset_mesh() -> CylindricalMesh {
    CylindricalMesh::new(
        [1.5, -2.0, 0.5],
        vec![0.0, 1.0, 2.5, 4.0, 7.0],
        vec![0.0, 0.5, 1.7, std::f64::consts::PI],
        vec![-4.0, -1.0, 0.5, 3.0],
    )
}

#[test]
fn twin_matches_reference_full_phi() {
    let m = full_phi_mesh();

    // Through the axis (phi flips by pi): both the radial perigee event and the
    // seam-wrap must be reproduced.
    assert_same(&m, [-8.0, -0.3, 0.0], [8.0, 0.3, 0.0]);
    assert_same(&m, [-6.0, 2.0, -4.0], [6.0, -2.0, 4.0]);
    // Straight through the centre exactly along +x.
    assert_same(&m, [-9.0, 0.0, 1.0], [9.0, 0.0, 1.0]);
    // Axis-parallel z sweep (rho constant, no radial/phi crossing).
    assert_same(&m, [3.0, 3.0, -9.0], [3.0, 3.0, 9.0]);
    // Radial outward from near the axis.
    assert_same(&m, [0.1, 0.1, 0.0], [7.0, 5.0, 2.0]);
    // Tangential chord that never reaches the inner rings.
    assert_same(&m, [-9.0, 6.0, 0.0], [9.0, 6.0, 0.0]);
    // Starting outside, entering through the outer shell.
    assert_same(&m, [-20.0, 1.3, -2.0], [20.0, -1.3, 2.0]);
    // Fully outside (misses entirely).
    assert_same(&m, [15.0, 15.0, 0.0], [15.0, 15.0, 5.0]);
    // Short segment inside one voxel.
    assert_same(&m, [4.0, 1.0, 2.0], [4.05, 1.02, 2.01]);
}

#[test]
fn twin_matches_reference_central_hole() {
    let m = central_hole_mesh();

    // Straight through the middle: exits the inner shell, crosses the untallied
    // hole, re-enters -> two crossing groups.
    assert_same(&m, [-9.0, 0.0, 0.0], [9.0, 0.0, 0.0]);
    assert_same(&m, [-8.0, 1.5, -3.0], [8.0, -1.5, 3.0]);
    // Off-centre chord that stays in the outer rings (never enters the hole).
    assert_same(&m, [-9.0, 5.0, 0.0], [9.0, 5.0, 0.0]);
    // Grazing the inner shell.
    assert_same(&m, [-9.0, 2.0, 1.0], [9.0, 2.0, 1.0]);
    // Diagonal through the hole with z motion.
    assert_same(&m, [-7.0, -7.0, -5.0], [7.0, 7.0, 5.0]);
}

#[test]
fn twin_matches_reference_half_phi_wedge() {
    let m = half_phi_mesh();

    // Enter through a phi end wall (the y=0 half-plane) from below.
    assert_same(&m, [3.0, -4.0, 0.0], [3.0, 6.0, 0.0]);
    assert_same(&m, [-5.0, -1.0, 1.0], [5.0, 4.0, -1.0]);
    // A chord entirely within the upper half.
    assert_same(&m, [-6.0, 1.0, 0.0], [6.0, 3.0, 0.0]);
    // Track that stays in the lower (out-of-mesh) half: no crossings.
    assert_same(&m, [-6.0, -1.0, 0.0], [6.0, -3.0, 0.0]);
    // Along +y across the wedge.
    assert_same(&m, [1.0, -6.0, 2.0], [1.0, 6.0, 2.0]);
}

#[test]
fn twin_matches_reference_nonuniform_offset() {
    let m = nonuniform_offset_mesh();

    assert_same(&m, [-6.0, -1.5, -3.0], [8.0, 0.5, 2.5]);
    assert_same(&m, [1.5, -10.0, 0.0], [1.5, 10.0, 0.0]); // +y sweep through origin
    assert_same(&m, [1.6, -1.9, -3.9], [6.0, 1.0, 2.9]);
    assert_same(&m, [-4.0, 3.0, 0.5], [7.0, -5.0, 0.5]);
}

#[test]
fn twin_matches_reference_bin_at() {
    // Point-binning (collision estimator path) over the full-phi and central
    // hole meshes, including on-axis, in-hole, and out-of-mesh points.
    let full = full_phi_mesh();
    for pos in [
        [0.0, 0.0, 0.0], // on axis
        [3.0, 4.0, 1.0], // interior
        [-5.0, -5.0, -2.0],
        [1.0, -1.0, 8.5],
        [9.99, 0.0, 0.0], // near outer shell
        [12.0, 0.0, 0.0], // outside (r too big)
        [0.0, 0.0, 11.0], // outside (z too big)
    ] {
        assert_bin_at(&full, pos);
    }

    let hole = central_hole_mesh();
    for pos in [
        [0.0, 0.0, 0.0], // in the hole -> None
        [1.0, 0.5, 0.0], // still in hole
        [5.0, 0.0, 0.0],
        [-4.0, 3.0, -2.0],
        [8.0, 0.0, 5.5],
    ] {
        assert_bin_at(&hole, pos);
    }
}

#[test]
fn twin_matches_reference_random_sweep() {
    // A deterministic pseudo-random spread of segments over the full-phi and
    // central-hole meshes. Simple LCG so the test needs no rng dependency.
    let meshes = [full_phi_mesh(), central_hole_mesh(), half_phi_mesh()];
    let mut s: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = |lo: f64, hi: f64| {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        lo + ((s >> 11) as f64 / (1u64 << 53) as f64) * (hi - lo)
    };
    for m in &meshes {
        for _ in 0..3000 {
            let r0 = [next(-14.0, 14.0), next(-14.0, 14.0), next(-9.0, 9.0)];
            let r1 = [next(-14.0, 14.0), next(-14.0, 14.0), next(-9.0, 9.0)];
            let dl = ((r1[0] - r0[0]).powi(2) + (r1[1] - r0[1]).powi(2) + (r1[2] - r0[2]).powi(2))
                .sqrt();
            if dl < 1e-2 {
                continue;
            }
            assert_same(m, r0, r1);
        }
    }
}
