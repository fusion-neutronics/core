//! A unit square with a square hole, meshed in fine mode (cad-to-dagmc-mesher#184).
//!
//! Refinement ran on to its 50,000-iteration cap, each pass walking every
//! triangle, so the eight-edge face took minutes. It must finish promptly,
//! keep the hole, and cover the square less the hole.

use std::sync::mpsc;
use std::time::Duration;

use yamm::cdt::BoundarySplits;
use yamm::mesher::{mesh_face, mesh_face_with, EdgeInput, FaceInput, MeshMode};

/// A square wire of side `size` from `offset` along both axes, each edge
/// given from its end to its start and flagged reversed, as the issue's
/// reproduction gives it.
fn wire(size: f64, offset: f64, first_id: u64) -> Vec<EdgeInput> {
    let v = [
        [offset, offset],
        [offset + size, offset],
        [offset + size, offset + size],
        [offset, offset + size],
    ];
    (0..4)
        .map(|i| EdgeInput {
            edge_id: first_id + i as u64,
            uv_points: vec![v[(i + 1) % 4], v[i]],
            reversed: true,
        })
        .collect()
}

/// What a mesh of the face came to.
struct Meshed {
    triangles: usize,
    area: f64,
    covers_hole: bool,
    longest_edge: f64,
    smallest_angle_deg: f64,
    split_requests: Vec<(usize, usize, usize)>,
}

fn mesh(target_edge_length: f64, splits: BoundarySplits) -> Meshed {
    let face = FaceInput {
        face_id: 0,
        boundary_edges: wire(1.0, 0.0, 0),
        holes: vec![wire(0.2, 0.4, 4)],
        is_planar: true,
        mode: MeshMode::Fine { target_edge_length },
    };
    let out = if splits == BoundarySplits::Never {
        mesh_face(&face)
    } else {
        mesh_face_with(&face, splits)
    }
    .expect("the face meshes");
    let mut points = out.boundary_uv_vertices.clone();
    points.extend(&out.interior_uv_vertices);
    let len = |p: [f64; 2], q: [f64; 2]| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt();
    let mut m = Meshed {
        triangles: out.triangles.len(),
        area: 0.0,
        covers_hole: false,
        longest_edge: 0.0,
        smallest_angle_deg: 180.0,
        split_requests: out.split_requests.clone(),
    };
    for t in &out.triangles {
        let [a, b, c] = [points[t[0]], points[t[1]], points[t[2]]];
        m.area += 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])).abs();
        let centroid = [(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0];
        if (0.4..0.6).contains(&centroid[0]) && (0.4..0.6).contains(&centroid[1]) {
            m.covers_hole = true;
        }
        let (ab, bc, ca) = (len(a, b), len(b, c), len(c, a));
        m.longest_edge = m.longest_edge.max(ab).max(bc).max(ca);
        for (x, y, z) in [(ab, ca, bc), (ab, bc, ca), (bc, ca, ab)] {
            let cos = ((x * x + y * y - z * z) / (2.0 * x * y)).clamp(-1.0, 1.0);
            m.smallest_angle_deg = m.smallest_angle_deg.min(cos.acos().to_degrees());
        }
    }
    m
}

/// Mesh in a thread, failing if it takes more than 20 s.
fn within_20_s(target: f64, splits: BoundarySplits) -> Meshed {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || tx.send(mesh(target, splits)).unwrap());
    rx.recv_timeout(Duration::from_secs(20))
        .unwrap_or_else(|_| panic!("{splits:?}, target {target}: still meshing after 20 s"))
}

#[test]
fn a_face_on_its_own_splits_its_boundary_and_meshes_to_the_target() {
    // Ruppert's method: refinement splits a boundary segment it comes too
    // close to, so the mesh reaches the target however coarse the boundary
    // was given. Without it, 0.3 looped in the long-edge split and 0.1 ran
    // refinement to its 50,000-iteration cap.
    for target in [0.5, 0.3, 0.1, 0.05] {
        let m = within_20_s(target, BoundarySplits::InPlace);
        assert!(
            !m.covers_hole,
            "target {target}: a triangle covers the hole"
        );
        assert!(
            (m.area - 0.96).abs() < 1e-9,
            "target {target}: area {}",
            m.area
        );
        assert!(
            m.longest_edge <= 1.5 * target,
            "target {target}: an edge of {}",
            m.longest_edge
        );
        assert!(
            m.smallest_angle_deg > 15.0,
            "target {target}: an angle of {} degrees",
            m.smallest_angle_deg
        );
        assert!(
            m.triangles as f64 > 0.5 * 0.96 / (0.433 * target * target),
            "target {target}: {} triangles",
            m.triangles
        );
    }
}

#[test]
fn a_shared_boundary_is_reported_not_split() {
    // Where the face's edges are shared, a split made here alone would open
    // a crack, so the segments are reported for the caller to split on the
    // shared edge. The mesh still finishes promptly, and keeps the hole.
    let m = within_20_s(0.1, BoundarySplits::Report);
    assert!(!m.covers_hole);
    assert!((m.area - 0.96).abs() < 1e-9, "area {}", m.area);
    // Every edge of both wires is longer than the target; each request
    // names a real segment of a real edge (all edges here are one segment).
    assert!(!m.split_requests.is_empty());
    for &(wire, edge, segment) in &m.split_requests {
        assert!(
            wire < 2 && edge < 4 && segment == 0,
            "{:?}",
            (wire, edge, segment)
        );
    }
}

#[test]
fn never_splitting_still_finishes_where_the_long_edge_split_looped() {
    // A batch of faces shares its edges and does not split them on purpose.
    // At 0.3 the long-edge split looped forever; at smaller targets
    // refinement still runs long next to the coarse boundary, which the
    // caller avoids by meshing with `BoundarySplits::Report` and splitting
    // the shared edges itself.
    for target in [0.5, 0.3] {
        let m = within_20_s(target, BoundarySplits::Never);
        assert!(!m.covers_hole, "target {target}");
        assert!(
            (m.area - 0.96).abs() < 1e-9,
            "target {target}: area {}",
            m.area
        );
        assert!(m.split_requests.is_empty());
    }
}
