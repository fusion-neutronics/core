//! Benchmark: Arrow IPC file read performance.
//!
//! Usage:
//!     cargo run --example bench_arrow_read --release --features arrow -- <path_to_arrow_file>
//!
//! Generates the test file with cad_to_yamc first (see README).

use std::path::Path;
use std::time::Instant;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("Usage: bench_arrow_read <path_to_arrow_file>");
    let path = Path::new(&path);

    let file_size = std::fs::metadata(path).expect("Cannot stat file").len();
    println!(
        "File: {} ({:.1} MB)",
        path.display(),
        file_size as f64 / 1e6
    );

    // Warm the OS page cache by reading the file once.
    {
        let _ = std::fs::read(path).expect("Cannot read file");
    }
    println!();

    // ---------------------------------------------------------------
    // 1. read_arrow_mesh (File + FileReader)
    // ---------------------------------------------------------------
    const ROUNDS: u32 = 5;

    let mut times_file = Vec::new();
    for i in 0..ROUNDS {
        // Drop the page cache between rounds (best-effort).
        drop_caches();

        let t0 = Instant::now();
        let data = yamt::io::arrow::read_arrow_mesh(path).expect("read_arrow_mesh failed");
        let dt = t0.elapsed();
        times_file.push(dt.as_secs_f64());
        if i == 0 {
            print_data_summary(&data);
        }
    }

    let median_file = median(&mut times_file);
    println!(
        "read_arrow_mesh (File):  {:.1} ms  (median of {ROUNDS}, throughput {:.0} MB/s)",
        median_file * 1e3,
        file_size as f64 / 1e6 / median_file,
    );

    // ---------------------------------------------------------------
    // 2. Full pipeline: read + build_topology + BVH
    // ---------------------------------------------------------------
    println!();
    println!("--- Full pipeline (read + topology + BVH) ---");

    drop_caches();
    let t0 = Instant::now();
    let data = yamt::io::arrow::read_arrow_mesh(path).expect("read failed");
    let t_read = t0.elapsed();
    let t1 = Instant::now();
    let topo = yamt::io::arrow::build_topology(data).expect("build_topology failed");
    let t_topo = t1.elapsed();
    let t2 = Instant::now();
    let _geom = yamt::MeshGeometry::from_topology(topo);
    let t_bvh = t2.elapsed();
    let t_total = t0.elapsed();

    println!(
        "  File read:      {:>7.1} ms  ({:.0}%)",
        t_read.as_secs_f64() * 1e3,
        t_read.as_secs_f64() / t_total.as_secs_f64() * 100.0,
    );
    println!(
        "  build_topology: {:>7.1} ms  ({:.0}%)",
        t_topo.as_secs_f64() * 1e3,
        t_topo.as_secs_f64() / t_total.as_secs_f64() * 100.0,
    );
    println!(
        "  BVH build:      {:>7.1} ms  ({:.0}%)",
        t_bvh.as_secs_f64() * 1e3,
        t_bvh.as_secs_f64() / t_total.as_secs_f64() * 100.0,
    );
    println!("  Total:          {:>7.1} ms", t_total.as_secs_f64() * 1e3,);
}

fn print_data_summary(data: &yamt::io::arrow::ArrowMeshData) {
    println!(
        "  Vertices: {}  Triangles: {}  Tetrahedra: {}",
        data.vertices.len(),
        data.triangles.len(),
        data.tetrahedra.len(),
    );
    let data_bytes = data.vertices.len() * 24
        + data.triangles.len() * 12
        + data.tetrahedra.len() * 16
        + data.triangle_surface_ids.len() * 4
        + data.triangle_physical_groups.len() * 4
        + data.tet_volume_ids.len() * 4
        + data.tet_physical_groups.len() * 4;
    println!("  In-memory data: {:.1} MB", data_bytes as f64 / 1e6);
    println!();
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n.is_multiple_of(2) {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    } else {
        v[n / 2]
    }
}

fn drop_caches() {
    // Best-effort: only works if running as root or with appropriate
    // permissions.  On normal user accounts this is a no-op, meaning
    // the benchmark measures hot-cache performance (which is the
    // realistic scenario for repeated reads).
    let _ = std::fs::write("/proc/sys/vm/drop_caches", "3");
}
