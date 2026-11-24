//! Peak memory of transport on a mesh whose volumes share one material.
//!
//! Run as:
//!
//!     cargo run --release -p yamc --features mesh --example mesh_shared_material_memory -- FILLED [--volumes N]
//!
//! Builds N disjoint boxes (32 by default) as one surface mesh, a mesh volume
//! per box, and gives the first FILLED of them the same `Arc<Material>`: a
//! tungsten tool steel of 12 nuclides. It then runs 5000 14 MeV
//! neutrons from the centre of the first box on 4 threads and prints the
//! number of slots in the geometry's material store, the flux in the steel and
//! the peak resident set size of the process (`VmHWM` in `/proc/self/status`,
//! so Linux only).
//!
//! Transport builds one set of cross-section tables per material slot, and
//! volumes given one material share a slot, so the slot count stays at 1 and
//! the peak does not grow with FILLED. The peak is a whole-process figure, so
//! each value of FILLED needs a run of its own.
//!
//! Measured (AMD Ryzen AI Max+ 395, peak varies by a few MiB between runs):
//!
//!   filled   slots   peak RSS
//!        1       1   329 MiB
//!       32       1   335 MiB
//!
//! Needs the nuclear-data fixtures that `python scripts/fetch_test_fixtures.py`
//! fetches, and exits naming the first nuclide that is not cached.

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geometry::mesh::MeshGeometry;
use yamc::model::{Model, TransportSettings, Verbose};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::Filter;
use yamc_tallies::tally::{FluxScore, Score, Tally};
use yamc_tallies::MaterialFilter;
use yamt::mesh::topology::{build_tet_aabbs, build_triangle_aabbs};

const USAGE: &str = "usage: mesh_shared_material_memory FILLED [--volumes N]";

/// Box side and the gap between neighbouring boxes, cm.
const SIDE: f64 = 4.0;
const GAP: f64 = 1.0;

/// The faces of a box as outward-wound quads. Corner `c` is at the high end
/// of axis `a` when bit `a` of `c` is set.
const FACES: [[u32; 4]; 6] = [
    [0, 2, 3, 1],
    [4, 5, 7, 6],
    [0, 1, 5, 4],
    [3, 2, 6, 7],
    [0, 4, 6, 2],
    [1, 3, 7, 5],
];

/// A tungsten tool steel by atom fraction, from nuclides in the fixture set:
/// iron and silicon over their natural isotopes, with Cr52, Mo92, W184 and
/// W186 standing for their elements.
const STEEL: [(&str, f64); 12] = [
    ("Fe54", 0.80 * 0.05845),
    ("Fe56", 0.80 * 0.91754),
    ("Fe57", 0.80 * 0.02119),
    ("Fe58", 0.80 * 0.00282),
    ("Cr52", 0.08),
    ("Mo92", 0.03),
    ("W184", 0.03),
    ("W186", 0.03),
    ("Si28", 0.01 * 0.92223),
    ("Si29", 0.01 * 0.04685),
    ("Si30", 0.01 * 0.03092),
    ("C12", 0.02),
];

fn fail(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(1)
}

/// `volumes` disjoint boxes on a lattice four boxes wide, box `i` being the
/// mesh volume named `v{i}`.
fn boxes(volumes: usize) -> yamt::MeshGeometry {
    let mut data = yamt::ArrowMeshData::default();
    let (mut groups, mut surface_volumes) = (Vec::new(), Vec::new());
    for i in 0..volumes {
        let low = [i % 4, (i / 4) % 4, i / 16].map(|n| n as f64 * (SIDE + GAP));
        let first = data.vertices.len() as u32;
        for c in 0..8 {
            let corner = std::array::from_fn(|a| low[a] + ((c >> a) & 1) as f64 * SIDE);
            data.vertices.push(corner);
        }
        for [a, b, c, d] in FACES.map(|face| face.map(|corner| first + corner)) {
            data.triangles.extend([[a, b, c], [a, c, d]]);
        }
        // One surface and one physical group per box, numbered from 1.
        data.triangle_surface_ids.extend([i as u32 + 1; 12]);
        data.triangle_physical_groups.extend([i as i32 + 1; 12]);
        groups.push(format!(r#""{}": {{"name": "mat:v{i}", "dim": 3}}"#, i + 1));
        surface_volumes.push(format!("[{i}, null]"));
    }
    data.triangle_aabbs = build_triangle_aabbs(&data.triangles, &data.vertices);
    data.tet_aabbs = build_tet_aabbs(&[], &data.vertices);
    data.physical_groups_json = format!("{{{}}}", groups.join(", "));
    data.surface_volumes_json = format!("[{}]", surface_volumes.join(", "));
    yamt::MeshGeometry::from_topology(yamt::build_topology(data).expect("mesh topology"))
}

/// The steel at 8.0 g/cm3 and 294 K, its nuclear data read from the fixture
/// cache.
fn steel() -> Arc<Material> {
    let fractions = STEEL.iter().map(|&(n, f)| (n.to_string(), f)).collect();
    let mut steel = Material::new(fractions, "atom", "g/cm3", Some(8.0)).expect("material");
    steel.set_material_id(1);
    steel.set_temperature("294");
    let mut paths = HashMap::new();
    for (name, _) in STEEL {
        let Some(path) = yamc_test_cache::nuclide(name) else {
            let cache = yamc_test_cache::root();
            let cache = cache.display();
            fail(&format!(
                "{name} is not in the nuclear-data fixture cache {cache}"
            ))
        };
        paths.insert(name.to_string(), path);
    }
    steel.read_nuclear_data(&paths, None).expect("nuclear data");
    Arc::new(steel)
}

/// Peak resident set size of this process in MiB, where the system reports it.
fn peak_rss_mib() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kib: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib / 1024.0)
}

fn main() {
    let number = |text: Option<String>| -> usize {
        let parsed = text.and_then(|t| t.parse().ok());
        parsed.unwrap_or_else(|| fail(USAGE))
    };
    let mut args = std::env::args().skip(1);
    let filled = number(args.next());
    let volumes = match args.next().as_deref() {
        None => 32,
        Some("--volumes") => number(args.next()),
        Some(_) => fail(USAGE),
    };
    if filled < 1 || filled > volumes {
        fail(&format!("FILLED must be from 1 to {volumes}, the volumes"));
    }

    // The same `Arc` under every filled volume's name.
    let steel = steel();
    let materials = (0..filled).map(|i| (format!("v{i}"), Arc::clone(&steel)));
    let geometry =
        MeshGeometry::new(boxes(volumes), &materials.collect(), None).expect("mesh geometry");
    let slots = geometry.materials.len();

    let mut flux = Tally::new();
    let in_steel = MaterialFilter::new(&steel).expect("filter");
    flux.filters = vec![Filter::Material(in_steel)];
    flux.scores = vec![Score::Flux(FluxScore)];
    flux.reset_accumulation();
    let flux = Arc::new(flux);
    let source = ParticleSource::Neutron(Source {
        // Centre of box 0, which is always filled.
        space: SourceSpatialDistribution::Point(Point::new([SIDE / 2.0; 3])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });
    let mut model = Model::new_with_mesh(geometry, vec![source], vec![Arc::clone(&flux)]);
    model.verbose = Verbose::silent();
    // A fixed thread count, so the peak does not follow the machine's cores.
    let settings = TransportSettings {
        total_particles: Some(5000),
        seed: 11,
        threads: Some(4),
        ..Default::default()
    };
    model.simulate_transport(&settings).expect("transport");

    println!("volumes {volumes}, filled {filled}, material slots {slots}");
    println!(
        "flux in steel {:.6} +/- {:.6}, {} lost particles",
        flux.get_mean()[0],
        flux.get_std_dev()[0],
        model.lost_particles.len()
    );
    match peak_rss_mib() {
        Some(peak) => println!("peak RSS {peak:.1} MiB"),
        None => println!("peak RSS not available: /proc/self/status is Linux only"),
    }
}
