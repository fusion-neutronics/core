//! Transport parity for the committed Arrow mesh fixtures.
//!
//! The fixtures in `crates/yamt/tests/data/` were mostly exercised by loader,
//! topology and option-matrix tests: they proved a file parses and that a run
//! completes, not that transport THROUGH the mesh lands on the right answer.
//! The tests here close that gap by running each fixture against an
//! independent representation of the same solid:
//!
//! * `cube.arrow` and `two_region_tets.arrow` are exact axis-aligned
//!   polyhedra, so a pure-CSG twin built from planes describes the identical
//!   solid with no discretisation error, and a structured
//!   `RegularRectangularMesh` over the identical volume bins the identical
//!   region. Both references are used at once: the mesh runs as GEOMETRY with
//!   a tet-mesh tally, the reference runs as CSG with a structured tally.
//!
//! Agreement is judged with a combined-standard-error z test,
//! `|a - b| / sqrt(sa^2 + sb^2)`, never a fixed percentage: a fixed percentage
//! either passes on a lucky stream or fails on an unlucky one, and several
//! bounds in this repository have done both.
//!
//! In practice these pairs run MATCHED streams: same source, same seed, same
//! physics along the same path, so the same random draws happen in the same
//! order and both sides currently agree to the last bit (measured difference
//! exactly 0.000%). That is a stronger property than the criterion asks for,
//! but it must not BE the criterion: renumbering the shared RNG stream in one
//! path and not the other would decorrelate them without anything being wrong,
//! and the z test is what stays valid when it does.
//!
//! `two_region.arrow` is deliberately absent: `hybrid_mesh_fill.rs`
//! (`hybrid_matches_pure_csg_twin`) already pins it against an exact pure-CSG
//! twin with the same z criterion.
#![cfg(feature = "mesh")]

use std::collections::HashMap;
use std::sync::Arc;
use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings, Verbose};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::Filter;
use yamc_tallies::mesh::RegularRectangularMesh;
use yamc_tallies::tally::{FluxScore, Score, Tally};
use yamc_tallies::{Estimator, MeshFilter, UnstructuredMeshFilter};

const CUBE: &str = "../yamt/tests/data/cube.arrow";
const TWO_REGION_TETS: &str = "../yamt/tests/data/two_region_tets.arrow";

/// Histories per run. Chosen for POWER, not for speed. Measured relative
/// error at this count: 0.025% per side on the cube, 0.030% on the fuel half
/// and 0.183% on the moderator half, so the combined standard error is 0.035%,
/// 0.042% and 0.259% and a 5% bias would land at 141, 118 and 19 sigma. The
/// four transport runs together take about 0.55 s: the fixtures are 1 cm cubes
/// with a vacuum boundary, so histories are short.
const N: usize = 1_000_000;

/// Agreement bound. The comparisons here are same-source, same-physics pairs,
/// so any difference is Monte Carlo noise; 4 sigma leaves room for a future
/// stream renumbering to decorrelate the two sides while still catching a real
/// geometry or attribution bias, which shows up at many tens of sigma (see the
/// perturbation figures in the per-test notes).
const Z_MAX: f64 = 4.0;

fn material(name: &str, id: u32, nuclide: &str, density: f64) -> Arc<Material> {
    let mut m = Material::new(
        HashMap::from([(nuclide.to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(density),
    )
    .unwrap();
    m.set_material_id(id);
    m.set_name(name);
    m.set_temperature("294");
    m.read_nuclear_data(
        &HashMap::from([(nuclide.to_string(), format!("tests/{nuclide}.arrow"))]),
        None,
    )
    .unwrap();
    Arc::new(m)
}

fn point_source(position: [f64; 3]) -> ParticleSource {
    ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(yamc_source::distribution::spatial::Point::new(
            position,
        )),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    })
}

fn flux_tally(name: &str, filter: Filter) -> Arc<Tally> {
    let mut t = Tally::new();
    t.estimator = Estimator::TrackLength;
    t.filters = vec![filter];
    t.scores = vec![Score::Flux(FluxScore)];
    t.name = Some(name.to_string());
    Arc::new(t)
}

/// Tet-mesh tally over one volume of an Arrow fixture. Bins are global tet
/// ids; only the queried volume's tets ever score, so the bin total is that
/// volume's track length.
fn tet_tally(name: &str, path: &str, volume_id: yamt::VolumeId) -> Arc<Tally> {
    let mesh = Arc::new(yamt::MeshGeometry::from_arrow(std::path::Path::new(path)).unwrap());
    flux_tally(
        name,
        Filter::UnstructuredMesh(UnstructuredMeshFilter::new(mesh, volume_id)),
    )
}

/// Structured rectangular tally over the same extent, as the independent
/// reference binning.
fn structured_tally(name: &str, lower: [f64; 3], upper: [f64; 3], shape: [usize; 3]) -> Arc<Tally> {
    flux_tally(
        name,
        Filter::Mesh(MeshFilter::new(RegularRectangularMesh::new(
            lower, upper, shape,
        ))),
    )
}

fn run(mut model: Model, tallies: &[Arc<Tally>]) -> Vec<yamc_tallies::TallyResult> {
    model.verbose = Verbose::silent();
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(N),
            seed: 42,
            ..Default::default()
        })
        .unwrap();
    assert!(
        model.lost_particles.is_empty(),
        "run lost particles: {:?}",
        model.lost_particles
    );
    tallies.iter().map(|t| t.finalize()).collect()
}

/// Tally total and its standard error. `aggregate_*` is the per-history total
/// statistic, so it carries the within-history correlation between bins; a
/// quadrature sum over bins would assume independence and over-state the error
/// of a track that crosses several tets, weakening the comparison.
fn bin_total(result: &yamc_tallies::TallyResult) -> (f64, f64) {
    (result.aggregate_mean(), result.aggregate_std_dev())
}

/// One bin of a result, for the structured reference mesh.
fn one_bin(result: &yamc_tallies::TallyResult, bin: usize) -> (f64, f64) {
    (result.mean[bin], result.standard_deviation[bin])
}

/// The z test used by every comparison here.
fn assert_agrees(label: &str, a: (f64, f64), b: (f64, f64)) {
    assert!(
        a.0 > 0.0 && b.0 > 0.0,
        "{label}: both sides must score (mesh {}, reference {})",
        a.0,
        b.0
    );
    let combined = (a.1 * a.1 + b.1 * b.1).sqrt();
    assert!(
        combined > 0.0,
        "{label}: both sides must report a standard error (mesh {}, reference {})",
        a.1,
        b.1
    );
    let z = (a.0 - b.0).abs() / combined;
    let rel = (a.0 - b.0).abs() / b.0;
    eprintln!(
        "{label}: mesh {:.6e} +/- {:.2e} ({:.3}%), reference {:.6e} +/- {:.2e} ({:.3}%), \
         diff {:.3}% = {z:.2} sigma",
        a.0,
        a.1,
        100.0 * a.1 / a.0,
        b.0,
        b.1,
        100.0 * b.1 / b.0,
        100.0 * rel,
    );
    assert!(
        z < Z_MAX,
        "{label}: mesh {} +/- {} vs reference {} +/- {} differ by {:.3}% ({z:.2} sigma)",
        a.0,
        a.1,
        b.0,
        b.1,
        100.0 * rel,
    );
}

fn above(s: Surface) -> Region {
    Region::new_from_halfspace(HalfspaceType::Above(Arc::new(s)))
}

fn below(s: Surface) -> Region {
    Region::new_from_halfspace(HalfspaceType::Below(Arc::new(s)))
}

/// Vacuum-bounded axis-aligned box from six planes: the CSG twin of a mesh
/// body whose whole skin carries the `boundary:vacuum` physical group.
fn vacuum_box(lower: [f64; 3], upper: [f64; 3]) -> Region {
    let vac = || Some(BoundaryType::Vacuum);
    above(Surface::x_plane(lower[0], None, vac()))
        .intersection(&below(Surface::x_plane(upper[0], None, vac())))
        .intersection(&above(Surface::y_plane(lower[1], None, vac())))
        .intersection(&below(Surface::y_plane(upper[1], None, vac())))
        .intersection(&above(Surface::z_plane(lower[2], None, vac())))
        .intersection(&below(Surface::z_plane(upper[2], None, vac())))
}

/// The y and z faces of the unit cube, all vacuum. Shared by both halves of
/// the split-box twin so the interface between them stays internal.
fn unit_cube_yz_faces() -> Region {
    let vac = || Some(BoundaryType::Vacuum);
    above(Surface::y_plane(0.0, None, vac()))
        .intersection(&below(Surface::y_plane(1.0, None, vac())))
        .intersection(&above(Surface::z_plane(0.0, None, vac())))
        .intersection(&below(Surface::z_plane(1.0, None, vac())))
}

fn mesh_model(
    path: &str,
    materials: &HashMap<String, Arc<Material>>,
    source: ParticleSource,
    tallies: Vec<Arc<Tally>>,
) -> Model {
    let mesh =
        yamc::geometry::mesh::MeshGeometry::from_arrow(std::path::Path::new(path), materials)
            .unwrap_or_else(|e| panic!("load {path}: {e}"));
    Model::new_with_mesh(mesh, vec![source], tallies)
}

/// yamt volume id containing a point, so the tet tally is aimed by geometry
/// rather than by a hard-coded index.
fn volume_at(path: &str, point: [f64; 3]) -> yamt::VolumeId {
    let mesh = yamt::MeshGeometry::from_arrow(std::path::Path::new(path)).unwrap();
    let vol = mesh.find_volume(point);
    assert_ne!(
        vol, mesh.topology.implicit_complement,
        "{point:?} is outside every volume of {path}"
    );
    vol
}

/// `cube.arrow` end to end: the fixture drives BOTH the geometry and the tally
/// binning, and is compared against a CSG unit cube tallied on a structured
/// 1x1x1 mesh over the identical `[0, 1]^3`.
///
/// What existed before covered the two halves separately and never together:
/// `test_unstructured_estimator_validation.rs` uses the fixture as a tally
/// overlay on a CSG *sphere* (issue #316), and `geometry/mesh.rs` only asks
/// the fixture-as-geometry for point location and closest-boundary distances,
/// with no transport at all. This is the first check that a particle tracked
/// through the mesh skin (six `boundary:vacuum` surfaces, adjacency handoff at
/// each crossing) deposits the same track length as the analytic solid.
///
/// Measured: 0.025% relative error on each side, agreeing at 0.00 sigma
/// (matched streams, see the module note). Teeth: shrinking the CSG twin by
/// 3% in x alone shifts it by 1.09%, which is 30.5 sigma and fails.
#[test]
fn cube_arrow_geometry_and_tet_tally_match_csg_with_structured_mesh() {
    let water = material("water", 1, "Fe56", 7.8);
    let materials = HashMap::from([("water".to_string(), Arc::clone(&water))]);
    let source = [0.5, 0.5, 0.5];

    let tet = tet_tally("cube_tets", CUBE, volume_at(CUBE, source));
    let mesh_stats = run(
        mesh_model(CUBE, &materials, point_source(source), vec![tet.clone()]),
        &[tet],
    );

    // Pure-CSG twin: the same unit cube as six vacuum-bounded planes.
    let cell = Cell::new(
        Some(1),
        vacuum_box([0.0; 3], [1.0; 3]),
        Some("cube".to_string()),
        Some(0),
    );
    let csg = Geometry::new(vec![cell], vec![water]).unwrap();
    let structured = structured_tally("cube_structured", [0.0; 3], [1.0; 3], [1, 1, 1]);
    let csg_stats = run(
        Model::new(csg, vec![point_source(source)], vec![structured.clone()]),
        &[structured],
    );

    assert_agrees(
        "cube.arrow total flux",
        bin_total(&mesh_stats[0]),
        bin_total(&csg_stats[0]),
    );
}

/// `two_region_tets.arrow` end to end, per region.
///
/// This fixture had NO transport coverage of any kind: every reference to it
/// lived inside `crates/yamt` and tested the loader, the topology build or the
/// element walk in isolation. It is the only fixture carrying two tetrahedral
/// volumes that share a conformal interface, which is exactly the geometry
/// that issue #316 broke (the walk leaked across the volume boundary and read
/// 33% low), so a per-region comparison is the check it was missing.
///
/// Reference: the identical solid as two CSG half boxes, tallied on a
/// structured 2x1x1 mesh over `[0, 1]^3` whose split plane coincides with the
/// mesh interface at x = 0.5. The source sits off centre inside the fuel half
/// so the two regions carry genuinely different flux (measured ratio about
/// 2.4), which is what makes a mis-attributed or leaked segment visible
/// instead of cancelling between the halves.
///
/// Measured: 0.030% relative error per side on the fuel half and 0.183% on the
/// moderator half, both agreeing at 0.00 sigma (matched streams). Teeth:
/// comparing the fuel tets against the WRONG structured bin fails at 1363
/// sigma, and swapping the two materials in the CSG twin fails at 41.8 sigma.
///
/// Known blind spot, stated rather than papered over: moving the CSG twin's
/// split plane to x = 0.52 (a 4% shift of the fuel half) is only a 0.05% flux
/// change and passes at 1.3 sigma. A 1 cm cube of Li6 / Fe56 is optically thin
/// at 14 MeV (mean free paths of 12 cm and 4 cm), so track length is set
/// almost entirely by streaming and barely notices WHICH material a thin slab
/// holds. This test is therefore sharp on per-volume attribution and on
/// material identity, and blunt on the exact interface position; no tolerance
/// was widened to hide that, and making it sharp would need a contrived
/// density rather than a better test.
#[test]
fn two_region_tets_per_volume_matches_csg_twin_with_structured_mesh() {
    // The fixture's fuel/moderator interface. Named so the teeth of this test
    // can be re-checked by moving the CSG twin's split off the fixture's.
    const SPLIT: f64 = 0.5;

    let fuel = material("fuel", 1, "Li6", 0.534);
    let moderator = material("moderator", 2, "Fe56", 7.8);
    let materials = HashMap::from([
        ("fuel".to_string(), Arc::clone(&fuel)),
        ("moderator".to_string(), Arc::clone(&moderator)),
    ]);
    // Off centre inside the fuel half: a flat source would let an attribution
    // error cancel between the two halves.
    let source = [0.25, 0.5, 0.5];

    let fuel_tets = tet_tally(
        "fuel_tets",
        TWO_REGION_TETS,
        volume_at(TWO_REGION_TETS, [0.25, 0.5, 0.5]),
    );
    let mod_tets = tet_tally(
        "moderator_tets",
        TWO_REGION_TETS,
        volume_at(TWO_REGION_TETS, [0.75, 0.5, 0.5]),
    );
    let tallies = vec![fuel_tets, mod_tets];
    let mesh_stats = run(
        mesh_model(
            TWO_REGION_TETS,
            &materials,
            point_source(source),
            tallies.clone(),
        ),
        &tallies,
    );

    // Pure-CSG twin: two half boxes meeting at x = 0.5. Only the outer skin
    // is vacuum; the split plane is an ordinary internal surface, exactly as
    // in the fixture (surfaces 1-10 carry `boundary:vacuum`, surface 11 is the
    // fuel/moderator interface).
    let vac = || Some(BoundaryType::Vacuum);
    let faces = unit_cube_yz_faces();
    let fuel_box = above(Surface::x_plane(0.0, None, vac()))
        .intersection(&below(Surface::x_plane(SPLIT, None, None)))
        .intersection(&faces);
    let mod_box = above(Surface::x_plane(SPLIT, None, None))
        .intersection(&below(Surface::x_plane(1.0, None, vac())))
        .intersection(&faces);
    let csg = Geometry::new(
        vec![
            Cell::new(Some(1), fuel_box, Some("fuel".to_string()), Some(0)),
            Cell::new(Some(2), mod_box, Some("moderator".to_string()), Some(1)),
        ],
        vec![fuel, moderator],
    )
    .unwrap();
    let structured = structured_tally("split_structured", [0.0; 3], [1.0; 3], [2, 1, 1]);
    let csg_stats = run(
        Model::new(csg, vec![point_source(source)], vec![structured.clone()]),
        &[structured],
    );
    let reference = &csg_stats[0];
    assert_eq!(
        reference.mean.len(),
        2,
        "2x1x1 structured mesh has two bins"
    );

    // Bin 0 is x in [0, 0.5] (fuel), bin 1 is x in [0.5, 1] (moderator).
    assert_agrees(
        "two_region_tets.arrow fuel half",
        bin_total(&mesh_stats[0]),
        one_bin(reference, 0),
    );
    assert_agrees(
        "two_region_tets.arrow moderator half",
        bin_total(&mesh_stats[1]),
        one_bin(reference, 1),
    );
    // The comparison only has power if the halves actually differ: a flat
    // profile would make a left/right swap invisible.
    let ratio = reference.mean[0] / reference.mean[1];
    assert!(
        ratio > 1.5,
        "the two halves must carry clearly different flux for this test to \
         discriminate, got a ratio of {ratio:.2}"
    );
}
