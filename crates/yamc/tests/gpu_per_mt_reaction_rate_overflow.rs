//! GPU per-MT reaction-rate fixed-point overflow regression (issue #307).
//!
//! The kernel accumulates tallies into a `u64` atomic reinterpreted as a
//! two's-complement `i64`, with a per-MT fixed-point scale sized host-side.
//! Pre-fix that scale was `2^30 · Σ_t_max / peak_Σ_MT`, comparing a channel's
//! peak against the maximum of `Σ_t` over the WHOLE energy grid. On a strong
//! 1/v absorber those two live at opposite ends of the grid: a 1 g/cc B10
//! sphere has `Σ_t_max ≈ 1.16e4 /cm` (the `(n,α)` tail at 1e-5 eV) but
//! `peak Σ_MT51 ≈ 5.0e-3 /cm` around 1 MeV, so MT 51 got scale `2.5e15`.
//! One 100k-history launch then accumulated `~1.05e19` quanta, past
//! `2^63 = 9.22e18`, and wrapped `2^64` exactly once. The tally read back
//! `-3.18e-2` against the CPU's `+4.18e-2`: a NEGATIVE production rate.
//!
//! MT 51/52/56..61 on B10 all flipped sign, and the same wrap hit Gd157
//! (MT 51/52/53/107 negative) and Xe135 -- where some channels wrapped to a
//! plausible-looking POSITIVE value 40x too small, which no positivity check
//! would have caught.
//!
//! Post-fix the scale is anchored on each channel's largest share of the
//! macroscopic total at the SAME energy (`2^30 / max_E(Σ_MT/Σ_t)`), which
//! bounds every per-step contribution by a DEFAULT-scaled total tally's.
//!
//! This test needs a real GPU. The always-run counterpart is
//! `per_mt_scale_tests` in `yamc-gpu`'s `common::tallies`, which pins the
//! scale-sizing invariant with no adapter and no nuclear data.
//!
//! Run it:
//!   cargo test -p yamc --features gpu --release \
//!       --test gpu_per_mt_reaction_rate_overflow -- --test-threads=1

#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::{ReactionRateScore, Score};
use yamc_tallies::tally::Tally;
use yamc_tallies::Mt;

/// B10's discrete-level inelastic channels, the ones that flipped negative.
const MTS: [u16; 8] = [51, 52, 56, 57, 58, 59, 60, 61];

/// Must exceed the dispatch's fixed 100k-history launch chunk, since the
/// accumulator is zeroed per launch: at 50k histories the pre-fix build
/// still returned the right answer.
const N_PARTICLES: usize = 200_000;

const SEED: u64 = 7;

fn b10_sphere() -> (Model, Vec<Arc<Tally>>, TransportSettings) {
    let surface = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 35.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(surface)));
    let mut material = Material::new(
        HashMap::from([("B10".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(1.0),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let mut nuclide_map = HashMap::new();
    nuclide_map.insert("B10".to_string(), "tests/B10.arrow".to_string());
    material.read_nuclear_data(&nuclide_map, None).unwrap();
    let cell = Cell::new(Some(1), region, Some("sphere".into()), Some(0));
    let geometry = Geometry::new(vec![cell.clone()], vec![Arc::new(material)]).unwrap();
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });
    // One tally per MT: the GPU dispatch takes a single score per tally.
    let tallies: Vec<Arc<Tally>> = MTS
        .iter()
        .map(|&mt| {
            let mut tally = Tally::new();
            tally
                .filters
                .push(Filter::Cell(CellFilter::from_id(cell.cell_id.unwrap())));
            tally.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(Mt::new(mt)))];
            tally.initialize_batches(4);
            Arc::new(tally)
        })
        .collect();
    let mut model = Model::new(geometry, vec![source], tallies.clone());
    model.gpu_max_steps_per_particle = 10_000;
    let settings = TransportSettings {
        total_particles: Some(N_PARTICLES),
        seed: SEED,
        threads: Some(1),
        ..Default::default()
    };
    (model, tallies, settings)
}

#[test]
fn gpu_b10_discrete_inelastic_rates_are_positive_and_match_cpu() {
    // Before the CPU run, not after. The file header says this test needs a
    // real GPU, and every other gpu test in this directory opens with this
    // guard, but this one never had it: `run_on_gpu` below returns
    // `GpuUnavailable` on a host with no f64 Vulkan adapter and the `expect`
    // turns that into a panic. Harmless while nothing compiled the `gpu`
    // feature in CI, and a guaranteed failure the moment something does.
    // Placed first because `b10_sphere()` plus the CPU solve is 200k histories
    // of work that only exists to be compared against a GPU result.
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }

    let (mut cpu_model, cpu_tallies, settings) = b10_sphere();
    cpu_model.simulate_transport(&settings).unwrap();
    let cpu: Vec<f64> = cpu_tallies.iter().map(|t| t.get_mean()[0]).collect();

    let (mut gpu_model, gpu_tallies, settings) = b10_sphere();
    yamc::gpu::run_on_gpu(&mut gpu_model, &settings).expect("GPU dispatch");
    let gpu: Vec<f64> = gpu_tallies.iter().map(|t| t.get_mean()[0]).collect();

    for (i, &mt) in MTS.iter().enumerate() {
        eprintln!(
            "B10 MT{mt:<3} CPU = {:+.6e}  GPU = {:+.6e}  ratio = {:.4}",
            cpu[i],
            gpu[i],
            gpu[i] / cpu[i]
        );
    }
    for (i, &mt) in MTS.iter().enumerate() {
        // The CPU reference must itself be a real, resolved signal, or the
        // comparison below would pass vacuously.
        assert!(
            cpu[i] > 1e-4,
            "CPU B10 MT{mt} rate {:.6e} is too small for this test to mean anything",
            cpu[i]
        );
        // A production reaction rate cannot be negative. This is the
        // assertion the wrapped accumulator failed.
        assert!(
            gpu[i] > 0.0,
            "GPU B10 MT{mt} reaction rate is negative ({:.6e}); the fixed-point \
             accumulator wrapped (issue #307)",
            gpu[i]
        );
        // Positivity alone is not enough: a wrap can land back in positive
        // territory (Xe135 MT 52 did). Pin the value against the CPU.
        let ratio = gpu[i] / cpu[i];
        assert!(
            (ratio - 1.0).abs() < 0.02,
            "GPU B10 MT{mt} = {:.6e} disagrees with CPU {:.6e} (ratio {ratio:.4})",
            gpu[i],
            cpu[i]
        );
    }
}
