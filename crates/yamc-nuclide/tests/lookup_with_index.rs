//! [`FastXSGrid::lookup_with_index`] is one grid search standing in for
//! [`FastXSGrid::lookup`] followed by [`FastXSGrid::lookup_grid_index`], on the
//! transport hot path. It must return their results bit for bit, including at
//! and beyond both ends of the grid, where `lookup` clamps to the end point
//! instead of interpolating.

use std::collections::HashMap;
use std::sync::Arc;

use yamc_nuclide::buffer::F64Buffer;
use yamc_nuclide::nuclide::FastXSGrid;
use yamc_nuclide::reaction::Reaction;
use yamc_nuclide::reaction_product::ReactionProduct;
use yamc_nuclide::ParticleType;

/// A reaction over the whole grid, with irregular values so an interpolation
/// between the wrong pair of points cannot match by accident.
fn reaction(
    mt: i32,
    energies: &[f64],
    scale: f64,
    products: Vec<ReactionProduct>,
) -> Arc<Reaction> {
    let values: Vec<f64> = energies
        .iter()
        .enumerate()
        .map(|(i, e)| scale * (1.0 + (i as f64 * 0.37).sin().abs() + 1.0 / e.sqrt()))
        .collect();
    Arc::new(Reaction {
        cross_section: F64Buffer::from_slice(&values),
        threshold_idx: 0,
        energy: F64Buffer::from_slice(energies),
        mt_number: mt,
        q_value: 0.0,
        products: products.into(),
        scatter_in_cm: true,
        redundant: false,
    })
}

fn neutron() -> Vec<ReactionProduct> {
    vec![ReactionProduct {
        particle: ParticleType::Neutron,
        emission_mode: "prompt".to_string(),
        decay_rate: 0.0,
        applicability: Vec::new(),
        distribution: Vec::new(),
        product_yield: None,
    }]
}

#[test]
fn lookup_with_index_matches_lookup_and_lookup_grid_index() {
    // Log-spaced, with a repeated point (a resonance-style discontinuity),
    // which `lookup_grid_index` steps over.
    let mut energies: Vec<f64> = (0..400).map(|i| 1e-5 * 1.07_f64.powi(i)).collect();
    energies.insert(200, energies[200]);
    let grid = F64Buffer::from_slice(&energies);
    let mut reactions = HashMap::new();
    reactions.insert(2, reaction(2, &energies, 4.0, neutron()));
    reactions.insert(102, reaction(102, &energies, 0.3, Vec::new()));
    reactions.insert(107, reaction(107, &energies, 0.1, Vec::new()));
    let fast = FastXSGrid::build(&grid, &reactions, None, "Xx1").expect("the fixture builds");

    let first = energies[0];
    let last = energies[energies.len() - 1];
    let mut probes = vec![0.0, first * 0.5, first, last, last * 2.0, f64::INFINITY];
    for w in energies.windows(2) {
        probes.push(w[0]);
        probes.push(0.5 * (w[0] + w[1]));
        probes.push(w[0] + 0.999 * (w[1] - w[0]));
    }

    for e in probes {
        let (values, (i_grid, f)) = fast.lookup_with_index(e);
        let (want_i, want_f) = fast.lookup_grid_index(e);
        let want = fast.lookup(e);
        assert_eq!(i_grid, want_i, "grid index at {e:e}");
        assert_eq!(f.to_bits(), want_f.to_bits(), "fraction at {e:e}");
        for (got, want) in [
            (values.0, want.0),
            (values.1, want.1),
            (values.2, want.2),
            (values.3, want.3),
        ] {
            assert_eq!(got.to_bits(), want.to_bits(), "cross section at {e:e}");
        }
    }
}
