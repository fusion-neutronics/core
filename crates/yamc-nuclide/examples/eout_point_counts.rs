//! Print incident-energy / outgoing-energy point counts for every
//! inelastic + fission MT on the actinide verification nuclides, so we
//! can see how often the GPU's `MAX_EOUT_AE` / `MAX_EOUT_X` caps
//! actually cause stride-subsampling.

use yamc_nuclide::nuclide::load_nuclide;
use yamc_nuclide::reaction_product::{
    AngleEnergyDistribution, EnergyDistribution, TabulatedProbability,
};

const MAX_EOUT_AE: usize = 128; // mirror yamc-gpu's cap
const MAX_EOUT_X: usize = 64; // mirror yamc-gpu's cap

const INTERESTING_MTS: &[i32] = &[
    18, // fission
    16, 17, 22, 28, 32, 33, 34, // multi-particle inelastic
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
    75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, // discrete levels
    91, // continuum
];

fn main() {
    for nm in &["Ac225", "Ac226", "Ac227"] {
        let path = format!(
            "/home/jon/nuclear_data/endf-b8.0-arrow/neutron/{}.arrow",
            nm
        );
        let nuclide = load_nuclide(&path, &yamc_nuclide::LoadScope::full()).expect("load");
        println!("\n=== {} ===", nm);
        let rxns_by_temp = nuclide
            .loaded_temperatures
            .iter()
            .zip(nuclide.reactions.iter())
            .next()
            .unwrap();
        let temp = rxns_by_temp.0;
        let rxns = rxns_by_temp.1;
        println!("  temp: {}", temp);
        for &mt in INTERESTING_MTS {
            let Some(rxn) = rxns.get(&mt) else {
                continue;
            };
            for p in &rxn.products {
                if !p.is_particle_type(&yamc_nuclide::particle_type::ParticleType::Neutron) {
                    continue;
                }
                for ae in &p.distribution {
                    let energy_dist = match ae {
                        AngleEnergyDistribution::UncorrelatedAngleEnergy { energy, .. } => {
                            energy.as_ref()
                        }
                        _ => None,
                    };
                    if let Some(EnergyDistribution::ContinuousTabular {
                        energy, energy_out, ..
                    }) = energy_dist
                    {
                        let n_ae = energy.len();
                        let max_x = energy_out
                            .iter()
                            .map(|t| match t {
                                TabulatedProbability::Tabulated { x, .. } => x.len(),
                            })
                            .max()
                            .unwrap_or(0);
                        let exceeds_ae = if n_ae > MAX_EOUT_AE { "*" } else { " " };
                        let exceeds_x = if max_x > MAX_EOUT_X { "*" } else { " " };
                        println!(
                            "    MT {:>3}: n_ae={:>4}{}  max_n_x={:>4}{}",
                            mt, n_ae, exceeds_ae, max_x, exceeds_x
                        );
                    }
                }
            }
        }
    }
}
