//! Print incident-energy / cosine point counts for the MT 2 (elastic)
//! angular distribution on the verification nuclides. Historical
//! diagnostic from when the GPU flat buffers used fixed per-axis caps
//! and stride-subsampled anything larger; the GPU now stores these
//! tables tight / variable-length (issue #104), so the reference
//! counts below are kept only as a data-resolution audit.
//!
//! Source: the elastic reaction's first neutron product's
//! `AngleEnergyDistribution::UncorrelatedAngleEnergy { angle, .. }`
//! -- `angle.energy.len()` is `n_ae`, and each `angle.mu[i]` is a
//! `Tabulated { x, p, c, .. }` whose `x.len()` is the cosine count
//! for that incident-energy slice.

use yamc_nuclide::nuclide::load_nuclide;
use yamc_nuclide::particle_type::ParticleType;
use yamc_nuclide::reaction_product::AngleEnergyDistribution;

// Reference resolution thresholds (historical GPU cap values).
const MAX_AE: usize = 128;
const MAX_MU: usize = 256;

const MT_ELASTIC: i32 = 2;

fn main() {
    for nm in &["Ac225", "Ac226", "Ac227", "Fe56", "Pb208", "U235", "U238"] {
        let path = format!(
            "/home/jon/nuclear_data/endf-b8.0-arrow/neutron/{}.arrow",
            nm
        );
        let nuclide = load_nuclide(&path, &yamc_nuclide::LoadScope::full()).expect("load");
        println!("\n=== {} ===", nm);
        let (temp, rxns) = nuclide
            .loaded_temperatures
            .iter()
            .zip(nuclide.reactions.iter())
            .next()
            .unwrap();
        println!("  temp: {}", temp);

        let Some(rxn) = rxns.get(&MT_ELASTIC) else {
            println!("  MT 2: <not present>");
            continue;
        };

        let mut found_any = false;
        for p in &rxn.products {
            if !p.is_particle_type(&ParticleType::Neutron) {
                continue;
            }
            for d in &p.distribution {
                let AngleEnergyDistribution::UncorrelatedAngleEnergy { angle, .. } = d else {
                    continue;
                };
                let n_ae = angle.energy.len();
                let max_n_mu = angle.mu.iter().map(|t| t.x.len()).max().unwrap_or(0);
                let exceeds_ae = if n_ae > MAX_AE { "*" } else { " " };
                let exceeds_mu = if max_n_mu > MAX_MU { "*" } else { " " };
                let ae_ratio = n_ae as f64 / MAX_AE as f64;
                let mu_ratio = max_n_mu as f64 / MAX_MU as f64;
                println!(
                    "  MT  2: n_ae={:>4}{} ({:>5.2}x cap)  max_n_mu={:>4}{} ({:>5.2}x cap)",
                    n_ae, exceeds_ae, ae_ratio, max_n_mu, exceeds_mu, mu_ratio
                );
                found_any = true;
            }
        }
        if !found_any {
            println!("  MT 2: no UncorrelatedAngleEnergy neutron product");
        }
    }
}
