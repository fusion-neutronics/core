//! Diagnostic: compare σ_capture(E) under three interpolation laws.
//! 1. CPU exact: linear-in-linear-E on the actinide's own grid.
//! 2. GPU OLD: linear-in-log-E (current pre-Phase-6 behavior).
//! 3. GPU NEW: linear-in-linear-E via exp() of stored log grid
//!    (Phase 6's reconstruction).
//!
//! For 1/v-like capture cross sections, (2) should over-estimate vs (1);
//! (3) should match (1) exactly.

use yamc_nuclide::nuclide::load_nuclide;

fn cpu_linear(e: f64, grid: &[f64], xs: &[f64]) -> f64 {
    // Mirror of CPU's `FastXSGrid::lookup_total`: linear-linear on
    // the linear-E grid. partition_point + linear frac in linear-E.
    let n = grid.len();
    if e <= grid[0] {
        return xs[0];
    }
    if e >= grid[n - 1] {
        return xs[n - 1];
    }
    let i = grid.partition_point(|&x| x < e).max(1) - 1;
    let frac = (e - grid[i]) / (grid[i + 1] - grid[i]);
    xs[i] + (xs[i + 1] - xs[i]) * frac
}

fn gpu_old_log(e: f64, log_grid: &[f64], xs: &[f64]) -> f64 {
    // Pre-Phase-6 GPU: linear-in-log frac.
    let log_e = e.ln();
    let n = log_grid.len();
    if log_e <= log_grid[0] {
        return xs[0];
    }
    if log_e >= log_grid[n - 1] {
        return xs[n - 1];
    }
    let i = log_grid.partition_point(|&x| x < log_e).max(1) - 1;
    let frac = (log_e - log_grid[i]) / (log_grid[i + 1] - log_grid[i]);
    xs[i] + (xs[i + 1] - xs[i]) * frac
}

fn gpu_new_lin_via_exp(e: f64, log_grid: &[f64], xs: &[f64]) -> f64 {
    // Phase 6: bracket search in log-E, frac in linear-E reconstructed via exp.
    let log_e = e.ln();
    let n = log_grid.len();
    if log_e <= log_grid[0] {
        return xs[0];
    }
    if log_e >= log_grid[n - 1] {
        return xs[n - 1];
    }
    let i = log_grid.partition_point(|&x| x < log_e).max(1) - 1;
    let e_lo = log_grid[i].exp();
    let e_hi = log_grid[i + 1].exp();
    let denom = e_hi - e_lo;
    let frac = if denom > 0.0 { (e - e_lo) / denom } else { 0.0 };
    xs[i] + (xs[i + 1] - xs[i]) * frac
}

fn main() {
    for nm in &["Ac225", "Ac226", "Ac227"] {
        let path = format!(
            "/home/jon/nuclear_data/endf-b8.0-arrow/neutron/{}.arrow",
            nm
        );
        let nuclide = load_nuclide(&path, &yamc_nuclide::LoadScope::full()).expect("load");
        let temp = nuclide.loaded_temperatures[0].clone();
        let temp_idx = 0;

        let energy_grid: &[f64] = nuclide
            .energy
            .as_ref()
            .expect("nuclide.energy")
            .get(&temp)
            .expect("temp grid");
        let log_grid: Vec<f64> = energy_grid.iter().map(|e: &f64| e.ln()).collect();
        let rxns = &nuclide.reactions[temp_idx];
        let mt102 = rxns.get(&102).expect("MT 102 present on actinide");
        // The reaction stores its XS on its own energy grid; for a fair
        // comparison build it onto the nuclide's main grid via
        // cross_section_at, mirroring how the GPU per-MT score table is built.
        let xs102_on_main: Vec<f64> = energy_grid
            .iter()
            .map(|&e| mt102.cross_section_at(e).unwrap_or(0.0))
            .collect();

        println!(
            "\n=== {} @ {} (MT 102 on main grid: {} pts) ===",
            nm,
            temp,
            energy_grid.len()
        );

        // Probe at log-spaced energies from 1 eV to 14 MeV.
        let log_e_min = 1.0_f64.ln();
        let log_e_max = (14.0e6_f64).ln();
        let n_probes = 12;
        println!(
            "  {:>11}  {:>12}  {:>12}  {:>12}  {:>9}  {:>9}",
            "E (eV)", "σ_CPU", "σ_GPU_OLD", "σ_GPU_NEW", "OLD diff", "NEW diff"
        );
        for k in 0..n_probes {
            let log_e = log_e_min + (log_e_max - log_e_min) * (k as f64) / (n_probes as f64 - 1.0);
            let e = log_e.exp();
            let cpu = cpu_linear(e, energy_grid, &xs102_on_main);
            let old = gpu_old_log(e, &log_grid, &xs102_on_main);
            let new = gpu_new_lin_via_exp(e, &log_grid, &xs102_on_main);
            let old_pct = if cpu > 0.0 {
                100.0 * (old - cpu) / cpu
            } else {
                0.0
            };
            let new_pct = if cpu > 0.0 {
                100.0 * (new - cpu) / cpu
            } else {
                0.0
            };
            println!(
                "  {:>11.3e}  {:>12.4e}  {:>12.4e}  {:>12.4e}  {:>+8.3}%  {:>+8.3}%",
                e, cpu, old, new, old_pct, new_pct
            );
        }
    }
}
