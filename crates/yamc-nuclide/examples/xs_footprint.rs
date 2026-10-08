//! Throwaway: per-nuclide size of the FastXSGrid matrices, zeros, trimmed cost.
use std::collections::HashSet;
use yamc_nuclide::load_scope::LoadScope;
use yamc_nuclide::storage::nuclide_loader::load_nuclide;

fn trimmed(flat: &[f64], n_mts: usize) -> (usize, usize) {
    // (nonzero count, entries kept if each column is stored from its first to last nonzero)
    if n_mts == 0 { return (0, 0); }
    let n_e = flat.len() / n_mts;
    let mut nz = 0;
    let mut kept = 0;
    for j in 0..n_mts {
        let mut first = None;
        let mut last = 0;
        for i in 0..n_e {
            let v = flat[i * n_mts + j];
            if v != 0.0 { nz += 1; if first.is_none() { first = Some(i); } last = i; }
        }
        if let Some(f) = first { kept += last - f + 1; }
    }
    (nz, kept)
}

fn main() {
    let cache = std::env::var("HOME").unwrap() + "/.cache/yamc";
    let temp = std::env::args().nth(1).unwrap_or("294K".into());
    let mb = |n: usize| n as f64 * 8.0 / 1e6;
    let mut tot = [0.0f64; 6];
    println!("{:8} {:>7} | {:>14} {:>14} {:>14} {:>14} | {:>8}", "nuclide", "n_E", "scatter n/MB/trim", "photon", "absorb", "fission", "rxn MB");
    for name in std::env::args().skip(2) {
        let dir = format!("{cache}/endf-b8.1-{name}.arrow");
        let scope = LoadScope::full().with_temperatures(Some(HashSet::from([temp.clone()])));
        let nuc = match load_nuclide(&dir, &scope) { Ok(n) => n, Err(e) => { println!("{name}: {e}"); continue; } };
        let g = &nuc.fast_xs[0];
        let n_e = g.energy.len();
        let rxn: usize = nuc.reactions[0].values().map(|r| r.cross_section.len()).sum();
        let mut row = format!("{:8} {:>7} |", name, n_e);
        for (k, (flat, n)) in [
            (g.scatter_mt_xs.as_slice(), g.scatter_mt_numbers.len()),
            (g.photon_rxn_xs.as_slice(), g.photon_rxn_mt_numbers.len()),
            (g.absorption_mt_xs.as_slice(), g.absorption_mt_numbers.len()),
            (g.fission_mt_xs.as_slice(), g.fission_mt_numbers.len()),
        ].into_iter().enumerate() {
            let (_nz, kept) = trimmed(flat, n);
            row += &format!(" {:>3}/{:>5.0}/{:>4.0}", n, mb(flat.len()), mb(kept));
            tot[k] += mb(flat.len());
            if k == 0 { tot[4] += mb(kept); }
        }
        tot[5] += mb(rxn);
        row += &format!(" | {:>8.1}", mb(rxn));
        println!("{row}");
    }
    println!("TOTAL MB: scatter {:.0} (trimmed {:.0}), photon {:.0}, absorption {:.0}, fission {:.0}, reactions as stored {:.0}", tot[0], tot[4], tot[1], tot[2], tot[3], tot[5]);
}
