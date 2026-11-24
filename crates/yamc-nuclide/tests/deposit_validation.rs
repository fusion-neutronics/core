//! The per-reaction deposits against the stored rows they split.
//!
//! Ignored: it prints the table the pull request reports. Point
//! `YAMC_VALIDATION_DATA` at a release tree (`<root>/neutron/<Nuclide>.arrow`)
//! and run
//!
//! ```text
//! YAMC_VALIDATION_DATA=... cargo test --release -p yamc-nuclide --test deposit_validation -- --ignored --nocapture
//! ```
//!
//! `YAMC_VALIDATION_NUCLIDES` (comma-separated) picks the nuclides.

use std::collections::BTreeSet;
use std::path::PathBuf;

use yamc_nuclide::deposit::{deposits, stored_row, unexplained_share, Response};
use yamc_nuclide::nuclide::load_nuclide;
use yamc_nuclide::LoadScope;

/// One of HEATR's partial groupings: its MT and which reactions it sums.
type Group = (i32, fn(i32) -> bool);

const ENERGIES: [f64; 7] = [0.0253, 1e3, 1e5, 1e6, 2.45e6, 8e6, 14.1e6];

#[test]
#[ignore]
fn partial_sums_against_the_stored_rows() {
    let Ok(root) = std::env::var("YAMC_VALIDATION_DATA") else {
        eprintln!("skipping: set YAMC_VALIDATION_DATA to a release tree");
        return;
    };
    let names = std::env::var("YAMC_VALIDATION_NUCLIDES")
        .unwrap_or_else(|_| "Fe56,W184,Li6,Be9,Cu63,U238".to_string());
    for name in names.split(',') {
        let dir = PathBuf::from(&root)
            .join("neutron")
            .join(format!("{name}.arrow"));
        let scope = LoadScope::full().with_temperatures(Some(["294".to_string()].into()));
        let nuclide = load_nuclide(&dir, &scope).expect("load");
        let t = 0;
        let label = &nuclide.loaded_temperatures[t];
        let grid = nuclide.energy.as_ref().unwrap()[label].as_slice().to_vec();
        let reactions = &nuclide.reactions[t];
        let mts: BTreeSet<i32> = reactions
            .iter()
            .filter(|(_, r)| !r.redundant)
            .map(|(mt, _)| *mt)
            .collect();
        for response in Response::ALL {
            let Some(stored) = stored_row(&nuclide, t, response) else {
                continue;
            };
            let start = std::time::Instant::now();
            let d = deposits(&nuclide, t, response, &mts);
            let elapsed = start.elapsed();
            let share = unexplained_share(&nuclide, t, response, &d, (0.0, f64::INFINITY))
                .unwrap_or(f64::NAN);
            let bands: Vec<String> = yamc_nuclide::deposit::BANDS
                .iter()
                .map(|b| match unexplained_share(&nuclide, t, response, &d, *b) {
                    Some(x) => format!("{:.2}%", 100.0 * x),
                    None => "-".to_string(),
                })
                .collect();
            println!("BANDS {name} {} {}", response.mt(), bands.join(" "));
            let unsplit: Vec<i32> = mts.iter().filter(|m| !d.contains_key(m)).copied().collect();
            println!(
                "{name} MT {}: unexplained {:.3}% over lethargy, {:.2} s, not split {unsplit:?}",
                response.mt(),
                100.0 * share,
                elapsed.as_secs_f64()
            );
            if std::env::var("YAMC_VALIDATION_PROFILE").is_ok() {
                // Residual and stored magnitude per decade, lethargy-weighted.
                let mut explained = vec![0.0; grid.len()];
                for (mt, h) in &d {
                    let r = &reactions[mt];
                    for (j, x) in r.cross_section.as_slice().iter().enumerate() {
                        let i = r.threshold_idx + j;
                        if i < grid.len() {
                            explained[i] += x * h[i];
                        }
                    }
                }
                let mut rows = std::collections::BTreeMap::<i32, (f64, f64)>::new();
                for i in 0..grid.len() - 1 {
                    if grid[i + 1] <= grid[i] {
                        continue;
                    }
                    let du = (grid[i + 1] / grid[i]).ln();
                    let row = rows.entry(grid[i].log10().floor() as i32).or_default();
                    row.0 += du * (stored[i] - explained[i]).abs();
                    row.1 += du * stored[i].abs();
                }
                for (dec, (res, tot)) in rows {
                    println!("    decade 1e{dec}: residual {res:.3e} of {tot:.3e}");
                }
            }
            if std::env::var("YAMC_VALIDATION_GROUPS").is_ok() {
                // HEATR's partial groupings, to compare with its own partials.
                let groups: Vec<Group> = match response {
                    Response::DamageEnergy => vec![
                        (445, |m| m == 2),
                        (446, |m| (51..=91).contains(&m)),
                        (447, |m| {
                            (102..=120).contains(&m) || (600..=849).contains(&m)
                        }),
                    ],
                    _ => vec![
                        (302, |m| m == 2),
                        (304, |m| (51..=91).contains(&m)),
                        (316, |m| m == 16),
                        (317, |m| m == 17),
                        (322, |m| m == 22),
                        (328, |m| m == 28),
                        (391, |m| m == 91),
                        (402, |m| m == 102),
                        (403, |m| (600..=649).contains(&m)),
                        (407, |m| (800..=849).contains(&m)),
                    ],
                };
                for &e in &ENERGIES {
                    let i = grid.partition_point(|&x| x < e).min(grid.len() - 1);
                    let mut line = format!("GROUP {name} {} {:.6e}", response.mt(), grid[i]);
                    for (g, f) in &groups {
                        let mut v = 0.0;
                        for (mt, h) in &d {
                            if f(*mt) {
                                let r = &reactions[mt];
                                let xs = i
                                    .checked_sub(r.threshold_idx)
                                    .and_then(|j| r.cross_section.as_slice().get(j))
                                    .copied()
                                    .unwrap_or(0.0);
                                v += xs * h[i];
                            }
                        }
                        line.push_str(&format!(" {g}:{v:.6e}"));
                    }
                    println!("{line}");
                }
            }
            for &e in &ENERGIES {
                let i = grid.partition_point(|&x| x < e).min(grid.len() - 1);
                let mut explained = 0.0;
                let mut parts: Vec<(f64, i32)> = Vec::new();
                for (mt, h) in &d {
                    let r = &reactions[mt];
                    let xs = i
                        .checked_sub(r.threshold_idx)
                        .and_then(|j| r.cross_section.as_slice().get(j))
                        .copied()
                        .unwrap_or(0.0);
                    explained += xs * h[i];
                    parts.push((xs * h[i], *mt));
                }
                parts.sort_by(|a, b| b.0.abs().total_cmp(&a.0.abs()));
                let top: Vec<String> = parts
                    .iter()
                    .take(
                        std::env::var("YAMC_VALIDATION_TOP")
                            .ok()
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(4),
                    )
                    .map(|(v, mt)| format!("{mt}:{:.3e}", v))
                    .collect();
                println!(
                    "  {:>10.4e} eV  stored {:>11.4e}  partials {:>11.4e}  ratio {:>8.5}  [{}]",
                    grid[i],
                    stored[i],
                    explained,
                    if stored[i] != 0.0 {
                        explained / stored[i]
                    } else {
                        f64::NAN
                    },
                    top.join(" ")
                );
            }
        }
    }
}
