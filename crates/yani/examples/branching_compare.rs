//! Scratch: load one decay subsection with two branching subsections and
//! compare the curves the run would use, channel by channel.
//! cargo run --release -p yani --example branching_compare -- DECAY NEW OLD

use std::collections::BTreeMap;
use std::path::Path;

fn curves(
    decay: &Path,
    branching: &Path,
) -> BTreeMap<(String, String, String), (Vec<f64>, Vec<f64>, Vec<String>)> {
    let (_, table) = yani::chain_arrow::parse_chain_parts(decay, None, None, &[("x", branching)])
        .expect("loads");
    let mut out = BTreeMap::new();
    for (parent, kinds) in table.curves() {
        for (kind, list) in kinds {
            for c in list {
                let routes = c.states.iter().map(|s| s.level_route.clone()).collect();
                out.insert(
                    (parent.clone(), kind.clone(), c.target.clone()),
                    (c.energy.clone(), c.values.clone(), routes),
                );
            }
        }
    }
    out
}

fn at(e: &[f64], v: &[f64], x: f64) -> f64 {
    if e.is_empty() || x < e[0] || x > e[e.len() - 1] {
        return 0.0;
    }
    let i = e.partition_point(|&p| p <= x).clamp(1, e.len() - 1);
    let (x0, x1) = (e[i - 1], e[i]);
    if x1 == x0 {
        return v[i];
    }
    v[i - 1] + (x - x0) / (x1 - x0) * (v[i] - v[i - 1])
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let decay = Path::new(&args[1]);
    let new = curves(decay, Path::new(&args[2]));
    let old = curves(decay, Path::new(&args[3]));
    let only_new: Vec<_> = new.keys().filter(|k| !old.contains_key(*k)).collect();
    let only_old: Vec<_> = old.keys().filter(|k| !new.contains_key(*k)).collect();
    let mut worst = 0.0f64;
    let mut differ = 0;
    for (k, (e, v, _)) in &old {
        let Some((e2, v2, _)) = new.get(k) else {
            continue;
        };
        let mut grid: Vec<f64> = e.iter().chain(e2).copied().collect();
        grid.sort_by(f64::total_cmp);
        grid.dedup();
        let d = grid
            .iter()
            .map(|&x| {
                let (a, b) = (at(e, v, x), at(e2, v2, x));
                (a - b).abs() / a.abs().max(b.abs()).max(1e-30)
            })
            .fold(0.0, f64::max);
        if d > 1e-12 {
            differ += 1;
            if differ <= 5 {
                println!("differs {k:?}: {d:.3e}");
            }
        }
        worst = worst.max(d);
    }
    println!(
        "curves: new {} old {}; only new {}; only old {}; shared differing {differ}; worst rel {worst:.3e}",
        new.len(),
        old.len(),
        only_new.len(),
        only_old.len()
    );
    for k in only_new.iter().take(8) {
        println!("  only new {k:?} routes {:?}", new[*k].2);
    }
    for k in only_old.iter().take(8) {
        println!("  only old {k:?} routes {:?}", old[*k].2);
    }
}
