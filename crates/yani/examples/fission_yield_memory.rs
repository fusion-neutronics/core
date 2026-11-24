//! Memory probe for loading a fission-yields subsection.
//!
//! Run as:
//! `cargo run --release -p yani --example fission_yield_memory -- DECAY_DIR FPY_DIR`
//!
//! Prints the resident set size before and after the load, and its peak,
//! read from `/proc/self/status` (Linux only).

fn status(field: &str) -> String {
    std::fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with(field))
        .map(|l| l.split_whitespace().skip(1).collect::<Vec<_>>().join(" "))
        .unwrap_or_else(|| "n/a".into())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (decay, fpy) = (&args[1], &args[2]);
    println!("before: VmRSS {}", status("VmRSS"));
    let start = std::time::Instant::now();
    let loaded = yani::load_chain_parts(decay, None, Some(fpy), &[]).expect("load");
    let elapsed = start.elapsed();
    let (mut parents, mut points, mut products) = (0usize, 0usize, 0usize);
    for nuc in loaded.chain.values() {
        if let Some(set) = &nuc.fission_yields {
            parents += 1;
            points += set.yields.len();
            for y in &set.yields {
                products += y.products.len()
                    + y.independent.as_ref().map_or(0, |e| e.products.len())
                    + y.cumulative.as_ref().map_or(0, |e| e.products.len());
            }
        }
    }
    println!("loaded in {elapsed:.1?}: {parents} parents, {points} yield sets, {products} product entries");
    println!(
        "after:  VmRSS {}  VmHWM {}",
        status("VmRSS"),
        status("VmHWM")
    );
}
