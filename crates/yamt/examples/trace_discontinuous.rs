//! Research harness: walk rays through a mesh with discontinuous tracking and
//! report the volume sequence each ray sees.
//!
//! TOL=<contact tolerance cm> cargo run -p yamt --release --example trace_discontinuous -- <mesh.arrow> [n_rays] [verbose]

use yamt::{add_vacuum_boundary, build_topology, read_arrow_mesh, MeshGeometry};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = std::path::Path::new(&args[1]);
    let n: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1000);
    let verbose = args.get(3).is_some();
    let ox: f64 = std::env::var("OX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let mut data = read_arrow_mesh(path).unwrap();
    add_vacuum_boundary(&mut data, 1.0);
    let mut geom = MeshGeometry::from_topology(build_topology(data).unwrap());
    if let Some(tol) = std::env::var("TOL")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
    {
        let t0 = std::time::Instant::now();
        geom.set_contact_tolerance(Some(tol));
        eprintln!("contacts built in {:.3} s", t0.elapsed().as_secs_f64());
    }
    let ic = geom.implicit_complement();
    eprintln!(
        "volumes={} surfaces={} contact_tolerance={:?} contact entries={} ic={ic}",
        geom.num_volumes(),
        geom.num_surfaces(),
        geom.contact_tolerance(),
        geom.contacts().len()
    );
    let mut seqs: std::collections::BTreeMap<Vec<u32>, usize> = Default::default();
    // Deterministic quasi-random directions (golden-angle spiral).
    for k in 0..n {
        let z = 1.0 - 2.0 * (k as f64 + 0.5) / n as f64;
        let r = (1.0 - z * z).sqrt();
        let phi = k as f64 * 2.399963229728653;
        let dir = [r * phi.cos(), r * phi.sin(), z];
        let mut pos = [ox, 0.0, 0.0];
        let mut vol = geom.find_volume(pos);
        let mut seq = vec![vol];
        for _step in 0..50 {
            match geom.next_crossing(vol, pos, dir) {
                None => {
                    seq.push(u32::MAX);
                    break;
                }
                Some((t, surf, beyond)) => {
                    let next = match beyond {
                        yamt::Beyond::Volume(v) => v,
                        yamt::Beyond::Relocate(tri) => {
                            let x = [
                                pos[0] + t * dir[0],
                                pos[1] + t * dir[1],
                                pos[2] + t * dir[2],
                            ];
                            Some(geom.relocate(vol, tri, x, dir))
                        }
                    };
                    if verbose && k < 20 {
                        let x = [
                            pos[0] + t * dir[0],
                            pos[1] + t * dir[1],
                            pos[2] + t * dir[2],
                        ];
                        let rr = (x[0] * x[0] + x[1] * x[1]).sqrt();
                        eprintln!(
                            "  ray {k} vol {vol} t={t:.6} surf={surf} next={next:?} r_xy={rr:.5} z={:.4}",
                            x[2]
                        );
                    }
                    let bc = geom.boundary_condition(surf);
                    if bc == yamt::BoundaryCondition::Vacuum {
                        seq.push(u32::MAX - 1);
                        break;
                    }
                    for i in 0..3 {
                        pos[i] += (t + 1e-8) * dir[i];
                    }
                    match next {
                        Some(v) => {
                            vol = v;
                            seq.push(v);
                        }
                        None => {
                            seq.push(u32::MAX - 2);
                            break;
                        }
                    }
                }
            }
        }
        *seqs.entry(seq).or_default() += 1;
    }
    let mut v: Vec<_> = seqs.into_iter().collect();
    v.sort_by_key(|entry| std::cmp::Reverse(entry.1));
    for (seq, c) in v.iter().take(15) {
        let s: Vec<String> = seq
            .iter()
            .map(|&x| match x {
                u32::MAX => "NONE".into(),
                x if x == u32::MAX - 1 => "VAC".into(),
                x if x == u32::MAX - 2 => "NOVOL".into(),
                x if x == ic => "IC".into(),
                x => format!("{x}"),
            })
            .collect();
        println!("{c:6}  {}", s.join(" > "));
    }
}
