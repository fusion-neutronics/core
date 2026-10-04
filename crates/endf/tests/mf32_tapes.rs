//! Every MF=32 section of six evaluated libraries, read to its SEND record.
//!
//! No fixture is small enough to cover what the libraries actually write in
//! MF=32, and the Python reader the goldens come from does not parse it at
//! all, so this is the check that the parser matches the tapes: every section
//! of ENDF/B-VIII.1, JEFF-4.0, JENDL-5.0, TENDL-2017, TENDL-2025 and
//! FENDL-3.2d is parsed, must consume exactly the lines up to its SEND
//! record, and the counts of what was read must match an independent survey
//! of the same tapes.
//!
//! The tapes are tens of gigabytes and live outside the repository, so this
//! is ignored by default. Point `ENDF_TAPES` at a directory laid out as
//! `nuclear_data_generation_scripts/data` is and run
//!
//! ```text
//! ENDF_TAPES=/path/to/data cargo test -p endf --release --test mf32_tapes -- --ignored
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use endf::mf::mf32::{parse_mf32, Covariance, Defect};
use endf::Reader;

/// A library, where it lives under `ENDF_TAPES`, and what the survey found.
struct Library {
    name: &'static str,
    dir: &'static str,
    files_with_mf32: usize,
    /// Ranges by `(LRU, LRF, LCOMP)`, with LCOMP -1 for an unresolved range.
    ranges: &'static [((i64, i64, i64), usize)],
    /// LCOMP=2 ranges by NDIGIT.
    ndigit: &'static [(i64, usize)],
    /// Files with an INTG line outside its matrix, and how many such lines.
    rows_outside: (usize, usize),
    /// Files with a negative variance, and how many each has.
    negative_variances: &'static [(&'static str, usize)],
    /// Non-zero correlations below the diagonal, over every compact matrix.
    correlations: usize,
}

const LIBRARIES: [Library; 6] = [
    Library {
        name: "ENDF/B-VIII.1",
        dir: "endfb-viii.1-endf/neutrons-version.VIII.1",
        files_with_mf32: 130,
        ranges: &[
            ((1, 2, 0), 20),
            ((1, 2, 1), 6),
            ((1, 2, 2), 62),
            ((1, 3, 1), 31),
            ((1, 3, 2), 1),
            ((1, 7, 1), 1),
            ((1, 7, 2), 9),
            ((2, 1, -1), 44),
        ],
        ndigit: &[(2, 65), (3, 5), (4, 1), (5, 1)],
        rows_outside: (0, 0),
        negative_variances: &[],
        correlations: 850_643,
    },
    Library {
        name: "JEFF-4.0",
        dir: "jeff-4.0-endf/neutron",
        files_with_mf32: 510,
        ranges: &[
            ((1, 2, 0), 17),
            ((1, 2, 1), 5),
            ((1, 2, 2), 439),
            ((1, 3, 1), 17),
            ((1, 3, 2), 20),
            ((1, 7, 2), 11),
            ((2, 1, -1), 436),
        ],
        ndigit: &[(2, 468), (4, 2)],
        rows_outside: (14, 45),
        negative_variances: &[("n_91-Pa-233g.jeff", 93)],
        correlations: 855_915,
    },
    Library {
        name: "JENDL-5.0",
        dir: "jendl-5.0-endf/neutron",
        files_with_mf32: 43,
        ranges: &[
            ((1, 2, 0), 30),
            ((1, 2, 1), 2),
            ((1, 3, 1), 9),
            ((1, 3, 2), 1),
            ((1, 7, 2), 1),
        ],
        ndigit: &[(2, 1), (4, 1)],
        rows_outside: (0, 0),
        negative_variances: &[],
        correlations: 8_518,
    },
    Library {
        name: "TENDL-2017",
        dir: "tendl-2017-endf/neutron_file",
        files_with_mf32: 10,
        ranges: &[((1, 3, 1), 1), ((1, 3, 2), 9), ((2, 1, -1), 1)],
        ndigit: &[(2, 8), (5, 1)],
        rows_outside: (0, 0),
        negative_variances: &[],
        correlations: 1_901_998,
    },
    Library {
        name: "TENDL-2025",
        dir: "tendl-2025-endf",
        files_with_mf32: 2850,
        ranges: &[((1, 2, 2), 2803), ((1, 3, 2), 47), ((2, 1, -1), 2715)],
        ndigit: &[(2, 2850)],
        rows_outside: (17, 49),
        negative_variances: &[],
        correlations: 737,
    },
    Library {
        name: "FENDL-3.2d",
        dir: "fendl-3.2d-endf/neutron",
        files_with_mf32: 35,
        ranges: &[
            ((1, 2, 1), 4),
            ((1, 2, 2), 7),
            ((1, 3, 1), 4),
            ((1, 3, 2), 16),
            ((1, 7, 2), 4),
            ((2, 1, -1), 11),
        ],
        ndigit: &[(2, 26), (4, 1)],
        rows_outside: (4, 19),
        negative_variances: &[],
        correlations: 634_206,
    },
];

/// What one file's MF=32 section held.
#[derive(Default)]
struct Summary {
    ranges: BTreeMap<(i64, i64, i64), usize>,
    ndigit: BTreeMap<i64, usize>,
    rows_outside: usize,
    negative_variances: usize,
    correlations: usize,
}

/// Incident-neutron evaluations under `dir`, the way the survey chose them.
fn evaluations(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).unwrap_or_else(|e| panic!("{}: {e}", d.display())) {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            if (name.starts_with("n-") || name.starts_with("n_"))
                && !name.ends_with(".md")
                && !name.ends_with(".json")
            {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

/// The MF=32 MT=151 lines of a file, or `None` if it has none, checking that
/// they are contiguous and closed by a SEND record.
fn mf32_section(text: &str) -> Result<Option<String>, String> {
    let control = |line: &str| {
        let f = |a: usize, b: usize| {
            line.get(a..b.min(line.len()))
                .unwrap_or("")
                .trim()
                .to_string()
        };
        (f(70, 72), f(72, 75))
    };
    let lines: Vec<&str> = text.lines().collect();
    let Some(start) = lines
        .iter()
        .position(|l| control(l) == ("32".to_string(), "151".to_string()))
    else {
        return Ok(None);
    };
    let mut end = start;
    while end < lines.len() && control(lines[end]) == ("32".to_string(), "151".to_string()) {
        end += 1;
    }
    match lines.get(end).map(|l| control(l)) {
        Some((mf, mt)) if mf == "32" && (mt == "0" || mt.is_empty()) => {}
        other => return Err(format!("MF=32 MT=151 is followed by {other:?}, not SEND")),
    }
    if lines[end..]
        .iter()
        .any(|l| control(l) == ("32".to_string(), "151".to_string()))
    {
        return Err("MF=32 MT=151 appears twice".into());
    }
    let mut body = String::new();
    for line in &lines[start..end] {
        body.push_str(line);
        body.push('\n');
    }
    Ok(Some(body))
}

/// A file's summary, `None` if it has no MF=32, or why it could not be read.
type Walked = Result<Option<Summary>, String>;

fn walk(path: &Path) -> Walked {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&bytes);
    let Some(body) = mf32_section(&text)? else {
        return Ok(None);
    };
    let mut reader = Reader::new(&body);
    let mf32 = parse_mf32(&mut reader).map_err(|e| e.to_string())?;
    if !reader.is_empty() {
        return Err(format!("{} lines left before SEND", reader.remaining()));
    }

    let mut s = Summary::default();
    for iso in &mf32.isotopes {
        for range in &iso.ranges {
            let lcomp = match &range.covariance {
                Covariance::Compatible(_) => 0,
                Covariance::General(_) | Covariance::GeneralRMatrix(_) => 1,
                Covariance::Compact(c) => {
                    *s.ndigit.entry(c.correlation.ndigit).or_default() += 1;
                    s.correlations += c.correlation.entries().count();
                    2
                }
                Covariance::CompactRMatrix(c) => {
                    *s.ndigit.entry(c.correlation.ndigit).or_default() += 1;
                    s.correlations += c.correlation.entries().count();
                    2
                }
                Covariance::Unresolved(_) => -1,
            };
            *s.ranges.entry((range.lru, range.lrf, lcomp)).or_default() += 1;
        }
    }
    for d in &mf32.defects {
        match d {
            Defect::CorrelationRowOutsideMatrix { .. } => s.rows_outside += 1,
            Defect::NegativeVariance { .. } => s.negative_variances += 1,
        }
    }
    Ok(Some(s))
}

#[test]
#[ignore = "reads tens of GB of local tapes; set ENDF_TAPES and run with --ignored"]
fn every_mf32_section_on_the_local_tapes_is_read_to_send() {
    let root = PathBuf::from(
        std::env::var_os("ENDF_TAPES")
            .expect("set ENDF_TAPES to the directory holding the six libraries"),
    );
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut failures = Vec::new();
    let mut mismatches = Vec::new();

    for lib in &LIBRARIES {
        let files = evaluations(&root.join(lib.dir));
        let next = AtomicUsize::new(0);
        let results: Mutex<Vec<(PathBuf, Walked)>> = Mutex::new(Vec::new());
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(path) = files.get(i) else { break };
                    let r = walk(path);
                    results.lock().unwrap().push((path.clone(), r));
                });
            }
        });

        let mut with_mf32 = 0usize;
        let mut ranges: BTreeMap<(i64, i64, i64), usize> = BTreeMap::new();
        let mut ndigit: BTreeMap<i64, usize> = BTreeMap::new();
        let mut rows_outside = (0usize, 0usize);
        let mut negative = Vec::new();
        let mut correlations = 0usize;
        for (path, r) in results.into_inner().unwrap() {
            match r {
                Ok(None) => {}
                Ok(Some(s)) => {
                    with_mf32 += 1;
                    for (k, v) in s.ranges {
                        *ranges.entry(k).or_default() += v;
                    }
                    for (k, v) in s.ndigit {
                        *ndigit.entry(k).or_default() += v;
                    }
                    correlations += s.correlations;
                    if s.rows_outside > 0 {
                        rows_outside.0 += 1;
                        rows_outside.1 += s.rows_outside;
                    }
                    if s.negative_variances > 0 {
                        let name = path.file_name().unwrap().to_string_lossy().to_string();
                        negative.push((name, s.negative_variances));
                    }
                }
                Err(e) => failures.push(format!("{}: {e}", path.display())),
            }
        }
        negative.sort();
        println!(
            "{}: {} files, {with_mf32} with MF=32; ranges {ranges:?}; NDIGIT {ndigit:?}; \
             {correlations} correlations; INTG lines outside the matrix: {} in {} files; \
             negative variances: {negative:?}",
            lib.name,
            files.len(),
            rows_outside.1,
            rows_outside.0,
        );

        let expected_ranges: BTreeMap<_, _> = lib.ranges.iter().copied().collect();
        let expected_ndigit: BTreeMap<_, _> = lib.ndigit.iter().copied().collect();
        if with_mf32 != lib.files_with_mf32 {
            mismatches.push(format!(
                "{}: {with_mf32} files with MF=32, the survey found {}",
                lib.name, lib.files_with_mf32
            ));
        }
        if ranges != expected_ranges {
            mismatches.push(format!(
                "{}: ranges {ranges:?}, expected {expected_ranges:?}",
                lib.name
            ));
        }
        if ndigit != expected_ndigit {
            mismatches.push(format!(
                "{}: NDIGIT {ndigit:?}, expected {expected_ndigit:?}",
                lib.name
            ));
        }
        if rows_outside != lib.rows_outside {
            mismatches.push(format!(
                "{}: (files, lines) with INTG lines outside the matrix {rows_outside:?}, \
                 the survey found {:?}",
                lib.name, lib.rows_outside
            ));
        }
        if correlations != lib.correlations {
            mismatches.push(format!(
                "{}: {correlations} non-zero correlations, the survey found {}",
                lib.name, lib.correlations
            ));
        }
        let expected_negative: Vec<(String, usize)> = lib
            .negative_variances
            .iter()
            .map(|&(f, n)| (f.to_string(), n))
            .collect();
        if negative != expected_negative {
            mismatches.push(format!(
                "{}: negative variances {negative:?}, expected {expected_negative:?}",
                lib.name
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{} files failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}

/// Every resolved MF=32 range of ENDF/B-VIII.1 and JEFF-4.0 is matched to its
/// MF=2 range and read into one parameter covariance, with no MF=2 parameter
/// given to two MF=32 ones. ENDF/B-VIII.1 matches every resonance exactly,
/// including the resonances of one spin MF=2 lists twice at one energy
/// (Ne22, Ti49, Cr52, Pb208), which only their widths tell apart. JEFF-4.0
/// has 63 resonances whose MF=32 energy differs from MF=2's within the
/// tolerance (Xe135 rounded, U236 from another parameter set) and 253 that
/// MF=2 does not list at all.
#[test]
#[ignore = "reads GB of local tapes; set ENDF_TAPES and run with --ignored"]
fn every_resolved_mf32_range_reads_into_a_parameter_covariance() {
    use endf::resonance_covariance::resolved_covariances;
    let root = PathBuf::from(
        std::env::var_os("ENDF_TAPES")
            .expect("set ENDF_TAPES to the directory holding the libraries"),
    );
    for (dir, files, ranges, approximate, unmatched) in [
        ("endfb-viii.1-endf/neutrons-version.VIII.1", 130, 130, 0, 0),
        ("jeff-4.0-endf/neutron", 510, 509, 63, 253),
    ] {
        let (mut f, mut r, mut a, mut u) = (0, 0, 0, 0);
        for path in evaluations(&root.join(dir)) {
            let Ok(m) = endf::material::Material::from_file(&path) else {
                continue;
            };
            let (Some(mf2), Some(mf32)) = (m.mf2(), m.mf32()) else {
                continue;
            };
            let v = resolved_covariances(mf2, mf32)
                .unwrap_or_else(|e| panic!("{}: {e:?}", path.display()));
            for c in &v {
                let mut seen = std::collections::HashSet::new();
                for p in &c.parameters {
                    assert!(
                        seen.insert(format!("{:?} {:?}", p.location, p.quantity)),
                        "{}: {:?} {:?} appears twice",
                        path.display(),
                        p.location,
                        p.quantity
                    );
                }
            }
            f += 1;
            r += v.len();
            a += v.iter().map(|c| c.approximate).sum::<usize>();
            u += v.iter().map(|c| c.unmatched.len()).sum::<usize>();
        }
        assert_eq!(
            (f, r, a, u),
            (files, ranges, approximate, unmatched),
            "{dir}"
        );
    }
}

/// ENDF/B-VIII.1 Pb208's elastic and capture group variances, in barns
/// squared (relative times the group cross section squared, since ERRORR
/// relativizes by the cross section with its MF=3 background), against
/// NJOY 2016 ERRORR's resonance-parameter contribution on nine groups over
/// the resolved range. The scattering radius uncertainty follows ERRORR's
/// reading: one parameter moving the p-wave radius by 0.0027 and the d- and
/// f-wave radii by 0.027. ERRORR takes the radius step at a whole standard
/// deviation, which above 300 keV is no longer small, so the two highest
/// groups' elastic agree less closely.
#[test]
#[ignore = "reads a local tape; set ENDF_TAPES and run with --ignored"]
fn pb208_group_covariance_matches_errorr() {
    use endf::resonance::ReichMooreRange;
    use endf::resonance_covariance::{group_covariance, resolved_covariances};
    let root = PathBuf::from(std::env::var_os("ENDF_TAPES").expect("set ENDF_TAPES"));
    let m = endf::material::Material::from_file(
        root.join("endfb-viii.1-endf/neutrons-version.VIII.1/n-082_Pb_208.endf"),
    )
    .unwrap();
    let cov = &resolved_covariances(m.mf2().unwrap(), m.mf32().unwrap()).unwrap()[0];
    assert_eq!(cov.radius_steps, vec![0.0, 0.0027, 0.027, 0.027]);
    let rm = ReichMooreRange::new(&m.mf2().unwrap().isotopes[0].ranges[0]).unwrap();
    let edges = [1e-5, 1.0, 1e2, 1e3, 1e4, 5e4, 1e5, 3e5, 1e6, 1.5e6];
    let g = group_covariance(cov, &rm, &edges).unwrap();
    // (relative variance, group cross section) as ERRORR prints them.
    let elastic = [
        (5.080e-4, 1.1301e1),
        (5.080e-4, 1.1300e1),
        (5.075e-4, 1.1293e1),
        (5.031e-4, 1.1230e1),
        (4.773e-4, 1.0860e1),
        (3.016e-4, 1.2204e1),
        (2.892e-4, 8.8376e0),
        (3.453e-4, 5.8307e0),
        (2.503e-3, 5.0234e0),
    ];
    let capture = [
        (1.163e-3, 2.0455e-3),
        (9.196e-4, 1.6761e-5),
        (3.568e-5, 1.2941e-5),
        (5.416e-7, 3.4620e-5),
        (2.869e-8, 1.6769e-4),
        (2.856e-4, 4.3271e-4),
        (1.546e-4, 9.2848e-4),
        (2.719e-5, 1.5737e-3),
        (2.288e-6, 3.8768e-4),
    ];
    for h in 0..9 {
        for (a, (rel, xs), tolerance) in [
            (0, elastic[h], if h >= 7 { 0.1 } else { 0.01 }),
            (1, capture[h], 0.05),
        ] {
            let ours = g.get(a, h, a, h) * g.cross_sections[a][h].powi(2);
            let njoy = rel * xs * xs;
            assert!(
                (ours / njoy - 1.0).abs() < tolerance,
                "reaction {a} group {h}: {ours:e} against ERRORR's {njoy:e}"
            );
        }
    }
}
