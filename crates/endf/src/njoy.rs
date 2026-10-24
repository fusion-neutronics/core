//! Driving NJOY: writing its input deck and running it.
//!
//! NJOY turns an evaluation into the processed forms a transport code reads:
//! a pointwise ENDF file, or an ACE table at a given temperature. It takes its
//! instructions on standard input and refers to files by unit number, so the
//! work here is composing the deck and staging the tapes.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{Error, Result};
use crate::material::Material;

/// The library each NLIB value names.
pub const LIBRARY: [(i64, &str); 19] = [
    (0, "ENDF/B"),
    (1, "ENDF/A"),
    (2, "JEFF"),
    (3, "EFF"),
    (4, "ENDF/B High Energy"),
    (5, "CENDL"),
    (6, "JENDL"),
    (17, "TENDL"),
    (18, "ROSFOND"),
    (21, "SG-23"),
    (31, "INDL/V"),
    (32, "INDL/A"),
    (33, "FENDL"),
    (34, "IRDF"),
    (35, "BROND"),
    (36, "INGDB-90"),
    (37, "FENDL/A"),
    (38, "IAEA/PD"),
    (41, "BROND"),
];

/// The library an NLIB value names, or `"Unknown"`.
pub fn library_name(nlib: i64) -> &'static str {
    LIBRARY
        .iter()
        .find(|&&(k, _)| k == nlib)
        .map_or("Unknown", |&(_, name)| name)
}

/// Where each NJOY module's output tape should go.
///
/// `None` means the module is not run at all; `Some(None)` runs it and writes
/// its tape under `output_dir` with the module's own name; `Some(Some(path))`
/// runs it and writes the tape there.
pub type ModuleOutput = Option<Option<PathBuf>>;

/// What to ask NJOY for.
#[derive(Debug, Clone)]
pub struct AceOptions {
    /// Temperatures in kelvin. Empty means room temperature alone.
    pub temperatures: Vec<f64>,
    /// Fractional tolerance for reconstruction and broadening.
    pub error: f64,
    pub pendf: ModuleOutput,
    pub broadr: ModuleOutput,
    pub heatr: ModuleOutput,
    pub gaspr: ModuleOutput,
    pub purr: ModuleOutput,
    pub acer: ModuleOutput,
    /// Where the ACE file goes. Defaults to `output_dir/ace`.
    pub ace: Option<PathBuf>,
    /// Where the xsdir goes. Defaults to sitting beside the ACE file.
    pub xsdir: Option<PathBuf>,
    /// Where the tapes go by default. NJOY also works in a directory of its
    /// own inside this one while it runs (see [`run`]).
    pub output_dir: PathBuf,
    /// Whether ACER thins and smooths the elastic and capture cross sections
    /// at low energy.
    pub smoothing: bool,
    /// The NJOY executable.
    pub njoy_exec: String,
    /// Where to keep a copy of the input deck, for a person to read.
    pub input_filename: Option<PathBuf>,
}

impl Default for AceOptions {
    /// Every module on, room temperature, output in the current directory:
    /// the same defaults the Python package has.
    fn default() -> Self {
        AceOptions {
            temperatures: Vec::new(),
            error: 0.001,
            pendf: None,
            broadr: Some(None),
            heatr: Some(None),
            gaspr: Some(None),
            purr: Some(None),
            acer: Some(None),
            ace: None,
            xsdir: None,
            output_dir: PathBuf::from("."),
            smoothing: true,
            njoy_exec: "njoy".to_string(),
            input_filename: None,
        }
    }
}

impl AceOptions {
    /// The options for a pointwise ENDF file: every module but `reconr` off.
    pub fn pendf_only(pendf: impl Into<PathBuf>) -> AceOptions {
        AceOptions {
            pendf: Some(Some(pendf.into())),
            broadr: None,
            heatr: None,
            gaspr: None,
            purr: None,
            acer: None,
            ..Default::default()
        }
    }
}

/// The input deck, and where its tapes come from and go.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Deck {
    /// The commands, as NJOY reads them on standard input.
    pub commands: String,
    /// Input files, by the unit number NJOY calls them.
    pub tapein: BTreeMap<i32, PathBuf>,
    /// Output files, likewise.
    pub tapeout: BTreeMap<i32, PathBuf>,
    /// The temperatures ACER was asked for, in the order it writes them.
    pub temperatures: Vec<f64>,
    /// Whether the target is metastable, which ACER does not record in the
    /// ZAID it writes.
    pub isomeric_state: i64,
}

/// Compose the NJOY input deck for an evaluation.
///
/// Separated from running it so the deck can be inspected, written out or
/// tested without NJOY installed.
pub fn ace_deck(filename: &Path, material: &Material, options: &AceOptions) -> Result<Deck> {
    let metadata = material.mf1_mt451().ok_or(Error::Unsupported {
        what: "an evaluation with no MF=1 MT=451 to process",
    })?;
    let mat = material.mat;
    // The Python reader leaves ZSYMAM as the file wrote it, padding and all,
    // and NJOY takes it that way.
    let zsymam = metadata.zsymam.clone().unwrap_or_default();
    let library = format!(
        "{}-{}.{}",
        library_name(metadata.nlib),
        metadata.nver,
        metadata.lrel
    );

    let temperatures = if options.temperatures.is_empty() {
        vec![293.6]
    } else {
        options.temperatures.clone()
    };
    let temps = temperatures
        .iter()
        .map(|t| crate::data::python_float_str(*t))
        .collect::<Vec<_>>()
        .join(" ");
    let error = options.error;

    let out = &options.output_dir;
    // `Some(None)` means the module runs and names its own tape.
    let named = |module: &ModuleOutput, default: &str| -> Option<PathBuf> {
        module
            .as_ref()
            .map(|path| path.clone().unwrap_or_else(|| out.join(default)))
    };

    // NJOY refers to files by unit number, and each module reads the tape the
    // one before it wrote, so the numbers thread through in sequence.
    let (nendf, npendf) = (20, 21);
    let mut deck = Deck {
        tapein: BTreeMap::from([(nendf, filename.to_path_buf())]),
        temperatures: temperatures.clone(),
        isomeric_state: metadata.liso,
        ..Default::default()
    };

    if let Some(path) = named(&options.pendf, "pendf") {
        deck.tapeout.insert(npendf, path);
    }

    deck.commands.push_str(&format!(
        "
reconr / %%%%%%%%%%%%%%%%%%% Reconstruct XS for neutrons %%%%%%%%%%%%%%%%%%%%%%%
{nendf} {npendf}
'{library} PENDF for {zsymam}'/
{mat} 2/
{error}/ err
'{library}: {zsymam}'/
'Processed by NJOY'/
0/
"
    ));
    let mut nlast = npendf;

    if let Some(path) = named(&options.broadr, "broadr") {
        let nbroadr = nlast + 1;
        deck.tapeout.insert(nbroadr, path);
        let num_temp = temperatures.len();
        deck.commands.push_str(&format!(
            "
broadr / %%%%%%%%%%%%%%%%%%%%%%% Doppler broaden XS %%%%%%%%%%%%%%%%%%%%%%%%%%%%
{nendf} {npendf} {nbroadr}
{mat} {num_temp} 0 0 0. /
{error}/ errthn
{temps}
0/
"
        ));
        nlast = nbroadr;
    }

    if options.heatr.is_some() {
        // Two runs: one where photons deposit their energy locally, one where
        // they carry it away.
        let nheatr_in = nlast;
        let nheatr_local = nheatr_in + 1;
        let local = match &options.heatr {
            Some(Some(path)) => {
                let mut name = path.clone().into_os_string();
                name.push("_local");
                PathBuf::from(name)
            }
            _ => out.join("heatr_local"),
        };
        deck.tapeout.insert(nheatr_local, local);
        deck.commands.push_str(&format!(
            "
heatr / %%%%%%%%%%%%%%%%% Add heating kerma (local photons) %%%%%%%%%%%%%%%%%%%%
{nendf} {nheatr_in} {nheatr_local} /
{mat} 4 0 0 1 /
302 318 402 444 /
"
        ));

        let nheatr = nheatr_local + 1;
        deck.tapeout
            .insert(nheatr, named(&options.heatr, "heatr").expect("heatr is on"));
        deck.commands.push_str(&format!(
            "
heatr / %%%%%%%%%%%%%%%%%%%%%%%%% Add heating kerma %%%%%%%%%%%%%%%%%%%%%%%%%%%%
{nendf} {nheatr_in} {nheatr} /
{mat} 4 0 0 0 /
302 318 402 444 /
"
        ));
        nlast = nheatr;
    }

    if let Some(path) = named(&options.gaspr, "gaspr") {
        let ngaspr_in = nlast;
        let ngaspr = ngaspr_in + 1;
        deck.tapeout.insert(ngaspr, path);
        deck.commands.push_str(&format!(
            "
gaspr / %%%%%%%%%%%%%%%%%%%%%%%%% Add gas production %%%%%%%%%%%%%%%%%%%%%%%%%%%
{nendf} {ngaspr_in} {ngaspr} /
"
        ));
        nlast = ngaspr;
    }

    if let Some(path) = named(&options.purr, "purr") {
        let npurr_in = nlast;
        let npurr = npurr_in + 1;
        deck.tapeout.insert(npurr, path);
        let num_temp = temperatures.len();
        deck.commands.push_str(&format!(
            "
purr / %%%%%%%%%%%%%%%%%%%%%%%% Add probability tables %%%%%%%%%%%%%%%%%%%%%%%%%
{nendf} {npurr_in} {npurr} /
{mat} {num_temp} 1 20 64 /
{temps}
1.e10
0/
"
        ));
        nlast = npurr;
    }

    if options.acer.is_some() {
        let nacer_in = nlast;
        let ismooth = i32::from(options.smoothing);
        for (i, &temperature) in temperatures.iter().enumerate() {
            // One ACER run per temperature, each writing its own ACE and
            // xsdir tape.
            let nace = nacer_in + 1 + 2 * i as i32;
            let ndir = nace + 1;
            let ext = format!("{:02}", i + 1);
            let t = crate::data::python_float_str(temperature);
            deck.commands.push_str(&format!(
                "
acer / %%%%%%%%%%%%%%%%%%%%%%%% Write out in ACE format %%%%%%%%%%%%%%%%%%%%%%%%
{nendf} {nacer_in} 0 {nace} {ndir}
1 0 1 .{ext} /
'{library}: {zsymam} at {t}'/
{mat} {t}
1 1 {ismooth}/
/
"
            ));
            deck.tapeout
                .insert(nace, out.join(format!("ace_{temperature:.1}")));
            deck.tapeout
                .insert(ndir, out.join(format!("xsdir_{temperature:.1}")));
        }
    }

    deck.commands.push_str("stop\n");
    Ok(deck)
}

/// Run NJOY with the given deck.
///
/// NJOY reads and writes its tapes in the directory it runs in, as `tape20`,
/// `tape21` and so on, so it runs in a [`Scratch`] directory made inside
/// `scratch`: the inputs are copied in, the outputs are moved out afterwards,
/// and the directory goes whether NJOY succeeded or not. Its scratch units, 10
/// to 19, are not named tapes: the Fortran runtime makes them in `TMPDIR`, or
/// `/tmp` without it, so `TMPDIR` is set to that directory for NJOY too.
///
/// `scratch` should be on the disk the output goes to, which is why
/// [`make_ace`] passes its output directory. One evaluation's tapes can run to
/// hundreds of MB, and the system temporary directory is often a small
/// RAM-backed tmpfs or under a per-user quota. Filling it does not fail with a
/// message that names the cause: NJOY exits with an error or leaves a tape cut
/// short, and the largest evaluations are the ones that fail.
pub fn run(
    deck: &Deck,
    njoy_exec: &str,
    input_filename: Option<&Path>,
    scratch: &Path,
) -> Result<String> {
    if let Some(path) = input_filename {
        std::fs::write(path, &deck.commands)?;
    }

    let work = Scratch::new_in(scratch, "run")?;
    run_in(deck, njoy_exec, work.path())
}

fn run_in(deck: &Deck, njoy_exec: &str, work: &Path) -> Result<String> {
    for (unit, filename) in &deck.tapein {
        std::fs::copy(filename, work.join(format!("tape{unit}")))?;
    }

    let mut child = std::process::Command::new(njoy_exec)
        .current_dir(work)
        // Where gfortran makes the files behind NJOY's scratch units. It
        // unlinks each as it opens it, so they take space while NJOY runs and
        // none is ever left behind.
        .env("TMPDIR", work)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()?;
    child
        .stdin
        .as_mut()
        .ok_or(Error::Unsupported {
            what: "an NJOY process with no standard input",
        })?
        .write_all(deck.commands.as_bytes())?;

    let output = child.wait_with_output()?;
    let log = String::from_utf8_lossy(&output.stdout).into_owned();
    if !output.status.success() {
        // Standard error as well. NJOY prints its own errors on standard
        // output, but the Fortran runtime prints on standard error, and a
        // scratch disk that filled up is named only there: standard output
        // stops at the banner and the module that died.
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stderr = stderr.trim();
        let what = if stderr.is_empty() {
            format!("NJOY exited with {}: {log}", output.status)
        } else {
            format!(
                "NJOY exited with {}: {log}\nstandard error:\n{stderr}",
                output.status
            )
        };
        return Err(Error::BadAceTable { what });
    }

    for (unit, filename) in &deck.tapeout {
        let written = work.join(format!("tape{unit}"));
        if written.is_file() {
            if let Some(parent) = filename.parent() {
                std::fs::create_dir_all(parent)?;
            }
            // Rename fails across filesystems, so fall back to a copy.
            if std::fs::rename(&written, filename).is_err() {
                std::fs::copy(&written, filename)?;
                std::fs::remove_file(&written)?;
            }
        }
    }
    Ok(log)
}

/// Generate an ACE file from an evaluation, running NJOY to do it.
///
/// The per-temperature ACE tapes are concatenated into one file and one xsdir,
/// as the Python package does.
pub fn make_ace(filename: &Path, material: &Material, options: &AceOptions) -> Result<()> {
    if !options.output_dir.is_dir() {
        return Err(Error::Unsupported {
            what: "an output directory that is not a directory",
        });
    }
    let deck = ace_deck(filename, material, options)?;
    run(
        &deck,
        &options.njoy_exec,
        options.input_filename.as_deref(),
        &options.output_dir,
    )?;

    if options.acer.is_none() {
        return Ok(());
    }
    let ace = match (&options.ace, &options.acer) {
        (Some(path), _) => path.clone(),
        (None, Some(Some(path))) => path.clone(),
        _ => options.output_dir.join("ace"),
    };
    let xsdir = options.xsdir.clone().unwrap_or_else(|| {
        ace.parent()
            .unwrap_or(Path::new("."))
            .join("xsdir")
            .to_path_buf()
    });

    let mut ace_text = String::new();
    let mut xsdir_text = String::new();
    for &temperature in &deck.temperatures {
        let per_temperature = options.output_dir.join(format!("ace_{temperature:.1}"));
        let text = std::fs::read_to_string(&per_temperature)?;
        ace_text.push_str(&metastable_zaid(&text, deck.isomeric_state));
        xsdir_text.push_str(&std::fs::read_to_string(
            options.output_dir.join(format!("xsdir_{temperature:.1}")),
        )?);
    }
    std::fs::write(&ace, ace_text)?;
    std::fs::write(&xsdir, xsdir_text)?;

    for &temperature in &deck.temperatures {
        let _ = std::fs::remove_file(options.output_dir.join(format!("ace_{temperature:.1}")));
        let _ = std::fs::remove_file(options.output_dir.join(format!("xsdir_{temperature:.1}")));
    }
    Ok(())
}

/// Mark a metastable target in the ZAID an ACE table opens with.
///
/// ACER does not record the isomeric state, so 400 is added to the mass
/// number the way MCNP libraries do. A first digit above two would carry, so
/// those are left alone, the same guard the Python package has.
fn metastable_zaid(text: &str, isomeric_state: i64) -> String {
    if isomeric_state == 0 {
        return text.to_string();
    }
    let bytes = text.as_bytes();
    let Some(digit) = bytes.get(3).and_then(|b| (*b as char).to_digit(10)) else {
        return text.to_string();
    };
    if digit > 2 {
        return text.to_string();
    }
    format!("{}{}{}", &text[..3], digit + 4, &text[4..])
}

/// A working directory for NJOY, made inside a directory the caller names and
/// removed when dropped.
///
/// The caller names it so it can sit beside the output, not in the system
/// temporary directory (see [`run`] for why). The name starts with a dot, so
/// `ls` and shell globs pass it by, and ends in a number, so nothing that
/// looks for `*.arrow` folders takes it for a nuclide. It carries a label,
/// the process id and a counter, so conversions running at the same time, in
/// this process or another, each get their own. A name that is already taken
/// is skipped, never reused, so a directory that another process made is
/// never written into or deleted.
///
/// Dropping it removes it: on success, on an early return and on a panic.
/// Only a process that is killed leaves one behind.
///
/// Hand-rolled because the crate has no dependencies.
#[derive(Debug)]
pub struct Scratch {
    path: PathBuf,
}

impl Scratch {
    /// Make a fresh directory inside `parent`, which must exist. `label` goes
    /// into the name, so a directory left by a killed process says what it
    /// was for.
    pub fn new_in(parent: &Path, label: &str) -> Result<Scratch> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Scratch::numbered_by(parent, label, &COUNTER)
    }

    /// [`Scratch::new_in`] with the counter passed in, so a test knows which
    /// names it will try.
    fn numbered_by(parent: &Path, label: &str, counter: &AtomicU64) -> Result<Scratch> {
        // Absolute, since NJOY is handed it as TMPDIR from inside it, where a
        // relative path would mean somewhere else.
        let parent = std::path::absolute(parent).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!(
                    "could not make NJOY's working directory in {}: {e}",
                    parent.display()
                ),
            )
        })?;
        let pid = std::process::id();
        for _ in 0..64 {
            let n = counter.fetch_add(1, Ordering::Relaxed);
            let path = parent.join(format!(".njoy-{label}-{pid}-{n}"));
            match std::fs::create_dir(&path) {
                Ok(()) => return Ok(Scratch { path }),
                // Left by an earlier process that had the same id.
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => {
                    return Err(std::io::Error::new(
                        e.kind(),
                        format!(
                            "could not make NJOY's working directory {}: {e}",
                            path.display()
                        ),
                    )
                    .into());
                }
            }
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!(
                "64 names for NJOY's working directory in {} were taken \
                 (.njoy-{label}-{pid}-*). Killed runs leave these behind; \
                 remove the ones no running process is using",
                parent.display()
            ),
        )
        .into())
    }

    /// The directory, as an absolute path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        // There is no one to tell from a destructor. A directory that cannot
        // be removed stays, under a name that says what made it.
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const AM244: &[u8] = include_bytes!("../fixtures/n-095_Am_244.endf.xz");
    /// The deck the Python `make_ace` composes for the same evaluation and the
    /// same options, captured with its `run` stubbed out.
    const REFERENCE_DECK: &[u8] = include_bytes!("../tests/reference/njoy-deck.txt.xz");

    fn am244() -> Material {
        Material::from_str(&crate::testdata::text(AM244)).unwrap()
    }

    #[test]
    fn composes_the_same_deck_as_the_python_package() {
        let options = AceOptions {
            temperatures: vec![293.6, 900.0],
            ..Default::default()
        };
        let deck = ace_deck(Path::new("tests/n-095_Am_244.endf.xz"), &am244(), &options).unwrap();
        assert_eq!(deck.commands, crate::testdata::text(REFERENCE_DECK));
    }

    #[test]
    fn the_tapes_thread_through_in_sequence() {
        let options = AceOptions {
            temperatures: vec![293.6, 900.0],
            ..Default::default()
        };
        let deck = ace_deck(Path::new("in.endf"), &am244(), &options).unwrap();

        // One input, on unit 20.
        assert_eq!(
            deck.tapein.keys().copied().collect::<Vec<_>>(),
            [20],
            "the evaluation goes in on tape20"
        );
        // reconr writes 21 but pendf is off, so 21 is not kept. Then broadr,
        // the two heatr runs, gaspr, purr, and an ACE and xsdir per
        // temperature.
        assert_eq!(
            deck.tapeout.keys().copied().collect::<Vec<_>>(),
            [22, 23, 24, 25, 26, 27, 28, 29, 30]
        );
        assert!(deck.tapeout[&27].ends_with("ace_293.6"));
        assert!(deck.tapeout[&28].ends_with("xsdir_293.6"));
        assert!(deck.tapeout[&29].ends_with("ace_900.0"));
        assert_eq!(deck.temperatures, [293.6, 900.0]);
    }

    #[test]
    fn turning_a_module_off_closes_the_gap_it_leaves() {
        let options = AceOptions {
            heatr: None,
            gaspr: None,
            ..Default::default()
        };
        let deck = ace_deck(Path::new("in.endf"), &am244(), &options).unwrap();
        assert!(!deck.commands.contains("heatr"));
        assert!(!deck.commands.contains("gaspr"));
        // purr now reads broadr's tape directly, and the numbering has no hole.
        assert!(deck.commands.contains("20 22 23 /"));
        assert_eq!(
            deck.tapeout.keys().copied().collect::<Vec<_>>(),
            [22, 23, 24, 25]
        );
    }

    #[test]
    fn a_pointwise_run_is_reconr_alone() {
        let options = AceOptions::pendf_only("out.pendf");
        let deck = ace_deck(Path::new("in.endf"), &am244(), &options).unwrap();
        assert!(deck.commands.contains("reconr"));
        for module in ["broadr", "heatr", "gaspr", "purr", "acer"] {
            assert!(!deck.commands.contains(module), "{module} should be off");
        }
        assert_eq!(deck.tapeout[&21], PathBuf::from("out.pendf"));
        assert!(deck.commands.ends_with("stop\n"));
    }

    #[test]
    fn smoothing_shows_up_in_the_acer_card() {
        let on = ace_deck(Path::new("f"), &am244(), &AceOptions::default()).unwrap();
        assert!(on.commands.contains("1 1 1/"));
        let off = ace_deck(
            Path::new("f"),
            &am244(),
            &AceOptions {
                smoothing: false,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(off.commands.contains("1 1 0/"));
    }

    #[test]
    fn a_metastable_target_is_marked_in_the_zaid() {
        // ACER writes no isomeric state, so 400 goes onto the mass number.
        // The header right-justifies the ZAID in ten columns, so the mass
        // number's first digit is the fourth character.
        assert_eq!(&metastable_zaid(" 95242.00c rest", 1)[..6], " 95642");
        // The ground state is left alone.
        assert_eq!(&metastable_zaid(" 95242.00c rest", 0)[..6], " 95242");
        // A first digit above two would carry, so it is left alone.
        assert_eq!(&metastable_zaid(" 95942.00c rest", 1)[..6], " 95942");
    }

    #[test]
    fn names_the_libraries_nlib_numbers() {
        assert_eq!(library_name(0), "ENDF/B");
        assert_eq!(library_name(2), "JEFF");
        assert_eq!(library_name(17), "TENDL");
        assert_eq!(library_name(99), "Unknown");
    }

    /// A fresh directory for one test, under the system temporary directory.
    fn fresh_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("endf-njoy-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// What a directory holds, sorted.
    fn entries(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    /// A stand-in for NJOY, so where it runs and what it leaves can be checked
    /// without NJOY installed: a shell script that writes the directory it was
    /// started in to `ran_in` beside itself and the one its `TMPDIR` names to
    /// `tmpdir`, saves its deck where it runs, then runs `body` there.
    ///
    /// Written by `sh`, not by this process. A process another test starts
    /// inherits, for a moment, every file this one has open, and running the
    /// script while a copy of it is open for writing fails with "Text file
    /// busy".
    #[cfg(unix)]
    fn stand_in(dir: &Path, body: &str) -> PathBuf {
        let script = dir.join("njoy");
        let text = format!(
            "#!/bin/sh\npwd -P > '{}'\n(cd \"${{TMPDIR:-/tmp}}\" && pwd -P) > '{}'\n\
             cat > deck\n{body}\n",
            dir.join("ran_in").display(),
            dir.join("tmpdir").display()
        );
        let mut sh = std::process::Command::new("sh")
            .args(["-c", "cat > \"$1\" && chmod 755 \"$1\"", "sh"])
            .arg(&script)
            .stdin(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        sh.stdin.take().unwrap().write_all(text.as_bytes()).unwrap();
        assert!(sh.wait().unwrap().success());
        script
    }

    /// What the stand-in under `root` wrote down: where it ran (`ran_in`) or
    /// where its TMPDIR pointed (`tmpdir`), with symlinks resolved.
    #[cfg(unix)]
    fn recorded(root: &Path, what: &str) -> PathBuf {
        let text = std::fs::read_to_string(root.join(what)).expect("NJOY was run");
        PathBuf::from(text.trim_end())
    }

    /// `path`, which is absolute, as a path from the current directory.
    #[cfg(unix)]
    fn relative_to_here(path: &Path) -> PathBuf {
        let here = std::env::current_dir().unwrap();
        let mut relative: PathBuf = here.components().skip(1).map(|_| "..").collect();
        relative.push(path.strip_prefix("/").unwrap());
        relative
    }

    /// The evaluation on disk, an empty output directory, and the options for
    /// a pointwise run of `njoy` that writes its one tape there.
    #[cfg(unix)]
    fn pendf_into(root: &Path, njoy: &Path) -> (PathBuf, PathBuf, AceOptions) {
        let evaluation = root.join("in.endf");
        std::fs::write(&evaluation, crate::testdata::text(AM244)).unwrap();
        let out = root.join("out");
        std::fs::create_dir(&out).unwrap();
        let options = AceOptions {
            output_dir: out.clone(),
            njoy_exec: njoy.to_str().unwrap().to_string(),
            ..AceOptions::pendf_only(out.join("pendf"))
        };
        (evaluation, out, options)
    }

    #[cfg(unix)]
    #[test]
    fn njoy_runs_in_a_directory_of_its_own_inside_the_output_directory() {
        let root = fresh_dir("runs-inside");
        let njoy = stand_in(&root, "echo pointwise > tape21");
        let (evaluation, out, options) = pendf_into(&root, &njoy);

        make_ace(&evaluation, &am244(), &options).unwrap();

        let ran_in = recorded(&root, "ran_in");
        assert_eq!(
            ran_in.parent(),
            Some(std::fs::canonicalize(&out).unwrap().as_path()),
            "NJOY ran in {}, not in a directory of its own inside {}",
            ran_in.display(),
            out.display()
        );
        let name = ran_in.file_name().unwrap().to_string_lossy();
        assert!(name.starts_with(".njoy-run-"), "{name}");
        // Its scratch units go there too, not to /tmp.
        assert_eq!(recorded(&root, "tmpdir"), ran_in);
        // The tape the deck asked for came out, and the working directory went.
        assert_eq!(entries(&out), ["pendf"]);
        assert_eq!(
            std::fs::read_to_string(out.join("pendf")).unwrap(),
            "pointwise\n"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn a_relative_output_directory_reaches_njoy_as_an_absolute_tmpdir() {
        let root = fresh_dir("relative");
        // NJOY reads TMPDIR from inside its own directory, where a relative
        // one names a directory that is not there.
        let njoy = stand_in(
            &root,
            "case \"$TMPDIR\" in /*) ;; *) exit 9 ;; esac\necho pointwise > tape21",
        );
        let (evaluation, out, mut options) = pendf_into(&root, &njoy);
        options.output_dir = relative_to_here(&out);

        make_ace(&evaluation, &am244(), &options).unwrap();

        assert_eq!(recorded(&root, "tmpdir"), recorded(&root, "ran_in"));
        assert_eq!(entries(&out), ["pendf"]);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn a_failed_njoy_run_leaves_nothing_in_the_output_directory() {
        let root = fresh_dir("fails");
        // As NJOY dies when its scratch disk fills: the module on standard
        // output, the cause from the Fortran runtime on standard error.
        let njoy = stand_in(
            &root,
            "echo partial > tape21\necho ' reconr...'\n\
             echo 'Fortran runtime error: No space left on device' >&2\nexit 2",
        );
        let (evaluation, out, options) = pendf_into(&root, &njoy);

        let err = make_ace(&evaluation, &am244(), &options)
            .unwrap_err()
            .to_string();

        assert!(err.contains("NJOY exited with"), "{err}");
        assert!(err.contains("reconr..."), "{err}");
        assert!(
            err.contains("Fortran runtime error: No space left on device"),
            "{err}"
        );
        // It ran in the output directory, and neither its working directory
        // nor the tape it cut short is left there.
        let ran_in = recorded(&root, "ran_in");
        assert!(
            ran_in.starts_with(std::fs::canonicalize(&out).unwrap()),
            "NJOY ran in {}",
            ran_in.display()
        );
        assert_eq!(entries(&out), Vec::<String>::new());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_scratch_directory_is_hidden_its_own_and_gone_once_dropped() {
        let root = fresh_dir("scratch");
        let a = Scratch::new_in(&root, "Fe56").unwrap();
        let b = Scratch::new_in(&root, "Fe56").unwrap();
        assert_ne!(a.path(), b.path(), "two conversions shared a directory");
        for scratch in [&a, &b] {
            assert!(scratch.path().is_dir());
            assert!(scratch.path().is_absolute());
            assert_eq!(scratch.path().parent(), Some(root.as_path()));
            let name = scratch.path().file_name().unwrap().to_string_lossy();
            assert!(name.starts_with(".njoy-Fe56-"), "{name}");
        }

        // Dropped with a tape in it, it goes, and takes nothing else with it.
        std::fs::write(a.path().join("tape20"), "evaluation").unwrap();
        let b_name = b.path().file_name().unwrap().to_string_lossy().into_owned();
        drop(a);
        assert_eq!(entries(&root), [b_name]);
        drop(b);
        assert_eq!(entries(&root), Vec::<String>::new());

        // A parent that does not exist is an error that names the path.
        let err = Scratch::new_in(&root.join("absent"), "x")
            .unwrap_err()
            .to_string();
        assert!(err.contains("absent"), "{err}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_name_already_taken_is_skipped_and_left_alone() {
        let root = fresh_dir("taken");
        let pid = std::process::id();
        let name = |n: u64| root.join(format!(".njoy-x-{pid}-{n}"));
        // What a killed run that had this process id would leave.
        std::fs::create_dir(name(0)).unwrap();
        std::fs::write(name(0).join("tape20"), "left by a killed run").unwrap();

        let fresh = Scratch::numbered_by(&root, "x", &AtomicU64::new(0)).unwrap();
        assert_eq!(fresh.path(), name(1));
        drop(fresh);
        assert_eq!(
            std::fs::read_to_string(name(0).join("tape20")).unwrap(),
            "left by a killed run"
        );

        // With every name it would try taken, it says so rather than reuse one.
        for n in 1..64 {
            std::fs::create_dir(name(n)).unwrap();
        }
        let err = Scratch::numbered_by(&root, "x", &AtomicU64::new(0))
            .unwrap_err()
            .to_string();
        assert!(err.contains("were taken"), "{err}");
        assert_eq!(entries(&root).len(), 64);
        std::fs::remove_dir_all(&root).unwrap();
    }
}
