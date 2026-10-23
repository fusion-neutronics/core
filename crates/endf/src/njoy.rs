//! Driving NJOY: checking which NJOY it is, writing its input deck and
//! running it.
//!
//! NJOY turns an evaluation into the processed forms a transport code reads:
//! a pointwise ENDF file, or an ACE table at a given temperature. It takes its
//! instructions on standard input and refers to files by unit number, so the
//! work here is composing the deck and staging the tapes. Which NJOY runs
//! matters as much as the deck, so [`require`] refuses one of the wrong
//! flavour or too old for a library before any work starts.

use std::collections::BTreeMap;
use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};

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
/// The tapes are staged in a temporary directory as `tape20`, `tape21` and so
/// on, since that is how NJOY names them, and the outputs are moved out
/// afterwards.
pub fn run(deck: &Deck, njoy_exec: &str, input_filename: Option<&Path>) -> Result<String> {
    if let Some(path) = input_filename {
        std::fs::write(path, &deck.commands)?;
    }

    let tmpdir = temp_dir()?;
    let result = run_in(deck, njoy_exec, &tmpdir);
    // The temporary directory goes whether NJOY succeeded or not.
    let _ = std::fs::remove_dir_all(&tmpdir);
    result
}

fn run_in(deck: &Deck, njoy_exec: &str, tmpdir: &Path) -> Result<String> {
    for (unit, filename) in &deck.tapein {
        std::fs::copy(filename, tmpdir.join(format!("tape{unit}")))?;
    }

    let mut child = std::process::Command::new(njoy_exec)
        .current_dir(tmpdir)
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
        return Err(Error::BadAceTable {
            what: format!("NJOY exited with {}: {log}", output.status),
        });
    }

    for (unit, filename) in &deck.tapeout {
        let written = tmpdir.join(format!("tape{unit}"));
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
    run(&deck, &options.njoy_exec, options.input_filename.as_deref())?;

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

/// An NJOY release, as its banner reports it.
///
/// NJOY2016 numbers its releases 2016.0, 2016.1 and onwards, so the part after
/// the point is a whole number and not a decimal fraction: 2016.9 (2017) is
/// older than 2016.80 (2026).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Version {
    /// 2016 for NJOY2016.
    pub year: u32,
    /// The release within that line: 80 for 2016.80.
    pub minor: u32,
    /// Whether this is the IAEA-NDS fork, which ends its version in `+`: its
    /// `src/vers.f90` holds `vers='2016.77+'` where upstream's holds a bare
    /// `'2016.78'`. That suffix is the only thing the binary itself exposes,
    /// so it is what an NJOY is judged on.
    pub iaea_fork: bool,
}

impl Version {
    /// Whether this is `needed`'s flavour, at its minimum or later.
    fn meets(self, needed: Flavour) -> bool {
        let minimum = needed.minimum();
        self.iaea_fork == minimum.iaea_fork
            && (self.year, self.minor) >= (minimum.year, minimum.minor)
    }
}

/// As NJOY prints it: `2016.80`, or `2016.77+` for the fork.
impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fork = if self.iaea_fork { "+" } else { "" };
        write!(f, "{}.{}{fork}", self.year, self.minor)
    }
}

/// Which NJOY a library is processed with.
///
/// Getting this wrong is quiet: the conversion succeeds and the numbers are
/// subtly wrong, which is why [`require`] judges the executable rather than
/// trusting whatever `njoy` is first on PATH.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Flavour {
    /// LANL's NJOY2016 (github.com/njoy/NJOY2016), for every library but
    /// FENDL.
    Upstream,
    /// The IAEA Nuclear Data Section's fork (github.com/IAEA-NDS/NJOY2016),
    /// which FENDL is processed with. Upstream does not reproduce the official
    /// release: measured on FENDL-3.2d against LANL's 2016.78, the fork changes
    /// the URR probability tables by up to 100% (La139), and MT 301 heating
    /// and MT 444 damage by up to 77% below about 1 keV.
    IaeaFork,
}

impl Flavour {
    /// The oldest release of this flavour a library may be processed with.
    ///
    /// Upstream needs 2016.80. Older releases sum the harmonic numbers in
    /// samm.f90's `bigeta` with integer division (njoy/NJOY2016#415, fixed by
    /// #417), which gives wrong Coulomb penetrabilities in R-matrix limited
    /// evaluations with charged-particle channels: Ca40 (n,p) at 925 keV came
    /// out at 3.1e-3 b instead of 5.6e-7 b. The fork needs 2016.77+, which
    /// carries 2016.77's fix to the URR background.
    pub const fn minimum(self) -> Version {
        match self {
            Flavour::Upstream => Version {
                year: 2016,
                minor: 80,
                iaea_fork: false,
            },
            Flavour::IaeaFork => Version {
                year: 2016,
                minor: 77,
                iaea_fork: true,
            },
        }
    }

    /// How to get an NJOY of this flavour, which a refusal ends with.
    pub(crate) fn how_to_get(self) -> String {
        match self {
            Flavour::Upstream => format!(
                "Install it with\n    {PIP_INSTALL}\nor build tag {} of \
                 https://github.com/njoy/NJOY2016, and pass its path as njoy_exec \
                 if it is not the njoy on PATH",
                self.minimum()
            ),
            Flavour::IaeaFork => IAEA_BUILD.to_string(),
        }
    }
}

/// Installs upstream NJOY2016: the njoy2016 wheel puts an `njoy` beside the
/// Python interpreter. `--upgrade` because an older wheel is the usual reason
/// to be refused.
pub(crate) const PIP_INSTALL: &str =
    "pip install --upgrade njoy2016 --extra-index-url https://shimwell.github.io/wheels";

/// Builds the IAEA-NDS fork, which has no wheel.
const IAEA_BUILD: &str = concat!(
    "Build it (this needs git, cmake and gfortran) with\n",
    "    git clone https://github.com/IAEA-NDS/NJOY2016\n",
    "    cmake -S NJOY2016 -B NJOY2016/build -DCMAKE_BUILD_TYPE=Release\n",
    "    cmake --build NJOY2016/build\n",
    "and pass the full path to NJOY2016/build/njoy as njoy_exec",
);

/// The version of an NJOY executable, read from the banner it prints.
///
/// NJOY2016 has no `--version`: run with it, NJOY prints its banner,
/// ` njoy 2016.80  08Oct26` and then the date and time, and stops at the end
/// of its empty input deck with a non-zero exit. So the exit status says
/// nothing and is not checked, and the version is the first
/// `njoy <year>.<minor>` printed, with the `+` the IAEA fork appends. Ported
/// from `njoy_version` in nuclear_data_generation_scripts' ensure_njoy.sh.
///
/// `njoy_exec` is a name looked up on PATH or a path, a relative one taken
/// from the current directory. Fails with [`Error::NjoyMissing`] if it cannot
/// be started and [`Error::NjoyNoVersion`] if it prints no version.
pub fn version(njoy_exec: &str) -> Result<Version> {
    let program = program(njoy_exec)?;
    // NJOY writes its listing to a file called `output` in the directory it
    // runs in before it reads anything, so it runs in a scratch directory
    // rather than over a file of that name in the caller's.
    let scratch = temp_dir()?;
    let ran = std::process::Command::new(program)
        .arg("--version")
        .current_dir(&scratch)
        // NJOY reads its deck from standard input. Closed, it stops at once
        // instead of reading the caller's.
        .stdin(std::process::Stdio::null())
        .output();
    let _ = std::fs::remove_dir_all(&scratch);
    let ran = ran.map_err(|source| Error::NjoyMissing {
        njoy_exec: njoy_exec.to_string(),
        source,
    })?;

    // The banner goes to standard output. Standard error is read as well, as
    // ensure_njoy.sh reads both.
    let stdout = String::from_utf8_lossy(&ran.stdout);
    let stderr = String::from_utf8_lossy(&ran.stderr);
    parse_version(&stdout)
        .or_else(|| parse_version(&stderr))
        .ok_or_else(|| Error::NjoyNoVersion {
            njoy_exec: njoy_exec.to_string(),
            status: ran.status,
            output: [stdout.trim(), stderr.trim()]
                .into_iter()
                .filter(|text| !text.is_empty())
                .collect::<Vec<_>>()
                .join("\n"),
        })
}

/// Check that `njoy_exec` is the NJOY a library needs, and return its version
/// for the record.
///
/// Refuses one of the other flavour, or older than [`Flavour::minimum`], with
/// [`Error::NjoyUnsuitable`], whose message names the executable, the version
/// found, the version needed and how to get it. Ported from `verify` in
/// ensure_njoy.sh, and fails as [`version`] does when there is no version to
/// judge.
pub fn require(njoy_exec: &str, needed: Flavour) -> Result<Version> {
    let found = version(njoy_exec)?;
    if found.meets(needed) {
        Ok(found)
    } else {
        Err(Error::NjoyUnsuitable {
            njoy_exec: njoy_exec.to_string(),
            found,
            needed,
        })
    }
}

/// The first `njoy <year>.<minor>` in `text`, with a `+` if one follows: the
/// match ensure_njoy.sh makes with `njoy +[0-9]{4}\.[0-9]+\+?`.
fn parse_version(text: &str) -> Option<Version> {
    text.match_indices("njoy").find_map(|(at, word)| {
        let rest = text[at + word.len()..]
            .strip_prefix(' ')?
            .trim_start_matches(' ');
        let (year, rest) = rest.split_at_checked(4)?;
        let rest = rest.strip_prefix('.')?;
        let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
        if digits == 0 || !year.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        Some(Version {
            year: year.parse().ok()?,
            minor: rest[..digits].parse().ok()?,
            iaea_fork: rest[digits..].starts_with('+'),
        })
    })
}

/// `njoy_exec` as a program to start in another working directory.
///
/// A path with a directory part is made absolute against the current
/// directory, since a child given a working directory of its own resolves a
/// relative one there instead (Linux does). A bare name is left to the PATH
/// search.
fn program(njoy_exec: &str) -> Result<PathBuf> {
    let path = Path::new(njoy_exec);
    if path.components().count() > 1 {
        Ok(std::path::absolute(path)?)
    } else {
        Ok(path.to_path_buf())
    }
}

/// A fresh temporary directory.
///
/// Hand-rolled because the crate has no dependencies: the name comes from the
/// process id and a counter, and `create_dir` fails rather than reuses if the
/// name is somehow taken.
fn temp_dir() -> Result<PathBuf> {
    use std::sync::atomic::{AtomicU32, Ordering};
    static COUNTER: AtomicU32 = AtomicU32::new(0);

    let base = std::env::temp_dir();
    for _ in 0..64 {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = base.join(format!("endf-njoy-{}-{n}", std::process::id()));
        match std::fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Err(Error::Unsupported {
        what: "finding an unused temporary directory name",
    })
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

    // What `njoy --version` prints on standard output, captured from real
    // builds: the banner, then NJOY stops at the end of its empty deck.

    /// LANL's 2016.80, built from its tag (ab27c64) by ensure_njoy.sh.
    const UPSTREAM_2016_80: &str = concat!(
        "\n njoy 2016.80  08Oct26                                       10/23/26 19:56:22\n",
        " *****************************************************************************\n",
    );
    /// LANL's 2016.78, from before the bigeta fix.
    const UPSTREAM_2016_78: &str = concat!(
        "\n njoy 2016.78  03Feb25                                       10/23/26 19:56:22\n",
        " *****************************************************************************\n",
    );
    /// The IAEA-NDS fork at 3835cff, built by ensure_njoy.sh. The `+` fills
    /// the version's eight columns, so one space follows it rather than two.
    const IAEA_FORK: &str = concat!(
        "\n njoy 2016.77+ 23Oct24                                       10/23/26 19:56:22\n",
        " *****************************************************************************\n",
    );
    /// The njoy2016 2016.80.post8 wheel on the extra index, the NJOY users are
    /// told to install. Its binary holds the tag's `vers` and `vday`
    /// (`2016.80 08Oct26 `) and prints them through the same format, so its
    /// banner is the tag's line.
    const WHEEL_2016_80: &str = UPSTREAM_2016_80;
    /// The 2016.79.post8 wheel the extra index served before it.
    const WHEEL_2016_79: &str = concat!(
        "\n njoy 2016.79  18Mar25                                       10/23/26 19:56:22\n",
        " *****************************************************************************\n",
    );

    fn v(year: u32, minor: u32, iaea_fork: bool) -> Version {
        Version {
            year,
            minor,
            iaea_fork,
        }
    }

    #[test]
    fn reads_the_banners_real_builds_print() {
        for (banner, want, upstream, fork) in [
            (UPSTREAM_2016_80, v(2016, 80, false), true, false),
            (UPSTREAM_2016_78, v(2016, 78, false), false, false),
            (IAEA_FORK, v(2016, 77, true), false, true),
            // Passes for every library but FENDL as it is.
            (WHEEL_2016_80, v(2016, 80, false), true, false),
            (WHEEL_2016_79, v(2016, 79, false), false, false),
        ] {
            let found = parse_version(banner).unwrap_or_else(|| panic!("no version in {banner:?}"));
            assert_eq!(found, want, "{banner:?}");
            assert_eq!(
                found.meets(Flavour::Upstream),
                upstream,
                "{found} for upstream"
            );
            assert_eq!(found.meets(Flavour::IaeaFork), fork, "{found} for the fork");
        }
    }

    #[test]
    fn prints_a_version_as_njoy_does() {
        assert_eq!(v(2016, 80, false).to_string(), "2016.80");
        assert_eq!(v(2016, 77, true).to_string(), "2016.77+");
        assert_eq!(v(2016, 8, false).to_string(), "2016.8");
        assert_eq!(Flavour::Upstream.minimum().to_string(), "2016.80");
        assert_eq!(Flavour::IaeaFork.minimum().to_string(), "2016.77+");
    }

    #[test]
    fn finds_nothing_in_what_is_not_a_banner() {
        for text in [
            "",
            "njoy",
            "njoy 2016",
            "njoy 2016.",
            "njoy2016.80",
            "NJOY 2016.80",
            "njoy 16.80",
            "njoy 20166.80",
            "njoy: 2016.80",
            "true (GNU coreutils) 9.4\nWritten by Jim Meyering.\n",
        ] {
            assert_eq!(parse_version(text), None, "{text:?}");
        }
        // An "njoy" that is not a version does not end the search.
        assert_eq!(
            parse_version(&format!("/usr/bin/njoy: started\n{UPSTREAM_2016_80}")),
            Some(v(2016, 80, false))
        );
    }

    #[test]
    fn upstream_needs_2016_80_and_the_fork_2016_77() {
        for (found, upstream, fork) in [
            (v(2016, 80, false), true, false),
            (v(2016, 81, false), true, false),
            (v(2017, 0, false), true, false),
            (v(2016, 79, false), false, false),
            // 2016.9 (2017) and 2016.8 are older than 2016.80, which a
            // comparison of decimals would get wrong.
            (v(2016, 9, false), false, false),
            (v(2016, 8, false), false, false),
            (v(2016, 77, true), false, true),
            (v(2016, 80, true), false, true),
            (v(2016, 76, true), false, false),
        ] {
            assert_eq!(
                found.meets(Flavour::Upstream),
                upstream,
                "{found} for upstream"
            );
            assert_eq!(found.meets(Flavour::IaeaFork), fork, "{found} for the fork");
        }
    }

    #[test]
    fn a_refusal_names_the_binary_both_versions_and_how_to_get_njoy() {
        let refusal = |found, needed| {
            Error::NjoyUnsuitable {
                njoy_exec: "/opt/njoy/bin/njoy".to_string(),
                found,
                needed,
            }
            .to_string()
        };
        let upstream = ["pip install --upgrade njoy2016", "github.com/njoy/NJOY2016"];
        let fork = [
            "git clone https://github.com/IAEA-NDS/NJOY2016",
            "cmake --build",
        ];
        for (message, reported, minimum, how) in [
            (
                refusal(v(2016, 78, false), Flavour::Upstream),
                "2016.78",
                "2016.80",
                upstream,
            ),
            (
                refusal(v(2016, 77, true), Flavour::Upstream),
                "2016.77+",
                "2016.80",
                upstream,
            ),
            (
                refusal(v(2016, 80, false), Flavour::IaeaFork),
                "2016.80",
                "2016.77+",
                fork,
            ),
            (
                refusal(v(2016, 76, true), Flavour::IaeaFork),
                "2016.76+",
                "2016.77+",
                fork,
            ),
        ] {
            for part in ["\"/opt/njoy/bin/njoy\"", reported, minimum, how[0], how[1]] {
                assert!(message.contains(part), "{part:?} is not in: {message}");
            }
        }
        assert!(refusal(v(2016, 77, true), Flavour::Upstream).contains("the IAEA-NDS fork,"));
        assert!(refusal(v(2016, 80, false), Flavour::IaeaFork).contains("upstream NJOY2016,"));
    }

    #[test]
    fn a_missing_executable_is_named_with_how_to_get_njoy() {
        let missing = std::env::temp_dir()
            .join(format!("endf-no-njoy-here-{}", std::process::id()))
            .join("njoy");
        let missing = missing.to_str().expect("a UTF-8 temporary directory");
        let error = version(missing).unwrap_err();
        assert!(matches!(error, Error::NjoyMissing { .. }), "{error:?}");
        let message = error.to_string();
        assert!(message.contains(missing), "{message}");
        assert!(message.contains(PIP_INSTALL), "{message}");
        assert!(message.contains("IAEA-NDS"), "{message}");
        // A refusal fails the same way when there is nothing to judge.
        assert!(matches!(
            require(missing, Flavour::Upstream),
            Err(Error::NjoyMissing { .. })
        ));
    }

    #[cfg(unix)]
    #[test]
    fn an_executable_that_is_not_njoy_is_refused() {
        // GNU `true --version` prints coreutils' version and the BSD one
        // prints nothing; neither is a banner.
        let error = version("true").unwrap_err();
        assert!(matches!(error, Error::NjoyNoVersion { .. }), "{error:?}");
        assert!(error
            .to_string()
            .starts_with("\"true\" printed no NJOY version"));
    }

    #[test]
    fn a_relative_path_is_kept_relative_to_the_callers_directory() {
        // NJOY is started in a scratch directory, so a relative path would
        // otherwise be looked for there.
        assert_eq!(program("njoy").unwrap(), PathBuf::from("njoy"));
        let relative = program("bin/njoy").unwrap();
        assert!(relative.is_absolute(), "{relative:?}");
        assert_eq!(
            relative,
            std::env::current_dir().unwrap().join("bin").join("njoy")
        );
    }

    /// The NJOY on PATH, when this machine has one. Announces a loud skip
    /// otherwise, as the NJOY-gated tests in yamc-convert do. PATH is searched
    /// rather than NJOY run to find out, since running it here would leave its
    /// listing in this directory, which is what this checks the probe does
    /// not do.
    #[test]
    fn reads_the_version_of_the_njoy_on_path() {
        let on_path = std::env::var_os("PATH").is_some_and(|path| {
            std::env::split_paths(&path)
                .any(|dir| dir.join("njoy").is_file() || dir.join("njoy.exe").is_file())
        });
        if !on_path {
            eprintln!("SKIP: no njoy on PATH; this test checked nothing");
            return;
        }

        // NJOY rewrites its listing wherever it runs, so an `output` here that
        // appears or changes would be the probe's.
        let listing = || {
            Path::new("output")
                .metadata()
                .and_then(|m| m.modified())
                .ok()
        };
        let before = listing();
        let found = version("njoy").expect("the njoy on PATH reports its version");
        assert_eq!(found.year, 2016, "{found} is not an NJOY2016");
        // The gate judges what the probe read.
        for needed in [Flavour::Upstream, Flavour::IaeaFork] {
            assert_eq!(
                require("njoy", needed).is_ok(),
                found.meets(needed),
                "{found} for {needed:?}"
            );
        }
        assert_eq!(
            listing(),
            before,
            "the probe wrote NJOY's listing into the working directory"
        );
    }
}
