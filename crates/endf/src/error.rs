//! Errors raised while reading an ENDF-6 file.

use std::fmt;

use crate::njoy::{Flavour, Version};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    /// A record was expected but the section ended first.
    UnexpectedEof {
        /// What the reader was trying to read.
        expected: &'static str,
    },
    /// A control field (MAT/MF/MT) could not be read as an integer.
    BadControlField {
        line: String,
    },
    /// An INTG record used an NDIGIT value the format does not define.
    BadNdigit {
        ndigit: i64,
    },
    /// An ACE table that could not be read.
    BadAceTable {
        what: String,
    },
    /// An interpolation scheme the reader does not know.
    BadInterpolation {
        name: String,
    },
    /// Two lists that had to be the same length were not.
    Mismatched {
        what: &'static str,
    },
    /// A name that is not a nuclide in GNDS convention.
    BadNuclideName {
        name: String,
    },
    /// A chemical symbol with no atomic number.
    UnknownElement {
        symbol: String,
    },
    /// A representation the reader does not implement, matching where the
    /// Python reader raises `NotImplementedError`.
    Unsupported {
        what: &'static str,
    },
    /// An argument outside what a computation accepts.
    InvalidArgument {
        what: String,
    },
    /// An NJOY executable that could not be started: not found on PATH or at
    /// its path, or not executable.
    NjoyMissing {
        njoy_exec: String,
        source: std::io::Error,
    },
    /// An executable that ran but printed no NJOY version.
    NjoyNoVersion {
        njoy_exec: String,
        status: std::process::ExitStatus,
        /// What it printed, standard output then standard error.
        output: String,
    },
    /// An NJOY of the other flavour from the one a library needs, or older
    /// than that flavour's minimum.
    NjoyUnsuitable {
        njoy_exec: String,
        found: Version,
        needed: Flavour,
    },
    Io(std::io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::UnexpectedEof { expected } => {
                write!(f, "ENDF section ended while reading {expected}")
            }
            Error::BadControlField { line } => {
                write!(
                    f,
                    "could not read MAT/MF/MT control fields from line: {line:?}"
                )
            }
            Error::BadNdigit { ndigit } => {
                write!(f, "INTG record has NDIGIT={ndigit}, expected 2 through 6")
            }
            Error::BadAceTable { what } => write!(f, "could not read the ACE table: {what}"),
            Error::BadInterpolation { name } => {
                write!(
                    f,
                    "{name:?} is not an interpolation scheme this reader knows"
                )
            }
            Error::Mismatched { what } => write!(f, "{what} must match"),
            Error::BadNuclideName { name } => {
                write!(f, "{name:?} is not a nuclide name in GNDS format")
            }
            Error::UnknownElement { symbol } => {
                write!(f, "{symbol:?} is not a recognised element symbol")
            }
            Error::Unsupported { what } => {
                write!(f, "this reader does not implement {what}")
            }
            Error::InvalidArgument { what } => write!(f, "{what}"),
            Error::NjoyMissing { njoy_exec, source } => write!(
                f,
                "could not run NJOY \"{njoy_exec}\": {source}. Install NJOY2016 with\n    {}\n\
                 (FENDL needs the IAEA-NDS fork instead, https://github.com/IAEA-NDS/NJOY2016), \
                 or pass njoy_exec the path to an njoy executable",
                crate::njoy::PIP_INSTALL
            ),
            Error::NjoyNoVersion {
                njoy_exec,
                status,
                output,
            } => {
                write!(
                    f,
                    "\"{njoy_exec}\" printed no NJOY version when run with --version \
                     ({status}), where NJOY2016 prints a banner such as \
                     \" njoy 2016.80  08Oct26\""
                )?;
                if output.is_empty() {
                    write!(f, "; it printed nothing")
                } else {
                    write!(f, "; it printed:\n{output}")
                }
            }
            Error::NjoyUnsuitable {
                njoy_exec,
                found,
                needed,
            } => {
                let minimum = needed.minimum();
                match (needed, found.iaea_fork) {
                    (Flavour::Upstream, true) => write!(
                        f,
                        "\"{njoy_exec}\" is NJOY {found}, the IAEA-NDS fork, but this library \
                         needs upstream NJOY2016 {minimum} or later"
                    )?,
                    (Flavour::Upstream, false) => write!(
                        f,
                        "\"{njoy_exec}\" is NJOY {found}, but this library needs {minimum} or \
                         later: older releases get the Coulomb penetrabilities of R-matrix \
                         limited evaluations wrong (njoy/NJOY2016#415)"
                    )?,
                    (Flavour::IaeaFork, false) => write!(
                        f,
                        "\"{njoy_exec}\" is NJOY {found}, upstream NJOY2016, but this library \
                         needs the IAEA-NDS fork, {minimum} or later, because upstream does not \
                         reproduce its official processing"
                    )?,
                    (Flavour::IaeaFork, true) => write!(
                        f,
                        "\"{njoy_exec}\" is NJOY {found}, but this library needs the IAEA-NDS \
                         fork at {minimum} or later"
                    )?,
                }
                write!(f, ". {}", needed.how_to_get())
            }
            Error::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::NjoyMissing { source, .. } => Some(source),
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
