//! Auxiliary photon data: Compton profiles and bremsstrahlung.
//!
//! None of this is ENDF. The photoatomic sublibrary does not carry Compton
//! profiles, bremsstrahlung cross sections or the density effect correction,
//! so a transport code that needs them takes them from separate tabulations:
//! Biggs, Mendelsohn and Mann (1975); Seltzer and Berger (1986); Sternheimer,
//! Berger and Seltzer (1984). The Python package attaches them to an
//! [`crate::IncidentPhoton`] by atomic number after reading the evaluation,
//! and so does this.
//!
//! Three files in `packages/yamc-core/python/yamc/data`, one per tabulation, each named for what
//! it holds and where it came from. All plain text, so this crate stays
//! dependency-free. See `packages/yamc-core/python/yamc/data/README.md` for provenance.
//!
//! * `compton_profiles_biggs1975.txt`
//! * `density_effect_sternheimer1982.txt`
//! * `bremsstrahlung_seltzer_berger1986.txt`
//!
//! They are not embedded in the crate. Together they are about 2.4 MB, which
//! does not belong in every binary that links this, and a consumer reading
//! nuclear data is opening files anyway.

use std::collections::BTreeMap;
use std::path::Path;

use crate::data::EV_PER_MEV;
use crate::error::{Error, Result};
use crate::spline::CubicSpline;

/// The highest atomic number the tabulations cover.
pub const MAX_Z: i64 = 100;

/// Compton profiles for one element, per subshell.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ComptonProfile {
    /// Electrons in each subshell.
    pub num_electrons: Vec<f64>,
    /// Binding energy of each subshell, in eV.
    pub binding_energy: Vec<f64>,
    /// J(pz) for each subshell, on the shared [`PhotonData::pz`] grid.
    pub j: Vec<Vec<f64>>,
}

/// Bremsstrahlung and density-effect data for one element.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Bremsstrahlung {
    /// Mean excitation energy, in eV.
    pub i: f64,
    /// Electrons in each subshell.
    pub num_electrons: Vec<f64>,
    /// Ionization energy of each subshell, in eV.
    pub ionization_energy: Vec<f64>,
    /// Scaled differential cross sections in barns, resampled onto
    /// [`PhotonData::electron_energy`]: one row per incident electron energy,
    /// one column per reduced photon energy.
    pub dcs: Vec<Vec<f64>>,
}

/// The auxiliary photon data for every element the tabulations cover.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PhotonData {
    /// The projected momentum grid the Compton profiles are given on.
    pub pz: Vec<f64>,
    /// Incident electron kinetic energies the cross sections are resampled
    /// onto, in eV: 200 points logarithmically spaced from 1 keV to 1 GeV.
    pub electron_energy: Vec<f64>,
    /// Reduced photon energies, the columns of each `dcs`.
    pub photon_energy: Vec<f64>,
    /// By atomic number.
    pub compton: BTreeMap<i64, ComptonProfile>,
    /// By atomic number.
    pub bremsstrahlung: BTreeMap<i64, Bremsstrahlung>,
}

impl PhotonData {
    /// Read the three tabulations.
    ///
    /// One argument per source rather than one per format: the Compton
    /// profiles and the density effect come from unrelated measurements and
    /// only shared a file because they shared an HDF5 container upstream.
    pub fn from_files(
        compton_profiles: impl AsRef<Path>,
        density_effect: impl AsRef<Path>,
        bremsstrahlung: impl AsRef<Path>,
    ) -> Result<PhotonData> {
        let mut data = PhotonData::default();
        parse_aux_into(&mut data, &std::fs::read_to_string(compton_profiles)?)?;
        parse_aux_into(&mut data, &std::fs::read_to_string(density_effect)?)?;
        data.add_bremsstrahlung(&std::fs::read_to_string(bremsstrahlung)?)?;
        Ok(data)
    }
}

fn bad(what: &'static str) -> Error {
    Error::Unsupported { what }
}

/// Take `n` floats off the front of an iterator.
fn take_floats<'a>(
    tokens: &mut impl Iterator<Item = &'a str>,
    n: usize,
    what: &'static str,
) -> Result<Vec<f64>> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let token = tokens.next().ok_or_else(|| bad(what))?;
        out.push(token.parse::<f64>().map_err(|_| bad(what))?);
    }
    Ok(out)
}

/// Parse one tabulation into `data`.
///
/// Each file still opens with its own `COMPTON` or `DENSITY` marker, so the
/// same reader takes either and the split into per-source files needed no
/// change to the format.
fn parse_aux_into(data: &mut PhotonData, text: &str) -> Result<()> {
    let mut section = "";

    for line in text.lines() {
        let line = line.trim();
        // Each file opens with a provenance header naming its source.
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line == "COMPTON" || line == "DENSITY" {
            section = if line == "COMPTON" {
                "compton"
            } else {
                "density"
            };
            continue;
        }

        let mut tokens = line.split_whitespace();
        let tag = tokens
            .next()
            .ok_or_else(|| bad("an empty photon data line"))?;

        if tag == "pz" {
            let n: usize = tokens
                .next()
                .and_then(|t| t.parse().ok())
                .ok_or_else(|| bad("the pz count in the photon data"))?;
            data.pz = take_floats(&mut tokens, n, "the pz grid in the photon data")?;
            continue;
        }
        if tag != "Z" {
            return Err(bad("an unrecognised line in the photon data"));
        }

        let z: i64 = tokens
            .next()
            .and_then(|t| t.parse().ok())
            .ok_or_else(|| bad("an atomic number in the photon data"))?;
        let nss: usize = tokens
            .next()
            .and_then(|t| t.parse().ok())
            .ok_or_else(|| bad("a subshell count in the photon data"))?;

        match section {
            "compton" => {
                let num_electrons = take_floats(&mut tokens, nss, "Compton num_electrons")?;
                let binding_energy = take_floats(&mut tokens, nss, "Compton binding_energy")?;
                let flat = take_floats(&mut tokens, nss * data.pz.len(), "Compton J")?;
                let j = flat.chunks(data.pz.len()).map(<[f64]>::to_vec).collect();
                data.compton.insert(
                    z,
                    ComptonProfile {
                        num_electrons,
                        binding_energy,
                        j,
                    },
                );
            }
            "density" => {
                let i = take_floats(&mut tokens, 1, "the mean excitation energy")?[0];
                let num_electrons = take_floats(&mut tokens, nss, "density num_electrons")?;
                let ionization_energy = take_floats(&mut tokens, nss, "ionization_energy")?;
                data.bremsstrahlung.insert(
                    z,
                    Bremsstrahlung {
                        i,
                        num_electrons,
                        ionization_energy,
                        dcs: Vec::new(),
                    },
                );
            }
            _ => return Err(bad("a photon data line before its section header")),
        }
    }
    Ok(())
}

impl PhotonData {
    /// Read `BREMX.DAT` and resample its cross sections onto the common grid.
    ///
    /// The file gives the scaled cross sections on 57 tabulated electron
    /// energies. They are interpolated with a not-a-knot cubic spline in log
    /// energy (linear in the cross section itself) onto 200 logarithmically
    /// spaced points, which is exactly what the Python package does with
    /// `scipy.interpolate.CubicSpline`.
    fn add_bremsstrahlung(&mut self, text: &str) -> Result<()> {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        let number = |i: usize, what: &'static str| -> Result<f64> {
            tokens
                .get(i)
                .ok_or_else(|| bad(what))?
                .parse::<f64>()
                .map_err(|_| bad(what))
        };

        // The counts sit at fixed offsets in the header.
        let n = number(37, "the electron energy count in BREMX.DAT")? as usize;
        let k = number(38, "the photon energy count in BREMX.DAT")? as usize;
        let mut p = 39;

        // 200 points from 1 keV to 1 GeV, matching `np.logspace(3, 9, 200)`.
        self.electron_energy = (0..200)
            .map(|i| 10f64.powf(3.0 + 6.0 * i as f64 / 199.0))
            .collect();
        let log_energy: Vec<f64> = self.electron_energy.iter().map(|e| e.ln()).collect();

        // Tabulated energies are in MeV; the spline runs in log eV.
        let mut logx = Vec::with_capacity(n);
        for i in 0..n {
            logx.push((number(p + i, "an electron energy in BREMX.DAT")? * EV_PER_MEV).ln());
        }
        p += n;

        self.photon_energy = (0..k)
            .map(|i| number(p + i, "a photon energy in BREMX.DAT"))
            .collect::<Result<_>>()?;
        p += k;

        for z in 1..=MAX_Z {
            // Row-major, `n` electron energies by `k` photon energies, in
            // millibarns.
            let mut y = vec![vec![0.0; k]; n];
            for (row, values) in y.iter_mut().enumerate() {
                for (col, value) in values.iter_mut().enumerate() {
                    *value = number(p + row * k + col, "a cross section in BREMX.DAT")? * 1.0e-3;
                }
            }
            p += n * k;

            // One spline per reduced photon energy, down the electron energy
            // axis. Built column by column, then transposed into rows so the
            // result is indexed the way the Python package indexes it.
            let mut dcs = vec![vec![0.0; k]; log_energy.len()];
            let mut column = vec![0.0; n];
            for j in 0..k {
                for (i, value) in column.iter_mut().enumerate() {
                    *value = y[i][j];
                }
                let spline = CubicSpline::new(&logx, &column)?;
                for (row, &q) in log_energy.iter().enumerate() {
                    dcs[row][j] = spline.eval(q);
                }
            }

            self.bremsstrahlung
                .get_mut(&z)
                .ok_or_else(|| bad("BREMX.DAT covers an element the density data does not"))?
                .dcs = dcs;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn datafiles() -> PathBuf {
        // In `packages/yamc-core/python/yamc/data`, not beside this crate. They are not the
        // parser's data: it takes paths and never looks a file up by name.
        // They are yamc's data, and living inside the Python package is what
        // puts them in the wheel under `yamc/data` rather than in a top-level
        // `crates/` directory in site-packages.
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../packages/yamc-core/python/yamc/data")
    }

    fn load() -> PhotonData {
        let d = datafiles();
        PhotonData::from_files(
            d.join("compton_profiles_biggs1975.txt"),
            d.join("density_effect_sternheimer1982.txt"),
            d.join("bremsstrahlung_seltzer_berger1986.txt"),
        )
        .unwrap()
    }

    #[test]
    fn reads_every_element() {
        let data = load();
        assert_eq!(data.pz.len(), 31);
        assert_eq!(data.compton.len(), MAX_Z as usize);
        assert_eq!(data.bremsstrahlung.len(), MAX_Z as usize);
        assert_eq!(data.photon_energy.len(), 30);
        assert_eq!(data.electron_energy.len(), 200);

        // The grid is 1 keV to 1 GeV inclusive.
        assert!((data.electron_energy[0] - 1.0e3).abs() < 1e-9);
        assert!((data.electron_energy[199] - 1.0e9).abs() < 1.0);

        // Hydrogen has one subshell, uranium 27.
        let h = &data.compton[&1];
        assert_eq!(h.num_electrons, [1.0]);
        assert_eq!(h.j.len(), 1);
        assert_eq!(h.j[0].len(), 31);
        assert_eq!(data.compton[&92].num_electrons.len(), 27);

        // Every element's dcs is the full resampled grid.
        for z in 1..=MAX_Z {
            let b = &data.bremsstrahlung[&z];
            assert_eq!(b.dcs.len(), 200, "Z={z}");
            assert_eq!(b.dcs[0].len(), 30, "Z={z}");
            assert!(b.i > 0.0, "Z={z} should have a mean excitation energy");
        }
    }

    /// Attaching the data to an element gives it what the Python reader's
    /// `from_endf` attaches at the end of its own read.
    #[test]
    fn attaches_to_an_incident_photon() {
        let data = load();
        let material = crate::Material::from_str(&crate::testdata::text(include_bytes!(
            "../fixtures/photoat-001_H_000.endf.xz"
        )))
        .unwrap();
        let mut photon = crate::IncidentPhoton::from_endf(&material, None).unwrap();

        // An ENDF evaluation carries neither, so both are absent until asked
        // for -- which is the difference from the Python package, where the
        // lookup happens inside `from_endf`.
        assert!(photon.compton_profiles.is_none());
        assert!(photon.bremsstrahlung.is_none());

        photon.add_photon_data(&data);

        let profiles = photon.compton_profiles.as_ref().expect("hydrogen profiles");
        assert_eq!(profiles.num_electrons, [1.0]);
        // J comes across as a Tabulated1D against pz, as it does in Python.
        assert_eq!(profiles.j.len(), 1);
        assert_eq!(profiles.j[0].x, data.pz);
        assert_eq!(profiles.j[0].y, data.compton[&1].j[0]);

        let brem = photon.bremsstrahlung.as_ref().expect("hydrogen brem");
        assert_eq!(brem.i, data.bremsstrahlung[&1].i);
        assert_eq!(brem.electron_energy.len(), 200);
        assert_eq!(brem.dcs.len(), 200);
        assert_eq!(brem.dcs[0].len(), 30);
    }

    /// Spot values taken from the Python package, which reads the HDF5 files
    /// with h5py and resamples with `scipy.interpolate.CubicSpline`. Two
    /// different source formats and two different spline implementations
    /// agreeing is the point of the check.
    #[test]
    fn agrees_with_the_python_reader() {
        let data = load();

        // Compton profiles come straight off the file.
        assert_eq!(data.pz[0], 0.0);
        assert_eq!(data.pz[30], 100.0);
        assert_eq!(data.compton[&92].binding_energy[0], 116110.0);

        // The resampled cross sections go through the spline.
        let u = &data.bremsstrahlung[&92];
        assert_eq!(u.i, 890.0);
        for (row, col, want) in [
            (0usize, 0usize, 0.00046007999999999997),
            (0, 29, 0.0010523499999999999),
            (100, 15, 0.0038985821050669185),
            (199, 29, 0.00147492),
        ] {
            let got = u.dcs[row][col];
            assert!(
                (got - want).abs() <= 1e-12 * want.abs(),
                "dcs[{row}][{col}]: got {got}, want {want}"
            );
        }
    }
}
