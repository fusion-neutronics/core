//! Unresolved resonance region probability tables.
//!
//! In the unresolved region the individual resonances are not resolved, so the
//! cross section is described statistically: at each energy a set of
//! equiprobable bands is given and one is sampled.
//!
//! Read from an ACE table rather than from ENDF. The ENDF form is MF=2 with
//! LRU=2, which [`crate::mf::mf2`] handles; this is the processed form NJOY
//! writes, and the one a transport code samples.

use crate::ace::Table;
use crate::data::EV_PER_MEV;

/// Which quantity each row of a probability table holds.
///
/// The second axis of [`ProbabilityTables::table`] is indexed by these.
pub const BAND_QUANTITIES: [&str; 6] = [
    "cumulative probability",
    "total",
    "elastic",
    "fission",
    "capture",
    "heating",
];

/// Probability tables for the unresolved resonance region.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbabilityTables {
    /// Energies in eV at which tables exist.
    pub energy: Vec<f64>,
    /// The tables, flattened from `(n_energy, 6, n_band)` in C order.
    ///
    /// Kept flat with an explicit shape because that is how the converted data
    /// stores it, and reshaping twice helps nobody.
    pub table: Vec<f64>,
    /// `[n_energy, 6, n_band]`.
    pub shape: [usize; 3],
    /// 2 for lin-lin, 5 for log-log.
    pub interpolation: i64,
    /// Below zero, the inelastic cross section is zero across the unresolved
    /// range. Above zero, the MT whose cross section to use there.
    pub inelastic_flag: i64,
    /// The same, for the other absorption cross section.
    pub absorption_flag: i64,
    /// Whether the values are cross sections (false) or factors multiplying
    /// the smooth background (true).
    pub multiply_smooth: bool,
}

impl ProbabilityTables {
    /// Number of energies at which tables are given.
    pub fn n_energy(&self) -> usize {
        self.shape[0]
    }

    /// Number of equiprobable bands.
    pub fn n_band(&self) -> usize {
        self.shape[2]
    }

    /// One value, by energy index, quantity index and band index.
    pub fn get(&self, energy: usize, quantity: usize, band: usize) -> Option<f64> {
        let [_, n_q, n_b] = self.shape;
        if energy >= self.shape[0] || quantity >= n_q || band >= n_b {
            return None;
        }
        self.table
            .get((energy * n_q + quantity) * n_b + band)
            .copied()
    }

    /// Read the probability tables from an ACE table.
    ///
    /// `None` when the table has no unresolved region, which is the usual case
    /// for a light nuclide.
    pub fn from_ace(table: &Table) -> Option<ProbabilityTables> {
        // JXS(23) locates the URR block, and is zero when there is none.
        let start = *table.jxs.get(23)?;
        if start <= 0 {
            return None;
        }
        let xss = &table.xss;
        let at = |i: i64| -> Option<f64> { xss.get(usize::try_from(i).ok()?).copied() };

        let n_energy = at(start)? as usize;
        let n_band = at(start + 1)? as usize;
        let interpolation = at(start + 2)? as i64;
        let inelastic_flag = at(start + 3)? as i64;
        let absorption_flag = at(start + 4)? as i64;
        let multiply_smooth = at(start + 5)? as i64 == 1;

        let mut idx = start + 6;
        // The file stores energies in MeV.
        let mut energy = Vec::with_capacity(n_energy);
        for i in 0..n_energy {
            energy.push(at(idx + i as i64)? * EV_PER_MEV);
        }
        idx += n_energy as i64;

        let count = n_energy * 6 * n_band;
        let mut values = Vec::with_capacity(count);
        for i in 0..count {
            values.push(at(idx + i as i64)?);
        }

        // Row 5 of each energy's block is the heating number, also in MeV.
        for e in 0..n_energy {
            for b in 0..n_band {
                let i = (e * 6 + 5) * n_band + b;
                values[i] *= EV_PER_MEV;
            }
        }

        Some(ProbabilityTables {
            energy,
            table: values,
            shape: [n_energy, 6, n_band],
            interpolation,
            inelastic_flag,
            absorption_flag,
            multiply_smooth,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_light_nuclide_has_no_unresolved_region() {
        // Li6 has no URR: JXS(23) is zero, and the reader says so rather than
        // reading whatever happens to sit at index zero.
        let tables = crate::testdata::ace_tables(crate::testdata::LI6_ACE);
        assert_eq!(tables[0].jxs[23], 0);
        assert!(ProbabilityTables::from_ace(&tables[0]).is_none());
    }
}
