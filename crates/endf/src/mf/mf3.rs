//! MF=3, reaction cross sections.

use crate::error::Result;
use crate::function::Tabulated1D;
use crate::records::Reader;

/// A reaction cross section as given in MF=3.
#[derive(Debug, Clone, PartialEq)]
pub struct Mf3 {
    /// ZA identifier, 1000*Z + A.
    pub za: i64,
    /// Ratio of the target mass to that of a neutron.
    pub awr: f64,
    /// Mass-difference Q value, eV.
    pub qm: f64,
    /// Reaction Q value for the lowest energy state, eV.
    pub qi: f64,
    /// Complex-break-up flag.
    pub lr: i64,
    /// Cross section as a function of incident energy, barns against eV.
    pub sigma: Tabulated1D,
}

/// Parse an MF=3 section.
pub fn parse_mf3(reader: &mut Reader) -> Result<Mf3> {
    let head = reader.head_record()?;
    let tab = reader.tab1_record()?;
    Ok(Mf3 {
        za: head.za,
        awr: head.awr,
        qm: tab.c1,
        qi: tab.c2,
        lr: tab.l2,
        sigma: tab.table,
    })
}

#[cfg(test)]
mod tests {
    use crate::material::Material;

    const FIXTURE: &[u8] = include_bytes!("../../fixtures/n-095_Am_244.endf.xz");

    #[test]
    fn reads_the_total_cross_section() {
        let m = Material::from_str(&crate::testdata::text(FIXTURE)).unwrap();
        let total = m.mf3(1).expect("MF=3 MT=1 is present");

        assert_eq!(total.za, 95244);
        // MT=1 is the total cross section, so it has no threshold and no Q.
        assert_eq!(total.qm, 0.0);
        assert_eq!(total.qi, 0.0);
        assert_eq!(total.lr, 0);

        let x = &total.sigma.x;
        let y = &total.sigma.y;
        assert_eq!(x.len(), y.len());
        assert!(!x.is_empty());
        // Cross sections are tabulated on an ascending energy grid.
        assert!(x.windows(2).all(|w| w[1] >= w[0]));
        assert!(y.iter().all(|&v| v >= 0.0));
        // The evaluation runs from 1e-5 eV to 20 MeV.
        assert!((x[0] - 1e-5).abs() < 1e-12);
        assert!((x[x.len() - 1] - 2.0e7).abs() < 1.0);
    }

    #[test]
    fn the_sum_rule_holds_at_the_first_energy() {
        // MT=1 is total, and for this evaluation elastic (2), fission (18) and
        // capture (102) are its parts at thermal energies.
        let m = Material::from_str(&crate::testdata::text(FIXTURE)).unwrap();
        let total = m.mf3(1).unwrap();
        let parts: f64 = [2, 18, 102]
            .iter()
            .filter_map(|&mt| m.mf3(mt))
            .map(|r| r.sigma.y[0])
            .sum();
        let got = total.sigma.y[0];
        // MF=3 values are written in 11-character fields, so the evaluation
        // carries six significant figures and its own total is rounded: here
        // 146212.0 against 146211.6223 summed. The agreement to ~3e-6 is the
        // format's, not the reader's, and the Python reader gives the same two
        // numbers.
        assert!(
            (got - parts).abs() <= 1e-5 * got.abs(),
            "total {got} != sum of parts {parts}"
        );
    }

    #[test]
    fn interpolating_returns_a_tabulated_point() {
        let m = Material::from_str(&crate::testdata::text(FIXTURE)).unwrap();
        let total = m.mf3(1).unwrap();
        let x0 = total.sigma.x[0];
        assert_eq!(total.sigma.eval(x0), total.sigma.y[0]);
    }
}
