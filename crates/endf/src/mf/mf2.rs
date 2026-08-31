//! MF=2 MT=151, resonance parameters.

use crate::error::{Error, Result};
use crate::function::Tabulated1D;
use crate::records::Reader;

/// MF=2 MT=151.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf2 {
    pub za: i64,
    pub awr: f64,
    pub nis: i64,
    pub isotopes: Vec<Isotope>,
}

/// One isotope of the material, with its energy ranges.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Isotope {
    pub zai: f64,
    pub abn: f64,
    pub lfw: i64,
    pub ner: i64,
    pub ranges: Vec<ResonanceRange>,
}

/// One energy range, and the parameters given over it.
#[derive(Debug, Clone, PartialEq)]
pub struct ResonanceRange {
    pub el: f64,
    pub eh: f64,
    /// 1 resolved, 2 unresolved, 0 neither.
    pub lru: i64,
    /// Which representation the parameters use.
    pub lrf: i64,
    pub nro: i64,
    pub naps: i64,
    pub parameters: ResonanceParameters,
}

/// The parameters of a range, in whichever representation it uses.
#[derive(Debug, Clone, PartialEq)]
pub enum ResonanceParameters {
    /// LRF=0: the scattering radius and nothing else.
    ScatteringRadius { spi: f64, ap: f64, nls: i64 },
    /// LRF=1 single-level or LRF=2 multi-level Breit-Wigner. The two are read
    /// identically; only the reconstruction differs.
    BreitWigner(BreitWigner),
    /// LRF=3.
    ReichMoore(ReichMoore),
    /// LRF=7.
    RMatrixLimited(Box<RMatrixLimited>),
    /// LRU=2.
    Unresolved(Box<Unresolved>),
    /// Nothing was read for this range.
    ///
    /// Reached for an unresolved range with LRF=1, which the Python reader
    /// dispatches past without reading: see
    /// <https://github.com/shimwell/endf-python/issues/15>. [`Unresolved`]
    /// below implements those cases already, so correcting the dispatch is a
    /// one-line change once that is settled upstream.
    Absent,
}

/// LRF=1 and LRF=2: Breit-Wigner parameters.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BreitWigner {
    /// NRO/=0: energy-dependent scattering radius.
    pub ape: Option<Tabulated1D>,
    pub spi: f64,
    pub ap: f64,
    pub nls: i64,
    pub sections: Vec<BreitWignerSection>,
}

/// The resonances of one orbital angular momentum.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BreitWignerSection {
    pub awri: f64,
    pub qx: f64,
    pub l: i64,
    pub lrx: i64,
    pub nrs: i64,
    /// Resonance energy.
    pub er: Vec<f64>,
    /// Spin.
    pub aj: Vec<f64>,
    /// Total width.
    pub gt: Vec<f64>,
    /// Neutron width.
    pub gn: Vec<f64>,
    /// Radiation width.
    pub gg: Vec<f64>,
    /// Fission width.
    pub gf: Vec<f64>,
}

/// LRF=3: Reich-Moore parameters.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ReichMoore {
    pub ape: Option<Tabulated1D>,
    pub spi: f64,
    pub ap: f64,
    pub lad: i64,
    pub nls: i64,
    pub nlsc: i64,
    pub sections: Vec<ReichMooreSection>,
}

/// The resonances of one orbital angular momentum.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ReichMooreSection {
    pub awri: f64,
    pub apl: f64,
    pub l: i64,
    pub nrs: i64,
    pub er: Vec<f64>,
    pub aj: Vec<f64>,
    pub gn: Vec<f64>,
    pub gg: Vec<f64>,
    /// First fission width.
    pub gfa: Vec<f64>,
    /// Second fission width.
    pub gfb: Vec<f64>,
}

/// The particle pairs of an R-matrix limited evaluation, column-wise.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ParticlePairs {
    pub ma: Vec<f64>,
    pub mb: Vec<f64>,
    pub za: Vec<f64>,
    pub zb: Vec<f64>,
    pub ia: Vec<f64>,
    pub ib: Vec<f64>,
    pub q: Vec<f64>,
    pub pnt: Vec<f64>,
    pub shf: Vec<f64>,
    pub mt: Vec<f64>,
    pub pa: Vec<f64>,
    pub pb: Vec<f64>,
}

/// The channels of one spin group, column-wise.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Channels {
    pub ppi: Vec<f64>,
    pub l: Vec<f64>,
    pub sch: Vec<f64>,
    pub bnd: Vec<f64>,
    pub ape: Vec<f64>,
    pub apt: Vec<f64>,
}

/// One spin group of an R-matrix limited evaluation.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SpinGroup {
    pub aj: f64,
    pub pj: f64,
    pub kbk: i64,
    pub kps: i64,
    pub nch: i64,
    pub channels: Channels,
    pub nrs: i64,
    pub nx: i64,
    /// Resonance energies.
    pub er: Vec<f64>,
    /// Widths, `nch` rows of `nrs`, the transpose of how the file stores them.
    pub gam: Vec<Vec<f64>>,
    // Background R-matrix, present when KBK > 0.
    pub lch: Option<i64>,
    pub lbk: Option<i64>,
    pub rbr: Option<Tabulated1D>,
    pub rbi: Option<Tabulated1D>,
    pub ed: Option<f64>,
    pub eu: Option<f64>,
    // Tabulated phase shifts, present when KPS > 0.
    pub lps: Option<i64>,
    pub psr: Option<Tabulated1D>,
    pub psi: Option<Tabulated1D>,
}

/// LRF=7: R-matrix limited parameters.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RMatrixLimited {
    pub ifg: i64,
    pub krm: i64,
    pub njs: i64,
    pub krl: i64,
    pub npp: i64,
    pub particle_pairs: ParticlePairs,
    pub spin_groups: Vec<SpinGroup>,
}

/// LRU=2: unresolved resonance parameters.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Unresolved {
    pub ape: Option<Tabulated1D>,
    pub spi: f64,
    pub ap: f64,
    pub lssf: i64,
    pub nls: i64,
    /// Case B only: the energies the fission widths are given on.
    pub ne: Option<i64>,
    pub es: Vec<f64>,
    pub ranges: Vec<UnresolvedRange>,
}

/// The unresolved parameters of one orbital angular momentum.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct UnresolvedRange {
    pub awri: f64,
    pub l: i64,
    pub njs: i64,
    /// Case A: energy-independent parameters, one entry per J.
    pub d: Vec<f64>,
    pub aj: Vec<f64>,
    pub amun: Vec<f64>,
    pub gno: Vec<f64>,
    pub gg: Vec<f64>,
    /// Cases B and C: one entry per J.
    pub parameters: Vec<UnresolvedParameters>,
}

/// The unresolved parameters of one spin, in Case B or Case C form.
#[derive(Debug, Clone, PartialEq)]
pub enum UnresolvedParameters {
    /// LFW=1, LRF=1: only the fission widths are energy dependent.
    CaseB {
        muf: i64,
        d: f64,
        aj: f64,
        amun: f64,
        gn0: f64,
        gg: f64,
        gf: Vec<f64>,
    },
    /// LRF=2: every parameter is energy dependent.
    CaseC {
        aj: f64,
        interpolation: i64,
        ne: i64,
        amux: f64,
        amun: f64,
        amuf: f64,
        e: Vec<f64>,
        d: Vec<f64>,
        gx: Vec<f64>,
        gn0: Vec<f64>,
        gg: Vec<f64>,
        gf: Vec<f64>,
    },
}

/// Take every `stride`-th value starting at `offset`.
fn column(values: &[f64], offset: usize, stride: usize) -> Vec<f64> {
    values
        .iter()
        .skip(offset)
        .step_by(stride)
        .copied()
        .collect()
}

/// Read the energy-dependent scattering radius that NRO/=0 puts first.
fn parse_ape(reader: &mut Reader, nro: i64) -> Result<Option<Tabulated1D>> {
    if nro != 0 {
        Ok(Some(reader.tab1_record()?.table))
    } else {
        Ok(None)
    }
}

fn parse_breit_wigner(reader: &mut Reader, nro: i64) -> Result<BreitWigner> {
    let ape = parse_ape(reader, nro)?;
    let c = reader.cont_record()?;
    let mut data = BreitWigner {
        ape,
        spi: c.c1,
        ap: c.c2,
        nls: c.n1,
        sections: Vec::new(),
    };
    for _ in 0..data.nls.max(0) {
        let list = reader.list_record()?;
        let v = &list.values;
        data.sections.push(BreitWignerSection {
            awri: list.cont.c1,
            qx: list.cont.c2,
            l: list.cont.l1,
            lrx: list.cont.l2,
            nrs: list.cont.n2,
            er: column(v, 0, 6),
            aj: column(v, 1, 6),
            gt: column(v, 2, 6),
            gn: column(v, 3, 6),
            gg: column(v, 4, 6),
            gf: column(v, 5, 6),
        });
    }
    Ok(data)
}

fn parse_reich_moore(reader: &mut Reader, nro: i64) -> Result<ReichMoore> {
    let ape = parse_ape(reader, nro)?;
    let c = reader.cont_record()?;
    let mut data = ReichMoore {
        ape,
        spi: c.c1,
        ap: c.c2,
        lad: c.l1,
        nls: c.n1,
        nlsc: c.n2,
        sections: Vec::new(),
    };
    for _ in 0..data.nls.max(0) {
        let list = reader.list_record()?;
        let v = &list.values;
        data.sections.push(ReichMooreSection {
            awri: list.cont.c1,
            apl: list.cont.c2,
            l: list.cont.l1,
            nrs: list.cont.n2,
            er: column(v, 0, 6),
            aj: column(v, 1, 6),
            gn: column(v, 2, 6),
            gg: column(v, 3, 6),
            gfa: column(v, 4, 6),
            gfb: column(v, 5, 6),
        });
    }
    Ok(data)
}

fn parse_r_matrix_limited(reader: &mut Reader) -> Result<RMatrixLimited> {
    let c = reader.cont_record()?;
    let mut data = RMatrixLimited {
        ifg: c.l1,
        krm: c.l2,
        njs: c.n1,
        krl: c.n2,
        ..Default::default()
    };

    let list = reader.list_record()?;
    let v = &list.values;
    data.npp = list.cont.l1;
    data.particle_pairs = ParticlePairs {
        ma: column(v, 0, 12),
        mb: column(v, 1, 12),
        za: column(v, 2, 12),
        zb: column(v, 3, 12),
        ia: column(v, 4, 12),
        ib: column(v, 5, 12),
        q: column(v, 6, 12),
        pnt: column(v, 7, 12),
        shf: column(v, 8, 12),
        mt: column(v, 9, 12),
        pa: column(v, 10, 12),
        pb: column(v, 11, 12),
    };

    for _ in 0..data.njs.max(0) {
        let list = reader.list_record()?;
        let v = &list.values;
        let nch = list.cont.n2;
        let mut group = SpinGroup {
            aj: list.cont.c1,
            pj: list.cont.c2,
            kbk: list.cont.l1,
            kps: list.cont.l2,
            nch,
            channels: Channels {
                ppi: column(v, 0, 6),
                l: column(v, 1, 6),
                sch: column(v, 2, 6),
                bnd: column(v, 3, 6),
                ape: column(v, 4, 6),
                apt: column(v, 5, 6),
            },
            ..Default::default()
        };

        // Resonance energies and widths: each resonance is one energy followed
        // by `nch` widths.
        let list = reader.list_record()?;
        let v = &list.values;
        group.nrs = list.cont.l2;
        group.nx = list.cont.n2;
        let width = nch.max(0) as usize + 1;
        group.er = column(v, 0, width);
        // Stored resonance-major; transposed to channel-major here, matching
        // what the Python reader returns.
        let nrs = group.nrs.max(0) as usize;
        let nch_u = nch.max(0) as usize;
        let mut gam = vec![Vec::with_capacity(nrs); nch_u];
        for j in 0..nrs {
            for (c, row) in gam.iter_mut().enumerate() {
                row.push(v.get(1 + width * j + c).copied().unwrap_or(0.0));
            }
        }
        group.gam = gam;

        // Background R-matrix.
        if group.kbk > 0 {
            let list = reader.list_record()?;
            group.lch = Some(list.cont.l1);
            group.lbk = Some(list.cont.l2);
            match list.cont.l2 {
                1 => {
                    group.rbr = Some(reader.tab1_record()?.table);
                    group.rbi = Some(reader.tab1_record()?.table);
                }
                2 | 3 => {
                    let list = reader.list_record()?;
                    group.ed = Some(list.cont.c1);
                    group.eu = Some(list.cont.c2);
                }
                _ => {}
            }
        }

        // Tabulated phase shifts.
        if group.kps > 0 {
            let list = reader.list_record()?;
            let lps = list.cont.n1;
            group.lps = Some(lps);
            if lps == 1 {
                group.psr = Some(reader.tab1_record()?.table);
                group.psi = Some(reader.tab1_record()?.table);
            }
        }

        data.spin_groups.push(group);
    }

    Ok(data)
}

fn parse_unresolved(reader: &mut Reader, lfw: i64, lrf: i64, nro: i64) -> Result<Unresolved> {
    let mut data = Unresolved {
        ape: parse_ape(reader, nro)?,
        ..Default::default()
    };

    // Case B carries these in its own LIST record instead of a CONT.
    if !(lfw == 1 && lrf == 1) {
        let c = reader.cont_record()?;
        data.spi = c.c1;
        data.ap = c.c2;
        data.lssf = c.l1;
        data.nls = c.n1;
    }

    if lfw == 0 && lrf == 1 {
        // Case A: no fission widths, everything energy independent.
        for _ in 0..data.nls.max(0) {
            let list = reader.list_record()?;
            let v = &list.values;
            data.ranges.push(UnresolvedRange {
                awri: list.cont.c1,
                l: list.cont.l1,
                njs: list.cont.n2,
                d: column(v, 0, 6),
                aj: column(v, 1, 6),
                amun: column(v, 2, 6),
                gno: column(v, 3, 6),
                gg: column(v, 4, 6),
                parameters: Vec::new(),
            });
        }
    } else if lfw == 1 && lrf == 1 {
        // Case B: only the fission widths are energy dependent.
        let list = reader.list_record()?;
        data.spi = list.cont.c1;
        data.ap = list.cont.c2;
        data.lssf = list.cont.l1;
        data.ne = Some(list.cont.n1);
        data.nls = list.cont.n2;
        data.es = list.values;

        for _ in 0..data.nls.max(0) {
            let c = reader.cont_record()?;
            let mut range = UnresolvedRange {
                awri: c.c1,
                l: c.l1,
                njs: c.n1,
                ..Default::default()
            };
            for _ in 0..range.njs.max(0) {
                let list = reader.list_record()?;
                let v = &list.values;
                range.parameters.push(UnresolvedParameters::CaseB {
                    muf: list.cont.l2,
                    d: v.first().copied().unwrap_or(0.0),
                    aj: v.get(1).copied().unwrap_or(0.0),
                    amun: v.get(2).copied().unwrap_or(0.0),
                    gn0: v.get(3).copied().unwrap_or(0.0),
                    gg: v.get(4).copied().unwrap_or(0.0),
                    // v[5] is unused by the format.
                    gf: v.get(6..).unwrap_or(&[]).to_vec(),
                });
            }
            data.ranges.push(range);
        }
    } else if lrf == 2 {
        // Case C: every parameter is energy dependent.
        for _ in 0..data.nls.max(0) {
            let c = reader.cont_record()?;
            let mut range = UnresolvedRange {
                awri: c.c1,
                l: c.l1,
                njs: c.n1,
                ..Default::default()
            };
            for _ in 0..range.njs.max(0) {
                let list = reader.list_record()?;
                let v = &list.values;
                range.parameters.push(UnresolvedParameters::CaseC {
                    aj: list.cont.c1,
                    interpolation: list.cont.l1,
                    ne: list.cont.n2,
                    amux: v.get(2).copied().unwrap_or(0.0),
                    amun: v.get(3).copied().unwrap_or(0.0),
                    amuf: v.get(5).copied().unwrap_or(0.0),
                    e: column(v, 6, 6),
                    d: column(v, 7, 6),
                    gx: column(v, 8, 6),
                    gn0: column(v, 9, 6),
                    gg: column(v, 10, 6),
                    gf: column(v, 11, 6),
                });
            }
            data.ranges.push(range);
        }
    }

    Ok(data)
}

/// Parse MF=2 MT=151.
pub fn parse_mf2(reader: &mut Reader) -> Result<Mf2> {
    let head = reader.head_record()?;
    let mut data = Mf2 {
        za: head.za,
        awr: head.awr,
        nis: head.n1,
        isotopes: Vec::new(),
    };

    for _ in 0..data.nis.max(0) {
        let c = reader.cont_record()?;
        let mut iso = Isotope {
            zai: c.c1,
            abn: c.c2,
            lfw: c.l2,
            ner: c.n1,
            ranges: Vec::new(),
        };

        for _ in 0..iso.ner.max(0) {
            let c = reader.cont_record()?;
            let (lru, lrf, nro) = (c.l1, c.l2, c.n1);

            let parameters = if lrf == 0 {
                let c = reader.cont_record()?;
                ResonanceParameters::ScatteringRadius {
                    spi: c.c1,
                    ap: c.c2,
                    nls: c.n1,
                }
            } else if lru == 0 || lru == 1 {
                match lrf {
                    1 | 2 => ResonanceParameters::BreitWigner(parse_breit_wigner(reader, nro)?),
                    3 => ResonanceParameters::ReichMoore(parse_reich_moore(reader, nro)?),
                    4 => {
                        return Err(Error::Unsupported {
                            what: "the Adler-Adler resonance formalism (MF=2, LRF=4)",
                        })
                    }
                    7 => ResonanceParameters::RMatrixLimited(Box::new(parse_r_matrix_limited(
                        reader,
                    )?)),
                    _ => {
                        return Err(Error::Unsupported {
                            what: "an unrecognised MF=2 resonance formalism",
                        })
                    }
                }
            } else if lru == 2 {
                // `lru`, because LRU is what says resolved or unresolved; LRF
                // only selects the formalism within the range. Both readers
                // tested LRF here, which skipped Cases A and B (LRU=2 with
                // LRF=1) and left their records unread, so the next range was
                // parsed from the middle of this one. Fixed in
                // <https://github.com/shimwell/endf-python/issues/15>.
                ResonanceParameters::Unresolved(Box::new(parse_unresolved(
                    reader, iso.lfw, lrf, nro,
                )?))
            } else {
                ResonanceParameters::Absent
            };

            iso.ranges.push(ResonanceRange {
                el: c.c1,
                eh: c.c2,
                lru,
                lrf,
                nro,
                naps: c.n2,
                parameters,
            });
        }

        data.isotopes.push(iso);
    }

    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::Material;

    const FIXTURE: &[u8] = include_bytes!("../../fixtures/n-095_Am_244.endf.xz");

    #[test]
    fn reads_a_scattering_radius_only_range() {
        let m = Material::from_str(&crate::testdata::text(FIXTURE)).unwrap();
        let d = m.mf2().expect("MF=2 MT=151 is present");
        assert_eq!(d.za, 95244);
        assert_eq!(d.isotopes.len(), 1);
        let iso = &d.isotopes[0];
        assert_eq!(iso.ranges.len(), iso.ner as usize);

        // This evaluation gives no resonance parameters, only the radius.
        let range = &iso.ranges[0];
        assert_eq!(range.lru, 0);
        assert_eq!(range.lrf, 0);
        match &range.parameters {
            ResonanceParameters::ScatteringRadius { ap, .. } => assert!(*ap > 0.0),
            other => panic!("expected a scattering radius, got {other:?}"),
        }
    }

    /// An unresolved range with LRF=1 (Case A) is read, and its records are
    /// consumed so a following range stays aligned.
    ///
    /// Both readers used to dispatch on LRF here where the format uses LRU, so
    /// this range matched no branch: its parameters were dropped and its
    /// records were left on the stream. See
    /// <https://github.com/shimwell/endf-python/issues/15>.
    #[test]
    fn an_unresolved_range_with_lrf_1_is_read() {
        // Plain decimals, right-justified in the format's 11-column fields.
        // `float_endf` reads these as readily as the e-less exponential form.
        fn f(v: f64) -> String {
            format!("{v:>11}")
        }
        fn i(v: i64) -> String {
            format!("{v:>11}")
        }
        fn line(fields: [String; 6]) -> String {
            format!("{:<66}9999 2151\n", fields.concat())
        }

        // NIS=1; LFW=0 and NER=2, so a Case A range and then a resolved one.
        let text = line([f(95244.0), f(241.968), i(0), i(0), i(1), i(0)])
            + &line([f(95244.0), f(1.0), i(0), i(0), i(2), i(0)])
            // Range 1: LRU=2, LRF=1.
            + &line([f(100.0), f(1000.0), i(2), i(1), i(0), i(0)])
            + &line([f(2.5), f(0.9), i(0), i(0), i(1), i(0)])
            + &line([f(241.968), f(0.0), i(0), i(0), i(6), i(1)])
            + &line([f(10.0), f(3.0), f(1.0), f(0.5), f(0.04), f(0.0)])
            // Range 2: LRU=1, LRF=2, one resonance. Only reachable if the
            // range above consumed exactly its own two records.
            + &line([f(1000.0), f(9000.0), i(1), i(2), i(0), i(0)])
            + &line([f(2.5), f(0.9), i(0), i(0), i(1), i(0)])
            + &line([f(241.968), f(0.0), i(0), i(0), i(6), i(1)])
            + &line([f(500.0), f(3.0), f(1.2), f(1.0), f(0.2), f(0.0)]);

        let d = parse_mf2(&mut Reader::new(&text)).unwrap();
        assert_eq!(d.isotopes[0].ranges.len(), 2);

        let range = &d.isotopes[0].ranges[0];
        assert_eq!((range.lru, range.lrf), (2, 1));
        match &range.parameters {
            ResonanceParameters::Unresolved(u) => {
                assert_eq!(u.spi, 2.5);
                assert_eq!(u.ap, 0.9);
                assert_eq!(u.nls, 1);
                // Case A: energy-independent parameters, not a `parameters`
                // list, which is what Cases B and C build.
                assert_eq!(u.ranges[0].d, [10.0]);
                assert_eq!(u.ranges[0].gno, [0.5]);
                assert!(u.ranges[0].parameters.is_empty());
            }
            other => panic!("expected unresolved Case A parameters, got {other:?}"),
        }

        // The dangerous half: before the fix this came back as the leftovers
        // of the range above rather than as its own header.
        let after = &d.isotopes[0].ranges[1];
        assert_eq!((after.lru, after.lrf), (1, 2));
        assert_eq!((after.el, after.eh), (1000.0, 9000.0));
        match &after.parameters {
            ResonanceParameters::BreitWigner(b) => assert_eq!(b.sections[0].er, [500.0]),
            other => panic!("expected Breit-Wigner parameters, got {other:?}"),
        }
    }
}
