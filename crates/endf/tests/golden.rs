//! Hold the Rust reader to what the Python reader produces.
//!
//! Every file in `tests/golden/` is a reference dump written by
//! `tools/dump_golden.py`. This test finds each one, reads the ENDF file it
//! names, builds the same `path -> value` map from its own parse, and compares
//! the two maps whole.
//!
//! Comparing maps rather than walking records is what makes this scale to the
//! whole package: a field that is renamed, dropped or added shows up as a path
//! on one side and not the other, without the test needing to know what the
//! field was for.
//!
//! Values are compared exactly. The dump records the shortest round-tripping
//! decimal and both readers parse decimals with correct rounding, so any
//! difference is a real one. The single exception is `…/evaly`, the sampled
//! interpolation, where the two languages evaluate the same expression through
//! their own `ln` and `exp`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use endf::ace;
use endf::mf::atomic::ElectroAtomicDistribution;
use endf::mf::covariance::Mf33Subsection;
use endf::mf::mf1::{FissionEnergyRelease, Nu, FISSION_ENERGY_COMPONENTS};
use endf::mf::mf2::{ResonanceParameters, UnresolvedParameters};
use endf::mf::mf4::{AngleAtEnergy, AngleDistribution};
use endf::mf::mf5::EnergyDistribution;
use endf::mf::mf6::Distribution as Mf6Distribution;
use endf::univariate::Univariate;
use endf::AngleEnergy;
use endf::{materials_from_str, Material, Section, Tabulated1D, Tabulated2D};

/// Read a file, decompressing it when the name says it is compressed.
///
/// Both the fixtures and the golden dumps are stored xz-compressed: an
/// evaluation is highly repetitive and compresses about six to one, and the
/// dumps about seven. `lzma-rs` is a dev-dependency, so nothing that uses the
/// crate pays for it.
fn read_text(path: &Path) -> String {
    let raw = std::fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    if path.extension().is_some_and(|e| e == "xz") {
        let mut out = Vec::new();
        lzma_rs::xz_decompress(&mut raw.as_slice(), &mut out)
            .unwrap_or_else(|e| panic!("decompressing {}: {e}", path.display()));
        return String::from_utf8(out)
            .unwrap_or_else(|e| panic!("{} is not UTF-8: {e}", path.display()));
    }
    String::from_utf8(raw).unwrap_or_else(|e| panic!("{} is not UTF-8: {e}", path.display()))
}

/// Whether a path names a fixture of the given format, `.xz` or not.
fn has_kind(path: &Path, kind: &str) -> bool {
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let name = name.strip_suffix(".xz").unwrap_or(&name);
    std::path::Path::new(name)
        .extension()
        .is_some_and(|e| e == kind)
}

/// Mirrors `MAX_SAMPLES` in `tools/dump_golden.py`.
const MAX_SAMPLES: usize = 24;

/// Relative tolerance for sampled interpolation. Everything else is exact.
const EVAL_TOL: f64 = 1e-12;

#[derive(Debug, Clone, PartialEq)]
enum Value {
    Floats(Vec<f64>),
    Ints(Vec<i64>),
    Text(String),
}

impl Value {
    fn kind(&self) -> &'static str {
        match self {
            Value::Floats(_) => "F",
            Value::Ints(_) => "I",
            Value::Text(_) => "T",
        }
    }
}

#[derive(Default)]
struct Dump {
    map: BTreeMap<String, Value>,
}

impl Dump {
    fn put(&mut self, path: String, value: Value) {
        if let Some(old) = self.map.insert(path.clone(), value) {
            panic!("the Rust dump wrote {path} twice (previously {old:?})");
        }
    }

    fn floats(&mut self, path: String, values: Vec<f64>) {
        self.put(path, Value::Floats(values));
    }

    fn float(&mut self, path: String, value: f64) {
        self.put(path, Value::Floats(vec![value]));
    }

    fn ints(&mut self, path: String, values: Vec<i64>) {
        self.put(path, Value::Ints(values));
    }

    fn int(&mut self, path: String, value: i64) {
        self.put(path, Value::Ints(vec![value]));
    }

    fn text(&mut self, path: String, value: &str) {
        self.put(path, Value::Text(value.to_string()));
    }

    fn tab1(&mut self, path: &str, t: &Tabulated1D) {
        self.floats(format!("{path}/x"), t.x.clone());
        self.floats(format!("{path}/y"), t.y.clone());
        self.ints(
            format!("{path}/bp"),
            t.breakpoints.iter().map(|&v| v as i64).collect(),
        );
        self.ints(
            format!("{path}/int"),
            t.interpolation.iter().map(|&v| v as i64).collect(),
        );
        let points = sample_points(t);
        if !points.is_empty() {
            let values = points.iter().map(|&p| t.eval(p)).collect();
            self.floats(format!("{path}/evalx"), points);
            self.floats(format!("{path}/evaly"), values);
        }
    }

    fn tab2(&mut self, path: &str, t: &Tabulated2D) {
        self.ints(
            format!("{path}/bp"),
            t.breakpoints.iter().map(|&v| v as i64).collect(),
        );
        self.ints(
            format!("{path}/int"),
            t.interpolation.iter().map(|&v| v as i64).collect(),
        );
    }
}

/// Mirrors `sample_points` in `tools/dump_golden.py`, index for index. The
/// arithmetic is the same in both languages, so the abscissae come out
/// bit-identical and only the ordinates need a tolerance.
fn sample_points(t: &Tabulated1D) -> Vec<f64> {
    let x = &t.x;
    if x.len() < 2 {
        return x.clone();
    }
    let n_bins = x.len() - 1;

    let mut wanted: BTreeSet<usize> = BTreeSet::new();
    for &b in &t.breakpoints {
        for i in [b as i64 - 2, b as i64 - 1] {
            if i >= 0 && (i as usize) < n_bins {
                wanted.insert(i as usize);
            }
        }
    }
    let step = (n_bins / MAX_SAMPLES).max(1);
    wanted.extend((0..n_bins).step_by(step));

    let mut points = Vec::with_capacity(wanted.len() + 4);
    points.push(x[0] * 0.5);
    points.push(x[0]);
    points.extend(wanted.iter().map(|&i| 0.5 * (x[i] + x[i + 1])));
    points.push(x[x.len() - 1]);
    points.push(x[x.len() - 1] * 2.0);
    points
}

// --------------------------------------------------------------------------
// One dumper per ENDF file, mirroring tools/dump_golden.py.
// --------------------------------------------------------------------------

fn dump_nu(d: &mut Dump, path: &str, nu: &Nu) {
    match nu {
        Nu::Polynomial(c) => d.floats(format!("{path}/poly"), c.clone()),
        Nu::Tabulated(t) => d.tab1(&format!("{path}/tab"), t),
        Nu::Absent => {}
    }
}

fn dump_mf2_parameters(d: &mut Dump, rp: &str, p: &ResonanceParameters) {
    match p {
        ResonanceParameters::ScatteringRadius { spi, ap, nls } => {
            d.float(format!("{rp}/SPI"), *spi);
            d.float(format!("{rp}/AP"), *ap);
            d.int(format!("{rp}/NLS"), *nls);
        }
        ResonanceParameters::BreitWigner(b) => {
            if let Some(ape) = &b.ape {
                d.tab1(&format!("{rp}/APE"), ape);
            }
            d.float(format!("{rp}/SPI"), b.spi);
            d.float(format!("{rp}/AP"), b.ap);
            d.int(format!("{rp}/NLS"), b.nls);
            for (i, s) in b.sections.iter().enumerate() {
                let sp = format!("{rp}/sections/{i}");
                d.float(format!("{sp}/AWRI"), s.awri);
                d.float(format!("{sp}/QX"), s.qx);
                d.int(format!("{sp}/L"), s.l);
                d.int(format!("{sp}/LRX"), s.lrx);
                d.int(format!("{sp}/NRS"), s.nrs);
                d.floats(format!("{sp}/ER"), s.er.clone());
                d.floats(format!("{sp}/AJ"), s.aj.clone());
                d.floats(format!("{sp}/GT"), s.gt.clone());
                d.floats(format!("{sp}/GN"), s.gn.clone());
                d.floats(format!("{sp}/GG"), s.gg.clone());
                d.floats(format!("{sp}/GF"), s.gf.clone());
            }
        }
        ResonanceParameters::ReichMoore(r) => {
            if let Some(ape) = &r.ape {
                d.tab1(&format!("{rp}/APE"), ape);
            }
            d.float(format!("{rp}/SPI"), r.spi);
            d.float(format!("{rp}/AP"), r.ap);
            d.int(format!("{rp}/LAD"), r.lad);
            d.int(format!("{rp}/NLS"), r.nls);
            d.int(format!("{rp}/NLSC"), r.nlsc);
            for (i, s) in r.sections.iter().enumerate() {
                let sp = format!("{rp}/sections/{i}");
                d.float(format!("{sp}/AWRI"), s.awri);
                d.float(format!("{sp}/APL"), s.apl);
                d.int(format!("{sp}/L"), s.l);
                d.int(format!("{sp}/NRS"), s.nrs);
                d.floats(format!("{sp}/ER"), s.er.clone());
                d.floats(format!("{sp}/AJ"), s.aj.clone());
                d.floats(format!("{sp}/GN"), s.gn.clone());
                d.floats(format!("{sp}/GG"), s.gg.clone());
                d.floats(format!("{sp}/GFA"), s.gfa.clone());
                d.floats(format!("{sp}/GFB"), s.gfb.clone());
            }
        }
        ResonanceParameters::RMatrixLimited(r) => {
            d.int(format!("{rp}/IFG"), r.ifg);
            d.int(format!("{rp}/KRM"), r.krm);
            d.int(format!("{rp}/NJS"), r.njs);
            d.int(format!("{rp}/KRL"), r.krl);
            d.int(format!("{rp}/NPP"), r.npp);
            let pp = &r.particle_pairs;
            for (key, values) in [
                ("MA", &pp.ma),
                ("MB", &pp.mb),
                ("ZA", &pp.za),
                ("ZB", &pp.zb),
                ("IA", &pp.ia),
                ("IB", &pp.ib),
                ("Q", &pp.q),
                ("PNT", &pp.pnt),
                ("SHF", &pp.shf),
                ("MT", &pp.mt),
                ("PA", &pp.pa),
                ("PB", &pp.pb),
            ] {
                d.floats(format!("{rp}/particle_pairs/{key}"), values.clone());
            }
            for (i, g) in r.spin_groups.iter().enumerate() {
                let gp = format!("{rp}/spin_groups/{i}");
                d.float(format!("{gp}/AJ"), g.aj);
                d.float(format!("{gp}/PJ"), g.pj);
                d.int(format!("{gp}/KBK"), g.kbk);
                d.int(format!("{gp}/KPS"), g.kps);
                d.int(format!("{gp}/NCH"), g.nch);
                d.int(format!("{gp}/NRS"), g.nrs);
                d.int(format!("{gp}/NX"), g.nx);
                let ch = &g.channels;
                for (key, values) in [
                    ("PPI", &ch.ppi),
                    ("L", &ch.l),
                    ("SCH", &ch.sch),
                    ("BND", &ch.bnd),
                    ("APE", &ch.ape),
                    ("APT", &ch.apt),
                ] {
                    d.floats(format!("{gp}/channels/{key}"), values.clone());
                }
                d.floats(format!("{gp}/ER"), g.er.clone());
                for (c, row) in g.gam.iter().enumerate() {
                    d.floats(format!("{gp}/GAM/{c}"), row.clone());
                }
                for (key, value) in [("LCH", g.lch), ("LBK", g.lbk), ("LPS", g.lps)] {
                    if let Some(v) = value {
                        d.int(format!("{gp}/{key}"), v);
                    }
                }
                for (key, value) in [("ED", g.ed), ("EU", g.eu)] {
                    if let Some(v) = value {
                        d.float(format!("{gp}/{key}"), v);
                    }
                }
                for (key, table) in [
                    ("RBR", &g.rbr),
                    ("RBI", &g.rbi),
                    ("PSR", &g.psr),
                    ("PSI", &g.psi),
                ] {
                    if let Some(t) = table {
                        d.tab1(&format!("{gp}/{key}"), t);
                    }
                }
            }
        }
        ResonanceParameters::Unresolved(u) => {
            if let Some(ape) = &u.ape {
                d.tab1(&format!("{rp}/APE"), ape);
            }
            d.float(format!("{rp}/SPI"), u.spi);
            d.float(format!("{rp}/AP"), u.ap);
            d.int(format!("{rp}/LSSF"), u.lssf);
            d.int(format!("{rp}/NLS"), u.nls);
            if let Some(ne) = u.ne {
                d.int(format!("{rp}/NE"), ne);
                d.floats(format!("{rp}/ES"), u.es.clone());
            }
            for (i, r) in u.ranges.iter().enumerate() {
                let up = format!("{rp}/ranges/{i}");
                d.float(format!("{up}/AWRI"), r.awri);
                d.int(format!("{up}/L"), r.l);
                d.int(format!("{up}/NJS"), r.njs);
                if !r.d.is_empty() {
                    d.floats(format!("{up}/D"), r.d.clone());
                    d.floats(format!("{up}/AJ"), r.aj.clone());
                    d.floats(format!("{up}/AMUN"), r.amun.clone());
                    d.floats(format!("{up}/GNO"), r.gno.clone());
                    d.floats(format!("{up}/GG"), r.gg.clone());
                }
                for (j, p) in r.parameters.iter().enumerate() {
                    let pp = format!("{up}/parameters/{j}");
                    match p {
                        UnresolvedParameters::CaseB {
                            muf,
                            d: dd,
                            aj,
                            amun,
                            gn0,
                            gg,
                            gf,
                        } => {
                            d.int(format!("{pp}/MUF"), *muf);
                            d.float(format!("{pp}/D"), *dd);
                            d.float(format!("{pp}/AJ"), *aj);
                            d.float(format!("{pp}/AMUN"), *amun);
                            d.float(format!("{pp}/GN0"), *gn0);
                            d.float(format!("{pp}/GG"), *gg);
                            d.floats(format!("{pp}/GF"), gf.clone());
                        }
                        UnresolvedParameters::CaseC {
                            aj,
                            interpolation,
                            ne,
                            amux,
                            amun,
                            amuf,
                            e,
                            d: dd,
                            gx,
                            gn0,
                            gg,
                            gf,
                        } => {
                            d.float(format!("{pp}/AJ"), *aj);
                            d.int(format!("{pp}/INT"), *interpolation);
                            d.int(format!("{pp}/NE"), *ne);
                            d.float(format!("{pp}/AMUX"), *amux);
                            d.float(format!("{pp}/AMUN"), *amun);
                            d.float(format!("{pp}/AMUF"), *amuf);
                            d.floats(format!("{pp}/E"), e.clone());
                            d.floats(format!("{pp}/D"), dd.clone());
                            d.floats(format!("{pp}/GX"), gx.clone());
                            d.floats(format!("{pp}/GN0"), gn0.clone());
                            d.floats(format!("{pp}/GG"), gg.clone());
                            d.floats(format!("{pp}/GF"), gf.clone());
                        }
                    }
                }
            }
        }
        ResonanceParameters::Absent => {}
    }
}

fn dump_mf6_distribution(d: &mut Dump, dp: &str, dist: &Mf6Distribution) {
    match dist {
        Mf6Distribution::None => {}
        Mf6Distribution::ContinuumEnergyAngle(c) => {
            d.int(format!("{dp}/LANG"), c.lang);
            d.int(format!("{dp}/LEP"), c.lep);
            d.int(format!("{dp}/NR"), c.nr);
            d.int(format!("{dp}/NE"), c.ne);
            d.tab2(&format!("{dp}/E_int"), &c.e_int);
            d.floats(format!("{dp}/E"), c.energy.clone());
            for (j, s) in c.distribution.iter().enumerate() {
                let sp = format!("{dp}/distribution/{j}");
                d.int(format!("{sp}/ND"), s.nd);
                d.int(format!("{sp}/NA"), s.na);
                d.int(format!("{sp}/NW"), s.nw);
                d.int(format!("{sp}/NEP"), s.nep);
                d.floats(format!("{sp}/Eout"), s.e_out.clone());
                for (r, row) in s.b.iter().enumerate() {
                    d.floats(format!("{sp}/b/{r}"), row.clone());
                }
            }
        }
        Mf6Distribution::DiscreteTwoBody(t) => {
            d.int(format!("{dp}/NR"), t.nr);
            d.int(format!("{dp}/NE"), t.ne);
            d.tab2(&format!("{dp}/E_int"), &t.e_int);
            d.floats(format!("{dp}/E"), t.energy.clone());
            for (j, s) in t.distribution.iter().enumerate() {
                let sp = format!("{dp}/distribution/{j}");
                d.int(format!("{sp}/LANG"), s.lang);
                d.int(format!("{sp}/NW"), s.nw);
                d.int(format!("{sp}/NL"), s.nl);
                d.floats(format!("{sp}/A_l"), s.a_l.clone());
            }
        }
        Mf6Distribution::ChargedParticleElastic(c) => {
            d.float(format!("{dp}/SPI"), c.spi);
            d.int(format!("{dp}/LIDP"), c.lidp);
            d.int(format!("{dp}/NE"), c.ne);
            d.tab2(&format!("{dp}/E_int"), &c.e_int);
            for (j, s) in c.distribution.iter().enumerate() {
                let sp = format!("{dp}/distribution/{j}");
                d.float(format!("{sp}/E"), s.energy);
                d.int(format!("{sp}/LTP"), s.ltp);
                d.int(format!("{sp}/NW"), s.nw);
                d.int(format!("{sp}/NL"), s.nl);
                d.floats(format!("{sp}/A"), s.a.clone());
            }
        }
        Mf6Distribution::NBodyPhaseSpace { apsx, npsx } => {
            d.float(format!("{dp}/APSX"), *apsx);
            d.int(format!("{dp}/NPSX"), *npsx);
        }
        Mf6Distribution::LaboratoryAngleEnergy(l) => {
            d.int(format!("{dp}/NE"), l.ne);
            d.tab2(&format!("{dp}/E_int"), &l.e_int);
            for (j, s) in l.distribution.iter().enumerate() {
                let sp = format!("{dp}/distribution/{j}");
                d.float(format!("{sp}/E"), s.energy);
                d.int(format!("{sp}/NRM"), s.nrm);
                d.int(format!("{sp}/NMU"), s.nmu);
                d.tab2(&format!("{sp}/mu_int"), &s.mu_int);
                for (k, entry) in s.mu.iter().enumerate() {
                    d.float(format!("{sp}/mu/{k}/mu"), entry.mu);
                    d.tab1(&format!("{sp}/mu/{k}/f"), &entry.f);
                }
            }
        }
    }
}

/// One MF=33 subsection. Shared with MF=40, which reuses the format.
fn dump_mf33_subsection(d: &mut Dump, sp: &str, sub: &Mf33Subsection) {
    d.float(format!("{sp}/XMF1"), sub.xmf1);
    d.float(format!("{sp}/XLFS1"), sub.xlfs1);
    for (key, value) in [
        ("MAT1", sub.mat1),
        ("MT1", sub.mt1),
        ("NC", sub.nc),
        ("NI", sub.ni),
    ] {
        d.int(format!("{sp}/{key}"), value);
    }
    for (i, nc) in sub.nc_subsections.iter().enumerate() {
        let np = format!("{sp}/nc/{i}");
        d.int(format!("{np}/LTY"), nc.lty);
        d.float(format!("{np}/E1"), nc.e1);
        d.float(format!("{np}/E2"), nc.e2);
        if nc.lty == 0 {
            d.int(format!("{np}/NCI"), nc.nci);
            d.floats(format!("{np}/CI"), nc.ci.clone());
            d.floats(format!("{np}/XMTI"), nc.xmti.clone());
        } else {
            d.int(format!("{np}/MATS"), nc.mats);
            d.int(format!("{np}/MTS"), nc.mts);
            d.int(format!("{np}/NEI"), nc.nei);
            d.float(format!("{np}/XMFS"), nc.xmfs);
            d.float(format!("{np}/XLFSS"), nc.xlfss);
            d.floats(format!("{np}/EI"), nc.ei.clone());
            d.floats(format!("{np}/WEI"), nc.wei.clone());
        }
    }
    for (i, ni) in sub.ni_subsections.iter().enumerate() {
        let ip = format!("{sp}/ni/{i}");
        d.int(format!("{ip}/LB"), ni.lb);
        d.int(format!("{ip}/NT"), ni.nt);
        match ni.lb {
            0..=4 => {
                d.int(format!("{ip}/LT"), ni.lt);
                d.int(format!("{ip}/NP"), ni.np);
                d.floats(format!("{ip}/Ek"), ni.ek.clone());
                d.floats(format!("{ip}/Fk"), ni.fk.clone());
                d.floats(format!("{ip}/El"), ni.el.clone());
                d.floats(format!("{ip}/Fl"), ni.fl.clone());
            }
            5 => {
                d.int(format!("{ip}/LS"), ni.ls);
                d.int(format!("{ip}/NE"), ni.ne);
                d.floats(format!("{ip}/Ek"), ni.ek.clone());
                d.floats(format!("{ip}/Fkk"), ni.fkk.clone());
            }
            6 => {
                d.int(format!("{ip}/NER"), ni.ner);
                d.int(format!("{ip}/NEC"), ni.nec);
                d.floats(format!("{ip}/ER"), ni.er.clone());
                d.floats(format!("{ip}/EC"), ni.ec.clone());
                d.floats(format!("{ip}/Fkl"), ni.fkl.clone());
            }
            _ => {
                d.int(format!("{ip}/LT"), ni.lt);
                d.int(format!("{ip}/NP"), ni.np);
                d.floats(format!("{ip}/Ek"), ni.ek.clone());
                d.floats(format!("{ip}/Fk"), ni.fk.clone());
            }
        }
    }
}

/// Mirrors `dump_angle_distribution` in `tools/dump_golden.py`.
fn dump_angle_distribution(d: &mut Dump, path: &str, dist: &AngleDistribution) {
    d.floats(format!("{path}/energy"), dist.energy.clone());
    d.int(format!("{path}/n_mu"), dist.mu.len() as i64);
    for (i, mu) in dist.mu.iter().enumerate() {
        let p = format!("{path}/mu/{i}");
        match mu {
            AngleAtEnergy::Legendre(l) => {
                d.text(format!("{p}/kind"), "legendre");
                d.floats(format!("{p}/coef"), l.coefficients.clone());
            }
            AngleAtEnergy::Tabulated(f) => {
                d.text(format!("{p}/kind"), "tabulated");
                d.tab1(&format!("{p}/f"), f);
            }
            AngleAtEnergy::Tabular(t) => {
                dump_univariate(d, &p, &Univariate::Tabular(t.clone()));
            }
            AngleAtEnergy::Isotropic(u) => {
                dump_univariate(d, &p, &Univariate::Uniform(u.clone()));
            }
        }
    }
}

/// Mirrors `dump_reactions` in `tools/dump_golden.py`.
fn dump_reactions(d: &mut Dump, path: &str, material: &Material) {
    let mts: Vec<i32> = material
        .section_data
        .keys()
        .filter(|&&(mf, _)| mf == 3)
        .map(|&(_, mt)| mt)
        .collect();
    d.ints(
        format!("{path}/mts"),
        mts.iter().map(|&mt| mt as i64).collect(),
    );
    for mt in mts {
        let rx = endf::Reaction::from_endf(mt, material).unwrap();
        dump_reaction(d, &format!("{path}/{mt}"), &rx);
    }
}

/// Mirrors `dump_reaction` in `tools/dump_golden.py`.
fn dump_reaction(d: &mut Dump, path: &str, rx: &endf::Reaction) {
    d.int(format!("{path}/MT"), rx.mt as i64);
    d.float(format!("{path}/q_reaction"), rx.q_reaction);
    d.float(format!("{path}/q_massdiff"), rx.q_massdiff);
    d.int(format!("{path}/redundant"), i64::from(rx.redundant));
    d.int(
        format!("{path}/center_of_mass"),
        i64::from(rx.center_of_mass),
    );
    for (temperature, xs) in &rx.xs {
        d.tab1(&format!("{path}/xs/{temperature}"), xs);
    }
    for (kind, products) in [
        ("products", &rx.products),
        ("derived_products", &rx.derived_products),
    ] {
        d.int(format!("{path}/n_{kind}"), products.len() as i64);
        for (i, product) in products.iter().enumerate() {
            dump_product(d, &format!("{path}/{kind}/{i}"), product);
        }
    }
}

/// Mirrors `dump_product` in `tools/dump_golden.py`.
fn dump_product(d: &mut Dump, path: &str, product: &endf::Product) {
    d.text(format!("{path}/name"), &product.name);
    d.text(
        format!("{path}/emission_mode"),
        product.emission_mode.name(),
    );
    d.float(format!("{path}/decay_rate"), product.decay_rate);
    match &product.yield_ {
        endf::Yield::Tabulated(t) => {
            d.text(format!("{path}/yield/kind"), "tabulated");
            d.tab1(&format!("{path}/yield/f"), t);
        }
        endf::Yield::Polynomial(p) => {
            d.text(format!("{path}/yield/kind"), "polynomial");
            d.floats(format!("{path}/yield/coef"), p.coefficients.clone());
        }
    }
    for (i, applicability) in product.applicability.iter().enumerate() {
        d.tab1(&format!("{path}/applicability/{i}"), applicability);
    }
    d.int(
        format!("{path}/n_distribution"),
        product.distribution.len() as i64,
    );
    for (i, dist) in product.distribution.iter().enumerate() {
        dump_angle_energy(d, &format!("{path}/distribution/{i}"), dist);
    }
}

/// Mirrors `dump_radionuclide_production` in `tools/dump_golden.py`.
fn dump_radionuclide_production(d: &mut Dump, path: &str, material: &Material) {
    let production = endf::radionuclide_production(material);
    d.ints(
        format!("{path}/mts"),
        production.keys().map(|&mt| mt as i64).collect(),
    );
    for (mt, states) in &production {
        for (i, state) in states.iter().enumerate() {
            let sp = format!("{path}/{mt}/{i}");
            d.int(format!("{sp}/ZAP"), state.zap);
            d.int(format!("{sp}/LFS"), state.lfs);
            d.float(format!("{sp}/QM"), state.qm);
            d.float(format!("{sp}/QI"), state.qi);
            if let Some(elfs) = state.elfs {
                d.float(format!("{sp}/ELFS"), elfs);
            }
            d.float(format!("{sp}/excitation_energy"), state.excitation_energy());
            if let Some(y) = &state.yields {
                d.tab1(&format!("{sp}/yields"), y);
            }
            if let Some(xs) = &state.cross_section {
                d.tab1(&format!("{sp}/cross_section"), xs);
            }
        }
    }
}

/// Mirrors `dump_univariate` in `tools/dump_golden.py`.
fn dump_univariate(d: &mut Dump, p: &str, u: &Univariate) {
    let c = match u {
        Univariate::Discrete(t) => {
            d.text(format!("{p}/kind"), "discrete");
            d.floats(format!("{p}/x"), t.x.clone());
            d.floats(format!("{p}/p"), t.p.clone());
            d.floats(format!("{p}/cdf"), t.cdf());
            &t.c
        }
        Univariate::Tabular(t) => {
            d.text(format!("{p}/kind"), "tabular");
            d.text(format!("{p}/interpolation"), t.interpolation.name());
            d.floats(format!("{p}/x"), t.x.clone());
            d.floats(format!("{p}/p"), t.p.clone());
            d.floats(format!("{p}/cdf"), t.cdf());
            &t.c
        }
        Univariate::Uniform(t) => {
            d.text(format!("{p}/kind"), "uniform");
            d.float(format!("{p}/a"), t.a);
            d.float(format!("{p}/b"), t.b);
            &None
        }
        Univariate::Mixture(m) => {
            d.text(format!("{p}/kind"), "mixture");
            d.floats(format!("{p}/probability"), m.probability.clone());
            for (j, sub) in m.distribution.iter().enumerate() {
                dump_univariate(d, &format!("{p}/distribution/{j}"), sub);
            }
            &None
        }
    };
    // The CDF as the file gave it, where there was one.
    if let Some(c) = c {
        d.floats(format!("{p}/c"), c.clone());
    }
}

/// Mirrors `dump_energy_distribution` in `tools/dump_golden.py`.
fn dump_energy_distribution(d: &mut Dump, p: &str, dist: &EnergyDistribution) {
    match dist {
        EnergyDistribution::ArbitraryTabulated { energy, g, .. } => {
            d.text(format!("{p}/kind"), "arbitrary-tabulated");
            d.floats(format!("{p}/E"), energy.clone());
            for (j, t) in g.iter().enumerate() {
                d.tab1(&format!("{p}/g/{j}"), t);
            }
        }
        EnergyDistribution::GeneralEvaporation { u, theta, g } => {
            d.text(format!("{p}/kind"), "general-evaporation");
            d.float(format!("{p}/U"), *u);
            d.tab1(&format!("{p}/theta"), theta);
            d.tab1(&format!("{p}/g"), g);
        }
        EnergyDistribution::MaxwellEnergy { u, theta } => {
            d.text(format!("{p}/kind"), "maxwell");
            d.float(format!("{p}/U"), *u);
            d.tab1(&format!("{p}/theta"), theta);
        }
        EnergyDistribution::Evaporation { u, theta } => {
            d.text(format!("{p}/kind"), "evaporation");
            d.float(format!("{p}/U"), *u);
            d.tab1(&format!("{p}/theta"), theta);
        }
        EnergyDistribution::WattEnergy { u, a, b } => {
            d.text(format!("{p}/kind"), "watt");
            d.float(format!("{p}/U"), *u);
            d.tab1(&format!("{p}/a"), a);
            d.tab1(&format!("{p}/b"), b);
        }
        EnergyDistribution::MadlandNix { efl, efh, t_m } => {
            d.text(format!("{p}/kind"), "madland-nix");
            d.float(format!("{p}/EFL"), *efl);
            d.float(format!("{p}/EFH"), *efh);
            d.tab1(&format!("{p}/T_M"), t_m);
        }
        EnergyDistribution::LevelInelastic {
            threshold,
            mass_ratio,
        } => {
            d.text(format!("{p}/kind"), "level-inelastic");
            d.float(format!("{p}/threshold"), *threshold);
            d.float(format!("{p}/mass_ratio"), *mass_ratio);
        }
        EnergyDistribution::DiscretePhoton {
            primary_flag,
            energy,
            atomic_weight_ratio,
        } => {
            d.text(format!("{p}/kind"), "discrete-photon");
            d.int(format!("{p}/primary_flag"), *primary_flag);
            d.float(format!("{p}/energy"), *energy);
            d.float(format!("{p}/atomic_weight_ratio"), *atomic_weight_ratio);
        }
        EnergyDistribution::ContinuousTabular {
            breakpoints,
            interpolation,
            energy,
            energy_out,
        } => {
            d.text(format!("{p}/kind"), "continuous-tabular");
            d.ints(
                format!("{p}/bp"),
                breakpoints.iter().map(|&v| v as i64).collect(),
            );
            d.ints(
                format!("{p}/int"),
                interpolation.iter().map(|&v| v as i64).collect(),
            );
            d.floats(format!("{p}/E"), energy.clone());
            for (j, eout) in energy_out.iter().enumerate() {
                dump_univariate(d, &format!("{p}/energy_out/{j}"), eout);
            }
        }
    }
}

/// Mirrors `dump_angle_energy` in `tools/dump_golden.py`.
fn dump_angle_energy(d: &mut Dump, p: &str, ae: &AngleEnergy) {
    match ae {
        AngleEnergy::Uncorrelated(u) => {
            d.text(format!("{p}/kind"), "uncorrelated");
            if let Some(angle) = &u.angle {
                dump_angle_distribution(d, &format!("{p}/angle"), angle);
            }
            if let Some(energy) = &u.energy {
                dump_energy_distribution(d, &format!("{p}/energy"), energy);
            }
        }
        AngleEnergy::KalbachMann(k) => {
            d.text(format!("{p}/kind"), "kalbach-mann");
            d.ints(
                format!("{p}/bp"),
                k.breakpoints.iter().map(|&v| v as i64).collect(),
            );
            d.ints(
                format!("{p}/int"),
                k.interpolation.iter().map(|&v| v as i64).collect(),
            );
            d.floats(format!("{p}/E"), k.energy.clone());
            for (j, eout) in k.energy_out.iter().enumerate() {
                dump_univariate(d, &format!("{p}/energy_out/{j}"), eout);
            }
            for (j, r) in k.precompound.iter().enumerate() {
                d.tab1(&format!("{p}/precompound/{j}"), r);
            }
            for (j, a) in k.slope.iter().enumerate() {
                d.tab1(&format!("{p}/slope/{j}"), a);
            }
        }
        AngleEnergy::Correlated(c) => {
            d.text(format!("{p}/kind"), "correlated");
            d.ints(
                format!("{p}/bp"),
                c.breakpoints.iter().map(|&v| v as i64).collect(),
            );
            d.ints(
                format!("{p}/int"),
                c.interpolation.iter().map(|&v| v as i64).collect(),
            );
            d.floats(format!("{p}/E"), c.energy.clone());
            for (j, eout) in c.energy_out.iter().enumerate() {
                dump_univariate(d, &format!("{p}/energy_out/{j}"), eout);
            }
            for (j, mu_j) in c.mu.iter().enumerate() {
                for (k, mu_jk) in mu_j.iter().enumerate() {
                    dump_univariate(d, &format!("{p}/mu/{j}/{k}"), mu_jk);
                }
            }
        }
        AngleEnergy::NBodyPhaseSpace(n) => {
            d.text(format!("{p}/kind"), "nbody");
            d.float(format!("{p}/total_mass"), n.total_mass);
            d.int(format!("{p}/n_particles"), n.n_particles);
            d.float(format!("{p}/atomic_weight_ratio"), n.atomic_weight_ratio);
            d.float(format!("{p}/q_value"), n.q_value);
        }
    }
}

/// Mirrors `dump_incident_neutron_ace` in `tools/dump_golden.py`.
fn dump_incident_neutron_ace(d: &mut Dump, path: &str, t: &ace::Table) {
    // ESZ (JXS(1)) is the energy grid and MTR (JXS(3)) the reaction list;
    // without them there is no nuclide to build.
    if t.data_type().ok() != Some(ace::TableType::NeutronContinuous) {
        return;
    }
    if t.jxs[1] <= 0 || t.jxs[3] <= 0 {
        return;
    }
    let n = endf::IncidentNeutron::from_ace(t, endf::ace::MetastableScheme::Mcnp).unwrap();

    d.text(format!("{path}/name"), &n.name());
    d.int(format!("{path}/atomic_number"), n.atomic_number as i64);
    d.int(format!("{path}/mass_number"), n.mass_number as i64);
    d.int(format!("{path}/metastable"), n.metastable as i64);
    d.float(
        format!("{path}/atomic_weight_ratio"),
        n.atomic_weight_ratio.unwrap(),
    );
    d.floats(format!("{path}/kTs"), n.k_ts.clone());
    for (i, temperature) in n.temperatures().iter().enumerate() {
        d.text(format!("{path}/temperatures/{i}"), temperature);
        d.floats(
            format!("{path}/energy/{temperature}"),
            n.energy[temperature].clone(),
        );
        if let Some(urr) = n.urr.get(temperature) {
            d.floats(
                format!("{path}/urr/{temperature}/energy"),
                urr.energy.clone(),
            );
        }
    }

    let mts: Vec<i32> = n.reactions.keys().copied().collect();
    d.ints(
        format!("{path}/mts"),
        mts.iter().map(|&m| m as i64).collect(),
    );
    d.ints(
        format!("{path}/redundant"),
        mts.iter()
            .map(|m| i64::from(n.reactions[m].redundant))
            .collect(),
    );
    for mt in &mts {
        d.ints(
            format!("{path}/components/{mt}"),
            n.reaction_components(*mt)
                .iter()
                .map(|&m| m as i64)
                .collect(),
        );
    }

    // The removal cross section is deliberately not dumped here. It folds the
    // elastic angular distribution into the total, and for ACE data the Python
    // `forward_fraction` returns uninitialized memory (see issue #21) so
    // there is nothing stable to compare against. It is dumped on the ENDF
    // path, where the answer is well defined.

    // Only the reactions this type synthesises are dumped in full; the ones
    // `Reaction::from_ace` builds are compared elsewhere.
    let mut from_ace_mts: BTreeSet<i32> = BTreeSet::from([2]);
    from_ace_mts.extend((1..=t.nxs[4]).map(|i| t.xss[(t.jxs[3] + i - 1) as usize] as i32));
    for mt in mts.iter().filter(|m| !from_ace_mts.contains(m)) {
        dump_reaction(d, &format!("{path}/synthesised/{mt}"), &n.reactions[mt]);
    }
}

/// Mirrors `dump_chain` in `tools/dump_chain_golden.py`.
fn dump_chain(d: &mut Dump, path: &str, chain: &endf::Chain) {
    d.int(format!("{path}/n"), chain.nuclides.len() as i64);
    for (i, nuclide) in chain.nuclides.iter().enumerate() {
        let np = format!("{path}/{i}");
        d.text(format!("{np}/name"), &nuclide.name);
        if let Some(half_life) = nuclide.half_life {
            d.float(format!("{np}/half_life"), half_life);
        }
        d.float(format!("{np}/decay_energy"), nuclide.decay_energy);
        for (j, mode) in nuclide.decay_modes.iter().enumerate() {
            let mp = format!("{np}/decay/{j}");
            d.text(format!("{mp}/type"), &mode.kind);
            if let Some(target) = &mode.target {
                d.text(format!("{mp}/target"), target);
            }
            d.float(format!("{mp}/branching_ratio"), mode.branching_ratio);
        }
        for (j, rx) in nuclide.reactions.iter().enumerate() {
            let rp = format!("{np}/reaction/{j}");
            d.text(format!("{rp}/type"), &rx.kind);
            if let Some(target) = &rx.target {
                d.text(format!("{rp}/target"), target);
            }
            d.float(format!("{rp}/Q"), rx.q_value);
            d.float(format!("{rp}/branching_ratio"), rx.branching_ratio);
        }
    }
}

/// Mirrors `dump_incident_photon` in `tools/dump_golden.py`.
fn dump_incident_photon(d: &mut Dump, path: &str, material: &Material) {
    let has_photoatomic = material.section_data.keys().any(|&(mf, _)| mf == 23);
    if has_photoatomic {
        let n = endf::IncidentPhoton::from_endf(material, None).unwrap();
        d.int(format!("{path}/atomic_number"), n.atomic_number);
        d.text(format!("{path}/name"), n.name());
        let mts: Vec<i32> = n.reactions.keys().copied().collect();
        d.ints(
            format!("{path}/mts"),
            mts.iter().map(|&m| m as i64).collect(),
        );
        for mt in &mts {
            let rx = &n.reactions[mt];
            let rp = format!("{path}/{mt}");
            if let Some(name) = rx.name() {
                d.text(format!("{rp}/name"), name);
            }
            for (key, value) in [
                ("xs", &rx.xs),
                ("scattering_factor", &rx.scattering_factor),
                ("anomalous_real", &rx.anomalous_real),
                ("anomalous_imag", &rx.anomalous_imag),
            ] {
                if let Some(value) = value {
                    d.tab1(&format!("{rp}/{key}"), value);
                }
            }
            if let Some(value) = rx.subshell_binding_energy {
                d.float(format!("{rp}/subshell_binding_energy"), value);
            }
            if let Some(value) = rx.fluorescence_yield {
                d.float(format!("{rp}/fluorescence_yield"), value);
            }
            d.ints(
                format!("{rp}/components"),
                n.reaction_components(*mt)
                    .iter()
                    .map(|&m| m as i64)
                    .collect(),
            );
        }
    }

    if material.mf28(533).is_some() {
        let relaxation = endf::AtomicRelaxation::from_endf(material).unwrap();
        dump_atomic_relaxation(d, &format!("{path}/relaxation"), &relaxation);
    }
}

/// Mirrors `dump_atomic_relaxation` in `tools/dump_golden.py`.
fn dump_atomic_relaxation(d: &mut Dump, path: &str, relaxation: &endf::AtomicRelaxation) {
    for (i, shell) in relaxation.subshells().iter().enumerate() {
        d.text(format!("{path}/subshells/{i}"), shell);
    }
    for (shell, value) in &relaxation.binding_energy {
        d.float(format!("{path}/binding_energy/{shell}"), *value);
    }
    for (shell, value) in &relaxation.num_electrons {
        d.float(format!("{path}/num_electrons/{shell}"), *value);
    }
    for (shell, t) in &relaxation.transitions {
        let tp = format!("{path}/transitions/{shell}");
        for (i, s) in t.secondary_subshell.iter().enumerate() {
            d.text(format!("{tp}/secondary/{i}"), s);
        }
        for (i, s) in t.tertiary_subshell.iter().enumerate() {
            d.text(format!("{tp}/tertiary/{i}"), s);
        }
        d.floats(format!("{tp}/energy"), t.energy.clone());
        d.floats(format!("{tp}/probability"), t.probability.clone());
    }
}

/// Mirrors `dump_decay` in `tools/dump_golden.py`.
fn dump_decay(d: &mut Dump, path: &str, material: &Material) {
    if material.mf8_mt457().is_some() {
        dump_decay_section(d, path, &endf::Decay::from_material(material).unwrap());
    }
    if material.mf8_mt454(454).is_some() || material.mf8_mt454(459).is_some() {
        let fpy = endf::FissionProductYields::from_material(material).unwrap();
        d.floats(format!("{path}/fpy/energies"), fpy.energies.clone());
        for (kind, sets) in [
            ("independent", &fpy.independent),
            ("cumulative", &fpy.cumulative),
        ] {
            for (i, yields) in sets.iter().enumerate() {
                // Sorted by name, as the Python dumper walks a dict.
                let mut yields = yields.clone();
                yields.sort_by(|a, b| a.name.cmp(&b.name));
                for (j, product) in yields.iter().enumerate() {
                    let yp = format!("{path}/fpy/{kind}/{i}/{j}");
                    d.text(format!("{yp}/name"), &product.name);
                    d.floats(
                        format!("{yp}/yield"),
                        vec![product.yield_.0, product.yield_.1],
                    );
                }
            }
        }
    }
}

/// Mirrors `dump_decay_section` in `tools/dump_golden.py`.
fn dump_decay_section(d: &mut Dump, path: &str, decay: &endf::Decay) {
    let pair = |v: (f64, f64)| vec![v.0, v.1];
    let n = &decay.nuclide;
    d.text(format!("{path}/name"), &n.name);
    d.int(format!("{path}/atomic_number"), n.atomic_number);
    d.int(format!("{path}/mass_number"), n.mass_number);
    d.int(format!("{path}/isomeric_state"), n.isomeric_state);
    d.int(format!("{path}/excited_state"), n.excited_state);
    d.float(format!("{path}/mass"), n.mass);
    d.int(format!("{path}/stable"), i64::from(n.stable));
    if let Some(spin) = n.spin {
        d.float(format!("{path}/spin"), spin);
    }
    d.float(format!("{path}/parity"), n.parity);

    if !n.stable {
        d.floats(format!("{path}/half_life"), pair(decay.half_life.unwrap()));
        // A half-life of zero means it was not evaluated, and both readers
        // return nothing for it rather than dividing by it (issue #23). Dumped
        // either way, so the absence is compared rather than skipped on both
        // sides at once.
        d.floats(
            format!("{path}/decay_constant"),
            match decay.decay_constant() {
                Some(constant) => pair(constant).to_vec(),
                None => Vec::new(),
            },
        );
    }
    d.floats(format!("{path}/decay_energy"), pair(decay.decay_energy()));
    for (key, value) in &decay.average_energies {
        d.floats(format!("{path}/average_energies/{key}"), pair(*value));
    }

    for (i, mode) in decay.modes.iter().enumerate() {
        let mp = format!("{path}/modes/{i}");
        d.text(format!("{mp}/parent"), &mode.parent);
        d.text(format!("{mp}/modes"), &mode.modes.join(","));
        d.text(format!("{mp}/daughter"), &mode.daughter().unwrap());
        d.floats(format!("{mp}/energy"), pair(mode.energy));
        d.floats(format!("{mp}/branching_ratio"), pair(mode.branching_ratio));
    }

    for (radiation, spectrum) in &decay.spectra {
        let sp = format!("{path}/spectra/{radiation}");
        d.text(
            format!("{sp}/continuous_flag"),
            spectrum.continuous_flag.name(),
        );
        d.floats(
            format!("{sp}/discrete_normalization"),
            pair(spectrum.discrete_normalization),
        );
        d.floats(
            format!("{sp}/energy_average"),
            pair(spectrum.energy_average),
        );
        d.floats(
            format!("{sp}/continuous_normalization"),
            pair(spectrum.continuous_normalization),
        );
        for (j, line) in spectrum.discrete.iter().enumerate() {
            let lp = format!("{sp}/discrete/{j}");
            d.floats(format!("{lp}/energy"), pair(line.energy));
            d.text(format!("{lp}/from_mode"), &line.from_mode.join(","));
            if let Some(kind) = line.transition_type {
                d.text(format!("{lp}/type"), kind);
            }
            d.floats(format!("{lp}/intensity"), pair(line.intensity));
            for (key, value) in [
                ("positron_intensity", line.positron_intensity),
                ("internal_pair", line.internal_pair),
                ("total_internal_conversion", line.total_internal_conversion),
                ("k_shell_conversion", line.k_shell_conversion),
                ("l_shell_conversion", line.l_shell_conversion),
            ] {
                if let Some(value) = value {
                    d.floats(format!("{lp}/{key}"), pair(value));
                }
            }
        }
        if let Some(continuous) = &spectrum.continuous {
            d.text(
                format!("{sp}/continuous_from_mode"),
                &spectrum.continuous_from_mode.join(","),
            );
            d.tab1(&format!("{sp}/continuous"), continuous);
        }
    }

    for (particle, dist) in decay.sources().unwrap() {
        dump_univariate(d, &format!("{path}/sources/{particle}"), &dist);
    }
}

/// Mirrors `dump_incident_neutron_endf` in `tools/dump_golden.py`.
fn dump_incident_neutron_endf(d: &mut Dump, path: &str, material: &Material) {
    if material.mf1_mt451().is_none() {
        return;
    }
    let n = endf::IncidentNeutron::from_endf(material).unwrap();
    d.text(format!("{path}/name"), &n.name());
    d.int(format!("{path}/atomic_number"), n.atomic_number as i64);
    d.int(format!("{path}/mass_number"), n.mass_number as i64);
    d.int(format!("{path}/metastable"), n.metastable as i64);
    d.text(format!("{path}/atomic_symbol"), n.atomic_symbol());
    let mts: Vec<i32> = n.reactions.keys().copied().collect();
    d.ints(
        format!("{path}/mts"),
        mts.iter().map(|&m| m as i64).collect(),
    );
    for mt in &mts {
        d.ints(
            format!("{path}/components/{mt}"),
            n.reaction_components(*mt)
                .iter()
                .map(|&m| m as i64)
                .collect(),
        );
    }

    // The removal cross section, which folds the elastic angular distribution
    // into the total. Several cutoffs, since each picks a different slice of
    // the forward cone.
    if n.contains(1) && n.contains(2) {
        for cutoff in [-1.0, 0.0, 0.5] {
            let name = format!("{cutoff:+.1}");
            d.tab1(
                &format!("{path}/removal_xs/{name}"),
                &n.removal_xs("0K", cutoff).unwrap(),
            );
        }
    }
}

/// Mirrors `dump_ace_reactions` in `tools/dump_golden.py`.
fn dump_ace_reactions(d: &mut Dump, path: &str, t: &ace::Table) {
    // MTR (JXS(3)) lists the reactions; without it there are none to read.
    if t.data_type().ok() != Some(ace::TableType::NeutronContinuous) || t.jxs[3] <= 0 {
        return;
    }
    let n = t.nxs[4];
    d.int(format!("{path}/n"), n);
    for i_reaction in 0..=n {
        let rx = endf::Reaction::from_ace(t, i_reaction).unwrap();
        dump_reaction(d, &format!("{path}/{i_reaction}"), &rx);
    }
}

/// Mirrors `dump_ace_dlw` in `tools/dump_golden.py`.
fn dump_ace_dlw(d: &mut Dump, path: &str, t: &ace::Table) {
    if t.data_type().ok() != Some(ace::TableType::NeutronContinuous) {
        return;
    }
    let (ldlw, dlw) = (t.jxs[10], t.jxs[11]);
    if ldlw <= 0 {
        return;
    }
    let at = |i: i64| t.xss.get(i as usize).copied().unwrap_or(0.0);

    let n = t.nxs[5];
    d.int(format!("{path}/n"), n);
    for i_reaction in 1..=n {
        let rp = format!("{path}/{i_reaction}");
        let mut lnw = at(ldlw + i_reaction - 1) as i64;
        let mut chain: Vec<i64> = Vec::new();
        while lnw > 0 {
            let k = chain.len();
            chain.push(lnw);
            d.tab1(
                &format!("{rp}/{k}/applicability"),
                &Tabulated1D::from_ace(&t.xss, (dlw + lnw + 2).max(0) as usize, true),
            );
            // Law 66 wants the reaction's Q value; the Python dumper passes a
            // fixed stand-in, so this passes the same one.
            let ae = AngleEnergy::from_ace(t, dlw, lnw, Some(0.0)).unwrap();
            dump_angle_energy(d, &format!("{rp}/{k}"), &ae);
            lnw = at(dlw + lnw - 1) as i64;
        }
        d.ints(format!("{rp}/chain"), chain);
    }
}

/// Mirrors `dump_ace_angle` in `tools/dump_golden.py`.
fn dump_ace_angle(d: &mut Dump, path: &str, t: &ace::Table) {
    if t.data_type().ok() != Some(ace::TableType::NeutronContinuous) {
        return;
    }
    let (land, and) = (t.jxs[8], t.jxs[9]);
    if land <= 0 {
        return;
    }

    let n = t.nxs[5] + 1;
    d.int(format!("{path}/n"), n);
    let locators: Vec<i64> = (0..n).map(|i| t.xss[(land + i) as usize] as i64).collect();
    d.ints(format!("{path}/locators"), locators.clone());
    for (i, &locator) in locators.iter().enumerate() {
        if locator <= 0 {
            continue;
        }
        let dist = AngleDistribution::from_ace(t, and, locator).unwrap();
        dump_angle_distribution(d, &format!("{path}/{i}"), &dist);
    }
}

/// Mirrors `ACE_XSS_SAMPLES` in `tools/dump_golden.py`.
const ACE_XSS_SAMPLES: usize = 2000;

/// Mirrors `ace_xss_indices` in `tools/dump_golden.py`, index for index.
fn ace_xss_indices(n: usize, jxs: &[i64]) -> Vec<usize> {
    let mut idx: BTreeSet<usize> = (0..n).step_by((n / ACE_XSS_SAMPLES).max(1)).collect();
    idx.extend(0..50.min(n));
    idx.extend(n.saturating_sub(50)..n);
    // The JXS values are offsets into XSS: where a consumer actually looks.
    for &j in jxs {
        if j >= 0 && (j as usize) < n {
            idx.insert(j as usize);
        }
    }
    idx.into_iter().collect()
}

fn dump_ace_table(d: &mut Dump, path: &str, t: &ace::Table) {
    d.text(format!("{path}/name"), &t.name);
    d.float(format!("{path}/atomic_weight_ratio"), t.atomic_weight_ratio);
    d.float(format!("{path}/kT"), t.kt);
    d.float(format!("{path}/temperature"), t.temperature());
    d.int(format!("{path}/zaid"), t.zaid().unwrap());
    d.text(
        format!("{path}/data_type"),
        &t.data_type().unwrap().suffix().to_string(),
    );
    d.ints(
        format!("{path}/pairs_iz"),
        t.pairs.iter().map(|p| p.0).collect(),
    );
    d.floats(
        format!("{path}/pairs_aw"),
        t.pairs.iter().map(|p| p.1).collect(),
    );
    d.ints(format!("{path}/nxs"), t.nxs.clone());
    d.ints(format!("{path}/jxs"), t.jxs.clone());
    d.int(format!("{path}/xss_len"), t.xss.len() as i64);
    let idx = ace_xss_indices(t.xss.len(), &t.jxs);
    let values: Vec<f64> = idx.iter().map(|&i| t.xss[i]).collect();
    d.ints(
        format!("{path}/xss_idx"),
        idx.iter().map(|&i| i as i64).collect(),
    );
    d.floats(format!("{path}/xss_val"), values);

    dump_ace_angle(d, &format!("{path}/and"), t);
    dump_ace_dlw(d, &format!("{path}/dlw"), t);
    dump_ace_reactions(d, &format!("{path}/reaction"), t);
    dump_incident_neutron_ace(d, &format!("{path}/nuclide"), t);

    // The unresolved resonance block, when the table has one.
    if let Some(urr) = endf::urr::ProbabilityTables::from_ace(t) {
        d.floats(format!("{path}/urr/energy"), urr.energy.clone());
        d.ints(
            format!("{path}/urr/shape"),
            urr.shape.iter().map(|&v| v as i64).collect(),
        );
        d.floats(format!("{path}/urr/table"), urr.table.clone());
        d.int(format!("{path}/urr/interpolation"), urr.interpolation);
        d.int(format!("{path}/urr/inelastic_flag"), urr.inelastic_flag);
        d.int(format!("{path}/urr/absorption_flag"), urr.absorption_flag);
        d.int(
            format!("{path}/urr/multiply_smooth"),
            i64::from(urr.multiply_smooth),
        );
    }
}

fn dump_section(d: &mut Dump, path: &str, section: &Section) {
    match section {
        Section::Mf1Mt451(s) => {
            for (key, value) in [
                ("ZA", s.za),
                ("LRP", s.lrp),
                ("LFI", s.lfi),
                ("NLIB", s.nlib),
                ("NMOD", s.nmod),
                ("LIS", s.lis),
                ("LISO", s.liso),
                ("NFOR", s.nfor),
                ("LREL", s.lrel),
                ("NSUB", s.nsub),
                ("NVER", s.nver),
                ("LDRV", s.ldrv),
                ("NWD", s.nwd),
                ("NXC", s.nxc),
            ] {
                d.int(format!("{path}/{key}"), value);
            }
            for (key, value) in [
                ("AWR", s.awr),
                ("ELIS", s.elis),
                ("STA", s.sta),
                ("AWI", s.awi),
                ("EMAX", s.emax),
                ("TEMP", s.temp),
            ] {
                d.float(format!("{path}/{key}"), value);
            }
            if let Some(zsymam) = &s.zsymam {
                d.text(format!("{path}/ZSYMAM"), zsymam);
                for (key, value) in [
                    ("ALAB", &s.alab),
                    ("EDATE", &s.edate),
                    ("AUTH", &s.auth),
                    ("REF", &s.reference),
                    ("DDATE", &s.ddate),
                    ("RDATE", &s.rdate),
                    ("ENDATE", &s.endate),
                ] {
                    d.text(format!("{path}/{key}"), value.as_deref().unwrap_or(""));
                }
                for (i, line) in s.hsub.iter().enumerate() {
                    d.text(format!("{path}/HSUB/{i}"), line);
                }
                for (i, line) in s.description.iter().enumerate() {
                    d.text(format!("{path}/description/{i}"), line);
                }
            }
            for (i, &(mf, mt, nc, md)) in s.section_list.iter().enumerate() {
                d.ints(format!("{path}/section_list/{i}"), vec![mf, mt, nc, md]);
            }
        }

        Section::Mf1Mt452(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.int(format!("{path}/LNU"), s.lnu);
            dump_nu(d, path, &s.nu);
        }

        Section::Mf1Mt455(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.int(format!("{path}/LDG"), s.ldg);
            d.int(format!("{path}/LNU"), s.lnu);
            if s.ldg == 0 {
                d.floats(format!("{path}/lambda"), s.lambda.clone());
            }
            if let Some(e_int) = &s.e_int {
                d.tab2(&format!("{path}/E_int"), e_int);
            }
            for (i, c) in s.constants.iter().enumerate() {
                d.float(format!("{path}/constants/{i}/E"), c.energy);
                d.floats(format!("{path}/constants/{i}/lambda"), c.lambda.clone());
                d.floats(format!("{path}/constants/{i}/alpha"), c.alpha.clone());
            }
            dump_nu(d, path, &s.nu);
        }

        Section::Mf1Mt458(s) => {
            d.float(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.int(format!("{path}/LFC"), s.lfc);
            d.int(format!("{path}/NPLY"), s.nply);
            if s.lfc == 1 {
                d.int(format!("{path}/NFC"), s.nfc);
            }
            for (name, component) in FISSION_ENERGY_COMPONENTS.iter().zip(&s.components) {
                match component {
                    FissionEnergyRelease::Polynomial(pairs) => {
                        let flat = pairs.iter().flat_map(|&(c, u)| [c, u]).collect();
                        d.floats(format!("{path}/{name}/poly"), flat);
                    }
                    FissionEnergyRelease::Tabulated { ldrv, eifc } => {
                        d.int(format!("{path}/{name}/LDRV"), *ldrv);
                        d.tab1(&format!("{path}/{name}/EIFC"), eifc);
                    }
                }
            }
        }

        Section::Mf1Mt460(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.int(format!("{path}/LO"), s.lo);
            if s.lo == 1 {
                d.int(format!("{path}/NG"), s.ng);
                d.floats(format!("{path}/E"), s.energy.clone());
            }
            for (i, t) in s.time.iter().enumerate() {
                d.tab1(&format!("{path}/T/{i}"), t);
            }
            if s.lo == 2 {
                d.floats(format!("{path}/lambda"), s.lambda.clone());
            }
        }

        Section::Mf2(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.int(format!("{path}/NIS"), s.nis);
            for (a, iso) in s.isotopes.iter().enumerate() {
                let ip = format!("{path}/isotopes/{a}");
                d.float(format!("{ip}/ZAI"), iso.zai);
                d.float(format!("{ip}/ABN"), iso.abn);
                d.int(format!("{ip}/LFW"), iso.lfw);
                d.int(format!("{ip}/NER"), iso.ner);
                for (b, r) in iso.ranges.iter().enumerate() {
                    let rp = format!("{ip}/ranges/{b}");
                    d.float(format!("{rp}/EL"), r.el);
                    d.float(format!("{rp}/EH"), r.eh);
                    d.int(format!("{rp}/LRU"), r.lru);
                    d.int(format!("{rp}/LRF"), r.lrf);
                    d.int(format!("{rp}/NRO"), r.nro);
                    d.int(format!("{rp}/NAPS"), r.naps);
                    dump_mf2_parameters(d, &rp, &r.parameters);
                }
            }
        }

        Section::Mf3(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.float(format!("{path}/QM"), s.qm);
            d.float(format!("{path}/QI"), s.qi);
            d.int(format!("{path}/LR"), s.lr);
            d.tab1(&format!("{path}/sigma"), &s.sigma);
        }

        Section::Mf4(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.int(format!("{path}/LTT"), s.ltt);
            d.int(format!("{path}/LI"), s.li);
            d.int(format!("{path}/LCT"), s.lct);
            if let Some(l) = &s.legendre {
                let sp = format!("{path}/legendre");
                d.tab2(&format!("{sp}/E_int"), &l.e_int);
                d.float(format!("{sp}/T"), l.t);
                d.int(format!("{sp}/LT"), l.lt);
                d.floats(format!("{sp}/E"), l.energy.clone());
                for (i, a) in l.a_l.iter().enumerate() {
                    d.floats(format!("{sp}/a_l/{i}"), a.clone());
                }
            }
            if let Some(t) = &s.tabulated {
                let sp = format!("{path}/tabulated");
                d.tab2(&format!("{sp}/E_int"), &t.e_int);
                d.float(format!("{sp}/T"), t.t);
                d.int(format!("{sp}/LT"), t.lt);
                d.floats(format!("{sp}/E"), t.energy.clone());
                for (i, mu) in t.mu.iter().enumerate() {
                    d.tab1(&format!("{sp}/mu/{i}"), mu);
                }
            }

            // The interpreted form, which is what a consumer actually uses.
            let dist = AngleDistribution::from_mf4(s);
            dump_angle_distribution(d, &format!("{path}/angle"), &dist);
            if !dist.energy.is_empty() {
                for cutoff in [-1.0, -0.5, 0.0, 0.5] {
                    let name = format!("{cutoff:+.1}");
                    d.floats(
                        format!("{path}/angle/forward_fraction/{name}"),
                        dist.forward_fraction(cutoff),
                    );
                }
            }
        }

        Section::Mf5(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.int(format!("{path}/NK"), s.nk);
            for (i, sub) in s.subsections.iter().enumerate() {
                let sp = format!("{path}/subsections/{i}");
                d.int(format!("{sp}/LF"), sub.lf);
                d.tab1(&format!("{sp}/p"), &sub.p);
                let dp = format!("{sp}/distribution");
                match &sub.distribution {
                    EnergyDistribution::ArbitraryTabulated { e_int, energy, g } => {
                        d.tab2(&format!("{dp}/E_int"), e_int);
                        d.floats(format!("{dp}/E"), energy.clone());
                        for (j, t) in g.iter().enumerate() {
                            d.tab1(&format!("{dp}/g/{j}"), t);
                        }
                    }
                    EnergyDistribution::GeneralEvaporation { u, theta, g } => {
                        d.float(format!("{dp}/U"), *u);
                        d.tab1(&format!("{dp}/theta"), theta);
                        d.tab1(&format!("{dp}/g"), g);
                    }
                    EnergyDistribution::MaxwellEnergy { u, theta }
                    | EnergyDistribution::Evaporation { u, theta } => {
                        d.float(format!("{dp}/U"), *u);
                        d.tab1(&format!("{dp}/theta"), theta);
                    }
                    EnergyDistribution::WattEnergy { u, a, b } => {
                        d.float(format!("{dp}/U"), *u);
                        d.tab1(&format!("{dp}/a"), a);
                        d.tab1(&format!("{dp}/b"), b);
                    }
                    EnergyDistribution::MadlandNix { efl, efh, t_m } => {
                        d.float(format!("{dp}/EFL"), *efl);
                        d.float(format!("{dp}/EFH"), *efh);
                        d.tab1(&format!("{dp}/T_M"), t_m);
                    }
                    // The remaining three have no ENDF law and can only come
                    // from an ACE table, where `dump_energy_distribution`
                    // handles them.
                    other => unreachable!("{dp}: {other:?} is not an ENDF law"),
                }
            }
        }

        Section::Mf6(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.int(format!("{path}/JP"), s.jp);
            d.int(format!("{path}/LCT"), s.lct);
            d.int(format!("{path}/NK"), s.nk);
            for (i, p) in s.products.iter().enumerate() {
                let pp = format!("{path}/products/{i}");
                d.int(format!("{pp}/ZAP"), p.zap);
                d.float(format!("{pp}/AWP"), p.awp);
                d.int(format!("{pp}/LIP"), p.lip);
                d.int(format!("{pp}/LAW"), p.law);
                d.tab1(&format!("{pp}/y_i"), &p.yield_);
                dump_mf6_distribution(d, &format!("{pp}/distribution"), &p.distribution);
            }
        }

        Section::Mf7Mt2(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.int(format!("{path}/LTHR"), s.lthr);
            if let Some(c) = &s.coherent {
                d.float(format!("{path}/coherent/0/T"), c.t);
                d.int(format!("{path}/coherent/0/LT"), c.lt);
                d.tab1(&format!("{path}/coherent/0/S"), &c.s);
                for (i, o) in c.others.iter().enumerate() {
                    let tp = format!("{path}/coherent/{}", i + 1);
                    d.float(format!("{tp}/T"), o.t);
                    d.int(format!("{tp}/LI"), o.li);
                    d.floats(format!("{tp}/S"), o.s.clone());
                }
            }
            if let Some(i) = &s.incoherent {
                d.float(format!("{path}/incoherent/SB"), i.sb);
                d.tab1(&format!("{path}/incoherent/W"), &i.w);
            }
        }

        Section::Mf7Mt4(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            for (key, value) in [
                ("LAT", s.lat),
                ("LASYM", s.lasym),
                ("LLN", s.lln),
                ("NI", s.ni),
                ("NS", s.ns),
            ] {
                d.int(format!("{path}/{key}"), value);
            }
            d.floats(format!("{path}/B"), s.b.clone());
            if let Some(bi) = &s.beta_int {
                d.tab2(&format!("{path}/beta_int"), bi);
                d.int(format!("{path}/NB"), s.nb);
            }
            for (i, law) in s.beta_data.iter().enumerate() {
                let tp = format!("{path}/beta_data/{i}/0");
                d.float(format!("{tp}/T"), law.t);
                d.float(format!("{tp}/beta"), law.beta);
                d.int(format!("{tp}/LT"), law.lt);
                d.tab1(&format!("{tp}/S"), &law.s);
                for (j, o) in law.others.iter().enumerate() {
                    let tp = format!("{path}/beta_data/{i}/{}", j + 1);
                    d.float(format!("{tp}/T"), o.t);
                    d.float(format!("{tp}/beta"), o.beta);
                    d.int(format!("{tp}/LT"), o.lt);
                    d.floats(format!("{tp}/S"), o.s.clone());
                }
            }
            for (i, t) in s.teff.iter().enumerate() {
                d.tab1(&format!("{path}/Teff/{i}"), t);
            }
        }

        Section::Mf7Mt451(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.int(format!("{path}/NA"), s.na);
            for (i, e) in s.elements.iter().enumerate() {
                let ep = format!("{path}/elements/{i}");
                d.int(format!("{ep}/NAS"), e.nas);
                d.int(format!("{ep}/NI"), e.ni);
                for (key, values) in [
                    ("ZAI", &e.zai),
                    ("LISI", &e.lisi),
                    ("AFI", &e.afi),
                    ("AWRI", &e.awri),
                    ("SFI", &e.sfi),
                ] {
                    d.floats(format!("{ep}/{key}"), values.clone());
                }
            }
        }

        Section::Mf8(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            for (key, value) in [("LIS", s.lis), ("LISO", s.liso), ("NS", s.ns), ("NO", s.no)] {
                d.int(format!("{path}/{key}"), value);
            }
            for (i, sub) in s.subsections.iter().enumerate() {
                let sp = format!("{path}/subsections/{i}");
                d.float(format!("{sp}/ZAP"), sub.zap);
                d.float(format!("{sp}/ELFS"), sub.elfs);
                d.int(format!("{sp}/LMF"), sub.lmf);
                d.int(format!("{sp}/LFS"), sub.lfs);
                if let Some(nd) = sub.nd {
                    d.int(format!("{sp}/ND"), nd);
                    for (key, values) in [
                        ("HL", &sub.hl),
                        ("RTYP", &sub.rtyp),
                        ("ZAN", &sub.zan),
                        ("BR", &sub.br),
                        ("END", &sub.end),
                        ("CT", &sub.ct),
                    ] {
                        d.floats(format!("{sp}/{key}"), values.clone());
                    }
                }
            }
        }

        Section::Mf8Mt454(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.int(format!("{path}/LE"), s.le);
            for (i, y) in s.yields.iter().enumerate() {
                let yp = format!("{path}/yields/{i}");
                d.float(format!("{yp}/E"), y.energy);
                d.int(format!("{yp}/NN"), y.nn);
                d.int(format!("{yp}/NFP"), y.nfp);
                d.int(format!("{yp}/LE_or_I"), y.le_or_interpolation);
                for (j, p) in y.products.iter().enumerate() {
                    let pp = format!("{yp}/products/{j}");
                    d.float(format!("{pp}/ZAFP"), p.zafp);
                    d.float(format!("{pp}/FPS"), p.fps);
                    d.floats(format!("{pp}/Y"), vec![p.y.0, p.y.1]);
                }
            }
        }

        Section::Mf8Mt457(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            for (key, value) in [
                ("LIS", s.lis),
                ("LISO", s.liso),
                ("NST", s.nst),
                ("NSP", s.nsp),
            ] {
                d.int(format!("{path}/{key}"), value);
            }
            d.float(format!("{path}/SPI"), s.spi);
            d.float(format!("{path}/PAR"), s.par);
            if s.nst == 1 {
                return;
            }
            if let Some(hl) = s.half_life {
                d.floats(format!("{path}/T1_2"), vec![hl.0, hl.1]);
            }
            d.int(format!("{path}/NC"), s.nc);
            d.floats(
                format!("{path}/Ex"),
                s.ex.iter().flat_map(|&(a, b)| [a, b]).collect(),
            );
            d.int(format!("{path}/NDK"), s.ndk);
            for (i, m) in s.modes.iter().enumerate() {
                let mp = format!("{path}/modes/{i}");
                d.float(format!("{mp}/RTYP"), m.rtyp);
                d.float(format!("{mp}/RFS"), m.rfs);
                d.floats(format!("{mp}/Q"), vec![m.q.0, m.q.1]);
                d.floats(format!("{mp}/BR"), vec![m.br.0, m.br.1]);
            }
            for (i, sp_) in s.spectra.iter().enumerate() {
                let sp = format!("{path}/spectra/{i}");
                d.float(format!("{sp}/STYP"), sp_.styp);
                d.int(format!("{sp}/LCON"), sp_.lcon);
                d.int(format!("{sp}/LCOV"), sp_.lcov);
                d.int(format!("{sp}/NER"), sp_.ner);
                d.floats(format!("{sp}/FD"), vec![sp_.fd.0, sp_.fd.1]);
                d.floats(format!("{sp}/ER_AV"), vec![sp_.er_av.0, sp_.er_av.1]);
                d.floats(format!("{sp}/FC"), vec![sp_.fc.0, sp_.fc.1]);
                for (j, r) in sp_.discrete.iter().enumerate() {
                    let rp = format!("{sp}/discrete/{j}");
                    d.floats(format!("{rp}/ER"), vec![r.er.0, r.er.1]);
                    d.float(format!("{rp}/RTYP"), r.rtyp);
                    d.float(format!("{rp}/TYPE"), r.type_);
                    d.floats(format!("{rp}/RI"), vec![r.ri.0, r.ri.1]);
                    for (key, value) in [
                        ("RIS", r.ris),
                        ("RICC", r.ricc),
                        ("RICK", r.rick),
                        ("RICL", r.ricl),
                    ] {
                        if let Some(v) = value {
                            d.floats(format!("{rp}/{key}"), vec![v.0, v.1]);
                        }
                    }
                }
                if let Some(c) = &sp_.continuous {
                    d.float(format!("{sp}/continuous/RTYP"), c.rtyp);
                    d.tab1(&format!("{sp}/continuous/RP"), &c.rp);
                }
                if let Some(c) = &sp_.continuous_covariance {
                    d.int(format!("{sp}/cont_cov/LB"), c.lb);
                    d.floats(format!("{sp}/cont_cov/Ek"), c.ek.clone());
                    d.floats(format!("{sp}/cont_cov/Fk"), c.fk.clone());
                }
                if let Some(c) = &sp_.discrete_covariance {
                    d.int(format!("{sp}/disc_cov/LS"), c.ls);
                    d.int(format!("{sp}/disc_cov/LB"), c.lb);
                    d.int(format!("{sp}/disc_cov/NE"), c.ne);
                    d.int(format!("{sp}/disc_cov/NERP"), c.nerp);
                    d.floats(format!("{sp}/disc_cov/Ek"), c.ek.clone());
                    d.floats(format!("{sp}/disc_cov/Fkk"), c.fkk.clone());
                }
            }
        }

        Section::Mf9Mf10(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.int(format!("{path}/LIS"), s.lis);
            d.int(format!("{path}/NS"), s.ns);
            for (i, level) in s.levels.iter().enumerate() {
                let lp = format!("{path}/levels/{i}");
                d.float(format!("{lp}/QM"), level.qm);
                d.float(format!("{lp}/QI"), level.qi);
                d.int(format!("{lp}/IZAP"), level.izap);
                d.int(format!("{lp}/LFS"), level.lfs);
                d.tab1(&format!("{lp}/func"), &level.func);
            }
        }

        Section::Mf12(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.int(format!("{path}/LO"), s.lo);
            d.int(format!("{path}/NK"), s.nk);
            if let Some(y) = &s.total_yield {
                d.tab1(&format!("{path}/Y"), y);
            }
            for (i, k) in s.multiplicities.iter().enumerate() {
                let kp = format!("{path}/multiplicities/{i}");
                d.float(format!("{kp}/Eg"), k.eg);
                d.float(format!("{kp}/ES"), k.es);
                d.int(format!("{kp}/LP"), k.lp);
                d.int(format!("{kp}/LF"), k.lf);
                d.tab1(&format!("{kp}/y"), &k.y);
            }
            if let Some(lg) = s.lg {
                d.int(format!("{path}/LG"), lg);
                d.float(format!("{path}/ES_NS"), s.es_ns);
                d.int(format!("{path}/LP"), s.lp);
                d.int(format!("{path}/NT"), s.nt);
                for (i, t) in s.transitions.iter().enumerate() {
                    let tp = format!("{path}/transitions/{i}");
                    d.float(format!("{tp}/ES"), t.es);
                    d.float(format!("{tp}/TP"), t.tp);
                    if let Some(gp) = t.gp {
                        d.float(format!("{tp}/GP"), gp);
                    }
                }
            }
        }

        Section::Mf13(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.int(format!("{path}/NK"), s.nk);
            if let Some(t) = &s.sigma_total {
                d.tab1(&format!("{path}/sigma_total"), t);
            }
            for (i, p) in s.photons.iter().enumerate() {
                let pp = format!("{path}/photons/{i}");
                d.float(format!("{pp}/EG"), p.eg);
                d.float(format!("{pp}/ES"), p.es);
                d.int(format!("{pp}/LP"), p.lp);
                d.int(format!("{pp}/LF"), p.lf);
                d.tab1(&format!("{pp}/sigma"), &p.sigma);
            }
        }

        Section::Mf14(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.int(format!("{path}/LI"), s.li);
            d.int(format!("{path}/NK"), s.nk);
            if let (Some(ltt), Some(ni)) = (s.ltt, s.ni) {
                d.int(format!("{path}/LTT"), ltt);
                d.int(format!("{path}/NI"), ni);
            }
            for (i, sub) in s.subsections.iter().enumerate() {
                let sp = format!("{path}/subsections/{i}");
                d.float(format!("{sp}/EG"), sub.eg);
                d.float(format!("{sp}/ES"), sub.es);
                if let Some(e_int) = &sub.e_int {
                    d.tab2(&format!("{sp}/E_int"), e_int);
                    d.int(format!("{sp}/NE"), sub.ne);
                    d.floats(format!("{sp}/E"), sub.energy.clone());
                }
                if !sub.nl.is_empty() {
                    d.floats(format!("{sp}/NL"), sub.nl.clone());
                }
                for (j, a) in sub.a_lk.iter().enumerate() {
                    d.floats(format!("{sp}/a_lk/{j}"), a.clone());
                }
                for (j, p) in sub.p_k.iter().enumerate() {
                    d.tab1(&format!("{sp}/p_k/{j}"), p);
                }
            }
        }

        Section::Mf15(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.int(format!("{path}/NC"), s.nc);
            for (i, sub) in s.subsections.iter().enumerate() {
                let sp = format!("{path}/subsections/{i}");
                d.int(format!("{sp}/LF"), sub.lf);
                d.tab1(&format!("{sp}/p"), &sub.p);
                d.tab2(&format!("{sp}/E_int"), &sub.e_int);
                d.int(format!("{sp}/NE"), sub.ne);
                d.floats(format!("{sp}/E"), sub.energy.clone());
                for (j, g) in sub.g.iter().enumerate() {
                    d.tab1(&format!("{sp}/g/{j}"), g);
                }
            }
        }

        Section::Mf23(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.float(format!("{path}/EPE"), s.epe);
            d.float(format!("{path}/EFL"), s.efl);
            d.tab1(&format!("{path}/sigma"), &s.sigma);
        }

        Section::Mf26(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.int(format!("{path}/NK"), s.nk);
            for (i, p) in s.products.iter().enumerate() {
                let pp = format!("{path}/products/{i}");
                d.int(format!("{pp}/ZAP"), p.zap);
                d.float(format!("{pp}/AWI"), p.awi);
                d.int(format!("{pp}/LAW"), p.law);
                d.tab1(&format!("{pp}/y"), &p.yield_);
                let dp = format!("{pp}/distribution");
                match &p.distribution {
                    ElectroAtomicDistribution::None => {}
                    ElectroAtomicDistribution::EnergyTransfer(t) => d.tab1(&format!("{dp}/ET"), t),
                    ElectroAtomicDistribution::ContinuumEnergyAngle(c) => {
                        dump_mf6_distribution(
                            d,
                            &dp,
                            &Mf6Distribution::ContinuumEnergyAngle((**c).clone()),
                        );
                    }
                    ElectroAtomicDistribution::DiscreteTwoBody(t) => {
                        dump_mf6_distribution(
                            d,
                            &dp,
                            &Mf6Distribution::DiscreteTwoBody((**t).clone()),
                        );
                    }
                }
            }
        }

        Section::Mf27(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.float(format!("{path}/Z"), s.z);
            d.tab1(&format!("{path}/H"), &s.h);
        }

        Section::Mf28(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.int(format!("{path}/NSS"), s.nss);
            for (i, sh) in s.shells.iter().enumerate() {
                let sp = format!("{path}/shells/{i}");
                d.float(format!("{sp}/SUBI"), sh.subi);
                d.int(format!("{sp}/NTR"), sh.ntr);
                d.float(format!("{sp}/EBI"), sh.ebi);
                d.float(format!("{sp}/ELN"), sh.eln);
                for (key, values) in [
                    ("SUBJ", &sh.subj),
                    ("SUBK", &sh.subk),
                    ("ETR", &sh.etr),
                    ("FTR", &sh.ftr),
                ] {
                    d.floats(format!("{sp}/{key}"), values.clone());
                }
            }
        }

        Section::Mf33(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.int(format!("{path}/MTL"), s.mtl);
            d.int(format!("{path}/NL"), s.nl);
            for (i, sub) in s.subsections.iter().enumerate() {
                dump_mf33_subsection(d, &format!("{path}/subsections/{i}"), sub);
            }
        }

        Section::Mf34(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.int(format!("{path}/LTT"), s.ltt);
            d.int(format!("{path}/NMT1"), s.nmt1);
            // Always empty, matching upstream; see issue #18.
            for (i, sub) in s.subsections.iter().enumerate() {
                let sp = format!("{path}/subsections/{i}");
                for (key, value) in [
                    ("MAT1", sub.mat1),
                    ("MT1", sub.mt1),
                    ("NL", sub.nl),
                    ("NSS", sub.nss),
                    ("LCT", sub.lct),
                ] {
                    d.int(format!("{sp}/{key}"), value);
                }
                for (key, values) in [("L", &sub.l), ("L1", &sub.l1), ("NI", &sub.ni)] {
                    d.floats(format!("{sp}/{key}"), values.clone());
                }
                for (j, ss) in sub.subsubsections.iter().enumerate() {
                    let ssp = format!("{sp}/subsubsections/{j}");
                    for (key, values) in [
                        ("LS", &ss.ls),
                        ("LB", &ss.lb),
                        ("NT", &ss.nt),
                        ("NE", &ss.ne),
                    ] {
                        d.floats(format!("{ssp}/{key}"), values.clone());
                    }
                    for (k, values) in ss.data.iter().enumerate() {
                        d.floats(format!("{ssp}/Data/{k}"), values.clone());
                    }
                }
            }
        }

        Section::Mf40(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.int(format!("{path}/LIS"), s.lis);
            d.int(format!("{path}/NS"), s.ns);
            for (i, sub) in s.subsections.iter().enumerate() {
                let sp = format!("{path}/subsections/{i}");
                d.float(format!("{sp}/QM"), sub.qm);
                d.float(format!("{sp}/QI"), sub.qi);
                d.int(format!("{sp}/IZAP"), sub.izap);
                d.int(format!("{sp}/LFS"), sub.lfs);
                d.int(format!("{sp}/NL"), sub.nl);
                for (j, ss) in sub.subsubsections.iter().enumerate() {
                    dump_mf33_subsection(d, &format!("{sp}/subsubsections/{j}"), ss);
                }
            }
        }

        Section::Unparsed { .. } => {}
    }
}

// --------------------------------------------------------------------------

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
}

/// The evaluations and ACE tables the goldens are generated from.
///
/// Upstream keeps these in `tests/` at the repository root. In yamc that name
/// is a symlink to `crates/yamc/tests`, so they live inside the crate instead.
/// Paths recorded inside a golden are still resolved through [`repo_root`],
/// which is why they read `crates/endf/fixtures/...`; this is only for the scan
/// that has no golden to read the path from.
fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures")
}

/// The golden file, split into the structural records and the value map.
struct Golden {
    /// "ace" for an ACE fixture; ENDF materials otherwise.
    kind: String,
    n_tables: usize,
    source: String,
    n_materials: usize,
    /// (material index, MF, MT) -> body line count.
    sections: BTreeMap<(usize, i32, i32), usize>,
    /// material index -> MAT number.
    mats: BTreeMap<usize, i32>,
    values: BTreeMap<String, Value>,
    /// `KIND chain` only: the evaluations the chain is built from.
    decay: Vec<String>,
    neutron: Vec<String>,
    chain_reactions: Vec<String>,
}

fn parse_golden(text: &str, name: &str) -> Golden {
    let mut g = Golden {
        kind: String::new(),
        n_tables: 0,
        decay: Vec::new(),
        neutron: Vec::new(),
        chain_reactions: Vec::new(),
        source: String::new(),
        n_materials: 0,
        sections: BTreeMap::new(),
        mats: BTreeMap::new(),
        values: BTreeMap::new(),
    };

    for (i, line) in text.lines().enumerate() {
        let at = format!("{name}:{}", i + 1);
        if line.trim_start().starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let key = parts.next().unwrap_or("");
        let f: Vec<&str> = parts.collect();
        match key {
            "KIND" => g.kind = f[0].to_string(),
            "DECAY" => g.decay.push(f[0].to_string()),
            "NEUTRON" => g.neutron.push(f[0].to_string()),
            "REACTION" => g.chain_reactions.push(f[0].to_string()),
            "TABLES" => g.n_tables = f[0].parse().unwrap(),
            "SOURCE" => g.source = f[0].to_string(),
            "MATERIALS" => g.n_materials = f[0].parse().unwrap(),
            "MAT" => {
                g.mats.insert(f[0].parse().unwrap(), f[1].parse().unwrap());
            }
            "SECTION" => {
                let k = (
                    f[0].parse().unwrap(),
                    f[1].parse().unwrap(),
                    f[2].parse().unwrap(),
                );
                g.sections.insert(k, f[3].parse().unwrap());
            }
            "V" => {
                let path = f[0].to_string();
                let rest = &f[2..];
                let value = match f[1] {
                    "F" => Value::Floats(
                        rest.iter()
                            .map(|s| {
                                s.parse()
                                    .unwrap_or_else(|_| panic!("{at}: bad float {s:?}"))
                            })
                            .collect(),
                    ),
                    "I" => Value::Ints(
                        rest.iter()
                            .map(|s| s.parse().unwrap_or_else(|_| panic!("{at}: bad int {s:?}")))
                            .collect(),
                    ),
                    "T" => {
                        let hex = rest.first().copied().unwrap_or("");
                        let bytes: Vec<u8> = (0..hex.len())
                            .step_by(2)
                            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                            .collect();
                        Value::Text(String::from_utf8(bytes).unwrap())
                    }
                    other => panic!("{at}: unknown value tag {other:?}"),
                };
                g.values.insert(path, value);
            }
            other => panic!("{at}: unknown golden record {other:?}"),
        }
    }
    g
}

/// Compare the two `path -> value` maps whole.
///
/// Shared by the ENDF and ACE paths: a field renamed, dropped or added shows up
/// as a path on one side and not the other, whatever produced it.
/// Whether a path holds values computed rather than read off the file.
///
/// Two kinds qualify. `…/evaly` is the sampled interpolation the dump takes of
/// every TAB1. The reaction yields are the other: a product given by MF=10 as
/// a production cross section becomes a yield by dividing two interpolated
/// cross sections on their union grid, and the Python reader does that with
/// NumPy's array `log`, whose SIMD implementation differs from scalar `log` in
/// the last bit. The tolerance is 1e-12 relative, so a real disagreement (a
/// wrong value, a wrong index, a wrong law) still fails; only last-bit noise
/// passes. It also covers the MF=9 yields, which are read verbatim and would
/// otherwise be compared exactly.
fn is_interpolated(path: &str) -> bool {
    path.ends_with("/evaly")
        || (path.contains("/reaction/") && path.contains("/yield/f/"))
        // The propagated uncertainties. The Python package gets these from
        // `uncertainties`, which accumulates a variance and takes its square
        // root, where the port applies the derivative directly. The two agree
        // to the last bit or two, and the nominal values (the first entry of
        // each pair) are exact either way.
        || path.ends_with("/decay_constant")
        || path.ends_with("/decay_energy")
        // The forward-scattered fraction, and the removal cross section that
        // folds it into the total. Both integrate a Legendre series, and NumPy
        // changed the association of the Clenshaw recurrence between 2.2 and
        // 2.4, the results differ in the last bit or two depending on which
        // NumPy wrote the golden, so no single implementation can match both
        // exactly. The values are all within 1e-15 of each other.
        || path.contains("/forward_fraction/")
        || path.contains("/removal_xs/")
}

fn compare(name: &str, ours: &BTreeMap<String, Value>, theirs: &BTreeMap<String, Value>) {
    let ours_keys: BTreeSet<&String> = ours.keys().collect();
    let theirs_keys: BTreeSet<&String> = theirs.keys().collect();
    let missing: Vec<&&String> = theirs_keys.difference(&ours_keys).take(10).collect();
    let extra: Vec<&&String> = ours_keys.difference(&theirs_keys).take(10).collect();
    assert!(
        missing.is_empty(),
        "{name}: the Rust reader did not produce {} paths, e.g. {missing:?}",
        theirs_keys.difference(&ours_keys).count()
    );
    assert!(
        extra.is_empty(),
        "{name}: the Rust reader produced {} paths the Python reader does not, e.g. {extra:?}",
        ours_keys.difference(&theirs_keys).count()
    );

    for (path, want) in theirs {
        let got = &ours[path];
        assert_eq!(
            got.kind(),
            want.kind(),
            "{name}: {path} is {} in Rust and {} in Python",
            got.kind(),
            want.kind()
        );
        match (got, want) {
            // Interpolation is arithmetic, not parsing: the two languages
            // evaluate the same expression but need not round identically once
            // logs and exps are involved.
            (Value::Floats(a), Value::Floats(b)) if is_interpolated(path) => {
                assert_eq!(a.len(), b.len(), "{name}: {path} length");
                for (i, (&got, &want)) in a.iter().zip(b).enumerate() {
                    // What is being checked is that the two readers agree, not
                    // that the answer is finite. A tabulated S(alpha, beta) can
                    // hold zeros, and log-linear interpolation across one gives
                    // NaN in both languages alike, a real property of the
                    // evaluation, reproduced faithfully. Comparing those with
                    // subtraction would fail on NaN != NaN and hide it.
                    let agree = if want.is_nan() {
                        got.is_nan()
                    } else if want.is_infinite() {
                        got == want
                    } else {
                        (got - want).abs() <= EVAL_TOL * want.abs().max(1.0)
                    };
                    assert!(agree, "{name}: {path}[{i}]: rust {got} != python {want}");
                }
            }
            _ => assert_eq!(got, want, "{name}: {path}"),
        }
    }
}

/// Compare one golden file against the Rust reader. Returns paths compared.
fn check(golden_path: &Path) -> usize {
    let text = read_text(golden_path);
    let name = golden_path
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let g = parse_golden(&text, &name);

    let source_path = repo_root().join(&g.source);

    if g.kind == "chain" {
        let read = |paths: &[String]| -> Vec<Material> {
            paths
                .iter()
                .map(|p| Material::from_str(&read_text(&repo_root().join(p))).unwrap())
                .collect()
        };
        let reactions: Vec<&str> = g.chain_reactions.iter().map(String::as_str).collect();
        let chain =
            endf::Chain::from_endf(&read(&g.decay), &[], &read(&g.neutron), &reactions).unwrap();
        let mut d = Dump::default();
        dump_chain(&mut d, "chain", &chain);
        compare(&name, &d.map, &g.values);
        return g.values.len();
    }

    if g.kind == "ace" {
        let tables = ace::tables_from_str(&read_text(&source_path), None)
            .unwrap_or_else(|e| panic!("{name}: the Rust reader failed on {}: {e}", g.source));
        assert_eq!(tables.len(), g.n_tables, "{name}: table count");
        let mut d = Dump::default();
        for (i, table) in tables.iter().enumerate() {
            dump_ace_table(&mut d, &i.to_string(), table);
        }
        compare(&name, &d.map, &g.values);
        return g.values.len();
    }

    let endf_text = read_text(&source_path);

    let materials = materials_from_str(&endf_text)
        .unwrap_or_else(|e| panic!("{name}: the Rust reader failed on {}: {e}", g.source));

    assert_eq!(materials.len(), g.n_materials, "{name}: material count");

    let mut d = Dump::default();
    let mut sections: BTreeMap<(usize, i32, i32), usize> = BTreeMap::new();
    for (m, material) in materials.iter().enumerate() {
        assert_eq!(
            Some(&material.mat),
            g.mats.get(&m),
            "{name}: material {m} number"
        );
        for (&(mf, mt), body) in &material.section_text {
            sections.insert((m, mf, mt), body.lines().count());
        }
        for (&(mf, mt), section) in &material.section_data {
            dump_section(&mut d, &format!("{m}/{mf}/{mt}"), section);
        }
        dump_radionuclide_production(&mut d, &format!("{m}/production"), material);
        dump_reactions(&mut d, &format!("{m}/reaction"), material);
        dump_incident_neutron_endf(&mut d, &format!("{m}/nuclide"), material);
        dump_decay(&mut d, &format!("{m}/decay"), material);
        dump_incident_photon(&mut d, &format!("{m}/photon"), material);
    }

    assert_eq!(sections, g.sections, "{name}: section splitting differs");

    compare(&name, &d.map, &g.values);

    g.values.len()
}

#[test]
fn matches_the_python_reader() {
    let dir = golden_dir();
    let mut goldens: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| has_kind(p, "txt"))
        .collect();
    goldens.sort();

    assert!(!goldens.is_empty(), "no golden files in {}", dir.display());

    let mut total = 0usize;
    for path in &goldens {
        total += check(path);
    }
    println!("{} golden files, {total} paths compared", goldens.len());
}

#[test]
fn unported_files_keep_their_text() {
    // The port proceeds file by file, so a section with no Rust parser must
    // still round-trip its text for the Python reader to fall back to.
    //
    // Built synthetically rather than taken from a fixture: every file in
    // every fixture on this branch is now ported, and a test that depends on
    // that not being true stops testing anything the moment it stops holding.
    // MF=32 (resonance parameter covariances) is not parsed by the Python
    // reader either (its dispatch warns and ignores) so it is a stable
    // choice rather than one the next commit invalidates.
    const MF: i32 = 32;
    let line =
        |body: &str, mat: i32, mf: i32, mt: i32| format!("{body:<66}{mat:>4}{mf:>2}{mt:>3}\n");
    let text = line(" tape id", 1, 0, 0)
        + &line(" 1.001000+3 9.991673-1          0          0          1          0", 125, MF, 2)
        + &line(" 0.000000+0 0.000000+0          0          2          1          1", 125, MF, 2)
        + &line("", 125, MF, 0)   // SEND
        + &line("", 0, 0, 0); // MEND

    let m = Material::from_str(&text).unwrap();
    assert_eq!(m.mat, 125);

    let unparsed: Vec<(i32, i32)> = m
        .section_data
        .iter()
        .filter(|(_, s)| matches!(s, Section::Unparsed { .. }))
        .map(|(&k, _)| k)
        .collect();
    assert_eq!(unparsed, vec![(MF, 2)], "MF={MF} should not have a parser");

    // The body is kept whole, SEND excluded, so the Python reader can take it.
    let body = &m.section_text[&(MF, 2)];
    assert_eq!(body.lines().count(), 2);
    assert!(body.contains("1.001000+3"));
}

/// The files that have a Rust parser but which no fixture exercises.
///
/// These are written and structurally complete but have never been run against
/// a real evaluation, so nothing here is checked against the Python reader.
/// Kept as an explicit list rather than a remark in a commit message: the test
/// below fails when a fixture starts covering one of them, which is the moment
/// the entry should be deleted.
const UNCOVERED_BY_ANY_FIXTURE: [i32; 1] = [40];

/// The MF files that have a Rust parser at all.
const PORTED: [i32; 21] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 23, 26, 27, 28, 33, 34, 40,
];

#[test]
fn the_uncovered_parser_list_is_accurate() {
    let mut seen: BTreeSet<i32> = BTreeSet::new();
    for entry in std::fs::read_dir(fixture_dir()).unwrap() {
        let path = entry.unwrap().path();
        if !has_kind(&path, "endf") {
            continue;
        }
        let text = read_text(&path);
        for m in materials_from_str(&text).unwrap() {
            seen.extend(m.section_data.keys().map(|&(mf, _)| mf));
        }
    }

    let uncovered: BTreeSet<i32> = PORTED
        .iter()
        .copied()
        .filter(|mf| !seen.contains(mf))
        .collect();
    let declared: BTreeSet<i32> = UNCOVERED_BY_ANY_FIXTURE.into_iter().collect();
    assert_eq!(
        uncovered, declared,
        "the list of parsers no fixture exercises has changed. If a fixture now \
         covers one of these, delete it from UNCOVERED_BY_ANY_FIXTURE; if a new \
         parser has no coverage, add it."
    );
}

/// Every distribution shape the object dumpers can write.
///
/// Three ENDF laws are missing on purpose. LF=1, LF=5 and LF=12 reach the
/// golden files through the MF=5 section dump, which is driven by the Python
/// reader's dictionaries and writes no `kind` line, so this scan cannot see
/// them. Their fixture coverage is tracked in `golden/README.md` instead.
///
/// ACE law 5 is missing for a different reason: the Python reader has no
/// `from_ace` for the general evaporation spectrum and dies with an
/// AttributeError, so there is nothing to compare against (issue #19). The
/// Rust reader refuses that law by name.
const DISTRIBUTION_SHAPES: [&str; 16] = [
    // Univariate shapes.
    "discrete",
    "tabular",
    "uniform",
    "mixture",
    // Angular distribution shapes, from ENDF.
    "legendre",
    "tabulated",
    // Energy distribution laws an ACE table can carry.
    "maxwell",
    "evaporation",
    "watt",
    "level-inelastic",
    "discrete-photon",
    "continuous-tabular",
    // Joint angle-energy shapes.
    "uncorrelated",
    "kalbach-mann",
    "correlated",
    "nbody",
];

/// Every distribution shape is reached by some fixture.
///
/// The golden files are the evidence: a shape that no fixture produces leaves
/// no `kind` line, and this says which one rather than the gap going unnoticed.
#[test]
fn every_distribution_shape_has_a_fixture() {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for entry in std::fs::read_dir(repo_root().join("crates/endf/tests/golden")).unwrap() {
        let path = entry.unwrap().path();
        if !has_kind(&path, "txt") {
            continue;
        }
        for line in read_text(&path).lines() {
            // `V <path>/kind T <hex>`
            let mut parts = line.split_whitespace();
            let (Some("V"), Some(p), Some("T"), Some(hex)) =
                (parts.next(), parts.next(), parts.next(), parts.next())
            else {
                continue;
            };
            if !p.ends_with("/kind") {
                continue;
            }
            let bytes: Vec<u8> = (0..hex.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                .collect();
            seen.insert(String::from_utf8(bytes).unwrap());
        }
    }

    let missing: Vec<&&str> = DISTRIBUTION_SHAPES
        .iter()
        .filter(|s| !seen.contains(**s))
        .collect();
    assert!(
        missing.is_empty(),
        "no fixture produces these distribution shapes: {missing:?}"
    );
}

/// Every fixture is now fully parsed. This guards the claim: if a fixture is
/// added that contains a file with no Rust parser, it says so by name rather
/// than the coverage quietly slipping.
#[test]
fn every_fixture_section_has_a_parser() {
    let mut missing: BTreeSet<(String, i32, i32)> = BTreeSet::new();
    for entry in std::fs::read_dir(fixture_dir()).unwrap() {
        let path = entry.unwrap().path();
        if !has_kind(&path, "endf") {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let text = read_text(&path);
        for m in materials_from_str(&text).unwrap() {
            for (&(mf, mt), section) in &m.section_data {
                if matches!(section, Section::Unparsed { .. }) {
                    missing.insert((name.clone(), mf, mt));
                }
            }
        }
    }
    assert!(
        missing.is_empty(),
        "fixtures contain files with no Rust parser: {missing:?}"
    );
}
