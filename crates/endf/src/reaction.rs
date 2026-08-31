//! Reactions: a cross section, its Q values, and what comes out of them.
//!
//! An ENDF evaluation scatters one reaction across several files: MF=3 has
//! the cross section, MF=4/5/6 the distributions of what is emitted, MF=8/9/10
//! the radioactive products. [`Reaction`] is those gathered into one place.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use crate::ace::Table;
use crate::angle_energy::{AngleEnergy, UncorrelatedAngleEnergy};
use crate::data::{gnds_name, temperature_str, ATOMIC_SYMBOL, EV_PER_MEV};
use crate::error::{Error, Result};
use crate::function::{Polynomial, Tabulated1D};
use crate::material::Material;
use crate::mf::mf1::Nu;
use crate::mf::mf4::{AngleAtEnergy, AngleDistribution};
use crate::mf::mf5::EnergyDistribution;
use crate::product::{EmissionMode, Product, Yield};
use crate::univariate::Uniform;

/// The MT numbers that mean fission.
pub const FISSION_MTS: [i32; 5] = [18, 19, 20, 21, 38];

/// The reactions whose names the format states outright.
///
/// The level reactions are not here: their names follow from the MT number by
/// the rules in [`LEVEL_FAMILIES`].
const REACTION_NAME_BASE: [(i32, &str); 111] = [
    (1, "(n,total)"),
    (2, "(n,elastic)"),
    (3, "(n,nonelastic)"),
    (4, "(n,level)"),
    (5, "(n,misc)"),
    (11, "(n,2nd)"),
    (16, "(n,2n)"),
    (17, "(n,3n)"),
    (18, "(n,fission)"),
    (19, "(n,f)"),
    (20, "(n,nf)"),
    (21, "(n,2nf)"),
    (22, "(n,na)"),
    (23, "(n,n3a)"),
    (24, "(n,2na)"),
    (25, "(n,3na)"),
    (27, "(n,absorption)"),
    (28, "(n,np)"),
    (29, "(n,n2a)"),
    (30, "(n,2n2a)"),
    (32, "(n,nd)"),
    (33, "(n,nt)"),
    (34, "(n,n3He)"),
    (35, "(n,nd2a)"),
    (36, "(n,nt2a)"),
    (37, "(n,4n)"),
    (38, "(n,3nf)"),
    (41, "(n,2np)"),
    (42, "(n,3np)"),
    (44, "(n,n2p)"),
    (45, "(n,npa)"),
    (91, "(n,nc)"),
    (101, "(n,disappear)"),
    (102, "(n,gamma)"),
    (103, "(n,p)"),
    (104, "(n,d)"),
    (105, "(n,t)"),
    (106, "(n,3He)"),
    (107, "(n,a)"),
    (108, "(n,2a)"),
    (109, "(n,3a)"),
    (111, "(n,2p)"),
    (112, "(n,pa)"),
    (113, "(n,t2a)"),
    (114, "(n,d2a)"),
    (115, "(n,pd)"),
    (116, "(n,pt)"),
    (117, "(n,da)"),
    (152, "(n,5n)"),
    (153, "(n,6n)"),
    (154, "(n,2nt)"),
    (155, "(n,ta)"),
    (156, "(n,4np)"),
    (157, "(n,3nd)"),
    (158, "(n,nda)"),
    (159, "(n,2npa)"),
    (160, "(n,7n)"),
    (161, "(n,8n)"),
    (162, "(n,5np)"),
    (163, "(n,6np)"),
    (164, "(n,7np)"),
    (165, "(n,4na)"),
    (166, "(n,5na)"),
    (167, "(n,6na)"),
    (168, "(n,7na)"),
    (169, "(n,4nd)"),
    (170, "(n,5nd)"),
    (171, "(n,6nd)"),
    (172, "(n,3nt)"),
    (173, "(n,4nt)"),
    (174, "(n,5nt)"),
    (175, "(n,6nt)"),
    (176, "(n,2n3He)"),
    (177, "(n,3n3He)"),
    (178, "(n,4n3He)"),
    (179, "(n,3n2p)"),
    (180, "(n,3n2a)"),
    (181, "(n,3npa)"),
    (182, "(n,dt)"),
    (183, "(n,npd)"),
    (184, "(n,npt)"),
    (185, "(n,ndt)"),
    (186, "(n,np3He)"),
    (187, "(n,nd3He)"),
    (188, "(n,nt3He)"),
    (189, "(n,nta)"),
    (190, "(n,2n2p)"),
    (191, "(n,p3He)"),
    (192, "(n,d3He)"),
    (193, "(n,3Hea)"),
    (194, "(n,4n2p)"),
    (195, "(n,4n2a)"),
    (196, "(n,4npa)"),
    (197, "(n,3p)"),
    (198, "(n,n3p)"),
    (199, "(n,3n2pa)"),
    (200, "(n,5n2p)"),
    (203, "(n,Xp)"),
    (204, "(n,Xd)"),
    (205, "(n,Xt)"),
    (206, "(n,X3He)"),
    (207, "(n,Xa)"),
    (301, "heating"),
    (444, "damage-energy"),
    (649, "(n,pc)"),
    (699, "(n,dc)"),
    (749, "(n,tc)"),
    (799, "(n,3Hec)"),
    (849, "(n,ac)"),
    (891, "(n,2nc)"),
    (901, "heating-local"),
];

/// The runs of MT numbers that count the excited level of the residual, as
/// `(first, last + 1, emitted particle, offset)`.
///
/// MT=51 is the first inelastic level and is named `(n,n1)`, so the name is
/// the particle followed by `MT - offset`.
const LEVEL_FAMILIES: [(i32, i32, &str, i32); 7] = [
    (51, 91, "n", 50),
    (600, 649, "p", 600),
    (650, 699, "d", 650),
    (700, 749, "t", 700),
    (750, 799, "3He", 750),
    (800, 849, "a", 800),
    (875, 891, "2n", 875),
];

/// The names by which a reaction may be asked for besides its own.
const REACTION_ALIASES: [(&str, i32); 5] = [
    ("total", 1),
    ("elastic", 2),
    ("fission", 18),
    ("absorption", 27),
    ("capture", 102),
];

/// The name of a reaction, e.g. `"(n,2n)"` for MT=16 or `"(n,n3)"` for MT=53.
///
/// `None` for an MT the format does not name, which includes every MT a
/// particular evaluation may have invented.
pub fn reaction_name(mt: i32) -> Option<String> {
    for &(first, end, particle, offset) in &LEVEL_FAMILIES {
        if (first..end).contains(&mt) {
            return Some(format!("(n,{particle}{})", mt - offset));
        }
    }
    REACTION_NAME_BASE
        .iter()
        .find(|&&(m, _)| m == mt)
        .map(|&(_, name)| name.to_string())
}

/// Every name a reaction goes by, to its MT. Built once, on first use.
fn reaction_mt_table() -> &'static BTreeMap<String, i32> {
    static TABLE: OnceLock<BTreeMap<String, i32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut map = BTreeMap::new();
        for &(mt, name) in &REACTION_NAME_BASE {
            map.insert(name.to_string(), mt);
        }
        for &(first, end, particle, offset) in &LEVEL_FAMILIES {
            for mt in first..end {
                map.insert(format!("(n,{particle}{})", mt - offset), mt);
            }
        }
        for &(name, mt) in &REACTION_ALIASES {
            map.insert(name.to_string(), mt);
        }
        map
    })
}

/// The MT of a named reaction, by its own name or by an alias.
pub fn reaction_mt(name: &str) -> Option<i32> {
    reaction_mt_table().get(name).copied()
}

/// One reaction channel of a nuclide.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Reaction {
    pub mt: i32,
    /// The cross section in barns, by temperature. The key is the temperature
    /// as [`crate::data::temperature_str`] writes it, `"0K"` for an
    /// unbroadened ENDF evaluation, `"294K"` for a processed table.
    pub xs: BTreeMap<String, Tabulated1D>,
    pub products: Vec<Product>,
    /// Products the evaluation gives but which duplicate others, such as the
    /// total fission neutron where prompt and delayed are both present.
    ///
    /// The Python reader computes these and then drops them, with a TODO
    /// saying they should be stored somewhere. They are stored here: nothing
    /// else changes, and a consumer that wants the total nu-bar of a fission
    /// evaluation would otherwise have to read MF=1 MT=452 itself.
    pub derived_products: Vec<Product>,
    /// Q value of the reaction in eV.
    pub q_reaction: f64,
    /// Mass-difference Q value in eV.
    pub q_massdiff: f64,
    /// True when the cross section is the sum of others rather than evaluated.
    pub redundant: bool,
    /// Whether the secondary distributions are in the centre-of-mass frame.
    pub center_of_mass: bool,
}

impl Reaction {
    /// An empty reaction with the given MT.
    pub fn new(mt: i32) -> Reaction {
        Reaction {
            mt,
            center_of_mass: true,
            ..Default::default()
        }
    }

    /// The reaction's name, where the format gives it one.
    pub fn name(&self) -> Option<String> {
        reaction_name(self.mt)
    }

    /// Gather one reaction from an ENDF material.
    ///
    /// The MF=3 section must exist; everything else is taken where present.
    pub fn from_endf(mt: i32, material: &Material) -> Result<Reaction> {
        let mf3 = material.mf3(mt).ok_or(Error::Unsupported {
            what: "a reaction with no MF=3 cross section",
        })?;
        let q_massdiff = mf3.qm;
        let q_reaction = mf3.qi;

        let is_fission = FISSION_MTS.contains(&mt);

        // Fission neutron yields and the delayed spectra come from MF=1.
        let (mut products, derived_products) = if is_fission {
            fission_products_endf(material)?
        } else {
            (Vec::new(), Vec::new())
        };

        if let Some(mf6) = material.mf6(mt) {
            for product in mf6_products(mf6) {
                // Fission neutrons already came from MF=1; MF=6 only adds
                // their distributions. The emptiness check is the one place
                // this differs from the Python reader, which indexes
                // unconditionally and raises IndexError for a fission
                // evaluation with no MF=1 nu-bar at all.
                if is_fission && product.name == "neutron" && !products.is_empty() {
                    products[0].applicability = product.applicability;
                    products[0].distribution = product.distribution;
                } else {
                    products.push(product);
                }
            }
        } else if material.mf4(mt).is_some() || material.mf5(mt).is_some() {
            let mut neutron = Product::new("neutron");

            if let Some(mf5) = material.mf5(mt) {
                // MT=455's energy distribution is read with the delayed
                // neutrons instead, in `fission_products_endf`.
                for sub in &mf5.subsections {
                    neutron.applicability.push(sub.p.clone());
                    neutron
                        .distribution
                        .push(AngleEnergy::Uncorrelated(UncorrelatedAngleEnergy {
                            angle: None,
                            energy: Some(sub.distribution.clone()),
                        }));
                }
            } else if mt == 2 {
                // Elastic scattering: the outgoing energy follows from the
                // kinematics, so no distribution is given.
                neutron
                    .distribution
                    .push(AngleEnergy::Uncorrelated(UncorrelatedAngleEnergy::default()));
            } else if (51..91).contains(&mt) {
                // Level inelastic scattering, likewise analytic. What is
                // needed is the threshold and the mass ratio.
                let a = material
                    .mf1_mt451()
                    .map(|m| m.awr)
                    .ok_or(Error::Unsupported {
                        what: "level inelastic scattering without MF=1 MT=451",
                    })?;
                neutron
                    .distribution
                    .push(AngleEnergy::Uncorrelated(UncorrelatedAngleEnergy {
                        angle: None,
                        energy: Some(EnergyDistribution::LevelInelastic {
                            threshold: (a + 1.0) / a * q_reaction.abs(),
                            mass_ratio: (a / (a + 1.0)).powi(2),
                        }),
                    }));
            }

            if let Some(mf4) = material.mf4(mt) {
                let angle = AngleDistribution::from_mf4(mf4);
                for dist in &mut neutron.distribution {
                    if let AngleEnergy::Uncorrelated(u) = dist {
                        u.angle = Some(angle.clone());
                    }
                }
            }

            if is_fission && material.mf5(mt).is_some() && !products.is_empty() {
                products[0].applicability = neutron.applicability;
                products[0].distribution = neutron.distribution;
            } else {
                products.push(neutron);
            }
        }

        if material.mf8(mt).is_some() {
            for act in activation_products(material, mt, &mf3.sigma) {
                // A product MF=6 already gave keeps its distribution; only the
                // yield is replaced.
                match products.iter_mut().find(|p| p.name == act.name) {
                    Some(existing) => existing.yield_ = act.yield_,
                    None => products.push(act),
                }
            }
        }

        Ok(Reaction {
            mt,
            xs: BTreeMap::from([("0K".to_string(), mf3.sigma.clone())]),
            products,
            derived_products,
            q_reaction,
            q_massdiff,
            redundant: false,
            center_of_mass: true,
        })
    }
}

/// The name of the particle a ZAP identifies.
fn product_name(za: i64) -> String {
    match za {
        0 => "photon".to_string(),
        1 => "neutron".to_string(),
        1000 => "electron".to_string(),
        _ => gnds_name((za / 1000) as u32, (za % 1000) as u32, 0),
    }
}

/// The products MF=6 lists, with their yields.
///
/// The distributions are not read here, matching the Python reader, which
/// leaves that to a consumer that needs them.
fn mf6_products(mf6: &crate::mf::mf6::Mf6) -> Vec<Product> {
    mf6.products
        .iter()
        .map(|data| Product {
            name: product_name(data.zap),
            yield_: Yield::Tabulated(data.yield_.clone()),
            ..Default::default()
        })
        .collect()
}

/// A nu-bar as a product yield.
fn nu_yield(nu: &Nu) -> Option<Yield> {
    match nu {
        Nu::Polynomial(c) => Some(Yield::Polynomial(Polynomial::new(c.clone()))),
        Nu::Tabulated(t) => Some(Yield::Tabulated(t.clone())),
        Nu::Absent => None,
    }
}

/// The fission neutrons of an evaluation: prompt, total and delayed.
///
/// Returns the products and the derived ones, the total neutron is derived
/// when prompt is also given, since it is then the sum of what is already
/// there.
fn fission_products_endf(material: &Material) -> Result<(Vec<Product>, Vec<Product>)> {
    let mut products: Vec<Product> = Vec::new();
    let mut derived_products: Vec<Product> = Vec::new();

    let prompt = material.mf1_mt452(456);
    if let Some(yield_) = prompt.and_then(|d| nu_yield(&d.nu)) {
        products.push(Product {
            name: "neutron".to_string(),
            yield_,
            ..Default::default()
        });
    }
    let has_prompt = prompt.is_some();

    if let Some(total) = material.mf1_mt452(452) {
        if let Some(yield_) = nu_yield(&total.nu) {
            let neutron = Product {
                name: "neutron".to_string(),
                yield_,
                emission_mode: EmissionMode::Total,
                ..Default::default()
            };
            if has_prompt {
                derived_products.push(neutron);
            } else {
                products.push(neutron);
            }
        }
    }

    let Some(delayed) = material.mf1_mt455() else {
        return Ok((products, derived_products));
    };
    if delayed.ldg != 0 {
        return Err(Error::Unsupported {
            what: "delayed neutrons with energy-dependent group constants",
        });
    }

    let decay_constants = delayed.lambda.clone();
    for &constant in &decay_constants {
        products.push(Product {
            name: "neutron".to_string(),
            decay_rate: constant,
            emission_mode: EmissionMode::Delayed,
            ..Default::default()
        });
    }

    // The delayed yield in MT=455 is the total across all precursor groups;
    // each group's share comes from the applicability in MF=5. The Python
    // reader writes the total onto the last six products, so this does too.
    if let Some(yield_) = nu_yield(&delayed.nu) {
        let start = products.len().saturating_sub(6);
        for neutron in &mut products[start..] {
            neutron.yield_ = yield_.clone();
        }
    }

    let Some(mf5) = material.mf5(455) else {
        return Ok((products, derived_products));
    };
    let nk = mf5.nk as usize;
    if nk > 1 && decay_constants.len() == 1 {
        // One precursor group listed but several spectra: the spectra are
        // what actually separates the groups, so the product is duplicated.
        let template = products[1].clone();
        for _ in 0..nk - 1 {
            products.push(template.clone());
        }
    } else if nk != decay_constants.len() {
        return Err(Error::Mismatched {
            what: "the number of delayed neutron spectra and precursors",
        });
    }

    for (i, sub) in mf5.subsections.iter().enumerate() {
        let neutron = &mut products[1 + i];
        let applicability = &sub.p;

        // The group's yield is the total yield times the applicability of its
        // spectrum.
        neutron.yield_ = match &neutron.yield_ {
            Yield::Tabulated(t) => {
                if applicability.y.iter().all(|&v| v == applicability.y[0]) {
                    let mut t = t.clone();
                    for v in &mut t.y {
                        *v *= applicability.y[0];
                    }
                    Yield::Tabulated(t)
                } else {
                    // Neither grid contains the other, so the product is taken
                    // on their union, cut where one of them runs out.
                    let max_energy =
                        t.x[t.x.len() - 1].min(applicability.x[applicability.x.len() - 1]);
                    let mut energy: Vec<f64> =
                        t.x.iter().chain(&applicability.x).copied().collect();
                    energy.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    energy.dedup();
                    energy.retain(|&e| e <= max_energy);
                    let group: Vec<f64> = energy
                        .iter()
                        .map(|&e| t.eval(e) * applicability.eval(e))
                        .collect();
                    Yield::Tabulated(Tabulated1D::new(energy, group))
                }
            }
            Yield::Polynomial(p) => {
                if p.coefficients.len() == 1 {
                    let mut t = applicability.clone();
                    for v in &mut t.y {
                        *v *= p.coefficients[0];
                    }
                    Yield::Tabulated(t)
                } else if applicability.y.iter().all(|&v| v == applicability.y[0]) {
                    let mut p = p.clone();
                    p.coefficients[0] *= applicability.y[0];
                    Yield::Polynomial(p)
                } else {
                    return Err(Error::Unsupported {
                        what: "an energy-dependent delayed yield and group probability together",
                    });
                }
            }
        };

        neutron
            .distribution
            .push(AngleEnergy::Uncorrelated(UncorrelatedAngleEnergy {
                angle: None,
                energy: Some(sub.distribution.clone()),
            }));
    }

    Ok((products, derived_products))
}

/// The radioactive products of a reaction, from MF=9 and MF=10.
///
/// MF=9 gives a yield directly; MF=10 gives a production cross section, which
/// becomes a yield once divided by the reaction's own.
fn activation_products(material: &Material, mt: i32, xs: &Tabulated1D) -> Vec<Product> {
    let Some(mf8) = material.mf8(mt) else {
        return Vec::new();
    };
    // MF=8 says which of the two files carries the data.
    let present = |lmf: i64| mf8.subsections.iter().any(|s| s.lmf == lmf);

    let mut products = Vec::new();
    for mf in [9, 10] {
        if !present(mf) {
            continue;
        }
        let section = match mf {
            9 => material.mf9(mt),
            _ => material.mf10(mt),
        };
        let Some(section) = section else { continue };

        for level in &section.levels {
            let (z, a) = (level.izap / 1000, level.izap % 1000);
            let symbol = ATOMIC_SYMBOL.get(z as usize).copied().unwrap_or("");
            // The excited state, not the isomeric state: see
            // `crate::radionuclide_production` for the difference.
            let name = if level.lfs > 0 {
                format!("{symbol}{a}_e{}", level.lfs)
            } else {
                format!("{symbol}{a}")
            };

            let yield_ = if mf == 9 {
                Yield::Tabulated(level.func.clone())
            } else {
                // Both cross sections onto their union grid, then the ratio
                // wherever the reaction actually happens.
                let mut energy: Vec<f64> = level.func.x.iter().chain(&xs.x).copied().collect();
                energy.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                energy.dedup();
                let y: Vec<f64> = energy
                    .iter()
                    .map(|&e| {
                        let neutron = xs.eval(e);
                        if neutron > 0.0 {
                            level.func.eval(e) / neutron
                        } else {
                            0.0
                        }
                    })
                    .collect();
                Yield::Tabulated(Tabulated1D::new(energy, y))
            };

            products.push(Product {
                name,
                yield_,
                ..Default::default()
            });
        }
    }
    products
}

impl Reaction {
    /// Read one reaction from an ACE table.
    ///
    /// `i_reaction` indexes the table's reaction list; index 0 is elastic
    /// scattering, which the format stores apart from the rest.
    pub fn from_ace(table: &Table, i_reaction: i64) -> Result<Reaction> {
        let xss = &table.xss;
        let at = |i: i64| -> f64 {
            usize::try_from(i)
                .ok()
                .and_then(|i| xss.get(i).copied())
                .unwrap_or(0.0)
        };
        let slice = |i: i64, n: usize| -> Vec<f64> { (0..n as i64).map(|k| at(i + k)).collect() };

        // The nuclide's own energy grid, which the cross sections index into.
        let n_grid = table.nxs[3].max(0) as usize;
        let grid: Vec<f64> = slice(table.jxs[1], n_grid)
            .into_iter()
            .map(|e| e * EV_PER_MEV)
            .collect();

        // `Table::temperature` is in kelvin; the raw kT in the file is in MeV.
        let temperature = temperature_str(table.temperature());

        let mut rx;
        // Which product the angular distribution belongs to, once one is read.
        let mut neutron: Option<usize> = None;

        if i_reaction > 0 {
            let mt = at(table.jxs[3] + i_reaction - 1) as i32;
            rx = Reaction::new(mt);
            rx.q_reaction = at(table.jxs[4] + i_reaction - 1) * EV_PER_MEV;

            // The cross section, which starts at a threshold rather than at
            // the bottom of the grid.
            let loc = at(table.jxs[6] + i_reaction - 1) as i64;
            // The stored index is one-based into the nuclide energy grid.
            let threshold_idx = (at(table.jxs[7] + loc - 1) as i64 - 1).max(0) as usize;
            let n_energy = at(table.jxs[7] + loc).max(0.0) as usize;
            let energy = grid
                .get(threshold_idx..(threshold_idx + n_energy).min(grid.len()))
                .unwrap_or_default()
                .to_vec();
            let mut sigma = slice(table.jxs[7] + loc + 1, n_energy);

            // Damage energy production is stored in MeV.
            if mt == 444 {
                for v in &mut sigma {
                    *v *= EV_PER_MEV;
                }
            }
            // Processed files occasionally carry a small negative cross
            // section, which is a processing artefact rather than physics.
            // The Python reader warns and zeroes; this zeroes.
            for v in &mut sigma {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
            let mut tabulated = Tabulated1D::new(energy, sigma);
            tabulated.threshold_idx = Some(threshold_idx);
            rx.xs.insert(temperature.clone(), tabulated);

            // TY is the multiplicity, and its sign records the frame.
            let ty = at(table.jxs[5] + i_reaction - 1) as i64;
            rx.center_of_mass = ty < 0;

            if i_reaction < table.nxs[5] + 1 {
                if ty != 19 {
                    let yield_ = if ty.abs() > 100 {
                        // An energy-dependent yield, stored in DLW.
                        Yield::Tabulated(Tabulated1D::from_ace(
                            xss,
                            (table.jxs[11] + ty.abs() - 101).max(0) as usize,
                            true,
                        ))
                    } else {
                        Yield::Polynomial(Polynomial::new(vec![ty.abs() as f64]))
                    };
                    rx.products.push(Product {
                        name: "neutron".to_string(),
                        yield_,
                        ..Default::default()
                    });
                    neutron = Some(rx.products.len() - 1);
                } else {
                    // TY = 19 means fission, whose neutrons come from the NU
                    // block rather than from a multiplicity.
                    let (products, derived) = fission_products_ace(table)?;
                    rx.products = products;
                    rx.derived_products = derived;
                    neutron = rx.products.iter().position(|p| {
                        matches!(p.emission_mode, EmissionMode::Prompt | EmissionMode::Total)
                    });
                    if neutron.is_none() {
                        return Err(Error::BadAceTable {
                            what: "a fission reaction with no prompt or total neutron".into(),
                        });
                    }
                }

                // DLW is a linked list: each entry's first word points at the
                // next distribution for this reaction.
                let i = neutron.expect("set on both branches above");
                let mut lnw = at(table.jxs[10] + i_reaction - 1) as i64;
                while lnw > 0 {
                    rx.products[i].applicability.push(Tabulated1D::from_ace(
                        xss,
                        (table.jxs[11] + lnw + 2).max(0) as usize,
                        true,
                    ));
                    let dist =
                        AngleEnergy::from_ace(table, table.jxs[11], lnw, Some(rx.q_reaction))?;
                    rx.products[i].distribution.push(dist);
                    lnw = at(table.jxs[11] + lnw - 1) as i64;
                }
            }
        } else {
            // Elastic scattering, whose cross section is the fourth column of
            // the main energy block.
            rx = Reaction::new(2);
            let mut elastic = slice(table.jxs[1] + 3 * n_grid as i64, n_grid);
            for v in &mut elastic {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
            let mut tabulated = Tabulated1D::new(grid.clone(), elastic);
            tabulated.threshold_idx = Some(0);
            rx.xs.insert(temperature.clone(), tabulated);

            // No energy distribution is given: it follows from the kinematics.
            rx.products.push(Product {
                name: "neutron".to_string(),
                distribution: vec![AngleEnergy::Uncorrelated(UncorrelatedAngleEnergy::default())],
                ..Default::default()
            });
            neutron = Some(0);
        }

        // The angular distribution, for the uncorrelated laws. A negative
        // locator means the angle is bound up with the energy in DLW instead,
        // which is why this only ever writes onto an uncorrelated
        // distribution.
        if i_reaction < table.nxs[5] + 1 {
            let loc = at(table.jxs[8] + i_reaction) as i64;
            let angle = match loc {
                l if l < 0 => None,
                0 => Some(isotropic_angle(0.0, grid.last().copied().unwrap_or(0.0))),
                l => Some(AngleDistribution::from_ace(table, table.jxs[9], l)?),
            };
            if let (Some(angle), Some(i)) = (angle, neutron) {
                for dist in &mut rx.products[i].distribution {
                    if let AngleEnergy::Uncorrelated(u) = dist {
                        u.angle = Some(angle.clone());
                    }
                }
            }
        }

        rx.products.extend(photon_products_ace(table, &rx)?);
        Ok(rx)
    }
}

/// An isotropic angular distribution spanning an energy range.
fn isotropic_angle(energy_low: f64, energy_high: f64) -> AngleDistribution {
    let mu = AngleAtEnergy::Isotropic(Uniform::new(-1.0, 1.0));
    AngleDistribution {
        energy: vec![energy_low, energy_high],
        mu: vec![mu.clone(), mu],
    }
}

/// A nu-bar in an ACE NU block, in whichever of the two forms it uses.
fn read_nu_ace(table: &Table, idx: i64) -> Yield {
    let at = |i: i64| -> f64 {
        usize::try_from(i)
            .ok()
            .and_then(|i| table.xss.get(i).copied())
            .unwrap_or(0.0)
    };
    if at(idx) as i64 == 1 {
        let nc = at(idx + 1) as usize;
        // The coefficients are per MeV of incident energy, so the term of
        // degree i converts by that power.
        let coefficients = (0..nc)
            .map(|i| at(idx + 2 + i as i64) * EV_PER_MEV.powi(-(i as i32)))
            .collect();
        Yield::Polynomial(Polynomial::new(coefficients))
    } else {
        Yield::Tabulated(Tabulated1D::from_ace(
            &table.xss,
            (idx + 1).max(0) as usize,
            true,
        ))
    }
}

/// The fission neutrons of an ACE table: prompt, total and delayed.
fn fission_products_ace(table: &Table) -> Result<(Vec<Product>, Vec<Product>)> {
    let xss = &table.xss;
    let at = |i: i64| -> f64 {
        usize::try_from(i)
            .ok()
            .and_then(|i| xss.get(i).copied())
            .unwrap_or(0.0)
    };

    let mut products: Vec<Product> = Vec::new();
    let mut derived_products: Vec<Product> = Vec::new();

    // No NU block at all.
    if table.jxs[2] == 0 {
        return Ok((products, derived_products));
    }

    let first = at(table.jxs[2]);
    if first > 0.0 {
        // One of prompt and total is given, and which it is depends on
        // whether the delayed block exists.
        products.push(Product {
            name: "neutron".to_string(),
            emission_mode: if table.jxs[24] > 0 {
                EmissionMode::Prompt
            } else {
                EmissionMode::Total
            },
            yield_: read_nu_ace(table, table.jxs[2]),
            ..Default::default()
        });
    } else if first < 0.0 {
        // Both are given, one after the other.
        products.push(Product {
            name: "neutron".to_string(),
            emission_mode: EmissionMode::Prompt,
            yield_: read_nu_ace(table, table.jxs[2] + 1),
            ..Default::default()
        });
        derived_products.push(Product {
            name: "neutron".to_string(),
            emission_mode: EmissionMode::Total,
            yield_: read_nu_ace(table, table.jxs[2] + first.abs() as i64 + 1),
            ..Default::default()
        });
    }

    if table.jxs[24] <= 0 {
        return Ok((products, derived_products));
    }

    let yield_delayed = Tabulated1D::from_ace(xss, (table.jxs[24] + 1).max(0) as usize, true);
    let mut idx = table.jxs[25];
    let n_group = table.nxs[8].max(0);
    let mut total_group_probability = 0.0;

    for group in 0..n_group {
        // Decay constants are stored in inverse shakes.
        let decay_rate = at(idx) * 1.0e8;
        let probability = Tabulated1D::from_ace(xss, (idx + 1).max(0) as usize, true);

        let yield_ = if probability.y.iter().all(|&v| v == probability.y[0]) {
            let mut y = yield_delayed.clone();
            let share = probability.y[0];
            for v in &mut y.y {
                *v *= share;
            }
            total_group_probability += share;
            y
        } else {
            // Neither grid contains the other, so the product is taken on
            // their union, cut where one of them runs out.
            let max_energy = yield_delayed.x[yield_delayed.x.len() - 1]
                .min(probability.x[probability.x.len() - 1]);
            let mut energy: Vec<f64> = yield_delayed
                .x
                .iter()
                .chain(&probability.x)
                .copied()
                .collect();
            energy.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            energy.dedup();
            energy.retain(|&e| e <= max_energy);
            let y: Vec<f64> = energy
                .iter()
                .map(|&e| yield_delayed.eval(e) * probability.eval(e))
                .collect();
            Tabulated1D::new(energy, y)
        };

        // Step past this group's probability record.
        let nr = at(idx + 1) as i64;
        let ne = at(idx + 2 + 2 * nr) as i64;
        idx += 3 + 2 * nr + 2 * ne;

        let location_start = at(table.jxs[26] + group) as i64;
        // No Q value is passed: the Python reader passes no reaction here
        // either, so law 66 would fail in both. It does not arise, a delayed
        // neutron spectrum is never N-body phase space.
        let distribution = AngleEnergy::from_ace(table, table.jxs[27], location_start, None)?;

        products.push(Product {
            name: "neutron".to_string(),
            emission_mode: EmissionMode::Delayed,
            decay_rate,
            yield_: Yield::Tabulated(yield_),
            distribution: vec![distribution],
            ..Default::default()
        });
    }

    // The group probabilities in an ACE file do not sum to exactly one, so the
    // delayed yields are renormalised against what they do sum to.
    if total_group_probability > 0.0 {
        for product in products.iter_mut().skip(1) {
            if let Yield::Tabulated(y) = &mut product.yield_ {
                for v in &mut y.y {
                    *v /= total_group_probability;
                }
            }
        }
    }

    Ok((products, derived_products))
}

/// The photons a reaction produces, from the ACE photon production blocks.
pub(crate) fn photon_products_ace(table: &Table, rx: &Reaction) -> Result<Vec<Product>> {
    let xss = &table.xss;
    let at = |i: i64| -> f64 {
        usize::try_from(i)
            .ok()
            .and_then(|i| xss.get(i).copied())
            .unwrap_or(0.0)
    };
    let slice = |i: i64, n: usize| -> Vec<f64> { (0..n as i64).map(|k| at(i + k)).collect() };

    let n_photon_reactions = table.nxs[6].max(0);
    let mut photons = Vec::new();
    for i in 0..n_photon_reactions {
        // The photon MT encodes the neutron reaction it belongs to.
        let photon_mt = at(table.jxs[13] + i) as i64;
        if (photon_mt / 1000) as i32 != rx.mt {
            continue;
        }

        // Either a yield or a production cross section, depending on which
        // ENDF file the processing took it from.
        let loca = at(table.jxs[14] + i) as i64;
        let idx = table.jxs[15] + loca - 1;
        let mftype = at(idx) as i64;
        let idx = idx + 1;

        let yield_ = match mftype {
            12 | 16 => Tabulated1D::from_ace(xss, (idx + 1).max(0) as usize, true),
            13 => {
                // A production cross section, which becomes a yield once
                // divided by the reaction's own.
                let threshold_idx = at(idx) as i64 - 1;
                let n_energy = at(idx + 1) as usize;
                let energy: Vec<f64> = slice(table.jxs[1] + threshold_idx, n_energy)
                    .into_iter()
                    .map(|e| e * EV_PER_MEV)
                    .collect();
                let production = slice(idx + 2, n_energy);
                let neutron_xs = rx.xs.values().next().ok_or(Error::BadAceTable {
                    what: "a photon production cross section with no reaction to divide by".into(),
                })?;
                let y: Vec<f64> = energy
                    .iter()
                    .zip(&production)
                    .map(|(&e, &p)| {
                        let n = neutron_xs.eval(e);
                        if n > 0.0 {
                            p / n
                        } else {
                            0.0
                        }
                    })
                    .collect();
                Tabulated1D::new(energy, y)
            }
            _ => {
                return Err(Error::BadAceTable {
                    what: format!("photon production MFTYPE {mftype}, expected 12, 13 or 16"),
                })
            }
        };

        let location_start = at(table.jxs[18] + i) as i64;
        // As with the delayed spectra, no Q value: photon production is never
        // given as N-body phase space, and the Python reader passes none.
        let mut distribution = AngleEnergy::from_ace(table, table.jxs[19], location_start, None)?;

        // The angular distribution, which is separate for photons.
        let loc = at(table.jxs[16] + i) as i64;
        let angle = if loc == 0 {
            // Nothing given, so isotropic in the laboratory frame across the
            // range the yield covers.
            isotropic_angle(
                yield_.x.first().copied().unwrap_or(0.0),
                yield_.x.last().copied().unwrap_or(0.0),
            )
        } else {
            AngleDistribution::from_ace(table, table.jxs[17], loc)?
        };
        match &mut distribution {
            AngleEnergy::Uncorrelated(u) => u.angle = Some(angle),
            _ => {
                return Err(Error::BadAceTable {
                    what: "a photon distribution that is not uncorrelated".into(),
                })
            }
        }

        photons.push(Product {
            name: "photon".to_string(),
            yield_: Yield::Tabulated(yield_),
            distribution: vec![distribution],
            ..Default::default()
        });
    }
    Ok(photons)
}

#[cfg(test)]
mod tests {
    use super::*;

    const AM244: &[u8] = include_bytes!("../fixtures/n-095_Am_244.endf.xz");
    const IN115: &[u8] = include_bytes!("../fixtures/n-049_In-115_trimmed.endf.xz");

    #[test]
    fn names_the_reactions_the_format_names() {
        assert_eq!(reaction_name(2).as_deref(), Some("(n,elastic)"));
        assert_eq!(reaction_name(16).as_deref(), Some("(n,2n)"));
        assert_eq!(reaction_name(102).as_deref(), Some("(n,gamma)"));
        assert_eq!(reaction_name(444).as_deref(), Some("damage-energy"));
        // An MT the format does not assign.
        assert_eq!(reaction_name(999), None);
    }

    #[test]
    fn names_the_level_reactions_by_their_level() {
        // The first inelastic level is MT=51 and is named for level 1.
        assert_eq!(reaction_name(51).as_deref(), Some("(n,n1)"));
        assert_eq!(reaction_name(90).as_deref(), Some("(n,n40)"));
        // MT=91 ends the run and is the continuum, not level 41.
        assert_eq!(reaction_name(91).as_deref(), Some("(n,nc)"));
        assert_eq!(reaction_name(600).as_deref(), Some("(n,p0)"));
        assert_eq!(reaction_name(648).as_deref(), Some("(n,p48)"));
        assert_eq!(reaction_name(649).as_deref(), Some("(n,pc)"));
        assert_eq!(reaction_name(750).as_deref(), Some("(n,3He0)"));
        assert_eq!(reaction_name(875).as_deref(), Some("(n,2n0)"));
        assert_eq!(reaction_name(890).as_deref(), Some("(n,2n15)"));
        assert_eq!(reaction_name(891).as_deref(), Some("(n,2nc)"));
    }

    #[test]
    fn looks_up_reactions_by_name_and_by_alias() {
        assert_eq!(reaction_mt("(n,elastic)"), Some(2));
        assert_eq!(reaction_mt("(n,n3)"), Some(53));
        // The aliases the package adds on top of the format's own names.
        assert_eq!(reaction_mt("total"), Some(1));
        assert_eq!(reaction_mt("elastic"), Some(2));
        assert_eq!(reaction_mt("fission"), Some(18));
        assert_eq!(reaction_mt("absorption"), Some(27));
        assert_eq!(reaction_mt("capture"), Some(102));
        assert_eq!(reaction_mt("(n,nonsense)"), None);
    }

    #[test]
    fn every_name_round_trips_to_its_own_mt() {
        for mt in 0..1200 {
            if let Some(name) = reaction_name(mt) {
                assert_eq!(reaction_mt(&name), Some(mt), "{name} came back wrong");
            }
        }
    }

    #[test]
    fn gathers_a_fission_reaction_from_its_files() {
        let m = Material::from_str(&crate::testdata::text(AM244)).unwrap();
        let rx = Reaction::from_endf(18, &m).unwrap();
        assert_eq!(rx.name().as_deref(), Some("(n,fission)"));
        assert_eq!(rx.xs.keys().collect::<Vec<_>>(), ["0K"]);

        // Prompt neutrons from MT=456, then one per delayed precursor group.
        assert_eq!(rx.products[0].emission_mode, EmissionMode::Prompt);
        let delayed: Vec<&Product> = rx
            .products
            .iter()
            .filter(|p| p.emission_mode == EmissionMode::Delayed)
            .collect();
        assert_eq!(delayed.len(), m.mf1_mt455().unwrap().lambda.len());
        assert!(delayed.iter().all(|p| p.decay_rate > 0.0));
        // This evaluation gives no MF=5 MT=455, so the groups have decay
        // constants and a shared yield but no spectra of their own.
        assert!(m.mf5(455).is_none());
        assert!(delayed.iter().all(|p| p.distribution.is_empty()));
        // The prompt neutron's spectrum comes from MF=5 MT=18.
        assert_eq!(rx.products[0].distribution.len(), 1);
        assert_eq!(rx.products[0].applicability.len(), 1);

        // The total neutron is derived: prompt and delayed are both given, so
        // it adds nothing a consumer cannot sum for itself.
        assert_eq!(rx.derived_products.len(), 1);
        assert_eq!(rx.derived_products[0].emission_mode, EmissionMode::Total);
    }

    #[test]
    fn each_delayed_group_gets_its_own_spectrum_and_share_of_the_yield() {
        const U235: &[u8] = include_bytes!("../fixtures/n-092_U_235_trimmed.endf.xz");
        let m = Material::from_str(&crate::testdata::text(U235)).unwrap();
        let rx = Reaction::from_endf(18, &m).unwrap();

        let lambda = &m.mf1_mt455().unwrap().lambda;
        let delayed: Vec<&Product> = rx
            .products
            .iter()
            .filter(|p| p.emission_mode == EmissionMode::Delayed)
            .collect();
        assert_eq!(delayed.len(), lambda.len());
        assert_eq!(
            delayed.iter().map(|p| p.decay_rate).collect::<Vec<_>>(),
            *lambda
        );

        // MF=5 MT=455 gives one spectrum per group, and each group's yield is
        // the total delayed yield scaled by that spectrum's applicability.
        let mf5 = m.mf5(455).unwrap();
        assert_eq!(mf5.subsections.len(), delayed.len());
        let total = match &m.mf1_mt455().unwrap().nu {
            Nu::Tabulated(t) => t.clone(),
            _ => panic!("this evaluation tabulates its delayed yield"),
        };
        for (product, sub) in delayed.iter().zip(&mf5.subsections) {
            assert_eq!(product.distribution.len(), 1);
            let Yield::Tabulated(y) = &product.yield_ else {
                panic!("a scaled tabulated yield stays tabulated");
            };
            // The applicability is a constant here, so the group yield is the
            // total times that constant at every energy.
            let share = sub.p.y[0];
            assert!(sub.p.y.iter().all(|&v| v == share));
            assert_eq!(y.x, total.x);
            for (got, want) in y.y.iter().zip(&total.y) {
                assert_eq!(*got, want * share);
            }
        }

        // The shares are a partition of the delayed neutrons.
        let sum: f64 = mf5.subsections.iter().map(|s| s.p.y[0]).sum();
        assert!((sum - 1.0).abs() < 1e-6, "the group shares sum to {sum}");
    }

    #[test]
    fn level_inelastic_scattering_gets_its_kinematics() {
        let m = Material::from_str(&crate::testdata::text(AM244)).unwrap();
        let rx = Reaction::from_endf(51, &m).unwrap();
        assert_eq!(rx.name().as_deref(), Some("(n,n1)"));

        let dist = &rx.products[0].distribution[0];
        let AngleEnergy::Uncorrelated(u) = dist else {
            panic!("level inelastic scattering is uncorrelated");
        };
        // MF=4 gives the angle; the energy follows from the Q value.
        assert!(u.angle.is_some());
        let Some(EnergyDistribution::LevelInelastic {
            threshold,
            mass_ratio,
        }) = u.energy
        else {
            panic!("the energy should be the level kinematics");
        };
        let a = m.mf1_mt451().unwrap().awr;
        assert_eq!(threshold, (a + 1.0) / a * rx.q_reaction.abs());
        assert_eq!(mass_ratio, (a / (a + 1.0)).powi(2));
    }

    #[test]
    fn an_mf10_production_cross_section_becomes_a_yield() {
        let m = Material::from_str(&crate::testdata::text(IN115)).unwrap();
        let rx = Reaction::from_endf(4, &m).unwrap();

        // In115 inelastic leaves In115 in its first excited state.
        let product = rx
            .products
            .iter()
            .find(|p| p.name == "In115_e1")
            .expect("the activation product is named for its excited state");
        let Yield::Tabulated(y) = &product.yield_ else {
            panic!("an MF=10 yield is tabulated");
        };

        // The yield is the ratio of the two cross sections, so at any energy
        // where the reaction happens it reproduces the production one.
        let production = &m.mf10(4).unwrap().levels[0].func;
        let xs = &m.mf3(4).unwrap().sigma;
        for &e in &[1.0e6, 5.0e6, 1.0e7, 1.9e7] {
            let want = production.eval(e);
            let got = y.eval(e) * xs.eval(e);
            assert!(
                (got - want).abs() <= 1e-9 * want.abs().max(1e-30),
                "at {e} eV: {got} != {want}"
            );
        }
        // Below the threshold the reaction cross section is zero, and the
        // yield is set to zero rather than left as a division by it.
        assert_eq!(y.y[0], 0.0);
    }

    #[test]
    fn a_reaction_with_no_cross_section_is_refused() {
        let m = Material::from_str(&crate::testdata::text(AM244)).unwrap();
        assert!(Reaction::from_endf(999, &m).is_err());
    }
}

#[cfg(test)]
mod ace_tests {
    use super::*;
    use crate::ace;

    fn li6() -> ace::Table {
        crate::testdata::ace_tables(crate::testdata::LI6_ACE).remove(0)
    }

    #[test]
    fn elastic_scattering_is_index_zero() {
        let t = li6();
        let rx = Reaction::from_ace(&t, 0).unwrap();
        assert_eq!(rx.mt, 2);
        assert_eq!(rx.name().as_deref(), Some("(n,elastic)"));

        // Its cross section spans the whole grid, unlike a threshold reaction.
        let xs = rx.xs.values().next().unwrap();
        assert_eq!(xs.x.len(), t.nxs[3] as usize);
        assert!(xs.y.iter().all(|&v| v >= 0.0));
        // The temperature keys the table, and it is the file's own.
        assert_eq!(
            rx.xs.keys().next().unwrap(),
            &crate::data::temperature_str(t.temperature())
        );

        // One neutron, with an angular distribution but no energy one: the
        // outgoing energy follows from the kinematics.
        assert_eq!(rx.products.len(), 1);
        let AngleEnergy::Uncorrelated(u) = &rx.products[0].distribution[0] else {
            panic!("elastic scattering is uncorrelated");
        };
        assert!(u.angle.is_some());
        assert!(u.energy.is_none());
    }

    #[test]
    fn a_threshold_reaction_starts_where_it_opens() {
        let t = li6();
        // Index 2 is MT=51, the first inelastic level.
        let rx = Reaction::from_ace(&t, 2).unwrap();
        assert_eq!(rx.mt, 51);
        assert!(rx.q_reaction < 0.0, "an inelastic level costs energy");

        let xs = rx.xs.values().next().unwrap();
        assert!(xs.x.len() < t.nxs[3] as usize);
        // It opens above the threshold the Q value implies.
        assert!(xs.x[0] > rx.q_reaction.abs());
    }

    #[test]
    fn a_constant_multiplicity_becomes_a_polynomial_yield() {
        let t = li6();
        let rx = Reaction::from_ace(&t, 2).unwrap();
        let neutron = rx.products.iter().find(|p| p.name == "neutron").unwrap();
        // TY = 1 for a level reaction: one neutron out, whatever the energy.
        assert_eq!(
            neutron.yield_,
            Yield::Polynomial(Polynomial::new(vec![1.0]))
        );
        assert_eq!(neutron.yield_.eval(1.0e6), 1.0);
    }

    #[test]
    fn every_reaction_in_the_table_reads() {
        let t = li6();
        let mts: Vec<i32> = (0..=t.nxs[4])
            .map(|i| Reaction::from_ace(&t, i).unwrap().mt)
            .collect();
        // Elastic first, then the reactions MTR lists, in its order.
        assert_eq!(mts[0], 2);
        assert_eq!(
            mts[1..],
            (1..=t.nxs[4])
                .map(|i| t.xss[(t.jxs[3] + i - 1) as usize] as i32)
                .collect::<Vec<_>>()
        );
        // Li6 has no fission, so nothing is a derived product.
        assert!((0..=t.nxs[4]).all(|i| Reaction::from_ace(&t, i)
            .unwrap()
            .derived_products
            .is_empty()));
    }

    #[test]
    fn photon_production_is_attached_to_the_reaction_that_makes_it() {
        let t = li6();
        let mut photons = 0;
        for i in 0..=t.nxs[4] {
            let rx = Reaction::from_ace(&t, i).unwrap();
            for photon in rx.products.iter().filter(|p| p.name == "photon") {
                photons += 1;
                // A photon always carries a distribution, and the angle is
                // filled in even when the file gives none.
                let AngleEnergy::Uncorrelated(u) = &photon.distribution[0] else {
                    panic!("photon production is uncorrelated");
                };
                assert!(u.angle.is_some());
                assert!(u.energy.is_some());
            }
        }
        // NXS(6) counts the photon production reactions, and each belongs to
        // exactly one neutron reaction.
        assert_eq!(photons, t.nxs[6] as usize);
    }
}
