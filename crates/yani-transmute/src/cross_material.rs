//! Covariance between the cross sections of two evaluations.
//!
//! An MF=33 subsection whose MAT1 names another material states the
//! covariance of this evaluation's reaction MT with that material's MT1
//! (ENDF-102 33.3.1). On the published libraries these are the neutron
//! cross-section standards and the evaluations fitted with them: ENDF/B-VIII.1
//! correlates U235 fission with U238 fission and capture, Pu239 fission,
//! Au197 capture and Li6 `(n,t)`; JENDL-5.0 correlates the fissions of U233,
//! U235, U238 and Pu239 to Pu241 with one another; FENDL-3.2d adds Li6 `(n,t)`
//! with B10.
//!
//! A run applies such a block where both evaluations are in it. The nuclides
//! the applied blocks link are then drawn jointly: each keeps its own
//! covariance exactly, and the covariance the blocks state between their
//! interval averages is added, made admissible beside the members' own where
//! it is not (see `JointDraw` in [`crate::covariance_sample`]). The group is
//! fixed by the whole run, not by a material, so a nuclide is drawn the same
//! way in every material that holds it, and a material with only one of the
//! pair still reads the same draw of it.
//!
//! Only relative blocks are applied, which is every one the libraries carry.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use yamc_materials::Material;
use yamc_nuclide::covariance::expand::{expand_ni, ExpandedBlock, Scale};
use yamc_nuclide::covariance::{CovarianceBlock, CovarianceData};
use yamc_nuclide::Nuclide;
use yani::ChainNuclide;

use crate::covariance_fold::{activation_field, nuclide_transport_field, CellField};

/// A block's identity within its evaluation: `(mt, subsection, block)`.
pub type BlockKey = (i32, i32, i32);

/// Per pair of a group's members `(a, b)`, `a < b`, the stated relative
/// covariance of `a`'s interval average `p` with `b`'s `q`, as
/// `(p, q, value)`.
pub type PairEntries = BTreeMap<(usize, usize), Vec<(usize, usize, f64)>>;

/// Per nuclide, the cross-material blocks of its evaluation a run applies,
/// by identity.
pub type AppliedCross = BTreeMap<String, BTreeSet<BlockKey>>;

/// One cross-material block as a run reads it: the relative covariance of
/// reaction `mt` of nuclide `row` with reaction `mt1` of nuclide `col`.
#[derive(Debug, Clone)]
struct Link {
    row: String,
    mt: i32,
    col: String,
    mt1: i32,
    block: ExpandedBlock,
    /// The blocks this link reads, by nuclide: the one it was read from and
    /// the other evaluation's copy of the same pair, when it states one.
    keys: Vec<(String, BlockKey)>,
}

/// One nuclide a link names, with the data its field is built from.
#[derive(Debug, Clone)]
struct Linked {
    nuclide: Arc<Nuclide>,
    temperature: String,
}

/// Every cross-material covariance block one run applies, and the nuclides
/// they link.
///
/// Built once from every material of the run, so that each material reads
/// the same groups: see the module documentation.
#[derive(Debug, Clone, Default)]
pub struct CrossMaterial {
    links: Vec<Link>,
    nuclides: BTreeMap<String, Linked>,
}

/// The nuclides one joint draw covers, and their covariance together.
pub struct JointGroup {
    /// Ascending by name.
    pub members: Vec<String>,
    /// Per member, the interval averages of its cross sections the links
    /// state a covariance for.
    pub averages: Vec<Vec<IntervalAverage>>,
    /// Per pair of members `(a, b)`, `a < b` by index into `members`, the
    /// stated relative covariance of `a`'s average `p` with `b`'s average
    /// `q`, as `(p, q, value)`, stated zeros included.
    pub links: PairEntries,
}

/// A member's cross section averaged over one interval of a link's grid,
/// as ENDF-102 33.2.2 defines the quantity an NI block's entry is a
/// covariance of, in terms of the member's relative cells.
#[derive(Debug, Clone, PartialEq)]
pub struct IntervalAverage {
    pub mt: i32,
    pub lo: f64,
    pub hi: f64,
    /// `(cell, weight)`: the share of the interval's lethargy each of the
    /// reaction's cells covers. They add up to less than one where the
    /// reaction has no cell over part of the interval, which states no
    /// uncertainty there.
    pub weights: Vec<(usize, f64)>,
}

impl JointGroup {
    /// The name the group's draw is keyed on.
    pub fn key(&self) -> String {
        self.members.join("+")
    }
}

impl CrossMaterial {
    /// The cross-material blocks of every nuclide in `materials` that names
    /// another nuclide among them, read from their covariance.
    ///
    /// A nuclide's MAT is read off its blocks, so one with no covariance, or
    /// from a file that does not record its MAT, cannot be named. A pair
    /// stated in both evaluations, as JENDL-5.0 states every fission pair, is
    /// read from the one with the lower MAT. A nuclide in several materials
    /// is read from the first that loaded its covariance.
    pub fn new(materials: &[&Material]) -> Self {
        let mut nuclides: BTreeMap<String, Linked> = BTreeMap::new();
        for material in materials {
            let mut names: Vec<&String> = material.nuclide_data.keys().collect();
            names.sort();
            for name in names {
                let nuclide = &material.nuclide_data[name];
                if nuclide.covariance.is_some() && !nuclides.contains_key(name) {
                    nuclides.insert(
                        name.clone(),
                        Linked {
                            nuclide: Arc::clone(nuclide),
                            temperature: material.temperature().to_string(),
                        },
                    );
                }
            }
        }
        let mat_of = |linked: &Linked| {
            linked
                .nuclide
                .covariance
                .as_ref()
                .and_then(|b| b.first())
                .map_or(0, |b| b.mat)
        };
        let by_mat: HashMap<i32, &String> = nuclides
            .iter()
            .map(|(name, l)| (mat_of(l), name))
            .filter(|(mat, _)| *mat != 0)
            .collect();

        // Every applicable block, keyed by its pair, lower MAT first, so a
        // pair both evaluations state is read once.
        let mut pairs: BTreeMap<(i32, i32, i32, i32), Vec<Link>> = BTreeMap::new();
        for (name, linked) in &nuclides {
            let mat = mat_of(linked);
            for block in linked.nuclide.covariance.iter().flat_map(|b| b.iter()) {
                let Some(partner) = applicable(block).and_then(|_| by_mat.get(&block.mat1)) else {
                    continue;
                };
                let CovarianceData::Ni(ni) = &block.data else {
                    continue;
                };
                let Ok(expanded) = expand_ni(ni) else {
                    continue;
                };
                if expanded.scale != Scale::Relative || expanded.is_empty() {
                    continue;
                }
                let (mt1, key) = (block.partner_mt(), key_of(block));
                let pair = if mat < block.mat1 {
                    (mat, block.mt, block.mat1, mt1)
                } else {
                    (block.mat1, mt1, mat, block.mt)
                };
                pairs.entry(pair).or_default().push(Link {
                    row: name.clone(),
                    mt: block.mt,
                    col: (*partner).clone(),
                    mt1,
                    block: expanded,
                    keys: vec![(name.clone(), key)],
                });
            }
        }
        let mut links = Vec::new();
        for ((low, ..), stated) in pairs {
            // The lower MAT's blocks of the pair, added as the blocks of one
            // subsection add; the other's are the same pair transposed.
            let (read, mirrored): (Vec<Link>, Vec<Link>) = stated
                .into_iter()
                .partition(|l| by_mat.get(&low) == Some(&&l.row));
            let mirror_keys: Vec<(String, BlockKey)> =
                mirrored.iter().flat_map(|l| l.keys.clone()).collect();
            let mut read = if read.is_empty() { mirrored } else { read };
            if let Some(first) = read.first_mut() {
                first.keys.extend(mirror_keys);
            }
            links.extend(read);
        }

        let linked: BTreeSet<&String> = links.iter().flat_map(|l| [&l.row, &l.col]).collect();
        let nuclides = nuclides
            .into_iter()
            .filter(|(name, _)| linked.contains(name))
            .collect();
        CrossMaterial { links, nuclides }
    }

    /// Whether the run has no cross-material block to apply.
    pub fn is_empty(&self) -> bool {
        self.links.is_empty()
    }

    /// The activation field of every linked nuclide, its cells fixed by the
    /// full `chain`, with no spectrum to read it: what a material that lacks
    /// one of a group's nuclides still needs to draw the group. The nuclides
    /// in `without_mf32_rows` are fielded from their tape blocks alone, as
    /// [`crate::covariance_fold::cell_fields`] fields them.
    pub fn activation_fields(
        &self,
        chain: &HashMap<String, ChainNuclide>,
        without_mf32_rows: &BTreeSet<String>,
    ) -> BTreeMap<String, CellField> {
        self.nuclides
            .iter()
            .filter_map(|(name, l)| {
                let field = activation_field(
                    name,
                    chain.get(name)?,
                    &l.nuclide,
                    &l.temperature,
                    without_mf32_rows.contains(name),
                    &[],
                    &[],
                )?;
                Some((name.clone(), field))
            })
            .collect()
    }

    /// The transport field of every linked nuclide, as
    /// [`crate::covariance_fold::transport_fields`] builds it, the ones in
    /// `without_mf32_rows` from their tape blocks alone.
    pub fn transport_fields(
        &self,
        without_mf32_rows: &BTreeSet<String>,
    ) -> BTreeMap<String, CellField> {
        self.nuclides
            .iter()
            .filter_map(|(name, l)| {
                let t = nuclide_transport_field(
                    name,
                    &l.nuclide,
                    &l.temperature,
                    &[],
                    without_mf32_rows.contains(name),
                )?;
                Some((name.clone(), t.field?))
            })
            .collect()
    }

    /// The blocks `fields` lets the run apply: those between two nuclides
    /// with fields, each with relative cells on its reaction.
    pub fn applied(&self, fields: &BTreeMap<String, &CellField>) -> AppliedCross {
        let has = |name: &str, mt: i32| {
            fields
                .get(name)
                .is_some_and(|f| f.relative_cells.iter().any(|c| c.mt == mt))
        };
        let mut out = AppliedCross::new();
        for link in &self.links {
            if has(&link.row, link.mt) && has(&link.col, link.mt1) {
                for (name, key) in &link.keys {
                    out.entry(name.clone()).or_default().insert(*key);
                }
            }
        }
        out
    }

    /// The groups of nuclides among `fields` the run's links correlate, each
    /// with the covariance its links state.
    ///
    /// A link states the covariance of one reaction's cross section
    /// averaged over each interval of its row grid with the other's averaged
    /// over each of its column grid (ENDF-102 33.2.2), and that is how it is
    /// kept: as a covariance between [`IntervalAverage`]s, each a weighted
    /// sum of the member's own cells. The cells are not split at the block's
    /// edges, since a cell is one random variable of its own evaluation,
    /// which states its parts fully correlated; splitting ENDF/B-VIII.1 U235
    /// and U238 fission that way stated a covariance no matrix has, a
    /// correlation repair of 1.1. Two nuclides are in one group when a link
    /// between them has a nonzero entry.
    pub fn joint_groups(&self, fields: &BTreeMap<String, &CellField>) -> Vec<JointGroup> {
        // Per nuclide, its averages, keyed by (MT, interval) so two links on
        // one grid share them.
        type Key = (i32, u64, u64);
        let mut averages: BTreeMap<String, Vec<IntervalAverage>> = BTreeMap::new();
        let mut index: BTreeMap<(String, Key), usize> = BTreeMap::new();
        let mut average = |name: &String, field: &CellField, mt: i32, lo: f64, hi: f64| {
            let key = (name.clone(), (mt, lo.to_bits(), hi.to_bits()));
            if let Some(&p) = index.get(&key) {
                return Some(p);
            }
            let width = (hi / lo).ln();
            let weights: Vec<(usize, f64)> = field
                .relative_cells
                .iter()
                .enumerate()
                .filter(|(_, c)| c.mt == mt)
                .filter_map(|(k, c)| {
                    let (a, b) = (c.lo.max(lo), c.hi.min(hi));
                    (a < b && width > 0.0).then(|| (k, (b / a).ln() / width))
                })
                .collect();
            if weights.is_empty() {
                return None;
            }
            let list = averages.entry(name.clone()).or_default();
            list.push(IntervalAverage {
                mt,
                lo,
                hi,
                weights,
            });
            index.insert(key, list.len() - 1);
            Some(list.len() - 1)
        };
        type Entries = Vec<(usize, usize, f64)>;
        let mut entries: BTreeMap<(String, String), Entries> = BTreeMap::new();
        for link in &self.links {
            let (Some(a), Some(b)) = (fields.get(&link.row), fields.get(&link.col)) else {
                continue;
            };
            let (er, ec) = (&link.block.row_energies, &link.block.col_energies);
            let mut list = Vec::new();
            for i in 0..link.block.n_rows() {
                for j in 0..link.block.n_cols() {
                    let v = link.block.get(i, j);
                    let p = average(&link.row, a, link.mt, er[i], er[i + 1]);
                    let q = average(&link.col, b, link.mt1, ec[j], ec[j + 1]);
                    if let (Some(p), Some(q)) = (p, q) {
                        list.push((p, q, v));
                    }
                }
            }
            entries
                .entry((link.row.clone(), link.col.clone()))
                .or_default()
                .extend(list);
        }
        // A stated zero is kept, as the fit holds it ([`crate::joint_fit`]),
        // but blocks of zeros alone link nothing.
        entries.retain(|_, list| list.iter().any(|e| e.2 != 0.0));

        // Connected components over the nuclides the entries join.
        let mut parent: BTreeMap<&String, &String> = BTreeMap::new();
        fn root<'a>(parent: &BTreeMap<&'a String, &'a String>, mut n: &'a String) -> &'a String {
            while let Some(&p) = parent.get(n) {
                if p == n {
                    break;
                }
                n = p;
            }
            n
        }
        for (a, b) in entries.keys() {
            parent.entry(a).or_insert(a);
            parent.entry(b).or_insert(b);
            let (ra, rb) = (root(&parent, a), root(&parent, b));
            if ra != rb {
                let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
                parent.insert(hi, lo);
            }
        }
        let mut components: BTreeMap<&String, Vec<String>> = BTreeMap::new();
        for &n in parent.keys() {
            components
                .entry(root(&parent, n))
                .or_default()
                .push(n.clone());
        }

        components
            .into_values()
            .map(|members| {
                let at = |name: &String| members.iter().position(|m| m == name);
                let mut links = PairEntries::new();
                for ((a, b), list) in &entries {
                    let (Some(ia), Some(ib)) = (at(a), at(b)) else {
                        continue;
                    };
                    let out = links.entry((ia.min(ib), ia.max(ib))).or_default();
                    for &(k, l, v) in list {
                        out.push(if ia < ib { (k, l, v) } else { (l, k, v) });
                    }
                }
                let averages = members
                    .iter()
                    .map(|m| averages.get(m).cloned().unwrap_or_default())
                    .collect();
                JointGroup {
                    members,
                    averages,
                    links,
                }
            })
            .collect()
    }
}

/// Whether `block` is a cross-material covariance of cross sections, given
/// explicitly, that a run could apply.
fn applicable(block: &CovarianceBlock) -> Option<()> {
    (block.is_cross_material()
        && block.names_cross_section()
        && matches!(block.data, CovarianceData::Ni(_)))
    .then_some(())
}

fn key_of(block: &CovarianceBlock) -> BlockKey {
    (block.mt, block.subsection_idx, block.block_idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::covariance_sample::Sampler;
    use endf::mf::covariance::NiSubsection;
    use yamc_nuclide::reaction::Reaction;

    const GRID: [f64; 3] = [1.0e-5, 1.0e6, 2.0e7];
    const U235: i32 = 9228;
    const U238: i32 = 9237;

    fn block(
        mat: i32,
        mt: i32,
        mat1: i32,
        mt1: i32,
        sub: i32,
        ni: NiSubsection,
    ) -> CovarianceBlock {
        CovarianceBlock {
            mt,
            subsection_idx: sub,
            block_idx: 0,
            mat1,
            mt1,
            xmf1: 0.0,
            xlfs1: 0.0,
            mtl: 0,
            mat,
            data: CovarianceData::Ni(ni),
        }
    }

    /// `lb = 5`, `ls = 1` on `GRID`, upper triangle `[a, b, c]`.
    fn own(mat: i32, upper: [f64; 3]) -> CovarianceBlock {
        let ni = NiSubsection {
            lb: 5,
            ls: 1,
            ne: GRID.len() as i64,
            ek: GRID.to_vec(),
            fkk: upper.to_vec(),
            ..Default::default()
        };
        block(mat, 18, 0, 18, 0, ni)
    }

    /// `lb = 6`: fission of `mat` with fission of `mat1`, on its own grids.
    fn cross(mat: i32, mat1: i32, er: &[f64], ec: &[f64], fkl: &[f64]) -> CovarianceBlock {
        let ni = NiSubsection {
            lb: 6,
            ner: er.len() as i64,
            nec: ec.len() as i64,
            nt: 1 + fkl.len() as i64,
            er: er.to_vec(),
            ec: ec.to_vec(),
            fkl: fkl.to_vec(),
            ..Default::default()
        };
        block(mat, 18, mat1, 18, 1, ni)
    }

    fn nuclide(name: &str, blocks: Vec<CovarianceBlock>) -> Nuclide {
        let fission = Reaction {
            cross_section: vec![1.0, 1.0, 1.0].into(),
            threshold_idx: 0,
            energy: GRID.to_vec().into(),
            mt_number: 18,
            q_value: 0.0,
            products: vec![].into(),
            scatter_in_cm: false,
            redundant: false,
        };
        let temperature = "294".to_string();
        Nuclide {
            name: Some(name.to_string()),
            element: None,
            atomic_symbol: None,
            atomic_number: None,
            neutron_number: None,
            mass_number: None,
            atomic_weight_ratio: None,
            library: None,
            energy: None,
            reactions: vec![HashMap::from([(18, Arc::new(fission))])],
            fissionable: true,
            available_temperatures: vec![temperature.clone()],
            loaded_temperatures: vec![temperature],
            data_path: None,
            data_source: None,
            fission_nu: None,
            fast_xs: vec![],
            urr_data: vec![],
            urr_present: false,
            fission_photon_release: None,
            covariance: Some(Arc::new(blocks)),
            angular_covariance: None,
            nubar_covariance: None,
            spectrum_covariance: None,
            resonance_parameters: None,
            elastic_flat_cache: Default::default(),
            fission_chi_flat_cache: Default::default(),
            delayed_neutron_cache: Default::default(),
            inelastic_angle_flat_cache: Default::default(),
            load_scope: Default::default(),
        }
    }

    fn material(nuclides: &[(&str, &Nuclide)]) -> Material {
        let mut m = Material::new(
            nuclides.iter().map(|(n, _)| (n.to_string(), 1.0)).collect(),
            "atom",
            "sum",
            None,
        )
        .expect("material");
        m.set_temperature("294");
        for (name, n) in nuclides {
            m.nuclide_data
                .insert(name.to_string(), Arc::new((*n).clone()));
        }
        m
    }

    const ER: [f64; 3] = [1.0e-5, 1.0e5, 2.0e7];
    const EC: [f64; 2] = [1.0e-5, 2.0e7];
    const FKL: [f64; 2] = [0.003, 0.004];

    /// U235 states its fission's covariance with U238's on a grid of its
    /// own, as ENDF/B-VIII.1 does, and U238 states nothing back.
    fn pair() -> (Nuclide, Nuclide) {
        (
            nuclide(
                "U235",
                vec![
                    own(U235, [0.01, 0.002, 0.04]),
                    cross(U235, U238, &ER, &EC, &FKL),
                ],
            ),
            nuclide("U238", vec![own(U238, [0.02, 0.0, 0.01])]),
        )
    }

    fn fields(m: &Material) -> BTreeMap<String, CellField> {
        crate::covariance_fold::transport_fields(m, &BTreeSet::new())
            .0
            .into_iter()
            .filter_map(|(n, t)| Some((n, t.field?)))
            .collect()
    }

    /// The block is kept as a covariance between interval averages of each
    /// nuclide's own cells: U235's interval below 1e5 eV is its first cell,
    /// the one above takes both cells by their share of its lethargy, and
    /// U238's one interval takes both of its own.
    #[test]
    fn a_link_is_kept_between_interval_averages() {
        let (a, b) = pair();
        let m = material(&[("U235", &a), ("U238", &b)]);
        let cross = CrossMaterial::new(&[&m]);
        assert!(!cross.is_empty());
        let f = fields(&m);
        assert_eq!(f["U235"].relative_cells.len(), 2);
        assert_eq!(f["U238"].relative_cells.len(), 2);
        let refs: BTreeMap<String, &CellField> = f.iter().map(|(n, c)| (n.clone(), c)).collect();
        let groups = cross.joint_groups(&refs);
        assert_eq!(groups.len(), 1);
        let g = &groups[0];
        assert_eq!(g.members, ["U235", "U238"]);
        let share = |lo: f64, hi: f64, a: f64, b: f64| (b / a).ln() / (hi / lo).ln();
        let (u235, u238) = (&g.averages[0], &g.averages[1]);
        assert_eq!(u235.len(), 2);
        assert_eq!(u235[0].weights, [(0, 1.0)]);
        let above = &u235[1].weights;
        assert_eq!(above.len(), 2);
        assert!((above[0].1 - share(1.0e5, 2.0e7, 1.0e5, 1.0e6)).abs() < 1e-15);
        assert!((above[1].1 - share(1.0e5, 2.0e7, 1.0e6, 2.0e7)).abs() < 1e-15);
        assert_eq!(u238.len(), 1);
        assert_eq!(g.links[&(0, 1)], [(0, 0, FKL[0]), (1, 0, FKL[1])]);
        let applied = cross.applied(&refs);
        assert_eq!(applied["U235"], BTreeSet::from([(18, 1, 0)]));
    }

    /// The averages' log-space covariance between two nuclides' cells,
    /// `Σ_kl w_pk w_ql ln(1 + S_kl)`, from the drawn cell covariance `S`.
    fn averaged(g: &JointGroup, drawn: &[f64], nb: usize, p: usize, q: usize) -> f64 {
        let (wa, wb) = (&g.averages[0][p].weights, &g.averages[1][q].weights);
        wa.iter()
            .flat_map(|&(k, x)| {
                wb.iter()
                    .map(move |&(l, y)| x * y * drawn[k * nb + l].ln_1p())
            })
            .sum()
    }

    /// Without its partner in the run a nuclide's draw is exactly what it
    /// is with no cross-material block at all.
    #[test]
    fn without_the_partner_nothing_changes() {
        let (a, _) = pair();
        let m = material(&[("U235", &a)]);
        let cross = CrossMaterial::new(&[&m]);
        assert!(cross.is_empty());
        let coupled = fields(&m);
        let s = Sampler::coupled(
            &coupled,
            &[],
            &cross,
            &cross.transport_fields(&BTreeSet::new()),
        );
        assert_eq!(s.drawn_with("U235"), None);
        assert_eq!(s.deviates(7, 3), Sampler::new(&coupled, &[]).deviates(7, 3));
    }

    /// Drawn jointly, the two fissions' interval averages covary as the
    /// block states, and each nuclide keeps its own variance.
    #[test]
    fn the_joint_draw_carries_the_link() {
        let (a, b) = pair();
        let m = material(&[("U235", &a), ("U238", &b)]);
        let cross = CrossMaterial::new(&[&m]);
        let f = fields(&m);
        let s = Sampler::coupled(&f, &[], &cross, &cross.transport_fields(&BTreeSet::new()));
        assert_eq!(
            s.drawn_with("U238"),
            Some(&["U235".to_string(), "U238".to_string()][..])
        );
        // The averages covary in log space exactly as stated, the link being
        // admissible beside the two nuclides' own covariance.
        let refs: BTreeMap<String, &CellField> = f.iter().map(|(n, c)| (n.clone(), c)).collect();
        let g = &cross.joint_groups(&refs)[0];
        let drawn = s
            .sampled_cross_covariance("U235", "U238")
            .expect("both drawn");
        for (p, v) in FKL.iter().enumerate() {
            let got = averaged(g, &drawn, 2, p, 0);
            assert!((got - v.ln_1p()).abs() < 1e-12, "{got} vs {}", v.ln_1p());
        }
        assert!(s.joint_adjustments().is_empty());
        // The draws carry it, cell by cell, and each nuclide its own.
        let draws = 40_000;
        let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
        for r in 0..draws {
            let d = s.deviates(11, r);
            let (x, y) = (d["U235"][0], d["U238"][0]);
            sab += x * y;
            saa += x * x;
            sbb += y * y;
        }
        let n = draws as f64;
        // Sampling error of each moment is a few times 1e-4 at this count.
        assert!(
            (sab / n - drawn[0]).abs() < 6.0e-4,
            "{} vs {}",
            sab / n,
            drawn[0]
        );
        assert!(drawn[0] > 0.002, "{drawn:?}");
        assert!((saa / n - 0.01).abs() < 6.0e-4, "{}", saa / n);
        assert!((sbb / n - 0.02).abs() < 1.2e-3, "{}", sbb / n);
    }

    /// A stated covariance past a correlation of one beside the two
    /// evaluations' own is drawn at the closest admissible one, a
    /// correlation of one (to the fit's tolerance): each nuclide keeps its
    /// own covariance exactly, the two are drawn fully correlated, and the
    /// adjustment is reported.
    #[test]
    fn an_inadmissible_link_is_clipped_and_reported() {
        let one = [1.0e-5, 2.0e7];
        let flat = |mat: i32| {
            let ni = NiSubsection {
                lb: 5,
                ls: 1,
                ne: 2,
                ek: one.to_vec(),
                fkk: vec![0.01],
                ..Default::default()
            };
            block(mat, 18, 0, 18, 0, ni)
        };
        let a = nuclide(
            "U235",
            vec![flat(U235), cross(U235, U238, &one, &one, &[0.02])],
        );
        let b = nuclide("U238", vec![flat(U238)]);
        let m = material(&[("U235", &a), ("U238", &b)]);
        let cross = CrossMaterial::new(&[&m]);
        let f = fields(&m);
        let s = Sampler::coupled(&f, &[], &cross, &cross.transport_fields(&BTreeSet::new()));
        let drawn = s.sampled_cross_covariance("U235", "U238").unwrap();
        assert!((drawn[0] - 0.01).abs() < 1e-8, "{drawn:?}");
        for n in ["U235", "U238"] {
            assert_eq!(s.sampled_covariance(n).unwrap().0, [0.01]);
        }
        let adjusted = s.joint_adjustments();
        let j = adjusted["U235+U238"];
        let want = 1.02_f64.ln() / 1.01_f64.ln();
        assert!((j.largest_stated_correlation - want).abs() < 1e-9, "{j:?}");
        assert!(j.relative_change > 0.4, "{j:?}");
        assert!(
            (j.relative_change - j.least_relative_change).abs() < 1e-6,
            "{j:?}"
        );
        assert!(j.largest_correlation_change < 1e-6 && j.converged, "{j:?}");
        // The draws carry it: the two multipliers, each of sigma 0.1, move
        // together.
        for r in 0..20 {
            let d = s.deviates(5, r);
            let (x, y) = (d["U235"][0], d["U238"][0]);
            assert!((x - y).abs() < 1e-4, "{x} {y}");
        }
    }

    /// The fit moves only what it must: of two uncorrelated cells per
    /// nuclide, the first pair stated past a correlation of one is drawn at
    /// one, and the second pair, stated within it, is drawn as stated, as
    /// are the stated zeros between them.
    #[test]
    fn the_fit_leaves_an_admissible_direction_as_stated() {
        let diagonal = |mat: i32| {
            let ni = NiSubsection {
                lb: 1,
                np: 3,
                ek: GRID.to_vec(),
                fk: vec![0.01, 0.01, 0.0],
                ..Default::default()
            };
            block(mat, 18, 0, 18, 0, ni)
        };
        let a = nuclide(
            "U235",
            vec![
                diagonal(U235),
                cross(U235, U238, &GRID, &GRID, &[0.02, 0.0, 0.0, 0.005]),
            ],
        );
        let b = nuclide("U238", vec![diagonal(U238)]);
        let m = material(&[("U235", &a), ("U238", &b)]);
        let cross = CrossMaterial::new(&[&m]);
        let s = Sampler::coupled(
            &fields(&m),
            &[],
            &cross,
            &cross.transport_fields(&BTreeSet::new()),
        );
        let drawn = s.sampled_cross_covariance("U235", "U238").unwrap();
        assert!((drawn[0] - 0.01).abs() < 1e-8, "{drawn:?}");
        assert!(drawn[1].abs() < 1e-8 && drawn[2].abs() < 1e-8, "{drawn:?}");
        assert!((drawn[3] - 0.005).abs() < 1e-8, "{drawn:?}");
    }

    /// Three evaluations, U235 linked to U238 and to Pu239, are one group:
    /// every stated pair is carried at its interval averages, and the pair
    /// no block states, U238 with Pu239, covaries only through U235.
    #[test]
    fn a_group_of_three_carries_every_stated_pair() {
        const PU239: i32 = 9437;
        let (_, b) = pair();
        let a = nuclide(
            "U235",
            vec![
                own(U235, [0.01, 0.002, 0.04]),
                cross(U235, U238, &ER, &EC, &FKL),
                cross(U235, PU239, &EC, &EC, &[0.002]),
            ],
        );
        let c = nuclide("Pu239", vec![own(PU239, [0.03, 0.0, 0.02])]);
        let m = material(&[("U235", &a), ("U238", &b), ("Pu239", &c)]);
        let cross = CrossMaterial::new(&[&m]);
        let f = fields(&m);
        let s = Sampler::coupled(&f, &[], &cross, &cross.transport_fields(&BTreeSet::new()));
        assert_eq!(s.drawn_with("Pu239").map(|m| m.len()), Some(3));
        assert!(
            s.joint_adjustments().is_empty(),
            "{:?}",
            s.joint_adjustments()
        );
        let refs: BTreeMap<String, &CellField> = f.iter().map(|(n, c)| (n.clone(), c)).collect();
        let g = &cross.joint_groups(&refs)[0];
        assert_eq!(g.members, ["Pu239", "U235", "U238"]);
        // Pu239 (0) with U235 (1): one average each.
        let drawn = s.sampled_cross_covariance("Pu239", "U235").unwrap();
        let drawn = &drawn;
        let (wp, wu) = (&g.averages[0][0].weights, &g.averages[1]);
        let at = |p: usize| -> f64 {
            wp.iter()
                .flat_map(|&(k, x)| {
                    wu[p]
                        .weights
                        .iter()
                        .map(move |&(l, y)| x * y * drawn[k * 2 + l].ln_1p())
                })
                .sum()
        };
        let whole = wu
            .iter()
            .position(|a| a.lo == EC[0] && a.hi == EC[1])
            .unwrap();
        assert!(
            (at(whole) - 0.002_f64.ln_1p()).abs() < 1e-12,
            "{}",
            at(whole)
        );
        // U238 with Pu239 is not stated, and is not zero: both move with U235.
        let implied = s.sampled_cross_covariance("U238", "Pu239").unwrap();
        assert!(implied.iter().any(|v| *v != 0.0), "{implied:?}");
        for n in ["U235", "U238", "Pu239"] {
            let (own, _) = s.sampled_covariance(n).unwrap();
            let (alone, _) = Sampler::new(&f, &[])
                .sampled_covariance(n)
                .map(|(a, b)| (a.to_vec(), b.to_vec()))
                .unwrap();
            assert_eq!(own, &alone[..], "{n} keeps its own covariance");
        }
    }

    /// A material holding only U238 draws it exactly as one holding both
    /// does, replica by replica: the group is the run's, not the
    /// material's.
    #[test]
    fn every_material_draws_a_group_alike() {
        let (a, b) = pair();
        let both = material(&[("U235", &a), ("U238", &b)]);
        let alone = material(&[("U238", &b)]);
        let cross = CrossMaterial::new(&[&both, &alone]);
        let partners = cross.transport_fields(&BTreeSet::new());
        let s_both = Sampler::coupled(&fields(&both), &[], &cross, &partners);
        let s_alone = Sampler::coupled(&fields(&alone), &[], &cross, &partners);
        assert!(s_alone.drawn_with("U238").is_some());
        for r in 0..5 {
            assert_eq!(
                s_both.deviates(3, r)["U238"],
                s_alone.deviates(3, r)["U238"]
            );
        }
    }

    /// A pair both evaluations state, as JENDL-5.0 states its fissions, is
    /// read once, from the lower MAT, and both copies count as applied.
    #[test]
    fn a_pair_stated_by_both_is_read_once() {
        let (a, mut b) = pair();
        let mut blocks = (*b.covariance.take().unwrap()).clone();
        blocks.push(cross(U238, U235, &EC, &ER, &FKL));
        b.covariance = Some(Arc::new(blocks));
        let m = material(&[("U235", &a), ("U238", &b)]);
        let cross = CrossMaterial::new(&[&m]);
        let f = fields(&m);
        let refs: BTreeMap<String, &CellField> = f.iter().map(|(n, c)| (n.clone(), c)).collect();
        let g = &cross.joint_groups(&refs)[0];
        assert_eq!(
            g.links[&(0, 1)],
            [(0, 0, FKL[0]), (1, 0, FKL[1])],
            "not twice"
        );
        let applied = cross.applied(&refs);
        assert_eq!(applied["U235"].len(), 1);
        assert_eq!(applied["U238"].len(), 1);
    }
}
