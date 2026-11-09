//! Continuous processing: removal, transfer between materials, and feed.
//!
//! A [`Processing`] adds terms to the Bateman equations of the materials it
//! names. Over a step, for nuclide `i` of material `m`,
//!
//! ```text
//! dN_i/dt = (A N)_i - λ N_i                     removal (and transfer out)
//!                   + λ N_i^src V_src / V_m      transfer in
//!                   + s_i                        constant feed
//!                   + k f_i N_ref                feed proportional to N_ref
//! ```
//!
//! Every term is linear in the inventory and held constant over each
//! interval it is solved on, so the solve stays a single matrix exponential
//! and stays exact. A constant feed is the one inhomogeneous term: it is made
//! homogeneous by a unit pseudo-nuclide whose column holds `s`, so
//! `exp(A t)` of the augmented matrix carries `∫ exp(A τ) s dτ` exactly.
//! Materials a transfer links are solved together as one block matrix; every
//! other material keeps its own independent solve.
//!
//! A rate given as a table is piecewise constant in time, and a step is split
//! at its breakpoints, so a table is solved exactly rather than averaged.

use std::collections::{BTreeSet, HashMap};
use std::error::Error;

use yamc_materials::material::Material;
use yani::{
    build_matrix_triplets, cram48_sparse, ChainNuclide, ChainParts, FissionYieldWeights,
    ReactionRates,
};

use crate::schedule::duration_to_seconds;
use crate::transmutation_stepper::{
    fed_during_step, reachable, total_loss_rate, ForwardEulerStepper, TransmutationStepper,
    DENSITY_FLOOR,
};

const AVOGADRO: f64 = 6.022_140_76e23;
/// cm³ per barn-cm: atoms/cm³ divided by this is atoms/barn-cm.
const CM3_PER_BARN_CM: f64 = 1e24;

/// A rate: constant, or piecewise constant in time.
#[derive(Clone, Debug, PartialEq)]
pub enum Rate {
    Constant(f64),
    /// `values[i]` holds from `times[i]` (seconds from the start of the run)
    /// until `times[i + 1]`, and the last value from its time onwards. The rate
    /// is zero before `times[0]`.
    Table {
        times: Vec<f64>,
        values: Vec<f64>,
    },
}

impl Rate {
    fn check(&self) -> Result<(), String> {
        let bad = |v: &f64| !v.is_finite() || *v < 0.0;
        match self {
            Rate::Constant(v) if bad(v) => Err(format!("rate {v} must be finite and >= 0")),
            Rate::Constant(_) => Ok(()),
            Rate::Table { times, values } => {
                if times.is_empty() || times.len() != values.len() {
                    return Err(format!(
                        "a rate table needs one value per time and at least one of each, got \
                         {} times and {} values",
                        times.len(),
                        values.len()
                    ));
                }
                if times.iter().any(bad) || times.windows(2).any(|w| w[1] <= w[0]) {
                    return Err(
                        "rate table times must be finite, >= 0 and strictly ascending".into(),
                    );
                }
                match values.iter().find(|v| bad(v)) {
                    Some(v) => Err(format!("rate {v} must be finite and >= 0")),
                    None => Ok(()),
                }
            }
        }
    }

    /// Apply `f` to every value.
    fn map(self, f: impl Fn(f64) -> Result<f64, String>) -> Result<Rate, String> {
        Ok(match self {
            Rate::Constant(v) => Rate::Constant(f(v)?),
            Rate::Table { times, values } => Rate::Table {
                times,
                values: values.into_iter().map(f).collect::<Result<_, _>>()?,
            },
        })
    }

    /// The value at time `t` (s).
    pub fn at(&self, t: f64) -> f64 {
        match self {
            Rate::Constant(v) => *v,
            Rate::Table { times, values } => match times.partition_point(|&x| x <= t) {
                0 => 0.0,
                i => values[i - 1],
            },
        }
    }

    fn breakpoints(&self) -> &[f64] {
        match self {
            Rate::Constant(_) => &[],
            Rate::Table { times, .. } => times,
        }
    }
}

/// What a transfer moves: everything (the whole liquid, say), every isotope
/// of an element, or one nuclide.
#[derive(Clone, Debug, PartialEq)]
pub enum Component {
    All,
    Element(String),
    Nuclide(String),
}

impl Component {
    /// `"H"` or `"hydrogen"` is an element, `"H3"` a nuclide.
    pub fn parse(s: &str) -> Result<Component, String> {
        if s.chars().any(|c| c.is_ascii_digit()) {
            yamc_nuclide::composition::atomic_number(s)?;
            Ok(Component::Nuclide(s.to_string()))
        } else {
            yamc_nuclide::composition::resolve_element_symbol(s).map(Component::Element)
        }
    }

    pub fn matches(&self, nuclide: &str) -> bool {
        match self {
            Component::All => true,
            Component::Nuclide(n) => n == nuclide,
            Component::Element(e) => {
                let symbol_len = nuclide
                    .find(|c: char| !c.is_ascii_alphabetic())
                    .unwrap_or(nuclide.len());
                &nuclide[..symbol_len] == e
            }
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Component::All => "all",
            Component::Element(s) | Component::Nuclide(s) => s,
        }
    }
}

/// First-order removal from `source` at `rate` [1/s], into `destination`
/// when there is one and out of the system otherwise.
#[derive(Clone, Debug)]
pub struct Transfer {
    pub source: u32,
    pub component: Component,
    pub rate: Rate,
    pub destination: Option<u32>,
    /// The schedule steps it acts on; every step when `None`.
    pub steps: Option<BTreeSet<usize>>,
}

/// Feed into `material` of `composition` (atom fractions summing to one).
/// The rate is atoms/s in total, or with `proportional_to` a rate constant
/// [1/s] times that nuclide's inventory.
#[derive(Clone, Debug)]
pub struct Feed {
    pub material: u32,
    pub composition: Vec<(String, f64)>,
    pub rate: Rate,
    pub proportional_to: Option<String>,
    pub steps: Option<BTreeSet<usize>>,
}

/// The removals, transfers and feeds of a transmutation run.
#[derive(Clone, Debug, Default)]
pub struct Processing {
    pub transfers: Vec<Transfer>,
    pub feeds: Vec<Feed>,
}

/// Seconds per `unit`, for a `"1/<unit>"` or `"<amount>/<unit>"` string.
fn per_time(units: &str) -> Option<(&str, f64)> {
    let (amount, time) = units.split_once('/')?;
    duration_to_seconds(1.0, time.trim())
        .ok()
        .map(|s| (amount.trim(), s))
}

impl Processing {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.transfers.is_empty() && self.feeds.is_empty()
    }

    /// Remove `component` (everything when `None`) from material `source` at
    /// `rate`, into `destination` when given.
    ///
    /// `rate_units` is `"1/s"` (any duration unit after the slash), or
    /// `"half-time"` for a rate given as the time [s] to remove half of it.
    pub fn add_transfer(
        &mut self,
        source: u32,
        component: Option<&str>,
        rate: Rate,
        rate_units: &str,
        destination: Option<u32>,
        steps: Option<Vec<usize>>,
    ) -> Result<(), String> {
        if destination == Some(source) {
            return Err(format!("material {source} cannot transfer into itself"));
        }
        let rate = match (rate_units.trim(), per_time(rate_units)) {
            ("half-time", _) => rate.map(|t| {
                if t > 0.0 {
                    Ok(std::f64::consts::LN_2 / t)
                } else {
                    Err("a half-time must be > 0".into())
                }
            })?,
            (_, Some(("1", seconds))) => rate.map(|v| Ok(v / seconds))?,
            _ => {
                return Err(format!(
                    "rate_units {rate_units:?} is not a removal rate; use \"1/s\" (or 1/min, \
                     1/h, 1/d, 1/a) or \"half-time\""
                ));
            }
        };
        rate.check()?;
        self.transfers.push(Transfer {
            source,
            component: component.map_or(Ok(Component::All), Component::parse)?,
            rate,
            destination,
            steps: steps.map(BTreeSet::from_iter),
        });
        Ok(())
    }

    /// Feed `composition` (nuclides or natural elements, by atom fraction)
    /// into `material`.
    ///
    /// `rate_units` is an amount per time, `"g/s"`, `"kg/d"`, `"mol/h"` or
    /// `"atoms/s"`; or with `proportional_to` a rate constant `"1/s"`, so
    /// that the feed is `rate * N(proportional_to)`.
    pub fn add_feed(
        &mut self,
        material: u32,
        composition: &HashMap<String, f64>,
        rate: Rate,
        rate_units: &str,
        proportional_to: Option<&str>,
        steps: Option<Vec<usize>>,
    ) -> Result<(), String> {
        let mut atoms: HashMap<String, f64> = HashMap::new();
        for (name, &fraction) in composition {
            if !fraction.is_finite() || fraction <= 0.0 {
                return Err(format!("feed fraction of {name} must be > 0"));
            }
            match Component::parse(name)? {
                Component::All => unreachable!("parse never returns All"),
                Component::Nuclide(n) => *atoms.entry(n).or_default() += fraction,
                Component::Element(e) => {
                    for (n, f) in yamc_nuclide::composition::expand_element(&e, fraction, "atom")? {
                        *atoms.entry(n).or_default() += f;
                    }
                }
            }
        }
        let total: f64 = atoms.values().sum();
        if total <= 0.0 {
            return Err("feed composition is empty".into());
        }
        let mut composition: Vec<(String, f64)> =
            atoms.into_iter().map(|(n, f)| (n, f / total)).collect();
        composition.sort_by(|a, b| a.0.cmp(&b.0));

        let unknown = || {
            format!(
                "rate_units {rate_units:?} is not a feed rate; use g, kg, mol or atoms per \
                 time (e.g. \"g/s\", \"kg/d\"), or \"1/s\" with proportional_to"
            )
        };
        let (amount, seconds) = per_time(rate_units).ok_or_else(unknown)?;
        let per_atom_mass = || -> Result<f64, String> {
            let grams_per_mol = composition.iter().try_fold(0.0, |acc, (n, f)| {
                yamc_nuclide::composition::atomic_mass(n).map(|m| acc + f * m)
            })?;
            Ok(AVOGADRO / grams_per_mol)
        };
        let to_atoms = match (amount, proportional_to) {
            ("1", Some(_)) => 1.0,
            ("1", None) => {
                return Err(format!(
                    "rate_units {rate_units:?} is a rate constant, which needs proportional_to"
                ));
            }
            (_, Some(_)) => {
                return Err(format!(
                    "a feed proportional_to a nuclide takes a rate constant (\"1/s\"), not \
                     {rate_units:?}"
                ));
            }
            ("atoms", None) => 1.0,
            ("mol", None) => AVOGADRO,
            ("g", None) => per_atom_mass()?,
            ("kg", None) => 1e3 * per_atom_mass()?,
            _ => return Err(unknown()),
        };
        let proportional_to = proportional_to
            .map(|n| match Component::parse(n)? {
                Component::Nuclide(n) => Ok(n),
                _ => Err(format!(
                    "proportional_to must name a nuclide (e.g. \"Li7\"), not the element {n:?}"
                )),
            })
            .transpose()?;
        let rate = rate.map(|v| Ok(v * to_atoms / seconds))?;
        rate.check()?;
        self.feeds.push(Feed {
            material,
            composition,
            rate,
            proportional_to,
            steps: steps.map(BTreeSet::from_iter),
        });
        Ok(())
    }

    /// Every material id a term names.
    pub fn material_ids(&self) -> BTreeSet<u32> {
        self.transfers
            .iter()
            .flat_map(|t| std::iter::once(t.source).chain(t.destination))
            .chain(self.feeds.iter().map(|f| f.material))
            .collect()
    }

    /// Refuse terms the run cannot solve: a material not in it, a volume a
    /// term needs and the material lacks, or a step past the schedule's end.
    pub fn check(&self, materials: &[&Material], n_steps: usize) -> Result<(), String> {
        let by_id: HashMap<u32, &Material> = materials
            .iter()
            .filter_map(|m| m.get_material_id().map(|id| (id, *m)))
            .collect();
        let volume = |id: u32, why: &str| -> Result<(), String> {
            let m = by_id.get(&id).ok_or_else(|| {
                format!("processing names material {id}, which is not in this transmutation run")
            })?;
            match m.volume {
                Some(v) if v > 0.0 => Ok(()),
                _ => Err(format!("material {id} needs a volume for {why}")),
            }
        };
        let steps_ok = |steps: &Option<BTreeSet<usize>>| match steps.as_ref().and_then(|s| s.last())
        {
            Some(&last) if last >= n_steps => Err(format!(
                "processing step {last} is past the schedule, which has {n_steps} steps"
            )),
            _ => Ok(()),
        };
        for t in &self.transfers {
            steps_ok(&t.steps)?;
            match t.destination {
                Some(d) => {
                    volume(t.source, "a transfer into another material")?;
                    volume(d, "a transfer into it")?;
                }
                None if !by_id.contains_key(&t.source) => volume(t.source, "")?,
                None => {}
            }
        }
        for f in &self.feeds {
            steps_ok(&f.steps)?;
            match f.proportional_to {
                None => volume(f.material, "an absolute feed")?,
                Some(_) if !by_id.contains_key(&f.material) => volume(f.material, "")?,
                Some(_) => {}
            }
        }
        Ok(())
    }

    /// The nuclides processing can put into material `id` that it may not
    /// otherwise hold, whose cross sections its rates need: what is fed into
    /// it, the chain's isotopes of an element or the nuclide a transfer into
    /// it carries, and for a transfer of everything the source's composition.
    /// Products arriving with it that are not among these still decay.
    pub fn nuclides_into(
        &self,
        id: u32,
        chain: &HashMap<String, ChainNuclide>,
        materials: &[&Material],
    ) -> BTreeSet<String> {
        let fed = self
            .feeds
            .iter()
            .filter(|f| f.material == id)
            .flat_map(|f| f.composition.iter().map(|(n, _)| n.clone()));
        let transferred = self
            .transfers
            .iter()
            .filter(|t| t.destination == Some(id))
            .flat_map(|t| match t.component {
                Component::All => materials
                    .iter()
                    .filter(|m| m.get_material_id() == Some(t.source))
                    .flat_map(|m| m.nuclides.keys().cloned())
                    .collect::<Vec<_>>(),
                _ => chain
                    .keys()
                    .filter(|n| t.component.matches(n))
                    .cloned()
                    .collect(),
            });
        fed.chain(transferred).collect()
    }

    fn active(steps: &Option<BTreeSet<usize>>, step: usize) -> bool {
        steps.as_ref().is_none_or(|s| s.contains(&step))
    }

    /// The step's intervals, split at every table breakpoint inside it, each
    /// with its duration and every term's rate over it.
    fn intervals(&self, step: usize, t0: f64, dt: f64) -> Vec<(f64, Vec<f64>, Vec<f64>)> {
        let t1 = t0 + dt;
        let tolerance = 1e-12 * t1.abs().max(1.0);
        let transfers: Vec<&Transfer> = self.transfers.iter().collect();
        let mut cuts: Vec<f64> = transfers
            .iter()
            .map(|t| &t.rate)
            .chain(self.feeds.iter().map(|f| &f.rate))
            .flat_map(Rate::breakpoints)
            .copied()
            .filter(|&t| t > t0 + tolerance && t < t1 - tolerance)
            .collect();
        cuts.sort_by(f64::total_cmp);
        cuts.dedup();
        let edges: Vec<f64> = std::iter::once(t0)
            .chain(cuts)
            .chain(std::iter::once(t1))
            .collect();
        edges
            .windows(2)
            .map(|w| {
                let mid = 0.5 * (w[0] + w[1]);
                let at = |rate: &Rate, steps: &Option<BTreeSet<usize>>| {
                    if Self::active(steps, step) {
                        rate.at(mid)
                    } else {
                        0.0
                    }
                };
                (
                    w[1] - w[0],
                    self.transfers
                        .iter()
                        .map(|t| at(&t.rate, &t.steps))
                        .collect(),
                    self.feeds.iter().map(|f| at(&f.rate, &f.steps)).collect(),
                )
            })
            .collect()
    }

    /// Step every member over `[t0, t0 + dt]` (step `step` of the schedule),
    /// one result per member in order.
    ///
    /// Members no term names take [`ForwardEulerStepper`] exactly; members a
    /// transfer links are solved together. Independent solves run in
    /// parallel.
    pub fn step(
        &self,
        members: &[Member<'_>],
        step: usize,
        t0: f64,
        dt: f64,
        parts: ChainParts,
    ) -> Vec<Result<Material, String>> {
        let named = self.material_ids();
        let index: HashMap<u32, usize> =
            members.iter().enumerate().map(|(i, m)| (m.id, i)).collect();
        // Union-find over the members a transfer links.
        let mut root: Vec<usize> = (0..members.len()).collect();
        fn find(root: &mut [usize], i: usize) -> usize {
            let mut r = i;
            while root[r] != r {
                r = root[r];
            }
            root[i] = r;
            r
        }
        for t in &self.transfers {
            if let (Some(&a), Some(&b)) = (
                index.get(&t.source),
                t.destination.and_then(|d| index.get(&d)),
            ) {
                let (ra, rb) = (find(&mut root, a), find(&mut root, b));
                root[ra] = rb;
            }
        }
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut group_of: HashMap<usize, usize> = HashMap::new();
        for i in 0..members.len() {
            let r = find(&mut root, i);
            let g = *group_of.entry(r).or_insert_with(|| {
                groups.push(Vec::new());
                groups.len() - 1
            });
            groups[g].push(i);
        }
        let solve = |group: &Vec<usize>| -> Vec<(usize, Result<Material, String>)> {
            let m = &members[group[0]];
            if group.len() == 1 && !named.contains(&m.id) {
                let r = ForwardEulerStepper
                    .step(m.material, m.chain, m.rates, m.fy_weights, parts, dt)
                    .map_err(|e| e.to_string());
                return vec![(group[0], r)];
            }
            let list: Vec<&Member<'_>> = group.iter().map(|&i| &members[i]).collect();
            match self.solve_group(&list, step, t0, dt, parts) {
                Ok(out) => group.iter().copied().zip(out.into_iter().map(Ok)).collect(),
                Err(e) => {
                    let e = e.to_string();
                    group.iter().map(|&i| (i, Err(e.clone()))).collect()
                }
            }
        };
        #[cfg(not(target_arch = "wasm32"))]
        let solved: Vec<Vec<(usize, Result<Material, String>)>> = {
            use rayon::prelude::*;
            groups.par_iter().map(solve).collect()
        };
        #[cfg(target_arch = "wasm32")]
        let solved: Vec<Vec<(usize, Result<Material, String>)>> =
            groups.iter().map(solve).collect();
        let mut out: Vec<Option<Result<Material, String>>> =
            (0..members.len()).map(|_| None).collect();
        for (i, r) in solved.into_iter().flatten() {
            out[i] = Some(r);
        }
        out.into_iter()
            .map(|r| r.expect("every member is in a group"))
            .collect()
    }

    /// One block matrix over the members of a group, solved interval by
    /// interval.
    fn solve_group(
        &self,
        members: &[&Member<'_>],
        step: usize,
        t0: f64,
        dt: f64,
        parts: ChainParts,
    ) -> Result<Vec<Material>, Box<dyn Error>> {
        let local: HashMap<u32, usize> =
            members.iter().enumerate().map(|(i, m)| (m.id, i)).collect();
        let intervals = self.intervals(step, t0, dt);
        let ever = |k: usize, transfer: bool| {
            intervals
                .iter()
                .any(|(_, t, f)| if transfer { t[k] > 0.0 } else { f[k] > 0.0 })
        };
        let volume = |m: usize| members[m].material.volume.unwrap_or(0.0);

        // The nuclides each member's block carries: its inventory and feeds,
        // what they reach, and what a transfer into it brings, to a fixed
        // point since a transfer can bring what the destination then reaches.
        let densities: Vec<HashMap<String, f64>> = members
            .iter()
            .map(|m| match m.material.nuclides.is_empty() {
                // An empty tank, say, which transfers will fill.
                true => Ok(HashMap::new()),
                false => m.material.get_atoms_per_barn_cm(),
            })
            .collect::<Result<_, _>>()?;
        let mut seeds: Vec<BTreeSet<String>> = densities
            .iter()
            .map(|d| {
                d.iter()
                    .filter(|&(_, &v)| v > 0.0)
                    .map(|(n, _)| n.clone())
                    .collect()
            })
            .collect();
        for (k, f) in self.feeds.iter().enumerate() {
            let Some(&m) = local.get(&f.material) else {
                continue;
            };
            if !ever(k, false) {
                continue;
            }
            for (n, _) in &f.composition {
                if !members[m].chain.contains_key(n) {
                    return Err(format!(
                        "{n} is fed into material {} but is not in the transmutation chain",
                        f.material
                    )
                    .into());
                }
                seeds[m].insert(n.clone());
            }
        }
        let names: Vec<Vec<String>> = loop {
            let names: Vec<Vec<String>> = members
                .iter()
                .zip(&seeds)
                .map(|(m, s)| reachable(s.iter().map(String::as_str), m.chain, m.rates))
                .collect();
            let mut grew = false;
            for (k, t) in self.transfers.iter().enumerate() {
                let (Some(&s), Some(&d)) = (
                    local.get(&t.source),
                    t.destination.and_then(|d| local.get(&d)),
                ) else {
                    continue;
                };
                if !ever(k, true) {
                    continue;
                }
                for n in names[s].iter().filter(|n| t.component.matches(n)) {
                    grew |= seeds[d].insert(n.clone());
                }
            }
            if !grew {
                break names;
            }
        };

        let mut offset = Vec::with_capacity(members.len());
        let mut size = 0;
        for n in &names {
            offset.push(size);
            size += n.len();
        }
        let index: Vec<HashMap<&str, usize>> = names
            .iter()
            .zip(&offset)
            .map(|(n, &o)| {
                n.iter()
                    .enumerate()
                    .map(|(i, s)| (s.as_str(), o + i))
                    .collect()
            })
            .collect();
        let feeds_absolute = self.feeds.iter().enumerate().any(|(k, f)| {
            f.proportional_to.is_none() && local.contains_key(&f.material) && ever(k, false)
        });
        let aux = size;
        if feeds_absolute {
            size += 1;
        }

        let mut base: Vec<(usize, usize, f64)> = Vec::new();
        let mut n: Vec<f64> = vec![0.0; size];
        for (m, member) in members.iter().enumerate() {
            let (triplets, _) = build_matrix_triplets(
                member.chain,
                &names[m],
                member.rates,
                member.fy_weights,
                parts,
            )?;
            base.extend(
                triplets
                    .into_iter()
                    .map(|(r, c, v)| (r + offset[m], c + offset[m], v)),
            );
            for (i, name) in names[m].iter().enumerate() {
                n[offset[m] + i] = densities[m].get(name).copied().unwrap_or(0.0);
            }
        }
        if feeds_absolute {
            n[aux] = 1.0;
        }

        for (sub_dt, transfer_rates, feed_rates) in &intervals {
            if *sub_dt <= 0.0 {
                continue;
            }
            let mut triplets = base.clone();
            for (t, &lambda) in self.transfers.iter().zip(transfer_rates) {
                let Some(&s) = local.get(&t.source) else {
                    continue;
                };
                if lambda == 0.0 {
                    continue;
                }
                let d = t.destination.and_then(|d| local.get(&d).copied());
                for name in names[s].iter().filter(|n| t.component.matches(n)) {
                    let i = index[s][name.as_str()];
                    triplets.push((i, i, -lambda));
                    if let Some(d) = d {
                        triplets.push((index[d][name.as_str()], i, lambda * volume(s) / volume(d)));
                    }
                }
            }
            for (f, &value) in self.feeds.iter().zip(feed_rates) {
                let Some(&m) = local.get(&f.material) else {
                    continue;
                };
                if value == 0.0 {
                    continue;
                }
                let (col, scale) = match &f.proportional_to {
                    Some(r) => match index[m].get(r.as_str()) {
                        Some(&c) => (c, value),
                        None => continue,
                    },
                    None => (aux, value / (volume(m) * CM3_PER_BARN_CM)),
                };
                for (name, fraction) in &f.composition {
                    triplets.push((index[m][name.as_str()], col, fraction * scale));
                }
            }
            let solved = cram48_sparse(&triplets, size, &n, *sub_dt)
                .map_err(|e| format!("CRAM48 solver error: {e}"))?;
            // The closed form where nothing feeds a nuclide, as the plain
            // stepper does: see `ForwardEulerStepper::step`.
            let fed = fed_during_step(&triplets, &n);
            let loss = total_loss_rate(&triplets, size);
            for i in 0..size {
                n[i] = if fed[i] {
                    solved[i]
                } else {
                    n[i] * (loss[i] * sub_dt).exp()
                };
            }
        }

        Ok(members
            .iter()
            .enumerate()
            .map(|(m, member)| {
                let densities: HashMap<String, f64> = names[m]
                    .iter()
                    .enumerate()
                    .map(|(i, name)| (name.clone(), n[offset[m] + i]))
                    .filter(|&(_, v)| v > DENSITY_FLOOR)
                    .collect();
                Material::from_nuclide_densities_without_data(densities, member.material)
            })
            .collect())
    }
}

/// A [`Processing`] as a run is handed it, with the materials it names: a
/// driver takes from `receivers` those it does not already hold (a storage
/// tank outside the geometry, say) and carries them unirradiated.
pub struct ProcessingRun<'a> {
    pub processing: &'a Processing,
    pub receivers: Vec<Material>,
}

/// One material's part of a [`Processing::step`]: its id, its composition at
/// the start of the step, and the chain, rates and yield weights it is solved
/// with.
pub struct Member<'a> {
    pub id: u32,
    pub material: &'a Material,
    pub chain: &'a HashMap<String, ChainNuclide>,
    pub rates: &'a ReactionRates,
    pub fy_weights: &'a FissionYieldWeights,
}

#[cfg(test)]
mod tests {
    use super::*;
    use yani::ChainReaction;

    fn nuclide(name: &str, half_life: Option<f64>, daughter: Option<&str>) -> ChainNuclide {
        ChainNuclide {
            name: name.to_string(),
            half_life,
            decay_energy: 0.0,
            reactions: vec![],
            decays: daughter
                .map(|d| ChainReaction {
                    kind: "beta-".to_string(),
                    target: Some(d.to_string()),
                    branching: 1.0,
                    q_value: None,
                    branching_uncertainty: None,
                    evaluated_branching: None,
                    multiplicity: None,
                })
                .into_iter()
                .collect(),
            fission_yields: None,
            sources: Vec::new(),
            half_life_uncertainty: None,
            decay_energy_uncertainty: None,
            decay_evaluated: true,
            excitation_energy: None,
            level_index: None,
            decay_energy_components: Default::default(),
        }
    }

    const T_HALF_H3: f64 = 3.888e8;

    fn chain() -> HashMap<String, ChainNuclide> {
        [
            nuclide("H1", None, None),
            nuclide("H2", None, None),
            nuclide("H3", Some(T_HALF_H3), Some("He3")),
            nuclide("He3", None, None),
            nuclide("Li6", None, None),
            nuclide("Li7", None, None),
            nuclide("Pb208", None, None),
        ]
        .into_iter()
        .map(|n| (n.name.clone(), n))
        .collect()
    }

    fn material(id: u32, nuclides: &[(&str, f64)], volume: f64) -> Material {
        let mut m = Material::new(
            nuclides.iter().map(|&(n, d)| (n.to_string(), d)).collect(),
            "atom",
            "sum",
            None,
        )
        .unwrap();
        m.set_material_id(id);
        m.volume = Some(volume);
        m
    }

    fn run(
        p: &Processing,
        materials: &[&Material],
        step: usize,
        t0: f64,
        dt: f64,
    ) -> Vec<Material> {
        let chain = chain();
        let none: ReactionRates = HashMap::new();
        let fy: FissionYieldWeights = HashMap::new();
        let members: Vec<Member<'_>> = materials
            .iter()
            .map(|m| Member {
                id: m.get_material_id().unwrap(),
                material: m,
                chain: &chain,
                rates: &none,
                fy_weights: &fy,
            })
            .collect();
        p.step(&members, step, t0, dt, ChainParts::default())
            .into_iter()
            .map(Result::unwrap)
            .collect()
    }

    fn get(m: &Material, n: &str) -> f64 {
        m.nuclides.get(n).copied().unwrap_or(0.0)
    }

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() <= 1e-9 * b.abs().max(1e-30), "{a} != {b}");
    }

    #[test]
    fn removal_is_exponential_and_element_wide() {
        let mut p = Processing::new();
        p.add_transfer(1, Some("H"), Rate::Constant(1.0), "1/h", None, None)
            .unwrap();
        let m = material(1, &[("H2", 1e-3), ("Pb208", 1e-2)], 1.0);
        let out = &run(&p, &[&m], 0, 0.0, 7200.0)[0];
        close(get(out, "H2"), 1e-3 * (-2.0_f64).exp());
        close(get(out, "Pb208"), 1e-2);
    }

    #[test]
    fn half_time_removes_half() {
        let mut p = Processing::new();
        p.add_transfer(
            1,
            Some("Pb208"),
            Rate::Constant(60.0),
            "half-time",
            None,
            None,
        )
        .unwrap();
        let out = &run(&p, &[&material(1, &[("Pb208", 1e-2)], 1.0)], 0, 0.0, 60.0)[0];
        close(get(out, "Pb208"), 5e-3);
    }

    #[test]
    fn transfer_conserves_atoms_and_decays_downstream() {
        let mut p = Processing::new();
        p.add_transfer(1, Some("H3"), Rate::Constant(1e-4), "1/s", Some(2), None)
            .unwrap();
        let loop_ = material(1, &[("H3", 1e-6), ("Pb208", 1e-2)], 2.0);
        let tank = material(2, &[("He3", 1e-12)], 5.0);
        let dt = 3.0e5;
        let out = run(&p, &[&loop_, &tank], 0, 0.0, dt);
        let decay = std::f64::consts::LN_2 / T_HALF_H3;
        let lambda = 1e-4;
        let in_loop = 1e-6 * (-(lambda + decay) * dt).exp();
        close(get(&out[0], "H3"), in_loop);
        // Everything not in the loop is in the tank, as H3 or He3, per volume.
        let total = |n: &[f64]| n[0] * 2.0 + n[1] * 5.0;
        let atoms_after = total(&[
            get(&out[0], "H3") + get(&out[0], "He3"),
            get(&out[1], "H3") + get(&out[1], "He3"),
        ]);
        close(atoms_after, total(&[1e-6, 1e-12]));
        // The tank's tritium: transferred in, decaying there.
        let tank_h3 = 1e-6 * 2.0 / 5.0 * lambda / lambda
            * ((-decay * dt).exp() - (-(lambda + decay) * dt).exp());
        close(get(&out[1], "H3"), tank_h3);
    }

    #[test]
    fn constant_feed_of_a_decaying_nuclide() {
        let mut p = Processing::new();
        // 1 mol/s into 10 cm3.
        p.add_feed(
            1,
            &HashMap::from([("H3".into(), 1.0)]),
            Rate::Constant(1.0),
            "mol/s",
            None,
            None,
        )
        .unwrap();
        let dt = 1e8;
        let out = &run(&p, &[&material(1, &[("Pb208", 1e-2)], 10.0)], 0, 0.0, dt)[0];
        let s = AVOGADRO / (10.0 * CM3_PER_BARN_CM);
        let decay = std::f64::consts::LN_2 / T_HALF_H3;
        close(get(out, "H3"), s / decay * (1.0 - (-decay * dt).exp()));
        close(get(out, "H3") + get(out, "He3"), s * dt);
    }

    #[test]
    fn grams_use_the_mean_atomic_mass() {
        let mut p = Processing::new();
        p.add_feed(
            1,
            &HashMap::from([("Li".into(), 1.0)]),
            Rate::Constant(6.94),
            "g/s",
            None,
            None,
        )
        .unwrap();
        let out = &run(&p, &[&material(1, &[("Pb208", 1e-2)], 1.0)], 0, 0.0, 1.0)[0];
        let atoms = (get(out, "Li6") + get(out, "Li7")) * CM3_PER_BARN_CM;
        assert!(
            (atoms / AVOGADRO - 1.0).abs() < 1e-3,
            "{}",
            atoms / AVOGADRO
        );
    }

    #[test]
    fn proportional_feed_holds_a_ratio() {
        let mut p = Processing::new();
        p.add_feed(
            1,
            &HashMap::from([("Li6".into(), 1.0)]),
            Rate::Constant(1e-3),
            "1/s",
            Some("Li7"),
            None,
        )
        .unwrap();
        let out = &run(&p, &[&material(1, &[("Li7", 1.0)], 1.0)], 0, 0.0, 10.0)[0];
        close(get(out, "Li6"), 1e-2);
    }

    #[test]
    fn tables_and_steps_switch_terms() {
        let mut p = Processing::new();
        p.add_transfer(
            1,
            Some("Pb208"),
            Rate::Table {
                times: vec![0.0, 50.0],
                values: vec![0.0, 1.0],
            },
            "1/s",
            None,
            None,
        )
        .unwrap();
        let m = material(1, &[("Pb208", 1.0)], 1.0);
        // Off for the first half of the step, on for the second.
        close(
            get(&run(&p, &[&m], 0, 0.0, 100.0)[0], "Pb208"),
            (-50.0_f64).exp(),
        );
        let mut q = Processing::new();
        q.add_transfer(
            1,
            Some("Pb208"),
            Rate::Constant(1.0),
            "1/s",
            None,
            Some(vec![1]),
        )
        .unwrap();
        close(get(&run(&q, &[&m], 0, 0.0, 1.0)[0], "Pb208"), 1.0);
        close(
            get(&run(&q, &[&m], 1, 1.0, 1.0)[0], "Pb208"),
            (-1.0_f64).exp(),
        );
    }

    #[test]
    fn a_whole_liquid_circulates_and_comes_back() {
        // Water through a core and a pool, flow Q both ways: the volumes stay
        // full, so both inventories tend to the mixed density.
        let mut p = Processing::new();
        let q = 10.0; // cm3/s
        p.add_transfer(1, None, Rate::Constant(q / 100.0), "1/s", Some(2), None)
            .unwrap();
        p.add_transfer(2, None, Rate::Constant(q / 400.0), "1/s", Some(1), None)
            .unwrap();
        let core = material(1, &[("H1", 1.0), ("H3", 1e-6)], 100.0);
        let pool = material(2, &[("H1", 1.0)], 400.0);
        let out = run(&p, &[&core, &pool], 0, 0.0, 1e4);
        close(get(&out[0], "H1"), 1.0);
        close(get(&out[1], "H1"), 1.0);
        let decay = std::f64::consts::LN_2 / T_HALF_H3;
        let mixed = 1e-6 * 100.0 / 500.0 * (-decay * 1e4).exp();
        assert!((get(&out[1], "H3") / mixed - 1.0).abs() < 1e-6);
    }

    #[test]
    fn unprocessed_members_match_the_plain_stepper() {
        let mut p = Processing::new();
        p.add_transfer(1, Some("H"), Rate::Constant(1.0), "1/s", None, None)
            .unwrap();
        let other = material(3, &[("H3", 1e-3)], 1.0);
        let out = run(
            &p,
            &[&material(1, &[("H2", 1.0)], 1.0), &other],
            0,
            0.0,
            1e7,
        );
        let plain = ForwardEulerStepper
            .step(
                &other,
                &chain(),
                &HashMap::new(),
                &HashMap::new(),
                ChainParts::default(),
                1e7,
            )
            .unwrap();
        assert_eq!(out[1].nuclides, plain.nuclides);
    }

    #[test]
    fn checks_refuse_what_cannot_be_solved() {
        let mut p = Processing::new();
        assert!(p
            .add_transfer(1, Some("H3"), Rate::Constant(1.0), "g/s", None, None)
            .is_err());
        assert!(p
            .add_transfer(1, Some("H3"), Rate::Constant(1.0), "1/s", Some(1), None)
            .is_err());
        assert!(p
            .add_feed(
                1,
                &HashMap::from([("Li6".into(), 1.0)]),
                Rate::Constant(1.0),
                "1/s",
                None,
                None
            )
            .is_err());
        p.add_transfer(
            1,
            Some("H3"),
            Rate::Constant(1.0),
            "1/s",
            Some(2),
            Some(vec![5]),
        )
        .unwrap();
        let mut a = material(1, &[("H3", 1.0)], 1.0);
        let b = material(2, &[("H3", 1.0)], 1.0);
        assert!(p
            .check(&[&a, &b], 3)
            .unwrap_err()
            .contains("past the schedule"));
        assert!(p
            .check(&[&a], 9)
            .unwrap_err()
            .contains("not in this transmutation run"));
        a.volume = None;
        assert!(p
            .check(&[&a, &b], 9)
            .unwrap_err()
            .contains("needs a volume"));
    }
}
