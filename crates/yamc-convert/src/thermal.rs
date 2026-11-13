//! Thermal scattering tables: one `thermal/<name>.arrow/` folder per bound
//! nuclide in a material (`c_H_in_H2O`), from a thermal scattering law
//! evaluation and the neutron evaluation of its principal atom.
//!
//! The inelastic part goes through NJOY (`endf::njoy::make_ace_thermal`):
//! S(alpha, beta) cannot be sampled as tabulated, and THERMR's integration
//! turns it into outgoing energy and cosine distributions. The elastic part is
//! closed form and is taken from MF=7 MT=2, with THERMR's and ACER's
//! normalisation (SB per principal atom, everything divided by `nmix`), so
//! transport can sample its exact angular distribution rather than ACER's
//! equiprobable cosines; [`Prepared::check_against_njoy`] holds it to the
//! elastic part NJOY wrote. The tabulated law itself is kept too
//! (`sab.arrow`).
//!
//! Every per-temperature section is one row and one record batch per
//! temperature, and `version.json` records where each batch sits
//! (`temperature_ranges`), so a reader fetches only the temperature it uses.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::builder::{Float32Builder, ListBuilder};
use arrow_array::ArrayRef;
use endf::data::temperature_str;
use endf::thermal_ace::InelasticDistribution;
use endf::thermal_scattering::SecondaryModel;
use endf::{Material, ThermalAce, ThermalElastic, ThermalScattering};

use crate::entry::Provenance;
use crate::sections::{
    bools, float_list, float_list_lists, float_lists, floats, int_list, ints, opt_floats,
    opt_strings, string_list, strings, write_section, write_section_per_row,
};

/// The per-temperature sections, in the order `temperature_ranges` lists them.
pub const PER_TEMPERATURE_SECTIONS: [&str; 3] = ["inelastic.arrow", "elastic.arrow", "sab.arrow"];

/// What a thermal table is built from.
#[derive(Debug, Clone)]
pub struct ThermalSource<'a> {
    /// The table's name, which is also its folder: `c_H_in_H2O`.
    pub name: &'a str,
    /// The thermal scattering law evaluation. Several files when a library
    /// publishes one file per temperature (JEFF-4.0's fuel tables); they are
    /// merged into one table, and must agree on everything but temperature.
    pub thermal: &'a [PathBuf],
    /// The incident-neutron evaluation of the principal atom.
    pub neutron: &'a Path,
    /// The nuclides of the neutron library the table applies to.
    pub bound_nuclides: &'a [String],
    /// The ZA values ACER records, and its six-character table name.
    pub zaids: &'a [i64],
    pub ace_name: &'a str,
    /// ACER's number of atom types in a mixed moderator.
    pub nmix: u32,
    pub njoy_exec: &'a str,
}

/// One temperature of a table, everything its rows need.
struct Temperature {
    label: String,
    kelvin: f64,
    ace: ThermalAce,
    /// Index of this temperature in `law`'s own list.
    index: usize,
    law: usize,
}

/// A table ready to write: its laws (one per source file) and temperatures,
/// sorted.
pub struct Prepared {
    pub name: String,
    pub bound_nuclides: Vec<String>,
    pub source: String,
    /// ACER's `nmix`: the elastic part is divided by it as ACER divides every
    /// cross section of the table, so elastic and inelastic stay in the ratio
    /// the evaluation gives them.
    nmix: f64,
    laws: Vec<ThermalScattering>,
    temperatures: Vec<Temperature>,
}

/// One temperature's elastic part as stored: its kind, the Bragg edges and
/// cumulative factors (eV b), and SB and W' for the incoherent part.
type ElasticRow = (
    &'static str,
    Option<(Vec<f64>, Vec<f64>)>,
    Option<(f64, f64)>,
);

fn same(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
}

impl Prepared {
    /// Pair each law's temperatures with the ACE tables NJOY wrote for them,
    /// in order, and check the laws describe one table.
    pub fn new(
        name: &str,
        bound_nuclides: &[String],
        source: &str,
        nmix: u32,
        laws: Vec<ThermalScattering>,
        aces: Vec<Vec<ThermalAce>>,
    ) -> Result<Self, Box<dyn Error>> {
        let first = laws
            .first()
            .ok_or("a thermal table needs at least one evaluation")?;
        if first.inelastic.is_none() {
            return Err(format!(
                "{name}: the principal atom has an analytic law (no tabulated S(alpha, beta)), \
                 which this converter does not process"
            )
            .into());
        }
        for law in &laws[1..] {
            let kind = |l: &ThermalScattering| l.elastic.as_ref().map(|e| e.lthr());
            if law.mat != first.mat
                || !same(law.awr, first.awr)
                || !same(law.energy_max, first.energy_max)
                || !same(law.principal_count, first.principal_count)
                || kind(law) != kind(first)
            {
                return Err(format!(
                    "{name}: its evaluation files disagree on more than temperature \
                     (material, mass, energy_max, M0 or elastic kind), so they are not \
                     one table"
                )
                .into());
            }
        }
        if aces.len() != laws.len() {
            return Err("one ACE file per evaluation file is needed".into());
        }
        let mut temperatures = Vec::new();
        for (l, (law, tables)) in laws.iter().zip(aces).enumerate() {
            if tables.len() != law.temperatures.len() {
                return Err(format!(
                    "{name}: NJOY wrote {} tables for {} temperatures",
                    tables.len(),
                    law.temperatures.len()
                )
                .into());
            }
            for (index, (ace, &kelvin)) in tables.into_iter().zip(&law.temperatures).enumerate() {
                if (ace.temperature - kelvin).abs() > 1e-3 * kelvin {
                    return Err(format!(
                        "{name}: the ACE table at {} K was paired with the {kelvin} K law",
                        ace.temperature
                    )
                    .into());
                }
                temperatures.push(Temperature {
                    label: temperature_str(kelvin),
                    kelvin,
                    ace,
                    index,
                    law: l,
                });
            }
        }
        temperatures.sort_by(|a, b| a.kelvin.total_cmp(&b.kelvin));
        for pair in temperatures.windows(2) {
            if pair[0].label == pair[1].label {
                return Err(
                    format!("{name}: two temperatures share the label {}", pair[0].label).into(),
                );
            }
        }
        Ok(Prepared {
            name: name.to_string(),
            bound_nuclides: bound_nuclides.to_vec(),
            source: source.to_string(),
            nmix: f64::from(nmix.max(1)),
            laws,
            temperatures,
        })
    }

    /// Temperature labels, in order.
    pub fn labels(&self) -> Vec<String> {
        self.temperatures.iter().map(|t| t.label.clone()).collect()
    }

    /// Write every section into `dir`.
    pub fn write(&self, dir: &Path) -> Result<(), Box<dyn Error>> {
        std::fs::create_dir_all(dir)?;
        self.write_header(dir)?;
        self.write_inelastic(dir)?;
        if self.laws[0].elastic.is_some() {
            self.write_elastic(dir)?;
        }
        self.write_sab(dir)?;
        Ok(())
    }

    fn write_header(&self, dir: &Path) -> Result<(), Box<dyn Error>> {
        let law = &self.laws[0];
        let inelastic = law.inelastic.as_ref().expect("checked in new");
        let model = |m: SecondaryModel| match m {
            SecondaryModel::ShortCollisionTime => "short_collision_time",
            SecondaryModel::FreeGas => "free_gas",
            SecondaryModel::Diffusive => "diffusive",
        };
        let kind = law.elastic.as_ref().map(|e| {
            match e {
                ThermalElastic::Coherent(_) => "coherent",
                ThermalElastic::Incoherent(_) => "incoherent",
                ThermalElastic::Mixed { .. } => "mixed",
            }
            .to_string()
        });
        let kelvin: Vec<f64> = self.temperatures.iter().map(|t| t.kelvin).collect();
        let kts: Vec<f64> = self.temperatures.iter().map(|t| t.ace.kt).collect();
        write_section(
            &dir.join("thermal.arrow"),
            "thermal.arrow",
            vec![
                strings(std::slice::from_ref(&self.name)),
                opt_strings(std::slice::from_ref(&law.zsymam)),
                ints(&[law.mat]),
                string_list(&self.bound_nuclides),
                floats(&[law.awr]),
                floats(&[law.free_atom_xs]),
                floats(&[law.energy_max]),
                floats(&[law.epsilon]),
                floats(&[law.principal_count]),
                ints(&[law.lat as i32]),
                bools(&[law.symmetric]),
                ints(&[law.lln as i32]),
                string_list(
                    &law.secondary
                        .iter()
                        .map(|s| model(s.model).to_string())
                        .collect::<Vec<_>>(),
                ),
                float_list(
                    &law.secondary
                        .iter()
                        .map(|s| s.free_atom_xs)
                        .collect::<Vec<_>>(),
                ),
                float_list(&law.secondary.iter().map(|s| s.awr).collect::<Vec<_>>()),
                float_list(&law.secondary.iter().map(|s| s.count).collect::<Vec<_>>()),
                string_list(&self.labels()),
                float_list(&kelvin),
                float_list(&kts),
                opt_strings(&[kind]),
                int_list(&inelastic.alpha_breakpoints),
                int_list(&inelastic.alpha_interpolation),
                int_list(&inelastic.beta_breakpoints),
                int_list(&inelastic.beta_interpolation),
                strings(std::slice::from_ref(&self.source)),
            ],
        )
    }

    fn write_inelastic(&self, dir: &Path) -> Result<(), Box<dyn Error>> {
        let mut rows = Vec::with_capacity(self.temperatures.len());
        for t in &self.temperatures {
            let InelasticDistribution::Continuous(dist) = &t.ace.inelastic else {
                return Err(format!(
                    "{}: the {} table has equally probable outgoing energies; the \
                     continuous form (ACER iwt=2) is the one this format stores",
                    self.name, t.label
                )
                .into());
            };
            let n_mu = dist.first().map_or(0, |d| d.n_mu);
            if dist.iter().any(|d| d.n_mu != n_mu) {
                return Err(
                    format!("{}: {} has a varying cosine count", self.name, t.label).into(),
                );
            }
            rows.push(vec![
                strings(std::slice::from_ref(&t.label)),
                floats(&[t.ace.kt]),
                float_list(&t.ace.inelastic_energy),
                float_list(&t.ace.inelastic_xs),
                float_list_lists(&[dist.iter().map(|d| d.energy_out.clone()).collect()]),
                float_list_lists(&[dist.iter().map(|d| d.pdf.clone()).collect()]),
                float_list_lists(&[dist.iter().map(|d| d.cdf.clone()).collect()]),
                f32_list_list(dist.iter().map(|d| d.mu.as_slice())),
                ints(&[n_mu as i32]),
            ]);
        }
        write_section_per_row(&dir.join("inelastic.arrow"), "inelastic.arrow", rows)
    }

    /// The elastic part at temperature `t`, per bound atom as ACER writes it:
    /// SB taken per principal atom (THERMR's `natom`, which the deck sets to
    /// M0) and everything divided by `nmix`.
    ///
    /// Only for a table with an elastic part: `new` checks every law has one
    /// or none, and the callers skip a table with none.
    fn elastic_row(&self, t: &Temperature) -> Result<ElasticRow, String> {
        let law = &self.laws[t.law];
        let Some(elastic) = law.elastic.as_ref() else {
            return Err(format!("{}: the evaluation has no elastic part", self.name));
        };
        let coherent = |c: &endf::thermal_scattering::CoherentElastic| {
            c.temperatures
                .iter()
                .position(|&k| same(k, t.kelvin))
                .map(|i| {
                    let f = c.factors[i].iter().map(|f| f / self.nmix).collect();
                    (c.bragg_edges.clone(), f)
                })
                .ok_or_else(|| {
                    format!(
                        "{}: the coherent elastic data has no {} K, which the inelastic law has",
                        self.name, t.kelvin
                    )
                })
        };
        let incoherent = |i: &endf::thermal_scattering::IncoherentElastic| {
            let x = &i.debye_waller.x;
            if x.is_empty()
                || t.kelvin < x[0] * (1.0 - 1e-9)
                || t.kelvin > x[x.len() - 1] * (1.0 + 1e-9)
            {
                return Err(format!(
                    "{}: {} K is outside the Debye-Waller table",
                    self.name, t.kelvin
                ));
            }
            Ok((
                i.bound_xs / (law.principal_count * self.nmix),
                i.debye_waller.eval(t.kelvin),
            ))
        };
        Ok(match elastic {
            ThermalElastic::Coherent(c) => ("coherent", Some(coherent(c)?), None),
            ThermalElastic::Incoherent(i) => ("incoherent", None, Some(incoherent(i)?)),
            ThermalElastic::Mixed {
                coherent: c,
                incoherent: i,
            } => ("mixed", Some(coherent(c)?), Some(incoherent(i)?)),
        })
    }

    /// Check the elastic part taken from MF=7 MT=2 against the one NJOY wrote
    /// into the ACE table beside the inelastic part: the cumulative Bragg
    /// factor at every edge ACER kept, and the incoherent cross section at
    /// every energy of its grid. The closed forms are kept rather than ACER's
    /// equiprobable cosines, so this is what holds them to NJOY's conventions.
    ///
    /// `tolerance` is the relative difference allowed, and is the fractional
    /// tolerance the deck gave THERMR (`error`): NJOY builds the elastic part
    /// it writes only to that accuracy, so the closed form, which is exact,
    /// cannot be held closer to it. That is still far below any convention
    /// mix-up (a factor of `nmix` or of M0).
    pub fn check_against_njoy(&self, tolerance: f64) -> Result<(), Box<dyn Error>> {
        use endf::thermal_ace::AceElastic;
        let off = |what: &str, e: f64, ours: f64, njoy: f64| -> Result<(), String> {
            if (ours - njoy).abs() > tolerance * njoy.abs().max(ours.abs()) {
                return Err(format!(
                    "{}: {what} at {e} eV is {ours}, NJOY wrote {njoy}",
                    self.name
                ));
            }
            Ok(())
        };
        for t in &self.temperatures {
            let law = &self.laws[t.law];
            // Many evaluations have no MF=7 MT=2 (D in D2O, H in HF, Al in
            // Al2O3): nothing to compare, as long as NJOY made none up either.
            if law.elastic.is_none() {
                if t.ace.elastic.is_some() {
                    return Err(format!(
                        "{}: the evaluation has no elastic part, but NJOY wrote one at {} K",
                        self.name, t.kelvin
                    )
                    .into());
                }
                continue;
            }
            let emax = law.energy_max;
            let (_, bragg, inc) = self.elastic_row(t)?;
            let (ace_bragg, ace_inc) = match &t.ace.elastic {
                None if bragg.is_none() && inc.is_none() => continue,
                None => return Err(format!("{}: NJOY wrote no elastic part", self.name).into()),
                Some(AceElastic::Coherent {
                    bragg_edges,
                    factors,
                }) => (Some((bragg_edges, factors)), None),
                Some(AceElastic::Incoherent { energy, xs, .. }) => (None, Some((energy, xs))),
                Some(AceElastic::Mixed {
                    bragg_edges,
                    factors,
                    energy,
                    xs,
                    ..
                }) => (Some((bragg_edges, factors)), Some((energy, xs))),
            };
            if bragg.is_some() != ace_bragg.is_some() || inc.is_some() != ace_inc.is_some() {
                return Err(format!(
                    "{}: NJOY wrote a different kind of elastic part at {} K",
                    self.name, t.kelvin
                )
                .into());
            }
            if let (Some((edges, factors)), Some((ace_edges, ace_factors))) = (&bragg, ace_bragg) {
                for (&e, &f) in ace_edges
                    .iter()
                    .zip(ace_factors)
                    .filter(|&(&e, _)| e <= emax)
                {
                    let n = edges.partition_point(|&x| x <= e * (1.0 + 1e-6));
                    let ours = if n == 0 { 0.0 } else { factors[n - 1] };
                    off("the cumulative Bragg factor", e, ours, f)?;
                }
            }
            if let (Some((sb, w)), Some((energy, xs))) = (inc, ace_inc) {
                for (&e, &njoy) in energy.iter().zip(xs).filter(|&(&e, _)| e <= emax) {
                    let x = 2.0 * e * w;
                    let ours = if x == 0.0 {
                        sb
                    } else {
                        sb / 2.0 * -(-2.0 * x).exp_m1() / x
                    };
                    off("the incoherent elastic cross section", e, ours, njoy)?;
                }
            }
        }
        Ok(())
    }

    fn write_elastic(&self, dir: &Path) -> Result<(), Box<dyn Error>> {
        let mut rows = Vec::with_capacity(self.temperatures.len());
        for t in &self.temperatures {
            let (kind, bragg, inc) = self.elastic_row(t)?;
            let (edges, factors) = match bragg {
                Some((e, f)) => (vec![e], vec![f]),
                None => (vec![], vec![]),
            };
            rows.push(vec![
                strings(std::slice::from_ref(&t.label)),
                strings(&[kind.to_string()]),
                nullable_list(&edges),
                nullable_list(&factors),
                opt_floats(&[inc.map(|(b, _)| b)]),
                opt_floats(&[inc.map(|(_, w)| w)]),
            ]);
        }
        write_section_per_row(&dir.join("elastic.arrow"), "elastic.arrow", rows)
    }

    fn write_sab(&self, dir: &Path) -> Result<(), Box<dyn Error>> {
        let mut rows = Vec::with_capacity(self.temperatures.len());
        for t in &self.temperatures {
            let law = &self.laws[t.law];
            let inelastic = law.inelastic.as_ref().expect("checked in new");
            let teff = law
                .effective_temperature
                .first()
                .map_or(t.kelvin, |f| f.eval(t.kelvin));
            rows.push(vec![
                strings(std::slice::from_ref(&t.label)),
                floats(&[t.kelvin]),
                float_list(&inelastic.alpha),
                float_list(&inelastic.beta),
                float_list(&inelastic.s[t.index]),
                ints(&[inelastic.temperature_interpolation[t.index] as i32]),
                floats(&[teff]),
            ]);
        }
        write_section_per_row(&dir.join("sab.arrow"), "sab.arrow", rows)
    }
}

/// A one-row `list<double>` column, null when `rows` is empty.
fn nullable_list(rows: &[Vec<f64>]) -> ArrayRef {
    if rows.is_empty() {
        crate::sections::float_lists_or_null(&[vec![]])
    } else {
        float_lists(rows)
    }
}

/// A one-row `list<list<float32>>` column.
fn f32_list_list<'a>(inner: impl Iterator<Item = &'a [f64]>) -> ArrayRef {
    let mut b = ListBuilder::new(ListBuilder::new(Float32Builder::new()));
    for values in inner {
        for &v in values {
            b.values().values().append_value(v as f32);
        }
        b.values().append(true);
    }
    b.append(true);
    Arc::new(b.finish())
}

/// Write `version.json` with the temperature index of every per-temperature
/// section present. Written last: its presence is what says the folder is
/// finished.
pub fn write_version(dir: &Path, provenance: &Provenance) -> Result<(), Box<dyn Error>> {
    let mut ranges = serde_json::Map::new();
    for section in PER_TEMPERATURE_SECTIONS {
        let path = dir.join(section);
        if path.exists() {
            let index = crate::energy_ranges::index_temperature_batches(&path, section)?;
            ranges.insert(section.to_string(), index.to_json());
        }
    }
    let marker = serde_json::json!({
        "format_version": nuclear_data_schema::FORMAT_VERSION,
        "library": provenance.library,
        "data_version": provenance.data_version,
        "converter_version": concat!("yamc-convert ", env!("CARGO_PKG_VERSION")),
        "created_utc": provenance.created_utc,
        "particle": "thermal",
        "temperature_ranges": ranges,
    });
    let tmp = dir.join("version.json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(&marker)?)?;
    std::fs::rename(tmp, dir.join("version.json"))?;
    Ok(())
}

/// Convert one thermal table into `output_dir/<name>.arrow/`, running NJOY on
/// every evaluation file at all of its temperatures. Returns the folder.
pub fn convert_thermal(
    source: &ThermalSource,
    output_dir: &Path,
    provenance: &Provenance,
) -> Result<PathBuf, Box<dyn Error>> {
    let neutron = Material::from_file(source.neutron)
        .map_err(|e| format!("{}: {e}", source.neutron.display()))?;
    std::fs::create_dir_all(output_dir)?;
    let mut laws = Vec::new();
    let mut aces = Vec::new();
    let error = endf::njoy::ThermalAceOptions::default().error;
    for file in source.thermal {
        let material = Material::from_file(file).map_err(|e| format!("{}: {e}", file.display()))?;
        let law = ThermalScattering::from_endf(&material)?;
        // NJOY works in a directory of its own inside the output directory,
        // removed when `work` drops (see `entry::read` for why not /tmp).
        let work = endf::njoy::Scratch::new_in(output_dir, source.name)?;
        let options = endf::njoy::ThermalAceOptions {
            table_name: source.ace_name.to_string(),
            zaids: source.zaids.to_vec(),
            nmix: source.nmix,
            output_dir: work.path().to_path_buf(),
            njoy_exec: source.njoy_exec.to_string(),
            error,
            ..Default::default()
        };
        let ace = endf::njoy::make_ace_thermal(source.neutron, &neutron, file, &material, &options)
            .map_err(|e| format!("NJOY failed on {}: {e}", file.display()))?;
        let text = std::fs::read_to_string(&ace)?;
        let tables = endf::ace::tables_from_str(&text, None)?
            .iter()
            .map(ThermalAce::from_ace)
            .collect::<Result<Vec<_>, _>>()?;
        laws.push(law);
        aces.push(tables);
    }
    let names: Vec<String> = source
        .thermal
        .iter()
        .filter_map(|p| p.file_name().and_then(|n| n.to_str()).map(str::to_string))
        .collect();
    let prepared = Prepared::new(
        source.name,
        source.bound_nuclides,
        &names.join(","),
        source.nmix,
        laws,
        aces,
    )?;
    prepared.check_against_njoy(error)?;
    let dir = output_dir.join(format!("{}.arrow", source.name));
    if dir.exists() {
        std::fs::remove_dir_all(&dir)?;
    }
    prepared.write(&dir)?;
    write_version(&dir, provenance)?;
    Ok(dir)
}

/// What a thermal folder holds, as the library builder's gates and index
/// read it.
#[derive(Debug, Clone, PartialEq)]
pub struct ThermalSummary {
    pub name: String,
    pub bound_nuclides: Vec<String>,
    pub temperatures: Vec<String>,
    pub energy_max: f64,
    pub elastic_kind: Option<String>,
}

/// Read a thermal folder's header and check that every per-temperature
/// section in it decodes, with one row and one batch for each of the header's
/// temperatures, in its order. `inelastic.arrow` and `sab.arrow` must be
/// there, and `elastic.arrow` exactly when the header names an elastic kind.
pub fn inspect_thermal(dir: &Path) -> Result<ThermalSummary, Box<dyn Error>> {
    use arrow_array::cast::AsArray;
    use arrow_array::Array;
    use arrow_ipc::reader::FileReader;

    let read = |section: &str| -> Result<Vec<arrow_array::RecordBatch>, Box<dyn Error>> {
        let path = dir.join(section);
        let reader = FileReader::try_new(
            std::fs::File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?,
            None,
        )?;
        let batches = reader.collect::<Result<Vec<_>, _>>()?;
        for b in &batches {
            nuclear_data_schema::check_batch(section, &b.schema())?;
        }
        Ok(batches)
    };
    let header = read("thermal.arrow")?;
    let [header] = header.as_slice() else {
        return Err("thermal.arrow does not hold one batch".into());
    };
    let text = |name: &str| -> Result<String, Box<dyn Error>> {
        Ok(header
            .column_by_name(name)
            .ok_or_else(|| format!("thermal.arrow has no {name}"))?
            .as_string_opt::<i32>()
            .ok_or_else(|| format!("thermal.arrow {name} is not text"))?
            .value(0)
            .to_string())
    };
    let texts = |name: &str| -> Result<Vec<String>, Box<dyn Error>> {
        let list = header
            .column_by_name(name)
            .ok_or_else(|| format!("thermal.arrow has no {name}"))?
            .as_list_opt::<i32>()
            .ok_or_else(|| format!("thermal.arrow {name} is not a list"))?
            .value(0);
        let strings = list
            .as_string_opt::<i32>()
            .ok_or_else(|| format!("thermal.arrow {name} items are not text"))?;
        Ok((0..strings.len())
            .map(|i| strings.value(i).to_string())
            .collect())
    };
    let temperatures = texts("temperatures")?;
    if temperatures.is_empty() {
        return Err("thermal.arrow lists no temperatures".into());
    }
    let kind_column = header
        .column_by_name("elastic_kind")
        .ok_or("thermal.arrow has no elastic_kind")?;
    let elastic_kind = (!kind_column.is_null(0))
        .then(|| text("elastic_kind"))
        .transpose()?;
    let energy_max = header
        .column_by_name("energy_max")
        .ok_or("thermal.arrow has no energy_max")?
        .as_primitive_opt::<arrow_array::types::Float64Type>()
        .ok_or("energy_max is not a float")?
        .value(0);
    let mut sections = vec!["inelastic.arrow", "sab.arrow"];
    let elastic_present = dir.join("elastic.arrow").exists();
    if elastic_kind.is_some() != elastic_present {
        return Err(format!(
            "elastic kind {elastic_kind:?}, and elastic.arrow is {}",
            if elastic_present { "there" } else { "missing" }
        )
        .into());
    }
    if elastic_present {
        sections.push("elastic.arrow");
    }
    for section in sections {
        let batches = read(section)?;
        let labels: Vec<String> = batches
            .iter()
            .map(|b| {
                if b.num_rows() != 1 {
                    return Err(format!("{section} has a batch of {} rows", b.num_rows()));
                }
                Ok(b.column_by_name("temperature")
                    .and_then(|c| c.as_string_opt::<i32>().map(|s| s.value(0).to_string()))
                    .unwrap_or_default())
            })
            .collect::<Result<_, _>>()?;
        if labels != temperatures {
            return Err(format!(
                "{section} holds temperatures {labels:?}, thermal.arrow lists {temperatures:?}"
            )
            .into());
        }
    }
    Ok(ThermalSummary {
        name: text("name")?,
        bound_nuclides: texts("bound_nuclides")?,
        temperatures,
        energy_max,
        elastic_kind,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::cast::AsArray;
    use arrow_array::Array;
    use arrow_ipc::reader::FileReader;

    fn xz(bytes: &[u8]) -> String {
        let mut out = Vec::new();
        lzma_rs::xz_decompress(&mut std::io::BufReader::new(bytes), &mut out).unwrap();
        String::from_utf8(out).unwrap()
    }

    const SIO2: &[u8] = include_bytes!("../../endf/fixtures/tsl-SiO2-beta.endf.xz");
    const ZR_ZRH: &[u8] = include_bytes!("../../endf/fixtures/tsl-ZrinZrH.endf.xz");
    const SYNTHETIC: &[u8] = include_bytes!("../../endf/fixtures/synthetic-thermal.ace.xz");

    /// The synthetic continuous table, given each of `law`'s temperatures in
    /// turn, standing in for what NJOY would write.
    fn aces_for(law: &ThermalScattering) -> Vec<ThermalAce> {
        let tables = endf::ace::tables_from_str(&xz(SYNTHETIC), None).unwrap();
        law.temperatures
            .iter()
            .map(|&t| {
                let mut table = tables[0].clone();
                table.kt = endf::data::K_BOLTZMANN * t / 1e6;
                ThermalAce::from_ace(&table).unwrap()
            })
            .collect()
    }

    fn law(fixture: &[u8]) -> ThermalScattering {
        ThermalScattering::from_endf(&Material::from_str(&xz(fixture)).unwrap()).unwrap()
    }

    fn rows(path: &Path) -> Vec<arrow_array::RecordBatch> {
        FileReader::try_new(std::fs::File::open(path).unwrap(), None)
            .unwrap()
            .map(|b| b.unwrap())
            .collect()
    }

    fn write(fixture: &[u8]) -> (tempfile::TempDir, PathBuf, ThermalScattering) {
        let l = law(fixture);
        let aces = aces_for(&l);
        let prepared = Prepared::new(
            "c_Test",
            &["Si28".into()],
            "t.endf",
            1,
            vec![l.clone()],
            vec![aces],
        )
        .unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("c_Test.arrow");
        prepared.write(&dir).unwrap();
        write_version(
            &dir,
            &Provenance {
                library: "test".into(),
                data_version: "2026-11-07".into(),
                created_utc: "now".into(),
            },
        )
        .unwrap();
        (tmp, dir, l)
    }

    #[test]
    fn a_coherent_table_writes_every_section_one_batch_per_temperature() {
        let (_tmp, dir, l) = write(SIO2);
        let n = l.temperatures.len();
        for section in ["inelastic.arrow", "elastic.arrow", "sab.arrow"] {
            let batches = rows(&dir.join(section));
            assert_eq!(batches.len(), n, "{section}");
            assert!(batches.iter().all(|b| b.num_rows() == 1), "{section}");
        }
        let header = &rows(&dir.join("thermal.arrow"))[0];
        let labels = header
            .column_by_name("temperatures")
            .unwrap()
            .as_list::<i32>()
            .value(0);
        let labels = labels.as_string::<i32>();
        assert_eq!(labels.len(), n);
        assert_eq!(
            header
                .column_by_name("elastic_kind")
                .unwrap()
                .as_string::<i32>()
                .value(0),
            "coherent"
        );
        // The coherent factors are the evaluation's at that temperature.
        let elastic = &rows(&dir.join("elastic.arrow"))[1];
        let factors = elastic
            .column_by_name("factors")
            .unwrap()
            .as_list::<i32>()
            .value(0);
        let factors = factors.as_primitive::<arrow_array::types::Float64Type>();
        let ThermalElastic::Coherent(c) = l.elastic.as_ref().unwrap() else {
            panic!()
        };
        assert_eq!(factors.values().to_vec(), c.factors[1]);
        // The index names every temperature of every section.
        let version: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("version.json")).unwrap())
                .unwrap();
        for section in PER_TEMPERATURE_SECTIONS {
            let index = nuclear_data_schema::energy_ranges::EnergyRanges::from_json(
                &version["temperature_ranges"][section],
            )
            .unwrap();
            assert_eq!(index.temperatures.len(), n, "{section}");
        }
    }

    #[test]
    fn an_incoherent_table_stores_the_bound_xs_per_principal_atom() {
        let (_tmp, dir, l) = write(ZR_ZRH);
        let ThermalElastic::Incoherent(i) = l.elastic.as_ref().unwrap() else {
            panic!()
        };
        let elastic = &rows(&dir.join("elastic.arrow"))[0];
        let bound = elastic
            .column_by_name("bound_xs")
            .unwrap()
            .as_primitive::<arrow_array::types::Float64Type>()
            .value(0);
        assert_eq!(bound, i.bound_xs / l.principal_count);
        let w = elastic
            .column_by_name("debye_waller")
            .unwrap()
            .as_primitive::<arrow_array::types::Float64Type>()
            .value(0);
        assert_eq!(w, i.debye_waller.eval(l.temperatures[0]));
    }

    #[test]
    fn a_written_folder_inspects_back() {
        let (_tmp, dir, l) = write(SIO2);
        let s = inspect_thermal(&dir).unwrap();
        assert_eq!(s.name, "c_Test");
        assert_eq!(s.bound_nuclides, ["Si28"]);
        assert_eq!(s.temperatures.len(), l.temperatures.len());
        assert_eq!(s.elastic_kind.as_deref(), Some("coherent"));
        std::fs::remove_file(dir.join("elastic.arrow")).unwrap();
        assert!(
            inspect_thermal(&dir).is_err(),
            "a coherent table with no elastic.arrow"
        );
    }

    #[test]
    fn the_transport_reader_reads_what_was_written() {
        use yamc_nuclide::thermal_scattering::{read_thermal_from_arrow, Elastic};
        let (_tmp, dir, l) = write(SIO2);
        let table = read_thermal_from_arrow(&dir, None).unwrap();
        assert_eq!(table.temperatures.len(), l.temperatures.len());
        assert_eq!(table.energy_max, l.energy_max);
        let first = &table.available[0];
        let t = &table.temperatures[&first.0];
        let ace = &aces_for(&l)[0];
        assert_eq!(t.inelastic.xs, ace.inelastic_xs);
        assert_eq!(t.inelastic.energy, ace.inelastic_energy);
        assert_eq!(t.kelvin, l.temperatures[0]);
        let Some(Elastic::Coherent { factors, .. }) = &t.elastic else {
            panic!("coherent")
        };
        let ThermalElastic::Coherent(c) = l.elastic.as_ref().unwrap() else {
            panic!()
        };
        assert_eq!(factors, &c.factors[0]);
        // One temperature asked for, one read; one the table lacks, refused.
        let one = read_thermal_from_arrow(&dir, Some(std::slice::from_ref(&first.0))).unwrap();
        assert_eq!(one.temperatures.len(), 1);
        assert!(read_thermal_from_arrow(&dir, Some(&["1K".to_string()])).is_err());
    }

    #[test]
    fn mismatched_ace_and_law_temperatures_are_refused() {
        let l = law(ZR_ZRH);
        let mut aces = aces_for(&l);
        aces.pop();
        assert!(Prepared::new("c", &[], "s", 1, vec![l], vec![aces]).is_err());
    }

    #[test]
    fn two_files_that_are_not_one_table_are_refused() {
        let (a, b) = (law(SIO2), law(ZR_ZRH));
        let (aa, ab) = (aces_for(&a), aces_for(&b));
        assert!(Prepared::new("c", &[], "s", 1, vec![a, b], vec![aa, ab]).is_err());
    }

    const S_CH4: &[u8] = include_bytes!("../../endf/fixtures/tsl-s-CH4.endf.xz");

    /// `law` paired with ACE tables whose elastic part is what THERMR and
    /// ACER make of MT=2 with `natom` = M0 and `nmix` (thermr.f90 `iel`:
    /// `c1 = sb/(2 natom)`; aceth.f90 `thrlod`: every cross section `/nmix`;
    /// Bragg factors `e * sigma`, kept only where they grow).
    fn prepared_against_njoy(fixture: &[u8], njoy_nmix: f64, nmix: u32) -> Prepared {
        use endf::thermal_ace::AceElastic;
        let l = law(fixture);
        let mut aces = aces_for(&l);
        for (i, ace) in aces.iter_mut().enumerate() {
            let kelvin = l.temperatures[i];
            ace.elastic = Some(match l.elastic.as_ref().unwrap() {
                ThermalElastic::Coherent(c) => {
                    let (mut edges, mut factors) = (vec![], vec![]);
                    let mut last = 0.0;
                    for (&e, &f) in c.bragg_edges.iter().zip(&c.factors[i]) {
                        if f > last {
                            edges.push(e);
                            factors.push(f / njoy_nmix);
                            last = f;
                        }
                    }
                    AceElastic::Coherent {
                        bragg_edges: edges,
                        factors,
                    }
                }
                ThermalElastic::Incoherent(inc) => {
                    let energy: Vec<f64> = (0..40).map(|k| 1e-5 * 1.3f64.powi(k)).collect();
                    let sb = inc.bound_xs / l.principal_count / njoy_nmix;
                    let xs = energy
                        .iter()
                        .map(|&e| inc.xs(kelvin, e) * sb / inc.bound_xs)
                        .collect();
                    AceElastic::Incoherent {
                        energy,
                        xs,
                        mu: vec![],
                        n_mu: 0,
                    }
                }
                ThermalElastic::Mixed { .. } => unreachable!(),
            });
        }
        Prepared::new("c_Test", &[], "t.endf", nmix, vec![l], vec![aces]).unwrap()
    }

    #[test]
    fn the_elastic_part_agrees_with_what_njoy_writes() {
        // SiO2: coherent, three atom types (nmix = 3).
        prepared_against_njoy(SIO2, 3.0, 3)
            .check_against_njoy(1e-3)
            .unwrap();
        // Solid CH4: incoherent with four principal atoms (M0 = 4).
        assert_eq!(law(S_CH4).principal_count, 4.0);
        prepared_against_njoy(S_CH4, 1.0, 1)
            .check_against_njoy(1e-3)
            .unwrap();
        prepared_against_njoy(ZR_ZRH, 1.0, 1)
            .check_against_njoy(1e-3)
            .unwrap();
    }

    #[test]
    fn the_check_allows_what_the_deck_allows_thermr_and_no_more() {
        use endf::thermal_ace::AceElastic;
        // NJOY's Bragg factors off by `d`, relative, as a THERMR built to a
        // tolerance of 0.1% may write them.
        let off_by = |d: f64| {
            let mut p = prepared_against_njoy(SIO2, 3.0, 3);
            for t in &mut p.temperatures {
                if let Some(AceElastic::Coherent { factors, .. }) = &mut t.ace.elastic {
                    factors.iter_mut().for_each(|f| *f *= 1.0 + d);
                }
            }
            p
        };
        off_by(2.5e-4).check_against_njoy(1e-3).unwrap();
        off_by(9e-4).check_against_njoy(1e-3).unwrap();
        let err = off_by(2e-3).check_against_njoy(1e-3).unwrap_err();
        assert!(err.to_string().contains("Bragg"), "{err}");
        assert_eq!(endf::njoy::ThermalAceOptions::default().error, 1e-3);
    }

    #[test]
    fn a_table_without_an_elastic_part_is_checked_without_one() {
        // As D in D2O or H in HF: MF=7 MT=4 only.
        let mut l = law(S_CH4);
        l.elastic = None;
        let mut aces = aces_for(&l);
        for a in &mut aces {
            a.elastic = None;
        }
        let p = Prepared::new("c_Test", &[], "t.endf", 1, vec![l.clone()], vec![aces]).unwrap();
        p.check_against_njoy(1e-3).unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("c_Test.arrow");
        p.write(&dir).unwrap();
        assert!(!dir.join("elastic.arrow").exists());

        // An elastic part NJOY wrote that the evaluation does not have.
        let mut aces: Vec<ThermalAce> = aces_for(&l)
            .into_iter()
            .map(|mut a| {
                a.elastic = None;
                a
            })
            .collect();
        aces[0].elastic = prepared_against_njoy(S_CH4, 1.0, 1).temperatures[0]
            .ace
            .elastic
            .clone();
        let err = Prepared::new("c_Test", &[], "t.endf", 1, vec![l], vec![aces])
            .unwrap()
            .check_against_njoy(1e-3)
            .unwrap_err();
        assert!(err.to_string().contains("no elastic part"), "{err}");
    }

    #[test]
    fn an_elastic_part_off_from_njoy_by_nmix_is_refused() {
        let err = prepared_against_njoy(SIO2, 3.0, 1)
            .check_against_njoy(1e-3)
            .unwrap_err();
        assert!(err.to_string().contains("Bragg"), "{err}");
        let err = prepared_against_njoy(S_CH4, 2.0, 1)
            .check_against_njoy(1e-3)
            .unwrap_err();
        assert!(err.to_string().contains("incoherent"), "{err}");
    }

    #[test]
    fn the_stored_elastic_part_is_divided_by_nmix() {
        let p = prepared_against_njoy(SIO2, 3.0, 3);
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("c_Test.arrow");
        p.write(&dir).unwrap();
        let elastic = &rows(&dir.join("elastic.arrow"))[0];
        let factors = elastic
            .column_by_name("factors")
            .unwrap()
            .as_list::<i32>()
            .value(0);
        let factors = factors.as_primitive::<arrow_array::types::Float64Type>();
        let ThermalElastic::Coherent(c) = law(SIO2).elastic.unwrap() else {
            panic!()
        };
        let want: Vec<f64> = c.factors[0].iter().map(|f| f / 3.0).collect();
        assert_eq!(factors.values().to_vec(), want);
    }
}
