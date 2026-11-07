//! A thermal scattering table, as transport reads it: `thermal/<name>.arrow/`.
//!
//! One table is the scattering law of one nuclide bound in one material
//! (`c_H_in_H2O`) at each of its temperatures. Its inelastic part is NJOY's
//! cross section and continuous outgoing distribution, its elastic part the
//! evaluation's closed form. The format is declared in `nuclear-data-schema`
//! (`thermal`, `thermal_inelastic`, `thermal_elastic`).
//!
//! An absent optional section is fine; a present one that does not read is a
//! hard error, never a table quietly missing its elastic part.

use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type};
use arrow_array::{Array, RecordBatch};

use crate::arrow_helpers::{
    get_f64, get_f64_list, get_i32, get_str, get_str_list, read_arrow_file, try_get_f64,
};

/// The outgoing distribution for one incident energy.
#[derive(Debug, Clone, PartialEq)]
pub struct OutgoingDistribution {
    /// Outgoing energies in eV, starting at 0.
    pub energy_out: Vec<f64>,
    /// pdf in 1/eV and cdf, as NJOY wrote them (not normalised).
    pub pdf: Vec<f64>,
    pub cdf: Vec<f64>,
    /// `n_mu` equiprobable cosines per outgoing energy, sorted:
    /// `mu[j * n_mu + k]`.
    pub mu: Vec<f32>,
}

/// The incoherent inelastic part at one temperature.
#[derive(Debug, Clone, PartialEq)]
pub struct Inelastic {
    /// Incident energies in eV.
    pub energy: Vec<f64>,
    /// Cross section in barns per principal atom.
    pub xs: Vec<f64>,
    pub distributions: Vec<OutgoingDistribution>,
    pub n_mu: usize,
}

/// The elastic part at one temperature.
#[derive(Debug, Clone, PartialEq)]
pub enum Elastic {
    /// Bragg edges (eV) and the cumulative factor at each (eV-barns).
    Coherent {
        bragg_edges: Vec<f64>,
        factors: Vec<f64>,
    },
    /// The bound cross section per principal atom (barns) and the
    /// Debye-Waller integral W' (1/eV).
    Incoherent { bound_xs: f64, debye_waller: f64 },
    Mixed {
        bragg_edges: Vec<f64>,
        factors: Vec<f64>,
        bound_xs: f64,
        debye_waller: f64,
    },
}

impl Elastic {
    /// The elastic cross section in barns at `energy` (eV).
    pub fn xs(&self, energy: f64) -> f64 {
        let coherent = |edges: &[f64], factors: &[f64]| {
            let n = edges.partition_point(|&e| e <= energy);
            if n == 0 {
                0.0
            } else {
                factors[n - 1] / energy
            }
        };
        let incoherent = |b: f64, w: f64| {
            let x = 2.0 * energy * w;
            if x == 0.0 {
                b
            } else {
                b / 2.0 * -(-2.0 * x).exp_m1() / x
            }
        };
        match self {
            Elastic::Coherent {
                bragg_edges,
                factors,
            } => coherent(bragg_edges, factors),
            Elastic::Incoherent {
                bound_xs,
                debye_waller,
            } => incoherent(*bound_xs, *debye_waller),
            Elastic::Mixed {
                bragg_edges,
                factors,
                bound_xs,
                debye_waller,
            } => coherent(bragg_edges, factors) + incoherent(*bound_xs, *debye_waller),
        }
    }
}

/// One temperature of a table.
#[derive(Debug, Clone, PartialEq)]
pub struct ThermalTemperature {
    /// The temperature in Kelvin, and kT in eV.
    pub kelvin: f64,
    pub kt: f64,
    pub inelastic: Inelastic,
    pub elastic: Option<Elastic>,
}

/// A thermal scattering table, with the temperatures that were read.
#[derive(Debug, Clone, PartialEq)]
pub struct ThermalTable {
    pub name: String,
    /// The nuclides the table applies to.
    pub bound_nuclides: Vec<String>,
    /// The principal atom's mass in neutron masses.
    pub atomic_weight_ratio: f64,
    /// Above this energy (eV) the table does not apply.
    pub energy_max: f64,
    /// Every temperature the table publishes, labels and Kelvin, in order.
    pub available: Vec<(String, f64)>,
    /// The temperatures read, by label.
    pub temperatures: BTreeMap<String, ThermalTemperature>,
}

/// A section's path: the whole file, or the ranged copy a temperature-ranged
/// download keeps under `temperatures/`.
fn section_path(dir: &Path, section: &str) -> Option<PathBuf> {
    let whole = dir.join(section);
    if crate::storage::exists(&whole) {
        return Some(whole);
    }
    let ranged = dir.join("temperatures").join(section);
    crate::storage::exists(&ranged).then_some(ranged)
}

fn row_of(batch: &RecordBatch, label: &str) -> Result<Option<usize>, Box<dyn Error>> {
    for row in 0..batch.num_rows() {
        if get_str(batch, "temperature", row)? == label {
            return Ok(Some(row));
        }
    }
    Ok(None)
}

fn nested_f64(batch: &RecordBatch, col: &str, row: usize) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    let outer = batch
        .column_by_name(col)
        .ok_or_else(|| format!("missing column {col}"))?
        .as_list_opt::<i32>()
        .ok_or_else(|| format!("{col} is not a list"))?
        .value(row);
    let inner = outer
        .as_list_opt::<i32>()
        .ok_or_else(|| format!("{col} items are not lists"))?;
    Ok((0..inner.len())
        .map(|i| {
            inner
                .value(i)
                .as_primitive::<Float64Type>()
                .values()
                .to_vec()
        })
        .collect())
}

fn nested_f32(batch: &RecordBatch, col: &str, row: usize) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let outer = batch
        .column_by_name(col)
        .ok_or_else(|| format!("missing column {col}"))?
        .as_list_opt::<i32>()
        .ok_or_else(|| format!("{col} is not a list"))?
        .value(row);
    let inner = outer
        .as_list_opt::<i32>()
        .ok_or_else(|| format!("{col} items are not lists"))?;
    Ok((0..inner.len())
        .map(|i| {
            inner
                .value(i)
                .as_primitive::<Float32Type>()
                .values()
                .to_vec()
        })
        .collect())
}

/// Read a thermal table from `dir`, at the temperature labels `wanted`
/// (`"294K"`), or every temperature present when `None`. A wanted label the
/// table does not publish, or whose rows are not in the folder (a ranged
/// download that did not fetch it), is an error.
pub fn read_thermal_from_arrow(
    dir: &Path,
    wanted: Option<&[String]>,
) -> Result<ThermalTable, Box<dyn Error>> {
    let at = |e: Box<dyn Error>| -> Box<dyn Error> { format!("{}: {e}", dir.display()).into() };
    let header = read_arrow_file(&dir.join("thermal.arrow")).map_err(at)?;
    let labels = get_str_list(&header, "temperatures", 0).map_err(at)?;
    let kelvin = get_f64_list(&header, "temperatures_k", 0).map_err(at)?;
    if labels.len() != kelvin.len() || labels.is_empty() {
        return Err(at(
            "thermal.arrow lists its temperatures inconsistently".into()
        ));
    }
    let available: Vec<(String, f64)> = labels.into_iter().zip(kelvin).collect();
    let elastic_kind = {
        let col = header
            .column_by_name("elastic_kind")
            .ok_or_else(|| at("thermal.arrow has no elastic_kind".into()))?;
        (!col.is_null(0))
            .then(|| get_str(&header, "elastic_kind", 0))
            .transpose()
            .map_err(at)?
    };

    let inelastic_path =
        section_path(dir, "inelastic.arrow").ok_or_else(|| at("no inelastic.arrow".into()))?;
    let inelastic = read_arrow_file(&inelastic_path).map_err(at)?;
    let elastic = match (&elastic_kind, section_path(dir, "elastic.arrow")) {
        (Some(_), Some(p)) => Some(read_arrow_file(&p).map_err(at)?),
        (Some(kind), None) => {
            return Err(at(
                format!("a {kind} elastic table with no elastic.arrow").into()
            ))
        }
        (None, _) => None,
    };

    let wanted: Vec<String> = match wanted {
        Some(w) => w.to_vec(),
        None => (0..inelastic.num_rows())
            .map(|r| get_str(&inelastic, "temperature", r))
            .collect::<Result<_, _>>()
            .map_err(at)?,
    };
    let mut temperatures = BTreeMap::new();
    for label in wanted {
        let kelvin = available
            .iter()
            .find(|(l, _)| *l == label)
            .map(|(_, k)| *k)
            .ok_or_else(|| {
                let list: Vec<&str> = available.iter().map(|(l, _)| l.as_str()).collect();
                at(format!("no {label} table; the table has {}", list.join(", ")).into())
            })?;
        let row = row_of(&inelastic, &label)
            .map_err(at)?
            .ok_or_else(|| at(format!("inelastic.arrow holds no {label} rows").into()))?;
        let n_mu = usize::try_from(get_i32(&inelastic, "n_mu", row).map_err(at)?)?;
        let e_out = nested_f64(&inelastic, "energy_out", row).map_err(at)?;
        let pdf = nested_f64(&inelastic, "pdf", row).map_err(at)?;
        let cdf = nested_f64(&inelastic, "cdf", row).map_err(at)?;
        let mu = nested_f32(&inelastic, "mu", row).map_err(at)?;
        let energy = get_f64_list(&inelastic, "energy", row).map_err(at)?;
        let xs = get_f64_list(&inelastic, "xs", row).map_err(at)?;
        if [e_out.len(), pdf.len(), cdf.len(), mu.len(), xs.len()]
            .iter()
            .any(|&n| n != energy.len())
        {
            return Err(at(format!(
                "{label}: inelastic columns of different lengths"
            )
            .into()));
        }
        let distributions = e_out
            .into_iter()
            .zip(pdf)
            .zip(cdf)
            .zip(mu)
            .map(|(((energy_out, pdf), cdf), mu)| {
                if pdf.len() != energy_out.len()
                    || cdf.len() != energy_out.len()
                    || mu.len() != energy_out.len() * n_mu
                {
                    return Err(at(format!(
                        "{label}: an outgoing distribution of inconsistent length"
                    )
                    .into()));
                }
                Ok(OutgoingDistribution {
                    energy_out,
                    pdf,
                    cdf,
                    mu,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let elastic = match (&elastic, &elastic_kind) {
            (Some(batch), Some(kind)) => {
                let row = row_of(batch, &label)
                    .map_err(at)?
                    .ok_or_else(|| at(format!("elastic.arrow holds no {label} rows").into()))?;
                let edges = || get_f64_list(batch, "bragg_edges", row).map_err(at);
                let factors = || get_f64_list(batch, "factors", row).map_err(at);
                let bound = || {
                    try_get_f64(batch, "bound_xs", row)
                        .ok_or_else(|| at(format!("{label}: no bound_xs").into()))
                };
                let w = || {
                    try_get_f64(batch, "debye_waller", row)
                        .ok_or_else(|| at(format!("{label}: no debye_waller").into()))
                };
                Some(match kind.as_str() {
                    "coherent" => Elastic::Coherent {
                        bragg_edges: edges()?,
                        factors: factors()?,
                    },
                    "incoherent" => Elastic::Incoherent {
                        bound_xs: bound()?,
                        debye_waller: w()?,
                    },
                    "mixed" => Elastic::Mixed {
                        bragg_edges: edges()?,
                        factors: factors()?,
                        bound_xs: bound()?,
                        debye_waller: w()?,
                    },
                    other => return Err(at(format!("unknown elastic kind {other:?}").into())),
                })
            }
            _ => None,
        };
        temperatures.insert(
            label.clone(),
            ThermalTemperature {
                kelvin,
                kt: get_f64(&inelastic, "kT", row).map_err(at)?,
                inelastic: Inelastic {
                    energy,
                    xs,
                    distributions,
                    n_mu,
                },
                elastic,
            },
        );
    }
    Ok(ThermalTable {
        name: get_str(&header, "name", 0).map_err(at)?,
        bound_nuclides: get_str_list(&header, "bound_nuclides", 0).map_err(at)?,
        atomic_weight_ratio: get_f64(&header, "atomic_weight_ratio", 0).map_err(at)?,
        energy_max: get_f64(&header, "energy_max", 0).map_err(at)?,
        available,
        temperatures,
    })
}

/// The table temperatures a material at `kelvin` uses, by label: the one it
/// sits on, or the two it falls between (the material blends them). Outside
/// the table's range is an error naming the range: a table is never
/// extrapolated.
pub fn labels_for_temperature(
    table_name: &str,
    available: &[(String, f64)],
    kelvin: f64,
) -> Result<Vec<String>, String> {
    let labels: Vec<String> = available.iter().map(|(l, _)| l.clone()).collect();
    // Bracket on the true temperatures, not the rounded labels: graphite's
    // "296K" is 296 K but water's "294K" is 293.6 K.
    let mut sorted: Vec<&(String, f64)> = available.iter().collect();
    sorted.sort_by(|a, b| a.1.total_cmp(&b.1));
    let (lo, hi) = (sorted[0].1, sorted[sorted.len() - 1].1);
    if let Some((l, _)) = sorted
        .iter()
        .find(|(_, k)| (k - kelvin).abs() <= crate::temperature::SNAP_TOLERANCE_K.max(0.5))
    {
        return Ok(vec![l.clone()]);
    }
    if kelvin < lo || kelvin > hi {
        return Err(format!(
            "{table_name} covers {lo} K to {hi} K ({}), and the material is at {kelvin} K; \
             a thermal table is not extrapolated",
            labels.join(", ")
        ));
    }
    let i = sorted.partition_point(|(_, k)| *k < kelvin);
    Ok(vec![sorted[i - 1].0.clone(), sorted[i].0.clone()])
}

/// Load thermal table `name` from `source` (a library keyword, a release
/// folder or a table folder) for a material at `kelvin`, reading only the
/// table temperatures the material uses (one, or the two it falls between).
///
/// On a keyword, only those temperatures are downloaded: the per-temperature
/// sections are fetched as byte ranges. A keyword whose library publishes no
/// thermal tables is refused, naming the libraries that do; tables are never
/// taken from another library.
pub fn load_thermal(source: &str, name: &str, kelvin: f64) -> Result<ThermalTable, Box<dyn Error>> {
    use crate::url_cache::{self, DataKind};
    if url_cache::is_keyword(source) && !url_cache::THERMAL_LIBRARIES.contains(&source) {
        return Err(url_cache::no_thermal_tables(source).into());
    }
    // Ask for the temperature as the labels spell it (whole kelvin), so a
    // material at 293.6 K fetches the "294K" table alone.
    let request = format!("{}", kelvin.round());
    let scope =
        crate::LoadScope::full().with_temperatures(Some(std::iter::once(request).collect()));
    let dir = url_cache::resolve_data_path(source, Some(name), DataKind::Thermal, &scope)?;
    let dir = Path::new(&dir);
    let header = read_arrow_file(&dir.join("thermal.arrow"))
        .map_err(|e| format!("thermal table {name}: {e}"))?;
    let labels = get_str_list(&header, "temperatures", 0)?;
    let kelvins = get_f64_list(&header, "temperatures_k", 0)?;
    let available: Vec<(String, f64)> = labels.into_iter().zip(kelvins).collect();
    let wanted = labels_for_temperature(name, &available, kelvin)?;
    read_thermal_from_arrow(dir, Some(&wanted))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ladder() -> Vec<(String, f64)> {
        vec![
            ("294K".into(), 293.6),
            ("400K".into(), 400.0),
            ("600K".into(), 600.0),
        ]
    }

    #[test]
    fn a_temperature_on_a_table_uses_it_and_between_two_blends_them() {
        assert_eq!(
            labels_for_temperature("t", &ladder(), 293.6).unwrap(),
            ["294K"]
        );
        // Within half a kelvin of a table's true temperature is that table:
        // the labels are rounded, and 294 K asks for water's 293.6 K table.
        assert_eq!(
            labels_for_temperature("t", &ladder(), 294.0).unwrap(),
            ["294K"]
        );
        assert_eq!(
            labels_for_temperature("t", &ladder(), 450.0).unwrap(),
            ["400K", "600K"]
        );
    }

    #[test]
    fn outside_the_table_is_refused_with_its_range() {
        let e = labels_for_temperature("c_Graphite", &ladder(), 250.0).unwrap_err();
        assert!(
            e.contains("293.6 K to 600 K") && e.contains("c_Graphite"),
            "{e}"
        );
        assert!(labels_for_temperature("t", &ladder(), 601.0).is_err());
    }

    #[test]
    fn elastic_cross_sections_are_the_closed_forms() {
        let coherent = Elastic::Coherent {
            bragg_edges: vec![1e-3, 2e-3],
            factors: vec![1e-3, 3e-3],
        };
        assert_eq!(coherent.xs(5e-4), 0.0);
        assert_eq!(coherent.xs(1.5e-3), 1e-3 / 1.5e-3);
        let incoherent = Elastic::Incoherent {
            bound_xs: 80.0,
            debye_waller: 10.0,
        };
        assert!((incoherent.xs(1e-12) - 80.0).abs() < 1e-6);
        let e = 1.0;
        assert!((incoherent.xs(e) - 40.0 / 20.0 * (1.0 - (-40.0f64).exp())).abs() < 1e-12);
    }
}
