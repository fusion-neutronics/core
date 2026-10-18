//! Evaluated photonuclear data, kept separate from photoatomic transport data.

use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int32Type};
use arrow_array::{Array, RecordBatch};
use arrow_ipc::reader::FileReader;
use endf::Material;

use crate::entry::Provenance;
use crate::sections::*;

/// Convert one photonuclear ENDF evaluation without neutron processing.
///
/// Cross sections retain their own grids and ENDF interpolation. Every source
/// section is also preserved, including the outgoing-particle distributions.
/// These are evaluated data, not a neutron or photoatomic transport library.
pub fn convert_photonuclear(
    input: &Path,
    output: &Path,
    provenance: &Provenance,
) -> Result<PathBuf, Box<dyn Error>> {
    let material = Material::from_file(input)?;
    let metadata = material
        .mf1_mt451()
        .ok_or("photonuclear evaluation has no MF=1 MT=451")?;
    if metadata.nsub != 0 {
        return Err(format!(
            "photonuclear data requires NSUB=0, got NSUB={}",
            metadata.nsub
        )
        .into());
    }
    let atomic_number = metadata.za / 1000;
    let mass_number = metadata.za % 1000;
    let name = endf::gnds_name(
        atomic_number as u32,
        mass_number as u32,
        metadata.liso as u32,
    );
    let mut rows = Vec::new();
    for &(mf, mt) in material.section_text.keys() {
        if mf != 3 {
            continue;
        }
        let section = material
            .mf3(mt)
            .ok_or_else(|| format!("{name}: MF=3 MT={mt} could not be read"))?;
        let xs = &section.sigma;
        if xs.x.is_empty()
            || xs.x.len() != xs.y.len()
            || xs.x.iter().any(|value| !value.is_finite() || *value < 0.0)
            || xs.y.iter().any(|value| !value.is_finite() || *value < 0.0)
            || xs.x.windows(2).any(|pair| pair[1] < pair[0])
            || xs.breakpoints.len() != xs.interpolation.len()
            || xs.breakpoints.last().copied() != Some(xs.x.len() as i32)
            || xs.breakpoints.windows(2).any(|pair| pair[1] <= pair[0])
            || xs.breakpoints.iter().any(|value| *value < 1)
            || xs
                .interpolation
                .iter()
                .any(|value| !(1..=5).contains(value))
        {
            return Err(format!("{name}: invalid photonuclear cross section at MT={mt}").into());
        }
        rows.push(vec![
            strings(std::slice::from_ref(&name)),
            ints(&[atomic_number as i32]),
            ints(&[mass_number as i32]),
            ints(&[metadata.liso as i32]),
            floats(&[metadata.awr]),
            ints(&[mt]),
            floats(&[section.qm]),
            floats(&[section.qi]),
            ints(&[section.lr as i32]),
            float_list(&xs.x),
            float_list(&xs.y),
            int_list(&xs.breakpoints),
            int_list(&xs.interpolation),
        ]);
    }
    if rows.is_empty() {
        return Err(format!("{name}: photonuclear evaluation has no MF=3 cross sections").into());
    }
    let dir = output.join(format!("{name}.arrow"));
    std::fs::create_dir_all(&dir)?;
    let marker = dir.join("version.json");
    if marker.exists() {
        std::fs::remove_file(&marker)?;
    }
    write_section_per_row(&dir.join("photonuclear.arrow"), "photonuclear.arrow", rows)?;
    let sections = material
        .section_text
        .iter()
        .map(|(&(mf, mt), text)| {
            vec![
                ints(&[mf]),
                ints(&[mt]),
                strings(std::slice::from_ref(text)),
            ]
        })
        .collect();
    write_section_per_row(
        &dir.join("photonuclear_sections.arrow"),
        "photonuclear_sections.arrow",
        sections,
    )?;
    let version = serde_json::json!({
        "format_version": 2,
        "particle": "photonuclear",
        "scope": "evaluated",
        "converter_version": concat!("yamc-convert ", env!("CARGO_PKG_VERSION")),
        "library": provenance.library,
        "data_version": provenance.data_version,
        "created_utc": provenance.created_utc,
    });
    let staged = dir.join("version.json.tmp");
    std::fs::write(&staged, serde_json::to_string_pretty(&version)?)?;
    std::fs::rename(staged, marker)?;
    Ok(dir)
}

/// A photonuclear folder validated against its preserved evaluation.
#[derive(Debug)]
pub struct PhotonuclearSummary {
    pub name: String,
    pub atomic_number: i32,
    pub mass_number: i32,
    pub energy_points: BTreeMap<i32, usize>,
    pub sections: Vec<(i32, i32)>,
}

fn read_batches(dir: &Path, section: &str) -> Result<Vec<RecordBatch>, Box<dyn Error>> {
    let reader = FileReader::try_new(std::fs::File::open(dir.join(section))?, None)?;
    if reader.schema().as_ref() != &section_schema(section) {
        return Err(format!("{section}: schema does not match the photonuclear format").into());
    }
    let batches = reader.collect::<Result<Vec<_>, _>>()?;
    if batches.is_empty() {
        return Err(format!("{section}: no record batches").into());
    }
    for batch in &batches {
        if batch.num_rows() != 1
            || batch
                .columns()
                .iter()
                .any(|column| column.null_count() != 0)
        {
            return Err(format!("{section}: expected one non-null row per record batch").into());
        }
    }
    Ok(batches)
}

/// Read and validate evaluated photonuclear Arrow data, without transport.
pub fn inspect_photonuclear(dir: &Path) -> Result<PhotonuclearSummary, Box<dyn Error>> {
    let marker: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("version.json"))?)?;
    if marker["particle"] != "photonuclear"
        || marker["scope"] != "evaluated"
        || marker["format_version"] != 2
    {
        return Err("not a supported evaluated photonuclear folder".into());
    }
    let mut sections = BTreeMap::new();
    for batch in read_batches(dir, "photonuclear_sections.arrow")? {
        let mf = batch
            .column_by_name("mf")
            .unwrap()
            .as_primitive::<Int32Type>()
            .value(0);
        let mt = batch
            .column_by_name("mt")
            .unwrap()
            .as_primitive::<Int32Type>()
            .value(0);
        let text = batch
            .column_by_name("text")
            .unwrap()
            .as_string::<i32>()
            .value(0)
            .to_string();
        if sections.insert((mf, mt), text).is_some() {
            return Err(format!("duplicate preserved section MF={mf} MT={mt}").into());
        }
    }
    let header = sections
        .get(&(1, 451))
        .ok_or("missing preserved MF=1 MT=451")?;
    let metadata = endf::mf::mf1::parse_mf1_mt451(&mut endf::Reader::new(header))?;
    if metadata.nsub != 0 {
        return Err("preserved evaluation is not photonuclear (NSUB=0)".into());
    }
    let name = endf::gnds_name(
        (metadata.za / 1000) as u32,
        (metadata.za % 1000) as u32,
        metadata.liso as u32,
    );
    let mut energy_points = BTreeMap::new();
    for batch in read_batches(dir, "photonuclear.arrow")? {
        let integer = |column: &str| {
            batch
                .column_by_name(column)
                .unwrap()
                .as_primitive::<Int32Type>()
                .value(0)
        };
        let scalar = |column: &str| {
            batch
                .column_by_name(column)
                .unwrap()
                .as_primitive::<Float64Type>()
                .value(0)
        };
        let mt = integer("mt");
        if batch
            .column_by_name("name")
            .unwrap()
            .as_string::<i32>()
            .value(0)
            != name
            || integer("atomic_number") != (metadata.za / 1000) as i32
            || integer("mass_number") != (metadata.za % 1000) as i32
            || integer("metastable") != metadata.liso as i32
            || scalar("atomic_weight_ratio") != metadata.awr
        {
            return Err(format!("MT={mt}: identity differs from the preserved evaluation").into());
        }
        let text = sections
            .get(&(3, mt))
            .ok_or_else(|| format!("MT={mt}: missing preserved MF=3 section"))?;
        let source = endf::mf::mf3::parse_mf3(&mut endf::Reader::new(text))?;
        if scalar("Q_mass_difference") != source.qm
            || scalar("Q_value") != source.qi
            || integer("lr") != source.lr as i32
        {
            return Err(format!(
                "MT={mt}: reaction metadata differs from the preserved evaluation"
            )
            .into());
        }
        for (column, expected) in [("energy", &source.sigma.x), ("xs", &source.sigma.y)] {
            let values = batch
                .column_by_name(column)
                .unwrap()
                .as_list::<i32>()
                .value(0);
            if values.null_count() != 0
                || values.as_primitive::<Float64Type>().values().as_ref() != expected
            {
                return Err(
                    format!("MT={mt}: {column} differs from the preserved evaluation").into(),
                );
            }
        }
        for (column, expected) in [
            ("breakpoints", &source.sigma.breakpoints),
            ("interpolation", &source.sigma.interpolation),
        ] {
            let values = batch
                .column_by_name(column)
                .unwrap()
                .as_list::<i32>()
                .value(0);
            if values.null_count() != 0
                || values.as_primitive::<Int32Type>().values().as_ref() != expected
            {
                return Err(
                    format!("MT={mt}: {column} differs from the preserved evaluation").into(),
                );
            }
        }
        if energy_points.insert(mt, source.sigma.x.len()).is_some() {
            return Err(format!("duplicate cross section MT={mt}").into());
        }
    }
    let source_mts: Vec<i32> = sections
        .keys()
        .filter_map(|&(mf, mt)| (mf == 3).then_some(mt))
        .collect();
    if source_mts != energy_points.keys().copied().collect::<Vec<_>>() {
        return Err("cross sections do not cover the preserved MF=3 sections".into());
    }
    Ok(PhotonuclearSummary {
        name,
        atomic_number: (metadata.za / 1000) as i32,
        mass_number: (metadata.za % 1000) as i32,
        energy_points,
        sections: sections.into_keys().collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use endf::material::Section;

    fn evaluation(nsub: i64) -> tempfile::NamedTempFile {
        let compressed = include_bytes!("../../endf/fixtures/n-003_Li_006_trimmed.endf.xz");
        let mut decoded = Vec::new();
        lzma_rs::xz_decompress(&mut &compressed[..], &mut decoded).unwrap();
        let text = String::from_utf8(decoded).unwrap();
        let mut lines: Vec<String> = text.lines().map(str::to_owned).collect();
        let header = lines
            .iter()
            .position(|line| line.len() >= 75 && &line[70..75] == " 1451")
            .unwrap();
        lines[header + 2].replace_range(44..55, &format!("{nsub:11}"));
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), lines.join("\n")).unwrap();
        file
    }

    #[test]
    fn rejects_neutron_data() {
        let input = evaluation(10);
        let output = tempfile::tempdir().unwrap();
        let error =
            convert_photonuclear(input.path(), output.path(), &Provenance::default()).unwrap_err();
        assert!(error.to_string().contains("requires NSUB=0"));
        assert_eq!(std::fs::read_dir(output.path()).unwrap().count(), 0);
    }

    #[test]
    fn preserves_cross_sections_and_source_sections() {
        use arrow_array::cast::AsArray;
        use arrow_array::types::{Float64Type, Int32Type};
        use arrow_ipc::reader::FileReader;

        let input = evaluation(0);
        let material = Material::from_file(input.path()).unwrap();
        let output = tempfile::tempdir().unwrap();
        let dir =
            convert_photonuclear(input.path(), output.path(), &Provenance::default()).unwrap();
        let reader = FileReader::try_new(
            std::fs::File::open(dir.join("photonuclear.arrow")).unwrap(),
            None,
        )
        .unwrap();
        let mut count = 0;
        for batch in reader {
            let batch = batch.unwrap();
            let mt = batch
                .column_by_name("mt")
                .unwrap()
                .as_primitive::<Int32Type>()
                .value(0);
            let section = material.mf3(mt).unwrap();
            for (column, expected) in [("energy", &section.sigma.x), ("xs", &section.sigma.y)] {
                let values = batch
                    .column_by_name(column)
                    .unwrap()
                    .as_list::<i32>()
                    .value(0);
                assert_eq!(
                    values.as_primitive::<Float64Type>().values().as_ref(),
                    expected
                );
            }
            for (column, expected) in [
                ("breakpoints", &section.sigma.breakpoints),
                ("interpolation", &section.sigma.interpolation),
            ] {
                let values = batch
                    .column_by_name(column)
                    .unwrap()
                    .as_list::<i32>()
                    .value(0);
                assert_eq!(
                    values.as_primitive::<Int32Type>().values().as_ref(),
                    expected
                );
            }
            count += 1;
        }
        assert_eq!(
            count,
            material
                .section_data
                .values()
                .filter(|section| matches!(section, Section::Mf3(_)))
                .count()
        );
        let reader = FileReader::try_new(
            std::fs::File::open(dir.join("photonuclear_sections.arrow")).unwrap(),
            None,
        )
        .unwrap();
        let mut count = 0;
        for batch in reader {
            let batch = batch.unwrap();
            let mf = batch
                .column_by_name("mf")
                .unwrap()
                .as_primitive::<Int32Type>()
                .value(0);
            let mt = batch
                .column_by_name("mt")
                .unwrap()
                .as_primitive::<Int32Type>()
                .value(0);
            let text = batch
                .column_by_name("text")
                .unwrap()
                .as_string::<i32>()
                .value(0);
            assert_eq!(text, material.section_text[&(mf, mt)]);
            count += 1;
        }
        assert_eq!(count, material.section_text.len());
        let marker: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("version.json")).unwrap())
                .unwrap();
        assert_eq!(marker["particle"], "photonuclear");
        assert_eq!(marker["scope"], "evaluated");
        let summary = inspect_photonuclear(&dir).unwrap();
        assert_eq!(summary.name, "Li6");
        assert_eq!(
            summary.energy_points.len(),
            material
                .section_data
                .values()
                .filter(|section| matches!(section, Section::Mf3(_)))
                .count()
        );
        assert_eq!(summary.sections.len(), material.section_text.len());
    }
}
