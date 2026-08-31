//! Render the schema declarations as the JSON manifest the converter reads.
//!
//! The rendering lives here rather than in the library so the library stays
//! declarations plus a lookup, and so serde_json is a dependency of this binary
//! rather than of every crate that wants a schema. Regenerate with:
//!
//! ```text
//! cargo run -p nuclear-data-schema --bin emit-schema-manifest \
//!     > packages/nuclear_data_to_arrow/src/nuclear_data_to_arrow/schema/manifest.json
//! ```

use arrow_schema::DataType;
use nuclear_data_schema::all_sections;
use serde_json::{json, Map, Value};

/// Arrow type as a stable string, matched by the Python side's reader.
///
/// Deliberately not the Debug spelling, which is not part of arrow-rs's API and
/// would silently rewrite the manifest on a version bump.
pub fn render_type(dt: &DataType) -> String {
    match dt {
        DataType::Utf8 => "string".to_string(),
        DataType::Float64 => "double".to_string(),
        DataType::Int32 => "int32".to_string(),
        DataType::Boolean => "bool".to_string(),
        DataType::List(field) => format!("list<{}>", render_type(field.data_type())),
        other => panic!("unmapped Arrow type in a schema: {other:?}"),
    }
}

/// The declarations as the JSON document committed alongside the converter.
pub fn render_manifest() -> String {
    let mut sections = Map::new();
    for (path, schema) in all_sections() {
        let fields: Vec<Value> = schema
            .fields()
            .iter()
            .map(|f| {
                json!({
                    "name": f.name(),
                    "nullable": f.is_nullable(),
                    "type": render_type(f.data_type()),
                })
            })
            .collect();
        let metadata: Map<String, Value> = schema
            .metadata()
            .iter()
            .map(|(k, v)| (k.clone(), Value::String(v.clone())))
            .collect();
        sections.insert(
            path.to_string(),
            json!({"fields": fields, "metadata": metadata}),
        );
    }
    // One section per line rather than fully pretty-printed. Nobody hand-edits
    // this file, since a test regenerates and diffs it, and at indent=2 the same
    // 201 fields ran to 1123 lines: a second copy of the declarations by volume.
    // Per line keeps it greppable by section and keeps diffs meaningful at
    // section granularity.
    let body: Vec<String> = sections
        .iter()
        .map(|(path, value)| {
            format!(
                "    {}: {}",
                serde_json::to_string(path).expect("serialises"),
                serde_json::to_string(value).expect("serialises")
            )
        })
        .collect();
    format!(
        "{{\n  \"format_version\": 1,\n  \"sections\": {{\n{}\n  }}\n}}\n",
        body.join(",\n")
    )
}

fn main() {
    print!("{}", render_manifest());
}
