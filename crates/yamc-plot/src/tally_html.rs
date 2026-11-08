//! Pure-Rust core of the interactive mesh-tally viewer (`tally.plot()`).
//!
//! Sister module to [`crate::viewer_html`] -- different surface but same
//! shape: builds a self-contained HTML doc with an inline JS viewer.
//! Shares `escape_js`, `parse_hex_color`, and the surface table
//! (`build_surface_table`) so tally-overlay outlines get the same
//! hover-tooltip surface lookup as `model.plot()`.
//!
//! Pyo3 wrapper `PyInteractiveTallyPlot` stays in
//! `yamc-python::interactive_tally_viewer` and calls into this module.

use std::collections::HashMap;

use crate::viewer_html::{escape_js, parse_hex_color, PresampledGrid};

#[cfg(feature = "mesh")]
static YAMT_WASM_BYTES: &[u8] =
    include_bytes!("../../../packages/yamc-core/python/yamc/_wasm/yamt_bg.wasm");
#[cfg(feature = "mesh")]
static YAMT_WASM_JS: &str = include_str!("../../../packages/yamc-core/python/yamc/_wasm/yamt.js");

/// Mesh-tally viewer JS body -- helpers + DOM bindings + init logic.
/// Lives in `tally_viewer.js` so it can be edited as real JavaScript;
/// inlined into the HTML built by [`build_interactive_tally_html`]
/// after the per-tally data globals (SLICES, MESH_LL, INITIAL_*, …)
/// are declared.
///
/// Mirror of [`crate::viewer_html::VIEWER_JS`] -- sister file, same
/// pattern.
pub const TALLY_VIEWER_JS: &str = include_str!("tally_viewer.js");

/// Metadata for the mesh tally embedded in the HTML.
pub struct MeshMeta {
    pub lower_left: [f64; 3],
    pub upper_right: [f64; 3],
    pub shape: [usize; 3],
    pub width: [f64; 3],
}

/// A single embedded slice: (basis, bin_index, flat_f64_data).
pub struct EmbeddedSlice {
    pub basis: String,
    pub bin_index: usize,
    pub data: Vec<f64>,
}

/// Parameters for the interactive tally HTML builder.
pub struct InteractiveTallyParams {
    pub initial_basis: String,
    pub initial_slice_index: usize,
    pub colorscale: String,
    pub log_scale: bool,
    pub outline: Option<String>,
    pub outline_color: String,
    pub outline_thickness: usize,
    pub axis_units: String,
    pub outline_pixels: usize,
    pub title: String,
    pub colorbar_title: String,
    pub scaling_factor: f64,
    pub font_size: usize,
    pub show_colorbar: bool,
    pub display_value: String,
}

/// Extra value slices for tooltip (e.g. standard_deviation, relative_error alongside the display value).
pub struct ExtraValueSlices {
    pub value_name: String,
    pub slices: Vec<EmbeddedSlice>,
}

/// Build a fully self-contained HTML string for the interactive tally viewer.
// Arg count + `format!(..)` inside another `format!(..)` are pre-existing
// patterns from when this lived in yamc-python; clippy didn't run with
// -D warnings against that file. Both are intentional and changing them
// would mean splitting the giant HTML template needlessly.
#[allow(clippy::too_many_arguments, clippy::format_in_format_args)]
pub fn build_interactive_tally_html(
    geometry_json: Option<&str>,
    geometry_kind: Option<&str>,
    mesh: &MeshMeta,
    slices: &[EmbeddedSlice],
    extra_values: &[ExtraValueSlices],
    params: &InteractiveTallyParams,
    cell_names: &HashMap<i32, String>,
    material_names: &HashMap<i32, String>,
    presampled_outline: Option<&PresampledGrid>,
    surface_table: Option<&[crate::surfaces::SurfaceTableEntry]>,
) -> String {
    use base64::Engine;

    // Build name maps JSON
    let cell_names_json = {
        let entries: Vec<String> = cell_names
            .iter()
            .map(|(id, name)| format!("\"{}\":\"{}\"", id, escape_js(name)))
            .collect();
        format!("{{{}}}", entries.join(","))
    };
    let material_names_json = {
        let entries: Vec<String> = material_names
            .iter()
            .map(|(id, name)| format!("\"{}\":\"{}\"", id, escape_js(name)))
            .collect();
        format!("{{{}}}", entries.join(","))
    };

    // Surface table for hover tooltips -- same shape build_interactive_html
    // emits, JS reads via surfacesAtPoint(). Empty when no caller passed
    // a table (e.g. tally without a geometry overlay).
    let surface_table_json = match surface_table {
        Some(entries) if !entries.is_empty() => {
            let rows: Vec<String> = entries
                .iter()
                .map(|e| {
                    let surf_json =
                        serde_json::to_string(&*e.surface).unwrap_or_else(|_| "null".to_string());
                    format!(
                        "{{\"surface\":{},\"name\":\"{}\"}}",
                        surf_json,
                        escape_js(&e.display_name)
                    )
                })
                .collect();
            format!("[{}]", rows.join(","))
        }
        _ => "[]".to_string(),
    };

    // Encode slices as base64 f64 little-endian
    let mut slices_json_entries = Vec::new();
    let mut slice_indices_map: HashMap<String, Vec<usize>> = HashMap::new();
    for s in slices {
        let bytes: Vec<u8> = s.data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        slices_json_entries.push(format!("\"{}:{}\":\"{}\"", s.basis, s.bin_index, b64));
        slice_indices_map
            .entry(s.basis.clone())
            .or_default()
            .push(s.bin_index);
    }
    let slices_json = format!("{{{}}}", slices_json_entries.join(","));

    // Encode extra value slices: {"standard_deviation": {"xy:0": "b64...", ...}, ...}
    let extra_slices_json = if extra_values.is_empty() {
        "{}".to_string()
    } else {
        let entries: Vec<String> = extra_values
            .iter()
            .map(|ev| {
                let inner: Vec<String> = ev
                    .slices
                    .iter()
                    .map(|s| {
                        let bytes: Vec<u8> = s.data.iter().flat_map(|v| v.to_le_bytes()).collect();
                        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
                        format!("\"{}:{}\":\"{}\"", s.basis, s.bin_index, b64)
                    })
                    .collect();
                format!("\"{}\":{{{}}}", ev.value_name, inner.join(","))
            })
            .collect();
        format!("{{{}}}", entries.join(","))
    };

    // Build ordered indices per basis
    let mut slice_indices_json_entries = Vec::new();
    for basis in &["xy", "xz", "yz"] {
        if let Some(indices) = slice_indices_map.get(*basis) {
            let mut sorted = indices.clone();
            sorted.sort();
            let vals: Vec<String> = sorted.iter().map(|i| i.to_string()).collect();
            slice_indices_json_entries.push(format!("\"{}\":[{}]", basis, vals.join(",")));
        }
    }
    let slice_indices_json = format!("{{{}}}", slice_indices_json_entries.join(","));

    // Slice dimensions per basis
    let [nx, ny, nz] = mesh.shape;
    let slice_dims_json = format!(
        "{{\"xy\":[{},{}],\"xz\":[{},{}],\"yz\":[{},{}]}}",
        nx, ny, nx, nz, ny, nz
    );

    // Build WASM section for mesh geometries
    let (geometry_section, wasm_section) = if let Some(geo_json) = geometry_json {
        let kind = geometry_kind.unwrap_or("csg");
        let geo_literal = geo_json.to_string();

        #[cfg(feature = "mesh")]
        let wasm_part = if kind == "mesh" {
            let wasm_b64 = base64::engine::general_purpose::STANDARD.encode(YAMT_WASM_BYTES);
            let js_glue = YAMT_WASM_JS
                .replace("export { initSync, __wbg_init as default };", "")
                .replace("WasmMeshPlotter", "YamtWasmMeshPlotter")
                .replace("export class", "class");
            let js_glue = if let Some(start) = js_glue.find("async function __wbg_init(") {
                let before = &js_glue[..start];
                let rest = &js_glue[start..];
                let mut depth = 0;
                let mut end = rest.len();
                for (i, ch) in rest.char_indices() {
                    match ch {
                        '{' => depth += 1,
                        '}' => {
                            depth -= 1;
                            if depth == 0 {
                                end = i + 1;
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                format!("{}{}", before, &rest[end..])
            } else {
                js_glue
            };
            format!(
                "const YAMT_WASM_BASE64 = \"{}\";\n{}\nfunction yamtWasmInitSync(bytes) {{ initSync(bytes); }}\n",
                wasm_b64, js_glue
            )
        } else {
            String::new()
        };
        #[cfg(not(feature = "mesh"))]
        let wasm_part = String::new();

        (
            format!(
                "const GEOMETRY_JSON = {};\nconst GEOMETRY_KIND = \"{}\";",
                geo_literal, kind
            ),
            wasm_part,
        )
    } else {
        (
            "const GEOMETRY_JSON = null;\nconst GEOMETRY_KIND = null;".to_string(),
            String::new(),
        )
    };

    // Encode presampled outline grid as base64 for instant first render
    let presampled_section = if let Some(grid) = presampled_outline {
        let bytes: Vec<u8> = grid.data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        format!(
            "const PRESAMPLED_OUTLINE_B64 = \"{}\";\nconst PRESAMPLED_OUTLINE_PH = {};\nconst PRESAMPLED_OUTLINE_PV = {};",
            b64, grid.ph, grid.pv
        )
    } else {
        "const PRESAMPLED_OUTLINE_B64 = null;\nconst PRESAMPLED_OUTLINE_PH = 0;\nconst PRESAMPLED_OUTLINE_PV = 0;".to_string()
    };

    let oc = parse_hex_color(&params.outline_color);

    format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>YAMC Interactive Mesh Tally Viewer</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #e0e0e0; display: flex; height: 100vh; overflow: hidden; }}
#controls {{ width: 240px; padding: 12px; background: #16213e; overflow-y: auto; flex-shrink: 0; border-right: 1px solid #0f3460; }}
#controls h3 {{ color: #e94560; margin: 12px 0 6px 0; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; }}
#controls h3:first-child {{ margin-top: 0; }}
#controls label {{ display: block; font-size: 12px; margin: 4px 0 2px 0; color: #aaa; }}
#controls input[type=number] {{ width: 100%; padding: 4px 6px; background: #0f3460; border: 1px solid #1a1a4e; color: #e0e0e0; border-radius: 3px; font-size: 12px; }}
#controls input[type=number]:focus {{ outline: none; border-color: #e94560; }}
#controls select {{ width: 100%; padding: 4px 6px; background: #0f3460; border: 1px solid #1a1a4e; color: #e0e0e0; border-radius: 3px; font-size: 12px; }}
#controls input[type=range] {{ width: 100%; }}
.btn-group {{ display: flex; gap: 2px; margin: 4px 0; }}
.btn-group button {{ flex: 1; padding: 5px 0; background: #0f3460; border: 1px solid #1a1a4e; color: #e0e0e0; cursor: pointer; font-size: 12px; border-radius: 3px; }}
.btn-group button.active {{ background: #e94560; border-color: #e94560; color: white; }}
.btn-group button:hover:not(.active) {{ background: #1a1a4e; }}
.btn-group button:disabled {{ opacity: 0.3; cursor: not-allowed; }}
#controls .radio-group {{ margin: 4px 0; }}
#controls .radio-group label {{ display: inline-flex; align-items: center; gap: 4px; margin-right: 8px; cursor: pointer; font-size: 12px; }}
#download-btn, #copy-btn, #reset-btn {{ width: 100%; padding: 6px; margin-top: 6px; background: #0f3460; border: 1px solid #1a1a4e; color: #e0e0e0; cursor: pointer; border-radius: 3px; font-size: 12px; }}
#download-btn:hover, #copy-btn:hover, #reset-btn:hover {{ background: #1a1a4e; }}
#main {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; }}
#canvas-area {{ flex: 1; display: flex; position: relative; overflow: hidden; }}
#canvas-container {{ flex: 1; position: relative; overflow: hidden; background: #c0c0c0; }}
#plot-canvas {{ position: absolute; image-rendering: pixelated; cursor: crosshair; }}
#outline-overlay {{ position: absolute; pointer-events: none; cursor: crosshair; }}
#tooltip {{ position: fixed; pointer-events: none; background: rgba(22,33,62,0.95); border: 1px solid #e94560; border-radius: 4px; padding: 8px 12px; font-size: 14px; color: #e0e0e0; display: none; white-space: nowrap; z-index: 10000; }}
#axis-label-h {{ position: absolute; bottom: 0; font-size: 11px; color: #222; z-index: 10; pointer-events: none; }}
#axis-label-v {{ position: absolute; font-size: 11px; color: #222; z-index: 10; pointer-events: none; writing-mode: vertical-rl; transform: rotate(180deg); }}
.tick-h {{ position: absolute; font-size: 10px; color: #222; white-space: nowrap; transform: translateX(-50%); pointer-events: none; z-index: 10; }}
.tick-v {{ position: absolute; font-size: 10px; color: #222; white-space: nowrap; transform: translateY(-50%); text-align: right; pointer-events: none; z-index: 10; }}
#colorbar-container {{ width: 100px; background: #c0c0c0; position: relative; flex-shrink: 0; }}
#colorbar-canvas {{ position: absolute; left: 4px; }}
.cb-tick {{ position: absolute; font-size: 9px; color: #222; white-space: nowrap; pointer-events: none; left: 24px; transform: translateY(-50%); text-align: left; }}
#colorbar-label {{ position: absolute; font-size: 10px; color: #222; writing-mode: vertical-rl; transform: rotate(180deg); pointer-events: none; }}
#status {{ position: absolute; top: 8px; right: 88px; font-size: 11px; color: #666; z-index: 50; }}
#slice-info {{ font-size: 11px; color: #888; margin-top: 2px; }}
</style>
</head>
<body>

<div id="controls">
  <h3>Slice Plane</h3>
  <div class="btn-group" id="basis-btns">
    <button data-val="xy" {basis_xy_active}>XY</button>
    <button data-val="xz" {basis_xz_active}>XZ</button>
    <button data-val="yz" {basis_yz_active}>YZ</button>
  </div>

  <h3>Slice Position</h3>
  <input type="range" id="slice-slider" min="0" max="0" value="0" step="1">
  <div id="slice-info">Bin 0</div>

  <h3>Scale</h3>
  <div class="radio-group">
    <label><input type="radio" name="scale" value="log" {log_checked}> Log</label>
    <label><input type="radio" name="scale" value="linear" {linear_checked}> Linear</label>
  </div>

  <h3>Value Range</h3>
  <div style="display:flex;align-items:center;gap:4px;margin:4px 0;">
    <label style="font-size:12px;color:#aaa;margin:0;width:30px;">Min</label>
    <input type="number" id="vmin-input" step="any" placeholder="auto" style="flex:1;">
  </div>
  <div style="display:flex;align-items:center;gap:4px;margin:4px 0;">
    <label style="font-size:12px;color:#aaa;margin:0;width:30px;">Max</label>
    <input type="number" id="vmax-input" step="any" placeholder="auto" style="flex:1;">
  </div>

  <h3>Scaling Factor</h3>
  <input type="number" id="scaling-factor" value="{scaling_factor}" step="any">

  <h3>Colorscale</h3>
  <select id="colorscale-select">
    <option value="Viridis" {cs_viridis}>Viridis</option>
    <option value="Cividis" {cs_cividis}>Cividis</option>
    <option value="Hot" {cs_hot}>Hot</option>
    <option value="Jet" {cs_jet}>Jet</option>
    <option value="Blues" {cs_blues}>Blues</option>
    <option value="Reds" {cs_reds}>Reds</option>
    <option value="Greens" {cs_greens}>Greens</option>
    <option value="Greys" {cs_greys}>Greys</option>
    <option value="YlGnBu" {cs_ylgnbu}>YlGnBu</option>
    <option value="YlOrRd" {cs_ylorrd}>YlOrRd</option>
    <option value="Plasma" {cs_plasma}>Plasma</option>
    <option value="Inferno" {cs_inferno}>Inferno</option>
    <option value="Bluered" {cs_bluered}>Bluered</option>
    <option value="RdBu" {cs_rdbu}>RdBu</option>
    <option value="Picnic" {cs_picnic}>Picnic</option>
    <option value="Rainbow" {cs_rainbow}>Rainbow</option>
    <option value="Portland" {cs_portland}>Portland</option>
    <option value="Blackbody" {cs_blackbody}>Blackbody</option>
  </select>
  <label>Levels</label>
  <select id="levels-select">
    <option value="0" selected>Continuous</option>
    <option value="2">2</option>
    <option value="3">3</option>
    <option value="4">4</option>
    <option value="5">5</option>
    <option value="6">6</option>
    <option value="8">8</option>
    <option value="10">10</option>
    <option value="12">12</option>
    <option value="15">15</option>
    <option value="20">20</option>
  </select>

  <h3>Outline</h3>
  <div class="radio-group">
    <label><input type="radio" name="outline" value="none" {outline_none_checked}> None</label>
    <label><input type="radio" name="outline" value="material" {outline_mat_checked}> Material</label>
    <label><input type="radio" name="outline" value="cell" {outline_cell_checked}> Cell</label>
  </div>
  <div style="display:flex;align-items:center;gap:8px;margin:6px 0;">
    <label style="font-size:12px;color:#aaa;margin:0;">Color</label>
    <input type="color" id="outline-color" value="{outline_color_value}" style="width:28px;height:22px;padding:0;border:1px solid #1a1a4e;background:#0f3460;cursor:pointer;border-radius:3px;">
    <label style="font-size:12px;color:#aaa;margin:0;">Thickness</label>
    <input type="number" id="outline-width" value="{outline_thickness_value}" min="1" max="5" step="1" style="width:44px;">
  </div>
  <label>Pixels</label>
  <input type="number" id="outline-pixels" value="{outline_pixels}" min="1000" max="4000000" step="10000">

  <h3>Units</h3>
  <div class="radio-group">
    <label><input type="radio" name="units" value="mm" {units_mm_checked}> mm</label>
    <label><input type="radio" name="units" value="cm" {units_cm_checked}> cm</label>
    <label><input type="radio" name="units" value="m" {units_m_checked}> m</label>
    <label><input type="radio" name="units" value="km" {units_km_checked}> km</label>
  </div>

  <h3>Font Size</h3>
  <input type="number" id="font-size" value="{font_size}" min="6" max="36" step="1">

  <h3>Colorbar Label</h3>
  <input type="text" id="colorbar-label-input" value="{colorbar_title_escaped}" style="width:100%;padding:4px 6px;background:#0f3460;border:1px solid #1a1a4e;color:#e0e0e0;border-radius:3px;font-size:12px;">

  <label style="display:flex;align-items:center;gap:6px;margin-top:8px;font-size:12px;color:#aaa;cursor:pointer;">
    <input type="checkbox" id="show-colorbar" {show_colorbar_checked}> Show Colorbar
  </label>

  <button id="reset-btn">Reset View</button>
  <button id="copy-btn">Copy Python Code</button>
  <button id="download-btn">Download PNG</button>
</div>

<div id="main">
  <div id="canvas-area">
    <div id="canvas-container">
      <canvas id="plot-canvas"></canvas>
      <canvas id="outline-overlay"></canvas>
      <div id="tooltip"></div>
      <div id="tick-container-h"></div>
      <div id="tick-container-v"></div>
      <div id="axis-label-h"></div>
      <div id="axis-label-v"></div>
      <div id="status"></div>
    </div>
    <div id="colorbar-container">
      <canvas id="colorbar-canvas" width="16" height="256"></canvas>
      <div id="colorbar-ticks"></div>
      <div id="colorbar-label">{colorbar_title_escaped}</div>
    </div>
  </div>
</div>

<script>
// ---- Embedded data ----
{geometry_section}
{presampled_section}
const CELL_NAMES = {cell_names_json};
const MATERIAL_NAMES = {material_names_json};
// Surface name + identity table -- drives hover tooltips. Same shape
// model.plot() embeds, same surfaceEval consumer below. Empty when
// the tally was plotted without a geometry overlay.
const SURFACE_TABLE = {surface_table_json};

const MESH_LL = [{mesh_ll}];
const MESH_UR = [{mesh_ur}];
const MESH_DIM = [{mesh_dim}];
const MESH_WIDTH = [{mesh_width}];

const SLICES = {slices_json};
const EXTRA_SLICES = {extra_slices_json};
const DISPLAY_VALUE = "{display_value}";
const SLICE_INDICES = {slice_indices_json};
const SLICE_DIMS = {slice_dims_json};
const OUTLINE_PIXELS = {outline_pixels};
const INITIAL_SLICE_BIN = {initial_slice_index};
const NATIVE_BASIS = "{initial_basis}";
const INITIAL_SCALING_FACTOR = {scaling_factor};
const INITIAL_FONT_SIZE = {font_size};

const COLORSCALES = {{
  'Viridis': [[0,[68,1,84]],[0.06,[72,24,106]],[0.12,[71,45,123]],[0.18,[66,64,134]],[0.24,[59,82,139]],[0.3,[51,99,141]],[0.36,[44,114,142]],[0.42,[38,130,142]],[0.48,[33,145,140]],[0.54,[31,160,136]],[0.6,[40,174,128]],[0.66,[63,188,115]],[0.72,[94,201,98]],[0.78,[132,212,75]],[0.84,[173,220,48]],[0.9,[216,226,25]],[1,[253,231,37]]],
  'Cividis': [[0,[0,32,76]],[0.25,[66,76,108]],[0.5,[124,123,120]],[0.75,[188,175,111]],[1,[253,234,69]]],
  'Hot': [[0,[0,0,0]],[0.33,[230,0,0]],[0.66,[255,210,0]],[1,[255,255,255]]],
  'Jet': [[0,[0,0,131]],[0.125,[0,0,255]],[0.25,[0,130,255]],[0.375,[0,255,255]],[0.5,[130,255,130]],[0.625,[255,255,0]],[0.75,[255,130,0]],[0.875,[255,0,0]],[1,[128,0,0]]],
  'Blues': [[0,[247,251,255]],[0.25,[186,214,235]],[0.5,[107,174,214]],[0.75,[33,113,181]],[1,[8,48,107]]],
  'Reds': [[0,[255,245,240]],[0.25,[252,174,145]],[0.5,[251,106,74]],[0.75,[203,24,29]],[1,[103,0,13]]],
  'Greens': [[0,[247,252,245]],[0.25,[186,228,179]],[0.5,[116,196,118]],[0.75,[35,139,69]],[1,[0,68,27]]],
  'Greys': [[0,[255,255,255]],[0.5,[150,150,150]],[1,[0,0,0]]],
  'YlGnBu': [[0,[255,255,217]],[0.25,[161,218,180]],[0.5,[65,182,196]],[0.75,[34,94,168]],[1,[8,29,88]]],
  'YlOrRd': [[0,[255,255,178]],[0.25,[254,204,92]],[0.5,[253,141,60]],[0.75,[227,26,28]],[1,[128,0,38]]],
  'Plasma': [[0,[13,8,135]],[0.13,[75,3,161]],[0.25,[126,3,168]],[0.38,[168,34,150]],[0.5,[203,70,121]],[0.63,[229,107,93]],[0.75,[248,148,65]],[0.88,[253,195,40]],[1,[240,249,33]]],
  'Inferno': [[0,[0,0,4]],[0.13,[20,11,53]],[0.25,[58,12,96]],[0.38,[99,25,108]],[0.5,[140,41,99]],[0.63,[184,55,78]],[0.75,[221,81,58]],[0.88,[243,137,36]],[1,[252,255,164]]],
  'Bluered': [[0,[0,0,255]],[0.5,[220,220,220]],[1,[255,0,0]]],
  'RdBu': [[0,[103,0,31]],[0.25,[214,96,77]],[0.5,[247,247,247]],[0.75,[67,147,195]],[1,[5,48,97]]],
  'Picnic': [[0,[0,0,255]],[0.25,[51,153,255]],[0.5,[255,255,255]],[0.75,[255,102,102]],[1,[255,0,0]]],
  'Rainbow': [[0,[150,0,90]],[0.14,[0,0,200]],[0.28,[0,25,255]],[0.42,[0,152,255]],[0.56,[44,255,150]],[0.7,[151,255,0]],[0.84,[255,234,0]],[1,[255,0,0]]],
  'Portland': [[0,[12,51,131]],[0.25,[10,136,186]],[0.5,[242,211,56]],[0.75,[242,143,56]],[1,[217,30,30]]],
  'Blackbody': [[0,[0,0,0]],[0.2,[230,0,0]],[0.4,[255,210,0]],[0.7,[255,255,255]],[1,[160,200,255]]]
}};

const TITLE = "{title_escaped}";

// ---- State ----
let state = {{
  basis: "{initial_basis}",
  sliceIdx: 0, // index into SLICE_INDICES[basis] array
  logScale: {log_scale_js},
  colorscale: "{initial_colorscale}",
  outline: "{initial_outline}",
  outlineColor: [{outline_r},{outline_g},{outline_b}],
  outlineWidth: {outline_thickness_value},
  outlinePixels: OUTLINE_PIXELS,
  scalingFactor: INITIAL_SCALING_FACTOR,
  fontSize: INITIAL_FONT_SIZE,
  colorbarLabel: "{colorbar_title_escaped}",
  showColorbar: {show_colorbar_js},
  levels: 0,
  vmin: null,
  vmax: null,
  units: "{axis_units}",
  // View (zoom/pan) -- in cm world coords
  originH: 0, originV: 0,
  widthH: 0, widthV: 0,
  // Data range for colorbar
  dataMin: 0, dataMax: 1,
  // Current frame data (for tooltip lookup)
  currentSliceData: null,
  currentSliceMask: null,
}};

// Helpers + DOM + init logic live in tally_viewer.js (an editable
// JS file) and are inlined here. They depend on the consts above
// and on {wasm_section}'s YAMT_WASM_BASE64 (mesh path).
{wasm_section}
{tally_viewer_js_body}
init();
</script>
</body>
</html>"##,
        // Template substitutions
        initial_basis = params.initial_basis,
        basis_xy_active = if params.initial_basis == "xy" {
            r#"class="active""#
        } else {
            ""
        },
        basis_xz_active = if params.initial_basis == "xz" {
            r#"class="active""#
        } else {
            ""
        },
        basis_yz_active = if params.initial_basis == "yz" {
            r#"class="active""#
        } else {
            ""
        },
        log_checked = if params.log_scale { "checked" } else { "" },
        linear_checked = if !params.log_scale { "checked" } else { "" },
        initial_colorscale = params.colorscale,
        cs_viridis = if params.colorscale == "Viridis" {
            "selected"
        } else {
            ""
        },
        cs_cividis = if params.colorscale == "Cividis" {
            "selected"
        } else {
            ""
        },
        cs_hot = if params.colorscale == "Hot" {
            "selected"
        } else {
            ""
        },
        cs_jet = if params.colorscale == "Jet" {
            "selected"
        } else {
            ""
        },
        cs_blues = if params.colorscale == "Blues" {
            "selected"
        } else {
            ""
        },
        cs_reds = if params.colorscale == "Reds" {
            "selected"
        } else {
            ""
        },
        cs_greens = if params.colorscale == "Greens" {
            "selected"
        } else {
            ""
        },
        cs_greys = if params.colorscale == "Greys" {
            "selected"
        } else {
            ""
        },
        cs_ylgnbu = if params.colorscale == "YlGnBu" {
            "selected"
        } else {
            ""
        },
        cs_ylorrd = if params.colorscale == "YlOrRd" {
            "selected"
        } else {
            ""
        },
        cs_plasma = if params.colorscale == "Plasma" {
            "selected"
        } else {
            ""
        },
        cs_inferno = if params.colorscale == "Inferno" {
            "selected"
        } else {
            ""
        },
        cs_bluered = if params.colorscale == "Bluered" {
            "selected"
        } else {
            ""
        },
        cs_rdbu = if params.colorscale == "RdBu" {
            "selected"
        } else {
            ""
        },
        cs_picnic = if params.colorscale == "Picnic" {
            "selected"
        } else {
            ""
        },
        cs_rainbow = if params.colorscale == "Rainbow" {
            "selected"
        } else {
            ""
        },
        cs_portland = if params.colorscale == "Portland" {
            "selected"
        } else {
            ""
        },
        cs_blackbody = if params.colorscale == "Blackbody" {
            "selected"
        } else {
            ""
        },
        outline_none_checked = if params.outline.is_none() {
            "checked"
        } else {
            ""
        },
        outline_mat_checked = if params.outline.as_deref() == Some("material") {
            "checked"
        } else {
            ""
        },
        outline_cell_checked = if params.outline.as_deref() == Some("cell") {
            "checked"
        } else {
            ""
        },
        outline_color_value = params.outline_color,
        outline_thickness_value = params.outline_thickness.max(1),
        outline_r = oc[0],
        outline_g = oc[1],
        outline_b = oc[2],
        axis_units = params.axis_units,
        units_mm_checked = if params.axis_units == "mm" {
            "checked"
        } else {
            ""
        },
        units_cm_checked = if params.axis_units == "cm" {
            "checked"
        } else {
            ""
        },
        units_m_checked = if params.axis_units == "m" {
            "checked"
        } else {
            ""
        },
        units_km_checked = if params.axis_units == "km" {
            "checked"
        } else {
            ""
        },
        log_scale_js = if params.log_scale { "true" } else { "false" },
        initial_outline = params.outline.as_deref().unwrap_or("none"),
        geometry_section = geometry_section,
        presampled_section = presampled_section,
        wasm_section = wasm_section,
        tally_viewer_js_body = TALLY_VIEWER_JS,
        cell_names_json = cell_names_json,
        material_names_json = material_names_json,
        surface_table_json = surface_table_json,
        mesh_ll = format!(
            "{},{},{}",
            mesh.lower_left[0], mesh.lower_left[1], mesh.lower_left[2]
        ),
        mesh_ur = format!(
            "{},{},{}",
            mesh.upper_right[0], mesh.upper_right[1], mesh.upper_right[2]
        ),
        mesh_dim = format!("{},{},{}", mesh.shape[0], mesh.shape[1], mesh.shape[2]),
        mesh_width = format!("{},{},{}", mesh.width[0], mesh.width[1], mesh.width[2]),
        slices_json = slices_json,
        extra_slices_json = extra_slices_json,
        display_value = escape_js(&params.display_value),
        slice_indices_json = slice_indices_json,
        slice_dims_json = slice_dims_json,
        outline_pixels = params.outline_pixels,
        initial_slice_index = params.initial_slice_index,
        scaling_factor = params.scaling_factor,
        font_size = params.font_size,
        show_colorbar_checked = if params.show_colorbar { "checked" } else { "" },
        show_colorbar_js = if params.show_colorbar {
            "true"
        } else {
            "false"
        },
        title_escaped = escape_js(&params.title),
        colorbar_title_escaped = escape_js(&params.colorbar_title),
    )
}
