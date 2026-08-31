//! Pure-Rust core of the interactive geometry viewer.
//!
//! Builds a self-contained HTML page with embedded WASM (CSG path) or
//! base64-bundled WASM (mesh path) for interactive geometry visualization.
//! Used by both:
//!
//! - `yamc-python::PyInteractivePlot` (Python `model.plot()`)
//! - the browser-side editor, via `yamc::wasm::WasmSimulation`
//!   (which exposes raw slice samples; the HTML/JS template here is
//!   read at runtime and assembled in JS).
//!
//! See the docstring on [`build_interactive_html`] for the data
//! contract the JS expects.

use std::collections::HashMap;

/// 16-color discrete palette (same as plot_helper.rs)
pub const DISCRETE_COLORS: &[&str] = &[
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94",
];

/// Parameters for initial view
pub struct InteractiveViewParams {
    pub origin: (f64, f64, f64),
    pub width: (f64, f64),
    pub total_pixels: usize,
    pub basis: String,
    pub color_by: String,
    pub outline: Option<String>,
    pub axis_units: String,
    /// Full 3D bounding box widths [wx, wy, wz] in cm.
    pub bbox_widths: [f64; 3],
    /// Outline colour as "#rrggbb" (from contour_kwargs["colors"]).
    pub outline_color: String,
    /// Outline thickness in pixels (from contour_kwargs["linewidths"]).
    pub outline_thickness: usize,
    /// Number of source particles to plot (model only, None = hidden).
    pub n_samples: Option<usize>,
    /// Tolerance for source particles on plot plane, in cm (model only).
    pub plane_tolerance: f64,
    /// Source point colour as "#rrggbb" (from source_kwargs["color"]).
    pub source_color: String,
    /// Source point size in pixels (from source_kwargs["size"]).
    pub source_size: usize,
}

// Compile-time embedded yamt WASM binary (only included in mesh HTML).
#[cfg(feature = "mesh")]
static YAMT_WASM_BYTES: &[u8] =
    include_bytes!("../../../packages/yamc-core/python/yamc/_wasm/yamt_bg.wasm");
// Compile-time embedded yamt wasm-bindgen JS glue (only included in mesh HTML).
#[cfg(feature = "mesh")]
static YAMT_WASM_JS: &str = include_str!("../../../packages/yamc-core/python/yamc/_wasm/yamt.js");

/// Viewer JavaScript body -- model-independent rendering / interaction
/// code. Lives in `viewer.js` so it can be edited as a real JS file;
/// inlined into the HTML built by [`build_interactive_html`] after the
/// per-model data globals (GEOMETRY_JSON, INITIAL, etc.) are declared.
///
/// Also exposed as `VIEWER_JS` so the browser-side editor can mount the
/// same renderer (see `yamc::wasm::WasmSimulation`).
pub const VIEWER_JS: &str = include_str!("viewer.js");

/// Pre-sampled grid data for instant initial render.
/// Interleaved [cell_id, material_id, ...] with rows top-to-bottom.
pub struct PresampledGrid {
    /// Interleaved [cell_id, material_id, ...] -- length = ph * pv * 2.
    pub data: Vec<i32>,
    /// Horizontal pixel count.
    pub ph: usize,
    /// Vertical pixel count.
    pub pv: usize,
}

/// Parsed contour kwargs (outline appearance).
pub struct ContourParams {
    pub colors: String,
    pub linewidths: usize,
}

impl Default for ContourParams {
    fn default() -> Self {
        Self {
            colors: "#000000".to_string(),
            linewidths: 1,
        }
    }
}

/// Parsed source kwargs (source point appearance).
pub struct SourceParams {
    pub color: String,
    pub size: usize,
}

impl Default for SourceParams {
    fn default() -> Self {
        Self {
            color: "#ff0000".to_string(),
            size: 3,
        }
    }
}

/// Parse a color string to [r, g, b].
/// Accepts "#rrggbb" hex codes and CSS named colors.
pub fn parse_hex_color(color: &str) -> [u8; 3] {
    let s = color.trim();
    let hex = s.trim_start_matches('#');
    if hex.len() >= 6 && hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(128);
        let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(128);
        let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(128);
        return [r, g, b];
    }
    // CSS named colors (common subset)
    match s.to_ascii_lowercase().as_str() {
        "black" => [0, 0, 0],
        "white" => [255, 255, 255],
        "red" => [255, 0, 0],
        "green" => [0, 128, 0],
        "blue" => [0, 0, 255],
        "yellow" => [255, 255, 0],
        "cyan" | "aqua" => [0, 255, 255],
        "magenta" | "fuchsia" => [255, 0, 255],
        "orange" => [255, 165, 0],
        "purple" => [128, 0, 128],
        "brown" => [165, 42, 42],
        "grey" | "gray" => [128, 128, 128],
        "pink" => [255, 192, 203],
        "lime" => [0, 255, 0],
        "navy" => [0, 0, 128],
        "teal" => [0, 128, 128],
        "maroon" => [128, 0, 0],
        "olive" => [128, 128, 0],
        "silver" => [192, 192, 192],
        "coral" => [255, 127, 80],
        "salmon" => [250, 128, 114],
        "gold" => [255, 215, 0],
        "violet" => [238, 130, 238],
        "indigo" => [75, 0, 130],
        "tan" => [210, 180, 140],
        "crimson" => [220, 20, 60],
        "turquoise" => [64, 224, 208],
        "khaki" => [240, 230, 140],
        "tomato" => [255, 99, 71],
        "orchid" => [218, 112, 214],
        "sienna" => [160, 82, 45],
        "plum" => [221, 160, 221],
        "peru" => [205, 133, 63],
        "chocolate" => [210, 105, 30],
        "firebrick" => [178, 34, 34],
        "steelblue" => [70, 130, 180],
        "slategray" | "slategrey" => [112, 128, 144],
        "darkgreen" => [0, 100, 0],
        "darkblue" => [0, 0, 139],
        "darkred" => [139, 0, 0],
        "darkorange" => [255, 140, 0],
        "lightblue" => [173, 216, 230],
        "lightgreen" => [144, 238, 144],
        "lightgray" | "lightgrey" => [211, 211, 211],
        "darkgray" | "darkgrey" => [169, 169, 169],
        _ => [128, 128, 128],
    }
}

/// Pre-parsed discrete color palette.
fn discrete_palette() -> Vec<[u8; 3]> {
    DISCRETE_COLORS.iter().map(|c| parse_hex_color(c)).collect()
}

/// Render pre-sampled grid data to RGBA pixels using the same logic as the JS renderGrid.
pub fn render_grid_rgba(
    grid: &PresampledGrid,
    color_by: &str,
    outline: Option<&str>,
    colors: Option<&HashMap<i32, String>>,
    outline_color: &str,
    outline_thickness: usize,
) -> Vec<u8> {
    let ph = grid.ph;
    let pv = grid.pv;
    let palette = discrete_palette();
    let mut rgba = vec![255u8; ph * pv * 4]; // white opaque default

    // Build auto-color assignment: first-encounter order (matches JS)
    let mut id_to_color_idx: HashMap<i32, usize> = HashMap::new();
    let mut next_idx = 0usize;
    for i in 0..pv {
        for j in 0..ph {
            let idx = (i * ph + j) * 2;
            let cell_id = grid.data[idx];
            let mat_id = grid.data[idx + 1];
            if cell_id == -1 {
                continue;
            }
            let display_id = if color_by == "cell" { cell_id } else { mat_id };
            if display_id != -1 && !id_to_color_idx.contains_key(&display_id) {
                id_to_color_idx.insert(display_id, next_idx);
                next_idx += 1;
            }
        }
    }

    // Render pixels
    for i in 0..pv {
        for j in 0..ph {
            let idx = (i * ph + j) * 2;
            let cell_id = grid.data[idx];
            let mat_id = grid.data[idx + 1];
            let pix = (i * ph + j) * 4;

            if cell_id == -1 {
                // Background: transparent
                rgba[pix + 3] = 0;
                continue;
            }

            let display_id = if color_by == "cell" { cell_id } else { mat_id };
            let rgb = if let Some(custom) = colors.and_then(|c| c.get(&display_id)) {
                parse_hex_color(custom)
            } else if let Some(&ci) = id_to_color_idx.get(&display_id) {
                palette[ci % palette.len()]
            } else {
                [200, 200, 200] // fallback grey
            };

            rgba[pix] = rgb[0];
            rgba[pix + 1] = rgb[1];
            rgba[pix + 2] = rgb[2];
            rgba[pix + 3] = 255;
        }
    }

    // Apply outline if requested
    if let Some(outline_mode) = outline {
        let outline_mode = if outline_mode == "none" {
            return rgba;
        } else {
            outline_mode
        };
        let oc = parse_hex_color(outline_color);
        let ot = outline_thickness.max(1);
        // Build ID grid for boundary detection
        let id_grid: Vec<i32> = (0..pv * ph)
            .map(|k| {
                let idx = k * 2;
                if outline_mode == "cell" {
                    grid.data[idx]
                } else {
                    grid.data[idx + 1]
                }
            })
            .collect();

        for i in 0..pv {
            for j in 0..ph {
                let id = id_grid[i * ph + j];
                let mut is_boundary = false;
                for d in 1..=ot {
                    if j >= d && id_grid[i * ph + j - d] != id {
                        is_boundary = true;
                        break;
                    }
                    if j + d < ph && id_grid[i * ph + j + d] != id {
                        is_boundary = true;
                        break;
                    }
                    if i >= d && id_grid[(i - d) * ph + j] != id {
                        is_boundary = true;
                        break;
                    }
                    if i + d < pv && id_grid[(i + d) * ph + j] != id {
                        is_boundary = true;
                        break;
                    }
                }
                if is_boundary {
                    let pix = (i * ph + j) * 4;
                    rgba[pix] = oc[0];
                    rgba[pix + 1] = oc[1];
                    rgba[pix + 2] = oc[2];
                    rgba[pix + 3] = 255;
                }
            }
        }
    }

    rgba
}

/// Source points to overlay on the PNG grid.
pub struct SourceOverlay {
    /// Source positions [[x, y, z], ...] -- pre-sampled.
    pub points: Vec<[f64; 3]>,
    /// Max number to plot.
    pub n_samples: usize,
    /// Tolerance distance from slice plane (cm).
    pub plane_tolerance: f64,
    /// Marker colour "#rrggbb".
    pub color: String,
    /// Marker radius in pixels.
    pub size: usize,
    /// Slice basis ("xy", "xz", "yz").
    pub basis: String,
    /// Plot origin (x, y, z) in cm.
    pub origin: (f64, f64, f64),
    /// Plot width (h, v) in cm.
    pub width: (f64, f64),
}

/// Draw source point markers onto a grid RGBA buffer.
pub fn draw_source_overlay(rgba: &mut [u8], ph: usize, pv: usize, src: &SourceOverlay) {
    let c = parse_hex_color(&src.color);
    let r = src.size.max(1) as f64;
    let r2 = r * r;
    let n = src.n_samples.min(src.points.len());

    let (h_origin, v_origin, slice_coord) = match src.basis.as_str() {
        "xz" => (src.origin.0, src.origin.2, src.origin.1),
        "yz" => (src.origin.1, src.origin.2, src.origin.0),
        _ => (src.origin.0, src.origin.1, src.origin.2),
    };

    for pt in src.points.iter().take(n) {
        let (x, y, z) = (pt[0], pt[1], pt[2]);

        // Check distance from slice plane
        let dist = match src.basis.as_str() {
            "xz" => (y - slice_coord).abs(),
            "yz" => (x - slice_coord).abs(),
            _ => (z - slice_coord).abs(),
        };
        if dist > src.plane_tolerance {
            continue;
        }

        // Map to horizontal/vertical world coords
        let (h, v) = match src.basis.as_str() {
            "xz" => (x, z),
            "yz" => (y, z),
            _ => (x, y),
        };

        // Convert to pixel coords (same mapping as JS)
        let frac_x = (h - h_origin + src.width.0 / 2.0) / src.width.0;
        let frac_y = (v - v_origin + src.width.1 / 2.0) / src.width.1;
        if !(0.0..=1.0).contains(&frac_x) || !(0.0..=1.0).contains(&frac_y) {
            continue;
        }

        let px = frac_x * ph as f64;
        let py = (1.0 - frac_y) * pv as f64;

        // Stamp filled circle
        let imin = ((py - r).floor() as isize).max(0) as usize;
        let imax = ((py + r).ceil() as isize).min(pv as isize - 1) as usize;
        let jmin = ((px - r).floor() as isize).max(0) as usize;
        let jmax = ((px + r).ceil() as isize).min(ph as isize - 1) as usize;
        for i in imin..=imax {
            for j in jmin..=jmax {
                let dx = j as f64 - px;
                let dy = i as f64 - py;
                if dx * dx + dy * dy <= r2 {
                    let idx = (i * ph + j) * 4;
                    rgba[idx] = c[0];
                    rgba[idx + 1] = c[1];
                    rgba[idx + 2] = c[2];
                    rgba[idx + 3] = 255;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Bitmap font for PNG axis labels (5x7 pixel glyphs)
// Each glyph is 7 rows of 5 bits (MSB-first), stored as [u8; 7].
// ---------------------------------------------------------------------------
fn glyph(ch: char) -> [u8; 7] {
    match ch {
        '0' => [
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ],
        '1' => [
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        '2' => [
            0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111,
        ],
        '3' => [
            0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110,
        ],
        '4' => [
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ],
        '5' => [
            0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
        ],
        '6' => [
            0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
        ],
        '7' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
        ],
        '8' => [
            0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
        ],
        '9' => [
            0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100,
        ],
        '.' => [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100,
        ],
        '-' => [
            0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000,
        ],
        'e' => [
            0b00000, 0b00000, 0b01110, 0b10001, 0b11111, 0b10000, 0b01110,
        ],
        'x' => [
            0b00000, 0b00000, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001,
        ],
        'y' => [
            0b00000, 0b00000, 0b10001, 0b01010, 0b00100, 0b01000, 0b10000,
        ],
        'z' => [
            0b00000, 0b00000, 0b11111, 0b00010, 0b00100, 0b01000, 0b11111,
        ],
        'c' => [
            0b00000, 0b00000, 0b01110, 0b10000, 0b10000, 0b10001, 0b01110,
        ],
        'm' => [
            0b00000, 0b00000, 0b11010, 0b10101, 0b10101, 0b10001, 0b10001,
        ],
        'k' => [
            0b10000, 0b10000, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010,
        ],
        ' ' => [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000,
        ],
        '(' => [
            0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010,
        ],
        ')' => [
            0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000,
        ],
        _ => [
            0b11111, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11111,
        ], // box
    }
}

const GLYPH_W: usize = 5;
const GLYPH_H: usize = 7;
const GLYPH_SPACING: usize = 1;

/// Set a scaled block of pixels (scale x scale) at a given position.
fn set_scaled_pixel(buf: &mut [u8], buf_w: usize, x: i32, y: i32, scale: usize, color: [u8; 4]) {
    for dy in 0..scale as i32 {
        for dx in 0..scale as i32 {
            let px = x + dx;
            let py = y + dy;
            if px >= 0 && py >= 0 && (px as usize) < buf_w {
                let idx = (py as usize * buf_w + px as usize) * 4;
                if idx + 3 < buf.len() {
                    buf[idx] = color[0];
                    buf[idx + 1] = color[1];
                    buf[idx + 2] = color[2];
                    buf[idx + 3] = color[3];
                }
            }
        }
    }
}

/// Draw a text string onto an RGBA buffer at (x, y) with integer scale factor.
fn draw_text(
    buf: &mut [u8],
    buf_w: usize,
    x: i32,
    y: i32,
    text: &str,
    color: [u8; 4],
    scale: usize,
) {
    let sw = GLYPH_W * scale;
    let sp = GLYPH_SPACING * scale;
    let mut cx = x;
    for ch in text.chars() {
        let g = glyph(ch);
        for (row, &bits) in g.iter().enumerate() {
            for col in 0..GLYPH_W {
                if bits & (1 << (GLYPH_W - 1 - col)) != 0 {
                    set_scaled_pixel(
                        buf,
                        buf_w,
                        cx + (col * scale) as i32,
                        y + (row * scale) as i32,
                        scale,
                        color,
                    );
                }
            }
        }
        cx += (sw + sp) as i32;
    }
}

/// Measure text width in pixels at a given scale.
fn text_width(text: &str, scale: usize) -> usize {
    if text.is_empty() {
        return 0;
    }
    let sw = GLYPH_W * scale;
    let sp = GLYPH_SPACING * scale;
    text.len() * (sw + sp) - sp
}

/// Scaled glyph height.
fn text_height(scale: usize) -> usize {
    GLYPH_H * scale
}

/// Draw a text string centered horizontally at (cx, y).
fn draw_text_centered(
    buf: &mut [u8],
    buf_w: usize,
    cx: i32,
    y: i32,
    text: &str,
    color: [u8; 4],
    scale: usize,
) {
    let w = text_width(text, scale) as i32;
    draw_text(buf, buf_w, cx - w / 2, y, text, color, scale);
}

/// Draw a text string rotated 90 degrees CCW, centered vertically at (x, cy).
#[allow(clippy::too_many_arguments)]
fn draw_text_vertical(
    buf: &mut [u8],
    buf_w: usize,
    _buf_h: usize,
    x: i32,
    cy: i32,
    text: &str,
    color: [u8; 4],
    scale: usize,
) {
    let sw = GLYPH_W * scale;
    let sp = GLYPH_SPACING * scale;
    let total_w = text_width(text, scale) as i32;
    let start_y = cy + total_w / 2; // bottom of the vertical text
    let mut char_x = 0i32;
    for ch in text.chars() {
        let g = glyph(ch);
        for (row, &bits) in g.iter().enumerate() {
            for col in 0..GLYPH_W {
                if bits & (1 << (GLYPH_W - 1 - col)) != 0 {
                    // Rotate 90° CCW: (col, row) → (row, total_w - col - char_x)
                    set_scaled_pixel(
                        buf,
                        buf_w,
                        x + (row * scale) as i32,
                        start_y - char_x - ((col + 1) * scale) as i32 + 1,
                        scale,
                        color,
                    );
                }
            }
        }
        char_x += (sw + sp) as i32;
    }
}

/// Format a tick value nicely (no trailing zeros).
fn format_tick(val: f64) -> String {
    if val.abs() < 1e-10 {
        return "0".to_string();
    }
    if val.abs() >= 1000.0 || (val.abs() < 0.01 && val.abs() > 0.0) {
        format!("{:.1e}", val)
    } else if (val - val.round()).abs() < 1e-10 {
        format!("{}", val as i64)
    } else {
        let s = format!("{:.2}", val);
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// Compute nice tick positions for a range.
fn nice_ticks(min_val: f64, max_val: f64, max_ticks: usize) -> Vec<f64> {
    let range = max_val - min_val;
    if range <= 0.0 {
        return vec![min_val];
    }
    let rough_step = range / max_ticks as f64;
    let mag = 10f64.powf(rough_step.log10().floor());
    let residual = rough_step / mag;
    let nice_step = if residual <= 1.5 {
        mag
    } else if residual <= 3.5 {
        2.0 * mag
    } else if residual <= 7.5 {
        5.0 * mag
    } else {
        10.0 * mag
    };
    let start = (min_val / nice_step).ceil() * nice_step;
    let mut ticks = Vec::new();
    let mut v = start;
    while v <= max_val + nice_step * 0.001 {
        ticks.push(v);
        v += nice_step;
    }
    ticks
}

/// Unit conversion factor from cm.
fn unit_scale(axis_units: &str) -> f64 {
    match axis_units {
        "mm" => 10.0,
        "m" => 0.01,
        "km" => 1e-5,
        _ => 1.0, // cm
    }
}

/// Render a full PNG image: plot grid + border + tick marks + axis labels.
///
/// `font_scale` is an integer multiplier on the 5x7 bitmap glyphs (1 = tiny, 2 = normal, 3 = large).
#[allow(clippy::too_many_arguments)]
pub fn render_png_rgba(
    grid_rgba: &[u8],
    ph: usize,
    pv: usize,
    origin: (f64, f64, f64),
    width: (f64, f64),
    basis: &str,
    axis_units: &str,
    font_scale: usize,
) -> (Vec<u8>, u32, u32) {
    let uscale = unit_scale(axis_units);
    let fs = font_scale.max(1);
    let th = text_height(fs); // scaled glyph height
    let tick_len = 4 + fs; // tick mark length scales slightly
    let gap = 2 + fs; // gap between elements

    // Axis ranges in display units
    let (h_label, v_label) = match basis {
        "xz" => ("x", "z"),
        "yz" => ("y", "z"),
        _ => ("x", "y"),
    };
    let (h_origin, v_origin) = match basis {
        "xz" => (origin.0, origin.2),
        "yz" => (origin.1, origin.2),
        _ => (origin.0, origin.1),
    };
    let h_min = (h_origin - width.0 / 2.0) * uscale;
    let h_max = (h_origin + width.0 / 2.0) * uscale;
    let v_min = (v_origin - width.1 / 2.0) * uscale;
    let v_max = (v_origin + width.1 / 2.0) * uscale;

    // Estimate widest vertical tick label for left margin
    let v_ticks = nice_ticks(v_min, v_max, 6);
    let max_vtick_w = v_ticks
        .iter()
        .map(|&v| text_width(&format_tick(v), fs))
        .max()
        .unwrap_or(0);

    // Margins scaled to font size
    let margin_left = max_vtick_w + tick_len + gap + th + gap * 2; // vtick labels + tick + gap + axis label + padding
    let margin_right = gap * 3;
    let margin_top = gap * 3;
    let margin_bottom = tick_len + gap + th + gap + th + gap; // tick + gap + tick labels + gap + axis label + padding

    let total_w = margin_left + ph + margin_right;
    let total_h = margin_top + pv + margin_bottom;

    let black = [0u8, 0, 0, 255];
    let grey = [153u8, 153, 153, 255];

    // Always transparent background (buffer is zero-initialized = fully transparent)
    let mut buf = vec![0u8; total_w * total_h * 4];

    // Copy plot grid
    for row in 0..pv {
        for col in 0..ph {
            let src = (row * ph + col) * 4;
            let dst_x = margin_left + col;
            let dst_y = margin_top + row;
            let dst = (dst_y * total_w + dst_x) * 4;
            buf[dst] = grid_rgba[src];
            buf[dst + 1] = grid_rgba[src + 1];
            buf[dst + 2] = grid_rgba[src + 2];
            buf[dst + 3] = grid_rgba[src + 3];
        }
    }

    // Draw border around plot area
    let plot_left = margin_left;
    let plot_right = margin_left + ph - 1;
    let plot_top = margin_top;
    let plot_bottom = margin_top + pv - 1;

    let set_px = |buf: &mut Vec<u8>, x: usize, y: usize, c: [u8; 4]| {
        let idx = (y * total_w + x) * 4;
        buf[idx] = c[0];
        buf[idx + 1] = c[1];
        buf[idx + 2] = c[2];
        buf[idx + 3] = c[3];
    };
    for x in plot_left..=plot_right {
        set_px(&mut buf, x, plot_top, grey);
        set_px(&mut buf, x, plot_bottom, grey);
    }
    for y in plot_top..=plot_bottom {
        set_px(&mut buf, plot_left, y, grey);
        set_px(&mut buf, plot_right, y, grey);
    }

    // Horizontal ticks (bottom edge)
    let h_ticks = nice_ticks(h_min, h_max, 6);
    for &tv in &h_ticks {
        let frac = (tv - h_min) / (h_max - h_min);
        let px = plot_left + (frac * (ph - 1) as f64) as usize;
        if px >= plot_left && px <= plot_right {
            for dy in 0..tick_len {
                set_px(&mut buf, px, plot_bottom + 1 + dy, black);
            }
            let label = format_tick(tv);
            draw_text_centered(
                &mut buf,
                total_w,
                px as i32,
                (plot_bottom + tick_len + gap) as i32,
                &label,
                black,
                fs,
            );
        }
    }

    // Vertical ticks (left edge) -- v_max at top, v_min at bottom
    for &tv in &v_ticks {
        let frac = (tv - v_min) / (v_max - v_min);
        let py = plot_bottom - (frac * (pv - 1) as f64) as usize;
        if py >= plot_top && py <= plot_bottom {
            for dx in 0..tick_len {
                if plot_left > dx {
                    set_px(&mut buf, plot_left - 1 - dx, py, black);
                }
            }
            let label = format_tick(tv);
            let tw = text_width(&label, fs);
            draw_text(
                &mut buf,
                total_w,
                (plot_left as i32) - (tick_len as i32) - (gap as i32) - (tw as i32),
                py as i32 - (th as i32) / 2,
                &label,
                black,
                fs,
            );
        }
    }

    // Axis labels
    let h_axis_label = format!("{} ({})", h_label, axis_units);
    draw_text_centered(
        &mut buf,
        total_w,
        (plot_left + ph / 2) as i32,
        (plot_bottom + tick_len + gap + th + gap) as i32,
        &h_axis_label,
        black,
        fs,
    );

    let v_axis_label = format!("{} ({})", v_label, axis_units);
    draw_text_vertical(
        &mut buf,
        total_w,
        total_h,
        gap as i32,
        (plot_top + pv / 2) as i32,
        &v_axis_label,
        black,
        fs,
    );

    (buf, total_w as u32, total_h as u32)
}

/// Inlined as a JS template literal in the HTML; used by `JsCsgPlotter` to
/// spawn a Worker pool. Self-contained: duplicates the region/surface eval
/// functions so the worker needs no imports.
const CSG_WORKER_SOURCE: &str = r#"
let cells = null;
function regionContains(region, x, y, z) { return evalExpr(region.expr, x, y, z); }
function evalExpr(expr, x, y, z) {
  if (expr.Halfspace) {
    const hs = expr.Halfspace;
    if (hs.Above) return surfaceEval(hs.Above, x, y, z) > 0;
    if (hs.Below) return surfaceEval(hs.Below, x, y, z) < 0;
  }
  if (expr.Intersection) return evalExpr(expr.Intersection[0], x, y, z) && evalExpr(expr.Intersection[1], x, y, z);
  if (expr.Union) return evalExpr(expr.Union[0], x, y, z) || evalExpr(expr.Union[1], x, y, z);
  if (expr.Complement) return !evalExpr(expr.Complement, x, y, z);
  return false;
}
function surfaceEval(surface, x, y, z) {
  const k = surface.kind;
  if (k.Plane) return k.Plane.a * x + k.Plane.b * y + k.Plane.c * z - k.Plane.d;
  if (k.Sphere) {
    const dx = x - k.Sphere.x0, dy = y - k.Sphere.y0, dz = z - k.Sphere.z0;
    return Math.sqrt(dx*dx + dy*dy + dz*dz) - k.Sphere.radius;
  }
  if (k.Cylinder) {
    const ax = k.Cylinder.axis, or = k.Cylinder.origin;
    const vx = x-or[0], vy = y-or[1], vz = z-or[2];
    const dot = vx*ax[0] + vy*ax[1] + vz*ax[2];
    const dx = vx - dot*ax[0], dy = vy - dot*ax[1], dz = vz - dot*ax[2];
    return Math.sqrt(dx*dx + dy*dy + dz*dz) - k.Cylinder.radius;
  }
  if (k.ZTorus) {
    const t = k.ZTorus;
    const dx = x - t.x0, dy = y - t.y0, dz = z - t.z0;
    const rho = Math.sqrt(dx*dx + dy*dy);
    return (rho - t.a) * (rho - t.a) / (t.c * t.c) + dz * dz / (t.b * t.b) - 1.0;
  }
  return 0;
}
self.onmessage = function(ev) {
  const m = ev.data;
  if (m.type === 'init') { cells = m.cells; return; }
  if (m.type === 'sample') {
    const basis = m.basis, ox = m.ox, oy = m.oy, oz = m.oz;
    const wh = m.wh, wv = m.wv, ph = m.ph, pv = m.pv;
    const rowStart = m.rowStart, rowEnd = m.rowEnd;
    const halfH = wh / 2, halfV = wv / 2;
    const slab = new Int32Array((rowEnd - rowStart) * ph * 2);
    for (let i = rowStart; i < rowEnd; i++) {
      const v = pv > 1 ? halfV - wv * i / (pv - 1) : 0;
      for (let j = 0; j < ph; j++) {
        const h = ph > 1 ? -halfH + wh * j / (ph - 1) : 0;
        let px, py, pz;
        switch(basis) {
          case 'xy': px = ox+h; py = oy+v; pz = oz; break;
          case 'xz': px = ox+h; py = oy; pz = oz+v; break;
          case 'yz': px = ox; py = oy+h; pz = oz+v; break;
          default: px = ox+h; py = oy+v; pz = oz;
        }
        const idx = ((i - rowStart) * ph + j) * 2;
        let found = false;
        for (const cell of cells) {
          if (regionContains(cell.region, px, py, pz)) {
            slab[idx] = cell.cell_id;
            slab[idx+1] = cell.material_id;
            found = true;
            break;
          }
        }
        if (!found) { slab[idx] = -1; slab[idx+1] = -1; }
      }
    }
    self.postMessage({rid: m.rid, rowStart, slab}, [slab.buffer]);
  }
};
"#;

/// Inlined in the HTML; mirrors the Rust samplers in `distribution_spatial`,
/// `distribution_energy`, and `source.rs` so users can change the `n_samples`
/// input box and get more points without re-running Python. Drift risk:
/// adding a new spatial distribution (e.g. `BoxSource`) or energy
/// distribution (e.g. `Muir`) in Rust requires a matching JS branch here.
const SOURCE_SAMPLER_SOURCE: &str = r#"
function _sampleDiscrete(d, r) {
  const n = d.energies.length;
  if (n === 1) return d.energies[0];
  const u = r();
  const bin = Math.min(Math.floor(u * n), n - 1);
  return r() < d.prob_scaled[bin] ? d.energies[bin] : d.energies[d.alias[bin]];
}
function _sampleUniform(u, r) { return u.a + r() * (u.b - u.a); }
function _sampleNormal(nrm, r) {
  while (true) {
    const x = 2 * r() - 1;
    const y = 2 * r() - 1;
    const r2 = x * x + y * y;
    if (r2 > 0 && r2 < 1) {
      const z = Math.sqrt(-2 * Math.log(r2) / r2);
      return nrm.mean_val + nrm.std_dev * z * x;
    }
  }
}
function _sampleUnivariate(u, r) {
  if (u.Discrete) return _sampleDiscrete(u.Discrete, r);
  if (u.Uniform) return _sampleUniform(u.Uniform, r);
  if (u.Normal) return _sampleNormal(u.Normal, r);
  return 0;
}
function _samplePosition(s, r) {
  if (s.Point) return s.Point.xyz;
  if (s.CylindricalRing) {
    const c = s.CylindricalRing;
    const rad = _sampleUnivariate(c.r, r);
    const phi = _sampleUnivariate(c.phi, r);
    const zv = _sampleUnivariate(c.z, r);
    return [c.origin[0] + rad * Math.cos(phi), c.origin[1] + rad * Math.sin(phi), c.origin[2] + zv];
  }
  return [0, 0, 0];
}
function _pickSource(sources, r) {
  if (sources.length === 1) return sources[0];
  let total = 0;
  for (const s of sources) total += s.strength;
  let t = r() * total;
  let cum = 0;
  for (const s of sources) {
    cum += s.strength;
    if (t <= cum) return s;
  }
  return sources[sources.length - 1];
}
function sampleSourcePoints(dist, n) {
  if (!dist || !dist.sources || dist.sources.length === 0) return [];
  const r = Math.random;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const src = _pickSource(dist.sources, r);
    const pos = _samplePosition(src.spatial, r);
    const en = _sampleUnivariate(src.energy, r);
    out[i] = [pos[0], pos[1], pos[2], en];
  }
  return out;
}
"#;

/// Build a fully self-contained HTML string for the interactive viewer.
///
/// # Arguments
/// * `geometry_json` -- serialized CsgGeometry or GeoMesh
/// * `geometry_kind` -- `"csg"` or `"mesh"`
/// * `params` -- initial view parameters
/// * `cell_names` -- map of cell_id -> name
/// * `material_names` -- map of material_id -> name
/// * `colors` -- optional map of ID -> color string
#[allow(clippy::too_many_arguments)]
pub fn build_interactive_html(
    geometry_json: &str,
    geometry_kind: &str,
    _wasm_base64: &str,
    params: &InteractiveViewParams,
    cell_names: &HashMap<i32, String>,
    material_names: &HashMap<i32, String>,
    colors: Option<&HashMap<i32, String>>,
    source_dist_json: Option<&str>,
    presampled: Option<&PresampledGrid>,
    // `true` when the geometry has mesh-filled cells (issue #291). The browser
    // sampler works from `geometry_json`, whose fills carry only an identity
    // fingerprint rather than triangles, so it cannot see a fill body: it would
    // draw the bare CSG frame and hide exactly the geometry the particles see.
    // For such a model the server-side `presampled` raster (which does resolve
    // fills) is authoritative and live re-sampling is disabled.
    has_mesh_fills: bool,
    surface_table: Option<&[crate::surfaces::SurfaceTableEntry]>,
) -> String {
    // Build color map JSON
    let color_map_json = if let Some(cm) = colors {
        let entries: Vec<String> = cm
            .iter()
            .map(|(id, color)| format!("\"{}\":\"{}\"", id, color))
            .collect();
        format!("{{{}}}", entries.join(","))
    } else {
        "{}".to_string()
    };

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

    // Build WASM section for mesh geometries (empty for CSG)
    #[cfg(feature = "mesh")]
    let mesh_wasm_section = if geometry_kind == "mesh" {
        use base64::Engine;
        let wasm_b64 = base64::engine::general_purpose::STANDARD.encode(YAMT_WASM_BYTES);
        // Adapt the wasm-bindgen JS glue for inline use:
        // - Remove ES module exports (export class, export { ... })
        // - Replace class name to avoid conflicts
        // - Add initSync wrapper
        let js_glue = YAMT_WASM_JS
            .replace("export { initSync, __wbg_init as default };", "")
            // Rename all occurrences (class, FinalizationRegistry, Symbol.dispose)
            .replace("WasmMeshPlotter", "YamtWasmMeshPlotter")
            // Remove 'export' keyword from class declaration
            .replace("export class", "class");
        // Remove the async __wbg_init function -- it contains `import.meta` which
        // causes a parse error in non-module <script> tags. We only use initSync.
        let js_glue = if let Some(start) = js_glue.find("async function __wbg_init(") {
            let before = &js_glue[..start];
            // Find the matching closing brace by counting braces
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
            "// --- yamt WASM mesh engine ---\nconst YAMT_WASM_BASE64 = \"{}\";\n{}\nfunction yamtWasmInitSync(bytes) {{ initSync(bytes); }}\n",
            wasm_b64, js_glue
        )
    } else {
        String::new()
    };
    #[cfg(not(feature = "mesh"))]
    let mesh_wasm_section = String::new();

    // Build pre-sampled grid data (base64-encoded little-endian i32 pairs)
    let presampled_section = if let Some(grid) = presampled {
        use base64::Engine;
        let bytes: Vec<u8> = grid.data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        format!(
            "const PRESAMPLED_B64 = \"{}\";\nconst PRESAMPLED_PH = {};\nconst PRESAMPLED_PV = {};",
            b64, grid.ph, grid.pv
        )
    } else {
        "const PRESAMPLED_B64 = null;".to_string()
    };

    // Surface name + identity table -- drives hover tooltips. Each entry
    // carries the full Surface JSON (so viewer.js can evaluate `|f|` at
    // the hover point via its existing `surfaceEval`) plus a display
    // name (already disambiguated by build_surface_table).
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

    // Build the HTML
    format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>YAMC Interactive Geometry Viewer</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #e0e0e0; display: flex; height: 100vh; overflow: hidden; }}
#controls {{ width: 240px; padding: 12px; background: #16213e; overflow-y: auto; flex-shrink: 0; border-right: 1px solid #0f3460; }}
#controls h3 {{ color: #e94560; margin: 12px 0 6px 0; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; }}
#controls h3:first-child {{ margin-top: 0; }}
#controls label {{ display: block; font-size: 12px; margin: 4px 0 2px 0; color: #aaa; }}
#controls input[type=number] {{ width: 100%; padding: 4px 6px; background: #0f3460; border: 1px solid #1a1a4e; color: #e0e0e0; border-radius: 3px; font-size: 12px; }}
#controls input[type=number]:focus {{ outline: none; border-color: #e94560; }}
.btn-group {{ display: flex; gap: 2px; margin: 4px 0; }}
.btn-group button {{ flex: 1; padding: 5px 0; background: #0f3460; border: 1px solid #1a1a4e; color: #e0e0e0; cursor: pointer; font-size: 12px; border-radius: 3px; }}
.btn-group button.active {{ background: #e94560; border-color: #e94560; color: white; }}
.btn-group button:hover:not(.active) {{ background: #1a1a4e; }}
#controls .radio-group {{ margin: 4px 0; }}
#controls .radio-group label {{ display: inline-flex; align-items: center; gap: 4px; margin-right: 8px; cursor: pointer; font-size: 12px; }}
#apply-btn {{ width: 100%; padding: 8px; margin-top: 12px; background: #6b2232; border: none; color: #aaa; cursor: default; border-radius: 3px; font-size: 13px; font-weight: bold; transition: background 0.2s, color 0.2s; }}
#apply-btn.dirty {{ background: #e94560; color: white; cursor: pointer; }}
#apply-btn.dirty:hover {{ background: #c73650; }}
#reset-btn {{ width: 100%; padding: 6px; margin-top: 6px; background: #0f3460; border: 1px solid #1a1a4e; color: #e0e0e0; cursor: pointer; border-radius: 3px; font-size: 12px; }}
#reset-btn:hover {{ background: #1a1a4e; }}
#copy-btn {{ width: 100%; padding: 6px; margin-top: 6px; background: #0f3460; border: 1px solid #1a1a4e; color: #e0e0e0; cursor: pointer; border-radius: 3px; font-size: 12px; }}
#copy-btn:hover {{ background: #1a1a4e; }}
#download-btn {{ width: 100%; padding: 6px; margin-top: 6px; background: #0f3460; border: 1px solid #1a1a4e; color: #e0e0e0; cursor: pointer; border-radius: 3px; font-size: 12px; }}
#download-btn:hover {{ background: #1a1a4e; }}
#main {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; }}
#canvas-container {{ flex: 1; position: relative; overflow: hidden; background: #c0c0c0; }}
#plot-canvas {{ position: absolute; image-rendering: pixelated; cursor: crosshair; }}
#tooltip {{ position: fixed; pointer-events: none; background: rgba(22,33,62,0.95); border: 1px solid #e94560; border-radius: 4px; padding: 8px 12px; font-size: 14px; color: #e0e0e0; display: none; white-space: nowrap; z-index: 10000; }}
#axis-label-h {{ position: absolute; bottom: 0; font-size: 11px; color: #222; z-index: 10; pointer-events: none; }}
#axis-label-v {{ position: absolute; font-size: 11px; color: #222; z-index: 10; pointer-events: none; writing-mode: vertical-rl; transform: rotate(180deg); }}
.tick-h {{ position: absolute; font-size: 10px; color: #222; white-space: nowrap; transform: translateX(-50%); pointer-events: none; z-index: 10; }}
.tick-v {{ position: absolute; font-size: 10px; color: #222; white-space: nowrap; transform: translateY(-50%); text-align: right; pointer-events: none; z-index: 10; }}
#legend {{ padding: 8px 12px; background: #16213e; border-top: 1px solid #0f3460; display: flex; flex-wrap: wrap; gap: 8px; max-height: 80px; overflow-y: auto; }}
.legend-item {{ display: flex; align-items: center; gap: 4px; font-size: 11px; }}
.legend-swatch {{ width: 12px; height: 12px; border-radius: 2px; border: 1px solid #444; }}
#status {{ position: absolute; top: 8px; right: 8px; font-size: 11px; color: #666; z-index: 50; }}
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

  <h3>Origin</h3>
  <label>X <input type="number" id="origin-x" step="any" value="{origin_x}"></label>
  <label>Y <input type="number" id="origin-y" step="any" value="{origin_y}"></label>
  <label>Z <input type="number" id="origin-z" step="any" value="{origin_z}"></label>

  <h3>Width</h3>
  <label>Horizontal <input type="number" id="width-h" step="any" value="{width_h}" min="0.001"></label>
  <label>Vertical <input type="number" id="width-v" step="any" value="{width_v}" min="0.001"></label>

  <h3>Pixels</h3>
  <label>Total <input type="number" id="total-pixels" value="{total_pixels}" min="100" max="4000000" step="1000"></label>
  <div id="pixel-dims" style="font-size:11px;color:#666;margin-top:2px;"></div>

  <h3>Color By</h3>
  <div class="radio-group">
    <label><input type="radio" name="color-by" value="material" {color_by_mat_checked}> Material</label>
    <label><input type="radio" name="color-by" value="cell" {color_by_cell_checked}> Cell</label>
  </div>

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

  <h3>Units</h3>
  <div class="radio-group">
    <label><input type="radio" name="units" value="mm" {units_mm_checked}> mm</label>
    <label><input type="radio" name="units" value="cm" {units_cm_checked}> cm</label>
    <label><input type="radio" name="units" value="m" {units_m_checked}> m</label>
    <label><input type="radio" name="units" value="km" {units_km_checked}> km</label>
  </div>

  <div id="source-section" style="display:{source_section_display};">
    <h3>Source</h3>
    <div class="radio-group">
      <label><input type="checkbox" id="source-show" {source_show_checked}> Show source points</label>
    </div>
    <div id="source-options" style="display:{source_options_display};margin-top:6px;">
      <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
        <label style="font-size:12px;color:#aaa;margin:0;">Color</label>
        <input type="color" id="source-color" value="{source_color_value}" style="width:28px;height:22px;padding:0;border:1px solid #1a1a4e;background:#0f3460;cursor:pointer;border-radius:3px;">
        <label style="font-size:12px;color:#aaa;margin:0;">Size</label>
        <input type="number" id="source-size" value="{source_size_value}" min="1" max="20" step="1" style="width:44px;">
      </div>
      <label style="font-size:12px;color:#aaa;">n_samples <input type="number" id="source-samples" value="{n_samples_value}" min="1" step="1" style="width:90px;margin-left:4px;"></label>
      <label style="font-size:12px;color:#aaa;display:block;margin-top:4px;">plane_tolerance (cm) <input type="number" id="source-tolerance" value="{plane_tolerance_value}" min="0.01" step="0.1" style="width:60px;margin-left:4px;"></label>
    </div>
  </div>

  <button id="apply-btn">Apply Changes</button>
  <button id="reset-btn">Reset View</button>
  <button id="copy-btn">Copy Python Code</button>
  <button id="download-btn">Download PNG</button>
</div>

<div id="main">
  <div id="canvas-container">
    <canvas id="plot-canvas"></canvas>
    <div id="tooltip"></div>
    <div id="tick-container-h"></div>
    <div id="tick-container-v"></div>
    <div id="axis-label-h"></div>
    <div id="axis-label-v"></div>
    <div id="status">Loading WASM...</div>
  </div>
  <div id="legend"></div>
</div>

<script>
// Embedded data
const GEOMETRY_JSON = {geometry_json_literal};
const GEOMETRY_KIND = "{geometry_kind}";
// Mesh-filled CSG cells (issue #291): the browser sampler cannot resolve fill
// bodies, so the server-rendered raster below is the only correct view and
// re-sampling is refused rather than silently drawing the bare CSG frame.
const HAS_MESH_FILLS = {has_mesh_fills};
const CUSTOM_COLORS = {color_map_json};
const CELL_NAMES = {cell_names_json};
const MATERIAL_NAMES = {material_names_json};
const DISCRETE_COLORS = {discrete_colors_json};
// Source distribution: JS samples from this on page load and on every
// change of the "n_samples" input box. Shape: null (no source) or
// {{sources: [{{strength, spatial, energy}}, ...]}}.
const SOURCE_DIST = {source_dist_literal};
{source_sampler_source}
let sourcePoints = null;
let sourcePointsN = 0;
function ensureSourceSamples(n) {{
  if (!SOURCE_DIST) return;
  n = Math.max(1, Math.floor(n));
  if (sourcePoints && sourcePointsN === n) return;
  sourcePoints = sampleSourcePoints(SOURCE_DIST, n);
  sourcePointsN = n;
}}
const BBOX_WIDTHS = [{bbox_wx}, {bbox_wy}, {bbox_wz}];
// Per-surface metadata for hover tooltips: each entry is the full
// Surface JSON (so viewer.js can call surfaceEval on it) plus a display
// name disambiguated by yamc_plot::build_surface_table. Empty list when
// no surfaces are found / no caller passed a table.
const SURFACE_TABLE = {surface_table_json};
{presampled_section}

const INITIAL = {{
  basis: "{basis}",
  originX: {origin_x}, originY: {origin_y}, originZ: {origin_z},
  widthH: {width_h}, widthV: {width_v},
  totalPixels: {total_pixels},
  colorBy: "{color_by}",
  outline: "{outline}",
  outlineColor: [{outline_color_r}, {outline_color_g}, {outline_color_b}],
  outlineWidth: {outline_thickness_value},
  units: "{axis_units}"
}};

// The viewer JS body lives in viewer.js (an editable JS file) and
// is inlined here. It depends on the consts declared above.
const CSG_WORKER_SOURCE = `{csg_worker_source}`;
{viewer_js_body}
{mesh_wasm_section}

init();
</script>
</body>
</html>"##,
        // Template substitutions
        basis = params.basis,
        basis_xy_active = if params.basis == "xy" {
            r#"class="active""#
        } else {
            ""
        },
        basis_xz_active = if params.basis == "xz" {
            r#"class="active""#
        } else {
            ""
        },
        basis_yz_active = if params.basis == "yz" {
            r#"class="active""#
        } else {
            ""
        },
        origin_x = params.origin.0,
        origin_y = params.origin.1,
        origin_z = params.origin.2,
        width_h = params.width.0,
        width_v = params.width.1,
        total_pixels = params.total_pixels,
        color_by = params.color_by,
        color_by_mat_checked = if params.color_by == "material" {
            "checked"
        } else {
            ""
        },
        color_by_cell_checked = if params.color_by == "cell" {
            "checked"
        } else {
            ""
        },
        outline = params.outline.as_deref().unwrap_or("none"),
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
        geometry_json_literal = geometry_json,
        geometry_kind = geometry_kind,
        mesh_wasm_section = mesh_wasm_section,
        presampled_section = presampled_section,
        has_mesh_fills = if has_mesh_fills { "true" } else { "false" },
        csg_worker_source = CSG_WORKER_SOURCE,
        viewer_js_body = VIEWER_JS,
        color_map_json = color_map_json,
        cell_names_json = cell_names_json,
        material_names_json = material_names_json,
        discrete_colors_json =
            serde_json::to_string(DISCRETE_COLORS).unwrap_or_else(|_| "[]".to_string()),
        source_dist_literal = source_dist_json.unwrap_or("null"),
        source_sampler_source = SOURCE_SAMPLER_SOURCE,
        source_section_display = if source_dist_json.is_some() {
            "block"
        } else {
            "none"
        },
        surface_table_json = surface_table_json,
        bbox_wx = params.bbox_widths[0],
        bbox_wy = params.bbox_widths[1],
        bbox_wz = params.bbox_widths[2],
        outline_color_value = params.outline_color,
        outline_color_r = parse_hex_color(&params.outline_color)[0],
        outline_color_g = parse_hex_color(&params.outline_color)[1],
        outline_color_b = parse_hex_color(&params.outline_color)[2],
        outline_thickness_value = params.outline_thickness.max(1),
        source_show_checked = if params.n_samples.is_some() {
            "checked"
        } else {
            ""
        },
        source_options_display = if params.n_samples.is_some() {
            "block"
        } else {
            "none"
        },
        n_samples_value = params.n_samples.unwrap_or(1000),
        plane_tolerance_value = params.plane_tolerance,
        source_color_value = params.source_color,
        source_size_value = params.source_size,
    )
}

pub fn escape_js(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}
