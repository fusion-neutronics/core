use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotParams {
    pub origin: (f64, f64, f64),
    pub width: (f64, f64),
    pub pixels: (usize, usize),
    pub basis: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotSample {
    pub cell_id: i32,
    pub material_id: i32,
    pub hover_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotGrid {
    pub cell_ids: Vec<Vec<i32>>,
    pub material_ids: Vec<Vec<i32>>,
    pub hover_text: Vec<Vec<String>>,
}

pub fn sample_plot_grid<F>(params: &PlotParams, mut sample_fn: F) -> Result<PlotGrid, String>
where
    F: FnMut((f64, f64, f64)) -> PlotSample,
{
    let (pixels_h, pixels_v) = params.pixels;
    let (width_h, width_v) = params.width;
    let origin = params.origin;

    let half_width_h = width_h / 2.0;
    let half_width_v = width_v / 2.0;

    let h_vals: Vec<f64> = (0..pixels_h)
        .map(|i| {
            if pixels_h > 1 {
                -half_width_h + width_h * (i as f64) / ((pixels_h - 1) as f64)
            } else {
                0.0
            }
        })
        .collect();

    let v_vals: Vec<f64> = (0..pixels_v)
        .map(|i| {
            if pixels_v > 1 {
                -half_width_v + width_v * (i as f64) / ((pixels_v - 1) as f64)
            } else {
                0.0
            }
        })
        .collect();

    let mut cell_ids = vec![vec![-1i32; pixels_h]; pixels_v];
    let mut material_ids = vec![vec![-1i32; pixels_h]; pixels_v];
    let mut hover_text = vec![vec![String::new(); pixels_h]; pixels_v];

    for (i, &v) in v_vals.iter().enumerate() {
        for (j, &h) in h_vals.iter().enumerate() {
            let point = match params.basis.as_str() {
                "xy" => (origin.0 + h, origin.1 + v, origin.2),
                "xz" => (origin.0 + h, origin.1, origin.2 + v),
                "yz" => (origin.0, origin.1 + h, origin.2 + v),
                _ => return Err("basis must be 'xy', 'xz', or 'yz'".to_string()),
            };

            let sample = sample_fn(point);
            cell_ids[i][j] = sample.cell_id;
            material_ids[i][j] = sample.material_id;
            hover_text[i][j] = sample.hover_text;
        }
    }

    Ok(PlotGrid {
        cell_ids,
        material_ids,
        hover_text,
    })
}
