mod tests {
    use yamc::geo::*;

    #[test]
    fn sample_plot_grid_shapes_and_values() {
        let params = PlotParams {
            origin: (0.0, 0.0, 0.0),
            width: (2.0, 2.0),
            pixels: (3, 2),
            basis: "xy".to_string(),
        };

        let grid = sample_plot_grid(&params, |point| PlotSample {
            cell_id: if point.0 >= 0.0 { 1 } else { -1 },
            material_id: 42,
            hover_text: format!("x={:.1}", point.0),
        })
        .expect("plot grid should succeed");

        assert_eq!(grid.cell_ids.len(), 2);
        assert_eq!(grid.cell_ids[0].len(), 3);
        assert_eq!(grid.material_ids.len(), 2);
        assert_eq!(grid.hover_text.len(), 2);
    }

    #[test]
    fn sample_plot_grid_invalid_basis() {
        let params = PlotParams {
            origin: (0.0, 0.0, 0.0),
            width: (1.0, 1.0),
            pixels: (2, 2),
            basis: "bad".to_string(),
        };

        let err = match sample_plot_grid(&params, |_point| PlotSample {
            cell_id: 1,
            material_id: 1,
            hover_text: "x".to_string(),
        }) {
            Ok(_) => panic!("invalid basis should error"),
            Err(err) => err,
        };

        assert!(err.contains("basis must be"));
    }
}
