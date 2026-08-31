//! Compiled reductions for mesh-tally VTK-HDF export.
//!
//! The Python `mesh_tally_to_vtkhdf` writer keeps the h5py I/O and dataset
//! naming, but the numeric reduction (summing over parent-nuclide and energy
//! bins, volume normalization, scaling, and relative error) runs here. A whole
//! score block is reduced across all requested parent/energy combinations in a
//! single call, so the flat data crosses the Python<->Rust boundary once.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

/// Reduce one score block of a mesh tally over parent/energy bins.
///
/// The block is laid out `[parent][energy][mesh]` (mesh fastest), matching the
/// tally's flat result order. For every `(parent_index, energy_index)` pair in
/// the cartesian product of `parent_iter` x `energy_iter` (parent outer, energy
/// inner), the mesh-length mean is summed over the collapsed axes and the
/// standard deviation is combined in quadrature (`sqrt(sum(std^2))`). A `None`
/// index means "sum over that whole axis"; a `Some(i)` selects a single bin.
///
/// When `vol` is given (length `n_mesh`), mean and std are divided by the
/// per-voxel volume; `scaling` then multiplies both. Relative error is
/// `std/mean` where mean is non-zero, else 0 (both volume and scaling cancel in
/// the ratio, matching the pure-numpy path).
///
/// Returns one `(mean, std, relative_error)` triple per combination, in the
/// same parent-outer/energy-inner order the caller iterates.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (block_mean, block_std, n_parent, n_energy, n_mesh, parent_iter, energy_iter, vol=None, scaling=1.0))]
#[allow(clippy::too_many_arguments)]
pub fn reduce_mesh_tally_block(
    block_mean: Vec<f64>,
    block_std: Vec<f64>,
    n_parent: usize,
    n_energy: usize,
    n_mesh: usize,
    parent_iter: Vec<Option<usize>>,
    energy_iter: Vec<Option<usize>>,
    vol: Option<Vec<f64>>,
    scaling: f64,
) -> PyResult<Vec<(Vec<f64>, Vec<f64>, Vec<f64>)>> {
    let expected = n_parent * n_energy * n_mesh;
    if block_mean.len() != expected || block_std.len() != expected {
        return Err(PyValueError::new_err(format!(
            "block length mismatch: expected n_parent*n_energy*n_mesh = {expected}, \
             got mean={} std={}",
            block_mean.len(),
            block_std.len()
        )));
    }
    if let Some(v) = &vol {
        if v.len() != n_mesh {
            return Err(PyValueError::new_err(format!(
                "vol length {} does not match n_mesh {n_mesh}",
                v.len()
            )));
        }
    }

    let mut out = Vec::with_capacity(parent_iter.len() * energy_iter.len());
    for &p_idx in &parent_iter {
        if let Some(p) = p_idx {
            if p >= n_parent {
                return Err(PyValueError::new_err(format!(
                    "parent index {p} out of range (n_parent={n_parent})"
                )));
            }
        }
        for &e_idx in &energy_iter {
            if let Some(e) = e_idx {
                if e >= n_energy {
                    return Err(PyValueError::new_err(format!(
                        "energy index {e} out of range (n_energy={n_energy})"
                    )));
                }
            }
            let p_range = match p_idx {
                Some(p) => p..p + 1,
                None => 0..n_parent,
            };

            let mut mean = vec![0.0f64; n_mesh];
            let mut var = vec![0.0f64; n_mesh];
            for p in p_range {
                let e_range = match e_idx {
                    Some(e) => e..e + 1,
                    None => 0..n_energy,
                };
                for e in e_range {
                    let base = (p * n_energy + e) * n_mesh;
                    for k in 0..n_mesh {
                        mean[k] += block_mean[base + k];
                        let s = block_std[base + k];
                        var[k] += s * s;
                    }
                }
            }

            let mut std: Vec<f64> = var.iter().map(|v| v.sqrt()).collect();
            if let Some(v) = &vol {
                for k in 0..n_mesh {
                    mean[k] /= v[k];
                    std[k] /= v[k];
                }
            }
            if scaling != 1.0 {
                for k in 0..n_mesh {
                    mean[k] *= scaling;
                    std[k] *= scaling;
                }
            }
            let rel: Vec<f64> = (0..n_mesh)
                .map(|k| {
                    if mean[k] != 0.0 {
                        std[k] / mean[k]
                    } else {
                        0.0
                    }
                })
                .collect();
            out.push((mean, std, rel));
        }
    }
    Ok(out)
}
