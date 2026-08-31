/// CRAM (Chebyshev Rational Approximation Method) solvers.
///
/// Solves the matrix exponential: N(t) = exp(A*t) * N(0)
/// where A is the transmutation matrix containing decay constants and reaction rates.
///
/// Provides both CRAM16 (order 16, 8 conjugate pairs) and CRAM48 (order 48,
/// 24 conjugate pairs). CRAM48 is the default and recommended solver.
///
/// Both dense and sparse variants are provided. The sparse variants use
/// `faer`'s sparse LU solver with symbolic factorization reuse across poles.
///
/// Based on:
/// - Pusa, M. (2010). "Rational Approximations to the Matrix Exponential
///   in Burnup Calculations"
/// - Pusa, M. (2015). "Higher-Order Chebyshev Rational Approximation Method
///   and Application to Burnup Equations" (Nucl. Sci. Eng., 182:3, 297-318)
use num_complex::Complex64;

use faer::c64;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::sparse::linalg::lu as sparse_lu;
use faer::sparse::{SparseColMatRef, SymbolicSparseColMat};
use faer::{Conj, Mat, Par};

/// Solve the matrix exponential using CRAM with the given coefficients.
///
/// Shared implementation for both CRAM16 and CRAM48.
fn cram_solve(
    a: &[f64],
    n: usize,
    n0: &[f64],
    dt: f64,
    alpha: &[Complex64],
    theta: &[Complex64],
    alpha0: f64,
) -> Result<Vec<f64>, String> {
    if dt == 0.0 {
        return Ok(n0.to_vec());
    }

    let mut y = n0.to_vec();
    let a_dt: Vec<f64> = a.iter().map(|v| v * dt).collect();

    for (alpha_i, theta_i) in alpha.iter().zip(theta.iter()) {
        let mut mat = vec![Complex64::new(0.0, 0.0); n * n];
        for row in 0..n {
            for col in 0..n {
                mat[row * n + col] = Complex64::new(a_dt[row * n + col], 0.0);
            }
            mat[row * n + row] -= *theta_i;
        }

        let b: Vec<Complex64> = y.iter().map(|v| Complex64::new(*v, 0.0)).collect();
        let x = solve_complex(&mat, n, &b)?;
        for i in 0..n {
            let contrib = *alpha_i * x[i];
            y[i] += 2.0 * contrib.re;
        }
    }

    for val in &mut y {
        *val *= alpha0;
    }

    Ok(y)
}

/// Solve the matrix exponential N(t) = exp(A*dt) * N(0) using CRAM48.
///
/// This is the default and recommended solver. Uses 24 conjugate pairs
/// (48th order) for high accuracy even with stiff transmutation problems.
///
/// # Arguments
/// * `a` - Transmutation matrix (n x n, row-major, flattened)
/// * `n` - Matrix dimension (number of nuclides)
/// * `n0` - Initial nuclide densities [atoms/barn-cm]
/// * `dt` - Time step [s]
///
/// # Returns
/// Final nuclide densities after time dt
#[allow(clippy::excessive_precision)]
pub fn cram48(a: &[f64], n: usize, n0: &[f64], dt: f64) -> Result<Vec<f64>, String> {
    let alpha: [Complex64; 24] = [
        Complex64::new(6.387380733878774e+2, -6.743912502859256e+2),
        Complex64::new(1.909896179065730e+2, -3.973203432721332e+2),
        Complex64::new(4.236195226571914e+2, -2.041233768918671e+3),
        Complex64::new(4.645770595258726e+2, -1.652917287299683e+3),
        Complex64::new(7.765163276752433e+2, -1.783617639907328e+4),
        Complex64::new(1.907115136768522e+3, -5.887068595142284e+4),
        Complex64::new(2.909892685603256e+3, -9.953255345514560e+3),
        Complex64::new(1.944772206620450e+2, -1.427131226068449e+3),
        Complex64::new(1.382799786972332e+5, -3.256885197214938e+6),
        Complex64::new(5.628442079602433e+3, -2.924284515884309e+4),
        Complex64::new(2.151681283794220e+2, -1.121774011188224e+3),
        Complex64::new(1.324720240514420e+3, -6.370088443140973e+4),
        Complex64::new(1.617548476343347e+4, -1.008798413156542e+6),
        Complex64::new(1.112729040439685e+2, -8.837109731680418e+1),
        Complex64::new(1.074624783191125e+2, -1.457246116408180e+2),
        Complex64::new(8.835727765158191e+1, -6.388286188419360e+1),
        Complex64::new(9.354078136054179e+1, -2.195424319460237e+2),
        Complex64::new(9.418142823531573e+1, -6.719055740098035e+2),
        Complex64::new(1.040012390717851e+2, -1.693747595553868e+2),
        Complex64::new(6.861882624343235e+1, -1.177598523430493e+1),
        Complex64::new(8.766654491283722e+1, -4.596464999363902e+3),
        Complex64::new(1.056007619389650e+2, -1.738294585524067e+3),
        Complex64::new(7.738987569039419e+1, -4.311715386228984e+1),
        Complex64::new(1.041366366475571e+2, -2.777743732451969e+2),
    ];
    let theta: [Complex64; 24] = [
        Complex64::new(-4.465731934165702e+1, 6.233225190695437e+1),
        Complex64::new(-5.284616241568964e+0, 4.057499381311059e+1),
        Complex64::new(-8.867715667624458e+0, 4.325515754166724e+1),
        Complex64::new(3.493013124279215e+0, 3.281615453173585e+1),
        Complex64::new(1.564102508858634e+1, 1.558061616372237e+1),
        Complex64::new(1.742097597385893e+1, 1.076629305714420e+1),
        Complex64::new(-2.834466755180654e+1, 5.492841024648724e+1),
        Complex64::new(1.661569367939544e+1, 1.316994930024688e+1),
        Complex64::new(8.011836167974721e+0, 2.780232111309410e+1),
        Complex64::new(-2.056267541998229e+0, 3.794824788914354e+1),
        Complex64::new(1.449208170441839e+1, 1.799988210051809e+1),
        Complex64::new(1.853807176907916e+1, 5.974332563100539e+0),
        Complex64::new(9.932562704505182e+0, 2.532823409972962e+1),
        Complex64::new(-2.244223871767187e+1, 5.179633600312162e+1),
        Complex64::new(8.590014121680897e-1, 3.536456194294350e+1),
        Complex64::new(-1.286192925744479e+1, 4.600304902833652e+1),
        Complex64::new(1.164596909542055e+1, 2.287153304140217e+1),
        Complex64::new(1.806076684783089e+1, 8.368200580099821e+0),
        Complex64::new(5.870672154659249e+0, 3.029700159040121e+1),
        Complex64::new(-3.542938819659747e+1, 5.834381701800013e+1),
        Complex64::new(1.901323489060250e+1, 1.194282058271408e+0),
        Complex64::new(1.885508331552577e+1, 3.583428564427879e+0),
        Complex64::new(-1.734689708174982e+1, 4.883941101108207e+1),
        Complex64::new(1.316284237125190e+1, 2.042951874827759e+1),
    ];
    let alpha0 = 2.258038182743983e-47_f64;

    cram_solve(a, n, n0, dt, &alpha, &theta, alpha0)
}

/// Solve a complex linear system Ax = b using Gaussian elimination with partial pivoting.
fn solve_complex(
    matrix: &[Complex64],
    n: usize,
    b: &[Complex64],
) -> Result<Vec<Complex64>, String> {
    let mut a = matrix.to_vec();
    let mut rhs = b.to_vec();

    for k in 0..n {
        let mut pivot = k;
        let mut max = a[k * n + k].norm();
        for i in (k + 1)..n {
            let val = a[i * n + k].norm();
            if val > max {
                max = val;
                pivot = i;
            }
        }
        if max == 0.0 {
            return Err("singular matrix".to_string());
        }
        if pivot != k {
            for j in 0..n {
                a.swap(k * n + j, pivot * n + j);
            }
            rhs.swap(k, pivot);
        }

        let pivot_val = a[k * n + k];
        let rhs_pivot = rhs[k];
        for i in (k + 1)..n {
            let factor = a[i * n + k] / pivot_val;
            a[i * n + k] = Complex64::new(0.0, 0.0);
            for j in (k + 1)..n {
                let pivot_row_val = a[k * n + j];
                a[i * n + j] -= factor * pivot_row_val;
            }
            rhs[i] -= factor * rhs_pivot;
        }
    }

    let mut x = vec![Complex64::new(0.0, 0.0); n];
    for i in (0..n).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..n {
            sum -= a[i * n + j] * x[j];
        }
        x[i] = sum / a[i * n + i];
    }

    Ok(x)
}

/// Sparse CRAM solver core using faer's sparse LU.
///
/// Key optimization: symbolic factorization is computed once and reused
/// for all poles, since the sparsity pattern (A*dt - theta*I) is
/// identical across poles.
fn cram_solve_sparse(
    triplets: &[(usize, usize, f64)],
    n: usize,
    n0: &[f64],
    dt: f64,
    alpha: &[Complex64],
    theta: &[Complex64],
    alpha0: f64,
) -> Result<Vec<f64>, String> {
    if dt == 0.0 {
        return Ok(n0.to_vec());
    }

    // Build augmented sparsity pattern: A's nonzeros ∪ full diagonal.
    // We need the full diagonal because (A*dt - theta*I) always has
    // nonzero diagonal, even if A's diagonal is zero (stable nuclides).
    //
    // Step 1: Collect unique (row, col) positions including all diagonal entries.
    let mut positions: Vec<(usize, usize)> = Vec::with_capacity(triplets.len() + n);
    for &(r, c, _) in triplets {
        positions.push((r, c));
    }
    for i in 0..n {
        positions.push((i, i));
    }
    // Sort by (col, row) for CSC and deduplicate
    positions.sort_unstable_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
    positions.dedup();
    let nnz = positions.len();

    // Step 2: Build CSC index arrays from sorted unique positions.
    let mut col_ptr: Vec<usize> = vec![0; n + 1];
    let mut row_idx: Vec<usize> = Vec::with_capacity(nnz);
    for &(r, c) in &positions {
        col_ptr[c + 1] += 1;
        row_idx.push(r);
    }
    // Cumulative sum for col_ptr
    for j in 0..n {
        col_ptr[j + 1] += col_ptr[j];
    }

    // Record diagonal indices via binary search on sorted positions.
    let diag_idx: Vec<usize> = (0..n)
        .map(|i| {
            positions
                .binary_search_by(|p| p.1.cmp(&i).then(p.0.cmp(&i)))
                .expect("diagonal must be present")
        })
        .collect();

    // Step 3: Build symbolic sparsity pattern and do symbolic LU factorization (once).
    let symbolic_mat =
        SymbolicSparseColMat::<usize>::new_checked(n, n, col_ptr.clone(), None, row_idx.clone());
    let symbolic_lu = sparse_lu::factorize_symbolic_lu(symbolic_mat.as_ref(), Default::default())
        .map_err(|e| format!("Sparse LU symbolic factorization error: {e:?}"))?;

    // Step 4: Allocate reusable buffers.
    let par = Par::Seq;
    let mut numeric_lu = sparse_lu::NumericLu::<usize, c64>::new();
    let factor_req = symbolic_lu.factorize_numeric_lu_scratch::<c64>(par, Default::default());
    let solve_req = symbolic_lu.solve_in_place_scratch::<c64>(1, par);
    let total_req = factor_req.or(solve_req);
    let mut mem_buf =
        MemBuffer::try_new(total_req).map_err(|_| "Failed to allocate sparse LU workspace")?;

    // Pre-compute A*dt values for each CSC position using binary search.
    let mut base_vals = vec![0.0f64; nnz];
    for &(r, c, v) in triplets {
        // Triplets may have duplicates -- accumulate.
        // Binary search on sorted (col, row) positions.
        let idx = positions
            .binary_search_by(|p| p.1.cmp(&c).then(p.0.cmp(&r)))
            .expect("triplet position must exist in positions");
        base_vals[idx] += v * dt;
    }

    // Step 5: For each pole, fill values and solve.
    // Pre-allocate buffers reused across all poles.
    let mut y = n0.to_vec();
    let mut vals = vec![c64::new(0.0, 0.0); nnz];
    let mut rhs = Mat::<c64>::zeros(n, 1);

    for (alpha_i, theta_i) in alpha.iter().zip(theta.iter()) {
        // Fill complex values: A*dt for all entries, then subtract theta on diagonal.
        let theta_c = c64::new(theta_i.re, theta_i.im);
        for (v, &bv) in vals.iter_mut().zip(base_vals.iter()) {
            *v = c64::new(bv, 0.0);
        }
        for &di in &diag_idx {
            vals[di] -= theta_c;
        }

        // Create SparseColMatRef from the symbolic pattern + values.
        let sym_ref = symbolic_mat.as_ref();
        let mat_ref = SparseColMatRef::<usize, c64>::new(sym_ref, &vals);

        // Numeric LU factorization (reuses symbolic + numeric buffer).
        let lu_ref = symbolic_lu
            .factorize_numeric_lu(
                &mut numeric_lu,
                mat_ref,
                par,
                MemStack::new(&mut mem_buf),
                Default::default(),
            )
            .map_err(|e| format!("Sparse LU numeric factorization error: {e:?}"))?;

        // Fill RHS and solve in place.
        for i in 0..n {
            rhs[(i, 0)] = c64::new(y[i], 0.0);
        }
        lu_ref.solve_in_place_with_conj(Conj::No, rhs.as_mut(), par, MemStack::new(&mut mem_buf));

        // Accumulate: y += 2 * Re(alpha * x)
        let alpha_c = c64::new(alpha_i.re, alpha_i.im);
        for i in 0..n {
            let contrib = alpha_c * rhs[(i, 0)];
            y[i] += 2.0 * contrib.re;
        }
    }

    // Scale by alpha0
    for val in &mut y {
        *val *= alpha0;
    }

    Ok(y)
}

/// Solve the matrix exponential N(t) = exp(A*dt) * N(0) using sparse CRAM48.
///
/// Uses faer's sparse LU solver. Symbolic factorization is computed once
/// and reused for all 24 poles. Much faster than dense CRAM48 for sparse
/// transmutation matrices (which are typically 96-97% zeros).
///
/// # Arguments
/// * `triplets` - Transmutation matrix in COO format: `(row, col, value)` entries
/// * `n` - Matrix dimension (number of nuclides)
/// * `n0` - Initial nuclide densities [atoms/barn-cm]
/// * `dt` - Time step [s]
///
/// # Returns
/// Final nuclide densities after time dt
#[allow(clippy::excessive_precision)]
pub fn cram48_sparse(
    triplets: &[(usize, usize, f64)],
    n: usize,
    n0: &[f64],
    dt: f64,
) -> Result<Vec<f64>, String> {
    let alpha: [Complex64; 24] = [
        Complex64::new(6.387380733878774e+2, -6.743912502859256e+2),
        Complex64::new(1.909896179065730e+2, -3.973203432721332e+2),
        Complex64::new(4.236195226571914e+2, -2.041233768918671e+3),
        Complex64::new(4.645770595258726e+2, -1.652917287299683e+3),
        Complex64::new(7.765163276752433e+2, -1.783617639907328e+4),
        Complex64::new(1.907115136768522e+3, -5.887068595142284e+4),
        Complex64::new(2.909892685603256e+3, -9.953255345514560e+3),
        Complex64::new(1.944772206620450e+2, -1.427131226068449e+3),
        Complex64::new(1.382799786972332e+5, -3.256885197214938e+6),
        Complex64::new(5.628442079602433e+3, -2.924284515884309e+4),
        Complex64::new(2.151681283794220e+2, -1.121774011188224e+3),
        Complex64::new(1.324720240514420e+3, -6.370088443140973e+4),
        Complex64::new(1.617548476343347e+4, -1.008798413156542e+6),
        Complex64::new(1.112729040439685e+2, -8.837109731680418e+1),
        Complex64::new(1.074624783191125e+2, -1.457246116408180e+2),
        Complex64::new(8.835727765158191e+1, -6.388286188419360e+1),
        Complex64::new(9.354078136054179e+1, -2.195424319460237e+2),
        Complex64::new(9.418142823531573e+1, -6.719055740098035e+2),
        Complex64::new(1.040012390717851e+2, -1.693747595553868e+2),
        Complex64::new(6.861882624343235e+1, -1.177598523430493e+1),
        Complex64::new(8.766654491283722e+1, -4.596464999363902e+3),
        Complex64::new(1.056007619389650e+2, -1.738294585524067e+3),
        Complex64::new(7.738987569039419e+1, -4.311715386228984e+1),
        Complex64::new(1.041366366475571e+2, -2.777743732451969e+2),
    ];
    let theta: [Complex64; 24] = [
        Complex64::new(-4.465731934165702e+1, 6.233225190695437e+1),
        Complex64::new(-5.284616241568964e+0, 4.057499381311059e+1),
        Complex64::new(-8.867715667624458e+0, 4.325515754166724e+1),
        Complex64::new(3.493013124279215e+0, 3.281615453173585e+1),
        Complex64::new(1.564102508858634e+1, 1.558061616372237e+1),
        Complex64::new(1.742097597385893e+1, 1.076629305714420e+1),
        Complex64::new(-2.834466755180654e+1, 5.492841024648724e+1),
        Complex64::new(1.661569367939544e+1, 1.316994930024688e+1),
        Complex64::new(8.011836167974721e+0, 2.780232111309410e+1),
        Complex64::new(-2.056267541998229e+0, 3.794824788914354e+1),
        Complex64::new(1.449208170441839e+1, 1.799988210051809e+1),
        Complex64::new(1.853807176907916e+1, 5.974332563100539e+0),
        Complex64::new(9.932562704505182e+0, 2.532823409972962e+1),
        Complex64::new(-2.244223871767187e+1, 5.179633600312162e+1),
        Complex64::new(8.590014121680897e-1, 3.536456194294350e+1),
        Complex64::new(-1.286192925744479e+1, 4.600304902833652e+1),
        Complex64::new(1.164596909542055e+1, 2.287153304140217e+1),
        Complex64::new(1.806076684783089e+1, 8.368200580099821e+0),
        Complex64::new(5.870672154659249e+0, 3.029700159040121e+1),
        Complex64::new(-3.542938819659747e+1, 5.834381701800013e+1),
        Complex64::new(1.901323489060250e+1, 1.194282058271408e+0),
        Complex64::new(1.885508331552577e+1, 3.583428564427879e+0),
        Complex64::new(-1.734689708174982e+1, 4.883941101108207e+1),
        Complex64::new(1.316284237125190e+1, 2.042951874827759e+1),
    ];
    let alpha0 = 2.258038182743983e-47_f64;

    cram_solve_sparse(triplets, n, n0, dt, &alpha, &theta, alpha0)
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- CRAM48 tests ---

    #[test]
    fn test_cram48_zero_dt() {
        let a = vec![0.0; 4];
        let n0 = vec![1.0, 2.0];
        let result = cram48(&a, 2, &n0, 0.0).unwrap();
        assert_eq!(result, n0);
    }

    #[test]
    fn test_cram48_pure_decay() {
        let half_life = 3600.0;
        let lambda = std::f64::consts::LN_2 / half_life;
        let a = vec![-lambda];
        let n0 = vec![1.0e20];

        let result = cram48(&a, 1, &n0, half_life).unwrap();
        let expected = 0.5 * n0[0];
        let rel_error = (result[0] - expected).abs() / expected;
        assert!(
            rel_error < 1e-14,
            "CRAM48 relative error {rel_error} too large"
        );
    }

    #[test]
    fn test_cram48_decay_chain() {
        let half_life_a = 3600.0;
        let lambda_a = std::f64::consts::LN_2 / half_life_a;
        let a = vec![-lambda_a, 0.0, lambda_a, 0.0];
        let n0 = vec![1.0e20, 0.0];

        let result = cram48(&a, 2, &n0, half_life_a).unwrap();

        let expected_a = 0.5 * n0[0];
        let rel_error_a = (result[0] - expected_a).abs() / expected_a;
        assert!(
            rel_error_a < 1e-14,
            "CRAM48 A relative error {rel_error_a} too large"
        );

        let expected_b = n0[0] - result[0];
        let rel_error_b = (result[1] - expected_b).abs() / expected_b;
        assert!(
            rel_error_b < 1e-14,
            "CRAM48 B relative error {rel_error_b} too large"
        );
    }

    #[test]
    fn test_cram48_stiff_accuracy() {
        // CRAM48 should match the analytic solution on a stiff problem.
        let half_life = 1.0; // very short half-life (stiff problem)
        let lambda = std::f64::consts::LN_2 / half_life;
        let a = vec![-lambda];
        let n0 = vec![1.0e20];
        let dt = 10.0 * half_life; // 10 half-lives

        let result_48 = cram48(&a, 1, &n0, dt).unwrap();
        let expected = n0[0] * (-lambda * dt).exp();

        let err_48 = (result_48[0] - expected).abs() / expected;

        assert!(
            err_48 < 1e-6,
            "CRAM48 relative error ({err_48:.2e}) too large on stiff problem"
        );
    }

    // --- Sparse vs Dense comparison tests ---

    /// Convert a dense row-major matrix to COO triplets (only non-zero entries).
    fn dense_to_triplets(a: &[f64], n: usize) -> Vec<(usize, usize, f64)> {
        let mut triplets = Vec::new();
        for row in 0..n {
            for col in 0..n {
                let val = a[row * n + col];
                if val != 0.0 {
                    triplets.push((row, col, val));
                }
            }
        }
        triplets
    }

    #[test]
    fn test_cram48_sparse_vs_dense_pure_decay() {
        let half_life = 3600.0;
        let lambda = std::f64::consts::LN_2 / half_life;
        let a = vec![-lambda];
        let n0 = vec![1.0e20];

        let dense_result = cram48(&a, 1, &n0, half_life).unwrap();
        let triplets = dense_to_triplets(&a, 1);
        let sparse_result = cram48_sparse(&triplets, 1, &n0, half_life).unwrap();

        let rel_error = (sparse_result[0] - dense_result[0]).abs() / dense_result[0];
        assert!(
            rel_error < 1e-12,
            "Sparse vs dense relative error {rel_error:.2e} too large"
        );
    }

    #[test]
    fn test_cram48_sparse_vs_dense_decay_chain() {
        // A -> B decay chain
        let half_life_a = 3600.0;
        let lambda_a = std::f64::consts::LN_2 / half_life_a;
        let a = vec![-lambda_a, 0.0, lambda_a, 0.0];
        let n0 = vec![1.0e20, 0.0];

        let dense_result = cram48(&a, 2, &n0, half_life_a).unwrap();
        let triplets = dense_to_triplets(&a, 2);
        let sparse_result = cram48_sparse(&triplets, 2, &n0, half_life_a).unwrap();

        for i in 0..2 {
            if dense_result[i].abs() > 1e-30 {
                let rel_error = (sparse_result[i] - dense_result[i]).abs() / dense_result[i].abs();
                assert!(
                    rel_error < 1e-12,
                    "Nuclide {i}: sparse vs dense relative error {rel_error:.2e} too large"
                );
            }
        }
    }

    #[test]
    fn test_cram48_sparse_vs_dense_larger_system() {
        // 5-nuclide chain: A -> B -> C -> D -> E with different half-lives
        let n = 5;
        let half_lives = [3600.0, 7200.0, 1800.0, 86400.0]; // A, B, C, D decay; E is stable
        let lambdas: Vec<f64> = half_lives
            .iter()
            .map(|t| std::f64::consts::LN_2 / t)
            .collect();

        // Build dense matrix
        let mut a = vec![0.0; n * n];
        for i in 0..4 {
            a[i * n + i] = -lambdas[i]; // diagonal loss
            a[(i + 1) * n + i] = lambdas[i]; // production in next nuclide
        }

        let n0 = vec![1.0e20, 0.0, 0.0, 0.0, 0.0];
        let dt = 7200.0; // 2 hours

        let dense_result = cram48(&a, n, &n0, dt).unwrap();
        let triplets = dense_to_triplets(&a, n);
        let sparse_result = cram48_sparse(&triplets, n, &n0, dt).unwrap();

        for i in 0..n {
            if dense_result[i].abs() > 1e-30 {
                let rel_error = (sparse_result[i] - dense_result[i]).abs() / dense_result[i].abs();
                assert!(
                    rel_error < 1e-12,
                    "Nuclide {i}: sparse vs dense relative error {rel_error:.2e} too large"
                );
            }
        }
    }

    #[test]
    fn test_cram48_sparse_zero_dt() {
        let triplets = vec![(0, 0, -1.0e-4)];
        let n0 = vec![1.0, 2.0];
        let result = cram48_sparse(&triplets, 2, &n0, 0.0).unwrap();
        assert_eq!(result, n0);
    }
}
