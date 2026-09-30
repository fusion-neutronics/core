/// CRAM (Chebyshev Rational Approximation Method) solvers.
///
/// Solves the matrix exponential: N(t) = exp(A*t) * N(0)
/// where A is the transmutation matrix containing decay constants and reaction rates.
///
/// Provides both CRAM16 (order 16, 8 conjugate pairs) and CRAM48 (order 48,
/// 24 conjugate pairs). CRAM48 is the default and recommended solver.
///
/// Both dense and sparse variants are provided. The sparse variants solve
/// block by block over the production graph, with `faer`'s sparse LU (its
/// symbolic factorization reused across poles) for each block with a cycle.
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

/// A diagonal block of the transmutation matrix in block lower triangular
/// form.
enum Block {
    /// A nuclide on no production cycle. Once its feeders are solved its row
    /// of `(A dt - theta I) x = y` has one unknown.
    Single { row: usize, diag: f64 },
    /// Nuclides that feed one another round a cycle, factored together.
    Coupled(Box<CoupledBlock>),
}

/// A strongly connected set of nuclides and the sparse LU of its own
/// submatrix, the symbolic part computed once and reused for every pole.
struct CoupledBlock {
    rows: Vec<usize>,
    symbolic_mat: SymbolicSparseColMat<usize>,
    symbolic_lu: sparse_lu::SymbolicLu<usize>,
    numeric_lu: sparse_lu::NumericLu<usize, c64>,
    mem_buf: MemBuffer,
    base_vals: Vec<f64>,
    diag_idx: Vec<usize>,
    vals: Vec<c64>,
    rhs: Mat<c64>,
}

impl CoupledBlock {
    /// `local` holds the block's own entries of `A * dt` in block-local
    /// indices, duplicates allowed.
    fn new(rows: Vec<usize>, local: &[(usize, usize, f64)]) -> Result<Self, String> {
        let n = rows.len();
        // The pattern is A's nonzeros and the full diagonal, since
        // (A*dt - theta*I) has a nonzero diagonal even where A has none.
        let mut positions: Vec<(usize, usize)> = Vec::with_capacity(local.len() + n);
        positions.extend(local.iter().map(|&(r, c, _)| (r, c)));
        positions.extend((0..n).map(|i| (i, i)));
        // Sorted by (col, row) for CSC.
        positions.sort_unstable_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
        positions.dedup();
        let find = |r: usize, c: usize| {
            positions
                .binary_search_by(|p| p.1.cmp(&c).then(p.0.cmp(&r)))
                .expect("position is in the pattern")
        };

        let mut col_ptr: Vec<usize> = vec![0; n + 1];
        let mut row_idx: Vec<usize> = Vec::with_capacity(positions.len());
        for &(r, c) in &positions {
            col_ptr[c + 1] += 1;
            row_idx.push(r);
        }
        for j in 0..n {
            col_ptr[j + 1] += col_ptr[j];
        }
        let diag_idx: Vec<usize> = (0..n).map(|i| find(i, i)).collect();
        let mut base_vals = vec![0.0f64; positions.len()];
        for &(r, c, v) in local {
            base_vals[find(r, c)] += v;
        }

        let symbolic_mat = SymbolicSparseColMat::<usize>::new_checked(n, n, col_ptr, None, row_idx);
        let symbolic_lu =
            sparse_lu::factorize_symbolic_lu(symbolic_mat.as_ref(), Default::default())
                .map_err(|e| format!("Sparse LU symbolic factorization error: {e:?}"))?;
        let par = Par::Seq;
        let factor_req = symbolic_lu.factorize_numeric_lu_scratch::<c64>(par, Default::default());
        let solve_req = symbolic_lu.solve_in_place_scratch::<c64>(1, par);
        let mem_buf = MemBuffer::try_new(factor_req.or(solve_req))
            .map_err(|_| "Failed to allocate sparse LU workspace")?;

        Ok(Self {
            rows,
            symbolic_mat,
            symbolic_lu,
            numeric_lu: sparse_lu::NumericLu::new(),
            mem_buf,
            vals: vec![c64::new(0.0, 0.0); base_vals.len()],
            base_vals,
            diag_idx,
            rhs: Mat::<c64>::zeros(n, 1),
        })
    }

    /// Solve `(A_BB dt - theta I) x = rhs` in place in `self.rhs`.
    fn solve_in_place(&mut self, theta: c64) -> Result<(), String> {
        let par = Par::Seq;
        for (v, &bv) in self.vals.iter_mut().zip(&self.base_vals) {
            *v = c64::new(bv, 0.0);
        }
        for &di in &self.diag_idx {
            self.vals[di] -= theta;
        }
        let mat_ref = SparseColMatRef::<usize, c64>::new(self.symbolic_mat.as_ref(), &self.vals);
        let lu_ref = self
            .symbolic_lu
            .factorize_numeric_lu(
                &mut self.numeric_lu,
                mat_ref,
                par,
                MemStack::new(&mut self.mem_buf),
                Default::default(),
            )
            .map_err(|e| format!("Sparse LU numeric factorization error: {e:?}"))?;
        lu_ref.solve_in_place_with_conj(
            Conj::No,
            self.rhs.as_mut(),
            par,
            MemStack::new(&mut self.mem_buf),
        );
        Ok(())
    }
}

/// The strongly connected components of the production graph, upstream first.
///
/// An edge runs from `col` to `row` for every off-diagonal nonzero `A[row,
/// col]`: `col` makes `row`. Every block's feeders from outside it come in an
/// earlier block. Tarjan's algorithm, iterative so a long decay chain cannot
/// overflow the stack; it closes a component only after everything downstream
/// of it, so its order is reversed on the way out.
fn production_blocks(n: usize, triplets: &[(usize, usize, f64)]) -> Vec<Vec<usize>> {
    let mut succ_ptr = vec![0usize; n + 1];
    for &(r, c, v) in triplets {
        if r != c && v != 0.0 {
            succ_ptr[c + 1] += 1;
        }
    }
    for i in 0..n {
        succ_ptr[i + 1] += succ_ptr[i];
    }
    let mut fill = succ_ptr.clone();
    let mut succ = vec![0usize; succ_ptr[n]];
    for &(r, c, v) in triplets {
        if r != c && v != 0.0 {
            succ[fill[c]] = r;
            fill[c] += 1;
        }
    }

    const UNSEEN: usize = usize::MAX;
    let mut index = vec![UNSEEN; n];
    let mut low = vec![0usize; n];
    let mut on_stack = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut call: Vec<(usize, usize)> = Vec::new();
    let mut blocks: Vec<Vec<usize>> = Vec::new();
    let mut next = 0usize;
    for root in 0..n {
        if index[root] != UNSEEN {
            continue;
        }
        index[root] = next;
        low[root] = next;
        next += 1;
        stack.push(root);
        on_stack[root] = true;
        call.push((root, succ_ptr[root]));
        while let Some(top) = call.last_mut() {
            let v = top.0;
            if top.1 < succ_ptr[v + 1] {
                let w = succ[top.1];
                top.1 += 1;
                if index[w] == UNSEEN {
                    index[w] = next;
                    low[w] = next;
                    next += 1;
                    stack.push(w);
                    on_stack[w] = true;
                    call.push((w, succ_ptr[w]));
                } else if on_stack[w] {
                    low[v] = low[v].min(index[w]);
                }
                continue;
            }
            call.pop();
            if let Some(&(parent, _)) = call.last() {
                low[parent] = low[parent].min(low[v]);
            }
            if low[v] == index[v] {
                let mut block = Vec::new();
                loop {
                    let w = stack.pop().expect("v is on the stack");
                    on_stack[w] = false;
                    block.push(w);
                    if w == v {
                        break;
                    }
                }
                block.sort_unstable();
                blocks.push(block);
            }
        }
    }
    blocks.reverse();
    blocks
}

/// Sparse CRAM solver core: block forward substitution over the production
/// graph, with faer's sparse LU inside each block that has a cycle.
///
/// Each pole solves `(A*dt - theta*I) x = y`. Factoring that whole matrix with
/// partial pivoting is not safe for a short-lived nuclide. Its column holds
/// `-lambda*dt - theta` on the diagonal and `+lambda*dt` in its daughter's row,
/// and once `lambda*dt` passes 65.75 the pole at `-44.66 + 62.33i` makes the
/// daughter's entry the larger, so the pivot swaps the daughter's row in. When
/// the daughter is the stable bulk of the material, the short-lived nuclide's
/// unknown is then read off the bulk's equation as a difference of bulk-sized
/// numbers, and what comes back is a residue near 1e-19 of the bulk rather
/// than its population. Zr90m held in equilibrium by Y90m in a cooling
/// zirconium foil is 7e-23 of the bulk; it came back as noise of either sign,
/// decided by the last bit of the step length, and carried up to a tenth of
/// the foil's decay heat.
///
/// The matrix is block lower triangular in the order the production graph
/// runs. A nuclide on no cycle is its own block, and once its feeders are
/// solved its row has one unknown: no pivoting, nothing downstream mixed in,
/// so it is as accurate as its feeders. Only a genuine cycle (reactions during
/// irradiation, mostly) is factored, and pivoting inside it can only mix
/// nuclides that feed one another. It is the same linear system; only where
/// the rounding lands moves. Where only decay connects nuclides, as in a
/// cooldown, every block is normally a single nuclide.
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

    let components = production_blocks(n, triplets);
    let mut block_of = vec![0usize; n];
    let mut local_of = vec![0usize; n];
    for (b, rows) in components.iter().enumerate() {
        for (k, &i) in rows.iter().enumerate() {
            block_of[i] = b;
            local_of[i] = k;
        }
    }

    // Each row's entries of A*dt split three ways: its diagonal, feeders in
    // its own block (into the block's matrix) and feeders upstream (onto the
    // right-hand side).
    let mut diag = vec![0.0f64; n];
    let mut upstream: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut local: Vec<Vec<(usize, usize, f64)>> = vec![Vec::new(); components.len()];
    for &(r, c, v) in triplets {
        let value = v * dt;
        if r == c {
            diag[r] += value;
        } else if block_of[r] == block_of[c] {
            local[block_of[r]].push((local_of[r], local_of[c], value));
        } else {
            upstream[r].push((c, value));
        }
    }
    let mut blocks = Vec::with_capacity(components.len());
    for (b, rows) in components.into_iter().enumerate() {
        if rows.len() == 1 {
            blocks.push(Block::Single {
                row: rows[0],
                diag: diag[rows[0]],
            });
        } else {
            let mut entries = std::mem::take(&mut local[b]);
            for (k, &i) in rows.iter().enumerate() {
                entries.push((k, k, diag[i]));
            }
            blocks.push(Block::Coupled(Box::new(CoupledBlock::new(rows, &entries)?)));
        }
    }

    let mut y = n0.to_vec();
    let mut x = vec![c64::new(0.0, 0.0); n];
    for (alpha_i, theta_i) in alpha.iter().zip(theta.iter()) {
        let theta_c = c64::new(theta_i.re, theta_i.im);
        let rhs_of = |i: usize, x: &[c64]| {
            let mut b = c64::new(y[i], 0.0);
            for &(j, value) in &upstream[i] {
                b -= x[j] * value;
            }
            b
        };
        for block in &mut blocks {
            match block {
                Block::Single { row, diag } => {
                    x[*row] = rhs_of(*row, &x) / (c64::new(*diag, 0.0) - theta_c);
                }
                Block::Coupled(coupled) => {
                    for (k, &i) in coupled.rows.iter().enumerate() {
                        coupled.rhs[(k, 0)] = rhs_of(i, &x);
                    }
                    coupled.solve_in_place(theta_c)?;
                    for (k, &i) in coupled.rows.iter().enumerate() {
                        x[i] = coupled.rhs[(k, 0)];
                    }
                }
            }
        }

        // Accumulate: y += 2 * Re(alpha * x)
        let alpha_c = c64::new(alpha_i.re, alpha_i.im);
        for i in 0..n {
            let contrib = alpha_c * x[i];
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

    /// Y90m (3.19 h) feeding Zr90m (0.808 s), the pair a cooling zirconium
    /// foil holds in equilibrium.
    const LAMBDA_PARENT: f64 = std::f64::consts::LN_2 / 11_484.0;
    const LAMBDA_DAUGHTER: f64 = 0.857_643;

    /// The last member of a Bateman chain after `t`, starting from `parent`
    /// atoms of the first and none of the rest, for distinct decay constants.
    fn bateman_last(parent: f64, lambdas: &[f64], t: f64) -> f64 {
        let sum: f64 = lambdas
            .iter()
            .enumerate()
            .map(|(i, &li)| {
                let denom: f64 = lambdas
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .map(|(_, &lj)| lj - li)
                    .product();
                (-li * t).exp() / denom
            })
            .sum();
        parent * lambdas[..lambdas.len() - 1].iter().product::<f64>() * sum
    }

    fn assert_close(what: &str, got: f64, want: f64, tol: f64) {
        let rel = ((got - want) / want).abs();
        assert!(
            rel < tol,
            "{what}: {got:e} against {want:e}, relative {rel:e}"
        );
    }

    /// Parent, short-lived daughter, stable bulk 1e16 times the daughter.
    fn parent_daughter_bulk() -> (Vec<(usize, usize, f64)>, Vec<f64>) {
        let triplets = vec![
            (0, 0, -LAMBDA_PARENT),
            (1, 0, LAMBDA_PARENT),
            (1, 1, -LAMBDA_DAUGHTER),
            (2, 1, LAMBDA_DAUGHTER),
        ];
        (triplets, vec![1.0e-13, 0.0, 0.1])
    }

    /// A short-lived daughter held in equilibrium by a long-lived parent, on
    /// top of the stable bulk it decays into. Past `lambda * dt` of about 66 a
    /// pivoted LU of the whole matrix read the daughter off the bulk's
    /// equation and returned a residue of about 1e-19 of the bulk: 0.7% of
    /// the daughter here, and of either sign.
    #[test]
    fn a_short_lived_daughter_is_not_read_off_the_bulk() {
        let (triplets, n0) = parent_daughter_bulk();
        let dt = 97.0; // lambda * dt = 83
        let result = cram48_sparse(&triplets, 3, &n0, dt).unwrap();
        let exact = bateman_last(n0[0], &[LAMBDA_PARENT, LAMBDA_DAUGHTER], dt);
        assert_close("daughter", result[1], exact, 1e-12);
        assert_close(
            "parent",
            result[0],
            n0[0] * (-LAMBDA_PARENT * dt).exp(),
            1e-12,
        );
    }

    /// Stretching the step by a few parts in 1e13 moves the daughter by about
    /// `lambda_parent * dt` times that, as it moves its parent, and not by
    /// its own size.
    #[test]
    fn a_short_lived_daughter_does_not_hang_on_the_last_bit_of_dt() {
        let (triplets, n0) = parent_daughter_bulk();
        let dt = 607.0;
        let nominal = cram48_sparse(&triplets, 3, &n0, dt).unwrap()[1];
        for k in 1..=4 {
            let stretched = dt * (1.0 + k as f64 * 1e-13);
            let result = cram48_sparse(&triplets, 3, &n0, stretched).unwrap();
            assert_close("stretched daughter", result[1], nominal, 1e-12);
        }
    }

    /// A short-lived daughter of a short-lived daughter, and a short-lived
    /// nuclide with two parents: each is solved from what feeds it, so the
    /// chain and the sum over parents come out as Bateman gives them.
    #[test]
    fn chains_and_several_parents_of_short_lived_nuclides_match_bateman() {
        let lambda_granddaughter = std::f64::consts::LN_2 / 0.2;
        let lambda_other_parent = std::f64::consts::LN_2 / 3_600.0;
        let dt = 300.0;
        // 0 parent, 1 daughter, 2 granddaughter, 3 stable bulk, 4 a second
        // parent of the granddaughter.
        let triplets = vec![
            (0, 0, -LAMBDA_PARENT),
            (1, 0, LAMBDA_PARENT),
            (1, 1, -LAMBDA_DAUGHTER),
            (2, 1, LAMBDA_DAUGHTER),
            (2, 2, -lambda_granddaughter),
            (3, 2, lambda_granddaughter),
            (4, 4, -lambda_other_parent),
            (2, 4, lambda_other_parent),
        ];
        let n0 = vec![1.0e-13, 0.0, 0.0, 1.0, 1.0e-14];
        let result = cram48_sparse(&triplets, 5, &n0, dt).unwrap();
        assert_close(
            "daughter",
            result[1],
            bateman_last(n0[0], &[LAMBDA_PARENT, LAMBDA_DAUGHTER], dt),
            1e-12,
        );
        let granddaughter =
            bateman_last(
                n0[0],
                &[LAMBDA_PARENT, LAMBDA_DAUGHTER, lambda_granddaughter],
                dt,
            ) + bateman_last(n0[4], &[lambda_other_parent, lambda_granddaughter], dt);
        assert_close("granddaughter", result[2], granddaughter, 1e-12);
    }

    /// A pair that transmute into each other is one block, factored together,
    /// upstream of the short-lived nuclide it feeds; the answer agrees with
    /// the dense solve.
    #[test]
    fn a_production_cycle_is_solved_as_one_block() {
        let n = 4;
        let mut a = vec![0.0; n * n];
        // 0 <-> 1 by reactions, 1 -> 2 (short-lived) -> 3 stable.
        a[0] = -2.0e-4;
        a[n] = 2.0e-4;
        a[n + 1] = -3.0e-4;
        a[1] = 1.0e-4;
        a[2 * n + 1] = 2.0e-4;
        a[2 * n + 2] = -LAMBDA_DAUGHTER;
        a[3 * n + 2] = LAMBDA_DAUGHTER;
        let n0 = vec![1.0e20, 0.0, 0.0, 0.0];
        let dt = 3_600.0;
        let dense = cram48(&a, n, &n0, dt).unwrap();
        let sparse = cram48_sparse(&dense_to_triplets(&a, n), n, &n0, dt).unwrap();
        for i in 0..n {
            assert_close(&format!("nuclide {i}"), sparse[i], dense[i], 1e-12);
        }
    }

    /// Tarjan's order: a block comes after every block that feeds it, and a
    /// cycle is one block.
    #[test]
    fn production_blocks_run_upstream_first() {
        // 3 -> 0 <-> 1 -> 2
        let triplets = vec![(0, 3, 1.0), (1, 0, 1.0), (0, 1, 1.0), (2, 1, 1.0)];
        assert_eq!(
            production_blocks(4, &triplets),
            vec![vec![3], vec![0, 1], vec![2]]
        );
    }
}
