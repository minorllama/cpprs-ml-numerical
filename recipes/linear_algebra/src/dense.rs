use crate::*;

/// Partial-pivoting LU: P A = L U. Exact zero pivots return Singular.
#[derive(Debug, Clone)]
pub struct Lu {
    factors: Matrix,
    permutation: Vec<usize>,
    sign: f64,
}
impl Lu {
    pub fn decompose(a: &Matrix) -> Result<Self> {
        let n = a.square()?;
        let mut f = a.clone();
        let mut p = (0..n).collect::<Vec<_>>();
        let mut sign = 1.;
        for k in 0..n {
            let pivot = (k..n)
                .max_by(|&i, &j| f[(i, k)].abs().total_cmp(&f[(j, k)].abs()))
                .unwrap();
            if f[(pivot, k)] == 0. {
                return Err(Error::Singular);
            }
            if pivot != k {
                f.swap_rows(k, pivot);
                p.swap(k, pivot);
                sign = -sign;
            }
            for i in k + 1..n {
                f[(i, k)] /= f[(k, k)];
                for j in k + 1..n {
                    f[(i, j)] -= f[(i, k)] * f[(k, j)];
                }
            }
        }
        finite(f.as_slice())?;
        Ok(Self {
            factors: f,
            permutation: p,
            sign,
        })
    }
    pub fn factors(&self) -> &Matrix {
        &self.factors
    }
    pub fn permutation(&self) -> &[usize] {
        &self.permutation
    }
    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>> {
        let n = self.permutation.len();
        vector(b, n)?;
        let mut x = self.permutation.iter().map(|&i| b[i]).collect::<Vec<_>>();
        for i in 0..n {
            for j in 0..i {
                x[i] -= self.factors[(i, j)] * x[j];
            }
        }
        for i in (0..n).rev() {
            for j in i + 1..n {
                x[i] -= self.factors[(i, j)] * x[j];
            }
            x[i] /= self.factors[(i, i)];
        }
        finite(&x)?;
        Ok(x)
    }
    pub fn determinant(&self) -> f64 {
        (0..self.permutation.len()).fold(self.sign, |d, i| d * self.factors[(i, i)])
    }
    pub fn inverse(&self) -> Result<Matrix> {
        let n = self.permutation.len();
        let mut inv = Matrix::zeros(n, n);
        for j in 0..n {
            let mut e = vec![0.; n];
            e[j] = 1.;
            let x = self.solve(&e)?;
            for i in 0..n {
                inv[(i, j)] = x[i];
            }
        }
        Ok(inv)
    }
}
/// Gaussian elimination and back substitution; shares the LU elimination kernel.
pub fn gaussian_solve(a: &Matrix, b: &[f64]) -> Result<Vec<f64>> {
    Lu::decompose(a)?.solve(b)
}

/// Full-pivoting Gauss-Jordan elimination. Returns A^-1 and X solving A X = B.
pub fn gauss_jordan(a: &Matrix, b: &Matrix) -> Result<(Matrix, Matrix)> {
    let n = a.square()?;
    if b.rows() != n {
        return Err(Error::Dimension);
    }
    finite(b.as_slice())?;
    let mut w = a.clone();
    let mut rhs = Matrix::zeros(n, n + b.cols());
    let mut perm = (0..n).collect::<Vec<_>>();
    for i in 0..n {
        rhs[(i, i)] = 1.;
        for j in 0..b.cols() {
            rhs[(i, n + j)] = b[(i, j)];
        }
    }
    for k in 0..n {
        let mut pivot = (k, k);
        for i in k..n {
            for j in k..n {
                if w[(i, j)].abs() > w[pivot].abs() {
                    pivot = (i, j);
                }
            }
        }
        if w[pivot] == 0. {
            return Err(Error::Singular);
        }
        w.swap_rows(k, pivot.0);
        rhs.swap_rows(k, pivot.0);
        for i in 0..n {
            let z = w[(i, k)];
            w[(i, k)] = w[(i, pivot.1)];
            w[(i, pivot.1)] = z;
        }
        perm.swap(k, pivot.1);
        let d = w[(k, k)];
        for j in 0..n {
            w[(k, j)] /= d;
        }
        for j in 0..rhs.cols() {
            rhs[(k, j)] /= d;
        }
        for i in 0..n {
            if i == k {
                continue;
            }
            let s = w[(i, k)];
            for j in 0..n {
                w[(i, j)] -= s * w[(k, j)];
            }
            for j in 0..rhs.cols() {
                rhs[(i, j)] -= s * rhs[(k, j)];
            }
        }
    }
    let mut inv = Matrix::zeros(n, n);
    let mut x = Matrix::zeros(n, b.cols());
    for i in 0..n {
        for j in 0..n {
            inv[(perm[i], j)] = rhs[(i, j)];
        }
        for j in 0..b.cols() {
            x[(perm[i], j)] = rhs[(i, n + j)];
        }
    }
    finite(inv.as_slice())?;
    finite(x.as_slice())?;
    Ok((inv, x))
}

/// Iterative refinement using compensated residual sums and a reusable LU.
/// Returns the best iterate; stops when improvement stalls or `steps` is reached.
pub fn refine(a: &Matrix, lu: &Lu, b: &[f64], initial: &[f64], steps: usize) -> Result<Vec<f64>> {
    let n = a.square()?;
    vector(b, n)?;
    vector(initial, n)?;
    let mut x = initial.to_vec();
    let mut err = backward_error(a, &x, b)?;
    for _ in 0..steps {
        let mut residual = vec![0.; n];
        for i in 0..n {
            let mut s = b[i];
            let mut c = 0.;
            for j in 0..n {
                let term = -a[(i, j)] * x[j] - c;
                let t = s + term;
                c = (t - s) - term;
                s = t;
            }
            residual[i] = s;
        }
        let dx = lu.solve(&residual)?;
        let candidate = x.iter().zip(dx).map(|(v, d)| v + d).collect::<Vec<_>>();
        let next = backward_error(a, &candidate, b)?;
        if next >= err {
            break;
        }
        x = candidate;
        err = next;
    }
    Ok(x)
}

#[derive(Debug, Clone)]
pub struct Cholesky {
    l: Matrix,
}
impl Cholesky {
    pub fn decompose(a: &Matrix) -> Result<Self> {
        let n = a.square()?;
        let mut l = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..i {
                let scale = a[(i, j)].abs().max(a[(j, i)].abs());
                if (a[(i, j)] - a[(j, i)]).abs() > 64. * f64::EPSILON * scale {
                    return Err(Error::NotSymmetric);
                }
            }
        }
        for i in 0..n {
            for j in 0..=i {
                let mut s = a[(i, j)];
                for k in 0..j {
                    s -= l[(i, k)] * l[(j, k)];
                }
                if i == j {
                    if s <= 0. {
                        return Err(Error::NotPositiveDefinite);
                    }
                    l[(i, j)] = s.sqrt();
                } else {
                    l[(i, j)] = s / l[(j, j)];
                }
            }
        }
        finite(l.as_slice())?;
        Ok(Self { l })
    }
    pub fn lower(&self) -> &Matrix {
        &self.l
    }
    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>> {
        let n = self.l.rows();
        vector(b, n)?;
        let mut x = b.to_vec();
        for i in 0..n {
            for j in 0..i {
                x[i] -= self.l[(i, j)] * x[j];
            }
            x[i] /= self.l[(i, i)];
        }
        for i in (0..n).rev() {
            for j in i + 1..n {
                x[i] -= self.l[(j, i)] * x[j];
            }
            x[i] /= self.l[(i, i)];
        }
        finite(&x)?;
        Ok(x)
    }
}
/// Upper-triangular back substitution, with a relative diagonal cutoff.
pub fn upper_solve(r: &Matrix, b: &[f64], rtol: f64) -> Result<Vec<f64>> {
    let n = r.square()?;
    vector(b, n)?;
    tolerance(rtol)?;
    let scale = r.as_slice().iter().map(|v| v.abs()).fold(0., f64::max);
    let mut x = b.to_vec();
    for i in (0..n).rev() {
        if r[(i, i)].abs() <= rtol * scale {
            return Err(Error::Singular);
        }
        for j in i + 1..n {
            x[i] -= r[(i, j)] * x[j];
        }
        x[i] /= r[(i, i)];
    }
    finite(&x)?;
    Ok(x)
}
/// Orthogonal row rotation: [row_i; row_j] <- [c s; -s c] [row_i; row_j].
pub fn rotate_rows(a: &mut Matrix, i: usize, j: usize, c: f64, s: f64) -> Result<()> {
    if i >= a.rows() || j >= a.rows() || i == j {
        return Err(Error::Dimension);
    }
    finite(&[c, s])?;
    if (c.hypot(s) - 1.).abs() > 64. * f64::EPSILON {
        return Err(Error::InvalidParameter);
    }
    for k in 0..a.cols() {
        let x = a[(i, k)];
        let y = a[(j, k)];
        a[(i, k)] = c * x + s * y;
        a[(j, k)] = -s * x + c * y;
    }
    Ok(())
}
/// Householder QR with explicit full Q, for m >= n.
#[derive(Debug, Clone)]
pub struct Qr {
    q: Matrix,
    r: Matrix,
}
impl Qr {
    pub fn decompose(a: &Matrix) -> Result<Self> {
        finite(a.as_slice())?;
        let (m, n) = (a.rows(), a.cols());
        if m < n {
            return Err(Error::Dimension);
        }
        let mut q = Matrix::identity(m);
        let mut r = a.clone();
        for k in 0..n {
            let scale = (k..m).map(|i| r[(i, k)].abs()).fold(0., f64::max);
            if scale == 0. {
                continue;
            }
            let mut v = (k..m).map(|i| r[(i, k)] / scale).collect::<Vec<_>>();
            let length = norm2(&v);
            v[0] += length.copysign(v[0]);
            let vn = norm2(&v);
            for x in &mut v {
                *x /= vn;
            }
            for j in k..n {
                let d = 2. * (k..m).map(|i| v[i - k] * r[(i, j)]).sum::<f64>();
                for i in k..m {
                    r[(i, j)] -= d * v[i - k];
                }
            }
            for i in 0..m {
                let d = 2. * (k..m).map(|j| q[(i, j)] * v[j - k]).sum::<f64>();
                for j in k..m {
                    q[(i, j)] -= d * v[j - k];
                }
            }
            for i in k + 1..m {
                r[(i, k)] = 0.;
            }
        }
        finite(q.as_slice())?;
        finite(r.as_slice())?;
        Ok(Self { q, r })
    }
    pub fn q(&self) -> &Matrix {
        &self.q
    }
    pub fn r(&self) -> &Matrix {
        &self.r
    }
    pub fn solve(&self, b: &[f64], rtol: f64) -> Result<Vec<f64>> {
        vector(b, self.q.rows())?;
        let y = self.q.transpose().mul_vec(b)?;
        let n = self.r.cols();
        let mut r = Matrix::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                r[(i, j)] = self.r[(i, j)];
            }
        }
        upper_solve(&r, &y[..n], rtol)
    }
    /// Givens rank-one update A <- A + u v^T, O(m^2 + mn).
    pub fn rank_one_update(&mut self, u: &[f64], v: &[f64]) -> Result<()> {
        let m = self.q.rows();
        let n = self.r.cols();
        vector(u, m)?;
        vector(v, n)?;
        let mut w = self.q.transpose().mul_vec(u)?;
        // Reduce Q^T u to its first coordinate; counterrotate Q's columns.
        for i in (1..m).rev() {
            let h = w[i - 1].hypot(w[i]);
            if h == 0. {
                continue;
            }
            let (c, s) = (w[i - 1] / h, w[i] / h);
            self.apply_rotation(i - 1, i, c, s)?;
            w[i - 1] = h;
            w[i] = 0.;
        }
        for (j, &vj) in v.iter().enumerate() {
            self.r[(0, j)] += w[0] * vj;
        }
        // R is upper Hessenberg after the bottom-up rotations.
        for i in 0..n.min(m - 1) {
            let h = self.r[(i, i)].hypot(self.r[(i + 1, i)]);
            if h == 0. {
                continue;
            }
            let (c, s) = (self.r[(i, i)] / h, self.r[(i + 1, i)] / h);
            self.apply_rotation(i, i + 1, c, s)?;
            self.r[(i + 1, i)] = 0.;
        }
        finite(self.r.as_slice())?;
        Ok(())
    }
    fn apply_rotation(&mut self, i: usize, j: usize, c: f64, s: f64) -> Result<()> {
        rotate_rows(&mut self.r, i, j, c, s)?;
        for k in 0..self.q.rows() {
            let x = self.q[(k, i)];
            let y = self.q[(k, j)];
            self.q[(k, i)] = c * x + s * y;
            self.q[(k, j)] = -s * x + c * y;
        }
        Ok(())
    }
}
