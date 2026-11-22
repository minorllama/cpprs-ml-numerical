use crate::*;
use std::collections::BTreeMap;

/// Thomas algorithm. lower/upper have length n-1; no pivoting.
pub fn tridiagonal(lower: &[f64], diag: &[f64], upper: &[f64], b: &[f64]) -> Result<Vec<f64>> {
    let n = diag.len();
    if n == 0 {
        return Err(Error::Dimension);
    }
    vector(lower, n - 1)?;
    vector(upper, n - 1)?;
    vector(b, n)?;
    finite(diag)?;
    let mut d = diag.to_vec();
    let mut x = b.to_vec();
    for i in 1..n {
        if d[i - 1] == 0. {
            return Err(Error::Breakdown);
        }
        let t = lower[i - 1] / d[i - 1];
        d[i] -= t * upper[i - 1];
        x[i] -= t * x[i - 1];
    }
    if d[n - 1] == 0. {
        return Err(Error::Breakdown);
    }
    x[n - 1] /= d[n - 1];
    for i in (0..n - 1).rev() {
        if d[i] == 0. {
            return Err(Error::Breakdown);
        }
        x[i] = (x[i] - upper[i] * x[i + 1]) / d[i];
    }
    finite(&x)?;
    Ok(x)
}
/// Cyclic tridiagonal solve via Sherman-Morrison.
/// `top_right` = A[0,n-1], `bottom_left` = A[n-1,0]; n >= 3.
pub fn cyclic_tridiagonal(
    lower: &[f64],
    diag: &[f64],
    upper: &[f64],
    top_right: f64,
    bottom_left: f64,
    b: &[f64],
) -> Result<Vec<f64>> {
    let n = diag.len();
    if n < 3 {
        return Err(Error::Dimension);
    }
    finite(&[top_right, bottom_left])?;
    finite(diag)?;
    let gamma = if diag[0] != 0. { -diag[0] } else { -1. };
    let mut d = diag.to_vec();
    d[0] -= gamma;
    d[n - 1] -= top_right * bottom_left / gamma;
    let mut x = tridiagonal(lower, &d, upper, b)?;
    let mut u = vec![0.; n];
    u[0] = gamma;
    u[n - 1] = bottom_left;
    let z = tridiagonal(lower, &d, upper, &u)?;
    let den = 1. + z[0] + top_right * z[n - 1] / gamma;
    if den == 0. {
        return Err(Error::Breakdown);
    }
    let t = (x[0] + top_right * x[n - 1] / gamma) / den;
    for i in 0..n {
        x[i] -= t * z[i];
    }
    finite(&x)?;
    Ok(x)
}
/// Compact band storage: entry (i,j) is data[i*(lower+upper+1)+j+lower-i].
/// Out-of-matrix padding is ignored and must be finite. O(n*(lower+upper+1)) storage.
#[derive(Debug, Clone)]
pub struct BandMatrix {
    n: usize,
    lower: usize,
    upper: usize,
    data: Vec<f64>,
}
impl BandMatrix {
    pub fn new(n: usize, lower: usize, upper: usize, data: Vec<f64>) -> Result<Self> {
        if n == 0 || lower >= n || upper >= n {
            return Err(Error::Dimension);
        }
        let width = lower
            .checked_add(upper)
            .and_then(|v| v.checked_add(1))
            .ok_or(Error::Dimension)?;
        if n.checked_mul(width) != Some(data.len()) {
            return Err(Error::Dimension);
        }
        finite(&data)?;
        Ok(Self {
            n,
            lower,
            upper,
            data,
        })
    }
    pub fn from_dense(a: &Matrix, lower: usize, upper: usize) -> Result<Self> {
        let n = a.square()?;
        if lower >= n || upper >= n {
            return Err(Error::Dimension);
        }
        let width = lower + upper + 1;
        let mut data = vec![0.; n * width];
        for i in 0..n {
            for j in 0..n {
                if j + lower >= i && j <= i + upper {
                    data[i * width + j + lower - i] = a[(i, j)];
                } else if a[(i, j)] != 0. {
                    return Err(Error::InvalidParameter);
                }
            }
        }
        Self::new(n, lower, upper, data)
    }
    pub fn get(&self, i: usize, j: usize) -> f64 {
        assert!(i < self.n && j < self.n);
        if j + self.lower < i || j > i + self.upper {
            0.
        } else {
            self.data[i * (self.lower + self.upper + 1) + j + self.lower - i]
        }
    }
    pub fn mul_vec(&self, x: &[f64]) -> Result<Vec<f64>> {
        vector(x, self.n)?;
        let mut y = vec![0.; self.n];
        for (i, yi) in y.iter_mut().enumerate() {
            for (j, &xj) in x
                .iter()
                .enumerate()
                .take((i + self.upper + 1).min(self.n))
                .skip(i.saturating_sub(self.lower))
            {
                *yi += self.get(i, j) * xj;
            }
        }
        finite(&y)?;
        Ok(y)
    }
    pub fn decompose(&self) -> Result<BandLu> {
        BandLu::decompose(self)
    }
}
/// Pivoted band elimination using ordered sparse rows; no dense expansion.
/// Stores pivot sequence and elimination multipliers separately, preserving fill.
#[derive(Debug, Clone)]
pub struct BandLu {
    rows: Vec<BTreeMap<usize, f64>>,
    pivots: Vec<usize>,
    multipliers: Vec<Vec<(usize, f64)>>,
    sign: f64,
}
impl BandLu {
    pub fn decompose(a: &BandMatrix) -> Result<Self> {
        let n = a.n;
        let mut rows = vec![BTreeMap::new(); n];
        for (i, row) in rows.iter_mut().enumerate() {
            for j in i.saturating_sub(a.lower)..(i + a.upper + 1).min(n) {
                let v = a.get(i, j);
                if v != 0. {
                    row.insert(j, v);
                }
            }
        }
        let mut pivots = Vec::new();
        let mut multipliers = Vec::new();
        let mut sign = 1.;
        for k in 0..n {
            let end = (k + a.lower + 1).min(n);
            let p = (k..end)
                .max_by(|&i, &j| {
                    rows[i]
                        .get(&k)
                        .unwrap_or(&0.)
                        .abs()
                        .total_cmp(&rows[j].get(&k).unwrap_or(&0.).abs())
                })
                .unwrap();
            let pivot = *rows[p].get(&k).unwrap_or(&0.);
            if pivot == 0. {
                return Err(Error::Singular);
            }
            rows.swap(k, p);
            if p != k {
                sign = -sign;
            }
            pivots.push(p);
            let tail = rows[k]
                .range(k + 1..)
                .map(|(&j, &v)| (j, v))
                .collect::<Vec<_>>();
            let mut mult = Vec::new();
            for (i, row) in rows.iter_mut().enumerate().take(end).skip(k + 1) {
                let v = row.remove(&k).unwrap_or(0.);
                if v == 0. {
                    continue;
                }
                let t = v / pivot;
                mult.push((i, t));
                for &(j, u) in &tail {
                    let z = row.entry(j).or_insert(0.);
                    *z -= t * u;
                }
            }
            multipliers.push(mult);
        }
        for r in &rows {
            finite(&r.values().copied().collect::<Vec<_>>())?;
        }
        Ok(Self {
            rows,
            pivots,
            multipliers,
            sign,
        })
    }
    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>> {
        let n = self.rows.len();
        vector(b, n)?;
        let mut x = b.to_vec();
        for k in 0..n {
            x.swap(k, self.pivots[k]);
            for &(i, t) in &self.multipliers[k] {
                x[i] -= t * x[k];
            }
        }
        for i in (0..n).rev() {
            for (&j, &v) in self.rows[i].range(i + 1..) {
                x[i] -= v * x[j];
            }
            x[i] /= self.rows[i][&i];
        }
        finite(&x)?;
        Ok(x)
    }
    pub fn determinant(&self) -> f64 {
        self.rows
            .iter()
            .enumerate()
            .fold(self.sign, |s, (i, r)| s * r[&i])
    }
}
/// Solve `sum_j nodes[j]^i * x[j] = moments[i]`, i=0..n-1.
/// O(n^2) Lagrange-polynomial dual construction; distinct nodes required.
pub fn vandermonde(nodes: &[f64], moments: &[f64]) -> Result<Vec<f64>> {
    let n = nodes.len();
    if n == 0 {
        return Err(Error::Dimension);
    }
    finite(nodes)?;
    vector(moments, n)?;
    let mut polynomial = vec![0.; n + 1];
    polynomial[0] = 1.;
    for (degree, &t) in nodes.iter().enumerate() {
        for i in (1..=degree + 1).rev() {
            polynomial[i] = polynomial[i - 1] - t * polynomial[i];
        }
        polynomial[0] *= -t;
    }
    let mut x = vec![0.; n];
    let mut q = vec![0.; n];
    for j in 0..n {
        q[n - 1] = polynomial[n];
        for i in (1..n).rev() {
            q[i - 1] = polynomial[i] + nodes[j] * q[i];
        }
        let mut denominator = 1.;
        for k in 0..n {
            if k != j {
                if nodes[j] == nodes[k] {
                    return Err(Error::Singular);
                }
                denominator *= nodes[j] - nodes[k];
            }
        }
        if denominator == 0. {
            return Err(Error::Breakdown);
        }
        x[j] = dot(&q, moments) / denominator;
    }
    finite(&x)?;
    Ok(x)
}
/// General nonsymmetric Toeplitz solve, `T[i,j]=col[i-j]` or `row[j-i]`.
/// O(n^2) bordering/Levinson recurrence; requires nonsingular leading blocks.
/// Breakdown does not imply that the whole Toeplitz matrix is singular.
pub fn toeplitz(col: &[f64], row: &[f64], b: &[f64]) -> Result<Vec<f64>> {
    let n = col.len();
    if n == 0 {
        return Err(Error::Dimension);
    }
    vector(row, n)?;
    vector(b, n)?;
    finite(col)?;
    if row[0] != col[0] {
        return Err(Error::InvalidParameter);
    }
    if col[0] == 0. {
        return Err(Error::Breakdown);
    }
    let mut x = vec![b[0] / col[0]];
    if n == 1 {
        finite(&x)?;
        return Ok(x);
    }
    let mut f = vec![1. / col[0]];
    let mut ft = f.clone();
    let mut u = vec![row[1] / col[0]];
    let mut v = vec![col[1] / col[0]];
    for k in 1..n {
        let d = (0..k).map(|i| col[k - i]).collect::<Vec<_>>();
        let c = (0..k).map(|i| row[k - i]).collect::<Vec<_>>();
        let schur = col[0] - dot(&d, &u);
        if schur == 0. {
            return Err(Error::Breakdown);
        }
        let last = (b[k] - dot(&d, &x)) / schur;
        for i in 0..k {
            x[i] -= u[i] * last;
        }
        x.push(last);
        if k + 1 == n {
            break;
        }
        let df = dot(&d, &f) / schur;
        let cf = dot(&c, &ft) / schur;
        for i in 0..k {
            f[i] += u[i] * df;
            ft[i] += v[i] * cf;
        }
        f.push(-df);
        ft.push(-cf);
        let ru = row[k + 1] - (0..k).map(|i| row[i + 1] * u[i]).sum::<f64>();
        let cv = col[k + 1] - (0..k).map(|i| col[i + 1] * v[i]).sum::<f64>();
        u.insert(0, 0.);
        v.insert(0, 0.);
        for i in 0..=k {
            u[i] += ru * f[i];
            v[i] += cv * ft[i];
        }
    }
    finite(&x)?;
    Ok(x)
}
