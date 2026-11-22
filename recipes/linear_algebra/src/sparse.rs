use crate::*;
use std::collections::BTreeMap;
/// Canonical compressed sparse row matrix: columns strictly increasing per row.
#[derive(Debug, Clone)]
pub struct Csr {
    rows: usize,
    cols: usize,
    offsets: Vec<usize>,
    columns: Vec<usize>,
    values: Vec<f64>,
}
impl Csr {
    pub fn new(
        rows: usize,
        cols: usize,
        offsets: Vec<usize>,
        columns: Vec<usize>,
        values: Vec<f64>,
    ) -> Result<Self> {
        if rows == 0
            || cols == 0
            || offsets.len() != rows + 1
            || offsets[0] != 0
            || offsets[rows] != values.len()
            || columns.len() != values.len()
        {
            return Err(Error::Dimension);
        }
        finite(&values)?;
        for i in 0..rows {
            if offsets[i] > offsets[i + 1] || offsets[i + 1] > values.len() {
                return Err(Error::Dimension);
            }
            let r = &columns[offsets[i]..offsets[i + 1]];
            if r.iter().any(|&j| j >= cols) || r.windows(2).any(|v| v[0] >= v[1]) {
                return Err(Error::InvalidParameter);
            }
        }
        Ok(Self {
            rows,
            cols,
            offsets,
            columns,
            values,
        })
    }
    pub fn from_dense(a: &Matrix, threshold: f64) -> Result<Self> {
        tolerance(threshold)?;
        finite(a.as_slice())?;
        let mut offsets = vec![0];
        let mut columns = Vec::new();
        let mut values = Vec::new();
        for i in 0..a.rows() {
            for j in 0..a.cols() {
                if a[(i, j)].abs() > threshold {
                    columns.push(j);
                    values.push(a[(i, j)]);
                }
            }
            offsets.push(values.len());
        }
        Self::new(a.rows(), a.cols(), offsets, columns, values)
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }
    pub fn columns(&self) -> &[usize] {
        &self.columns
    }
    pub fn values(&self) -> &[f64] {
        &self.values
    }
    pub fn to_dense(&self) -> Matrix {
        let mut a = Matrix::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for p in self.offsets[i]..self.offsets[i + 1] {
                a[(i, self.columns[p])] = self.values[p];
            }
        }
        a
    }
    pub fn mul_vec(&self, x: &[f64]) -> Result<Vec<f64>> {
        vector(x, self.cols)?;
        let y = (0..self.rows)
            .map(|i| {
                (self.offsets[i]..self.offsets[i + 1])
                    .map(|p| self.values[p] * x[self.columns[p]])
                    .sum()
            })
            .collect::<Vec<_>>();
        finite(&y)?;
        Ok(y)
    }
    pub fn transpose_mul_vec(&self, x: &[f64]) -> Result<Vec<f64>> {
        vector(x, self.rows)?;
        let mut y = vec![0.; self.cols];
        for (i, &xi) in x.iter().enumerate() {
            for p in self.offsets[i]..self.offsets[i + 1] {
                y[self.columns[p]] += self.values[p] * xi;
            }
        }
        finite(&y)?;
        Ok(y)
    }
    pub fn transpose(&self) -> Self {
        let mut offsets = vec![0; self.cols + 1];
        for &j in &self.columns {
            offsets[j + 1] += 1;
        }
        for i in 0..self.cols {
            offsets[i + 1] += offsets[i];
        }
        let mut next = offsets.clone();
        let mut columns = vec![0; self.values.len()];
        let mut values = vec![0.; self.values.len()];
        for i in 0..self.rows {
            for p in self.offsets[i]..self.offsets[i + 1] {
                let j = self.columns[p];
                let q = next[j];
                columns[q] = i;
                values[q] = self.values[p];
                next[j] += 1;
            }
        }
        Self {
            rows: self.cols,
            cols: self.rows,
            offsets,
            columns,
            values,
        }
    }
    /// Gustavson row accumulation; drops entries with |value| <= threshold.
    pub fn multiply(&self, b: &Self, threshold: f64) -> Result<Self> {
        tolerance(threshold)?;
        if self.cols != b.rows {
            return Err(Error::Dimension);
        }
        let mut offsets = vec![0];
        let mut columns = Vec::new();
        let mut values = Vec::new();
        for i in 0..self.rows {
            let mut row = BTreeMap::new();
            for p in self.offsets[i]..self.offsets[i + 1] {
                let k = self.columns[p];
                for q in b.offsets[k]..b.offsets[k + 1] {
                    *row.entry(b.columns[q]).or_insert(0.) += self.values[p] * b.values[q];
                }
            }
            for (j, v) in row {
                if !v.is_finite() {
                    return Err(Error::NonFinite);
                }
                if v.abs() > threshold {
                    columns.push(j);
                    values.push(v);
                }
            }
            offsets.push(values.len());
        }
        Self::new(self.rows, b.cols, offsets, columns, values)
    }
    /// Computes A B only at locations stored in `pattern`; stored zeros are kept.
    pub fn multiply_pattern(&self, b: &Self, pattern: &Self) -> Result<Self> {
        if self.cols != b.rows || pattern.rows != self.rows || pattern.cols != b.cols {
            return Err(Error::Dimension);
        }
        let bt = b.transpose();
        let mut values = Vec::with_capacity(pattern.values.len());
        for i in 0..self.rows {
            for p in pattern.offsets[i]..pattern.offsets[i + 1] {
                let j = pattern.columns[p];
                let mut a = self.offsets[i];
                let mut b = bt.offsets[j];
                let mut s = 0.;
                while a < self.offsets[i + 1] && b < bt.offsets[j + 1] {
                    match self.columns[a].cmp(&bt.columns[b]) {
                        std::cmp::Ordering::Less => a += 1,
                        std::cmp::Ordering::Greater => b += 1,
                        std::cmp::Ordering::Equal => {
                            s += self.values[a] * bt.values[b];
                            a += 1;
                            b += 1;
                        }
                    }
                }
                values.push(s);
            }
        }
        Self::new(
            pattern.rows,
            pattern.cols,
            pattern.offsets.clone(),
            pattern.columns.clone(),
            values,
        )
    }
    /// Jacobi preconditioner. A missing/zero diagonal uses identity at that row.
    pub fn diagonal_solve(&self, b: &[f64]) -> Result<Vec<f64>> {
        if self.rows != self.cols {
            return Err(Error::Dimension);
        }
        vector(b, self.rows)?;
        let mut x = b.to_vec();
        for (i, xi) in x.iter_mut().enumerate() {
            for p in self.offsets[i]..self.offsets[i + 1] {
                if self.columns[p] == i && self.values[p] != 0. {
                    *xi /= self.values[p];
                    break;
                }
            }
        }
        finite(&x)?;
        Ok(x)
    }
}
#[derive(Debug, Clone, Copy)]
pub enum StopCriterion {
    ResidualL2,
    PreconditionedL2,
    EstimatedErrorL2,
    EstimatedErrorInf,
    ResidualInf,
    PreconditionedInf,
}
#[derive(Debug, Clone)]
pub struct Iteration {
    pub x: Vec<f64>,
    pub iterations: usize,
    pub relative_residual: f64,
    pub stopping_error: f64,
}
fn measure(a: &Csr, x: &[f64], criterion: StopCriterion) -> Result<f64> {
    let p = matches!(
        criterion,
        StopCriterion::PreconditionedL2
            | StopCriterion::PreconditionedInf
            | StopCriterion::EstimatedErrorL2
            | StopCriterion::EstimatedErrorInf
    );
    let v = if p { a.diagonal_solve(x)? } else { x.to_vec() };
    Ok(
        if matches!(
            criterion,
            StopCriterion::ResidualInf
                | StopCriterion::PreconditionedInf
                | StopCriterion::EstimatedErrorInf
        ) {
            v.iter().map(|x| x.abs()).fold(0., f64::max)
        } else {
            norm2(&v)
        },
    )
}
/// Jacobi-preconditioned BiCG for a real nonsymmetric square CSR matrix.
/// Checks the true residual on every iteration. Zero RHS uses absolute error.
/// BiCG can break down on nonsingular systems; returns Breakdown explicitly.
pub fn bicg(
    a: &Csr,
    b: &[f64],
    initial: &[f64],
    tol: f64,
    max_iter: usize,
    criterion: StopCriterion,
) -> Result<Iteration> {
    if a.rows != a.cols {
        return Err(Error::Dimension);
    }
    let n = a.rows;
    vector(b, n)?;
    vector(initial, n)?;
    tolerance(tol)?;
    if tol == 0. {
        return Err(Error::InvalidParameter);
    }
    let mut x = initial.to_vec();
    let ax = a.mul_vec(&x)?;
    let mut r = b.iter().zip(ax).map(|(b, a)| b - a).collect::<Vec<_>>();
    let mut shadow = r.clone();
    let bn = norm2(b);
    let denom = measure(a, b, criterion)?;
    let denom = if denom == 0. { 1. } else { denom };
    let mut p = vec![0.; n];
    let mut pt = p.clone();
    let mut old_rho = 1.;
    let estimated = matches!(
        criterion,
        StopCriterion::EstimatedErrorL2 | StopCriterion::EstimatedErrorInf
    );
    let correction_norm = |v: &[f64]| {
        if matches!(criterion, StopCriterion::EstimatedErrorInf) {
            v.iter().map(|x| x.abs()).fold(0., f64::max)
        } else {
            norm2(v)
        }
    };
    let mut previous_measure: Option<f64> = None;
    let mut last_correction = 0.;
    for iter in 0..=max_iter {
        let actual = b
            .iter()
            .zip(a.mul_vec(&x)?)
            .map(|(b, v)| b - v)
            .collect::<Vec<_>>();
        let metric = measure(a, &actual, criterion)?;
        let mut error = metric / denom;
        if estimated {
            if let Some(previous) = previous_measure {
                let difference = (previous - metric).abs();
                if difference > f64::EPSILON * metric {
                    let estimate = last_correction * metric / difference;
                    let xn = correction_norm(&x);
                    // Use the correction estimate only when it is small compared
                    // with the iterate; otherwise retain the residual criterion.
                    if xn > 0. && estimate <= 0.5 * xn {
                        error = estimate / xn;
                    }
                }
            }
        }
        previous_measure = Some(metric);
        if error <= tol {
            return Ok(Iteration {
                x,
                iterations: iter,
                relative_residual: norm2(&actual) / if bn == 0. { 1. } else { bn },
                stopping_error: error,
            });
        }
        if iter == max_iter {
            return Err(Error::NoConvergence {
                iterations: iter,
                residual: error,
            });
        }
        let z = a.diagonal_solve(&r)?;
        let zt = a.diagonal_solve(&shadow)?;
        let rho = dot(&shadow, &z);
        if rho == 0. || !rho.is_finite() {
            return Err(Error::Breakdown);
        }
        let beta = if iter == 0 { 0. } else { rho / old_rho };
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
            pt[i] = zt[i] + beta * pt[i];
        }
        let ap = a.mul_vec(&p)?;
        let atp = a.transpose_mul_vec(&pt)?;
        let den = dot(&pt, &ap);
        if den == 0. || !den.is_finite() {
            return Err(Error::Breakdown);
        }
        let alpha = rho / den;
        last_correction = alpha.abs() * correction_norm(&p);
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
            shadow[i] -= alpha * atp[i];
        }
        finite(&x)?;
        old_rho = rho;
    }
    unreachable!()
}
