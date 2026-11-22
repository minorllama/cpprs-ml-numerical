use numerical_algorithms::*;

fn mat(r: usize, c: usize, x: &[f64]) -> Matrix {
    Matrix::new(r, c, x.to_vec()).unwrap()
}
fn close(x: &[f64], y: &[f64], tol: f64) {
    assert_eq!(x.len(), y.len());
    for (a, b) in x.iter().zip(y) {
        assert!(
            (a - b).abs() <= tol * (1. + b.abs()),
            "{a} != {b}, tolerance {tol}"
        );
    }
}
fn same(a: &Matrix, b: &Matrix, t: f64) {
    assert_eq!((a.rows(), a.cols()), (b.rows(), b.cols()));
    close(a.as_slice(), b.as_slice(), t);
}
fn sample() -> Matrix {
    mat(3, 3, &[0., 2., 1., 3., -1., 2., 1., 1., 4.])
}
fn generated(n: usize) -> Matrix {
    let mut a = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = ((i * 17 + j * 11 + 3) as f64).sin();
        }
        a[(i, i)] += n as f64;
    }
    a
}
fn truth(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i + 2) as f64).cos()).collect()
}

#[test]
fn strassen_odd_sizes_and_leaf_sizes() {
    for n in 1..10 {
        let a = generated(n);
        let b = a.transpose();
        for leaf in [1, 2, 8] {
            same(
                &fast::strassen(&a, &b, leaf).unwrap(),
                &a.mul(&b).unwrap(),
                1e-11,
            );
        }
    }
}


#[test]
fn sparse_matvec_transpose_conversion() {
    let a = mat(3, 4, &[0., 2., 0., -1., 3., 0., 0., 4., 0., 0., 5., 0.]);
    let s = Csr::from_dense(&a, 0.).unwrap();
    same(&s.to_dense(), &a, 0.);
    same(&s.transpose().to_dense(), &a.transpose(), 0.);
    close(
        &s.mul_vec(&truth(4)).unwrap(),
        &a.mul_vec(&truth(4)).unwrap(),
        1e-14,
    );
    close(
        &s.transpose_mul_vec(&truth(3)).unwrap(),
        &a.transpose().mul_vec(&truth(3)).unwrap(),
        1e-14,
    );
}
#[test]
fn sparse_products_threshold_and_pattern() {
    let a = Csr::from_dense(&sample(), 0.).unwrap();
    let b = a.transpose();
    let dense = sample().mul(&sample().transpose()).unwrap();
    same(&a.multiply(&b, 0.).unwrap().to_dense(), &dense, 1e-14);
    let p = Csr::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![0.; 3]).unwrap();
    let c = a.multiply_pattern(&b, &p).unwrap().to_dense();
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(c[(i, j)], if i == j { dense[(i, j)] } else { 0. });
        }
    }
    assert_eq!(a.multiply(&b, 1e9).unwrap().values().len(), 0);
}

#[test]
fn lu_factor_reconstruction() {
    for n in 1..12 {
        let a = generated(n);
        let f = Lu::decompose(&a).unwrap();
        let mut l = Matrix::identity(n);
        let mut u = Matrix::zeros(n, n);
        let mut pa = a.clone();
        for i in 0..n {
            for j in 0..n {
                pa[(i, j)] = a[(f.permutation()[i], j)];
                if i > j {
                    l[(i, j)] = f.factors()[(i, j)];
                } else {
                    u[(i, j)] = f.factors()[(i, j)];
                }
            }
        }
        same(&l.mul(&u).unwrap(), &pa, 1e-12);
    }
}
#[test]
fn gaussian_solve_known() {
    let a = sample();
    let x = truth(3);
    close(
        &gaussian_solve(&a, &a.mul_vec(&x).unwrap()).unwrap(),
        &x,
        1e-12,
    );
}
#[test]
fn gauss_jordan_full_pivots_multiple_rhs() {
    for n in 1..10 {
        let a = generated(n);
        let b = mat(
            n,
            2,
            &(0..2 * n).map(|x| (x as f64).cos()).collect::<Vec<_>>(),
        );
        let (inv, x) = gauss_jordan(&a, &b).unwrap();
        same(&a.mul(&x).unwrap(), &b, 1e-12);
        same(&a.mul(&inv).unwrap(), &Matrix::identity(n), 1e-12);
    }
}
#[test]
fn gauss_jordan_column_permutations() {
    let a = mat(3, 3, &[0., 0., 100., 2., 0., 0., 0., 3., 0.]);
    let b = Matrix::identity(3);
    let (inv, x) = gauss_jordan(&a, &b).unwrap();
    same(&inv, &x, 1e-14);
    same(&a.mul(&inv).unwrap(), &b, 1e-14);
}
#[test]
fn refinement_improves_inaccurate_guess() {
    let a = generated(8);
    let x = truth(8);
    let b = a.mul_vec(&x).unwrap();
    let f = Lu::decompose(&a).unwrap();
    let r = refine(&a, &f, &b, &[0.; 8], 5).unwrap();
    close(&r, &x, 1e-12);
    assert!(backward_error(&a, &r, &b).unwrap() < 1e-14);
}
#[test]
fn cholesky_reconstruct_and_solve() {
    for n in 1..10 {
        let a = generated(n);
        let a = a.transpose().mul(&a).unwrap();
        let f = Cholesky::decompose(&a).unwrap();
        same(&f.lower().mul(&f.lower().transpose()).unwrap(), &a, 1e-12);
        let x = truth(n);
        close(&f.solve(&a.mul_vec(&x).unwrap()).unwrap(), &x, 1e-12);
    }
}


#[test]
fn householder_qr_square_and_tall() {
    for (m, n) in [(1, 1), (4, 4), (7, 3), (10, 5)] {
        let mut a = Matrix::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                a[(i, j)] = ((i * 7 + j * 3 + 1) as f64).sin() + if i == j { 4. } else { 0. };
            }
        }
        let q = Qr::decompose(&a).unwrap();
        same(&q.q().mul(q.r()).unwrap(), &a, 1e-12);
        same(
            &q.q().transpose().mul(q.q()).unwrap(),
            &Matrix::identity(m),
            1e-12,
        );
        let x = truth(n);
        close(&q.solve(&a.mul_vec(&x).unwrap(), 1e-13).unwrap(), &x, 1e-12);
    }
}
#[test]
fn qr_least_squares_normal_residual() {
    let a = mat(4, 2, &[1., 0., 1., 1., 1., 2., 1., 3.]);
    let b = [0., 1., 1., 4.];
    let q = Qr::decompose(&a).unwrap();
    let x = q.solve(&b, 1e-13).unwrap();
    let r = a
        .mul_vec(&x)
        .unwrap()
        .iter()
        .zip(b)
        .map(|(x, y)| x - y)
        .collect::<Vec<_>>();
    close(&a.transpose().mul_vec(&r).unwrap(), &[0., 0.], 1e-12);
}
#[test]
fn givens_rank_one_updates() {
    for (m, n) in [(1, 1), (5, 5), (8, 3)] {
        let mut a = Matrix::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                a[(i, j)] = if i == j {
                    3.
                } else {
                    ((i + j + 1) as f64).cos()
                };
            }
        }
        let mut q = Qr::decompose(&a).unwrap();
        for step in 0..3 {
            let u = (0..m)
                .map(|i| ((i + step + 1) as f64).sin())
                .collect::<Vec<_>>();
            let v = truth(n);
            q.rank_one_update(&u, &v).unwrap();
            for i in 0..m {
                for j in 0..n {
                    a[(i, j)] += u[i] * v[j];
                }
            }
            same(&q.q().mul(q.r()).unwrap(), &a, 1e-11);
            same(
                &q.q().transpose().mul(q.q()).unwrap(),
                &Matrix::identity(m),
                1e-11,
            );
            for i in 0..m {
                for j in 0..n.min(i) {
                    assert!(q.r()[(i, j)].abs() < 1e-11);
                }
            }
        }
    }
}
#[test]
fn upper_solve_and_rotation() {
    let r = mat(3, 3, &[2., 1., 3., 0., 4., 1., 0., 0., 2.]);
    let x = truth(3);
    close(
        &dense::upper_solve(&r, &r.mul_vec(&x).unwrap(), 1e-14).unwrap(),
        &x,
        1e-13,
    );
    let mut a = Matrix::identity(2);
    dense::rotate_rows(&mut a, 0, 1, 0.6, 0.8).unwrap();
    same(&a.transpose().mul(&a).unwrap(), &Matrix::identity(2), 1e-14);
}
#[test]
fn thomas_and_singleton() {
    close(&tridiagonal(&[], &[2.], &[], &[4.]).unwrap(), &[2.], 0.);
    for n in 2..14 {
        let mut a = Matrix::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = 4.;
            if i + 1 < n {
                a[(i, i + 1)] = 2.;
                a[(i + 1, i)] = -1.;
            }
        }
        let x = truth(n);
        close(
            &tridiagonal(
                &vec![-1.; n - 1],
                &vec![4.; n],
                &vec![2.; n - 1],
                &a.mul_vec(&x).unwrap(),
            )
            .unwrap(),
            &x,
            1e-12,
        );
    }
}
#[test]
fn cyclic_asymmetric_corners() {
    for n in 3..15 {
        let mut a = Matrix::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = 5.;
            if i + 1 < n {
                a[(i, i + 1)] = 1.;
                a[(i + 1, i)] = -2.;
            }
        }
        a[(0, n - 1)] = 0.7;
        a[(n - 1, 0)] = -0.4;
        let x = truth(n);
        close(
            &cyclic_tridiagonal(
                &vec![-2.; n - 1],
                &vec![5.; n],
                &vec![1.; n - 1],
                0.7,
                -0.4,
                &a.mul_vec(&x).unwrap(),
            )
            .unwrap(),
            &x,
            1e-12,
        );
    }
}
#[test]
fn band_product_and_factorization() {
    for n in 3..15 {
        for (lo, hi) in [(0, 0), (1, 2), (2, 1)] {
            let mut a = Matrix::zeros(n, n);
            for i in 0..n {
                for j in i.saturating_sub(lo)..(i + hi + 1).min(n) {
                    a[(i, j)] = if i == j {
                        5.
                    } else {
                        ((i * 3 + j + 1) as f64).cos()
                    };
                }
            }
            let band = BandMatrix::from_dense(&a, lo, hi).unwrap();
            let x = truth(n);
            let b = a.mul_vec(&x).unwrap();
            close(&band.mul_vec(&x).unwrap(), &b, 1e-14);
            close(&band.decompose().unwrap().solve(&b).unwrap(), &x, 1e-12);
        }
    }
}
#[test]
fn band_requires_pivoting_and_fill() {
    let a = mat(
        4,
        4,
        &[
            0., 2., 1., 0., 3., 1., 2., 1., 0., 2., 3., 2., 0., 0., 4., 1.,
        ],
    );
    let band = BandMatrix::from_dense(&a, 1, 2).unwrap();
    let x = truth(4);
    let f = band.decompose().unwrap();
    close(&f.solve(&a.mul_vec(&x).unwrap()).unwrap(), &x, 1e-12);
    assert!((f.determinant() - Lu::decompose(&a).unwrap().determinant()).abs() < 1e-10);
}
#[test]
fn vandermonde_moment_orientation() {
    let nodes = [-2., -0.5, 0.3, 1.2, 2.];
    let x = truth(5);
    let b = (0..5)
        .map(|i| {
            nodes
                .iter()
                .zip(&x)
                .map(|(&t, &v)| f64::powi(t, i) * v)
                .sum()
        })
        .collect::<Vec<_>>();
    close(&vandermonde(&nodes, &b).unwrap(), &x, 1e-11);
    assert_eq!(vandermonde(&[1., 1.], &[1., 2.]), Err(Error::Singular));
}
#[test]
fn nonsymmetric_toeplitz_recurrence() {
    for n in 1..20 {
        let mut col = (0..n)
            .map(|i| ((i + 1) as f64).sin() / 3.)
            .collect::<Vec<_>>();
        let mut row = (0..n)
            .map(|i| ((i + 2) as f64).cos() / 2.)
            .collect::<Vec<_>>();
        col[0] = n as f64 + 2.;
        row[0] = col[0];
        let mut a = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = if i >= j { col[i - j] } else { row[j - i] };
            }
        }
        let x = truth(n);
        close(
            &toeplitz(&col, &row, &a.mul_vec(&x).unwrap()).unwrap(),
            &x,
            1e-11,
        );
    }
}
#[test]
fn toeplitz_breakdown_nonsingular() {
    assert_eq!(
        toeplitz(&[0., 1.], &[0., 2.], &[2., 1.]),
        Err(Error::Breakdown)
    );
}

#[test]
fn bicg_all_stopping_modes() {
    let a = generated(8);
    let s = Csr::from_dense(&a, 0.).unwrap();
    let x = truth(8);
    let b = a.mul_vec(&x).unwrap();
    for c in [
        StopCriterion::ResidualL2,
        StopCriterion::PreconditionedL2,
        StopCriterion::EstimatedErrorL2,
        StopCriterion::EstimatedErrorInf,
        StopCriterion::ResidualInf,
        StopCriterion::PreconditionedInf,
    ] {
        let r = bicg(&s, &b, &[0.; 8], 1e-11, 100, c).unwrap();
        close(&r.x, &x, 1e-10);
        assert!(r.relative_residual < 1e-10);
    }
}
#[test]
fn bicg_zero_rhs_and_iteration_limit() {
    let s = Csr::from_dense(&sample(), 0.).unwrap();
    assert_eq!(
        bicg(&s, &[0.; 3], &[0.; 3], 1e-12, 10, StopCriterion::ResidualL2)
            .unwrap()
            .iterations,
        0
    );
    assert!(matches!(
        bicg(&s, &[1.; 3], &[0.; 3], 1e-12, 0, StopCriterion::ResidualL2),
        Err(Error::NoConvergence { .. })
    ));
}