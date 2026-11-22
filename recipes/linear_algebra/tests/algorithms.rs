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