use std::ops::{Add, Mul, Neg, Sub};

use crate::linalg::vector::Vector;

use super::Order;
use super::constant::ConstantMatrix;
use super::dense::DenseMatrix;
use super::diagonal::DiagonalMatrix;
use super::identity::IdentityMatrix;
use super::sparse::SparseMatrix;
use super::zero::ZeroMatrix;

#[derive(Clone, Debug)]
pub enum Matrix<const R: usize, const C: usize> where [(); R*C]: Sized {
    Constant(ConstantMatrix<R, C>),
    Dense(DenseMatrix<R, C>),
    Diagonal(DiagonalMatrix<R, C>),
    Identity(IdentityMatrix<R, C>),
    Sparse(SparseMatrix<R, C>),
    Zero(ZeroMatrix<R, C>),
}

impl<const R: usize, const C: usize> Matrix<R, C> where [(); R*C]: Sized {
    // constructor
    pub fn constant(c: f32) -> Self {
        Self::Constant(ConstantMatrix(c))
    }

    // constructor
    pub fn from_arr(arr: [f32; R*C]) -> Self {
        Self::Dense(DenseMatrix::from_arr(arr))
    }

    // constructor
    pub fn from_cols(cols: &[[f32; R]; C]) -> Self {
        Self::Dense(DenseMatrix::from_cols(cols))
    }

    // constructor
    pub fn sparse() -> Self {
        Self::Sparse(SparseMatrix(Vec::new(), Vec::new(), Order::COLS))
    }

    // constructor
    pub fn zero() -> Self {
        Self::Zero(ZeroMatrix(0f32))
    }

    /// Matrix transpose.
    pub fn T(&self) -> Matrix<C, R> where [(); C*R]: Sized {
        use Matrix as M;
        match self {
            M::Constant(m) => M::Constant::<C, R>(m.T()),
            M::Dense(m) => M::Dense(m.T()),
            M::Diagonal(m) => M::Diagonal(m.T()),
            M::Identity(m) => M::Identity(m.T()),
            M::Sparse(m) => M::Sparse(m.T()),
            M::Zero(m) => M::Zero(m.T()),
        }
    }
}

impl<const D: usize> Matrix<D, D> where [(); D*D]: Sized {
    pub fn diag(main_diag: Vector<D>) -> Self {
        Self::Diagonal(main_diag.into())
    }
    
    pub fn I() -> Self {
        Self::Identity(IdentityMatrix(0f32, 1f32))
    }
}

impl<const R: usize, const C: usize> Add<&Matrix<R, C>> for &Matrix<R, C> where [(); R*C]: Sized {
    type Output = Matrix<R, C>;

    // Some matrix flavors can only be instantiated as square, but we must provide
    // rectangular impls for all of them to satisfy impl coherence.
    fn add(self, rhs: &Matrix<R, C>) -> Self::Output {
        use Matrix as M;
        match (self, rhs) {
            (M::Constant(m1), M::Constant(m2)) => M::Constant(m1 + m2),
            (M::Constant(m1), M::Dense(m2)) => M::Dense(m1 + m2),
            (M::Constant(m1), M::Diagonal(m2)) => M::Constant(m1 + m2),
            (M::Constant(m1), M::Identity(m2)) => M::Constant(m1 + m2),
            (M::Constant(m1), M::Sparse(m2)) => M::Sparse(m1 + m2),
            (M::Constant(m1), M::Zero(m2)) => M::Zero(m1 + m2),

            (M::Dense(m1), M::Constant(m2)) => M::Dense(m1 + m2),
            (M::Dense(m1), M::Dense(m2)) => M::Dense(m1 + m2),
            (M::Dense(m1), M::Diagonal(m2)) => M::Dense(m1 + m2),
            (M::Dense(m1), M::Identity(m2)) => M::Dense(m1 + m2),
            (M::Dense(m1), M::Sparse(m2)) => M::Dense(m1 + m2),
            (M::Dense(m1), M::Zero(m2)) => M::Dense(m1 + m2),

            (M::Diagonal(m1), M::Constant(m2)) => M::Dense(m1 + m2),
            (M::Diagonal(m1), M::Dense(m2)) => M::Dense(m1 + m2),
            (M::Diagonal(m1), M::Diagonal(m2)) => M::Diagonal(m1 + m2),
            (M::Diagonal(m1), M::Identity(m2)) => M::Diagonal(m1 + m2),
            (M::Diagonal(m1), M::Sparse(m2)) => M::Sparse(m1 + m2),
            (M::Diagonal(m1), M::Zero(m2)) => M::Diagonal(m1 + m2),

            (M::Identity(m1), M::Constant(m2)) => M::Dense(m1 + m2),
            (M::Identity(m1), M::Dense(m2)) => M::Dense(m1 + m2),
            (M::Identity(m1), M::Diagonal(m2)) => M::Diagonal(m1 + m2),
            (M::Identity(m1), M::Identity(m2)) => M::Diagonal(m1 + m2),
            (M::Identity(m1), M::Sparse(m2)) => M::Sparse(m1 + m2),
            (M::Identity(m1), M::Zero(m2)) => M::Identity(m1 + m2),

            (M::Sparse(m1), M::Constant(m2)) => M::Dense(m1 + m2),
            (M::Sparse(m1), M::Dense(m2)) => M::Dense(m1 + m2),
            (M::Sparse(m1), M::Diagonal(m2)) => M::Sparse(m1 + m2),
            (M::Sparse(m1), M::Identity(m2)) => M::Sparse(m1 + m2),
            (M::Sparse(m1), M::Sparse(m2)) => M::Sparse(m1 + m2),
            (M::Sparse(m1), M::Zero(m2)) => M::Sparse(m1 + m2),

            (M::Zero(m1), M::Constant(m2)) => M::Constant(m1 + m2),
            (M::Zero(m1), M::Dense(m2)) => M::Dense(m1 + m2),
            (M::Zero(m1), M::Diagonal(m2)) => M::Diagonal(m1 + m2),
            (M::Zero(m1), M::Identity(m2)) => M::Identity(m1 + m2),
            (M::Zero(m1), M::Sparse(m2)) => M::Sparse(m1 + m2),
            (M::Zero(m1), M::Zero(m2)) => M::Zero(m1 + m2),
        }
    }
}

impl<const R: usize, const C: usize> Sub<&Matrix<R, C>> for &Matrix<R, C> where [(); R*C]: Sized {
    type Output = Matrix<R, C>;

    // Some matrix flavors can only be instantiated as square, but we must provide
    // rectangular impls for all of them to satisfy impl coherence.
    fn sub(self, rhs: &Matrix<R, C>) -> Self::Output {
        use Matrix as M;
        match (self, rhs) {
            (M::Constant(m1), M::Constant(m2)) => M::Constant(m1 - m2),
            (M::Constant(m1), M::Dense(m2)) => M::Dense(m1 - m2),
            (M::Constant(m1), M::Diagonal(m2)) => M::Constant(m1 - m2),
            (M::Constant(m1), M::Identity(m2)) => M::Constant(m1 - m2),
            (M::Constant(m1), M::Sparse(m2)) => M::Sparse(m1 - m2),
            (M::Constant(m1), M::Zero(m2)) => M::Zero(m1 - m2),

            (M::Dense(m1), M::Constant(m2)) => M::Dense(m1 - m2),
            (M::Dense(m1), M::Dense(m2)) => M::Dense(m1 - m2),
            (M::Dense(m1), M::Diagonal(m2)) => M::Dense(m1 - m2),
            (M::Dense(m1), M::Identity(m2)) => M::Dense(m1 - m2),
            (M::Dense(m1), M::Sparse(m2)) => M::Dense(m1 - m2),
            (M::Dense(m1), M::Zero(m2)) => M::Dense(m1 - m2),

            (M::Diagonal(m1), M::Constant(m2)) => M::Dense(m1 - m2),
            (M::Diagonal(m1), M::Dense(m2)) => M::Dense(m1 - m2),
            (M::Diagonal(m1), M::Diagonal(m2)) => M::Diagonal(m1 - m2),
            (M::Diagonal(m1), M::Identity(m2)) => M::Diagonal(m1 - m2),
            (M::Diagonal(m1), M::Sparse(m2)) => M::Sparse(m1 - m2),
            (M::Diagonal(m1), M::Zero(m2)) => M::Diagonal(m1 - m2),

            (M::Identity(m1), M::Constant(m2)) => M::Dense(m1 - m2),
            (M::Identity(m1), M::Dense(m2)) => M::Dense(m1 - m2),
            (M::Identity(m1), M::Diagonal(m2)) => M::Diagonal(m1 - m2),
            (M::Identity(m1), M::Identity(m2)) => M::Zero(m1 - m2),
            (M::Identity(m1), M::Sparse(m2)) => M::Sparse(m1 - m2),
            (M::Identity(m1), M::Zero(m2)) => M::Identity(m1 - m2),

            (M::Sparse(m1), M::Constant(m2)) => M::Dense(m1 - m2),
            (M::Sparse(m1), M::Dense(m2)) => M::Dense(m1 - m2),
            (M::Sparse(m1), M::Diagonal(m2)) => M::Sparse(m1 - m2),
            (M::Sparse(m1), M::Identity(m2)) => M::Sparse(m1 - m2),
            (M::Sparse(m1), M::Sparse(m2)) => M::Sparse(m1 - m2),
            (M::Sparse(m1), M::Zero(m2)) => M::Sparse(m1 - m2),

            (M::Zero(m1), M::Constant(m2)) => M::Constant(m1 - m2),
            (M::Zero(m1), M::Dense(m2)) => M::Dense(m1 - m2),
            (M::Zero(m1), M::Diagonal(m2)) => M::Diagonal(m1 - m2),
            (M::Zero(m1), M::Identity(m2)) => M::Diagonal(m1 - m2),
            (M::Zero(m1), M::Sparse(m2)) => M::Sparse(m1 - m2),
            (M::Zero(m1), M::Zero(m2)) => M::Zero(m1 - m2),
        }
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&Matrix<C, C2>> for &Matrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
        [(); R*C2]: Sized,
{
    type Output = Matrix<R, C2>;
    
    // Some matrix flavors can only be instantiated as square, but we must provide
    // rectangular impls for all of them to satisfy impl coherence.
    fn mul(self, rhs: &Matrix<C, C2>) -> Self::Output {
        use Matrix as M;
        match (self, rhs) {
            (M::Constant(m1), M::Constant(m2)) => M::Constant(m1 * m2),
            (M::Constant(m1), M::Dense(m2)) => M::Dense(m1 * m2),
            (M::Constant(m1), M::Diagonal(m2)) => M::Constant(m1 * m2),
            (M::Constant(m1), M::Identity(m2)) => M::Constant(m1 * m2),
            (M::Constant(m1), M::Sparse(m2)) => M::Sparse(m1 * m2),
            (M::Constant(m1), M::Zero(m2)) => M::Zero(m1 * m2),

            (M::Dense(m1), M::Constant(m2)) => M::Dense(m1 * m2),
            (M::Dense(m1), M::Dense(m2)) => M::Dense(m1 * m2),
            (M::Dense(m1), M::Diagonal(m2)) => M::Dense(m1 * m2),
            (M::Dense(m1), M::Identity(m2)) => M::Dense(m1 * m2),
            (M::Dense(m1), M::Sparse(m2)) => M::Dense(m1 * m2),
            (M::Dense(m1), M::Zero(m2)) => M::Zero(m1 * m2),

            (M::Diagonal(m1), M::Constant(m2)) => M::Constant(m1 * m2),
            (M::Diagonal(m1), M::Dense(m2)) => M::Dense(m1 * m2),
            (M::Diagonal(m1), M::Diagonal(m2)) => M::Diagonal(m1 * m2),
            (M::Diagonal(m1), M::Identity(m2)) => M::Diagonal(m1 * m2),
            (M::Diagonal(m1), M::Sparse(m2)) => M::Sparse(m1 * m2),
            (M::Diagonal(m1), M::Zero(m2)) => M::Zero(m1 * m2),

            (M::Identity(m1), M::Constant(m2)) => M::Constant(m1 * m2),
            (M::Identity(m1), M::Dense(m2)) => M::Dense(m1 * m2),
            (M::Identity(m1), M::Diagonal(m2)) => M::Diagonal(m1 * m2),
            (M::Identity(m1), M::Identity(m2)) => M::Identity(m1 * m2),
            (M::Identity(m1), M::Sparse(m2)) => M::Sparse(m1 * m2),
            (M::Identity(m1), M::Zero(m2)) => M::Zero(m1 * m2),

            (M::Sparse(m1), M::Constant(m2)) => M::Sparse(m1 * m2),
            (M::Sparse(m1), M::Dense(m2)) => M::Dense(m1 * m2),
            (M::Sparse(m1), M::Diagonal(m2)) => M::Sparse(m1 * m2),
            (M::Sparse(m1), M::Identity(m2)) => M::Sparse(m1 * m2),
            (M::Sparse(m1), M::Sparse(m2)) => M::Sparse(m1 * m2),
            (M::Sparse(m1), M::Zero(m2)) => M::Zero(m1 * m2),

            (M::Zero(m1), M::Constant(m2)) => M::Zero(m1 * m2),
            (M::Zero(m1), M::Dense(m2)) => M::Zero(m1 * m2),
            (M::Zero(m1), M::Diagonal(m2)) => M::Zero(m1 * m2),
            (M::Zero(m1), M::Identity(m2)) => M::Zero(m1 * m2),
            (M::Zero(m1), M::Sparse(m2)) => M::Zero(m1 * m2),
            (M::Zero(m1), M::Zero(m2)) => M::Zero(m1 * m2),
        }
    }
}

impl<const R: usize, const C: usize> Mul<&Matrix<R, C>> for f32 where [(); R*C]: Sized {
    type Output = Matrix<R, C>;

    fn mul(self, rhs: &Matrix<R, C>) -> Self::Output {
        use Matrix as M;
        match rhs {
            M::Constant(m) => M::Constant(m * self),
            M::Dense(m) => M::Dense(m * self),
            M::Diagonal(m) => M::Diagonal(m * self),
            M::Identity(m) => M::Diagonal(m * self),
            M::Sparse(m) => M::Sparse(m * self),
            M::Zero(m) => M::Zero(m * self),
        }
    }
}

impl<const R: usize, const C: usize> Neg for &Matrix<R, C> where [(); R*C]: Sized {
    type Output = Matrix<R, C>;

    fn neg(self) -> Self::Output {
        use Matrix as M;
        match self {
            M::Constant(m) => M::Constant(-m),
            M::Dense(m) => M::Dense(-m),
            M::Diagonal(m) => M::Diagonal(-m),
            M::Identity(m) => M::Diagonal(-m),
            M::Sparse(m) => M::Sparse(-m),
            M::Zero(m) => M::Zero(-m), // This is just clone.
        }
    }
}

impl<const R: usize, const C: usize> Mul<&Vector<C>> for &Matrix<R, C> where [(); R*C]: Sized {
    type Output = Vector<R>;
    
    fn mul(self, rhs: &Vector<C>) -> Self::Output {
        use Matrix as M;
        use Vector as V;
        match (self, rhs) {
            (M::Dense(m), V::Dense(v)) => V::Dense(m * v),
            (M::Diagonal(m), V::Dense(v)) => V::Dense(m * v),
            // TODO: matrix-vector mul impls.
            _ => unimplemented!(),
        }
    }
}

impl<const R: usize, const C: usize> PartialEq for Matrix<R, C> where [(); R*C]: Sized {
    fn eq(&self, other: &Self) -> bool {
        use Matrix as M;
        match (self, other) {
            (M::Constant(m1), M::Constant(m2)) => m1 == m2,
            (M::Dense(m1), M::Dense(m2)) => m1 == m2,
            (M::Diagonal(m1), M::Diagonal(m2)) => m1 == m2,
            (M::Identity(m1), M::Identity(m2)) => m1 == m2,
            (M::Sparse(m1), M::Sparse(m2)) => m1 == m2,
            (M::Zero(m1), M::Zero(m2)) => m1 == m2,
            _ => false, // TODO: equality between mixed flavors.
        }
    }
}

impl<const R: usize, const C: usize> From<[f32; R*C]> for Matrix<R, C> where [(); R*C]: Sized {
    fn from(arr: [f32; R*C]) -> Self {
        Matrix::Dense(DenseMatrix::from_arr(arr))
    }
}
