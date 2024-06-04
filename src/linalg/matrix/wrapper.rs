use std::ops::Mul;

use crate::linalg::vector::DenseVector;
use crate::linalg::vector::VectorWrapper;

use super::Order;
use super::constant::ConstantMatrix;
use super::dense::DenseMatrix;
use super::diagonal::DiagonalMatrix;
use super::identity::IdentityMatrix;
use super::sparse::SparseMatrix;
use super::zero::ZeroMatrix;

#[derive(Clone, Debug)]
pub enum MatrixWrapper<const R: usize, const C: usize> where [(); R*C]: Sized {
    Constant(ConstantMatrix<R, C>),
    Dense(DenseMatrix<R, C>),
    Diagonal(DiagonalMatrix<R, C>),
    Identity(IdentityMatrix<R, C>),
    Sparse(SparseMatrix<R, C>),
    Zero(ZeroMatrix<R, C>),
}

impl<const R: usize, const C: usize> MatrixWrapper<R, C> where [(); R*C]: Sized {
    // constructor
    pub fn constant(c: f32) -> Self {
        Self::Constant(ConstantMatrix(c))
    }

    // constructor
    pub fn sparse() -> Self {
        Self::Sparse(SparseMatrix(Vec::new(), Vec::new(), Order::COLUMNS))
    }

    // constructor
    pub fn zero() -> Self {
        Self::Zero(ZeroMatrix(0f32))
    }

    /// Matrix transpose.
    pub fn T(&self) -> MatrixWrapper<C, R>
        where
            [(); R*C]: Sized,
            [(); C*R]: Sized,
    {
        use MatrixWrapper as M;
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

impl<const D: usize> MatrixWrapper<D, D> where [(); D*D]: Sized {
    pub fn diagonal(v: VectorWrapper<D>) -> Self {
        Self::Diagonal(DiagonalMatrix(v))
    }
    
    pub fn identity() -> Self {
        Self::Identity(IdentityMatrix(0f32, 1f32))
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&MatrixWrapper<C, C2>> for &MatrixWrapper<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
        [(); R*C2]: Sized,
{
    type Output = MatrixWrapper<R, C2>;
    
    // Some matrix flavors can only be instantiated as square, but we must provide
    // rectangular impls for all of them to satisfy impl coherence.
    fn mul(self, rhs: &MatrixWrapper<C, C2>) -> Self::Output {
        use MatrixWrapper as M;
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

impl<const R: usize, const C: usize> Mul<&VectorWrapper<C>> for &MatrixWrapper<R, C> where [(); R*C]: Sized {
    type Output = VectorWrapper<R>;
    
    fn mul(self, rhs: &VectorWrapper<C>) -> Self::Output {
        match (self, rhs) {
            //TODO: matrix-vector mul impls.
            _ => unimplemented!(),
        }
    }
}

impl<const R: usize, const C: usize> PartialEq for MatrixWrapper<R, C> where [(); R*C]: Sized {
    fn eq(&self, other: &Self) -> bool {
        use MatrixWrapper as M;
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
