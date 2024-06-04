use std::ops::Mul;

use super::Order;

use super::constant::ConstantMatrix;
use super::dense::DenseMatrix;
use super::diagonal::DiagonalMatrix;
use super::identity::IdentityMatrix;
use super::zero::ZeroMatrix;

#[derive(Clone, Debug)]
pub struct SparseMatrix<const R: usize, const C: usize>(
    pub(super) Vec<usize>,
    pub(super) Vec<f32>,
    pub(super) Order,
);

impl<const R: usize, const C: usize> SparseMatrix<R, C> {
    pub(super) fn T(&self) -> SparseMatrix<C, R> {
        todo!() //TODO
    }
}

impl<const R: usize, const C: usize> PartialEq for SparseMatrix<R, C> where [(); R*C]: Sized {
    fn eq(&self, _other: &Self) -> bool {
        todo!() // TODO: handle transposed args.
    }
}

///////////////////////////////
/// SPARSE MATRIX MUL IMPLS ///
///////////////////////////////

impl<const R: usize, const C: usize, const C2: usize> Mul<&ConstantMatrix<C, C2>> for &SparseMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
        [(); R*C2]: Sized,
{
    type Output = SparseMatrix<R, C2>;

    fn mul(self, _rhs: &ConstantMatrix<C, C2>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&DenseMatrix<C, C2>> for &SparseMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
        [(); R*C2]: Sized,
{
    type Output = DenseMatrix<R, C2>;

    fn mul(self, _rhs: &DenseMatrix<C, C2>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&DiagonalMatrix<C, C2>> for &SparseMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
        [(); R*C2]: Sized,
{
    type Output = SparseMatrix<R, C2>;

    fn mul(self, _rhs: &DiagonalMatrix<C, C2>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&IdentityMatrix<C, C2>> for &SparseMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
        [(); R*C2]: Sized,
{
    type Output = SparseMatrix<R, C2>;

    fn mul(self, _rhs: &IdentityMatrix<C, C2>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&SparseMatrix<C, C2>> for &SparseMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
        [(); R*C2]: Sized,
{
    type Output = SparseMatrix<R, C2>;

    fn mul(self, _rhs: &SparseMatrix<C, C2>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&ZeroMatrix<C, C2>> for &SparseMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
{
    type Output = ZeroMatrix<R, C2>;

    fn mul(self, _rhs: &ZeroMatrix<C, C2>) -> Self::Output {
        todo!()
    }
}