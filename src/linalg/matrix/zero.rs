use std::fmt::Display;
use std::ops::{Add, Mul, Neg, Sub};

use super::constant::ConstantMatrix;
use super::dense::DenseMatrix;
use super::diagonal::DiagonalMatrix;
use super::identity::IdentityMatrix;
use super::sparse::SparseMatrix;

#[derive(Clone, Debug)]
pub struct ZeroMatrix<const R: usize, const C: usize>(
    pub(super) f32,
);

impl<const R: usize, const C: usize> ZeroMatrix<R, C> {
    pub(super) fn T(&self) -> ZeroMatrix<C, R> {
        ZeroMatrix(self.0)
    }
}

impl<const R: usize, const C: usize> PartialEq for ZeroMatrix<R, C> where [(); R*C]: Sized {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

///////////////////////////////
/// SPARSE MATRIX ADD IMPLS ///
///////////////////////////////

impl<const R: usize, const C: usize> Add<&ConstantMatrix<R, C>> for &ZeroMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = ConstantMatrix<R, C>;

    fn add(self, _rhs: &ConstantMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Add<&DenseMatrix<R, C>> for &ZeroMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = DenseMatrix<R, C>;

    fn add(self, _rhs: &DenseMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Add<&DiagonalMatrix<R, C>> for &ZeroMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = DiagonalMatrix<R, C>;

    fn add(self, _rhs: &DiagonalMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Add<&IdentityMatrix<R, C>> for &ZeroMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = IdentityMatrix<R, C>;

    fn add(self, _rhs: &IdentityMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Add<&SparseMatrix<R, C>> for &ZeroMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = SparseMatrix<R, C>;

    fn add(self, _rhs: &SparseMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Add<&ZeroMatrix<R, C>> for &ZeroMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = ZeroMatrix<R, C>;

    fn add(self, _rhs: &ZeroMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

///////////////////////////////
/// SPARSE MATRIX SUB IMPLS ///
///////////////////////////////

impl<const R: usize, const C: usize> Sub<&ConstantMatrix<R, C>> for &ZeroMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = ConstantMatrix<R, C>;

    fn sub(self, _rhs: &ConstantMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Sub<&DenseMatrix<R, C>> for &ZeroMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = DenseMatrix<R, C>;

    fn sub(self, _rhs: &DenseMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Sub<&DiagonalMatrix<R, C>> for &ZeroMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = DiagonalMatrix<R, C>;

    fn sub(self, _rhs: &DiagonalMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Sub<&IdentityMatrix<R, C>> for &ZeroMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = DiagonalMatrix<R, C>;

    fn sub(self, _rhs: &IdentityMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Sub<&SparseMatrix<R, C>> for &ZeroMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = SparseMatrix<R, C>;

    fn sub(self, _rhs: &SparseMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Sub<&ZeroMatrix<R, C>> for &ZeroMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = ZeroMatrix<R, C>;

    fn sub(self, _rhs: &ZeroMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

/////////////////////////////
/// ZERO MATRIX MUL IMPLS ///
/////////////////////////////

impl<const R: usize, const C: usize, const C2: usize> Mul<&ConstantMatrix<C, C2>> for &ZeroMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
        [(); R*C2]: Sized,
{
    type Output = ZeroMatrix<R, C2>;

    fn mul(self, _rhs: &ConstantMatrix<C, C2>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&DenseMatrix<C, C2>> for &ZeroMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
        [(); R*C2]: Sized,
{
    type Output = ZeroMatrix<R, C2>;

    fn mul(self, _rhs: &DenseMatrix<C, C2>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&DiagonalMatrix<C, C2>> for &ZeroMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
        [(); R*C2]: Sized,
{
    type Output = ZeroMatrix<R, C2>;

    fn mul(self, _rhs: &DiagonalMatrix<C, C2>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&IdentityMatrix<C, C2>> for &ZeroMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
        [(); R*C2]: Sized,
{
    type Output = ZeroMatrix<R, C2>;

    fn mul(self, _rhs: &IdentityMatrix<C, C2>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&SparseMatrix<C, C2>> for &ZeroMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
        [(); R*C2]: Sized,
{
    type Output = ZeroMatrix<R, C2>;

    fn mul(self, _rhs: &SparseMatrix<C, C2>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&ZeroMatrix<C, C2>> for &ZeroMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
{
    type Output = ZeroMatrix<R, C2>;

    fn mul(self, _rhs: &ZeroMatrix<C, C2>) -> Self::Output {
        todo!()
    }
}

//////////////////////////////
/// ZERO MATRIX MATH IMPLS ///
//////////////////////////////

impl<const R: usize, const C: usize> Add<&ZeroMatrix<R, C>> for f32 where [(); R*C]: Sized {
    type Output = ConstantMatrix<R, C>;

    fn add(self, _rhs: &ZeroMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Add<f32> for &ZeroMatrix<R, C> where [(); R*C]: Sized {
    type Output = ConstantMatrix<R, C>;

    fn add(self, _rhs: f32) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Sub<f32> for &ZeroMatrix<R, C> where [(); R*C]: Sized {
    type Output = ConstantMatrix<R, C>;

    fn sub(self, _rhs: f32) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Mul<&ZeroMatrix<R, C>> for f32 where [(); R*C]: Sized {
    type Output = ZeroMatrix<R, C>;

    fn mul(self, _rhs: &ZeroMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Mul<f32> for &ZeroMatrix<R, C> where [(); R*C]: Sized {
    type Output = ZeroMatrix<R, C>;

    fn mul(self, _rhs: f32) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Neg for &ZeroMatrix<R, C> where [(); R*C]: Sized {
    type Output = ZeroMatrix<R, C>;

    fn neg(self) -> Self::Output {
        todo!()
    }
}

/////////////////////////////////
/// ZERO MATRIX UTILITY IMPLS ///
/////////////////////////////////

impl<const R: usize, const C: usize> Display for ZeroMatrix<R, C> where [(); R*C]: Sized {
    /// Displays the rows of a zero matrix.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]", (0..R).into_iter()
            .map(|_| std::iter::repeat("0")
                .take(C)
                .collect::<Vec<_>>()
                .join(","))
            .collect::<Vec<_>>()
            .join("],["))
    }
}
