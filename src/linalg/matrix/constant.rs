use std::fmt::Display;
use std::ops::{Add, Mul, Neg, Sub};

use super::dense::DenseMatrix;
use super::diagonal::DiagonalMatrix;
use super::identity::IdentityMatrix;
use super::sparse::SparseMatrix;
use super::zero::ZeroMatrix;

#[derive(Clone, Debug)]
pub struct ConstantMatrix<const R: usize, const C: usize>(
    pub(super) f32,
);

impl<const R: usize, const C: usize> ConstantMatrix<R, C> {
    pub(super) fn T(&self) -> ConstantMatrix<C, R> {
        ConstantMatrix(self.0)
    }
}

impl<const R: usize, const C: usize> PartialEq for ConstantMatrix<R, C> where [(); R*C]: Sized {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

/////////////////////////////////
/// CONSTANT MATRIX ADD IMPLS ///
/////////////////////////////////

impl<const R: usize, const C: usize> Add<&ConstantMatrix<R, C>> for &ConstantMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = ConstantMatrix<R, C>;

    fn add(self, _rhs: &ConstantMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Add<&DenseMatrix<R, C>> for &ConstantMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = DenseMatrix<R, C>;

    fn add(self, _rhs: &DenseMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Add<&DiagonalMatrix<R, C>> for &ConstantMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = ConstantMatrix<R, C>;

    fn add(self, _rhs: &DiagonalMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Add<&IdentityMatrix<R, C>> for &ConstantMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = ConstantMatrix<R, C>;

    fn add(self, _rhs: &IdentityMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Add<&SparseMatrix<R, C>> for &ConstantMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = SparseMatrix<R, C>;

    fn add(self, _rhs: &SparseMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Add<&ZeroMatrix<R, C>> for &ConstantMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = ZeroMatrix<R, C>;

    fn add(self, _rhs: &ZeroMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

/////////////////////////////////
/// CONSTANT MATRIX SUB IMPLS ///
/////////////////////////////////

impl<const R: usize, const C: usize> Sub<&ConstantMatrix<R, C>> for &ConstantMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = ConstantMatrix<R, C>;

    fn sub(self, _rhs: &ConstantMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Sub<&DenseMatrix<R, C>> for &ConstantMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = DenseMatrix<R, C>;

    fn sub(self, _rhs: &DenseMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Sub<&DiagonalMatrix<R, C>> for &ConstantMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = ConstantMatrix<R, C>;

    fn sub(self, _rhs: &DiagonalMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Sub<&IdentityMatrix<R, C>> for &ConstantMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = ConstantMatrix<R, C>;

    fn sub(self, _rhs: &IdentityMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Sub<&SparseMatrix<R, C>> for &ConstantMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = SparseMatrix<R, C>;

    fn sub(self, _rhs: &SparseMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Sub<&ZeroMatrix<R, C>> for &ConstantMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = ZeroMatrix<R, C>;

    fn sub(self, _rhs: &ZeroMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

/////////////////////////////////
/// CONSTANT MATRIX MUL IMPLS ///
/////////////////////////////////

impl<const R: usize, const C: usize, const C2: usize> Mul<&ConstantMatrix<C, C2>> for &ConstantMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
{
    type Output = ConstantMatrix<R, C2>;

    fn mul(self, _rhs: &ConstantMatrix<C, C2>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&DenseMatrix<C, C2>> for &ConstantMatrix<R, C>
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

impl<const R: usize, const C: usize, const C2: usize> Mul<&DiagonalMatrix<C, C2>> for &ConstantMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
        [(); R*C]: Sized,
{
    type Output = ConstantMatrix<R, C2>;

    fn mul(self, _rhs: &DiagonalMatrix<C, C2>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&IdentityMatrix<C, C2>> for &ConstantMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
        [(); R*C2]: Sized,
{
    type Output = ConstantMatrix<R, C2>;

    fn mul(self, _rhs: &IdentityMatrix<C, C2>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&SparseMatrix<C, C2>> for &ConstantMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
{
    type Output = SparseMatrix<R, C2>;

    fn mul(self, _rhs: &SparseMatrix<C, C2>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&ZeroMatrix<C, C2>> for &ConstantMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
{
    type Output = ZeroMatrix<R, C2>;

    fn mul(self, _rhs: &ZeroMatrix<C, C2>) -> Self::Output {
        todo!()
    }
}

//////////////////////////////////
/// CONSTANT MATRIX MATH IMPLS ///
//////////////////////////////////

impl<const R: usize, const C: usize> Add<&ConstantMatrix<R, C>> for f32 where [(); R*C]: Sized {
    type Output = ConstantMatrix<R, C>;

    fn add(self, _rhs: &ConstantMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Add<f32> for &ConstantMatrix<R, C> where [(); R*C]: Sized {
    type Output = ConstantMatrix<R, C>;

    fn add(self, _rhs: f32) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Sub<f32> for &ConstantMatrix<R, C> where [(); R*C]: Sized {
    type Output = ConstantMatrix<R, C>;

    fn sub(self, _rhs: f32) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Mul<&ConstantMatrix<R, C>> for f32 where [(); R*C]: Sized {
    type Output = ConstantMatrix<R, C>;

    fn mul(self, _rhs: &ConstantMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Mul<f32> for &ConstantMatrix<R, C> where [(); R*C]: Sized {
    type Output = ConstantMatrix<R, C>;

    fn mul(self, _rhs: f32) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Neg for &ConstantMatrix<R, C> where [(); R*C]: Sized {
    type Output = ConstantMatrix<R, C>;

    fn neg(self) -> Self::Output {
        todo!()
    }
}

/////////////////////////////////////
/// CONSTANT MATRIX UTILITY IMPLS ///
/////////////////////////////////////

impl<const R: usize, const C: usize> Display for ConstantMatrix<R, C> where [(); R*C]: Sized {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]", (0..R).into_iter()
            .map(|_| std::iter::repeat(&self.0)
                .map(<f32>::to_string)
                .take(C)
                .collect::<Vec<_>>()
                .join(","))
            .collect::<Vec<_>>()
            .join("],["))
    }
}
