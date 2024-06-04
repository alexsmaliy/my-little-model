use std::fmt::Display;
use std::ops::{Add, Mul, Neg, Sub};

use crate::linalg::vector::VectorWrapper;

use super::constant::ConstantMatrix;
use super::dense::DenseMatrix;
use super::identity::IdentityMatrix;
use super::sparse::SparseMatrix;
use super::zero::ZeroMatrix;

#[derive(Clone, Debug)]
pub struct DiagonalMatrix<const R: usize, const C: usize>(
    pub(super) VectorWrapper<R>,
) where [(); R*C]: Sized;

// Impl is provided for possibly unequal R and C,
// even though only square diagonal matrices can be instantiated.
// This impl clones the underlying array and, if needed,
// extends it with 0f32 or truncates it to the needed size.
// If R == C, as it ought to be, this is just a clone.
// Wishing Rust had impl specialization...
impl<const R: usize, const C: usize> DiagonalMatrix<R, C>
    where
        [(); C*R]: Sized,
        [(); R*C]: Sized,
{
    pub(super) fn T(&self) -> DiagonalMatrix<C, R> {
        let arr: [f32; C] = self.0.into_iter()
            .chain(std::iter::repeat(0f32))
            .take(C)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        DiagonalMatrix(VectorWrapper::from_arr(arr))
    }
}

impl<const R: usize, const C: usize> PartialEq for DiagonalMatrix<R, C> where [(); R*C]: Sized {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

/////////////////////////////////
/// DIAGONAL MATRIX ADD IMPLS ///
/////////////////////////////////

impl<const R: usize, const C: usize> Add<&ConstantMatrix<R, C>> for &DiagonalMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = DenseMatrix<R, C>;

    fn add(self, _rhs: &ConstantMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Add<&DenseMatrix<R, C>> for &DiagonalMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = DenseMatrix<R, C>;

    fn add(self, _rhs: &DenseMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Add<&DiagonalMatrix<R, C>> for &DiagonalMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = DiagonalMatrix<R, C>;

    fn add(self, _rhs: &DiagonalMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Add<&IdentityMatrix<R, C>> for &DiagonalMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = DiagonalMatrix<R, C>;

    fn add(self, _rhs: &IdentityMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Add<&SparseMatrix<R, C>> for &DiagonalMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = SparseMatrix<R, C>;

    fn add(self, _rhs: &SparseMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Add<&ZeroMatrix<R, C>> for &DiagonalMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = DiagonalMatrix<R, C>;

    fn add(self, _rhs: &ZeroMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

/////////////////////////////////
/// DIAGONAL MATRIX SUB IMPLS ///
/////////////////////////////////

impl<const R: usize, const C: usize> Sub<&ConstantMatrix<R, C>> for &DiagonalMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = DenseMatrix<R, C>;

    fn sub(self, _rhs: &ConstantMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Sub<&DenseMatrix<R, C>> for &DiagonalMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = DenseMatrix<R, C>;

    fn sub(self, _rhs: &DenseMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Sub<&DiagonalMatrix<R, C>> for &DiagonalMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = DiagonalMatrix<R, C>;

    fn sub(self, _rhs: &DiagonalMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Sub<&IdentityMatrix<R, C>> for &DiagonalMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = DiagonalMatrix<R, C>;

    fn sub(self, _rhs: &IdentityMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Sub<&SparseMatrix<R, C>> for &DiagonalMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = SparseMatrix<R, C>;

    fn sub(self, _rhs: &SparseMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Sub<&ZeroMatrix<R, C>> for &DiagonalMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = DiagonalMatrix<R, C>;

    fn sub(self, _rhs: &ZeroMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

/////////////////////////////////
/// DIAGONAL MATRIX MUL IMPLS ///
/////////////////////////////////

impl<const R: usize, const C: usize, const C2: usize> Mul<&ConstantMatrix<C, C2>> for &DiagonalMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
        [(); R*C2]: Sized,
{
    type Output = ConstantMatrix<R, C2>;

    fn mul(self, _rhs: &ConstantMatrix<C, C2>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&DenseMatrix<C, C2>> for &DiagonalMatrix<R, C>
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

impl<const R: usize, const C: usize, const C2: usize> Mul<&DiagonalMatrix<C, C2>> for &DiagonalMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
        [(); R*C2]: Sized,
{
    type Output = DiagonalMatrix<R, C2>;

    fn mul(self, _rhs: &DiagonalMatrix<C, C2>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&IdentityMatrix<C, C2>> for &DiagonalMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
        [(); R*C2]: Sized,
{
    type Output = DiagonalMatrix<R, C2>;

    fn mul(self, _rhs: &IdentityMatrix<C, C2>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&SparseMatrix<C, C2>> for &DiagonalMatrix<R, C>
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

impl<const R: usize, const C: usize, const C2: usize> Mul<&ZeroMatrix<C, C2>> for &DiagonalMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
        [(); R*C2]: Sized,
{
    type Output = ZeroMatrix<R, C2>;

    fn mul(self, _rhs: &ZeroMatrix<C, C2>) -> Self::Output {
        todo!()
    }
}

//////////////////////////////////
/// DIAGONAL MATRIX MATH IMPLS ///
//////////////////////////////////

impl<const R: usize, const C: usize> Neg for &DiagonalMatrix<R, C> where [(); R*C]: Sized {
    type Output = DiagonalMatrix<R, C>;

    fn neg(self) -> Self::Output {
        todo!()
    }
}

/////////////////////////////////////
/// DIAGONAL MATRIX UTILITY IMPLS ///
/////////////////////////////////////

impl<const D: usize> Display for DiagonalMatrix<D, D> where [(); D*D]: Sized {
    /// Displays the rows of a diagonal matrix.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]", (0..D).into_iter()
            .map(|r| {
                let before = (0..r).into_iter().map(|_| 0f32.to_string());
                let val = std::iter::once(self.0[r].to_string());
                let after = ((r+1)..D).into_iter().map(|_| 0f32.to_string());
                before.chain(val).chain(after).collect::<Vec<_>>().join(",")
            }).collect::<Vec<_>>().join("],["))
    }
}
