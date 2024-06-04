use std::fmt::Display;
use std::ops::Mul;

use super::constant::ConstantMatrix;
use super::dense::DenseMatrix;
use super::diagonal::DiagonalMatrix;
use super::sparse::SparseMatrix;
use super::zero::ZeroMatrix;

#[derive(Clone, Debug)]
pub struct IdentityMatrix<const R: usize, const C: usize>(
    pub(super) f32,
    pub(super) f32,
);

// Impl is provided for possibly unequal R and C,
// even though only square diagonal matrices can be instantiated.
impl<const R: usize, const C: usize> IdentityMatrix<R, C> {
    pub(super) fn T(&self) -> IdentityMatrix<C, R> {
        IdentityMatrix(self.0, self.1)
    }
}

impl<const R: usize, const C: usize> PartialEq for IdentityMatrix<R, C> where [(); R*C]: Sized {
    fn eq(&self, _rhs: &Self) -> bool {
        true
    }
}

/////////////////////////////////
/// IDENTITY MATRIX MUL IMPLS ///
/////////////////////////////////

impl<const R: usize, const C: usize, const C2: usize> Mul<&ConstantMatrix<C, C2>> for &IdentityMatrix<R, C>
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

impl<const R: usize, const C: usize, const C2: usize> Mul<&DenseMatrix<C, C2>> for &IdentityMatrix<R, C>
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

impl<const R: usize, const C: usize, const C2: usize> Mul<&DiagonalMatrix<C, C2>> for &IdentityMatrix<R, C>
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

impl<const R: usize, const C: usize, const C2: usize> Mul<&IdentityMatrix<C, C2>> for &IdentityMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
        [(); R*C2]: Sized,
{
    type Output = IdentityMatrix<R, C2>;

    fn mul(self, _rhs: &IdentityMatrix<C, C2>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&SparseMatrix<C, C2>> for &IdentityMatrix<R, C>
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

impl<const R: usize, const C: usize, const C2: usize> Mul<&ZeroMatrix<C, C2>> for &IdentityMatrix<R, C>
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

/////////////////////////////////////
/// IDENTITY MATRIX UTILITY IMPLS ///
/////////////////////////////////////

impl<const D: usize> Display for IdentityMatrix<D, D> where [(); D*D]: Sized {
    /// Displays the rows of an identity matrix.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]", (0..D).into_iter()
            .map(|r| {
                let before = (0..r).into_iter().map(|_| 0f32.to_string());
                let val = std::iter::once(self.0.to_string());
                let after = ((r+1)..D).into_iter().map(|_| 0f32.to_string());
                before.chain(val).chain(after).collect::<Vec<_>>().join(",")
            }).collect::<Vec<_>>().join("],["))
    }
}