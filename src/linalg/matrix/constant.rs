use std::fmt::Display;
use std::ops::Mul;

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
