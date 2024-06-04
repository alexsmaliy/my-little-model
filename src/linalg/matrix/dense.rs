use std::fmt::Display;
use std::ops::Mul;

use super::Order;

use super::constant::ConstantMatrix;
use super::diagonal::DiagonalMatrix;
use super::identity::IdentityMatrix;
use super::sparse::SparseMatrix;
use super::zero::ZeroMatrix;

#[derive(Clone, Debug)]
pub struct DenseMatrix<const R: usize, const C: usize>(
    pub(super) [f32; R*C],
    pub(super) Order,
) where [(); R*C]: Sized;

impl<const R: usize, const C: usize> DenseMatrix<R, C>
    where
        [(); C*R]: Sized,
        [(); R*C]: Sized,
{
    pub(super) fn T(&self) -> DenseMatrix<C, R> {
        let arr: [f32; C*R] = self.0.to_vec().try_into().unwrap();
        DenseMatrix(arr, -self.1)
    }
}

impl<const R: usize, const C: usize> PartialEq for DenseMatrix<R, C> where [(); R*C]: Sized {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1 //TODO: handle transposed args.
    }
}

//////////////////////////////
/// DENSE MATRIX MUL IMPLS ///
//////////////////////////////

impl<const R: usize, const C: usize, const C2: usize> Mul<&ConstantMatrix<C, C2>> for &DenseMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
        [(); R*C2]: Sized,
{
    type Output = DenseMatrix<R, C2>;

    fn mul(self, _rhs: &ConstantMatrix<C, C2>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&DenseMatrix<C, C2>> for &DenseMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
        [(); R*C2]: Sized,
{
    type Output = DenseMatrix<R, C2>;

    fn mul(self, rhs: &DenseMatrix<C, C2>) -> Self::Output {
        // TODO: handle transposed args.
        let mut arr = [0f32; R*C2];
        for i in 0..(R * C2) {
            let from = (i / R) * C;
            let to = from + C;
            let x: f32 = rhs.0[from..to]
                .into_iter()
                .enumerate()
                .map(|(j, x)| x * self.0[j * R + i % R])
                .sum();
            arr[i] = x;
        }
        DenseMatrix(arr, self.1)
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&DiagonalMatrix<C, C2>> for &DenseMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
        [(); R*C2]: Sized,
{
    type Output = DenseMatrix<R, C2>;

    fn mul(self, _rhs: &DiagonalMatrix<C, C2>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&IdentityMatrix<C, C2>> for &DenseMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
        [(); R*C2]: Sized,
{
    type Output = DenseMatrix<R, C2>;

    fn mul(self, _rhs: &IdentityMatrix<C, C2>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&SparseMatrix<C, C2>> for &DenseMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
        [(); R*C2]: Sized,
{
    type Output = DenseMatrix<R, C2>;

    fn mul(self, _rhs: &SparseMatrix<C, C2>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&ZeroMatrix<C, C2>> for &DenseMatrix<R, C>
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
/// DENSE MATRIX UTILITY IMPLS ///
//////////////////////////////////

impl<const R: usize, const C: usize> Display for DenseMatrix<R, C> where [(); R*C]: Sized {
    /// Displays the rows of a dense matrix.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for r in 0..R {
            let mut arr = [0f32; C];
            for c in 0..C {
                arr[c] = self.0[c*R+r];
            }
            write!(f, "[{}]", arr.map(|n| n.to_string()).join(","))?;
        }
        write!(f, "]")
    }
}