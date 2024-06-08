use std::fmt::Display;
use std::marker::PhantomData;
use std::ops::{Add, Mul, Neg, Sub};

use crate::linalg::vector::DenseVector;
use crate::linalg::Vector;

use super::constant::ConstantMatrix;
use super::dense::DenseMatrix;
use super::identity::IdentityMatrix;
use super::sparse::SparseMatrix;
use super::zero::ZeroMatrix;

#[derive(Clone, Debug)]
pub struct DiagonalMatrix<const R: usize, const C: usize> where [(); R*C]: Sized {
    pub(crate) diagonal_data: Box<[f32]>,
    pub(crate) size_marker: PhantomData<[[f32; R]; C]>
}

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
        let copy = self.diagonal_data.iter().copied()
            .chain(std::iter::repeat(0f32))
            .take(C);

        let mut container = Vec::with_capacity(C*R);
        container.extend(copy);

        DiagonalMatrix {
            diagonal_data: container.into_boxed_slice(),
            size_marker: PhantomData,
        }
    }
}

impl<const R: usize, const C: usize> PartialEq for DiagonalMatrix<R, C> where [(); R*C]: Sized {
    fn eq(&self, other: &Self) -> bool {
        self.diagonal_data == other.diagonal_data
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

impl<const R: usize, const C: usize> Mul<&DenseVector<C>> for &DiagonalMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = DenseVector<R>;

    fn mul(self, rhs: &DenseVector<C>) -> Self::Output {
        let f = |(x, y)| x * y;
        let us = self.diagonal_data.iter();
        let them = rhs.data.iter();

        let mut container = Vec::with_capacity(R);
        let mapped = us.zip(them).take(R).map(f);
        container.extend(mapped);

        DenseVector {
            data: container.into_boxed_slice(),
            size_marker: PhantomData,
        }
    }
}

//////////////////////////////////
/// DIAGONAL MATRIX MATH IMPLS ///
//////////////////////////////////

impl<const R: usize, const C: usize> Add<&DiagonalMatrix<R, C>> for f32 where [(); R*C]: Sized {
    type Output = DenseMatrix<R, C>;

    fn add(self, _rhs: &DiagonalMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Add<f32> for &DiagonalMatrix<R, C> where [(); R*C]: Sized {
    type Output = DenseMatrix<R, C>;

    fn add(self, _rhs: f32) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Sub<f32> for &DiagonalMatrix<R, C> where [(); R*C]: Sized {
    type Output = DenseMatrix<R, C>;

    fn sub(self, _rhs: f32) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Mul<&DiagonalMatrix<R, C>> for f32 where [(); R*C]: Sized {
    type Output = DiagonalMatrix<R, C>;

    fn mul(self, _rhs: &DiagonalMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Mul<f32> for &DiagonalMatrix<R, C> where [(); R*C]: Sized {
    type Output = DiagonalMatrix<R, C>;

    fn mul(self, _rhs: f32) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Neg for &DiagonalMatrix<R, C> where [(); R*C]: Sized {
    type Output = DiagonalMatrix<R, C>;

    fn neg(self) -> Self::Output {
        todo!()
    }
}

/////////////////////////////////////
/// DIAGONAL MATRIX UTILITY IMPLS ///
/////////////////////////////////////

impl<const D: usize> From<DenseVector<D>> for DiagonalMatrix<D, D> where [(); D*D]: Sized {
    fn from(vector: DenseVector<D>) -> Self {
        Self {
            diagonal_data: vector.data,
            size_marker: PhantomData,
        }
    }
}

impl<const D: usize> From<Vector<D>> for DiagonalMatrix<D, D> where [(); D*D]: Sized {
    fn from(vector: Vector<D>) -> Self {
        vector.to_dense().into()
    }
}

impl<const D: usize> Display for DiagonalMatrix<D, D> where [(); D*D]: Sized {
    /// Displays the rows of a diagonal matrix.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]", (0..D).into_iter()
            .map(|r| {
                let before = (0..r).into_iter().map(|_| 0f32.to_string());
                let val = std::iter::once(self.diagonal_data[r].to_string());
                let after = ((r+1)..D).into_iter().map(|_| 0f32.to_string());
                before.chain(val).chain(after).collect::<Vec<_>>().join(",")
            }).collect::<Vec<_>>().join("],["))
    }
}
