use std::ops::{Add, Index, Mul, Sub};

use crate::linalg::matrix::DenseMatrix;

use super::{CanDotProduct, CanMap, CanOuterProduct, ConstantVector, DenseVector, SparseVector, ZeroVector};

#[derive(Clone, Debug, PartialEq)]
pub struct OneHotVector<const D: usize> {
    pub(super) zero: f32,      // Index impls must return a ref to a sentinel value
    pub(super) one: f32,       // as above
    pub(super) index: usize,   // the non-zero index
}

impl<const D: usize> OneHotVector<D> {
    // constructor
    pub fn at_index(index: usize) -> Self {
        assert!(index < D);
        OneHotVector {
            zero: 0f32,
            one: 1f32,
            index,
        }
    }

    pub(super) fn sum(&self) -> f32 {
        1f32
    }

    pub(super) fn sum_of_squares(&self) -> f32 {
        1f32
    }
}

impl<const D: usize> CanMap for &OneHotVector<D> {
    type Output = DenseVector<D>;

    fn map(&self, _f: impl Fn(f32) -> f32) -> Self::Output {
        todo!()
    }
}

////////////////////////////
/// ONEHOT VEC ADD IMPLS ///
////////////////////////////

impl<const D: usize> Add<f32> for &OneHotVector<D> {
    type Output = DenseVector<D>;

    fn add(self, _rhs: f32) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Add<&ConstantVector<D>> for &OneHotVector<D> {
    type Output = DenseVector<D>;

    fn add(self, _rhs: &ConstantVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Add<&DenseVector<D>> for &OneHotVector<D> {
    type Output = DenseVector<D>;

    fn add(self, _rhs: &DenseVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Add<&OneHotVector<D>> for &OneHotVector<D> {
    type Output = SparseVector<D>;

    fn add(self, _rhs: &OneHotVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Add<&SparseVector<D>> for &OneHotVector<D> {
    type Output = SparseVector<D>;

    fn add(self, _rhs: &SparseVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Add<&ZeroVector<D>> for &OneHotVector<D> {
    type Output = OneHotVector<D>;

    fn add(self, _rhs: &ZeroVector<D>) -> Self::Output {
        todo!()
    }
}

////////////////////////////
/// ONEHOT VEC SUB IMPLS ///
////////////////////////////

impl<const D: usize> Sub<f32> for &OneHotVector<D> {
    type Output = DenseVector<D>;

    fn sub(self, _rhs: f32) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Sub<&ConstantVector<D>> for &OneHotVector<D> {
    type Output = DenseVector<D>;

    fn sub(self, _rhs: &ConstantVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Sub<&DenseVector<D>> for &OneHotVector<D> {
    type Output = DenseVector<D>;

    fn sub(self, _rhs: &DenseVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Sub<&OneHotVector<D>> for &OneHotVector<D> {
    type Output = SparseVector<D>;

    fn sub(self, _rhs: &OneHotVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Sub<&SparseVector<D>> for &OneHotVector<D> {
    type Output = SparseVector<D>;

    fn sub(self, _rhs: &SparseVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Sub<&ZeroVector<D>> for &OneHotVector<D> {
    type Output = OneHotVector<D>;

    fn sub(self, _rhs: &ZeroVector<D>) -> Self::Output {
        todo!()
    }
}

////////////////////////////////////
/// ONEHOT VEC DOT PRODUCT IMPLS ///
////////////////////////////////////

impl<const D: usize> CanDotProduct<&ConstantVector<D>> for &OneHotVector<D> {
    fn dot(&self, _other: &ConstantVector<D>) -> f32 {
        todo!()
    }
}

impl<const D: usize> CanDotProduct<&DenseVector<D>> for &OneHotVector<D> {
    fn dot(&self, _other: &DenseVector<D>) -> f32 {
        todo!()
    }
}

impl<const D: usize> CanDotProduct<&OneHotVector<D>> for &OneHotVector<D> {
    fn dot(&self, _other: &OneHotVector<D>) -> f32 {
        todo!()
    }
}

impl<const D: usize> CanDotProduct<&SparseVector<D>> for &OneHotVector<D> {
    fn dot(&self, _other: &SparseVector<D>) -> f32 {
        todo!()
    }
}

impl<const D: usize> CanDotProduct<&ZeroVector<D>> for &OneHotVector<D> {
    fn dot(&self, _other: &ZeroVector<D>) -> f32 {
        todo!()
    }
}

////////////////////////////////////////
/// CONSTANT VEC OUTER PRODUCT IMPLS ///
////////////////////////////////////////

impl<const D: usize, const D2: usize> CanOuterProduct<&ConstantVector<D2>> for &OneHotVector<D>
    where [(); D*D2]: Sized
{
    type Output = DenseMatrix<D, D2>;

    fn outer(self, _other: &ConstantVector<D2>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize, const D2: usize> CanOuterProduct<&DenseVector<D2>> for &OneHotVector<D>
    where [(); D*D2]: Sized
{
    type Output = DenseMatrix<D, D2>;

    fn outer(self, _other: &DenseVector<D2>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize, const D2: usize> CanOuterProduct<&OneHotVector<D2>> for &OneHotVector<D>
    where [(); D*D2]: Sized
{
    type Output = DenseMatrix<D, D2>;

    fn outer(self, _other: &OneHotVector<D2>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize, const D2: usize> CanOuterProduct<&SparseVector<D2>> for &OneHotVector<D>
    where [(); D*D2]: Sized
{
    type Output = DenseMatrix<D, D2>;

    fn outer(self, _other: &SparseVector<D2>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize, const D2: usize> CanOuterProduct<&ZeroVector<D2>> for &OneHotVector<D>
    where [(); D*D2]: Sized
{
    type Output = DenseMatrix<D, D2>;
    
    fn outer(self, _other: &ZeroVector<D2>) -> Self::Output {
        todo!()
    }
}

//////////////////////////////
/// ONEHOT VEC ARITH IMPLS ///
//////////////////////////////

impl<const D: usize> Mul<f32> for &OneHotVector<D> {
    type Output = SparseVector<D>;

    fn mul(self, _rhs: f32) -> Self::Output {
        todo!()
    }
}

////////////////////////////////
/// ONEHOT VEC UTILITY IMPLS ///
////////////////////////////////

impl<const D: usize> Index<usize> for OneHotVector<D> {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        if index == self.index { &self.one } else { &self.zero }
    }
}
