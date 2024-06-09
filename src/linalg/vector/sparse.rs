use std::{collections::HashMap, ops::{Add, Index, Mul, Sub}};

use ahash::RandomState;

use crate::linalg::matrix::DenseMatrix;

use super::{CanDotProduct, CanAppend, CanMap, CanOuterProduct, ConstantVector, DenseVector, OneHotVector, ZeroVector};

#[derive(Clone, Debug, PartialEq)]
pub struct SparseVector<const D: usize> {
    pub(super) elems: HashMap<usize, f32, RandomState>,
    pub(super) zero: f32,
}

impl<const D: usize> SparseVector<D> {
    pub(super) fn sum(&self) -> f32 {
        todo!()
    }

    pub(super) fn sum_of_squares(&self) -> f32 {
        todo!()
    }
}

impl<const D: usize> CanAppend for &SparseVector<D> where [(); D+1]: Sized {
    type Output = SparseVector<{D+1}>;
    fn append(&self, _extra_val: f32) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> CanMap for &SparseVector<D> {
    type Output = DenseVector<D>;

    fn map(&self, _f: impl Fn(f32) -> f32) -> Self::Output {
        todo!()
    }
}

////////////////////////////
/// SPARSE VEC ADD IMPLS ///
////////////////////////////

impl<const D: usize> Add<f32> for &SparseVector<D> {
    type Output = DenseVector<D>;

    fn add(self, _rhs: f32) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Add<&ConstantVector<D>> for &SparseVector<D> {
    type Output = DenseVector<D>;

    fn add(self, _rhs: &ConstantVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Add<&DenseVector<D>> for &SparseVector<D> {
    type Output = DenseVector<D>;

    fn add(self, _rhs: &DenseVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Add<&OneHotVector<D>> for &SparseVector<D> {
    type Output = SparseVector<D>;

    fn add(self, _rhs: &OneHotVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Add<&SparseVector<D>> for &SparseVector<D> {
    type Output = SparseVector<D>;

    fn add(self, _rhs: &SparseVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Add<&ZeroVector<D>> for &SparseVector<D> {
    type Output = SparseVector<D>;

    fn add(self, _rhs: &ZeroVector<D>) -> Self::Output {
        todo!()
    }
}

////////////////////////////
/// SPARSE VEC SUB IMPLS ///
////////////////////////////

impl<const D: usize> Sub<f32> for &SparseVector<D> {
    type Output = DenseVector<D>;

    fn sub(self, _rhs: f32) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Sub<&ConstantVector<D>> for &SparseVector<D> {
    type Output = DenseVector<D>;

    fn sub(self, _rhs: &ConstantVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Sub<&DenseVector<D>> for &SparseVector<D> {
    type Output = DenseVector<D>;

    fn sub(self, _rhs: &DenseVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Sub<&OneHotVector<D>> for &SparseVector<D> {
    type Output = SparseVector<D>;

    fn sub(self, _rhs: &OneHotVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Sub<&SparseVector<D>> for &SparseVector<D> {
    type Output = SparseVector<D>;

    fn sub(self, _rhs: &SparseVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Sub<&ZeroVector<D>> for &SparseVector<D> {
    type Output = SparseVector<D>;

    fn sub(self, _rhs: &ZeroVector<D>) -> Self::Output {
        todo!()
    }
}

////////////////////////////////////
/// SPARSE VEC DOT PRODUCT IMPLS ///
////////////////////////////////////

impl<const D: usize> CanDotProduct<&ConstantVector<D>> for &SparseVector<D> {
    fn dot(&self, _other: &ConstantVector<D>) -> f32 {
        todo!()
    }
}

impl<const D: usize> CanDotProduct<&DenseVector<D>> for &SparseVector<D> {
    fn dot(&self, _other: &DenseVector<D>) -> f32 {
        todo!()
    }
}

impl<const D: usize> CanDotProduct<&OneHotVector<D>> for &SparseVector<D> {
    fn dot(&self, _other: &OneHotVector<D>) -> f32 {
        todo!()
    }
}

impl<const D: usize> CanDotProduct<&SparseVector<D>> for &SparseVector<D> {
    fn dot(&self, _other: &SparseVector<D>) -> f32 {
        todo!()
    }
}

impl<const D: usize> CanDotProduct<&ZeroVector<D>> for &SparseVector<D> {
    fn dot(&self, _other: &ZeroVector<D>) -> f32 {
        todo!()
    }
}

//////////////////////////////////////
/// CONSTANT VEC DOT PRODUCT IMPLS ///
//////////////////////////////////////

impl<const D: usize, const D2: usize> CanOuterProduct<&ConstantVector<D2>> for &SparseVector<D>
    where [(); D*D2]: Sized
{
    type Output = DenseMatrix<D, D2>;

    fn outer(self, _other: &ConstantVector<D2>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize, const D2: usize> CanOuterProduct<&DenseVector<D2>> for &SparseVector<D>
    where [(); D*D2]: Sized
{
    type Output = DenseMatrix<D, D2>;

    fn outer(self, _other: &DenseVector<D2>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize, const D2: usize> CanOuterProduct<&OneHotVector<D2>> for &SparseVector<D>
    where [(); D*D2]: Sized
{
    type Output = DenseMatrix<D, D2>;

    fn outer(self, _other: &OneHotVector<D2>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize, const D2: usize> CanOuterProduct<&SparseVector<D2>> for &SparseVector<D>
    where [(); D*D2]: Sized
{
    type Output = DenseMatrix<D, D2>;

    fn outer(self, _other: &SparseVector<D2>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize, const D2: usize> CanOuterProduct<&ZeroVector<D2>> for &SparseVector<D>
    where [(); D*D2]: Sized
{
    type Output = DenseMatrix<D, D2>;
    
    fn outer(self, _other: &ZeroVector<D2>) -> Self::Output {
        todo!()
    }
}

//////////////////////////////
/// SPARSE VEC ARITH IMPLS ///
//////////////////////////////

impl<const D: usize> Mul<f32> for &SparseVector<D> {
    type Output = SparseVector<D>;

    fn mul(self, _rhs: f32) -> Self::Output {
        todo!()
    }
}

////////////////////////////////
/// SPARSE VEC UTILITY IMPLS ///
////////////////////////////////

impl<const D: usize> Index<usize> for SparseVector<D> {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        self.elems.get(&index).unwrap_or(&self.zero)
    }
}
