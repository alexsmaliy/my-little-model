use std::ops::{Add, AddAssign, Index, Mul, Sub};

use crate::linalg::matrix::DenseMatrix;

use super::{DenseVector, OneHotVector, SparseVector, ZeroVector};
use super::traits::{CanDotProduct, CanAppend, CanMap, CanOuterProduct};

#[derive(Clone, Debug, PartialEq)]
pub struct ConstantVector<const D: usize>(
    pub(super) f32,
);

impl<const D: usize> ConstantVector<D> {
    pub(super) fn sum(&self) -> f32 {
        todo!()
    }

    pub(super) fn sum_of_squares(&self) -> f32 {
        todo!()
    }
}

impl<const D: usize> CanAppend for ConstantVector<D> where [(); D+1]: Sized {
    type Output = DenseVector<{D+1}>;
    fn append(&self, _extra_val: f32) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> CanMap for ConstantVector<D> {
    type Output = DenseVector<D>;

    fn map(&self, _f: impl Fn(f32) -> f32) -> Self::Output {
        todo!()
    }
}

//////////////////////////////
/// CONSTANT VEC ADD IMPLS ///
//////////////////////////////

impl<const D: usize> Add<f32> for &ConstantVector<D> {
    type Output = ConstantVector<D>;

    fn add(self, _rhs: f32) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Add<&ConstantVector<D>> for &ConstantVector<D> {
    type Output = ConstantVector<D>;

    fn add(self, _rhs: &ConstantVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Add<&DenseVector<D>> for &ConstantVector<D> {
    type Output = DenseVector<D>;

    fn add(self, _rhs: &DenseVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Add<&OneHotVector<D>> for &ConstantVector<D> {
    type Output = DenseVector<D>;

    fn add(self, _rhs: &OneHotVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Add<&SparseVector<D>> for &ConstantVector<D> {
    type Output = DenseVector<D>;

    fn add(self, _rhs: &SparseVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Add<&ZeroVector<D>> for &ConstantVector<D> {
    type Output = ConstantVector<D>;

    fn add(self, _rhs: &ZeroVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> AddAssign<&ConstantVector<D>> for ConstantVector<D> {
    fn add_assign(&mut self, rhs: &ConstantVector<D>) {
        self.0 += rhs.0
    }
}

impl<const D: usize> AddAssign<f32> for ConstantVector<D> {
    fn add_assign(&mut self, rhs: f32) {
        self.0 += rhs
    }
}

//////////////////////////////
/// CONSTANT VEC SUB IMPLS ///
//////////////////////////////

impl<const D: usize> Sub<f32> for &ConstantVector<D> {
    type Output = ConstantVector<D>;

    fn sub(self, _rhs: f32) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Sub<&ConstantVector<D>> for &ConstantVector<D> {
    type Output = ConstantVector<D>;

    fn sub(self, _rhs: &ConstantVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Sub<&DenseVector<D>> for &ConstantVector<D> {
    type Output = DenseVector<D>;

    fn sub(self, _rhs: &DenseVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Sub<&OneHotVector<D>> for &ConstantVector<D> {
    type Output = DenseVector<D>;

    fn sub(self, _rhs: &OneHotVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Sub<&SparseVector<D>> for &ConstantVector<D> {
    type Output = DenseVector<D>;

    fn sub(self, _rhs: &SparseVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Sub<&ZeroVector<D>> for &ConstantVector<D> {
    type Output = ConstantVector<D>;

    fn sub(self, _rhs: &ZeroVector<D>) -> Self::Output {
        todo!()
    }
}

//////////////////////////////////////
/// CONSTANT VEC DOT PRODUCT IMPLS ///
//////////////////////////////////////

impl<const D: usize> CanDotProduct<&ConstantVector<D>> for &ConstantVector<D> {
    fn dot(&self, _other: &ConstantVector<D>) -> f32 {
        todo!()
    }
}

impl<const D: usize> CanDotProduct<&DenseVector<D>> for &ConstantVector<D> {
    fn dot(&self, _other: &DenseVector<D>) -> f32 {
        todo!()
    }
}

impl<const D: usize> CanDotProduct<&OneHotVector<D>> for &ConstantVector<D> {
    fn dot(&self, _other: &OneHotVector<D>) -> f32 {
        todo!()
    }
}

impl<const D: usize> CanDotProduct<&SparseVector<D>> for &ConstantVector<D> {
    fn dot(&self, _other: &SparseVector<D>) -> f32 {
        todo!()
    }
}

impl<const D: usize> CanDotProduct<&ZeroVector<D>> for &ConstantVector<D> {
    fn dot(&self, _other: &ZeroVector<D>) -> f32 {
        todo!()
    }
}

//////////////////////////////////////
/// CONSTANT VEC DOT PRODUCT IMPLS ///
//////////////////////////////////////

impl<const D: usize, const D2: usize> CanOuterProduct<&ConstantVector<D2>> for ConstantVector<D>
    where [(); D*D2]: Sized
{
    type Output = DenseMatrix<D, D2>;

    fn outer(&self, _other: &ConstantVector<D2>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize, const D2: usize> CanOuterProduct<&DenseVector<D2>> for ConstantVector<D>
    where [(); D*D2]: Sized
{
    type Output = DenseMatrix<D, D2>;

    fn outer(&self, _other: &DenseVector<D2>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize, const D2: usize> CanOuterProduct<&OneHotVector<D2>> for ConstantVector<D>
    where [(); D*D2]: Sized
{
    type Output = DenseMatrix<D, D2>;

    fn outer(&self, _other: &OneHotVector<D2>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize, const D2: usize> CanOuterProduct<&SparseVector<D2>> for ConstantVector<D>
    where [(); D*D2]: Sized
{
    type Output = DenseMatrix<D, D2>;

    fn outer(&self, _other: &SparseVector<D2>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize, const D2: usize> CanOuterProduct<&ZeroVector<D2>> for ConstantVector<D>
    where [(); D*D2]: Sized
{
    type Output = DenseMatrix<D, D2>;

    fn outer(&self, _other: &ZeroVector<D2>) -> Self::Output {
        todo!()
    }
}

////////////////////////////////
/// CONSTANT VEC ARITH IMPLS ///
////////////////////////////////

impl<const D: usize> Mul<f32> for &ConstantVector<D> {
    type Output = ConstantVector<D>;

    fn mul(self, _rhs: f32) -> Self::Output {
        todo!()
    }
}

//////////////////////////////////
/// CONSTANT VEC UTILITY IMPLS ///
//////////////////////////////////

impl<const D: usize> Index<usize> for ConstantVector<D> {
    type Output = f32;

    fn index(&self, _index: usize) -> &Self::Output {
        &self.0
    }
}

impl<const D: usize> From<ConstantVector<D>> for f32 {
    fn from(v: ConstantVector<D>) -> Self {
        v.0
    }
}