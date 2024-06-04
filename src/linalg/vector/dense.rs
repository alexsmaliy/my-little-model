use std::ops::{Add, Index, IndexMut, Mul, Sub};

use crate::linalg::matrix::DenseMatrix;

use super::{CanDotProduct, CanMap, CanOuterProduct, ConstantVector, OneHotVector, SparseVector, ZeroVector};

#[derive(Clone, Debug, PartialEq)]
pub struct DenseVector<const D: usize>(
    pub(super) [f32; D],
);

impl<const D: usize> DenseVector<D> {
    pub(super) fn from_arr(arr: [f32; D]) -> Self {
        Self(arr)
    }

    pub(super) fn sum(&self) -> f32 {
        todo!()
    }

    pub(super) fn sum_of_squares(&self) -> f32 {
        todo!()
    }
}

impl<const D: usize> CanMap for &DenseVector<D> {
    type Output = DenseVector<D>;

    fn map(&self, _f: impl Fn(f32) -> f32) -> Self::Output {
        todo!()
    }
}

///////////////////////////
/// DENSE VEC ADD IMPLS ///
///////////////////////////

impl<const D: usize> Add<f32> for &DenseVector<D> {
    type Output = DenseVector<D>;

    fn add(self, _rhs: f32) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Add<&ConstantVector<D>> for &DenseVector<D> {
    type Output = DenseVector<D>;

    fn add(self, _rhs: &ConstantVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Add<&DenseVector<D>> for &DenseVector<D> {
    type Output = DenseVector<D>;

    fn add(self, _rhs: &DenseVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Add<&OneHotVector<D>> for &DenseVector<D> {
    type Output = DenseVector<D>;

    fn add(self, _rhs: &OneHotVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Add<&SparseVector<D>> for &DenseVector<D> {
    type Output = DenseVector<D>;

    fn add(self, _rhs: &SparseVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Add<&ZeroVector<D>> for &DenseVector<D> {
    type Output = DenseVector<D>;

    fn add(self, _rhs: &ZeroVector<D>) -> Self::Output {
        todo!()
    }
}

///////////////////////////
/// DENSE VEC SUB IMPLS ///
///////////////////////////

impl<const D: usize> Sub<f32> for &DenseVector<D> {
    type Output = DenseVector<D>;

    fn sub(self, _rhs: f32) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Sub<&ConstantVector<D>> for &DenseVector<D> {
    type Output = DenseVector<D>;

    fn sub(self, _rhs: &ConstantVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Sub<&DenseVector<D>> for &DenseVector<D> {
    type Output = DenseVector<D>;

    fn sub(self, _rhs: &DenseVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Sub<&OneHotVector<D>> for &DenseVector<D> {
    type Output = DenseVector<D>;

    fn sub(self, _rhs: &OneHotVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Sub<&SparseVector<D>> for &DenseVector<D> {
    type Output = DenseVector<D>;

    fn sub(self, _rhs: &SparseVector<D>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> Sub<&ZeroVector<D>> for &DenseVector<D> {
    type Output = DenseVector<D>;

    fn sub(self, _rhs: &ZeroVector<D>) -> Self::Output {
        todo!()
    }
}

///////////////////////////////////
/// DENSE VEC DOT PRODUCT IMPLS ///
///////////////////////////////////

impl<const D: usize> CanDotProduct<&ConstantVector<D>> for &DenseVector<D> {
    fn dot(&self, _other: &ConstantVector<D>) -> f32 {
        todo!()
    }
}

impl<const D: usize> CanDotProduct<&DenseVector<D>> for &DenseVector<D> {
    fn dot(&self, _other: &DenseVector<D>) -> f32 {
        todo!()
    }
}

impl<const D: usize> CanDotProduct<&OneHotVector<D>> for &DenseVector<D> {
    fn dot(&self, _other: &OneHotVector<D>) -> f32 {
        todo!()
    }
}

impl<const D: usize> CanDotProduct<&SparseVector<D>> for &DenseVector<D> {
    fn dot(&self, _other: &SparseVector<D>) -> f32 {
        todo!()
    }
}

impl<const D: usize> CanDotProduct<&ZeroVector<D>> for &DenseVector<D> {
    fn dot(&self, _other: &ZeroVector<D>) -> f32 {
        todo!()
    }
}

//////////////////////////////////////
/// CONSTANT VEC DOT PRODUCT IMPLS ///
//////////////////////////////////////

impl<const D: usize, const D2: usize> CanOuterProduct<&ConstantVector<D2>> for &DenseVector<D>
    where [(); D*D2]: Sized
{
    type Output = DenseMatrix<D, D2>;

    fn outer(&self, _other: &ConstantVector<D2>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize, const D2: usize> CanOuterProduct<&DenseVector<D2>> for &DenseVector<D>
    where [(); D*D2]: Sized
{
    type Output = DenseMatrix<D, D2>;

    fn outer(&self, _other: &DenseVector<D2>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize, const D2: usize> CanOuterProduct<&OneHotVector<D2>> for &DenseVector<D>
    where [(); D*D2]: Sized
{
    type Output = DenseMatrix<D, D2>;

    fn outer(&self, _other: &OneHotVector<D2>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize, const D2: usize> CanOuterProduct<&SparseVector<D2>> for &DenseVector<D>
    where [(); D*D2]: Sized
{
    type Output = DenseMatrix<D, D2>;

    fn outer(&self, _other: &SparseVector<D2>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize, const D2: usize> CanOuterProduct<&ZeroVector<D2>> for &DenseVector<D>
    where [(); D*D2]: Sized
{
    type Output = DenseMatrix<D, D2>;
    
    fn outer(&self, _other: &ZeroVector<D2>) -> Self::Output {
        todo!()
    }
}

/////////////////////////////
/// DENSE VEC ARITH IMPLS ///
/////////////////////////////

impl<const D: usize> Mul<f32> for &DenseVector<D> {
    type Output = DenseVector<D>;

    fn mul(self, _rhs: f32) -> Self::Output {
        todo!()
    }
}

///////////////////////////////
/// DENSE VEC UTILITY IMPLS ///
///////////////////////////////

impl<const D: usize> Index<usize> for DenseVector<D> {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const D: usize> IndexMut<usize> for DenseVector<D> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}
