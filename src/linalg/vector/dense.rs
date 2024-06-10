use std::{marker::PhantomData, ops::{Add, AddAssign, Index, IndexMut, Mul, Sub}};

use crate::linalg::matrix::DenseMatrix;

use super::{ConstantVector, OneHotVector, SparseVector, ZeroVector};
use super::traits::{CanDotProduct, CanAppend, CanMap, CanOuterProduct};

#[derive(Clone, Debug, PartialEq)]
pub struct DenseVector<const D: usize> {
    pub(crate) data: Box<[f32]>,
    pub(crate) size_marker: PhantomData<[f32; D]>,
}

impl<const D: usize> DenseVector<D> {
    pub(crate) fn from_arr(arr: [f32; D]) -> Self {
        Self {
            data: Box::new(arr),
            size_marker: PhantomData,
        }
    }

    pub(crate) fn from_boxed_slice(slice: Box<[f32]>) -> Self {
        assert_eq!(slice.len(), D);
        Self {
            data: slice,
            size_marker: PhantomData,
        }
    }

    pub fn from_fun(f: impl Fn(usize) -> f32) -> Self {
        let mut v = Vec::with_capacity(D);
        v.extend((0..D).map(f));
        DenseVector::from_boxed_slice(v.into_boxed_slice())
    }

    pub(super) fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub(super) fn sum_of_squares(&self) -> f32 {
        self.data.iter().copied().map(|n| <f32>::powi(n, 2)).sum()
    }
}

impl<const D: usize> CanAppend for &DenseVector<D> where [(); D+1]: Sized {
    type Output = DenseVector<{D+1}>;
    fn append(&self, _extra_val: f32) -> Self::Output {
        todo!()
    }
}

impl<const D: usize> CanMap for &DenseVector<D> {
    type Output = DenseVector<D>;

    fn map(&self, f: impl Fn(f32) -> f32) -> Self::Output {
        let mapped = self.data.iter().copied().map(f).collect::<Box<[f32]>>();
        DenseVector {
            data: mapped,
            size_marker: PhantomData,
        }
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

    fn add(self, rhs: &DenseVector<D>) -> Self::Output {
        let f = |(a, b)| a + b;
        let us = self.data.iter().copied();
        let them = rhs.data.iter().copied();
        let mapped = us.zip(them).map(f).collect::<Box<[f32]>>();
        DenseVector {
            data: mapped,
            size_marker: PhantomData,
        }
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

impl<const D: usize> AddAssign<&ConstantVector<D>> for DenseVector<D> {
    fn add_assign(&mut self, _rhs: &ConstantVector<D>) {
        todo!()
    }
}

impl<const D: usize> AddAssign<&DenseVector<D>> for DenseVector<D> {
    fn add_assign(&mut self, _rhs: &DenseVector<D>) {
        todo!()
    }
}

impl<const D: usize> AddAssign<&OneHotVector<D>> for DenseVector<D> {
    fn add_assign(&mut self, _rhs: &OneHotVector<D>) {
        todo!()
    }
}

impl<const D: usize> AddAssign<&SparseVector<D>> for DenseVector<D> {
    fn add_assign(&mut self, _rhs: &SparseVector<D>) {
        todo!()
    }
}

impl<const D: usize> AddAssign<f32> for DenseVector<D> {
    fn add_assign(&mut self, _rhs: f32) {
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

    fn sub(self, rhs: &DenseVector<D>) -> Self::Output {
        let f = |(a, b)| a - b;
        let us = self.data.iter().copied();
        let them = rhs.data.iter().copied();
        let mapped = us.zip(them).map(f).collect::<Box<[f32]>>();
        DenseVector {
            data: mapped,
            size_marker: PhantomData,
        }
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

    fn outer(self, _other: &ConstantVector<D2>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize, const D2: usize> CanOuterProduct<&DenseVector<D2>> for &DenseVector<D>
    where [(); D*D2]: Sized
{
    type Output = DenseMatrix<D, D2>;

    fn outer(self, other: &DenseVector<D2>) -> Self::Output {
        let mut container = Vec::with_capacity(D*D2);
        for col in 0..D2 {
            let scalar = other[col];
            let x = self.data.into_iter().map(|x| x * scalar);
            container.extend(x);
        }
        DenseMatrix::from_boxed_slice(container.into_boxed_slice())
    }
}

impl<const D: usize, const D2: usize> CanOuterProduct<&OneHotVector<D2>> for &DenseVector<D>
    where [(); D*D2]: Sized
{
    type Output = DenseMatrix<D, D2>;

    fn outer(self, _other: &OneHotVector<D2>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize, const D2: usize> CanOuterProduct<&SparseVector<D2>> for &DenseVector<D>
    where [(); D*D2]: Sized
{
    type Output = DenseMatrix<D, D2>;

    fn outer(self, _other: &SparseVector<D2>) -> Self::Output {
        todo!()
    }
}

impl<const D: usize, const D2: usize> CanOuterProduct<&ZeroVector<D2>> for &DenseVector<D>
    where [(); D*D2]: Sized
{
    type Output = DenseMatrix<D, D2>;
    
    fn outer(self, _other: &ZeroVector<D2>) -> Self::Output {
        todo!()
    }
}

/////////////////////////////
/// DENSE VEC ARITH IMPLS ///
/////////////////////////////

impl<const D: usize> Mul<f32> for &DenseVector<D> {
    type Output = DenseVector<D>;

    fn mul(self, rhs: f32) -> Self::Output {
        let f = |x| x * rhs;
        let us = self.data.iter().copied();
        let mapped = us.map(f);

        let mut container = Vec::with_capacity(D);
        container.extend(mapped);

        DenseVector {
            data: container.into_boxed_slice(),
            size_marker: PhantomData,
        }
    }
}

///////////////////////////////
/// DENSE VEC UTILITY IMPLS ///
///////////////////////////////

impl<const D: usize> Index<usize> for DenseVector<D> {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<const D: usize> IndexMut<usize> for DenseVector<D> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}
