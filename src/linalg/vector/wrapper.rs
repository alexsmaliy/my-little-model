use std::collections::HashMap;
use std::ops::Index;

use super::constant::ConstantVector;
use super::dense::DenseVector;
use super::onehot::OneHotVector;
use super::sparse::SparseVector;
use super::zero::ZeroVector;

#[derive(Clone, Debug)]
pub enum VectorWrapper<const D: usize> {
    Constant(ConstantVector<D>),
    Dense(DenseVector<D>),
    OneHot(OneHotVector<D>),
    Sparse(SparseVector<D>),
    Zero(ZeroVector<D>),
}

impl<const D: usize> VectorWrapper<D> {
    pub fn from_arr(arr: [f32; D]) -> Self {
        Self::Dense(DenseVector::from_arr(arr))
    }

    pub fn onehot(i: usize) -> Self {
        Self::OneHot(OneHotVector { zero: 0f32, one: 1f32, index: i })
    }

    pub fn sparse() -> Self {
        Self::Sparse(SparseVector { elems: HashMap::default(), zero: 0f32 })
    }

    pub fn zero() -> Self {
        Self::Zero(ZeroVector(0f32))
    }
}

///////////////////
/// VECTOR ITER ///
///////////////////

impl<'a, const D: usize> IntoIterator for &'a VectorWrapper<D> {
    type Item = f32;
    type IntoIter = VectorWrapperIterator<'a, D>;

    fn into_iter(self) -> Self::IntoIter {
        VectorWrapperIterator::new(self)
    }
}

pub struct VectorWrapperIterator<'a, const D: usize> {
    pos: usize,
    vec: &'a VectorWrapper<D>,
}

impl<'a, const D: usize> VectorWrapperIterator<'a, D> {
    fn new(vec: &'a VectorWrapper<D>) -> Self {
        VectorWrapperIterator { pos: 0, vec }
    }
}

impl<'a, const D: usize> Iterator for VectorWrapperIterator<'a, D> {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos == D {
            None
        } else {
            use VectorWrapper as V;
            let ind = self.pos;
            self.pos += 1;
            match self.vec {
                V::Constant(v) => Some(v[ind]),
                V::Dense(v) => Some(v[ind]),
                V::OneHot(v) => Some(v[ind]),
                V::Sparse(v) => Some(v[ind]),
                V::Zero(v) => Some(v[ind]),
            }
        }
    }
}

////////////////////////////
/// VECTOR UTILITY IMPLS ///
////////////////////////////

impl<const D: usize> Index<usize> for VectorWrapper<D> {
    type Output = f32;

    fn index(&self, i: usize) -> &Self::Output {
        use VectorWrapper as V;
        match self {
            V::Constant(v) => &v[i],
            V::Dense(v) => &v[i],
            V::OneHot(v) => &v[i],
            V::Sparse(v) => &v[i],
            V::Zero(v) => &v[i],
        }
    }
}

impl<const D: usize> PartialEq for VectorWrapper<D> {
    fn eq(&self, other: &Self) -> bool {
        use VectorWrapper as V;
        match (self, other) {
            (V::Constant(v1), V::Constant(v2)) => v1 == v2,
            (V::Dense(v1), V::Dense(v2)) => v1 == v2,
            (V::OneHot(v1), V::OneHot(v2)) => v1 == v2,
            (V::Sparse(v1), V::Sparse(v2)) => v1 == v2,
            (V::Zero(v1), V::Zero(v2)) => v1 == v2,
            _ => false, // TODO: euqality between mixed flavors.
        }
    }
}
