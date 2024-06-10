use std::collections::HashMap;
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, Sub};

use crate::linalg::Matrix;

use super::{CanDotProduct, CanAppend, CanMap, CanOuterProduct};
use super::constant::ConstantVector;
use super::dense::DenseVector;
use super::onehot::OneHotVector;
use super::sparse::SparseVector;
use super::zero::ZeroVector;

#[derive(Clone, Debug)]
pub enum Vector<const D: usize> {
    Constant(ConstantVector<D>),
    Dense(DenseVector<D>),
    OneHot(OneHotVector<D>),
    Sparse(SparseVector<D>),
    Zero(ZeroVector<D>),
}

impl<const D: usize> Vector<D> {
    // constructor
    pub fn from_arr(arr: [f32; D]) -> Self {
        Self::Dense(DenseVector::from_arr(arr))
    }

    // constructor
    pub(crate) fn from_boxed_slice(slice: Box<[f32]>) -> Self {
        Self::Dense(DenseVector::from_boxed_slice(slice))
    }

    // constructor
    pub fn from_fun(f: impl Fn(usize) -> f32) -> Self {
        Self::Dense(DenseVector::from_fun(f))
    }

    // constructor
    pub fn one_hot(i: usize) -> Self {
        Self::OneHot(OneHotVector::at_index(i))
    }

    // constructor
    pub fn sparse() -> Self {
        Self::Sparse(SparseVector { elems: HashMap::default(), zero: 0f32 })
    }

    // constructor
    pub fn zero() -> Self {
        Self::Zero(ZeroVector(0f32))
    }

    pub fn dot(&self, other: &Vector<D>) -> f32 {
        use Vector as V;
        match (self, other) {
            (V::Constant(v1), V::Constant(v2)) => v1.dot(v2),
            (V::Constant(v1), V::Dense(v2)) => v1.dot(v2),
            (V::Constant(v1), V::OneHot(v2)) => v1.dot(v2),
            (V::Constant(v1), V::Sparse(v2)) => v1.dot(v2),
            (V::Constant(v1), V::Zero(v2)) => v1.dot(v2),

            (V::Dense(v1), V::Constant(v2)) => v1.dot(v2),
            (V::Dense(v1), V::Dense(v2)) => v1.dot(v2),
            (V::Dense(v1), V::OneHot(v2)) => v1.dot(v2),
            (V::Dense(v1), V::Sparse(v2)) => v1.dot(v2),
            (V::Dense(v1), V::Zero(v2)) => v1.dot(v2),

            (V::OneHot(v1), V::Constant(v2)) => v1.dot(v2),
            (V::OneHot(v1), V::Dense(v2)) => v1.dot(v2),
            (V::OneHot(v1), V::OneHot(v2)) => v1.dot(v2),
            (V::OneHot(v1), V::Sparse(v2)) => v1.dot(v2),
            (V::OneHot(v1), V::Zero(v2)) => v1.dot(v2),

            (V::Sparse(v1), V::Constant(v2)) => v1.dot(v2),
            (V::Sparse(v1), V::Dense(v2)) => v1.dot(v2),
            (V::Sparse(v1), V::OneHot(v2)) => v1.dot(v2),
            (V::Sparse(v1), V::Sparse(v2)) => v1.dot(v2),
            (V::Sparse(v1), V::Zero(v2)) => v1.dot(v2),

            (V::Zero(v1), V::Constant(v2)) => v1.dot(v2),
            (V::Zero(v1), V::Dense(v2)) => v1.dot(v2),
            (V::Zero(v1), V::OneHot(v2)) => v1.dot(v2),
            (V::Zero(v1), V::Sparse(v2)) => v1.dot(v2),
            (V::Zero(v1), V::Zero(v2)) => v1.dot(v2),
        }
    }

    pub fn extend(&self, extra_val: f32) -> Vector<{D+1}> {
        use Vector as V;
        match self {
            V::Constant(v) => V::Dense(v.append(extra_val)),
            V::Dense(v) => V::Dense(v.append(extra_val)),
            V::OneHot(v) => V::Sparse(v.append(extra_val)),
            V::Sparse(v) => V::Sparse(v.append(extra_val)),
            V::Zero(v) => V::Sparse(v.append(extra_val)),
        }
    }

    pub fn map(&self, f: impl Fn(f32) -> f32) -> Vector<D> {
        use Vector as V;
        match self {
            V::Constant(v) => V::Dense(v.map(f)),
            V::Dense(v) => V::Dense(v.map(f)),
            V::OneHot(v) => V::Dense(v.map(f)),
            V::Sparse(v) => V::Dense(v.map(f)),
            V::Zero(v) => V::Dense(v.map(f)),
        }
    }

    pub fn outer<const D2: usize>(&self, other: &Vector<D2>) -> Matrix<D, D2> where [(); D*D2]: Sized {
        use Matrix as M;
        use Vector as V;
        match (self, other) {
            (V::Constant(v1), V::Constant(v2)) => M::Dense(v1.outer(v2)),
            (V::Constant(v1), V::Dense(v2)) => M::Dense(v1.outer(v2)),
            (V::Constant(v1), V::OneHot(v2)) => M::Dense(v1.outer(v2)),
            (V::Constant(v1), V::Sparse(v2)) => M::Dense(v1.outer(v2)),
            (V::Constant(v1), V::Zero(v2)) => M::Dense(v1.outer(v2)),

            (V::Dense(v1), V::Constant(v2)) => M::Dense(v1.outer(v2)),
            (V::Dense(v1), V::Dense(v2)) => M::Dense(v1.outer(v2)),
            (V::Dense(v1), V::OneHot(v2)) => M::Dense(v1.outer(v2)),
            (V::Dense(v1), V::Sparse(v2)) => M::Dense(v1.outer(v2)),
            (V::Dense(v1), V::Zero(v2)) => M::Dense(v1.outer(v2)),

            (V::OneHot(v1), V::Constant(v2)) => M::Dense(v1.outer(v2)),
            (V::OneHot(v1), V::Dense(v2)) => M::Dense(v1.outer(v2)),
            (V::OneHot(v1), V::OneHot(v2)) => M::Dense(v1.outer(v2)),
            (V::OneHot(v1), V::Sparse(v2)) => M::Dense(v1.outer(v2)),
            (V::OneHot(v1), V::Zero(v2)) => M::Dense(v1.outer(v2)),

            (V::Sparse(v1), V::Constant(v2)) => M::Dense(v1.outer(v2)),
            (V::Sparse(v1), V::Dense(v2)) => M::Dense(v1.outer(v2)),
            (V::Sparse(v1), V::OneHot(v2)) => M::Dense(v1.outer(v2)),
            (V::Sparse(v1), V::Sparse(v2)) => M::Dense(v1.outer(v2)),
            (V::Sparse(v1), V::Zero(v2)) => M::Dense(v1.outer(v2)),

            (V::Zero(v1), V::Constant(v2)) => M::Dense(v1.outer(v2)),
            (V::Zero(v1), V::Dense(v2)) => M::Dense(v1.outer(v2)),
            (V::Zero(v1), V::OneHot(v2)) => M::Dense(v1.outer(v2)),
            (V::Zero(v1), V::Sparse(v2)) => M::Dense(v1.outer(v2)),
            (V::Zero(v1), V::Zero(v2)) => M::Dense(v1.outer(v2)),
        }
    }

    pub fn sum(&self) -> f32 {
        use Vector as V;
        match self {
            V::Constant(v) => v.sum(),
            V::Dense(v) => v.sum(),
            V::OneHot(v) => v.sum(),
            V::Sparse(v) => v.sum(),
            V::Zero(v) => v.sum(),
        }
    }

    pub fn sum_of_squares(&self) -> f32 {
        use Vector as V;
        match self {
            V::Constant(v) => v.sum_of_squares(),
            V::Dense(v) => v.sum_of_squares(),
            V::OneHot(v) => v.sum_of_squares(),
            V::Sparse(v) => v.sum_of_squares(),
            V::Zero(v) => v.sum_of_squares(),
        }
    }

    pub(crate) fn to_dense(self) -> DenseVector<D> {
        use Vector as V;
        match self {
            V::Constant(_v) => todo!(),
            V::Dense(v) => v,
            V::OneHot(_v) => todo!(),
            V::Sparse(_v) => todo!(),
            V::Zero(_v) => todo!(),
        }
    }
}

impl<const D: usize> Add<&Vector<D>> for &Vector<D> {
    type Output = Vector<D>;

    fn add(self, rhs: &Vector<D>) -> Self::Output {
        use Vector as V;
        match (self, rhs) {
            (V::Constant(v1), V::Constant(v2)) => V::Constant(v1 + v2),
            (V::Constant(v1), V::Dense(v2)) => V::Dense(v1 + v2),
            (V::Constant(v1), V::OneHot(v2)) => V::Dense(v1 + v2),
            (V::Constant(v1), V::Sparse(v2)) => V::Dense(v1 + v2),
            (V::Constant(v1), V::Zero(v2)) => V::Constant(v1 + v2),

            (V::Dense(v1), V::Constant(v2)) => V::Dense(v1 + v2),
            (V::Dense(v1), V::Dense(v2)) => V::Dense(v1 + v2),
            (V::Dense(v1), V::OneHot(v2)) => V::Dense(v1 + v2),
            (V::Dense(v1), V::Sparse(v2)) => V::Dense(v1 + v2),
            (V::Dense(v1), V::Zero(v2)) => V::Dense(v1 + v2),

            (V::OneHot(v1), V::Constant(v2)) => V::Dense(v1 + v2),
            (V::OneHot(v1), V::Dense(v2)) => V::Dense(v1 + v2),
            (V::OneHot(v1), V::OneHot(v2)) => V::Sparse(v1 + v2),
            (V::OneHot(v1), V::Sparse(v2)) => V::Sparse(v1 + v2),
            (V::OneHot(v1), V::Zero(v2)) => V::OneHot(v1 + v2),

            (V::Sparse(v1), V::Constant(v2)) => V::Dense(v1 + v2),
            (V::Sparse(v1), V::Dense(v2)) => V::Dense(v1 + v2),
            (V::Sparse(v1), V::OneHot(v2)) => V::Sparse(v1 + v2),
            (V::Sparse(v1), V::Sparse(v2)) => V::Sparse(v1 + v2),
            (V::Sparse(v1), V::Zero(v2)) => V::Sparse(v1 + v2),

            (V::Zero(v1), V::Constant(v2)) => V::Constant(v1 + v2),
            (V::Zero(v1), V::Dense(v2)) => V::Dense(v1 + v2),
            (V::Zero(v1), V::OneHot(v2)) => V::OneHot(v1 + v2),
            (V::Zero(v1), V::Sparse(v2)) => V::Sparse(v1 + v2),
            (V::Zero(v1), V::Zero(v2)) => V::Zero(v1 + v2),
        }
    }
}

impl<const D: usize> Add<&Vector<D>> for f32 {
    type Output = Vector<D>;

    fn add(self, rhs: &Vector<D>) -> Self::Output {
        use Vector as V;
        match rhs {
            V::Constant(v) => V::Constant(v + self),
            V::Dense(v) => V::Dense(v + self),
            V::OneHot(v) => V::Dense(v + self),
            V::Sparse(v) => V::Dense(v + self),
            V::Zero(v) => V::Constant(v + self),
        }
    }
}

impl<const D: usize> Add<f32> for &Vector<D> {
    type Output = Vector<D>;

    fn add(self, rhs: f32) -> Self::Output {
        use Vector as V;
        match self {
            V::Constant(v) => V::Constant(v + rhs),
            V::Dense(v) => V::Dense(v + rhs),
            V::OneHot(v) => V::Dense(v + rhs),
            V::Sparse(v) => V::Dense(v + rhs),
            V::Zero(v) => V::Constant(v + rhs),
        }
    }
}

impl<const D: usize> AddAssign<&Vector<D>> for Vector<D> {
    fn add_assign(&mut self, rhs: &Vector<D>) {
        use Vector as V;
        match (&mut *self, rhs) {
            (V::Constant(ref mut v1), V::Constant(v2)) => *v1 += v2,
            (V::Constant(v1), V::Dense(v2)) => *self = V::Dense(&*v1 + v2),
            (V::Constant(v1), V::OneHot(v2)) => *self = V::Dense(&*v1 + v2),
            (V::Constant(v1), V::Sparse(v2)) => *self = V::Dense(&*v1 + v2),
            (V::Constant(_), V::Zero(_)) => {}, // no-op

            (V::Dense(ref mut v1), V::Constant(v2)) => *v1 += v2,
            (V::Dense(ref mut v1), V::Dense(v2)) => *v1 += v2,
            (V::Dense(ref mut v1), V::OneHot(v2)) => *v1 += v2,
            (V::Dense(ref mut v1), V::Sparse(v2)) => *v1 += v2,
            (V::Dense(_), V::Zero(_)) => {}, // no-op

            (V::OneHot(v1), V::Constant(v2)) => *self = V::Dense(&*v1 + v2),
            (V::OneHot(v1), V::Dense(v2)) => *self = V::Dense(&*v1 + v2),
            (V::OneHot(v1), V::OneHot(v2)) => *self = V::Sparse(&*v1 + v2),
            (V::OneHot(v1), V::Sparse(v2)) => *self = V::Sparse(&*v1 + v2),
            (V::OneHot(_), V::Zero(_)) => {}, // no-op

            (V::Sparse(v1), V::Constant(v2)) => *self = V::Dense(&*v1 + v2),
            (V::Sparse(v1), V::Dense(v2)) => *self = V::Dense(&*v1 + v2),
            (V::Sparse(ref mut v1), V::OneHot(v2)) => *v1 += v2,
            (V::Sparse(ref mut v1), V::Sparse(v2)) => *v1 += v2,
            (V::Sparse(_), V::Zero(_)) => {}, // no-op

            (V::Zero(_), rhs @ V::Constant(_)) => *self = rhs.clone(),
            (V::Zero(_), rhs @ V::Dense(_)) => *self = rhs.clone(),
            (V::Zero(_), rhs @ V::OneHot(_)) => *self = rhs.clone(),
            (V::Zero(_), rhs @ V::Sparse(_)) => *self = rhs.clone(),
            (V::Zero(_), V::Zero(_)) => {}, // no-op
        }
    }
}

impl<const D: usize> AddAssign<f32> for Vector<D> {
    fn add_assign(&mut self, rhs: f32) {
        use Vector as V;
        match self {
            V::Constant(ref mut v) => *v += rhs,
            V::Dense(ref mut v) => *v += rhs,
            V::OneHot(v) => *self = V::Dense(&*v + rhs),
            V::Sparse(v) => *self = V::Dense(&*v + rhs),
            V::Zero(v) => *self = V::Constant(&*v + rhs),
        }
    }
}

impl<const D: usize> Sub<&Vector<D>> for &Vector<D> {
    type Output = Vector<D>;

    fn sub(self, rhs: &Vector<D>) -> Self::Output {
        use Vector as V;
        match (self, rhs) {
            (V::Constant(v1), V::Constant(v2)) => V::Constant(v1 - v2),
            (V::Constant(v1), V::Dense(v2)) => V::Dense(v1 - v2),
            (V::Constant(v1), V::OneHot(v2)) => V::Dense(v1 - v2),
            (V::Constant(v1), V::Sparse(v2)) => V::Dense(v1 - v2),
            (V::Constant(v1), V::Zero(v2)) => V::Constant(v1 - v2),

            (V::Dense(v1), V::Constant(v2)) => V::Dense(v1 - v2),
            (V::Dense(v1), V::Dense(v2)) => V::Dense(v1 - v2),
            (V::Dense(v1), V::OneHot(v2)) => V::Dense(v1 - v2),
            (V::Dense(v1), V::Sparse(v2)) => V::Dense(v1 - v2),
            (V::Dense(v1), V::Zero(v2)) => V::Dense(v1 - v2),

            (V::OneHot(v1), V::Constant(v2)) => V::Dense(v1 - v2),
            (V::OneHot(v1), V::Dense(v2)) => V::Dense(v1 - v2),
            (V::OneHot(v1), V::OneHot(v2)) => V::Sparse(v1 - v2),
            (V::OneHot(v1), V::Sparse(v2)) => V::Sparse(v1 - v2),
            (V::OneHot(v1), V::Zero(v2)) => V::OneHot(v1 - v2),

            (V::Sparse(v1), V::Constant(v2)) => V::Dense(v1 - v2),
            (V::Sparse(v1), V::Dense(v2)) => V::Dense(v1 - v2),
            (V::Sparse(v1), V::OneHot(v2)) => V::Sparse(v1 - v2),
            (V::Sparse(v1), V::Sparse(v2)) => V::Sparse(v1 - v2),
            (V::Sparse(v1), V::Zero(v2)) => V::Sparse(v1 - v2),

            (V::Zero(v1), V::Constant(v2)) => V::Constant(v1 - v2),
            (V::Zero(v1), V::Dense(v2)) => V::Dense(v1 - v2),
            (V::Zero(v1), V::OneHot(v2)) => V::Sparse(v1 - v2),
            (V::Zero(v1), V::Sparse(v2)) => V::Sparse(v1 - v2),
            (V::Zero(v1), V::Zero(v2)) => V::Zero(v1 - v2),
        }
    }
}

impl<const D: usize> Sub<&Vector<D>> for f32 {
    type Output = Vector<D>;

    fn sub(self, rhs: &Vector<D>) -> Self::Output {
        use Vector as V;
        match rhs {
            V::Constant(v) => V::Constant(v - self),
            V::Dense(v) => V::Dense(v - self),
            V::OneHot(v) => V::Dense(v - self),
            V::Sparse(v) => V::Dense(v - self),
            V::Zero(v) => V::Constant(v - self),
        }
    }
}

impl<const D: usize> Sub<f32> for &Vector<D> {
    type Output = Vector<D>;

    fn sub(self, rhs: f32) -> Self::Output {
        use Vector as V;
        match self {
            V::Constant(v) => V::Constant(v - rhs),
            V::Dense(v) => V::Dense(v - rhs),
            V::OneHot(v) => V::Dense(v - rhs),
            V::Sparse(v) => V::Dense(v - rhs),
            V::Zero(v) => V::Constant(v - rhs),
        }
    }
}

impl<const D: usize> Mul<&Vector<D>> for f32 {
    type Output = Vector<D>;

    fn mul(self, rhs: &Vector<D>) -> Self::Output {
        use Vector as V;
        match rhs {
            V::Constant(v) => V::Constant(v * self),
            V::Dense(v) => V::Dense(v * self),
            V::OneHot(v) => V::Sparse(v * self),
            V::Sparse(v) => V::Sparse(v * self),
            V::Zero(v) => V::Zero(v * self),
        }
    }
}

///////////////////
/// VECTOR ITER ///
///////////////////

impl<'a, const D: usize> IntoIterator for &'a Vector<D> {
    type Item = f32;
    type IntoIter = VectorWrapperIterator<'a, D>;

    fn into_iter(self) -> Self::IntoIter {
        VectorWrapperIterator::new(self)
    }
}

pub struct VectorWrapperIterator<'a, const D: usize> {
    pos: usize,
    vec: &'a Vector<D>,
}

impl<'a, const D: usize> VectorWrapperIterator<'a, D> {
    fn new(vec: &'a Vector<D>) -> Self {
        VectorWrapperIterator { pos: 0, vec }
    }
}

impl<'a, const D: usize> Iterator for VectorWrapperIterator<'a, D> {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos == D {
            None
        } else {
            use Vector as V;
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

impl<const D: usize> Index<usize> for Vector<D> {
    type Output = f32;

    fn index(&self, i: usize) -> &Self::Output {
        use Vector as V;
        match self {
            V::Constant(v) => &v[i],
            V::Dense(v) => &v[i],
            V::OneHot(v) => &v[i],
            V::Sparse(v) => &v[i],
            V::Zero(v) => &v[i],
        }
    }
}

impl<const D: usize> IndexMut<usize> for Vector<D> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        use Vector as V;
        match self {
            V::Constant(_v) => todo!(),
            V::Dense(v) => &mut v[index],
            V::OneHot(_v) => todo!(),
            V::Sparse(_v) => todo!(),
            V::Zero(_v) => todo!(),
        }
    }
}

impl<const D: usize> PartialEq for Vector<D> {
    fn eq(&self, other: &Self) -> bool {
        use Vector as V;
        match (self, other) {
            (V::Constant(v1), V::Constant(v2)) => v1 == v2,
            (V::Dense(v1), V::Dense(v2)) => v1 == v2,
            (V::OneHot(v1), V::OneHot(v2)) => v1 == v2,
            (V::Sparse(v1), V::Sparse(v2)) => v1 == v2,
            (V::Zero(v1), V::Zero(v2)) => v1 == v2,
            _ => false, // TODO: equality between mixed flavors.
        }
    }
}
