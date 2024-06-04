use std::fmt::Display;
use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

use super::Matrix;

#[allow(unused_imports)] pub(super) use constant::ConstantVector;
#[allow(unused_imports)] pub(super) use dense::DenseVector;
#[allow(unused_imports)] pub(super) use onehot::OneHotVector;
#[allow(unused_imports)] pub(super) use sparse::SparseVector;
#[allow(unused_imports)] pub(super) use wrapper::{VectorWrapper, VectorWrapperIterator};
#[allow(unused_imports)] pub(super) use zero::ZeroVector;

mod constant;
mod dense;
mod onehot;
mod sparse;
mod wrapper;
mod zero;

trait CanDotProduct<V> {
    fn dot(&self, other: V) -> f32;
}

trait CanMap {
    type Output;
    fn map(&self, f: impl Fn(f32) -> f32) -> Self::Output;
}

trait CanOuterProduct<V> {
    type Output;
    fn outer(&self, other: V) -> Self::Output;
}

#[derive(Clone, Debug)]
pub struct Vector<const N: usize>(pub(super) [f32; N]);

impl<const N: usize> Vector<N> {
    pub fn zero() -> Vector<N> {
        Self([0f32; N])
    }
    
    pub fn one_hot(n: usize) -> Vector<N> {
        let mut arr = [0f32; N];
        arr[n] = 1f32;
        Self(arr)
    }

    pub fn from_arr(arr: [f32; N]) -> Vector<N> {
        Vector(arr)
    }

    pub fn from_fun(f: impl Fn() -> f32) -> Vector<N> {
        let mut arr = [0f32; N];
        for i in 0..N {
            arr[i] = f();
        }
        Self(arr)
    }

    /**
        Ordinary dot product.
     */
    pub fn dot(&self, other: &Vector<N>) -> f32 {
        let mut product = 0f32;
        for i in 0..N {
            product += self.0[i] * other.0[i];
        }
        product
    }

    /**
        Vectorizes `f` over the elements of `self`.
     */
    pub fn map(&self, f: impl Fn(f32) -> f32) -> Self {
        Self(self.0.clone().map(f))
    }

    pub fn sum(&self) -> f32 {
        self.0.iter().sum()
    }

    /**
        Squared 2-norm.
     */
    pub fn sum_of_squares(&self) -> f32 {
        let x = self.0.iter().fold(0f32, |acc, x| acc + x*x);
        x
    }

    /**
        Vector outer product: `u.outer(&v)` creates a matrix of |`v`| copies of `u`, each scaled by `v[i]`.
     */
    pub fn outer<const M: usize>(&self, other: &Vector<M>) -> Matrix<N, M> where [(); N*M]: Sized {
        let mut arr = [0f32; N*M];
        for m in 0..M {
            arr[m*N..(m+1)*N].copy_from_slice(&(other[m] * self).0);
        }
        Matrix::from_arr(arr)
    }

    // #[allow(non_snake_case)]
    // pub fn T(&self) -> Matrix<1, N> where [(); 1*N]: Sized {
    //     let arr: [f32; 1*N] = self.0.iter().copied().collect::<Vec<f32>>().try_into().unwrap(); // TODO: fix
    //     Matrix::<1, N>::from_arr(arr)
    // }
}

/////////////////////////
/// &Vector + &Vector ///
/////////////////////////

impl<const N: usize> Add for &Vector<N> {
    type Output = Vector<N>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut arr = [0f32; N];
        self.0.iter()
            .zip(rhs.0.iter())
            .map(|(n, m)| n + m)
            .enumerate()
            .for_each(|(i, x)| arr[i] = x);
        Vector(arr)
    }
}

/////////////////////
/// f32 + &Vector ///
/////////////////////

impl<const N: usize> Add<&Vector<N>> for f32 {
    type Output = Vector<N>;

    fn add(self, rhs: &Vector<N>) -> Self::Output {
        let mut arr = [0f32; N];
        rhs.0.iter()
           .enumerate()
           .for_each(|(i, x)| arr[i] = x + self);
        Vector(arr)
    }
}

/////////////////////
/// &Vector + f32 ///
/////////////////////

impl<const N: usize> Add<f32> for &Vector<N> {
    type Output = Vector<N>;

    fn add(self, rhs: f32) -> Self::Output {
        let mut arr = [0f32; N];
        self.0.iter()
            .enumerate()
            .for_each(|(i, x)| arr[i] = x + rhs);
        Vector(arr)
    }
}

/////////////////////////
/// &Vector - &Vector ///
/////////////////////////

impl<const N: usize> Sub for &Vector<N> {
    type Output = Vector<N>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut arr = [0f32; N];
        self.0.iter()
            .zip(rhs.0.iter())
            .map(|(n, m)| n - m)
            .enumerate()
            .for_each(|(i, x)| arr[i] = x);
        Vector(arr)
    }
}

/////////////////////
/// &Vector - f32 ///
/////////////////////

impl<const N: usize> Sub<f32> for &Vector<N> {
    type Output = Vector<N>;

    fn sub(self, rhs: f32) -> Self::Output {
        self + (-rhs)
    }
}

/////////////////////////
/// &Vector * &Vector ///
/////////////////////////

impl<const N: usize> Mul for &Vector<N> {
    type Output = f32;

    fn mul(self, rhs: Self) -> Self::Output {
        self.dot(rhs) // TODO: reconsider if this is conceptually ambiguous.
    }
}

/////////////////////
/// f32 * &Vector ///
/////////////////////

impl<const N: usize> Mul<&Vector<N>> for f32 {
    type Output = Vector<N>;

    fn mul(self, rhs: &Vector<N>) -> Self::Output {
        let mut arr = [0f32; N];
        rhs.0.iter()
           .enumerate()
           .for_each(|(i, x)| arr[i] = x * self);
        Vector(arr)
    }
}

/////////////////////
/// &Vector * f32 ///
/////////////////////

impl<const N: usize> Mul<f32> for &Vector<N> {
    type Output = Vector<N>;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut arr = [0f32; N];
        self.0.iter()
            .enumerate()
            .for_each(|(i, x)| arr[i] = x * rhs);
        Vector(arr)
    }
}

////////////////////
/// -1 * &Vector ///
////////////////////

impl<const N: usize> Neg for &Vector<N> {
    type Output = Vector<N>;

    fn neg(self) -> Self::Output {
        Vector(self.0.map(|x| -x))
    }
}

/////////////////////
/// UTILITY IMPLS ///
/////////////////////

impl<const N: usize> Display for Vector<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // include superscript T to indicate that it's a column vector
        write!(f, "[{}]\u{1D40}", self.0.map(|n| n.to_string()).join(","))
    }
}

impl<const N: usize> From<[f32; N]> for Vector<N> {
    fn from(arr: [f32; N]) -> Self {
        Vector(arr)
    }
}

impl<const N: usize> From<Vector<N>> for [f32; N] {
    fn from(vector: Vector<N>) -> Self {
        vector.0
    }
}

impl<const N: usize> Index<usize> for Vector<N> {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const N: usize> IndexMut<usize> for Vector<N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const N: usize> PartialEq for Vector<N> {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}
