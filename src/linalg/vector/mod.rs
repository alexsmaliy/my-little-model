use std::fmt::Display;
use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

use super::OldMatrixDoNotUse;

#[allow(unused_imports)] pub(super) use constant::ConstantVector;
#[allow(unused_imports)] pub(super) use dense::DenseVector;
#[allow(unused_imports)] pub(super) use onehot::OneHotVector;
#[allow(unused_imports)] pub(super) use sparse::SparseVector;
#[allow(unused_imports)] pub use wrapper::{Vector, VectorWrapperIterator};
#[allow(unused_imports)] pub(super) use zero::ZeroVector;

mod constant;
mod dense;
mod onehot;
mod sparse;
pub mod traits;
mod wrapper;
mod zero;

#[derive(Clone, Debug)]
pub struct OldVectorDoNotUse<const N: usize>(pub(super) [f32; N]);

impl<const N: usize> OldVectorDoNotUse<N> {
    pub fn zero() -> OldVectorDoNotUse<N> {
        Self([0f32; N])
    }
    
    pub fn one_hot(n: usize) -> OldVectorDoNotUse<N> {
        let mut arr = [0f32; N];
        arr[n] = 1f32;
        Self(arr)
    }

    pub fn from_arr(arr: [f32; N]) -> OldVectorDoNotUse<N> {
        OldVectorDoNotUse(arr)
    }

    pub fn from_fun(f: impl Fn() -> f32) -> OldVectorDoNotUse<N> {
        let mut arr = [0f32; N];
        for i in 0..N {
            arr[i] = f();
        }
        Self(arr)
    }

    /**
        Ordinary dot product.
     */
    pub fn dot(&self, other: &OldVectorDoNotUse<N>) -> f32 {
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
    pub fn outer<const M: usize>(&self, other: &OldVectorDoNotUse<M>) -> OldMatrixDoNotUse<N, M> where [(); N*M]: Sized {
        let mut arr = [0f32; N*M];
        for m in 0..M {
            arr[m*N..(m+1)*N].copy_from_slice(&(other[m] * self).0);
        }
        OldMatrixDoNotUse::from_arr(arr)
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

impl<const N: usize> Add for &OldVectorDoNotUse<N> {
    type Output = OldVectorDoNotUse<N>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut arr = [0f32; N];
        self.0.iter()
            .zip(rhs.0.iter())
            .map(|(n, m)| n + m)
            .enumerate()
            .for_each(|(i, x)| arr[i] = x);
        OldVectorDoNotUse(arr)
    }
}

/////////////////////
/// f32 + &Vector ///
/////////////////////

impl<const N: usize> Add<&OldVectorDoNotUse<N>> for f32 {
    type Output = OldVectorDoNotUse<N>;

    fn add(self, rhs: &OldVectorDoNotUse<N>) -> Self::Output {
        let mut arr = [0f32; N];
        rhs.0.iter()
           .enumerate()
           .for_each(|(i, x)| arr[i] = x + self);
        OldVectorDoNotUse(arr)
    }
}

/////////////////////
/// &Vector + f32 ///
/////////////////////

impl<const N: usize> Add<f32> for &OldVectorDoNotUse<N> {
    type Output = OldVectorDoNotUse<N>;

    fn add(self, rhs: f32) -> Self::Output {
        let mut arr = [0f32; N];
        self.0.iter()
            .enumerate()
            .for_each(|(i, x)| arr[i] = x + rhs);
        OldVectorDoNotUse(arr)
    }
}

/////////////////////////
/// &Vector - &Vector ///
/////////////////////////

impl<const N: usize> Sub for &OldVectorDoNotUse<N> {
    type Output = OldVectorDoNotUse<N>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut arr = [0f32; N];
        self.0.iter()
            .zip(rhs.0.iter())
            .map(|(n, m)| n - m)
            .enumerate()
            .for_each(|(i, x)| arr[i] = x);
        OldVectorDoNotUse(arr)
    }
}

/////////////////////
/// &Vector - f32 ///
/////////////////////

impl<const N: usize> Sub<f32> for &OldVectorDoNotUse<N> {
    type Output = OldVectorDoNotUse<N>;

    fn sub(self, rhs: f32) -> Self::Output {
        self + (-rhs)
    }
}

/////////////////////////
/// &Vector * &Vector ///
/////////////////////////

impl<const N: usize> Mul for &OldVectorDoNotUse<N> {
    type Output = f32;

    fn mul(self, rhs: Self) -> Self::Output {
        self.dot(rhs) // TODO: reconsider if this is conceptually ambiguous.
    }
}

/////////////////////
/// f32 * &Vector ///
/////////////////////

impl<const N: usize> Mul<&OldVectorDoNotUse<N>> for f32 {
    type Output = OldVectorDoNotUse<N>;

    fn mul(self, rhs: &OldVectorDoNotUse<N>) -> Self::Output {
        let mut arr = [0f32; N];
        rhs.0.iter()
           .enumerate()
           .for_each(|(i, x)| arr[i] = x * self);
        OldVectorDoNotUse(arr)
    }
}

/////////////////////
/// &Vector * f32 ///
/////////////////////

impl<const N: usize> Mul<f32> for &OldVectorDoNotUse<N> {
    type Output = OldVectorDoNotUse<N>;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut arr = [0f32; N];
        self.0.iter()
            .enumerate()
            .for_each(|(i, x)| arr[i] = x * rhs);
        OldVectorDoNotUse(arr)
    }
}

////////////////////
/// -1 * &Vector ///
////////////////////

impl<const N: usize> Neg for &OldVectorDoNotUse<N> {
    type Output = OldVectorDoNotUse<N>;

    fn neg(self) -> Self::Output {
        OldVectorDoNotUse(self.0.map(|x| -x))
    }
}

/////////////////////
/// UTILITY IMPLS ///
/////////////////////

impl<const N: usize> Display for OldVectorDoNotUse<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // include superscript T to indicate that it's a column vector
        write!(f, "[{}]\u{1D40}", self.0.map(|n| n.to_string()).join(","))
    }
}

impl<const N: usize> From<[f32; N]> for OldVectorDoNotUse<N> {
    fn from(arr: [f32; N]) -> Self {
        OldVectorDoNotUse(arr)
    }
}

impl<const N: usize> From<OldVectorDoNotUse<N>> for [f32; N] {
    fn from(vector: OldVectorDoNotUse<N>) -> Self {
        vector.0
    }
}

impl<const N: usize> Index<usize> for OldVectorDoNotUse<N> {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const N: usize> IndexMut<usize> for OldVectorDoNotUse<N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const N: usize> PartialEq for OldVectorDoNotUse<N> {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}
