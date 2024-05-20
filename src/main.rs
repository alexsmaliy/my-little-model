#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::{fmt::Display, ops::{Add, Index, IndexMut, Mul, Neg}};

fn main() {
    println!("Hello, world!");
}

// a dense column-ordered matrix
#[derive(Debug)]
pub struct Matrix<const R: usize, const C: usize>([f32; R*C]) where [(); R*C]: ;

impl<const R: usize, const C: usize> Matrix<R, C> where [(); R*C]: {
    #[allow(dead_code)]
    pub fn zero() -> Self {
        Matrix([0f32; R*C])
    }

    #[allow(dead_code)]
    pub fn from_arr(arr: [f32; R*C]) -> Self {
        Matrix(arr)
    }

    pub fn from_cols(cols: &[[f32; R]; C]) -> Self {
        let mut arr = [0f32; R*C];
        for c_ind in 0..C {
            let from = c_ind * R;
            let to = from + R;
            let col = &cols[c_ind];
            arr[from..to].copy_from_slice(col);
        }
        Matrix(arr)
    }

    #[allow(non_snake_case)]
    pub fn T(&self) -> Matrix<C, R> where [(); C*R]: {
        let mut arr = [0f32; C*R];
        for i in 0..(C * R) {
            arr[i] = self.0[(i % C) * R + (i / C)];
        }
        Matrix(arr)
    }
}

impl<const D: usize> Matrix<D, D> where [(); D*D]: {
    #[allow(dead_code, non_snake_case)]
    pub fn I() -> Self {
        let mut arr = [0f32; D*D];
        for i in 0..D {
            arr[i * D + i] = 1f32;
        }
        Matrix(arr)
    }
}

impl<const R: usize, const C: usize> Add for &Matrix<R, C> where [(); R*C]: {
    type Output = Matrix<R, C>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut arr = [0f32; R*C];
        self.0.iter()
            .zip(rhs.0.iter())
            .map(|(n, m)| n + m)
            .enumerate()
            .for_each(|(i, x)| arr[i] = x);
        Matrix(arr)
    }
}

impl<const R: usize, const C: usize> Add<&Matrix<R, C>> for f32 where [(); R*C]: {
    type Output = Matrix<R, C>;

    fn add(self, rhs: &Matrix<R, C>) -> Self::Output {
        let mut arr = [0f32; R*C];
        rhs.0.iter()
           .enumerate()
           .for_each(|(i, x)| arr[i] = x + self);
        Matrix(arr)
    }
}

impl<const R: usize, const C: usize> Add<f32> for &Matrix<R, C> where [(); R*C]: {
    type Output = Matrix<R, C>;

    fn add(self, rhs: f32) -> Self::Output {
        let mut arr = [0f32; R*C];
        self.0.iter()
            .enumerate()
            .for_each(|(i, x)| arr[i] = x + rhs);
        Matrix(arr)
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&Matrix<C, C2>> for &Matrix<R, C>
    where [(); R*C]: Sized, [(); C*C2]: Sized, [(); R*C2]: Sized // boilerplate
{
    type Output = Matrix<R, C2>;

    fn mul(self, rhs: &Matrix<C, C2>) -> Self::Output {
        let mut arr = [0f32; R*C2];
        for i in 0..(R * C2) {
            let from = (i / R) * C;
            let to = from + C;
            let x: f32 = rhs.0[from..to]
                .into_iter()
                .enumerate()
                .map(|(j, x)| x * self.0[j * R + i % R])
                .sum();
            arr[i] = x;
        }
        Matrix(arr)
    }
}

impl<const R: usize, const C: usize> Mul<&Vector<C>> for &Matrix<R, C> where [(); R*C]: {
    type Output = Vector<R>;

    fn mul(self, rhs: &Vector<C>) -> Self::Output {
        let mut arr = [0f32; R];
        for i in 0..R {
            let x: f32 = rhs.0
                .into_iter()
                .enumerate()
                .map(|(j, x)| x * self.0[j * R + i])
                .sum();
            arr[i] = x;
        }
        Vector(arr)
    }
} 

impl<const R: usize, const C: usize> Mul<&Matrix<R, C>> for f32 where [(); R*C]: {
    type Output = Matrix<R, C>;

    fn mul(self, rhs: &Matrix<R, C>) -> Self::Output {
        let mut arr = [0f32; R*C];
        rhs.0.iter()
           .enumerate()
           .for_each(|(i, x)| arr[i] = x * self);
        Matrix(arr)
    }
}

impl<const R: usize, const C: usize> Mul<f32> for &Matrix<R, C> where [(); R*C]: {
    type Output = Matrix<R, C>;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut arr = [0f32; R*C];
        self.0.iter()
            .enumerate()
            .for_each(|(i, x)| arr[i] = x * rhs);
        Matrix(arr)
    }
}

impl<const R: usize, const C: usize> Display for Matrix<R, C> where [(); R*C]: {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for r in 0..R {
            let mut arr = [0f32; C];
            for c in 0..C {
                arr[c] = self.0[c*R+r];
            }
            write!(f, "[{}]", arr.map(|n| n.to_string()).join(","))?;
        }
        write!(f, "]")
    }
}

impl<const R: usize, const C: usize> From<[f32; R*C]> for Matrix<R, C> {
    fn from(arr: [f32; R*C]) -> Self {
        Matrix(arr)
    }
}

impl<const R: usize, const C: usize> From<[[f32; R]; C]> for Matrix<R, C> where [(); R*C]: {
    fn from(cols: [[f32; R]; C]) -> Self {
        Self::from_cols(&cols)
    }
}

impl<const R: usize, const C: usize> PartialEq for Matrix<R, C> where [(); R*C]: {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

// a dense column vector
#[derive(Debug)]
struct Vector<const N: usize>([f32; N]);

impl<const N: usize> Vector<N> {
    #[allow(dead_code)]
    pub fn zero() -> Vector<N> {
        Self([0f32; N])
    }
    
    #[allow(dead_code)]
    pub fn one_hot(n: usize) -> Vector<N> {
        let mut arr = [0f32; N];
        arr[n] = 1f32;
        Self(arr)
    }

    #[allow(dead_code)]
    pub fn from_arr(arr: [f32; N]) -> Vector<N> {
        Vector(arr)
    }

    #[allow(dead_code)]
    pub fn from_fun(f: impl Fn() -> f32) -> Vector<N> {
        let mut arr = [0f32; N];
        for i in 0..N {
            arr[i] = f();
        }
        Self(arr)
    }

    pub fn dot(&self, other: &Vector<N>) -> f32 {
        let mut product = 0f32;
        for i in 0..N {
            product += self.0[i] * other.0[i];
        }
        product
    }
}

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

impl<const N: usize> Mul for &Vector<N> {
    type Output = f32;

    fn mul(self, rhs: Self) -> Self::Output {
        self.dot(rhs)
    }
}

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

impl<const N: usize> Neg for &Vector<N> {
    type Output = Vector<N>;

    fn neg(self) -> Self::Output {
        Vector(self.0.map(|x| -x))
    }
}

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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_twice() {
        let m = Matrix::from_cols(&[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
        ]);
        let mt = m.T();
        let mtt = mt.T();
        assert_eq!(m, mtt);
    }

    #[test]
    fn matrix_add_multiply_scalar() {
        let m = Matrix::<3, 3>::I();
        let m2 = 3f32 + &(2f32 * &(&(&m + 1f32) * 5f32));
        let expected = Matrix::from_cols(&[
            [23., 13., 13.],
            [13., 23., 13.],
            [13., 13., 23.],
        ]);
        assert_eq!(m2, expected);
    }

    #[test]
    fn matrix_multiplication() {
        let m1 = Matrix::from_cols(&[
            [1., 2.],
            [3., 4.],
            [5., 6.],
        ]);
        let m2 = Matrix::from_cols(&[
            [1., 2., 3.],
            [4., 5., 6.],
        ]);
        let expected = Matrix::from_cols(&[
            [22., 28.],
            [49., 64.],
        ]);
        assert_eq!(&m1 * &m2, expected);
    }

    #[test]
    fn matrix_vector_multiplication() {
        let m = Matrix::from_cols(&[
            [1., 2.],
            [3., 4.],
            [5., 6.],
        ]);
        let v = Vector::from_arr([1., 0., 2.]);
        let expected = Vector::from_arr([11., 14.]);
        assert_eq!(&m * &v, expected);
    }

    #[test]
    fn vector_indexing() {
        let v = &Vector::<5>::zero() + 5f32;
        for i in 0..5 {
            assert_eq!(v[i], 5.0);
        }
    }

    #[test]
    fn vector_mut_indexing() {
        let mut v = Vector::<5>::zero();
        for i in 0..5 {
            v[i] = 5. - i as f32;
        }
        for i in 0..5 {
            assert_eq!(v[i], 5. - i as f32);
        }
    }
}
