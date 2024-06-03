use std::fmt::Display;
use std::marker::PhantomData;
use std::ops::{Add, Mul, Sub};

use super::Vector;

//////////////////

pub enum MatrixWrapper<const R: usize, const C: usize> where [(); R*C]: Sized {
    Constant(ConstantMatrix<R, C>),
    Dense(DenseMatrix<R, C>),
    Diagonal(DiagonalMatrix<R, C>),
    Identity(IdentityMatrix<R, C>),
    Sparse(SparseMatrix<R, C>),
    Zero(ZeroMatrix<R, C>),
}

pub enum VectorWrapper<const D: usize> {
    Constant(ConstantVector<D>),
    Dense(DenseVector<D>),
    OneHot(OneHotVector<D>),
    Sparse(SparseVector<D>),
    Zero(ZeroVector<D>),
}

impl<const R: usize, const C: usize> MatrixWrapper<R, C> where [(); R*C]: Sized {
    pub fn constant(c: f32) -> Self {
        Self::Constant(ConstantMatrix(c))
    }

    pub fn sparse() -> Self {
        Self::Sparse(SparseMatrix(Vec::new(), Vec::new()))
    }

    pub fn zero() -> Self {
        Self::Zero(ZeroMatrix(0f32))
    }
}

impl<const D: usize> MatrixWrapper<D, D> where [(); D*D]: Sized {
    pub fn diagonal(v: DenseVector<D>) -> Self {
        Self::Diagonal(DiagonalMatrix(v))
    }
    
    pub fn identity() -> Self {
        Self::Identity(IdentityMatrix(0f32, 1f32))
    }
}

impl<const D: usize> VectorWrapper<D> {
    pub fn onehot(i: usize) -> Self {
        Self::OneHot(OneHotVector(0f32, 1f32, i))
    }

    pub fn sparse() -> Self {
        Self::Sparse(SparseVector(Vec::new(), Vec::new()))
    }

    pub fn zero() -> Self {
        Self::Zero(ZeroVector(0f32))
    }
}

impl<const R: usize, const C: usize> Mul<&VectorWrapper<C>> for &MatrixWrapper<R, C> where [(); R*C]: Sized {
    type Output = VectorWrapper<R>;
    
    fn mul(self, rhs: &VectorWrapper<C>) -> Self::Output {
        use MatrixWrapper as M;
        use VectorWrapper as V;
        match (self, rhs) {
            (&M::Identity(ref m), &V::Constant(ref v)) => V::Constant(ConstantVector::<R>(0f32)),
            (&M::Dense(ref m), &V::Constant(ref v)) => V::Constant(ConstantVector::<R>(0f32)),
            (&M::Identity(ref m), &V::Dense(ref v)) => V::Constant(ConstantVector::<R>(0f32)),
            (&M::Dense(ref m), &V::Dense(ref v)) => V::Constant(ConstantVector::<R>(0f32)),
            _ => unimplemented!(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConstantMatrix<const R: usize, const C: usize>(f32);

#[derive(Clone, Debug)]
pub struct DenseMatrix<const R: usize, const C: usize>(pub(super) [f32; R*C]) where [(); R*C]: Sized;

#[derive(Clone, Debug)]
pub struct DiagonalMatrix<const R: usize, const C: usize>(DenseVector<R>) where [(); R*C]: Sized;

#[derive(Clone, Debug)]
pub struct IdentityMatrix<const R: usize, const C: usize>(f32, f32);

#[derive(Clone, Debug)]
pub struct SparseMatrix<const R: usize, const C: usize>(Vec<usize>, Vec<f32>);

#[derive(Clone, Debug)]
pub struct ZeroMatrix<const R: usize, const C: usize>(f32);

#[derive(Clone, Debug)]
pub struct ConstantVector<const D: usize>(f32);

#[derive(Clone, Debug)]
pub struct DenseVector<const D: usize>(pub(super) [f32; D]);

#[derive(Clone, Debug)]
pub struct OneHotVector<const D: usize>(f32, f32, usize);

#[derive(Clone, Debug)]
pub struct SparseVector<const D: usize>(Vec<usize>, Vec<f32>);

#[derive(Clone, Debug)]
pub struct ZeroVector<const D: usize>(f32);

//////////////////
impl<const R: usize, const C: usize> Display for ConstantMatrix<R, C> where [(); R*C]: {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        (0..R).into_iter()
            .map(|_| std::iter::repeat(&self.0)
                .map(<f32>::to_string)
                .take(C)
                .collect::<Vec<_>>()
                .join(","))
            .collect::<Vec<_>>()
            .join("],[");
        write!(f, "]")
    }
}

impl<const R: usize, const C: usize> Display for DenseMatrix<R, C> where [(); R*C]: {
    /// Displays the rows of a dense matrix.
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

impl<const D: usize> Display for DiagonalMatrix<D, D> where [(); D*D]: {
    /// Displays the rows of a diagonal matrix.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        (0..D).into_iter()
            .map(|r| {
                let before = (0..r).into_iter().map(|_| 0f32.to_string());
                let val = std::iter::once(self.0.0[r].to_string()); // TODO: index dense vector
                let after = ((r+1)..D).into_iter().map(|_| 0f32.to_string());
                before.chain(val).chain(after).collect::<Vec<_>>().join(",")
            }).collect::<Vec<_>>().join("],[");
        write!(f, "]")
    }
}

impl<const D: usize> Display for IdentityMatrix<D, D> where [(); D*D]: {
    /// Displays the rows of an identity matrix.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        (0..D).into_iter()
            .map(|r| {
                let before = (0..r).into_iter().map(|_| 0f32.to_string());
                let val = std::iter::once(self.0.to_string());
                let after = ((r+1)..D).into_iter().map(|_| 0f32.to_string());
                before.chain(val).chain(after).collect::<Vec<_>>().join(",")
            }).collect::<Vec<_>>().join("],[");
        write!(f, "]")
    }
}

impl<const R: usize, const C: usize> Display for ZeroMatrix<R, C> where [(); R*C]: {
    /// Displays the rows of a zero matrix.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        (0..R).into_iter()
            .map(|_| std::iter::repeat("0")
                .take(C)
                .collect::<Vec<_>>()
                .join(","))
            .collect::<Vec<_>>()
            .join("],[");
        write!(f, "]")
    }
}

//////////////////

#[derive(Clone, Debug)]
pub struct Matrix<const R: usize, const C: usize>(pub(super) [f32; R*C])
    where [(); R*C]: Sized;

impl<const R: usize, const C: usize> Matrix<R, C> where [(); R*C]: {
    pub fn zero() -> Self {
        Matrix([0f32; R*C])
    }

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

    /**
        Matrix transpose (eagerly).
     */
    pub fn T(&self) -> Matrix<C, R> where [(); C*R]: {
        let mut arr = [0f32; C*R];
        for i in 0..(C * R) {
            arr[i] = self.0[(i % C) * R + (i / C)];
        }
        Matrix(arr)
    }
}

impl<const D: usize> Matrix<D, D> where [(); D*D]: {
    #[allow(non_snake_case)]
    pub fn I() -> Self {
        let mut arr = [0f32; D*D];
        for i in 0..D {
            arr[i * D + i] = 1f32;
        }
        Matrix(arr)
    }

    /**
        Returns a square matrix whose main diagonal is `main_diag`.
     */
    pub fn diag(main_diag: &[f32; D]) -> Self {
        let mut arr = [0f32; D*D];
        for i in 0..D {
            arr[i * D + i] = main_diag[i];
        }
        Matrix(arr)
    }
}

/////////////////////////
/// &Matrix + &Matrix ///
/////////////////////////

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

/////////////////////
/// f32 + &Matrix ///
/////////////////////

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

/////////////////////
/// &Matrix + f32 ///
/////////////////////

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

/////////////////////////
/// &Matrix - &Matrix ///
/////////////////////////

impl<const R: usize, const C: usize> Sub for &Matrix<R, C> where [(); R*C]: {
    type Output = Matrix<R, C>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut arr = [0f32; R*C];
        self.0.iter()
            .zip(rhs.0.iter())
            .map(|(n, m)| n - m)
            .enumerate()
            .for_each(|(i, x)| arr[i] = x);
        Matrix(arr)
    }
}

/////////////////////////
/// &Matrix * &Matrix ///
/////////////////////////

impl<const R: usize, const C: usize, const C2: usize> Mul<&Matrix<C, C2>> for &Matrix<R, C>
    where
        [(); R*C]: Sized,  // boilerplate
        [(); C*C2]: Sized, // boilerplate
        [(); R*C2]: Sized, // boilerplate
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

/////////////////////////
/// &Matrix * &Vector ///
/////////////////////////

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

/////////////////////
/// f32 * &Matrix ///
/////////////////////

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

/////////////////////
/// &Matrix * f32 ///
/////////////////////

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

/////////////////////
/// UTILITY IMPLS ///
/////////////////////

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
