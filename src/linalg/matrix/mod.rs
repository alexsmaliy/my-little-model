use std::fmt::Display;
use std::ops::{Add, Mul, Sub};

use super::order::Order;
use super::Vector;

pub(super) use constant::ConstantMatrix;
pub(super) use dense::DenseMatrix;
pub(super) use diagonal::DiagonalMatrix;
pub(super) use identity::IdentityMatrix;
pub(super) use sparse::SparseMatrix;
pub(super) use wrapper::MatrixWrapper;
pub(super) use zero::ZeroMatrix;

mod constant;
mod dense;
mod diagonal;
mod identity;
mod sparse;
mod wrapper;
mod zero;

mod test {
    #[test]
    fn dense_matrix_wrapper_multiply() {
        use super::dense::DenseMatrix;
        use super::wrapper::MatrixWrapper;
        
        let m1 = MatrixWrapper::Dense(DenseMatrix::<2, 3>([1., 2., 3., 4., 5., 6.], super::Order::COLUMNS));
        let m2 = MatrixWrapper::Dense(DenseMatrix::<3, 2>([1., 2., 3., 4., 5., 6.], super::Order::COLUMNS));
        let expected = MatrixWrapper::Dense(DenseMatrix::<2, 2>([22., 28., 49., 64.], super::Order::COLUMNS));
        assert_eq!(&m1 * &m2, expected);
    }
}

#[derive(Clone, Debug)]
pub struct Matrix<const R: usize, const C: usize>(pub(super) [f32; R*C])
    where [(); R*C]: Sized;

impl<const R: usize, const C: usize> Matrix<R, C> where [(); R*C]: Sized {
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
    pub fn T(&self) -> Matrix<C, R> where [(); C*R]: Sized {
        let mut arr = [0f32; C*R];
        for i in 0..(C * R) {
            arr[i] = self.0[(i % C) * R + (i / C)];
        }
        Matrix(arr)
    }
}

impl<const D: usize> Matrix<D, D> where [(); D*D]: Sized {
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

impl<const R: usize, const C: usize> Add for &Matrix<R, C> where [(); R*C]: Sized {
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

impl<const R: usize, const C: usize> Add<&Matrix<R, C>> for f32 where [(); R*C]: Sized {
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

impl<const R: usize, const C: usize> Add<f32> for &Matrix<R, C> where [(); R*C]: Sized {
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

impl<const R: usize, const C: usize> Sub for &Matrix<R, C> where [(); R*C]: Sized {
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
        [(); R*C]:  Sized, // boilerplate
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

impl<const R: usize, const C: usize> Mul<&Vector<C>> for &Matrix<R, C> where [(); R*C]: Sized {
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

impl<const R: usize, const C: usize> Mul<&Matrix<R, C>> for f32 where [(); R*C]: Sized {
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

impl<const R: usize, const C: usize> Mul<f32> for &Matrix<R, C> where [(); R*C]: Sized {
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

impl<const R: usize, const C: usize> Display for Matrix<R, C> where [(); R*C]: Sized {
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

impl<const R: usize, const C: usize> From<[[f32; R]; C]> for Matrix<R, C> where [(); R*C]: Sized {
    fn from(cols: [[f32; R]; C]) -> Self {
        Self::from_cols(&cols)
    }
}

impl<const R: usize, const C: usize> PartialEq for Matrix<R, C> where [(); R*C]: Sized {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}
