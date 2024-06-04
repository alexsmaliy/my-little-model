use std::fmt::Display;
use std::ops::{Add, Mul, Sub};

use super::order::Order;
use super::OldVectorDoNotUse;

#[allow(unused_imports)] pub(super) use constant::ConstantMatrix;
#[allow(unused_imports)] pub(super) use dense::DenseMatrix;
#[allow(unused_imports)] pub(super) use diagonal::DiagonalMatrix;
#[allow(unused_imports)] pub(super) use identity::IdentityMatrix;
#[allow(unused_imports)] pub(super) use sparse::SparseMatrix;
#[allow(unused_imports)] pub use wrapper::MatrixWrapper;
#[allow(unused_imports)] pub(super) use zero::ZeroMatrix;

mod constant;
mod dense;
mod diagonal;
mod identity;
mod sparse;
mod wrapper;
mod zero;

#[derive(Clone, Debug)]
pub struct OldMatrixDoNotUse<const R: usize, const C: usize>(pub(super) [f32; R*C])
    where [(); R*C]: Sized;

impl<const R: usize, const C: usize> OldMatrixDoNotUse<R, C> where [(); R*C]: Sized {
    pub fn zero() -> Self {
        OldMatrixDoNotUse([0f32; R*C])
    }

    pub fn from_arr(arr: [f32; R*C]) -> Self {
        OldMatrixDoNotUse(arr)
    }

    pub fn from_cols(cols: &[[f32; R]; C]) -> Self {
        let mut arr = [0f32; R*C];
        for c_ind in 0..C {
            let from = c_ind * R;
            let to = from + R;
            let col = &cols[c_ind];
            arr[from..to].copy_from_slice(col);
        }
        OldMatrixDoNotUse(arr)
    }

    /**
        Matrix transpose (eagerly).
     */
    pub fn T(&self) -> OldMatrixDoNotUse<C, R> where [(); C*R]: Sized {
        let mut arr = [0f32; C*R];
        for i in 0..(C * R) {
            arr[i] = self.0[(i % C) * R + (i / C)];
        }
        OldMatrixDoNotUse(arr)
    }
}

impl<const D: usize> OldMatrixDoNotUse<D, D> where [(); D*D]: Sized {
    #[allow(non_snake_case)]
    pub fn I() -> Self {
        let mut arr = [0f32; D*D];
        for i in 0..D {
            arr[i * D + i] = 1f32;
        }
        OldMatrixDoNotUse(arr)
    }

    /**
        Returns a square matrix whose main diagonal is `main_diag`.
     */
    pub fn diag(main_diag: &[f32; D]) -> Self {
        let mut arr = [0f32; D*D];
        for i in 0..D {
            arr[i * D + i] = main_diag[i];
        }
        OldMatrixDoNotUse(arr)
    }
}

/////////////////////////
/// &Matrix + &Matrix ///
/////////////////////////

impl<const R: usize, const C: usize> Add for &OldMatrixDoNotUse<R, C> where [(); R*C]: Sized {
    type Output = OldMatrixDoNotUse<R, C>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut arr = [0f32; R*C];
        self.0.iter()
            .zip(rhs.0.iter())
            .map(|(n, m)| n + m)
            .enumerate()
            .for_each(|(i, x)| arr[i] = x);
        OldMatrixDoNotUse(arr)
    }
}

/////////////////////
/// f32 + &Matrix ///
/////////////////////

impl<const R: usize, const C: usize> Add<&OldMatrixDoNotUse<R, C>> for f32 where [(); R*C]: Sized {
    type Output = OldMatrixDoNotUse<R, C>;

    fn add(self, rhs: &OldMatrixDoNotUse<R, C>) -> Self::Output {
        let mut arr = [0f32; R*C];
        rhs.0.iter()
           .enumerate()
           .for_each(|(i, x)| arr[i] = x + self);
        OldMatrixDoNotUse(arr)
    }
}

/////////////////////
/// &Matrix + f32 ///
/////////////////////

impl<const R: usize, const C: usize> Add<f32> for &OldMatrixDoNotUse<R, C> where [(); R*C]: Sized {
    type Output = OldMatrixDoNotUse<R, C>;

    fn add(self, rhs: f32) -> Self::Output {
        let mut arr = [0f32; R*C];
        self.0.iter()
            .enumerate()
            .for_each(|(i, x)| arr[i] = x + rhs);
        OldMatrixDoNotUse(arr)
    }
}

/////////////////////////
/// &Matrix - &Matrix ///
/////////////////////////

impl<const R: usize, const C: usize> Sub for &OldMatrixDoNotUse<R, C> where [(); R*C]: Sized {
    type Output = OldMatrixDoNotUse<R, C>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut arr = [0f32; R*C];
        self.0.iter()
            .zip(rhs.0.iter())
            .map(|(n, m)| n - m)
            .enumerate()
            .for_each(|(i, x)| arr[i] = x);
        OldMatrixDoNotUse(arr)
    }
}

/////////////////////////
/// &Matrix * &Matrix ///
/////////////////////////

impl<const R: usize, const C: usize, const C2: usize> Mul<&OldMatrixDoNotUse<C, C2>> for &OldMatrixDoNotUse<R, C>
    where
        [(); R*C]:  Sized, // boilerplate
        [(); C*C2]: Sized, // boilerplate
        [(); R*C2]: Sized, // boilerplate
{
    type Output = OldMatrixDoNotUse<R, C2>;

    fn mul(self, rhs: &OldMatrixDoNotUse<C, C2>) -> Self::Output {
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
        OldMatrixDoNotUse(arr)
    }
}

/////////////////////////
/// &Matrix * &Vector ///
/////////////////////////

impl<const R: usize, const C: usize> Mul<&OldVectorDoNotUse<C>> for &OldMatrixDoNotUse<R, C> where [(); R*C]: Sized {
    type Output = OldVectorDoNotUse<R>;

    fn mul(self, rhs: &OldVectorDoNotUse<C>) -> Self::Output {
        let mut arr = [0f32; R];
        for i in 0..R {
            let x: f32 = rhs.0
                .into_iter()
                .enumerate()
                .map(|(j, x)| x * self.0[j * R + i])
                .sum();
            arr[i] = x;
        }
        OldVectorDoNotUse(arr)
    }
} 

/////////////////////
/// f32 * &Matrix ///
/////////////////////

impl<const R: usize, const C: usize> Mul<&OldMatrixDoNotUse<R, C>> for f32 where [(); R*C]: Sized {
    type Output = OldMatrixDoNotUse<R, C>;

    fn mul(self, rhs: &OldMatrixDoNotUse<R, C>) -> Self::Output {
        let mut arr = [0f32; R*C];
        rhs.0.iter()
           .enumerate()
           .for_each(|(i, x)| arr[i] = x * self);
        OldMatrixDoNotUse(arr)
    }
}

/////////////////////
/// &Matrix * f32 ///
/////////////////////

impl<const R: usize, const C: usize> Mul<f32> for &OldMatrixDoNotUse<R, C> where [(); R*C]: Sized {
    type Output = OldMatrixDoNotUse<R, C>;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut arr = [0f32; R*C];
        self.0.iter()
            .enumerate()
            .for_each(|(i, x)| arr[i] = x * rhs);
        OldMatrixDoNotUse(arr)
    }
}

/////////////////////
/// UTILITY IMPLS ///
/////////////////////

impl<const R: usize, const C: usize> Display for OldMatrixDoNotUse<R, C> where [(); R*C]: Sized {
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

impl<const R: usize, const C: usize> From<[f32; R*C]> for OldMatrixDoNotUse<R, C> {
    fn from(arr: [f32; R*C]) -> Self {
        OldMatrixDoNotUse(arr)
    }
}

impl<const R: usize, const C: usize> From<[[f32; R]; C]> for OldMatrixDoNotUse<R, C> where [(); R*C]: Sized {
    fn from(cols: [[f32; R]; C]) -> Self {
        Self::from_cols(&cols)
    }
}

impl<const R: usize, const C: usize> PartialEq for OldMatrixDoNotUse<R, C> where [(); R*C]: Sized {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

mod tests {
    #[test]
    fn matrix_from_cols_multiply() {
        use super::wrapper::MatrixWrapper;

        let m1 = MatrixWrapper::from_cols(&[
            [1., 2.],
            [3., 4.],
            [5., 6.],
        ]);

        let m2 = MatrixWrapper::from_cols(&[
            [1., 2., 3.],
            [4., 5., 6.],
        ]);

        let expected = MatrixWrapper::from_cols(&[
            [22., 28.],
            [49., 64.],
        ]);

        assert_eq!(&m1 * &m2, expected);
    }
}
