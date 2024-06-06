use std::fmt::Display;
use std::marker::PhantomData;
use std::ops::{Add, Mul, Neg, Sub};

use crate::linalg::vector::DenseVector;

use super::Order;

use super::constant::ConstantMatrix;
use super::diagonal::DiagonalMatrix;
use super::identity::IdentityMatrix;
use super::sparse::SparseMatrix;
use super::zero::ZeroMatrix;

#[derive(Clone, Debug)]
pub struct DenseMatrix<const R: usize, const C: usize> where [(); R*C]: Sized {
    pub(super) data: Box<[f32]>,
    pub(super) order: Order,
    pub(super) size_marker: PhantomData<[[f32; R]; C]>,
}

impl<const R: usize, const C: usize> DenseMatrix<R, C>
    where [(); R*C]: Sized
{
    // constructor
    pub(crate) fn from_arr(arr: [f32; R*C]) -> Self {
        DenseMatrix {
            data: Box::new(arr),
            order: Order::COLS,
            size_marker: PhantomData,
        }
    }

    // constructor
    pub(super) fn from_cols(cols: &[[f32; R]; C]) -> Self {
        let mut arr = [0f32; R*C];
        for c_ind in 0..C {
            let from = c_ind * R;
            let to = from + R;
            let col = &cols[c_ind];
            arr[from..to].copy_from_slice(col);
        }
        DenseMatrix {
            data: Box::new(arr),
            order: Order::COLS,
            size_marker: PhantomData,
        }
    }
    
    pub(super) fn T(&self) -> DenseMatrix<C, R> where [(); C*R]: Sized {
        DenseMatrix {
            data: self.data.clone(),
            order: Order::ROWS,
            size_marker: PhantomData,
        }
    }
}

impl<const R: usize, const C: usize> PartialEq for DenseMatrix<R, C> where [(); R*C]: Sized {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.order == other.order // TODO: handle transposed args.
    }
}

//////////////////////////////
/// DENSE MATRIX ADD IMPLS ///
//////////////////////////////

impl<const R: usize, const C: usize> Add<&ConstantMatrix<R, C>> for &DenseMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = DenseMatrix<R, C>;

    fn add(self, _rhs: &ConstantMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Add<&DenseMatrix<R, C>> for &DenseMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = DenseMatrix<R, C>;

    fn add(self, _rhs: &DenseMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Add<&DiagonalMatrix<R, C>> for &DenseMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = DenseMatrix<R, C>;

    fn add(self, _rhs: &DiagonalMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Add<&IdentityMatrix<R, C>> for &DenseMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = DenseMatrix<R, C>;

    fn add(self, _rhs: &IdentityMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Add<&SparseMatrix<R, C>> for &DenseMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = DenseMatrix<R, C>;

    fn add(self, _rhs: &SparseMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Add<&ZeroMatrix<R, C>> for &DenseMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = DenseMatrix<R, C>;

    fn add(self, _rhs: &ZeroMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

//////////////////////////////
/// DENSE MATRIX SUB IMPLS ///
//////////////////////////////

impl<const R: usize, const C: usize> Sub<&ConstantMatrix<R, C>> for &DenseMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = DenseMatrix<R, C>;

    fn sub(self, _rhs: &ConstantMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Sub<&DenseMatrix<R, C>> for &DenseMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = DenseMatrix<R, C>;

    fn sub(self, rhs: &DenseMatrix<R, C>) -> Self::Output {
        let f = |(a, b)| a - b;
        let us = self.data.iter().copied();
        let them = rhs.data.iter().copied();
        let mapped = us.zip(them).map(f).collect::<Box<[f32]>>();
        DenseMatrix {
            data: mapped,
            order: self.order,
            size_marker: PhantomData,
        }
    }
}

impl<const R: usize, const C: usize> Sub<&DiagonalMatrix<R, C>> for &DenseMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = DenseMatrix<R, C>;

    fn sub(self, _rhs: &DiagonalMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Sub<&IdentityMatrix<R, C>> for &DenseMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = DenseMatrix<R, C>;

    fn sub(self, _rhs: &IdentityMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Sub<&SparseMatrix<R, C>> for &DenseMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = DenseMatrix<R, C>;

    fn sub(self, _rhs: &SparseMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Sub<&ZeroMatrix<R, C>> for &DenseMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = DenseMatrix<R, C>;

    fn sub(self, _rhs: &ZeroMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

//////////////////////////////
/// DENSE MATRIX MUL IMPLS ///
//////////////////////////////

impl<const R: usize, const C: usize, const C2: usize> Mul<&ConstantMatrix<C, C2>> for &DenseMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
        [(); R*C2]: Sized,
{
    type Output = DenseMatrix<R, C2>;

    fn mul(self, _rhs: &ConstantMatrix<C, C2>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&DenseMatrix<C, C2>> for &DenseMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
        [(); R*C2]: Sized,
{
    type Output = DenseMatrix<R, C2>;

    fn mul(self, rhs: &DenseMatrix<C, C2>) -> Self::Output {
        // TODO: handle transposed args.
        let product = (0..R*C2).map(|i| {
            let from = (i / R) * C;
            let to = from + C;
            rhs.data[from..to].into_iter()
               .enumerate()
               .map(|(j, x)| x * self.data[j * R + i % R])
               .sum()
        }).collect();
        DenseMatrix {
            data: product,
            order: self.order,
            size_marker: PhantomData,
        }

        // let mut arr = [0f32; R*C2];
        // for i in 0..(R * C2) {
        //     let from = (i / R) * C;
        //     let to = from + C;
        //     let x: f32 = rhs.0[from..to]
        //         .into_iter()
        //         .enumerate()
        //         .map(|(j, x)| x * self.0[j * R + i % R])
        //         .sum();
        //     arr[i] = x;
        // }
        // DenseMatrix(arr, self.1)
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&DiagonalMatrix<C, C2>> for &DenseMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
        [(); R*C2]: Sized,
{
    type Output = DenseMatrix<R, C2>;

    fn mul(self, _rhs: &DiagonalMatrix<C, C2>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&IdentityMatrix<C, C2>> for &DenseMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
        [(); R*C2]: Sized,
{
    type Output = DenseMatrix<R, C2>;

    fn mul(self, _rhs: &IdentityMatrix<C, C2>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&SparseMatrix<C, C2>> for &DenseMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
        [(); R*C2]: Sized,
{
    type Output = DenseMatrix<R, C2>;

    fn mul(self, _rhs: &SparseMatrix<C, C2>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&ZeroMatrix<C, C2>> for &DenseMatrix<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
{
    type Output = ZeroMatrix<R, C2>;

    fn mul(self, _rhs: &ZeroMatrix<C, C2>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Mul<&DenseVector<C>> for &DenseMatrix<R, C>
    where [(); R*C]: Sized
{
    type Output = DenseVector<R>;

    fn mul(self, rhs: &DenseVector<C>) -> Self::Output {
        let product = (0..R).map(|i| {
            rhs.data.into_iter()
               .enumerate()
               .map(|(j, x)| x * self.data[j * R + i])
               .sum()
        }).collect();

        DenseVector {
            data: product,
            size_marker: PhantomData,
        }
    }
}

///////////////////////////////
/// DENSE MATRIX MATH IMPLS ///
///////////////////////////////

impl<const R: usize, const C: usize> Add<&DenseMatrix<R, C>> for f32 where [(); R*C]: Sized {
    type Output = DenseMatrix<R, C>;

    fn add(self, rhs: &DenseMatrix<R, C>) -> Self::Output {
        let f = |x| x + self;
        let data = rhs.data.iter().map(f).collect();
        DenseMatrix {
            data,
            order: rhs.order,
            size_marker: PhantomData,
        }
    }
}

impl<const R: usize, const C: usize> Add<f32> for &DenseMatrix<R, C> where [(); R*C]: Sized {
    type Output = DenseMatrix<R, C>;

    fn add(self, _rhs: f32) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Sub<f32> for &DenseMatrix<R, C> where [(); R*C]: Sized {
    type Output = DenseMatrix<R, C>;

    fn sub(self, _rhs: f32) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Mul<&DenseMatrix<R, C>> for f32 where [(); R*C]: Sized {
    type Output = DenseMatrix<R, C>;

    fn mul(self, _rhs: &DenseMatrix<R, C>) -> Self::Output {
        todo!()
    }
}

impl<const R: usize, const C: usize> Mul<f32> for &DenseMatrix<R, C> where [(); R*C]: Sized {
    type Output = DenseMatrix<R, C>;

    fn mul(self, rhs: f32) -> Self::Output {
        let f = |x| x * rhs;
        let data = self.data.iter().map(f).collect();
        DenseMatrix {
            data,
            order: self.order,
            size_marker: PhantomData,
        }
    }
}

impl<const R: usize, const C: usize> Neg for &DenseMatrix<R, C> where [(); R*C]: Sized {
    type Output = DenseMatrix<R, C>;

    fn neg(self) -> Self::Output {
        todo!()
    }
}

//////////////////////////////////
/// DENSE MATRIX UTILITY IMPLS ///
//////////////////////////////////

impl<const R: usize, const C: usize> Display for DenseMatrix<R, C> where [(); R*C]: Sized {
    /// Displays the rows of a dense matrix.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for r in 0..R {
            let mut arr = [0f32; C];
            for c in 0..C {
                arr[c] = self.data[c*R+r];
            }
            write!(f, "[{}]", arr.map(|n| n.to_string()).join(","))?;
        }
        write!(f, "]")
    }
}

mod tests {
    #[test]
    fn dense_matrix_multiply() {
        use super::DenseMatrix;

        let m1 = DenseMatrix::from_cols(&[
            [1., 2.],
            [3., 4.],
            [5., 6.],
        ]);

        let m2 = DenseMatrix::from_cols(&[
            [1., 2., 3.],
            [4., 5., 6.],
        ]);

        let expected = DenseMatrix::from_cols(&[
            [22., 28.],
            [49., 64.],
        ]);

        assert_eq!(&m1 * &m2, expected);
    }

    #[test]
    fn dense_matrix_scalar_add_multiply() {
        use super::DenseMatrix;

        let m = DenseMatrix::from_cols(&[
            [1., 2.],
            [3., 4.],
            [5., 6.],
        ]);

        let expected = DenseMatrix::from_cols(&[
            [12., 14.],
            [16., 18.],
            [20., 22.],
        ]);

        assert_eq!(&(5. + &m) * 2., expected);
    }
}
