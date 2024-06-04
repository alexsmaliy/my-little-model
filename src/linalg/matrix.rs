use std::fmt::Display;
use std::ops::{Add, Mul, Neg, Sub};

use crate::linalg::matrix::inner::{ConstantMatrix, DenseMatrix, SparseMatrix};

use super::Vector;

//////////////////
#[derive(Copy, Clone, Debug)]
enum Order {
    COLUMNS, ROWS
}

impl Neg for Order {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            Order::COLUMNS => Order::ROWS,
            Order::ROWS => Order::COLUMNS,
        }
    }
}

#[derive(Clone, Debug)]
pub enum MatrixWrapper<const R: usize, const C: usize> where [(); R*C]: Sized {
    Constant(inner::ConstantMatrix<R, C>),
    Dense(inner::DenseMatrix<R, C>),
    Diagonal(inner::DiagonalMatrix<R, C>),
    Identity(inner::IdentityMatrix<R, C>),
    Sparse(inner::SparseMatrix<R, C>),
    Zero(inner::ZeroMatrix<R, C>),
}

pub enum VectorWrapper<const D: usize> {
    Constant(inner::ConstantVector<D>),
    Dense(inner::DenseVector<D>),
    OneHot(inner::OneHotVector<D>),
    Sparse(inner::SparseVector<D>),
    Zero(inner::ZeroVector<D>),
}

impl<const R: usize, const C: usize> MatrixWrapper<R, C> where [(); R*C]: Sized {
    // constructor
    pub fn constant(c: f32) -> Self {
        Self::Constant(inner::ConstantMatrix(c))
    }

    // constructor
    pub fn sparse() -> Self {
        Self::Sparse(inner::SparseMatrix(Vec::new(), Vec::new(), Order::COLUMNS))
    }

    // constructor
    pub fn zero() -> Self {
        Self::Zero(inner::ZeroMatrix(0f32))
    }

    /// Matrix transpose.
    pub fn T(&self) -> MatrixWrapper<C, R>
        where
            [(); R*C]: Sized,
            [(); C*R]: Sized,
    {
        use MatrixWrapper as M;
        match self {
            M::Constant(m) => M::Constant::<C, R>(m.T()),
            M::Dense(m) => M::Dense(m.T()),
            M::Diagonal(m) => M::Diagonal(m.T()),
            M::Identity(m) => M::Identity(m.T()),
            M::Sparse(m) => M::Sparse(m.T()),
            M::Zero(m) => M::Zero(m.T()),
        }
    }
}

impl<const D: usize> MatrixWrapper<D, D> where [(); D*D]: Sized {
    pub fn diagonal(v: inner::DenseVector<D>) -> Self {
        Self::Diagonal(inner::DiagonalMatrix(v))
    }
    
    pub fn identity() -> Self {
        Self::Identity(inner::IdentityMatrix(0f32, 1f32))
    }
}

impl<const D: usize> VectorWrapper<D> {
    pub fn onehot(i: usize) -> Self {
        Self::OneHot(inner::OneHotVector { zero: 0f32, one: 1f32, index: i })
    }

    pub fn sparse() -> Self {
        Self::Sparse(inner::SparseVector(Vec::new(), Vec::new()))
    }

    pub fn zero() -> Self {
        Self::Zero(inner::ZeroVector(0f32))
    }
}

impl<const R: usize, const C: usize, const C2: usize> Mul<&MatrixWrapper<C, C2>> for &MatrixWrapper<R, C>
    where
        [(); R*C]: Sized,
        [(); C*C2]: Sized,
        [(); R*C2]: Sized,
{
    type Output = MatrixWrapper<R, C2>;
    
    // Some matrix flavors can only be instantiated as square, but we must provide
    // rectangular impls for all of them to satisfy impl coherence.
    fn mul(self, rhs: &MatrixWrapper<C, C2>) -> Self::Output {
        use MatrixWrapper as M;
        match (self, rhs) {
            (M::Constant(m1), M::Constant(m2)) => M::Constant(m1 * m2),
            (M::Constant(m1), M::Dense(m2)) => M::Dense(m1 * m2),
            (M::Constant(m1), M::Diagonal(m2)) => M::Constant(m1 * m2),
            (M::Constant(m1), M::Identity(m2)) => M::Constant(m1 * m2),
            (M::Constant(m1), M::Sparse(m2)) => M::Sparse(m1 * m2),
            (M::Constant(m1), M::Zero(m2)) => M::Zero(m1 * m2),

            (M::Dense(m1), M::Constant(m2)) => M::Dense(m1 * m2),
            (M::Dense(m1), M::Dense(m2)) => M::Dense(m1 * m2),
            (M::Dense(m1), M::Diagonal(m2)) => M::Dense(m1 * m2),
            (M::Dense(m1), M::Identity(m2)) => M::Dense(m1 * m2),
            (M::Dense(m1), M::Sparse(m2)) => M::Dense(m1 * m2),
            (M::Dense(m1), M::Zero(m2)) => M::Zero(m1 * m2),

            (M::Diagonal(m1), M::Constant(m2)) => M::Constant(m1 * m2),
            (M::Diagonal(m1), M::Dense(m2)) => M::Dense(m1 * m2),
            (M::Diagonal(m1), M::Diagonal(m2)) => M::Diagonal(m1 * m2),
            (M::Diagonal(m1), M::Identity(m2)) => M::Diagonal(m1 * m2),
            (M::Diagonal(m1), M::Sparse(m2)) => M::Sparse(m1 * m2),
            (M::Diagonal(m1), M::Zero(m2)) => M::Zero(m1 * m2),

            (M::Identity(m1), M::Constant(m2)) => M::Constant(m1 * m2),
            (M::Identity(m1), M::Dense(m2)) => M::Dense(m1 * m2),
            (M::Identity(m1), M::Diagonal(m2)) => M::Diagonal(m1 * m2),
            (M::Identity(m1), M::Identity(m2)) => M::Identity(m1 * m2),
            (M::Identity(m1), M::Sparse(m2)) => M::Sparse(m1 * m2),
            (M::Identity(m1), M::Zero(m2)) => M::Zero(m1 * m2),

            (M::Sparse(m1), M::Constant(m2)) => M::Sparse(m1 * m2),
            (M::Sparse(m1), M::Dense(m2)) => M::Dense(m1 * m2),
            (M::Sparse(m1), M::Diagonal(m2)) => M::Sparse(m1 * m2),
            (M::Sparse(m1), M::Identity(m2)) => M::Sparse(m1 * m2),
            (M::Sparse(m1), M::Sparse(m2)) => M::Sparse(m1 * m2),
            (M::Sparse(m1), M::Zero(m2)) => M::Zero(m1 * m2),

            (M::Zero(m1), M::Constant(m2)) => M::Zero(m1 * m2),
            (M::Zero(m1), M::Dense(m2)) => M::Zero(m1 * m2),
            (M::Zero(m1), M::Diagonal(m2)) => M::Zero(m1 * m2),
            (M::Zero(m1), M::Identity(m2)) => M::Zero(m1 * m2),
            (M::Zero(m1), M::Sparse(m2)) => M::Zero(m1 * m2),
            (M::Zero(m1), M::Zero(m2)) => M::Zero(m1 * m2),
        }
    }
}

impl<const R: usize, const C: usize> PartialEq for MatrixWrapper<R, C> where [(); R*C]: Sized {
    fn eq(&self, other: &Self) -> bool {
        use MatrixWrapper as M;
        match (self, other) {
            (M::Constant(m1), M::Constant(m2)) => m1 == m2,
            (M::Dense(m1), M::Dense(m2)) => m1 == m2,
            (M::Diagonal(m1), M::Diagonal(m2)) => m1 == m2,
            (M::Identity(m1), M::Identity(m2)) => m1 == m2,
            (M::Sparse(m1), M::Sparse(m2)) => m1 == m2,
            (M::Zero(m1), M::Zero(m2)) => m1 == m2,
            _ => false, // TODO: equality between mixed flavors.
        }
    }
}

impl<const R: usize, const C: usize> Mul<&VectorWrapper<C>> for &MatrixWrapper<R, C> where [(); R*C]: Sized {
    type Output = VectorWrapper<R>;
    
    fn mul(self, rhs: &VectorWrapper<C>) -> Self::Output {
        match (self, rhs) {
            //TODO: matrix-vector mul impls.
            _ => unimplemented!(),
        }
    }
}

mod inner {
    use super::*;

    mod test {
        #[test]
        fn dense_matrix_wrapper_multiply() {
            use super::*;
            let m1 = MatrixWrapper::Dense(DenseMatrix::<2, 3>([1., 2., 3., 4., 5., 6.], super::Order::COLUMNS));
            // let m1 = Matrix::from_cols(&[
            //     [1., 2.],
            //     [3., 4.],
            //     [5., 6.],
            // ]);
            let m2 = MatrixWrapper::Dense(DenseMatrix::<3, 2>([1., 2., 3., 4., 5., 6.], super::Order::COLUMNS));
            let expected = MatrixWrapper::Dense(DenseMatrix::<2, 2>([22., 28., 49., 64.], super::Order::COLUMNS));
            assert_eq!(&m1 * &m2, expected);
        }
    }

    ///////////////////////
    /// CONSTANT MATRIX ///
    ///////////////////////

    #[derive(Clone, Debug)]
    pub struct ConstantMatrix<const R: usize, const C: usize>(
        pub(super) f32,
    );

    impl<const R: usize, const C: usize> ConstantMatrix<R, C> {
        pub(super) fn T(&self) -> ConstantMatrix<C, R> {
            ConstantMatrix(self.0)
        }
    }

    impl<const R: usize, const C: usize> PartialEq for ConstantMatrix<R, C> where [(); R*C]: Sized {
        fn eq(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }

    ////////////////////
    /// DENSE MATRIX ///
    ////////////////////

    #[derive(Clone, Debug)]
    pub struct DenseMatrix<const R: usize, const C: usize>(
        pub(super) [f32; R*C],
        pub(super) Order,
    ) where [(); R*C]: Sized;

    impl<const R: usize, const C: usize> DenseMatrix<R, C>
        where
            [(); C*R]: Sized,
            [(); R*C]: Sized,
    {
        pub(super) fn T(&self) -> DenseMatrix<C, R> {
            let arr: [f32; C*R] = self.0.to_vec().try_into().unwrap();
            DenseMatrix(arr, -self.1)
        }
    }

    impl<const R: usize, const C: usize> PartialEq for DenseMatrix<R, C> where [(); R*C]: Sized {
        fn eq(&self, _other: &Self) -> bool {
            todo!() //TODO: handle transposed args.
        }
    }

    ///////////////////////
    /// DIAGONAL MATRIX ///
    ///////////////////////

    #[derive(Clone, Debug)]
    pub struct DiagonalMatrix<const R: usize, const C: usize>(
        pub(super) DenseVector<R>,
    ) where [(); R*C]: Sized;

    // Impl is provided for possibly unequal R and C,
    // even though only square diagonal matrices can be instantiated.
    // This impl clones the underlying array and, if needed,
    // extends it with 0f32 or truncates it to the needed size.
    // If R == C, as it ought to be, this is just a clone.
    // Wishing Rust had impl specialization...
    impl<const R: usize, const C: usize> DiagonalMatrix<R, C>
        where
            [(); C*R]: Sized,
            [(); R*C]: Sized,
    {
        pub(super) fn T(&self) -> DiagonalMatrix<C, R> {
            let arr: [f32; C] = self.0.0.iter()
                .copied()
                .chain(std::iter::repeat(0f32))
                .take(C)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            DiagonalMatrix(DenseVector(arr))
        }
    }

    impl<const R: usize, const C: usize> PartialEq for DiagonalMatrix<R, C> where [(); R*C]: Sized {
        fn eq(&self, other: &Self) -> bool {
            self.0.0 == other.0.0 // TODO: PartialEq for DenseVector
        }
    }

    ///////////////////////
    /// IDENTITY MATRIX ///
    ///////////////////////

    #[derive(Clone, Debug)]
    pub struct IdentityMatrix<const R: usize, const C: usize>(
        pub(super) f32,
        pub(super) f32,
    );

    // Impl is provided for possibly unequal R and C,
    // even though only square diagonal matrices can be instantiated.
    impl<const R: usize, const C: usize> IdentityMatrix<R, C> {
        pub(super) fn T(&self) -> IdentityMatrix<C, R> {
            IdentityMatrix(self.0, self.1)
        }
    }

    impl<const R: usize, const C: usize> PartialEq for IdentityMatrix<R, C> where [(); R*C]: Sized {
        fn eq(&self, _: &Self) -> bool {
            true
        }
    }

    /////////////////////
    /// SPARSE MATRIX ///
    /////////////////////

    #[derive(Clone, Debug)]
    pub struct SparseMatrix<const R: usize, const C: usize>(
        pub(super) Vec<usize>,
        pub(super) Vec<f32>,
        pub(super) Order,
    );

    impl<const R: usize, const C: usize> SparseMatrix<R, C> {
        pub(super) fn T(&self) -> SparseMatrix<C, R> {
            todo!() //TODO
        }
    }

    impl<const R: usize, const C: usize> PartialEq for SparseMatrix<R, C> where [(); R*C]: Sized {
        fn eq(&self, _other: &Self) -> bool {
            todo!() // TODO: handle transposed args.
        }
    }

    ///////////////////
    /// ZERO MATRIX ///
    ///////////////////

    #[derive(Clone, Debug)]
    pub struct ZeroMatrix<const R: usize, const C: usize>(
        pub(super) f32,
    );

    impl<const R: usize, const C: usize> ZeroMatrix<R, C> {
        pub(super) fn T(&self) -> ZeroMatrix<C, R> {
            ZeroMatrix(self.0)
        }
    }

    impl<const R: usize, const C: usize> PartialEq for ZeroMatrix<R, C> where [(); R*C]: Sized {
        fn eq(&self, _other: &Self) -> bool {
            true
        }
    }

    ///// vector impls

    #[derive(Clone, Debug)]
    pub struct ConstantVector<const D: usize>(
        pub(super) f32,
    );

    #[derive(Clone, Debug)]
    pub struct DenseVector<const D: usize>(
        pub(super) [f32; D],
    );

    #[derive(Clone, Debug)]
    pub struct OneHotVector<const D: usize> {
        pub(super) zero: f32,      // Index impls must return a ref to a sentinel value
        pub(super) one: f32,       // as above
        pub(super) index: usize,   // the non-zero index
    }

    #[derive(Clone, Debug)]
    pub struct SparseVector<const D: usize>(
        pub(super) Vec<usize>,
        pub(super) Vec<f32>,
    );

    #[derive(Clone, Debug)]
    pub struct ZeroVector<const D: usize>(
        pub(super) f32,
    );

    /////////////////////////////////
    /// CONSTANT MATRIX MUL IMPLS ///
    /////////////////////////////////

    impl<const R: usize, const C: usize, const C2: usize> Mul<&ConstantMatrix<C, C2>> for &ConstantMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
    {
        type Output = ConstantMatrix<R, C2>;

        fn mul(self, rhs: &ConstantMatrix<C, C2>) -> Self::Output {
            todo!()
        }
    }

    impl<const R: usize, const C: usize, const C2: usize> Mul<&DenseMatrix<C, C2>> for &ConstantMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
            [(); R*C2]: Sized,
    {
        type Output = DenseMatrix<R, C2>;

        fn mul(self, rhs: &DenseMatrix<C, C2>) -> Self::Output {
            todo!()
        }
    }

    impl<const R: usize, const C: usize, const C2: usize> Mul<&DiagonalMatrix<C, C2>> for &ConstantMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
            [(); R*C]: Sized,
    {
        type Output = ConstantMatrix<R, C2>;

        fn mul(self, rhs: &DiagonalMatrix<C, C2>) -> Self::Output {
            todo!()
        }
    }

    impl<const R: usize, const C: usize, const C2: usize> Mul<&IdentityMatrix<C, C2>> for &ConstantMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
            [(); R*C2]: Sized,
    {
        type Output = ConstantMatrix<R, C2>;

        fn mul(self, rhs: &IdentityMatrix<C, C2>) -> Self::Output {
            todo!()
        }
    }

    impl<const R: usize, const C: usize, const C2: usize> Mul<&SparseMatrix<C, C2>> for &ConstantMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
    {
        type Output = SparseMatrix<R, C2>;

        fn mul(self, rhs: &SparseMatrix<C, C2>) -> Self::Output {
            todo!()
        }
    }

    impl<const R: usize, const C: usize, const C2: usize> Mul<&ZeroMatrix<C, C2>> for &ConstantMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
    {
        type Output = ZeroMatrix<R, C2>;

        fn mul(self, rhs: &ZeroMatrix<C, C2>) -> Self::Output {
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

        fn mul(self, rhs: &ConstantMatrix<C, C2>) -> Self::Output {
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
            let mut arr = [0f32; R*C2];
            for i in 0..(R * C) {
                let from = (i / R) * C;
                let to = from + C;
                let x: f32 = rhs.0[from..to]
                    .into_iter()
                    .enumerate()
                    .map(|(j, x)| x * self.0[j * R + i % R])
                    .sum();
                arr[i] = x;
            }
            DenseMatrix(arr, self.1)
        }
    }

    impl<const R: usize, const C: usize, const C2: usize> Mul<&DiagonalMatrix<C, C2>> for &DenseMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
            [(); R*C2]: Sized,
    {
        type Output = DenseMatrix<R, C2>;

        fn mul(self, rhs: &DiagonalMatrix<C, C2>) -> Self::Output {
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

        fn mul(self, rhs: &IdentityMatrix<C, C2>) -> Self::Output {
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

        fn mul(self, rhs: &SparseMatrix<C, C2>) -> Self::Output {
            todo!()
        }
    }

    impl<const R: usize, const C: usize, const C2: usize> Mul<&ZeroMatrix<C, C2>> for &DenseMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
    {
        type Output = ZeroMatrix<R, C2>;

        fn mul(self, rhs: &ZeroMatrix<C, C2>) -> Self::Output {
            todo!()
        }
    }

    /////////////////////////////////
    /// DIAGONAL MATRIX MUL IMPLS ///
    /////////////////////////////////

    impl<const R: usize, const C: usize, const C2: usize> Mul<&ConstantMatrix<C, C2>> for &DiagonalMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
            [(); R*C2]: Sized,
    {
        type Output = ConstantMatrix<R, C2>;

        fn mul(self, rhs: &ConstantMatrix<C, C2>) -> Self::Output {
            todo!()
        }
    }

    impl<const R: usize, const C: usize, const C2: usize> Mul<&DenseMatrix<C, C2>> for &DiagonalMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
            [(); R*C2]: Sized,
    {
        type Output = DenseMatrix<R, C2>;

        fn mul(self, rhs: &DenseMatrix<C, C2>) -> Self::Output {
            todo!()
        }
    }

    impl<const R: usize, const C: usize, const C2: usize> Mul<&DiagonalMatrix<C, C2>> for &DiagonalMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
            [(); R*C2]: Sized,
    {
        type Output = DiagonalMatrix<R, C2>;

        fn mul(self, rhs: &DiagonalMatrix<C, C2>) -> Self::Output {
            todo!()
        }
    }

    impl<const R: usize, const C: usize, const C2: usize> Mul<&IdentityMatrix<C, C2>> for &DiagonalMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
            [(); R*C2]: Sized,
    {
        type Output = DiagonalMatrix<R, C2>;

        fn mul(self, rhs: &IdentityMatrix<C, C2>) -> Self::Output {
            todo!()
        }
    }

    impl<const R: usize, const C: usize, const C2: usize> Mul<&SparseMatrix<C, C2>> for &DiagonalMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
            [(); R*C2]: Sized,
    {
        type Output = SparseMatrix<R, C2>;

        fn mul(self, rhs: &SparseMatrix<C, C2>) -> Self::Output {
            todo!()
        }
    }

    impl<const R: usize, const C: usize, const C2: usize> Mul<&ZeroMatrix<C, C2>> for &DiagonalMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
            [(); R*C2]: Sized,
    {
        type Output = ZeroMatrix<R, C2>;

        fn mul(self, rhs: &ZeroMatrix<C, C2>) -> Self::Output {
            todo!()
        }
    }

    /////////////////////////////////
    /// IDENTITY MATRIX MUL IMPLS ///
    /////////////////////////////////

    impl<const R: usize, const C: usize, const C2: usize> Mul<&ConstantMatrix<C, C2>> for &IdentityMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
            [(); R*C2]: Sized,
    {
        type Output = ConstantMatrix<R, C2>;

        fn mul(self, rhs: &ConstantMatrix<C, C2>) -> Self::Output {
            todo!()
        }
    }

    impl<const R: usize, const C: usize, const C2: usize> Mul<&DenseMatrix<C, C2>> for &IdentityMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
            [(); R*C2]: Sized,
    {
        type Output = DenseMatrix<R, C2>;

        fn mul(self, rhs: &DenseMatrix<C, C2>) -> Self::Output {
            todo!()
        }
    }

    impl<const R: usize, const C: usize, const C2: usize> Mul<&DiagonalMatrix<C, C2>> for &IdentityMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
            [(); R*C2]: Sized,
    {
        type Output = DiagonalMatrix<R, C2>;

        fn mul(self, rhs: &DiagonalMatrix<C, C2>) -> Self::Output {
            todo!()
        }
    }

    impl<const R: usize, const C: usize, const C2: usize> Mul<&IdentityMatrix<C, C2>> for &IdentityMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
            [(); R*C2]: Sized,
    {
        type Output = IdentityMatrix<R, C2>;

        fn mul(self, rhs: &IdentityMatrix<C, C2>) -> Self::Output {
            todo!()
        }
    }

    impl<const R: usize, const C: usize, const C2: usize> Mul<&SparseMatrix<C, C2>> for &IdentityMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
            [(); R*C2]: Sized,
    {
        type Output = SparseMatrix<R, C2>;

        fn mul(self, rhs: &SparseMatrix<C, C2>) -> Self::Output {
            todo!()
        }
    }

    impl<const R: usize, const C: usize, const C2: usize> Mul<&ZeroMatrix<C, C2>> for &IdentityMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
            [(); R*C2]: Sized,
    {
        type Output = ZeroMatrix<R, C2>;

        fn mul(self, rhs: &ZeroMatrix<C, C2>) -> Self::Output {
            todo!()
        }
    }

    ///////////////////////////////
    /// SPARSE MATRIX MUL IMPLS ///
    ///////////////////////////////

    impl<const R: usize, const C: usize, const C2: usize> Mul<&ConstantMatrix<C, C2>> for &SparseMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
            [(); R*C2]: Sized,
    {
        type Output = SparseMatrix<R, C2>;

        fn mul(self, rhs: &ConstantMatrix<C, C2>) -> Self::Output {
            todo!()
        }
    }

    impl<const R: usize, const C: usize, const C2: usize> Mul<&DenseMatrix<C, C2>> for &SparseMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
            [(); R*C2]: Sized,
    {
        type Output = DenseMatrix<R, C2>;

        fn mul(self, rhs: &DenseMatrix<C, C2>) -> Self::Output {
            todo!()
        }
    }

    impl<const R: usize, const C: usize, const C2: usize> Mul<&DiagonalMatrix<C, C2>> for &SparseMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
            [(); R*C2]: Sized,
    {
        type Output = SparseMatrix<R, C2>;

        fn mul(self, rhs: &DiagonalMatrix<C, C2>) -> Self::Output {
            todo!()
        }
    }

    impl<const R: usize, const C: usize, const C2: usize> Mul<&IdentityMatrix<C, C2>> for &SparseMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
            [(); R*C2]: Sized,
    {
        type Output = SparseMatrix<R, C2>;

        fn mul(self, rhs: &IdentityMatrix<C, C2>) -> Self::Output {
            todo!()
        }
    }

    impl<const R: usize, const C: usize, const C2: usize> Mul<&SparseMatrix<C, C2>> for &SparseMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
            [(); R*C2]: Sized,
    {
        type Output = SparseMatrix<R, C2>;

        fn mul(self, rhs: &SparseMatrix<C, C2>) -> Self::Output {
            todo!()
        }
    }

    impl<const R: usize, const C: usize, const C2: usize> Mul<&ZeroMatrix<C, C2>> for &SparseMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
    {
        type Output = ZeroMatrix<R, C2>;

        fn mul(self, rhs: &ZeroMatrix<C, C2>) -> Self::Output {
            todo!()
        }
    }

    /////////////////////////////
    /// ZERO MATRIX MUL IMPLS ///
    /////////////////////////////

    impl<const R: usize, const C: usize, const C2: usize> Mul<&ConstantMatrix<C, C2>> for &ZeroMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
            [(); R*C2]: Sized,
    {
        type Output = ZeroMatrix<R, C2>;

        fn mul(self, rhs: &ConstantMatrix<C, C2>) -> Self::Output {
            todo!()
        }
    }

    impl<const R: usize, const C: usize, const C2: usize> Mul<&DenseMatrix<C, C2>> for &ZeroMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
            [(); R*C2]: Sized,
    {
        type Output = ZeroMatrix<R, C2>;

        fn mul(self, rhs: &DenseMatrix<C, C2>) -> Self::Output {
            todo!()
        }
    }

    impl<const R: usize, const C: usize, const C2: usize> Mul<&DiagonalMatrix<C, C2>> for &ZeroMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
            [(); R*C2]: Sized,
    {
        type Output = ZeroMatrix<R, C2>;

        fn mul(self, rhs: &DiagonalMatrix<C, C2>) -> Self::Output {
            todo!()
        }
    }

    impl<const R: usize, const C: usize, const C2: usize> Mul<&IdentityMatrix<C, C2>> for &ZeroMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
            [(); R*C2]: Sized,
    {
        type Output = ZeroMatrix<R, C2>;

        fn mul(self, rhs: &IdentityMatrix<C, C2>) -> Self::Output {
            todo!()
        }
    }

    impl<const R: usize, const C: usize, const C2: usize> Mul<&SparseMatrix<C, C2>> for &ZeroMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
            [(); R*C2]: Sized,
    {
        type Output = ZeroMatrix<R, C2>;

        fn mul(self, rhs: &SparseMatrix<C, C2>) -> Self::Output {
            todo!()
        }
    }

    impl<const R: usize, const C: usize, const C2: usize> Mul<&ZeroMatrix<C, C2>> for &ZeroMatrix<R, C>
        where
            [(); R*C]: Sized,
            [(); C*C2]: Sized,
    {
        type Output = ZeroMatrix<R, C2>;

        fn mul(self, rhs: &ZeroMatrix<C, C2>) -> Self::Output {
            todo!()
        }
    }
    
    ////////////////////////////////////
    /// MATRIX WRAPPER DISPLAY IMPLS ///
    ////////////////////////////////////

    impl<const R: usize, const C: usize> Display for ConstantMatrix<R, C> where [(); R*C]: Sized {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "[{}]", (0..R).into_iter()
                .map(|_| std::iter::repeat(&self.0)
                    .map(<f32>::to_string)
                    .take(C)
                    .collect::<Vec<_>>()
                    .join(","))
                .collect::<Vec<_>>()
                .join("],["))
        }
    }

    impl<const R: usize, const C: usize> Display for DenseMatrix<R, C> where [(); R*C]: Sized {
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

    impl<const D: usize> Display for DiagonalMatrix<D, D> where [(); D*D]: Sized {
        /// Displays the rows of a diagonal matrix.
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "[{}]", (0..D).into_iter()
                .map(|r| {
                    let before = (0..r).into_iter().map(|_| 0f32.to_string());
                    let val = std::iter::once(self.0.0[r].to_string()); // TODO: index dense vector
                    let after = ((r+1)..D).into_iter().map(|_| 0f32.to_string());
                    before.chain(val).chain(after).collect::<Vec<_>>().join(",")
                }).collect::<Vec<_>>().join("],["))
        }
    }

    impl<const D: usize> Display for IdentityMatrix<D, D> where [(); D*D]: Sized {
        /// Displays the rows of an identity matrix.
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "[{}]", (0..D).into_iter()
                .map(|r| {
                    let before = (0..r).into_iter().map(|_| 0f32.to_string());
                    let val = std::iter::once(self.0.to_string());
                    let after = ((r+1)..D).into_iter().map(|_| 0f32.to_string());
                    before.chain(val).chain(after).collect::<Vec<_>>().join(",")
                }).collect::<Vec<_>>().join("],["))
        }
    }

    impl<const R: usize, const C: usize> Display for ZeroMatrix<R, C> where [(); R*C]: Sized {
        /// Displays the rows of a zero matrix.
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "[{}]", (0..R).into_iter()
                .map(|_| std::iter::repeat("0")
                    .take(C)
                    .collect::<Vec<_>>()
                    .join(","))
                .collect::<Vec<_>>()
                .join("],["))
        }
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
        for i in 0..(R * C) {
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
