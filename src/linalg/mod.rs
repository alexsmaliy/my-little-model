pub use matrix::OldMatrixDoNotUse; // re-export
pub use vector::OldVectorDoNotUse; // re-export

pub(super) mod matrix;
mod order;
pub(super) mod vector;

pub use matrix::Matrix;
pub use vector::Vector;

//////////////////////////////////////////
/// TESTS OF LINEAR ALGEBRA OPERATIONS ///
//////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_twice() {
        let m = OldMatrixDoNotUse::from_cols(&[
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
        let m = OldMatrixDoNotUse::<3, 3>::I();
        let m2 = 3f32 + &(2f32 * &(&(&m + 1f32) * 5f32));
        let expected = OldMatrixDoNotUse::from_cols(&[
            [23., 13., 13.],
            [13., 23., 13.],
            [13., 13., 23.],
        ]);
        assert_eq!(m2, expected);
    }

    #[test]
    fn matrix_multiplication() {
        let m1 = OldMatrixDoNotUse::from_cols(&[
            [1., 2.],
            [3., 4.],
            [5., 6.],
        ]);
        let m2 = OldMatrixDoNotUse::from_cols(&[
            [1., 2., 3.],
            [4., 5., 6.],
        ]);
        let expected = OldMatrixDoNotUse::from_cols(&[
            [22., 28.],
            [49., 64.],
        ]);
        assert_eq!(&m1 * &m2, expected);
    }

    #[test]
    fn matrix_vector_multiplication() {
        let m = OldMatrixDoNotUse::from_cols(&[
            [1., 2.],
            [3., 4.],
            [5., 6.],
        ]);
        let v = OldVectorDoNotUse::from_arr([1., 0., 2.]);
        let expected = OldVectorDoNotUse::from_arr([11., 14.]);
        assert_eq!(&m * &v, expected);
    }

    #[test]
    fn vector_indexing() {
        let v = &OldVectorDoNotUse::<5>::zero() + 5f32;
        for i in 0..5 {
            assert_eq!(v[i], 5.0);
        }
    }

    #[test]
    fn vector_mut_indexing() {
        let mut v = OldVectorDoNotUse::<5>::zero();
        for i in 0..5 {
            v[i] = 5. - i as f32;
        }
        for i in 0..5 {
            assert_eq!(v[i], 5. - i as f32);
        }
    }

    #[test]
    fn vector_times_transpose() {
        let v = OldVectorDoNotUse::from_arr([1., 2., 3., 4., 5.]);
        let u = OldVectorDoNotUse::from_arr([1., 0., 2.]);

        let expected = OldMatrixDoNotUse::from_cols(&[
            [1., 2., 3., 4., 5.],
            [0., 0., 0., 0., 0.],
            [2., 4., 6., 8., 10.],
        ]);

        assert_eq!(v.outer(&u), expected);
    }
}
