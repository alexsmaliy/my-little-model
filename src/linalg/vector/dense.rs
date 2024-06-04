use std::ops::Index;

#[derive(Clone, Debug, PartialEq)]
pub struct DenseVector<const D: usize>(
    pub(super) [f32; D],
);

impl<const D: usize> DenseVector<D> {
    pub(super) fn from_arr(arr: [f32; D]) -> Self {
        Self(arr)
    }
}

impl<const D: usize> Index<usize> for DenseVector<D> {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
