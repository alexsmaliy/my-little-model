use std::ops::Index;

#[derive(Clone, Debug, PartialEq)]
pub struct ConstantVector<const D: usize>(
    pub(super) f32,
);

impl<const D: usize> Index<usize> for ConstantVector<D> {
    type Output = f32;

    fn index(&self, _index: usize) -> &Self::Output {
        &self.0
    }
}
