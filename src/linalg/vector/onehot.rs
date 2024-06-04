use std::ops::Index;

#[derive(Clone, Debug, PartialEq)]
pub struct OneHotVector<const D: usize> {
    pub(super) zero: f32,      // Index impls must return a ref to a sentinel value
    pub(super) one: f32,       // as above
    pub(super) index: usize,   // the non-zero index
}

impl<const D: usize> Index<usize> for OneHotVector<D> {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        if index == self.index { &self.one } else { &self.zero }
    }
}
