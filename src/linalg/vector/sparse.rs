use std::{collections::HashMap, ops::Index};

use ahash::RandomState;

#[derive(Clone, Debug, PartialEq)]
pub struct SparseVector<const D: usize> {
    pub(super) elems: HashMap<usize, f32, RandomState>,
    pub(super) zero: f32,
}

impl<const D: usize> Index<usize> for SparseVector<D> {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        self.elems.get(&index).unwrap_or(&self.zero)
    }
}
