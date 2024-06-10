pub(crate) trait CanAppend {
    type Output;
    fn append(&self, extra_val: f32) -> Self::Output; // TODO: fix self
}

pub(crate) trait CanDotProduct<V> {
    fn dot(&self, other: V) -> f32; // TODO: fix self
}

pub(crate) trait CanMap {
    type Output;
    fn map(&self, f: impl Fn(f32) -> f32) -> Self::Output; // TODO: fix self
    // TODO: map_self, also via <<= ?
}

pub(crate) trait CanOuterProduct<V> {
    type Output;
    fn outer(&self, other: V) -> Self::Output; // TODO: fix self
}

pub(crate) trait CanStackHorizontally<T> {
    type Output;
    fn hstack(&mut self, other: T) -> Self::Output;
    // TODO: hstack mut
}

pub(crate) trait CanStackVertically<T> {
    type Output;
    fn vstack(&mut self, other: T) -> Self::Output;
    // TODO: vstack mut
}
