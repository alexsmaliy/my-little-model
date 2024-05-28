use crate::linalg::Vector;

type V<const N: usize> = Vector<N>;

pub trait LossFunction: Copy {
    fn get_L<const DIM: usize>(&self) -> impl Fn(&V<DIM>, &V<DIM>) -> (f32, V<DIM>) + 'static;
    fn get_dL_da<const DIM: usize>(&self) -> impl Fn(&V<DIM>, &V<DIM>) -> V<DIM> + 'static;
}

#[derive(Clone, Copy)]
pub struct MSELoss;

impl LossFunction for MSELoss {
    fn get_L<const DIM: usize>(&self) -> impl Fn(&V<DIM>, &V<DIM>) -> (f32, V<DIM>) + 'static {
        |target: &V<DIM>, output: &V<DIM>| {
            let errors = target - output;
            (errors.sum_of_squares(), errors)
        }
    }

    fn get_dL_da<const DIM: usize>(&self) -> impl Fn(&V<DIM>, &V<DIM>) -> V<DIM> + 'static {
        |target: &V<DIM>, output: &V<DIM>| -2f32 * &(target - output)
    }
}
