use crate::linalg::{Matrix, Vector};
use super::ModelOutput;

pub mod connected;

pub trait ModelLayer<const IN: usize, const OUT: usize>
    where
        [(); IN*OUT]: Sized,
        [(); OUT*IN]: Sized,
        [(); OUT*OUT]: Sized,
{
    fn forward(&mut self, input_src: &Vector<IN>);
    fn backward(&mut self, upstream_Wᵀs: &Vector<OUT>);
    fn update_weights(&mut self, learning_rate: f32, a_prev: &Vector<IN>);
    fn nonlinear_output(&self) -> &Vector<OUT>;
    fn linear_output(&self) -> &Vector<OUT>;
    fn f(&self) -> Box<dyn Fn(f32) -> f32>;
    fn df(&self) -> Box<dyn Fn(f32) -> f32>;
    fn set_sensitivities(&mut self, s: Vector<OUT>);
    fn sensitivities(&self) -> &Vector<IN>;
}

pub trait ModelLayerChain<const IN: usize, const OUT: usize, T> {
    fn run_once(&mut self, input: &Vector<IN>, target: &Vector<OUT>) -> ModelOutput<OUT>;
}

impl<
    const A: usize,
    const B: usize,
    const C: usize,
    const D: usize,
    L0: ModelLayer<A, B>,
    L1: ModelLayer<B, C>,
    L2: ModelLayer<C, D>,
> ModelLayerChain<A, D, (
    Box<dyn ModelLayer<A, B>>,
    Box<dyn ModelLayer<B, C>>,
    Box<dyn ModelLayer<C, D>>,
)> for (
    L0, L1, L2,
) where
    [(); A*B]: Sized,
    [(); B*A]: Sized,
    [(); B*B]: Sized,
    [(); B*C]: Sized,
    [(); C*B]: Sized,
    [(); C*C]: Sized,
    [(); C*D]: Sized,
    [(); D*C]: Sized,
    [(); D*D]: Sized,
{
    fn run_once(&mut self, item: &Vector<A>, target: &Vector<D>) -> ModelOutput<D> {
        let learning_rate: f32 = 0.001; // TODO

        self.0.forward(item);
        self.1.forward(self.0.nonlinear_output());
        self.2.forward(self.1.nonlinear_output());

        let (n_last, df_last) = (self.2.linear_output(), self.2.df());
        let tmp = (-2f32 * &n_last.map(df_last)).into();
        let tmp = Matrix::diag(&tmp);
        let errors = target - self.2.nonlinear_output();
        let s_last = &tmp * &errors;
        let loss = errors.sum_of_squares();

        self.2.set_sensitivities(s_last);
        self.1.backward(self.2.sensitivities());
        self.0.backward(self.1.sensitivities());

        self.0.update_weights(learning_rate, item);
        self.1.update_weights(learning_rate, self.0.nonlinear_output());
        self.2.update_weights(learning_rate, self.1.nonlinear_output());

        return ModelOutput {
            loss,
            errors,
            output: self.2.nonlinear_output().clone()
        }
    }
}
