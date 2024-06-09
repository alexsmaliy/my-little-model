use crate::linalg::{Matrix, Vector};
use crate::model::loss::LossFunction;
use crate::model::ModelOutput;

pub mod connected;

pub trait ModelLayer<const IN: usize, const OUT: usize>
    where
        [(); IN*OUT]: Sized,
        [(); OUT*IN]: Sized,
        [(); OUT*OUT]: Sized,
{
    fn forward(&mut self, input_src: &Vector<IN>);
    fn backward(&mut self, upstream_Wᵀs: &Vector<OUT>);
    fn update_params(&mut self, learning_rate: f32, a_prev: &Vector<IN>);
    fn nonlinear_output(&self) -> &Vector<OUT>;
    fn linear_output(&self) -> &Vector<OUT>;
    fn f(&self) -> Box<dyn Fn(f32) -> f32 + 'static>;
    fn df(&self) -> Box<dyn Fn(f32) -> f32 + 'static>;
    fn get_sensitivities(&self) -> &Vector<IN>;
    fn set_sensitivities(&mut self, s: Vector<OUT>);
}

pub trait ModelLayerChain<const IN: usize, const OUT: usize, T> {
    fn train_single<L: LossFunction>(
        &mut self,
        input_pair: (&Vector<IN>, &Vector<OUT>),
        loss_function: L,
        learning_rate: f32,
    ) -> ModelOutput<OUT>;
}

// This is prime target for a macro.
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
    fn train_single<LF: LossFunction>(
        &mut self,
        input_pair: (&Vector<A>, &Vector<D>),
        loss_function: LF,
        learning_rate: f32,
    ) -> ModelOutput<D> {
        let (item, target) = input_pair;

        self.0.forward(item);
        self.1.forward(self.0.nonlinear_output());
        self.2.forward(self.1.nonlinear_output());

        let last_layer = &self.2;
        let model_output = last_layer.nonlinear_output().clone();

        let (L, dL_da) = (loss_function.get_L(), loss_function.get_dL_da());
        let (loss, errors) = L(target, &model_output);
        // let errors = target - model_output;
        // let loss = errors.sum_of_squares();

        let (n_last, df_last) = (self.2.linear_output(), self.2.df());

        // This doesn't depend on choice of L.
        let da_dn = n_last.map(df_last).into();
        let da_dn = Matrix::diag(da_dn);
        // This depends on choice of L.
        let dL_da = dL_da(target, &model_output);
        // let dL_da = -2f32 * &errors;

        let s_last = &da_dn * &dL_da;

        self.2.set_sensitivities(s_last);
        self.1.backward(self.2.get_sensitivities());
        self.0.backward(self.1.get_sensitivities());

        self.0.update_params(learning_rate, item);
        self.1.update_params(learning_rate, self.0.nonlinear_output());
        self.2.update_params(learning_rate, self.1.nonlinear_output());

        return ModelOutput {
            loss,
            errors,
            output: model_output,
        }
    }
}
