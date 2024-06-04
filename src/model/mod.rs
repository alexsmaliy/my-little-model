pub mod activation;
pub mod loss;
pub mod manual;
pub mod weights;

use std::marker::PhantomData;

use crate::linalg::VectorWrapper;
use crate::layer::ModelLayerChain;
use loss::LossFunction;

pub struct ModelOutput<const DIM: usize> {
    pub loss: f32,
    pub errors: VectorWrapper<DIM>,
    pub output: VectorWrapper<DIM>,
}

pub struct Model<const IN: usize, const OUT: usize, T, L: ModelLayerChain<IN, OUT, T>, LF: LossFunction> {
    pub layers: L,
    pub last_input: VectorWrapper<IN>,
    pub last_output: VectorWrapper<OUT>,
    pub errors: VectorWrapper<OUT>,
    pub loss: f32,
    pub loss_function: LF,

    _ph: PhantomData<T>, // dummy field denoting hard-to-inscribe type T
}

impl<const IN: usize, const OUT: usize, T, L: ModelLayerChain<IN, OUT, T>, LF: LossFunction> Model<IN, OUT, T, L, LF> {
    pub fn new(layers: L, loss_function: LF) -> Self {
        Model {
            layers,
            last_input: VectorWrapper::zero(),
            last_output: VectorWrapper::zero(),
            errors: VectorWrapper::zero(),
            loss: 0f32,
            loss_function,
            _ph: PhantomData::<T>,
        }
    }

    pub fn run_once(&mut self, input: &VectorWrapper<IN>, target: &VectorWrapper<OUT>) {
        let ModelOutput { loss, errors, output } = self.layers.run_once(
            (input, target),
            self.loss_function,
            0.01,
        );
        self.last_input = input.clone();
        self.last_output = output;
        self.loss = loss;
        self.errors = errors;
    }
}
