use crate::linalg::{Matrix, Vector};
use crate::model::transfer_function::TransferFunction;
use crate::model::weights::{Biases, Weights};
use super::ModelLayer;

#[allow(dead_code)]
pub struct FullyConnectedLayer<const IN: usize, const OUT: usize, F: TransferFunction>
    where [(); OUT*IN]: Sized
{
    pub W: Matrix<OUT, IN>, // weights
    pub b: Vector<OUT>,     // biases
    pub n: Vector<OUT>,     // net linear outputs
    pub a: Vector<OUT>,     // net nonlinear outputs
    pub s: Vector<OUT>,     // dL/dn of this layer
    pub Wᵀs: Vector<IN>,    // weighted dL/dn for backwards pass
    // pub f: fn(f32) -> f32,
    // pub df: fn(f32) -> f32,
    pub f: F,
}

impl<const IN: usize, const OUT: usize, F: TransferFunction> FullyConnectedLayer<IN, OUT, F>
    where
        [(); IN*OUT]: Sized,
        [(); OUT*IN]: Sized,
        [(); OUT*OUT]: Sized,
{
    pub fn new(weights: Weights<IN, OUT>, biases: Biases<OUT>, f: F) -> Self {
        FullyConnectedLayer {
            W: weights.into(),
            b: biases.into(),
            n: Vector::zero(),
            a: Vector::zero(),
            s: Vector::zero(),
            Wᵀs: Vector::zero(),
            f,
        }
    }
}

impl<const IN: usize, const OUT: usize, F: TransferFunction> ModelLayer<IN, OUT> for FullyConnectedLayer<IN, OUT, F>
    where
        [(); IN*OUT]: Sized,
        [(); OUT*IN]: Sized,
        [(); OUT*OUT]: Sized,
{
    fn forward(&mut self, prev_output: &Vector<IN>) {
        self.n = &(&self.W * prev_output) + &self.b;
        self.a = self.n.map(self.f.get_f());
    }

    fn backward(&mut self, upstream_Wᵀs: &Vector<OUT>) {
        /*
        // needs: n_i and df_i, but WT_i+1 and s_i+1
        // thus: get upstream Wᵀs to compute and set own Wᵀs

        let x = self.n1.map(self.df1).into();
        let y = Matrix::diag(&x);
        self.s1 = &y * &(&self.w2.T() * &self.s2);
         */
        let x = self.n.map(self.f.get_df()).into();
        let y = Matrix::diag(&x);
        self.s = &y * upstream_Wᵀs;
        self.Wᵀs = &self.W.T() * &self.s;
    }

    fn update_weights(&mut self, learning_rate: f32, a_prev: &Vector<IN>) {
        self.W = &self.W - &(learning_rate * &(self.s.outer(a_prev)));
        self.b = &self.b - &(learning_rate * &self.s);
    }

    fn nonlinear_output(&self) -> &Vector<OUT> {
        &self.a
    }

    fn linear_output(&self) -> &Vector<OUT> {
        &self.n
    }

    fn f(&self) -> Box<dyn Fn(f32) -> f32 + 'static> {
        Box::new(self.f.get_f())
    }

    fn df(&self) -> Box<dyn Fn(f32) -> f32 + 'static> {
        Box::new(self.f.get_df())
    }

    fn set_sensitivities(&mut self, s: Vector<OUT>) {
        self.Wᵀs = &self.W.T() * &s;
        self.s = s;
    }

    fn sensitivities(&self) -> &Vector<IN> {
        &self.Wᵀs
    }
}
