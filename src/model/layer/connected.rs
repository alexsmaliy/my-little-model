use crate::linalg::{Matrix, Vector};
use super::ModelLayer;

#[allow(dead_code)]
pub struct FullyConnectedLayer<const IN: usize, const OUT: usize>
    where [(); OUT*IN]: Sized
{
    pub W: Matrix<OUT, IN>, // weights
    pub b: Vector<OUT>,     // biases
    pub n: Vector<OUT>,     // net linear outputs
    pub a: Vector<OUT>,     // net nonlinear outputs
    pub s: Vector<OUT>,     // dL/dn of this layer
    pub Wᵀs: Vector<IN>,    // weighted dL/dn for backwards pass
    pub f: fn(f32) -> f32,
    pub df: fn(f32) -> f32,
}

impl<const IN: usize, const OUT: usize> ModelLayer<IN, OUT> for FullyConnectedLayer<IN, OUT>
    where
        [(); IN*OUT]: Sized,
        [(); OUT*IN]: Sized,
        [(); OUT*OUT]: Sized,
        [(); 1*IN]: Sized,
        [(); IN*1]: Sized,
{
    fn forward(&mut self, input_src: &Vector<IN>) {
        self.n = &(&self.W * input_src) + &self.b;
        self.a = self.n.map(self.f);
    }

    fn backward(&mut self, upstream_Wᵀs: &Vector<OUT>) {
        /*
        // needs: n_i and df_i, but WT_i+1 and s_i+1
        // thus: get upstream Wᵀs to compute and set own Wᵀs

        let x = self.n1.map(self.df1).into();
        let y = Matrix::diag(&x);
        self.s1 = &y * &(&self.w2.T() * &self.s2);
         */
        let x = self.n.map(self.df).into();
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

    fn f(&self) -> fn(f32) -> f32 {
        self.f
    }

    fn df(&self) -> fn(f32) -> f32 {
        self.df
    }

    fn set_sensitivities(&mut self, s: Vector<OUT>) {
        self.Wᵀs = &self.W.T() * &s;
        self.s = s;
    }

    fn sensitivities(&self) -> &Vector<IN> {
        &self.Wᵀs
    }
}
