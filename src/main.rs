#![allow(non_snake_case, uncommon_codepoints)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::marker::PhantomData;

use linalg::{Matrix, Vector};

mod linalg;

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
    fn f(&self) -> fn(f32) -> f32;
    fn df(&self) -> fn(f32) -> f32;
    fn set_sensitivities(&mut self, s: Vector<OUT>);
    fn sensitivities(&self) -> &Vector<IN>;
}

#[allow(dead_code)]
struct FullyConnectedLayer<const IN: usize, const OUT: usize>
    where [(); OUT*IN]: Sized
{
    W: Matrix<OUT, IN>, // weights
    b: Vector<OUT>,     // biases
    n: Vector<OUT>,     // net linear outputs
    a: Vector<OUT>,     // net nonlinear outputs
    s: Vector<OUT>,     // dL/dn of this layer
    Wᵀs: Vector<IN>,    // weighted dL/dn for backwards pass
    f: fn(f32) -> f32,
    df: fn(f32) -> f32,
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

pub struct Model<const IN: usize, const OUT: usize, T, L: ModelLayerChain<IN, OUT, T>> {
    layers: L,
    last_input: Vector<IN>,
    last_output: Vector<OUT>,
    errors: Vector<OUT>,
    loss: f32,

    _ph: PhantomData<T>,
}

impl<const IN: usize, const OUT: usize, T, L: ModelLayerChain<IN, OUT, T>> Model<IN, OUT, T, L> {
    pub fn new(layers: L) -> Self {
        Model {
            layers,
            last_input: Vector::zero(),
            last_output: Vector::zero(),
            errors: Vector::zero(),
            loss: 0f32,
            _ph: PhantomData::<T>
        }
    }

    pub fn run_once(&mut self, input: &Vector<IN>, target: &Vector<OUT>) {
        let ModelOutput { loss, errors, output } = self.layers.run_once(input, target);
        self.last_input = input.clone();
        self.last_output = output;
        self.loss = loss;
        self.errors = errors;
    }
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

pub struct ModelOutput<const DIM: usize> {
    loss: f32,
    errors: Vector<DIM>,
    output: Vector<DIM>,
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

fn main() {
    let mut chain = (
        FullyConnectedLayer {
            W: Matrix::from_cols(&[[0.05, 0.1, 0.2], [0.04, 0.1, 0.22], [0.71, 0.03, 0.23]]),
            b: Vector::from_arr([0.12, 0.23, 0.45]),
            n: Vector::zero(),
            a: Vector::zero(),
            s: Vector::zero(),
            Wᵀs: Vector::zero(),
            f: |x: f32| if x > 0f32 { x } else { 0.2 * x },
            df: |x: f32| if x > 0f32 { 1f32 } else { 0.2 },
        },
        FullyConnectedLayer {
            W: Matrix::from_cols(&[[0.15, 0.12, 0.21], [0.4, 0.01, 0.42], [0.75, 0.3, 0.3]]),
            b: Vector::from_arr([0.1, 0.3, 0.5]),
            n: Vector::zero(),
            a: Vector::zero(),
            s: Vector::zero(),
            Wᵀs: Vector::zero(),
            f: |x: f32| if x > 0f32 { 0.8 * x } else { 0.1 * x },
            df: |x: f32| if x > 0f32 { 0.8 } else { 0.1 },
        },
        FullyConnectedLayer {
            W: Matrix::from_cols(&[[0.715, 0.172, 0.271], [0.47, 0.07, 0.72], [0.5, 0.37, 0.73]]),
            b: Vector::from_arr([0.17, 0.37, 0.75]),
            n: Vector::zero(),
            a: Vector::zero(),
            s: Vector::zero(),
            Wᵀs: Vector::zero(),
            f: |x: f32| if x > 0f32 { 0.9 * x } else { 0.15 * x },
            df: |x: f32| if x > 0f32 { 0.9 } else { 0.15 },
        }
    );

    // let mut model = ManualModel {
    //     w1: Matrix::from_cols(&[[0.05, 0.1, 0.2], [0.04, 0.1, 0.22], [0.71, 0.03, 0.23]]),
    //     b1: Vector::from_arr([0.12, 0.23, 0.45]),
        
    //     n1: Vector::zero(),
    //     a1: Vector::zero(),
    //     s1: Vector::zero(),
        
    //     f1: |x: f32| if x > 0f32 { x } else { 0.2 * x },
    //     df1: |x: f32| if x > 0f32 { 1f32 } else { 0.2 },

    //     w2: Matrix::from_cols(&[[0.15, 0.12, 0.21], [0.4, 0.01, 0.42], [0.75, 0.3, 0.3]]),
    //     b2: Vector::from_arr([0.1, 0.3, 0.5]),
        
    //     n2: Vector::zero(),
    //     a2: Vector::zero(),
    //     s2: Vector::zero(),
        
    //     f2: |x: f32| if x > 0f32 { 0.8 * x } else { 0.1 * x },
    //     df2: |x: f32| if x > 0f32 { 0.8 } else { 0.1 },

    //     w3: Matrix::from_cols(&[[0.715, 0.172, 0.271], [0.47, 0.07, 0.72], [0.5, 0.37, 0.73]]),
    //     b3: Vector::from_arr([0.17, 0.37, 0.75]),
        
    //     n3: Vector::zero(),
    //     a3: Vector::zero(),
    //     s3: Vector::zero(),
        
    //     f3: |x: f32| if x > 0f32 { 0.9 * x } else { 0.15 * x },
    //     df3: |x: f32| if x > 0f32 { 0.9 } else { 0.15 },

    //     curr_input: Vector::zero(),
    //     curr_output: Vector::zero(),
    //     curr_target: Vector::zero(),
    //     curr_errors: Vector::zero(),
    //     curr_loss: 0f32,
    // };

    // let input = Vector::from_arr([1., 2., 3.]);
    // let target = Vector::from_arr([1., 2., 3.]);
    
    // model.forward(&input, &target);
    // model.backward();

    // println!("Input: {input}\nTarget: {target}\nErrors: {}\nLoss: {}\nS_1: {}\nS_2: {}\nS_3: {}",
    //     model.curr_errors, model.curr_loss, model.s1, model.s2, model.s3
    // );

    // let ModelOutput { loss: chain_loss, errors: chain_errors, .. } = chain.run_once(&input, &target);
    // println!("");
    // println!("Input: {input}\nTarget: {target}\nErrors: {chain_errors}\nLoss: {chain_loss}\nS_1: {}\nS_2: {}\nS_3: {}",
    //     chain.0.s, chain.1.s, chain.2.s);
    // model.n1[2] = model.n1[2] + 0.0001;
    // model.forward_dummy_n1();

    // println!("Input: {input}\nTarget: {target},Errors: {}\nLoss: {}\nS_3: {}",
    //     model.curr_errors, model.curr_loss, model.s1
    // );

    let input = Vector::from_arr([1., 2., 3.]);
    let target = Vector::from_arr([1., 2., 3.]);

    let mut model = Model::new(chain);
    // println!("{}\t{}", model.layers.0.W, model.layers.0.b);
    // println!("{}\t{}", model.layers.1.W, model.layers.1.b);
    // println!("{}\t{}", model.layers.2.W, model.layers.2.b);
    // println!("");
    for i in 0..5000 {
        model.run_once(&input, &target);
        if i % 100 == 0 {
            println!("[{i:>4}/5000] Loss: {}", model.loss);
        }
    }
    // println!("");
    // println!("{}\t{}", model.layers.0.W, model.layers.0.b);
    // println!("{}\t{}", model.layers.1.W, model.layers.1.b);
    // println!("{}\t{}", model.layers.2.W, model.layers.2.b);

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct ManualModel<const IN: usize, const MID1: usize, const MID2: usize, const OUT: usize>
    where [(); MID1*IN]: Sized, [(); MID2*MID1]: Sized, [(); OUT*MID2]: Sized
{
    pub w1: Matrix<MID1, IN>,
    pub b1: Vector<MID1>,

    pub n1: Vector<MID1>,
    pub a1: Vector<MID1>,
    pub s1: Vector<MID1>,
    
    pub f1: fn(f32) -> f32,
    pub df1: fn(f32) -> f32,
    
    pub w2: Matrix<MID2, MID1>,
    pub b2: Vector<MID2>,
    
    pub n2: Vector<MID2>,
    pub a2: Vector<MID2>,
    pub s2: Vector<MID2>,
    
    pub f2: fn(f32) -> f32,
    pub df2: fn(f32) -> f32,
    
    pub w3: Matrix<OUT, MID2>,
    pub b3: Vector<OUT>,
    
    pub n3: Vector<OUT>,
    pub a3: Vector<OUT>,
    pub s3: Vector<OUT>,
    
    pub f3: fn(f32) -> f32,
    pub df3: fn(f32) -> f32,
    
    pub curr_input: Vector<IN>,
    pub curr_output: Vector<OUT>,
    pub curr_target: Vector<OUT>,
    pub curr_errors: Vector<OUT>,
    
    pub curr_loss: f32,
}

impl<
    const IN: usize,
    const MID1: usize,
    const MID2: usize,
    const OUT: usize,
> ManualModel<IN, MID1, MID2, OUT>
    where
        [(); MID1*IN]: Sized,
        [(); MID2*MID1]: Sized,
        [(); OUT*MID2]: Sized,
{
    #[allow(dead_code)]
    pub fn forward(&mut self, input: &Vector<IN>, target: &Vector<OUT>) {
        self.curr_input = input.clone();
        self.curr_target = target.clone();

        self.n1 = &(&self.w1 * &self.curr_input) + &self.b1;
        self.a1 = self.n1.map(self.f1);

        self.n2 = &(&self.w2 * &self.a1) + &self.b2;
        self.a2 = self.n2.map(self.f2);

        self.n3 = &(&self.w3 * &self.a2) + &self.b3;
        self.a3 = self.n3.map(self.f3);

        self.curr_output = self.a3.clone();
        self.curr_errors = &self.curr_target - &self.curr_output;
        self.curr_loss = self.curr_errors.sum_of_squares();
    }

    #[allow(dead_code)]
    pub fn forward_dummy_n1(&mut self) {
        self.a1 = self.n1.map(self.f1);

        self.n2 = &(&self.w2 * &self.a1) + &self.b2;
        self.a2 = self.n2.map(self.f2);

        self.n3 = &(&self.w3 * &self.a2) + &self.b3;
        self.a3 = self.n3.map(self.f3);

        self.curr_output = self.a3.clone();
        self.curr_errors = &self.curr_target - &self.curr_output;
        self.curr_loss = self.curr_errors.sum_of_squares();
    }

    #[allow(dead_code)]
    pub fn forward_dummy_n2(&mut self) {
        self.a2 = self.n2.map(self.f2);

        self.n3 = &(&self.w3 * &self.a2) + &self.b3;
        self.a3 = self.n3.map(self.f3);

        self.curr_output = self.a3.clone();
        self.curr_errors = &self.curr_target - &self.curr_output;
        self.curr_loss = self.curr_errors.sum_of_squares();
    }

    #[allow(dead_code)]
    pub fn forward_dummy_n3(&mut self) {
        self.a3 = self.n3.map(self.f3);

        self.curr_output = self.a3.clone();
        self.curr_errors = &self.curr_target - &self.curr_output;
        self.curr_loss = self.curr_errors.sum_of_squares();
    }

    #[allow(dead_code)]
    pub fn backward(&mut self)
    where
        [(); MID1*IN]: Sized,
        [(); MID2*MID2]: Sized,
        [(); MID1*MID2]: Sized,
        [(); MID1*MID1]: Sized,
        [(); OUT*MID2]: Sized,
        [(); OUT*OUT]: Sized,
        [(); MID2*OUT]: Sized,
    {
        let x = (-2f32 * &self.n3.map(self.df3)).into();
        let y = Matrix::diag(&x);
        self.s3 = &y * &self.curr_errors;

        let x = self.n2.map(self.df2).into();
        let y = Matrix::diag(&x);
        self.s2 = &y * &(&self.w3.T() * &self.s3);

        let x = self.n1.map(self.df1).into();
        let y = Matrix::diag(&x);
        self.s1 = &y * &(&self.w2.T() * &self.s2);
    }
}