#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use linalg::{Matrix, Vector};

mod linalg;

fn main() {
    let mut model = Model {
        w1: Matrix::from_cols(&[[0.05, 0.1, 0.2], [0.04, 0.1, 0.22], [0.71, 0.03, 0.23]]),
        b1: Vector::from_arr([0.12, 0.23, 0.45]),
        n1: Vector::zero(), a1: Vector::zero(), s1: Vector::zero(),
        f1: |x: f32| if x > 0f32 { x } else { 0.2 * x },
        df1: |x: f32| if x > 0f32 { 1f32 } else { 0.2 },

        w2: Matrix::from_cols(&[[0.15, 0.12, 0.21], [0.4, 0.01, 0.42], [0.75, 0.3, 0.3]]),
        b2: Vector::from_arr([0.1, 0.3, 0.5]),
        n2: Vector::zero(), a2: Vector::zero(), s2: Vector::zero(),
        f2: |x: f32| if x > 0f32 { 0.8 * x } else { 0.1 * x },
        df2: |x: f32| if x > 0f32 { 0.8 } else { 0.1 },

        w3: Matrix::from_cols(&[[0.715, 0.172, 0.271], [0.47, 0.07, 0.72], [0.5, 0.37, 0.73]]),
        b3: Vector::from_arr([0.17, 0.37, 0.75]),
        n3: Vector::zero(), a3: Vector::zero(), s3: Vector::zero(),
        f3: |x: f32| if x > 0f32 { 0.9 * x } else { 0.15 * x },
        df3: |x: f32| if x > 0f32 { 0.9 } else { 0.15 },

        curr_input: Vector::zero(),
        curr_output: Vector::zero(),
        curr_target: Vector::zero(),
        curr_errors: Vector::zero(),
        curr_loss: 0f32,
    };

    let input = Vector::from_arr([1., 2., 3.]);
    let target = Vector::from_arr([1., 2., 3.]);
    model.forward(&input, &target);
    model.backward();

    println!("Input: {input}\nTarget: {target},Errors: {}\nLoss: {}\nS_3: {}",
        model.curr_errors, model.curr_loss, model.s1
    );

    model.n1[2] = model.n1[2] + 0.0001;
    model.forward_dummy_n1();

    println!("Input: {input}\nTarget: {target},Errors: {}\nLoss: {}\nS_3: {}",
        model.curr_errors, model.curr_loss, model.s1
    );
}

struct Model<const IN: usize, const MID1: usize, const MID2: usize, const OUT: usize>
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

impl<const IN: usize, const MID1: usize, const MID2: usize, const OUT: usize> Model<IN, MID1, MID2, OUT>
    where [(); MID1*IN]: Sized, [(); MID2*MID1]: Sized, [(); OUT*MID2]: Sized
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
    where [(); MID1*IN]: Sized, [(); MID2*MID2]: Sized, [(); MID1*MID2]: Sized, [(); MID1*MID1]: Sized, [(); OUT*MID2]: Sized, [(); OUT*OUT]: Sized, [(); MID2*OUT]: Sized
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