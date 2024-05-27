#![allow(non_snake_case, uncommon_codepoints)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use linalg::{Matrix, Vector};
use model::Model;
use layer::connected::FullyConnectedLayer;
use model::transfer_function::LeakyReLU;
use model::weights::{Biases, Weights};

mod layer;
mod linalg;
mod model;

fn main() {
    let (lo, hi) = (-1f32/f32::sqrt(6.), 1f32/f32::sqrt(6.));

    let chain = (
        FullyConnectedLayer::new(
            Weights::<3, 4>::uniformly_random(lo, hi),
            Biases::uniformly_random(lo, hi),
            LeakyReLU { slope_lt0: 0.2, slope_gte0: 1.0 },
        ),
        // FullyConnectedLayer {
        //     W: Matrix::from_cols(&[[0.05, 0.1, 0.2], [0.04, 0.1, 0.22], [0.71, 0.03, 0.23]]),
        //     b: Vector::from_arr([0.12, 0.23, 0.45]),
        //     n: Vector::zero(),
        //     a: Vector::zero(),
        //     s: Vector::zero(),
        //     Wᵀs: Vector::zero(),
        //     f: |x: f32| if x > 0f32 { x } else { 0.2 * x },
        //     df: |x: f32| if x > 0f32 { 1f32 } else { 0.2 },
        // },
        FullyConnectedLayer::new(
            Weights::<4, 4>::uniformly_random(lo, hi),
            Biases::uniformly_random(lo, hi),
            LeakyReLU { slope_lt0: 0.1, slope_gte0: 0.8 },
        ),
        // FullyConnectedLayer {
        //     W: Matrix::from_cols(&[[0.15, 0.12, 0.21], [0.4, 0.01, 0.42], [0.75, 0.3, 0.3]]),
        //     b: Vector::from_arr([0.1, 0.3, 0.5]),
        //     n: Vector::zero(),
        //     a: Vector::zero(),
        //     s: Vector::zero(),
        //     Wᵀs: Vector::zero(),
        //     f: |x: f32| if x > 0f32 { 0.8 * x } else { 0.1 * x },
        //     df: |x: f32| if x > 0f32 { 0.8 } else { 0.1 },
        // },
        FullyConnectedLayer::new(
            Weights::<4, 3>::uniformly_random(lo, hi),
            Biases::uniformly_random(lo, hi),
            LeakyReLU { slope_lt0: 0.15, slope_gte0: 0.9 },
        ),
        // FullyConnectedLayer {
        //     W: Matrix::from_cols(&[[0.715, 0.172, 0.271], [0.47, 0.07, 0.72], [0.5, 0.37, 0.73]]),
        //     b: Vector::from_arr([0.17, 0.37, 0.75]),
        //     n: Vector::zero(),
        //     a: Vector::zero(),
        //     s: Vector::zero(),
        //     Wᵀs: Vector::zero(),
        //     f: |x: f32| if x > 0f32 { 0.9 * x } else { 0.15 * x },
        //     df: |x: f32| if x > 0f32 { 0.9 } else { 0.15 },
        // }
    );

    let mut model = model::manual::ManualModelDoNotUse {
        w1: Matrix::from_cols(&[[0.05, 0.1, 0.2], [0.04, 0.1, 0.22], [0.71, 0.03, 0.23]]),
        b1: Vector::from_arr([0.12, 0.23, 0.45]),
        
        n1: Vector::zero(),
        a1: Vector::zero(),
        s1: Vector::zero(),
        
        f1: |x: f32| if x > 0f32 { x } else { 0.2 * x },
        df1: |x: f32| if x > 0f32 { 1f32 } else { 0.2 },

        w2: Matrix::from_cols(&[[0.15, 0.12, 0.21], [0.4, 0.01, 0.42], [0.75, 0.3, 0.3]]),
        b2: Vector::from_arr([0.1, 0.3, 0.5]),
        
        n2: Vector::zero(),
        a2: Vector::zero(),
        s2: Vector::zero(),
        
        f2: |x: f32| if x > 0f32 { 0.8 * x } else { 0.1 * x },
        df2: |x: f32| if x > 0f32 { 0.8 } else { 0.1 },

        w3: Matrix::from_cols(&[[0.715, 0.172, 0.271], [0.47, 0.07, 0.72], [0.5, 0.37, 0.73]]),
        b3: Vector::from_arr([0.17, 0.37, 0.75]),
        
        n3: Vector::zero(),
        a3: Vector::zero(),
        s3: Vector::zero(),
        
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
