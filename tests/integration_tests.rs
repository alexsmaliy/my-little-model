#![allow(non_snake_case, uncommon_codepoints)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use mylittlemodel::linalg::{OldMatrixDoNotUse, OldVectorDoNotUse, Vector};
use mylittlemodel::model::loss::MeanSquaredErrorLoss;
use mylittlemodel::model::Model;
use mylittlemodel::layer::connected::FullyConnectedLayer;
use mylittlemodel::model::activation::LeakyReLU;
use mylittlemodel::model::weights::{Biases, Weights};

// Run with `cargo test -- --nocapture` to inspect stdout.

#[test]
fn kitchen_sink_test() {
    let chain = (
        FullyConnectedLayer::with(
            Weights::<3, 50>::default(),
            Biases::default(),
            LeakyReLU { slope_lt0: 0.2, slope_gte0: 1.0 },
        ),
        FullyConnectedLayer::with(
            Weights::<50, 50>::default(),
            Biases::default(),
            LeakyReLU { slope_lt0: 0.1, slope_gte0: 0.8 },
        ),
        FullyConnectedLayer::with(
            Weights::<50, 3>::default(),
            Biases::default(),
            LeakyReLU { slope_lt0: 0.15, slope_gte0: 0.9 },
        ),
    );

    // an example fully specified by hand for comparing correctness
    let mut model = mylittlemodel::model::manual::ManualModelDoNotUse {
        w1: OldMatrixDoNotUse::from_cols(&[[0.05, 0.1, 0.2], [0.04, 0.1, 0.22], [0.71, 0.03, 0.23]]),
        b1: OldVectorDoNotUse::from_arr([0.12, 0.23, 0.45]),
        
        n1: OldVectorDoNotUse::zero(),
        a1: OldVectorDoNotUse::zero(),
        s1: OldVectorDoNotUse::zero(),
        
        f1: |x: f32| if x > 0f32 { x } else { 0.2 * x },
        df1: |x: f32| if x > 0f32 { 1f32 } else { 0.2 },

        w2: OldMatrixDoNotUse::from_cols(&[[0.15, 0.12, 0.21], [0.4, 0.01, 0.42], [0.75, 0.3, 0.3]]),
        b2: OldVectorDoNotUse::from_arr([0.1, 0.3, 0.5]),
        
        n2: OldVectorDoNotUse::zero(),
        a2: OldVectorDoNotUse::zero(),
        s2: OldVectorDoNotUse::zero(),
        
        f2: |x: f32| if x > 0f32 { 0.8 * x } else { 0.1 * x },
        df2: |x: f32| if x > 0f32 { 0.8 } else { 0.1 },

        w3: OldMatrixDoNotUse::from_cols(&[[0.715, 0.172, 0.271], [0.47, 0.07, 0.72], [0.5, 0.37, 0.73]]),
        b3: OldVectorDoNotUse::from_arr([0.17, 0.37, 0.75]),
        
        n3: OldVectorDoNotUse::zero(),
        a3: OldVectorDoNotUse::zero(),
        s3: OldVectorDoNotUse::zero(),
        
        f3: |x: f32| if x > 0f32 { 0.9 * x } else { 0.15 * x },
        df3: |x: f32| if x > 0f32 { 0.9 } else { 0.15 },

        curr_input: OldVectorDoNotUse::zero(),
        curr_output: OldVectorDoNotUse::zero(),
        curr_target: OldVectorDoNotUse::zero(),
        curr_errors: OldVectorDoNotUse::zero(),
        curr_loss: 0f32,
    };

    let input = OldVectorDoNotUse::from_arr([1., 2., 3.]);
    let target = OldVectorDoNotUse::from_arr([1., 2., 3.]);
    
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
    let target = Vector::from_arr([0., 0., 1.]);

    let mut model = Model::new(chain, MeanSquaredErrorLoss);
    // println!("{}\t{}", model.layers.0.W, model.layers.0.b);
    // println!("{}\t{}", model.layers.1.W, model.layers.1.b);
    // println!("{}\t{}", model.layers.2.W, model.layers.2.b);
    // println!("");
    for i in 0..5000 {
        model.train_single(&input, &target);
        if i % 100 == 0 {
            println!("[{i:>4}/5000] Loss: {}", model.loss);
        }
    }
    // println!("");
    // println!("{}\t{}", model.layers.0.W, model.layers.0.b);
    // println!("{}\t{}", model.layers.1.W, model.layers.1.b);
    // println!("{}\t{}", model.layers.2.W, model.layers.2.b);

}
