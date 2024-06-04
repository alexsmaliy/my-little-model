#![allow(non_snake_case, uncommon_codepoints)]  // for symbols and one-letter variables standard in the ML literature
#![allow(incomplete_features)]                  // for dimension analysis of matrix math
#![feature(generic_const_exprs)]

// TODO: restrict module visibility through selective re-exports.
pub mod layer;
pub mod linalg;
pub mod model;
