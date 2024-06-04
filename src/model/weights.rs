use num_traits::cast::AsPrimitive;
use rand::thread_rng;
use rand::distributions::{Distribution, Uniform};

use crate::linalg::vector::Vector;
use crate::linalg::matrix::Matrix;

pub enum Biases<const DIM: usize> {
    Zeros,
    UniformlyRandom { lo: f32, hi: f32 },
}

impl<const DIM: usize> Biases<DIM> {
    pub fn zeros() -> Self {
        Self::Zeros
    }

    pub fn uniformly_random(lo: f32, hi: f32) -> Self {
        Self::UniformlyRandom { lo, hi }
    }

    fn lazy_uniform_random(lo: f32, hi: f32) -> Vector<DIM> {
        let rng = thread_rng();
        let mut uniform = Uniform::<f32>::new(lo, hi).sample_iter(rng);
        let mut arr: [f32; DIM] = [0f32; DIM];
        for i in 0..DIM {
            arr[i] = uniform.next().unwrap();
        }
        Vector::from_arr(arr)
    }
}

// No From impl, this is a one-way conversion.
impl<const DIM: usize> Into<Vector<DIM>> for Biases<DIM> {
    fn into(self) -> Vector<DIM> {
        match self {
            Self::Zeros => Vector::zero(),
            Self::UniformlyRandom { lo, hi } => Biases::lazy_uniform_random(lo, hi),
        }
    }
}

impl<const DIM: usize> Default for Biases<DIM> {
    /// For biases vector of dim N, set elements uniformly at random in \[-1/√N, 1/√N\]
    fn default() -> Self {
        let n: f32 = <usize as AsPrimitive<f32>>::as_(DIM);
        Self::UniformlyRandom { lo: -1f32 / n.sqrt(), hi: 1f32 / n.sqrt() }
    }
}

pub enum Weights<const IN: usize, const OUT: usize> where [(); OUT*IN]: Sized {
    Zeros,
    UniformlyRandom { lo: f32, hi: f32 },
}

impl<const IN: usize, const OUT: usize> Weights<IN, OUT> where [(); OUT*IN]: Sized {
    pub fn zeros() -> Self {
        Self::Zeros
    }

    pub fn uniformly_random(lo: f32, hi: f32) -> Self {
        Self::UniformlyRandom { lo, hi }
    }

    fn lazy_uniform_random(lo: f32, hi: f32) -> Matrix<OUT, IN> {
        let rng = thread_rng();
        let mut uniform = Uniform::<f32>::new(lo, hi).sample_iter(rng);
        let mut arr: [f32; OUT*IN] = [0f32; OUT*IN];
        for i in 0..OUT*IN {
            arr[i] = uniform.next().unwrap();
        }
        // uniform.collect::<Vec<_>>().try_into().unwrap(); // TODO: this panics.
        Matrix::from_arr(arr)
    }
}

impl<const IN: usize, const OUT: usize> Default for Weights<IN, OUT> where [(); OUT*IN]: Sized {
    /// For weights matrix of R×C, set elements uniformly at random in \[-1/√C, 1/√C\]
    fn default() -> Self {
        let n: f32 = <usize as AsPrimitive<f32>>::as_(IN);
        Self::UniformlyRandom { lo: -1f32 / n.sqrt(), hi: 1f32 / n.sqrt() }
    }
}

// No From impl, this is a one-way conversion.
impl<const IN: usize, const OUT: usize> Into<Matrix<OUT, IN>> for Weights<IN, OUT> where [(); OUT*IN]: Sized {
    fn into(self) -> Matrix<OUT, IN> {
        match self {
            Self::Zeros => Matrix::zero(),
            Self::UniformlyRandom { lo, hi } => Weights::lazy_uniform_random(lo, hi),
        }
    }
}
