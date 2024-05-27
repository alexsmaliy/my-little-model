use rand::thread_rng;
use rand::distributions::{Distribution, Uniform};

use crate::linalg::{Matrix, Vector};

pub enum Biases<const DIM: usize> {
    Zeros(Vector<DIM>),
    UniformlyRandom(Vector<DIM>),
}

impl<const DIM: usize> Biases<DIM> {
    pub fn zeros() -> Self {
        Self::Zeros(Vector::zero())
    }

    pub fn uniformly_random(lo: f32, hi: f32) -> Self {
        let rng = thread_rng();
        let mut uniform = Uniform::<f32>::new(lo, hi).sample_iter(rng);
        let mut arr: [f32; DIM] = [0f32; DIM];
        for i in 0..DIM {
            arr[i] = uniform.next().unwrap();
        }
        Self::UniformlyRandom(Vector::from_arr(arr))
    }
}

impl<const DIM: usize> Into<Vector<DIM>> for Biases<DIM> {
    fn into(self) -> Vector<DIM> {
        match self {
            Self::Zeros(v) => v,
            Self::UniformlyRandom(v) => v,
        }
    }
}

pub enum Weights<const IN: usize, const OUT: usize> where [(); OUT*IN]: Sized {
    Zeros(Matrix<OUT, IN>),
    UniformlyRandom(Matrix<OUT, IN>),
}

impl<const IN: usize, const OUT: usize> Weights<IN, OUT> where [(); OUT*IN]: Sized {
    pub fn zeros() -> Self {
        Self::Zeros(Matrix::zero())
    }

    pub fn uniformly_random(lo: f32, hi: f32) -> Self {
        let rng = thread_rng();
        let mut uniform = Uniform::<f32>::new(lo, hi).sample_iter(rng);
        let mut arr: [f32; OUT*IN] = [0f32; OUT*IN];
        for i in 0..OUT*IN {
            arr[i] = uniform.next().unwrap();
        }
        // uniform.collect::<Vec<_>>().try_into().unwrap(); // TODO: this panics.
        Self::UniformlyRandom(Matrix::from_arr(arr))
    }
}

// No From impl, this is a one-way conversion.
impl<const IN: usize, const OUT: usize> Into<Matrix<OUT, IN>> for Weights<IN, OUT> where [(); OUT*IN]: Sized {
    fn into(self) -> Matrix<OUT, IN> {
        match self {
            Self::Zeros(m) => m,
            Self::UniformlyRandom(m) => m,
        }
    }
}
