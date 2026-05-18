use ndarray::{Array1, Array2};
use crate::math::*;

pub struct Layer {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
    pub z: Option<Array2<f64>>,      // pre-activation
    pub a: Option<Array2<f64>>,      // post-activation
}

impl Layer {
    pub fn new(in_size: usize, out_size: usize) -> Self {
        Self {
            weights: he_init(out_size, in_size),
            biases: Array1::zeros(out_size),
            z: None,
            a: None,
        }
    }

    pub fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        let z = input.dot(&self.weights.t()) + &self.biases;
        self.z = Some(z.clone());
        let a = relu(&z);
        self.a = Some(a.clone());
        a
    }

    pub fn forward_output(&mut self, input: &Array2<f64>) -> Array2<f64> {
        let z = input.dot(&self.weights.t()) + &self.biases;
        self.z = Some(z.clone());
        let a = softmax(&z);
        self.a = Some(a.clone());
        a
    }
}
