use ndarray::{Array1, Array2, Axis};
use crate::math::relu_derivative;
use crate::network::Layer;

pub struct Gradients {
    pub dw: Array2<f64>,
    pub db: Array1<f64>,
}

pub fn backward(
    layers: &mut [Layer],
    input: &Array2<f64>,
    target: &Array2<f64>,
) -> Vec<Gradients> {
    let mut grads = vec![];
    let n_samples = input.nrows() as f64;

    // Output layer gradient
    let output = layers.last().unwrap().a.as_ref().unwrap();
    let mut dz = output - target;  // Softmax + CrossEntropy simplifies to this

    for i in (0..layers.len()).rev() {
        let prev_a = if i == 0 {
            input.clone()
        } else {
            layers[i - 1].a.as_ref().unwrap().clone()
        };

        let dw = prev_a.t().dot(&dz) / n_samples;
        let db = dz.mean_axis(Axis(0)).unwrap();

        grads.push(Gradients { dw, db });

        if i > 0 {
            let w = layers[i].weights.clone();
            let da_prev = dz.dot(&w);
            let z = layers[i - 1].z.as_ref().unwrap();
            dz = da_prev * relu_derivative(z);
        }
    }

    grads.reverse();
    grads
}

pub fn update_weights(layers: &mut [Layer], grads: &[Gradients], lr: f64) {
    for (layer, grad) in layers.iter_mut().zip(grads.iter()) {
        layer.weights -= &(grad.dw.t().to_owned() * lr);
        layer.biases -= &(grad.db.clone() * lr);
    }
}
