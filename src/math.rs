use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use rand::rngs::StdRng;
use rand::SeedableRng;

pub fn he_init(rows: usize, cols: usize) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(42);
    let limit = (2.0 / rows as f64).sqrt();
    Array2::random_using((rows, cols), Uniform::new(-limit, limit), &mut rng)
}

pub fn relu(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| v.max(0.0))
}

pub fn relu_derivative(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
}

pub fn softmax(x: &Array2<f64>) -> Array2<f64> {
    let max = x.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp = x.mapv(|v| (v - max).exp());
    let sum = exp.sum_axis(Axis(1)).insert_axis(Axis(1));
    exp / sum
}

pub fn one_hot(labels: &Array1<usize>, classes: usize) -> Array2<f64> {
    let mut out = Array2::zeros((labels.len(), classes));
    for (i, &l) in labels.iter().enumerate() {
        out[[i, l]] = 1.0;
    }
    out
}

pub fn cross_entropy_loss(pred: &Array2<f64>, target: &Array2<f64>) -> f64 {
    let epsilon = 1e-12;
    let clipped = pred.mapv(|v| v.clamp(epsilon, 1.0 - epsilon));
    -(target * clipped.mapv(|v| v.ln())).sum() / pred.nrows() as f64
}

pub fn accuracy(pred: &Array2<f64>, labels: &Array1<usize>) -> usize {
    pred.rows()
        .into_iter()
        .zip(labels.iter())
        .filter(|(row, &label)| argmax(*row) == label)
        .count()
}

pub fn argmax(row: ndarray::ArrayView1<f64>) -> usize {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}
