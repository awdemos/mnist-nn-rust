mod math;
mod network;
mod data;
mod train;

use network::*;
use data::*;
use train::*;
use math::*;

fn main() {
    println!("Loading MNIST dataset...");
    let (x_train, y_train_raw) = load_csv("data/mnist_train.csv");
    let (x_test, y_test_raw) = load_csv("data/mnist_test.csv");
    println!("Train: {} samples, Test: {} samples", x_train.nrows(), x_test.nrows());

    let y_train = one_hot(&y_train_raw, 10);
    let _y_test = one_hot(&y_test_raw, 10);

    let mut layers = vec![
        Layer::new(784, 64),
        Layer::new(64, 10),
    ];
    let lr = 0.1;
    let epochs = 20;
    let batch_size = 64;

    println!("\nTraining neural network: 784 -> 64 -> 10");
    println!("Learning rate: {}, Epochs: {}, Batch size: {}\n", lr, epochs, batch_size);

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut correct = 0;
        let mut num_batches = 0;

        for batch_start in (0..x_train.nrows()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(x_train.nrows());
            let x_batch = x_train.slice(ndarray::s![batch_start..batch_end, ..]).to_owned();
            let y_batch = y_train.slice(ndarray::s![batch_start..batch_end, ..]).to_owned();

            // Forward
            let h = layers[0].forward(&x_batch);
            let pred = layers[1].forward_output(&h);

            // Loss & accuracy
            total_loss += cross_entropy_loss(&pred, &y_batch);
            correct += accuracy(&pred, &y_train_raw.slice(ndarray::s![batch_start..batch_end]).to_owned());
            num_batches += 1;

            // Backward
            let grads = backward(&mut layers, &x_batch, &y_batch);
            update_weights(&mut layers, &grads, lr);
        }

        let avg_loss = total_loss / num_batches as f64;
        let train_acc = 100.0 * correct as f64 / x_train.nrows() as f64;
        println!(
            "Epoch {:2} | Loss: {:.4} | Train Acc: {:.2}%",
            epoch + 1,
            avg_loss,
            train_acc
        );
    }

    // Test evaluation
    println!("\n--- Test Evaluation ---");
    let h = layers[0].forward(&x_test);
    let pred = layers[1].forward_output(&h);
    let test_acc = accuracy(&pred, &y_test_raw);
    println!("Test Accuracy: {:.2}%", 100.0 * test_acc as f64 / x_test.nrows() as f64);
}
