# MNIST Neural Network in Rust

A from-scratch neural network classifier for the MNIST handwritten digits dataset, implemented in pure Rust using only `ndarray` for matrix operations. No external ML frameworks — just linear algebra and calculus.

## Architecture

- **Input**: 784 neurons (28x28 pixel images)
- **Hidden**: 64 neurons with ReLU activation
- **Output**: 10 neurons with Softmax activation
- **Loss**: Cross-entropy
- **Optimizer**: Stochastic Gradient Descent (SGD)

## Performance

After 20 epochs with a learning rate of 0.1 and batch size of 64:

```
Epoch 20 | Loss: 0.0344 | Train Acc: 99.17%
--- Test Evaluation ---
Test Accuracy: 97.37%
```

## Project Structure

```
mnist_nn/
├── Cargo.toml
├── README.md
├── LICENSE
├── data/
│   ├── mnist_train.csv    # 60,000 training samples
│   └── mnist_test.csv     # 10,000 test samples
└── src/
    ├── main.rs      # Training loop and evaluation
    ├── math.rs      # Activation functions, loss, accuracy, He initialization
    ├── network.rs   # Layer struct and forward propagation
    ├── train.rs     # Backpropagation and SGD weight updates
    └── data.rs      # CSV loader with pixel normalization
```

## Key Implementation Details

- **He Initialization**: Prevents dying ReLUs by scaling initial weights based on layer size
- **Softmax + Cross-Entropy**: The output layer gradient simplifies elegantly to `pred - target`
- **Batch Training**: Processes data in mini-batches of 64 for stable gradient estimates
- **Pure Math**: All matrix operations implemented with `ndarray` — no TensorFlow, PyTorch, or JAX

## Building & Running

```bash
# Clone the repository
git clone <repo-url>
cd mnist_nn

# Build in release mode
cargo build --release

# Run training
cargo run --release
```

## Dependencies

- `ndarray` — N-dimensional arrays for Rust
- `ndarray-rand` — Random array generation
- `rand` — Random number utilities
- `csv` — CSV parsing for MNIST data

## License

MIT License — see [LICENSE](LICENSE) for details.

## Credits

Based on the tutorial "I Made A Neural Network In Rust... (From Scratch)" — implementing every layer, activation, and gradient calculation by hand to understand how neural networks actually work under the hood.
