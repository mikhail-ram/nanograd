# nanograd

`nanograd` is a lightweight, zero-dependency neural network library in Haskell, inspired by Andrej Karpathy's `micrograd`. It provides the foundational components for building and training simple neural networks in Haskell with a clear, functional API.

## Core Concepts

The library is built around a few core data types:

-   `Neuron`: Represents a single neuron with a set of weights and a bias.
-   `Layer`: An abstraction for a layer of neurons, which can be a `DenseLayer` (fully connected) or an `ActivationLayer` (e.g., `Sigmoid`).
-   `Network`: A list of `Layer`s, forming a multi-layer perceptron (MLP).

## API Overview

The primary functions for operating on the network are:

-   `forward`: Performs a forward pass through the network, taking an `InputVector` and producing an `OutputVector`.
-   `backward`: Computes the gradients of the network's parameters with respect to a loss, using backpropagation.
-   `update`: Updates the network's weights and biases based on the computed gradients and a learning rate.

## Building and Running

To build and run the project, you can use `cabal`:

```bash
# Build the project
cabal build

# Run the XOR example
cabal run xor

# Run the Iris example
cabal run iris
```

## Todos:
- [x] proof-of-concept for XOR
- [x] proof-of-concept for iris classification
- [ ] efficient n-dimensional tensors
- [ ] Grenade-style dependent types
- [ ] ReLU and Softmax activations
- [ ] batching
- [ ] train-test-validation split
- [ ] optimizers - momentum
- [ ] convolutions
