# variational_quantum_circuits_enhanced_generative_adversarial_network

This example uses the standard MNIST IDX files. The repository no longer stores
the local `mnist.rar` archive in order to keep the branch smaller.

1. Create a `mnist/` directory next to `QGAN.py`.
2. Download the standard MNIST files from:
   - `https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz`
   - `https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz`
   - `https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz`
   - `https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz`
3. Extract the `.gz` files into `mnist/`, so the directory contains files such
   as `train-images-idx3-ubyte` and `train-labels-idx1-ubyte`.
4. Run `QGAN.py`.
