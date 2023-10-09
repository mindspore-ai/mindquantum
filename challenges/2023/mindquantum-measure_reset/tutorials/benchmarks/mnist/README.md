# Description

These scripts are going to test the performance between MindQuantum and TensorFlow Quantum on mnist dataset.

Please install TensorFlow Quantum before running these scripts.

## MindQuantum

Run the command below to run mnist classification with MindQuantum.

```bash
python3 mnist.py -n -1 -b 2 -o 8 -p 2
```

## TensorFlow Quantum

Run the command below to run mnist classification with TensorFlow Quantum.

```bash
python3 mnist_tf.py -n -1 -b 2 -o 16
```
