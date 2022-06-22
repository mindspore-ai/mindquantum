# Description

These scripts are going to test the performance between MindQuantum and TensorFlow Quantum about gradient calculation on mnist dataset.

Please install TensorFlow Quantum before running these scripts.

## MindQuantum

Run the command below to run mnist classification with MindQuantum.

```bash
python3 mindquantum_grad.py -n 500 -o 8
```

## TensorFlow Quantum

Run the command below to run mnist classification with TensorFlow Quantum.

```bash
python3 tensorflow_quantum_grad.py -n 500 -o 8
```
