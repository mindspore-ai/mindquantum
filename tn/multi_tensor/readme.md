# Example implementation of multi-tensor contraction algorithm 

This example contains a Python implementation for CPU of the multi-tensor simulation algorithm from [arXiv:2108.05665](https://arxiv.org/abs/2108.05665) 
and its usage for simulation of a simple circuit from 
[Google's quantum supremacy experiment](https://www.nature.com/articles/s41586-019-1666-5).

## Files description

The example contains the following files:

- utils.py -- contains auxiliary procedures and tables
- tensor_network.py -- contains structures and functions for working with tensor networks
- tree.py -- contains data structure and functions for preparing multi-tensor contraction
- contraction.py -- contains multi-amplitude contraction algorithm and an example of the multi-amplitude simulation

## Requirements

- python >= 3.8
- numpy
- yaml (pyyaml if install using conda)
- pytorch (CPU version)
- intbitset

## Running example

To run the example, use the command (python of version >= 3.8 is required):
> python contraction.py

It does the following steps:
- reads the circuit form *.qsim file and creates tensor network (applying gate fusion) corresponding 
  to this circuit (elided circuit for m=14 from Google's data);
- reads bitstrings with amplitudes from file (first 1000 bitstrings from the [Google's data](https://datadryad.org/stash/dataset/doi:10.5061/dryad.k6t1rj8));
- reads the contraction tree for this tensor network
- calculates amplitudes using multi-tensor contraction algorithm
- compares the calculated amplitudes with the amplitudes provided by Google 
  and prints the relative and absolute difference.
