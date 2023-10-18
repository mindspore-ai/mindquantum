# Variational ansatz-based quantum simulation of imaginary time evolution

## Introduction

This paper proposed ansatz-based imaginary time evolution method for the optimization of VQE. In this recurrence, we implement this method, and reproduce the experiment on LiH and H2 system.

## 主要结果
<script
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</script>

In this recurrence, we implemented the ansatz-based imaginary time evolution method and tested its performance on VQE of H2 and LiH systems. We proposed two methods to calculate the forward mode gradients: 1. use of an helper function 2. direct manipulation of the mindquantum circuit object. Our test shows that these two methods work well and provide exactly the same gradients as the reverse mode method built into mindquantum. To compute the Hessian-like preconditioner, we proposed a hybrid mode computation with only $N$ passes for $N$ parameters, in addition to the widely used forward mode method with $N^2$ passes for $N$ parameters. The test shows that the method works well. In addition to the ansatz-based imaginary time evolution, we have also implemented the quantum gradient descent method and tested its performance.

Our experiment shows that the ansatz-based imaginary time evolution converges much faster than the gradient descent method, in agreement with the argumentation in this paper.


## 创新点


Calculating Hessian $\frac{\partial <\phi| \partial |\phi>}{\partial \theta_i \partial \theta_j}$ is a notoriously expensive operation for forward mode differentiation, since we need to calculate $N^2$ expectation for $N$ parameters. However, good news is that for reverse mode differentiation, we only need to calculate one expectation for all $N$ parameters, this would be beneficial if we could use such mode in Hessian calculation. Here I propose a new hybrid mode for this calculation: For each parameter $\theta_i$, we firstly setup the expectation target using forward mode(which means the expectation value itself is $\frac{\partial <\phi|\phi>}{\partial \theta_i}$), and then we use reverse model AD to calculate $\frac{\partial <\phi| \partial |\phi>}{\partial \theta_i \partial \theta_j}$ with respect to all parameter $\theta_j$ through one pass. In this case, only $N$ expectation calculation is needed! 
Experiments show that hybridMode is 20x faster than forwardMode for H2 simulation.

In addition to ansatz-based imaginary time evolution, we also implemented the quantum natural gradient descent method, and tested its performance.

邮箱地址：le.niu@hotmail.com