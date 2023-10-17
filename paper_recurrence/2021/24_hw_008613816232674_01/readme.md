<script
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</script>

# Quantum simulation with hybrid tensor networks

## 项目介绍

This paper proposes a general framework denoting hybrid tensor networks, which is the combination of classical tensor networks and quantum tensor networks. More precisely, the paper summarizes the general strategy to contract indices between different types of tensors:

    1. classical tensor + quantum tensor, with classical indices
    2. classical tensor + quantum tensor, with classical indices + quantum indices 
    3. classical tensor + quantum tensor, with quantum indices
    4. quantum tensor + quantum tensor, with quantum indices + classical indices
    5. quantum tensor + quantum tensor, with quantum indices

These operations can be performed with a combination of classical tensor contraction and quantum circuit measurement. The combination of these operations provides great flexibility and expressiveness to the hybrid tensor networks. More precisely, one can use a tree-structured tensor network where each layer is represented by a quantum tensor (prepared by a quantum circuit). Thus, each building block is a large full-rank tensor and the tensor network is much more powerful after contraction than conventional low-rank proposals such as the MPS approach (where all building blocks are just small pieces due to the limitation of computational resources).

In this repetition, we show how to define a hybrid tensor network with two layers of quantum tensors (quantum circuit), which serves as the $|\Phi(\theta)>$ ansatz for VQE problem, and the optimization of the circuit parameters using the ansatz-based imaginary time evolution method.


## 主要结果

We implemented the hybrid quantum tree tensor network(HQTTN) ansatz proposed in the paper, and tried ansatz based imaginary time evolution for VQE problem with ising-like Hamiltonian. There are still problems of matching our optimization result with that of tensor network based MPS ansatz with DMRG optimization method. We will solve these problems in the future versions.


## 创新点

Instead of using case 4 (quantum tensor + quantum tensor, with quantum indices + classical indices) for the contraction, as described in the numerical experiments section of the paper, we used case 5 (quantum tensor + quantum tensor, with quantum indices) for the contraction. The side effect of this choice is that the resulting tensor network is no longer guaranteed to be normalized. This was confirmed in our experiment.


邮箱地址：le.niu@hotmail.com