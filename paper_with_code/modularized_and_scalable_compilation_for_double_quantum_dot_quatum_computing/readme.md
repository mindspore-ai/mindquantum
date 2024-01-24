# Paper: Modularized and Scalable Compilation for Double Quantum Dot Quantum Computing  

## Abstract: 

Any quantum program on a realistic quantum device must be compiled into an executable form while taking into account the underlying hardware constraints. Stringent restrictions on architecture and control imposed by physical platforms make this very challenging. In this paper, based on the quantum variational algorithm, we propose a novel scheme to train an Ansatz circuit and realize high-fidelity compilation of a set of universal quantum gates for singlet-triplet qubits in semiconductor double quantum dots, a fairly heavily constrained system. Furthermore, we propose a scalable architecture for a modular implementation of quantum programs in this constrained systems and validate its performance with two representative demonstrations, the Grover's algorithm for the database searching (static compilation) and a variant of variational quantum eigensolver for the Max-Cut optimization (dynamic compilation). Our methods are potentially applicable to a wide range of physical devices.  This work constitutes an important stepping-stone for exploiting the potential for advanced and complicated quantum algorithms on near-term devices.

## ArXiv: 
[https://arxiv.org/abs/2211.05300](https://arxiv.org/abs/2211.05300)

## QST:

https://iopscience.iop.org/article/10.1088/2058-9565/acfe38

**代码说明**：

原始代码基于 MindQuantum 0.6.0 开发，由于涉及修改 MindQuantum 部分源代码，所以读者可能无法直接运行。作者正逐步将代码迁移到 MindQuantum 0.10 版本。目前已完成对 `one_qubit_case.ipynb` 的更新，后续将继续更新。