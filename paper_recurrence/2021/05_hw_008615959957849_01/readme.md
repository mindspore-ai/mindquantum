<script
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</script>

# Paper05: Microcanonical and finite-temperature ab initio molecular dynamics simulations on quantum computers

## 项目介绍

This paper mainly used different types of VQE to simulate hydrogen and trihydrogen ion, which are respectively through the method of tapering off number of qubits the from parity form of Hamiltonian of hydrogen and seniority-zero Hamiltonian of trihydrogen ion in the first part. In another part of this paper, they experimented with algorithms on real a quantum computer and provide solutions for the alleviation of the statistical noise associated with the measurements of the expectation values of energies and forces, as well as schemes for the mitigation of the hardware noise sources.

This recurrence mainly reproduced FIG.3. in this paper,i.e. the simulation of hydrogen.


## 主要结果

After the work was done, I found:

1. VQE is a good method to simulate molecule, which produces a high accuracy.
2. Transform Hamiltonian to QubitOperator with the method of parity is good for the further taperting off to reduce qubits that are needed to simulate  hydrogen or maybe other symmetrical molecules.
3. Mindquantum is very convenient for VQE. Maybe it is better to add the function of tapering off.


## 创新点

A simple function for tapering off.

邮箱地址：Franke_cdk@outlook.com