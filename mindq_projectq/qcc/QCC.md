## <center>Qubit Coupled Cluster (QCC) method</center>

## 1. Introduction

​	Compared with unitary coupled cluster (UCC) ansatz, QCC method build ansatz directly from "qubit space". "entanglers" in the "qubit space" are ranked according to their estimated contributions into the correlation energy.

## 2. algorithm

### 	2.1 The mean-field wavefunction 

​		The mean-field wavefunction $|\mathbf{\Omega}\rangle$:
$$
|\mathbf{\Omega}\rangle=\prod_{i=1}^{N_q}|\Omega_i\rangle
$$
​		Where $|\Omega_i\rangle=\cos(\frac{\theta_i}{2})|\alpha\rangle+e^{i\phi_i}\sin(\frac{\theta_i}{2})|\beta\rangle$​​, $\phi_i$​​​ and $\theta_i$​ are azimuthal and polar angles one the "Bloch sphere" of the $i^{th}$​​ qubit. $|\alpha\rangle$​ and $|\beta\rangle$​​ are spin-up and spin-down eigenstates of the $\frac{\hat{z_i}}{2}$​​	. This layer have $2N_q$ independent parameters.

### 2.2 entanglers

​	Correlation is introduced by entanglers.

​		Entangler:  $e^{-i\tau\hat{P}/2}$​, where $\hat{P}$ is pauli words.​​

​		Many entanglers in the ansatz under construction:

$$
\hat{U}(\mathbf{\tau})=\prod_{k=1}^{N_{ent}}e^{-i\tau_k\hat{P_k}/2}
$$

​	The expectation value of the Hamiltonian for the QCC parameterization is:

$$
E(\mathbf{\tau}, \mathbf{\Omega})=\langle\mathbf{\Omega}|\hat{U}^\dagger(\mathbf{\tau})\hat{H}\hat{U}(\mathbf{\tau})|\mathbf{\Omega}\rangle
$$

### 2.3 entangler ranking

​	The quantity of all entanglers is $4^{N_q}-3N_q-1$. A pre-screening based on the first and second terms of the Taylor expansion by evaluating $\frac{d E(\tau, \hat{P})}{d\tau} |_{\tau=0}$ and $\frac{d^2 E(\tau, \hat{P})}{d^2 \tau} |_{\tau=0}$.

## 3. Reference

origin paper: [Qubit Coupled Cluster Method: A Systematic Approach to Quantum Chemistry on a Quantum Computer]([Qubit Coupled Cluster Method: A Systematic Approach to Quantum Chemistry on a Quantum Computer | Journal of Chemical Theory and Computation (acs.org)](https://pubs.acs.org/doi/full/10.1021/acs.jctc.8b00932))



















