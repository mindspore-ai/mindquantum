# <center>Unitary Coupled Cluster (UCC)</center>

## 1. Introduction

The Unitary Coupled Cluster (UCC) ansatz is widely used from the very beginning of the VQE study. The UCC ansatz is constructed from classical Coupled Cluster(CC) operators. It can be written as

$$
|\Psi(\bold{\theta})\rangle= e^{\hat{T}(\bold{\theta})-\hat{T}^\dagger(\bold{\theta})}|\Psi_0\rangle= \hat{U}(\bold{\theta})\left|\Psi_{0}\right\rangle,
$$

where $\left|\Psi_{0}\right\rangle$ is an initial state, $\hat{T}(\bold{\theta})$ is the cluster operator.
$\hat{T}(\bold{\theta})$ can be written as
$$
\hat{T}(\bold{\theta})=\sum\limits_{k}\hat{T}_{k}(\bold{\theta}),
$$ 
where k will be truncated to some order in practice.

$\hat{T}_{k}(\bold{\theta})$ are excitation operators with different orders like

$$
\hat{T}_{1}(\bold{\theta})=
\sum\limits_{i,j}\hat{t}_{ij}=
\sum\limits_{i,j}\theta_{ij}\hat{a}_i^{\dagger}\hat{a}_j
$$
$$
\hat{T}_{2}(\bold{\theta})=
\sum\limits_{i,j}\hat{t}_{ijmn}=
\sum\limits_{i,j,m,n}
\theta_{ijmn}\hat{a}_i^{\dagger}\hat{a}_m^{\dagger}\hat{a}_j \hat{a}_n
$$

and so on. Here $a$ and $a^\dagger$ are annihilation and creation operators. As a default index of UCCSD, index j,n denote to occupied orbitals, i,m denote to unoccupied orbitals. The adjustable parameter series of $\theta$ work as variables in the variational algorithm. Noticing that the form $\hat{U}(\bold{\theta})=e^{T(\theta)-T^\dagger(\theta)}$ permits the operator to be unitary. 

## 2. Settings

In this project, we truncate the operator to k = 2, and only consider exciations that preserve the singlet spin. This is called singlet-UCCSD. In singlet-UCCSD, some excitations have the same effect to the energy, like single excitation of spin up and down electrons that share the same index i,j. Thus in practice, these operators are only considered once.

## 3. Reference
Peruzzo, A., McClean, J., Shadbolt, P., Yung, M.-H. H., Zhou, X.-Q. Q., Love, P. J., Aspuru-Guzik, A., O’Brien, J. L., & O’Brien, J. L. (2014). A variational eigenvalue solver on a photonic quantum processor. Nature Communications, 5(1), 4213. https://doi.org/10.1038/ncomms5213

Taube, A. G., & Bartlett, R. J. (2006). New perspectives on unitary coupled-cluster theory. International Journal of Quantum Chemistry, 106(15), 3393–3401. https://doi.org/10.1002/qua.21198