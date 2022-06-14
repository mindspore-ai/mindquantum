# <center> k-UpCCGSD</center>

## 1. Introduction

  k-UpCCGSD is k (a chosen integer) Unitary Pair Coupled Cluster with Generalized Singles and Doubles Product Wave functions. 

## 2. Algorithm

To understand k-UpCCGSD easily, we make a brief discussion about pCCD and UpCCSD.
### 2.1 pCCD

  The method of pair coupled-cluster double excitations (pCCD) extends a widely used quantum chemistry method known as generalized valence-bond perfect-pairing  (GVB-PP). pCCD is a coupled cluster wave function with a very limited number of doubles amplitudes (containing only the two body excitations that move a pair of electrons from one spatial orbital to another)

$$
\hat{T}_{2}=\sum_{i a} t_{i_{\alpha} i_{\beta}}^{a_{a} a_{\beta}} a_{a_{\alpha}}^{\dagger} \hat{a}_{a_{\beta}}^{\dagger} \hat{a}_{i_{\beta}} \hat{a}_{i_{\alpha}}
$$

where the indexes runs over occupied and unoccupied spatial orbitals. 
Even if this method has  high computational efficiency and reduced incidence of nonvariationality, pCCD loses  invariance to unitary transformation within the occupied occupied and virtual−virtual subspaces present in CCD, and it does not recover the dynamic correlation that CCD has.

### 2.2 UpCCSD

UpCCSD (Unitary Pair Coupled Cluster with Singles and Doubles Product Wave functions) has the full singles operators together with the doubles operator.(containing the single excitations and the two body excitations that move a pair of electrons from one spatial orbital to another)
$$
	\hat{T}=\sum_{i a} t_{i}^{a} \hat{a}_{a}^{\dagger} \hat{a}_{i}+\sum_{i a} t_{i_{\alpha} i_{\beta}}^{a_{a} a_{\beta}} a_{a_{\alpha}}^{\dagger} \hat{a}_{a_{\beta}}^{\dagger} \hat{a}_{i_{\beta}} \hat{a}_{i_{\alpha}}
$$

### 2.3 k-UpCCGSD

To reduce the errors of UpCCSD in the absolute energies to the threshold for  chemical accuracy: (i)  We use use the generalized singles and doubles operators, and (ii) we take e a product of a total of k unitary operators to increase the flexibility of the wave function. We shall refer to this model as k-UpCCGSD.
$$
|\psi\rangle=\Pi_{\alpha=1}^{k}\left(e^{\hat{T}^{(\alpha)}-\hat{T}^{(\alpha) \dagger}}\right)\left|\phi_{0}\right\rangle
$$
where each $\hat{T}^{(k)}$ contains an independent set of variational parameters ($\text{i.e.}$, the singles and paired doubles amplitudes, the $t^q_p$ values and the $t_{p_{a} p_{\beta}}^{q_{a} q_{\beta}}$ values, respectively). Since the doubles operator in UpCCGSD is very sparse, the circuit depth required to prepare a k-UpCCGSD state still scales linearly with the system size, with a prefactor that is increased by a factor of k. This is similar in spirit to other recently proposed low depth ansatz.

## 3. Reference 

[1] Lee, J., Huggins, W., Head-Gordon, M., & Whaley, K. (2019). Generalized Unitary Coupled Cluster Wave functions for Quantum Computation. Journal of Chemical Theory and Computation, 15(1), 311-324. https://pubs.acs.org/doi/10.1021/acs.jctc.8b01004

