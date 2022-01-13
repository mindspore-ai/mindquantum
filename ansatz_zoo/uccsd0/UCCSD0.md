# <center>singlet-paired Unitary Coupled Cluster (UCCSD0)</center>

## 1. Introduction
Singlet-paired UCC (UCCSD0) will generate ansatz operators using CCD0 ansatz. It is similar to the UCC ansatz: the single excitation operators in UCCSD0 is the same with UCC ansatz, but the second excitation operators is a little different. The parameters ahead of second excitation operators may be dependent like
$$
(\theta_{1}-\theta_{2})\hat{a}_i^{\dagger}\hat{a}_m^{\dagger}\hat{a}_j \hat{a}_n,
$$
where different terms may be affected by same parameters, but the types of second excitation operators is the same with operators in UCCSD ansatz.

Also refer to [UCCSD](UCCSD.md) ansatz.

## 2. Settings
UCCSD0 ansatz used in the benchmark will use the standard way introduced in the reference articles to generate the ansatz operator.


## 3. Reference
Igor O. Sokolov, Panagiotis Kl. Barkoutsos, Pauline J. Ollitrault, Donny Greenberg, Julia Rice, Marco Pistoia, and Ivano Tavernelli (2020). "Quantum orbital-optimized unitary coupled cluster methods in the strongly correlated regime: Can quantum algorithms outperform their classical equivalents?", The Journal of Chemical Physics 152, 124107 
https://doi.org/10.1063/1.5141835

Bulik, I. W., Henderson, T. M., & Scuseria, G. E. (2015). Can Single-Reference Coupled Cluster Theory Describe Static Correlation? Journal of Chemical Theory and Computation, 11(7), 3171–3179. https://doi.org/10.1021/ACS.JCTC.5B00422