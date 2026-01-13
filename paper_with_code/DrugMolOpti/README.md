# DMET-VQE-MGO
### *Large-scale Efficient Molecule Geometry Optimization with Hybrid Quantum-Classical Computing*

This repository provides an efficient and scalable framework for **Molecule Geometry Optimization (MGO)** using a hybrid **DMET + VQE** scheme.  
Our method integrates classical electronic-structure embedding (DMET) with quantum-variational solvers (VQE), enabling geometry optimization for molecular systems beyond the reach of purely classical approaches.
Geometry files with the suffix _ref correspond to classical reference structures, whereas files with the suffix _our_work correspond to geometries optimized by the DMET–VQE method.

---

## Example

The main implementation of the geometry optimization workflow can be found in:

- **`main/geo_fun.py`** — core functions for DMET-VQE molecular geometry optimization  
- **`main/H4.py`** — example script demonstrating the geometry optimization of the $\mathrm{H_4}$ molecule  
- **`mol.py`** — definitions of molecular structures and helper functions for building geometries ($\mathrm{H_2O_2}$ and $\mathrm{C_2H_4O_3}$)


---

## Dependencies

This project relies on the following library:

* **PySCF:** `>= 2.8`
* **OpenFermion:** `>= 1.7.0`
* **OpenFermion-PySCF:** `>= 0.5`
* **h5py:** `>= 3.13.0`
* **MindQuantum:** `>= 0.10.0`
* **Tangelo-GC:** `>= 0.4.3`
