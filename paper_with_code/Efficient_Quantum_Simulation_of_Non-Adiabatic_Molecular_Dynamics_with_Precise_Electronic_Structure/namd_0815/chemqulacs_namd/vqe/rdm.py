# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from importlib.metadata import version
from itertools import product

import numpy as np
from openfermion.ops import FermionOperator


def get_1rdm(state, fermion_qubit_mapping, estimator, n_electrons, sz=0):
    """
    Compute 1-RDM of a given state with user specfied fermion to qubit mapping.

    Args:
        state (quri_parts.GeneralCircuitQuantumState)
        fermion_qubit_mapping (quri_parts.openfermion.transforms.OpenFermionQubitMapping)
        estimator (quri_parts.qulacs.estimator.ConcurrentQuantumEstimator)
        n_electrons (int)
        n_spin_orbitals (int):
            number of spin-orbitals. if None, it is set to # of qubits in QuantumState

    Return:
        numpy.ndarray of shape (n_spin_orbitals, n_spin_orbitals):
            1-RDM
    """
    _n_spin_orbitals = fermion_qubit_mapping.n_spin_orbitals(state.qubit_count)
    if version("quri-parts-openfermion") >= "0.19.0":
        op_mapper = fermion_qubit_mapping.get_of_operator_mapper(
            n_spin_orbitals=_n_spin_orbitals, n_fermions=n_electrons, sz=sz
        )
    else:
        op_mapper = fermion_qubit_mapping.get_of_operator_mapper(
            n_spin_orbitals=_n_spin_orbitals, n_fermions=n_electrons
        )
    ret = np.zeros((_n_spin_orbitals, _n_spin_orbitals), dtype=np.complex128)
    ops_est = []
    for i in range(_n_spin_orbitals):
        for j in range(i + 1):
            one_body_op = FermionOperator(((i, 1), (j, 0)))
            ops_est.append(op_mapper(one_body_op))

    estimates = iter(estimator(ops_est, [state]))






    for i in range(_n_spin_orbitals):
        for j in range(i + 1):
            tmp = next(estimates).value
            ret[i, j] = tmp
            ret[j, i] = tmp.conjugate()
    return ret


def get_2rdm(state, fermion_qubit_mapping, estimator, n_electrons, sz=0):
    """
    Compute 2-RDM of a given state with user specfied fermion to qubit mapping.

    Args:
        state (quri_parts.GeneralCircuitQuantumState):
        fermion_qubit_mapping (quri_parts.openfermion.transforms.OpenFermionQubitMapping)
        estimator (quri_parts.qulacs.estimator.ConcurrentQuantumEstimator)
        n_electrons (int)
        n_spin_orbitals (int):
            number of spin-orbitals. if None, it is set to # of qubits in QuantumState
    Return:
        numpy.ndarray of shape (n_spin_orbitals, n_spin_orbitals, n_spin_orbitals, n_spin_orbitals):
            2-RDM
    """
    _n_spin_orbitals = fermion_qubit_mapping.n_spin_orbitals(state.qubit_count)
    if version("quri-parts-openfermion") >= "0.19.0":
        op_mapper = fermion_qubit_mapping.get_of_operator_mapper(
            n_spin_orbitals=_n_spin_orbitals, n_fermions=n_electrons, sz=sz
        )
    else:
        op_mapper = fermion_qubit_mapping.get_of_operator_mapper(
            n_spin_orbitals=_n_spin_orbitals, n_fermions=n_electrons
        )
    ret = np.zeros(
        (_n_spin_orbitals, _n_spin_orbitals, _n_spin_orbitals, _n_spin_orbitals),
        dtype=np.complex128,
    )
    ops_est = []
    for i, k in product(range(_n_spin_orbitals), range(_n_spin_orbitals)):
        for j, l in product(range(i), range(k)):
            two_body_op = FermionOperator(((i, 1), (j, 1), (k, 0), (l, 0)))
            ops_est.append(op_mapper(two_body_op))

    from time import time
    start=time()

    estimates = iter(estimator(ops_est, [state]))
    end=time()
    print("Time taken for 2rdm calculation:",str(end-start))


    # estimates = iter(estimator(ops_est, [state]))

    for i, k in product(range(_n_spin_orbitals), range(_n_spin_orbitals)):
        for j, l in product(range(i), range(k)):
            two_body_op = FermionOperator(((i, 1), (j, 1), (k, 0), (l, 0)))
            two_body_op = op_mapper(two_body_op)
            tmp = next(estimates).value
            ret[i, j, k, l] = tmp
            ret[i, j, l, k] = -tmp
            ret[j, i, l, k] = tmp
            ret[j, i, k, l] = -tmp
            ret[l, k, j, i] = tmp.conjugate()
            ret[k, l, j, i] = -tmp.conjugate()
            ret[k, l, i, j] = tmp.conjugate()
            ret[l, k, i, j] = -tmp.conjugate()

    return ret
