import os
import sys
import time
import itertools
import numpy as np
from scipy.optimize import minimize
from collections import OrderedDict as ordict
import matplotlib.pyplot as plt


from hiqfermion.drivers import MolecularData
from openfermionpyscf import run_pyscf

from mindquantum import Simulator, Hamiltonian, X, Circuit
from mindquantum.core.operators import FermionOperator
from mindquantum.core.operators.utils import down_index, up_index, get_fermion_operator
from mindquantum.algorithm.nisq.chem.transform import Transform
from mindquantum.third_party.interaction_operator import InteractionOperator
from mindquantum.core.circuit.utils import decompose_single_term_time_evolution
from mindquantum.core.circuit import Circuit
from mindquantum.core.operators.fermion_operator import FermionOperator
from mindquantum.core.operators.qubit_operator import QubitOperator

from mindquantum.framework import MQAnsatzOnlyLayer
import mindspore as ms
import mindspore.context as context
from mindspore.common.parameter import Parameter

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

os.environ['OMP_NUM_THREADS'] = '1'


'''
我们对构造出的anstaz是基于UCCSD，下面三条分别对原来的UCCSD做出的改变

1. 首先考虑分子轨道冻结，所有分子的能级最低的轨道对构造波函数是不起作用的，
   我们不考虑最低能级的跃迁情况。

2. 通过研究发现，算符作用到初始Hartree Fork态上的顺序对优化anstaz是有影响的，
   用过对operator先的系数的大小进行排序对其进行降序排列。

3  不仅operator的顺序对优化anstaz有影响，其系数的大小对构造anstaz有重要影响，
   之所以full UCCSD能达到化学精度，是因为其构造出的anstaz几乎能遍历整个希尔伯特空间。
   不同的operator会遍历不同的希尔伯特空间，较大系数operator构造出的anstaz所遍历的空间中是存在
   我们想要求解的目标态，所以忽略掉不需要的operator我们依然能得到目标态。
'''


def _para_uccsd_singlet_generator(mol, th, prefix, blen):

    n_qubits = mol.n_qubits
    n_electrons = mol.n_electrons
    params = {}
    if n_qubits % 2 != 0:
        raise ValueError('The total number of spin-orbitals should be even.')
    out = []
    out_tmp = []
    n_spatial_orbitals = n_qubits // 2
    n_occupied = int(np.ceil(n_electrons / 2))
    n_virtual = n_spatial_orbitals - n_occupied

    # Unpack amplitudes
    n_single_amplitudes = n_occupied * n_virtual
    # Generate excitations
    spin_index_functions = [up_index, down_index]

    # Generate all spin-conserving double excitations derived
    # from two spatial occupied-virtual pairs

    for i, (p, q) in enumerate(itertools.product(range(n_virtual), range(n_occupied))):
        
        # Get indices of spatial orbitals
        virtual_spatial = n_occupied + p
        occupied_spatial = q

        virtual_up = virtual_spatial * 2
        virtual_down = virtual_spatial * 2 + 1
        occupied_up = occupied_spatial * 2
        occupied_down = occupied_spatial * 2 + 1

        single_amps = mol.ccsd_single_amps[virtual_up, occupied_up]
        double1_amps = mol.ccsd_double_amps[virtual_up, occupied_up, virtual_down, occupied_down]

        single_amps_name = 'p' + str(i)
        double1_amps_name = 'p' + str(i+n_occupied*n_virtual)
        

        if q != 0:

            for spin in range(2):
                # Get the functions which map a spatial orbital index to a
                # spin orbital index
                this_index = spin_index_functions[spin]
                #other_index = spin_index_functions[1 - spin]

                # Get indices of spin orbitals
                virtual_this = this_index(virtual_spatial)
                #virtual_other = other_index(virtual_spatial)
                occupied_this = this_index(occupied_spatial)
                #occupied_other = other_index(occupied_spatial)

                # Generate single excitations
                if abs(single_amps) > th:
                    params[single_amps_name] = single_amps
                    fermion_ops1 = FermionOperator(
                        ((occupied_this, 1), (virtual_this, 0)), 1)
                    fermion_ops2 = FermionOperator(
                        ((virtual_this, 1), (occupied_this, 0)), 1)
                    out.append([fermion_ops2 - fermion_ops1, single_amps_name])

            # Generate double excitation
            if abs(double1_amps) > th:
                params[double1_amps_name] = double1_amps
                fermion_ops1 = FermionOperator(
                    ((occupied_up, 1), (virtual_up, 0), (occupied_down, 1),
                    (virtual_down, 0)), 1)
                fermion_ops2 = FermionOperator(
                    ((virtual_up, 1), (occupied_up, 0),
                    (virtual_down, 1), (occupied_down, 0)), 1)
            out.append([fermion_ops2 - fermion_ops1, double1_amps_name])

#     Generate all spin-conserving double excitations derived
#     from two spatial occupied-virtual pairs
    for i, ((p, q), (r, s)) in enumerate(
            itertools.combinations(
                itertools.product(range(n_virtual), range(n_occupied)), 2)):
        
        # 当LiH的键长为0.4的时候用这种方法差一点点达到化学精度，所以为了使这个键长达到化学精度，可以少筛选一点门

        if q != 0 and s != 0:
      
            virtual_spatial_1 = n_occupied + p
            occupied_spatial_1 = q
            virtual_spatial_2 = n_occupied + r
            occupied_spatial_2 = s

            virtual_1_up = virtual_spatial_1 * 2
            occupied_1_up = occupied_spatial_1 * 2
            virtual_2_up = virtual_spatial_2 * 2 + 1
            occupied_2_up = occupied_spatial_2 * 2 + 1

            double2_amps = mol.ccsd_double_amps[virtual_1_up, occupied_1_up, virtual_2_up, occupied_2_up]
            double2_amps_name = 'p' + str(i + 2 * n_single_amplitudes)

            # Generate double excitations
            for (spin_a, spin_b) in itertools.product(range(2), repeat=2):
                # Get the functions which map a spatial orbital index to a
                # spin orbital index  # 0 - 6
                index_a = spin_index_functions[spin_a]   # function 2n or 2n + 1
                index_b = spin_index_functions[spin_b]   # function 2n or 2n + 1

                # Get indices of spin orbitals
                virtual_1_a = index_a(virtual_spatial_1)
                occupied_1_a = index_a(occupied_spatial_1)
                virtual_2_b = index_b(virtual_spatial_2)
                occupied_2_b = index_b(occupied_spatial_2)

                if occupied_1_a != occupied_2_b and virtual_1_a != virtual_2_b:

                    if abs(double2_amps) > th:
                        params[double2_amps_name] = double2_amps
                        fermion_ops1 = FermionOperator(
                            ((virtual_1_a, 1), (occupied_1_a, 0),
                             (virtual_2_b, 1), (occupied_2_b, 0)), 1)
                        fermion_ops2 = FermionOperator(
                            ((occupied_2_b, 1), (virtual_2_b, 0),
                             (occupied_1_a, 1), (virtual_1_a, 0)), 1)
                        out.append([fermion_ops1 - fermion_ops2, double2_amps_name])


    '''
    通过研究发现，算符作用到初始Hartree Fork态上的顺序对优化anstaz是有影响的，用过对operator先的系数的大小进行排序
    对其进行降序排列
    '''
    parameters_order = {k:v for k, v in sorted(params.items(), key = lambda value: abs(value[1]), reverse=True)}
    
    '''
    不仅operator的顺序对优化anstaz有影响，其系数的大小对构造anstaz有重要影响，之所以full UCCSD能达到化学精度，是因为其构造出的
    anstaz几乎能遍历整个希尔伯特空间。不同的operator会遍历不同的希尔伯特空间，较大系数operator构造出的anstaz所遍历的空间中是存在
    我们想要求解的目标态，所以忽略掉不需要的operator我们依然能得到目标态
    '''

    # 当LiH的键长为0.4的时候用这种方法差一点点达到化学精度，所以为了使这个键长达到化学精度，我们少筛选一点门

    parameters_reduced_order = {}
    for key, value in parameters_order.items():
        if abs(value) > 0.001:
            parameters_reduced_order[key] = value


    fermion_ansatz_oder = []
    for order in parameters_reduced_order.keys():
        for j in range(len(out)):
            if out[j][1] == order:
                fermion_ansatz_oder.append(out[j])
    return fermion_ansatz_oder, parameters_reduced_order


def _transform2pauli(fermion_ansatz):
    """
    Transform a fermion ansatz to pauli ansatz based on jordan-wigner
    transformation.
    """
    out = ordict()
    for i in fermion_ansatz:
        qubit_generator = Transform(i[0]).jordan_wigner()
        if qubit_generator.terms != {}:
            for key, term in qubit_generator.terms.items():
                if key not in out:
                    out[key] = ordict({i[1]: float(term.imag)})
                else:
                    if i[1] in out[key]:
                        out[key][i[1]] += float(term.imag)
                    else:
                        out[key][i[1]] = float(term.imag)
    return out


def _pauli2circuit(pauli_ansatz):
    """Transform a pauli ansatz to parameterized quantum circuit."""
    circuit = Circuit()
    for k, v in pauli_ansatz.items():
        circuit += decompose_single_term_time_evolution(k, v)
    return circuit

def generate_uccsd(molecular, th, prefix, blen):

    if isinstance(molecular, str):
        mol = MolecularData(filename=molecular)
        mol.load()
    else:
        mol = molecular
    print("ccsd:{}.".format(mol.ccsd_energy))
    print("fci:{}.".format(mol.fci_energy))
    fermion_ansatz, parameters = _para_uccsd_singlet_generator(mol, th, prefix, blen)
    # print(len(parameters))
    pauli_ansatz = _transform2pauli(fermion_ansatz)
    uccsd_circuit = _pauli2circuit(pauli_ansatz)
    ham_of = mol.get_molecular_hamiltonian()
    inter_ops = InteractionOperator(*ham_of.n_body_tensors.values())
    ham_hiq = get_fermion_operator(inter_ops)
    qubit_hamiltonian = Transform(ham_hiq).jordan_wigner()
    qubit_hamiltonian.compress()

    parameters_name = list(parameters.keys())
    initial_amplitudes = [parameters[i] for i in parameters_name]
    return uccsd_circuit, \
        initial_amplitudes, \
        parameters_name, \
        qubit_hamiltonian,\
        mol.n_qubits,\
        mol.n_electrons