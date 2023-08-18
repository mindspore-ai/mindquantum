from mindquantum.core.operators import FermionOperator
from mindquantum.algorithm.nisq.chem import Transform, QubitUCCAnsatz
from mindquantum.core.operators import QubitExcitationOperator

from mindquantum.algorithm.nisq.chem import Transform
from mindquantum.algorithm.nisq.chem import get_qubit_hamiltonian
from mindquantum.algorithm.nisq.chem import uccsd_singlet_generator, uccsd_singlet_get_packed_amplitudes
from mindquantum.core.operators import TimeEvolution
from mindquantum.framework import MQAnsatzOnlyLayer
from mindquantum.core import Circuit, X, Y, Z, CNOT, RX, RY, RZ






import itertools

def Q_ij_to_circ(i, j, theta, coeff=1):
    # (i j^)
    circ = Circuit()
    circ += CNOT(i, j)
    circ += RY({theta: coeff}).on(j, i)
    circ += CNOT(i, j)
    return circ


def Q_ijkl_to_circ(i, j, k, l, theta, coeff=1):
    # i j k^ l^
    circ = Circuit()
    circ += CNOT.on(k, l)
    circ += CNOT.on(i, j)
    circ += CNOT.on(j, l)
    circ += X.on(k)
    circ += X.on(i)
    circ += RY({theta: coeff}).on(l, ctrl_qubits=[i, j, k])
    circ += X.on(k)
    circ += X.on(i)
    circ += CNOT.on(j, l)
    circ += CNOT.on(i, j)
    circ += CNOT.on(k, l)
    return circ



def F_ij_to_circ(i, j, theta, coeff=1):
    # (i j^)
    # make sure i < j
    # prepare for computing parity
    if i > j:
        ip, jp = j, i
    else:
        ip, jp = i, j
    if jp - ip > 1:
        stair = True
    else:
        stair = False
    circ = Circuit()
    # compute the parity of q_{i+1} ... q_{k-1}
    for x in range(ip + 1, jp - 1):
        circ += CNOT(x + 1, x)
        
    ##############
    circ += CNOT(i, j)
    if stair:
        circ += Z.on(j, jp - 1)
    circ += RY({theta: coeff}).on(j, i)
    if stair:
        circ += Z.on(j, jp - 1)
    circ += CNOT(i, j)
    ##############
    
    # compute the parity of q_{i+1} ... q_{k-1}
    for x in range(jp - 2, ip, -1):
        circ += CNOT(x + 1 , x)    
    return circ



def F_ijkl_to_circ(i, j, k, l, theta, coeff=1):
    # i j k^ l^    
    circ = Circuit()
    # prepare for computing parity
    if i > j:
        ip, jp = j, i
    else:
        ip, jp = i, j
    if k > l:
        kp, lp = l, k
    else:
        kp, lp = k, l
    # compute the parity of q_{i+1} ... q_{j-1} q_{k+1} ... q_{l-1}
    stair_coo = []
    stair = False
    if jp - ip > 1:
        stair = True
        stair_coo += [x for x in range(ip + 1, jp)]
    if lp - kp > 1:
        stair = True
        stair_coo += [x for x in range(kp + 1, lp)]
        
    if stair == True:
        for x in range(len(stair_coo) - 1):
            circ += CNOT(stair_coo[x + 1], stair_coo[x])
        y = stair_coo[-1]
    ##########################
    circ += CNOT.on(k, l)
    circ += CNOT.on(i, j)
    circ += CNOT.on(j, l)
    circ += X.on(k)
    circ += X.on(i)
    if stair == True:
        circ += Z.on(l, y)
    circ += RY({theta: coeff}).on(l, ctrl_qubits=[i, j, k])
    if stair == True:
        circ += Z.on(l, y)
    circ += X.on(k)
    circ += X.on(i)
    circ += CNOT.on(j, l)
    circ += CNOT.on(i, j)
    circ += CNOT.on(k, l)
    ##########################
    # compute the parity of q_{k+1} ... q_{l-1}
    if stair == True:
        for x in range(len(stair_coo) - 1)[::-1]:
            circ += CNOT(stair_coo[x + 1], stair_coo[x])

    return circ


import numpy as np




def gen_uccsd_circuit_from_F_ijkl(n_qubit, n_electron):
    # not Q, but F
    # simply replace FermionOper with Q_ij and Q_ijkl, regardless of parity
    # efficient circuit to implement Q_ij and Q_ijkl
    #f = QubitExcitationOperator()
    circ = Circuit()
    occupied = (n_electron + 1) // 2 #    2e -> 1, 3e -> 2
    ground = range(0, 2 * occupied, 2) #, dtype=int)
    excited = range(2 * occupied, n_qubit, 2) #, dtype=int)
    #print(ground, excited)
    
    # single excitation
    for i in ground:
        for j in excited:
            #print(i, j, type(i))
            i, j = int(i), int(j)
            #f += QubitExcitationOperator(f'{i}^ {j}', f's_{i}_{j}')
            #f -= QubitExcitationOperator(f'{j}^ {i}', f's_{i}_{j}')
            circ += F_ij_to_circ(j, i, f's_{i}_{j}', 1)
            circ += F_ij_to_circ(i, j, f's_{i}_{j}', -1)
            # 系数和上面相同
            #f += QubitExcitationOperator(f'{i+1}^ {j+1}', f's_{i}_{j}')
            #f -= QubitExcitationOperator(f'{j+1}^ {i+1}', f's_{i}_{j}')
            circ += F_ij_to_circ(j+1, i+1, f's_{i}_{j}', 1)
            circ += F_ij_to_circ(i+1, j+1, f's_{i}_{j}', -1)
            
       
    
    # double excitation 重写
    ground_even = range(0, 2 * occupied, 2) #, dtype=int)
    ground_odd =  range(1, 2 * occupied , 2) #, dtype=int)
    excited_even = range(2 * occupied    , n_qubit, 2) #, dtype=int)
    excited_odd =  range(2 * occupied + 1, n_qubit, 2) #, dtype=int)
    #print(ground_even, ground_odd, excited_even, excited_odd)
    # i odd, j even
    for i_ in ground_odd:
        for j_ in ground_even:
            if i_ > j_ :
                i, j = j_, i_
            else:
                i, j = i_, j_
            for k_e_ in excited_odd:
                for l_e_ in excited_even:
                    #if k_e_ > l_e_: #大小顺序也很重要， # 奇偶性问题，0^ 4 2^ 6 不等于 0^ 6 2^ 4 只会发生在同样是偶数/奇数的情况？倒也不是。
                    #    k_e, l_e = l_e_, k_e_
                    #else:
                    #    k_e, l_e = k_e_, l_e_
                    # 系数也要去重复，因为 0^2 和 1^3 的系数是相同的 (0 和 1 轨道相同自旋不同)
                    i, j, k_e, l_e = i_, j_, k_e_, l_e_
                    #f += QubitExcitationOperator(f"{i}^ {k_e} {j}^ {l_e}", f'd_{i//2}_{j//2}_{k_e//2}_{l_e//2}')
                    #f -= QubitExcitationOperator(f"{k_e}^ {i} {l_e}^ {j}", f'd_{i//2}_{j//2}_{k_e//2}_{l_e//2}')
                    # FermionOper 对 k，l 为 6，8 和 8，6 两种情况是不同的，但是 Q_ijkl 或许是相同的？先不改，测试一下
                    circ += F_ijkl_to_circ(k_e, l_e, i, j, f'd_{i//2}_{j//2}_{k_e//2}_{l_e//2}', 1)
                    circ += F_ijkl_to_circ(i, j, k_e, l_e, f'd_{i//2}_{j//2}_{k_e//2}_{l_e//2}', -1)
    # i, j both even
    for i_ in ground_even:
        for j_ in ground_even:
            if i_ < j_:
                for k_e_ in excited_even:
                    for l_e_ in excited_even:
                        #if k_e_ < l_e_:
                        if k_e_ != l_e_:
                        #if True:
                        
                            #print(i_, j_, k_e_, l_e_)
                            #i, j, k_e, l_e = int(i_), int(j_), int(k_e_), int(l_e_)
                            #print(i, j, k_e, l_e)
                            i, j, k_e, l_e = i_, j_, k_e_, l_e_
                            #f += QubitExcitationOperator(f"{i}^ {k_e} {j}^ {l_e}", f'd_{i//2}_{j//2}_{k_e//2}_{l_e//2}')
                            #f -= QubitExcitationOperator(f"{k_e}^ {i} {l_e}^ {j}", f'd_{i//2}_{j//2}_{k_e//2}_{l_e//2}')
                            circ += F_ijkl_to_circ(k_e, l_e, i, j, f'd_{i//2}_{j//2}_{k_e//2}_{l_e//2}', 1)
                            circ += F_ijkl_to_circ(i, j, k_e, l_e, f'd_{i//2}_{j//2}_{k_e//2}_{l_e//2}', -1)
                            # i, j both odd
                            i, j, k_e, l_e = i_ + 1, j_ + 1, k_e_ + 1, l_e_ + 1
                            #f += QubitExcitationOperator(f"{i}^ {k_e} {j}^ {l_e}", f'd_{i//2}_{j//2}_{k_e//2}_{l_e//2}')
                            #f -= QubitExcitationOperator(f"{k_e}^ {i} {l_e}^ {j}", f'd_{i//2}_{j//2}_{k_e//2}_{l_e//2}')
                            circ += F_ijkl_to_circ(k_e, l_e, i, j, f'd_{i//2}_{j//2}_{k_e//2}_{l_e//2}', 1)
                            circ += F_ijkl_to_circ(i, j, k_e, l_e, f'd_{i//2}_{j//2}_{k_e//2}_{l_e//2}', -1)


    return circ







def gen_qucc_ladder_ops(n_qubit, n_electron):
    # number of occupied and virtual orbits
    occ = int( (n_electron + 1) // 2 * 2)
    vir = n_qubit - occ
    q = QubitExcitationOperator()
    
    # single excitation
    for i in range(occ):
        for j in range(occ, n_qubit):
            q += QubitExcitationOperator(((j, 1), (i, 0)), f's_{i}_{j}')
            q -= QubitExcitationOperator(((i, 1), (j, 0)), f's_{i}_{j}')
            
    # double excitations
    for (i, j) in itertools.combinations(range(occ), 2):
        for (k, l) in itertools.combinations(range(occ, n_qubit), 2):
            q += QubitExcitationOperator(((k, 1), (l, 1), (i, 0), (j, 0)), f'd_{i}_{j}_{k}_{l}')
            q -= QubitExcitationOperator(((i, 1), (j, 1), (k, 0), (l, 0)), f'd_{i}_{j}_{k}_{l}')
    return q
            



def gen_qucc_excite_circuit(n_qubit, n_electron):
    # number of occupied and virtual orbits
    occ = int( (n_electron + 1) // 2 * 2)
    vir = n_qubit - occ
    
    circ = Circuit()
    # single excitation
    for i in range(occ):
        for j in range(occ, n_qubit):
            #q += QubitExcitationOperator(((j, 1), (i, 0)), f's_{i}_{j}')
            circ += Q_ij_to_circ(i, j, f's_{i}_{j}', 1)
            #q -= QubitExcitationOperator(((i, 1), (j, 0)), f's_{i}_{j}')
            #circ += Q_ij_to_circ(j, i, f's_{i}_{j}', -1)
            
    # double excitations
    for (i, j) in itertools.combinations(range(occ), 2):
        for (k, l) in itertools.combinations(range(occ, n_qubit), 2):
            #q += QubitExcitationOperator(((k, 1), (l, 1), (i, 0), (j, 0)), f'd_{i}_{j}_{k}_{l}')
            circ += Q_ijkl_to_circ(i, j, k, l, f'd_{i}_{j}_{k}_{l}', 1)
            #q -= QubitExcitationOperator(((i, 1), (j, 1), (k, 0), (l, 0)), f'd_{i}_{j}_{k}_{l}')
            #circ += Q_ijkl_to_circ(k, l, i, j, f'd_{i}_{j}_{k}_{l}', -1)
    return circ


    
def gen_qucc_circuit(n_qubit, n_electron):
    return gen_qucc_excite_circuit(n_qubit, n_electron)







def gen_uccsd_singlet(n_qubit, n_electron):
    # generate a uccsd fermion operator 
    # to be debugged
    f = FermionOperator()
    occupied = (n_electron + 1) // 2 #    2e -> 1, 3e -> 2
    ground = np.arange(0, 2 * occupied, 2)
    excited = np.arange(2 * occupied, n_qubit, 2)
    print(ground, excited)
    
    # single excitation
    for i in ground:
        for j in excited:
            print(i, j)
            f += FermionOperator(f'{i}^ {j}', f's_{i}_{j}')
            f -= FermionOperator(f'{j}^ {i}', f's_{i}_{j}')
            # 系数和上面相同
            f += FermionOperator(f'{i+1}^ {j+1}', f's_{i}_{j}')
            f -= FermionOperator(f'{j+1}^ {i+1}', f's_{i}_{j}')
            
    
    # double excitation 
    ground_even = np.arange(0, 2 * occupied, 2)
    ground_odd =  np.arange(1, 2 * occupied , 2)
    excited_even = np.arange(2 * occupied    , n_qubit, 2)
    excited_odd =  np.arange(2 * occupied + 1, n_qubit, 2)
    print(ground_even, ground_odd, excited_even, excited_odd)
    # i odd, j even
    for i_ in ground_odd:
        for j_ in ground_even:
            if i_ > j_ :
                i, j = j_, i_
            else:
                i, j = i_, j_
            for k_e_ in excited_odd:
                for l_e_ in excited_even:
                    #if k_e_ > l_e_: #大小顺序也很重要， # 奇偶性问题，0^ 4 2^ 6 不等于 0^ 6 2^ 4 
                    #    k_e, l_e = l_e_, k_e_
                    #else:
                    #    k_e, l_e = k_e_, l_e_
                    # 系数也要去重复，因为 0^2 和 1^3 的系数是相同的 (0 和 1 轨道相同自旋不同)
                    i, j, k_e, l_e = i_, j_, k_e_, l_e_
                    f += FermionOperator(f"{i}^ {k_e} {j}^ {l_e}", f'd_{i//2}_{j//2}_{k_e//2}_{l_e//2}')
                    f -= FermionOperator(f"{k_e}^ {i} {l_e}^ {j}", f'd_{i//2}_{j//2}_{k_e//2}_{l_e//2}')
    # i, j both even
    for i_ in ground_even:
        for j_ in ground_even:
            if i_ < j_:
                for k_e_ in excited_even:
                    for l_e_ in excited_even:
                        #if k_e_ < l_e_:
                        if k_e_ != l_e_:
                        #if True:
                            i, j, k_e, l_e = i_, j_, k_e_, l_e_
                            f += FermionOperator(f"{i}^ {k_e} {j}^ {l_e}", f'd_{i//2}_{j//2}_{k_e//2}_{l_e//2}')
                            f -= FermionOperator(f"{k_e}^ {i} {l_e}^ {j}", f'd_{i//2}_{j//2}_{k_e//2}_{l_e//2}')
                            # i, j both odd
                            i, j, k_e, l_e = i_ + 1, j_ + 1, k_e_ + 1, l_e_ + 1
                            f += FermionOperator(f"{i}^ {k_e} {j}^ {l_e}", f'd_{i//2}_{j//2}_{k_e//2}_{l_e//2}')
                            f -= FermionOperator(f"{k_e}^ {i} {l_e}^ {j}", f'd_{i//2}_{j//2}_{k_e//2}_{l_e//2}')



    return f







