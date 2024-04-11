from mindquantum import *
from mindquantum.core.circuit import Circuit
import numpy as np
import random
from mindquantum.core.circuit import AP, A, Circuit, add_prefix, add_suffix
from mindquantum.core.gates import BasicGate, X
from mindquantum.utils.type_value_check import (
    _check_input_type,
    _check_int_type,
    _check_value_should_not_less,
)

from mindquantum.algorithm.nisq._ansatz import Ansatz

"""                 Implement SG ansatz.              
        The SG ansatz consists of multiple variational quantum circuit blocks, each of which
        is a parametrized quantum circuit applied to several adjacent qubits. 
        With such a structure, the SG ansatz naturally adapts to quantum many-body problems. 
        Specifically, for 1D quantum systems, the SG ansatz can efficiently generate any matrix
        product states with a fixed bond dimension.For 2D systems, the SG ansatz can generate string-bond states.
        For more detail, please refers `A sequentially generated variational quantumcircuit with polynomial complexity 
        <https://arxiv.org/abs/2305.12856>`_.
        
"""



class SGAnsatz(Ansatz):
    """
        SG ansatz for 1D quantum systems

        Args:
            nqubits (int): Number of qubits in the ansatz.
            k (int): log(R)+1, where R is the bond dimension of a MPS state
            nlayers (int): Number of layers in each block. Default: ``1``
            prefix (str): The prefix of parameters. Default: ``''``.
            suffix (str): The suffix of parameters. Default: ``''``.  
        Examples:
            >>> from mindquantum.core.gates import RY, RZ, Z
            >>> from (……).SG_Ansatz import SGAnsatz, SGAnsatz_2D
            >>> SG=SGAnsatz(4,3,1)
            >>> SG.circuit
            q0: ──RY(a1_00)─────────●────────RX(b1_000)────────●───────────────────────────────────────────────────────────────
                                    │                          │
            q1: ──RZ(a1_01)─────RY(a2_00)────RZ(b1_001)────RX(b2_000)────────●─────────RZ(b1_101)────────●─────────────────────
                                                                            │                           │
            q2: ──RX(b1_002)─────────────────────────────────────────────RZ(b2_001)────RZ(b1_102)────RY(b2_101)────────●───────
                                                                                                                    │
            q3: ──RY(b1_103)───────────────────────────────────────────────────────────────────────────────────────RY(b2_102)── 

    """
        # pylint: disable=too-many-arguments
    def __init__(
        self,
        nqubits,
        k,
        nlayers=1,
        prefix: str = '',
        suffix: str = '',
    ):
        """Initialize a SGAnsatz object."""
        _check_int_type('nlayers', nlayers)
        _check_int_type('k', k)
        _check_value_should_not_less('nlayers', 1, nlayers)
        _check_input_type('prefix', str, prefix)
        _check_input_type('suffix', str, suffix)
        self.prefix = prefix
        self.suffix = suffix
        self.nlayers = nlayers
        self.nqubits = nqubits
        self.k = k 
        super().__init__('SGAnsatz', nqubits, nlayers, k)



    def _implement(self, nlayers, k):
        """Implement of SG ansatz."""
        circ = Circuit()
        self._circuit = self.random_block()
        if self.prefix:
            self._circuit = add_prefix(self._circuit, self.prefix)
        if self.suffix:
            self._circuit = add_suffix(self._circuit, self.suffix)

    

    def random_block(self):
        if not len(self._circuit)==0:
            raise TypeError(f"There already exists a circuit ansatz!")
        # 0:X 1:Y, 2:Z
        rand_list=[0,1,2]

        # Construt the 1-st block on k-1 sites
        for j in range(self.nlayers):
            # firstly apply H gate on each sites
            for i in range(self.k-1):
                flg=random.choice(rand_list)
                if flg == 0:
                    self._circuit += RX('a1_{}{}'.format(j,i)).on(i)
                elif flg == 1:
                    self._circuit += RY('a1_{}{}'.format(j,i)).on(i)
                else:
                    self._circuit += RZ('a1_{}{}'.format(j,i)).on(i)
            
            if self.k!= 2:
                for i in range(self.k-2):
                    flg=random.choice(rand_list)
                    if flg == 0:
                        self._circuit += RX('a2_{}{}'.format(j,i)).on(i+1,i)
                    elif flg == 1:
                        self._circuit += RY('a2_{}{}'.format(j,i)).on(i+1,i)
                    else:
                        self._circuit += RZ('a2_{}{}'.format(j,i)).on(i+1,i)

        # Construct the N-k+1 (k-1)-local block
        for d in range(self.nqubits-self.k+1):
            for j in range(self.nlayers):
                for i in range(d,d+self.k):
                    flg=random.choice(rand_list)
                    if flg == 0:
                        self._circuit += RX('b1_{}{}{}'.format(d,j,i)).on(i)
                    elif flg == 1:
                        self._circuit += RY('b1_{}{}{}'.format(d,j,i)).on(i)
                    else:
                        self._circuit += RZ('b1_{}{}{}'.format(d,j,i)).on(i)

                for i in range(d,d+self.k-1):
                    flg=random.choice(rand_list)
                    if flg == 0:
                        self._circuit += RX('b2_{}{}{}'.format(d,j,i)).on(i+1,i)
                    elif flg == 1:
                        self._circuit += RY('b2_{}{}{}'.format(d,j,i)).on(i+1,i)
                    else:
                        self._circuit += RZ('b2_{}{}{}'.format(d,j,i)).on(i+1,i)

        return self._circuit

class SGAnsatz_2D(Ansatz):
    """
        SG ansatz for 2D quantum systems.

        Args:
            nqubits (int): Number of qubits in the ansatz.
            k (int): log(R)+1, where R is the fixed bond dimension
            line_set: A list set of qubits' lines to generate a specific type of string-bond state.
                        Default: ``[[0,1,2,...,(nqubits)]]``
            nlayers (int): Number of layers in each block. Default: ``1``
            prefix (str): The prefix of parameters. Default: ``''``.
            suffix (str): The suffix of parameters. Default: ``''``.

        Examples:
            >>> from mindquantum.core.gates import RY, RZ, Z
            >>> from (……).SG_Ansatz import SGAnsatz, SGAnsatz_2D
            >>> SG=SGAnsatz_2D(4,2)
            >>> SG.circuit
            q0: ───RY(a1_000)────RY(b1_0000)─────────●───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
                                                    │
            q1: ──RX(b1_0001)───────────────────RZ(b2_0000)────RX(b1_0101)─────────●─────────────────────────────────────────────────────────────────────────────────────────────────
                                                                                │
            q2: ──RZ(b1_0102)─────────────────────────────────────────────────RZ(b2_0101)────RX(b1_0202)─────────●───────────────────────────────────────────────────────────────────
                                                                                                                │
            q3: ──RX(b1_0203)───────────────────────────────────────────────────────────────────────────────RY(b2_0202)────RX(b1_0303)─────────●─────────────────────────────────────
                                                                                                                                            │
            q4: ──RY(b1_0304)─────────────────────────────────────────────────────────────────────────────────────────────────────────────RZ(b2_0303)────RY(b1_0404)─────────●───────
                                                                                                                                                                            │
            q5: ──RZ(b1_0405)───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────RX(b2_0404)──
            >>>from (……).SG_Ansatz import generate_line_set
            >>>line_set=generate_line_set(2,3)
            >>> SG=SGAnsatz_2D(6, 2, line_set)
            >>> SG.circuit
            q0: ───RZ(a1_000)────RX(b1_0000)─────────●──────────RX(a1_100)────RY(b1_1000)────────────────────────────────────────────────────────────────────────────────────●─────────────────────────────────────>>
                                                    │                                                                                                                       │                                     >>
            q1: ─────────────────────────────────────┼─────────RY(b1_0203)──────────────────────────────────RX(b2_0202)────RX(b1_0303)─────────●─────────RX(b1_1001)────RX(b2_1000)────RY(b1_1101)─────────●───────>>
                                                    │                                                           │                             │                                                           │       >>
            q2: ─────────────────────────────────────┼───────────────────────────────────────────────────────────┼─────────RX(b1_0304)────RZ(b2_0303)────RY(b1_0404)─────────●─────────RX(b1_1102)────RY(b2_1101)──>>
                                                    │                                                           │                                                           │                                     >>
            q3: ──RX(b1_0001)───────────────────RZ(b2_0000)────RZ(b1_0101)─────────●─────────────────────────────┼───────────────────────────────────────────────────────────┼─────────────────────────────────────>>
                                                                                │                             │                                                           │                                     >>
            q4: ──RZ(b1_0102)─────────────────────────────────────────────────RY(b2_0101)────RZ(b1_0202)─────────●───────────────────────────────────────────────────────────┼─────────────────────────────────────>>
                                                                                                                                                                            │                                     >>
            q5: ──RZ(b1_0405)───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────RZ(b2_0404)────RX(b1_1203)─────────────────>>
            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            q0: <<──────────────────────────────────────────────────────────────────────────────────────────
                <<
            q1: <<──────────────────────────────────────────────────────────────────────────────────────────
                <<
            q2: <<──RX(b1_1202)─────────●───────────────────────────────────────────────────────────────────
                <<                      │
            q3: <<──────────────────────┼─────────RZ(b1_1405)──────────────────────────────────RX(b2_1404)──
                <<                      │                                                           │
            q4: <<──────────────────────┼─────────RX(b1_1304)────RY(b2_1303)────RZ(b1_1404)─────────●───────
                <<                      │                             │
            q5: <<─────────────────RY(b2_1202)────RZ(b1_1303)─────────●─────────────────────────────────────

    """
    def __init__(
        self,
        nqubits,
        k,
        line_set=None,
        nlayers=1,
        prefix: str = '',
        suffix: str = '',
    ):
        """Initialize a SGAnsatz_2D object."""
        _check_int_type('nlayers', nlayers)
        _check_int_type('nqubits', nqubits)
        _check_int_type('k', k)
        _check_value_should_not_less('nlayers', 1, nlayers)
        _check_input_type('prefix', str, prefix)
        _check_input_type('suffix', str, suffix)
        self.prefix = prefix
        self.suffix = suffix
        self.nlayers = nlayers
        self.k = k 
        if line_set is None:
            line_set = self.generate_line_set(nqubits)

        self.line_set = line_set
        self.nqubits = nqubits

        super().__init__('SGAnsatz_2D', nqubits, nlayers, k, line_set)

    def _implement(self, nlayers, k, line_set):
        """Implement of SG ansatz."""
        circ = Circuit()
        for chain_idx in range(len(self.line_set)):
            self._circuit = self.random_block(chain_idx,self.line_set[chain_idx])

        if self.prefix:
            self._circuit = add_prefix(self._circuit, self.prefix)
        if self.suffix:
            self._circuit = add_suffix(self._circuit, self.suffix)


    def generate_line_set(self, nqubits):
        line = list(range(nqubits))  
        line_set = [line] 
        return line_set



    def random_block(self,idx,chain):
        rand_list=[0,1,2]

        # Construt the 1-st block on k-1 sites
        for j in range(self.nlayers):
            # firstly apply H gate on each sites
            for i in range(self.k-1):
                flg=random.choice(rand_list)
                if flg == 0:
                    self._circuit += RX('a1_{}{}{}'.format(idx,j,i)).on(chain[i])
                elif flg == 1:
                    self._circuit += RY('a1_{}{}{}'.format(idx,j,i)).on(chain[i])
                else:
                    self._circuit += RZ('a1_{}{}{}'.format(idx,j,i)).on(chain[i])
            
            if self.k!= 2:
                for i in range(self.k-2):
                    flg=random.choice(rand_list)
                    if flg == 0:
                        self._circuit += RX('a2_{}{}{}'.format(idx,j,i)).on(chain[i+1],chain[i])
                    elif flg == 1:
                        self._circuit += RY('a2_{}{}{}'.format(idx,j,i)).on(chain[i+1],chain[i])
                    else:
                        self._circuit += RZ('a2_{}{}{}'.format(idx,j,i)).on(chain[i+1],chain[i])

        # Construct the N-k+1 (k-1)-local block
        for d in range(len(chain)-self.k+1):
            for j in range(self.nlayers):
                for i in range(d,d+self.k):
                    flg=random.choice(rand_list)
                    if flg == 0:
                        self._circuit += RX('b1_{}{}{}{}'.format(idx,d,j,i)).on(chain[i])
                    elif flg == 1:
                        self._circuit += RY('b1_{}{}{}{}'.format(idx,d,j,i)).on(chain[i])
                    else:
                        self._circuit += RZ('b1_{}{}{}{}'.format(idx,d,j,i)).on(chain[i])

                for i in range(d,d+self.k-1):
                    flg=random.choice(rand_list)
                    if flg == 0:
                        self._circuit += RX('b2_{}{}{}{}'.format(idx,d,j,i)).on(chain[i+1],chain[i])
                    elif flg == 1:
                        self._circuit += RY('b2_{}{}{}{}'.format(idx,d,j,i)).on(chain[i+1],chain[i])
                    else:
                        self._circuit += RZ('b2_{}{}{}{}'.format(idx,d,j,i)).on(chain[i+1],chain[i])

        return self._circuit


def generate_line_set(nrow,ncol):
    nqubits=nrow*ncol
    a = np.arange(0, nqubits, 1)
    b=a.reshape(nrow,ncol)
    b = b.tolist()
    line1=[]
    for i in range(ncol):
        if i % 2 == 0:
            for j in range(nrow):
                line1.append(b[j][i])
        else:
            for j in range(nrow-1,-1,-1):
                line1.append(b[j][i])

    line2=[]
    for i in range(nrow):
        if i % 2 == 0:
            for j in range(ncol):
                line2.append(b[i][j])
        else:
            for j in range(ncol-1,-1,-1):
                line2.append(b[i][j])

    line_set=[]
    line_set.append(line1)
    line_set.append(line2)

    return line_set

