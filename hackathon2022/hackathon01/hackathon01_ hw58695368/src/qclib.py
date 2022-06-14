# -*- coding: utf-8 -*-
"""
Author: NoEvaa
Date: 2022-03-15 20:30:12
LastEditTime: 2022-04-06 23:34:43
Description: A library for 
             generating 
             quantum circuits.
FilePath: /src/qclib.py
"""
from mindquantum.core.circuit import Circuit, UN, add_prefix
from mindquantum import gates as G
import numpy as np

class QCircuitLib_basic: 
    def entanglements(self, nqubits, mode='full'): # CX纠缠层
        '''
        CX entanglements.

        Args:
            nqubits (int): Number of qubits.
            mode (Union[str, list]): Specifies the entanglement structure. 
                        Can be a string ('full', 'linear' or 'circular'), 
                        or a list of integer-pairs specifying 
                        the indices of qubits entangled with one another.
                        Default: 'full'.

        Returns:
            Circuit, a quantum circuit.

        Raises:
            TypeError: If nqubits is not int.
            ValueError: If nqubits is smaller than 2.
        '''
        if mode == None:
            return Circuit()
        if isinstance(mode, Circuit):
            return mode
        if isinstance(mode, list):
            return UN(G.X, mode[0], mode[1])

        if not isinstance(nqubits, int):
            try:
                nqubits = int(nqubits)
            except TypeError:
                raise TypeError("Not supported type:{}".format(type(nqubits)))
        if nqubits < 2:
            raise ValueError("The nqubits must be greater than 1, but get {}".format(nqubits))

        mode = mode if mode in ['full', 'linear', 'circular'] else 'full'
        if mode == 'full':
            return self.entanglements_full(nqubits)
        if mode == 'linear':
            return self.entanglements_linear(nqubits)
        if mode == 'circular':
            return self.entanglements_circular(nqubits)

    def QVC_core(self, nqubits, gatelist, mode='full', 
                        reps=1, pname=['h', 'p'], le=False, re=False): # 变分量子线路核
        '''
        Quantum variational circuit core.

        Args:
            nqubits (int): Number of qubits.
            gatelist (list): .
            mode (Union[str, list]): Specifies the entanglement structure. 
                        Can be a string ('full', 'linear' or 'circular'), 
                        or a list of integer-pairs specifying 
                        the indices of qubits entangled with one another.
                        Default: 'full'.
            reps (int): Specifies how often the structure of a rotation layer 
                        followed by an entanglement layer is repeated.
                        Default: 1.

        Returns:
            Circuit, a quantum circuit.
        '''
        # 省略 Fun:参数输入检查
        circ = self.basic_para_circ(nqubits, gatelist, pname[1])
        layer = self.entanglements(nqubits, mode) + circ
        circ = add_prefix(layer, pname[0]+'0') if le else add_prefix(circ, pname[0]+'0') #起始纠缠
        for i in range(1, reps+1):
            circ += add_prefix(layer, pname[0]+f'{i}')
        if re:
            circ += self.entanglements(nqubits, mode)
        return circ

    def join(self, circlist, prefix=None): # 拼接多个个QVC
        prefix = prefix or 'noe'
        circ = Circuit()
        for i in range(len(circlist)):
            circ += add_prefix(circlist[i], prefix+f'{i}')
        return circ

    def basic_para_circ(self, nqubits, gatelist, pname='p', coef=1): # 基本含参量子线路
        # 省略 Fun:参数输入检查
        circ = Circuit()
        j = -1
        for i in range(nqubits):
            for g in gatelist:
                j += 1
                circ += g({pname+f'{j}':coef}).on(i)
        return circ    

    def entanglements_full(self, nq):
        circ = Circuit()
        for i in range(nq):
            circ += UN(G.X, [k for k in range(i+1,nq)], [i]*(nq-1-i))
        return circ
    def entanglements_linear(self, nq):
        return UN(G.X, [k for k in range(1, nq)], [k for k in range(nq-1)])
    def entanglements_circular(self, nq):
        return UN(G.X, [k for k in range(nq)], [nq-1] + [k for k in range(nq-1)])


class QCircuitLib_pure(QCircuitLib_basic):
    def __init__(self):
        QCircuitLib_basic.__init__(self)
        self.qvc = self.QVC_core

    def entanglements_cu(self, nqubits): # 常用CX纠缠层
        circ = Circuit()
        circ += UN(G.X, [k*2+1 for k in range(int(nqubits/2))], [k*2 for k in range(int(nqubits/2))])
        circ += UN(G.X, [k*2+2 for k in range(int((nqubits-1)/2))], [k*2+1 for k in range(int((nqubits-1)/2))])
        return circ

    def encoder_mini(self, nqubits, gate, coef=1): # 极简单门编码器
        # 省略 Fun:参数输入检查
        circ = Circuit()
        for i in range(nqubits):
            circ += gate({f'e{i}':coef}).on(i)
        return circ

    def encoder_mli(self, nqubits, gatelist, layers, mode='full', coef=1): # Multilayer Input + CX Entanglements
        '''
        A simple encoder.
        Number of classical input parameters of quantum circuits is equal to (nqubits * layers).

        Args:
            nqubits (int): Number of qubits.
            gatelist (list): .
            layers (int): Number of encoder layers.
            mode (Union[str, list]): Specifies the entanglement structure. 
                        Can be a string ('full', 'linear' or 'circular'), 
                        or a list of integer-pairs specifying 
                        the indices of qubits entangled with one another.
                        Default: 'full'.
            coef (float)

        Returns:
            Circuit, a quantum circuit.
        '''
        # 省略 Fun:参数输入检查
        ecl = self.basic_para_circ(nqubits, gatelist, coef=coef) + self.entanglements(nqubits, mode)
        circ = Circuit()
        for i in range(layers):
            circ += add_prefix(ecl, f'e{i}')
        return circ

    def RealAmplitudes(self, nqubits, mode='full', reps=1): # 实振幅
        '''
        The 'RealAmplitudes' circuit consists of alternating layers of Y rotations 
        and CX entanglements. 
        The prepared quantum states will only have real amplitudes, 
        the complex part is always 0.

        Args:
            nqubits (int): Number of qubits.
            mode (Union[str, list]): Specifies the entanglement structure. 
                        Can be a string ('full', 'linear' or 'circular'), 
                        or a list of integer-pairs specifying 
                        the indices of qubits entangled with one another.
                        Default: 'full'.
            reps (int): Specifies how often the structure of a rotation layer 
                        followed by an entanglement layer is repeated.
                        Default: 1.

        Returns:
            Circuit, a quantum circuit.
        '''
        return self.QVC_core(nqubits, [G.RY], mode, reps)
        
    def IQP(self, nqubits, coef=1): # IQP编码(-v-)  RZ  circular
        '''
        Instantaneous Quantum Polynomial encoding

        Args:
            nqubits (int): Number of qubits.
            coef (float)

        Returns:
            Circuit, a quantum circuit.
        '''
        circ = Circuit()
        circ += UN(G.H, nqubits)
        for i in range(nqubits):
            circ += G.RZ({f'e{i}':coef}).on(i)
        for j in range(nqubits-1):
            circ += G.X.on(j+1, j)
            circ += G.RZ({f'e{j+nqubits}':coef}).on(j+1)
            circ += G.X.on(j+1, j)
        circ += G.X.on(0, j+1)
        circ += G.RZ({f'e{j+1+nqubits}':coef}).on(0)
        circ += G.X.on(0, j+1)
        return circ

    def encoder_bs(self, bpq=4, nqubits=4, init=None, gatelist=None, params=None):
        '''
        Encoder for binary system.

        Args:
            bpq (int): Number of bits encoded per qubit. Default: 4.
            nqubits (int): Number of qubits.
            init (list): Default: None.
            gatelist (list): Default: None.
            params (list): Default: None.

        Returns:
            Circuit, a quantum circuit.

        Raises:
            ValueError: If the parameter 'bpq' is not supported.
            ValueError: If the lengths of 'gatelist' and 'params' are not equal.
        '''
        if bpq not in [1, 2, 3, 4]:
            raise ValueError(f'A circuit encoding of {bpq} bits is not supported.')
        if None in (init, gatelist, params):
            init, gatelist, params = self.__e_bs_table(bpq)
        else:
            if len(gatelist) != len(params):
                raise ValueError("The lengths of 'gatelist' and 'params' are not equal")

        circ = Circuit()
        for i in range(nqubits):
            # initialize
            for g in init:
                circ += g.on(i)
            # prepare
            for j in range(len(params)):
                circ += gatelist[j]({f'a{i}_e{j}':params[j]}).on(i)
        return circ

    def __e_bs_table(self, n):
        if n == 4:
            return [G.H, G.T, G.H], [G.RY, G.RZ, G.RZ, G.RZ], [np.pi/2, np.pi/2, np.pi, np.pi/4]
        if n == 3:
            return [G.H, G.T, G.H], [G.RY, G.RZ, G.RZ], [np.pi/2, np.pi/2, np.pi]
        if n == 2:
            return [G.H], [G.RZ, G.RZ], [np.pi/2, np.pi]
        if n == 1:
            return [], [G.RY], [np.pi]
        raise
