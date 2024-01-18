# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Stabilizer tableau class. A binary table represents a stabilizer circuit.
Reference:
    S. Aaronson, D. Gottesman, Improved Simulation of Stabilizer Circuits,
        Phys. Rev. A 70, 052328 (2004).  arXiv:quant-ph/0406196
"""
import numpy as np
try:
    from mindquantum import *
except:
    raise ImportError("mindquantum is NOT implemented !!!")


class StabilizerTableau:
    def __init__(self, table=None, phase=None, num_qubits=None):
        if table is None:
            if num_qubits is None:
                self._table = np.eye(2, dtype=bool)
                self._phase = np.zeros((2, ), dtype=bool)
            else:
                # initialize by 'num_qubits' only
                self._table = np.eye(2*num_qubits, dtype=bool)
                self._phase = np.zeros((2*num_qubits, ), dtype=bool)
        elif phase is None:
            assert table.ndim==2
            assert table.shape[0]%2==0
            assert table.shape[1]%2==0
            self._table = table
            self._phase = np.zeros((table.shape[0], ), dtype=bool)
        else:
            assert table.ndim==2
            assert table.shape[0]%2==0
            assert table.shape[1]%2==0
            assert phase.ndim==1
            assert phase.shape[0]==table.shape[0]

            self._table = table
            self._phase = phase   

    def initialize_state(self, num_qubits, size=None):
        """
        initialize a M*2N tableau.
        for a 2N*2N eye-type tableau (identity matrix), the tableau means a |0> state for all n qubits
        """
        if size is None:
            # default as 2N*2N table
            self._table = np.zeros((2*num_qubits, 2*num_qubits), dtype=bool)
            self._phase = np.zeros((2*num_qubits, ), dtype=bool)
            for ii in range(2*num_qubits):
                self._table[ii,ii] = 1
        else:
            # M*2N table
            self._table = np.zeros((size, 2*num_qubits), dtype=bool)
            self._phase = np.zeros((size, ), dtype=bool)
            for ii in range(min(size,2*num_qubits)):
                self._table[ii,ii] = 1

    @property
    def size(self):
        return self._table.shape[0]

    @property
    def shape(self):
        return self._table.shape

    @property
    def num_qubits(self):
        return self._table.shape[1]//2

    def copy(self):
        newStab = StabilizerTableau(num_qubits=self.num_qubits)
        newStab._table = self._table.copy()
        newStab._phase = self._phase.copy()
        return newStab

    @staticmethod
    def swap(a, b):
        # TODO: bad swap, NOT change a and b
        # DO NOT USE IT !!!
        t = a;  b = a;  a = t

    @staticmethod
    def g(x1,z1,x2,z2):
        """
        returns the exponent to which i is raised (either 0, 1, or −1)
            when the Pauli matrices represented by x1 z1 and x2 z2 are multiplied.
        Example:
            X*Y = iZ  =>  g(1,0,1,1) = 1
            X*Z =-iY  =>  g(1,0,0,1) =-1
        """
        if   (x1==0 and z1==0): # I
            return 0
        elif (x1==1 and z1==0): # X
            return z2*(2*x2-1)
        elif (x1==0 and z1==1): # Z
            return x2*(1-2*z2)
        elif (x1==1 and z1==1): # Y
            return int(z2) - int(x2)

    def rowsum(self, h, j):
        """
        apply row sum for h-th and j-th Pauli string (add j-th to h-th Pauli)
        """
        n = self.num_qubits
        r0 = 2*(self._phase[h]+self._phase[j])
        for i in range(n):
            r0 += self.g(self._table[h,i], self._table[h,i+n], \
                    self._table[j,i], self._table[j,i+n])
            self._table[h,i] = self._table[h,i] ^ self._table[j,i]
            self._table[h,i+n] = self._table[h,i+n] ^ self._table[j,i+n]
        self._phase[h] = (int(r0)%4)//2


    def CNOT(self, control, target):
        """
        apply a CNOT gate on the qubit 'target' controled by the qubit 'control'.
        """
        n = self.num_qubits
        for i in range(2*n):
            # phase
            self._phase[i] = self._phase[i] \
                ^ ( self._table[i,control]*self._table[i,target+n]
                    *(self._table[i,target] ^ self._table[i,control+n] ^ 1) )
            # table
            self._table[i,target] = self._table[i,target] ^ self._table[i,control]
            self._table[i,control+n] = self._table[i,control+n] ^ self._table[i,target+n]

    def Hadamard(self, a):
        """
        apply a Hadamard gate on the qubit 'a'.
        
        """
        n = self.num_qubits
        for i in range(2*n):
            # swap x_{ia} with z_{ia} in the table
            # phase
            self._phase[i] = self._phase[i] ^ (self._table[i,a]*self._table[i,a+n])
            # table
            # self.swap(self._table[i,a], self._table[i,a+n])
            temp = self._table[i,a]
            self._table[i,a] = self._table[i,a+n]
            self._table[i,a+n] = temp
                    
    def PhaseGate(self, a):
        """
        apply a Phase gate P on the qubit 'a'.

        """
        n = self.num_qubits
        for i in range(2*n):
            # phase
            self._phase[i] = self._phase[i] ^ (self._table[i,a]*self._table[i,a+n])
            # table
            self._table[i,a+n] = self._table[i,a+n] ^ self._table[i,a]

    def Pdagger(self, a):
        """
        apply a Phase_dagger gate P^dagger on the qubit 'a'.  P^dagger = PPP

        """
        n = self.num_qubits
        for i in range(2*n):
            # phase
            self._phase[i] = self._phase[i] ^ (self._table[i,a]*self._table[i,a+n]) ^ self._table[i,a]
            # table
            self._table[i,a+n] = self._table[i,a+n] ^ self._table[i,a]

    def ZGate(self, a):
        """
        apply a Z gate on the qubit 'a'.  note Z=PP

        """
        n = self.num_qubits
        for i in range(2*n):
            # phase
            self._phase[i] = self._phase[i] ^ self._table[i,a]
            # table: nothing to do

    def XGate(self, a):
        """
        apply a X gate on the qubit 'a'.  note X=HZH=HPPH

        """
        n = self.num_qubits
        for i in range(2*n):
            # phase
            self._phase[i] = self._phase[i] ^ self._table[i,a+n]
            # table: nothing to do

    def Measurement(self, a):
        """
        take a measurement on the qubit 'a'.
        """
        n = self.num_qubits
        for p in range(n,2*n):
            # check whether there exists a p in {n+1,...,2n} such that xpa = 1
            if self._table[p,a]==1 :
                # First call rowsum(i,p) for all i ∈ {1,...,2n} such that i!= p and xia = 1
                for i in range(2*n):
                    if (self._table[i,a]==1) and (i!=p):
                        self.rowsum(i, p)
                
                # Second, set entire the (p − n)-th row equal to the p=th row.
                self._table[p-n,:] = self._table[p,:]
                self._phase[p-n]   = self._phase[p]
                
                # Third, set the pth row to be identically 0, 
                #        except that rp is 0 or 1 with equal probability, and zpa = 1
                self._table[p,:] = 0
                self._table[p,a+n] = 1
                self._phase[p] = np.random.randint(2)  # 0 or 1 with equal probability        
                return self._phase[p]                
        
        # Such an p (xpa=1) does not exist
        # expand to 2n+1 rows
        # TODO: bad expand to (2n+1)*2n
        tableau_temp = StabilizerTableau()
        tableau_temp.initialize_state(num_qubits=n, size=2*n+1)  # (2n+1)*2n
        tableau_temp._table[:2*n] = self._table
        tableau_temp._phase[:2*n] = self._phase
        for i in range(n):
            if tableau_temp._table[i,a]==1:
                tableau_temp.rowsum(2*n, i+n)
        return tableau_temp._phase[2*n]


    def apply_gate(self, gate):
        """
        apply mindquantum gate on the tableau
        """
        try:
            import mindquantum
        except:
            raise ImportError("mindquantum is NOT implemented !!!")
        # TODO: interface to mindquantum
        if isinstance(gate, HGate):
            self.Hadamard(gate.obj_qubits[0])
        elif isinstance(gate, SGate):
            self.PhaseGate(gate.obj_qubits[0])
        elif isinstance(gate, XGate) and len(gate.ctrl_qubits)==1:
            self.CNOT(gate.ctrl_qubits[0], gate.obj_qubits[0])
        elif isinstance(gate, XGate) and len(gate.ctrl_qubits)==0:
            self.XGate(gate.obj_qubits[0])
        elif isinstance(gate, ZGate) and len(gate.ctrl_qubits)==0:
            self.ZGate(gate.obj_qubits[0])
        elif isinstance(gate, Measure):
            self.Measurement(gate.obj_qubits[0])
        else:
            raise ValueError(f'The gate {gate} is not support.')

    
    def print_tableau(self, with_phase=True):
        """
        print the stabilizer tableau
        """
        n = self.num_qubits
        for i in range(2*n):
            table_str = ''
            if i==n: print( "-" * (4*n+4) )
            for j in range(2*n):
                if j==n: table_str += '| '
                table_str += str(int(self._table[i,j])) + ' '
            if with_phase:
                table_str += '| '
                table_str += str(int(self._phase[i])) + ' '
            print(table_str)


    def print_stabilizer(self):
        """
        print the stabilizer
        """
        n = self.num_qubits
        for i in range(2*n):
            stabilizer_str = ''
            if i==0: print('destabilizer:')
            if i==n: print('stabilizer:')
            # phase
            if self._phase[i]==0 :
                stabilizer_str += '+'
            else:
                stabilizer_str += '-'
            # transform the symplectic representation in the tableau
            for j in range(n):
                if   (self._table[i,j]==0) and (self._table[i,j+n]==0) :
                    stabilizer_str += 'I'   # X^0 Z^0 = I
                elif (self._table[i,j]==1) and (self._table[i,j+n]==0) :
                    stabilizer_str += 'X'   # X^1 Z^0 = X
                elif (self._table[i,j]==0) and (self._table[i,j+n]==1) :
                    stabilizer_str += 'Z'   # X^0 Z^1 = Z
                elif (self._table[i,j]==1) and (self._table[i,j+n]==1) :
                    stabilizer_str += 'Y'   # X^1 Z^1 = Y *(-i)
            print(stabilizer_str)
 

    def print(self):
        """
        print the stabilizer
        """
        self.print_stabilizer()


if __name__=="__main__":
    ################################
    ######  test a Hadamard gate ###
    ################################
    # H|0> = (|0> + |1>)/sqrt(2)
    # 0:|0>---H---M---
    num_measure = 10000
    count = 0
    for i in range(num_measure):
        H0 = StabilizerTableau()
        H0.Hadamard(0)
        m0 = H0.Measurement(0)
        count += int(m0)
    print("probability of |1>:", count/num_measure)

    Hp = H0.copy()

    ################################
    ######  test a Bell state  #####
    ################################
    # |Bell> = (|00> + |11>)/sqrt(2)
    # 0:|0>---H---#---M---
    #             |   |
    # 1:|0>-------X---M---
    num_measure = 10000
    count = 0
    for i in range(num_measure):
        bell = StabilizerTableau(num_qubits=2)
        bell.Hadamard(0)
        bell.CNOT(0,1)
        m0 = bell.Measurement(0)
        m1 = bell.Measurement(1)
        if m0==1 and m1==1:
            count +=1
    print("probability of |11>:", count/num_measure)



    ################################
    ######  test a GHZ state  ######
    ################################
    # |GHZ> = (|000> + |111>)/sqrt(2)
    # 0:|0>-----------X---M---
    #                 |
    # 1:|0>-------X---|---M---
    #             |   |
    # 2:|0>---H---#---#---M---
    
    ghz = StabilizerTableau()
    ghz.initialize_state(num_qubits=3, size=6) # |000>, 3 qubits with 6 Pauli generators

    ghz.Hadamard(2)
    ghz.CNOT(2,1)
    ghz.CNOT(2,0)

    print("tableau before measurement:")
    #ghz.print_stabilizer()
    ghz.print_tableau()

    m0 = ghz.Measurement(0)
    m1 = ghz.Measurement(1)
    m2 = ghz.Measurement(2)

    print("tableau after measurement:")
    #ghz.print_stabilizer()
    ghz.print_tableau()
    print("measurement 0:", int(m0), " 1:", int(m1), " 2:", int(m2))


    # test mindquantum gate
    ghz.apply_gate(X.on(0,1))
    ghz.apply_gate(H.on(0))
    ghz.apply_gate(S.on(0))
    ghz.apply_gate(S.on(0).hermitian()) # S^dagger
    ghz.apply_gate(X.on(0))
    ghz.apply_gate(Z.on(0))
    ghz.apply_gate(XGate().on(0,1))
    ghz.apply_gate(HGate().on(0))
    ghz.apply_gate(SGate().on(1))
    ghz.apply_gate(SGate().on(1).hermitian())
    ghz.apply_gate(XGate().on(0))
    ghz.apply_gate(ZGate().on(0))

