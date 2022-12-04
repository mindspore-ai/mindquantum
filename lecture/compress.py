#!/usr/bin/env python
# coding: utf-8

# In[24]:


import math
import cmath
import numpy as np
   
from mindquantum.core.gates import BasicGate,X, H, RY,Z,CNOT,RX,RZ    # 导入量子门H, X, RY
from mindquantum.core.circuit import Circuit
from mindquantum.dagcircuit import DAGCircuit
from mindquantum.dagcircuit import circuit_to_dag1


# In[25]:


def _mod_2pi(angle: float, atol: float = 0):
    #Wrap angle into interval [-π,π). If within atol of the endpoint, clamp to -π
    wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
    if abs(wrapped - np.pi) < atol:
        wrapped = -np.pi
    return wrapped

def paramsof_zyz(mat):
    import scipy.linalg as la
    # We rescale the input matrix to be special unitary (det(U) = 1)
    # This ensures that the quaternion representation is real
    coeff = la.det(mat) ** (-0.5)
    phase = -cmath.phase(coeff)
    su_mat = coeff * mat 
    theta = 2 * math.atan2(abs(su_mat[1, 0]), abs(su_mat[0, 0]))
    phiplambda2 = cmath.phase(su_mat[1, 1])
    phimlambda2 = cmath.phase(su_mat[1, 0])
    phi = phiplambda2 + phimlambda2
    lam = phiplambda2 - phimlambda2
    return theta, phi, lam

def circuitof_zyz(
        theta,
        phi,
        lam,
        qubit,
        atol=1e-10,
        k_gate=RZ,
        a_gate=RY,
    ):
        circuit = Circuit()
        if abs(theta) < atol:
            lam, phi = lam + phi, 0
            lam = _mod_2pi(lam, atol)
            if abs(lam) > atol:
                circuit.append(k_gate(lam).on(qubit))
            return circuit
        if abs(theta - np.pi) < atol:
            lam, phi = lam - phi, 0
        if (abs(_mod_2pi(lam + np.pi)) < atol or abs(_mod_2pi(phi + np.pi)) < atol):
            lam, theta, phi = lam + np.pi, -theta, phi + np.pi
        lam = _mod_2pi(lam, atol)
        if abs(lam) > atol:
            circuit.append(k_gate(lam).on(qubit))
        circuit.append(a_gate(theta).on(qubit))
        phi = _mod_2pi(phi, atol)
        if abs(phi) > atol:
            circuit.append(k_gate(phi).on(qubit))
        return circuit

def dispose_run(run):
    operator = run[0].op.matrix()
    for gate in run[1:]:
        operator = gate.op.matrix().dot(operator)
    qubit=run[0].op.obj_qubits[0]
    [t,p,l]=paramsof_zyz(operator)
    new_circ =circuitof_zyz(t,p,l,qubit)

    return new_circ

def compress(circ):
    #compress adjacent one-qubit gates
    [dag,que]=circuit_to_dag1(circ)
    runs = dag.collect_1q_runs()
    for run in runs:
        new_circ = dispose_run(run)
        sub_circ=Circuit()
        if new_circ is not None and len(new_circ)<len(run): 
            nodelist=[]
            for j in range(len(run)):
                nodelist.append(run[j]._node_id)
            j=0
            newque=[]
            for i in range(len(circ)):
                nodeid=que[i]
                if nodeid not in nodelist:
                    sub_circ.append(circ[i])
                    newque.append(que[i])
                elif j<len(new_circ):
                    sub_circ.append(new_circ[j])
                    newque.append(max(que)+1)
                    j+=1
            que=newque.copy()
        if len(sub_circ)>0:
            circ=Circuit()
            circ=sub_circ.__copy__()
            
    return circ




