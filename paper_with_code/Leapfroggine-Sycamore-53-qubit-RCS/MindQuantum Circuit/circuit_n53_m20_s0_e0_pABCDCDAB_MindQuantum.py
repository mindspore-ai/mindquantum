from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import FSim
from mindquantum.io import OpenQASM
from mindquantum.utils import fdopen

with fdopen("qasm_circuit_n53_m20_s0_e0_pABCDCDAB.txt", 'r') as fd:
    cmds = fd.readlines()

for i in range(15):
    if cmds[i].startswith("qreg"):
        qreg_index = i
        break

CIRCUIT = Circuit()
skip = 0
for i in range(len(cmds)):
    if cmds[i].startswith("// Gate: cirq.FSimGate"):
        theta_index = cmds[i].find("theta")
        phi_index = cmds[i].find(", phi")
        theta = float(cmds[i][theta_index + 6 : phi_index])
        phi = float(cmds[i][phi_index + 6 : -2])
        obj0 = int(cmds[i+1].split('[')[1].split(']')[0])
        obj1 = int(cmds[i+2].split('[')[1].split(']')[0])
        CIRCUIT += FSim(theta, -phi).on([obj0, obj1])
        skip = 36
    else:
        if skip != 0:
            skip -= 1
        else:
            CIRCUIT += OpenQASM().from_string(cmds[qreg_index] + cmds[i])

print(CIRCUIT)