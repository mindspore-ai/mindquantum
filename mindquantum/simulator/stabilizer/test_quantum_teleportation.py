import numpy as np
from stabilizer_tableau import StabilizerTableau


if __name__=="__main__":
    """
    test the quantum teleportation.
    Referrence:
        C. H. Bennett, G. Brassard, C. Crepeau, et.al.,
            Phys. Rev. Lett. 70, 1895 (1993).
    """
    # 0:|0>---H-------#---H---M-----------#-------|---
    #                 |       |           |       |
    # 1:|a>---H---#---X-------M---#-------|-------|---
    #             |           |   |       |       |
    # 2:|b>-------X-----------|---X---H---X---H---M---

    nloop = 2000
    count = 0
    for n in range(nloop):
        tele = StabilizerTableau(num_qubits=3)
        # prepare the teleported state
        tele.Hadamard(0)
        #tele.PhaseGate(0)
        # EPR pair (qubit 1 is Alice’s half; qubit 2 is Bob’s half)
        tele.Hadamard(1)
        tele.CNOT(1,2)
        # Alice interacts qubit 0 (the state to be teleported) with her half of the EPR pair
        tele.CNOT(0,1)
        tele.Hadamard(0)
        tele.Measurement(0)
        tele.Measurement(1)
        # Bob uses the bits from Alice to recover the teleported state
        tele.CNOT(1,2)
        tele.Hadamard(2)
        tele.CNOT(0,2)
        tele.Hadamard(2)
        m2 = tele.Measurement(2)
        count += m2
    print("probability of |1> for Bob:", count/nloop)



