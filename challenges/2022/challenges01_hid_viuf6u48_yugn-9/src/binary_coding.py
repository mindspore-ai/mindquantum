import numpy as np

def get_code(nmodes, nocc=0):
    from openfermion.transforms import binary_code_transform, \
        jordan_wigner_code, parity_code, bravyi_kitaev_code, \
        checksum_code, interleaved_code, weight_one_binary_addressing_code
    #code = checksum_code(nmodes,0)
    code = checksum_code(nmodes//2,nocc%2) + checksum_code(nmodes//2,nocc%2)
    code = interleaved_code(nmodes) * code
    #code_col = code.encoder.tocoo().col
    #print(np.where(code_col<nocc*2))
    return code

def fermi_to_qubit_coding(fermi_ham, nmodes, nocc=0, code=None):
    from openfermion.transforms import binary_code_transform, \
        jordan_wigner_code, parity_code, bravyi_kitaev_code, \
        checksum_code, interleaved_code, weight_one_binary_addressing_code
    from mindquantum.core.operators.qubit_operator import QubitOperator as QOP1
    from mindquantum.core.operators import FermionOperator as FOP1
    from openfermion.ops import FermionOperator as FOP2
    fermi_ham2 = FOP2()
    fermi_ham2.terms = fermi_ham.terms.copy()
    if (code is None):
        #code = checksum_code(nmodes,0)
        code = checksum_code(nmodes//2,nocc%2) + checksum_code(nmodes//2,nocc%2)
        code = interleaved_code(nmodes) * code
    #code_col = code.encoder.tocoo().col
    #print(np.where(code_col<nocc*2))
    qubit_ham2 = binary_code_transform(fermi_ham2, code)
    qubit_ham = QOP1()
    qubit_ham.terms = qubit_ham2.terms.copy()
    return qubit_ham


