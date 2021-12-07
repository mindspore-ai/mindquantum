from mindquantum.ops import FermionOperator, QubitOperator
from mindquantum.utils import hermitian_conjugated, normal_ordered
import numpy as np
from mindquantum.hiqfermion.transforms.transform import Transform


def check_operator_requirement(ferop):
    if not ferop.terms:
            # Zero operator
            return 0
    else:
        return max(
            len(term)
            for term, coeff in ferop.terms.items()
            if (abs(coeff) > 1e-12))

class OperatorPool(object):
    def __init__(self, n_orb, n_occ, n_vir):
        self.n_orb = n_orb
        self.n_occ = n_occ
        self.n_vir = n_vir

        self.n_spin_orb = 2*self.n_orb

        self.n_occ_a = n_occ
        self.n_occ_b = n_occ

        self.n_vir_a = n_vir
        self.n_vir_b = n_vir

        self.n_ops = 0

        self.fermi_ops = []


class singlet_SD(OperatorPool):
    def __init__(self, n_orb, n_occ, n_vir):
        super(singlet_SD, self).__init__(n_orb, n_occ, n_vir)
        
    def generate_operators(self):
        n_occ = self.n_occ
        n_vir = self.n_vir

        for i in range(0,n_occ):
            ia = 2*i
            ib = 2*i+1

            for a in range(0,n_vir):
                aa = 2*n_occ + 2*a
                ab = 2*n_occ + 2*a+1

                termA =  FermionOperator(((aa,1),(ia,0)), 1/np.sqrt(2))
                termA += FermionOperator(((ab,1),(ib,0)), 1/np.sqrt(2))

                termA -= hermitian_conjugated(termA)

                termA = normal_ordered(termA)

                # without normalization
                # if check_operator_requirement(termA) > 0:
                #     self.fermi_ops.append(termA)

                # Normalization
                coeffA = 0
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t

                if check_operator_requirement(termA) > 0:
                    termA = termA/np.sqrt(coeffA)
                    self.fermi_ops.append(termA)
        for i in range(0,n_occ):
            ia = 2*i
            ib = 2*i+1

            for j in range(i,n_occ):
                ja = 2*j
                jb = 2*j+1

                for a in range(0,n_vir):
                    aa = 2*n_occ + 2*a
                    ab = 2*n_occ + 2*a+1

                    for b in range(a,n_vir):
                        ba = 2*n_occ + 2*b
                        bb = 2*n_occ + 2*b+1

                        termA =  FermionOperator(((aa,1),(ba,1),(ia,0),(ja,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((ab,1),(bb,1),(ib,0),(jb,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((aa,1),(bb,1),(ia,0),(jb,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((ab,1),(ba,1),(ib,0),(ja,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((aa,1),(bb,1),(ib,0),(ja,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((ab,1),(ba,1),(ia,0),(jb,0)), 1/np.sqrt(12))

                        termB  = FermionOperator(((aa,1),(bb,1),(ia,0),(jb,0)), 1/2)
                        termB += FermionOperator(((ab,1),(ba,1),(ib,0),(ja,0)), 1/2)
                        termB += FermionOperator(((aa,1),(bb,1),(ib,0),(ja,0)), -1/2)
                        termB += FermionOperator(((ab,1),(ba,1),(ia,0),(jb,0)), -1/2)

                        termA -= hermitian_conjugated(termA)
                        termB -= hermitian_conjugated(termB)

                        termA = normal_ordered(termA)
                        termB = normal_ordered(termB)

                        # Without Normalization
                        # if check_operator_requirement(termA) > 0:
                        #     self.fermi_ops.append(termA)
                        # if check_operator_requirement(termB) > 0:    
                        #     self.fermi_ops.append(termB)

                        # Normalize
                        coeffA = 0
                        coeffB = 0
                        for t in termA.terms:
                            coeff_t = termA.terms[t]
                            coeffA += coeff_t * coeff_t
                        for t in termB.terms:
                            coeff_t = termB.terms[t]
                            coeffB += coeff_t * coeff_t


                        if check_operator_requirement(termA) > 0:
                            termA = termA/np.sqrt(coeffA)
                            self.fermi_ops.append(termA)

                        if check_operator_requirement(termB) > 0:
                            termB = termB/np.sqrt(coeffB)
                            self.fermi_ops.append(termB)
        
        self.n_ops = len(self.fermi_ops)
        # print(" Number of operators: ", self.n_ops)
        return 


class singlet_GSD(OperatorPool):
    def __init__(self, n_orb, n_occ, n_vir):
        super(singlet_GSD, self).__init__(n_orb, n_occ, n_vir)
    
    def generate_operators(self):
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1

            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1

                termA =  FermionOperator(((pa,1),(qa,0)))
                termA += FermionOperator(((pb,1),(qb,0)))

                termA -= hermitian_conjugated(termA)

                termA = normal_ordered(termA)

                # Without Normalization
               
                # if check_operator_requirement(termA) > 0:
                #     self.fermi_ops.append(termA)

                # Normalization
                coeffA = 0
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t

                if check_operator_requirement(termA) > 0:
                    termA = termA/np.sqrt(coeffA)
                    self.fermi_ops.append(termA)


        pq = -1
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1

            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1

                pq += 1

                rs = -1
                for r in range(0,self.n_orb):
                    ra = 2*r
                    rb = 2*r+1

                    for s in range(r,self.n_orb):
                        sa = 2*s
                        sb = 2*s+1

                        rs += 1

                        if(pq > rs):
                            continue

#                        oplist = []
#                        oplist.append(FermionOperator(((ra,1),(pa,0),(sa,1),(qa,0)), 2/np.sqrt(12)))
#                        oplist.append(FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)), 2/np.sqrt(12)))
#                        oplist.append(FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)), 1/np.sqrt(12)))
#                        oplist.append(FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)), 1/np.sqrt(12)))
#                        oplist.append(FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), 1/np.sqrt(12)))
#                        oplist.append(FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), 1/np.sqrt(12)))
#
#                        print(p,q,r,s)
#                        for i in range(len(oplist)):
#                            oplist[i] -= hermitian_conjugated(oplist[i])
#                        for i in range(len(oplist)):
#                            for j in range(i+1,len(oplist)):
#                                opi = oplist[i]
#                                opj = oplist[i]
#                                opij = opi*opj - opj*opi
#                                opij = normal_ordered(opij)
#                                print(opij, end='')
#                        print()
                        termA =  FermionOperator(((ra,1),(pa,0),(sa,1),(qa,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), 1/np.sqrt(12))

                        termB =  FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)),  1/2.0)
                        termB += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)),  1/2.0)
                        termB += FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), -1/2.0)
                        termB += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), -1/2.0)

                        termA -= hermitian_conjugated(termA)
                        termB -= hermitian_conjugated(termB)

                        termA = normal_ordered(termA)
                        termB = normal_ordered(termB)

                        # #Normalize
                        # if check_operator_requirement(termA) > 0:
                        #     self.fermi_ops.append(termA)
                        # if check_operator_requirement(termB) > 0:
                        #     self.fermi_ops.append(termB)
                        # Normalize
                        coeffA = 0
                        coeffB = 0
                        for t in termA.terms:
                            coeff_t = termA.terms[t]
                            coeffA += coeff_t * coeff_t
                        for t in termB.terms:
                            coeff_t = termB.terms[t]
                            coeffB += coeff_t * coeff_t


                        if check_operator_requirement(termA) > 0:
                            termA = termA/np.sqrt(coeffA)
                            self.fermi_ops.append(termA)

                        if check_operator_requirement(termB) > 0:
                            termB = termB/np.sqrt(coeffB)
                            self.fermi_ops.append(termB)

        self.n_ops = len(self.fermi_ops)
        # print(" Number of operators: ", self.n_ops)
        return



def pauli_pool(collection_of_ferops):
    collection_of_jw_qubitops = [Transform(ferop).jordan_wigner() for ferop in collection_of_ferops]
    
    collection_of_pauli_strings_without_z = []
    for sub_collection in collection_of_jw_qubitops:
        for pauli_seq, coeff in sub_collection.terms.items():
            # print(pauli_seq, coeff)
            temp_pauli_seq = QubitOperator(())
            for qubit_id, pauli in pauli_seq:
                if pauli == 'Z':
                    continue
                else:
                    # print(qubit_id, pauli)
                    temp_pauli_seq *= QubitOperator(((qubit_id, pauli),))
            collection_of_pauli_strings_without_z.append(temp_pauli_seq)
    return collection_of_pauli_strings_without_z

def rename_pauli_string(single_pauli_string, iteration):
    renamed_single_pauli_string = QubitOperator(())
    for pauli_seq, coeff in single_pauli_string.terms.items():
        renamed_single_pauli_string = QubitOperator(pauli_seq, 'p'+str(iteration))
    return renamed_single_pauli_string


if __name__=="__main__":
    sd_ops = singlet_SD(4, 2, 2)
    sd_ops.generate_operators()

    collection_of_ferops = sd_ops.fermi_ops
    # print(collection_of_ferops)
    # for index, ferop in enumerate(collection_of_ferops):
    #     print('{}-th fermionic operators: {}'.format(index, ferop))
    #     print('correspondong pauli ops: {}'.format(Transform(ferop).jordan_wigner()))
    #     print('--------------------------------------------------')

    print(pauli_pool(collection_of_ferops))


