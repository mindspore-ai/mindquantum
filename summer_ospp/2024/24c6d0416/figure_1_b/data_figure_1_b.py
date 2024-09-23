from Figure1_b_functions import *

'''
    定义6比特的 Pauli basis
'''
paulix = np.array([[0,1],[1,0]])
pauliy = np.array([[0,-1j],[1j,0]])
pauliz = np.array([[1,0],[0,-1]])
paulii = np.array([[1,0],[0,1]])
pauli_single = [paulii, paulix, pauliy, pauliz]

Yn = [
    np.kron(pauliy,np.kron(paulii,np.kron(paulii,np.kron(paulii,np.kron(paulii,paulii))))),
    np.kron(paulii,np.kron(pauliy,np.kron(paulii,np.kron(paulii,np.kron(paulii,paulii))))),
    np.kron(paulii,np.kron(paulii,np.kron(pauliy,np.kron(paulii,np.kron(paulii,paulii))))),
    np.kron(paulii,np.kron(paulii,np.kron(paulii,np.kron(pauliy,np.kron(paulii,paulii))))),
    np.kron(paulii,np.kron(paulii,np.kron(paulii,np.kron(paulii,np.kron(pauliy,paulii))))),
    np.kron(paulii,np.kron(paulii,np.kron(paulii,np.kron(paulii,np.kron(paulii,pauliy))))),
]
pauli = [
    np.kron(a,np.kron(b,np.kron(c,np.kron(d,np.kron(e,f))))) 
    for a in pauli_single 
    for b in pauli_single 
    for c in pauli_single 
    for d in pauli_single
    for e in pauli_single
    for f in pauli_single
]

'''
    求Renyi 熵
'''
def entropy_fun(mats,PY):
    S_qubo = []
    for mat in mats:
        O = mat.conj().T @ PY @ mat
        coff = []
        for i in pauli:
            c = np.trace(O @ i) / 2**6
            coff.append(c)
        value = 0
        for j in coff:
            value += abs(j)**4

        S = -np.log2(value)
        S_qubo.append(S)
    return S_qubo

'''
    求Renyi 熵的平均值
'''
def entropy_avr(mats):
    lst = [entropy_fun(mats,Yn[0]),
    entropy_fun(mats,Yn[1]),
    entropy_fun(mats,Yn[2]),
    entropy_fun(mats,Yn[3]),
    entropy_fun(mats,Yn[4]),
    entropy_fun(mats,Yn[5])]

    a = np.average([lst[i][0] for i in range(6)])
    b = np.average([lst[i][1] for i in range(6)])
    c = np.average([lst[i][2] for i in range(6)])
    d = np.average([lst[i][3] for i in range(6)])
    e = np.average([lst[i][4] for i in range(6)])
    f = np.average([lst[i][5] for i in range(6)])
    g = np.average([lst[i][6] for i in range(6)])
    h = np.average([lst[i][7] for i in range(6)])
    i = np.average([lst[i][8] for i in range(6)])
    j = np.average([lst[i][9] for i in range(6)])
    return [a,b,c,d,e,f,g,h,i,j]


nodes = 6
pool = mixer_pool_single(nodes) + mixer_pool_pop(nodes)

S1_all = []
S2_all = []
for i in range(50):
    qubo = graph_complete(nodes, 'uniform')
    pr1, _, circ1 = ADAPT_QAOA(nodes,qubo,pool,10,'bfgs')
    pr2, _, circ2 = QAOA(qubo, nodes, 10, 'cobyla')
    mlst1 = mat_lst(pr1,circ1)
    mlst2 = mat_lst(pr2,circ2)
    S1_all.append(entropy_avr(mlst1))
    S2_all.append(entropy_avr(mlst2))
save_list_to_csv('data1b_adapt_ex.csv', S1_all)
save_list_to_csv('data1b_stand_ex.csv', S2_all)

S3_all = []
S4_all = []
for i in range(50):
    qubo = graph_complete(nodes, 'exponential')
    pr1, _, circ1 = ADAPT_QAOA(nodes,qubo,pool,10,'bfgs')
    pr2, _, circ2 = QAOA(qubo, nodes, 10, 'cobyla')
    mlst1 = mat_lst(pr1,circ1)
    mlst2 = mat_lst(pr2,circ2)
    S3_all.append(entropy_avr(mlst1))
    S4_all.append(entropy_avr(mlst2))
save_list_to_csv('data1b_adapt_ex.csv', S3_all)
save_list_to_csv('data1b_stand_ex.csv', S4_all)

S5_all = []
S6_all = []
for i in range(50):
    qubo = graph_complete(nodes, 'normal')
    pr1, _, circ1 = ADAPT_QAOA(nodes,qubo,pool,10,'bfgs')
    pr2, _, circ2 = QAOA(qubo, nodes, 10, 'cobyla')
    mlst1 = mat_lst(pr1,circ1)
    mlst2 = mat_lst(pr2,circ2)
    S5_all.append(entropy_avr(mlst1))
    S6_all.append(entropy_avr(mlst2))
save_list_to_csv('data1b_adapt_no.csv', S5_all)
save_list_to_csv('data1b_stand_no.csv', S6_all)

S7_all = []
S8_all = []
for i in range(50):
    qubo = graph_complete(nodes, 'uniform2')
    pr1, _, circ1 = ADAPT_QAOA(nodes,qubo,pool,10,'bfgs')
    pr2, _, circ2 = QAOA(qubo, nodes, 10, 'cobyla')
    mlst1 = mat_lst(pr1,circ1)
    mlst2 = mat_lst(pr2,circ2)
    S7_all.append(entropy_avr(mlst1))
    S8_all.append(entropy_avr(mlst2))
save_list_to_csv('data1b_adapt_un2.csv', S7_all)
save_list_to_csv('data1b_stand_un2.csv', S8_all)