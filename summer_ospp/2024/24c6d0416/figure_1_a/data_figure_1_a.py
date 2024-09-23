from Figure1_a_functions import *
'''
    四种分布分别求解，返回精确比
'''
POP = mixer_pool_single(6) + mixer_pool_multi(6)
def circlefunc_u():
    qubo = graph_complete(6,'uniform')
    _, v1, _ = QAOA(qubo, 6, 10, 'cobyla')
    _, v2, _ = ADAPT_QAOA(6, qubo, POP, 10, 'bfgs')
    exact, _ = ground_state_hamiltonian(qubo.hamiltonian)
    return [x / exact * 2 for x in v1],[x / exact * 2 for x in v2]
def circlefunc_e():
    qubo = graph_complete(6,'exponential')
    _, v1, _ = QAOA(qubo, 6, 10, 'cobyla')
    _, v2, _ = ADAPT_QAOA(6, qubo, POP, 10, 'bfgs')
    exact, _ = ground_state_hamiltonian(qubo.hamiltonian)
    return [x / exact * 2 for x in v1],[x / exact * 2 for x in v2]
def circlefunc_n():
    qubo = graph_complete(6,'normal')
    _, v1, _ = QAOA(qubo, 6, 10, 'cobyla')
    _, v2, _ = ADAPT_QAOA(6, qubo, POP, 10, 'bfgs')
    exact, _ = ground_state_hamiltonian(qubo.hamiltonian)
    return [x / exact * 2 for x in v1],[x / exact * 2 for x in v2]
def circlefunc_u2():
    qubo = graph_complete(6,'uniform2')
    _, v1, _ = QAOA(qubo, 6, 10, 'cobyla')
    _, v2, _ = ADAPT_QAOA(6, qubo, POP, 10, 'bfgs')
    exact, _ = ground_state_hamiltonian(qubo.hamiltonian)
    return [x / exact * 2 for x in v1],[x / exact * 2 for x in v2]

R_qaoa_u = []
R_qaoa_e = []
R_qaoa_n = []
R_qaoa_u2 = []
R_adapt_u = []
R_adapt_e = []
R_adapt_n = []
R_adapt_u2 = []

for i in range(5):
    r_qaoa_u, r_adapt_u = circlefunc_u()
    r_qaoa_e, r_adapt_e = circlefunc_e()
    r_qaoa_n, r_adapt_n = circlefunc_n()
    r_qaoa_u2, r_adapt_u2 = circlefunc_u2()

    R_qaoa_u.append(r_qaoa_u)
    R_qaoa_e.append(r_qaoa_e)
    R_qaoa_n.append(r_qaoa_n)
    R_qaoa_u2.append(r_qaoa_u2)

    R_adapt_u.append(r_adapt_u)
    R_adapt_e.append(r_adapt_e)
    R_adapt_n.append(r_adapt_n)
    R_adapt_u2.append(r_adapt_u2)

save_lists_to_csv('qaoa/u.csv', R_qaoa_u)
save_lists_to_csv('qaoa/e.csv', R_qaoa_e)
save_lists_to_csv('qaoa/n.csv', R_qaoa_n)
save_lists_to_csv('qaoa/u2.csv', R_qaoa_u2)

save_lists_to_csv('adapt/u.csv', R_adapt_u)
save_lists_to_csv('adapt/e.csv', R_adapt_e)
save_lists_to_csv('adapt/n.csv', R_adapt_n)
save_lists_to_csv('adapt/u2.csv', R_adapt_u2)


