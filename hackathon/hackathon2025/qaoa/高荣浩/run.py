from multiprocessing import Pool, cpu_count
import time

from answer import *
# from adapt_clifford import *
from samples.utils import build_ham_ising


def load_data(nqubit, seed):
    data_file = f'./samples/public/Q_ising_triu_uniform_complete_{nqubit}_{seed}.npz'
    data = np.load(data_file)
    Q_triu = data['arr_0']
    Q_triu = np.triu(Q_triu, k=0)
    return Q_triu


def single_score(args):
    nqubit, seed = args
    # print("nqubit", nqubit)
    # print("seed", seed)
    Q_triu = load_data(nqubit, seed)
    ham = build_ham_ising(Q_triu)
    solve_circ = solve(nqubit, Q_triu)
    # print("solve_circ = ", solve_circ)
    # sim = Simulator('stabilizer', solve_circ.n_qubits)
    sim = Simulator('stabilizer', nqubit)
    sim.reset()
    exp = sim.get_expectation(Hamiltonian(ham), solve_circ)
    energy = exp.real
    return -energy


def judgment():
    seed_length = 5
    tasks = []
    for nqubit in range(200, 1100, 200):
        for seed in range(seed_length):
            tasks.append((nqubit, seed))
    print('start')
    # size = min(len(tasks), cpu_count(), 4)
    size = 4
    pool = Pool(size)
    scores = pool.map(single_score, tasks)
    pool.close()
    pool.join()
    print(scores)
    return sum(scores)



if __name__ == '__main__':
    score_list = []
    for i in range(1):
        print(i)    
        start = time.time()
        score = judgment()
        end = time.time()
        print("use time：", str(end - start))
        print("score:", "%.4f" % score)
        
        score_list.append(score)
    print("score_list:", score_list)
