import os, sys
sys.path.append(os.path.abspath(__file__))

import time
import inspect
import argparse

parse = argparse.ArgumentParser(description="测试你的求解算法。")
parse.add_argument('-f', '--file', default='mol.csv', type=str, help="输入分子文件，默认值为 mol.csv")
parse.add_argument('--demo', action='store_true', help="运行官方提供的demo方案")
args = parse.parse_args()

import simulator
from simulator import HKSSimulator, init_shots_counter
from utils import read_mol_data, generate_molecule
if args.demo:
    from solution_demo import solution
else:
    from solution import solution


class Solver:

    def __init__(self):
        self.execution_time = 0
        self.result = 0
        self.mol = None

    def __enter__(self):
        init_shots_counter()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        n_shots = simulator.shots_counter.count
        E = self.result
        E_fci = self.mol.fci_energy
        print(f"shots: {n_shots}")
        print(f"time: {self.execution_time:.2f}")   # <= 2 hours
        print(f"energy: {E:.5f}")
        print(f"fci: {E_fci}")                      # -2.166387448634759 for the sample case
        score = 1 / abs(E - E_fci)                  # score function
        print(f">> score: {score:.3f}")

    def run(self, method, molecule):
        origin_sig = '(molecule, Simulator: simulator.HKSSimulator) -> float'
        if str(inspect.signature(method)) != origin_sig:
            raise RuntimeError("You can not change the signature of solution.")
        t0 = time.time()
        res = method(molecule, HKSSimulator)
        self.execution_time = time.time() - t0
        self.result = res
        self.mol = generate_molecule(molecule)
        return res


# 自动调用你的解答方法。
with Solver() as solver:
    # TODO: tmp test data
    #dist = 0.1
    #molecule = [
    #    ['H', [0.0, 0.0, 0.0 * dist]],
    #    ['H', [0.0, 0.0, 1.0 * dist]],
    #    ['H', [0.0, 0.0, 2.0 * dist]],
    #    ['H', [0.0, 0.0, 3.0 * dist]],
    #]
    molecule = read_mol_data(args.file)
    solver.run(solution, molecule)
