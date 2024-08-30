import os
import sys

sys.path.append(os.path.abspath(__file__))
import argparse

parse = argparse.ArgumentParser(description="测试你的求解算法。")
# parse.add_argument('-f', '--file', default='mol.csv', type=str, help="输入分子文件，默认值为 mol.csv")
parse.add_argument('-f', '--file', default='mol_test.csv', type=str, help="输入分子文件，默认值为 mol.csv")
parse.add_argument('--demo', default=0, action='store_true', help="运行官方提供的demo方案")
args = parse.parse_args()

import simulator
from simulator import HKSSimulator, init_shots_counter, Simulator
from utils import read_mol_data, generate_molecule
import inspect
import time
if args.demo:
    from solution_demo import solution
    print("Using solution_demo for mol.csv.")
    # print("Using solution_demo for mol_test.csv.")
else:
    # from solution import solution
    from solution import solution
    print("Using solution_mps_pretrain for mol_test.csv.")
    # print("Using solution_mps_pretrain for mol.csv.")
    # print("Using solution_fixing—5 for mol.csv + Z_ham False.")
    # print("Using solution_fixing—5 for mol.csv + Z_ham True.")


class Solver:

    def __init__(self):
        self.execution_time = 0
        self.result = 0
        self.mol = None

    def __enter__(self):
        init_shots_counter()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(f"shos: {simulator.shots_counter.count}")
        print(f"time: {self.execution_time}")
        print(f"energy: {self.result}")
        print(f"fci: {self.mol.fci_energy}")
        print(f"scaore: {1/abs(self.result - self.mol.fci_energy)}")

    def run(self, method, molecule):
        origin_sig = '(molecule, Simulator: simulator.HKSSimulator) -> float'
        if str(inspect.signature(method)) != origin_sig:
            raise RuntimeError("You can not change the signature of solution.")
        t0 = time.time()
        res = method(molecule, HKSSimulator)
        self.execution_time = time.time() - t0
        self.result = res
        self.mol = generate_molecule(molecule)
        res0 = method(molecule, Simulator)
        print(f"Simulator testing energy: {res0}")
        return res


# 自动调用你的解答方法。
with Solver() as solver:
    solver.run(solution, read_mol_data(args.file))
