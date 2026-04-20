from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

# 1. 计算 Hartree-Fock 态
driver = PySCFDriver(
    # atom="H 0 0 0; H 0 0 0.7414",
    atom="O 0 0 0; H 0 0 0.958*1.3; H 0 0 -0.958*1.3",
    basis="sto3g",
    charge=0,
    spin=0,
    unit=DistanceUnit.ANGSTROM,
)

es_problem = driver.run()

# 打印 Hartree-Fock 能量
hf_energy = es_problem.hamiltonian.nuclear_repulsion_energy + es_problem.reference_energy
print(f"HF Energy (Method 2, Qiskit Nature): {hf_energy}")

fermion_operator = es_problem.second_q_ops()[0]  # 获取费米子算符
print(f"Fermion Operator:\n{fermion_operator}")


# 2. 生成分子哈密顿量并进行 Jordan-Wigner 变换
mapper = JordanWignerMapper()
qubit_op = mapper.map(fermion_operator)  # Jordan-Wigner 转换后的量子比特哈密顿量
sparse_matrix = qubit_op.to_matrix()  # 转换为矩阵


# 打印 Jordan-Wigner 变换后的哈密顿量
print(f"JW Hamiltonian (Method 2, Qiskit Nature):\n{qubit_op}")
print(f"JW Matrix (Method 2, Qiskit Nature):\n{sparse_matrix}")
# 3. 使用 Qiskit 的求解器进行本征值求解
numpy_solver = NumPyMinimumEigensolver()
calc = GroundStateEigensolver(mapper, numpy_solver)
res = calc.solve(es_problem)

# 打印结果
print(f"FCI Energy (Method 2, Qiskit Nature): {res}")
