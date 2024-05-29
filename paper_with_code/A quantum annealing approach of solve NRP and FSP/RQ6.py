import sys
import os
import numpy as np
import qiskit_aer as Aer
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.translators import from_docplex_mp
from qiskit_algorithms import QAOA
from qiskit.primitives import Estimator

# Load the project path
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

# Define problem details
names_NRP = ['rp', 'ms', 'Baan', 'classic-1']
order_NRP = ['cost', 'revenue']
names_FSP = ['BerkeleyDB', 'ERS', 'WebPortal', 'Drupal', 'Amazon', 'E-Shop']
order_FSP = ['COST', 'USED_BEFORE', 'DEFECTS', 'DESELECTED']

def solve_problem(name, order):
    # Assuming you have a function to load problem details and convert to QUBO
    Q = np.random.randn(10, 10)  # Placeholder for the QUBO matrix for the problem

    # Create a Qiskit QuadraticProgram
    qp = QuadraticProgram(name=name)
    for i in range(Q.shape[0]):
        qp.binary_var(name=f"x{i}")
        for j in range(i, Q.shape[1]):
            qp.minimize(linear={f"x{i}": Q[i, i]}, quadratic={('x'+str(i), 'x'+str(j)): Q[i, j]})
    
    # Define quantum instance
    quantum_instance = Aer.get_backend('qasm_simulator')
    
    # Define QAOA with specific parameters
    qaoa = QAOA(quantum_instance=quantum_instance, reps=3)
    optimizer = MinimumEigenOptimizer(qaoa)
    
    # Solve the problem using the QAOA optimizer
    result = optimizer.solve(qp)
    
    return result

for name in names_NRP:
    order = order_NRP
    result_folder = 'EA-GA-{}'.format(name)

    print("start solve {} problem".format(name))
    result = solve_problem(name, order)
    print("end solve {} problem".format(name))
    print(result)

for name in names_FSP:
    order = order_FSP
    result_folder = 'EA-GA-{}'.format(name)

    print("start solve {} problem".format(name))
    result = solve_problem(name, order)
    print("end solve {} problem".format(name))
    print(result)