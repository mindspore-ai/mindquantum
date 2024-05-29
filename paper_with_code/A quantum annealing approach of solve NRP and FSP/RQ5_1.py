import json
import numpy as np
from dimod import BinaryQuadraticModel
import sys
import os

# Add the path to MOQASolver.py
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import the solver from MOQASolver.py
from nen.Solver import MOQASolver

# Load data from the JSON file
with open('D:/projects/ICSE2024-main/ICSE2024-main/code/data/uClinux.json', 'r') as file:
    data = json.load(file)

# Function to test solver capabilities with a single large QUBO problem
def test_solver_capability(data, max_qubits=5000):
    # Create a BQM from the JSON data
    linear = {var: data['linear'][var] for var in data['variables']}
    quadratic = {(pair[0], pair[1]): data['quadratic'][pair] for pair in data['quadratic']}
    
    bqm = BinaryQuadraticModel(linear, quadratic, 0.0, 'BINARY')

    # Check the size of the problem
    size = len(bqm.variables)
    if size > max_qubits:
        print(f"Problem size {size} exceeds the maximum number of qubits {max_qubits}.")
        return

    # Attempt to solve the problem using the MOQASolver
    try:
        solver = MOQASolver()
        result = solver.solve(bqm)
        print(f"Successfully solved problem of size {size} qubits.")
        return result
    except Exception as e:
        print(f"Failed to solve problem of size {size} qubits: {e}")
        return None

# Call the function and print the results
solver_capability = test_solver_capability(data)
print(solver_capability)
