import hubo_solver
import numpy as np
import json

base_path = r'/home/laiguohui24/demo_data/{}.json'
file_path = base_path.format(0)

print(dir(hubo_solver))


n_iterations = 1000
batch_size = 10
backend = "GPU" 

print("Initializing BSB solver on GPU...")
data = json.load(open(file_path, 'r'))
J = data['J']
coeff = data['coeff']
        
bsb_solver = hubo_solver.BSB(J, coeff, n_iterations, batch_size, be = backend)


print("Initializing spins")
bsb_solver.initial()

print("Running solver...")
bsb_solver.update() 


solution = bsb_solver.get_solution()
energy = bsb_solver.get_energy(solution)


