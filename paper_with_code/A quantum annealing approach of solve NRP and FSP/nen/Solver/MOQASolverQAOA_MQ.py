import time
import numpy as np
from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import QubitOperator
from mindquantum.simulator import Simulator
from mindquantum.algorithm.qaia import QAIA
from scipy.optimize import minimize
from typing import List, Dict
from random import random

from nen.Term import Constraint, Quadratic
from nen.Problem import QP
from nen.Result import Result
from nen.Solver.SimuEmbeddingSampler import EmbeddingSampler1
from nen.Solver.MetaSolver import SolverUtil

class MOQASolverQAOA_MQ:
    """MOQASolverQAOA, stands for Multi-Objective Quantum Annealling Solver using QAOA,
    it adopts random weighted sum of objectives and query on QPU for several times.

    Note that, the elapsed time just counts QPU time, it does not include the pre/post-process.

    The Quantum Annealling Solver is implemented with MindQuantum,
    make sure the environment is configured successfully accordingly.
    """

    @staticmethod
    def solve(problem: QP, num_reads: int, sample_times: int = 1, 
              p: int = 1, postprocess: bool = True) -> Result:
        """Solve the multi-objective problem using QAOA.
        num_reads: number of solution samples to read.
        sample_times: number of times to run the Multi-objectives Quantum Annealing Algorithm.
        """
        print("{} start MOQA to solve multi-objective problem using QAOA!!!".format(problem.name))
        # Scale objectives and get the basic
        basic_weights = SolverUtil.scaled_weights(problem.objectives)
        # Sample for sample_times times
        samplesets = []
        elapseds = []

        for _ in range(sample_times):
            try:
                # Generate random weights and calculate weighted sum objective
                weights = MOQASolverQAOA_MQ.random_normalized_weights(basic_weights)
                wso = Quadratic(linear=SolverUtil.weighted_sum_objective(problem.objectives, weights))
                # Calculate the penalty and add constraints to objective with penalty
                penalty = EmbeddingSampler1.calculate_penalty(wso, problem.constraint_sum)
                objective = Constraint.quadratic_weighted_add(1, penalty, wso, problem.constraint_sum)

                num_qubits = len(problem.variables)
                hamiltonian = QubitOperator()
                for (i, j), coeff in objective.quadratic.items():
                    hamiltonian += QubitOperator(f'Z{i} Z{j}', coeff)
                for i, coeff in objective.linear.items():
                    hamiltonian += QubitOperator(f'Z{i}', coeff)

                # Create QAOA ansatz using QAIA
                qaia = QAIA(hamiltonian, p)
                simulator = Simulator('mqvector', num_qubits)

                # Define the optimization problem
                def objective_function(params):
                    energy = simulator.get_expectation(qaia.circuit, hamiltonian, params)
                    return energy.asnumpy()[0]

                initial_params = np.random.rand(qaia.circuit.n_params)
                start_time = time.time()
                result = minimize(objective_function, initial_params, method='COBYLA')
                elapsed = time.time() - start_time

                # Simulate the final circuit with optimized parameters
                optimized_params = result.x
                simulator.apply_circuit(qaia.circuit, optimized_params)
                samples = simulator.sampling(shots=num_reads)

                elapseds.append(elapsed)
                samplesets.append(samples)

            except MemoryError as e:
                print(f"MemoryError encountered during sample {_}: {str(e)}")
                print(f"Dataset type: {type(problem)}")
                print(f"Memory block shape: (variables: {len(problem.variables)}, quadratic terms: {len(objective.quadratic)})")
                print(f"Memory allocation: {e}")
                break

        # Put samples into result
        result = Result(problem)
        for sampleset in samplesets:
            for sample in sampleset:
                values = {var: sample[var] for var in problem.variables}
                solution = problem.evaluate(values)
                result.wso_add(solution)
        # Add into method result
        result.elapsed = sum(elapseds)
        result.info['sample_times'] = sample_times
        result.info['num_reads'] = num_reads
        result.iterations = sample_times
        result.total_num_anneals = sample_times * num_reads
        print("MOQA with QAOA end!!!")
        return result

    @staticmethod
    def random_normalized_weights(basic_weights: Dict[str, float]) -> Dict[str, float]:
        """Get random weights with respect to the basic one,
        it is generated as random weights which sum to 1 and multiply with the basic weights.
        """
        random_weights = {k: random() for k in basic_weights}
        sum_weights = sum(random_weights.values())
        return {k: (random_weights[k] / sum_weights) * v for k, v in basic_weights.items()}
