from functools import partial
from typing import Dict, List
from random import random

from qiskit_aer import AerSimulator
from qiskit_aer.aerprovider import AerProvider
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.problems import QuadraticProgram
from qiskit_algorithms import QAOA
from qiskit.primitives import Estimator, Sampler
from qiskit_algorithms.optimizers import COBYLA

from nen.Term import Constraint, Quadratic
from nen.Problem import QP
from nen.Result import Result
from nen.Solver.SimuEmbeddingSampler import EmbeddingSampler1
from nen.Solver.MetaSolver import SolverUtil
from dimod import SampleSet

import time

class MOQASolverQAOA:
    """ MOQASolverQAOA, stands for Multi-Objective Quantum Annealling Solver using QAOA,
    it adopts random weighted sum of objectives and query on QPU for several times.

    Note that, the elapsed time just counts QPU time, it does not include the pre/post-process.

    The Quantum Annealling Solver is implemented with Qiskit,
    make sure the environment is configured successfully accordingly.
    """

    @staticmethod
    def solve(problem: QP, num_reads: int, sample_times: int = 1, 
              p: int = 1, postprocess: bool = True) -> Result:
        """solve the multi-objective problem using QAOA.
        num_reads: number of solution samples to read.
        sample_times: number of times to run the Multi-objectives Quantum Annealing Algorithm.
        """
        print("{} start MOQA to solve multi-objective problem using QAOA!!!".format(problem.name))
        # Scale objectives and get the basic
        basic_weights = SolverUtil.scaled_weights(problem.objectives)
        # Sample for sample_times times
        samplesets: List[SampleSet] = []
        elapseds: List[float] = []

        for _ in range(sample_times):
            try:
            # Generate random weights and calculate weighted sum objective
                weights = MOQASolverQAOA.random_normalized_weights(basic_weights)
                wso = Quadratic(linear=SolverUtil.weighted_sum_objective(problem.objectives, weights))
                # Calculate the penalty and add constraints to objective with penalty
                penalty = EmbeddingSampler1.calculate_penalty(wso, problem.constraint_sum)
                objective = Constraint.quadratic_weighted_add(1, penalty, wso, problem.constraint_sum)

                qp = QuadraticProgram()

                for var in problem.variables:
                    qp.binary_var(var)
        
            # print("Variables:", problem.variables)
            # print("Linear terms:", objective.linear)
            # print("Quadratic terms:", objective.quadratic)
        
            # 设置目标函数
                linear_terms = {i: coeff for i, coeff in objective.linear.items()}
                quadratic_terms = {(i, j): coeff for (i, j), coeff in objective.quadratic.items()}


            # Convert to QUBO
            # qp = QuadraticProgram()
            # for var in problem.variables:
            #     qp.binary_var(var)
            # for (i, j), coeff in objective.linear.items():
            #     qp.minimize(linear={i: coeff})
            # for (i, j), coeff in objective.quadratic.items():
            #     qp.minimize(quadratic={(i, j): coeff})
                
            # Convert to QUBO format
                conv = QuadraticProgramToQubo()
                qubo = conv.convert(qp)

            # Configure QAOA
            # Aer = AerProvider()
            # backend = Aer.get_backend('qasm_simulator')
                backend = AerSimulator()
                print(backend)
                optimizer = COBYLA()
                sampler = Sampler()
                qaoa = QAOA(optimizer=optimizer, reps=p, sampler=sampler)
                qaoa_solver = MinimumEigenOptimizer(qaoa)
            
                # Solve the QUBO problem
                start_time = time.time()
                result = qaoa_solver.solve(qubo)
                sampleset = result.samples
                # Assume time_taken attribute is available (it might need to be computed differently)
                elapsed = time.time()-start_time

                elapseds.append(elapsed)
                samplesets.append(sampleset)

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
        """get a random weights with respect to the basic one,
        it is generated as random weights which sum is 1 and multiply with the basic weights.
        """
        random_weights = {k: random() for k in basic_weights}
        sum_weights = sum(random_weights.values())
        return {k: (random_weights[k] / sum_weights) * v for k, v in basic_weights.items()}