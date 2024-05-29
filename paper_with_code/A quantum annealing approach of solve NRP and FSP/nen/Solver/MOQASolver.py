from functools import partial
from typing import Dict, List
from random import random

from dwave.system import ReverseAdvanceComposite, DWaveCliqueSampler, ReverseBatchStatesComposite, DWaveSampler, EmbeddingComposite

from nen.Term import Constraint, Quadratic
from nen.Problem import QP
from nen.Result import Result
from nen.Solver.MetaSolver import SolverUtil
from nen.Solver.EmbeddingSampler import EmbeddingSampler, SampleSet
from greedy import SteepestDescentSolver
from dimod.binary_quadratic_model import BinaryQuadraticModel


class MOQASolver:
    """ [summary] MOQASolver, stands for Multi-Objective Quantum Annealling Solver,
    it adopts random weighted sum of objectives and query on QPU for serveral times.

    Note that, the elapsed time just count qpu time, it does not include the pre/post-process.

    The Quantum Annealling Solver is implemeneted with D-Wave Leap,
    make sure the environment is configured successfully accordingly.
    """

    @staticmethod
    def solve(problem: QP, num_reads: int, sample_times: int = 1, 
              annealing_time: float = 20, postprocess: bool = True) -> Result:
        """solve [summary] solve qp, results are recorded in result.
        num_reads:
            read the num of solution from the solver result, not all the solution on the pareto.
        sample_times:
            run the Multi-objectives Quantum Annealing Algorithm times.
        """
        print("{} start MOQA to solve multi-objective problem!!!".format(problem.name))
        # scale objectives and get the basic
        basic_weights = SolverUtil.scaled_weights(problem.objectives)
        # sample for sample_times times
        samplesets: List[SampleSet] = []
        elapseds: List[float] = []
        # annealing_schedule = [[0.0, 0.0], [15, 0.5], [80, 0.5], [100, 1.0]]
        # annealing_schedule = [[0.0, 0.0], [48, 0.5], [annealing_time, 1.0]]
        for _ in range(sample_times):
            # generate random weights and calculate weighted sum obejctive
            weights = MOQASolver.random_normalized_weights(basic_weights)
            wso = Quadratic(linear=SolverUtil.weighted_sum_objective(problem.objectives, weights))
            # calculate the penalty and add constraints to objective with penalty
            penalty = EmbeddingSampler.calculate_penalty(wso, problem.constraint_sum)
            objective = Constraint.quadratic_weighted_add(1, penalty, wso, problem.constraint_sum)
            qubo = Constraint.quadratic_to_qubo_dict(objective)
            bqm = BinaryQuadraticModel.from_qubo(qubo)
            # Solve in QA
            sampler = EmbeddingSampler()
            # read num_reads from once sample, but some answer is duplicate,
            # so the number return less than num_reads
            sampleset, elapsed = sampler.sample(qubo, num_reads=num_reads, answer_mode='raw', 
                                                # anneal_schedule=annealing_schedule,
                                                annealing_time=annealing_time,
                                                )
            elapseds.append(elapsed)
            samplesets.append(sampleset)

            reverse_sampler = ReverseBatchStatesComposite(EmbeddingComposite(DWaveSampler()))
            r_schedule = [[0.0, 1.0], [5, 0.5], [15, 0.5], [20, 1.0]]
            r_sampleset, r_elapsed = reverse_sampler.sample(bqm, num_reads=num_reads,
                                                initial_states=sampleset,
                                                reinitialize_state=True,
                                                anneal_schedule=r_schedule)
            elapseds.append(elapsed)
            samplesets.append(r_sampleset)

            # if postprocess:
            #     local_solver = SteepestDescentSolver()
            #     start = SolverUtil.time()
            #     sampleset = local_solver.sample(bqm=bqm, initial_states=sampleset)
            #     end = SolverUtil.time()
            #     elapseds.append(start - end)
            #     samplesets.append(sampleset)
            

        # put samples into result
        result = Result(problem)
        for sampleset in samplesets:
            for values in EmbeddingSampler.get_values(sampleset, problem.variables):
                solution = problem.evaluate(values)
                result.wso_add(solution)
        # add into method result
        result.elapsed = sum(elapseds)
        for sampleset in samplesets:
            if 'solving info' not in result.info:
                result.info['solving info'] = [sampleset.info]
            else:
                result.info['solving info'].append(sampleset.info)
        # storage parameters
        result.info['sample_times'] = sample_times
        result.info['num_reads'] = num_reads
        result.iterations = sample_times
        result.total_num_anneals = sample_times * num_reads
        print("MOQA end!!!")
        return result

    @staticmethod
    def random_normalized_weights(basic_weights: Dict[str, float]) -> Dict[str, float]:
        """random_normalized_weights [summary] get a random weights with respect to the basic one,
        it is generated as random weights which sum is 1 and multiply with the basic weights.
        """
        random_weights = {k: random() for k in basic_weights}
        sum_weights = sum(random_weights.values())
        return {k: (random_weights[k] / sum_weights) * v for k, v in basic_weights.items()}

# # reverse anneal
# reverse_sampler = ReverseAdvanceComposite(sampler)
# bqm = BinaryQuadraticModel.from_qubo(qubo)
# embedding, bqm_embedded = sampler.embed(bqm)
# last_set = sampleset
# # select one from last set as initial state
# selected_state = EmbeddingSampler.select_by_energy(sampleset)[0]
# initial_state = {u: selected_state[v] for v, chain in embedding.items() for u in chain}
# # sample
# r_sampleset, r_elapsed = reverse_sampler.sample(bqm,
#                                                 num_reads=num_reads,
#                                                 reinitialize_state=True,
#                                                 initial_state=initial_state,
#                                                 anneal_schedules=anneal_schedule)
# # samplesets.append(r_sampleset)
# # compare current set and last set
# if EmbeddingSampler.engery_compare(last_set, r_sampleset):
#     samplesets.append(last_set)
# else:
#     samplesets.append(r_sampleset)
