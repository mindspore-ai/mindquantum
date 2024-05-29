import sys
sys.path.append('D:/projects/ICSE2024-main/ICSE2024-main/code')

import random
from typing import Dict, List, Tuple

import numpy as np
from dimod import BinaryQuadraticModel
# from dwave.system import DWaveSampler
from hybrid import State, min_sample, EnergyImpactDecomposer
from jmetal.core.solution import BinarySolution

from nen.Solver import MOQASolver, JarSolver, SASolver, QAWSOSolver
from nen.Solver.GASolver import GASolver
from nen.Solver.SAQPSolver import SAQPSolver
from nen.Solver.SOQASolver import SOQASolver
from nen.Term import Constraint, Quadratic
from nen.Problem import QP
from nen.Result import Result, MethodResult, ProblemResult
from nen.Solver.MetaSolver import SolverUtil
from nen.Solver.EmbeddingSampler import EmbeddingSampler, SampleSet
from dwave.system import DWaveSampler
from nen.DescribedProblem import DescribedProblem
import dimod
import minorminer
from greedy import SteepestDescentSolver
from tabu import TabuSampler
from dimod.binary_quadratic_model import BinaryQuadraticModel

class HybridSolver:
    """ [summary] HybridSolver, stands for Multi-Objective Quantum Annealling with HybridSolver,
    hybrid—quantum-classical hybrid; typically one or more classical algorithms run on the problem
    while outsourcing to a quantum processing unit (QPU) parts of the problem where it benefits most.

    The Quantum Annealling Solver is implemeneted with D-Wave Leap,
    make sure the environment is configured successfully accordingly.
    """
    @staticmethod
    def solve(problem: QP, num_reads: int, sub_size: int,
              sample_times: int = 1, rate: float = 1.0,
              steps: int = 1, **parameters) -> Result:
        """solve [summary] solve multi-objective qp, results are recorded in result.
        """
        print("{} start Hybrid Solver to solve multi-objective problem!!!".format(problem.name))
        # scale objectives and get the basic
        basic_weights = SolverUtil.scaled_weights(problem.objectives)
        # sample for sample_times times
        elapseds: List[float] = []
        solution_list: List[BinarySolution] = []
        # annealing_schedule = [[0.0, 0.0], [10, 0.5], [110, 0.5], [120, 1.0]]
        # initial_solutions = [SASolver.QPrandomSolution(problem) for _ in range(num_reads)]
        for _ in range(sample_times):
            # generate random weights and calculate weighted sum obejctive
            weights = MOQASolver.random_normalized_weights(basic_weights)
            wso = Quadratic(linear=SolverUtil.weighted_sum_objective(problem.objectives, weights))
            # calculate the penalty and add constraints to objective with penalty
            penalty = EmbeddingSampler.calculate_penalty(wso, problem.constraint_sum)
            objective = Constraint.quadratic_weighted_add(1, penalty, wso, problem.constraint_sum)
            qubo = Constraint.quadratic_to_qubo_dict(objective)
            # convert qubo to bqm
            bqm = BinaryQuadraticModel.from_qubo(qubo)
            t = 0
            initial_solution_list = [SASolver.QPrandomSolution(problem) for _ in range(num_reads)]

            '''Decomposer'''
            s1 = SolverUtil.time()
            states = HybridSolver.Decomposer(sub_size=sub_size, bqm=bqm, variables_num=bqm.num_variables,
                                            )    
            length = len(states)
            states = states[:int(length * rate)]
            e1 = SolverUtil.time() 
            '''Sampler'''
            sampleset, runtime = HybridSolver.Sampler_global(problem=problem, states=states, bqm=bqm, 
                                                             initial_solution_list=initial_solution_list, 
                                                             steps= steps, num_reads=num_reads,
                                                              # anneal_schedule=annealing_schedule, 
                                                              **parameters)
            HybridSolver.Composer(problem=problem, samplesets=[sampleset], num_reads=num_reads, solution_list=solution_list)
            # assert len(solution_list) == num_reads
            t = e1 - s1 + runtime
            elapseds.append(t)
            # print(i)
            
        # put samples into result
        result = Result(problem)
        for solution in solution_list:
            result.wso_add(solution)
        # add into method result
        result.elapsed = sum(elapseds)
        # storage parameters
        result.info['sample_times'] = sample_times
        result.info['num_reads'] = num_reads
        result.iterations = sample_times
        result.total_num_anneals = num_reads * sample_times
        print("{} Hybrid Solver end!!!".format(problem.name))
        return result

    @staticmethod
    def solve_parallel(problem: QP, num_reads: int, sub_size: int, maxEvaluations: int, populationSize: int,
              objectiveOrder: List[str], resultFolder: str, problem_result_path: str,
              sample_times: int = 1, rate: float = 1.0, **parameters) -> Result:
        """solve [summary] solve multi-objective qp, results are recorded in result.
        """
        print("{} start Hybrid Solver to solve multi-objective problem!!!".format(problem.name))
        # scale objectives and get the basic
        basic_weights = SolverUtil.scaled_weights(problem.objectives)
        # sample for sample_times times
        elapseds: List[float] = []
        solution_list: List[BinarySolution] = []
        # annealing_schedule = [[0.0, 0.0], [10, 0.5], [110, 0.5], [120, 1.0]]
        for i in range(sample_times):
            t = 0
            initial_solutions = [SASolver.randomSolution(problem) for _ in range(num_reads)]
            # generate random weights and calculate weighted sum obejctive
            weights = MOQASolver.random_normalized_weights(basic_weights)
            wso = Quadratic(linear=SolverUtil.weighted_sum_objective(problem.objectives, weights))
            # calculate the penalty and add constraints to objective with penalty
            penalty = EmbeddingSampler.calculate_penalty(wso, problem.constraint_sum)
            objective = Constraint.quadratic_weighted_add(1, penalty, wso, problem.constraint_sum)
            qubo = Constraint.quadratic_to_qubo_dict(objective)
            # convert qubo to bqm
            bqm = BinaryQuadraticModel.from_qubo(qubo)

            '''Decomposer'''
            s1 = SolverUtil.time()
            states = HybridSolver.Decomposer(sub_size=sub_size, bqm=bqm, variables_num=bqm.num_variables,
                                             )    
            length = len(states)
            states = states[:int(length * rate)]
            e1 = SolverUtil.time()
            '''Sampler'''
            subsamplesets, runtime = HybridSolver.Sampler(states=states, num_reads=num_reads, 
                                                          initial_solutions=initial_solutions, 
                                                            # anneal_schedule=annealing_schedule, 
                                                          **parameters)
            '''Composer'''
            s2 = SolverUtil.time()
            HybridSolver.Composer(problem=problem, subsamplesets=subsamplesets, num_reads=num_reads,
                                  solution_list=solution_list)
            e2 = SolverUtil.time()
            t = e1 - s1 + e2 - s2 + runtime
            elapseds.append(t)
            print(i)

        '''NSGA-II'''
        HybridSolver.NSGAII(populationSize=populationSize, maxEvaluations=maxEvaluations, problem=problem,
                            time=sum(elapseds), solution_list=solution_list, iterations=sample_times,
                            objectiveOrder=objectiveOrder, resultFolder=resultFolder, 
                            problem_result_path=problem_result_path)
        '''Selection'''
        # solution_list.sort(key=lambda x: (x.constraints[0], np.dot(x.objectives, list(weights.values()))))
        solution_list.sort(key=lambda x: (x.constraints[0], x.objectives))
        # solution_list.sort(key=lambda x: np.dot(x.objectives, list(weights.values())))
        # solution_list = solution_list[:sample_times * num_reads]

        # put samples into result
        result = Result(problem)
        for solution in solution_list:
            result.wso_add(solution)
        # add into method result
        result.elapsed = sum(elapseds)
        # storage parameters
        result.info['sample_times'] = sample_times
        result.info['num_reads'] = num_reads
        result.iterations = sample_times
        result.total_num_anneals = num_reads * sample_times
        print("{} Hybrid Solver end!!!".format(problem.name))
        return result

    @staticmethod
    def solve_rates(problem: QP, num_reads: int, sub_size: int, maxEvaluations: int, populationSize: int,
              objectiveOrder: List[str], resultFolder: str, problem_result_path: str,
              rates: List[float], sample_times: int = 1, **parameters) -> Dict[float, Result]:
        """solve [summary] solve multi-objective qp, results are recorded in result.
        """
        print("{} start Hybrid Solver to solve multi-objective problem!!!".format(problem.name))
        # scale objectives and get the basic
        basic_weights = SolverUtil.scaled_weights(problem.objectives)
        # sample for sample_times times
        solution_lists: Dict[float, List[BinarySolution]] = {}
        rate_elapseds: Dict[float, float] = {}
        # annealing_schedule = [[0.0, 0.0], [10, 0.5], [110, 0.5], [120, 1.0]]
        for rate in rates:
            solution_list: List[BinarySolution] = []
            elapseds: List[float] = []
            for _ in range(sample_times):
                t = 0
                # generate random weights and calculate weighted sum obejctive
                weights = MOQASolver.random_normalized_weights(basic_weights)
                wso = Quadratic(linear=SolverUtil.weighted_sum_objective(problem.objectives, weights))
                # calculate the penalty and add constraints to objective with penalty
                penalty = EmbeddingSampler.calculate_penalty(wso, problem.constraint_sum)
                objective = Constraint.quadratic_weighted_add(1, penalty, wso, problem.constraint_sum)
                qubo = Constraint.quadratic_to_qubo_dict(objective)
                # convert qubo to bqm
                bqm = BinaryQuadraticModel.from_qubo(qubo)

                '''Decomposer'''
                s1 = SolverUtil.time()
                states = HybridSolver.Decomposer(sub_size=sub_size, bqm=bqm, variables_num=bqm.num_variables,
                                                )    
                length = len(states)
                states = states[:int(length * rate)]
                e1 = SolverUtil.time()
                '''Sampler'''
                subsamplesets, runtime = HybridSolver.Sampler(states=states, num_reads=num_reads, 
                                                                # anneal_schedule=annealing_schedule, 
                                                                **parameters)
                '''Composer'''
                s2 = SolverUtil.time()
                HybridSolver.Composer(problem=problem, subsamplesets=subsamplesets, num_reads=num_reads,
                                    solution_list=solution_list)
                e2 = SolverUtil.time()
                t = e1 - s1 + e2 - s2 + runtime
                elapseds.append(t)
            solution_lists[rate] = solution_list
            rate_elapseds[rate] = sum(elapseds)
        '''NSGA-II'''
        nsgaii_soluion_list: List[BinarySolution] = []
        HybridSolver.NSGAII(populationSize=populationSize, maxEvaluations=maxEvaluations, problem=problem,
                            time=sum(elapseds), solution_list=nsgaii_soluion_list, iterations=sample_times,
                            objectiveOrder=objectiveOrder, resultFolder=resultFolder, 
                            problem_result_path=problem_result_path)
        '''Selection'''
        results: Dict[float, Result] = {}
        for rate in rates:
            rate_solution_list: List[BinarySolution] = solution_lists[rate]
            rate_solution_list += nsgaii_soluion_list
            rate_solution_list.sort(key=lambda x: (x.constraints[0], x.objectives))
            result = Result(problem)
            for solution in rate_solution_list:
                result.wso_add(solution)
            result.elapsed = rate_elapseds[rate]
            result.total_num_anneals = num_reads * sample_times
            results[rate] = result

        print("{} Hybrid Solver end!!!".format(problem.name))
        return results

    @staticmethod
    def single_solve(problem: QP, num_reads: int, sub_size: int, maxEvaluations: int, populationSize: int,
              objectiveOrder: List[str], resultFolder: str, problem_result_path: str, weights: Dict[str, float],
              sample_times: int = 1, rate: float = 1.0, **parameters) -> Result:
        print("{} start Hybrid Solver to solve single-objective problem!!!".format(problem.name))
        result = Result(problem)
        

        # generate random weights and calculate weighted sum obejctive
        wso = Quadratic(linear=SolverUtil.weighted_sum_objective(problem.offset_objectives, weights))
        # calculate the penalty and add constraints to objective with penalty
        penalty = EmbeddingSampler.calculate_penalty(wso, problem.constraint_sum)
        objective = Constraint.quadratic_weighted_add(1, penalty, wso, problem.constraint_sum)
        qubo = Constraint.quadratic_to_qubo_dict(objective)
        # convert qubo to bqm
        bqm = BinaryQuadraticModel.from_qubo(qubo)

        for _ in range(sample_times):
            solution_list: List[BinarySolution] = []
            '''Decomposer'''
            s1 = SolverUtil.time()
            states = HybridSolver.Decomposer(sub_size=sub_size, bqm=bqm, variables_num=bqm.num_variables)
            length = int(len(states) * rate)
            states = states[:length]
            e1 = SolverUtil.time()
            '''Sampler'''
            subsamplesets, runtime = HybridSolver.Sampler(states=states, num_reads=num_reads, **parameters)
            '''Composer'''
            s2 = SolverUtil.time()
            HybridSolver.Composer(problem=problem, subsamplesets=subsamplesets, num_reads=num_reads, 
                                solution_list=solution_list)
            e2 = SolverUtil.time()
            '''GA'''
            # x0 = QAWSOSolver.best_solution(solution_list, problem, weights)
            t = e1 - s1 + e2 - s2 + runtime
            # HybridSolver.SA(t_max=t_max, t_min=t_min, num_reads=num_reads, weight=weights,
            #                 time=e2-s1, problem=problem, solution_list=solution_list, alpha=alpha)
            HybridSolver.GA(populationSize=populationSize, maxEvaluations=maxEvaluations, problem=problem,
                            time=t, solution_list=solution_list, weights=weights, iterations=1,
                            objectiveOrder=objectiveOrder, resultFolder=resultFolder, problem_result_path=problem_result_path)
            '''Selection'''
            # solution_list.sort(key=lambda x: (x.constraints[0], np.dot(x.objectives, list(weights.values()))))
            res = QAWSOSolver.best_solution(solution_list, problem, weights)
            result.wso_add(res)
            result.elapsed += t
        
        # storage parameters
        print("{} Hybrid Solver end!!!".format(problem.name))
        return result

    @staticmethod
    def single_solve_rate(problem: QP, num_reads: int, sub_size: int, maxEvaluations: int, populationSize: int,
                        objectiveOrder: List[str], resultFolder: str, problem_result_path: str, weights: Dict[str, float],
                        rates: List[float], sample_times: int = 1, **parameters) -> Dict[float, Result]:
        print("{} start Hybrid Solver to solve single-objective problem!!!".format(problem.name))
        results: Dict[float, Result] = []
        solution_lists: Dict[float, List[BinarySolution]] = {}
        rate_elapseds: Dict[float, float] = {}

        # generate random weights and calculate weighted sum obejctive
        wso = Quadratic(linear=SolverUtil.weighted_sum_objective(problem.offset_objectives, weights))
        # calculate the penalty and add constraints to objective with penalty
        penalty = EmbeddingSampler.calculate_penalty(wso, problem.constraint_sum)
        objective = Constraint.quadratic_weighted_add(1, penalty, wso, problem.constraint_sum)
        qubo = Constraint.quadratic_to_qubo_dict(objective)
        # convert qubo to bqm
        bqm = BinaryQuadraticModel.from_qubo(qubo)

        for _ in range(sample_times):
            for rate in rates:
                solution_list: List[BinarySolution] = []
                '''Decomposer'''
                s1 = SolverUtil.time()
                states = HybridSolver.Decomposer(sub_size=sub_size, bqm=bqm, variables_num=bqm.num_variables)
                length = int(len(states) * rate)
                states = states[:length]
                e1 = SolverUtil.time()
                '''Sampler'''
                subsamplesets, runtime = HybridSolver.Sampler(states=states, num_reads=num_reads, **parameters)
                '''Composer'''
                s2 = SolverUtil.time()
                HybridSolver.Composer(problem=problem, subsamplesets=subsamplesets, num_reads=num_reads, 
                                    solution_list=solution_list)
                e2 = SolverUtil.time()
                t = e1 - s1 + e2 - s2 + runtime
                solution_lists[rate] = solution_list
                rate_elapseds[rate] = t
            '''GA'''
            ga_solution_list: List[BinarySolution] = []
            HybridSolver.GA(populationSize=populationSize, maxEvaluations=maxEvaluations, problem=problem,
                            time=t, solution_list=ga_solution_list, weights=weights, iterations=1,
                            objectiveOrder=objectiveOrder, resultFolder=resultFolder, problem_result_path=problem_result_path)
            '''Selection'''
            for rate in rates:
                rate_solution_list: List[BinarySolution] = solution_lists[rate]
                rate_solution_list += ga_solution_list
                res = QAWSOSolver.best_solution(rate_solution_list, problem, weights)
                results[rate].wso_add(res)
                results[rate].elapsed += rate_elapseds[rate]
        
        # storage parameters
        print("{} Hybrid Solver end!!!".format(problem.name))
        return results

    @staticmethod
    def Decomposer_decrease(sub_size: int, bqm: BinaryQuadraticModel, variables_num: int) -> List[State]:
        """Decomposer [summary]
        State is a dict subclass and usually contains at least three keys:
        samples: SampleSet, problem: BinaryQuadraticModel and subproblem: BinaryQuadraticModel.
        """
        length = variables_num
        if length < sub_size:
            sub_size = length
        state0 = State.from_sample(min_sample(bqm), bqm)
        sampler=DWaveSampler(solver='Advantage_system6.1')
        states = []
        embedding = False
        while length > 0:
            while not embedding:
                decomposer = EnergyImpactDecomposer(size=sub_size, rolling=True, rolling_history=1.0, traversal='pfs')
                state0 = decomposer.run(state0).result()
                sub_bqm = state0.subproblem
                source_edgelist = list(sub_bqm.quadratic) + [(v, v) for v in sub_bqm.linear]
                target_structure = dimod.child_structure_dfs(sampler)
                target_nodelist, target_edgelist, target_adjacency = target_structure
                # find embedding
                # embedding is {variable: (qubit, qubit, ...)} (single qubit or a chain of qubits)
                embedding = minorminer.find_embedding(source_edgelist, target_edgelist)
                if not embedding:
                    sub_size -= 100
                # print("subsize: ", sub_size)
            states.append(state0)
            length -= sub_size
        return states

    @staticmethod
    def Decomposer(sub_size: int, bqm: BinaryQuadraticModel, variables_num: int) -> List[State]:
        """Decomposer [summary]
        State is a dict subclass and usually contains at least three keys:
        samples: SampleSet, problem: BinaryQuadraticModel and subproblem: BinaryQuadraticModel.
        """
        length = variables_num
        if length < sub_size:
            sub_size = length
        state0 = State.from_sample(min_sample(bqm), bqm)
        states = []
        decomposer = EnergyImpactDecomposer(size=sub_size, rolling=True, rolling_history=1.0, traversal='pfs')
        while length > 0:
            state0 = decomposer.run(state0).result()
            states.append(state0)
            length -= sub_size
        return states

    @staticmethod
    def Sampler(states: List[State], num_reads: int, **parameters) -> Tuple[List[SampleSet], float]:
        """ just solve subproblems and return the answer
        """
        elapseds = 0.0
        subsamplesets = []
        for state in states:
            pro_bqm = state.subproblem

            qubo, offset = pro_bqm.to_qubo()
            sampler = EmbeddingSampler()
            sampleset, elapsed = sampler.sample(qubo, num_reads=num_reads, answer_mode='raw', 
                                                # postprocess='sampling', 
                                                **parameters)
            local_solver = SteepestDescentSolver()
            sampleset = local_solver.sample(bqm=pro_bqm, initial_states=sampleset)
            subsamplesets.append(sampleset)
            elapseds += elapsed
        return subsamplesets, elapseds
    
    @staticmethod
    def Sampler_global(problem: QP, states: List[State], bqm: BinaryQuadraticModel, 
                       initial_solution_list: List[BinarySolution], num_reads: int, 
                       steps: int, **parameters) -> Tuple[SampleSet, float]:
        """ solve the subproblems and update the solution
        """
        elapseds = 0.0
        converge_step = 0

        # calculate the variable index
        var_index_in_problem: Dict[str, int] = {}
        for var in problem.variables:
            var_index_in_problem[var] = problem.variables.index(var)
        length = len(var_index_in_problem)
        for var in problem.artificial_list:
            var_index_in_problem[var] = problem.artificial_list.index(var) + length

        # calculate initial energy
        best_solution_list = []
        for solution in initial_solution_list:
            best_solution_list.append({var: int(solution.variables[0][problem.all_variables_index[var]]) for ind, var in enumerate(bqm.variables)})
        best_solution_energy = bqm.energies(best_solution_list)

        step = 0
        while converge_step < 3 and step < steps:
            new_solution = []
            for solution in initial_solution_list:
                new_solution.append(solution.__copy__())
            for state in states:
                pro_bqm = state.subproblem
                qubo, offset = pro_bqm.to_qubo()
                sampler = EmbeddingSampler()
                sampleset, elapsed = sampler.sample(qubo, num_reads=num_reads, answer_mode='raw', 
                                                    # postprocess='sampling', 
                                                    **parameters)
                elapseds += elapsed
                HybridSolver.update_solution(problem=problem, sampleset=sampleset, bqm=bqm, 
                                             solution_list=new_solution, num_reads=num_reads)
                
                # postprocess
                local_solver = SteepestDescentSolver()
                variables = []
                for var in bqm.variables:
                    variables.append(var)
                s = SolverUtil.time()
                samples = []
                for k in range(num_reads):
                    dic = {}
                    for index, var in enumerate(variables):
                        dic[var] = np.int8(new_solution[k].variables[0][var_index_in_problem[var]])
                    samples.append(dic)
                sampleset = dimod.SampleSet.from_samples_bqm(samples_like=dimod.as_samples(samples), 
                                                            bqm=bqm)
                sampleset = local_solver.sample(bqm=bqm, initial_states=sampleset)
                elapseds += (SolverUtil.time() - s)

                HybridSolver.update_solution(problem=problem, sampleset=sampleset, bqm=bqm, 
                                             solution_list=new_solution, num_reads=num_reads)
                
            better_flag = True

            # # objective
            # for i in range(len(new_best_solution.objectives)):
            #     if new_best_solution.objectives[i] > best_solution.objectives[i]:
            #         better_flag = False
            #         converge_step += 1
            #         break
            # # sum 
            # w = [weights[s] for s in problem.objectives_order]
            # new_v = sum([new_best_solution.objectives[i] * w[i] for i in range(problem.objectives_num)])
            # v = sum([best_solution[k].objectives[i] * w[i] for i in range(problem.objectives_num)])
            # if new_v > v:
            #     better_flag = False
            #     converge_step += 1

            # energy of bqm
            for k in range(num_reads):
                new_values = {var: new_solution[k].variables[0][var_index_in_problem[var]] for ind, var in enumerate(bqm.variables)}
                new_solution_energy = bqm.energy(new_values)
                if new_solution_energy >= best_solution_energy[k]:
                    better_flag = False
                if better_flag:
                    best_solution_energy[k] = new_solution_energy
                    initial_solution_list[k] = new_solution[k]
            step += 1
        variables = []
        for var in bqm.variables:
            variables.append(var)
        
        samples = []
        for k in range(num_reads):
            dic = {}
            for index, var in enumerate(variables):
                dic[var] = np.int8(initial_solution_list[k].variables[0][var_index_in_problem[var]])
            samples.append(dic)
        sampleset = dimod.SampleSet.from_samples_bqm(samples_like=dimod.as_samples(samples), 
                                                     bqm=bqm)
        
        local_solver = SteepestDescentSolver()
        # local_solver = TabuSampler()
        start = SolverUtil.time()
        sampleset = local_solver.sample(bqm=bqm, initial_states=sampleset, large_sparse_opt=True)
        end = SolverUtil.time()
        elapseds += (end - start)

        return sampleset, elapseds
    
    def classic_solver(problem: QP, num_reads: int, bqm: BinaryQuadraticModel, 
                       initial_solutions: List[BinarySolution]) -> Tuple[SampleSet, float]:
        variables = problem.variables
        for var in problem.artificial_list:
            variables.append(var)
        
        samples = []
        for k in range(num_reads):
            dic = {}
            for index, var in enumerate(variables):
                dic[var] = np.int8(initial_solutions[k].variables[0][index])
            samples.append(dic)
        # samples = dimod.as_samples([{var: initial_solutions[k].variables[0][index] \
        #                             for index, var in enumerate(variables)} \
        #                             for k in range(num_reads)])
        sampleset = dimod.SampleSet.from_samples_bqm(samples_like=dimod.as_samples(samples), 
                                                     bqm=bqm)
        local_solver = SteepestDescentSolver()
        start = SolverUtil.time()
        sampleset = local_solver.sample(bqm=bqm, initial_states=sampleset)
        end = SolverUtil.time()
        return sampleset, end - start

    @staticmethod
    def update_solution(problem: QP, sampleset: SampleSet, 
                        solution_list: List[BinarySolution], num_reads: int, 
                        # weights: Dict[str, float]
                        bqm: BinaryQuadraticModel) -> bool:
        """ Bring the value of the variable of the subproblem into the original problem, 
        and if the new solution obtained has lower energy, update the solution
        """
        var_index_in_problem: Dict[str, int] = {}
        for var in problem.variables:
            var_index_in_problem[var] = problem.variables.index(var)
        length = len(var_index_in_problem)
        for var in problem.artificial_list:
            var_index_in_problem[var] = problem.artificial_list.index(var) + length
        var_index_in_sampleset: Dict[str, int] = {}
        for var in sampleset.variables:
            var_index_in_sampleset[var] = sampleset.variables.index(var)

        
        new_solution: BinarySolution = None
        for k in range(num_reads):
            subsample = sampleset.record[k]
            solution = solution_list[k]
        # for subsample in sampleset.record:
            new_solution = solution.__copy__()
            for var in sampleset.variables:
                assert var in var_index_in_problem
                new_solution.variables[0][var_index_in_problem[var]] = subsample[0][var_index_in_sampleset[var]]
            new_solution = problem.evaluate_solution(new_solution)
            m_better: bool = True

            # for i in range(len(new_solution.objectives)):
            #     if new_solution.objectives[i] > solution_list[k].objectives[i]:
            #         m_better = False
            #         break
            # w = [weights[s] for s in problem.objectives_order]
            # new_v = sum([new_solution.objectives[i] * w[i] for i in range(problem.objectives_num)])
            # v = sum([solution_list[k].objectives[i] * w[i] for i in range(problem.objectives_num)])
            # if new_v > v:
            #     m_better = False
            # energy of bqm
            new_values = {var: new_solution.variables[0][var_index_in_problem[var]] for ind, var in enumerate(bqm.variables)}
            new_energy = bqm.energy(new_values)
            best_values = {var: solution.variables[0][var_index_in_problem[var]] for ind, var in enumerate(bqm.variables)}
            best_energy = bqm.energy(best_values)
            if new_energy >= best_energy:
                m_better = False
            
            if m_better:
                solution_list[k] = new_solution
        return True

    @staticmethod
    def Composer(problem: QP, samplesets: List[SampleSet], num_reads: int, solution_list: List[BinarySolution]) -> bool:
        values_array: Dict[str, List[int]] = {}
        for var in problem.variables:
            values_array[var] = []
        for subsampleset in samplesets:
            var_index: Dict[str, int] = {}
            for var in subsampleset.variables:
                if var not in problem.variables:
                    continue
                var_index[var] = subsampleset.variables.index(var)
            # assert len(subsampleset.record) == num_reads
            for subsample in subsampleset.record:
                for var in subsampleset.variables:
                    if var not in problem.variables:
                        continue
                    values_array[var].append(subsample[0][var_index[var]])
        for var in problem.variables:
            if len(values_array[var]) == 0:
                values_array[var] += [0.0 for _ in range(len(samplesets[0].record))]
        pos = 0
        while pos < num_reads:
            values = {var: bool(values_array[var][pos]) for var in problem.variables}
            solution = problem.evaluate(values)
            solution_list.append(solution)
            pos += 1
        return True

    @staticmethod
    def NSGAII(populationSize: int, maxEvaluations: int, problem: QP,
               objectiveOrder: List[str], resultFolder: str, problem_result_path: str,
               time: float, solution_list: List[BinarySolution], iterations: int,
               ):
        JarSolver.solve(
        solver_name='NSGAII', config_name='tmp_config',
        problem=problem.name, objectiveOrder=objectiveOrder, iterations=iterations,
        populationSize=populationSize, maxEvaluations=maxEvaluations,
        crossoverProbability=0.8, mutationProbability=(1 / problem.variables_num),
        resultFolder=resultFolder, methodName='nsgaii', exec_time=time * 100,
        )
        # load results
        ea_result = MethodResult('nsgaii', problem_result_path, problem)
        ea_result.load(evaluate=True, single_flag=True, total_num_anneals=populationSize)
        for result in ea_result.results:
            for solution in result.solution_list:
                solution_list.append(solution)

    @staticmethod
    def SA(t_max: float, t_min: float, num_reads: int, alpha: float,
           time: float, problem: QP, solution_list: List[BinarySolution], weight: Dict[str, float]):
        result = SAQPSolver.solve(problem=problem, num_reads=num_reads, weights=weight, if_embed=False,
                                t_max=t_max, t_min=t_min, alpha=alpha, exec_time=time)
        for solution in result.solution_list:
            solution_list.append(solution)

    @staticmethod
    def GA(populationSize: int, maxEvaluations: int, problem: QP, weights: Dict[str, float],
           objectiveOrder: List[str], resultFolder: str, problem_result_path: str,
            time: float, solution_list: List[BinarySolution], iterations: int):
        JarSolver.solve(
        solver_name='GASingle', config_name='tmp_config',
        problem=problem.name, objectiveOrder=objectiveOrder, iterations=iterations,
        populationSize=populationSize, maxEvaluations=maxEvaluations, weights=weights,
        crossoverProbability=0.8, mutationProbability=(1 / problem.variables_num),
        resultFolder=resultFolder, methodName='ga', exec_time=time * 100
        )
        # load results
        ea_result = MethodResult('ga', problem_result_path, problem)
        ea_result.load(evaluate=True, single_flag=True, total_num_anneals=populationSize)
        for result in ea_result.results:
            for solution in result.solution_list:
                solution_list.append(solution)

