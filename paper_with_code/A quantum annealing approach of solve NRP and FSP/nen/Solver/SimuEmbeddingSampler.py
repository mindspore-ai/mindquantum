from typing import Tuple, Any, Dict, List, Callable
import numpy as np
from simanneal import Annealer
from dimod import BinaryQuadraticModel, SampleSet, Vartype
from nen.Problem import Problem
from nen.Term import Quadratic
from nen.Solver.MetaSolver import SolverUtil

Sample = Any

class QUBOAnnealer(Annealer):
    def __init__(self, state, bqm):
        self.bqm = bqm
        super().__init__(state)

    def move(self):
        i = np.random.randint(len(self.state))
        self.state[i] = 1 - self.state[i]

    def energy(self):
        energy = self.bqm.energy(self.state)
        return energy

class EmbeddingSampler1:
    """EmbeddingSampler1 class implementing simulated annealing on a classical platform."""
    def __init__(self):
        self.parameters = {}
        self.properties = {}
        self.target_structure = None

    def embed(self, bqm: BinaryQuadraticModel) -> Tuple[Any, Any]:
        return None, bqm

    def sample(self, qubo: Dict[Tuple[str, str], float], **parameters) -> Tuple[SampleSet, float]:
        bqm = BinaryQuadraticModel.from_qubo(qubo)
        initial_state = [np.random.choice([0, 1]) for _ in range(len(bqm.variables))]

        annealer = QUBOAnnealer(initial_state, bqm)
        annealer.steps = parameters.get('steps', 10000)
        annealer.Tmax = parameters.get('Tmax', 10.0)
        annealer.Tmin = parameters.get('Tmin', 0.1)
        annealer.copy_strategy = parameters.get('copy_strategy', 'slice')

        state, energy = annealer.anneal()

        samples = [{var: state[i] for i, var in enumerate(bqm.variables)}]
        energies = [energy]
        num_occurrences = [1]

        sampleset = SampleSet.from_samples(samples, vartype=Vartype.BINARY, energy=energies)
        sampleset.record.num_occurrences = num_occurrences
        return sampleset, annealer.steps / 1000.0

    def refinement_sample(self, qubo: Dict[Tuple[str, str], float], max_reverse_loop: int,
                          anneal_schedule: List[List[Any]], select: Callable[[SampleSet], Sample],
                          dominate: Callable[[SampleSet, SampleSet], bool], num_reads: int) -> List[SampleSet]:
        bqm = BinaryQuadraticModel.from_qubo(qubo)
        initial_state = [np.random.choice([0, 1]) for _ in range(len(bqm.variables))]

        sampleset_list = []

        for i in range(max_reverse_loop):
            print('annealing #', i)
            annealer = QUBOAnnealer(initial_state, bqm)
            annealer.steps = anneal_schedule[i][1]
            annealer.Tmax = anneal_schedule[i][0][0]
            annealer.Tmin = anneal_schedule[i][0][1]
            annealer.copy_strategy = 'slice'

            state, energy = annealer.anneal()

            samples = [{var: state[i] for i, var in enumerate(bqm.variables)}]
            energies = [energy]
            num_occurrences = [1]

            sampleset = SampleSet.from_samples(samples, vartype=Vartype.BINARY, energy=energies)
            sampleset.record.num_occurrences = num_occurrences
            sampleset_list.append(sampleset)

            if i > 0 and dominate(sampleset_list[-2], sampleset):
                break

            initial_state = select(sampleset)[0]

        return sampleset_list

    @staticmethod
    def sample_to_values(sample: Sample, variables: List[str]) -> Dict[str, bool]:
        return {var: bool(sample[var]) for var in variables}

    @staticmethod
    def get_values_and_occurrence(sampleset: SampleSet, variables: List[str]) -> List[Tuple[Dict[str, bool], int]]:
        var_index = {var: sampleset.variables.index(var) for var in variables}
        results = []
        for sample in sampleset.record:
            values = {var: bool(sample[0][var_index[var]]) for var in variables}
            results.append((values, sample[2]))
        return results

    @staticmethod
    def get_values(sampleset: SampleSet, variables: List[str]) -> List[Dict[str, bool]]:
        var_index = {var: sampleset.variables.index(var) for var in variables}
        results = []
        for sample in sampleset.record:
            values = {var: bool(sample[0][var_index[var]]) for var in variables}
            results.append(values)
        return results

    @staticmethod
    def get_qpu_time(sampleset: SampleSet) -> float:
        return sampleset.info['timing']['qpu_sampling_time'] / 1000_000

    @staticmethod
    def select_by_energy(sampleset: SampleSet) -> Sample:
        return sampleset.first

    @staticmethod
    def engery_compare(set_1: SampleSet, set_2: SampleSet) -> bool:
        return set_1.first[1] < set_2.first[1]

    @staticmethod
    def weighted_energy(sampleset: SampleSet, problem: Problem, weights: Dict[str, float]) -> float:
        values_occurrences = EmbeddingSampler1.get_values_and_occurrence(sampleset, problem.variables)
        all_occurence = sum([occ for _, occ in values_occurrences])
        result = 0.0
        for values, occurrence in values_occurrences:
            objectives = problem.evaluate(values).objectives
            obj = sum([objectives[ind] * weights[obj_name] for ind, obj_name in enumerate(problem.objectives_order)])
            result += (obj * (occurrence / all_occurence))
        return result

    @staticmethod
    def weighted_energy_compare(set_1: SampleSet, set_2: SampleSet, problem: Problem, weights: Dict[str, float]) -> bool:
        e_1 = EmbeddingSampler1.weighted_energy(set_1, problem, weights)
        e_2 = EmbeddingSampler1.weighted_energy(set_2, problem, weights)
        return e_1 < e_2

    @staticmethod
    def always_false(set_1: SampleSet, set_2: SampleSet) -> bool:
        return False

    @staticmethod
    def calculate_penalty(objective: Quadratic, constraint: Quadratic) -> float:
        ha, Ja = SolverUtil.collect_quadratic_coefs(objective)
        hb, Jb = SolverUtil.collect_quadratic_coefs(constraint)
        delta_a = SolverUtil.max_gradient(ha + Ja)
        delta_b = SolverUtil.min_gradient(hb + Jb)
        return delta_a / delta_b