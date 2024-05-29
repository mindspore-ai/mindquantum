from typing import Tuple, Any, Dict, List, Callable
import dimod
import minorminer
from dimod.sampleset import SampleSet
from dimod.binary_quadratic_model import BinaryQuadraticModel
from dwave.system import DWaveSampler, ReverseAdvanceComposite
from dwave.embedding import unembed_sampleset, EmbeddedStructure
from dwave.system.warnings import WarningHandler, WarningAction
from functools import partial

from nen.Problem import Problem
from nen.Term import Quadratic
from nen.Solver.MetaSolver import SolverUtil

Sample = Any


class EmbeddingSampler:
    """ [summary] Composite Embedding and Sampling.
    """
    # DW_2000Q_6, Advantage_system6.1
    def __init__(self, sampler=DWaveSampler(solver='Advantage_system4.1')) -> None:
        # choose the sampler
        self.sampler = sampler
        self.reverse_sampler = ReverseAdvanceComposite(self.sampler)

        # set the parameters
        self.parameters = self.sampler.parameters.copy()
        self.parameters.update(chain_strength=[],
                               chain_break_method=[],
                               chain_break_fraction=[],
                               embedding_parameters=[],
                               return_embedding=[],
                               warnings=[],
                               )
        # set the properties
        self.properties = dict(child_properties=self.sampler.properties.copy())

        # get sampler structure
        self.target_structure = dimod.child_structure_dfs(self.sampler)
        self.target_nodelist, self.target_edgelist, self.target_adjacency = self.target_structure

    def embed(self, bqm: BinaryQuadraticModel) -> Tuple[Any, Any]:
        """embed [summary] embedd bqm model on target structure.

        Return  embedding: {variable: (qubit, qubit, ...)},
                bqm_embeded: {qubit: bias} and {(qubit, qubit): bias} and offset
        """
        # add self-loops to edgelist to handle singleton variables
        source_edgelist = list(bqm.quadratic) + [(v, v) for v in bqm.linear]
        # find embedding
        # embedding is {variable: (qubit, qubit, ...)} (single qubit or a chain of qubits)
        embedding = minorminer.find_embedding(source_edgelist, self.target_edgelist)

        if bqm and not embedding:
            raise ValueError("no embedding found")
        if not hasattr(embedding, 'embed_bqm'):
            embedding = EmbeddedStructure(self.target_edgelist, embedding)

        # bqm is {var: bias}, {(var, var): bias}
        # embeded bqm is {qubit: bias} and {(qubit, qubit): bias} (and offset)
        bqm_embedded = embedding.embed_bqm(bqm, chain_strength=None, smear_vartype=dimod.SPIN)

        return embedding, bqm_embedded

    @staticmethod
    def async_unembed_complete(response, embedding, bqm, warninghandler):
        """async_unembed_complete [summary] handle response from dwave leap cloud in async way and uuembed.

        Return  unembed samplset.
        """
        # warning handle
        warninghandler.chain_break(response, embedding)
        # unembed
        sampleset = unembed_sampleset(response, embedding, source_bqm=bqm,
                                      chain_break_method=None,
                                      chain_break_fraction=True,
                                      return_embedding=False)
        # warning handle
        if len(sampleset):
            warninghandler.issue("All samples have broken chains",
                                 func=lambda: (sampleset.record.chain_break_fraction.all(), None))
        if warninghandler.action is WarningAction.SAVE:
            sampleset.info.setdefault('warnings', []).extend(warninghandler.saved)

        return sampleset

    def sample(self, qubo: Dict[Tuple[str, str], float], **parameters) -> Tuple[SampleSet, float]:
        """sample [summary] sample qubo with paramters passed to sampler.
        """
        # convert qubo to bqm
        bqm = BinaryQuadraticModel.from_qubo(qubo)
        # embed, embedding: {var -> (qi)}
        # bqm_embedded: BinaryQuadraticModel({q -> bias}, {(q, q) -> offset}, constant, type}
        # to access lp/qp part: bqm_embedded.linear, bqm_embedded.quadratic
        embedding, bqm_embedded = self.embed(bqm)

        # warings handle
        warnings = WarningAction.IGNORE
        warninghandler = WarningHandler(warnings)
        warninghandler.chain_strength(bqm, embedding.chain_strength, embedding)
        warninghandler.chain_length(embedding)

        # initialize state for reversed anneal
        if 'initial_state' in parameters:
            # state: variable_name -> {0, 1}
            # Here, for each var: (u1, u2, ...), u1 = u2 = ... = state[var]
            state = parameters['initial_state']
            parameters['initial_state'] = {u: state[v] for v, chain in embedding.items() for u in chain}

        # sample on QPU
        # self.sampler.parameters['anneal_offsets'] = [-0.5]
        response = self.sampler.sample(bqm_embedded, **parameters)

        # unembed
        async_unembed = partial(EmbeddingSampler.async_unembed_complete,
                                embedding=embedding,
                                bqm=bqm,
                                warninghandler=warninghandler)
        sampleset = dimod.SampleSet.from_future(response, async_unembed)
        return sampleset, EmbeddingSampler.get_qpu_time(sampleset)

    def refinement_sample(self,
                          qubo: Dict[Tuple[str, str], float],
                          max_reverse_loop: int,
                          anneal_schedule: List[List[Any]],
                          select: Callable[[SampleSet], Sample],
                          dominate: Callable[[SampleSet, SampleSet], bool],
                          num_reads: int) -> List[SampleSet]:
        """refinement_sample [summary] sample with refinement.

        max_reverse_loop restricts the reverse annealing times,
        anneal_schedule defines the reverse annealing schedule but NOT first sampling,
        select should be a function select a sample from sampleset,
        dominate is a function for comparing which one is better between two samplesets,
        num_reads is the number of sampling times per sample.

        Return the sampleset sequence.
        """
        # convert qubo to bqm and embed
        bqm = BinaryQuadraticModel.from_qubo(qubo)
        embedding, bqm_embedded = self.embed(bqm)

        # warings handle
        warnings = WarningAction.IGNORE
        warninghandler = WarningHandler(warnings)
        warninghandler.chain_strength(bqm, embedding.chain_strength, embedding)
        warninghandler.chain_length(embedding)

        # prepare sampleset list
        sampleset_list: List[SampleSet] = []

        # first sample
        response = self.sampler.sample(bqm_embedded, num_reads=num_reads)
        async_unembed = partial(EmbeddingSampler.async_unembed_complete,
                                embedding=embedding,
                                bqm=bqm,
                                warninghandler=warninghandler)
        sampleset = dimod.SampleSet.from_future(response, async_unembed)
        # sampleset_list.append(sampleset)

        # reverse anneal
        for i in range(max_reverse_loop):
            print('reverse annealing #', i)
            # store old sampleset
            last_set = sampleset
            # select one from last set as initial state
            selected_state = select(sampleset)[0]
            initial_state = {u: selected_state[v] for v, chain in embedding.items() for u in chain}
            # sample
            response = self.reverse_sampler.sample(bqm_embedded,
                                                   num_reads=num_reads,
                                                   reinitialize_state=True,
                                                   initial_state=initial_state,
                                                   anneal_schedules=anneal_schedule)
            # unembed
            async_unembed = partial(EmbeddingSampler.async_unembed_complete,
                                    embedding=embedding,
                                    bqm=bqm,
                                    warninghandler=warninghandler)
            sampleset = dimod.SampleSet.from_future(response, async_unembed)
            sampleset_list.append(sampleset)
            # compare current set and last set
            if dominate(last_set, sampleset):
                sampleset_list.append(last_set)
                break
            else:
                sampleset_list.append(sampleset)
        return sampleset_list

    @staticmethod
    def sample_to_values(sample: Sample, variables: List[str]) -> Dict[str, bool]:
        """sample_to_values [summary] turn sample result to variable values mapping.
        """
        return {var: bool(sample[0][var]) for var in variables}

    @staticmethod
    def get_values_and_occurrence(sampleset: SampleSet, variables: List[str]) -> List[Tuple[Dict[str, bool], int]]:
        """get_values_and_occurrence [summary] get variables values and its occurrence from sampleset.
        """
        # prepare variables index mapping
        var_index: Dict[str, int] = {}
        for var in variables:
            var_index[var] = sampleset.variables.index(var)
        # get values
        results: List[Tuple[Dict[str, bool], int]] = []
        for sample in sampleset.record:
            values = {var: bool(sample[0][var_index[var]]) for var in variables}
            results.append((values, sample[2]))
        return results

    @staticmethod
    def get_values(sampleset: SampleSet, variables: List[str]) -> List[Dict[str, bool]]:
        """get_values [summary] get variables values from sampleset.
        """
        # prepare variables index mapping
        var_index: Dict[str, int] = {}
        for var in variables:
            var_index[var] = sampleset.variables.index(var)
        # get values
        results: List[Dict[str, bool]] = []
        for sample in sampleset.record:
            values = {var: bool(sample[0][var_index[var]]) for var in variables}
            results.append(values)
        return results

    @staticmethod
    def get_qpu_time(sampleset: SampleSet) -> float:
        """get_qpu_time [summary] get qpu time from sampleset
        The unit is seconds
        """
        return sampleset.info['timing']['qpu_sampling_time'] / 1000_000

    @staticmethod
    def select_by_energy(sampleset: SampleSet) -> Sample:
        """select_by_energy [summary] return the sampling with the lowest energy.
        """
        return sampleset.first

    @staticmethod
    def engery_compare(set_1: SampleSet, set_2: SampleSet) -> bool:
        """engery_compare [summary] return True if set_1 dominate set_2.
        """
        return set_1.first[1] < set_2.first[1]

    @staticmethod
    def weighted_energy(sampleset: SampleSet, problem: Problem, weights: Dict[str, float]) -> float:
        """weighted_energy [summary] calculate the weighted objectives value for all samples with their probablilty.
        """
        values_occurrences = EmbeddingSampler.get_values_and_occurrence(sampleset, problem.variables)
        all_occurence = sum([occ for _, occ in values_occurrences])
        result = 0.0
        for values, occurrence in values_occurrences:
            objectives = problem.evaluate(values).objectives
            obj = sum([objectives[ind] * weights[obj_name] for ind, obj_name in enumerate(problem.objectives_order)])
            result += (obj * (occurrence / all_occurence))
        return result

    @staticmethod
    def weighted_energy_compare(set_1: SampleSet, set_2: SampleSet,
                                problem: Problem, weights: Dict[str, float]
                                ) -> bool:
        """weighted_engery_compare [summary] calculate engery of the sampleset with each probability of the sample.
        """
        e_1 = EmbeddingSampler.weighted_energy(set_1, problem, weights)
        e_2 = EmbeddingSampler.weighted_energy(set_2, problem, weights)
        return e_1 < e_2

    @staticmethod
    def always_false(set_1: SampleSet, set_2: SampleSet) -> bool:
        """always_false [summary] False, make sure reverse annealing to the maximum loop.
        """
        return False

    @staticmethod
    def calculate_penalty(objective: Quadratic, constraint: Quadratic) -> float:
        """calculate_penalty [summary] calculate the min penalty according to given objective and constraint.
        """
        ha, Ja = SolverUtil.collect_quadratic_coefs(objective)
        hb, Jb = SolverUtil.collect_quadratic_coefs(constraint)
        delta_a = SolverUtil.max_gradient(ha + Ja)
        delta_b = SolverUtil.min_gradient(hb + Jb)
        return delta_a / delta_b
