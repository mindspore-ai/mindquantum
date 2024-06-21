import csv
import math
import time
import random
import bisect
import individual
import numpy as np
import numpy.random
import multiprocessing
from math import pi
from tqdm import tqdm
from copy import deepcopy
from functools import partial
from deap import creator, base
from mindquantum.core.gates import X, Y, Z, H
from mindquantum.io import OpenQASM
from mindquantum.core.circuit import Circuit
from mindquantum.algorithm.library import amplitude_encoder
from mindquantum.simulator import Simulator
from deap.tools.emo import sortNondominated as sort_nondominated
from projectq.ops import (H, X, Y, Z, CNOT, CX, Rx,
                          Ry, Rz, Swap, SwapGate)

# 创建适应度类
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# 创建个体类，并指定适应度类作为其适应度属性
creator.create("Individual", list, fitness=creator.FitnessMax)


class Evolution:
    def __init__(self, triplets, number_of_qubits, connectivity="ALL", EMC=2.0, ESL=2.0, ):
        self.EMC = EMC
        self.ESL = ESL  # ESL 代表预期序列长度
        self.CMW = 0.2  # 电路变异权重
        self.batch = 2
        self.optimized = True
        self.triplets = triplets
        self.population_size = 5
        self.connectivity = connectivity
        self.number_of_generations = 1
        self.crossover_rate = 0.4
        self.number_of_qubits = number_of_qubits
        self.allowed_gates = [H, X, Y, Z, CNOT, CX,
                              Rx, Ry, Rz, Swap]
        self.permutation = random.sample(
            range(number_of_qubits), number_of_qubits)
        self.base_path = "D:\pycharm\projects\mindquantun_RGB/"

    def continue_code(self, population_size):
        next_pop = []
        file_name = f'D:\pycharm\projects\mindquantun_RGB/result/qasm/RGB_individual_10.qasm'
        qc_ind = OpenQASM().from_file(file_name)
        # print(OpenQASM().from_string(string))
        # qc_ind = QuantumCircuit.from_qasm_file(file_name)
        for _ in range(population_size - 1):
            cir = qc_ind
            decodeqc = self.from_circuit1(cir)  # 解码
            New_QC = self.mutate_ind(individual.Individual(decodeqc, self.number_of_qubits, connectivity="ALL", EMC=2.0,
                                                           ESL=2.0))  # 将类Individual实例化，并调用mutate变异功能，在类中对量子线路进行变异
            new_qc = New_QC.circuit  # 将变异后的量子门组合输出

            qc = self.to_circuit(new_qc)  # 对量子门组合再次进行编码
            print('new_qc', qc)
            new_ind = qc
            next_pop.append(new_ind)
        pop = next_pop
        return pop

    def runcircuit(self, ind, flattened, number_of_qubits):
        start_time = time.time()
        circuit = Circuit()
        flattened_array = np.array(flattened)
        desired_length = 2 ** 14
        padded = np.pad(flattened_array, (0, desired_length - len(flattened_array)), mode='constant')
        features = padded / np.max(np.abs(padded))

        encoder, parameterResolver = amplitude_encoder(features, number_of_qubits)
        sim = Simulator('mqvector', number_of_qubits)
        sim.apply_circuit(encoder, parameterResolver)
        sim.get_qs(True)
        # encoder.summary()
        # print(sim.get_qs(True))
        circuit += encoder
        circuit.barrier(show=True)
        circuit += ind
        ################################################
        circuit.measure_all()
        res = sim.sampling(circuit, pr=parameterResolver, shots=500)
        sampling = (res.data.get("00", 0) + res.data.get("01", 1) + res.data.get("10", 2) + res.data.get("11", 3))
        expectation = [res.data.get("01", 0) / sampling, res.data.get("01", 1) / sampling,
                       res.data.get("10", 2) / sampling, res.data.get("11", 3) / sampling]

        end_time1 = time.time()
        elapsed_time1 = end_time1 - start_time
        # print('\n runcircuit time\n', elapsed_time1)
        return expectation

    def generate_random_circuit(self, initialize=True):
        if initialize:
            p = 1 / 30
        else:
            p = 1 / self.ESL
        cir_length = numpy.random.geometric(p)
        produced_circuit = []
        quantum_circuit = Circuit()
        for i in range(cir_length):
            # Choose a gate to add from allowed_gates
            gate = random.choice(self.allowed_gates)
            if gate in [CNOT, CX, Swap]:
                # if gate to add is CNOT we need to choose control and target indices
                if self.connectivity == "ALL":
                    control, target = random.sample(
                        range(self.number_of_qubits), 2)
                else:
                    control, target = random.choice(self.connectivity)
                    print("control, target:", control, target)
                # TFG stands for Two Qubit Fixed Gate
                produced_circuit.append(("TFG", gate, control, target))

            elif gate in [H, X, Y, Z, ]:
                # choose the index to apply
                target = random.choice(range(self.number_of_qubits))
                # SFG stands for Single Qubit Fixed Gate
                produced_circuit.append(("SFG", gate, target))
            elif gate in [Rx, Ry, Rz]:
                # choose the index to apply the operator
                target = random.choice(range(self.number_of_qubits))
                # choose the rotation parameter between 0 and 2pi up to 3 significiant figures
                # !! We may change this significant figure things later on
                significant_figure = 2  # 有效数字
                parameter = round(pi * random.uniform(0, 2),
                                  significant_figure)
                produced_circuit.append(("SG", gate, target, parameter))
                quantum_circuit.rx(parameter, target)
            else:
                print("WHAT ARE YOU DOING HERE!!")
                print("GATE IS:", gate)

        return produced_circuit

    def to_circuit(self, produced_circuit):
        qc = Circuit()
        for op in produced_circuit:
            if op[0] == "TFG":
                # can be CNOT,CX,Swap,SwapGate
                if op[1] in [CX, CNOT]:
                    qc.x(op[2], op[3])
                elif op[1] in [Swap, SwapGate]:
                    qc.swap([op[2], op[3]])
                else:
                    print("Problem in to_circuit:", op[1])

            elif op[0] == "SFG":
                # can be H,X,Y,Z,T,T^d,S,S^d,sqrtX,sqrtXdagger
                if op[1] == H:
                    qc.h(op[2])
                elif op[1] == X:
                    qc.x(op[2])
                elif op[1] == Y:
                    qc.y(op[2])
                elif op[1] == Z:
                    qc.z(op[2])
                else:
                    print("Problem in to_circuit:", op[1])

            elif op[0] == "SG":
                # can be Rx,Ry,Rz
                if op[1] == Rx:
                    qc.rx(op[3], op[2])
                elif op[1] == Ry:
                    qc.ry(op[3], op[2])
                elif op[1] == Rz:
                    qc.rz(op[3], op[2])
                else:
                    print("Problem in to_circuit:", op[1])

        return qc

    def from_circuit1(self, encoded_circuit):
        decoded_circuit = []

        for gate in encoded_circuit.gates:
            if gate.name in ['X', 'CX']:  # Assuming 'X' corresponds to Pauli-X gate and 'CX' corresponds to CNOT gate
                decoded_circuit.append(("TFG", CX, gate.qubits[0], gate.qubits[1]))

            elif gate.name == 'Swap':
                decoded_circuit.append(("TFG", Swap, gate.qubits[0], gate.qubits[1]))

            elif gate.name in ['H', 'X', 'Y', 'Z', 'T', 'S']:
                decoded_circuit.append(("SFG", getattr(Circuit, gate.name.lower()), gate.qubits[0]))

            elif gate.name in ['RX', 'RY', 'RZ']:
                rotation_angle = gate.params['theta']
                if gate.name == 'RX':
                    decoded_circuit.append(("SG", Rx, gate.qubits[0], rotation_angle))
                elif gate.name == 'RY':
                    decoded_circuit.append(("SG", Ry, gate.qubits[0], rotation_angle))
                elif gate.name == 'RZ':
                    decoded_circuit.append(("SG", Rz, gate.qubits[0], rotation_angle))

        return decoded_circuit

    def new_individual(self):
        circuit = Circuit()
        for qubit in range(1, 14):
            circuit.x(0, qubit)
        circuit += self.to_circuit(self.generate_random_circuit(initialize=True))
        # circuit.barrier()
        return circuit

    def new_pop(self, toolbox):
        first_pop = []
        for i in range(self.population_size):
            circuit = self.new_individual()
            ind = creator.Individual(circuit, self.number_of_qubits)
            first_pop.append(ind)
            x_train_batch = random.sample(self.triplets, self.batch)
            ind.fitness.values = toolbox.evaluate(self, ind=circuit, features=x_train_batch)
            # (self, ind, features)
            string = OpenQASM().to_string(circuit)
            OpenQASM().from_string(string)
            first_circuit = fr'D:\pycharm_projects\paper1\ylh_mindspore\mindquantun_RGB\result/qasm/individual{i}.qasm'
            openqasm = OpenQASM()
            openqasm.to_file(first_circuit, circuit, version='2.0')
            # print(circuit)  # 打印Encoder
            # circuit.summary()  # 总结Encoder量子线路
            circuit.svg().to_file(
                fr"D:\pycharm_projects\paper1\ylh_mindspore\mindquantun_RGB/result/svg/filename{i}.svg")
        fitness_pop = [ind.fitness.values[0] for ind in first_pop]
        return first_pop, fitness_pop

    def triplet_loss(self, z_out):
        return np.abs(z_out[0] - z_out[2]) + np.abs(z_out[1] - z_out[3])

    def process_feature(self, ind, number_of_qubits, im):
        flattened = []
        for i, j, k in zip(im[0], im[1], im[2]):
            flattened.append(i)
            flattened.append(j)
        z_out1 = self.runcircuit(ind, flattened, self.number_of_qubits)
        z1_loss = self.triplet_loss(z_out1)

        flattened = []
        for i, j, k in zip(im[0], im[1], im[2]):
            flattened.append(k)
            flattened.append(i)
        z_out2 = self.runcircuit(ind, flattened, self.number_of_qubits)

        z2_loss = self.triplet_loss(z_out2)
        siam_loss = (z1_loss - z2_loss)
        consistancy_loss = (np.abs((z_out1[0] - z_out2[2])) + np.abs((z_out1[1] - z_out2[3])))
        loss = 0.9 * siam_loss + 0.1 * consistancy_loss
        return loss

    def evaluate(self, ind, features):
        start_time = time.time()
        num_processes = 3
        # num_processes = multiprocessing.cpu_count()

        with multiprocessing.Pool(processes=num_processes) as pool:
            partial_process_feature = partial(self.process_feature, ind, self.number_of_qubits)
            loss_list = list(tqdm(pool.imap(partial_process_feature, features), total=len(features)))

        loss = sum(loss_list)
        print('loss:', loss)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total fitness calculation time: {elapsed_time} seconds\nAverage fitness: {loss / len(features)}")
        evaluate = loss / len(features)
        fitness = 1 / (1 + evaluate)
        return [fitness]

    def mutate_ind(self, individual):
        '''
        随机对电路进行 12 次突变中的一次突变
        :param individual:种群中的单个个体
        :return:变异后的个体
        '''
        if individual.optimizedx:
            individual.parameter_mutation()
            return individual
        mutation_choice_fn = random.choice([
            individual.discrete_uniform_mutation,  # 创建离散均匀突变
            individual.continuous_uniform_mutation,  # 连续均匀突变
            individual.sequence_insertion,  # 序列插入
            individual.sequence_and_inverse_insertion,  # 序列和逆插入
            individual.insert_mutate_invert,  # 插入_变异_反转
            individual.sequence_deletion,  # 序列删除
            individual.sequence_replacement,  # 序列替换
            individual.sequence_swap,  # 序列交换_就算是交叉了
            individual.sequence_scramble,  # 序列重组
            individual.permutation_mutation,  # 置换_变异
            individual.clean,  # 清除
            individual.move_gate  # 移动量子门
        ])
        mutation_choice_fn()
        print('mutation_choice_fn', mutation_choice_fn)
        return individual

    def mate(self, parent1, parent2, toolbox):
        return parent1.cross_over(parent2, toolbox), parent2.cross_over(parent1, toolbox)

    def select_parents(self, pop, num_parents):
        """
        从种群中选择父辈。
        """
        parents = []
        fitness_values = [ind.fitness.values[0] for ind in pop]
        sorted_indices = sorted(range(len(fitness_values)), key=lambda k: fitness_values[k],
                                reverse=True) 
        for i in range(num_parents):
            parents.append(pop[sorted_indices[i]])
        return parents


    def select(self, pop, to_carry, fitness_of_pop):
        """
        ranks = sort_nondominated(pop, len(pop))
        to_carry = len(ranks[0])
        individuals = self.select(pop, to_carry)
        next_generation = individuals
        crossover = len(pop) // 13
        current_rank = 1
        N = len(pop)-to_carry - 2*crossover
        """
        normalized_fitness_values = [fitness / sum(fitness_of_pop) for fitness in fitness_of_pop]
        cumulative_fitness_values = [sum(normalized_fitness_values[:i + 1]) for i in
                                     range(len(normalized_fitness_values))]
        selected_individuals = []
        selected_ind_fit = []
        for i in range(to_carry):
            random_number = random.random()
            index = bisect.bisect_left(cumulative_fitness_values, random_number)
            selected_individuals.append(pop[index])
            selected_ind_fit.append(fitness_of_pop[index])
        return selected_individuals, selected_ind_fit

    def tournament_selection(self, pop, to_carry, fitness_of_pop, tournament_size):
        selected_individuals = []
        selected_ind_fit = []

        for i in range(to_carry):
            tournament = random.sample(list(zip(pop, fitness_of_pop)), tournament_size)
            tournament.sort(key=lambda x: x[1], reverse=True)
            winner = tournament[0][0]
            winner_fitness = tournament[0][1]
            selected_individuals.append(winner)
            selected_ind_fit.append(winner_fitness)

        return selected_individuals, selected_ind_fit

    def elitism_selection(self, pop, to_carry, fitness_of_pop):
        best_index = np.argmax(fitness_of_pop)
        selected_individuals = [pop[best_index]] * to_carry
        selected_ind_fit = [fitness_of_pop[best_index]] * to_carry
        return selected_individuals, selected_ind_fit

    def crossover(self, pop, crossover, toolbox):
        """
        执行交叉操作。
        """
        mate_individual = []
        num_cross = int(crossover)
        num_parents = 4
        parents = self.select_parents(pop, num_parents)
        for _ in range(num_cross):
            parent1, parent2 = random.sample(parents, 2)
            circuit1 = self.from_circuit1(parent1.circuit) 
            individuals1 = individual.Individual(circuit1, self.number_of_qubits,
                                                 connectivity="ALL", EMC=2.0, ESL=2.0)
            circuit2 = self.from_circuit1(parent1.circuit) 
            individuals2 = individual.Individual(circuit2, self.number_of_qubits,
                                                 connectivity="ALL", EMC=2.0, ESL=2.0)
            child1, child2 = self.mate(individuals1, individuals2, toolbox)  

            mate_individual.append(child1)
            mate_individual.append(child2)
        return mate_individual

    def mutate_individuals(self, ranks, N, toolbox, current_rank=1):
        """This function takes a nested of list of individuals, sorted according to
        their ranks and chooses a random individual. Each individual's probability
        to be chosen is proportional to exp(-(individual's rank)).
        """
        # FIXME what is the indexing of current_rank, clarify
        L = len(ranks)
        T = 0
        # Calculate the summation of exponential terms
        for i in range(L):
            T += math.exp(-current_rank - i)

        cps = []
        cp_fitness = []
        list_indexes = []
        element_indexes = []

        for _ in range(N):
            # Choose a random number between 0 and T
            random_number = random.uniform(0, T)
            # Find out which sublist this random number corresponds to
            # Let's say T = exp(-1) + exp(-2) + exp(-3) + exp(-4) and if our
            # random number is between 0 and 1, than it belongs to first list,
            # ranks[0]. If it is between exp(-1) and exp(-1)+exp(-2) than it
            # belongs to second list etc.

            # FIXME Refactor list index and everything else
            list_index = -1
            right_border = 0
            for i in range(L):
                right_border += math.exp(-current_rank - i)
                if random_number <= right_border:
                    list_index = i
                    break
            if list_index == -1:
                list_index = L - 1
            left_border = right_border - math.exp(-current_rank - list_index)
            # Now, we will find out which index approximately the chosen number
            # corresponds to by using a simple relation.
            element_index = math.floor(
                len(ranks[list_index]) * (random_number - left_border) / (right_border - left_border))

            while len(ranks[list_index]) == 0:
                list_index += 1
                if len(ranks[list_index]) != 0:
                    element_index = random.choice(range(len(ranks[list_index])))

            if element_index >= len(ranks[list_index]):
                element_index = -1

            # Copies the individual
            print('list_index', list_index)
            print('element_index', element_index)
            cp = deepcopy(ranks[list_index][element_index])
            print('type(cp)', type(cp))  # <class 'deap.creator.Individual'>
            print('cpcp', cp.circuit)
            if isinstance(cp.circuit, list):
                individuals = cp.circuit
            else:
                circuit = self.from_circuit1(cp.circuit)
                individuals = individual.Individual(circuit, self.number_of_qubits, connectivity="ALL", EMC=2.0,
                                                    ESL=2.0)
            cp_mutat = toolbox.mutate_ind(individuals)
            new_circuit = cp_mutat.circuit
            print('cp.circuit', new_circuit)  # produce_circuit
            circuit = self.to_circuit(new_circuit)
            x_train_batch = random.sample(self.triplets, self.batch)
            cp_class = creator.Individual(circuit, self.number_of_qubits)
            cp_class.fitness.values = toolbox.evaluate(self, ind=circuit, features=x_train_batch)
            # cps.append(circuit)
            cps.append(cp_class)
            list_indexes.append(list_index)
            element_indexes.append(element_index)
            cp_fitness.append(cp.fitness.values[0])
        return cps, cp_fitness

    def select_and_evolve(self, pop, fitness_of_pop, toolbox):
        """
        Apply nondominated sorting to rank individuals and select individuals for the next generation,
        then mutate and perform crossover to generate the next generation.
        """
        # Apply nondominated sorting to rank individuals
        ranks = sort_nondominated(pop, len(pop))
        to_carry = len(ranks[0])  # Number of individuals to carry to the next generation
        print('to_carry', to_carry)
        print(f'before: select pop\n{pop},\nfitness_of_pop\n {fitness_of_pop}')
        # print('len(pop)',len(pop))
        # Select individuals from nondominated ranks
        individuals, individuals_fit = self.select(pop, to_carry, fitness_of_pop)
        print('len_select_individuals', len(individuals_fit))
        print(f'after: select pop\n{individuals},\nfitness_of_pop\n {individuals_fit}')
        # Initialize the next generation and fitness values with the selected individuals
        next_generation, fitness_values = individuals, individuals_fit

        # Determine number of individuals for mutation and crossover
        crossover = int(len(pop) * self.crossover_rate)
        N = len(pop) - to_carry - 2 * crossover
        print(f'before mutate individuals: pop\n{next_generation},\n fitness_values\n '
              f'{fitness_values}')
        # Mutate individuals and calculate their fitness values
        mutated_individuals, mutated_fitness_values = self.mutate_individuals(ranks, N, toolbox, current_rank=1)
        print('len_select_mutated_individuals', len(mutated_individuals))
        print(f'after mutate individuals: mutate pop\n{mutated_individuals},\n mutated_fitness_values\n '
              f'{mutated_fitness_values}')
        # Add mutated individuals to the next generation
        next_generation.extend(mutated_individuals)
        fitness_values.extend(mutated_fitness_values)

        mate_individuals = self.crossover(pop, crossover, toolbox)  # individual class

        mate_fitness_values = []
        x_train_batch = random.sample(self.triplets, self.batch)

        for child in mate_individuals:
            print(type(child))
            childs = self.to_circuit(child.circuit)
            # Display the circuits in the next generation
            child.fitness.values = toolbox.evaluate(self, ind=childs, features=x_train_batch)
            mate_fitness_values.append(child.fitness.values[0])

        next_generation.extend(mate_individuals)
        fitness_values.extend(mate_fitness_values)
        print('len_next_generation', len(next_generation))
        return next_generation, fitness_values

    def evolution(self, toolbox):
        initial_pop, fitness_pop = self.new_pop(toolbox)

        next_generation = initial_pop
        sorted_next_generation = []
        sorted_fitness_values = []
        for i in tqdm(range(self.number_of_generations)):
            print('number_of_generations:', i)
            next_generation, fitness_values = self.select_and_evolve(next_generation, fitness_pop, toolbox)
            print(f'next_generation:{next_generation}\n fitness_values{fitness_values}')
            combined_data = list(zip(next_generation, fitness_values))
            sorted_data = sorted(combined_data, key=lambda x: x[1], reverse=True)
            sorted_next_generation, sorted_fitness_values = zip(*sorted_data)

            filename = f'{self.base_path}/result/csv/fitness_values_ranks_{i}.csv'
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Index', 'Fitness Value'])
                    writer.writerow([j + 1, fitness])
            count = 1
            for ind in sorted_next_generation:
                circuit = self.to_circuit(ind.circuit)
                first_circuit = f'{self.base_path}/result/qasm/next_generation_{i + 1}_{count}.qasm'
                OpenQASM().to_file(first_circuit, circuit, version='2.0')
                circuit.svg().to_file(f'{self.base_path}/result/png/next_generation_{i + 1}_{count}.svg')
                count += 1
        print(f'sorted_next_generation{sorted_next_generation},\n sorted_fitness_values{sorted_fitness_values}')
        return sorted_next_generation, sorted_fitness_values
