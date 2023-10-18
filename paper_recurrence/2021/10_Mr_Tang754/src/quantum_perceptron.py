import random
import numpy as np
from itertools import combinations
from mindquantum import Circuit, UN, H, Z, X, Measure
from mindquantum.simulator import Simulator


class perceptron:

    @staticmethod
    def gate_combinations():

        n_qubit = 4
        # possible
        z_combinations = []
        control_z_combinations = []
        control_2_z_combinatons = []
        control_3_z_combinations = []

        ## 排列组合
        qubits_list = np.arange(0, n_qubit, 1)
        z_all = [1] * n_qubit
        control_z_all = list(combinations(qubits_list, 2))
        control_2_z_all = list(combinations(qubits_list, 3))
        control_3_z_all = list(combinations(qubits_list, 4))

        for i in range(2**len(z_all)):
            z_combination = list('{:04b}'.format(i))
            z_combinations.append(list(map(int, z_combination)))

        for i in range(2**len(control_z_all)):
            control_z_combination = list('{:06b}'.format(i))
            control_z_combination = list(map(int, control_z_combination))
            for index, possible in enumerate(control_z_combination):
                if possible != 0:
                    control_z_combination[index] = list(control_z_all[index])
            control_z_combinations.append(control_z_combination)

        for i in range(2**len(control_2_z_all)):
            control_2_z_combinaton = list('{:04b}'.format(i))
            control_2_z_combinaton = list(map(int, control_2_z_combinaton))
            for index, possible in enumerate(control_2_z_combinaton):
                if possible != 0:
                    control_2_z_combinaton[index] = list(
                        control_2_z_all[index])
            control_2_z_combinatons.append(control_2_z_combinaton)

        for i in range(2**len(control_3_z_all)):
            control_3_z_combination = list('{:01b}'.format(i))
            control_3_z_combination = list(map(int, control_3_z_combination))
            for index, possible in enumerate(control_3_z_combination):
                if possible != 0:
                    control_3_z_combination[index] = list(
                        control_3_z_all[index])
            control_3_z_combinations.append(control_3_z_combination)

        return z_combinations, control_z_combinations, control_2_z_combinatons, control_3_z_combinations

    @staticmethod
    def random_dic(dicts):
        dict_key_ls = list(dicts.keys())
        random.shuffle(dict_key_ls)
        new_dic = {}
        for key in dict_key_ls:
            new_dic[key] = dicts.get(key)
        return new_dic

    @staticmethod
    def training_data(input_dataset):

        # generate positve data
        w_target = np.array([1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1])

        number = np.array([6, 9, 10, 11, 14])
        possible = list(combinations(number, 2))

        positive_data = []
        for index in range(16):
            data_copy = w_target.copy()
            if index == 6 or index == 9 or index == 10 or index == 11 or index == 14:  # 少一个
                data_copy[index] = 1
                positive_data.append(tuple(data_copy))

        for index in range(16):
            data_copy = w_target.copy()
            if index == 5 or index == 7 or index == 13 or index == 15:  # 多一个
                data_copy[index] = 0
                positive_data.append(tuple(data_copy))

        more = [5, 7, 13, 15]
        less = [6, 9, 10, 11, 14]
        for i in more:  # 多一个少一个
            for j in less:
                data_copy = w_target.copy()
                data_copy[i] = 0
                data_copy[j] = 1
                positive_data.append(tuple(data_copy))

        element = [[5, 7], [5, 13], [7, 15], [13, 15]]
        for i in element:  # 多两个
            data_copy = w_target.copy()
            for j in i:
                data_copy[j] = 0
            positive_data.append(tuple(data_copy))

        for i in possible:
            data_copy = w_target.copy()
            data_copy[list(i)[0]] = 1
            data_copy[list(i)[1]] = 1
            positive_data.append(tuple(data_copy))
        positive_data.append(tuple(w_target))

        #put the label to the data
        global traing_label

        input_dataset2 = input_dataset.copy()

        training_positive_circuits = {}
        traing_negtive_circuits = {}
        traing_label = {}
        traing_negtive_label = {}

        for i in positive_data:
            training_positive_circuits[i] = input_dataset2[i]
            traing_label[i] = 1
            del input_dataset2[i]

        traing_datas_negtive_key = random.sample(input_dataset2.keys(), 500)
        for key in traing_datas_negtive_key:
            traing_negtive_circuits[key] = input_dataset2[key]
            traing_negtive_label[key] = 0

        training_positive_circuits.update(traing_negtive_circuits)
        traing_label.update(traing_negtive_label)

        training_data = perceptron.random_dic(training_positive_circuits)

        return positive_data, training_data, traing_label
        #return positive_data

    @staticmethod
    def Quantum_Hypergraph_cicuits(z_position, control_z_positon,
                                   control_2_z_positon, control_3_z_positon,
                                   flag):

        n_qubits = 4  # 设定量子比特数为3
        #sim = Simulator('mqvector', n_qubits)        # 使用mqvector模拟器，命名为sim
        circuit = Circuit()  # 初始化量子线路，命名为circuit

        if flag == True:
            circuit += UN(H, n_qubits)

        for index, position in enumerate(z_position):
            if position == 1:
                circuit += Z.on([index])

        for index, position in enumerate(control_z_positon):
            if isinstance(position, list):
                target = int(position[1])
                control = int(position[0])
                circuit += Z.on([target], [control])

        for index, position in enumerate(control_2_z_positon):
            if isinstance(position, list):
                target = int(position[-1])
                control = [int(i) for i in position]
                del control[-1]
                circuit += Z.on([target], control)

        for index, position in enumerate(control_3_z_positon):
            if isinstance(position, list):
                circuit += Z.on([3], [0, 1, 2])
        return circuit

    @staticmethod
    def active_function(input_circuits, weight_circuits):

        n_qubits = 5  # 设定量子比特数为3
        sim = Simulator('mqvector', n_qubits)  # 使用mqvector模拟器，命名为sim
        circuit = Circuit()  # 初始化量子线路，命名为circuit

        circuit += input_circuits
        circuit += weight_circuits
        circuit += UN(H, n_qubits - 1)
        circuit += UN(X, n_qubits - 1)
        circuit += X.on([4], [0, 1, 2, 3])
        circuit += Measure().on(4)

        res = sim.sampling(circuit, shots=1000)
        counts = res.data
        key_number = []
        for i in counts.keys():
            key_number.append(i)
        if len(key_number) == 2:
            one_weight = counts["1"] / (counts["0"] + counts["1"])
        elif len(key_number) == 1:
            for key in counts.keys():
                if key == '1':
                    one_weight = 1
                else:
                    one_weight = 0

        if one_weight > 0.5:  ### Threshoud value
            pred = 1
        else:
            pred = 0
        return pred

    @staticmethod
    def update(in_vector, w_vector, pred, input_dataset, weight_dataset):

        w_vector = w_vector
        if pred == 1 and traing_label[in_vector] == 1:
            return w_vector

        elif pred == 0 and traing_label[in_vector] == 0:
            return w_vector

        elif pred == 1 and traing_label[in_vector] == 0:  ## move far away

            w_forward = w_vector
            for i in range(80):
                w_vector = list(w_vector)
                for i in range(16):
                    if w_vector[i] == in_vector[i]:
                        w_vector[i] = random.choice([0, 1])
                w_vector = tuple(w_vector)
                input_circuits = input_dataset[in_vector]
                try:
                    weight_circuits = weight_dataset[w_vector]
                except:
                    continue

                preds = perceptron.active_function(input_circuits,
                                                   weight_circuits)
                if preds == 0:
                    break
            try:
                weight_circuits = weight_dataset[w_vector]
            except:
                w_vector = w_forward
            return w_vector

        elif pred == 0 and traing_label[in_vector] == 1:  ## move far away

            w_forward = w_vector
            for i in range(80):
                w_vector = list(w_vector)
                for i in range(16):
                    if w_vector[i] != in_vector[i]:
                        w_vector[i] = random.choice([0, 1])

                w_vector = tuple(w_vector)
                input_circuits = input_dataset[in_vector]

                try:
                    weight_circuits = weight_dataset[w_vector]
                except:
                    continue

                preds = perceptron.active_function(input_circuits,
                                                   weight_circuits)
                if preds == 1:
                    break
            try:
                weight_circuits = weight_dataset[w_vector]
            except:
                w_vector = w_forward
            return w_vector

    @staticmethod
    def data_circuits(z_combinations, control_z_combinations,
                      control_2_z_combinatons, control_3_z_combinations):

        data_circuits_input = []
        data_circuits_weight = []
        for c_3_z_gate in control_3_z_combinations:
            for c_2_z_gate in control_2_z_combinatons:
                for c_z_gate in control_z_combinations:
                    for z_gate in z_combinations:
                        data_circuits_input.append(
                            perceptron.Quantum_Hypergraph_cicuits(
                                z_gate, c_z_gate, c_2_z_gate, c_3_z_gate,
                                True))
                        data_circuits_weight.append(
                            perceptron.Quantum_Hypergraph_cicuits(
                                z_gate, c_z_gate, c_2_z_gate, c_3_z_gate,
                                False))
        return data_circuits_input, data_circuits_weight

    @staticmethod
    def circuit_state(data_circuits_input):

        circuits_states = []
        for circ in data_circuits_input:

            n_qubits = 4
            sim = Simulator('mqvector', n_qubits)  # 使用mqvector模拟器，命名为sim
            sim.reset()
            sim.apply_circuit(circ)  # 通过模拟器sim运行搭建好的量子线路circuit
            result = sim.get_qs(True)
            state = []
            for value in result.splitlines():
                if value[0] == '-':
                    state.append(0)
                else:
                    state.append(1)
            state = tuple(state)
            circuits_states.append(state)
        return circuits_states
