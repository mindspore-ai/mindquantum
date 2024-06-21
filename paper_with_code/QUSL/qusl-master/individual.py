import random
import copy
import numpy.random
import numpy as np
import projectq
from projectq.ops import H, X, Y, Z, T, Tdagger, S, Sdagger, CNOT, CX, Rx, Ry, Rz, SqrtX, Measure, All, get_inverse, Swap, SwapGate
from math import pi
from mindquantum.core.circuit import Circuit
import RGB_genetic_algorithms

class Individual:
    """This class is a container for an individual in GA.
        用于表示遗传算法中个体的类，构造函数接受一些参数来初始化个体的属性，包括量子比特数量、允许的量子门、连接性等。类的实例化后，会自动生成随机排列、随机电路等属性，并设置一些默认值。
    """

    def __init__(self, circuit, number_of_qubits,connectivity="ALL", EMC=2.0, ESL=2.0,circuit2 = None):
        self.number_of_qubits = number_of_qubits
        self.allowed_gates = self.allowed_gates = [H, X, Y, Z,
                              Rx, Ry, Rz,
                              CNOT, CX, Swap]  # 共10种量子门可选择
        self.connectivity = connectivity
        self.permutation = random.sample(
            range(number_of_qubits), number_of_qubits)
        # EMC 代表预期突变计数
        self.EMC = EMC
        # ESL 代表预期序列长度
        self.ESL = ESL
        self.circuit = circuit
        self.circuit2 = circuit2
        self.CMW = 0.2 # 电路变异权重
        self.optimizedx = False

    def generate_random_circuit(self,initialize=True):
        '''
        生成一个随机电路，其长度从均值为 self.ESL 的几何分布中选择。
        其平均值为 self.ESL。
        params：none
        return:代表电路的图元列表。
        '''
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
                    print("control, target:",control, target)
                # TFG stands for Two Qubit Fixed Gate
                produced_circuit.append(("TFG", gate, control, target))

            elif gate in [H, X, Y, Z,]:
                # choose the index to apply
                target = random.choice(range(self.number_of_qubits))
                # SFG stands for Single Qubit Fixed Gate
                produced_circuit.append(("SFG", gate, target))
            elif gate in [Rx, Ry, Rz]:
                # choose the index to apply the operator
                target = random.choice(range(self.number_of_qubits))
                # choose the rotation parameter between 0 and 2pi up to 3 significiant figures
                # !! We may change this significant figure things later on
                significant_figure = 2 # 有效数字
                parameter = round(pi * random.uniform(0, 2),
                                  significant_figure)
                produced_circuit.append(("SG", gate, target, parameter))
                quantum_circuit.rx(parameter,target)
            else:
                print("WHAT ARE YOU DOING HERE!!")
                print("GATE IS:", gate)

        return produced_circuit

    def __str__(self):
        output = "number_of_qubits: " + str(self.number_of_qubits)
        output += "\nConnectivity = " + str(self.connectivity)
        output += "\nQubit Mapping = " + str(self.permutation)
        output += "\nallowedGates: ["
        for i in range(len(self.allowed_gates)):
            if self.allowed_gates[i] == Rx:
                output += "Rx, "
            elif self.allowed_gates[i] == Ry:
                output += "Ry, "
            elif self.allowed_gates[i] == Rz:
                output += "Rz, "
            elif self.allowed_gates[i] in [SwapGate, Swap]:
                output += "Swap, "
            elif self.allowed_gates[i] in [SqrtX]:
                output += "SqrtX, "
            elif self.allowed_gates[i] in [CNOT, CX]:
                output += "CX, "
            else:
                output += str(self.allowed_gates[i]) + ", "
        output = output[:-2]
        output += "]\nEMC: " + str(self.EMC) + ", ESL: " + str(self.ESL) + "\n"
        output += self.print_circuit()
        output += "\ncircuitLength: " + str(len(self.circuit))
        return output

    def print_circuit(self):
        output = f"Qubit Mapping: {self.permutation}\n"
        output += "Circuit: ["

        for operator in self.circuit:
            if operator[0] == "SFG":
                output += f"({operator[1]}, {operator[2]}), "
            elif operator[0] == "TFG":
                output += f"({operator[1]}, {operator[2]}, {operator[3]}), "
            elif operator[0] == "SG":
                output += f"({operator[1](round(operator[3], 3))}, {operator[2]}), "

        output = output[:-2]  # Remove the trailing comma and space
        output += "]"

        return output

    def clean(self):
        """
        Optimizes self.circuit by removing redundant gates
        """
        finished = False
        while not finished:
            finished = True
            i = 0
            while i < len(self.circuit) - 1:
                gate = self.circuit[i]

                if gate[1] == SqrtX:
                    j = i+1
                    while j < len(self.circuit):
                        if self.circuit[j][1] == gate[1] and self.circuit[j][2] == gate[2]:
                            self.circuit.pop(j)
                            self.circuit[i] = ("SFG", X, gate[2])
                            finished = False
                            break
                        elif self.circuit[j][1] == get_inverse(SqrtX) and self.circuit[j][2] == gate[2]:
                            self.circuit.pop(j)
                            self.circuit.pop(i)
                            finished = False
                            break
                        elif self.circuit[j][0] == "TFG":
                            if self.circuit[j][2] == gate[2] or self.circuit[j][3] == gate[2]:
                                break
                        elif self.circuit[j][2] == gate[2]:
                            break
                        j += 1
                elif gate[1] == Rz:
                    j = i+1
                    while j < len(self.circuit):
                        if self.circuit[j][1] == gate[1] and self.circuit[j][2] == gate[2]:
                            parameter = (self.circuit[j][3] + gate[3]) % (pi*2)
                            self.circuit.pop(j)
                            self.circuit[i] = ("SG", Rz, gate[2], parameter)
                            finished = False
                            break
                        elif self.circuit[j][0] == "TFG":
                            if self.circuit[j][2] == gate[2] or self.circuit[j][3] == gate[2]:
                                break
                        elif self.circuit[j][2] == gate[2]:
                            break
                        j += 1
                elif gate[1] == X:
                    j = i+1
                    while j < len(self.circuit):
                        if self.circuit[j][1] == gate[1] and self.circuit[j][2] == gate[2]:
                            self.circuit.pop(j)
                            self.circuit.pop(i)
                            finished = False
                            break
                        elif self.circuit[j][0] == "TFG":
                            if self.circuit[j][2] == gate[2] or self.circuit[j][3] == gate[2]:
                                break
                        elif self.circuit[j][2] == gate[2]:
                            break
                        j += 1
                elif gate[1] == CX:
                    j = i+1
                    while j < len(self.circuit):
                        if self.circuit[j][1] == gate[1] and self.circuit[j][2] == gate[2] and self.circuit[j][3] == gate[3]:
                            self.circuit.pop(j)
                            self.circuit.pop(i)
                            finished = False
                            break
                        elif self.circuit[j][2] == gate[2]:
                            break
                        elif self.circuit[j][2] == gate[3]:
                            break
                        elif self.circuit[j][0] == "TFG":
                            if self.circuit[j][3] == gate[2] or self.circuit[j][3] == gate[3]:
                                break
                        j += 1
                elif gate[1] == Swap:
                    j = i+1
                    while j < len(self.circuit):
                        if self.circuit[j][1] == gate[1] and self.circuit[j][2] == gate[2] and self.circuit[j][3] == gate[3]:
                            self.circuit.pop(j)
                            self.circuit.pop(i)
                            finished = False
                            break
                        if self.circuit[j][1] == gate[1] and self.circuit[j][2] == gate[3] and self.circuit[j][3] == gate[2]:
                            self.circuit.pop(j)
                            self.circuit.pop(i)
                            finished = False
                            break
                        elif self.circuit[j][2] == gate[2]:
                            break
                        elif self.circuit[j][2] == gate[3]:
                            break
                        elif self.circuit[j][0] == "TFG":
                            if self.circuit[j][3] == gate[2] or self.circuit[j][3] == gate[3]:
                                break
                        j += 1

                i += 1

    ###################### Mutations from here on ############################

    def permutation_mutation(self):
        '''
        随机改变个体的排列组合。
        '''
        self.permutation = random.sample(
            range(self.number_of_qubits), self.number_of_qubits)

    def discrete_uniform_mutation(self):
        """
        该函数遍历电路中定义的所有门，并 以 EMC / circuit_length 的概率随机改变目标和/或控制比特。
        Args:
          None
        Returns:
          None -> should I return sth ? maybe self?
        """
        circuit_length = len(self.circuit)
        if circuit_length == 0:
            mutation_prob = 0
        else:
            mutation_prob = self.EMC / circuit_length
        # I don't know if we really need this part
        if mutation_prob >= 1.0:
            mutation_prob = 0.5

        # We will loop over all the gates
        for i in range(circuit_length):
            if random.random() < mutation_prob:
                self.discrete_mutation(i)

    def sequence_insertion(self):
        """
        该函数通过从均值为 ESL 的几何分布中选择一个值，生成一个电路长度为 从均值为 ESL 的几何分布中选择一个值，然后将其插入中的一个随机点。
        """
        circuit_to_insert = self.generate_random_circuit(initialize=False)
        old_circuit_length = len(self.circuit)
        if old_circuit_length == 0:
            insertion_index = 0
        else:
            insertion_index = random.choice(range(old_circuit_length))
        self.circuit[insertion_index:] = circuit_to_insert + \
            self.circuit[insertion_index:]
        return self.circuit

    def sequence_and_inverse_insertion(self):
        """
        This function generates a random circuit with circuit length given by choosing
        a value from a geometric distribution with mean value ESL, it is inserted to a
        random point in self.circuit and its inverse is inserted to another point.
        """
        circuit_to_insert = self.generate_random_circuit(initialize=False)
        # MAYBE CONNECTIVITY IS NOT REFLECTIVE ?
        inverse_circuit = get_inverse_circuit(circuit_to_insert)
        old_circuit_length = len(self.circuit)
        if old_circuit_length >= 2:
            index1, index2 = random.sample(range(old_circuit_length), 2)
            if index1 > index2:
                index2, index1 = index1, index2
        else:
            index1, index2 = 0, 1
        new_circuit = (
            self.circuit[:index1]
            + circuit_to_insert
            + self.circuit[index1:index2]
            + inverse_circuit
            + self.circuit[index2:]
        )
        self.circuit = new_circuit

    def discrete_mutation(self, index):
        """
        This function applies a discrete mutation to the circuit element at index.
        Discrete mutation means that the control and/or target qubits are randomly changed.
        """
        if len(self.circuit) == 0:
            return
        while index >= len(self.circuit):
            index -= 1
        if self.circuit[index][0] == "SFG":
            # This means we have a single qubit fixed gate
            new_target = random.choice(range(self.number_of_qubits))
            self.circuit[index] = ("SFG", self.circuit[index][1], new_target)
        elif self.circuit[index][0] == "TFG":
            # This means we have two qubit fixed gate
            if self.connectivity == "ALL":
                new_control, new_target = random.sample(
                    range(self.number_of_qubits), 2)
            else:
                new_control, new_target = random.choice(self.connectivity)
            self.circuit[index] = (
                "TFG", self.circuit[index][1], new_control, new_target)
        elif self.circuit[index][0] == "SG":
            # This means we have a single rotation gate
            new_target = random.choice(range(self.number_of_qubits))
            self.circuit[index] = (
                "SG",
                self.circuit[index][1],
                new_target,
                self.circuit[index][3],
            )
        else:
            print("WRONG BRANCH IN discrete_mutation")

    def continuous_mutation(self, index):
        """
        This function applies a continuous mutation to the circuit element at index.
        Continuous mutation means that if the gate has a parameter, its parameter its
        changed randomly, if not a discrete_mutation is applied.
        """
        if len(self.circuit) == 0:
            return
        while index >= len(self.circuit):
            index -= 1

        if self.circuit[index][0] == "SG":
            # This means we have a single rotation gate
            newParameter = float(
                self.circuit[index][-1]) + numpy.random.normal(scale=self.CMW)
            self.circuit[index] = (
                "SG", self.circuit[index][1], self.circuit[index][2], newParameter)
        elif self.circuit[index][0] == "SFG":
            # This means we have a single qubit/two qubit fixed gate and we need to
            # apply a discrete_mutation.
            new_target = random.choice(range(self.number_of_qubits))
            self.circuit[index] = ("SFG", self.circuit[index][1], new_target)
        elif self.circuit[index][0] == "TFG":
            # This means we have two qubit fixed gate
            if self.connectivity == "ALL":
                new_control, new_target = random.sample(
                    range(self.number_of_qubits), 2)
            else:
                new_control, new_target = random.choice(self.connectivity)
            self.circuit[index] = (
                "TFG", self.circuit[index][1], new_control, new_target)
        else:
            print("WRONG BRANCH IN continuous_mutation")

    def parameter_mutation(self):
        '''
        该函数遍历电路中定义的所有栅极，并定期调整旋转栅极的参数。
        '''
        if len(self.circuit) == 0:
            return

        mutation_prob = self.EMC / len(self.circuit)
        for index in range(len(self.circuit)):
            if random.random() < mutation_prob:
                if self.circuit[index][0] == "SG":
                    # This means we have a single rotation gate
                    newParameter = float(
                        self.circuit[index][-1]) + numpy.random.normal(scale=self.CMW)
                    newParameter = newParameter % (2*pi)
                    self.circuit[index] = (
                        "SG", self.circuit[index][1], self.circuit[index][2], newParameter)

    def continuous_uniform_mutation(self):
        """
        This function iterates over all the gates defined in the circuit and
        randomly changes the parameter if possible, if not target and/or control qubits
        with probability EMC / circuit_length.
        Args:
          None
        Returns:
          None -> should I return sth ? maybe self?
        """
        circuit_length = len(self.circuit)
        if circuit_length == 0:
            mutation_prob = 0
        else:
            mutation_prob = self.EMC / circuit_length
        # I don't know if we really need this part
        if mutation_prob >= 1.0:
            mutation_prob = 0.5

        # We will loop over all the gates
        for i in range(circuit_length):
            if random.random() < mutation_prob:
                self.continuous_mutation(i)

    def insert_mutate_invert(self):
        """
        This function performs a discrete mutation on a single gate, then places a
        randomly selected gate immediately before it and its inverse immediately
        after it.
        """
        # index to apply discrete mutation
        if len(self.circuit) == 0:
            index = 0
        else:
            index = random.choice(range(len(self.circuit)))

        # Discrete Mutation
        self.discrete_mutation(index)

        # Generate the circuit to insert and its inverse
        circuit_to_insert = self.generate_random_circuit(initialize=False)
        while len(circuit_to_insert) == 0:
            circuit_to_insert = self.generate_random_circuit(initialize=False)
        circuit_to_insert = [circuit_to_insert[0]]
        inverse_circuit = get_inverse_circuit(circuit_to_insert)
        if index >= len(self.circuit):
            # This probably happens only when index = 0 and length of the circuit = 0
            if index == 0:
                new_circuit = circuit_to_insert + inverse_circuit
            else:
                print("\n\nIT SHOULD NEVER ENTER HEREE!!!\n\n")
        else:
            new_circuit = (
                self.circuit[:index]
                + circuit_to_insert
                + [self.circuit[index]]
                + inverse_circuit
                + self.circuit[(index + 1):]
            )
        self.circuit = new_circuit

    def swap_qubits(self):
        """
        This function swaps two randomly selected qubits.
        """
        qubit1, qubit2 = random.sample(range(self.number_of_qubits), 2)

        for operator in self.circuit:
            if operator[0] == "SFG":
                if operator[2] == qubit1:
                    operator = operator[0:2] + (qubit2,)
                elif operator[2] == qubit2:
                    operator = operator[0:2] + (qubit1,)

            elif operator[0] == "TFG":
                if operator[2] == qubit1 and operator[3] == qubit2:
                    operator = operator[0:2] + (qubit2, qubit1)

                elif operator[2] == qubit2 and operator[3] == qubit1:
                    operator = operator[0:2] + (qubit1, qubit2)

                elif operator[2] == qubit1:
                    operator = (
                        operator[0:2] + (qubit2,) + operator[3:]
                    )

                elif operator[2] == qubit2:
                    operator = (
                        operator[0:2] + (qubit1,) + operator[3:]
                    )

                elif operator[3] == qubit1:
                    operator = operator[0:3] + (qubit2,)

                elif operator[3] == qubit2:
                    operator = operator[0:3] + (qubit1,)

            elif operator[0] == "SG":
                if operator[2] == qubit1:
                    operator = (
                        operator[0:2] +
                        (qubit2,) + (operator[3],)
                    )
                elif operator[2] == qubit2:
                    operator = (
                        operator[0:2] +
                        (qubit1,) + (operator[3],)
                    )

    def sequence_deletion(self):
        """
        This function deletes a randomly selected interval of the circuit.
        """
        if len(self.circuit) < 2:
            return

        circuit_length = len(self.circuit)
        index = random.choice(range(circuit_length))
        # If this is the case, we'll simply remove the last element
        if index == (circuit_length - 1):
            self.circuit = self.circuit[:-1]
        else:
            sequence_length = numpy.random.geometric(p=(1 / self.ESL))
            if (index + sequence_length) >= circuit_length:
                self.circuit = self.circuit[: (-circuit_length + index)]
            else:
                self.circuit = (
                    self.circuit[:index] +
                    self.circuit[(index + sequence_length):]
                )

    def sequence_replacement(self):
        """
        This function first applies sequence_deletion, then applies a sequence_insertion.
        """
        self.sequence_deletion()
        self.sequence_insertion()

    def sequence_swap(self):
        """
        This function randomly chooses two parts of the circuit and swaps them.
        """
        if len(self.circuit) < 4:
            return

        indices = random.sample(range(len(self.circuit)), 4)
        indices.sort()
        i1, i2, i3, i4 = indices[0], indices[1], indices[2], indices[3]

        self.circuit = (
            self.circuit[0:i1]
            + self.circuit[i3:i4]
            + self.circuit[i2:i3]
            + self.circuit[i1:i2]
            + self.circuit[i4:]
        )

    def sequence_scramble(self):
        """
        该函数随机选择一个索引，并从几何距离 w/ 均值 ESL 中选择一个长度，然后对电路该部分的门电路进行置换。
        """
        circuit_length = len(self.circuit)
        if circuit_length < 2:
            index1 = 0
        else:
            index1 = random.choice(range(circuit_length - 1))

        sequence_length = numpy.random.geometric(p=(1 / self.ESL))
        if (index1 + sequence_length) >= circuit_length:
            index2 = circuit_length - 1
        else:
            index2 = index1 + sequence_length

        toShuffle = self.circuit[index1:index2]
        random.shuffle(toShuffle)

        self.circuit = self.circuit[:index1] + \
            toShuffle + self.circuit[index2:]

    def move_gate(self):
        """
        This function randomly moves a gate from one point to another point.
        """
        circuit_length = len(self.circuit)
        if circuit_length < 2:
            return
        old_index, new_index = random.sample(range(circuit_length), 2)

        temp = self.circuit.pop(old_index)
        self.circuit.insert(new_index, temp)

    def cross_over(self, parent2):
        """
        该函数获取两个父解，创建一个空子解，随机从每个父代方案中随机选取门的数量，
        并从第一个父代方案中选取该数量的门，从第二个父代方案中放弃该数量的门。
        第二个父节点中的门数。如此重复，直到父代方案用完为止。
        """
        self_circuit = self.circuit[:]
        parent2_circuit = parent2[:]
        p1 = p2 = 1.0

        if len(self_circuit) != 0:
            p1 = self.EMC / len(self.circuit)
        if (p1 <= 0) or (p1 > 1):
            p1 = 1.0
        #
        # if len(parent2_circuit) != 0:
        #     p2 = parent2.EMC / len(parent2.circuit)
        # if (p2 <= 0) or (p2 > 1):
        #     p2 = 1.0

        Child = []
        turn = 1
        while len(self_circuit) or len(parent2_circuit):
            if turn == 1:
                number_of_gates_to_select = numpy.random.geometric(p1)
                Child += self_circuit[:number_of_gates_to_select]
                turn = 2
            else:
                number_of_gates_to_select = numpy.random.geometric(p2)
                Child += parent2_circuit[:number_of_gates_to_select]
                turn = 1
            self_circuit = self_circuit[number_of_gates_to_select:]
            parent2_circuit = parent2_circuit[number_of_gates_to_select:]
        return Child


def print_circuit(circuit):
    output = "Circuit: ["
    for i in range(len(circuit)):
        if circuit[i][0] == "SFG":
            output += "(" + str(circuit[i][1]) + \
                "," + str(circuit[i][2]) + "), "
        elif circuit[i][0] == "TFG":
            output += (
                "("
                + str(circuit[i][1])
                + ","
                + str(circuit[i][2])
                + ","
                + str(circuit[i][3])
                + "), "
            )
        elif circuit[i][0] == "SG":
            output += (
                "("
                + str(circuit[i][1](round(circuit[i][3], 3)))
                + ","
                + str(circuit[i][2])
                + "), "
            )
    output = output[:-2]
    output += "]"
    return output


def get_inverse_circuit(circuit):
    """
    This function takes a circuit and returns a circuit which is the inverse circuit.
    """
    if len(circuit) == 0:
        return []

    reversed_circuit = circuit[::-1]
    for gate in reversed_circuit:
        if gate[1] in [H, X, Y, Z, CX, Swap, SwapGate]:
            continue
        elif gate[1] == S:
            gate = ("SFG", Sdagger, gate[2])
        elif gate[1] == Sdagger:
            gate = ("SFG", S, gate[2])
        elif gate[1] == T:
            gate = ("SFG", Tdagger, gate[2])
        elif gate[1] == Tdagger:
            gate = ("SFG", T, gate[2])
        elif gate[1] in [Rx, Ry, Rz]:
            gate = (
                "SG",
                gate[1],
                gate[2],
                round(2 * pi - gate[3], 3),
            )
        elif gate[1] in [SqrtX]:
            gate = ("SFG", get_inverse(
                SqrtX), gate[2])
        elif gate[1] in [get_inverse(SqrtX)]:
            gate = ("SFG", SqrtX, gate[2])
        else:
            print("\nWRONG BRANCH IN get_inverse_circuit\n")

    return reversed_circuit
