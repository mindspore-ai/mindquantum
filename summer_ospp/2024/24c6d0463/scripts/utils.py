"""
Utility module for handling quantum circuits.
Contains the QuantumCircuitManager class for managing quantum circuits and utility functions.
"""
import copy
import os
import pandas as pd
import numpy as np
from mindquantum import Simulator
from mindquantum.algorithm.compiler import (compile_circuit, BasicDecompose,
                                            decompose, GateReplacer, cu_decompose)
from mindquantum.core.circuit import Circuit
from mindquantum.core.parameterresolver import ParameterResolver
from mindquantum.core.gates import U3, SWAP, X, Measure
pd.options.mode.chained_assignment = None  # default='warn'


class QCManager:
    """
    A class to manage quantum circuits, including loading, parsing, and processing QASM files.

    Attributes:
        program_path (str): The path to the QASM file.
        result_path (str): The path where results will be saved.
        circuit: The quantum circuit object (loaded from the QASM file).
    """
    def __init__(self, program_path, new_program_path):
        self.program_path = program_path
        self.result_path = new_program_path
        self.circuit = None

    def num_qubits_from_qasm(self):
        """
        Function to return the number of qubits in a program
        """
        circ = self.read_qasm(self.program_path)
        total = circ.n_qubits
        return total

    def read_qasm(self, filepath):
        """
        Function to read a QASM into a Quantum circuit object
        """
        circ = Circuit.from_openqasm(filepath)
        # circ.summary()
        # if verbo:
        #     print('Number of Qubits in program:', circ.n_qubits)
        return circ

    def recursive_compile_noise_adaptive(self, quantum_circs):
        """
        Function to recursively compile a list of quantum circuits
        """
        post_compile_circs = []
        for quantum_circ in quantum_circs:
            post_circs = compile_circuit(BasicDecompose(), quantum_circ)
            swap_decomposed_circ = decompose.swap_decompose(SWAP.on([0, 5]))
            compiler_swap = GateReplacer(SWAP.on([0, 5]), swap_decomposed_circ[0])
            post_circs = compile_circuit(compiler_swap, post_circs)
            theta = ParameterResolver(0)
            phi = ParameterResolver(0)
            lamda = ParameterResolver(np.pi / 2)
            u3_two_decomposed_circ = cu_decompose(U3(theta, phi, lamda).on(0, 5))
            compiler_u3_two = GateReplacer(U3(theta, phi, lamda).on(0, 5), u3_two_decomposed_circ)
            post_circs = compile_circuit(compiler_u3_two, post_circs)
            post_compile_circs.append(post_circs)

        return post_compile_circs

    def create_instruction_table(self, qubit_set, circuit_depth):
        """
        Create an empty instruction table.

        Input:
        - qubit_set: Set of all qubits in the circuit
        - max_time_steps(circuit_depth): Maximum number of time steps to be considered in the table

        Output:
        - An empty instruction table (IDT_Frame) initialized with empty strings
        """

        empty_frame = {qubit: [""] * circuit_depth for qubit in qubit_set}
        return empty_frame

    def populate_instruction_table(self, qubit_set, dag, empty_frame):
        """
        Function to populate the Instruction Table (IDT) with the operations from DAG.

        Args:
            qubit_set (list): List of qubit indices used in the DAG.
            dag (DAGCircuit): The directed acyclic graph representation of the quantum circuit.
            empty_frame (DataFrame): Empty Instruction Table frame to be populated with operations.

        Returns:
            DataFrame: Populated Instruction Table (IDT) with operations from the DAG.
        """
        populate_frame = copy.deepcopy(empty_frame)

        # Dictionary of all indices in the DAG to time step 0
        index_list = {ele: 0 for ele in qubit_set}

        # Iterating over all gate nodes in the DAG
        for node in dag.topological_sort():
            if node.gate.name == 'X':
                # Check if there is a control qubit
                if len(node.local) > 1:
                    q1 = self.extract_qubit_index(node.local[0])
                    q2 = self.extract_qubit_index(node.local[1])
                    cur_index = max(index_list[q1], index_list[q2])
                    # Change 'X' to 'CX' if there is a control qubit
                    populate_frame[q1][cur_index] = f"CX ({q1})({q2})"
                    populate_frame[q2][cur_index] = f"CX ({q1})({q2})"
                    index_list[q1] = cur_index + 1
                    index_list[q2] = cur_index + 1
                else:
                    q1 = self.extract_qubit_index(node.local[0])
                    populate_frame[q1][index_list[q1]] = node.gate.name
                    index_list[q1] = index_list[q1] + 1

            elif node.gate.name == 'U3':
                q1 = self.extract_qubit_index(node.local[0])
                populate_frame[q1][index_list[q1]] = \
                    (f"{node.gate.name} ({node.gate.theta.const})"
                     f"({node.gate.phi.const})({node.gate.lamda.const})")
                index_list[q1] = index_list[q1] + 1

            elif node.gate.name == 'CX':
                q1 = self.extract_qubit_index(node.local[0])
                q2 = self.extract_qubit_index(node.local[1])
                cur_index = max(index_list[q1], index_list[q2])
                populate_frame[q1][cur_index] = f"CX ({q1})({q2})"
                populate_frame[q2][cur_index] = f"CX ({q1})({q2})"
                index_list[q1] = cur_index + 1
                index_list[q2] = cur_index + 1

            elif node.gate.name == 'BARRIER':
                qargs_list = [self.extract_qubit_index(q) for q in node.local]
                cur_index = max(index_list[q] for q in qargs_list)
                for q in qargs_list:
                    populate_frame[q][cur_index] = node.gate.name
                    index_list[q] = cur_index + 1

            elif node.gate.name in ['RX', 'RZ']:
                q1 = self.extract_qubit_index(node.local[0])
                rx_param = node.gate.coeff
                populate_frame[q1][index_list[q1]] = f"{node.gate.name} ({rx_param})"
                index_list[q1] = index_list[q1] + 1

            elif node.gate.name in ['I', 'SX', 'H', 'Measure']:
                q1 = self.extract_qubit_index(node.local[0])
                populate_frame[q1][index_list[q1]] = node.gate.name
                index_list[q1] = index_list[q1] + 1

        return populate_frame

    def extract_qubit_index(self, qubit_obj):
        """
        Extract the index of a qubit from a mindquantum qubit object.

        Input:
        - qubit_obj: A mindquantum qubit object

        Output:
        - The index of the qubit
        """
        return qubit_obj

    def get_instruction_table(self, qubit_set, circuit_depth):
        """生成并填充指令表"""
        empty_table = self.create_instruction_table(qubit_set, circuit_depth)
        populated_table = self.populate_instruction_table(qubit_set, self.circuit, empty_table)
        return populated_table

    def zero_filter(self, df):
        """
        Function to filter out zero values (no operations) from the Instruction Table (IDT).

        Args:
            df (DataFrame): Instruction Table (IDT) frame with potential zero values.

        Returns:
            DataFrame: Filtered Instruction Table (IDT) with zero values removed.
        """
        # If df is a dictionary, convert it to a DataFrame
        if isinstance(df, dict):
            df = pd.DataFrame(df)
        # Remove rows with all zero values
        # df = df[(df.T != 0).any()]
        df = df.loc[(df != 0).any(axis=1)]
        # Iterate over columns and remove columns with only zero and 'barrier' values
        for ele in df.columns:
            if 0 in set(df[ele]) and 'barrier' in set(df[ele]) and len(set(df[ele])) == 2:
                df.drop(columns=[ele], inplace=True)
        return df

    def get_all_instruction_lengths(self):
        """
        Function to get lengths of all instructions from the device properties.

        Returns:
        dict: A dictionary containing the gate names as keys and their respective
              durations as values.
        """
        device_properties = {
            'gates': [
                {'name': 'CX', 'duration': 185},
                {'name': 'X', 'duration': 84},
                {'name': 'H', 'duration': 84},
                {'name': 'I', 'duration': 185},
                {'name': 'RZ', 'duration': 84},
                {'name': 'U3', 'duration': 84},
                {'name': 'Y', 'duration': 84}
            ]
        }

        cx_lengths = self.get_gate_length(device_properties, 'CX')
        x_lengths = self.get_gate_length(device_properties, 'X')
        h_lengths = self.get_gate_length(device_properties, 'H')
        id_lengths = self.get_gate_length(device_properties, 'I')
        rz_lengths = self.get_gate_length(device_properties, 'RZ')
        u3_lengths = self.get_gate_length(device_properties, 'U3')
        y_lengths = self.get_gate_length(device_properties, 'Y')

        gate_lengths = {'CX': cx_lengths, 'H': h_lengths, 'I': id_lengths,
                    'RZ': rz_lengths, 'X': x_lengths,
                    'U3': u3_lengths, 'Y': y_lengths}
        return gate_lengths

    def get_gate_length(self, device_properties, gate_name):
        """
        Function to get the lengths of a specific gate from the device properties.

        Args:
            device_properties (dict): Dictionary containing device gate properties.
            gate_name (str): Name of the gate to get the length for.

        Returns:
            list: List of durations for the specified gate.
        """
        lengths = []
        for gate in device_properties['gates']:
            if gate['name'] == gate_name:
                lengths.append(gate['duration'])
        return lengths

    def discrete_to_analog(self, dataframe, gate_lengths):
        """
        Function to convert a discrete IDT to analog time values.

        Args:
            dataframe: Instruction Table with discrete time steps.
            gate_lengths (dict): Dictionary containing gate lengths for each type of gate.

        Returns:
            DataFrame: IDT table with time values where operations end.
        """
        # Convert IDT to DataFrame if it is not already
        if isinstance(dataframe, dict):
            dataframe = pd.DataFrame(dataframe)

        # initiating the time indices to t = 0
        new_time_indices = [0]

        # Initialize an empty DataFrame for the analog times
        populate_frame = copy.deepcopy(dataframe)

        # Initialize a dictionary to keep track of the current time for each qubit
        # current_time = {qubit: 0 for qubit in IDT.columns}
        current_time = 0

        # Iterate over each row (time step)
        for row in dataframe.index:
            max_time_step_duration = 0
            # Iterate over each column (qubit) in the current time step
            for qubit in dataframe.columns:
                gate = dataframe.loc[row, qubit]
                if gate and gate != '':
                    if ' ' in gate:
                        gate_name = gate.split(' ')[0]
                    elif '(' in gate:
                        gate_name = gate.split(' ')[0]
                    else:
                        gate_name = gate

                    if gate_name in gate_lengths:
                        gate_length = gate_lengths[gate_name][0]
                    else:
                        gate_length = 0  # Default to 0 if the gate length is not specified

                    max_time_step_duration = max(max_time_step_duration, gate_length)

            # Update the current time for each qubit after the current time step

            current_time += max_time_step_duration
            # analog_IDT.loc[row, qubit] = f'{gate} @ {current_time[qubit]}'
            new_time_indices.append(current_time)

        # removing the t = 0 index
        new_time_indices = new_time_indices[1:]
        populate_frame.index = new_time_indices

        return populate_frame

    def gather_all_gate_counts(self, circuit):
        """
        Function to gather important gate counts
        """
        op_counts = {}
        for op in circuit:
            op_name = str(op).split()[0]
            if op_name not in op_counts:
                op_counts[op_name] = 0
            op_counts[op_name] += 1
        return op_counts

    def save_circuit_svg(self, circuit, filename):
        """Save the quantum circuit in SVG format"""
        circuit.svg().to_file(f"{self.result_path}_{filename}.svg")

    def circ_to_qasm(self, circuits, sequence_strings, name):
        """
        Convert a list of quantum circuits to OpenQASM format and save them as files.

        Args:
            circuits (list): A list of quantum circuit objects.
            sequence_strings (list): A list of strings used to uniquely name each QASM file.
            name (str) : The name of qasm_file

        """
        i = 0
        path_without_extension, _ = os.path.splitext(self.result_path)
        for circ in circuits:
            if name == 'skeleton':
                circ = circ[0]
            base_cir_name = f"{path_without_extension}_{sequence_strings[i]}_{name}.qasm"
            circ.to_openqasm(base_cir_name)
            i += 1

    def circ_to_delay_qasm(self, circuits, sequence_strings, name):
        """
        Convert a list of MindQuantum circuits to OpenQASM format,
        insert delay instructions after DD(XX) gates,
        and save them as files.

        Args:
            circuits (list): A list of quantum circuit objects.
            sequence_strings (list): A list of strings used to uniquely name each QASM file.
            name : Saved file identification
        """
        i = 0
        path_without_extension, _ = os.path.splitext(self.result_path)

        for circ in circuits:
            if name == 'skeleton':
                circ = circ[0]
            # Step 1: Convert the circuit to OpenQASM string
            qasm_string = circ.to_openqasm()

            # Step 2: Split the QASM into lines for processing
            qasm_lines = qasm_string.splitlines()

            # Step 3: Initialize a new list to hold modified QASM lines
            modified_qasm_lines = []

            # Step 4: Initialize a dictionary to track consecutive X gates on each qubit
            consecutive_x_count = {}

            # Step 5: Process each line in the QASM file
            for line in qasm_lines:

                # Check if the line contains an X gate
                if line.startswith('x'):
                    qubit = int(line.split('[')[1].split(']')[0])
                    # Check if this qubit has had consecutive X gates
                    if qubit not in consecutive_x_count:
                        consecutive_x_count[qubit] = 0

                    # Increment the X gate count for this qubit
                    consecutive_x_count[qubit] += 1

                    # Check if this qubit had a previous X gate
                    if consecutive_x_count[qubit] == 2:
                        modified_qasm_lines.append(f"delay(8.5) q[{qubit}];")
                        modified_qasm_lines.append(line)
                        modified_qasm_lines.append(f"delay(8.5) q[{qubit}];")
                    elif consecutive_x_count[qubit] > 2:
                        consecutive_x_count[qubit] = 0
                        modified_qasm_lines.append(line)
                    else:
                        modified_qasm_lines.append(line)

                else:
                    # Reset the X gate tracker for all qubits if it's not an X gate
                    for qubit in [int(q.split('[')[1].split(']')[0])
                                  for q in line.split() if 'q[' in q]:
                        consecutive_x_count[qubit] = 0
                    modified_qasm_lines.append(line)

            base_cir_name = f"{path_without_extension}_{sequence_strings[i]}_delay_{name}.qasm"
            with open(base_cir_name, 'w', encoding='utf-8') as f:
                f.write("\n".join(modified_qasm_lines))

            i += 1

    def total_variation_distance(self, p, q):
        """
        Calculate the Total Variation Distance (TVD) between two probability distributions P and Q.

        Args:
            p (array-like): Ideal probability distribution.
            q (array-like): Real experiment's probability distribution.

        Returns:
            float: Total Variation Distance.
        """
        # Get all possible quantum states
        all_keys = set(p.keys()).union(set(q.keys()))

        # Normalized probability distribution
        norm_dist1 = self.normalize_distribution(p)
        norm_dist2 = self.normalize_distribution(q)

        # Calculate the tvd
        tvd = 0
        for key in all_keys:
            p1 = norm_dist1.get(key, 0.0)
            p2 = norm_dist2.get(key, 0.0)
            tvd += abs(p1 - p2)

        return tvd / 2.0

    def normalize_distribution(self, distribution):
        """
        Normalize the values in a distribution so that they sum up to 1.

        Args:
            distribution (dict): A dictionary where keys are categories
            and values are numeric counts or frequencies.

        Returns:
            dict: A new dictionary where the values are normalized
        """
        total_count = sum(distribution.values())
        return {key: value / total_count for key, value in distribution.items()}

    def execute_on_ideal_machine(self, circuits, shots, mode='normal'):
        """
        Function to get the histogram data of a simulator. (no noise)
        """
        counts = []

        if mode == 'normal':
            for circ in circuits:
                sim = Simulator('mqvector', circ.n_qubits)
                counts_ = sim.sampling(circ, shots=shots)
                counts.append(counts_)
                # sim.apply_circuit(circ)
                # counts.append(sim.get_qs())

        elif mode == 'dd':
            for circ in circuits:
                circ_temp_2 = Circuit()
                circ_temp_1 = circ[1]
                for qubit in range(circ_temp_1.n_qubits):
                    circ_temp_1 += Measure().on(qubit)
                sim = Simulator('mqvector', circ_temp_1.n_qubits)
                counts_ = sim.sampling(circ_temp_1, shots=shots)
                most_counts_result = max(counts_.data, key=counts_.data.get)
                for i in range(circ[1].n_qubits):
                    if most_counts_result[i] == '1':
                        circ_temp_2 += X.on(i)
                circ_temp_2 = circ_temp_2 + circ[2]
                sim_stab = Simulator('stabilizer', circ_temp_2.n_qubits)
                # sim_stab.set_qs(final_state)
                counts_ = sim_stab.sampling(circ_temp_2, shots=shots)
                counts.append(counts_)

        return counts
