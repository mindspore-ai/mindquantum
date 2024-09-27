"""
Module for handling Dynamical Decoupling (DD) in quantum circuits.
Contains the DDManager class responsible for applying DD sequences to circuits.
"""
import itertools
import pandas as pd
import numpy as np
from sympy import sympify
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import (X, Y, RX, U3, Measure,
                                    I, RZ, H, PhaseDampingChannel, DepolarizingChannel)
from mindquantum.algorithm.error_mitigation import query_single_qubit_clifford_elem
from mindquantum.simulator import decompose_stabilizer
pd.options.mode.chained_assignment = None  # default='warn'

class DDManager:
    """
    A class to manage the application of Dynamical Decoupling (DD) to quantum circuits.

    Attributes:
        gate_lengths (dict): A dictionary containing the length of various quantum gates.
        tab (list): A list to store the state of qubits considered for DD.
        qubits_to_consider (list): A list of qubit indices to be considered for DD applications.
        circ (Circuit): An instance of the Circuit class representing the main quantum circuit.
        circ_skeleton (Circuit): An instance of the Circuit class representing
        a skeletal version of the quantum circuit.
        layer_set (int): The number of layers to be applied in the DD sequence.
        ts (int): A timestamp or counter used in the DD process.
    """

    def __init__(self, gate_lengths, qubits_in_device):
        self.gate_lengths = gate_lengths
        self.tab = [0] * qubits_in_device
        self.qubits_to_consider = list(range(qubits_in_device))
        self.circ = Circuit()
        self.circ_skeleton = Circuit()
        self.layer_set = 5
        self.ts = 0


    def parse_param(self, param):
        """
        Parse a parameter that may contain the 'π' and convert it
        into a float or a numerical expression involving pi.

        Args:
            param (str): A string representing a numerical value, possibly containing 'π'.

        Returns:
            float: The numerical value after replacing 'π' with np.pi and evaluating the expression.
        """
        if 'π' in param:
            param = param.replace('π', 'pi')
            try:
                # Safely parse the expression using ast.literal_eval
                # return eval(param, {"__builtins__": None}, {"np": np})
                return float(sympify(param))
            except Exception as e:
                raise ValueError(f"Invalid expression: {param}") from e

        return float(param)

    def extract_gate_params(self, gate_info):
        """
        Extract the gate name and its parameters from a gate information string.

        Args:
            gate_info (str): A string containing the gate name and parameters
            in the format 'GateName(param1, param2, ...)'.

        Returns:
            tuple: A tuple containing the gate name and a list of parameters.
        """
        if '(' in gate_info:
            gate_name, params = gate_info.split(' ', 1)
            params = params.strip('()').split(')(')
            params = [self.parse_param(p) for p in params]
        else:
            gate_name = gate_info
            params = []
        return gate_name, params

    def dataframe_to_circ(self, analog_table, qubits_to_consider, mode='normal'):
        """
        Function to convert analog_table to MindQuantum Circuit

        Args:
            analog_table (DataFrame): Analog Instruction Table with time values.
            mode (str): Mode parameter to determine if dynamic decoupling
                        should be applied ('normal' or 'dynamic_decoupling').
            qubits_to_consider (list): A list of qubit indices to be considered for DD applications.

        Returns:
            Circuit: MindQuantum Circuit.
        """
        # dd_modes = ['xyxy']
        self.circ = Circuit()
        # Number of rows and qubits in the analog IDT
        n_rows, qubits = analog_table.shape
        self.tab = [0] * qubits
        # The analog time values
        # analog_time = list(analog_table.index)
        # The qubits names
        qubits_in_table = list(analog_table.columns)
        self.qubits_to_consider = qubits_to_consider
        skeleton_flag = None
        # Adding the operations to the quantum circuit
        for ts in range(n_rows):
            # ts_val = analog_time[ts]
            # diff = ts_val if ts == 0 else ts_val - analog_time[ts - 1]
            self.ts = ts
            # Process each gate in the current timestep
            self.process_gates_in_timestep(analog_table, qubits_in_table, mode, skeleton_flag)
            # Add measurement operations
        self.finalize_measurements(analog_table, skeleton_flag)

        return self.circ

    def dataframe_to_skeleton_circ(self, analog_table,
                                   qubits_to_consider=None, mode='normal'):
        """
        Function to convert analog_IDT to MindQuantum Skeleton Circuit
        Generate a seed Clifford dummy circuit (SDC) based on the given analog table.

            Args:
            analog_table (DataFrame): Analog Instruction Table with time values.
            qubits_in_device (int): Total number of qubits in the device.
            mode (str): Mode parameter to determine if dd should be applied ('normal' or 'dd').

        Returns:
            Circuit: The generated seed Clifford dummy MindQuantum circuit (SDC).
        """
        # dd_modes = ['xyxy']
        self.circ = Circuit()
        self.circ_skeleton = Circuit()
        # Number of rows and qubits in the analog IDT
        n_rows, qubits = analog_table.shape
        self.tab = [0] * qubits
        # The analog time values
        # analog_time = list(analog_table.index)
        # The qubits names
        qubits_in_table = list(analog_table.columns)
        self.qubits_to_consider = qubits_to_consider
        skeleton_flag = True
        # Adding the operations to the quantum circuit
        for ts in range(n_rows):
            # ts_val = analog_time[ts]
            # diff = ts_val if ts == 0 else ts_val - analog_time[ts - 1]
            self.ts = ts
            # Process each gate in the current timestep
            self.process_gates_in_timestep(analog_table, qubits_in_table, mode, skeleton_flag)
            # Add measurement operations
        self.finalize_measurements(analog_table, skeleton_flag)
        return self.circ + self.circ_skeleton, self.circ, self.circ_skeleton

    def process_gates_in_timestep(self, analog_table, qubits_in_table, mode, skeleton_flag):
        """
        Process the gates for each qubit in a given timestep and update the circuit.

        Args:
            analog_table (DataFrame): The analog instruction table.
            qubits_in_table (list): List of qubits in the analog table.
            mode (str): 'normal' for no DD, 'xyxy' for dynamic decoupling.
            skeleton_flag : Select whether to generate a skeleton circuit.
        """
        for idx, qubit_val in enumerate(qubits_in_table):
            gate_info = analog_table.iloc[self.ts, idx]
            if gate_info:
                gate_name, params = self.extract_gate_params(gate_info)

                if gate_name == 'CX' and int(params[0]) == qubit_val:
                    self.handle_cx_gate(params, qubit_val, mode)
                if gate_name == 'I':
                    self.handle_i_gate(qubit_val, mode)
                else:
                    if skeleton_flag:
                        self.handle_non_clifford_gate(gate_name, params, qubit_val)
                    else:
                        self.handle_single_qubit_gate(gate_name, params, qubit_val)

    def handle_cx_gate(self, params, qubit_val, mode):
        """
        Handle a CX gate by inserting DD if necessary and updating the circuit.

        Args:
            params (list): Parameters of the CX gate (qubits involved).
            qubit_val (int): The current qubit.
            mode (str): 'normal' for no DD, 'xyxy' for dynamic decoupling.
        """
        sec_qubit = int(params[1])
        qubit_exec_time = self.tab[qubit_val]
        sec_qubit_exec_time = self.tab[sec_qubit]

        # #Calculate time difference between the two qubits involved in CX
        time_diff = abs(qubit_exec_time - sec_qubit_exec_time)

        # Insert DD if necessary
        if qubit_exec_time < sec_qubit_exec_time:
            dd_qubit = qubit_val

            if time_diff > 0 and mode != 'normal':
                self.apply_dynamic_decoupling(dd_qubit, mode, time_diff)
            self.tab[sec_qubit] += self.gate_lengths['CX'][0]
            self.tab[qubit_val] = self.tab[sec_qubit]
        else:
            dd_qubit = sec_qubit

            if time_diff > 0 and mode != 'normal':
                self.apply_dynamic_decoupling(dd_qubit, mode, time_diff)
            self.tab[qubit_val] += self.gate_lengths['CX'][0]
            self.tab[sec_qubit] = self.tab[qubit_val]

        # Add the CX gate to the circuit
        self.circ += X.on(qubit_val, sec_qubit)

    def handle_i_gate(self, qubit_val, mode):
        """
        Handle a CX gate by inserting DD if necessary and updating the circuit.

        Args:
            qubit_val (int): The current qubit.
            mode (str): 'normal' for no DD, 'xyxy' for dynamic decoupling.
        """
        if mode != 'normal' and qubit_val in self.qubits_to_consider:
            self.circ += DepolarizingChannel(0.002).on(qubit_val)
        if mode != 'normal' and qubit_val not in self.qubits_to_consider:
            self.circ += DepolarizingChannel(0.02).on(qubit_val)
        # Insert DD if necessary

        self.tab[qubit_val] += self.gate_lengths['I'][0]
        # Add the CX gate to the circuit
        self.circ += I.on(qubit_val)

    def apply_dynamic_decoupling(self, qubit_val, mode, time_diff):
        """
        Apply dynamic decoupling (DD) gates to a qubit during its idle time.

        Args:
            qubit_val (int): The qubit on which to apply DD.
            mode (str): 'normal' for no DD, 'xyxy' for dynamic decoupling.
            time_diff (float): The duration of idle time.
        """
        idle_time = time_diff
        dd_seq_length = self.gate_lengths['X'][0] * 2
        num_pulses = int(idle_time // dd_seq_length)
        if qubit_val not in self.qubits_to_consider:
            # self.circ += ThermalRelaxationChannel(0.20, 0.15,
            #                                       idle_time * 20).on(qubit_val)
            self.circ += PhaseDampingChannel(0.2).on(qubit_val)
        if mode == 'xx' and qubit_val in self.qubits_to_consider:
            for _ in range(num_pulses):
                self.circ += X.on(qubit_val)
                # self.circ += ThermalRelaxationChannel(0.20, 0.15,
                #                         (idle_time - num_pulses * dd_seq_length)/2).on(qubit_val)
                self.circ += PhaseDampingChannel(0.02).on(qubit_val)
                self.circ += X.on(qubit_val)
                # self.circ += ThermalRelaxationChannel(0.20, 0.15,
                #                         (idle_time - num_pulses * dd_seq_length)/2).on(qubit_val)
                self.circ += PhaseDampingChannel(0.02).on(qubit_val)
        elif mode =='xyxy' and qubit_val in self.qubits_to_consider:
            dd_seq_length = self.gate_lengths['X'][0] * 2 + self.gate_lengths['Y'][0] * 2
            num_pulses = int(idle_time // dd_seq_length)
            for _ in range(num_pulses):
                self.circ += X.on(qubit_val)
                self.circ += Y.on(qubit_val)
                self.circ += X.on(qubit_val)
                self.circ += Y.on(qubit_val)

    def handle_single_qubit_gate(self, gate_name, params, qubit_val):
        """
        Handle a single qubit gate and update the circuit and time tabs.

        Args:
            gate_name (str): The name of the gate.
            params (): Parameters of the gate (e.g., angles for U3 gate).
            qubit_val (int): The qubit to apply the gate to.
        """
        if gate_name == 'U3':
            theta, phi, lamda = params
            self.circ.u3(theta, phi, lamda, qubit_val)
            tab_gate = self.gate_lengths.get(gate_name, [0])[0]
            self.tab[qubit_val] += tab_gate
        elif gate_name == 'X':
            self.circ += X.on(qubit_val)
            tab_gate = self.gate_lengths.get(gate_name, [0])[0]
            self.tab[qubit_val] += tab_gate
        elif gate_name == 'RZ':
            theta_ = params[0]
            self.circ += RZ(theta_).on(qubit_val)
            tab_gate = self.gate_lengths.get(gate_name, [0])[0]
            # self.tab[qubit_val] += tab_gate
        elif gate_name == 'RX':
            theta_ = params[0]
            self.circ += RX(theta_).on(qubit_val)
            tab_gate = self.gate_lengths.get(gate_name, [0])[0]
            self.tab[qubit_val] += tab_gate
        elif gate_name == 'H':
            self.circ += H.on(qubit_val)
            tab_gate = self.gate_lengths.get(gate_name, [0])[0]
            self.tab[qubit_val] += tab_gate
        elif gate_name == 'I':
            self.circ += I.on(qubit_val)
            tab_gate = self.gate_lengths.get(gate_name, [0])[0]
            self.tab[qubit_val] += tab_gate


    def handle_non_clifford_gate(self, gate_name, params, qubit_val):
        """
        Handle a single qubit gate and update the circuit and time tabs.

        Args:
            gate_name (str): The name of the gate.
            params (): Parameters of the gate (e.g., angles for U3 gate).
            qubit_val (int): The qubit to apply the gate to.
        """
        if gate_name == 'U3':
            if self.ts < self.layer_set:
                theta, phi, lamda = params
                self.circ.u3(theta, phi, lamda, qubit_val)
            elif self.ts >= self.layer_set:
                closest_gate = self.get_closest_clifford_gate(gate_name, qubit_val, params)
                self.circ_skeleton += closest_gate
            tab_gate = self.gate_lengths.get(gate_name, [0])[0]
            self.tab[qubit_val] += tab_gate
        elif gate_name == 'X':
            self.circ += X.on(qubit_val)
            tab_gate = self.gate_lengths.get(gate_name, [0])[0]
            self.tab[qubit_val] += tab_gate
        elif gate_name == 'RZ':
            if self.ts < self.layer_set:
                theta_ = params[0]
                self.circ += RZ(theta_).on(qubit_val)
            elif self.ts >= self.layer_set:
                closest_gate = self.get_closest_clifford_gate(gate_name, qubit_val, params)
                self.circ_skeleton += closest_gate
            tab_gate = self.gate_lengths.get(gate_name, [0])[0]
            # self.tab[qubit_val] += tab_gate
        elif gate_name == 'RX':
            if self.ts < self.layer_set:
                theta_ = params[0]
                self.circ += RX(theta_).on(qubit_val)
            elif self.ts >= self.layer_set:
                closest_gate = self.get_closest_clifford_gate(gate_name, qubit_val, params)
                self.circ_skeleton += closest_gate
            tab_gate = self.gate_lengths.get(gate_name, [0])[0]
            self.tab[qubit_val] += tab_gate
        elif gate_name == 'H':
            self.circ += H.on(qubit_val)
            tab_gate = self.gate_lengths.get(gate_name, [0])[0]
            self.tab[qubit_val] += tab_gate
        elif gate_name == 'I':
            self.circ += I.on(qubit_val)
            tab_gate = self.gate_lengths.get(gate_name, [0])[0]
            self.tab[qubit_val] += tab_gate

        # Update time tab for this qubit
        # tab_gate = max(diff,self.gate_lengths.get(gate_name, [0])[0])
        # tab_gate =  self.gate_lengths.get(gate_name, [0])[0]
        # self.tab[qubit_val] += tab_gate

    def finalize_measurements(self, analog_table, skeleton_flag):
        """
        Finalize the circuit by adding measurement operations.

        Args:
            analog_table (DataFrame): The analog instruction table.
            to determine qubits for measurement.
            skeleton_flag: Select whether to generate a skeleton circuit.
        """
        qubit_to_measure = []
        for col in analog_table.columns:
            if 'Measure' in analog_table[col].values:
                qubit_to_measure.append(int(col))
        for qubit in sorted(qubit_to_measure, reverse=True):
            if skeleton_flag:
                self.circ_skeleton += Measure().on(qubit)
            else:
                self.circ += Measure().on(qubit)

    def get_closest_clifford_gate(self, gate, qubit, params=None):
        """
        Get the closest Clifford gate for a given non-Clifford gate using operator norm.

        Args:
            gate (str): The name of the non-Clifford gate.
            qubit (int): The qubit on which the gate is applied.
            params (tuple): Parameters of the gate if any.

        Returns:
            function: The closest Clifford gate function.
        """
        # Define Clifford gates
        clifford_group = self.generate_clifford_group()

        # Compute target matrix
        target_matrix = self.compute_target_matrix(gate, params)

        # all single qubit clifford gates
        closest_gate = None
        min_norm = float('inf')
        for circ, matrix in clifford_group:
            norm = self.operator_norm(target_matrix, matrix)
            if norm < min_norm:
                min_norm = norm
                closest_gate = circ
        closest_circ = Circuit()
        for gate_ in closest_gate:
            closest_circ += gate_.on(qubit)

        return closest_circ

    def generate_clifford_group(self):
        """
        Generate the group of 24 single-qubit Clifford elements.

        Returns:
            list: A list of tuples where each tuple contains a Clifford circuit
            and its corresponding matrix representation.
        """
        clifford_group = []

        for i in range(24):
            ele = query_single_qubit_clifford_elem(i)
            circ_clifford = decompose_stabilizer(ele)
            matrix = circ_clifford.matrix()
            clifford_group.append((circ_clifford, matrix))

        return clifford_group

    def compute_target_matrix(self, gate, params):
        """
        Compute the target matrix for the given gate and parameters.
        """
        if gate == 'U3':
            theta, phi, lamda = params
            return U3(theta, phi, lamda).matrix()

        if gate == 'RZ':
            return RZ(params[0]).matrix()

        if gate == 'RX':
            return RX(params[0]).matrix()

        raise ValueError(f"Unrecognized gate: {gate}")

    def operator_norm(self, u, v):
        """
        Input: Unitary matrices U, V
        Output: The operator norm of M which is M = U-V.
        """
        m = u - v
        mdm = np.matmul(m.conj().T, m)
        e_vals, _ = np.linalg.eig(mdm)
        max_e_val = max(list(e_vals))

        return np.sqrt(max_e_val)

    def generate_combinations(self, analog_frame):
        """
        Function to generate combinations
        """
        qubit_list = [int(i) for i in analog_frame.columns]
        all_combinations = []
        rename_table = [(qubit_list[i], i) for i in range(len(qubit_list))]

        sequence_strings, sequence_ids = [], []
        for r in range(len(qubit_list) + 1):
            combinations_object = itertools.combinations(qubit_list, r)
            comb_list = list(combinations_object)
            all_combinations += comb_list
            for curr_combo in comb_list:
                seq_str, seq_id = (self.generate_seq_id_from_combination
                                   (curr_combo, len(qubit_list), rename_table))
                sequence_strings.append(seq_str)
                sequence_ids.append(seq_id)

        return all_combinations, sequence_strings, sequence_ids

    def generate_seq_id_from_combination(self, combo, sequence_length, rename_list):
        """
        Function to generate sequence ID from combination
        """
        bitstring = [0 for _ in range(sequence_length)]
        for i in combo:
            for remap_entry in rename_list:
                if i == remap_entry[0]:
                    bitstring[remap_entry[1]] = 1
                    break
        bitstring.reverse()
        sequence_string = ''.join(str(bit) for bit in bitstring)
        sequence_id = self.convert_key_to_decimal(sequence_string, len(sequence_string))
        return sequence_string, sequence_id

    def convert_key_to_decimal(self, string, width):
        """
        Function to convert a key to decimal
        """
        power = width - 1
        dec_key = 0
        for c in string:  # go through every character
            dec_key = dec_key + np.power(2, power) * int(c)
            power = power - 1
        return dec_key
