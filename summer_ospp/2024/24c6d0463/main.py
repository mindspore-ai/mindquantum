"""
Main module for executing quantum circuit experiments with Dynamical Decoupling (DD) applied.
This script orchestrates the flow of operations, invoking relevant utilities from other modules.
"""
import os
import glob
import pandas as pd
from mindquantum.algorithm.compiler import DAGCircuit
from scripts import QCManager
from scripts import DDManager
pd.options.mode.chained_assignment = None  # default='warn'


def initialize_params(circ_prep):
    """
    Initialize parameters like qubits, shots, and repeats.

     Args:
        circ_prep (QCManager): An instance of the QuantumCircuitManager
                                           class to manage quantum circuit operations.
    Returns:
        program_qubits (int): The number of qubits in the quantum circuit.
        total_shots (int): The total number of shots (experiments) to be executed.
    """
    shots = 2192
    repeats = 2
    program_qubits = circ_prep.num_qubits_from_qasm()
    if program_qubits > 8:
        repeats = 4
    total_shots = int(repeats * shots)
    return program_qubits, total_shots

def compile_circuit(circ_prep, program):
    """
    Compile the QASM circuit and analyze its depth and structure.

    Args:
        circ_prep (QuantumCircuitManager): An instance of the QuantumCircuitManager class used
                                           for managing quantum circuit compilation.
        program (str): The QASM representation of the quantum circuit.

    Returns:
        dag (DAGCircuit): A DAG representation of the compiled quantum circuit.
        circuit_depth (int): The depth of the compiled circuit, representing the
                             longest path through the DAG.
    """
    qc_out = circ_prep.recursive_compile_noise_adaptive([circ_prep.read_qasm(program)])[0]
    dag = DAGCircuit(qc_out)
    circuit_depth = dag.depth()
    return dag, circuit_depth

def generate_gate_lengths_and_frames(circ_prep, dag, circuit_depth, new_program_path):
    """
    Generate gate lengths and convert discrete frames to analog frames.

    Args:
        circ_prep (QuantumCircuitManager): Instance used for managing circuit operations.
        dag (DAGCircuit): DAG representation of the quantum circuit.
        circuit_depth (int): Depth of the compiled circuit.
        new_program_path (str): Path to save the analog frame CSV.

    Returns:
        gate_lengths (dict): Dictionary of gate types and their lengths.
        analog_frame (DataFrame): Analog frame created from the discrete frame.
    """
    qubit_set = list(range(circ_prep.num_qubits_from_qasm()))
    empty_frame = circ_prep.create_instruction_table(qubit_set, circuit_depth + 10)
    p_frame = circ_prep.populate_instruction_table(qubit_set, dag, empty_frame)

    discrete_frame = circ_prep.zero_filter(p_frame)

    gate_lengths = circ_prep.get_all_instruction_lengths()

    analog_frame = circ_prep.discrete_to_analog(discrete_frame, gate_lengths)
    frame_name = f"{new_program_path}_analog_IDT_Frame.csv"
    analog_frame.to_csv(frame_name, index=False)
    return gate_lengths, analog_frame

def sim_baseline_circ(circ_prep, total_shots, all_circuits, baseline_counts):
    """
    Simulate the baseline circuit and calculate fidelity.

    Args:
        circ_prep (QuantumCircuitManager): Instance used for managing circuit operations.
        total_shots (int): Total number of shots for the simulation.
        all_circuits (list): List of circuits to be executed.
        baseline_counts (list): Baseline counts for fidelity comparison.

    Returns:
        None: Saves the counts and fidelity data to files.
    """
    all_el_fidelity = []
    counts_vector = circ_prep.execute_on_ideal_machine(
        all_circuits, shots=total_shots, mode='normal')

    counts_vector_txt = pd.DataFrame(counts_vector)
    counts_vector_txt.to_csv(f"{circ_prep.result_path}_counts_vector.txt", index=False)

    for el_count in counts_vector:
        fide = 1 - circ_prep.total_variation_distance(baseline_counts[0].data, el_count.data)
        all_el_fidelity.append(fide)
    all_el_fidelity_txt = pd.DataFrame(all_el_fidelity)
    all_el_fidelity_txt.to_csv(f'{circ_prep.result_path}_all_el_fidelity.txt', index=False)

def sim_skeleton_circ(circ_prep, total_shots, all_skeleton_circuits, baseline_counts):
    """
    Simulate the skeleton circuit and calculate fidelity.

    Args:
        circ_prep (QuantumCircuitManager): Instance used for managing circuit operations.
        total_shots (int): Total number of shots for the simulation.
        all_skeleton_circuits (list): List of circuits to be executed.
        baseline_counts (list): Baseline counts for fidelity comparison.

    Returns:
        None: Saves the counts and fidelity data to files.
    """

    all_skeleton_el_fidelity = []
    counts_vector_skeleton = circ_prep.execute_on_ideal_machine(
        all_skeleton_circuits, shots=total_shots, mode='dd')

    counts_vector_skeleton_txt = pd.DataFrame(counts_vector_skeleton)
    (counts_vector_skeleton_txt.to_csv
     (f'{circ_prep.result_path}_counts_vector_skeleton.txt', index=False))

    for el_count in counts_vector_skeleton:
        fide = 1 - circ_prep.total_variation_distance(baseline_counts[0].data, el_count.data)
        all_skeleton_el_fidelity.append(fide)
    all_skeleton_el_fidelity_txt = pd.DataFrame(all_skeleton_el_fidelity)
    (all_skeleton_el_fidelity_txt.to_csv
     (f'{circ_prep.result_path}_all_skeleton_el_fidelity.txt', index=False))

def dd_for_baseline_circ(all_combinations, dd_manager, analog_table, circ_prep):
    """
    Generate DD sequences for the baseline circuit.

    Args:
        all_combinations (list): List of all possible qubit combinations for DD.
        dd_manager (DDManager): Manager responsible for handling DD operations.
        analog_table (pd.DataFrame): Analog instruction table for the circuit.
        circ_prep (QuantumCircuitManager): Instance used for managing circuit operations.

    Returns:
        list: List of circuits generated with DD sequences applied.
    """
    all_circuits = []
    all_gate_counts = []
    for el in all_combinations:
        # Creating the circuit for particular DD combination -- XX
        baseline_circ = dd_manager.dataframe_to_circ(analog_table,
                                              qubits_to_consider=list(el), mode='xx')
        all_circuits.append(baseline_circ)
        op_dicts = circ_prep.gather_all_gate_counts(baseline_circ)
        all_gate_counts.append(op_dicts)

    return all_circuits

def dd_for_skeleton_circ(all_combinations, dd_manager, analog_table, circ_prep):
    """
    Generate DD sequences for the skeleton circuit.

    Args:
        all_combinations (list): List of all possible qubit combinations for DD.
        dd_manager (DDManager): Manager responsible for handling DD operations.
        analog_table (pd.DataFrame): Analog instruction table for the circuit.
        circ_prep (QuantumCircuitManager): Instance used for managing circuit operations.

    Returns:
        list: List of circuits generated with DD sequences applied.
    """
    all_circuits = []
    all_skeleton_gate_counts = []
    for el in all_combinations:
        # Creating the circuit for particular DD combination -- XX
        circ = dd_manager.dataframe_to_skeleton_circ(analog_table,
                                                         qubits_to_consider=list(el), mode='xx')
        all_circuits.append(circ)
        op_dicts = circ_prep.gather_all_gate_counts(circ[0])
        all_skeleton_gate_counts.append(op_dicts)

    return all_circuits

def generate_baseline_circuit(circ_prep, dd_manager, analog_frame, program_qubits, total_shots):
    """
    Generate the baseline circuit and execute it on an ideal machine.

    Args:
        circ_prep (QuantumCircuitManager): Manager handling circuit operations.
        dd_manager (DDManager): Manager responsible for handling DD operations.
        analog_frame (pd.DataFrame): Analog instruction table for the circuit.
        program_qubits (int): Number of qubits in the program.
        total_shots (int): Total number of shots for the experiment.

    Returns:
        list: Measurement counts from executing the baseline circuit on the ideal machine.
    """
    baseline_circ = dd_manager.dataframe_to_circ(analog_frame,
                                qubits_to_consider=list(range(program_qubits)), mode='normal')
    baseline_counts = circ_prep.execute_on_ideal_machine([baseline_circ],
                                        total_shots, mode='normal')
    return baseline_counts

def run_all_experiments(program, new_program_path):
    """
    Execute all experiments on the given program and save results.

    Args:
        program: The program or circuit to be executed in the experiments.
        new_program_path: The file path where the results of the program will be saved.
    Returns:
        dict: A dictionary containing data from all experiments.
    """
    circ_prep = QCManager(program, new_program_path)
    program_qubits, total_shots = initialize_params(circ_prep)
    dag, circuit_depth = compile_circuit(circ_prep, program)
    gate_lengths, analog_frame = generate_gate_lengths_and_frames(circ_prep,
                                        dag, circuit_depth, new_program_path)
    dd_manager = DDManager(gate_lengths, program_qubits)
    # Find all possible DD combinations
    # all combinations is a list of all possible combinations of qubits
    all_combinations, _, _ = dd_manager.generate_combinations(analog_frame)
    # Generate baseline circuit and counts
    baseline_counts = generate_baseline_circuit(circ_prep, dd_manager, analog_frame, program_qubits,
                                                               total_shots)
    # Generating all DD sequences for circuit
    print('Generating all DD sequences for baseline circuit')
    all_circuits = dd_for_baseline_circ(all_combinations, dd_manager,
                                        analog_frame, circ_prep)
    circ_prep.save_circuit_svg(all_circuits[0], 'baseline')
    circ_prep.save_circuit_svg(all_circuits[-1], 'baseline_all')
    print('Generating all DD sequences for skeleton circuit')
    all_skeleton_circuits = dd_for_skeleton_circ(all_combinations, dd_manager,
                                                 analog_frame, circ_prep)
    circ_prep.save_circuit_svg(all_skeleton_circuits[0][0], 'skeleton')

    # Circuit conversion qasm for particular DD combination
    # circ_prep.circ_to_qasm(all_circuits, sequence_strings, "baseline")
    # circ_prep.circ_to_delay_qasm(all_circuits, sequence_strings, "baseline")
    # circ_prep.circ_to_qasm(all_skeleton_circuits, sequence_strings, "skeleton")
    # circ_prep.circ_to_delay_qasm(all_skeleton_circuits, sequence_strings, "skeleton")

    print('Real Machine Simulation Step for baseline circ')
    sim_baseline_circ(circ_prep, total_shots, all_circuits, baseline_counts)
    print('Real Machine Simulation Step for skeleton circ')
    sim_skeleton_circ(circ_prep, total_shots, all_skeleton_circuits, baseline_counts)
    print(f'DD successfully saved to {new_program_path}')

if __name__ == '__main__':
    PREFIX_PATH = "benchmarks/"
    filelist = glob.glob(os.path.join(PREFIX_PATH, '*.qasm'))
    NEW_PATH = 'result/'

    for program_ in filelist:
        path, program_name = os.path.split(program_)
        each_program_path = os.path.join(NEW_PATH, program_name)
        run_all_experiments(program_, each_program_path)
