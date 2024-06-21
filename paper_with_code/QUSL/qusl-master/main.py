import os
import time
import argparse
from os import listdir
import data_processing
import genetic_algorithms
from functools import partial
from multiprocessing import Pool
from os.path import isfile, join
from mindquantum.io import OpenQASM
from mindquantum.core.circuit import Circuit
from constants import NUMBER_OF_GENERATIONS, NUMBER_OF_QUBITS, \
    POPULATION_SIZE, NUM_TRIPLETS
from toolbox import initialize_toolbox


def main():
    device = 'window'
    dataset = 'landscape'
    if device == 'window':
        base_path = "/"
    elif device == 'linux':
        base_path = "/home/ubuntu/qusl/"
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        raise ValueError("Unknown device type")

    start = time.perf_counter()

    """Runs the genetic algorithm based on the global constants
    """
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-p", "--POPSIZE", help="Size of the population")
    parser.add_argument("-g", "--NGEN", help="The number of generations")
    parser.add_argument("-q", "--NQUBIT", help="The number of qubits")
    parser.add_argument("-i", "--INDEX", help="Index of desired state")
    # FIXME -id is illegal (it means -i -d)
    parser.add_argument("-id", "--ID", help="ID of the saved file")

    # Read arguments from command line
    args = parser.parse_args()

    population_size = int(args.POPSIZE) if args.POPSIZE else POPULATION_SIZE
    number_of_generations = int(
        args.NGEN) if args.NGEN else NUMBER_OF_GENERATIONS
    number_of_qubits = int(args.NQUBIT) if args.NQUBIT else NUMBER_OF_QUBITS
    triplets, indecies = RGB_data_processing.generate_landscape_triplets(base_path, dataset, num_triplets=5000, testing=False)

    print('##########starting Evolution##########')
    runtime1 = round(time.perf_counter() - start, 2)
    print("runtime first", runtime1)
    toolbox = initialize_toolbox(number_of_qubits)
    EVO = RGB_genetic_algorithms.Evolution(triplets, number_of_qubits)

    # evolution_instance.evolution(toolbox)
    pop, fitness_ranks = EVO.evolution(toolbox) 

    combined_data = list(zip(pop, fitness_ranks))

    sorted_data = sorted(combined_data, key=lambda x: x[1], reverse=True)

    sorted_pop, sorted_fitness_ranks = zip(*sorted_data)

    next_individual = sorted_pop
    print('fitness of last population', sorted_fitness_ranks)
    count = 1
    for qc in next_individual:
        cir = qc.circuit
        circuit = EVO.to_circuit(cir)
        string = OpenQASM().to_string(circuit)
        OpenQASM().from_string(string)
        first_circuit = f'{base_path}/result/qasm/best_Candidate_{count}.qasm'
        openqasm = OpenQASM()
        openqasm.to_file(first_circuit, circuit, version='2.0')
        # print(circuit)  # print Encoder
        # circuit.summary()  # summary Encoder quantum circuit
        circuit.svg().to_file(f'{base_path}/result/svg/best_Candidate_{count}.svg')
    runtime = round(time.perf_counter() - start, 2)
    print("runtime train", runtime)


    print('##########starting test##########')
    test_cir = Circuit()
    number_of_qubits = NUMBER_OF_QUBITS
    evolution = RGB_genetic_algorithms.Evolution(triplets, number_of_qubits=number_of_qubits)
    file_name = f'{base_path}/result/qasm/best_Candidate_1.qasm'
    cir = OpenQASM().from_file(file_name)

    path = f'{base_path}/datasets/landscapes_test'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    files.sort()
    triplets, indecies = RGB_data_processing.generate_landscape_triplets(base_path, dataset, num_triplets=100, testing=False)
    with Pool(processes=3) as pool:
        partial_process_feature = partial(RGB_data_processing.process_feature, triplets=triplets, indecies=indecies, evolution=evolution,
                                          circuit=test_cir, number_of_qubits=number_of_qubits, path=path, files=files)
        pool.map(partial_process_feature, range(2))
    runtime2 = round(time.perf_counter() - start, 2)
    print("runtime test", runtime2)


if __name__ == '__main__':
    main()
