import os
import pickle
import csv
import numpy as np

# Constants
randseeds = list(range(5))
Dict = {
    # '--Output_path': '/mnt/nas-new/home/yange/nature_comm/pack2/output/{}/{}_{}layers_{}_seed{}_{}.pkl'
    '--Output_path': '/mnt/nas-new/home/yange/nature_comm/pack3/output/{}/{}/{}_{}_s{}_l{}.pkl'
}

# Parameters
# runid = 'paramhiq'
runid = 'layerhiq'
# points = ['F2', 'BeH2'] 
# points = ['F2'] 
# points=['H2'] 
points=['LiH'] 
# points = ['BeH2'] 
# layers = [50,60,66,70,72,74,76,78,80,84,90,100,110,120]
# layers = [25, 30, 35, 40, 45, 50, 55, 60]
# layers = [300, 350, 400, 450, 500]
# layers = [150, 175, 200, 225, 250]

# layers = [25,50,80,90,100,110,120,130,140,150,160,170,250,350,450]
layers = [25,40,45,50,55,60,65,70,75,80,85,125,175,225]

# layers = [1,2,3,4,5,6,7,8,9,10]
# layers = [1,2,3,4,5]
# gate_types = ['BS', 'RBS', 'RCNOT']
# gate_types = ['BS', 'RBS']
# gate_types = ['BS', 'RBS']
gate_types = ['RCNOT']
circuit_type = 'newhea'

# Functions to load data
def load_energy(file_path):
    with open(file_path, 'rb') as f:
        energy = pickle.load(f)
    return energy

def load_standard_energy(file_path):
    with open(file_path, 'rb') as f:
        fci_energy = pickle.load(f)
    return fci_energy

# Function to process data for a given gate type
def process_layer_data(point, gate_type, circuit_type, layers, runid, randseeds):
    # Define output path for each molecule and gate type
    output_csv_path = f"/mnt/nas-new/home/yange/nature_comm/pack3/excel/{point}_{gate_type}_layer.csv"

    # Initialize the header
    header = ['Layers', 'Gate Type', 'Mean Energy', 'Standard Deviation', 'Max Energy', 'Min Energy', 'Standard Energy', 'Difference (Mean - Standard)','Difference (Min - Standard)', 'Log Mean Difference', 'Log Standard Deviation']
    data = []
    # Load standard energy
    # standard_energy_path = os.path.join(f"/mnt/nas-new/home/yange/nature_comm/pack2/energy/{point}_dist1.34.pkl")
    # standard_energy_path = os.path.join(f"/mnt/nas-new/home/yange/nature_comm/pack3/energy/{point}_dist1.42.pkl")
    standard_energy_path = os.path.join(f"/mnt/nas-new/home/yange/nature_comm/pack3/data/pks_data/{point}/energy/{point}_1.pkl")
    
    print(f"Loading standard energy from: {standard_energy_path}")
    if os.path.exists(standard_energy_path):
        standard_energy = load_standard_energy(standard_energy_path)
        print(f"Loaded standard energy: {standard_energy}")
    else:
        standard_energy = None
        print(f"Standard energy file does not exist: {standard_energy_path}")
    # Fill data list
    for layer in layers:
        energies = []
        log_differences = []
        for seed in randseeds:
            file_path = Dict['--Output_path'].format(runid, point, gate_type, circuit_type, seed, layer)
            # file_path = Dict['--Output_path'].format(runid, point, layer, gate_type, seed, circuit_type)
            print(f"Trying to access file: {file_path}") 
            if os.path.exists(file_path):
                print(f"File exists: {file_path}") 
                energy = load_energy(file_path)
                print(f"Successfully loaded energy data from: {file_path}")
                # final_energy = energy[-1][-1]
                final_energy = np.min(energy[-1])
                energies.append(final_energy)
                if standard_energy is not None:
                    difference = final_energy - standard_energy
                    log_differences.append(np.log10(np.abs(difference)))
            else:
                print(f"File does not exist: {file_path}") 
                energies.append(None)
                log_differences.append(None)

        # Calculate mean, std, max, and min excluding None values
        valid_energies = [e for e in energies if e is not None]
        print(f"Energies for layer {layer}: {energies}")
        valid_log_differences = [d for d in log_differences if d is not None]
        if valid_energies:
            mean_energy = np.mean(valid_energies)
            std_energy = np.std(valid_energies)
            max_energy = np.max(valid_energies)
            min_energy = np.min(valid_energies)
            min_difference=min_energy-standard_energy
        else:
            mean_energy = None
            std_energy = None
            max_energy = None
            min_energy = None
            min_difference=None


        if valid_log_differences:
            log_mean_difference = np.mean(valid_log_differences)
            log_std_difference = np.std(valid_log_differences)
        else:
            log_mean_difference = None
            log_std_difference = None



        # Calculate difference
        if mean_energy is not None and standard_energy is not None:
            difference = mean_energy - standard_energy
        else:
            difference = None

        row = [layer, gate_type, mean_energy, std_energy, max_energy, min_energy, standard_energy, difference, min_difference, log_mean_difference, log_std_difference]
        data.append(row)

    # Write data to CSV file
    print(f"Data to write for {output_csv_path}: {data}") 
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Write header
        writer.writerows(data)   # Write data rows

    print(f"CSV file saved to {output_csv_path}")

# Process each molecule and gate type
for point in points:
    for gate_type in gate_types:
        process_layer_data(point, gate_type, circuit_type, layers, runid, randseeds)

