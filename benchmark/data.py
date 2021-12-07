import json
import os


# initialize json files
def initdata(mole):
    meta = {
        "transformation": "jordan_wigner",
        "optimizer": "bfgs",
        "simulator": "mindspore",
        "package": "mindquantum",
        "mol_name": mole
    }

    energies = {
        "meta": meta,
        "bond_lengths": [],
        "energies": {
            "Hartree-Fock": [],
            "full-CI": [],
            "1-UpCCGSD": [],
            "HEA": [],
            "CCSD": [],
            "UCCSD": []
        }
    }

    times = {
        "meta": meta,
        "bond_lengths": [],
        "times": {
            "Hartree-Fock": [],
            "full-CI": [],
            "HEA": [],
            "1-UpCCGSD": []
        }
    }

    parameters = {
        "meta": meta,
        "bond_lengths": [],
        "parameters": {
            "HEA": [],
            "1-UpCCGSD": []
        }
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           r'data/mindquantum_energies_{}.json'.format(mole)),
              'w+',
              newline='') as f:
        b = json.dumps(energies)
        f.write(b)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           r'data/mindquantum_times_{}.json'.format(mole)),
              'w+',
              newline='') as f:
        b = json.dumps(times)
        f.write(b)

    with open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         r'data/mindquantum_parameters_{}.json'.format(mole)),
            'w+',
            newline='') as f:
        b = json.dumps(parameters)
        f.write(b)


def savedata(mole_name, results, method, init=False):
    '''
    Method used for saving data in json files for different molecules.

    Args:
        mole_name(string): molecule name
        results(2d-list): for i-th term in the list, [i][0] should be bond length, [i][1] should be energy,
            [i][2] should be time being used, [i][3] should be parameter number
        method(string): 
    '''
    with open(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            r'data/mindquantum_energies_{}.json'.format(mole_name)),
              'r+',
              newline='') as f:
        data = f.read()
        energies = json.loads(data)

    with open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         r'data/mindquantum_times_{}.json'.format(mole_name)),
            'r+',
            newline='') as f:
        data = f.read()
        times = json.loads(data)

    with open(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            r'data/mindquantum_parameters_{}.json'.format(mole_name)),
              'r+',
              newline='') as f:
        data = f.read()
        parameters = json.loads(data)

    # Adjust for different methods
    if init:
        energies["energies"][method] = []
        times["times"][method] = []
        parameters["parameters"][method] = []

    for i in range(len(results)):
        """ energies["bond_lengths"].append(results[i][0])
        times["bond_lengths"].append(results[i][0])
        parameters["bond_lengths"].append(results[i][0]) """
        energies["energies"][method].append(results[i][1])
        times["times"][method].append(results[i][2])
        parameters["parameters"][method].append(results[i][3])

    with open(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            r'data/mindquantum_energies_{}.json'.format(mole_name)),
              'w+',
              newline='') as f:
        b = json.dumps(energies)
        f.write(b)

    with open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         r'data/mindquantum_times_{}.json'.format(mole_name)),
            'w+',
            newline='') as f:
        b = json.dumps(times)
        f.write(b)

    with open(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            r'data/mindquantum_parameters_{}.json'.format(mole_name)),
              'w+',
              newline='') as f:
        b = json.dumps(parameters)
        f.write(b)


def rounddata(mole_name):
    '''
    Method used for round data in json files.

    Args:
        mole_name (string): molecule name
    '''
    with open(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            r'data/mindquantum_energies_{}.json'.format(mole_name)),
              'r+',
              newline='') as f:
        data = f.read()
        energies = json.loads(data)

    with open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         r'data/mindquantum_times_{}.json'.format(mole_name)),
            'r+',
            newline='') as f:
        data = f.read()
        times = json.loads(data)

    with open(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            r'data/mindquantum_parameters_{}.json'.format(mole_name)),
              'r+',
              newline='') as f:
        data = f.read()
        parameters = json.loads(data)

    for terms in energies["energies"]:
        energies["energies"][terms] = [
            round(num, 6) for num in energies["energies"][terms]
        ]
        print(energies["energies"][terms])

    for terms in times["times"]:
        times["times"][terms] = [
            round(num, 3) for num in times["times"][terms]
        ]
        print(times["times"][terms])

    for terms in parameters["parameters"]:
        parameters["parameters"][terms] = [
            round(num, 0) for num in parameters["parameters"][terms]
        ]
        print(parameters["parameters"][terms])

    energies["bond_lengths"] = [
        round(num, 1) for num in energies["bond_lengths"]
    ]
    times["bond_lengths"] = [round(num, 1) for num in times["bond_lengths"]]
    parameters["bond_lengths"] = [
        round(num, 1) for num in parameters["bond_lengths"]
    ]
    print(energies["bond_lengths"])
    print(times["bond_lengths"])
    print(parameters["bond_lengths"])

    with open(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            r'data/mindquantum_energies_{}.json'.format(mole_name)),
              'w+',
              newline='') as f:
        b = json.dumps(energies)
        f.write(b)

    with open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         r'data/mindquantum_times_{}.json'.format(mole_name)),
            'w+',
            newline='') as f:
        b = json.dumps(times)
        f.write(b)

    with open(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            r'data/mindquantum_parameters_{}.json'.format(mole_name)),
              'w+',
              newline='') as f:
        b = json.dumps(parameters)
        f.write(b)


if __name__ == "__main__":
    bond_len = 0.8
    hf = 12.3
    mole = 'H4'
    rounddata(mole)
