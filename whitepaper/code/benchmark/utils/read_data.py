# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Read pytest-benchmark data."""
import json
import re
import numpy as np
import matplotlib.pyplot as plt

framework_color_mark = {
    'mindquantum_complex128':['#f00909', 'o'],
    'mindquantum_complex64':['#f00909', '*'],
    'qiskit':['#5a5a5a', '>'],
    'pennylane':['#5a5a5a', '<'],
    'pyqpanda':['#5a5a5a', '.'],
    'qulacs':['#5a5a5a', '4'],
    'tensorcircuit_complex128':['#5a5a5a', 'v'],
    'tensorcircuit_complex64':['#5a5a5a', '^'],
    'tensorflow_quantum':['#5a5a5a', 'p'],
    'intel':['#5a5a5a', 'H'],
}

def load_file(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

def ext_qubit(file_name):
    if 'simple_circuit' in file_name:
        match = re.search(r"(\w+)_qubit_\d+_size_\d+\.json", file_name)
        n_qubits = int(match.group(0).split('_')[3])
        return n_qubits

    if 'random_circuit' in file_name:
        match = re.search(r"(\w+)_qubit_\d+_size_\d+\.json", file_name)
        n_qubits = int(match.group(0).split('_')[3])
        return n_qubits

    if 'regular_4' in file_name:
        match = re.search(r"(\w+)_qubit_\d+\.json", file_name)
        n_qubits = int(match.group(0).split('_')[3].split('.')[0])
        return n_qubits

    raise ValueError(f'Unknown task: {file_name}')

def ext_task_type(file_name):
    for i in ['simple_circuit', 'random_circuit', 'regular_4']:
        if i in file_name:
            return i
    raise ValueError(f'Unknown task: {file_name}')

def ext_framework(file_name):
    return file_name.split('_')[1]

def _ext_info(file_path):
    data = load_file(file_path)
    out = []
    for b in data['benchmarks']:
        name = b['name']
        mean_time = b['stats']['mean']
        stddev = b['stats']['stddev']
        framework = ext_framework(name)
        if framework == 'mindquantum':
            framework += "_"
            framework += b["params"]['dtype'][15:-1].split('.')[-1]
        elif framework == 'tensorcircuit':
            framework += "_"
            framework += b["params"]['dtype']
        if 'gpu' in name:
            framework += '_gpu'
        out.append([framework, ext_task_type(name), ext_qubit(name), mean_time,  stddev])
    return out

def append_data(out, new_data, strategy='override'):
    hit = False
    for i in out:
        if i[:3] == new_data[:3]:
            hit = True
            print(f"Hit same task configuration: {i[:3]}. Strategy: {strategy}")
            if strategy == 'override':
                i[3] = new_data[3]
                i[4] = new_data[4]
    if not hit:
        out.append(new_data)

def ext_info(file_name):
    if isinstance(file_name, str):
        file_name = [file_name]
    out = []
    for f in file_name:
        for i in _ext_info(f):
            append_data(out, i)
    return out

def get_task(task_name, data):
    out = {}
    for i in data:
        if i[1] == task_name:
            d = out.get(i[0], [])
            d.append(i[2:])
            out[i[0]] = d
    return {i:np.array(j) for i, j in out.items()}

def show_task(task_name, data, log=False):
    data = get_task(task_name, data)
    for framework, j in data.items():
        n_qubits = j[:, 0].astype(int)
        mean_time = j[:, 1]
        stddev = j[:, 2]
        color, mark = framework_color_mark.get(framework[:-4], ['#5a5a5a', 'D'])
        plt.plot(n_qubits, mean_time, "--", color=color)
        plt.plot(n_qubits, mean_time, mark, color=color, label=framework)
    if log:
        plt.yscale('log')
    plt.legend()
    plt.show()

def data_to_csv(file_path, data):
    if not file_path.endswith('.csv'):
        file_path += '.csv'
    out = '\n'.join(','.join([str(j) for j in i]) for i in data)
    with open(file_path, 'w') as f:
        f.writelines(out)

# file_path1 = '../src/.benchmarks/Linux-CPython-3.9-64bit/0002_mindquantum.json'
# file_path2 = '../src/.benchmarks/Linux-CPython-3.9-64bit/0015_mindquantum.json'
# data = ext_info([file_path1, file_path2])
# print(get_task('random_circuit', data))
# show_task('random_circuit', data)
# ext_info(file_path1)
