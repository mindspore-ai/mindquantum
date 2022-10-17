import ast
import json
import os
import time
from math import sin, cos
import numpy as np


def fsim(teta, phi):
    return [[1,                0,                0,                          0],
            [0,        cos(teta), -1.j * sin(teta),                          0],
            [0, -1.j * sin(teta),        cos(teta),                          0],
            [0,                0,                0, cos(phi) - 1.0j * sin(phi)]]


gates = {
    'cx': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    'cz': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
    'is': [[1, 0, 0, 0], [0, 0, 1.0j, 0], [0, 1.0j, 0, 0], [0, 0, 0, 1]],
    'i': [[1, 0], [0, 1]],
    'x': [[0, 1], [1, 0]],
    'y': [[0, 1j], [-1j, 0]],
    'x_1_2': np.array([[1, -1j], [-1j, 1]]) * 0.5j ** 0.5,
    'y_1_2': np.array([[1, -1], [1, 1]]) * 0.5j ** 0.5,
    'hz_1_2': np.array([[1, -1.j ** 0.5], [(-1.j) ** 0.5, 1]]) * 0.5j ** 0.5,
    'fs': (2, fsim),
    'rz': (1, lambda teta: [[cos(teta / 2) - 1.j * sin(teta / 2), 0], [0, cos(teta / 2) + 1.j * sin(teta / 2)]]),
    'z': [[1, 0], [0, -1]],
    'h': np.array([[1, 1], [1, -1]]) / 2. ** 0.5,
    't': [[1, 0], [0, (1 + 1.0j) / 2. ** 0.5]],
    's': [[1, 0], [0, 1.0j]]
}


_ein_ind = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


def contract2_numpy(x, xi, y, yi, out):
    en = {s: i for (i, s) in enumerate(set(list(xi) + list(yi) + list(out)))}
    xind = ''.join([_ein_ind[en[i]] for i in xi])
    yind = ''.join([_ein_ind[en[i]] for i in yi])
    oind = ''.join([_ein_ind[en[i]] for i in out])
    h = f'{xind},{yind} -> {oind}'
    res = np.einsum(h, x, y, optimize=(len(en) > 15))
    return res, out


def transpose_numpy(x, inp, out):
    en = {s: i for (i, s) in enumerate(inp)}
    return np.array(x).transpose(*[en[x] for x in out])


def read_table(fn: str) -> list:
    """
    Reads a table from a file and returns it as a list of lists.
    If element is a Python literal, it is converted to the corresponding type;
    otherwise it is left as a string.
    """
    def elem(x: str):  # convert string to Python literal if possible
        try:
            return ast.literal_eval(x)
        except (ValueError, TypeError, SyntaxError):
            return x

    res = []
    with open(fn, 'r') as f:
        for line in f:
            row = [elem(x) for x in line.split()]
            if len(row) > 0:
                res.append(row)
    return res


def read_amps(fn, limit=0):
    """
    Reads a list of bitstrings and amplitudes from a file.

    If limit > 0, only the first limit amplitudes are read.
    Each row is either a bitstring (string of 0s and 1s)
             or has the form: bitstring amplitude_real amplitude_imag
    """
    res = []
    with open(fn) as f:
        for line in f:
            sp = line.split()
            if not sp:
                continue
            if len(sp) == 1:
                res.append([int(c) for c in sp[0]])
            else:
                amp, re, im = sp
                res.append(([int(c) for c in amp], float(re) + 1.0j * float(im)))
            if limit == 1:
                break
            limit -= 1
    return res


class PerformanceProfile:
    """
    Measures the performance of the computer:
    - CPU fp32 performance in GFLOP/s
    - Memory bandwidth in GB/s
    """
    def __init__(self, file='profile.json'):
        self.flop_per_second = 0
        self.memory_bandwidth = 0
        if os.path.exists(file):
            self.load(file)
        else:
            self.measure()
            self.save(file)

    def load(self, file):
        with open(file) as f:
            data = json.load(f)
            self.flop_per_second = data['flop_per_second']
            self.memory_bandwidth = data['memory_bandwidth']

    def save(self, file):
        with open(file, 'w') as f:
            json.dump({'flop_per_second': self.flop_per_second, 'memory_bandwidth': self.memory_bandwidth}, f)

    def measure(self):
        # measure flop per second, multiply matrices 1024x1024 with pytorch
        import torch
        print('Measuring performance...')
        m, n, k = 1024, 256, 256
        ntry = 1024
        a = torch.randn(m, n, dtype=torch.float32)
        b = torch.randn(n, k, dtype=torch.float32)
        t0 = time.time()
        for i in range(ntry):
            c = a @ b
        t1 = time.time()
        self.flop_per_second = ntry * 2 * m * n * k / (t1-t0)
        print(f'flop_per_second = {self.flop_per_second*1e-9:.0f} GFLOP/s')

        # measure memory bandwidth by copying long vector
        print('Measuring memory bandwidth...')
        size, ntry = 2**28, 2
        a = torch.randn(size, dtype=torch.float32)
        b = torch.clone(a)
        t0 = time.time()
        for i in range(ntry):
            b = torch.clone(a)
        t1 = time.time()
        self.memory_bandwidth = ntry * size * 8 / (t1-t0)
        print(f'memory_bandwidth = {self.memory_bandwidth*1e-9:.0f} GB/s')


def estimate_time(cost, memrw, dtype='c64'):
    profile = PerformanceProfile()
    rw_count = 4  # each element is read and written twice (once in multiplication and once in index permutation)
    bytes_per_element = {'c64': 8, 'c128': 16, 'f32': 4, 'f64': 8}[dtype]
    flops_per_element = {'c64': 8, 'c128': 16, 'f32': 2, 'f64': 4}[dtype]  # equivalent number of fp32 operations
    return cost*flops_per_element/profile.flop_per_second + rw_count*memrw*bytes_per_element/profile.memory_bandwidth
