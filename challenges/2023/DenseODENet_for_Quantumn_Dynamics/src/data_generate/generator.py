import numpy as np
import mindspore as ms
from typing import List
from .hamiltonian import random_hamiltonian, random_initial_state, construct_with_eigenvector


class Generator:
    def __init__(self, dim: int, dtype=np.float32):
        self.dim = dim
        self.dtype = dtype

    def evolution(self, init_s, h, t_list):
        r"""
        S = exp(-iHt) * S0
        return: (len(t_list) + 1, dim + dim)
        """
        eigenvalues, eigenvectors = np.linalg.eigh(h)

        st_list = [np.concatenate((init_s.real, init_s.imag), axis=0)[:, 0]]
        for t in t_list:
            ut_eigenvalues = np.exp(complex(0, -1) * t * eigenvalues)
            ut = construct_with_eigenvector(diag=ut_eigenvalues, eigenvectors=eigenvectors)
            st = np.dot(ut, init_s)
            st = np.concatenate((st.real, st.imag), axis=0, dtype=self.dtype)[:, 0]
            st_list.append(st)
        st_list = np.stack(st_list, axis=0)
        return st_list

    def generate_trajectory(self, num: int, t_list: np.ndarray):
        r"""
        return: (2, dim, dim); (num, t_len + 1, dim + dim)
        """
        trajectories = []
        h = random_hamiltonian(self.dim)
        for idx in range(num):
            s0 = random_initial_state(self.dim)
            sts = self.evolution(init_s=s0, h=h, t_list=t_list)
            trajectories.append(sts)
        trajectories = np.stack(trajectories, axis=0)
        time_points = np.concatenate((np.array([0., ], dtype=self.dtype), t_list), axis=0)
        h_real, h_imag = h.real, h.imag
        h = np.stack((h_real, h_imag), axis=0)
        return h, trajectories, time_points

    def generate_multi_h_trajectory(self, h_num, s0_num, t_list):
        r"""
        return: (h_num, 2, dim, dim); (h_num, s0_num, t_len + 1, dim + dim); (h_num, t_len + 1)
        """
        h_stack = []
        trajectories_stack = []
        time_points_stack = []

        for h_idx in range(h_num):
            h, trajectories, time_points = self.generate_trajectory(num=s0_num, t_list=t_list)
            h_stack.append(h)
            trajectories_stack.append(trajectories)
            time_points_stack.append(time_points)
        h_stack = np.stack(h_stack, axis=0)
        trajectories_stack = np.stack(trajectories_stack, axis=0)
        time_points_stack = np.stack(time_points_stack, axis=0)
        return h_stack, trajectories_stack, time_points_stack


class Dataset:
    def __init__(self, H_list, trajectories: np.ndarray, t_points, batch_size: int, dtype, shuffle: bool):
        r"""
        :param H_list: hamiltonian (h_num, 2, dim, dim);
        :param trajectories: (h_num, s0_num, t_len + 1, dim + dim)
        :param batch_size:
        :param t_points: (h_num, t_len + 1)
        :param dtype:
        :param shuffle:
        """
        self.H_list = ms.Tensor(H_list, dtype=dtype)
        self.trajectories = ms.Tensor(trajectories, dtype=dtype)
        self.t_points = ms.Tensor(t_points, dtype=dtype)
        self.h_num = self.trajectories.shape[0]
        self.s0_num = self.trajectories.shape[1]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def fetch(self):
        r"""
        yield: (dim, dim); (batch_size, t_len + 1, dim + dim); (t_len + 1, )
        """
        h_indices = list(range(self.h_num))
        s0_indices = list(range(self.s0_num))
        if self.shuffle:
            np.random.shuffle(h_indices)
            np.random.shuffle(s0_indices)

        for h_idx in h_indices:
            start = 0
            while start < self.s0_num:
                end = start + self.batch_size
                if end > self.s0_num:
                    end = self.s0_num
                batch_indices = s0_indices[start:end]
                yield self.H_list[h_idx], self.trajectories[h_idx, batch_indices], self.t_points[h_idx]
                start = end
