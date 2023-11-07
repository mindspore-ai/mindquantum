import numpy as np


def random_hamiltonian(dim: int):
    h_real = np.random.rand(dim, dim)
    h_imag = np.random.rand(dim, dim)
    h_real = np.triu(h_real)
    h_imag = np.triu(h_imag)
    h_real = h_real + h_real.T - np.diag(h_real.diagonal())
    h_imag = h_imag - h_imag.T

    h = np.zeros((dim, dim), dtype=complex)
    h.real = h_real
    h.imag = h_imag
    return h


def random_initial_state(dim: int):
    s_real = np.random.rand(dim, 1)
    s_imag = np.random.rand(dim, 1)
    modulus = np.sqrt(np.sum(s_real**2) + np.sum(s_imag**2))
    s_real = s_real / modulus
    s_imag = s_imag / modulus
    s = np.zeros((dim, 1), dtype=complex)
    s.real = s_real
    s.imag = s_imag
    return s


def check_unitary(u):
    m = np.dot(u.conj().T, u)
    if np.sum(np.abs(m.real - np.eye(m.shape[0]))) == 0 and np.sum(np.abs(m.imag - np.zeros(m.shape[0]))) == 0:
        return True
    else:
        return False


def check_emi(h):
    m = h.conj().T - h
    if np.sum(np.abs(m.real)) == 0 and np.sum(np.abs(m.imag)) == 0:
        return True
    else:
        return False


def construct_with_eigenvector(diag, eigenvectors):
    matrix = np.dot(eigenvectors, np.diag(diag))
    matrix = np.dot(matrix, eigenvectors.conj().T)
    return matrix


# aa = random_hamiltonian(6)
# d = check_unitary(aa)
# print(d)
# aa = np.zeros((2, 2), dtype=complex)
# aa.imag = np.array([[0, -1], [1, 0]])
# # aa.real = np.array([[0, 1], [1, 0]])
# aa.real = np.array([[1, 0], [0, -1]])
#
# m, v = np.linalg.eigh(aa)
# c3 = construct_with_eigenvector(m, v)
# print(c3)


# r, i = random_hamiltonian(dim=3)
# check_emi(r, i)
# check_unitary(r, i)


# print(r - r.T)
# print(i + i.T)
#
# hh_real = np.dot(r, r.T) - np.dot(i, -i.T)
# hh_img = np.dot(r, -i.T) + np.dot(i, r.T)
#
# anti_hh_real = np.dot(r.T, r) - np.dot(-i.T, i)
# anti_hh_img = np.dot(-i.T, r) + np.dot(r.T, i)
#
# print(hh_real - anti_hh_real)
# print(hh_img - anti_hh_img)
