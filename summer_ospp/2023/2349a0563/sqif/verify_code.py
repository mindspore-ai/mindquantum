import itertools
import sympy
import numpy as np
from numpy.linalg import norm
from schnorr_algorithm import SearchType
from schnorr_algorithm import lll_reduction, babai_algorithm_extension


####################### Question 2 verify code #######################
print("\n####################### Question 2 verify #######################")
# Question 2 verify code:
mat_b_54 = np.array([
    [2, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 3, 0, 0],
    [0, 0, 0, 2, 0],
    [0, 0, 0, 0, 1],
    [6931, 10986, 16094, 19459, 23979]
], np.float64)

t_5 = np.array([[0], [0], [0], [0], [0], [176985]], np.float64)

bop_paper = np.array([[2], [4], [9], [8], [0], [176993]], np.float64)

bop_5 = babai_algorithm_extension(
    lat_basis=mat_b_54,
    t=t_5,
    delta=0.75,
    search_type=SearchType.NONE)

print(f"Difference in original paper: {norm(bop_paper - t_5):.2f}")
print(f"Difference in implementation: {norm(bop_5 - t_5):.2f}")
print(f"`bop` in implementation: {bop_5.astype(int).flatten().tolist()}")


####################### Question 3 verify code #######################
print("\n####################### Question 3 verify #######################")
# Question 3 verify code:
def check_lll_conditon(mat_d, delta=0.75):
    n = mat_d.shape[1]
    ret = True
    _mat_q, mat_r = np.linalg.qr(mat_d)
    # condition 1
    for i in range(n):
        if not ret:
            break
        for j in range(i+1, n):
            if abs(mat_r[i, j] / mat_r[i, i]) > 0.5:
                ret = False
                print("break rule 1",
                      f"i={i}, j={j}, R[i, j]={mat_r[i, j]}, R[i, i]={mat_r[i,i]}")
                break
    # condition 2
    for i in range(n-1):
        if delta * mat_r[i, i]**2 > mat_r[i, i+1]**2 + mat_r[i+1, i+1]**2:
            ret = False
            print("break rule 2:",
                  f"i={i}, R[i, i]={mat_r[i, i]} {mat_r[i, i+1]} {mat_r[i+1,i+1]}")
            break
    # condition 3
    alpha = 1.0 / (delta - 0.25)
    for i in range(n-1):
        if mat_r[i, i]**2 > alpha * mat_r[i+1, i+1]**2:
            ret = False
            print("break rule 3.")
            break
    if ret:
        print("Pass all rules.")


mat_b_10 = np.array([
    [3,     0,     0,     0,     0,     0,     0,     0,     0,        0],
    [0,     2,     0,     0,     0,     0,     0,     0,     0,        0],
    [0,     0,     3,     0,     0,     0,     0,     0,     0,        0],
    [0,     0,     0,     1,     0,     0,     0,     0,     0,        0],
    [0,     0,     0,     0,     1,     0,     0,     0,     0,        0],
    [0,     0,     0,     0,     0,     3,     0,     0,     0,        0],
    [0,     0,     0,     0,     0,     0,     1,     0,     0,        0],
    [0,     0,     0,     0,     0,     0,     0,     1,     0,        0],
    [0,     0,     0,     0,     0,     0,     0,     0,     2,        0],
    [0,     0,     0,     0,     0,     0,     0,     0,     0,        2],
    [6931, 10986, 16094, 19459, 23979, 25649, 28332, 29444, 31355, 33673]
], np.float64)

mat_d_10 = np.array([
    [0,   0,  3,  0,  0,  0,  3,  0, -3, -3],
    [0,   0,  2,  0,  4, -4,  0,  4, -2,  4],
    [-3,  0,  0,  0,  0,  0, -3,  0,  0,  0],
    [1,   2,  1,  4, -4, -2, -2,  0, -1,  0],
    [2,   0,  2, -2,  0,  0,  1, -1,  0,  4],
    [0,   0, -3, -3,  0,  0,  0,  0, -3,  3],
    [-3,  3, -1,  0,  1,  2,  1,  2, -2, -1],
    [0,  -2,  0,  1,  2, -1,  1, -3,  3, -3],
    [0,  -2, -2,  0, -2,  0,  0,  0,  2,  2],
    [2,  -2,  0, -2,  0,  2, -2,  2,  0,  0],
    [0,  -2, -2,  0,  1,  3,  1, -2, -2, -1]], np.float64)
print("\nCheck the matrix in paper:")
check_lll_conditon(mat_d_10)

print("\nCheck the matrix in implementation:")
# Inplace `lll_reduction` with `lll_reduction_fpylll` return same result.
mat_d_10_im = lll_reduction(mat_b_10.copy(), delta=0.75)
check_lll_conditon(mat_d_10_im)

print("\nThe implemented LLL-reduction matrix for B_10_4:")
print(mat_d_10_im.astype(int))


####################### Question 4 verify code #######################
print("\n####################### Question 4 verify #######################")
mat_m_paper = np.linalg.inv(mat_b_10.T @ mat_b_10) @ mat_b_10.T @ mat_d_10
mat_m_im = np.linalg.inv(mat_b_10.T @ mat_b_10) @ mat_b_10.T @ mat_d_10_im

print("\nTransform matrix in paper is not all integer, it's:")
print(np.round(mat_m_paper, 2))

print("\nTransform matrix in implementation is  all integer, it's:")
print(np.round(mat_m_im, 2))


####################### Question 5 verify code #######################
print("\n####################### Question 5 verify #######################")
z1, z2, z3, z4, z5 = sympy.symbols('z1 z2 z3 z4 z5')
Hc5 = 781 - 142*z1 - 64*z2 - 81*z3 - 213*z4 - 4.5*z5 \
    - 13.5*z1*z2 + 3.5*z1*z3 + 18*z1*z4 + 17.5*z1*z5 \
    - 29*z2*z3 + 19.5*z2*z4 - 34*z2*z5 \
    - 31.5*z3*z4 - 2.5*z3*z5 \
    + 4.5*z4*z5
print("Hc5:", Hc5)
# 00110  -> 1 1 -1 -1 1
energy_00110 = Hc5.subs(z1, 1).subs(
    z2, 1).subs(z3, -1).subs(z4, -1).subs(z5, 1)
print("energy_00110:", energy_00110)  # 论文中是 186，实际得到 789
print("Run over all state and get their energy:")
for state in itertools.product(('0', '1'), repeat=5):
    # '0' -> 1, '1' -> -1
    v = [1 if s == '0' else -1 for s in state]
    energy = Hc5.subs(z1, v[0]).subs(z2, v[1]).subs(
        z3, v[2]).subs(z4, v[3]).subs(z5, v[4])
    print("".join(state), int(energy))
