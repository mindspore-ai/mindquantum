import pandas as pd
import numpy as np
from scipy.linalg import expm

# 载入数据，这里用温度数据为例。b每一行是一个样本，每一列是一维参数
csv_file_path = 'temperature.csv'
df = pd.read_csv(csv_file_path, header=None)
matrix_data = df.to_numpy()
b = matrix_data

# b作为数据矩阵来构造协方差矩阵，x是待降维矩阵。x每一列是一个样本，这里是对全样本降维，因此x就是b的厄密共轭。
x = np.transpose(b)
temp = b.shape
column = temp[0]
row = temp[1]
temp2 = x.shape

# 临时变量，不用管
t1 = np.ones((1, temp2[1]))
t2 = 1 / column * np.sum(b, axis=0)
t2 = t2.reshape(365, 1)

# 中心化
x = x - np.kron(t1, t2)
b = b - np.kron(np.ones((column, 1)) / column, np.sum(b, axis=0))
B = np.dot(np.transpose(b), b)

# 一些基本门操作
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Had = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
Z = np.array([[1, 0], [0, -1]])

# 数据维度、降维后维数等一些参数定义
L = len(B)
Hs = B
r = 16
c = 0.001
A = np.eye(L)
Qz = np.zeros((r, temp2[1]), dtype=complex)
Cz = Qz
D, U = np.linalg.eig(B)
l = round(2 ** np.ceil(np.log2(r)))
aim = np.diag(D[0:l])
Ht = aim
t3 = Had
for _ in range(round(np.log2(l) - 1)):
    t3 = np.kron(t3, Had)
INT = np.sqrt(l) * t3

# 降维的共振哈密顿量和演化算符
H = lambda HT: -np.kron(np.array([[1, 0], [0, 0]]), np.kron(np.diag([0] + [1] * (l - 1)), np.eye(L))) + np.kron(
    np.array([[0, 0], [0, 1]]),
    np.kron(-HT, np.eye(len(B))) + np.kron(np.eye(len(HT)), Hs)) + np.pi / 2 * c * np.kron(Y, np.kron(INT, A))
time = 1 / c
Ut = expm(-1j * H(Ht) * time)

# 降维中的一步子程序，采用其他文献中的方法，可以暂时忽略
xx = np.sum(U[:, 0:l], axis=1) / np.sqrt(l)
print(xx.shape)
inp = np.array([1] + [0] * (L - 1))
inp = inp / np.linalg.norm(inp)
print(inp.shape)
xx = xx / np.linalg.norm(xx)
al = np.dot(inp, xx)
inbar = xx - al * inp
inbar = inbar / np.linalg.norm(inbar)
xbar = np.sqrt(1 - al ** 2) * inp - al * inbar
Ur2 = np.outer(xx, inp) + np.outer(xbar, inbar)
Ur2 = Ur2 + np.eye(len(U)) - np.outer(xx, xx) - np.outer(xbar, xbar)
Ur = np.kron(np.eye(2 * l), Ur2)

# 对x中每一列样本进行降维
for nn in range(temp2[1]):
    # 初始化输入
    Phi = x[:, nn]
    Psi0 = np.kron(np.array([1, 0]), np.kron(np.array([1] + [0] * (l - 1)), Phi))

    # 时间演化，测量，得到降维后量子态
    Psi = np.dot(Ut, Psi0)
    Psi[:L * l] = 0
    Psi = Psi / np.linalg.norm(Psi)
    Psi = np.dot(Ur, Psi)
    z = np.zeros(l, dtype=complex)
    for t in range(l):
        z[t] = Psi[L * l + t * L]

    # 记录结果，计算经典降维结果作为理论值
    Qz[:, nn] = z / np.linalg.norm(z)
    cz = np.dot(Phi.T, U[:, 0:l])
    Cz[:, nn] = cz / np.linalg.norm(cz)

# 计算相似度、误差
error = np.sum(np.abs(Cz - Qz)) / np.size(Qz)
inner = 0
for mm in range(temp[0]):
    inner = inner + np.inner(Qz[:, mm], Cz[:, mm])
inner = abs(inner / temp[0])

# 输出计算结果
formatted_number = "{:s} {:.2e}".format("error:", error)
print(formatted_number)
formatted_number = "{:s} {:.2e}".format("inner:", inner)
print(formatted_number)

# 保存数据
df = pd.DataFrame(Qz)
# 文件路径
file_path = 'DR_temperature.csv'
df.to_csv(file_path, index=False, header=False)