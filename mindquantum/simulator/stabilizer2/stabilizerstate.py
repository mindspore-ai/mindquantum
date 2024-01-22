import numpy as np
import random
import time
from enum import Enum, unique


@unique
class Gate(Enum):
    CNOT = 0
    HADAMARD = 1
    PHASE = 2
    MEASURE = 3


class QuantumCircuit:
    # 从文件名 fn 读取量子电路，带有可选参数 params
    def __init__(self, fn: str, params: str):
        self.c = 0
        self.DISPQSTATE = 0
        self.DISPTIME = 0
        self.SILENT = 0
        self.DISPPROG = 0
        self.SUPPRESSM = 0
        if params:
            for t in range(len(params)):
                if params[t] == "q" or params[t] == "Q":
                    self.DISPQSTATE = 1
                if params[t] == "p" or params[t] == "P":
                    self.DISPPROG = 1
                if params[t] == "t" or params[t] == "T":
                    self.DISPQSTATE = 1
                if params[t] == "s" or params[t] == "S":
                    self.SILENT = 1
                if params[t] == "m" or params[t] == "M":
                    self.SUPPRESSM = 1
        fp = open(fn, "r")
        if not fp:
            fn2 = fn + ".chp"
            fp = open(fn2, "r")
            if not fp:
                print("error filename!")
                return
        self.numGates = 0
        self.numQubits = 0
        content = fp.readline()
        while content:
            if content[0] == "#" or content[0] == "\r" or content[0] == "\n":
                content = fp.readline()
                continue
            if int(content[2]) + 1 > self.numQubits:
                self.numQubits = int(content[2]) + 1
            if content[0] == "c" or content[0] == "C":
                if int(content[4]) + 1 > self.numQubits:
                    self.numQubits = int(content[4]) + 1
            self.numGates += 1
            content = fp.readline()
        fp.close()
        t = 0
        self.instOpode = np.zeros(self.numGates, dtype=np.uint8)
        self.qubit1 = np.zeros(self.numGates, dtype=np.int64)
        self.qubit2 = np.zeros(self.numGates, dtype=np.int64)
        fp = open(fn, "r")
        if not fp:
            fn2 = fn + ".chp"
            fp = open(fn2, "r")
            if not fp:
                print("error filename!")
                return
        content = fp.readline()
        while content:
            if content[0] == "#" or content[0] == "\r" or content[0] == "\n":
                content = fp.readline()
                continue
            if content[0] == "c" or content[0] == "C":
                self.instOpode[t] = np.uint8(Gate.CNOT.value)
                self.qubit2[t] = np.int64(content[4])
            if content[0] == "h" or content[0] == "H":
                self.instOpode[t] = np.uint8(Gate.HADAMARD.value)
            if content[0] == "p" or content[0] == "P":
                self.instOpode[t] = np.uint8(Gate.PHASE.value)
            if content[0] == "m" or content[0] == "M":
                self.instOpode[t] = np.uint8(Gate.MEASURE.value)
            self.qubit1[t] = np.int64(content[2])
            t += 1
            content = fp.readline()
        fp.close()


class QuantumState:
    # 构造函数，创建一个有n个量子位，输入状态根据s设置
    def __init__(self, n: np.int64, s: str):
        self.numQubits = n
        self.over32 = (n >> 5) + 1

        self.x = np.zeros((2 * n + 1, self.over32), dtype=np.int64)
        self.z = np.zeros((2 * n + 1, self.over32), dtype=np.int64)
        # 类型未定
        self.r = np.zeros(2 * n + 1, dtype=np.int64)
        self.pw = np.zeros(32, dtype=np.int64)
        for i in range(32):
            self.pw[i] = pow(2, i)
        for i in range(2 * n + 1):
            if i < n:
                self.x[i][i >> 5] = self.pw[i & 31]
            elif i < 2 * n:
                j = i - n
                self.z[i][j >> 5] = self.pw[j & 31]
        if s:
            self.preparestate(s)

    # 准备初始状态的输入
    def preparestate(self, s: str):
        for i in range(len(s)):
            if s[i] == "Z":
                self.hadamard(i)
                self.phase(i)
                self.phase(i)
                self.hadamard(i)
            if s[i] == "x":
                self.hadamard(i)
            if s[i] == "X":
                self.hadamard(i)
                self.phase(i)
                self.phase(i)
            if s[i] == "y":
                self.hadamard(i)
                self.phase(i)
            if s[i] == "Y":
                self.hadamard(i)
                self.phase(i)
                self.phase(i)
                self.phase(i)

    # 施加CNOT门,控制位b 目标位置c
    def cnot(self, b: np.int64, c: np.int64):
        b5 = b >> 5
        c5 = c >> 5
        pwb = self.pw[b & 31]
        pwc = self.pw[c & 31]
        for i in range(2 * self.numQubits):
            if self.x[i][b5] & pwb:
                self.x[i][c5] ^= pwc
            if self.z[i][c5] & pwc:
                self.z[i][b5] ^= pwb
            if (
                (self.x[i][b5] & pwb)
                and (self.z[i][c5] & pwc)
                and (self.x[i][c5] & pwc)
                and (self.z[i][b5] & pwb)
            ):
                self.r[i] = (self.r[i] + 2) % 4
            if (
                (self.x[i][b5] & pwb)
                and (self.z[i][c5] & pwc)
                and not (self.x[i][c5] & pwc)
                and not (self.z[i][b5] & pwb)
            ):
                self.r[i] = (self.r[i] + 2) % 4

    # 对比特b施加Hadamard门
    def hadamard(self, b: np.int64):
        b5 = b >> 5
        pw = self.pw[b & 31]
        for i in range(2 * self.numQubits):
            tmp = self.x[i][b5]
            self.x[i][b5] ^= (self.x[i][b5] ^ self.z[i][b5]) & pw
            self.z[i][b5] ^= (self.z[i][b5] ^ tmp) & pw
            if (self.x[i][b5] & pw) and (self.z[i][b5] & pw):
                self.r[i] = (self.r[i] + 2) % 4

    # 对比特b施加phase门(|0>->|0>, |1>->i|1>)
    def phase(self, b: np.int64):
        b5 = b >> 5
        pw = self.pw[b & 31]
        for i in range(2 * self.numQubits):
            if (self.x[i][b5] & pw) and (self.z[i][b5] & pw):
                self.r[i] = (self.r[i] + 2) % 4
            self.z[i][b5] ^= self.x[i][b5] & pw

    # 将第 i 行设置为等于第 k 行
    def rowcopy(self, i: np.int64, k: np.int64):
        for j in range(self.over32):
            self.x[i][j] = self.x[k][j]
            self.z[i][j] = self.z[k][j]
        self.r[i] = self.r[k]

    # 交换第 i 行和第 k 行
    def rowswap(self, i: np.int64, k: np.int64):
        self.rowcopy(2 * self.numQubits, k)
        self.rowcopy(k, i)
        self.rowcopy(i, 2 * self.numQubits)

    # 将第 i 行设置为等于第 b 个可观察值 (X_1,...X_n,Z_1,...,Z_n)
    def rowset(self, i: np.int64, b: np.int64):
        for j in range(self.over32):
            self.x[i][j] = 0
            self.z[i][j] = 0
        self.r[i] = 0
        if b < self.numQubits:
            b5 = b >> 5
            b31 = b & 31
            self.x[i][b5] = self.pw[b31]
        else:
            b5 = (b - self.numQubits) >> 5
            b31 = (b - self.numQubits) & 31
            self.z[i][b5] = self.pw[b31]


    # 返回第 i 行左乘第 k 行时的相位 (0,1,2,3)
    def clifford(self, i: np.int64, k: np.int64) -> np.int64:
        e = 0
        for j in range(self.over32):
            for l in range(32):
                pw = self.pw[l]
                if (self.x[k][j] & pw) and not (self.z[k][j] & pw):
                    if (self.x[i][j] & pw) and (self.z[i][j] & pw):
                        e += 1
                    if not (self.x[i][j] & pw) and (self.z[i][j] & pw):
                        e -= 1
                if (self.x[k][j] & pw) and (self.z[k][j] & pw):
                    if not (self.x[i][j] & pw) and (self.z[i][j] & pw):
                        e += 1
                    if (self.x[i][j] & pw) and not (self.z[i][j] & pw):
                        e -= 1
                if not (self.x[k][j] & pw) and (self.z[k][j] & pw):
                    if (self.x[i][j] & pw) and not (self.z[i][j] & pw):
                        e += 1
                    if (self.x[i][j] & pw) and (self.z[i][j] & pw):
                        e -= 1

        e = (e + self.r[i] + self.r[k]) % 4

        if e >= 0:
            return e
        else:
            return e + 4

    # 第 i 行左乘第 k 行
    def rowmult(self, i: np.int64, k: np.int64):
        self.r[i] = self.clifford(i, k)
        for j in range(self.over32):
            self.x[i][j] ^= self.x[k][j]
            self.z[i][j] ^= self.z[k][j]

    # 打印状态q的destabilizer和stabilizer
    def printstate(self):
        for i in range(2 * self.numQubits):
            if i == self.numQubits:
                print()
                for j in range(self.numQubits):
                    print("-", end="")
            if self.r[i] == 2:
                print()
                print("-", end="")
            else:
                print()
                print("+", end="")
            for j in range(self.numQubits):
                j5 = j >> 5
                pw = self.pw[j & 31]
                if not (self.x[i][j5] & pw) and not (self.z[i][j5] & pw):
                    print("I", end="")
                if (self.x[i][j5] & pw) and not (self.z[i][j5] & pw):
                    print("X", end="")
                if (self.x[i][j5] & pw) and (self.z[i][j5] & pw):
                    print("Y", end="")
                if not (self.x[i][j5] & pw) and (self.z[i][j5] & pw):
                    print("Z", end="")
        print()

    # 测量量子位 b
    # 如果结果始终为 0，则返回 0
    # 如果结果始终为 1，则返回 1
    # 如果结果是随机的并且选择了 0，则返回 2
    # 如果结果是随机的并且选择了 1，则返回 3
    # sup：如果应抑制确定的测量结果，则为 1，否则为 0
    def measure(self, b: np.int64, sup: np.bool_) -> np.int64:
        ran = 0
        b5 = b >> 5
        pw = self.pw[b & 31]
        for p in range(self.numQubits):
            if self.x[p + self.numQubits][b5] & pw:
                ran = 1
            if ran:
                break
        debugfunc(self, "measure01")
        if ran:
            self.rowcopy(p, p + self.numQubits)
            debugfunc(self, "measure02")
            self.rowset(p + self.numQubits, b + self.numQubits)
            debugfunc(self, "measure03")
            self.r[p + self.numQubits] = 2 * random.randint(0, 1)
            for i in range(2 * self.numQubits):
            
                if (i != p) and (self.x[i][b5] & pw):
                    self.rowmult(i, p)
                    debugfunc(self, "measure04")
            if self.r[p + self.numQubits]:
                debugfunc(self, "measure05")
                return 3
            else:
                return 2
        
        if not ran and not sup:
            for m in range(self.numQubits):
                if self.x[m][b5] & pw:
                    break
            self.rowcopy(2 * self.numQubits, m + self.numQubits)
            for i in range(m + 1, self.numQubits):
                if self.x[i][b5] & pw:
                    self.rowmult(2 * self.numQubits, i + self.numQubits)
                if self.r[2 * self.numQubits]:
                    return 1
                else:
                    return 0
        return 0

    # 进行高斯消去，使稳定器生成器具有以下形式：
    # 在顶部，包含 X 和 Y 的最小生成器集，采用“准上三角”形式。
    # （返回值=此类生成器的数量=非零基态数量的log_2）
    # 在底部，生成器仅包含准上三角形式的 Z。
    def gaussian(self) -> np.int64:
        i = self.numQubits
        falg=0
        for j in range(self.numQubits):
            j5 = j >> 5
            pw = self.pw[j & 31]
            for k in range(i, 2 * self.numQubits):
                if self.x[k][j5] & pw:
                    flag=1
                    break
            if(not falg):
                k+=1
            if k < 2 * self.numQubits:
                self.rowswap(i, k)
                self.rowswap(i - self.numQubits, k - self.numQubits)
                for k2 in range(i + 1, 2 * self.numQubits):
                    if self.x[k2][j5] & pw:
                        self.rowmult(k2, i)
                        self.rowmult(i - self.numQubits, k2 - self.numQubits)
                i += 1
        g = i - self.numQubits
        for j in range(self.numQubits):
            j5 = j >> 5
            pw = self.pw[j & 31]
            for k in range(i, 2 * self.numQubits):
                if self.z[k][j5] & pw:
                    break
            if k < 2 * self.numQubits:
                self.rowswap(i, k)
                self.rowswap(i - self.numQubits, k - self.numQubits)
                for k2 in range(i + 1, 2 * self.numQubits):
                    if self.z[k2][j5] & pw:
                        self.rowmult(k2, i)
                        self.rowmult(i - self.numQubits, k2 - self.numQubits)
                i += 1
        return g

    # 将 q 的“暂存空间”中应用泡利运算符的结果打印到 |0...0>
    def printbasisstate(self):
        e = self.r[2 * self.numQubits]
        for j in range(self.numQubits):
            j5 = j >> 5
            pw = self.pw[j & 31]
            if (self.x[2 * self.numQubits][j5] & pw) and (
                self.z[2 * self.numQubits][j5] & pw
            ):
                e = (e + 1) % 4
        if e == 0:
            print()
            print(" +|", end="")
        if e == 1:
            print()
            print("+i|", end="")
        if e == 2:
            print()
            print(" -|", end="")
        if e == 3:
            print()
            print("-i|", end="")
        for j in range(self.numQubits):
            j5 = j >> 5
            pw = self.pw[j & 31]
            if self.x[2 * self.numQubits][j5] & pw:
                print("1", end="")
            else:
                print("0", end="")
        print(">", end="")

    # 找到泡利算子 P，使得基础状态 P|0...0> 出现时 q 的振幅非零，并且
    # 将 P 写入 q 的暂存空间。为了实现这一点，高斯消去法必须已经是
    # 对 q 执行。 g 是 gaussian(q) 的返回值。
    def seed(self, g: np.int64):
        self.r[2 * self.numQubits] = 0
        for j in range(self.over32):
            self.x[2 * self.numQubits][j] = 0
            self.z[2 * self.numQubits][j] = 0
        for i in range(2 * self.numQubits - 1, self.numQubits + g - 1 ,- 1):
            f = self.r[i]
            for j in range(self.numQubits - 1, -1, -1):
                j5 = j >> 5
                pw = self.pw[j & 31]
                if self.z[i][j5] & pw:
                    min = j
                    if self.x[2 * self.numQubits][j5] & pw:
                        f = (f + 2) % 4
            if f == 2:
                j5 = min >> 5
                pw = self.pw[min & 31]
                self.x[2 * self.numQubits][j5] ^= pw

    # 以 ket 表示法打印状态
    def printket(self):
        g = self.gaussian()
        print()
        print("2^" + str(g) + " nonzero basis states", end="")
        if g > 31:
            print()
            print("State is WAY too big to print", end="")
            return
        self.seed(g)
        self.printbasisstate()
        for t in range(self.pw[g] - 1):
            t2 = t ^ (t + 1)
            for i in range(g):
                if t2 & self.pw[i]:
                    self.rowmult(2 * self.numQubits, self.numQubits + i)
            self.printbasisstate()
        print()


# Simulate the quantum circuit
def runprog(h: QuantumCircuit, q: QuantumState):
    mvirgin = 1
    start_time = time.time()
    debugfunc( q, "init", -1)
    for t in range(h.numGates):
        if h.instOpode[t] == Gate.CNOT.value:
            q.cnot(h.qubit1[t], h.qubit2[t])
            debugfunc(q, "cnot", t)

        if h.instOpode[t] == Gate.HADAMARD.value:
            q.hadamard(h.qubit1[t])
            debugfunc(q, "hadamard", t)

        if h.instOpode[t] == Gate.PHASE.value:
            q.phase(h.qubit1[t])
            debugfunc(q, "phase", t)

        if h.instOpode[t] == Gate.MEASURE.value:
            if mvirgin and h.DISPTIME:
                end_time = time.time()
                time_diff = end_time - start_time
                print("Gate time:" + str(time_diff) + "seconds")
                start_time = time.time()
            mvirgin = 0
            m = q.measure(h.qubit1[t], h.SUPPRESSM)
            debugfunc(q, "measure", t)
            if not h.SILENT:
                print("Outcome of measuring qubit " + str(h.qubit1[t]) + ": ", end="")
                if m > 1:
                    print(str(m - 2) + " (random)")
                else:
                    print(m)
        if h.DISPPROG:
            if h.instOpode[t] == Gate.CNOT:
                print("CNOT " + str(h.qubit1[t]) + "->" + str(h.qubit2[t]))
            if h.instOpode[t] == Gate.HADAMARD:
                print("Hadamard " + str(h.qubit1[t]))
            if h.instOpode[t] == Gate.PHASE:
                print("Phase " + str(h.qubit1[t]))
    if h.DISPTIME:
        end_time = time.time()
        time_diff = end_time - start_time
        print("Measurement time: " + str(time_diff) + " seconds")
    if h.DISPQSTATE:
        print("Final state: ")
        q.printstate()
        q.gaussian()
        q.printstate()
        q.printket()


def debugfunc( q: QuantumState, gatename: str, instnumber=-1):
    # print()
    # print()
    # print("the state after " + gatename)
    # print(instnumber)
    # print("the state of qubits:")
    # print("num of qubits:" + str(q.numQubits))
    # print(q.x)
    # print(q.z)
    # print(q.r)
    # print()
    # print()
    pass


if __name__ == "__main__":
    print("CHP: Efficient Simulator for Stabilizer Quantum Circuits")
    h = QuantumCircuit("test/teleport.chp", "q")
    q = QuantumState(h.numQubits, "")
    runprog(h, q)
